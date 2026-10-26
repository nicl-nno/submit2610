'Loss-based example difficulty (the ToolACE-style static data filter).\n\n\nBefore training, the unified pool can be trimmed by the teacher-forced\nreference loss of a DIFFICULTY model (the base model — the "untrained"\nreference in the ToolACE spirit): the easiest examples (the model already\nreproduces the reference → no learning gradient / saturated reward → the r2\nzero-variance groups) and the hardest examples (out-of-distribution or\nmis-labelled noise) carry little training value. The filter drops a\nper-class quantile share from EACH tail.\n\nThe scoring is INLINE, with a SEPARATE difficulty model (a CLI argument):\nload → score → unload from VRAM before the training model loads (the scorer\nis a pure bf16 forward pass, no gradients, deterministic).\n\nDevice and memory contract\n\n* :func:`load_difficulty_model` resolves the scoring device (CUDA when\n  available) and MOVES the model there — the model must never score on CPU\n  by accident; ``flash_attention_2`` is CUDA-only and falls back to ``sdpa``\n  on CPU;\n* the forward runs with ``use_cache=False`` (no KV-cache transient) and the\n  batches are planned by a PADDED-TOKEN budget\n  (:data:`DEFAULT_MAX_BATCH_TOKENS`), not only by the example count: the\n  ``[B, T, V]`` logits tensor over the 145k vocab is the dominant transient\n  (a padded batch of 8 x 31.4k tokens would be ~76 GB in bf16).\n\nReporting contract\n\n* :func:`score_example_difficulty` renders the unique examples once (tqdm\n  bar), scores them in token-budgeted batches (tqdm bar) and logs a one-line\n  summary (unique / duplicate-source / empty-reference counts, elapsed);\n* :func:`filter_by_difficulty` returns :class:`DifficultyFilterStats` —\n  per-class before / scored / unscored / duplicates / easy / hard / dropped /\n  share / loss min-mean-max plus totals; the entry points log\n  ``stats.format_report_lines()``.\n\nKey decisions (per the operator):\n\n* the difficulty model is a SEPARATE model passed as a CLI flag and freed\n  from VRAM after scoring (before the training model loads);\n* the loss is the LENGTH-NORMALISED reference loss — the mean per-token\n  cross-entropy over the reference span (``mask_example``\'s SFT labels) —\n  otherwise "hard" would just mean "long";\n* the quantiles are PER-CLASS (the catalog is imbalanced: 9 SGR classes x\n  200 vs ``menon_qa`` 31 — global quantiles would wipe out whole small\n  classes); the drop count per tail is ``floor(share * n_scored)``;\n* the warm-up applies ONLY the hard tail (a data-quality guard — its\n  rehearsal mix is there for retention, not for learning); GRPO applies\n  both tails;\n* the scoring is a pure forward pass → deterministic (no artifact cache);\n* fixed-val rows and the val/test splits are NEVER scored (the measuring\n  instrument) — the callers score only the TRAIN pool;\n* a scoring failure disables the filter (a warning, never an abort).'

from __future__ import annotations

import logging
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from llm_training.schema import UnifiedExample

logger = logging.getLogger(__name__)

#: Padded-token budget of one scoring batch. The dominant scoring transient is
#: the ``[B, T, V]`` logits tensor (bf16 over the ~145k vocab ≈ 290 KB per
#: token): 16384 padded tokens ≈ 4.75 GB of logits, plus the same again for
#: the internal log-softmax of the cross-entropy — safe next to the ~28 GB
#: bf16 difficulty model on an 80 GB card. A single example longer than the
#: budget gets a batch of its own (the 31.4k-token tail ≈ 9 GB of logits).


DEFAULT_MAX_BATCH_TOKENS = 16384


@dataclass(frozen=True)
class ClassDifficultyStats:
    'Per-class difficulty-filter row\n\n    ``n_before``/``n_scored``/``n_duplicates``/``n_dropped_*`` count ROWS of\n    the input sequence (the final-turn-boost duplicates each count; they\n    share one score but are independently droppable). ``loss_*`` is computed\n    over the UNIQUE scored sources of the class (a boost copy does not\n    double-weight the mean). ``n_dropped`` is the union of both tails;\n    ``n_dropped_easy``/``n_dropped_hard`` are the raw slice sizes (they may\n    overlap when ``drop_easy + drop_hard >= 1``).'

    task_class: str
    n_before: int
    n_scored: int
    n_unscored: int
    n_duplicates: int
    n_dropped_easy: int
    n_dropped_hard: int
    n_dropped: int
    loss_min: float | None
    loss_mean: float | None
    loss_max: float | None
    skip_reason: str | None

    @property
    def n_after(self) -> int:
        'Rows of the class after the filter'
        return self.n_before - self.n_dropped

    @property
    def dropped_share(self) -> float:
        'Dropped share of the class input'
        return self.n_dropped / self.n_before if self.n_before else 0.0


@dataclass(frozen=True)
class DifficultyFilterStats:
    'Aggregate difficulty-filter report\n\n    ``enabled`` is False only when both drop shares are <= 0 (the caller\n    never reaches that branch: the entry points guard it). ``n_tail_overlap``\n    exposes the pathological ``drop_easy + drop_hard >= 1`` case where the\n    easy and hard slices cover the same rows.'

    enabled: bool
    drop_easy: float
    drop_hard: float
    min_class_size: int
    n_input: int
    n_kept: int
    n_scored: int
    n_unscored: int
    n_duplicates: int
    n_dropped: int
    n_dropped_easy: int
    n_dropped_hard: int
    n_tail_overlap: int
    n_classes: int
    n_classes_skipped_small: int
    n_classes_unscored: int
    loss_min: float | None
    loss_mean: float | None
    loss_max: float | None
    classes: dict[str, ClassDifficultyStats]

    @property
    def dropped_share(self) -> float:
        'Dropped share of the whole input'
        return self.n_dropped / self.n_input if self.n_input else 0.0

    @property
    def loss_summary(self) -> str:
        '``"min/mean/max"`` or ``"-"``'
        return _fmt_loss(self.loss_min, self.loss_mean, self.loss_max)

    @property
    def dropped_per_class(self) -> dict[str, int]:
        '``{class: n_dropped}`` for classes with drops'
        return {
            cls: st.n_dropped for cls, st in self.classes.items() if st.n_dropped
        }

    def format_report_lines(self) -> list[str]:
        'The per-class report table'
        if not self.enabled:
            return [
                "disabled (drop_easy=%.3f and drop_hard=%.3f) — no examples "
                "dropped / фильтр выключен" % (self.drop_easy, self.drop_hard)
            ]
        header = (
            "%-28s %7s %7s %8s %5s %5s %5s %7s %6s  %s"
            % (
                "class", "before", "scored", "unscored", "dups", "easy",
                "hard", "dropped", "share", "loss min/mean/max",
            )
        )
        lines = [
            "per class (drop_easy=%.3f, drop_hard=%.3f, min_class_size=%d):"
            % (self.drop_easy, self.drop_hard, self.min_class_size),
            header,
        ]
        for cls in sorted(self.classes):
            st = self.classes[cls]
            lines.append(
                "%-28s %7d %7d %8d %5d %5d %5d %7d %5.1f%%  %s"
                % (
                    st.task_class, st.n_before, st.n_scored, st.n_unscored,
                    st.n_duplicates, st.n_dropped_easy, st.n_dropped_hard,
                    st.n_dropped, 100.0 * st.dropped_share,
                    _fmt_loss(st.loss_min, st.loss_mean, st.loss_max),
                )
            )
        lines.append(
            "%-28s %7d %7d %8d %5d %5d %5d %7d %5.1f%%  %s"
            % (
                "TOTAL", self.n_input, self.n_scored, self.n_unscored,
                self.n_duplicates, self.n_dropped_easy, self.n_dropped_hard,
                self.n_dropped, 100.0 * self.dropped_share, self.loss_summary,
            )
        )
        if self.n_tail_overlap:
            lines.append(
                "tails overlap by %d row(s) (drop_easy + drop_hard >= 1)"
                % self.n_tail_overlap
            )
        skipped = [
            cls for cls in sorted(self.classes)
            if self.classes[cls].skip_reason == "below_min_class_size"
        ]
        if skipped:
            lines.append(
                "skipped below min_class_size=%d: %s"
                % (
                    self.min_class_size,
                    ", ".join(
                        "%s (%d scored)" % (cls, self.classes[cls].n_scored)
                        for cls in skipped
                    ),
                )
            )
        unscored = [
            cls for cls in sorted(self.classes)
            if self.classes[cls].skip_reason == "unscored"
        ]
        if unscored:
            lines.append("unscored classes: %s" % ", ".join(unscored))
        return lines

    def format_report(self) -> str:
        'The whole report as one string'
        return "\n".join(self.format_report_lines())


def _loss_summary(
    values: Sequence[float],
) -> tuple[float | None, float | None, float | None]:
    '``(min, mean, max)`` or ``(None, None, None)``'
    if not values:
        return None, None, None
    return min(values), math.fsum(values) / len(values), max(values)


def _fmt_loss(
    loss_min: float | None, loss_mean: float | None, loss_max: float | None
) -> str:
    'Format a loss summary for the report'
    if loss_min is None or loss_mean is None or loss_max is None:
        return "-"
    return "%.3f/%.3f/%.3f" % (loss_min, loss_mean, loss_max)


def _resolve_scoring_device(device: str | None, cuda_available: bool) -> str:
    'Resolve the difficulty-scoring device\n\n    An explicit ``device`` wins; otherwise CUDA when available, else CPU.'
    if device:
        return device
    return "cuda" if cuda_available else "cpu"


def _resolve_scoring_attn(attn: str, device: str) -> str:
    'Flash-attention is CUDA-only\n\n    On CPU the requested ``flash_attention_2`` is replaced by ``sdpa`` (the\n    crash that silently disabled the r5 warm-up filter: the difficulty model\n    stayed on CPU while the flag asked for FA2).'
    if device.startswith("cpu") and attn == "flash_attention_2":
        logger.warning('flash_attention_2 is CUDA-only; using sdpa for the CPU difficulty scoring.')
        return "sdpa"
    return attn


def load_difficulty_model(
    model_id: str,
    *,
    attn: str = "sdpa",
    device: str | None = None,
) -> tuple[Any, Any]:
    'Load the difficulty model ON the scoring device\n\n    Returns ``(model, tokenizer)``. The model is bf16, ``eval()``, and MOVED\n    to the resolved device (CUDA when available) — scoring on CPU by accident\n    was the r5 warm-up failure. ``flash_attention_2`` falls back to ``sdpa``\n    on CPU. The caller frees the model from VRAM after scoring.'
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = _resolve_scoring_device(device, torch.cuda.is_available())
    resolved_attn = _resolve_scoring_attn(attn, resolved)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation=resolved_attn,
    )
    model.to(resolved)
    model.eval()
    logger.info(
        "Difficulty model %s on %s (attn=%s, bf16).",
        model_id, resolved, resolved_attn,
    )
    return model, tokenizer


def _plan_batches(
    lengths: Sequence[int],
    *,
    batch_size: int,
    max_batch_tokens: int,
) -> list[list[int]]:
    'Greedy token-budget batches\n\n    A batch is closed when adding the next example would exceed either the\n    example-count cap (``batch_size``) or the PADDED-token budget\n    (``(k + 1) * max(lengths)``, since the batch is padded to its longest\n    sequence). An example longer than the budget gets a batch of its own.\nDeterministic (input order).'
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for i, length in enumerate(lengths):
        new_max = max(current_max, length)
        if current and (
            len(current) + 1 > batch_size
            or (len(current) + 1) * new_max > max_batch_tokens
        ):
            batches.append(current)
            current = []
            new_max = length
        current.append(i)
        current_max = new_max
    if current:
        batches.append(current)
    return batches


def _log_scoring_summary(
    *,
    n_input: int,
    n_unique: int,
    n_duplicates: int,
    n_empty_ref: int,
    n_scored: int,
    elapsed_s: float,
) -> None:
    'One-line scoring summary'
    if n_scored == 0:
        logger.warning('Difficulty scoring: no example produced a score (%d input, %d unique, %d duplicate source(s), %d empty-reference); the filter will be a no-op.', n_input, n_unique, n_duplicates, n_empty_ref)
        return
    logger.info(
        "Difficulty scoring: scored %d of %d unique examples (%d duplicate "
        "source(s), %d empty-reference) in %.1fs.",
        n_scored, n_unique, n_duplicates, n_empty_ref, elapsed_s,
    )


def score_example_difficulty(
    model: Any,
    tokenizer: Any,
    examples: Sequence[UnifiedExample],
    *,
    batch_size: int = 8,
    max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS,
    progress: bool = True,
    desc: str = "Difficulty scoring",
) -> dict[str, float]:
    'Return ``{source: mean_reference_token_loss}`` for the unique examples.\n\n\n    Batched teacher-forced forward over ``prompt_ids + reference_ids`` (the\n    SFT labels from :func:`llm_training.masking.mask_example`), cross-entropy\n    summed over the reference positions, divided by the reference length.\n    Duplicate ``source`` ids (the final-turn boost) are scored once. The\n    forward runs with ``use_cache=False`` and in token-budgeted batches (see\n    :func:`_plan_batches`); two tqdm bars report the rendering and the\n    scoring progress (``progress=False`` silences them).\n    ``use_cache=False``'
    import numpy as np
    import torch
    from tqdm import tqdm

    from llm_training.masking import mask_example

    device = model.device
    unique: list[UnifiedExample] = []
    seen: set[str] = set()
    for ex in examples:
        if ex.source in seen:
            continue
        seen.add(ex.source)
        unique.append(ex)

    n_input = len(examples)
    n_duplicates = n_input - len(unique)
    n_empty_ref = 0
    started = time.monotonic()
    entries: list[tuple[UnifiedExample, Any, Any, int]] = []

    with tqdm(
        total=len(unique),
        desc=f"{desc}: rendering",
        unit="ex",
        disable=not progress,
        file=sys.stderr,
        mininterval=10.0,
    ) as render_bar:
        for ex in unique:
            res = mask_example(ex, tokenizer)
            if len(res.reference_ids) == 0:
                n_empty_ref += 1
            else:
                entries.append(
                    (
                        ex,
                        np.asarray(
                            [*res.prompt_ids, *res.reference_ids], dtype=np.int32
                        ),
                        np.asarray(res.labels, dtype=np.int32),
                        len(res.reference_ids),
                    )
                )
            render_bar.update(1)

    scores: dict[str, float] = {}
    model.eval()
    batches = _plan_batches(
        [len(entry[1]) for entry in entries],
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
    )
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.no_grad():
        with tqdm(
            total=len(entries),
            desc=desc,
            unit="ex",
            disable=not progress,
            file=sys.stderr,
            mininterval=10.0,
        ) as score_bar:
            for batch_idx in batches:
                batch = [entries[i] for i in batch_idx]
                max_len = max(len(entry[1]) for entry in batch)
                padded_ids = torch.full(
                    (len(batch), max_len), pad, dtype=torch.long, device=device
                )
                padded_labels = torch.full(
                    (len(batch), max_len), -100, dtype=torch.long, device=device
                )
                for row, (_, ids, labels, _) in enumerate(batch):
                    n = len(ids)
                    padded_ids[row, max_len - n :] = torch.as_tensor(
                        ids, dtype=torch.long, device=device
                    )
                    padded_labels[row, max_len - n :] = torch.as_tensor(
                        labels, dtype=torch.long, device=device
                    )
                attention_mask = padded_ids.ne(pad).long()
                logits = model(
                    input_ids=padded_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
                # Label-shift instead of slicing the logits (no `.contiguous()`
                # copy of the [B, T, V] transient): the label at position t is
                # the token at t + 1; the last position is ignored.
                
                shift_labels = torch.full_like(padded_labels, -100)
                shift_labels[:, :-1] = padded_labels[:, 1:]
                loss_per_token = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view(shift_labels.shape)
                mask = shift_labels.ne(-100)
                per_seq = (loss_per_token.float() * mask).sum(dim=1)
                for (ex, _, _, ref_len), total in zip(batch, per_seq):
                    scores[ex.source] = float(total / ref_len)
                score_bar.update(len(batch))

    _log_scoring_summary(
        n_input=n_input,
        n_unique=len(unique),
        n_duplicates=n_duplicates,
        n_empty_ref=n_empty_ref,
        n_scored=len(scores),
        elapsed_s=time.monotonic() - started,
    )
    return scores


def filter_by_difficulty(
    examples: Sequence[UnifiedExample],
    scores: dict[str, float] | None,
    *,
    drop_easy: float = 0.0,
    drop_hard: float = 0.0,
    min_class_size: int = 4,
) -> tuple[list[UnifiedExample], DifficultyFilterStats]:
    'Drop per-class tail quantiles by the scored difficulty.\n\n\n    Within each task class, the scored examples are sorted by the loss and\n    ``floor(drop_easy * n_scored)`` lowest + ``floor(drop_hard * n_scored)``\n    highest are dropped (stable order — deterministic). Examples without a\n    score are KEPT (the scoring may have failed for them). Classes with\n    fewer than ``min_class_size`` scored examples are untouched (a tail of a\n    3-example class is meaningless). Returns ``(kept, stats)`` where\n    :class:`DifficultyFilterStats` carries the per-class before/after and\n    tail counts, the unscored/duplicate counts and the loss summary (the\n    entry points log ``stats.format_report_lines()``).\n    ``floor(drop_easy * n)``\n``min_class_size``'
    scores = scores or {}
    enabled = drop_easy > 0.0 or drop_hard > 0.0
    n_input = len(examples)

    # Per-class bookkeeping over ALL input rows (duplicates included).
    
    rows: dict[str, list[int]] = defaultdict(list)
    class_duplicates: dict[str, int] = defaultdict(int)
    scored_rows: dict[str, list[tuple[float, int]]] = defaultdict(list)
    unique_source_scores: dict[str, float] = {}
    seen_sources: set[str] = set()
    n_duplicates = 0
    for i, ex in enumerate(examples):
        if ex.source in seen_sources:
            n_duplicates += 1
            class_duplicates[ex.task_class] += 1
        else:
            seen_sources.add(ex.source)
        rows[ex.task_class].append(i)
        s = scores.get(ex.source)
        if s is not None:
            scored_rows[ex.task_class].append((s, i))
            unique_source_scores[ex.source] = s

    dropped_idx: set[int] = set()
    per_class: dict[str, ClassDifficultyStats] = {}
    n_dropped_easy = 0
    n_dropped_hard = 0
    n_tail_overlap = 0
    n_classes_skipped_small = 0
    n_classes_unscored = 0
    for cls in sorted(rows):
        n_before = len(rows[cls])
        scored = scored_rows.get(cls, [])
        n_scored = len(scored)
        n_unscored = n_before - n_scored
        skip_reason: str | None = None
        if n_scored == 0:
            skip_reason = "unscored"
            n_classes_unscored += 1
        elif n_scored < min_class_size:
            skip_reason = "below_min_class_size"
            n_classes_skipped_small += 1

        n_easy = n_hard = n_union = 0
        if enabled and skip_reason is None:
            ordered = sorted(scored, key=lambda t: (t[0], t[1]))
            n_easy_slice = math.floor(drop_easy * n_scored)
            n_hard_slice = math.floor(drop_hard * n_scored)
            easy_idx = {i for _, i in ordered[:n_easy_slice]}
            hard_idx = (
                {i for _, i in ordered[-n_hard_slice:]} if n_hard_slice else set()
            )
            drop = easy_idx | hard_idx
            n_easy, n_hard, n_union = len(easy_idx), len(hard_idx), len(drop)
            dropped_idx |= drop

        class_sources = {examples[i].source: s for s, i in scored}
        loss_min, loss_mean, loss_max = _loss_summary(
            list(class_sources.values())
        )
        per_class[cls] = ClassDifficultyStats(
            task_class=cls,
            n_before=n_before,
            n_scored=n_scored,
            n_unscored=n_unscored,
            n_duplicates=class_duplicates.get(cls, 0),
            n_dropped_easy=n_easy,
            n_dropped_hard=n_hard,
            n_dropped=n_union,
            loss_min=loss_min,
            loss_mean=loss_mean,
            loss_max=loss_max,
            skip_reason=skip_reason,
        )
        n_dropped_easy += n_easy
        n_dropped_hard += n_hard
        n_tail_overlap += n_easy + n_hard - n_union

    n_dropped = len(dropped_idx)
    n_scored_rows = sum(len(v) for v in scored_rows.values())
    loss_min, loss_mean, loss_max = _loss_summary(
        list(unique_source_scores.values())
    )
    kept = [ex for i, ex in enumerate(examples) if i not in dropped_idx]
    stats = DifficultyFilterStats(
        enabled=enabled,
        drop_easy=drop_easy,
        drop_hard=drop_hard,
        min_class_size=min_class_size,
        n_input=n_input,
        n_kept=len(kept),
        n_scored=n_scored_rows,
        n_unscored=n_input - n_scored_rows,
        n_duplicates=n_duplicates,
        n_dropped=n_dropped,
        n_dropped_easy=n_dropped_easy,
        n_dropped_hard=n_dropped_hard,
        n_tail_overlap=n_tail_overlap,
        n_classes=len(rows),
        n_classes_skipped_small=n_classes_skipped_small,
        n_classes_unscored=n_classes_unscored,
        loss_min=loss_min,
        loss_mean=loss_mean,
        loss_max=loss_max,
        classes=per_class,
    )
    return kept, stats
