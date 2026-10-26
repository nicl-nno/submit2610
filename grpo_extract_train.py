'GRPO-extraction training entry — label-free extraction skills on an\nunlabeled corpus (the new GRPO line of the SFT(OPLoRA) → GRPO pivot).\n\n\nPipeline\n\n1. **Data**: the corpus built by ``scripts/build_extraction_corpus.py``\n   (``train.jsonl``; one-stage + stage-1 rows, later stage-2 rows from the\n   ``stage2`` subcommand) and the FROZEN anchor pool (``val.jsonl`` annotated\n   once with judge pseudo-gold by ``scripts/annotate_pseudo_gold.py``).\n   Rows carry the message-list prompt and the ``extra["extraction"]`` payload\n   (stage / language / source text / ontology schema / provided entities).\n2. **In-loop reward** — the deterministic label-free check\n   (:func:`llm_training.reward.extraction.extraction_check`): json envelope,\n   ontology membership + DOMAIN->RANGE signatures, stemmed entity grounding,\n   relation closure, optional spaCy normalization, and the anti-collapse\n   CONSENSUS coverage computed over each GRPO group (the reward is\n   group-aware: TRL submits ``num_generations`` consecutive completions per\n   prompt, so the group is sliced inside the reward function — no judge, no\n   embeddings, no network in the hot loop). NOTE: the descriptions-fluency\n   component of the design is NOT wired in-loop in v1 (the deterministic\n   components + the anchor-side pseudo-gold cover the gradient; wiring the\n   grammar classifier in is a Phase-2 task).\n3. **Mixed-CUTS** (:mod:`llm_training.runtime.mixed_cuts`): half of every\n   group\'s rollouts keep standard sampling, half sample uniformly from the\n   top-K — restoring intra-group advantage variance on the saturated\n   (post-SFT) prompts. Escape: ``--no-cuts``. CAVEAT: keep\n   ``num_iterations=1`` (the default) — see the module docstring.\n4. **Anchors** (``--anchor-every N``): greedy generation over the frozen val\n   pool on the TRAINING model, scored by the same deterministic check PLUS\n   the cached judge pseudo-gold (``pseudo_gold_score`` — the ctrl-point\n   recall). The ``val`` measure = per-example harmonic(extraction combined,\n   pseudo-gold combined), averaged. Checkpoint selection ``best-<step>/``\n   (through :func:`llm_training.runtime.checkpointing.sharded_checkpoint_saves`\n   — bf16 hard-check + pinned shard size) and early stop after\n   ``--anchor-patience``. The **drift probe** (the committed\n   ``src_data/drift_probe.jsonl``) runs alongside every anchor as a save GATE\n   (``--drift-probe-gate``) — the anti-forgetting instrument of this line:\n   the extraction-only stream has no rehearsal classes in v1, so the probe\n   plus KL (``--beta``) are the guards. A step-0 baseline anchor seeds\n   ``best_val`` (escape ``--no-baseline-anchor``).\n5. **Last save**: ``last/`` after training (the same sharded policy).\n\nthe train_slurm.sh\nmemory budget)::\n\n    accelerate launch grpo_extract_train.py --model /path/to/sft-oplora-merged         --corpus src_data/extraction/train.jsonl         --val src_data/extraction/val_annotated.jsonl         --output-dir checkpoints/grpo-extract --seed 24         --num-generations 16 --per-device-batch 8 --grad-accum 16         --cuts-k 32 --cuts-share 0.5 --anchor-every 50 --anchor-patience 5'

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Callable

from llm_training.extraction.qa_prompts import QA_TASK_CLASS
from llm_training.reward.summarization import (
    SUMMARY_TASK_CLASSES,
    summary_reference_semsim,
)

logger = logging.getLogger(__name__)

#: Where the committed drift-probe set lives (the frozen instrument).

_DRIFT_PROBE_PATH = Path(__file__).resolve().parent / "src_data" / "drift_probe.jsonl"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_extraction_rows(path: str) -> list[dict[str, Any]]:
    'Read a corpus/val JSONL built by ``build_extraction_corpus.py``.\n\n    is valid iff it has a prompt AND at least one scoring payload:\n    ``extra["extraction"]`` (label-free extraction), ``extra["summary"]``\n    (consolidation) or a non-empty ``reference`` (the QA class — the ONE\n    reference-bearing one). The row KIND drives the reward dispatch.'
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    def _valid(r: dict[str, Any]) -> bool:
        if not r.get("prompt"):
            return False
        extra = r.get("extra") or {}
        if isinstance(extra.get("extraction"), dict):
            return True
        if isinstance(extra.get("summary"), dict):
            return True
        return bool(str(r.get("reference") or "").strip())

    bad = [r.get("id") for r in rows if not _valid(r)]
    if bad:
        raise SystemExit(
            f'{path}: rows without prompt/scoring payload: {bad[:5]}...\n{path}'
        )
    return rows


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    'Render a message list exactly the way the anchor path does.\n\n    ``apply_chat_template`` call (``add_generation_prompt=True``) — the\n    training strings and the anchor/eval strings stay byte-identical (the\n    main-line invariant: "the SAME strings the eval/GRPO path renders").'
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_extraction_train_stream(
    rows: list[dict[str, Any]],
    train_examples: int,
    max_upsample_ratio: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    'Balanced per-class stream of ≈ ``train_examples`` presentations.\n\n    ``build_train_stream`` semantics, adapted to corpus rows): equal\n    per-class quotas, a class larger than its quota is subsampled (fresh\n    rows only), a smaller class is taken whole and topped up by duplication\n    capped at ``max_upsample_ratio`` × pool; the deficit of the capped\n    classes is redistributed to the classes that still have fresh rows.\n    Deterministic under the seed (per-class ``seed:stream:<class>`` RNG).\n\n    Returns ``(stream_rows, stats)`` — per class: pool, fresh, duplicated,\n    total presentations.'
    if train_examples <= 0:
        return list(rows), {}
    by_class: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_class.setdefault(row["task_class"], []).append(row)
    classes = sorted(by_class)
    n_classes = len(classes)
    if n_classes == 0:
        return [], {}
    share = max(1, train_examples // n_classes)
    rngs = {
        c: random.Random(f"{seed}:stream:{c}") for c in classes
    }
    chosen: dict[str, list[dict[str, Any]]] = {}
    fresh_used: dict[str, int] = {}
    # 1) fresh rows: subsample the surplus classes, take small classes whole.
    for c in classes:
        pool = by_class[c]
        if len(pool) > share:
            chosen[c] = rngs[c].sample(pool, share)
            fresh_used[c] = share
        else:
            chosen[c] = list(pool)
            fresh_used[c] = len(pool)
    # 2) duplication for the classes below the share, capped at
    #    max_upsample_ratio × pool; fill class-by-class in the deficit order.
    def _cap(c: str) -> int:
        return min(share, max_upsample_ratio * len(by_class[c]))

    for c in classes:
        cap = _cap(c)
        pool = by_class[c]
        have = len(chosen[c])
        if have >= cap:
            continue
        add = cap - have
        dup_rng = random.Random(f"{seed}:stream-dup:{c}")
        chosen[c].extend(dup_rng.choices(pool, k=add))
    # 3) the deficit of the capped classes goes to the classes that still
    #    have fresh (unused) rows — largest remaining first.
    deficit = train_examples - sum(len(chosen[c]) for c in classes)
    if deficit > 0:
        donors = sorted(
            (c for c in classes if fresh_used[c] < len(by_class[c])),
            key=lambda c: len(by_class[c]) - fresh_used[c],
            reverse=True,
        )
        for c in donors:
            if deficit <= 0:
                break
            take = min(deficit, len(by_class[c]) - fresh_used[c])
            # additional presentations sampled from the whole pool (they may
            # coincide with earlier picks — presentations, not identity,
            # are what the stream budget counts).
            chosen[c].extend(rngs[c].sample(by_class[c], take))
            fresh_used[c] += take
            deficit -= take
    stream: list[dict[str, Any]] = []
    stats: dict[str, dict[str, int]] = {}
    for c in classes:
        # interleave: shuffle the per-class list so classes mix downstream
        rngs[c].shuffle(chosen[c])
        stream.extend(chosen[c])
        fresh = min(fresh_used[c], len(chosen[c]))
        stats[c] = {
            "pool": len(by_class[c]),
            "fresh": fresh_used[c],
            "duplicated": len(chosen[c]) - fresh_used[c],
            "total": len(chosen[c]),
        }
    rng = random.Random(f"{seed}:stream-final")
    rng.shuffle(stream)
    return stream, stats


def build_extraction_dataset(
    rows: list[dict[str, Any]], tokenizer: Any = None
) -> Any:
    "Wrap corpus rows into a ``datasets.Dataset`` for GRPOTrainer.\n\n    a ``tokenizer``, the message-list prompts are pre-rendered to TEXT (the\n    main-line convention: standard-format prompts, and the full corpus no\n    longer rides as a nested list-of-structs column); without one, the raw\n    message lists ride as before (small-corpus/test path). ``task_class``,\n    ``extra`` and (when present) ``reference`` ride along (GRPOConfig keeps\n    extra columns) and reach the reward function as kwargs — the same\n    mechanism the main GRPO line uses. ``reference`` is empty for the\n    label-free rows.\n\n    ``extra`` rides as a JSON STRING: the 8-class corpus carries per-class\n    payload shapes (``extraction`` / three ``summary`` kinds / ``qa``), and\n    pyarrow's struct unification over 500k mixed rows dies with\n    ``ArrowInvalid: cannot mix list and non-list, non-null values`` (the same\n    heterogeneous-struct trap ``package_extraction_dataset.py`` dodges). The\n    reward fn parses it back via :func:`_coerce_extra`. The TEXT prompts\n    likewise keep the total column size within pyarrow's 2 GiB int32 offset\n    limit — pair this with :func:`build_extraction_train_stream` (the full\n    515k-row corpus rendered to text is ~6 GB and would still overflow on\n    dill's ``combine_chunks``)."
    from datasets import Dataset

    prompts = (
        [_render_prompt(tokenizer, r["prompt"]) for r in rows]
        if tokenizer is not None
        else [r["prompt"] for r in rows]
    )
    return Dataset.from_dict(
        {
            "prompt": prompts,
            "task_class": [r["task_class"] for r in rows],
            "extra": [json.dumps(r["extra"], ensure_ascii=False) for r in rows],
            "reference": [r.get("reference") or "" for r in rows],
        }
    )


def _coerce_extra(extra: Any) -> dict[str, Any]:
    "Accept the dataset's JSON-string ``extra`` encoding or a plain dict.\n\n    The TRL reward fn receives dataset columns verbatim, so the JSON-string\n    encoding of :func:`build_extraction_dataset` arrives here as a string."
    if isinstance(extra, str):
        return json.loads(extra)
    return extra or {}


def make_extraction_reward_fn(
    group_size: int,
    scorer: Any | None = None,
    fluency_fn: Any | None = None,
) -> Any:
    'Build the TRL-compatible async COMPOSITE reward function.\n\n    ``task_class`` + ``extra`` (the 8-class design of this line):\n\n    * extraction rows (``extra["extraction"]``) — the deterministic label-free\n      :func:`~llm_training.reward.extraction.extraction_check`, GROUP-AWARE:\n      TRL submits ``num_generations`` CONSECUTIVE completions per prompt, so\n      the consensus ``coverage`` is sliced per group (the anti-collapse\n      component). If a call\'s completion count is not group-aligned (a\n      defensive fallback — never expected with the RepeatSampler), coverage\n      degrades to N/A and only the per-completion components score;\n    * consolidation rows (``extra["summary"]``) — the deterministic\n      :func:`~llm_training.reward.summarization.summarization_check` (group-free),\n      optionally with ``fluency_fn`` (the grammar classifier — the Phase-2\n      signal of the design; a failing call records, never zeroes);\n    * QA rows (``fineweb_qa``, the ONE reference-bearing class) — the standard\n      reference path via ``scorer``: SemSim-vs-reference + fluency\n      (:func:`llm_training.reward.dispatch.make_reward_function`\'s scorer,\n      judge-free). Without a scorer the row scores 0.0 (startup refuses a\n      QA-containing corpus in that case, see ``main``).\n\n    No judge, no network in the hot loop (embeddings/fluency are in-process).'
    from llm_training.reward.dispatch import completion_to_text
    from llm_training.reward.extraction import extraction_check
    from llm_training.reward.summarization import summarization_check

    async def extraction_reward(
        prompts: list[Any], completions: list[Any], **kwargs: Any
    ) -> list[float]:
        texts = [completion_to_text(c) for c in completions]
        extras: list[Any] = kwargs.get("extra") or [None] * len(texts)
        classes: list[Any] = kwargs.get("task_class") or [None] * len(texts)
        refs: list[Any] = kwargs.get("reference") or [None] * len(texts)
        rewards: list[float] = []
        aligned = len(texts) % max(1, group_size) == 0 and group_size >= 2
        step = group_size if aligned else 1
        for start in range(0, len(texts), step):
            group = texts[start : start + step] if aligned else None
            for j in range(start, min(start + step, len(texts))):
                tc = classes[j] or ""
                extra = _coerce_extra(extras[j])
                # 1) Consolidation rows — deterministic, group-free.
                if tc in SUMMARY_TASK_CLASSES and isinstance(
                    extra.get("summary"), dict
                ):
                    flu_val: float | None = None
                    if fluency_fn is not None:
                        try:
                            flu_val = await fluency_fn(texts[j], tc)
                        except Exception:  # never zero the reward on fluency
                            flu_val = None
                    scored_s = summarization_check(
                        tc,
                        extra,
                        texts[j],
                        fluency=(lambda _t, v=flu_val: v)
                        if flu_val is not None
                        else None,
                    )
                    rewards.append(
                        float(scored_s.combined) if scored_s else 0.0
                    )
                # 2) QA rows — the standard reference path (SemSim + fluency).
                elif tc == QA_TASK_CLASS and scorer is not None:
                    reference = str(refs[j] or "").strip()
                    question = str((extra.get("qa") or {}).get("question") or "")
                    try:
                        metrics = await scorer.score(
                            user_input=question or texts[j],
                            response=texts[j],
                            reference=reference,
                            task_class=tc,
                            extra=extra,
                        )
                        rewards.append(float(metrics.combined))
                    except Exception:  # a scorer crash must not kill training
                        rewards.append(0.0)
                # 3) Extraction rows — deterministic, GROUP-aware.
                else:
                    scored = extraction_check(
                        texts[j], tc, extra, group_responses=group
                    )
                    rewards.append(float(scored.combined) if scored else 0.0)
        return rewards

    extraction_reward.__name__ = "extraction_reward"
    return extraction_reward


class _ColocateAnchorEngine:
    'Greedy anchor generation through the colocate vLLM engine (TRL 1.10).\n\n    engine already coexists with training on the card at every step (each\n    training generation runs with it awake), so an anchor through it is\n    strictly lighter than a training step — no backward. The greedy\n    sampling override follows the Mixed-CUTS pattern (temporary instance-\n    field override, restored in ``finally``): the engine reads\n    ``temperature/top_p/top_k/min_p`` from instance state at every call.\n    Greedy in vLLM is a pure argmax — deterministic and batch-invariant.'

    _GREEDY_FIELDS: tuple[tuple[str, Any], ...] = (
        ("temperature", 0.0),
        ("top_p", 1.0),
        ("top_k", 0),
        ("min_p", 0.0),
    )

    def __init__(self, vgen: Any, tokenizer: Any, max_new_tokens: int) -> None:
        self.vgen = vgen
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self._saved: dict[str, Any] = {}
        self._saved_max: Any = None
        # The UNWRAPPED generate: Mixed-CUTS patches vgen.generate to split
        # rows into standard/CUTS halves — an anchor must measure, not
        # explore, so it bypasses the split through the saved original.
        
        
        
        self._generate = getattr(vgen, "generate_original", None) or vgen.generate

    def __enter__(self) -> "_ColocateAnchorEngine":
        self._saved = {f: getattr(self.vgen, f) for f, _ in self._GREEDY_FIELDS}
        self._saved_max = getattr(self.vgen, "max_completion_length", None)
        try:
            # Push the CURRENT training weights; if the engine is in level-2
            # sleep the generate() wake path re-pushes them itself.
            
            
            self.vgen.sync_weights()
        except Exception as exc:  # noqa: BLE001 — the wake path covers it
            logger.warning("Anchor sync_weights failed (%s) — generate() re-pushes.", exc)
        for field, value in self._GREEDY_FIELDS:
            setattr(self.vgen, field, value)
        if self._saved_max is not None:
            self.vgen.max_completion_length = self.max_new_tokens
        return self

    def __exit__(self, *exc: Any) -> bool:
        for field, value in self._saved.items():
            setattr(self.vgen, field, value)
        if self._saved_max is not None:
            self.vgen.max_completion_length = self._saved_max
        return False

    def generate_texts(
        self, prompt_texts: list[str], batch_size: int = 16
    ) -> list[str]:
        'Greedy-generate rendered prompt texts; one completion per prompt.\n\nvLLM greedy is\n        batch-invariant.'
        from llm_training.tokenizer_utils import decode_generation

        out: list[str] = []
        for start in range(0, len(prompt_texts), batch_size):
            batch = prompt_texts[start : start + batch_size]
            ids = [
                self.tokenizer(t, add_special_tokens=False)["input_ids"]
                for t in batch
            ]
            _, completions, _, _ = self._generate(ids, None, 1)
            out.extend(decode_generation(self.tokenizer, seq) for seq in completions)
        return out

    def generate_message_lists(
        self, message_lists: list[list[dict[str, str]]], batch_size: int = 16
    ) -> list[str]:
        'The agent-suite contract: message lists in, responses out.'
        prompts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in message_lists
        ]
        return self.generate_texts(prompts, batch_size=batch_size)


def _make_anchor_engine(
    trainer: Any, tokenizer: Any, max_new_tokens: int
) -> _ColocateAnchorEngine | None:
    'The colocate engine when present, else ``None`` (the HF fallback).'
    vgen = getattr(trainer, "vllm_generation", None)
    if vgen is None:
        return None
    return _ColocateAnchorEngine(vgen, tokenizer, max_new_tokens)


def _anchor_rank_world(trainer: Any) -> tuple[int, int]:
    "(rank, world_size) of the trainer's accelerator (defaults 0, 1)."
    accelerator = getattr(trainer, "accelerator", None)
    world = int(getattr(accelerator, "num_processes", 1) or 1)
    rank = int(getattr(accelerator, "process_index", 0) or 0)
    return rank, world


def _anchor_shard(n_items: int, rank: int, world_size: int) -> list[int]:
    'Deterministic interleaved shard of item indices for one rank.\n\n    over ranks covers ``range(n_items)`` exactly once.'
    return list(range(rank, n_items, world_size))


def _gather_anchor_objects(obj: Any, world_size: int) -> list[Any]:
    '``all_gather_object`` under an initialized process group; else raise.\n\n    ``all_gather_object``'
    import torch

    if world_size <= 1 or not (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    ):
        raise RuntimeError(
            f'torch.distributed is not initialized, but a multi-rank anchor gather was requested (world_size={world_size}'
        )
    gathered: list[Any] = [None] * world_size
    torch.distributed.all_gather_object(gathered, obj)
    return gathered


def run_extraction_anchor(
    model: Any,
    tokenizer: Any,
    val_rows: list[dict[str, Any]],
    max_new_tokens: int,
    qa_scorer: Any | None = None,
    summary_semsim: Callable[[str, str], float] | None = None,
    engine: _ColocateAnchorEngine | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any] | None:
    'Generate + score the frozen val pool on the training model (greedy).\n\n    Two generation backends:\n\n    * ``engine`` (the colocate vLLM engine, the fast path) — prompts are\n      sharded across ranks (deterministic interleaved split), each rank\n      generates its shard through its own engine, the completions are\n      gathered to rank 0, which scores the WHOLE pool; non-zero ranks return\n      ``None`` (rank 0 owns the breakdown/checkpoint saves);\n    * no ``engine`` — the main-line HF loop, one example at a time, KV-cache\n      temporarily re-enabled, decoded via ``decode_generation``.\n\n    Per-row dispatch by ``task_class`` + ``extra``:\n\n    * extraction rows — the per-example ``val`` measure is the strict harmonic\n      mean of the deterministic extraction ``combined`` and the cached\n      pseudo-gold F1 (``pseudo_gold_score``) — judge-free at anchor time (the\n      annotation was computed once offline);\n    * QA rows (``fineweb_qa``) — the standard reference path via\n      ``qa_scorer`` (a :class:`~llm_training.runtime.reward_service.RewardService`\n      wrapping the judge-free scorer): SemSim-vs-reference + fluency. Rows are\n      scored in ONE batch at the end of the loop (the service is batched);\n      a scorer crash maps the row to 0.0 and records, never aborts the anchor;\n    * consolidation rows (``extra["summary"]``) — the strict harmonic of the\n      deterministic ``summarization_check`` combined AND, when the judge\n      wrote a reference (``annotate_pseudo_gold.py`` summary mode) and a\n      ``summary_semsim`` clip-cosine callable is wired, the SemSim of the\n      response prose vs that reference — judge-free at anchor time either\n      way (the reference was computed once offline; without reference or\n      embedder the structural ``combined`` alone scores the row).'
    import torch

    from llm_training.reward.extraction import (
        extraction_check,
        parse_extraction_output,
        pseudo_gold_score,
    )
    from llm_training.reward.scoring import harmonic_mean
    from llm_training.reward.summarization import summarization_check
    from llm_training.runtime.reward_service import ScoreRequest
    from llm_training.tokenizer_utils import decode_generation

    # QA rows are generated in-loop but SCORED in one batch at the end (the
    # service is batched + concurrency-limited; the embedder is in-process).
    
    qa_pending: list[tuple[int, dict[str, Any], str, str]] = []
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    vals: dict[int, float] = {}
    def _score_row(idx: int, row: dict[str, Any], response: str) -> None:
        'The per-row scoring dispatch shared by both generation backends.'
        extra = row.get("extra") or {}
        tc = row.get("task_class") or ""
        active: list[float] = []
        # 1) Consolidation rows — deterministic, judge-free; the
        #    cached judge-reference SemSim joins when wired.
        
        
        if tc in SUMMARY_TASK_CLASSES and isinstance(
            extra.get("summary"), dict
        ):
            scored_s = summarization_check(tc, extra, response)
            if scored_s is not None:
                sums["summary"] = sums.get("summary", 0.0) + scored_s.combined
                counts["summary"] = counts.get("summary", 0) + 1
                active.append(scored_s.combined)
                if summary_semsim is not None:
                    semsim = summary_reference_semsim(
                        summary_semsim, extra["summary"], response
                    )
                    if semsim is not None:
                        active.append(semsim)
                        sums["summary_semsim"] = (
                            sums.get("summary_semsim", 0.0) + semsim
                        )
                        counts["summary_semsim"] = (
                            counts.get("summary_semsim", 0) + 1
                        )
        # 2) QA rows — batched reference scoring at the end.
        elif tc == QA_TASK_CLASS:
            qa_pending.append(
                (idx, row, tc, response)
            )
            return  # scored below in one service batch
        # 3) Extraction rows — deterministic + cached pseudo-gold.
        else:
            payload = extra.get("extraction") or {}
            scored = extraction_check(response, tc, extra)
            pg = pseudo_gold_score(
                parse_extraction_output(response),
                payload.get("pseudo_gold"),
                payload.get("language"),
                payload.get("text"),
            )
            if scored is not None:
                active.append(scored.combined)
                for key in (
                    "json_score", "schema_score", "grounding", "closure",
                ):
                    value = getattr(scored, key)
                    if value is not None:
                        sums[key] = sums.get(key, 0.0) + value
                        counts[key] = counts.get(key, 0) + 1
            if pg is not None and pg.combined is not None:
                active.append(pg.combined)
                sums["pseudo_gold"] = sums.get("pseudo_gold", 0.0) + pg.combined
                counts["pseudo_gold"] = counts.get("pseudo_gold", 0) + 1
        vals[idx] = harmonic_mean(active)

    prompt_texts = [
        tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        for row in val_rows
    ]

    if engine is not None:
        # Fast path: the colocate engine, sharded across ranks (every rank
        # participates in the gather; rank 0 scores the whole pool).
        
        
        shard = _anchor_shard(len(val_rows), rank, world_size)
        with engine:
            shard_responses = engine.generate_texts(
                [prompt_texts[i] for i in shard]
            )
        if world_size > 1:
            gathered = _gather_anchor_objects(shard_responses, world_size)
            responses: list[Any] = [None] * len(val_rows)
            for r_ in range(world_size):
                for pos, i in enumerate(_anchor_shard(len(val_rows), r_, world_size)):
                    responses[i] = gathered[r_][pos]
        else:
            responses = shard_responses
        if rank != 0:
            return None  # rank 0 owns the scoring/breakdown/saves
        for idx, (row, response) in enumerate(zip(val_rows, responses)):
            _score_row(idx, row, response)
    else:
        was_training = model.training
        model.eval()
        prev_use_cache = getattr(model.config, "use_cache", None)
        model.config.use_cache = True
        try:
            with torch.no_grad():
                for idx, row in enumerate(val_rows):
                    inputs = tokenizer(
                        prompt_texts[idx],
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).to(model.device)
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                        or tokenizer.eos_token_id,
                    )
                    response = decode_generation(
                        tokenizer, out[0][inputs["input_ids"].shape[1]:]
                    )
                    _score_row(idx, row, response)
        finally:
            if prev_use_cache is not None:
                model.config.use_cache = prev_use_cache
            if was_training:
                model.train()
    # Batched QA scoring: SemSim-vs-reference + fluency, one service call.
    
    if qa_pending and qa_scorer is not None:
        requests = [
            ScoreRequest(
                prompt=row["prompt"],
                completion=response,
                task_class=tc,
                reference=str(row.get("reference") or ""),
                extra=row.get("extra") or {},
                source=str(row.get("id") or ""),
            )
            for (_idx, row, tc, response) in qa_pending
        ]
        try:
            metrics_list = qa_scorer.score_batch(requests)
        except Exception as exc:  # pragma: no cover - defensive, never abort
            logger.warning("Anchor QA scorer crash: %s (rows → 0.0)", exc)
            metrics_list = [None] * len(requests)
        for (idx, _row, _tc, _response), metrics in zip(
            qa_pending, metrics_list
        ):
            combined = float(metrics.combined) if metrics is not None else 0.0
            sums["qa"] = sums.get("qa", 0.0) + combined
            counts["qa"] = counts.get("qa", 0) + 1
            vals[idx] = combined
    elif qa_pending:
        # No scorer wired but the pool has QA rows: they measure 0.0 (visible
        # in the breakdown). Startup refuses this pairing — defensive only.
        
        logger.warning(
            "%d QA anchor rows scored 0.0 (no qa_scorer wired).", len(qa_pending)
        )
        for (idx, _row, _tc, _response) in qa_pending:
            vals[idx] = 0.0
    ordered = [vals[i] for i in sorted(vals)]
    breakdown = {"val": (sum(ordered) / len(ordered)) if ordered else 0.0, "n": len(ordered)}
    for key, total in sums.items():
        breakdown[key] = total / max(1, counts[key])
    return breakdown


def make_extraction_anchor_callback(
    *,
    tokenizer: Any,
    val_rows: list[dict[str, Any]],
    anchor_every: int,
    patience: int,
    anchor_max_new_tokens: int,
    output_dir: Path,
    max_shard_size: str,
    probe_items: list[Any] | None,
    probe_gate_tol: float,
    probe_warn_tol: float,
    baseline_val: float | None = None,
    qa_scorer: Any | None = None,
    summary_semsim: Callable[[str, str], float] | None = None,
    agent_scorer: Any | None = None,
) -> tuple[Any, dict[str, Any], Any]:
    'Build the ctrl-point callback (best-save + drift-probe gate + early stop).\n\n    OPLoRA lesson: ``CallbackHandler.call_event`` dispatches every event via\n    ``getattr`` — a duck-typed class dies with ``AttributeError``).'
    from transformers import TrainerCallback

    from llm_training.runtime.agent_suite import selection_key
    from llm_training.runtime.checkpointing import sharded_checkpoint_saves
    from llm_training.runtime.drift_probe import run_drift_probe

    state: dict[str, Any] = {
        "best_val": baseline_val if baseline_val is not None else -1.0,
        "best_key": baseline_val if baseline_val is not None else -1.0,
        "no_improve": 0,
        "probe_baseline": None,
        "anchor_runs": 0,
    }

    def _run_probe(trainer: Any) -> dict[str, Any] | None:
        if not probe_items:
            return None
        try:
            return run_drift_probe(
                trainer.model, tokenizer, probe_items, max_new_tokens=256
            )
        except Exception as exc:  # noqa: BLE001 - observability, never fatal
            logger.warning("Drift probe failed: %s: %s", type(exc).__name__, exc)
            return None

    def _anchor(trainer: Any, step: int, *, is_baseline: bool = False) -> None:
        engine = _make_anchor_engine(trainer, tokenizer, anchor_max_new_tokens)
        rank, world = _anchor_rank_world(trainer)
        agent_pair: tuple[Any, Any] | None = None
        # The context manager syncs the CURRENT weights into the engine and
        # overrides the sampling to greedy for the whole anchor (restored on
        # exit). Both gathers (val pool, agent suite) are entered by EVERY
        # rank; only rank 0 proceeds to scoring/saves.
        
        
        
        
        with engine or contextlib.nullcontext():
            breakdown = run_extraction_anchor(
                trainer.model, tokenizer, val_rows, anchor_max_new_tokens,
                qa_scorer=qa_scorer,
                summary_semsim=summary_semsim,
                engine=engine, rank=rank, world_size=world,
            )
            if agent_scorer is not None:
                gen_fn = (
                    engine.generate_message_lists
                    if engine is not None
                    else None
                )
                try:
                    agent_pair = agent_scorer.evaluate(
                        trainer.model, tokenizer,
                        generate_batch_fn=gen_fn, rank=rank, world_size=world,
                    )
                except Exception as exc:  # noqa: BLE001 - measurement failure
                    logger.warning(
                        "agent-suite rollout failed: %s: %s — the selection key "
                        "falls back to val for this anchor (a measurement "
                        "failure is not degradation).",
                        type(exc).__name__, exc,
                    )
        if breakdown is None:
            return  # non-zero rank: rank 0 owns the breakdown/saves
        agent_breakdown = None
        if agent_pair is not None:
            agent_breakdown, _agent_records = agent_pair
        agent_val = (
            agent_breakdown["agent_val"] if agent_breakdown is not None else None
        )
        key, agent_used = selection_key(
            breakdown["val"],
            agent_val,
            agent_breakdown["n_scored"] if agent_breakdown is not None else 0,
            agent_scorer.n_tasks if agent_scorer is not None else 0,
        )
        if agent_breakdown is not None and not agent_used:
            logger.warning(
                "agent-suite coverage below the floor (%d/%d scored) — the "
                "selection key falls back to val for this anchor.",
                agent_breakdown["n_scored"], agent_scorer.n_tasks,
            )
        probe = _run_probe(trainer)
        state["anchor_runs"] += 1
        probe_txt = (
            f" probe={probe['score']:.4f}" if probe is not None else ""
        )
        agent_txt = f" agent_val={agent_val:.4f}" if agent_val is not None else ""
        tag = " (baseline)" if is_baseline else ""
        logger.info(
            "ANCHOR step=%d%s: val=%.4f agent_val=%s key=%.4f json=%.4f "
            "grounding=%.4f closure=%.4f pg=%.4f ssim=%.4f%s",
            step, tag, breakdown["val"],
            f"{agent_val:.4f}" if agent_val is not None else "n/a",
            key, breakdown.get("json_score", 0.0),
            breakdown.get("grounding", 0.0), breakdown.get("closure", 0.0),
            breakdown.get("pseudo_gold", 0.0),
            breakdown.get("summary_semsim", 0.0), probe_txt,
        )
        print(
            f"ANCHOR step={step}{tag}: val={breakdown['val']:.4f} "
            f"pg={breakdown.get('pseudo_gold', 0.0):.4f} "
            f"ssim={breakdown.get('summary_semsim', 0.0):.4f}"
            f"{agent_txt}{probe_txt}",
            flush=True,
        )
        if probe is not None:
            if state["probe_baseline"] is None:
                state["probe_baseline"] = probe["score"]
            if (
                probe_warn_tol > 0.0
                and probe["score"] < state["probe_baseline"] - probe_warn_tol
            ):
                logger.warning(
                    "PROBE step=%d: score=%.4f dropped beyond the warn "
                    "tolerance (baseline %.4f − %.3f).",
                    step, probe["score"], state["probe_baseline"], probe_warn_tol,
                )
        if is_baseline:
            if breakdown["val"] > state["best_val"]:
                state["best_val"] = breakdown["val"]
            if key > state["best_key"]:
                state["best_key"] = key
            return
        probe_ok = True
        if (
            probe_items
            and probe_gate_tol > 0.0
            and state["probe_baseline"] is not None
            and probe is not None
        ):
            probe_ok = probe["score"] >= state["probe_baseline"] - probe_gate_tol
        key_improved = key > state["best_key"]
        if key_improved and probe_ok:
            state["best_key"] = key
            if breakdown["val"] > state["best_val"]:
                state["best_val"] = breakdown["val"]
            state["no_improve"] = 0
            best_dir = output_dir / f"best-{step}"
            logger.info(
                "New best key=%.4f (val=%.4f%s) → saving %s",
                key, breakdown["val"],
                f", agent_val={agent_val:.4f}" if agent_val is not None else "",
                best_dir,
            )
            with sharded_checkpoint_saves(trainer, max_shard_size):
                trainer.save_model(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))
        else:
            if key_improved and not probe_ok and probe is not None:
                logger.warning(
                    "Anchor step %d: key improved (%.4f > %.4f) but the "
                    "DRIFT-PROBE GATE blocked the best-save (probe %.4f < "
                    "baseline %.4f − tol %.3f); counted as no_improve.",
                    step, key, state["best_key"], probe["score"],
                    state["probe_baseline"], probe_gate_tol,
                )
            state["no_improve"] += 1
            if state["no_improve"] >= patience:
                logger.info(
                    "Early stop: %d anchors without key improvement.",
                    state["no_improve"],
                )
                trainer.control.should_training_stop = True

    class _ExtractionAnchorCallback(TrainerCallback):
        def on_step_end(self, args, state_, control, trainer=None, **kwargs) -> None:
            if anchor_every <= 0 or state_.global_step % anchor_every != 0:
                return
            if state_.global_step == 0:
                return
            _anchor(trainer, state_.global_step)

    return _ExtractionAnchorCallback(), state, _anchor


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'GRPO training of extraction skills on an unlabeled corpus (deterministic label-free reward, Mixed-CUTS, pseudo-gold anchors).'
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True, help='train.jsonl')
    parser.add_argument("--val", required=True, help='val_annotated.jsonl')
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=24)
    # --- GRPO knobs ---
    parser.add_argument("--num-generations", type=int, default=16)
    parser.add_argument("--per-device-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.001,
                        help="KL coefficient vs the reference policy (the plan pins 0.001; raise for stronger forgetting protection)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit")
    parser.add_argument("--attn", default="flash_attention_2")
    parser.add_argument("--liger", action="store_true", default=True)
    parser.add_argument("--no-liger", dest="liger", action="store_false")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--max-shard-size", default="5GB")
    # --- Mixed-CUTS ---
    parser.add_argument("--cuts-k", type=int, default=32)
    parser.add_argument("--cuts-share", type=float, default=0.5)
    parser.add_argument(
        "--cuts-temperature", type=float, default=2.0,
        help='CUTS sampling temperature; vLLM 0.27+ validates it to [0, 2]',
    )
    parser.add_argument("--no-cuts", dest="cuts", action="store_false", default=True)
    # --- Anchors / probe ---
    parser.add_argument("--anchor-every", type=int, default=50)
    parser.add_argument("--anchor-patience", type=int, default=5)
    parser.add_argument("--anchor-max-new-tokens", type=int, default=4096)
    parser.add_argument("--no-baseline-anchor", action="store_true")
    # --- Agent suite (the FOURTH validation element; spec sections 9/11) ---
    parser.add_argument(
        "--agent-suite", default=None,
        help=(
            'Directory of the frozen agent suite (suite.jsonl + index/) — enables agent_val in the anchor selection key (harmonic(val, agent_val)).'
        ),
    )
    parser.add_argument(
        "--agent-suite-embed-model", default="BAAI/bge-m3",
        help='Local query encoder (in-process; queries only).',
    )
    parser.add_argument(
        "--agent-suite-embed-device", default="cuda",
        help='Encoder device.',
    )
    parser.add_argument(
        "--agent-suite-max-batch", type=int, default=8,
        help='Mini-batch of the lockstep rollout.',
    )
    parser.add_argument(
        "--agent-suite-turns", type=int, default=4,
        help='Turn cap of the policy rollout.',
    )
    parser.add_argument(
        "--agent-suite-max-new-tokens", type=int, default=2048,
        help='Per-turn generation cap.',
    )
    parser.add_argument(
        "--agent-suite-recall-k", type=int, default=10,
        help='Retrieval recall window.',
    )
    parser.add_argument("--drift-probe", default=str(_DRIFT_PROBE_PATH))
    parser.add_argument("--no-drift-probe", action="store_true")
    parser.add_argument("--drift-probe-gate", type=float, default=0.10)
    parser.add_argument("--drift-probe-warn", type=float, default=0.05)
    # --- In-process scorer (QA SemSim + consolidation fluency; no judge) ---
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-concurrency", type=int, default=4)
    parser.add_argument("--no-fluency", action="store_true",
                        help="Disable the grammar-classifier fluency signal.")
    # --- Balanced train stream (the main-line convention) ---
    
    parser.add_argument("--train-examples", type=int, default=50000,
                        help='Balanced-stream size (prompt presentations); 0 = train over the whole corpus.')
    parser.add_argument("--max-upsample-ratio", type=int, default=8,
                        help='Per-class presentation cap, × pool.')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _seed_everything(args.seed)
    logger.info("Seed: %d (log it to re-play the run).", args.seed)

    train_rows = load_extraction_rows(args.corpus)
    val_rows = load_extraction_rows(args.val)
    logger.info(
        "Corpus: %d train rows, %d anchor rows (frozen pool).",
        len(train_rows), len(val_rows),
    )

    import torch
    from datasets import disable_progress_bar  # keep logs terse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    disable_progress_bar()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # The balanced stream caps the in-memory dataset (pyarrow's 2 GiB column
    # offset limit) AND balances class exposure — the main-line convention.
    
    
    if args.train_examples > 0:
        train_rows, stream_stats = build_extraction_train_stream(
            train_rows, args.train_examples, args.max_upsample_ratio, args.seed
        )
        for c in sorted(stream_stats):
            s = stream_stats[c]
            logger.info(
                "Stream %-26s pool=%-7d fresh=%-7d dup=%-6d total=%d",
                c, s["pool"], s["fresh"], s["duplicated"], s["total"],
            )
        logger.info(
            "Stream: %d presentations requested, %d built.",
            args.train_examples, len(train_rows),
        )
    dataset = build_extraction_dataset(train_rows, tokenizer=tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,  # the shared checkpoint policy: explicit bf16
        attn_implementation=args.attn,
    )

    from llm_training.runtime.generation_config import unset_greedy_sampling_flags

    unset_greedy_sampling_flags(model)

    from grpo_train import _build_grpo_kwargs

    config_kwargs = _build_grpo_kwargs(
        GRPOConfig,
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        lr=args.lr,
        epochs=args.epochs,
        per_device_batch=args.per_device_batch,
        grad_accum=args.grad_accum,
        optim=args.optim,
        seed=args.seed,
        max_steps=args.max_steps,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_server_host="127.0.0.1",
        vllm_server_port=8998,
        liger=args.liger,
    )
    qa_rows = [r for r in train_rows + val_rows if r["task_class"] == QA_TASK_CLASS]
    summary_rows = [
        r for r in train_rows + val_rows
        if isinstance((r.get("extra") or {}).get("summary"), dict)
    ]
    reward_scorer = None
    qa_service = None
    fluency_fn = None
    if qa_rows or summary_rows:
        from llm_training.reward.config import RewardConfig
        from llm_training.reward.judge import make_ragas_scorer

        # The judge-free scorer: SemSim (embeddings) + fluency for QA rows,
        # plus the fluency signal for consolidation rows. No judge (the hot
        # loop stays network-free).
        reward_cfg = RewardConfig(
            judge=None,
            compute_answer_correctness=False,
            local_embedding_model=args.embedding_model,
            local_embedding_device=args.embedding_device,
            enable_fluency=not args.no_fluency,
            fluency_classes=frozenset(
                {QA_TASK_CLASS} | set(SUMMARY_TASK_CLASSES)
            ),
            # The extraction corpus has no NEREL prose sub-configs — the
            # DEFAULT_FLUENCY_SUBTASKS key would be a dead config (the
            # RewardConfig guard rejects it).
            
            # fluency_subtasks
            
            fluency_subtasks={},
        )
        reward_scorer = make_ragas_scorer(reward_cfg)
        if summary_rows and not args.no_fluency:
            from llm_training.reward.fluency import make_fluency_scorer

            fluency_scorer = make_fluency_scorer(reward_cfg)

            async def fluency_fn(text: str, task_class: str) -> float | None:
                return await fluency_scorer.score(text, task_class)
        if qa_rows:
            from llm_training.runtime.reward_service import RewardService

            qa_service = RewardService(
                reward_scorer, concurrency=args.embedding_concurrency
            )
    summary_semsim = None
    if summary_rows:
        from llm_training.reward.summarization import (
            SUMMARY_REFERENCE_KEY,
            make_summary_semsim,
        )

        # The judge wrote reference summaries OFFLINE (annotate_pseudo_gold.py
        # summary mode); the anchor scores clipped-cosine SemSim of the
        # response prose vs the cached reference through the SAME in-process
        # embedder the QA scorer already holds — no second model copy, no
        # judge call at anchor time.
        
        
        has_reference = any(
            isinstance(
                ((r.get("extra") or {}).get("summary") or {}).get(
                    SUMMARY_REFERENCE_KEY
                ),
                dict,
            )
            for r in summary_rows
        )
        if has_reference:
            summary_semsim = make_summary_semsim(reward_scorer.base_embeddings)
            if summary_semsim is None:
                logger.warning(
                    "Judge summary references found, but the scorer exposes "
                    "no sync embedder — anchors fall back to the structural "
                    "summarization combined (no reference SemSim)."
                )
    reward_fn = make_extraction_reward_fn(
        args.num_generations,
        scorer=qa_service if qa_service is not None else reward_scorer,
        fluency_fn=fluency_fn,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=GRPOConfig(**config_kwargs),
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    if args.cuts:
        from llm_training.runtime.mixed_cuts import apply_mixed_cuts

        info = apply_mixed_cuts(
            trainer,
            cuts_k=args.cuts_k,
            cuts_share=args.cuts_share,
            cuts_temperature=args.cuts_temperature,
        )
        logger.info(info.describe())

    probe_items = None
    if not args.no_drift_probe and Path(args.drift_probe).exists():
        from llm_training.runtime.drift_probe import load_probe

        probe_items = load_probe(args.drift_probe)
        logger.info("Drift probe: %d items.", len(probe_items))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_scorer = None
    if args.agent_suite:
        from llm_training.runtime.agent_suite import AgentSuiteScorer

        agent_scorer = AgentSuiteScorer(
            suite_dir=args.agent_suite,
            corpus_paths=[args.corpus, args.val],
            local_embed_model=args.agent_suite_embed_model,
            embed_device=args.agent_suite_embed_device,
            max_turns=args.agent_suite_turns,
            mini_batch=args.agent_suite_max_batch,
            max_new_tokens=args.agent_suite_max_new_tokens,
            recall_k=args.agent_suite_recall_k,
        )
        logger.info(
            "Agent suite: %d tasks, index digest %s (agent_val joins the "
            "selection key: harmonic(val, agent_val)).",
            agent_scorer.n_tasks, agent_scorer.index_digest,
        )
    callback, state, run_anchor = make_extraction_anchor_callback(
        tokenizer=tokenizer,
        val_rows=val_rows,
        anchor_every=args.anchor_every,
        patience=args.anchor_patience,
        anchor_max_new_tokens=args.anchor_max_new_tokens,
        output_dir=output_dir,
        max_shard_size=args.max_shard_size,
        probe_items=probe_items,
        probe_gate_tol=args.drift_probe_gate,
        probe_warn_tol=args.drift_probe_warn,
        qa_scorer=qa_service,
        summary_semsim=summary_semsim,
        agent_scorer=agent_scorer,
    )
    trainer.add_callback(callback)

    if args.anchor_every > 0 and not args.no_baseline_anchor:
        run_anchor(trainer, 0, is_baseline=True)

    trainer.train()

    from llm_training.runtime.checkpointing import sharded_checkpoint_saves

    last_dir = output_dir / "last"
    with sharded_checkpoint_saves(trainer, args.max_shard_size):
        trainer.save_model(str(last_dir))
    tokenizer.save_pretrained(str(last_dir))
    logger.info(
        "Done. best_key=%.4f best_val=%.4f anchors_run=%d → %s",
        state["best_key"], state["best_val"], state["anchor_runs"], last_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
