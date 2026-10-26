"Build the unified GRPO dataset: load → length-filter → route → sample.\n\n\nTwo entry points\n\n* :func:`build_full_dataset` — load + length-filter + route, keeping the FULL\n  (uncapped) per-class train pools and the frozen capped val/test hold-outs.\n* :func:`build_train_stream` — assemble the balanced train stream of exactly\n  ``train_examples`` rows from the full pools: per-class equal-share quotas\n  capped at ``max_upsample_ratio`` × pool, the deficit redistributed to the\n  abundant classes (fresh data), scarce classes up-sampled by round-robin\n  duplication, NEREL's quota split evenly across its 6 sub-configs.\n* :func:`build_unified_dataset` — the legacy capped builder (load → filter →\n  cap → split) kept for ``run_eval.py`` / ``sft_warmup.py`` and their tests.\n\nPipeline (per AGENTS.md)\n\n1. **Load** every task class via :data:`llm_training.loaders.LOADERS`.\n2. **Length-discard (no truncation):** an example is dropped if its total\n   rendered length (system + tools + all messages including the reference and\n   ``<tool_response>``) exceeds ``max_total_tokens`` (32 000), **or** its\n   reference-answer length exceeds ``max_reference_tokens`` (8 000). Examples\n   whose differential-render split is unclean (``prefix_ok=False``) are also\n   dropped.\n3. **Route** the Meno-stand fixed-val rows into the val pool (uncapped); the\n   hold-out classes into val/test **capped** to ``max_per_class`` (the anchor\n   measuring instrument); everything else into the full train pools.\n4. **Balanced stream** (``build_train_stream``): equal-share quotas capped at\n   ``max_upsample_ratio`` × pool, deficit redistribution, up-sampling; SGR\n   composition shaping (reasoning downsample → final-turn boost) applied to\n   the full pools first; NEREL split evenly across its 6 sub-configs.\n5. **Bake tools** into the system message of tool-use examples (so the TRL\n   ``prompt`` column is self-contained) when materializing HF datasets."

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from llm_training.config import DataConfig
from llm_training.loaders import LOADERS, collect_tool_names, collect_tool_schemas
from llm_training.loaders.meno_stand import CLASSES as _MENO_STAND_CLASSES
from llm_training.reward.language import detect_target_language

#: Classes whose test-split rows are routed into the val pool (the Meno-stand
#: distillation: its classes participate in BOTH pools — train rows train,
#: test rows validate). Never valid as ``--val-task``/``--test-task``.


FIXED_VAL_CLASSES: frozenset[str] = frozenset(_MENO_STAND_CLASSES)
from llm_training.loaders.sgr_agent import CLASSES as _SGR_CLASSES
from llm_training.masking import mask_example
from llm_training.schema import TASK_CLASSES, UnifiedExample, to_dataset_row
from llm_training.tokenizer_utils import prepare_messages

if TYPE_CHECKING:  # pragma: no cover
    from datasets import DatasetDict
    from transformers import PreTrainedTokenizer

#: Task classes that require network access at load time (excluded from offline
#: builds unless explicitly requested).

NETWORK_CLASSES: frozenset[str] = frozenset({"nerel_instruct"})

#: The 9 SGR-agent task classes (the only ones carrying SGR ``finalanswertool``
#: turns, hence the only ones the final-turn boost applies to).


SGR_AGENT_CLASSES: frozenset[str] = frozenset(_SGR_CLASSES)


@dataclass(frozen=True)
class BuildResult:
    "Outcome of :func:`build_unified_dataset`.\n\n\n    The three splits hold :class:`UnifiedExample` objects (tools NOT yet baked\n    into the system message — baking happens in :func:`to_dataset_dict`).\n    ``tool_names`` is the union of every tool name declared by any tool-use\n    example across the three splits, used by the reward's :func:`format_check`\n    to validate ``<tool_call>`` names.\nsystem"

    train: list[UnifiedExample] = field(default_factory=list)
    val: list[UnifiedExample] = field(default_factory=list)
    test: list[UnifiedExample] = field(default_factory=list)
    seed: int = 0
    val_task: "str | tuple[str, ...] | None" = None
    test_task: str | None = None
    #: per-class counts: {class: {loaded, kept, capped, final_turn_boost_added?}}.
    
    
    stats: dict[str, dict[str, int]] = field(default_factory=dict)
    #: union of every declared tool name across all three splits (registry for
    #: :func:`llm_training.reward.scoring.format_check`).
    
    tool_names: frozenset[str] = frozenset()
    #: ``{tool name -> parameter JSON schema}`` registry across all three
    #: splits (for :func:`llm_training.reward.schema_check.schema_check`).
    
    tool_schemas: dict[str, dict] = field(default_factory=dict)


@dataclass(frozen=True)
class FullBuildResult:
    'Outcome of :func:`build_full_dataset`: FULL (uncapped) train pools.\n\n\n    Unlike :class:`BuildResult`, the train side is returned as a per-class\n    mapping of the FULL length-filtered pools (no ``max_per_class`` cap) so the\n    balanced train stream can be assembled without throwing examples away. The\n    ``val`` / ``test`` pools are the SAME frozen, capped hold-out pools as\n    :class:`BuildResult` (the anchor measuring instrument).'

    train_pools: dict[str, list[UnifiedExample]] = field(default_factory=dict)
    val: list[UnifiedExample] = field(default_factory=list)
    test: list[UnifiedExample] = field(default_factory=list)
    seed: int = 0
    val_task: "str | tuple[str, ...] | None" = None
    test_task: str | None = None
    stats: dict[str, dict[str, int]] = field(default_factory=dict)


def _passes_length_filter(
    example: UnifiedExample,
    tokenizer: "PreTrainedTokenizer",
    cfg: DataConfig,
) -> tuple[bool, str]:
    'Length-discard rule (AGENTS.md). Returns (kept, reason).'
    res = mask_example(example, tokenizer)
    # ``prefix_ok`` may be False on a benign BPE boundary merge (e.g. a reference
    # starting with "\n" merges with the trailing newline of the assistant
    # generation marker). The differential render falls back to re-tokenizing the
    # textual suffix, so the reference_ids are still valid; only discard when no
    # reference span could be computed at all.
    # ``prefix_ok``
    
    
    if len(res.reference_ids) == 0:
        return False, "malformed_split"
    if res.total_tokens > cfg.max_total_tokens:
        return False, "total_too_long"
    # Reference-answer length: count the RENDERED reference token span (per
    # AGENTS.md, "measured on the chat_template-rendered text … the reference
    # is the full assistant-generated content"). ``res.reference_ids`` is the
    # rendered reference span from mask_example, so tool-use references are
    # measured inside their <tool_call> rendering — consistent with the
    # total-length branch above.
    
    
    if len(res.reference_ids) > cfg.max_reference_tokens:
        return False, "reference_too_long"
    return True, "kept"


def _boost_final_turns(
    examples: list[UnifiedExample], share: float
) -> tuple[list[UnifiedExample], int]:
    "Top up SGR final-answer turns to ``share`` of the pool (deterministic).\n\n\n    SGR trajectories end with exactly one ``finalanswertool`` turn, so after\n    the per-turn split finals are a small minority of a class pool. This\n    appends round-robin duplicates of the final turns until they reach the\n    closed-form target ``F' = ceil(share * (N - F) / (1 - share))`` (N = pool\n    size, F = finals), which guarantees finals ≥ ``share`` of the RESULTING\n    pool — no RNG, no reordering of the originals; duplicates get fresh\n    sampled completions in GRPO every epoch, so repeated prompts are extra\n    groups, not dead weight. Classes already at or above the share (e.g. the\n    single-final-turn classes) are returned unchanged."
    finals = [ex for ex in examples if (ex.extra or {}).get("is_final_answer")]
    if not finals:
        return examples, 0
    n = len(examples)
    target = math.ceil(share * (n - len(finals)) / (1.0 - share))
    need = target - len(finals)
    if need <= 0:
        return examples, 0
    base, rem = divmod(need, len(finals))
    duplicates: list[UnifiedExample] = []
    for i, ex in enumerate(finals):
        copies = base + (1 if i < rem else 0)
        duplicates.extend([ex] * copies)
    return [*examples, *duplicates], need


def _downsample_reasoning(
    examples: list[UnifiedExample], keep_share: float
) -> tuple[list[UnifiedExample], int]:
    'Keep only ``keep_share`` of the SGR reasoning turns (deterministic).\n\n\n    The mirror of :func:`_boost_final_turns`: the r3 round established that\n    reasoning turns carry no quality gradient (the structural markers are\n    saturated, judge rubrics measured flat ~0.48), yet they consume the GRPO\n    generation budget like any other prompt. This DROPS the deterministic\n    prefix ``floor(keep_share * N_reasoning)`` — no RNG, stable across\n    re-runs — keeping every non-reasoning example. ``keep_share=1.0`` is a\n    no-op. Returns ``(new list, number dropped)``.\nno-op.'
    reasoning_idx = [
        i for i, ex in enumerate(examples)
        if (ex.extra or {}).get("is_reasoning_turn")
    ]
    n_reasoning = len(reasoning_idx)
    keep = math.floor(keep_share * n_reasoning)
    drop = n_reasoning - keep
    if drop <= 0:
        return examples, 0
    dropped = set(reasoning_idx[:drop])
    return [ex for i, ex in enumerate(examples) if i not in dropped], drop


def _cap_pool(
    examples: list[UnifiedExample], max_per_class: int, seed: int, cls: str
) -> list[UnifiedExample]:
    'Seeded per-class cap (used for the frozen hold-out pools and the legacy\n    ``build_unified_dataset`` train side).'
    if len(examples) <= max_per_class:
        return examples
    return random.Random(f"{seed}:{cls}").sample(examples, max_per_class)


def _normalize_val_holdouts(
    val_task: "str | Sequence[str] | None",
) -> tuple[str, ...]:
    'Normalize the val hold-out spec into a tuple of distinct class names.\n\n\n    ``val_task`` may be a single class (the legacy form) or a sequence of\n    classes (the merged-validation form — every listed class is held out into\n    the SAME val pool, so the anchor validation becomes statistically more\n    robust instead of splitting a separate test pool that nothing scores).\n    Each class is validated against the catalog and the fixed-val guard.\n    ``val_task``'
    if val_task is None:
        return ()
    items = (val_task,) if isinstance(val_task, str) else tuple(val_task)
    for task in items:
        if task not in TASK_CLASSES:
            raise ValueError(
                f'--val-task={task!r} is not a known task class (one of {sorted(TASK_CLASSES)}).\n--val-task={task!r}'
            )
        if task in FIXED_VAL_CLASSES:
            raise ValueError(
                f'--val-task={task!r} is a Meno-stand class with fixed val rows (its test split already joins the val pool); it cannot be a whole-class hold-out.\n--val-task={task!r}'
            )
    return items


def _resolve_classes(
    val_task: "str | Sequence[str] | None",
    test_task: str | None,
    task_classes: Iterable[str] | None,
    include_network: bool,
) -> list[str]:
    'Validate the hold-outs and resolve the (sorted) class list.'
    val_holdouts = _normalize_val_holdouts(val_task)
    if test_task is not None and test_task in val_holdouts:
        raise ValueError(
            '--test-task must not also be a --val-task hold-out.\n--test-task'
        )

    # Validate the held-out test class against the catalog up front: a typo
    # would otherwise silently produce an empty split (all examples → train) —
    # a hard-to-detect data leak. (The val hold-outs are already validated in
    # _normalize_val_holdouts.)
    
    
    
    if test_task is not None:
        if test_task not in TASK_CLASSES:
            raise ValueError(
                f'--test-task={test_task!r} is not a known task class (one of {sorted(TASK_CLASSES)}).\n--test-task={test_task!r}'
            )
        if test_task in FIXED_VAL_CLASSES:
            raise ValueError(
                f'--test-task={test_task!r} is a Meno-stand class with fixed val rows (its test split already joins the val pool); it cannot be a whole-class hold-out.\n--test-task={test_task!r}'
            )

    if task_classes is None:
        classes = [
            c
            for c in LOADERS.keys()
            if include_network or c not in NETWORK_CLASSES
        ]
    else:
        # Even with an explicit task_classes list, the network gate still
        # applies: nerel_instruct is excluded unless include_network is set, so
        # the --include-network flag keeps its documented effect.
        
        
        classes = [
            c for c in task_classes
            if include_network or c not in NETWORK_CLASSES
        ]
    # Deterministic class order
    return sorted(classes)


def _load_kept_pools(
    tokenizer: "PreTrainedTokenizer",
    cfg: DataConfig,
    classes: list[str],
) -> tuple[dict[str, tuple[list[UnifiedExample], list[UnifiedExample]]],
           dict[str, dict[str, int]]]:
    'Load + length-filter every class; split fixed-val from the train part.\n\n\n    Returns ``{cls: (fixed_val_rows, train_part)}`` (NO per-class cap) and the\n    per-class stats (loaded / discarded_* / fixed_val).'
    kept: dict[str, tuple[list[UnifiedExample], list[UnifiedExample]]] = {}
    stats: dict[str, dict[str, int]] = {}
    for cls in classes:
        loader = LOADERS[cls]
        loaded = list(loader(tokenizer, cfg))
        kept_examples: list[UnifiedExample] = []
        n_total_long = n_ref_long = n_malformed = 0
        for ex in loaded:
            ok, reason = _passes_length_filter(ex, tokenizer, cfg)
            if not ok:
                if reason == "total_too_long":
                    n_total_long += 1
                elif reason == "reference_too_long":
                    n_ref_long += 1
                else:
                    n_malformed += 1
                continue
            kept_examples.append(ex)
        # Fixed-val rows (extra["fixed_val"] — the Meno-stand test split):
        # routed into the VAL pool; the train part is what the cap / stream
        # sampling operates on.
        # Fixed-val
        
        fixed_val_rows = [ex for ex in kept_examples if (ex.extra or {}).get("fixed_val")]
        train_part = [ex for ex in kept_examples if not (ex.extra or {}).get("fixed_val")]
        kept[cls] = (fixed_val_rows, train_part)
        stats[cls] = {
            "loaded": len(loaded),
            "fixed_val": len(fixed_val_rows),
            "discarded_total_too_long": n_total_long,
            "discarded_reference_too_long": n_ref_long,
            "discarded_malformed": n_malformed,
        }
    return kept, stats


def _detect_languages(examples: list[UnifiedExample]) -> None:
    'Fill ``extra["target_language"]`` in place (no-op when lingua is absent).\n\nno-op).'
    for i, ex in enumerate(examples):
        lang = detect_target_language(ex.prompt, ex.task_class, ex.extra)
        if lang is not None:
            examples[i] = replace(ex, extra={**ex.extra, "target_language": lang})


def build_full_dataset(
    tokenizer: "PreTrainedTokenizer",
    cfg: DataConfig,
    val_task: "str | Sequence[str] | None" = None,
    test_task: str | None = None,
    *,
    task_classes: Iterable[str] | None = None,
    include_network: bool = False,
) -> FullBuildResult:
    'Build the FULL (uncapped) train pools + the frozen capped val/test.\n\n\n    The train side keeps every length-filtered example — no per-class cap —\n    so the balanced train stream can be assembled without throwing examples\n    away. The hold-out classes are capped to ``cfg.max_per_class`` exactly as\n    before, and the Meno-stand fixed-val rows join the val pool uncapped: the\n    anchor measuring instrument is byte-identical to\n    :func:`build_unified_dataset`. ``val_task`` may be a single class or a\n    sequence — every listed class is held out into the SAME val pool (the\n    merged-validation form; a separate test pool is kept only for the legacy\n    callers that still pass ``test_task``).\n    :func:`build_unified_dataset`. ``val_task``'
    classes = _resolve_classes(val_task, test_task, task_classes, include_network)
    val_holdouts = set(_normalize_val_holdouts(val_task))
    kept, stats = _load_kept_pools(tokenizer, cfg, classes)

    train_pools: dict[str, list[UnifiedExample]] = {}
    val: list[UnifiedExample] = []
    test: list[UnifiedExample] = []
    for cls in classes:
        fixed_val_rows, train_part = kept[cls]
        if fixed_val_rows:
            val.extend(fixed_val_rows)
        if cls in val_holdouts:
            capped = _cap_pool(train_part, cfg.max_per_class, cfg.seed, cls)
            val.extend(capped)
            stats[cls]["kept"] = len(fixed_val_rows) + len(capped)
        elif cls == test_task:
            capped = _cap_pool(train_part, cfg.max_per_class, cfg.seed, cls)
            test.extend(capped)
            stats[cls]["kept"] = len(fixed_val_rows) + len(capped)
        else:
            train_pools[cls] = train_part
            stats[cls]["kept"] = len(fixed_val_rows) + len(train_part)

    _detect_languages(val)
    _detect_languages(test)

    return FullBuildResult(
        train_pools=train_pools,
        val=val,
        test=test,
        seed=cfg.seed,
        val_task=_normalize_val_holdouts(val_task),
        test_task=test_task,
        stats=stats,
    )


def build_unified_dataset(
    tokenizer: "PreTrainedTokenizer",
    cfg: DataConfig,
    val_task: "str | Sequence[str] | None" = None,
    test_task: str | None = None,
    *,
    task_classes: Iterable[str] | None = None,
    include_network: bool = False,
) -> BuildResult:
    'Build the unified GRPO dataset (load → filter → cap → split).\n\n\n    This is the CAPPED builder kept for ``run_eval.py`` / ``sft_warmup.py``\n    (a bounded, fixed eval pool) and their tests. ``grpo_train.py`` uses\n    :func:`build_full_dataset` + :func:`build_train_stream` instead so the\n    balanced stream loses no examples.\n``sft_warmup.py``\n\n    Args:\n        tokenizer: the Meno-Lite tokenizer (for length accounting & baking).\n        cfg: build configuration (seed, caps, thresholds).\n        val_task: task class(es) held out for validation (a single class or a\n            sequence — every listed class goes into the val split).\n        test_task: task class held out for testing (whole class → test split).\n        task_classes: restrict to these classes (default: all classes except\n            network ones, unless ``include_network``).\n        include_network: also build network-only classes (e.g. ``nerel_instruct``).'
    full = build_full_dataset(
        tokenizer, cfg, val_task, test_task,
        task_classes=task_classes, include_network=include_network,
    )
    stats = full.stats
    train: list[UnifiedExample] = []
    for cls in sorted(full.train_pools):
        examples = _cap_pool(full.train_pools[cls], cfg.max_per_class, cfg.seed, cls)
        stats[cls]["kept"] = stats[cls]["fixed_val"] + len(examples)
        # Reasoning downsample — TRAINABLE SGR classes only (the same gating as
        # the final-turn boost: the hold-out pools are the measuring
        # instrument). Runs BEFORE the boost so the boost's closed-form target
        # operates on the already-thinned pool.
        
        if (
            cfg.reasoning_keep_share is not None
            and cfg.reasoning_keep_share < 1.0
            and cls in SGR_AGENT_CLASSES
        ):
            examples, dropped = _downsample_reasoning(
                examples, cfg.reasoning_keep_share
            )
            stats[cls]["reasoning_downsampled"] = dropped
        # Final-turn boost — TRAINABLE SGR classes only, never the held-out
        # ones (the anchor pool's composition is the measuring instrument).
        
        if cfg.final_turn_share is not None and cls in SGR_AGENT_CLASSES:
            examples, added = _boost_final_turns(
                examples, cfg.final_turn_share
            )
            stats[cls]["final_turn_boost_added"] = added
        train.extend(examples)

    _detect_languages(train)

    return BuildResult(
        train=train,
        val=full.val,
        test=full.test,
        seed=cfg.seed,
        val_task=_normalize_val_holdouts(val_task),
        test_task=test_task,
        stats=stats,
        tool_names=collect_tool_names([*train, *full.val, *full.test]),
        tool_schemas=collect_tool_schemas([*train, *full.val, *full.test]),
    )


def _balanced_allocations(
    pool_sizes: dict[str, int],
    train_examples: int,
    max_upsample_ratio: float,
) -> dict[str, int]:
    'Integer per-class allocations summing to ``train_examples`` (deterministic).\n\n\n    ``base = min(desired, ratio * pool)``; the deficit (classes too small to\n    reach the equal share) is redistributed among the classes with fresh\n    (unduplicated) examples left, proportionally to that slack; the final\n    integerization uses the largest-remainder method (deterministic tie-break\n    by class name).\n    ``base = min'
    if not pool_sizes:
        return {}
    n = len(pool_sizes)
    desired = train_examples / n
    base: dict[str, float] = {}
    for cls, size in pool_sizes.items():
        base[cls] = min(desired, max_upsample_ratio * size)
    slack = {cls: max(0.0, size - base[cls]) for cls, size in pool_sizes.items()}
    total_slack = sum(slack.values())
    deficit = train_examples - sum(base.values())
    if deficit > 1e-9 and total_slack > 1e-9:
        for cls in pool_sizes:
            base[cls] += deficit * slack[cls] / total_slack
    floors = {cls: math.floor(base[cls]) for cls in pool_sizes}
    rem = {cls: base[cls] - floors[cls] for cls in pool_sizes}
    total = sum(floors.values())
    for cls in sorted(pool_sizes, key=lambda c: (-rem[c], c)):
        if total >= train_examples:
            break
        floors[cls] += 1
        total += 1
    return floors


def _draw_pool(
    pool: list[UnifiedExample], allocated: int, seed: int, cls: str
) -> tuple[list[UnifiedExample], int]:
    'Draw ``allocated`` examples: fresh seeded sample or pool + up-sampling.'
    if len(pool) >= allocated:
        return random.Random(f"{seed}:stream:{cls}").sample(pool, allocated), 0
    out = list(pool)
    extra = allocated - len(pool)
    for i in range(extra):
        out.append(pool[i % len(pool)])
    return out, extra


def _draw_nerel(
    pool: list[UnifiedExample], allocated: int, seed: int
) -> tuple[list[UnifiedExample], int]:
    'Draw ``allocated`` NEREL examples split evenly across the 6 sub-configs.\n\n    ``entity_recognition``'
    by_config: dict[str, list[UnifiedExample]] = defaultdict(list)
    for ex in pool:
        by_config[(ex.extra or {}).get("config") or ""].append(ex)
    configs = sorted(by_config)
    per_config = _balanced_allocations(
        {c: len(by_config[c]) for c in configs}, allocated, max_upsample_ratio=1.0
    )
    out: list[UnifiedExample] = []
    total_extra = 0
    for c in configs:
        sub, extra = _draw_pool(by_config[c], per_config[c], seed, f"nerel:{c}")
        out.extend(sub)
        total_extra += extra
    return out, total_extra


def build_train_stream(
    train_pools: dict[str, list[UnifiedExample]],
    *,
    train_examples: int,
    max_upsample_ratio: float,
    seed: int,
    final_turn_share: float | None = None,
    reasoning_keep_share: float | None = None,
) -> tuple[list[UnifiedExample], dict[str, Any]]:
    "Assemble the balanced train stream of ``train_examples`` rows.\n\n\n    1. SGR composition shaping (reasoning downsample → final-turn boost) is\n       applied to each class's FULL pool first.\n    2. Per-class allocations target the equal share, capped at\n       ``max_upsample_ratio`` × pool, with the deficit redistributed to the\n       classes that still have fresh (unduplicated) examples — scarce classes\n       are topped up only up to the ratio, abundant classes absorb the rest\n       with fresh data.\n    3. Each class is drawn (fresh seeded sample; short pools up-sampled by\n       round-robin duplication); NEREL's quota is split evenly across its 6\n       sub-configs."
    if max_upsample_ratio < 1:
        raise ValueError(
            f'max_upsample_ratio must be >= 1, got {max_upsample_ratio}'
        )

    classes = sorted(train_pools)
    pools: dict[str, list[UnifiedExample]] = {}
    for cls in classes:
        pool = list(train_pools[cls])
        if (
            reasoning_keep_share is not None
            and reasoning_keep_share < 1.0
            and cls in SGR_AGENT_CLASSES
        ):
            pool, _ = _downsample_reasoning(pool, reasoning_keep_share)
        if final_turn_share is not None and cls in SGR_AGENT_CLASSES:
            pool, _ = _boost_final_turns(pool, final_turn_share)
        pools[cls] = pool

    allocations = _balanced_allocations(
        {cls: len(pools[cls]) for cls in classes},
        train_examples, max_upsample_ratio,
    )

    stream: list[UnifiedExample] = []
    per_class: dict[str, dict[str, int]] = {}
    for cls in classes:
        allocated = allocations[cls]
        if cls == "nerel_instruct":
            drawn, extra = _draw_nerel(pools[cls], allocated, seed)
        else:
            drawn, extra = _draw_pool(pools[cls], allocated, seed, cls)
        stream.extend(drawn)
        per_class[cls] = {
            "pool": len(pools[cls]),
            "allocated": allocated,
            "up_sampled": extra,
        }

    _detect_languages(stream)

    stats: dict[str, Any] = {
        "train_examples": train_examples,
        "actual_total": sum(allocations.values()),
        "n_classes": len(classes),
        "max_upsample_ratio": max_upsample_ratio,
        "per_class": per_class,
    }
    return stream, stats


def bake_example(
    example: UnifiedExample, tokenizer: "PreTrainedTokenizer"
) -> UnifiedExample:
    'Return a copy of ``example`` with its tool schemas baked into the system.\n\n\n    For non-tool examples this is a no-op. For tool-use examples the leading\n    system message content is replaced with the template-exact baked version so\n    the TRL ``prompt`` column is self-contained.'
    if not example.tools:
        return example
    prompt_msgs, _ = prepare_messages(example, tokenizer)
    return replace(example, prompt=prompt_msgs)


def to_dataset_dict(result: BuildResult, tokenizer: "PreTrainedTokenizer") -> "DatasetDict":
    'Materialize a :class:`BuildResult` into a TRL-ready ``DatasetDict``.\n\n\n    Each example is baked (tools → system) and flattened via\n    :func:`to_dataset_row`. The resulting columns include the conversational\n    ``prompt`` (which TRL renders) plus ``task_class`` / ``reference`` (which\n    reach the reward function as kwargs).'
    from datasets import Dataset, DatasetDict

    def _rows(examples: list[UnifiedExample]) -> dict[str, list]:
        rows = [to_dataset_row(bake_example(ex, tokenizer)) for ex in examples]
        if not rows:
            keys = ["prompt", "task_class", "reference", "tools",
                    "reference_messages", "source", "extra"]
            return {k: [] for k in keys}
        return {k: [r[k] for r in rows] for k in rows[0].keys()}

    return DatasetDict(
        {
            "train": Dataset.from_dict(_rows(result.train)),
            "validation": Dataset.from_dict(_rows(result.val)),
            "test": Dataset.from_dict(_rows(result.test)),
        }
    )
