'Stratified SGR-only validation split for the SFT warm-up.\n\n\nDesign (fixed with the operator):\n\n* Only the 9 **SGR-agent** classes are split into train/val — the validation\n  must measure exactly what the post-warm-up gate measures (the SGR\n  ``<tool_call>`` protocol), so the auxiliary envelope-diversity classes\n  (``tool_calls_multiturn``, ``opencode_gen``) go to train **in full**.\n* The split is **stratified by (task_class × turn kind)** (final / reasoning /\n  action) so every format is represented on both sides.\n* ``--val-ratio`` (default 0.1) of each stratum goes to val; the sampling is\n  seeded per stratum (``f"{seed}:{task_class}:{kind}"``) so adding an unrelated\n  class never shifts another stratum\'s sample.\n* ``exclude_tasks`` removes whole task classes (both sides) — used to mirror\n  the future GRPO ``--val-task`` / ``--test-task`` hold-outs so the warm-up\n  cannot leak into the leave-task-out generalization measurement.\n* Strata with a single example keep it in TRAIN (a 1-example stratum in val\n  would both starve training and give an all-or-nothing val point).'

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Hashable, Sequence

from llm_training.schema import UnifiedExample

#: The SGR-agent task classes whose data is split into train/val.

from llm_training.loaders.sgr_agent import CLASSES as SGR_CLASSES


def turn_kind(example: UnifiedExample) -> str:
    "Classify an SGR example's turn kind from its loader metadata."
    e = example.extra or {}
    if e.get("is_final_answer"):
        return "final"
    if e.get("is_reasoning_turn"):
        return "reasoning"
    if e.get("is_call_turn"):
        return "action"
    return "plain"


@dataclass
class WarmupSplit:
    'The result of splitting warm-up examples into train / val.\n\nval.'

    train: list[UnifiedExample] = field(default_factory=list)
    val: list[UnifiedExample] = field(default_factory=list)
    excluded_examples: int = 0
    excluded_classes: list[str] = field(default_factory=list)
    strata_stats: dict[str, dict[str, int]] = field(default_factory=dict)


def split_warmup_examples(
    examples: Sequence[UnifiedExample],
    *,
    seed: int,
    val_ratio: float = 0.1,
    exclude_tasks: Sequence[str] = (),
) -> WarmupSplit:
    'Split warm-up examples: stratified val from SGR classes; the rest → train.\n\n\nArgs\n        examples: the built unified examples (all included classes, already\n            capped/filtered by ``build_unified_dataset``).\n        seed: RNG seed (logged by the caller); each stratum samples with a\n            deterministic per-stratum seed derived from it.\n        val_ratio: fraction of each SGR stratum to hold out for validation\n        exclude_tasks: task classes to remove entirely (mirroring the future\n            GRPO hold-outs'
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(
            f'val_ratio must be in (0, 1), got {val_ratio}'
        )
    out = WarmupSplit(excluded_classes=sorted(set(exclude_tasks)))

    # Strata: (task_class, kind) -> examples, SGR classes only.
    
    strata: dict[tuple[str, str], list[UnifiedExample]] = defaultdict(list)
    for ex in examples:
        if ex.task_class in exclude_tasks:
            out.excluded_examples += 1
            continue
        if ex.task_class in SGR_CLASSES:
            strata[(ex.task_class, turn_kind(ex))].append(ex)
        else:
            # Auxiliary (non-SGR) classes: envelope-diversity data → train only.
            
            out.train.append(ex)

    for (cls, kind), items in sorted(strata.items()):
        n_val = round(len(items) * val_ratio)
        if len(items) > 1 and n_val > 0:
            rng = random.Random(f"{seed}:{cls}:{kind}")
            val_idx = set(rng.sample(range(len(items)), n_val))
        else:
            val_idx = set()  # single-example strata stay in train
        for i, ex in enumerate(items):
            (out.val if i in val_idx else out.train).append(ex)
        out.strata_stats[f"{cls}:{kind}"] = {
            "total": len(items),
            "val": len(val_idx),
        }
    return out
