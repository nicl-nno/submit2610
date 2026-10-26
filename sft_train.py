'SFT training over the SAME data/anchors as grpo_train.py (the ablation twin).\n\n\nPurpose\nin fine-tuning Meno-Lite, all else equal?". The script is a line-for-line twin\nof :mod:`grpo_train` — the SAME train/validation data (the leave-task-out\nhold-outs ``--val-task``, the balanced ``--train-examples`` stream, the\ndifficulty filter, the typo/SYSTEM-typo augmentation, ``--seed``), the SAME\nLLM-as-Judge anchor validation (``run_anchor_scoring`` /\n``make_judge_anchor_callback`` verbatim: the composite ``val`` measure, the\nstep-0 baseline, the drift probe + its save gate, ``best-<step>/`` selection,\nearly stop on ``--anchor-patience``), the SAME checkpoint discipline (bf16 +\n``--max-shard-size``) — with exactly one difference: the optimizer step is\nplain SFT (teacher-forced cross-entropy on the reference) instead of GRPO.\n\n**LoRA by default\nover the 145k vocab cannot fit ``--max-len 32768`` on a single 80 GB card\nwithout the liger fused CE kernel (the ``[B, T, V]`` logits transient alone is\n~28 GB at T=32768), and the target environment (the ``vllm_env`` conda stack:\ntrl 0.15.1, peft 0.14.0, transformers 4.49 — NO liger, NO flash-attn) does not\ncarry it. LoRA removes the dominant full-FT costs (gradients ~30 GB + optimizer\nstates) while keeping the whole length budget: the peak drops to ~50 GB and\n``--max-len 32768`` preserves the dataset byte-identically (the build layer\'s\n32 000-token discard rule stays the binding cap). Defaults follow the OPLoRA\npaper (arXiv:2510.13003, Table 8, validated on Qwen2.5-7B — the Meno-Lite\nancestor): ``r=64, alpha=64, dropout=0.05`` on\n``q/v/up/down/o_proj``, ``lr=2e-4``. The known CONFOUND of the ablation —\nGRPO trains all weights while its SFT twin trains adapters — is documented in\nthe README; ``--no-lora`` restores full FT for environments that carry liger\n(e.g. compute-node), at ``--max-len 16384`` or lower.\n``--max-len 32768``\n\n**OPLoRA (opt-in ``--oplora``)** constrains every adapter update to the\northogonal complement of the top-k singular subspace of the frozen weights\n(:mod:`llm_training.oplora`, arXiv:2510.13003): the top-k singular triples are\npreserved exactly, maximising the retained general ability — the paper\'s best\nforgetting resistance on Qwen2.5-7B. ``--oplora-k`` (default 128) is the\nnumber of protected directions.\n\n**Checkpoints\n``best-<step>/`` (peft\'s own dtype convention; resume-safe), and after\ntraining the adapters are MERGED into the base weights\n(``merge_and_unload``) and saved as full bfloat16 sharded models:\n``last-merged/`` (the final state) and ``best-merged/`` (reloaded from the\nwinning ``best-<step>/``, the ablation deliverable) — the downstream\n(run_eval.py / mergekit) consumes plain full models. ``--no-lora`` runs keep\nthe grpo_train.py save path verbatim.\n``last-merged/``\n\n**Judge anchors on the old ragas\nscore through the shared reward layer; on the ``vllm_env`` stack (ragas\n0.2.15) :mod:`llm_training.reward._ragas_compat` transparently adapts the\nmetric classes, so the anchor code is unchanged. NOTE: the same judge server\nscored through different ragas versions produces slightly different absolute\nAC/rubric numbers (prompt plumbing differs between ragas 0.2.x and 0.4.x) —\ncompare the ablation against GRPO qualitatively, or re-score both with\n``run_eval.py`` on one stack for exact numbers.\n\nUsage\n    accelerate launch sft_train.py --model /path/to/storage/models/meno-lite-0.2         --output-dir checkpoints/sft --val-task summarizer math_reasoner         --train-examples 50000 --max-len 32768 --seed 24\n\n    # OPLoRA (orthogonal-projection LoRA, arXiv:2510.13003):\n    accelerate launch sft_train.py ... --oplora --oplora-k 128\n\nSmoke run first\n    accelerate launch sft_train.py ... --max-steps 30 --judge-anchor-every 10'

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llm_training.augment.typo import (
    TypoConfig,
    augment_pool,
    augment_pool_system,
    resolve_classes,
    resolve_system_classes,
)
from llm_training.build import (
    FIXED_VAL_CLASSES,
    build_full_dataset,
    build_train_stream,
)
from llm_training.config import DataConfig
from llm_training.inference.generate import render_prompt
from llm_training.loaders import collect_tool_names, collect_tool_schemas
from llm_training.oplora import DEFAULT_OPLORA_K
from llm_training.prepare import load_grpo_cache
from llm_training.reward.scoring import harmonic_mean
from llm_training.runtime.anchor_generation import (
    DEFAULT_ANCHOR_GEN_BATCH_TOKENS,
    DEFAULT_ANCHOR_GEN_MAX_BATCH,
    generate_anchor_responses,
)
from llm_training.runtime.checkpointing import (
    DEFAULT_MAX_SHARD_SIZE,
    assert_bfloat16_model,
    merge_adapter_into_base,
    sharded_checkpoint_saves,
    unwrap_trainer_model,
)
from llm_training.schema import UnifiedExample
from llm_training.tokenizer_utils import prepare_messages, render
from llm_training.warmup.split import turn_kind

logger = logging.getLogger("sft_train")

#: The hold-out task classes — the SAME pair as grpo_train.py (keep mirrored
#: across all stages; both are held out into the ONE anchor val pool).


RECOMMENDED_VAL_TASKS = ("summarizer", "math_reasoner")

#: LoRA target modules — the OPLoRA paper's set for Qwen2.5-7B (q/v/up/down/o;
#: no k_proj/gate_proj — validated there; extend via --lora-target-modules).

DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "v_proj",
    "up_proj",
    "down_proj",
    "o_proj",
)

#: LoRA hyperparameter defaults (the OPLoRA paper's Table 8 for Qwen2.5-7B:
#: r=64/alpha=64 for the math configuration — the "new skills" regime; alpha
#: = r, i.e. the scaling factor 1.0, as in the paper).


DEFAULT_LORA_R = 64
DEFAULT_LORA_ALPHA = 64
DEFAULT_LORA_DROPOUT = 0.05

#: Learning-rate defaults: LoRA trains with ~20× the full-FT rate (the paper:
#: 2e-4; full FT — the sft_warmup.py regime, 1e-5).


DEFAULT_LORA_LR = 2e-4
DEFAULT_FULL_FT_LR = 1e-5


def _seed_everything(seed: int) -> None:
    'Seed every RNG (sft_train.py seeds CUDA too — no vLLM is forked here).'
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a core dep
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        from transformers import set_seed

        set_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def to_prompt_completion(
    examples: Sequence[UnifiedExample], tokenizer: Any
) -> list[dict[str, str]]:
    "Convert unified examples to TRL **standard** (pre-rendered text) rows.\n\n\n    Copied verbatim from :mod:`sft_warmup` (``to_prompt_completion``) — keep\n    the two in sync: the ablation's rendering MUST be byte-identical to the\n    warm-up's. ``prompt`` is the chat-template render with the generation\n    prompt appended (:func:`render_prompt`); ``completion`` is the reference\n    suffix — the difference between the full-conversation render and the\n    prompt render (the exact span :func:`llm_training.masking.mask_example`\n    localises). Plain strings also keep pyarrow out of the heterogeneous\n    ``tool_calls`` schema (see the sft_warmup.py docstring for the ArrowInvalid\n    landmine)."
    rows: list[dict[str, str]] = []
    for ex in examples:
        prompt_text = render_prompt(ex, tokenizer)
        _prompt_messages, full_messages = prepare_messages(ex, tokenizer)
        full_text = render(tokenizer, full_messages, add_generation_prompt=False)
        if not full_text.startswith(prompt_text):
            raise ValueError(
                f'Rendered full conversation does not start with the rendered prompt for {ex.source!r}'
            )
        rows.append(
            {
                "prompt": prompt_text,
                "completion": full_text[len(prompt_text):],
            }
        )
    return rows


def turn_kind_stats(examples: Sequence[UnifiedExample]) -> dict[str, Any]:
    'The turn-kind census of the train stream (grpo_train.py parity).'
    kinds = {"final": 0, "reasoning": 0, "action": 0, "plain": 0}
    for ex in examples:
        e = ex.extra or {}
        if e.get("is_final_answer"):
            kinds["final"] += 1
        elif e.get("is_reasoning_turn"):
            kinds["reasoning"] += 1
        elif e.get("is_call_turn"):
            kinds["action"] += 1
        else:
            kinds["plain"] += 1
    return {"n_rows": len(examples), "turn_kinds": kinds}


def _build_sft_kwargs(
    sft_config_cls: Any,
    *,
    output_dir: str,
    lr: float,
    per_device_batch: int,
    grad_accum: int,
    max_len: int,
    optim: str,
    seed: int,
    max_steps: int,
    liger: bool = False,
) -> dict[str, Any]:
    'Build SFTConfig kwargs adaptive to the installed TRL/transformers.\n\n\nVersion landmines handled here\n    * ``max_length`` / ``max_seq_length`` — set EXPLICITLY always (TRL 1.10\n      defaults to a truncating 1024); a hard error if the installed SFTConfig\n      declares neither name (silent truncation is unacceptable).\n    * ``completion_only_loss=True`` — the SFT twin must learn only the\n      reference span (the GRPO reward also sees only the completion). The\n      top-level field exists on TRL 0.14+/0.15.1/0.20/1.10; older stacks get\n      the ``dataset_kwargs`` fallback; neither name → a hard error (training\n      on the prompt tokens would silently change the semantics).\n    * ``warmup_ratio`` vs the v5 float-ratio ``warmup_steps`` fallback.\n    * ``num_train_epochs=1`` + ``max_steps`` — the training length is the\n      example budget ``--train-examples`` (the balanced stream), exactly as\n      grpo_train.py derives it; an epoch would be redundant.\n    * Unknown fields are dropped with a warning (forward compatibility).'
    from dataclasses import fields as dc_fields

    known = {f.name for f in dc_fields(sft_config_cls)}
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": 1,
        "learning_rate": lr,
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "lr_scheduler_type": "cosine",
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": 5,
        "save_strategy": "no",  # checkpoints are saved by the anchor callback
        "report_to": [],
        "seed": seed,
        "remove_unused_columns": False,
        # transformers 4.53.x cannot auto-derive label_names from a PeftModel
        # (its forward hides the base signature) — the trainer warns and uses
        # an EMPTY list. Harmless for the training loss (the model computes
        # it from the collator's `labels`; num_items_in_batch checks the
        # batch keys directly), but the explicit name silences the warning
        # and keeps the eval-machinery semantics correct.
        
        
        
        
        
        "label_names": ["labels"],
        "optim": optim,
        "weight_decay": 0.0,
        "max_length": max_len,
    }
    if max_steps > 0:
        kwargs["max_steps"] = max_steps

    # Completion-only loss is SEMANTIC: dropping it silently would train on
    # the prompt tokens and change the ablation. Field → dataset_kwargs →
    # hard error (same ladder as the length cap).
    # Completion-only loss
    
    if "completion_only_loss" in known:
        kwargs["completion_only_loss"] = True
    elif "dataset_kwargs" in known:
        kwargs["dataset_kwargs"] = {"completion_only_loss": True}
        logger.warning('SFTConfig has no completion_only_loss field; routing it through dataset_kwargs (older TRL) — verify the installed TRL honours it.')
    else:
        raise RuntimeError(
            'The installed SFTConfig declares neither completion_only_loss nor dataset_kwargs; cannot guarantee completion-only training. Refusing to run.'
        )

    # Warmup: prefer the exact ratio field; fall back to the v5 ratio-as-float.
    
    if "warmup_ratio" in known:
        kwargs["warmup_ratio"] = 0.03
    elif "warmup_steps" in known:
        kwargs["warmup_steps"] = 0.03  # v5: float < 1 == ratio of steps
    else:
        logger.warning('SFTConfig declares neither warmup_ratio nor warmup_steps; LR warmup is disabled for this run.')

    # Length cap is mandatory: TRL 1.10 defaults max_length to 1024, silently
    # truncating long prompts. Accept the legacy max_seq_length name; if
    # neither exists, refuse to run.
    
    if "max_length" not in known:
        if "max_seq_length" in known:
            kwargs["max_seq_length"] = kwargs.pop("max_length")
        else:
            raise RuntimeError(
                'The installed SFTConfig declares neither max_length nor max_seq_length; cannot guarantee the length cap (TRL would truncate to its own default). Refusing to run.'
            )

    if liger and "use_liger_kernel" in known:
        kwargs["use_liger_kernel"] = True

    # Drop anything the installed class does not know (forward compatibility).
    
    unknown = sorted(set(kwargs) - known)
    if unknown:
        logger.warning("SFTConfig dropped unsupported kwargs: %s", unknown)
        for key in unknown:
            kwargs.pop(key)
    return kwargs


def audit_anchor_val_budget(
    val_examples: Sequence[UnifiedExample],
    tokenizer: Any,
    *,
    max_new_tokens: int,
    max_positions: int | None = None,
) -> dict[str, Any]:
    'Audit the anchor val pool against the anchor generation budget.\n\n\n    Observability ONLY — the pool composition stays frozen (the anchor val\n    pool is the measuring instrument; dropping or altering examples would\n    break the anchor comparability). Two conditions are reported BEFORE the\n    first anchor, while the ANCHOR line\'s ``final_unparsed`` counter exposes\n    them only AFTER the fact:\n\n    * ``over_budget``: the reference span exceeds ``max_new_tokens`` — the\n      model cannot reproduce the turn within the budget, so a final-answer\n      turn\'s ``finalanswertool`` call gets truncated,\n      :func:`extract_final_answer` returns None and its AC is scored against\n      an EMPTY answer (AC≈0);\n    * ``over_context``: ``prompt + max_new_tokens`` exceeds the model\'s\n      position limit — :func:`run_anchor_scoring` does NOT clamp the budget\n      to the remaining context (unlike ``run_eval.py``\'s per-prompt auto\n      budget), so the whole anchor generation pass would fail.\n\n    ``max_positions=None`` (e.g. an unreadable model config) skips the\n    context branch. Logs the warnings/info line and returns the report\n    ``{"n", "max_ref_span", "max_prompt_tokens", "over_budget",\n    "over_context"}``.'
    from llm_training.masking import mask_example

    over_budget: list[tuple[UnifiedExample, int, int]] = []
    over_context: list[tuple[UnifiedExample, int]] = []
    max_ref_span = 0
    max_prompt_tokens = 0
    for ex in val_examples:
        res = mask_example(ex, tokenizer)
        ref_span = len(res.reference_ids)
        prompt_len = len(res.prompt_ids)
        max_ref_span = max(max_ref_span, ref_span)
        max_prompt_tokens = max(max_prompt_tokens, prompt_len)
        if ref_span > max_new_tokens:
            over_budget.append((ex, ref_span, prompt_len))
        if (
            max_positions is not None
            and prompt_len + max_new_tokens > max_positions
        ):
            over_context.append((ex, prompt_len))
    if over_budget:
        details = ", ".join(
            f"{ex.task_class}/{turn_kind(ex)} span={span} ({ex.source})"
            for ex, span, _ in over_budget[:5]
        )
        logger.warning(
            "Anchor val budget audit: %d of %d example(s) have a reference "
            "span longer than --anchor-max-new-tokens=%d — their "
            "finalanswertool calls will be truncated and AC scored against "
            "an EMPTY answer (final_unparsed). Raise --anchor-max-new-tokens "
            "to >= %d. Examples: %s",
            len(over_budget), len(val_examples), max_new_tokens,
            max_ref_span, details,
        )
    if over_context:
        details = ", ".join(
            f"{ex.task_class} prompt={prompt_len} ({ex.source})"
            for ex, prompt_len in over_context[:5]
        )
        logger.warning(
            "Anchor val budget audit: %d example(s) would exceed the model "
            "context during anchor generation (prompt + %d > "
            "max_position_embeddings=%s) — the whole anchor pass would "
            "fail. Examples: %s",
            len(over_context), max_new_tokens, max_positions, details,
        )
    if not over_budget and not over_context:
        logger.info(
            "Anchor val budget audit: n=%d, max ref span=%d, max prompt=%d; "
            "all fit --anchor-max-new-tokens=%d%s.",
            len(val_examples), max_ref_span, max_prompt_tokens,
            max_new_tokens,
            f" and the context {max_positions}" if max_positions else "",
        )
    return {
        "n": len(val_examples),
        "max_ref_span": max_ref_span,
        "max_prompt_tokens": max_prompt_tokens,
        "over_budget": over_budget,
        "over_context": over_context,
    }


def run_anchor_scoring(
    model: Any,
    tokenizer: Any,
    val_examples: Sequence[UnifiedExample],
    judge_scorer: Any,
    reward_service: Any,
    *,
    max_new_tokens: int,
    gen_batch_tokens: int = DEFAULT_ANCHOR_GEN_BATCH_TOKENS,
    gen_max_batch: int = DEFAULT_ANCHOR_GEN_MAX_BATCH,
) -> dict[str, float]:
    "Generate + score the anchor val pool; return the aggregate breakdown.\n\n    ``.generate``/``.eval``/``.train``\n\n    The generation runs via HF ``.generate`` on the TRAINING model (the exact\n    current weights, no vLLM weight-sync games) over LEFT-padded\n    length-bucketed packages (:func:`llm_training.runtime.anchor_generation.generate_anchor_responses`: deterministic packing under the\n    conservative KV-cache budget ``gen_batch_tokens``; ``0`` = the legacy\n    one-example-at-a-time mode) with the KV-cache temporarily re-enabled —\n    ``use_cache`` is ``False`` during training for gradient-checkpointing\n    compatibility, and cache-less greedy generation recomputes the full\n    prefix at every step (quadratic cost). Every touched state (``eval``/\n    ``train``, ``use_cache``, ``padding_side``) is restored in a ``finally``\n    block. Generations are decoded via\n    :func:`llm_training.tokenizer_utils.decode_generation` so the\n    ``<tool_call>`` envelope survives (it consists of added special tokens;\n    a ``skip_special_tokens=True`` decode deletes it and corrupts every\n    final/action turn's AC/SemSim). NOTE: batched kernels are not\n    bit-identical to single-stream ones — anchor series with packages are\n    NOT token-identical to the batch=1 series (inside one run every anchor,\n    including the step-0 baseline, packs identically).\n\n    The scoring reuses a **persistent** ``RewardService`` (see\n    :func:`make_judge_anchor_callback`): its single background event loop must\n    outlive every anchor, because the judge's ``AsyncOpenAI`` httpx client\n    binds to the first loop it serves — a fresh loop per anchor would break\n    every judge call from the second anchor on.\n\n    The returned breakdown separates REAL AC failures (``ac_failed``: the\n    judge call was attempted and failed, or the dispatcher crashed — the\n    ``failed_metrics`` marker / sentinel) from structural skips\n    (``ac_skipped``: SGR reasoning turns, where AC is not applicable by\n    design and stays ``None``), and reports ``final_unparsed``/``n_final``:\n    final-answer turns whose generation contains no valid\n    ``finalanswertool`` call — every content metric then scores an EMPTY\n    answer (AC≈0), usually because the generation budget truncated the\n    call (see ``--anchor-max-new-tokens``).\n\n    The headline selection measure is ``val``: EVERY example contributes via\n    its judge-side / structural metrics — the judge rubrics\n    (``rubric_score``, reasoning turns: Russian-language literacy +\n    consistency with the input context, scored by the same judge), AC\n    (action/final turns), ``format_check`` and ``code_check`` — combined per\n    example by the strict harmonic mean and averaged across the pool\n    (``n_val``). Semantic Similarity and the small-model fluency are\n    DELIBERATELY excluded: in grpo_train.py they already form the in-loop\n    reward, and a checkpoint-selection criterion overlapping the optimized\n    objective would lose its reward-hacking independence (kept identical here\n    for series comparability)."
    from llm_training.reward.dispatch import DISPATCHER_FAILURE
    from llm_training.reward.scoring import extract_final_answer
    from llm_training.runtime.reward_service import ScoreRequest

    generations = generate_anchor_responses(
        model,
        tokenizer,
        val_examples,
        render_fn=render_prompt,
        max_new_tokens=max_new_tokens,
        batch_tokens=gen_batch_tokens,
        max_batch=gen_max_batch,
    )

    requests: list[ScoreRequest] = []
    n_final = 0
    final_unparsed = 0
    for ex, gen in zip(val_examples, generations):
        # Final-answer turns carry the whole answer inside a
        # finalanswertool call; when extraction fails, every content
        # metric scores an EMPTY string (AC≈0). Count these
        # casualties so the ANCHOR line exposes the share instead of
        # silently biasing AC downward (usually the generation
        # budget truncated the call).
        if bool((ex.extra or {}).get("is_final_answer")):
            n_final += 1
            if extract_final_answer(gen) is None:
                final_unparsed += 1
        requests.append(
            ScoreRequest(
                prompt=ex.prompt,
                completion=gen,
                task_class=ex.task_class,
                reference=ex.reference,
                extra=dict(ex.extra),
                source=ex.source,
            )
        )

    metrics_all: list[Any] = reward_service.score_batch(requests)
    ac_vals = [m.answer_correctness for m in metrics_all if m.answer_correctness is not None]
    ss_vals = [m.semantic_similarity for m in metrics_all if m.semantic_similarity is not None]
    fl_vals = [m.fluency for m in metrics_all if m.fluency is not None]
    rb_vals = [m.rubric_score for m in metrics_all if m.rubric_score is not None]
    # AC failure visibility: a real failure is an attempted-and-failed AC (the
    # ``failed_metrics`` marker) or a dispatcher crash (the sentinel); anything
    # else that stayed None is a structural skip (e.g. SGR reasoning turns).
    ac_failed = 0
    ac_skipped = 0
    for m in metrics_all:
        if m.answer_correctness is not None:
            continue
        if (
            "answer_correctness" in m.failed_metrics
            or DISPATCHER_FAILURE in m.failed_metrics
        ):
            ac_failed += 1
        else:
            ac_skipped += 1  # structurally N/A (e.g. SGR reasoning turns)
    if ac_failed and not ac_vals:
        logger.warning(
            "Anchor: AC failed for ALL %d eligible val examples "
            "(ac=0.0 fallback); judge health is suspect.",
            ac_failed,
        )
    rubric_failed = sum(
        1 for m in metrics_all if "rubrics" in m.failed_metrics
    )
    if rubric_failed and not rb_vals:
        logger.warning(
            "Anchor: judge rubrics failed for ALL %d reasoning val examples; "
            "judge health is suspect.",
            rubric_failed,
        )
    # The ctrl-point selection measure ``val``: EVERY example contributes via
    # its judge-side / structural metrics — the judge rubrics (reasoning
    # turns), AC (action/final turns), format_check, code_check and the
    # deterministic rerank verdict grading (the Meno-stand fixed-val rows) —
    # while SemSim and the small-model fluency stay OUT (the grpo_train.py
    # reward-hacking-independence convention, kept for series comparability).
    # Strict harmonic mean per example, arithmetic mean across the pool; an
    # example whose every component failed to compute (or is not applicable)
    # drops out.
    val_vals: list[float] = []
    val_by_subtype: dict[str, list[float]] = {}
    for ex, m in zip(val_examples, metrics_all):
        parts = [
            v for v in (
                m.rubric_score,
                m.answer_correctness,
                m.format_score,
                m.code_check,
                m.rerank_em,
            )
            if v is not None
        ]
        if parts:
            val_i = harmonic_mean(parts)
            val_vals.append(val_i)
            # The per-subtype breakdown: the fixed-val Meno-stand rows are
            # reported separately from the leave-task-out hold-out rows.
            tc = ex.task_class
            subtype = tc if tc in FIXED_VAL_CLASSES else "holdout"
            val_by_subtype.setdefault(subtype, []).append(val_i)
    if n_final and final_unparsed == n_final:
        logger.warning(
            "Anchor: every final-answer turn (%d/%d) lacks a valid "
            "finalanswertool call — their AC is scored against an EMPTY "
            "answer; check the anchor generation budget "
            "(--anchor-max-new-tokens).",
            final_unparsed, n_final,
        )
    return {
        "val": sum(val_vals) / len(val_vals) if val_vals else 0.0,
        "val_by_subtype": {
            k: sum(v) / len(v)
            for k, v in sorted(val_by_subtype.items())
            if v
        },
        "ac": sum(ac_vals) / len(ac_vals) if ac_vals else 0.0,
        "rubric": sum(rb_vals) / len(rb_vals) if rb_vals else 0.0,
        "semsim": sum(ss_vals) / len(ss_vals) if ss_vals else 0.0,
        "fluency": sum(fl_vals) / len(fl_vals) if fl_vals else 0.0,
        "n": float(len(metrics_all)),
        "n_val": float(len(val_vals)),
        "ac_failed": float(ac_failed),
        "ac_skipped": float(ac_skipped),
        "rubric_failed": float(rubric_failed),
        "final_unparsed": float(final_unparsed),
        "n_final": float(n_final),
    }


def make_judge_anchor_callback(
    *,
    trainer: Any,
    tokenizer: Any,
    anchor_examples: Sequence[UnifiedExample],
    get_judge_runtime: Any,
    output_dir: Path,
    anchor_every: int,
    patience: int,
    max_new_tokens: int,
    gen_batch_tokens: int = DEFAULT_ANCHOR_GEN_BATCH_TOKENS,
    gen_max_batch: int = DEFAULT_ANCHOR_GEN_MAX_BATCH,
    max_shard_size: str | int = DEFAULT_MAX_SHARD_SIZE,
    logger_: Any = logger,
    run_anchor_scoring_fn: Any = None,
    probe_items: Sequence[Any] = (),
    probe_max_new_tokens: int = 256,
    probe_warn_drop: float = 0.05,
    probe_gate_tol: float = 0.10,
    probe_baseline: float | None = None,
    initial_best_val: float | None = None,
    run_probe_fn: Any = None,
    save_best: Any = None,
) -> Any:
    'Build the judge-anchor ctrl-point callback (the checkpoint selector).\n\n\n    Every ``anchor_every`` steps (main process only): generate the anchor val\n    pool on the training model, score it with the judge-enabled scorer, log\n    the val/AC/rubric/SemSim/fluency breakdown, save ``best-<step>/`` iff the\n    aggregate **``val``** measure improves, and stop training after\n    ``patience`` consecutive anchors without improvement. ``max_shard_size``\n    is pinned on every full-model save (:func:`llm_training.runtime.checkpointing.sharded_checkpoint_saves` — the bfloat16 check rides along).\n\n    **``save_best`` (the ONLY delta from grpo_train.py).** When ``None``\n    (default), the save is the grpo_train.py verbatim block —\n    ``sharded_checkpoint_saves`` + ``trainer.save_model`` + the tokenizer. A\n    callable ``(trainer, best_dir)`` replaces the save mechanics for the LoRA\n    path (sft_train.py saves the peft ADAPTER; the merged full model is\n    produced after training) while the selection logic — ``val``, the drift\n    gate, the patience counter — stays byte-identical.\n``save_best``\n\n    ``get_judge_runtime`` returns the ``(scorer, reward_service)`` pair\n    lazily; the service must be the SAME persistent instance across anchors\n    (one background event loop for the whole run). Any anchor failure is\n    logged and the anchor skipped — a ctrl-point problem must never abort a\n    long training run. Returns ``(callback, state)``.\n\n    **The drift-probe gate (the save POLICY, not the measure).** Alongside\n    every anchor the drift probe runs BEFORE the save decision: when\n    ``probe_items`` are given AND a baseline is known AND ``probe_gate_tol >\n    0``, a ``best-<step>/`` save additionally requires ``probe_score >=\n    probe_baseline - probe_gate_tol``. The frozen ``val`` NUMBER is never\n    blended with the probe; the probe only conditions WHICH checkpoints get\n    saved. A val-improving anchor blocked by the gate counts as\n    ``no_improve``. A FAILED probe run never vetoes a save;\n    ``probe_baseline`` normally comes from the step-0 run in ``main()`` and\n    falls back to the first successful anchor when ``None``.\n\n    **The step-0 baseline anchor.** ``initial_best_val`` seeds the selection\n    bar: without it ``best_val`` starts at -1.0 and the FIRST in-training\n    anchor always counts as an improvement and saves a ``best-<step>/``\n    checkpoint even when the model is WORSE than the starting checkpoint. No\n    checkpoint is saved for step 0 itself; a step-0 failure leaves ``None``;\n    the baseline never counts toward ``no_improve``.'
    from transformers import TrainerCallback

    if run_anchor_scoring_fn is None:
        run_anchor_scoring_fn = run_anchor_scoring  # DI seam for tests.
    if run_probe_fn is None:
        from llm_training.runtime.drift_probe import run_drift_probe

        run_probe_fn = run_drift_probe  # DI seam for tests.

    def _default_save_best(t: Any, best_dir: Path) -> None:
        with sharded_checkpoint_saves(t, max_shard_size):
            t.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))

    if save_best is None:
        save_best = _default_save_best

    state = {
        # The step-0 baseline seed: None → the legacy -1.0 start; a seeded
        # value means the pre-training anchor already ran and set the bar.
        "best_val": (
            -1.0 if initial_best_val is None else float(initial_best_val)
        ),
        "best_step": None,  # for the final log: no in-run best save yet
        "no_improve": 0,
        "probe_baseline": probe_baseline,
    }

    class _JudgeAnchorCallback(TrainerCallback):
        def on_step_end(self, targs, state_, control, **kwargs):
            if not trainer.accelerator.is_main_process:
                return
            if anchor_every <= 0:
                # --judge-anchor-every 0 disables the anchors.
                return
            if state_.global_step % anchor_every != 0:
                return
            if not anchor_examples:
                logger_.info("Anchor: no val examples; skipping.")
                return
            try:
                judge_scorer, reward_service = get_judge_runtime()
                breakdown = run_anchor_scoring_fn(
                    trainer.model, tokenizer, anchor_examples, judge_scorer,
                    reward_service, max_new_tokens=max_new_tokens,
                    gen_batch_tokens=gen_batch_tokens,
                    gen_max_batch=gen_max_batch,
                )
            except Exception as e:  # noqa: BLE001 - never abort the run
                logger_.warning(
                    "Anchor at step %d failed; skipping it: %s",
                    state_.global_step, e,
                )
                return

            # --- Drift probe — runs BEFORE the save decision (its score
            # gates the best-save below). A failed run maps to probe=None →
            # the gate passes (a measurement failure is not degradation).
            probe: dict[str, Any] | None = None
            if probe_items:
                try:
                    probe = run_probe_fn(
                        trainer.model, tokenizer, probe_items,
                        max_new_tokens=probe_max_new_tokens,
                    )
                except Exception as e:  # noqa: BLE001 - never abort the run
                    logger_.warning(
                        "Drift probe at step %d failed; skipping it: %s",
                        state_.global_step, e,
                    )
                    probe = None
                if probe is not None and state["probe_baseline"] is None:
                    # The step-0 baseline was unavailable — the first
                    # successful anchor becomes the reference point.
                    state["probe_baseline"] = probe["score"]

            logger_.info(
                "ANCHOR step=%d: val=%.4f AC=%.4f rubric=%.4f semsim=%.4f "
                "fluency=%.4f (n=%d, n_val=%d, ac_failed=%d, ac_skipped=%d, "
                "rubric_failed=%d, final_unparsed=%d/%d) "
                "val_by_subtype=%s [best_val=%.4f, no_improve=%d/%d]",
                state_.global_step, breakdown["val"], breakdown["ac"],
                breakdown["rubric"], breakdown["semsim"],
                breakdown["fluency"], int(breakdown["n"]),
                int(breakdown.get("n_val", 0)),
                int(breakdown.get("ac_failed", 0)),
                int(breakdown.get("ac_skipped", 0)),
                int(breakdown.get("rubric_failed", 0)),
                int(breakdown.get("final_unparsed", 0)),
                int(breakdown.get("n_final", 0)),
                {k: round(v, 4)
                 for k, v in breakdown.get("val_by_subtype", {}).items()},
                state["best_val"], state["no_improve"], patience,
            )

            # PROBE observability line + WARN (independent of the gate).
            if probe is not None:
                _baseline = state["probe_baseline"]
                if _baseline is not None:
                    delta = probe["score"] - _baseline
                    logger_.info(
                        "PROBE step=%d: score=%.4f (Δ%+.4f vs baseline "
                        "%.4f, n=%d) per_check=%s",
                        state_.global_step, probe["score"], delta,
                        _baseline, int(probe["n"]),
                        {k: round(v["pass_rate"], 3)
                         for k, v in probe["per_check"].items()},
                    )
                    if -delta > probe_warn_drop:
                        logger_.warning('general instruction-following is degrading while the val measure may not see it.')

            # --- The drift-probe gate: probe_ok=False blocks the save.
            # Inactive when the probe is off, the baseline is unknown, or
            # the tolerance is disabled (<=0); a failed probe run passes.
            probe_ok = True
            if (
                probe_items
                and probe_gate_tol > 0.0
                and state["probe_baseline"] is not None
                and probe is not None
            ):
                probe_ok = (
                    probe["score"]
                    >= state["probe_baseline"] - probe_gate_tol
                )

            val_improved = breakdown["val"] > state["best_val"]
            if val_improved and probe_ok:
                state["best_val"] = breakdown["val"]
                state["no_improve"] = 0
                state["best_step"] = state_.global_step
                best_dir = output_dir / f"best-{state_.global_step}"
                logger_.info(
                    "New best val=%.4f → saving %s",
                    state["best_val"], best_dir,
                )
                save_best(trainer, best_dir)
            else:
                if val_improved and not probe_ok:
                    logger_.warning('Anchor step %d: val improved (%.4f > %.4f) but the DRIFT-PROBE GATE blocked the best-save (probe %.4f < baseline %.4f − tol %.3f); counted as no_improve.', state_.global_step, breakdown['val'], state['best_val'], probe['score'], state['probe_baseline'], probe_gate_tol)
                state["no_improve"] += 1
                if state["no_improve"] >= patience:
                    logger_.info(
                        "Early stop: %d anchors without val improvement.",
                        state["no_improve"],
                    )
                    control.should_training_stop = True

    return _JudgeAnchorCallback(), state


def save_merged_model(
    merged_model: Any, tokenizer: Any, out_dir: Path, max_shard_size: str | int
) -> None:
    'Save a merged (adapter-free) model: bf16 hard-check + pinned shards.'
    merged_model.save_pretrained(str(out_dir), max_shard_size=max_shard_size)
    tokenizer.save_pretrained(str(out_dir))


def _merge_final_adapter(
    trainer: Any,
    tokenizer: Any,
    out_dir: Path,
    max_shard_size: str | int,
    oplora_projection: Any = None,
) -> Any:
    'Merge the trainer\'s LIVE LoRA adapter into the base; save full bf16.\n\n\n    ``unwrap_trainer_model`` takes the TRAINER — it reads ``trainer.model``\n    and ``trainer.accelerator.unwrap_model``. Passing ``trainer.model`` (the\n    peft wrapper) instead makes its ``getattr(model, "model")`` resolve\n    through peft\'s ``__getattr__`` chain to the BARE base causal LM (and\n    ``accelerator`` to ``None``), so the returned object has no\n    ``merge_and_unload`` — the 2026-09-26 cluster crash after a full 782-step\n    OPLoRA run. The OPLoRA repair before the merge is a no-op in practice\n    (the step-end callback repaired after the last optimizer step), kept as\n    the explicit invariant at merge time.\n\n    ``unwrap_trainer_model``\n    ``trainer.model`` (peft\n``merge_and_unload``\n    no-op (callback'
    if oplora_projection is not None:
        oplora_projection.repair_if_drifted()
    merged = unwrap_trainer_model(trainer).merge_and_unload()
    assert_bfloat16_model(merged)
    save_merged_model(merged, tokenizer, out_dir, max_shard_size)
    return merged


def _build_arg_parser() -> argparse.ArgumentParser:
    'Build the sft_train.py CLI parser (separate for testability).'
    p = argparse.ArgumentParser(
        description=(
            'SFT training over the same data/anchors as grpo_train.py (the GRPO ablation twin; LoRA by default, OPLoRA opt-in).'
        )
    )
    p.add_argument("--model", default="checkpoints/warmup/best",
                   help='The warmed checkpoint to start SFT from (the same starting point as the GRPO twin run).')
    p.add_argument("--data-dir", default="src_data",
                   help='Path to the src_data catalog.')
    p.add_argument("--prepared-dir", default=None, metavar="DIR",
                   help='Load the pre-built unified dataset from this cache directory (produced by prepare_dataset.py --pipeline grpo')
    p.add_argument("--output-dir", default="checkpoints/sft",
                   help='Output dir for best-<step>/ (adapters) and the merged last-merged/ + best-merged/ models.')
    p.add_argument("--max-shard-size", default=DEFAULT_MAX_SHARD_SIZE,
                   metavar="SIZE",
                   help='max_shard_size for saved merged checkpoints (transformers size string, e.g. 5GB; default 5GB; also hard-checks bfloat16 before every save).')
    p.add_argument("--val-task", nargs="+", default=list(RECOMMENDED_VAL_TASKS),
                   metavar="CLS",
                   help=f'Held-out validation class(es) (default: {list(RECOMMENDED_VAL_TASKS)}')
    p.add_argument("--max-per-class", type=int, default=200,
                   help='Per-class cap of the FROZEN val hold-out pools (the anchor measuring instrument).')
    p.add_argument("--sgr-loop-subdir", default=None,
                   help='subdir with the loop-format SGR catalog (reasoningtool; scripts/convert_sgr_to_loop.py); per-file preference of the SGR loader, classes without files there fall back to sgr_agent')
    p.add_argument("--include-network", action="store_true",
                   help="Include network-only classes (nerel_instruct, loaded from its 'train' split) in the train pool.")
    p.add_argument("--final-turn-share", type=float, default=None,
                   metavar="SHARE",
                   help='Minimum share of SGR final-answer turns in each TRAINABLE SGR class pool (deterministic duplication; None = off; the grpo_train.py data knob verbatim).')
    p.add_argument("--reasoning-keep-share", type=float, default=None,
                   metavar="SHARE",
                   help='Keep only SHARE of the reasoning turns in each TRAINABLE SGR class pool (deterministic prefix downsample; None = off; the grpo_train.py data knob verbatim).')
    p.add_argument("--difficulty-model", default="anonymous/base-model",
                   metavar="MODEL_ID",
                   help='The SEPARATE difficulty model whose reference loss scores the train pool; default the base Meno-Lite. Empty string disables the filter.')
    p.add_argument("--difficulty-drop-easy", type=float, default=0.05,
                   metavar="SHARE",
                   help='Per-class share of the EASIEST train examples to drop (the grpo_train.py twin knob). 0 = off.')
    p.add_argument("--difficulty-drop-hard", type=float, default=0.01,
                   metavar="SHARE",
                   help='Per-class share of the HARDEST train examples to drop (noise). 0 = off.')
    p.add_argument("--difficulty-batch-size", type=int, default=8,
                   help='Upper bound on the number of examples per difficulty scoring forward pass (the batches are also capped by a padded-token budget).')
    p.add_argument("--no-difficulty-progress", action="store_true",
                   help='Disable the tqdm progress bars of the difficulty scoring.')
    p.add_argument("--train-examples", type=int, default=0,
                   help='Total number of training examples: the balanced train stream is assembled to exactly this size (rounded up to a whole multiple of the effective batch)')
    p.add_argument("--max-upsample-ratio", type=float, default=8.0,
                   help="Maximum up-sampling multiple of a scarce class's pool when balancing the train stream (>= 1).")
    p.add_argument("--max-steps", type=int, default=0,
                   help='If >0, cap training steps (smoke runs).')
    p.add_argument("--lr", type=float, default=None, metavar="LR",
                   help=f'Learning rate (default: {DEFAULT_LORA_LR} with LoRA, {DEFAULT_FULL_FT_LR}')
    p.add_argument("--per-device-batch", type=int, default=1,
                   help='Per-device batch size (1 for the 32768-token length budget on one 80 GB card).')
    p.add_argument("--grad-accum", type=int, default=64,
                   help='Gradient accumulation steps (effective batch = per-device × accum).')
    p.add_argument("--optim", default="adamw_torch",
                   help='plain Adam states fit; --no-lora runs may want paged_adamw_8bit).')
    p.add_argument("--attn", default="sdpa",
                   choices=("sdpa", "flash_attention_2", "eager"),
                   help='Attention implementation passed to from_pretrained (sdpa for vllm_env')
    p.add_argument("--max-len", type=int, default=32768,
                   help="SFTConfig max_length (prompt+completion); 32768 keeps the whole dataset (the build layer's 32 000-token discard rule is the binding cap).")
    p.add_argument("--liger", action="store_true",
                   help='Enable Liger kernels (needs the liger-kernel package; absent in vllm_env')
    # --- LoRA / OPLoRA -----------------------------------------------------
    p.add_argument("--no-lora", action="store_true",
                   help='Disable LoRA and train ALL weights (the full-FT escape hatch; needs liger or a reduced --max-len to fit 80 GB).')
    p.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R,
                   help=f'LoRA rank (default {DEFAULT_LORA_R}')
    p.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA,
                   help=f'LoRA alpha (default {DEFAULT_LORA_ALPHA}')
    p.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT,
                   help=f'LoRA dropout (default {DEFAULT_LORA_DROPOUT}')
    p.add_argument("--lora-target-modules", nargs="+",
                   default=list(DEFAULT_LORA_TARGET_MODULES), metavar="MOD",
                   help=f'LoRA target modules (default: {list(DEFAULT_LORA_TARGET_MODULES)}')
    p.add_argument("--oplora", action="store_true",
                   help='Constrain the LoRA updates to the orthogonal complement of the top-k singular subspace of each frozen weight (OPLoRA, arXiv:2510.13003: exact preservation of the top-k singular triples')
    p.add_argument("--oplora-k", type=int, default=DEFAULT_OPLORA_K, metavar="K",
                   help=f'The number of protected top singular directions (default {DEFAULT_OPLORA_K}')
    # --- Judge anchors
    p.add_argument("--local-embedding-model", default=None,
                   help='HF id or LOCAL PATH of the in-process anchor embedder (default BAAI/bge-m3). NOTE vllm_env: torch 2.5.1 + transformers 4.53 carry the CVE-2025-32434 torch.load gate and the bge-m3 HF repo has no safetensors')
    p.add_argument("--local-embedding-device", default=None,
                   help="Device for the in-process bge-m3 embedder (the anchor scorer's SemSim). Env: LOCAL_EMBEDDING_DEVICE.")
    p.add_argument("--local-fluency-device", default=None,
                   help="Device for the in-process fluency model (the anchor scorer's fluency column). Env: LOCAL_FLUENCY_DEVICE.")
    p.add_argument("--embedding-concurrency", type=int, default=2,
                   help='Concurrent in-process embedding forwards.')
    p.add_argument("--judge-concurrency", type=int, default=8,
                   help="Max in-flight judge HTTP requests at the anchors; keep STRICTLY below the provider's limit (Z.AI GLM-5.3-Flash: 50).")
    p.add_argument("--judge-max-tokens", type=int, default=4096,
                   help='Max tokens for the judge LLM structured output (raise for thinking judges, e.g. 16384).')
    p.add_argument("--judge-timeout", type=float, default=600.0,
                   help='Per-request judge HTTP timeout, seconds (600 covers slow judges; scale with --judge-max-tokens, e.g. 1800 for 16384).')
    p.add_argument("--no-judge-preflight", action="store_true",
                   help='Skip the pre-flight judge ping (a one-message availability check run BEFORE the heavy setup; a failed ping aborts the run).')
    p.add_argument("--judge-anchor-every", type=int, default=500,
                   help='Judge anchor interval (steps).')
    p.add_argument("--anchor-max-new-tokens", type=int, default=6144,
                   help='Generation budget for anchor scoring (the grpo_train twin value: summarizer final turns carry the whole answer inside a finalanswertool call; 2048 truncated them).')
    p.add_argument("--anchor-gen-batch-tokens", type=int,
                   default=DEFAULT_ANCHOR_GEN_BATCH_TOKENS, metavar="N",
                   help='Anchor-generation batching: LEFT-padded packages of greedy HF .generate under the KV-cache budget (padded prompt + the FULL generation budget) x rows <= N tokens (~192 KiB/token for a 14B Qwen2.5-class model; default 65536 ~ 12 GB KV). Batched kernels are not bit-identical to single-stream ones')
    p.add_argument("--anchor-gen-max-batch", type=int,
                   default=DEFAULT_ANCHOR_GEN_MAX_BATCH, metavar="ROWS",
                   help='Row cap of one anchor-generation package (prefill padding waste grows with the spread; the default 8 sits on the diminishing-returns knee).')
    p.add_argument("--no-drift-probe", action="store_true",
                   help='Disable the drift probe (the synthetic instruction-following mini-set checked deterministically at every anchor).')
    p.add_argument("--drift-probe-warn", type=float, default=0.05,
                   metavar="DROP",
                   help='WARN when the drift-probe score drops more than DROP below the baseline.')
    p.add_argument("--drift-probe-gate", type=float, default=0.10,
                   metavar="TOL",
                   help='The drift-probe GATE tolerance: a best-<step>/ save additionally requires probe >= baseline')
    p.add_argument("--drift-probe-max-new-tokens", type=int, default=256,
                   help='Generation budget per drift-probe prompt.')
    p.add_argument("--anchor-patience", type=int, default=3,
                   help='Early stop after N anchors without val improvement.')
    p.add_argument("--no-baseline-anchor", action="store_true",
                   help='Skip the step-0 baseline anchor (the pre-training full-val measurement that seeds the best_val selection bar).')
    # --- Typo augmentation
    p.add_argument("--typo-share", type=float, default=0.0, metavar="SHARE",
                   help='Offline keyboard-typo augmentation of a SHARE fraction of the TRAIN pool (0 = off; the grpo_train twin knob).')
    p.add_argument("--typo-class", action="append", default=None, metavar="CLS",
                   help='Repeatable; restrict typo augmentation to these task classes.')
    p.add_argument("--no-typo-class", action="append", default=None,
                   metavar="CLS",
                   help='Repeatable; remove task classes from the typo allowlist.')
    p.add_argument("--system-typo-share", type=float, default=0.0,
                   metavar="SHARE",
                   help='Offline SYSTEM-prompt augmentation of a SHARE fraction of the TRAIN pool (0 = off).')
    p.add_argument("--system-typo-class", action="append", default=None,
                   metavar="CLS",
                   help='Repeatable; restrict SYSTEM-prompt augmentation to these classes.')
    p.add_argument("--no-system-typo-class", action="append", default=None,
                   metavar="CLS",
                   help='Repeatable; remove classes from the system-typo allowlist.')
    p.add_argument("--seed", type=int, default=42,
                   help='RNG seed (logged; grpo_train.py parity')
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    args = _build_arg_parser().parse_args(argv)
    if args.oplora and args.no_lora:
        print(
            'error: --oplora requires LoRA (--no-lora disables adapters).',
            file=sys.stderr,
        )
        return 2
    output_dir = Path(args.output_dir)
    logger.info("sft_train: seed=%d lora=%s oplora=%s(k=%d)", args.seed,
                not args.no_lora, args.oplora, args.oplora_k)
    _seed_everything(args.seed)

    # Lazy heavy imports (dev environments without the train extra stay light).
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "SFT training needs 'torch', 'transformers', 'datasets' and 'trl': pip install -e '.[train]'."
        ) from e
    if args.liger:
        try:
            import liger_kernel  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                "--liger requires the 'liger-kernel' package: pip install -U liger-kernel. Or drop the flag."
            ) from e
    use_lora = not args.no_lora
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                "LoRA requires the 'peft' package: pip install -U peft. Or pass --no-lora for full fine-tuning."
            ) from e

    import os

    lr = args.lr if args.lr is not None else (
        DEFAULT_LORA_LR if use_lora else DEFAULT_FULL_FT_LR
    )

    local_embedding_model = (
        args.local_embedding_model or os.environ.get("LOCAL_EMBEDDING_MODEL")
    )
    local_embedding_device = (
        args.local_embedding_device or os.environ.get("LOCAL_EMBEDDING_DEVICE")
    )
    local_fluency_device = (
        args.local_fluency_device or os.environ.get("LOCAL_FLUENCY_DEVICE")
    )

    # --- 0. Judge pre-flight: ping the judge BEFORE the heavy setup. ---
    
    if args.judge_anchor_every > 0 and not args.no_judge_preflight:
        from llm_training.runtime.judge_preflight import run_judge_preflight

        _ping_reply = run_judge_preflight(timeout_s=args.judge_timeout)
        if _ping_reply:
            logger.info("Judge pre-flight OK: %r", _ping_reply[:80])

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # --- 1. Data: full train pools + the frozen capped hold-outs. ---
    
    if args.prepared_dir:
        cache = load_grpo_cache(Path(args.prepared_dir))
        if cache.seed != args.seed:
            logger.warning('Prepared cache %s was built with seed=%d but --seed=%d; the pool composition follows the cache, the stream/difficulty RNG follows --seed.', args.prepared_dir, cache.seed, args.seed)
        train_pools = dict(cache.train_pools)
        val = list(cache.val)
        tool_names = cache.tool_names
        tool_schemas = cache.tool_schemas
        scores = cache.scores
        all_train = [
            ex for cls in sorted(train_pools) for ex in train_pools[cls]
        ]
        logger.info(
            "Data: loaded prepared cache %s (%d classes, %d train examples, "
            "%d val examples, %d scores); known tools=%d (%d with schemas); "
            "val_tasks=%s",
            args.prepared_dir, len(train_pools), len(all_train), len(val),
            len(scores), len(tool_names), len(tool_schemas),
            list(cache.val_tasks),
        )
    else:
        cfg = DataConfig(
            data_dir=Path(args.data_dir), seed=args.seed,
            max_per_class=args.max_per_class,
            final_turn_share=args.final_turn_share,
            reasoning_keep_share=args.reasoning_keep_share,
            nerel_split="train",
            sgr_loop_subdir=args.sgr_loop_subdir,
        )
        result = build_full_dataset(
            tokenizer, cfg,
            val_task=args.val_task,
            include_network=args.include_network,
        )
        train_pools = result.train_pools
        val = list(result.val)
        all_train = [
            ex for cls in sorted(train_pools) for ex in train_pools[cls]
        ]
        tool_names = collect_tool_names([*all_train, *val])
        tool_schemas = collect_tool_schemas([*all_train, *val])
        logger.info(
            "Data: full train pools across %d classes (total %d examples; "
            "val-task=%s with %d); known tools=%d (%d with schemas)",
            len(train_pools), len(all_train),
            list(args.val_task), len(val),
            len(tool_names), len(tool_schemas),
        )

        # Difficulty SCORING (the expensive bf16 forward pass) — run inline
        # only without --prepared-dir; the FILTER itself is shared below.
        scores = {}
        if args.difficulty_model and (
            args.difficulty_drop_easy > 0 or args.difficulty_drop_hard > 0
        ):
            logger.info(
                "Difficulty filter: scoring the train pool with %s "
                "(drop easy=%.3f, hard=%.3f)…",
                args.difficulty_model, args.difficulty_drop_easy,
                args.difficulty_drop_hard,
            )
            try:
                from llm_training.runtime.difficulty import (
                    load_difficulty_model,
                    score_example_difficulty,
                )

                diff_model, diff_tokenizer = load_difficulty_model(
                    args.difficulty_model, attn=args.attn,
                )
                try:
                    scores = score_example_difficulty(
                        diff_model, diff_tokenizer, all_train,
                        batch_size=args.difficulty_batch_size,
                        progress=not args.no_difficulty_progress,
                        desc="Difficulty scoring (train pool)",
                    )
                finally:
                    del diff_model
                    import gc

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001 - never abort the run
                logger.warning('Difficulty scoring failed (%s: %s); the filter is DISABLED and the full train pool is kept.', type(e).__name__, e)
                scores = {}

    if not all_train:
        print("No training examples after the hold-out split.", file=sys.stderr)
        return 1

    # --- 1a. Anchor val budget audit (observability ONLY). ---
    
    if args.judge_anchor_every > 0:
        max_positions: int | None = None
        try:
            max_positions = (
                int(
                    getattr(
                        AutoConfig.from_pretrained(args.model),
                        "max_position_embeddings", 0,
                    )
                    or 0
                )
                or None
            )
        except Exception as e:  # noqa: BLE001 - audit only, never abort
            logger.warning(
                "Anchor val budget audit: cannot read the model config of "
                "%s (%s: %s); the context check is skipped.",
                args.model, type(e).__name__, e,
            )
        audit_anchor_val_budget(
            val, tokenizer,
            max_new_tokens=args.anchor_max_new_tokens,
            max_positions=max_positions,
        )

    # --- 1b. Difficulty filter (loss-based, both tails; grpo twin). ---
    if (
        args.difficulty_model
        and (args.difficulty_drop_easy > 0 or args.difficulty_drop_hard > 0)
        and scores
    ):
        from llm_training.runtime.difficulty import filter_by_difficulty

        all_train, diff_stats = filter_by_difficulty(
            all_train, scores,
            drop_easy=args.difficulty_drop_easy,
            drop_hard=args.difficulty_drop_hard,
        )
        logger.info(
            "Difficulty filter: %d -> %d examples (dropped=%d: easy=%d, "
            "hard=%d, %.1f%% of the pool); scored=%d, unscored=%d, "
            "duplicates=%d; classes skipped: %d below min_class_size=%d, "
            "%d unscored; loss min/mean/max=%s.",
            diff_stats.n_input, diff_stats.n_kept, diff_stats.n_dropped,
            diff_stats.n_dropped_easy, diff_stats.n_dropped_hard,
            100.0 * diff_stats.dropped_share, diff_stats.n_scored,
            diff_stats.n_unscored, diff_stats.n_duplicates,
            diff_stats.n_classes_skipped_small, diff_stats.min_class_size,
            diff_stats.n_classes_unscored, diff_stats.loss_summary,
        )
        for line in diff_stats.format_report_lines():
            logger.info("Difficulty filter | %s", line)
        logger.info(
            "Difficulty filter: train pool now %d examples.",
            len(all_train),
        )

    # Re-group the (possibly filtered) flat train list back into per-class
    # pools for the balanced stream builder.
    train_pools = {}
    for ex in all_train:
        train_pools.setdefault(ex.task_class, []).append(ex)

    # --- 1c. Balanced train stream (verbatim grpo_train.py). ---
    prompts_per_step = args.per_device_batch * args.grad_accum
    if args.train_examples > 0:
        num_steps = math.ceil(args.train_examples / prompts_per_step)
        stream_examples = num_steps * prompts_per_step
    elif args.max_steps > 0:
        num_steps = args.max_steps
        stream_examples = num_steps * prompts_per_step
    else:
        print(
            'error: neither --train-examples nor --max-steps was given — the training length is undefined.',
            file=sys.stderr,
        )
        return 2

    train_pool, stream_stats = build_train_stream(
        train_pools,
        train_examples=stream_examples,
        max_upsample_ratio=args.max_upsample_ratio,
        seed=args.seed,
        final_turn_share=args.final_turn_share,
        reasoning_keep_share=args.reasoning_keep_share,
    )
    if stream_stats["actual_total"] < stream_examples:
        logger.warning('Balanced stream: only %d/%d examples reachable (the pools are exhausted even at --max-upsample-ratio %.1f); training proceeds with the smaller stream.', stream_stats['actual_total'], stream_examples, args.max_upsample_ratio)
    logger.info(
        "Train stream: %d examples across %d classes (requested %d, "
        "num_steps=%d, max-upsample-ratio=%.1f); per_class=%s",
        len(train_pool), stream_stats["n_classes"], stream_examples,
        num_steps, args.max_upsample_ratio, stream_stats["per_class"],
    )

    # --- 1d. Typo augmentation (offline, TRAIN stream only; verbatim). ---
    typo_cfg = TypoConfig(
        share=args.typo_share,
        seed=args.seed,
        classes=resolve_classes(args.typo_class, args.no_typo_class),
        system_share=args.system_typo_share,
        system_classes=resolve_system_classes(
            args.system_typo_class, args.no_system_typo_class
        ),
        copies=0,
    )
    train_pool, typo_stats = augment_pool(train_pool, typo_cfg)
    if typo_stats["enabled"]:
        logger.info(
            "Typo augmentation: eligible=%d selected=%d augmented=%d "
            "reverted_clean=%d per_class=%s",
            typo_stats["eligible"],
            typo_stats["selected"],
            typo_stats["augmented"],
            typo_stats["reverted_clean"],
            typo_stats["per_class"],
        )
    if args.system_typo_share > 0.0:
        train_pool, sys_stats = augment_pool_system(train_pool, typo_cfg)
        if sys_stats["enabled"]:
            logger.info(
                "System-typo augmentation: eligible=%d selected=%d "
                "augmented=%d reverted_clean=%d",
                sys_stats["eligible"], sys_stats["selected"],
                sys_stats["augmented"], sys_stats["reverted_clean"],
            )

    rows = to_prompt_completion(train_pool, tokenizer)
    logger.info("Training rows: %s", turn_kind_stats(train_pool))

    # --- 2. Judge anchor runtime (built lazily at the first anchor). ---
    # Autonomous twin of grpo_train.py's runtime: NO in-loop reward exists in
    # SFT, so the judge scorer is built standalone (no base-embeddings /
    # fluency-classifier sharing — private copies; the bge-m3 weights are
    # loaded once at the first anchor and live for the whole run).
    
    
    judge_runtime_holder: dict[str, Any] = {}

    def _get_judge_runtime() -> Any:
        if "scorer" not in judge_runtime_holder:
            from llm_training.reward.config import (
                DEFAULT_LOCAL_EMBEDDING_MODEL,
                RewardConfig,
            )
            from llm_training.reward.judge import make_ragas_scorer
            from llm_training.runtime.endpoints import OpenAIEndpoint
            from llm_training.runtime.reward_service import RewardService

            judge_base_url = os.environ.get("JUDGE_BASE_URL")
            judge_model = os.environ.get("JUDGE_MODEL")
            judge_api_key = os.environ.get("JUDGE_API_KEY")
            if not judge_base_url or not judge_model:
                raise RuntimeError(
                    'Judge anchors need JUDGE_BASE_URL and JUDGE_MODEL (env). Point them at the local vllm serve judge (see judge_serve_slurm.sh) or the cloud fallback.'
                )
            judge_cfg = RewardConfig(
                judge=OpenAIEndpoint(
                    base_url=judge_base_url,
                    model=judge_model,
                    api_key=judge_api_key or "EMPTY",
                ),
                compute_answer_correctness=True,
                known_tool_names=tool_names,
                # The anchor embedder: the flag/env overrides (on vllm_env a
                # LOCAL converted copy — the CVE-2025-32434 torch.load gate
                # + the .bin-only bge-m3 HF repo), the default otherwise.
                
                
                
                local_embedding_model=(
                    local_embedding_model or DEFAULT_LOCAL_EMBEDDING_MODEL
                ),
                embedding_concurrency=args.embedding_concurrency,
                local_embedding_device=local_embedding_device,
                local_fluency_device=local_fluency_device,
                judge_concurrency=args.judge_concurrency,
                judge_max_tokens=args.judge_max_tokens,
                judge_timeout_s=args.judge_timeout,
                # Judge rubrics for SGR reasoning turns (AC is structurally
                # skipped there) — the grpo_train.py anchor config verbatim;
                # the frozen anchor methodology (never enable the in-loop-only
                # flags here).
                
                # grpo_train.py
                enable_judge_rubrics=True,
            )
            judge_runtime_holder["scorer"] = make_ragas_scorer(judge_cfg)
            judge_runtime_holder["service"] = RewardService(
                judge_runtime_holder["scorer"],
                concurrency=max(
                    judge_cfg.embedding_concurrency,
                    judge_cfg.judge_concurrency,
                    judge_cfg.fluency_concurrency,
                ),
            )
        return judge_runtime_holder["scorer"], judge_runtime_holder["service"]

    # Anchor val pool: the WHOLE held-out val class, sorted by source for
    # deterministic anchor logs (verbatim grpo_train.py).
    anchor_examples: list[UnifiedExample] = sorted(
        val, key=lambda ex: ex.source
    ) if val else []

    # --- 3. Model (+ LoRA / OPLoRA). ---
    # torch_dtype is bf16 EXPLICITLY (not "auto") — the cluster checkpoint
    # policy (the merged saves hard-check bfloat16).
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )
    model.config.use_cache = False
    # Greedy HF-generate loops (drift probe + judge anchors) must not carry
    # the checkpoint's sampling defaults — silences the per-generate()
    # "generation flags are not valid" warning spam (greedy outputs are
    # unaffected by the flags).
    
    
    
    from llm_training.runtime.generation_config import (
        unset_greedy_sampling_flags,
    )

    unset_greedy_sampling_flags(model)
    logger.info(
        "Model loaded in %s with attn_implementation=%s",
        model.dtype, args.attn,
    )
    oplora_projection = None
    if use_lora:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=list(args.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "LoRA: r=%d alpha=%d dropout=%.2f targets=%s — %.2fM/%.2fB "
            "trainable params (%.3f%%)",
            args.lora_r, args.lora_alpha, args.lora_dropout,
            list(args.lora_target_modules), trainable / 1e6, total / 1e9,
            100.0 * trainable / total,
        )
        if args.oplora:
            from llm_training.oplora import OploraProjection

            oplora_projection = OploraProjection.from_peft_model(
                model, k=args.oplora_k
            )
            oplora_projection.project_weights()
            report = oplora_projection.interference()
            logger.info(
                "OPLoRA: %d layers projected onto the orthogonal complement "
                "of the top-%d singular subspaces (mean_rho=%s, max_rho=%s; "
                "rho is None while B=0 — freshly initialised).",
                report["n_layers"], report["k"], report["mean_rho"],
                report["max_rho"],
            )

    sft_kwargs = _build_sft_kwargs(
        SFTConfig,
        output_dir=args.output_dir,
        lr=lr,
        per_device_batch=args.per_device_batch,
        grad_accum=args.grad_accum,
        max_len=args.max_len,
        optim=args.optim,
        seed=args.seed,
        max_steps=num_steps,
        liger=args.liger,
    )
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**sft_kwargs),
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
    )

    # OPLoRA hooks FIRST (registration order): the repair callback must run
    # before the anchor callback's on_step_end so anchors/generations/saves
    # always observe projected parameters; the forward guard covers any other
    # observation path (anchor generation IS a forward — belt and braces).
    
    
    oplora_guard_handle = None
    if oplora_projection is not None:
        trainer.add_callback(oplora_projection.make_trainer_callback())
        oplora_guard_handle = oplora_projection.register_forward_guard(model)

    # --- 4. Judge-anchor ctrl-point callback. ---
    probe_items: list[Any] = []
    if not args.no_drift_probe:
        probe_path = Path(args.data_dir) / "drift_probe.jsonl"
        if probe_path.is_file():
            from llm_training.runtime.drift_probe import load_probe

            probe_items = load_probe(probe_path)
            logger.info(
                "Drift probe: %d items from %s (warn-drop=%.3f, gate=%.3f)",
                len(probe_items), probe_path,
                args.drift_probe_warn, args.drift_probe_gate,
            )
        else:
            logger.warning('Drift probe disabled: no %s (run scripts/build_drift_probe.py to generate it).', probe_path)
    # The step-0 baseline — the warmed, PRE-training checkpoint.
    probe_baseline: float | None = None
    if probe_items:
        try:
            from llm_training.runtime.drift_probe import run_drift_probe

            probe0 = run_drift_probe(
                trainer.model, tokenizer, probe_items,
                max_new_tokens=args.drift_probe_max_new_tokens,
            )
            probe_baseline = probe0["score"]
            logger.info(
                "Drift probe baseline (step 0): %.4f (n=%d); gate tol %.3f "
                "→ saves require probe >= %.4f",
                probe_baseline, int(probe0["n"]), args.drift_probe_gate,
                probe_baseline - args.drift_probe_gate,
            )
        except Exception as e:  # noqa: BLE001 - never abort the run
            logger.warning('Step-0 drift probe failed (%s); the gate baseline falls back to the first successful anchor.', e)
    # The step-0 BASELINE ANCHOR — the full val measurement on the starting
    # checkpoint (verbatim grpo_train.py).
    baseline_val: float | None = None
    if (
        args.judge_anchor_every > 0
        and anchor_examples
        and not args.no_baseline_anchor
    ):
        try:
            judge_scorer, reward_service = _get_judge_runtime()
            breakdown0 = run_anchor_scoring(
                trainer.model, tokenizer, anchor_examples, judge_scorer,
                reward_service, max_new_tokens=args.anchor_max_new_tokens,
                gen_batch_tokens=args.anchor_gen_batch_tokens,
                gen_max_batch=args.anchor_gen_max_batch,
            )
            baseline_val = breakdown0["val"]
            logger.info(
                "ANCHOR step=0 (baseline): val=%.4f AC=%.4f rubric=%.4f "
                "semsim=%.4f fluency=%.4f (n=%d, n_val=%d, ac_failed=%d, "
                "ac_skipped=%d, rubric_failed=%d, final_unparsed=%d/%d) "
                "val_by_subtype=%s — seeds best_val; no checkpoint saved "
                "(the weights live in %s)",
                breakdown0["val"], breakdown0["ac"], breakdown0["rubric"],
                breakdown0["semsim"], breakdown0["fluency"],
                int(breakdown0["n"]),
                int(breakdown0.get("n_val", 0)),
                int(breakdown0.get("ac_failed", 0)),
                int(breakdown0.get("ac_skipped", 0)),
                int(breakdown0.get("rubric_failed", 0)),
                int(breakdown0.get("final_unparsed", 0)),
                int(breakdown0.get("n_final", 0)),
                {k: round(v, 4)
                 for k, v in breakdown0.get("val_by_subtype", {}).items()},
                args.model,
            )
        except Exception as e:  # noqa: BLE001 - never abort the run
            logger.warning('Step-0 baseline anchor failed (%s); best_val falls back to -1.0 (the first in-training anchor saves unconditionally).', e)

    # The best-save mechanics: with LoRA the anchors save the peft ADAPTER
    # (the merged full models are produced after training); full-FT runs keep
    # the grpo_train.py verbatim block (the DI-seam default).
    if use_lora:
        def _save_peft_best(t: Any, best_dir: Path) -> None:
            t.save_model(str(best_dir))  # adapter + tokenizer + training args

        save_best_fn: Any = _save_peft_best
    else:
        save_best_fn = None
    anchor_callback, anchor_state = make_judge_anchor_callback(
        trainer=trainer,
        tokenizer=tokenizer,
        anchor_examples=anchor_examples,
        get_judge_runtime=_get_judge_runtime,
        output_dir=output_dir,
        anchor_every=args.judge_anchor_every,
        patience=args.anchor_patience,
        max_new_tokens=args.anchor_max_new_tokens,
        gen_batch_tokens=args.anchor_gen_batch_tokens,
        gen_max_batch=args.anchor_gen_max_batch,
        max_shard_size=args.max_shard_size,
        probe_items=probe_items,
        probe_max_new_tokens=args.drift_probe_max_new_tokens,
        probe_warn_drop=args.drift_probe_warn,
        probe_gate_tol=args.drift_probe_gate,
        probe_baseline=probe_baseline,
        initial_best_val=baseline_val,
        save_best=save_best_fn,
    )
    trainer.add_callback(anchor_callback)
    try:
        trainer.train()
    finally:
        if oplora_guard_handle is not None:
            oplora_guard_handle.remove()
        if "service" in judge_runtime_holder:
            judge_runtime_holder["service"].close()

    # --- 5. Final saves: last/ (+ merge) and best-merged/. ---
    last_dir = output_dir / "last"
    if use_lora:
        trainer.save_model(str(last_dir))  # adapter + tokenizer + training args
        _merge_final_adapter(
            trainer, tokenizer, output_dir / "last-merged",
            args.max_shard_size, oplora_projection,
        )
        logger.info(
            "Merged the final adapter into the base weights → %s",
            output_dir / "last-merged",
        )
    else:
        with sharded_checkpoint_saves(trainer, args.max_shard_size):
            trainer.save_model(str(last_dir))
        tokenizer.save_pretrained(str(last_dir))

    if use_lora and anchor_state["best_step"] is not None:
        # Reload the WINNING adapter onto a fresh base copy and merge — the
        # ablation deliverable (a plain full model for run_eval.py / mergekit).
        # The plain peft merge is exact for OPLoRA too: the anchors save the
        # adapter AFTER the repair callback (registered first), so the
        # persisted B @ A already lies in the orthogonal complement — see
        # merge_adapter_into_base.
        
        
        
        
        
        best_dir = output_dir / f"best-{anchor_state['best_step']}"
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            attn_implementation=args.attn,
        )
        try:
            best_merged = merge_adapter_into_base(
                base, best_dir, output_dir / "best-merged",
                tokenizer=tokenizer, max_shard_size=args.max_shard_size,
            )
            del best_merged
            logger.info(
                "Merged the best adapter (best-%d) into the base weights → %s",
                anchor_state["best_step"], output_dir / "best-merged",
            )
        finally:
            del base
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if anchor_state["best_step"] is None:
        logger.info(
            "SFT done. No anchor beat the selection bar (best val=%.4f); no "
            "best-<step>/ was saved — the INITIAL checkpoint %s remains the "
            "best candidate. checkpoints under %s (last: %s)",
            anchor_state["best_val"], args.model, output_dir, last_dir,
        )
    else:
        logger.info(
            "SFT done. best val=%.4f (best-%d); checkpoints under %s "
            "(last: %s%s)",
            anchor_state["best_val"], anchor_state["best_step"], output_dir,
            last_dir,
            ", best-merged: "
            f"{output_dir / 'best-merged'}" if use_lora else "",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
