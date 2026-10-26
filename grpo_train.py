"GRPO training over the reusable multi-task reward (the project's main stage).\n\n\nPipeline\n\n1. **Data.** ``build_full_dataset`` with the manual leave-task-out hold-outs\n   (repeatable ``--val-task`` — the SAME classes the SFT warm-up excluded via\n   ``--exclude-task``, all held out into the ONE anchor val pool so the\n   validation stays statistically robust; the former separate test split was\n   never scored and is merged into val),\n   then ``build_train_stream`` assembles the balanced stream of exactly\n   ``--train-examples`` prompt presentations (equal-share quotas capped at\n   ``--max-upsample-ratio`` × pool, the deficit redistributed to the abundant\n   classes; NEREL's quota split evenly across its 6 sub-configs). Each training\n   row carries a **pre-rendered TEXT prompt** (tools baked into the system\n   message, generation prompt appended — the same strings ``run_eval.py``\n   renders), plus the ``task_class`` / ``reference`` / ``extra`` / ``question``\n   columns TRL forwards to the reward function as kwargs.\n   ``--final-turn-share`` (optional) tops up the SGR ``finalanswertool`` turns\n   of the TRAINABLE SGR classes (deterministic duplication) so the envelope\n   gradient generalizes to the held-out classes. ``--typo-share`` (optional)\n   corrupts a share of the TRAIN prompts with light keyboard typos\n   (augmentex; user-typed spans only, external context untouched — see\n   :mod:`llm_training.augment.typo`); ``--system-typo-share`` (optional) is the\n   orthogonal SYSTEM-prompt (instruction-noise) dimension.\n2. **Reward (in-loop).** ``make_reward_function`` over the reusable RAGAS\n   scorer with ``judge=None``: Semantic Similarity (or the deterministic NEREL\n   F1) + ``format_check`` for tool-call turns + ``code_check`` for ``coder``\n   Python + the SGR reasoning/action/final turn dispatch — all cheap\n   in-process signals, no judge network in the hot path. The **fluency signal**\n   (raw ``P(grammatical)`` from the grammar classifier) enters the harmonic mean\n   directly as an improvement signal — the previous degradation guard\n   (``--fluency-guard``) now defaults to ``None`` (off). ``--reasoning-\n   semsim`` (optional) swaps the SGR reasoning turns from the structural\n   ``reasoning_check`` to Semantic Similarity — a quality signal for the half\n   of the pool that otherwise has none (r2 anchors: flat judge rubrics);\n   the judge anchors keep their frozen methodology either way.\n3. **Generation.** vLLM via TRL ``GRPOTrainer``: ``--vllm-mode colocate``\n   (default; single GPU pool, sleep mode swaps memory between the generation\n   and training phases — viable on one A100 80 GB with an 8-bit optimizer and\n   ``--liger``) or ``--vllm-mode server`` (a separate ``vllm serve`` pool, the\n   AGENTS GPU-split spec; see ``judge_serve_slurm.sh`` for the launch pattern).\n4. **Ctrl-points (judge anchors).** Every ``--judge-anchor-every`` steps the\n   whole held-out ``--val-task`` pool is generated (HF ``.generate`` on the\n   training model, with the KV-cache temporarily re-enabled and the\n   ``<tool_call>`` envelope preserved in the decode — the\n   ``llm_training.tokenizer_utils.decode_generation`` helper) and scored by\n   the SAME scorer re-invoked with the judge\n     endpoint (``JUDGE_*`` env vars; default: the local ``vllm serve`` judge —\n     see ``judge_serve_slurm.sh``). The aggregate Answer\n     Correctness drives: (a) drift detection (AC stalls while in-loop rewards\n     grow = reward hacking), (b) checkpoint selection (``best-<step>/`` saved\n     iff AC improves), (c) early stop after ``--anchor-patience`` anchors\n     without improvement. The judge-enabled scorer and its\n   ``RewardService`` (a single persistent background event loop — creating a\n   fresh loop per anchor would re-bind the judge's ``AsyncOpenAI`` httpx\n   client to a dead loop) are built lazily at the first anchor and live for\n   the whole run.\n5. **Liger.** ``--liger`` (default on) eliminates the dominant training-phase\n   transient — the ``[B, T, V]`` logits tensor over the 145k vocab (up to\n   ~9-19 GB at 8192-token completions) — via the fused GRPO loss\n   (``use_liger_loss`` on TRL 0.20, ``use_liger_kernel`` on TRL 1.10+).\n\nVersion compatibility\nbuilt adaptively (:func:`_build_grpo_kwargs`) — the script runs unchanged on\nTRL 0.20 + transformers 4.x (dev) and TRL 1.10 + transformers v5.2+ (the A100\ncluster): ``max_prompt_length`` was removed and the Liger flag renamed in\n1.10, and both versions default ``max_completion_length`` to a truncating 256\n(always set explicitly; a config without the field is a hard error).\n\nUsage\n    accelerate launch grpo_train.py --model checkpoints/warmup/best         --output-dir checkpoints/grpo         --val-task summarizer math_reasoner         --vllm-mode colocate --seed 24\n\n    # with a pre-built cache (prepare_dataset.py --pipeline grpo):\n    accelerate launch grpo_train.py --model checkpoints/warmup/best         --output-dir checkpoints/grpo         --prepared-dir cache/grpo --seed 24\n\nSmoke run first\n    accelerate launch grpo_train.py ... --max-steps 30 --judge-anchor-every 10\ndata-dependent OOM"

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
from llm_training.prepare import load_grpo_cache
from llm_training.reward.inputs import extract_user_question
from llm_training.reward.scoring import harmonic_mean
from llm_training.runtime.anchor_generation import (
    DEFAULT_ANCHOR_GEN_BATCH_TOKENS,
    DEFAULT_ANCHOR_GEN_MAX_BATCH,
    generate_anchor_responses,
)
from llm_training.runtime.checkpointing import (
    DEFAULT_MAX_SHARD_SIZE,
    sharded_checkpoint_saves,
)
from llm_training.schema import UnifiedExample
from llm_training.warmup.split import turn_kind

logger = logging.getLogger("train")

#: The hold-out task classes this pipeline was designed around (the SAME pair
#: the SFT warm-up excluded; keep them mirrored across all stages). Both are
#: held out into the SAME validation pool (the merged-validation design — the
#: former separate test split was never scored and its data now strengthens
#: the anchor validation).



RECOMMENDED_VAL_TASKS = ("summarizer", "math_reasoner")


def _seed_everything(seed: int) -> None:
    'Seed every RNG (grpo_train.py seeds CUDA too — it does not fork vLLM).'
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


def build_training_rows(
    examples: Sequence[UnifiedExample],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    'Convert unified examples into TRL training rows (pre-rendered prompts).\n\n\n    Each row carries: ``prompt`` (the chat-template render with tools baked\n    into the system message and the generation prompt appended — the same\n    string ``run_eval.py``/``mask_example`` produce; plain strings keep pyarrow\n    out of the heterogeneous ``tool_calls`` schema that once crashed\n    ``Dataset.from_list``), plus the ``task_class`` / ``reference`` / ``extra``\n    columns TRL forwards to the reward function, and ``question`` — the judge\n    question precomputed from the ORIGINAL message list (a rendered TEXT\n    prompt cannot be re-parsed reliably).'
    rows: list[dict[str, Any]] = []
    for ex in examples:
        rows.append(
            {
                "prompt": render_prompt(ex, tokenizer),
                "task_class": ex.task_class,
                "reference": ex.reference,
                "extra": ex.extra,
                "question": extract_user_question(ex.task_class, ex.prompt, ex.extra),
            }
        )
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
    stats = {"n_rows": len(rows), "turn_kinds": kinds}
    return rows, stats


def _build_grpo_kwargs(
    grpo_config_cls: Any,
    *,
    output_dir: str,
    num_generations: int,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    beta: float,
    lr: float,
    epochs: float,
    per_device_batch: int,
    grad_accum: int,
    optim: str,
    seed: int,
    max_steps: int,
    vllm_mode: str,
    vllm_gpu_memory_utilization: float,
    vllm_server_host: str,
    vllm_server_port: int,
    liger: bool,
    world_size: int = 1,
) -> dict[str, Any]:
    'Build GRPOConfig kwargs adaptive to the installed TRL/transformers.\n\n\nVersion landmines handled here\n    * ``max_completion_length`` defaults to a truncating **256** in both TRL\n      0.20 and 1.10 — always set explicitly (a hard error if the field is\n      missing; silently truncating every completion would zero format scores).\n    * ``max_prompt_length`` (truncating default 512) exists only in 0.20 →\n      set to ``None`` there; the field was removed in 1.10.\n    * Liger: the fused GRPO loss is ``use_liger_loss`` in 0.20 and\n      ``use_liger_kernel`` in 1.10 (single inherited flag).\n    * ``vllm_enable_sleep_mode`` exists only in 1.10 → set when present.\n    * Unknown fields are dropped with a warning (forward compatibility).'
    from dataclasses import fields as dc_fields

    known = {f.name for f in dc_fields(grpo_config_cls)}
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "num_generations": num_generations,
        "max_completion_length": max_completion_length,
        "temperature": temperature,
        "top_p": top_p,
        "beta": beta,
        "learning_rate": lr,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "optim": optim,
        "seed": seed,
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": 5,
        "save_strategy": "no",  # checkpoints are saved by the anchor callback
        "report_to": [],
        "use_vllm": True,
        "vllm_mode": vllm_mode,
    }
    if max_steps > 0:
        kwargs["max_steps"] = max_steps

    # The completion budget must be explicit — TRL's own default silently
    # truncates to 256 tokens, zeroing every tool-call format score.
    
    if "max_completion_length" not in known:
        raise RuntimeError(
            'The installed GRPOConfig declares no max_completion_length; cannot guarantee the completion budget (TRL would truncate to its own default). Refusing to run.'
        )

    # Prompt truncation: only 0.20 has the field (default 512 — truncates!);
    # 1.10 removed it (prompts are never truncated there).
    
    if "max_prompt_length" in known:
        kwargs["max_prompt_length"] = None

    if vllm_mode == "colocate":
        kwargs["vllm_gpu_memory_utilization"] = vllm_gpu_memory_utilization
        if "vllm_enable_sleep_mode" in known:
            # Swap vLLM memory out during the training phase — the single-GPU
            # colocate budget depends on it (weights+KV offloaded to CPU RAM).
            
            kwargs["vllm_enable_sleep_mode"] = True
        else:
            logger.warning('The installed GRPOConfig lacks vllm_enable_sleep_mode; colocate will keep both vLLM and training residents on the GPU (watch for OOM).')
    else:  # server
        if "vllm_server_host" in known:
            kwargs["vllm_server_host"] = vllm_server_host
            kwargs["vllm_server_port"] = vllm_server_port
        elif "vllm_server_base_url" in known:
            kwargs["vllm_server_base_url"] = (
                f"http://{vllm_server_host}:{vllm_server_port}"
            )
        else:
            raise RuntimeError(
                'vllm_mode=server requires vllm_server_host/port or vllm_server_base_url fields in GRPOConfig.'
            )

    # Liger: the fused GRPO loss flag was renamed between versions.
    
    if liger:
        if "use_liger_loss" in known:
            kwargs["use_liger_loss"] = True
            if "use_liger_kernel" in known:
                kwargs["use_liger_kernel"] = True  # fused model kernels
        elif "use_liger_kernel" in known:
            kwargs["use_liger_kernel"] = True  # 1.10: single flag drives both
        else:
            logger.warning('--liger: the installed GRPOConfig has neither use_liger_loss nor use_liger_kernel; Liger is disabled for this run.')

    # Batch divisibility: TRL requires every generation batch to decompose
    # into whole num_generations groups.
    
    effective = per_device_batch * grad_accum * world_size
    if effective % num_generations != 0:
        raise ValueError(
            f'per_device_batch({per_device_batch}) x grad_accum({grad_accum}) x world_size({world_size}) = {effective} must be divisible by num_generations({num_generations}'
        )

    unknown = sorted(set(kwargs) - known)
    if unknown:
        logger.warning("GRPOConfig dropped unsupported kwargs: %s", unknown)
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
    'Audit the anchor val pool against the anchor generation budget.\n\n\n    Observability ONLY — the pool composition stays frozen (the anchor val\n    pool is the measuring instrument; dropping or altering examples would\n    break the r2-r5 anchor comparability). Two conditions are reported\n    BEFORE the first anchor, while the ANCHOR line\'s ``final_unparsed``\n    counter exposes them only AFTER the fact:\n\n    * ``over_budget``: the reference span exceeds ``max_new_tokens`` — the\n      model cannot reproduce the turn within the budget, so a final-answer\n      turn\'s ``finalanswertool`` call gets truncated,\n      :func:`extract_final_answer` returns None and its AC is scored against\n      an EMPTY answer (AC≈0);\n    * ``over_context``: ``prompt + max_new_tokens`` exceeds the model\'s\n      position limit — :func:`run_anchor_scoring` does NOT clamp the budget\n      to the remaining context (unlike ``run_eval.py``\'s per-prompt auto\n      budget), so the whole anchor generation pass would fail.\n\n    ``max_positions=None`` (e.g. an unreadable model config) skips the\n    context branch. Logs the warnings/info line and returns the report\n    ``{"n", "max_ref_span", "max_prompt_tokens", "over_budget",\n    "over_context"}``.'
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
    "Generate + score the anchor val pool; return the aggregate breakdown.\n\n\n    The generation runs via HF ``.generate`` on the TRAINING model (the exact\n    current weights, no vLLM weight-sync games) over LEFT-padded\n    length-bucketed packages (:func:`llm_training.runtime.anchor_generation.generate_anchor_responses`: deterministic packing under the\n    conservative KV-cache budget ``gen_batch_tokens``; ``0`` = the legacy\n    one-example-at-a-time mode) with the KV-cache temporarily re-enabled —\n    ``use_cache`` is ``False`` during training for gradient-checkpointing\n    compatibility, and cache-less greedy generation recomputes the full\n    prefix at every step (quadratic cost). Every touched state (``eval``/\n    ``train``, ``use_cache``, ``padding_side``) is restored in a ``finally``\n    block. Generations are decoded via\n    :func:`llm_training.tokenizer_utils.decode_generation` so the\n    ``<tool_call>`` envelope survives (it consists of added special tokens;\n    a ``skip_special_tokens=True`` decode deletes it and corrupts every\n    final/action turn's AC/SemSim). NOTE: batched kernels are not\n    bit-identical to single-stream ones — anchor series with packages are\n    NOT token-identical to the batch=1 series (inside one run every anchor,\n    including the step-0 baseline, packs identically).\n    ``<tool_call>``\n\n    The scoring reuses a **persistent** ``RewardService`` (see\n    :func:`make_judge_anchor_callback`): its single background event loop must\n    outlive every anchor, because the judge's ``AsyncOpenAI`` httpx client\n    binds to the first loop it serves — a fresh loop per anchor would break\n    every judge call from the second anchor on.\n    event-loop\n\n    The returned breakdown separates REAL AC failures (``ac_failed``: the\n    judge call was attempted and failed, or the dispatcher crashed — the\n    ``failed_metrics`` marker / sentinel) from structural skips\n    (``ac_skipped``: SGR reasoning turns, where AC is not applicable by\n    design and stays ``None``), and reports ``final_unparsed``/``n_final``:\n    final-answer turns whose generation contains no valid\n    ``finalanswertool`` call — every content metric then scores an EMPTY\n    answer (AC≈0), usually because the generation budget truncated the\n    call (see ``--anchor-max-new-tokens``).\n\n    The headline selection measure is ``val``: EVERY example contributes via\n    its judge-side / structural metrics — the judge rubrics\n    (``rubric_score``, reasoning turns: Russian-language literacy +\n    consistency with the input context, scored by the same judge), AC\n    (action/final turns), ``format_check`` and ``code_check`` — combined per\n    example by the strict harmonic mean and averaged across the pool\n    (``n_val``). Semantic Similarity and the small-model fluency are\n    DELIBERATELY excluded: they already form the in-loop reward, and a\n    checkpoint-selection criterion overlapping the optimized objective would\n    lose its reward-hacking independence.\n    ``failed_metrics``"
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
    # AC failure visibility, refined: AC is structurally NOT applicable to
    # SGR reasoning turns (judge.py skips it by design), so a bare
    # ``answer_correctness is None`` count conflates N/A skips with real
    # judge failures (on the summarizer val pool the skips are HALF the
    # pool, which once read as "judge half-broken"). Classify by what
    # actually happened: a real failure is an attempted-and-failed AC (the
    # ``failed_metrics`` marker) or a dispatcher crash (the sentinel);
    # anything else that stayed None is a structural skip. The empty-sample
    # warning fires only when examples actually FAILED.
    
    
    
    
    
    
    
    
    
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
    # deterministic rerank verdict grading (the Meno-stand fixed-val rows —
    # a STRUCTURAL metric in both the in-loop reward and val, the same
    # precedent as format_check/code_check) — while SemSim and the small-model
    # fluency stay OUT (they already sit in the in-loop reward; a selection
    # criterion overlapping the optimized objective would lose its
    # reward-hacking independence). Strict harmonic mean per example,
    # arithmetic mean across the pool; an example whose every component
    # failed to compute (or is not applicable) drops out.
    
    
    # (action/final), format_check, code_check
    
    
    
    
    
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
            # reported separately from the leave-task-out hold-out rows, so
            # the anchor line shows WHAT moved (in-domain interpolation vs
            # cross-task transfer) instead of blending the two silently.
            
            
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
) -> Any:
    'Build the judge-anchor ctrl-point callback (the checkpoint selector).\n\n\n    Every ``anchor_every`` steps (main process only): generate the anchor val\n    pool on the training model, score it with the judge-enabled scorer, log\n    the val/AC/rubric/SemSim/fluency breakdown, save ``best-<step>/`` iff the\n    aggregate **``val``** measure improves, and stop training after\n    ``patience`` consecutive anchors without improvement. ``val`` covers\n    EVERY example via its judge-side / structural metrics (the judge rubrics\n    for reasoning turns, AC for action/final turns, ``format_check`` /\n    ``code_check``; SemSim and the small-model fluency are excluded — they\n    sit in the in-loop reward). ``max_shard_size`` is pinned on every\n    ``best-<step>/`` save (:func:`llm_training.runtime.checkpointing.sharded_checkpoint_saves` — the bfloat16 check rides along).\n    ``get_judge_runtime`` returns the\n    ``(scorer, reward_service)`` pair lazily; the service must be the SAME\n    persistent instance across anchors (one background event loop for the\n    whole run — see :func:`run_anchor_scoring`). Any anchor failure is logged\n    and the anchor skipped — a ctrl-point problem must never abort a long\n    training run. Returns ``(callback, state)`` — the dict tracks\n    ``best_val`` / ``no_improve`` for the final report.\n\n    **The drift-probe gate (the save POLICY, not the measure).** Alongside\n    every anchor the drift probe runs BEFORE the save decision (see\n    :mod:`llm_training.runtime.drift_probe`): when ``probe_items`` are given\n    AND a baseline is known AND ``probe_gate_tol > 0``, a ``best-<step>/``\n    save additionally requires ``probe_score >= probe_baseline -\n    probe_gate_tol``. The frozen ``val`` NUMBER is never blended with the\n    probe (the r2–r5 anchor series stays comparable); the probe only\n    conditions WHICH checkpoints get saved. A val-improving anchor blocked\n    by the gate counts as ``no_improve`` (a persistent degradation stops the\n    run instead of shipping a degraded checkpoint — the r2–r4 lesson:\n    parents with −15 pp instruction following passed a val-only selector).\n    A FAILED probe run never vetoes a save (a measurement failure is not\n    degradation); ``probe_baseline`` normally comes from the step-0 run in\n    ``main()`` (the warmed, pre-training checkpoint) and falls back to the\n    first successful anchor when ``None``.\n    **The step-0 baseline anchor.** ``initial_best_val`` seeds the\n    selection bar: without it ``best_val`` starts at -1.0 and the FIRST\n    in-training anchor always counts as an improvement and saves a\n    ``best-<step>/`` checkpoint even when the model is WORSE than the\n    warmed pre-training checkpoint. ``main()`` runs the full anchor\n    evaluation once before ``trainer.train()`` (the same pattern as the\n    drift-probe baseline) and passes the resulting ``val`` here; no\n    checkpoint is saved for step 0 itself (the weights already live at\n    ``--model``). A step-0 failure leaves ``None`` — the legacy\n    first-anchor-saves behaviour as the fallback. The baseline never\n    counts toward ``no_improve``; it only sets the bar.\n    ``initial_best_val``\n    no_improve / best_step.'
    from transformers import TrainerCallback

    if run_anchor_scoring_fn is None:
        run_anchor_scoring_fn = run_anchor_scoring  # DI seam for tests.
    if run_probe_fn is None:
        from llm_training.runtime.drift_probe import run_drift_probe

        run_probe_fn = run_drift_probe  # DI seam for tests.

    state = {
        # The step-0 baseline seed: None → the legacy -1.0 start (the first
        # in-training anchor saves unconditionally); a seeded value means
        # the pre-training anchor already ran and set the bar.
        
        
        
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
                # --judge-anchor-every 0 disables the anchors (the modulo
                # below would also ZeroDivisionError).
                
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
            # gates the best-save below). Synthetic machine-checkable
            # prompts generated on the training model (the 2026-09
            # attribution: the SFT/GRPO parents lost ~15 pp on IFEval-style
            # instruction following while neither the in-loop reward nor the
            # frozen val measure sees that dimension). A failed run maps to
            # probe=None → the gate passes (a measurement failure is not
            # degradation; the warning below is the visibility).
            
            # best-save
            
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
                    # The step-0 baseline was unavailable (probe disabled at
                    # startup or its run failed) — the first successful
                    # anchor becomes the reference point.
                    
                    
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

            # PROBE observability line + WARN (independent of the gate:
            # the WARN threshold is tighter than the gate tolerance).
            
            
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
                with sharded_checkpoint_saves(trainer, max_shard_size):
                    trainer.save_model(str(best_dir))
                tokenizer.save_pretrained(str(best_dir))
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


def _build_arg_parser() -> argparse.ArgumentParser:
    'Build the grpo_train.py CLI parser (separate for testability).'
    p = argparse.ArgumentParser(
        description=(
            'GRPO training over the reusable multi-task reward.'
        )
    )
    p.add_argument("--model", default="checkpoints/warmup/best",
                   help='The warmed checkpoint to start GRPO from (the sft_warmup.py best/ export).')
    p.add_argument("--data-dir", default="src_data",
                   help='Path to the src_data catalog.')
    p.add_argument("--prepared-dir", default=None, metavar="DIR",
                   help='Load the pre-built unified dataset from this cache directory (produced by prepare_dataset.py --pipeline grpo) instead of building + difficulty-scoring it inline. Skips both expensive phases; the difficulty filter is re-applied from the cached scores with --difficulty-drop-easy/hard.')
    p.add_argument("--output-dir", default="checkpoints/grpo",
                   help='Output dir for best-<step>/ checkpoints.')
    p.add_argument("--max-shard-size", default=DEFAULT_MAX_SHARD_SIZE,
                   metavar="SIZE",
                   help='pins the shard size across transformers versions: v5.2+ changed the library default to 50GB, which would write one monolithic ~30 GB file; also hard-checks bfloat16 before every save).')
    p.add_argument("--val-task", nargs="+", default=list(RECOMMENDED_VAL_TASKS),
                   metavar="CLS",
                   help=f'Held-out validation class(es) (default: {list(RECOMMENDED_VAL_TASKS)}')
    p.add_argument("--max-per-class", type=int, default=200,
                   help='Per-class cap of the FROZEN val hold-out pools (the anchor measuring instrument; the train stream is sized by --train-examples instead).')
    p.add_argument("--sgr-loop-subdir", default=None,
                   help='subdir with the loop-format SGR catalog (reasoningtool; scripts/convert_sgr_to_loop.py); per-file preference of the SGR loader, classes without files there fall back to sgr_agent')
    p.add_argument("--include-network", action="store_true",
                   help="Include network-only classes (nerel_instruct, loaded from its practically inexhaustible 'train' split) in the train pool.")
    p.add_argument("--final-turn-share", type=float, default=None,
                   metavar="SHARE",
                   help='Minimum share of SGR final-answer turns in each TRAINABLE SGR class pool (0<SHARE<1, computed on the RESULTING pool; deterministic duplication after the cap; None = off). Boosts the finalanswertool-envelope gradient so the format transfers to the held-out classes (r2 anchors: final_unparsed 3-5/10 there; held-out classes are never boosted).')
    p.add_argument("--reasoning-keep-share", type=float, default=None,
                   metavar="SHARE",
                   help='Keep only SHARE of the reasoning turns in each TRAINABLE SGR class pool (0<SHARE<=1, deterministic prefix downsample; None = off). The r3 rationale: reasoning turns carry no quality gradient (saturated markers, flat judge rubrics) yet consume the generation budget; thinning shifts the budget onto the action/final turns with structural signals. Held-out classes are never downsampled.')
    p.add_argument("--difficulty-model", default="anonymous/base-model",
                   metavar="MODEL_ID",
                   help="the 'untrained' reference. Empty string disables the filter. The model is loaded, scores, and is freed from VRAM before the policy/vLLM init.")
    p.add_argument("--difficulty-drop-easy", type=float, default=0.05,
                   metavar="SHARE",
                   help='Per-class share of the EASIEST (lowest reference loss) train examples to drop (the r2 zero-variance lesson: they already saturate the reward). 0 = off.')
    p.add_argument("--difficulty-drop-hard", type=float, default=0.01,
                   metavar="SHARE",
                   help='Per-class share of the HARDEST (highest reference loss) train examples to drop (out-of-distribution')
    p.add_argument("--difficulty-batch-size", type=int, default=8,
                   help='Upper bound on the number of examples per difficulty scoring forward pass; the actual batches are also capped by a padded-token budget (16384) so the [B, T, V] logits transient stays bounded.')
    p.add_argument("--no-difficulty-progress", action="store_true",
                   help='Disable the tqdm progress bars of the difficulty scoring (they are forced ON in non-TTY SLURM logs by default; TQDM_DISABLE does not apply).')
    p.add_argument("--num-generations", type=int, default=8,
                   help='GRPO group size.')
    p.add_argument("--max-completion-length", type=int, default=8192,
                   help='Generation budget per completion (TRL default 256 truncates')
    p.add_argument("--temperature", type=float, default=0.7,
                   help='greedy gives zero within-group variance).')
    p.add_argument("--top-p", type=float, default=0.95,
                   help='Nucleus sampling p.')
    p.add_argument("--beta", type=float, default=0.0,
                   help='KL coefficient to the reference policy (0 = off; degradation defence is the anchors + the strict harmonic reward).')
    p.add_argument("--lr", type=float, default=1e-6,
                   help='Learning rate.')
    p.add_argument("--train-examples", type=int, default=0,
                   help='Total number of training prompt presentations: the balanced train stream is assembled to exactly this size (rounded up to a whole multiple of the effective batch). Replaces --epochs. 0 = unset (only a --max-steps smoke run may omit it).')
    p.add_argument("--max-upsample-ratio", type=float, default=8.0,
                   help="Maximum up-sampling multiple of a scarce class's pool when balancing the train stream (must be >= 1). The deficit is redistributed to the abundant classes.")
    p.add_argument("--max-steps", type=int, default=0,
                   help='If >0, cap training steps (smoke runs).')
    p.add_argument("--per-device-batch", type=int, default=4,
                   help='Per-device batch (completions per micro-step).')
    p.add_argument("--grad-accum", type=int, default=8,
                   help='Gradient accumulation steps.')
    p.add_argument("--optim", default="paged_adamw_8bit",
                   help='Optimizer; the colocate single-A100 budget assumes an 8-bit optimizer (paged_adamw_8bit pages optimizer states out under peak pressure; fallback: adafactor).')
    p.add_argument("--attn", default="sdpa",
                   choices=("sdpa", "flash_attention_2", "eager"),
                   help='Attention implementation passed to from_pretrained. flash_attention_2 is recommended on A100: O(N) attention memory and no padded-batch fallbacks.')
    p.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate",
                   help='vLLM generation mode: colocate (single GPU pool, sleep mode) or server (separate vllm serve pool).')
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.4,
                   help='vLLM GPU memory fraction in colocate mode.')
    p.add_argument("--vllm-server-host", default="localhost",
                   help='vLLM server host (server mode).')
    p.add_argument("--vllm-server-port", type=int, default=8000,
                   help='vLLM server port (server mode).')
    p.add_argument("--liger", dest="liger", action="store_true", default=True,
                   help='Fused GRPO loss + model kernels (liger-kernel; default on).')
    p.add_argument("--no-liger", dest="liger", action="store_false",
                   help='Disable Liger.')
    p.add_argument("--fluency-guard", type=float, default=None, metavar="THRESHOLD",
                   help='the raw P(grammatical) enters the harmonic mean directly as an improvement signal (the grammar classifier is well-spread, unlike the saturated RuCoLa). Set e.g. 0.9 to re-enable the brake for ablations.')
    p.add_argument("--no-fluency", action="store_true",
                   help='Disable fluency entirely in the loop.')
    p.add_argument("--reasoning-semsim", action="store_true",
                   help='Score SGR-agent reasoning turns with Semantic Similarity in the IN-LOOP reward instead of the structural reasoning_check (default OFF: reasoning is open-ended, SemSim rewards paraphrase-imitation')
    p.add_argument("--no-schema-check", action="store_true",
                   help='Disable the in-loop tool-call schema_check (the argument validation against the prompt-declared JSON schemas; default ON in the loop only')
    p.add_argument("--no-language-guard", action="store_true",
                   help="Disable the in-loop fluency language guard (default ON in the loop only: the fluency score is multiplied by the confidence of the example's target language")
    p.add_argument("--no-rummlu-em", action="store_true",
                   help='Disable the deterministic rummlu_pro letter exact match (default ON')
    p.add_argument("--no-rerank-em", action="store_true",
                   help='0.3 valid JSON verdict + 0.7 graded by the ordinal label distance; replaces SemSim and skips AC for the class).')
    p.add_argument("--format-epsilon", type=float, default=0.0, metavar="EPS",
                   help='In-loop-ONLY format epsilon floor: the format component is floored to EPS before the strict harmonic combine, so a broken tool-call envelope no longer zeroes the whole example (the r2 zero-variance lesson). 0.0 (default) keeps the strict semantics; recommendation 0.1-0.15. The judge anchors')
    p.add_argument("--local-embedding-device", default=None,
                   help='Device for the in-process bge-m3 embedder (SemSim). Env: LOCAL_EMBEDDING_DEVICE.')
    p.add_argument("--local-fluency-device", default=None,
                   help='Device for the in-process fluency model. Env: LOCAL_FLUENCY_DEVICE.')
    p.add_argument("--embedding-concurrency", type=int, default=2,
                   help='Concurrent in-process embedding forwards.')
    p.add_argument("--judge-concurrency", type=int, default=8,
                   help="Max in-flight judge HTTP requests at the anchors. Keep STRICTLY below the provider's concurrency limit (e.g. Z.AI GLM-5.3-Flash: 50)")
    p.add_argument("--judge-max-tokens", type=int, default=4096,
                   help='rubrics). Raise substantially for thinking judges (GLM on Z.AI: reasoning counts toward the cap and truncates the JSON; e.g. 16384')
    p.add_argument("--judge-timeout", type=float, default=600.0,
                   help='Per-request judge HTTP timeout, seconds (the anchor default 600 covers slow judges, incl. cloud thinking models whose reasoning cannot be disabled; scale it with --judge-max-tokens, e.g. 1800 for 16384 tokens at ~15-60 tok/s).')
    p.add_argument("--no-judge-preflight", action="store_true",
                   help='Skip the pre-flight judge ping (a one-message availability check run BEFORE the heavy setup; a failed ping aborts the run).')
    p.add_argument("--judge-anchor-every", type=int, default=500,
                   help='Judge anchor interval (steps).')
    p.add_argument("--anchor-max-new-tokens", type=int, default=2048,
                   help='Generation budget for anchor scoring; the val pool is audited against it at startup (a reference span over the budget truncates finalanswertool calls and their AC is scored against an EMPTY answer).')
    p.add_argument("--anchor-gen-batch-tokens", type=int,
                   default=DEFAULT_ANCHOR_GEN_BATCH_TOKENS, metavar="N",
                   help='Anchor-generation batching: LEFT-padded packages of greedy HF .generate under the KV-cache budget (padded prompt + the FULL generation budget) x rows <= N tokens (~192 KiB/token for a 14B Qwen2.5-class model; default 65536 ~ 12 GB KV). Batched kernels are not bit-identical to single-stream ones')
    p.add_argument("--anchor-gen-max-batch", type=int,
                   default=DEFAULT_ANCHOR_GEN_MAX_BATCH, metavar="ROWS",
                   help='Row cap of one anchor-generation package (prefill padding waste grows with the spread; the default 8 sits on the diminishing-returns knee).')
    p.add_argument("--no-drift-probe", action="store_true",
                   help='Disable the drift probe (the synthetic instruction-following mini-set checked deterministically at every anchor; auto-loaded from <data-dir>/drift_probe.jsonl when present')
    p.add_argument("--drift-probe-warn", type=float, default=0.05,
                   metavar="DROP",
                   help='WARN when the drift-probe score drops more than DROP below the baseline (default 0.05).')
    p.add_argument("--drift-probe-gate", type=float, default=0.10,
                   metavar="TOL",
                   help='a persistent degradation stops the run instead of shipping a degraded checkpoint). Default 0.10 ≈ 2 sigma of the binomial noise of the 76-item probe; 0 disables the gate (the probe stays log+WARN).')
    p.add_argument("--drift-probe-max-new-tokens", type=int, default=256,
                   help='Generation budget per drift-probe prompt.')
    p.add_argument("--anchor-patience", type=int, default=3,
                   help='Early stop after N anchors without AC improvement.')
    p.add_argument("--no-baseline-anchor", action="store_true",
                   help='Skip the step-0 baseline anchor (the pre-training full-val measurement that seeds the best_val selection bar; without it the first in-training anchor saves a best-<step>/ checkpoint unconditionally, even when worse than the starting one).')
    p.add_argument("--typo-share", type=float, default=0.0, metavar="SHARE",
                   help='external context (retrieved docs, texts to summarise, tool results, the SGR date preamble) is never touched; val/test pools stay clean. Russian draws pc/mobile keyboards, English is pc-only (augmentex has no mobile+eng statistics). SHARE is exact: the fraction of prompts carrying >=1 visible typo.')
    p.add_argument("--typo-class", action="append", default=None, metavar="CLS",
                   help='Repeatable; restrict typo augmentation to these task classes (default: the built-in allowlist of classes with a known typed/external boundary).')
    p.add_argument("--no-typo-class", action="append", default=None,
                   metavar="CLS",
                   help='Repeatable; remove task classes from the typo allowlist.')
    p.add_argument("--system-typo-share", type=float, default=0.0,
                   metavar="SHARE",
                   help='Offline SYSTEM-prompt (instruction-noise) augmentation of a SHARE fraction of the TRAIN pool (0 = off, default). Light typos land in the SYSTEM message of the allowlisted classes; the <tools> block and fenced code/JSON never touched.')
    p.add_argument("--system-typo-class", action="append", default=None,
                   metavar="CLS",
                   help='Repeatable; restrict SYSTEM-prompt augmentation to these classes (default: the built-in allowlist of plain-instruction system prompts).')
    p.add_argument("--no-system-typo-class", action="append", default=None,
                   metavar="CLS",
                   help='Repeatable; remove classes from the system-typo allowlist.')
    p.add_argument("--seed", type=int, default=42,
                   help='RNG seed (logged).')
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    args = _build_arg_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    logger.info("train: seed=%d", args.seed)
    _seed_everything(args.seed)

    # Lazy heavy imports (dev environments without the train extra stay light).
    
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "GRPO training needs 'torch', 'transformers', 'datasets' and 'trl': pip install -e '.[train]'."
        ) from e
    if args.liger:
        try:
            import liger_kernel  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                "--liger requires the 'liger-kernel' package: pip install -U liger-kernel (>=0.7.0 for transformers v5). Or pass --no-liger."
            ) from e

    import os

    local_embedding_device = (
        args.local_embedding_device or os.environ.get("LOCAL_EMBEDDING_DEVICE")
    )
    local_fluency_device = (
        args.local_fluency_device or os.environ.get("LOCAL_FLUENCY_DEVICE")
    )

    # --- 0. Judge pre-flight: ping the judge BEFORE the heavy setup. ---
    # A broken judge (wrong JUDGE_BASE_URL root, missing key, 401/404, a
    # hung server) would stay invisible until the FIRST anchor — hours of
    # setup wasted with checkpoint selection silently dead. One trivial
    # question + a bounded wait; a failure exits with a bilingual error
    # (--no-judge-preflight escapes). Missing JUDGE_* env → a warning and
    # training continues (mirrors the anchor skip behaviour).
    
    
    
    
    
    
    
    if args.judge_anchor_every > 0 and not args.no_judge_preflight:
        from llm_training.runtime.judge_preflight import run_judge_preflight

        _ping_reply = run_judge_preflight(timeout_s=args.judge_timeout)
        if _ping_reply:
            logger.info("Judge pre-flight OK: %r", _ping_reply[:80])

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # --- 1. Data: full train pools + the frozen capped hold-outs. ---
    # The GRPO train side keeps the FULL length-filtered pools (no per-class
    # cap) so the balanced stream below loses no examples; the hold-out
    # classes stay capped — the anchor measuring instrument. NEREL (when
    # --include-network) loads its practically inexhaustible 'train' split.
    # With --prepared-dir the (expensive, deterministic) build + difficulty
    # SCORING are skipped: the cache carries the full train pools, the frozen
    # val pool and the per-source difficulty scores; the difficulty FILTER
    # below re-applies the instant quantile filter from the cached scores.
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
            sgr_loop_subdir=getattr(args, "sgr_loop_subdir", None),
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
    # The anchor val pool is the frozen measuring instrument: never filtered,
    # never altered. But its examples must FIT the anchor generation budget —
    # otherwise the ANCHOR line's final_unparsed counter exposes the damage
    # only after the first anchor (a truncated finalanswertool → AC scored
    # against an EMPTY answer). This audit warns BEFORE the model/vLLM init
    # when a reference span exceeds --anchor-max-new-tokens, or when
    # prompt + budget exceeds the model's position limit (run_anchor_scoring
    # has no context clamp). Skipped with --judge-anchor-every 0.
    
    
    
    
    
    
    
    
    
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

    # --- 1b. Difficulty filter (loss-based, ToolACE-style; both tails). ---
    # The TRAIN pool is trimmed by the reference loss of a SEPARATE
    # difficulty model (the base — the "untrained" reference): the easiest
    # examples (the reference is already reproduced → saturated reward → the
    # r2 zero-variance groups) and the hardest (out-of-distribution /
    # mis-labelled noise) are dropped per-class. The SCORING is either the
    # cached scores (--prepared-dir) or was just computed above; this filter
    # is a pure per-class quantile over ``scores`` — instant in both cases.
    # fixed-val rows are already in the val pool → never scored (the measuring
    # instrument). A scoring failure leaves ``scores`` empty → the filter is a
    # no-op (never an abort).
    
    
    
    
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

    # --- 1c. Balanced train stream (no data loss, example-budget driven). ---
    # The full per-class pools are assembled into a stream of exactly
    # ``stream_examples`` rows: equal-share quotas capped at
    # ``--max-upsample-ratio`` × pool, the deficit redistributed to the
    # abundant classes (fresh data), scarce classes up-sampled by round-robin
    # duplication, NEREL's quota split evenly across its 6 sub-configs. The
    # SGR reasoning-downsample / final-turn-boost are applied to the full
    # pools inside the builder.
    
    
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

    # --- 1d. Typo augmentation (offline, TRAIN stream only). ---
    # Corruption of a share of the train prompts with light keyboard typos
    # (augmentex); the val hold-outs are the measuring instrument and
    # are never passed through augment_pool. The rows below (prompt render +
    # the judge ``question`` column) are then derived from the augmented
    # messages, keeping everything consistent with what the policy saw.
    
    
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
    # SYSTEM-prompt (instruction-noise) augmentation — the orthogonal
    # dimension, applied on top of the user-typo corruption.
    
    if args.system_typo_share > 0.0:
        train_pool, sys_stats = augment_pool_system(train_pool, typo_cfg)
        if sys_stats["enabled"]:
            logger.info(
                "System-typo augmentation: eligible=%d selected=%d augmented=%d "
                "reverted_clean=%d",
                sys_stats["eligible"], sys_stats["selected"],
                sys_stats["augmented"], sys_stats["reverted_clean"],
            )

    rows, row_stats = build_training_rows(train_pool, tokenizer)
    logger.info("Training rows: %s", row_stats)

    # --- 2. In-loop reward (no judge). ---
    from llm_training.reward.config import RewardConfig
    from llm_training.reward.dispatch import make_reward_function
    from llm_training.reward.judge import make_ragas_scorer

    reward_cfg = RewardConfig(
        judge=None,
        compute_answer_correctness=False,
        known_tool_names=tool_names,
        known_tool_schemas=tool_schemas,
        embedding_concurrency=args.embedding_concurrency,
        local_embedding_device=local_embedding_device,
        local_fluency_device=local_fluency_device,
        enable_fluency=not args.no_fluency,
        fluency_guard_threshold=None if args.no_fluency else args.fluency_guard,
        # Quality signal for SGR reasoning turns: the structural
        # reasoning_check (4 SGR markers) carries no quality gradient — the
        # r2 anchors showed flat judge rubrics (~0.48) while every other
        # metric moved. With this flag the reasoning turns fall back to
        # SemSim (reference-anchored, gives within-group variance). Judge
        # anchors are NOT affected (see the note in _get_judge_runtime).
        
        
        
        enable_reasoning_semsim=args.reasoning_semsim,
        # Tool-call argument validation against the prompt-declared JSON
        # schemas (in-loop ONLY: the anchor config below never sets it —
        # the frozen anchor methodology; the same precedent as
        # enable_reasoning_semsim). The benchmark attribution showed the
        # SFT/GRPO parents nearly doubled the valid-JSON rate on hard
        # schemas while the merge lost the gain — this trains exactly that
        # dimension.
        
        enable_schema_check=not args.no_schema_check,
        # Fluency language guard (in-loop ONLY — the frozen anchor config
        # never sets it): multiply the fluency score by the confidence of the
        # example's target language (extra["target_language"], computed at
        # build time) — a reward-hacking defence against the policy drifting
        # to English, which the English-trained grammar classifier may score
        # slightly higher.
        
        enable_language_guard=not args.no_language_guard,
        # ruMMLU-pro: the deterministic letter EM replaces SemSim for the
        # class (a single option letter carries no usable cosine signal).
        # Default-on mirrors enable_nerel_f1 / enable_docsem_check; the
        # escape hatch mirrors --no-fluency.
        
        enable_rummlu_em=not args.no_rummlu_em,
        # Meno-stand rerank: the deterministic verdict grading (0.3 valid
        # JSON + 0.7 graded by the ordinal label distance) replaces SemSim
        # for the class; AC is structurally skipped there (a judge over a
        # JSON verdict is noise).
        
        
        enable_rerank_em=not args.no_rerank_em,
        # In-loop-ONLY format epsilon floor (the r2 zero-variance lesson:
        # frac_reward_zero_std failed at 0.375-0.675 — a broken envelope
        # zeroed the whole example). 0.0 keeps the strict semantics.
        
        format_epsilon_floor=args.format_epsilon,
    )
    reward_scorer = make_ragas_scorer(reward_cfg)
    reward_fn = make_reward_function(reward_scorer, reward_config=reward_cfg)
    logger.info("In-loop reward: %s", reward_cfg.describe())

    # --- 3. Judge anchor runtime (built lazily at the first anchor). ---
    # The scorer AND its RewardService are cached together and live for the
    # whole run: the service owns ONE persistent background event loop, and
    # the judge's AsyncOpenAI httpx client binds to the first loop it serves —
    # rebuilding the service per anchor would break every judge call from the
    # second anchor on ("Event loop is closed").
    
    
    
    judge_runtime_holder: dict[str, Any] = {}

    def _get_judge_runtime() -> Any:
        if "scorer" not in judge_runtime_holder:
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
                embedding_concurrency=args.embedding_concurrency,
                local_embedding_device=local_embedding_device,
                local_fluency_device=local_fluency_device,
                # Tunable judge-side knobs (cloud judges included):
                # --judge-concurrency caps in-flight judge HTTP requests — keep
                # it strictly below the provider's concurrency limit (e.g. Z.AI
                # GLM-5.3-Flash: 50; the default 8 leaves retry headroom);
                # --judge-max-tokens bounds the structured output (raise for
                # thinking judges whose reasoning may count toward the cap);
                # --judge-timeout defaults to 600 s.
                
                
                
                
                
                judge_concurrency=args.judge_concurrency,
                judge_max_tokens=args.judge_max_tokens,
                # Anchor completions reach ~4-7k tokens at ~15 tok/s per
                # request on the V0 + json_object judge, so the RewardConfig
                # default of 120 s provokes client-side "Request timed out";
                # the anchor runs off the hot loop — a 600 s budget is safe.
                
                
                
                
                judge_timeout_s=args.judge_timeout,
                # Judge rubrics for SGR reasoning turns (AC is structurally
                # skipped there): Russian-language literacy + consistency with
                # the input context, scored by the same judge. This is what
                # gives the `val` selection measure FULL val-pool coverage
                # with a methodology different from the in-loop reward.
                # NOTE: ``enable_reasoning_semsim`` is deliberately NOT set
                # here even when the in-loop reward uses it — enabling it in
                # this config would flip the reasoning turns from rubrics to
                # AC and change the `val` measure mid-flight. The anchor is
                # the frozen measuring instrument; only the in-loop reward
                # composition may change between runs.
                
                
                
                
                
                
                
                enable_judge_rubrics=True,
            )
            judge_runtime_holder["scorer"] = make_ragas_scorer(
                judge_cfg,
                # Share the heavy reward-model weights with the in-loop
                # scorer: the base bge-m3 embedder (~2.3 GB) and the (by now
                # loaded) grammar classifier (a tiny BERT, ~50 MB) stay
                # singletons — no second copy becomes resident for the rest of
                # the run.
                # Each scorer wraps the shared bases in its OWN
                # CachingEmbeddings / FluencyScorer (own semaphore + cache),
                # so asyncio primitives never cross the two RewardService
                # event loops. If the fluency classifier has not loaded by
                # the first anchor, None falls back to a private copy (the
                # previous behaviour).
                
                
                
                
                
                
                
                
                base_embeddings=reward_scorer.base_embeddings,
                fluency_classifier=reward_scorer.fluency_classifier,
            )
            judge_runtime_holder["service"] = RewardService(
                judge_runtime_holder["scorer"],
                concurrency=max(
                    judge_cfg.embedding_concurrency,
                    judge_cfg.judge_concurrency,
                    judge_cfg.fluency_concurrency,
                ),
            )
        return judge_runtime_holder["scorer"], judge_runtime_holder["service"]

    # Anchor val pool: the WHOLE held-out val class (calibration quality over
    # anchor speed — AC drives checkpoint selection and early stop, so the
    # honest full-population mean beats a noisy subsample; the pool is bounded
    # by --max-per-class). Sorted by source for deterministic anchor logs.
    
    
    anchor_examples: list[UnifiedExample] = sorted(
        val, key=lambda ex: ex.source
    ) if val else []

    # --- 4. Trainer. ---
    # torch_dtype is bf16 EXPLICITLY (not "auto"): the A100 cluster policy
    # requires every checkpoint in bfloat16, and "auto" would silently
    # follow the source checkpoint's config dtype (fp16/fp32 sources would
    # poison every save downstream; vLLM colocate weight sync also needs
    # the matching dtype).
    # torch_dtype
    
    # config.dtype
    
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )
    logger.info(
        "Model loaded in %s with attn_implementation=%s",
        model.dtype, args.attn,
    )
    model.config.use_cache = False
    # Greedy HF-generate loops (drift probe + judge anchors) must not carry
    # the checkpoint's sampling defaults — silences the per-generate()
    # "generation flags are not valid" warning spam (greedy outputs are
    # unaffected by the flags; the vLLM colocate path builds its own sampling
    # params and is not touched).
    
    
    
    
    from llm_training.runtime.generation_config import (
        unset_greedy_sampling_flags,
    )

    unset_greedy_sampling_flags(model)

    grpo_kwargs = _build_grpo_kwargs(
        GRPOConfig,
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        lr=args.lr,
        epochs=1.0,
        per_device_batch=args.per_device_batch,
        grad_accum=args.grad_accum,
        optim=args.optim,
        seed=args.seed,
        max_steps=num_steps,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        liger=args.liger,
    )
    trainer = GRPOTrainer(
        model=model,
        args=GRPOConfig(**grpo_kwargs),
        reward_funcs=[reward_fn],
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
    )

    # --- 5. Judge-anchor ctrl-point callback. ---
    # The drift probe items ride along with the anchors: the score gates
    # the best-<step>/ saves (the save POLICY — the frozen ``val`` number
    # is never blended with the probe) and prints an observability column.
    
    # best-<step>
    
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
    # The step-0 baseline — the warmed, PRE-training checkpoint: the
    # cleanest reference point for the gate (the first anchor would already
    # be a slightly trained model). A failure falls back to the first
    # successful anchor (see make_judge_anchor_callback).
    
    
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
    # The step-0 BASELINE ANCHOR — the full val measurement on the warmed,
    # PRE-training checkpoint (the same pattern as the drift-probe baseline
    # above). Before r5, best_val started at -1.0 and the FIRST in-training
    # anchor always saved a best-<step>/ checkpoint even when the model was
    # worse than the starting one; the baseline seeds the selection bar so a
    # checkpoint is only saved when it BEATS the pre-training model. No
    # checkpoint is written for step 0 itself (the weights already live at
    # --model); a failure falls back to the legacy -1.0 behaviour; the
    # baseline never counts toward no_improve. Escape: --no-baseline-anchor.
    
    
    
    
    
    
    
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
    )
    trainer.add_callback(anchor_callback)
    try:
        trainer.train()
    finally:
        # Tear down the anchor RewardService (its background event-loop
        # thread) if an anchor was ever run. The judge scorer's AsyncOpenAI
        # client is bound to that loop, so it must not outlive it.
        
        if "service" in judge_runtime_holder:
            judge_runtime_holder["service"].close()

    last_dir = output_dir / "last"
    with sharded_checkpoint_saves(trainer, args.max_shard_size):
        trainer.save_model(str(last_dir))
    tokenizer.save_pretrained(str(last_dir))
    if anchor_state["best_step"] is None:
        # No in-run best-<step>/ exists: either the step-0 baseline was
        # never beaten (its val is best_val) or the baseline anchor failed
        # and no in-training anchor improved on -1.0 — in both cases the
        # INITIAL checkpoint is the best candidate.
        
        
        
        
        logger.info(
            "GRPO done. No anchor beat the selection bar (best val=%.4f); "
            "no best-<step>/ was saved — the INITIAL checkpoint %s remains "
            "the best candidate. checkpoints under %s (last: %s)",
            anchor_state["best_val"], args.model, output_dir, last_dir,
        )
    else:
        logger.info(
            "GRPO done. best val=%.4f (best-%d); checkpoints under %s "
            "(last: %s)",
            anchor_state["best_val"], anchor_state["best_step"], output_dir,
            last_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
