'SFT warm-up: prime the tool-call format before GRPO (one stage, offline).\n\nSFT warm-up\n\nWhy this stage exists\n\nThe base Meno-Lite model produces high-quality *content* but never emits a\nvalid Qwen2.5 ``<tool_call>`` envelope (the cold-start probe: 0/1488 valid\ncalls in sampling mode; the model even *verbalises* "I will call the\nfinalanswertool function" without ever emitting the envelope — it knows the\nsemantics from the system schema but not the wire format). The GRPO reward is\na strict harmonic mean, so every group\'s rewards collapse to 0 with zero\nvariance — GRPO has no gradient to follow. A short full fine-tune on the\ntool-call-faithful references teaches the envelope in a few epochs without\ntouching the content knowledge (which stays GRPO\'s job); the default is 5\nepochs, and per-epoch val scoring makes the extra epochs safe (``best/`` is\nselected by the generative metric, not by the last epoch). GRPO remains the\nproject\'s main training method; this warm-up is a one-off format priming\nstage.\n``<tool_call>`` Qwen2.5\ntool-call-faithful\n\nData\n    ``build_unified_dataset`` over the 9 SGR-agent classes +\n    ``opencode_gen`` + ``tool_calls_multiturn`` (envelope diversity), capped\n    per class, length-filtered to ``--max-len``. Only SGR examples are split\n    into train/val (stratified by class × turn kind; the auxiliary classes go\n    to train in full); ``--exclude-task`` mirrors the future GRPO hold-outs.\n\nValidation\n    Per epoch a greedy generative check runs on the val split: each generated\n    turn is scored ``harmonic(format-side, chrF++)`` — ``format_check`` on the\n    raw generation for final/action turns (``reasoning_check`` for reasoning\n    turns) + chrF++ of the comparable content vs the reference\n    (:mod:`llm_training.warmup.metrics`). The checkpoint with the best\n    aggregate is kept as ``best/`` (``last/`` is also saved). The generated\n    text is decoded WITHOUT stripping special tokens\n    (:func:`decode_generation`): ``<tool_call>`` / ``</tool_call>`` are added\n    special tokens of the Meno-Lite tokenizer, so the legacy\n    ``skip_special_tokens=True`` decode deleted the envelope and zeroed every\n    final/action turn via the strict harmonic mean (the first cluster run\n    reported ``final=0.0000 action=0.0000`` while the model was fine). Only\n    the trailing stop tokens are stripped. Generation runs with the KV-cache\n    on (``use_cache`` is temporarily re-enabled; it is ``False`` during\n    training for gradient checkpointing). Raw generations are dumped to\n    ``<output-dir>/val_generations.jsonl`` per epoch for instant diagnosis.\n    The authoritative gate afterwards is the full sampling probe via\n    ``run_eval.py`` (≥80% valid final-turn calls).\n    (:mod:`llm_training.warmup.metrics`\n``</tool_call>``\n    Meno-Lite\n    ``<output-dir>/val_generations.jsonl``\n\nUsage\n    accelerate launch sft_warmup.py --model anonymous/base-model         --output-dir checkpoints/warmup --epochs 2 --seed 42         --attn flash_attention_2 --liger --optim paged_adamw_8bit\n\nMemory on a single A100 80 GB\n    The dominant transient peak is the ``[B, T, V]`` logits tensor over the\n    145k vocab (plus its fp32 CE upcast) — ``--liger`` replaces the lm_head +\n    cross-entropy with a fused kernel that never materializes it; FA2\n    (``--attn flash_attention_2``) makes attention memory O(N). The remaining\n    ladder: ``--per-device-batch 1 --grad-accum 16``, ``--optim\n    paged_adamw_8bit``, and ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``\n    (instead of the legacy ``garbage_collection_threshold``/``max_split_size_mb``\n    pair, which itself provokes allocator thrash warnings near the limit).\n    (``--attn flash_attention_2``\n    paged_adamw_8bit``\n    ``max_split_size_mb``\n\nVersion compatibility\n    The SFTConfig kwargs are built **adaptively** to the installed TRL /\n    transformers stack (:func:`_build_sft_kwargs`): the script runs unchanged\n    on TRL 0.20 + transformers 4.x (local dev) and on TRL 1.10 +\n    transformers v5.2+ (the A100 cluster), where ``warmup_ratio`` was removed\n    and ``max_length`` defaults to a truncating 1024. Kwargs the installed\n    SFTConfig does not declare are dropped with a warning; a missing length\n    cap is a hard error.\n    Kwargs SFTConfig\n    transformers (:func:`_build_sft_kwargs`\n    ``max_length``'

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import llm_training.warmup.metrics as warmup_metrics
from llm_training.build import build_unified_dataset
from llm_training.config import DataConfig
from llm_training.inference.generate import render_prompt
from llm_training.loaders import collect_tool_names
from llm_training.loaders.sgr_agent import CLASSES as SGR_CLASSES
from llm_training.masking import mask_example
from llm_training.prepare import load_warmup_cache
from llm_training.runtime.checkpointing import (
    DEFAULT_MAX_SHARD_SIZE,
    sharded_checkpoint_saves,
)
from llm_training.schema import UnifiedExample
from llm_training.tokenizer_utils import (
    decode_generation,
    prepare_messages,
    render,
)
from llm_training.warmup.split import WarmupSplit, split_warmup_examples

logger = logging.getLogger("sft_warmup")

#: Default task classes: the 9 SGR-agent classes + the two auxiliary
#: envelope-diversity tool-use classes.

DEFAULT_CLASSES: tuple[str, ...] = (
    *SGR_CLASSES,
    "opencode_gen",
    "tool_calls_multiturn",
)

#: Rehearsal classes (ON by default since the 2026-09 attribution): the first
#: warm-up run (5 epochs of full FT on the narrow SGR+tool pool) catastrophi-
#: cally forgot general capabilities — EN summarization collapsed 24→3 rouge1,
#: belebele −8 pp, ifeval −15 pp on the SFT/GRPO parents. Mixing ~39 % of
#: REHEARSAL data (all from the already-committed src_data catalog, with
#: references; the auxiliary-class routing sends them to train in full while
#: the val split stays SGR-only) preserves the general skills while the
#: envelope is being taught. RU: mtrageval (incl. its summarization subtask),
#: menon_qa, self_cognition, asr_summarization, rummlu_pro; EN: multihop_rag
#: (EN RAG QA), docsem_quant (EN PoT). The Meno-stand classes are NOT here —
#: their test split is fixed-val for GRPO anchors, and rehearsing them in the
#: warm-up would train on the anchor val rows.






# : fixed-val
REHEARSAL_CLASSES: tuple[str, ...] = (
    "mtrageval",
    "menon_qa",
    "self_cognition",
    "asr_summarization",
    "rummlu_pro",
    "multihop_rag",
    "docsem_quant",
)

#: Generation budget for the per-epoch validation (a closed JSON envelope +
#: the full answer must fit; the val split is pre-filtered so references fit
#: this budget). 4096 covers the longest reference span of the SGR val pool
#: (3718 tokens — a deep_research action turn; the coder finals are
#: 1989/2136) with room for the finalanswertool envelope. NOTE: this also
#: raises the ceiling for a degenerate generation (a rambling model can now
#: run up to 4096 tokens per val example).



# : deep_research


VAL_MAX_NEW_TOKENS = 4096

#: Generation budget per drift-probe prompt (the probe prompts are short;
#: the default mirrors grpo_train.py).

PROBE_MAX_NEW_TOKENS = 256


def _run_epoch_probe(
    model: "object",
    tokenizer: "object",
    probe_items: "list[object]",
    *,
    max_new_tokens: int = PROBE_MAX_NEW_TOKENS,
) -> dict | None:
    'Run the drift probe with failure containment; ``None`` on a failure.\n\n\n    A measurement failure is not degradation — the callers treat ``None`` as\n    "gate passes" and only log a warning (the same semantics as the GRPO\n    anchor probe). Kept module-level for test monkeypatching.'
    try:
        from llm_training.runtime.drift_probe import run_drift_probe

        return run_drift_probe(
            model, tokenizer, probe_items, max_new_tokens=max_new_tokens
        )
    except Exception as e:  # noqa: BLE001 - record + continue
        logger.warning("Drift probe failed; skipping it: %s", e)
        return None

#: Reference-length ceiling (tokens) for the val split — examples whose
#: reference is too long to be regenerated within VAL_MAX_NEW_TOKENS are
#: dropped from val (and RETURNED TO TRAIN by the caller — they are valid
#: SGR examples, only their references are budget-unfair for val) and
#: logged. 3800 = the 4096 generation budget minus ~296 tokens of the
#: finalanswertool envelope margin; the longest kept val reference span is
#: 3718 tokens.





VAL_MAX_REFERENCE_TOKENS = 3800


def _seed_everything(seed: int) -> None:
    'Seed Python / NumPy / torch (CPU) RNGs; log the seed (repo convention).\n\nNumPy'
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a core dep
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def build_warmup_data(
    tokenizer: Any,
    *,
    data_dir: Path,
    task_classes: Sequence[str],
    max_per_class: int,
    max_len: int,
    seed: int,
    val_ratio: float,
    exclude_tasks: Sequence[str],
) -> tuple[WarmupSplit, frozenset[str], dict[str, Any]]:
    'Build + split the warm-up dataset; return (split, tool_names, stats).'
    cfg = DataConfig(data_dir=data_dir, seed=seed, max_per_class=max_per_class)
    result = build_unified_dataset(
        tokenizer, cfg, task_classes=task_classes, include_network=False
    )
    pool = [*result.train, *result.val, *result.test]
    tool_names = collect_tool_names(pool)

    # Length filter: the warm-up trains on prompt+reference fits in max_len.
    
    kept: list[UnifiedExample] = []
    dropped_len = 0
    for ex in pool:
        res = mask_example(ex, tokenizer)
        if res.total_tokens <= max_len:
            kept.append(ex)
        else:
            dropped_len += 1

    split = split_warmup_examples(
        kept, seed=seed, val_ratio=val_ratio, exclude_tasks=exclude_tasks
    )
    stats = {
        "build": result.stats,
        "n_pool": len(pool),
        "n_after_length_filter": len(kept),
        "dropped_too_long": dropped_len,
        "n_train": len(split.train),
        "n_val": len(split.val),
        "excluded_examples": split.excluded_examples,
        "excluded_classes": split.excluded_classes,
        "strata": split.strata_stats,
    }
    return split, tool_names, stats


def to_prompt_completion(
    examples: Sequence[UnifiedExample], tokenizer: Any
) -> list[dict[str, str]]:
    'Convert unified examples to TRL **standard** (pre-rendered text) rows.\n\n\n    ``prompt`` is the chat-template render of the prompt messages with the\n    generation prompt appended (tools baked into the system message —\n    :func:`render_prompt`); ``completion`` is the reference suffix — the\n    difference between the full-conversation render and the prompt render,\n    which is exactly the reference span :func:`mask_example` localises (the\n    prompt/reference boundary is byte-identical to the eval/GRPO path).\n\n    Why pre-rendered TEXT and not the conversational message lists: pyarrow\n    (behind ``Dataset.from_list``) must unify one columnar schema across all\n    rows, and the ``tool_calls_multiturn`` records carry heterogeneous value\n    types in same-named ``arguments`` fields across records (int in the\n    Python-repr majority, string in the hybrid-JSON minority — e.g.\n    ``amount`` int vs str, ``product_ids[]`` int vs str) →\n    ``ArrowInvalid: cannot mix struct and non-struct, non-null values``.\n    A lossless cast is impossible (the template renders ``{"amount": 5}``\n    differently from ``{"amount": "5"}``), so the rows are rendered to plain\n    strings here — the very same strings the reward/eval pipeline renders —\n    which no pyarrow version can mis-type. TRL\'s standard prompt-completion\n    format keeps ``completion_only_loss`` support.\n    (int/str/None/bool) → ArrowInvalid; lossless'
    rows: list[dict[str, str]] = []
    for ex in examples:
        prompt_text = render_prompt(ex, tokenizer)
        prompt_messages, full_messages = prepare_messages(ex, tokenizer)
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


def filter_val_for_generation_budget(
    val_examples: Sequence[UnifiedExample], tokenizer: Any
) -> tuple[list[UnifiedExample], list[UnifiedExample]]:
    'Split val examples by the generation budget: (kept, dropped).\n\n\n    The dropped examples are returned as a LIST (not just counted) so the\n    caller can route them back to TRAIN — they are valid SGR examples, only\n    their references are too long to be regenerated fairly within\n    ``VAL_MAX_NEW_TOKENS``; without the routing they would silently vanish\n    from the warm-up entirely (the split put them in val, and the val filter\n    is the only thing that removes them).'
    kept: list[UnifiedExample] = []
    dropped: list[UnifiedExample] = []
    for ex in val_examples:
        res = mask_example(ex, tokenizer)
        if len(res.reference_ids) <= VAL_MAX_REFERENCE_TOKENS:
            kept.append(ex)
        else:
            dropped.append(ex)
    return kept, dropped


def run_generative_validation(
    model: Any,
    tokenizer: Any,
    val_examples: Sequence[UnifiedExample],
    known_tool_names: frozenset[str],
    *,
    dump_path: Path | None = None,
    epoch: float | None = None,
    step: int | None = None,
) -> dict[str, float]:
    'Greedy-generate the val split and score harmonic(format, chrF++).\n\n\n    Generation temporarily re-enables the KV-cache (``use_cache``): it is\n    ``False`` during training for gradient-checkpointing compatibility, but\n    greedy generation without the cache recomputes the full prefix at every\n    step (quadratic cost). The flag is restored in a ``finally`` block. When\n    ``dump_path`` is given, every scored generation is appended to that JSONL\n    (raw text + score + provenance) so metric artefacts are diagnosable\n    instantly.\n    ``dump_path``\n\n    Returns a breakdown ``{"aggregate": …, "final": …, "action": …,\n    "reasoning": …, "n": …}`` (per-kind means; kinds with no examples report\n    the aggregate\'s neutral 0.0 — check ``n`` fields via the log).'
    import json

    import torch

    from llm_training.warmup.split import turn_kind

    per_kind: dict[str, list[float]] = {"final": [], "action": [], "reasoning": []}
    model.eval()
    prev_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True
    dump_file = None
    try:
        if dump_path is not None:
            dump_file = open(dump_path, "a", encoding="utf-8")
        with torch.no_grad():
            for ex in val_examples:
                prompt_text = render_prompt(ex, tokenizer)
                inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
                out = model.generate(
                    **inputs,
                    max_new_tokens=VAL_MAX_NEW_TOKENS,
                    do_sample=False,
                )
                gen = decode_generation(
                    tokenizer, out[0][inputs["input_ids"].shape[1]:]
                )
                score = warmup_metrics.warmup_turn_score(
                    gen, ex.reference, ex.task_class, ex.extra, known_tool_names
                )
                per_kind[turn_kind(ex)].append(score)
                if dump_file is not None:
                    dump_file.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "step": step,
                                "task_class": ex.task_class,
                                "turn_kind": turn_kind(ex),
                                "source": ex.source,
                                "generation": gen,
                                "score": score,
                                "reference": ex.reference,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    dump_file.flush()
    finally:
        if dump_file is not None:
            dump_file.close()
        if prev_use_cache is not None:
            model.config.use_cache = prev_use_cache
        model.train()
    all_scores = [s for scores in per_kind.values() for s in scores]
    breakdown = {
        "aggregate": warmup_metrics.aggregate_warmup_scores(all_scores),
        "final": warmup_metrics.aggregate_warmup_scores(per_kind["final"]),
        "action": warmup_metrics.aggregate_warmup_scores(per_kind["action"]),
        "reasoning": warmup_metrics.aggregate_warmup_scores(per_kind["reasoning"]),
        "n": float(len(all_scores)),
    }
    return breakdown


def _build_sft_kwargs(
    sft_config_cls: Any,
    *,
    output_dir: str,
    epochs: int,
    lr: float,
    per_device_batch: int,
    grad_accum: int,
    max_len: int,
    optim: str,
    seed: int,
    liger: bool = False,
) -> dict[str, Any]:
    'Build SFTConfig kwargs adaptive to the installed TRL/transformers.\n\n\n    The stack moved under us between environments: the local dev env runs\n    TRL 0.20 over transformers 4.x, while the A100 cluster runs TRL 1.10 over\n    transformers v5.2+, where ``warmup_ratio`` was **removed** (passing it\n    raises ``TypeError: unexpected keyword argument``). Instead of pinning one\n    stack, build the kwargs and adapt them to the fields the installed\n    ``SFTConfig`` actually declares:\n\n    * ``warmup_ratio`` → kept where the field exists (transformers 4.x; the\n      local path is byte-identical to before); where it is gone, falls back to\n      ``warmup_steps=0.03`` — in transformers v5 a float < 1 is interpreted as\n      a **ratio** of training steps, preserving the semantics; if neither field\n      exists, warmup is dropped with a warning (non-critical).\n    * ``max_length`` (the TRL name in both 0.20 and 1.10; 1.10 defaults to\n      **1024**, which would silently truncate every long prompt — hence it is\n      always set explicitly). If the installed SFTConfig declares neither\n      ``max_length`` nor ``max_seq_length``, we raise: silent truncation to\n      some library default is unacceptable for this stage.\n    * ``gradient_checkpointing_kwargs={"use_reentrant": False}`` — the\n      non-reentrant checkpointing implementation plays nicer with inputs that\n      require grads and with Liger\'s patched modules (dropped automatically\n      on stacks that do not declare the field).\n    * ``use_liger_kernel=True`` only when ``liger=True`` AND the installed\n      SFTConfig declares the field; otherwise the caller patches the model\n      instance directly (see :func:`_apply_liger_to_model`).\n    * Any other kwarg the installed class does not know is dropped with a\n      warning (forward compatibility), never a crash.\n    ``warmup_steps``'
    from dataclasses import fields as dc_fields

    known = {f.name for f in dc_fields(sft_config_cls)}
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "learning_rate": lr,
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "lr_scheduler_type": "cosine",
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "completion_only_loss": True,
        "logging_steps": 5,
        "save_strategy": "no",  # checkpoints are saved by the callback
        "report_to": [],
        "seed": seed,
        "remove_unused_columns": False,
        "optim": optim,
        "max_length": max_len,
    }
    if liger:
        # Passed only when the field is declared (checked by the caller); the
        # generic drop below is a safety net, not the dispatch.
        
        
        kwargs["use_liger_kernel"] = True
    # Warmup: prefer the exact ratio field; fall back to the v5 ratio-as-float.
    
    if "warmup_ratio" in known:
        kwargs["warmup_ratio"] = 0.03
    elif "warmup_steps" in known:
        kwargs["warmup_steps"] = 0.03  # v5: float < 1 == ratio of steps
    else:
        logger.warning('SFTConfig declares neither warmup_ratio nor warmup_steps; LR warmup is disabled for this run.')

    # Length cap is mandatory: TRL 1.10 defaults max_length to 1024, silently
    # truncating long prompts. Accept the legacy max_seq_length name as a
    # fallback; if neither exists, refuse to run.
    
    
    if "max_length" not in known:
        if "max_seq_length" in known:
            kwargs["max_seq_length"] = kwargs.pop("max_length")
        else:
            raise RuntimeError(
                'The installed SFTConfig declares neither max_length nor max_seq_length; cannot guarantee the length cap (TRL would truncate to its own default). Refusing to run.'
            )

    # Drop anything the installed class does not know (forward compatibility).
    
    unknown = sorted(set(kwargs) - known)
    if unknown:
        logger.warning("SFTConfig dropped unsupported kwargs: %s", unknown)
        for key in unknown:
            kwargs.pop(key)
    return kwargs


def _apply_liger_to_model(model: Any) -> None:
    'Patch a loaded model instance with Liger kernels (the fallback path).\n\n\n    Used when the installed ``SFTConfig`` does not declare\n    ``use_liger_kernel`` (older stacks) but the operator passed ``--liger``.\n    ``apply_liger_kernel_to_qwen2(model=...)`` binds the fused\n    linear+cross-entropy forward to the instance — the ``[B, T, V]`` logits\n    tensor is never materialized — and swaps RMSNorm/SwiGLU/RoPE for fused\n    Triton kernels. Meno-Lite is a Qwen2 descendant, so the qwen2 patch\n    applies verbatim.\n    ``use_liger_kernel``\n    ``apply_liger_kernel_to_qwen2(model=...)``\nTriton. Meno-Lite'
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2
    except ImportError as e:
        raise SystemExit(
            "--liger requires the 'liger-kernel' package: pip install -U liger-kernel (needs >=0.7.0 for transformers v5), or drop the flag to train without Liger."
        ) from e
    apply_liger_kernel_to_qwen2(model=model)


def _build_arg_parser() -> argparse.ArgumentParser:
    'Build the sft_warmup.py CLI parser (separate for testability).'
    p = argparse.ArgumentParser(
        description=(
            'SFT warm-up: prime the tool-call format before GRPO (one-off stage).\nSFT warm-up'
        )
    )
    p.add_argument("--model", default="anonymous/base-model",
                   help='Base model to warm up.')
    p.add_argument("--data-dir", default="src_data",
                   help='Path to the src_data catalog.')
    p.add_argument("--prepared-dir", default=None, metavar="DIR",
                   help='Load the pre-built warm-up dataset from this cache directory (produced by prepare_dataset.py --pipeline warmup) instead of building + splitting + difficulty-scoring it inline. Skips all three phases; the hard-tail difficulty filter is re-applied from the cached scores with --difficulty-drop-hard.')
    p.add_argument("--class", action="append", dest="classes", default=None,
                   help='Task class to include (repeatable; default: 9 SGR-agent classes + opencode_gen + tool_calls_multiturn).')
    p.add_argument("--exclude-task", action="append", dest="exclude_tasks",
                   default=None,
                   help='Task class to exclude entirely, mirroring the future GRPO --val-task/--test-task hold-outs (repeatable).')
    p.add_argument("--rehearsal-task", action="append", dest="rehearsal_tasks",
                   default=None,
                   help='Extra rehearsal task class to ADD to the default rehearsal mix (repeatable).')
    p.add_argument("--no-rehearsal", action="store_true",
                   help='Disable the rehearsal mix entirely (the pre-r5 behaviour: 9 SGR classes + the two tool classes only')
    p.add_argument("--no-drift-probe", action="store_true",
                   help='Disable the drift probe in the per-epoch validation (auto-loaded from <data-dir>/drift_probe.jsonl).')
    p.add_argument("--warmup-probe-gate", type=float, default=0.10,
                   metavar="TOL",
                   help='The per-epoch probe gate: a best/ checkpoint is only saved for an epoch whose probe score stays within TOL below the pre-training baseline (default 0.10 ≈ 2 sigma of the 76-item binomial noise; 0 disables the gate')
    p.add_argument("--drift-probe-max-new-tokens", type=int,
                   default=PROBE_MAX_NEW_TOKENS,
                   help='Generation budget per drift-probe prompt.')
    p.add_argument("--difficulty-model", default="anonymous/base-model",
                   metavar="MODEL_ID",
                   help='The SEPARATE difficulty model whose reference loss scores the warm-up train pool (the ToolACE-style static filter); default the base Meno-Lite')
    p.add_argument("--difficulty-drop-hard", type=float, default=0.01,
                   metavar="SHARE",
                   help='Per-class share of the HARDEST (highest reference loss) train examples to drop (out-of-distribution')
    p.add_argument("--difficulty-drop-easy", type=float, default=0.0,
                   metavar="SHARE",
                   help='Per-class share of the EASIEST train examples to drop (default 0.0 in the warm-up')
    p.add_argument("--difficulty-batch-size", type=int, default=8,
                   help='Upper bound on the number of examples per difficulty scoring forward pass; the actual batches are also capped by a padded-token budget (16384) so the [B, T, V] logits transient stays bounded.')
    p.add_argument("--no-difficulty-progress", action="store_true",
                   help='Disable the tqdm progress bars of the difficulty scoring (they are forced ON in non-TTY SLURM logs by default; TQDM_DISABLE does not apply).')
    p.add_argument("--max-per-class", type=int, default=500,
                   help='Per-class cap applied by build_unified_dataset.')
    p.add_argument("--max-len", type=int, default=8192,
                   help='Max total tokens (prompt+reference); longer examples are dropped.')
    p.add_argument("--val-ratio", type=float, default=0.1,
                   help='Fraction of each SGR stratum (class × turn kind) held out for validation.')
    p.add_argument("--epochs", type=int, default=5,
                   help='Number of fine-tuning epochs (default 5; the best checkpoint is selected by the per-epoch generative val metric, so extra epochs cannot hurt the pick).')
    p.add_argument("--lr", type=float, default=1e-5,
                   help='Learning rate (full fine-tune).')
    p.add_argument("--per-device-batch", type=int, default=2,
                   help='Per-device batch size.')
    p.add_argument("--grad-accum", type=int, default=8,
                   help='Gradient accumulation steps (effective batch = per-device × devices × accum).')
    p.add_argument("--seed", type=int, default=42,
                   help='RNG seed (logged).')
    p.add_argument("--optim", default="adamw_torch",
                   help='Optimizer (transformers OptimizerNames). Default adamw_torch; on a single A100 an OOM fallback is --optim adamw_bnb_8bit (needs bitsandbytes) or adafactor (/ paged_adamw_8bit, whose optimizer states can be paged out under peak pressure).')
    p.add_argument("--attn", default="sdpa",
                   choices=("sdpa", "flash_attention_2", "eager"),
                   help='Attention implementation passed to from_pretrained. flash_attention_2 is recommended on A100: O(N) attention memory and no padded-batch fallbacks.')
    p.add_argument("--liger", action="store_true",
                   help='Enable Liger kernels (needs the liger-kernel package, >=0.7.0 for transformers v5): the fused linear+cross-entropy never materializes the [B, T, V] logits tensor')
    p.add_argument("--output-dir", default="checkpoints/warmup",
                   help='Where to save the best/ and last/ checkpoints.')
    p.add_argument("--max-shard-size", default=DEFAULT_MAX_SHARD_SIZE,
                   metavar="SIZE",
                   help='pins the shard size across transformers versions: v5.2+ changed the library default to 50GB, which would write one monolithic ~30 GB file; also hard-checks bfloat16 before every save).')
    p.add_argument("--save-steps", type=int, default=0,
                   help='If >0, also save intermediate checkpoints every N steps.')
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    args = _build_arg_parser().parse_args(argv)
    logger.info("sft_warmup: seed=%d", args.seed)
    _seed_everything(args.seed)

    # Lazy heavy imports (dev environments without the train extra stay light).
    
    try:
        import torch

        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "The warm-up needs 'torch', 'transformers', 'datasets' and 'trl': pip install -e '.[train]'."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    scores: dict[str, float] = {}
    if args.prepared_dir:
        # Load the pre-built cache: the FINAL warm-up pools (post stratified
        # split + val-budget return) + the per-source difficulty scores. The
        # build / split / difficulty-scoring phases are skipped entirely; the
        # hard-tail difficulty FILTER below is re-applied from the scores.
        
        
        cache = load_warmup_cache(Path(args.prepared_dir))
        if cache.seed != args.seed:
            logger.warning('Prepared cache %s was built with seed=%d but --seed=%d; the pool composition follows the cache, the RNG follows --seed.', args.prepared_dir, cache.seed, args.seed)
        split = WarmupSplit(train=list(cache.train), val=list(cache.val))
        val_examples = list(cache.val)
        tool_names = cache.tool_names
        scores = cache.scores
        logger.info(
            "Warm-up data: loaded prepared cache %s (train=%d, val=%d, "
            "%d scores); known tools=%d",
            args.prepared_dir, len(split.train), len(val_examples), len(scores),
            len(tool_names),
        )
        if not split.train:
            print("No training examples after filtering.", file=sys.stderr)
            return 1
    else:
        # Class composition: the explicit --task list wins (full manual
        # control); otherwise DEFAULT + REHEARSAL (the anti-forgetting mix,
        # --no-rehearsal reproduces the pre-r5 narrow configuration). The
        # Meno-stand classes are ALWAYS stripped — their test split is the
        # GRPO anchors' fixed-val pool; rehearsing them would train on the
        # anchor val rows.
        
        
        
        
        from llm_training.build import FIXED_VAL_CLASSES

        if args.classes:
            task_classes = list(args.classes)
        elif args.no_rehearsal:
            task_classes = list(DEFAULT_CLASSES)
        else:
            task_classes = [
                *DEFAULT_CLASSES,
                *REHEARSAL_CLASSES,
                *(args.rehearsal_tasks or []),
            ]
        dropped_fixed_val = [c for c in task_classes if c in FIXED_VAL_CLASSES]
        if dropped_fixed_val:
            task_classes = [c for c in task_classes if c not in FIXED_VAL_CLASSES]
            logger.warning("Dropping the Meno-stand class(es) %s from the warm-up pool: their test split is the GRPO anchors' fixed-val pool", dropped_fixed_val)
        exclude_tasks = args.exclude_tasks or []
        logger.info("Task classes: %s (excluded: %s)", task_classes, exclude_tasks)

        split, tool_names, stats = build_warmup_data(
            tokenizer,
            data_dir=Path(args.data_dir),
            task_classes=task_classes,
            max_per_class=args.max_per_class,
            max_len=args.max_len,
            seed=args.seed,
            val_ratio=args.val_ratio,
            exclude_tasks=exclude_tasks,
        )
        logger.info(
            "Warm-up data: train=%d, val=%d (pool=%d, dropped_too_long=%d, "
            "excluded=%d); known tools=%d",
            stats["n_train"], stats["n_val"], stats["n_pool"],
            stats["dropped_too_long"], stats["excluded_examples"],
            len(tool_names),
        )
        if stats["n_train"] == 0:
            print("No training examples after filtering.", file=sys.stderr)
            return 1

        # Val split: drop references too long for the val generation budget and
        # RETURN them to TRAIN — the measuring instrument stays budget-fair, and
        # the examples themselves (valid SGR turns, only their references are
        # too long to regenerate within the budget) are not lost. Runs BEFORE
        # the difficulty filter, so the returned examples are treated like any
        # other train example.
        
        
        
        
        
        
        val_examples, val_dropped = filter_val_for_generation_budget(
            split.val, tokenizer
        )
        if val_dropped:
            split.train.extend(val_dropped)
            logger.info(
                "Val: dropped %d examples with references longer than %d tokens "
                "(returned to train; warm-up train pool now %d).",
                len(val_dropped), VAL_MAX_REFERENCE_TOKENS, len(split.train),
            )

        # Difficulty SCORING (the expensive bf16 forward pass) — run inline
        # only without --prepared-dir; the FILTER itself is shared below.
        
        
        if args.difficulty_model and args.difficulty_drop_hard > 0:
            logger.info(
                "Difficulty filter: scoring the warm-up train pool with %s "
                "(drop hard=%.3f)…",
                args.difficulty_model, args.difficulty_drop_hard,
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
                        diff_model, diff_tokenizer, split.train,
                        batch_size=args.difficulty_batch_size,
                        progress=not args.no_difficulty_progress,
                        desc="Difficulty scoring (warm-up train pool)",
                    )
                finally:
                    del diff_model
                    import gc

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001 - never abort the run
                logger.warning('Difficulty scoring failed (%s: %s); the filter is DISABLED and the full warm-up pool is kept.', type(e).__name__, e)
                scores = {}

    # Difficulty filter (loss-based, ToolACE-style; the HARD tail only — a
    # data-quality guard). The rehearsal mix rides the train pool for
    # RETENTION, not learning, so the easy tail is NOT cut in the warm-up;
    # the hard tail (out-of-distribution / mis-labelled noise in the
    # distilled references) is dropped per-class by the reference loss of a
    # SEPARATE difficulty model (the base — the "untrained" reference). The
    # SCORING is either cached (--prepared-dir) or was just computed above;
    # this filter is a pure per-class quantile — instant in both cases. The
    # val split is never scored (the measuring instrument).
    
    
    
    
    
    if (
        args.difficulty_model
        and args.difficulty_drop_hard > 0
        and scores
    ):
        from llm_training.runtime.difficulty import filter_by_difficulty

        kept, diff_stats = filter_by_difficulty(
            split.train, scores,
            drop_easy=args.difficulty_drop_easy,
            drop_hard=args.difficulty_drop_hard,
        )
        split.train = kept
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
            "Difficulty filter: warm-up train pool now %d examples.",
            len(split.train),
        )

    # torch_dtype is bf16 EXPLICITLY (not "auto"): the A100 cluster policy
    # requires every checkpoint (best/ feeds GRPO) in bfloat16, and "auto"
    # would silently follow the source checkpoint's config dtype.
    # torch_dtype
    
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )
    model.config.use_cache = False  # gradient checkpointing compatibility
    # Greedy per-epoch validation must not carry the checkpoint's sampling
    # defaults — silences the per-generate() "generation flags are not
    # valid" warning spam (greedy outputs are unaffected by the flags).
    
    
    
    from llm_training.runtime.generation_config import (
        unset_greedy_sampling_flags,
    )

    unset_greedy_sampling_flags(model)
    logger.info(
        "Model loaded in %s with attn_implementation=%s",
        model.dtype, args.attn,
    )

    # Liger: prefer the Trainer-level use_liger_kernel flag (the installed
    # SFTConfig dispatches by model_type); when the field is absent (older
    # stacks), patch the loaded instance directly. Without --liger nothing
    # here runs — the default path stays byte-identical.
    
    # model_type
    
    
    liger_via_config = False
    if args.liger:
        from dataclasses import fields as dc_fields

        if "use_liger_kernel" in {f.name for f in dc_fields(SFTConfig)}:
            liger_via_config = True
            logger.info(
                "Liger: enabled via SFTConfig.use_liger_kernel "
                "(Trainer auto-patches by model_type)."
            )
        else:
            _apply_liger_to_model(model)
            logger.info(
                "Liger: applied apply_liger_kernel_to_qwen2 to the model "
                "instance (SFTConfig has no use_liger_kernel field)."
            )

    train_rows = to_prompt_completion(split.train, tokenizer)
    trainer_cfg = SFTConfig(
        **_build_sft_kwargs(
            SFTConfig,
            output_dir=args.output_dir,
            epochs=args.epochs,
            lr=args.lr,
            per_device_batch=args.per_device_batch,
            grad_accum=args.grad_accum,
            max_len=args.max_len,
            optim=args.optim,
            seed=args.seed,
            liger=liger_via_config,
        )
    )
    trainer = SFTTrainer(
        model=model,
        args=trainer_cfg,
        train_dataset=Dataset.from_list(train_rows),
        processing_class=tokenizer,
    )

    # Per-epoch generative validation + best-checkpoint selection.
    
    best_score = -1.0
    output_dir = Path(args.output_dir)

    # The drift probe (the 2026-09 attribution: the warm-up's SGR-only val
    # could not see the catastrophic forgetting). The BASELINE runs on the
    # loaded model BEFORE training — the pre-warm-up reference point; every
    # epoch then reports probe=… (Δ…) and the best/ selection is GATED: an
    # epoch may only claim best/ while its probe stays within the tolerance
    # below the baseline (a measurement failure passes the gate — the same
    # semantics as the GRPO anchor probe).
    
    
    probe_items: list = []
    if not args.no_drift_probe:
        probe_path = Path(args.data_dir) / "drift_probe.jsonl"
        if probe_path.is_file():
            from llm_training.runtime.drift_probe import load_probe

            probe_items = load_probe(probe_path)
            logger.info(
                "Drift probe: %d items from %s (gate=%.3f)",
                len(probe_items), probe_path, args.warmup_probe_gate,
            )
        else:
            logger.warning('Drift probe disabled: no %s (run scripts/build_drift_probe.py).', probe_path)
    probe_baseline: float | None = None
    if probe_items:
        base_probe = _run_epoch_probe(
            model, tokenizer, probe_items,
            max_new_tokens=args.drift_probe_max_new_tokens,
        )
        if base_probe is not None:
            probe_baseline = base_probe["score"]
            logger.info(
                "Drift probe baseline (pre-training): %.4f (n=%d); gate "
                "tol %.3f → best/ requires probe >= %.4f",
                probe_baseline, int(base_probe["n"]), args.warmup_probe_gate,
                probe_baseline - args.warmup_probe_gate,
            )

    class _GenerativeValCallback(TrainerCallback):  # noqa: D401 - tiny local class
        def on_epoch_end(self, targs, state, control, **kwargs):
            nonlocal best_score
            # Main-process only: generation, logging and saving must happen
            # exactly once even under a multi-process accelerate launch.
            
            
            if not trainer.accelerator.is_main_process:
                return
            if not val_examples:
                logger.info("Val split empty; skipping generative validation.")
                return
            m = kwargs.get("model") or trainer.model
            breakdown = run_generative_validation(
                m,
                tokenizer,
                val_examples,
                tool_names,
                dump_path=output_dir / "val_generations.jsonl",
                epoch=float(state.epoch or 0.0),
                step=int(state.global_step),
            )
            # The epoch's drift probe — BEFORE the best/ decision (the gate).
            
            probe = None
            if probe_items:
                probe = _run_epoch_probe(
                    m, tokenizer, probe_items,
                    max_new_tokens=args.drift_probe_max_new_tokens,
                )
            probe_ok = True
            if probe is not None:
                if probe_baseline is None:
                    probe_baseline_new = probe["score"]
                    logger.info(
                        "PROBE (no pre-training baseline; this epoch becomes "
                        "the reference): %.4f", probe_baseline_new,
                    )
                else:
                    delta = probe["score"] - probe_baseline
                    logger.info(
                        "PROBE epoch=%d: score=%.4f (Δ%+.4f vs %.4f, n=%d) "
                        "per_check=%s",
                        round(state.epoch or 0.0), probe["score"], delta,
                        probe_baseline, int(probe["n"]),
                        {k: round(v["pass_rate"], 3)
                         for k, v in probe["per_check"].items()},
                    )
                    if args.warmup_probe_gate > 0 and -delta > args.warmup_probe_gate:
                        probe_ok = False
            logger.info(
                "Epoch %d val: aggregate=%.4f final=%.4f action=%.4f "
                "reasoning=%.4f (n=%d)",
                round(state.epoch or 0.0), breakdown["aggregate"],
                breakdown["final"], breakdown["action"],
                breakdown["reasoning"], int(breakdown["n"]),
            )
            if breakdown["aggregate"] > best_score and probe_ok:
                best_score = breakdown["aggregate"]
                best_dir = output_dir / "best"
                logger.info("New best aggregate=%.4f → saving %s", best_score, best_dir)
                with sharded_checkpoint_saves(trainer, args.max_shard_size):
                    trainer.save_model(str(best_dir))
                tokenizer.save_pretrained(str(best_dir))
            elif breakdown["aggregate"] > best_score and not probe_ok:
                logger.warning('general instruction-following degraded beyond the tolerance; the epoch is skipped for selection.')

    trainer.add_callback(_GenerativeValCallback())
    trainer.train()

    last_dir = output_dir / "last"
    with sharded_checkpoint_saves(trainer, args.max_shard_size):
        trainer.save_model(str(last_dir))
    tokenizer.save_pretrained(str(last_dir))
    logger.info(
        "Warm-up done. best aggregate=%.4f; checkpoints: %s, %s",
        best_score, output_dir / "best", last_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
