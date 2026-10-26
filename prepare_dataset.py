"Offline dataset preparation: build + difficulty-score once, cache for reuse.\n\n\nThe expensive, deterministic data phases of ``grpo_train.py`` and\n``sft_warmup.py`` — building the unified dataset (loading every class + the\n``mask_example`` length filter) and the loss-based difficulty scoring (a bf16\nforward pass of a separate model over the train pool) — are run here ONCE and\ncached to disk. The training scripts then consume the cache via\n``--prepared-dir`` and skip both phases, re-applying only the cheap,\ninstant steps (the per-class quantile difficulty filter from the cached\nscores, the balanced-stream assembly, typo augmentation, prompt rendering).\n\nTwo pipelines\n\n* ``--pipeline grpo`` — ``build_full_dataset`` (full uncapped train pools +\n  the frozen anchor val pool, hold-out classes via repeatable ``--val-task``)\n  + difficulty scoring over the whole train pool. Cache layout:\n  ``train_pools.jsonl`` / ``val.jsonl`` / ``scores.json`` / ``meta.json``.\n* ``--pipeline warmup`` — the warm-up build + stratified SGR split +\n  val-generation-budget return + difficulty scoring of the train pool.\n  Cache layout: ``train.jsonl`` / ``val.jsonl`` / ``scores.json`` /\n  ``meta.json``.\n\nThe difficulty **scores** are cached, not the filtered result, so the training\nscripts keep their own ``--difficulty-drop-*`` shares (re-filtering is a pure\nPython quantile over the cached scores — instant).\n\n\nUsage\n    # GRPO (mirrors grpo_train.py's data flags; the tokenizer must match the\n    # one grpo_train.py / sft_warmup.py use — the Meno-Lite tokenizer):\n    python prepare_dataset.py --pipeline grpo --output-dir cache/grpo         --val-task summarizer math_reasoner --include-network --seed 24\n\n    # warm-up:\n    python prepare_dataset.py --pipeline warmup --output-dir cache/warmup         --seed 42 --max-len 8192"

from __future__ import annotations

import argparse
import gc
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from llm_training.build import FIXED_VAL_CLASSES, build_full_dataset
from llm_training.config import DataConfig
from llm_training.loaders import collect_tool_names, collect_tool_schemas
from llm_training.prepare import save_grpo_cache, save_warmup_cache
from llm_training.schema import UnifiedExample

logger = logging.getLogger("prepare_dataset")

#: The default GRPO hold-out classes (mirrors grpo_train.py's
#: ``RECOMMENDED_VAL_TASKS`` — keep them in sync across the stages).

RECOMMENDED_VAL_TASKS = ("summarizer", "math_reasoner")


def _score_difficulty(
    examples: Sequence[UnifiedExample],
    model_id: str,
    *,
    attn: str,
    batch_size: int,
    progress: bool,
    desc: str,
) -> dict[str, float]:
    'Score ``examples`` with the difficulty model; free it from VRAM.\n\n\n    Returns ``{}`` when ``model_id`` is empty (scoring disabled).'
    if not model_id:
        return {}
    import torch

    from llm_training.runtime.difficulty import (
        load_difficulty_model,
        score_example_difficulty,
    )

    diff_model, diff_tokenizer = load_difficulty_model(model_id, attn=attn)
    try:
        return score_example_difficulty(
            diff_model, diff_tokenizer, examples,
            batch_size=batch_size, progress=progress, desc=desc,
        )
    finally:
        del diff_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _import_sft_warmup() -> Any:
    'Load ``sft_warmup.py`` (repo-root script) by path, like the tests do.'
    script = Path(__file__).resolve().parent / "sft_warmup.py"
    spec = importlib.util.spec_from_file_location("_sft_warmup", script)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise SystemExit(f"Cannot load {script}.")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_grpo(args: argparse.Namespace) -> int:
    'Build + difficulty-score the GRPO dataset; write the cache.'
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cfg = DataConfig(
        data_dir=Path(args.data_dir), seed=args.seed,
        max_per_class=args.max_per_class, nerel_split="train",
        sgr_loop_subdir=args.sgr_loop_subdir,
    )
    result = build_full_dataset(
        tokenizer, cfg,
        val_task=args.val_task,
        include_network=args.include_network,
    )
    all_train = [
        ex for cls in sorted(result.train_pools) for ex in result.train_pools[cls]
    ]
    tool_names = collect_tool_names([*all_train, *result.val])
    tool_schemas = collect_tool_schemas([*all_train, *result.val])
    logger.info(
        "GRPO data: %d classes, %d train examples, %d val examples, "
        "%d tool names (%d with schemas); val_tasks=%s.",
        len(result.train_pools), len(all_train), len(result.val),
        len(tool_names), len(tool_schemas), list(result.val_task),
    )
    if not all_train:
        print("No training examples after the hold-out split.", file=sys.stderr)
        return 1

    scores = _score_difficulty(
        all_train, args.difficulty_model,
        attn=args.attn, batch_size=args.difficulty_batch_size,
        progress=not args.no_difficulty_progress,
        desc="Difficulty scoring (train pool)",
    )
    meta = save_grpo_cache(
        Path(args.output_dir),
        train_pools=result.train_pools,
        val=result.val,
        scores=scores,
        seed=args.seed,
        val_tasks=result.val_task,
        include_network=args.include_network,
        max_per_class=args.max_per_class,
        difficulty_model=args.difficulty_model,
        tool_names=tool_names,
        tool_schemas=tool_schemas,
        stats=result.stats,
    )
    logger.info(
        "GRPO cache written to %s (%d train, %d val, %d scores).",
        args.output_dir, meta["n_train"], meta["n_val"], meta["n_scores"],
    )
    return 0


def _run_warmup(args: argparse.Namespace) -> int:
    'Build + split + difficulty-score the warm-up dataset; write the cache.'
    from transformers import AutoTokenizer

    sw = _import_sft_warmup()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Class composition (mirrors sft_warmup.py main()): the explicit --class
    # list wins; otherwise DEFAULT + REHEARSAL (--no-rehearsal = narrow); the
    # Meno-stand classes are always stripped (their test split is fixed-val).
    
    if args.classes:
        task_classes = list(args.classes)
    elif args.no_rehearsal:
        task_classes = list(sw.DEFAULT_CLASSES)
    else:
        task_classes = [
            *sw.DEFAULT_CLASSES,
            *sw.REHEARSAL_CLASSES,
            *(args.rehearsal_tasks or []),
        ]
    task_classes = [c for c in task_classes if c not in FIXED_VAL_CLASSES]
    exclude_tasks = args.exclude_tasks or []

    split, tool_names, stats = sw.build_warmup_data(
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
        "excluded=%d); known tools=%d.",
        stats["n_train"], stats["n_val"], stats["n_pool"],
        stats["dropped_too_long"], stats["excluded_examples"], len(tool_names),
    )
    if stats["n_train"] == 0:
        print("No training examples after filtering.", file=sys.stderr)
        return 1

    # Val generation budget: drop too-long references and RETURN them to train
    # (mirrors sft_warmup.py main() — runs BEFORE the difficulty scoring).
    
    val_examples, val_dropped = sw.filter_val_for_generation_budget(
        split.val, tokenizer
    )
    if val_dropped:
        split.train.extend(val_dropped)
    stats["val_budget_dropped"] = len(val_dropped)

    scores = _score_difficulty(
        split.train, args.difficulty_model,
        attn=args.attn, batch_size=args.difficulty_batch_size,
        progress=not args.no_difficulty_progress,
        desc="Difficulty scoring (warm-up train pool)",
    )
    meta = save_warmup_cache(
        Path(args.output_dir),
        train=split.train,
        val=val_examples,
        scores=scores,
        seed=args.seed,
        difficulty_model=args.difficulty_model,
        tool_names=tool_names,
        stats=stats,
    )
    logger.info(
        "Warm-up cache written to %s (%d train, %d val, %d scores).",
        args.output_dir, meta["n_train"], meta["n_val"], meta["n_scores"],
    )
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    'Build the prepare_dataset.py CLI parser.'
    p = argparse.ArgumentParser(
        description=(
            'Offline dataset preparation: build + difficulty-score once, cache for grpo_train.py / sft_warmup.py (--prepared-dir).'
        )
    )
    p.add_argument("--pipeline", choices=("grpo", "warmup"), default="grpo",
                   help="Which training stage's dataset to prepare.")
    p.add_argument("--output-dir", required=True,
                   help='Cache directory.')
    p.add_argument("--model", default="anonymous/base-model",
                   help='Tokenizer model for length accounting/rendering (the Meno-Lite tokenizer; must match the training scripts).')
    p.add_argument("--data-dir", default="src_data",
                   help='Path to the src_data catalog.')
    p.add_argument("--seed", type=int, default=42,
                   help='RNG seed (logged).')
    p.add_argument("--max-per-class", type=int, default=200,
                   help='Per-class cap of the FROZEN hold-out pools (GRPO)')
    p.add_argument("--sgr-loop-subdir", default=None,
                   help='subdir with the loop-format SGR catalog (reasoningtool; see scripts/convert_sgr_to_loop.py) preferred per-file by the SGR loader; classes without files there fall back to sgr_agent')
    p.add_argument("--difficulty-model", default="anonymous/base-model",
                   metavar="MODEL_ID",
                   help='The separate difficulty model whose reference loss scores the train pool; empty string disables scoring.')
    p.add_argument("--difficulty-batch-size", type=int, default=8,
                   help='Upper bound on examples per difficulty forward pass.')
    p.add_argument("--no-difficulty-progress", action="store_true",
                   help='Disable the difficulty-scoring tqdm bars.')
    p.add_argument("--attn", default="sdpa",
                   choices=("sdpa", "flash_attention_2", "eager"),
                   help='Attention implementation for the difficulty model.')

    # GRPO pipeline
    p.add_argument("--val-task", nargs="+", default=list(RECOMMENDED_VAL_TASKS),
                   metavar="CLS",
                   help=f'Hold-out class(es) routed into the anchor val pool (default: {list(RECOMMENDED_VAL_TASKS)}')
    p.add_argument("--include-network", action="store_true",
                   help='Include network-only classes (nerel_instruct train split).')

    # Warm-up pipeline
    p.add_argument("--class", action="append", dest="classes", default=None,
                   help='Warm-up task class to include (repeatable).')
    p.add_argument("--exclude-task", action="append", dest="exclude_tasks",
                   default=None,
                   help='Warm-up task class to exclude entirely (repeatable).')
    p.add_argument("--rehearsal-task", action="append", dest="rehearsal_tasks",
                   default=None,
                   help='Extra rehearsal class to add (repeatable).')
    p.add_argument("--no-rehearsal", action="store_true",
                   help='Disable the rehearsal mix (the narrow pre-r5 configuration).')
    p.add_argument("--max-len", type=int, default=8192,
                   help='Max total tokens (prompt+reference) for the warm-up.')
    p.add_argument("--val-ratio", type=float, default=0.1,
                   help='Fraction of each SGR stratum held out for warm-up validation.')
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    args = _build_arg_parser().parse_args(argv)
    logger.info("prepare_dataset: pipeline=%s seed=%d", args.pipeline, args.seed)
    if args.pipeline == "grpo":
        return _run_grpo(args)
    return _run_warmup(args)


if __name__ == "__main__":
    raise SystemExit(main())
