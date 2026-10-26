'CPT stage 1: pre-tokenize the SFT+PT corpus into a disk cache (offline).\n\n\nRun BEFORE training\n\n    python cpt_tokenize.py         --tokenizer anonymous/base-model         --sft-data sft1.jsonl --sft-data sft2.jsonl         --pt-data pt1.jsonl --pt-data pt2.jsonl         --val-sft-data val.jsonl         --cache-dir cpt_cache --max-len 4096 --seed 42 --num-proc 32\n\nWhy a separate stage\n    The corpus reaches several billion tokens; tokenization is a one-off\n    CPU-bound pass, and training must not repeat it. The cache is a\n    memory-mapped Arrow dataset (``<cache-dir>/train``, ``<cache-dir>/val``):\n    billions of int32 ids (~4 bytes/token) are paged in by the OS as the\n    mixed sampler reads random rows — nothing large lives in RAM, and all 10\n    DeepSpeed ranks share one OS page cache.\n\n\nSpeed\n    ``--num-proc`` workers (default ``min(32, cpu_count)``) each run the\n    batched Rust fast tokenizer with ``TOKENIZERS_PARALLELISM=false`` (the\n    parallelism comes from the workers, not from within them); PT chunking\n    and SFT prompt masking happen inside the same map pass. Re-running with\n    an unchanged corpus/tokenizer/``--max-len`` is a no-op (fingerprint in\n    ``meta.json``); ``--overwrite`` rebuilds.\n\n    ``--num-proc``\n\nThe tokenizer choice MUST match the training run\n    ``--tokenizer <model_id>`` loads the foreign tokenizer (and, at train\n    time, the donor vocab-sized layers) from another model; without it the\n    tokenizer comes from ``--model``. ``cpt_train.py`` verifies the cache\n    ``meta.json`` against its own ``--tokenizer``/``--model`` and refuses to\n    run on a mismatch.'

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

logger = logging.getLogger("cpt_tokenize")

META_VERSION = 1


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


def _file_descriptor(path: str | Path) -> dict[str, object]:
    p = Path(path)
    st = p.stat()
    return {"path": str(p), "size": st.st_size, "mtime": st.st_mtime}


def _sources_match(
    meta: dict, tokenizer_id: str, max_len: int, sources: dict[str, list]
) -> bool:
    'Fingerprint check: same tokenizer, same max_len, same source files.'
    if meta.get("version") != META_VERSION:
        return False
    if meta.get("tokenizer") != tokenizer_id or meta.get("max_len") != max_len:
        return False
    for group, descriptors in sources.items():
        if meta.get("sources", {}).get(group) != descriptors:
            return False
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            'CPT stage 1: pre-tokenize the SFT+PT corpus into a disk cache.'
        )
    )
    p.add_argument(
        "--sft-data", action="append", default=None,
        help='Path to an SFT jsonl ({system, query, response, history}); repeatable.',
    )
    p.add_argument(
        "--pt-data", action="append", default=None,
        help='Path to an unlabeled PT jsonl ({response}); repeatable.',
    )
    p.add_argument(
        "--val-sft-data", action="append", default=None,
        help='Path to an SFT jsonl used for chrF++ validation; repeatable.',
    )
    p.add_argument(
        "--model", default="anonymous/base-model",
        help='Model whose tokenizer is used when --tokenizer is absent.',
    )
    p.add_argument(
        "--tokenizer", default=None,
        help='Foreign tokenizer (model id): the tokenizer AND the donor of the vocab-sized layers at train time. Must match cpt_train.py.',
    )
    p.add_argument(
        "--max-len", type=int, default=4096,
        help='Max tokens per row: SFT rows longer than this are dropped, PT documents are chunked.',
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help='RNG seed (logged).',
    )
    p.add_argument(
        "--cache-dir", default="cpt_cache",
        help='Where to write the train/ val/ cache and meta.json.',
    )
    p.add_argument(
        "--num-proc", type=int, default=0,
        help='Tokenizer map workers (0 = min(32, cpu_count)).',
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help='Rebuild the cache even if the fingerprint matches.',
    )
    return p


def _sum_tokens(dataset) -> int:
    'Sum the n_tokens column without materializing Python objects.'
    import pyarrow.compute as pc

    return int(pc.sum(dataset.data.column("n_tokens")).as_py() or 0)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    args = build_arg_parser().parse_args(argv)
    if not args.sft_data and not args.pt_data:
        print(
            'At least one --sft-data or --pt-data file is required.',
            file=sys.stderr,
        )
        return 1
    logger.info("cpt_tokenize: seed=%d", args.seed)
    _seed_everything(args.seed)

    cache_dir = Path(args.cache_dir)
    tokenizer_id = args.tokenizer or args.model
    sources = {
        group: [_file_descriptor(p) for p in paths or []]
        for group, paths in (
            ("sft", args.sft_data), ("pt", args.pt_data),
            ("val", args.val_sft_data),
        )
    }
    meta_path = cache_dir / "meta.json"
    if meta_path.exists() and not args.overwrite:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if _sources_match(meta, tokenizer_id, args.max_len, sources):
            logger.info('Cache at %s is up to date (fingerprint matches); skipping. Use --overwrite to rebuild.\n--overwrite.', cache_dir)
            return 0
        logger.warning('Cache fingerprint mismatch; rebuilding (use --overwrite to silence).')

    # Keep datasets' intermediate caches next to the final artifacts (the
    # default ~/.cache may be tiny on cluster nodes); set BEFORE importing
    # datasets. Removed on success, kept for post-mortem on a crash.
    
    
    
    intermediate_dir = cache_dir / "_intermediate"
    os.environ["HF_DATASETS_CACHE"] = str(intermediate_dir)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    num_proc = args.num_proc or min(32, os.cpu_count() or 1)
    # Workers provide the process-level parallelism; disable the Rust-internal
    # one to avoid oversubscription (with a single worker, enable it instead).
    
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false" if num_proc > 1 else "true"

    from datasets import Dataset, Features, Sequence, Value

    from llm_training.cpt.data import (
        build_val_rows,
        iter_pt_records,
        iter_sft_records,
        tokenize_pt_batch,
        tokenize_sft_batch,
    )

    out_features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "n_prompt": Value("int32"),
            "kind": Value("string"),
            "source": Value("string"),
            "n_tokens": Value("int32"),
        }
    )
    parts: list[Dataset] = []
    raw_counts: dict[str, int] = {}

    if args.sft_data:
        raw_sft = Dataset.from_generator(
            partial(iter_sft_records, tuple(args.sft_data)),
            features=Features(
                {
                    "system": Value("string"),
                    "query": Value("string"),
                    "response": Value("string"),
                    "history": Sequence(Sequence(Value("string"))),
                    "source": Value("string"),
                }
            ),
        )
        raw_counts["raw_sft"] = len(raw_sft)
        ds_sft = raw_sft.map(
            partial(tokenize_sft_batch, tokenizer=tokenizer, max_len=args.max_len),
            batched=True,
            batch_size=256,
            num_proc=num_proc,
            remove_columns=raw_sft.column_names,
            features=out_features,
            desc="Tokenizing SFT",
        )
        logger.info('SFT: %d records → %d rows (drops: too-long/boundary = %d).', raw_counts['raw_sft'], len(ds_sft), raw_counts['raw_sft'] - len(ds_sft))
        parts.append(ds_sft)

    if args.pt_data:
        raw_pt = Dataset.from_generator(
            partial(iter_pt_records, tuple(args.pt_data)),
            features=Features(
                {"response": Value("string"), "source": Value("string")}
            ),
        )
        raw_counts["raw_pt_docs"] = len(raw_pt)
        ds_pt = raw_pt.map(
            partial(tokenize_pt_batch, tokenizer=tokenizer, max_len=args.max_len),
            batched=True,
            batch_size=64,
            num_proc=num_proc,
            remove_columns=raw_pt.column_names,
            features=out_features,
            desc="Tokenizing PT",
        )
        logger.info('PT: %d documents → %d chunk rows (long documents are chunked, not dropped).', raw_counts['raw_pt_docs'], len(ds_pt))
        parts.append(ds_pt)

    if not parts:
        print(
            "Nothing to tokenize.", file=sys.stderr,
        )
        return 1
    train_ds = parts[0] if len(parts) == 1 else _concatenate(parts)

    val_rows: list[dict] = []
    if args.val_sft_data:
        val_records = list(iter_sft_records(args.val_sft_data))
        val_rows = build_val_rows(val_records, tokenizer)
        logger.info('Val: %d SFT records → %d rows.\nVal: %d SFT', len(val_records), len(val_rows), len(val_records))
    else:
        logger.warning('No --val-sft-data given: the training stage will have no chrF++ validation (best == last, early stopping disabled).')

    train_dir = cache_dir / "train"
    val_dir = cache_dir / "val"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)
    train_ds.save_to_disk(str(train_dir))

    val_features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "prompt_text": Value("string"),
            "reference": Value("string"),
            "source": Value("string"),
            "n_reference_tokens": Value("int32"),
        }
    )
    val_ds = Dataset.from_list(val_rows, features=val_features)
    val_ds.save_to_disk(str(val_dir))

    total_tokens = _sum_tokens(train_ds)
    kinds = train_ds.data.column("kind").to_pylist()
    meta = {
        "version": META_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "tokenizer": tokenizer_id,
        "vocab_size": len(tokenizer),
        "max_len": args.max_len,
        "seed": args.seed,
        "sources": sources,
        "counts": {
            **raw_counts,
            "sft_rows": kinds.count("sft"),
            "pt_rows": kinds.count("pt"),
            "train_rows": len(train_ds),
            "total_tokens": total_tokens,
            "val_rows": len(val_rows),
        },
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    logger.info('Cache written: %s (train=%d rows, %.2fB tokens, val=%d rows).', cache_dir, len(train_ds), total_tokens / 1000000000.0, len(val_rows))
    shutil.rmtree(intermediate_dir, ignore_errors=True)
    return 0


def _concatenate(parts: list) -> object:
    from datasets import concatenate_datasets

    return concatenate_datasets(parts)


if __name__ == "__main__":
    raise SystemExit(main())
