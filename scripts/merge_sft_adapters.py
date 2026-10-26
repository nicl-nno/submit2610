'Merge a saved SFT LoRA/OPLoRA adapter into the base weights, offline.\n\n\nWhy\n``last-merged/`` and ``best-merged/`` AFTER the training loop, and a crash in\nthat tail (the 2026-09-26 compute-node run: the OPLoRA SFT finished all 782 steps,\n``last/`` and every ``best-<step>/`` adapter were safely on disk, then the\nfinal merge died) must not cost a re-train. The adapter directories carry\neverything needed: peft re-loads them onto a fresh base copy, and for OPLoRA\nthe projection is rebuilt from the FROZEN base weights — a deterministic\ntruncated SVD, so the offline merge is mathematically identical to the\nin-training one. The heavy lifting lives in\n:func:`llm_training.runtime.checkpointing.merge_adapter_into_base`; this\nscript is a thin CLI over it.\n:func:`llm_training.runtime.checkpointing.merge_adapter_into_base`\n\nUsage\n\n    ls /path/to/storage/models/meno-lite-0.2-checkpoints/sft-full/   # find best-<step>\n    python scripts/merge_sft_adapters.py         --model /path/to/storage/models/meno-lite-0.1         --adapter /path/to/storage/models/meno-lite-0.2-checkpoints/sft-full/best-<N>         --output-dir /path/to/storage/models/meno-lite-0.2-checkpoints/sft-full/best-merged         --oplora --oplora-k 128\n    python scripts/merge_sft_adapters.py         --model /path/to/storage/models/meno-lite-0.1         --adapter /path/to/storage/models/meno-lite-0.2-checkpoints/sft-full/last         --output-dir /path/to/storage/models/meno-lite-0.2-checkpoints/sft-full/last-merged         --oplora --oplora-k 128\n\nMemory\nworkspace when ``--oplora`` is set — ``--device cpu`` wants ~2x model size of\nhost RAM, ``--device cuda`` (the default when a GPU is visible) a free card;\non a shared GPU node run inside your own allocation. Without ``--oplora``\nthere is no SVD and the merge is a single fused weight update.\n``--device cuda``'

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from llm_training.runtime.checkpointing import (
    DEFAULT_MAX_SHARD_SIZE,
    merge_adapter_into_base,
)

logger = logging.getLogger("merge_sft_adapters")

#: Tokenizer files that mark a directory as carrying a tokenizer (peft's
#: ``trainer.save_model`` writes them into the adapter directory).

# : (peft ``trainer.save_model``
_TOKENIZER_MARKERS: tuple[str, ...] = (
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
)


def resolve_tokenizer_path(
    explicit: str | None, adapter_dir: Path, model_path: str
) -> str | None:
    'Pick the tokenizer source: explicit flag > adapter dir > base model.'
    if explicit:
        return explicit
    if any((adapter_dir / name).is_file() for name in _TOKENIZER_MARKERS):
        return str(adapter_dir)
    return model_path


def build_arg_parser() -> argparse.ArgumentParser:
    'Build the CLI parser (separate for testability).'
    p = argparse.ArgumentParser(
        description=(
            "Merge a saved peft adapter (LoRA/OPLoRA) into the base weights and save a full bf16 sharded model — the offline twin of sft_train.py's last-merged/best-merged step (e.g. to recover them after a post-training crash without re-training)."
        )
    )
    p.add_argument("--model", required=True,
                   help='The BASE checkpoint the adapter was trained on (loaded in bfloat16).')
    p.add_argument("--adapter", required=True, metavar="DIR",
                   help="The adapter directory to merge (sft_train.py's best-<step>/ or last/).")
    p.add_argument("--output-dir", required=True, metavar="DIR",
                   help='Where to write the merged full model.')
    p.add_argument("--tokenizer", default=None, metavar="PATH",
                   help='Tokenizer source (default: the adapter directory when it carries tokenizer files, else --model).')
    p.add_argument("--max-shard-size", default=DEFAULT_MAX_SHARD_SIZE,
                   metavar="SIZE",
                   help='max_shard_size of the merged checkpoint (transformers size string; default 5GB; bfloat16 is hard-checked before the save).')
    p.add_argument("--oplora", action="store_true",
                   help='Re-apply the OPLoRA invariant before merging: rebuild the projection from the frozen base weights (a deterministic SVD) and re-project the adapter')
    p.add_argument("--oplora-k", type=int, default=128, metavar="K",
                   help='The OPLoRA subspace size (must match training; default 128).')
    p.add_argument("--device", choices=("auto", "cpu", "cuda"),
                   default="auto",
                   help='Where to run the merge and the OPLoRA SVD (auto = cuda when visible, else cpu; cpu needs ~2x model size of RAM, no SVD without --oplora).')
    p.add_argument("--seed", type=int, default=42, metavar="N",
                   help='RNG seed: torch.svd_lowrank of --oplora draws a random start matrix, so the seed makes the offline merge bit-reproducible (logged; default 42).')
    return p


def main(argv: list[str] | None = None) -> int:
    'Run the offline adapter merge.'
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = build_arg_parser().parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(args.adapter)
    if not adapter_dir.is_dir():
        raise SystemExit(
            f'adapter directory not found: {adapter_dir}'
        )
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    logger.info('seed=%s (seeds torch.svd_lowrank of --oplora; log it to re-play a run).', args.seed)

    tokenizer_path = resolve_tokenizer_path(
        args.tokenizer, adapter_dir, args.model
    )
    tokenizer = (
        AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer_path is not None else None
    )
    logger.info('Loading the base model %s in bfloat16 on %s (tokenizer from %s)...', args.model, device, tokenizer_path)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    )
    if device == "cuda":
        base.to("cuda")

    merged = merge_adapter_into_base(
        base, adapter_dir, args.output_dir,
        tokenizer=tokenizer, max_shard_size=args.max_shard_size,
        oplora=args.oplora, oplora_k=args.oplora_k,
    )
    n_params = sum(p.numel() for p in merged.parameters())
    logger.info('Merged %s into %s (oplora=%s, k=%s, device=%s, %.2fB params, dtype=%s).', adapter_dir, args.output_dir, args.oplora, args.oplora_k if args.oplora else '-', device, n_params / 1000000000.0, next(merged.parameters()).dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
