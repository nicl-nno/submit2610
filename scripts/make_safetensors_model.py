'Convert a .bin-only HF model into a local safetensors copy (one-off).\n\n\nWhy\n(``check_torch_load_is_safe``): loading a NON-safetensors checkpoint via\n``torch.load`` is refused outright unless torch >= 2.6. The frozen ``vllm_env``\ncluster stack has torch 2.5.1 (pinned by vllm 0.7.3 — cannot upgrade) and\ntransformers 4.53.2 (required by trl 0.20.0 — cannot downgrade), and the\n``BAAI/bge-m3`` HF repo ships NO ``model.safetensors`` — only\n``pytorch_model.bin`` — so the in-process anchor embedder can never load there\n(the judge anchors of sft_train.py died exactly there). The weights cannot be\nswapped for another model (the frozen anchor methodology — the SemSim/AC\nnumbers must stay comparable with the GRPO series), but a safetensors copy is\nBIT-FOR-BIT the same model: this script produces it.\n``model.safetensors``\n\nThe script itself MUST run on torch >= 2.6 (it is the one performing the\n``torch.load`` — doing it on a vulnerable torch would defeat the point of the\ngate). The dev machine (torch 2.8) qualifies; copy the output to the cluster\nand point the consumers at it.\n``torch.load``). Dev\n\nUsage\n\n    python scripts/make_safetensors_model.py --source BAAI/bge-m3         --out /tmp/opencode/bge-m3\n    rsync -av --progress /tmp/opencode/bge-m3/ <node>:/path/to/storage/models/bge-m3/\n    # then, per run:\n    #   sft_train_slurm.sh: EMBEDDING_MODEL=/path/to/storage/models/bge-m3\n    #   run_eval.py:        --local-embedding-model /path/to/storage/models/bge-m3\n\nThe output directory is a complete local model: the converted\n``model.safetensors`` (with the ``format=pt`` metadata transformers requires)\nplus every auxiliary file the snapshot provides — tokenizer, configs and the\nsentence-transformers scaffolding (``modules.json``, ``1_Pooling/``,\n``sentence_bert_config.json``). Missing patterns are skipped, so the script\nworks for plain transformers models too. The heavy ``.bin`` and the optional\n``colbert_linear.pt`` / ``sparse_linear.pt`` heads are NOT copied (plain\n``SentenceTransformer.encode`` never reads them).\n``model.safetensors``\n``format=pt``\n``colbert_linear.pt``'

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

#: The heavy checkpoint file this script converts (never copied to --out).

WEIGHTS_BIN = "pytorch_model.bin"

#: Auxiliary files that must ride along for a complete local model dir.
#: Missing patterns are simply skipped (usable for plain transformers models
#: too — only the tokenizer/config patterns will match there).


AUX_PATTERNS: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
    # sentence-transformers scaffolding
    "modules.json",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
    "1_Pooling/config.json",
)


def _torch_release() -> tuple[int, int, int]:
    'The running torch release as an (major, minor, patch) tuple.'
    import torch

    m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", torch.__version__)
    if m is None:  # pragma: no cover - unreadable version string
        raise RuntimeError(f"Cannot parse torch version: {torch.__version__!r}")
    return (
        int(m.group(1)),
        int(m.group(2)),
        int(m.group(3) or 0),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Convert a .bin-only HF model into a local safetensors copy (torch >= 2.6 required).'
    )
    parser.add_argument(
        "--source", default="BAAI/bge-m3",
        help='HF repo id (or a local snapshot path).'
    )
    parser.add_argument(
        "--out", required=True,
        help='Output model directory (created).'
    )
    parser.add_argument(
        "--revision", default=None,
        help='Optional HF revision.'
    )
    args = parser.parse_args(argv)

    if _torch_release() < (2, 6, 0):
        print(
            'REFUSED: torch >= 2.6 is required — this script performs the torch.load that the CVE-2025-32434 gate forbids on older torch; run it on the dev machine (torch 2.8) or another torch>=2.6 env and copy the result to the cluster.',
            file=sys.stderr,
        )
        return 2

    import torch
    from huggingface_hub import snapshot_download

    source = Path(args.source)
    if source.is_dir():
        snapshot = source
        print(f'Using the local snapshot {snapshot}')
    else:
        snapshot = Path(snapshot_download(
            repo_id=args.source,
            revision=args.revision,
            allow_patterns=[WEIGHTS_BIN, *AUX_PATTERNS],
        ))
        print(f'Using the HF snapshot {snapshot}')

    bin_path = snapshot / WEIGHTS_BIN
    if not bin_path.is_file():
        print(
            f'REFUSED: no {WEIGHTS_BIN}in the snapshot — nothing to convert (the source already ships safetensors?).',
            file=sys.stderr,
        )
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {bin_path} (torch.load, weights_only=True)...")
    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    print(f"Loaded {len(state)} tensors; converting to safetensors...")

    from safetensors.torch import save_file

    # contiguous(): safetensors refuses non-contiguous views; the "format=pt"
    # metadata is what transformers checks when loading safetensors files.
    # contiguous(): safetensors
    
    save_file(
        {k: v.contiguous() for k, v in state.items()},
        str(out / "model.safetensors"),
        metadata={"format": "pt"},
    )
    print(f"Wrote {out / 'model.safetensors'}")

    copied: list[str] = []
    for pattern in AUX_PATTERNS:
        for src_file in sorted(snapshot.glob(pattern)):
            dest = out / src_file.relative_to(snapshot)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest)
            copied.append(str(dest.relative_to(out)))
    print(f"Copied {len(copied)} auxiliary files: {', '.join(copied) or '-'}")

    print(
        'Ready'
        "\n  sft_train_slurm.sh: EMBEDDING_MODEL=" + str(out.resolve()) +
        "\n  run_eval.py: --local-embedding-model " + str(out.resolve())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
