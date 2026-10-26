'Package the prepared GRPO-extraction corpus as a Huggingface Dataset.\n\nDatasets (parquet +\n\nThe output is ONE directory in the hub layout, loadable by ``datasets``:\n\n    from datasets import load_dataset\n    ds = load_dataset("<out-dir>")            # auto-detects data/*.parquet\n    # or explicitly:\n    ds = load_dataset(\n        "parquet",\n        data_files={"train": "<out-dir>/data/train-*.parquet",\n                    "val": "<out-dir>/data/val-*.parquet"},\n    )\n\nRound-trip contract (LOSSLESS): every row is stored as five PLAIN columns —\n``id``, ``task_class``, ``prompt`` (JSON string of the message list),\n``reference`` (nullable string), ``extra`` (JSON string of the dict). Plain\nstring columns dodge the pyarrow heterogeneous-type trap (nested structs with\nmixed value types fail to unify); ``json.loads`` restores the original\nobjects byte-exactly. A row without ``reference`` comes back with an\nexplicit ``reference: None`` (the column is fixed; consumers read it via\n``row.get("reference")`` either way).\n\nReproducibility: ``meta.json`` records the row counts, composition stats and\nthe sha256 of every input file — the packaged artifact is verifiable against\nthe exact inputs that produced it.\n\nCard (README.md): YAML front matter (language, licenses, task category,\nsource datasets) + the license table, composition report and the loading\nsnippet. ``--push-to-hub <repo_id>`` uploads the directory as a dataset\nrepository (network; optional — the packaging itself is offline).\n\nInputs are the FINAL line-format files of the pipeline (``build`` output,\n``stage2``, ``summaries``, ``qa``, the train/val split, the annotated val):\n\n    python scripts/package_extraction_dataset.py         --train-jsonl src_data/extraction/train.jsonl         --val-jsonl src_data/extraction/val_annotated.jsonl         --out-dir hf_dataset/grpo-extraction-ruen         --dataset-name grpo-extraction-ruen         --push-to-hub anonymous/grpo-extraction-ruen   # optional'

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# The plain columns of the packaged dataset (see the module docstring).
_COLUMNS = ("id", "task_class", "prompt", "reference", "extra")

#: The license table (mirrors the AGENTS.md corpus license table).
_LICENSE_TABLE = """\
| Source | License | Rule |
|---|---|---|
| Wikipedia RU/EN (wikimedia/wikipedia) | CC BY-SA 4.0 | attribution required |
| Project Gutenberg books | public domain | free |
| FineWeb2-HQ-QA (epfml/FineWeb2-HQ-QA) | ODC-By v1.0 | attribution required; no share-alike on derivatives |
| NEREL-BIO (nerel-ds/NEREL-BIO) | research use | cite Bioinformatics 2023 |
| TED transcripts (if included) | CC BY-NC-ND | non-commercial |
"""


def read_rows(paths: list[str]) -> list[dict[str, Any]]:
    'Read final-format JSONL rows (strict on the required fields).'
    rows: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f'{path}:{line_no}: not valid JSON ({exc}).\n{path}:{line_no}'
                    ) from exc
                if not isinstance(rec, dict):
                    raise SystemExit(
                        f'{path}:{line_no}: row is not an object.\n{path}:{line_no}'
                    )
                for field in ("id", "task_class", "prompt"):
                    if not rec.get(field):
                        raise SystemExit(
                            f'{path}:{line_no}: missing required field {field!r}.\n{path}:{line_no}'
                        )
                if "extra" not in rec:
                    raise SystemExit(
                        f"{path}:{line_no}: missing required field 'extra'.\n{path}:{line_no}"
                    )
                rows.append(rec)
    return rows


def _encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def rows_to_arrow(rows: list[dict[str, Any]]):  # noqa: ANN202 - pyarrow type
    'Build the five-column Arrow table (plain strings, lossless JSON).'
    import pyarrow as pa

    return pa.table(
        {
            "id": [str(r["id"]) for r in rows],
            "task_class": [str(r["task_class"]) for r in rows],
            "prompt": [
                _encode(r["prompt"])
                if not isinstance(r["prompt"], str)
                else r["prompt"]
                for r in rows
            ],
            "reference": [
                None if r.get("reference") is None else str(r["reference"])
                for r in rows
            ],
            "extra": [
                _encode(r["extra"])
                if not isinstance(r["extra"], str)
                else r["extra"]
                for r in rows
            ],
        }
    )


def compute_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    'Composition report from the row metadata'
    classes: Counter[str] = Counter()
    styles: Counter[str] = Counter()
    langs: Counter[str] = Counter()
    cs = 0
    with_ref = 0
    for row in rows:
        classes[str(row["task_class"])] += 1
        extra = row.get("extra") or {}
        if isinstance(extra, str):
            extra = json.loads(extra)
        styles[str(extra.get("style") or "unknown")] += 1
        langs[str(extra.get("target_language") or "unknown")] += 1
        if extra.get("code_switched"):
            cs += 1
        if row.get("reference"):
            with_ref += 1
    return {
        "rows": len(rows),
        "with_reference": with_ref,
        "by_task_class": dict(sorted(classes.items())),
        "by_style": dict(sorted(styles.items())),
        "by_target_language": dict(sorted(langs.items())),
        "code_switched": cs,
    }


def sha256_file(path: str | Path) -> str:
    'Stable content hash of an input file'
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_card(
    out_dir: Path,
    name: str,
    train_stats: dict[str, Any],
    val_stats: dict[str, Any],
    inputs: list[dict[str, str]],
) -> None:
    'Write the dataset card (README.md)'
    def fmt_counts(counts: dict[str, Any]) -> str:
        return ", ".join(f"{key}: {value}" for key, value in counts.items())

    inputs_md = "\n".join(
        f"- `{item['path']}` (sha256 `{item['sha256']}`)" for item in inputs
    )
    card = f"""---
language:
- ru
- en
license:
- cc-by-sa-4.0
- other
task_categories:
- text-generation
tags:
- information-extraction
- nerel
- nerel-bio
- ragu
- grpo
- reinforcement-learning
size_category:
- 100K<n<1M
source_datasets:
- wikimedia/wikipedia
- epfml/FineWeb2-HQ-QA
- project_gutenberg
- nerel-ds/NEREL-BIO
---

# {name}

GRPO-extraction corpus (RAGU-style label-free extraction skills): 3 extraction
stages + 3 consolidation (`ragu_*_summary`) + QA (`fineweb_qa`) over a
language-balanced RU/EN open corpus (license table below).
The `prompt` column carries the full rendered conversation (JSON list of
messages); `extra` carries the payload the deterministic reward reads
(schema, gold-free metadata, pseudo-gold for the annotated val).

## Composition

| split | rows | with reference |
|---|---|---|
| train | {train_stats['rows']} | {train_stats['with_reference']} |
| val | {val_stats['rows']} | {val_stats['with_reference']} |

- train by task class: {fmt_counts(train_stats['by_task_class'])}
- val by task class: {fmt_counts(val_stats['by_task_class'])}
- train by style: {fmt_counts(train_stats['by_style'])}
- train by target language: {fmt_counts(train_stats['by_target_language'])}
- train code-switched rows (RU-EN mixed register): {train_stats['code_switched']}

## License

See the license table below; a source without a clear license is not
included. Derivative redistribution follows the strictest applicable term
per row (attribution strings are kept in `extra.source`).

{_LICENSE_TABLE}
## Loading

```python
from datasets import load_dataset
ds = load_dataset("<out-dir>")  # this directory or the hub repo id
row = ds["train"][0]
messages = json.loads(row["prompt"])   # the rendered conversation
extra = json.loads(row["extra"])       # the reward payload
```

## Provenance

Input files (exact bytes this packaging was built from):

{inputs_md}
"""
    (out_dir / "README.md").write_text(card, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Package the GRPO-extraction corpus as a Huggingface Dataset.'
        )
    )
    parser.add_argument(
        "--train-jsonl",
        action="append",
        required=True,
        help='Train JSONL (repeatable)',
    )
    parser.add_argument(
        "--val-jsonl",
        action="append",
        default=[],
        help='Val JSONL (repeatable)',
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--dataset-name",
        default="grpo-extraction-ruen",
        help='Card title',
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help=(
            'Upload to this dataset repo id (needs a HF token)'
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help='Create the hub repo private',
    )
    args = parser.parse_args(argv)

    import pyarrow.parquet as pq

    out_dir = Path(args.out_dir)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_rows(args.train_jsonl)
    val_rows = read_rows(args.val_jsonl) if args.val_jsonl else []
    if not train_rows:
        raise SystemExit(
            'No train rows read'
        )

    train_inputs = [
        {"path": p, "sha256": sha256_file(p)} for p in args.train_jsonl
    ]
    val_inputs = [{"path": p, "sha256": sha256_file(p)} for p in args.val_jsonl]

    pq.write_table(
        rows_to_arrow(train_rows), data_dir / "train-00000-of-00001.parquet"
    )
    if val_rows:
        pq.write_table(
            rows_to_arrow(val_rows), data_dir / "val-00000-of-00001.parquet"
        )

    train_stats = compute_stats(train_rows)
    val_stats = compute_stats(val_rows)
    meta = {
        "format": "hf-datasets/parquet",
        "columns": list(_COLUMNS),
        "round_trip": "prompt/extra are JSON strings — json.loads restores them",
        "dataset_name": args.dataset_name,
        "train": train_stats,
        "val": val_stats,
        "inputs": {"train": train_inputs, "val": val_inputs},
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_card(
        out_dir, args.dataset_name, train_stats, val_stats, train_inputs + val_inputs
    )

    print(f"packaged: {len(train_rows)} train / {len(val_rows)} val rows -> {out_dir}")

    if args.push_to_hub:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise SystemExit(
                '--push-to-hub needs `huggingface_hub` (pip install datasets).\n--push-to-hub'
            ) from exc
        api = HfApi()
        api.create_repo(
            repo_id=args.push_to_hub,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=args.push_to_hub,
            repo_type="dataset",
        )
        print(f"pushed to: https://huggingface.co/datasets/{args.push_to_hub}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
