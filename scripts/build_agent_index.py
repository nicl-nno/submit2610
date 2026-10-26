'Build the frozen hybrid index over the extraction corpus (agent suite).\n\n\nStage 2 of ``specifications/2026-10-17-agent-suite-glm53-validation.md``:\nONE index over ALL corpus chunks (train + val; 326k texts) backs the\n``searchdocs`` tool of the suite, the teacher pass and the anchors — the\nretrieval environment is identical by construction. The chunk texts ride in\nthe corpus rows (``extra["extraction"]["text"]``); the vectors come from the\nSERVED embeddings endpoint (``--embed-base-url`` — the bge-m3 service,\nspec section 16), batched inside :class:`OpenAIEmbeddingsEncoder`. The\nsnapshot plus ``digest.txt`` (the content sha256) land in ``--out-dir``; the\nsuite manifest pins the digest.\n\n\nUsage\n\n    python scripts/build_agent_index.py         --train-jsonl src_data/extraction/train_extract.jsonl         --val-jsonl src_data/extraction/val_extract.jsonl         --out work/agent_suite/index         --embed-base-url http://localhost:8025/v1         --embed-model-id bge-m3'

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from llm_training.extraction.retrieval import (
    HybridIndex,
    OpenAIEmbeddingsEncoder,
    RetrievalConfig,
)


def _dotted_get(row: dict[str, Any], dotted: str) -> Any:
    'Fetch a value by a dotted path (``extra.extraction.text``).'
    value: Any = row
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"dotted path {dotted!r} failed at {part!r}")
        value = value[part]
    return value


def iter_texts(
    paths: list[str], text_field: str, limit: int = 0
) -> Iterator[tuple[str, str]]:
    'Stream ``(doc_id, text)`` pairs from corpus row files.'
    n = 0
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                doc_id = str(row.get("id") or "").strip()
                text = str(_dotted_get(row, text_field) or "").strip()
                if not doc_id or not text:
                    raise SystemExit(
                        f'{path}: row without id or text (id={doc_id!r}).\n{path}'
                    )
                yield doc_id, text
                n += 1
                if limit and n >= limit:
                    return


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Build the frozen hybrid index over the extraction corpus (agent suite, stage 2).'
        )
    )
    parser.add_argument(
        "--train-jsonl", action="append", required=True,
        help='Train corpus JSONL (repeatable).',
    )
    parser.add_argument(
        "--val-jsonl", action="append", default=[],
        help='Val corpus JSONL (repeatable).',
    )
    parser.add_argument(
        "--text-field", default="extra.extraction.text",
        help='Dotted path to the chunk text.',
    )
    parser.add_argument("--out-dir", required=True,
                        help='Snapshot directory.')
    parser.add_argument("--embed-base-url", required=True,
                        help='Embeddings endpoint root.')
    parser.add_argument("--embed-model-id", required=True,
                        help='Served embeddings model id.')
    parser.add_argument("--embed-api-key", default="EMPTY",
                        help='API key (local vLLM: EMPTY).')
    parser.add_argument(
        "--embed-batch", type=int, default=64,
        help='Texts per embeddings request.',
    )
    parser.add_argument(
        "--outer-batch", type=int, default=8192,
        help='Progress/report granularity.',
    )
    parser.add_argument("--limit", type=int, default=0,
                        help='>0 = index only the first N docs (smoke).')
    parser.add_argument(
        "--no-self-test", action="store_true",
        help='Skip the post-build self-retrieval check.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    started = time.monotonic()
    encoder = OpenAIEmbeddingsEncoder(
        base_url=args.embed_base_url,
        model_id=args.embed_model_id,
        api_key=args.embed_api_key,
        batch_size=args.embed_batch,
    )

    texts: list[str] = []
    ids: list[str] = []
    vectors_chunks: list[np.ndarray] = []
    chunk_texts: list[str] = []
    chunk_ids: list[str] = []
    total = 0
    for total, (doc_id, text) in enumerate(
        iter_texts(
            list(args.train_jsonl) + list(args.val_jsonl),
            args.text_field,
            limit=args.limit,
        ),
        start=1,
    ):
        chunk_texts.append(text)
        chunk_ids.append(doc_id)
        if len(chunk_texts) >= args.outer_batch:
            vectors_chunks.append(
                np.asarray(encoder(chunk_texts), dtype=np.float32)
            )
            texts.extend(chunk_texts)
            ids.extend(chunk_ids)
            elapsed = time.monotonic() - started
            print(
                f"indexed {total} docs ({elapsed:.0f}s)", flush=True
            )
            chunk_texts, chunk_ids = [], []
    if chunk_texts:
        vectors_chunks.append(np.asarray(encoder(chunk_texts), dtype=np.float32))
        texts.extend(chunk_texts)
        ids.extend(chunk_ids)
    if not texts:
        raise SystemExit('no documents indexed.')
    vectors = np.vstack(vectors_chunks)
    index = HybridIndex.build(
        texts,
        ids,
        encode=encoder,
        config=RetrievalConfig(),
        vectors=vectors,
    )
    saved = index.save(args.out_dir)
    digest_path = Path(args.out_dir) / "digest.txt"
    digest_path.write_text(saved["digest"] + "\n", encoding="utf-8")
    elapsed = time.monotonic() - started
    print(
        f"OK: {len(ids)} docs, dim={vectors.shape[1]}, "
        f"digest={saved['digest']} -> {args.out_dir} ({elapsed:.0f}s)"
    )
    if not args.no_self_test:
        rng = random.Random(0)
        sample = rng.sample(list(range(len(ids))), min(5, len(ids)))
        missed = 0
        for i in sample:
            fragment = texts[i][:160]
            hits = index.search(fragment, top_k=5)
            if ids[i] not in {hit.doc_id for hit in hits}:
                missed += 1
        if missed:
            print(
                f'SELF-TEST: {missed}/{len(sample)}'
            )
            return 1
        print("SELF-TEST OK: 5/5 fragments self-retrieved in top-5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
