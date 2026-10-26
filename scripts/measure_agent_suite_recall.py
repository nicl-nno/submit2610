"Control recall@k measurement over the frozen agent suite and index.\n\n\nStage 2 of ``specifications/2026-10-17-agent-suite-glm53-validation.md``:\nthe INDEX quality gate, run BEFORE any training. Two deterministic target\nvariants over the same suite rows / index:\n\n* ``self`` — the seed-fragment anchor must retrieve its own source chunk\n  (the chunk is one of 326k; a fragment query finding it is the minimal\n  sanity of the embeddings service);\n* ``gold`` — a query derived from the GOLD (pseudo-gold entity names /\n  summary anchor / QA question) must surface chunks containing the gold\n  content stems (pseudo-gold entities / descriptions / reference answer).\n\nThe MODEL's own query quality is a training outcome and is NOT measured\nhere — the anchors score it; this script only certifies the instrument.\n\nUsage\n\n    python scripts/measure_agent_suite_recall.py         --suite work/agent_suite/suite.jsonl         --index-dir work/agent_suite/index         --corpus-jsonl src_data/extraction/train_extract.jsonl         --corpus-jsonl src_data/extraction/val_extract.jsonl         --embed-base-url http://localhost:8025/v1 --embed-model-id bge-m3         --out work/agent_suite/recall_report.json"

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llm_training.extraction.retrieval import (
    HybridIndex,
    OpenAIEmbeddingsEncoder,
    tokenize_stems,
)


def content_stems(text: str) -> set[str]:
    "Content stems of a text (len > 2; the probe's target tokens)."
    return {stem for stem in tokenize_stems(text) if len(stem) > 2}


def gold_query(row: dict[str, Any]) -> str:
    "A deterministic probe query derived from the row's gold."
    family = row.get("family")
    gold = row.get("gold") or {}
    if family == "extraction":
        entities = (gold.get("pseudo_gold") or {}).get("entities") or []
        names = [
            str(e.get("entity_name") or "")
            for e in entities
            if isinstance(e, dict)
        ]
        return "; ".join(n for n in names if n)[:250]
    if family == "summary":
        material = gold.get("material") or {}
        kind = material.get("kind")
        if kind == "relation":
            return f"{material.get('subject', '')} {material.get('object', '')}"[:250]
        if kind == "community":
            entities = material.get("entities") or []
            names = [
                str(item.get("name") or item.get("entity_name") or "")
                for item in entities[:4]
                if isinstance(item, dict)
            ]
            return ", ".join(n for n in names if n)[:250]
        return str(material.get("entity_name") or "")[:250]
    if family == "qa":
        return str(gold.get("question") or "")[:250]
    raise ValueError(f"unknown suite family {family!r}")


def gold_targets(row: dict[str, Any]) -> set[str]:
    'The gold content stems the top-k chunks must contain.'
    family = row.get("family")
    gold = row.get("gold") or {}
    if family == "extraction":
        entities = (gold.get("pseudo_gold") or {}).get("entities") or []
        text = " ".join(
            str(e.get("entity_name") or "") for e in entities if isinstance(e, dict)
        )
    elif family == "summary":
        material = gold.get("material") or {}
        parts = [str(material.get("entity_name") or material.get("subject") or "")]
        parts.extend(
            str(d) for d in (material.get("descriptions") or [])[:8]
        )
        text = " ".join(parts)
    elif family == "qa":
        text = str(gold.get("reference") or "")
    else:
        raise ValueError(f"unknown suite family {family!r}")
    return content_stems(text)


def self_recall(index: HybridIndex, row: dict[str, Any], k: int) -> float:
    '1.0 when the seed-fragment query retrieves the source chunk in top-k.'
    fragment = str((row.get("extra") or {}).get("seed_fragment") or "").strip()
    if not fragment:
        return 0.0
    hits = index.search(fragment, top_k=k)
    return 1.0 if any(hit.doc_id == row["source_row_id"] for hit in hits) else 0.0


def gold_recall(
    index: HybridIndex,
    texts_by_id: dict[str, str],
    row: dict[str, Any],
    k: int,
) -> float:
    'Share of gold stems found in the top-k retrieved chunk texts.'
    query = gold_query(row).strip()
    if len(query) < 8:
        return 0.0
    hits = index.search(query, top_k=k)
    found: set[str] = set()
    for hit in hits:
        found |= content_stems(texts_by_id.get(hit.doc_id, ""))
    targets = gold_targets(row)
    if not targets:
        return 0.0
    return len(targets & found) / len(targets)


def evaluate_suite(
    index: HybridIndex,
    suite_rows: list[dict[str, Any]],
    texts_by_id: dict[str, str],
    top_ks: list[int],
    targets: str = "all",
) -> dict[str, Any]:
    'Run the probe; returns the JSON-serializable report.'
    want_self = targets in ("all", "self")
    want_gold = targets in ("all", "gold")
    buckets: dict[str, dict[int, list[float]]] = {
        "self": {k: [] for k in top_ks},
        "gold": {k: [] for k in top_ks},
    }
    family_buckets: dict[str, dict[str, dict[int, list[float]]]] = {}
    for row in suite_rows:
        family = str(row.get("family"))
        fam = family_buckets.setdefault(
            family, {"self": {k: [] for k in top_ks}, "gold": {k: [] for k in top_ks}}
        )
        # The self variant needs the fragment anchor: only extraction rows
        # carry it, the others are SKIPPED (a zero would dilute the mean).
        
        
        has_fragment = bool(
            str((row.get("extra") or {}).get("seed_fragment") or "").strip()
        )
        for k in top_ks:
            if want_self and has_fragment:
                value = self_recall(index, row, k)
                buckets["self"][k].append(value)
                fam["self"][k].append(value)
            if want_gold:
                value = gold_recall(index, texts_by_id, row, k)
                buckets["gold"][k].append(value)
                fam["gold"][k].append(value)

    def _summarize(store: dict[int, list[float]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, values in store.items():
            out[str(k)] = {
                "mean_recall": round(sum(values) / len(values), 4) if values else None,
                "n": len(values),
            }
        return out

    report: dict[str, Any] = {
        "n_rows": len(suite_rows),
        "top_ks": top_ks,
        "variants": {name: _summarize(store) for name, store in buckets.items()},
        "per_family": {
            family: {name: _summarize(store) for name, store in stores.items()}
            for family, stores in family_buckets.items()
        },
    }
    return report


def _local_encoder(model_id: str) -> Callable[[list[str]], Any]:
    'sentence-transformers fallback (no service).'
    from sentence_transformers import SentenceTransformer  # lazy, heavy.

    model = SentenceTransformer(model_id)

    def encode(texts: list[str]) -> Any:
        return model.encode(list(texts), batch_size=64, show_progress_bar=False)

    return encode


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Control recall@k over the frozen agent suite and index (stage 2 gate).'
        )
    )
    parser.add_argument("--suite", required=True,
                        help='suite.jsonl (in).')
    parser.add_argument("--index-dir", required=True,
                        help='Frozen index snapshot.')
    parser.add_argument(
        "--corpus-jsonl", action="append", required=True,
        help='Corpus JSONL for chunk texts (repeatable).',
    )
    parser.add_argument("--embed-base-url",
                        help='Embeddings endpoint root.')
    parser.add_argument("--embed-model-id",
                        help='Served embeddings model id.')
    parser.add_argument("--embed-api-key", default="EMPTY")
    parser.add_argument("--embed-batch", type=int, default=64)
    parser.add_argument(
        "--local-embed-model",
        help='sentence-transformers fallback instead of the endpoint.',
    )
    parser.add_argument("--top-ks", default="1,5,10",
                        help='Comma-separated k values.')
    parser.add_argument("--max-queries", type=int, default=0,
                        help='>0 = cap the rows (smoke).')
    parser.add_argument("--targets", choices=("all", "self", "gold"),
                        default="all")
    parser.add_argument("--out", default="",
                        help='Report JSON path.')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.embed_base_url and not args.local_embed_model:
        raise SystemExit(
            'either --embed-base-url (+ --embed-model-id) or --local-embed-model is required.'
        )
    if args.local_embed_model:
        encode = _local_encoder(args.local_embed_model)
    else:
        if not args.embed_model_id:
            raise SystemExit("--embed-base-url requires --embed-model-id.")
        encode = OpenAIEmbeddingsEncoder(
            base_url=args.embed_base_url,
            model_id=args.embed_model_id,
            api_key=args.embed_api_key,
            batch_size=args.embed_batch,
        )
    index = HybridIndex.load(args.index_dir, encode=encode)

    texts_by_id: dict[str, str] = {}
    for path in args.corpus_jsonl:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row_id = str(row.get("id") or "")
                text = str(
                    ((row.get("extra") or {}).get("extraction") or {}).get("text")
                    or ""
                ).strip()
                if row_id and text:
                    texts_by_id[row_id] = text

    suite_rows: list[dict[str, Any]] = []
    with open(args.suite, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                suite_rows.append(json.loads(line))
    if args.max_queries and args.max_queries > 0:
        suite_rows = suite_rows[: args.max_queries]
    top_ks = sorted({int(k) for k in args.top_ks.split(",") if k.strip()})
    report = evaluate_suite(
        index, suite_rows, texts_by_id, top_ks, targets=args.targets
    )
    report["index_digest"] = (
        Path(args.index_dir) / "digest.txt"
    ).read_text(encoding="utf-8").strip()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=1),
            encoding="utf-8",
        )
    print(json.dumps(report["variants"], ensure_ascii=False, indent=1))
    for family, stores in report["per_family"].items():
        gold_k10 = stores["gold"].get("10") or {}
        self_k5 = stores["self"].get("5") or {}
        print(
            f"{family}: gold@10={gold_k10.get('mean_recall')} "
            f"(n={gold_k10.get('n')}), self@5={self_k5.get('mean_recall')}"
        )
    if args.out:
        print(f"OK -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
