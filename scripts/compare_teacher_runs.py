'Agreement of two teacher runs over the overlapping suite tasks.\n\n\nStage 3 of ``specifications/2026-10-17-agent-suite-glm53-validation.md``\n(hybrid teacher): the BULK run (glm-5.3-flash, all tasks) is checked\nagainst the SEED run (full glm-5.3, stratified subset) on the overlap —\nacceptance rates and the semantic similarity of the two answers per task\n(cosine over local bge-m3 embeddings). High agreement certifies the\nFlash gold; low agreement demotes it to solvability-only and shifts the\ndistillation core to the seed run.\n\n\nUsage\n\n    python scripts/compare_teacher_runs.py         --bulk work/agent_suite/teacher_full_flash.jsonl         --seed work/agent_suite/teacher_seed_glm53.jsonl         --local-embed-model BAAI/bge-m3         --out work/agent_suite/agreement_report.json'

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def load_indexed(path: str) -> dict[str, dict[str, Any]]:
    'Read a teacher run JSONL indexed by ``suite_id``.'
    rows: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[str(row.get("suite_id"))] = row
    return rows


def _local_encoder(model_id: str, batch_size: int = 16) -> Callable[[list[str]], Any]:
    'sentence-transformers encoder (sync; the report is small).\n\n    batch keeps a 12 GB consumer GPU clear of OOM on long answer batches\n    (bge-m3 attention is quadratic in sequence length).'
    from sentence_transformers import SentenceTransformer  # lazy, heavy.

    model = SentenceTransformer(model_id)

    def encode(texts: list[str]) -> Any:
        return model.encode(list(texts), batch_size=batch_size, show_progress_bar=False)

    return encode


def _cosine_matrix(a: Any, b: Any) -> Any:
    'Row-wise cosine similarity between two L2-normalizable matrices.'
    import numpy as np

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norms = np.linalg.norm(a, axis=1, keepdims=True)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True)
    a_norms[a_norms == 0] = 1.0
    b_norms[b_norms == 0] = 1.0
    return (a / a_norms) @ (b / b_norms).T


def compare(
    bulk_rows: dict[str, dict[str, Any]],
    seed_rows: dict[str, dict[str, Any]],
    encode: Callable[[list[str]], Any],
) -> dict[str, Any]:
    'Acceptance table + per-task answer cosine over the overlap.'
    overlap = sorted(set(bulk_rows) & set(seed_rows))
    if not overlap:
        raise SystemExit(
            'no overlapping suite tasks between the two runs.'
        )
    pairs = []
    for suite_id in overlap:
        b, s = bulk_rows[suite_id], seed_rows[suite_id]
        # 2000 chars is ample for an agreement cosine; full extraction JSONs
        # would only risk OOM on consumer GPUs.
        
        
        def _answer_text(final: Any) -> str:
            answer = (final or {}).get("answer")
            if not isinstance(answer, str):
                answer = json.dumps(answer or "", ensure_ascii=False)
            return answer[:2000]

        pairs.append((suite_id, _answer_text(b.get("final")), _answer_text(s.get("final"))))

    matrix = encode([p[1] for p in pairs] + [p[2] for p in pairs])
    left, right = matrix[: len(pairs)], matrix[len(pairs) :]
    sims = _cosine_matrix(left, right).diagonal().tolist()

    per_task: dict[str, Any] = {}
    by_family: dict[str, list[float]] = {}
    for (suite_id, _, _), sim in zip(pairs, sims, strict=True):
        family = str(bulk_rows[suite_id].get("family"))
        by_family.setdefault(family, []).append(sim)
        per_task[suite_id] = {
            "family": family,
            "answer_cosine": round(float(sim), 4),
            "bulk_accepted": bulk_rows[suite_id].get("accepted_envelope"),
            "seed_accepted": seed_rows[suite_id].get("accepted_envelope"),
        }

    def _acc(rows_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
        n = len(overlap)
        acc = sum(1 for i in overlap if rows_index[i].get("accepted_envelope"))
        return {"n": n, "accepted": acc, "rate": round(acc / n, 4) if n else None}

    return {
        "n_overlap": len(overlap),
        "bulk_acceptance": _acc(bulk_rows),
        "seed_acceptance": _acc(seed_rows),
        "answer_cosine": {
            "mean": round(sum(sims) / len(sims), 4) if sims else None,
            "min": round(min(sims), 4) if sims else None,
            "max": round(max(sims), 4) if sims else None,
        },
        "by_family_mean_cosine": {
            family: round(sum(values) / len(values), 4)
            for family, values in by_family.items()
        },
        "per_task": per_task,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Agreement of two teacher runs over the overlapping suite tasks.'
        )
    )
    parser.add_argument("--bulk", required=True,
                        help='Bulk run JSONL (e.g. Flash).')
    parser.add_argument("--seed", required=True,
                        help='Seed run JSONL (e.g. glm-5.3).')
    parser.add_argument("--local-embed-model", default="BAAI/bge-m3",
                        help='Local encoder for the answer cosine.')
    parser.add_argument("--out", required=True,
                        help='Report JSON path.')
    args = parser.parse_args(argv)

    bulk = load_indexed(args.bulk)
    seed = load_indexed(args.seed)
    encode = _local_encoder(args.local_embed_model)
    report = compare(bulk, seed, encode)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=1),
        encoding="utf-8",
    )
    compact = {
        k: v
        for k, v in report.items()
        if k not in ("per_task",)
    }
    print(json.dumps(compact, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
