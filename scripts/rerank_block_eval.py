#!/usr/bin/env python3
"Rerank-block evaluation over an OpenAI-compatible endpoint (WWW 2027 campaign).\n\n\nScores the ``meno_rerank`` fixed-val pool (``src_data/meno_stand_distil/test\n.jsonl`` — 63 rerank rows) of a served model with THREE deterministic\nmeasures (the frozen protocol ``paper_for_webconf/protocol.md``):\n\n* ``rerank_em`` — the repo's graded ordinal verdict score\n  (``llm_training.reward.rerank_em``: 0.3 valid JSON verdict + 0.7 scaled by\n  the ordinal distance to the gold label on the 0/1/2 scale);\n* ``logprob soft-score`` — the probability of the GOLD verdict after\n  renormalizing the model's first-token distribution over the three labels\n  {0, 1, 2} (labels absent from the top-k logprobs get probability 0);\n* ``top1_em`` — the share of rows whose argmax label equals the gold verdict.\n\nThe gold verdict is parsed from the row's REFERENCE response (both the\ninteger and the string spellings, the loader's semantics). Only rerank rows\nare scored: the rerank marker of the system message (the same string the\nloader keys on). Rows whose reference has no parseable label are skipped and\ncounted.\n\nGreedy decoding (temperature 0), no judge, no network except the model\nendpoint itself. Usage (three columns = three runs):\n\n\nExample\n\n    vllm serve ~/models/meno-lite-0.2 --served-model-name meno-0-2         --port 8000 --max-model-len 16384 --gpu-memory-utilization 0.90\n    python scripts/rerank_block_eval.py         --base-url http://localhost:8000/v1 --model meno-0-2         --rows src_data/meno_stand_distil/test.jsonl         --output rerank_meno-0-2.json"

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# The rerank marker of the meno-stand system prompt (mirror of the loader's
# ``llm_training.loaders.meno_stand._RERANK_MARKER``; kept literal here so the
# script has no package dependency beyond the metric module).


RERANK_MARKER: str = "насколько документ пригоден"

#: The valid verdict labels of the 0/1/2 ordinal scale.

VALID_LABELS: tuple[int, ...] = (0, 1, 2)

#: Logprobs requested per generated token (covers the three label digits).

TOP_LOGPROBS: int = 20


def load_rerank_rows(path: str | Path) -> tuple[list[dict[str, Any]], int, int]:
    'Read the fixed-val JSONL and return the rerank rows with gold labels.\n\n\n    Returns ``(rows, n_skipped_foreign, n_skipped_gold)``: foreign rows (not\n    rerank by the system marker) and rerank rows whose reference response has\n    no parseable label. Each row carries ``id`` (0-based source line number),\n    ``system``, ``query`` and ``gold``.'
    from llm_training.reward.rerank_em import parse_label

    rows: list[dict[str, Any]] = []
    n_foreign = 0
    n_gold = 0
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if RERANK_MARKER not in (record.get("system") or ""):
                n_foreign += 1
                continue
            gold = parse_label(record.get("response") or "")
            if gold is None:
                n_gold += 1
                continue
            rows.append(
                {
                    "id": line_no,
                    "system": record["system"],
                    "query": record["query"],
                    "gold": gold,
                }
            )
    return rows, n_foreign, n_gold


def extract_label_logprobs(top_logprobs: list[dict[str, Any]] | None) -> dict[int, float]:
    'Map the valid labels to their first-token logprobs (missing → absent).\n\n\n    ``top_logprobs`` is the OpenAI payload of the first generated token: a\n    list of ``{"token": str, "logprob": float}``. The FIRST entry matching\n    each label (stripped) wins.\n``top_logprobs``'
    found: dict[int, float] = {}
    if not top_logprobs:
        return found
    for entry in top_logprobs:
        token = str(entry.get("token", "")).strip()
        if token in {"0", "1", "2"}:
            label = int(token)
            if label not in found:
                found[label] = float(entry.get("logprob", float("-inf")))
    return found


def soft_scores(label_logprobs: dict[int, float]) -> dict[int, float] | None:
    'Renormalize the label logprobs into a probability mass over {0,1,2}.\n\n\n    Labels missing from the top-k get probability 0 (the model did not rank\n    them among the top continuations). Returns ``None`` when NO label appears\n    in the payload (the soft-score is undefined for the row).'
    if not label_logprobs:
        return None
    import math

    masses = {label: math.exp(label_logprobs.get(label, float("-inf"))) for label in VALID_LABELS}
    total = sum(masses.values())
    if total <= 0.0:
        return None
    return {label: masses[label] / total for label in VALID_LABELS}


def _first_token_top_logprobs(choice: Any) -> list[dict[str, Any]] | None:
    "First generated token's top-logprobs from an OpenAI choice (or None)."
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None) if logprobs is not None else None
    if not content:
        return None
    return content[0].top_logprobs


def _score_one(client: Any, model: str, row: dict[str, Any], max_tokens: int,
               retries: int) -> dict[str, Any]:
    'Score one row against the endpoint (greedy, with a small retry loop).'
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": row["system"]},
                    {"role": "user", "content": row["query"]},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=TOP_LOGPROBS,
            )
            break
        except Exception as exc:  # noqa: BLE001 — transient endpoint errors
            last_error = exc
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
    else:  # pragma: no cover - the loop always breaks or raises via last_error
        raise RuntimeError(f"row {row['id']}: endpoint failed: {last_error}")

    from llm_training.reward.rerank_em import parse_label, rerank_em

    choice = response.choices[0]
    content = choice.message.content or ""
    got = parse_label(content)
    soft = soft_scores(extract_label_logprobs(_first_token_top_logprobs(choice)))
    return {
        "id": row["id"],
        "gold": row["gold"],
        "response": content,
        "parsed_label": got,
        "rerank_em": rerank_em(content, "meno_rerank", {"gold_label": row["gold"]}),
        "soft_gold": (soft or {}).get(row["gold"]),
        "top1_correct": (got == row["gold"]) if got is not None else False,
    }


def main(argv: list[str] | None = None) -> int:
    'CLI entry point'
    parser = argparse.ArgumentParser(
        description='Rerank-block evaluation (rerank_em + logprob soft-score)')
    parser.add_argument("--base-url", required=True,
                        help='OpenAI-compatible API root, e.g. http://localhost:8000/v1')
    parser.add_argument("--model", required=True,
                        help='served model name')
    parser.add_argument("--api-key", default=None,
                        help='API key (default: the OPENAI_API_KEY env)')
    parser.add_argument("--rows", default="src_data/meno_stand_distil/test.jsonl",
                        help='fixed-val JSONL with the rerank pool')
    parser.add_argument("--output", default="rerank_report.json",
                        help='output JSON report')
    parser.add_argument("--concurrency", type=int, default=8,
                        help='parallel requests')
    parser.add_argument("--timeout", type=float, default=120.0,
                        help='per-request timeout, s')
    parser.add_argument("--max-tokens", type=int, default=64,
                        help='generation cap (a verdict is short)')
    parser.add_argument("--retries", type=int, default=2,
                        help='retries per row on transient errors')
    parser.add_argument("--limit", type=int, default=0,
                        help='score only the first N rows (smoke)')
    args = parser.parse_args(argv)

    from openai import OpenAI

    rows, n_foreign, n_gold = load_rerank_rows(args.rows)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        print(f'No scorable rerank rows in {args.rows} (foreign: {n_foreign}, no-gold: {n_gold}', file=sys.stderr)
        return 1

    client = OpenAI(base_url=args.base_url, api_key=args.api_key or "EMPTY",
                    timeout=args.timeout)
    results: list[dict[str, Any]] = []
    n_failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(_score_one, client, args.model, row,
                               args.max_tokens, args.retries) for row in rows]
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 — a row failure is recorded, not fatal
                n_failed += 1
                print(f'row failed', file=sys.stderr)

    scored = [r for r in results if r["rerank_em"] is not None]
    softs = [r["soft_gold"] for r in results if r["soft_gold"] is not None]
    report = {
        "model": args.model,
        "rows_file": str(args.rows),
        "base_url": args.base_url,
        "n_rerank_rows": len(rows),
        "n_foreign_rows": n_foreign,
        "n_no_gold_rows": n_gold,
        "n_scored": len(scored),
        "n_failed": n_failed,
        "em_mean": (sum(r["rerank_em"] for r in scored) / len(scored)) if scored else None,
        "soft_gold_mean": (sum(softs) / len(softs)) if softs else None,
        "top1_em_mean": (sum(1 for r in results if r["top1_correct"]) / len(results))
                        if results else None,
        "per_row": sorted(results, key=lambda r: r["id"]),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"scored {len(scored)}/{len(rows)} rows, failed {n_failed}; "
          f"em={report['em_mean']}, soft_gold={report['soft_gold_mean']}, "
          f"top1_em={report['top1_em_mean']} -> {args.output}")
    return 0 if not n_failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
