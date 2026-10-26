'Annotate the frozen anchor pool with judge references (once).\n\n\nThe one-off producer of the judge annotations for the GRPO-extraction anchor\nrows (see :mod:`llm_training.runtime.pseudo_gold`). The frozen val pool of\nthe 8-class line carries THREE row kinds; the script auto-detects and\nannotates the two that need a judge, each idempotently (so an interrupted\nrun resumes where it stopped AND a completed annotation is never recomputed\n— the measuring instrument stays frozen across the run series; freeze the\njudge per series, the AC convention):\n\n* extraction rows (``extra["extraction"]``) → ``extra["extraction"]\n  ["pseudo_gold"]`` (``annotate_rows``) — the recall ctrl-signal;\n* consolidation rows (``extra["summary"]``) → ``extra["summary"]\n  ["reference"]`` (``annotate_summary_references``) — the judge solves the\n  row\'s own prompt once; the anchor scores clipped-cosine SemSim against it;\n* QA rows (``fineweb_qa``) carry human references — untouched.\n\nUsage\n\n    python scripts/annotate_pseudo_gold.py         --val src_data/extraction/val.jsonl         --out src_data/extraction/val_annotated.jsonl         --judge-base-url http://localhost:8001/v1         --judge-model Qwen/Qwen2.5-72B-Instruct\n    # The judge flags fall back to JUDGE_BASE_URL / JUDGE_MODEL /\n    # JUDGE_API_KEY env vars (run_eval.py convention).'

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from llm_training.runtime.endpoints import OpenAIEndpoint
from llm_training.runtime.pseudo_gold import (
    annotate_rows,
    annotate_summary_references,
)


def _env_or(env_var: str, flag_value: str | None) -> str | None:
    return flag_value or os.environ.get(env_var) or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Annotate the anchor pool with judge references (pseudo-gold + summary references).'
        )
    )
    parser.add_argument("--val", required=True, help='Val JSONL (in).')
    parser.add_argument("--out", required=True, help='Annotated JSONL (out).')
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help='Judge API root (or JUDGE_BASE_URL).',
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help='Judge model id (or JUDGE_MODEL).',
    )
    parser.add_argument(
        "--judge-api-key",
        default=None,
        help='Judge API key (or JUDGE_API_KEY).',
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)

    base_url = _env_or("JUDGE_BASE_URL", args.judge_base_url)
    model = _env_or("JUDGE_MODEL", args.judge_model)
    if not base_url or not model:
        parser.error('A judge endpoint is required: pass --judge-base-url/--judge-model or export JUDGE_BASE_URL / JUDGE_MODEL.\nJUDGE_MODEL.')
    endpoint = OpenAIEndpoint(
        base_url=base_url,
        model=model,
        api_key=args.judge_api_key or os.environ.get("JUDGE_API_KEY"),
        kind="judge",
    )

    rows: list[dict] = []
    with open(args.val, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print(f"WARNING: {args.val} carries no rows — nothing to annotate.")
        return 0

    # Two idempotent passes over the same pool: extraction pseudo-gold, then
    # summary references (each pass re-reads what the previous one cached —
    # the rows list is shared, so a single write at the end covers both).
    
    # BOTH judge passes share ONE event loop: two ``asyncio.run`` calls over
    # one AsyncOpenAI client would serve the second pass from a dead loop
    # (the pooled keep-alive sockets are bound to the closed first loop —
    # the documented judge-service lesson).
    
    
    # (keep-alive
    

    async def _annotate_all():
        nonlocal rows
        rows, ext_report_local = await annotate_rows(
            rows,
            endpoint,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            retries=args.retries,
            timeout_s=args.timeout,
        )
        rows, sum_report_local = await annotate_summary_references(
            rows,
            endpoint,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            retries=args.retries,
            timeout_s=args.timeout,
        )
        return ext_report_local, sum_report_local

    ext_report, sum_report = asyncio.run(_annotate_all())
    # A mode with nothing of its own (e.g. an extraction-only pool) stays
    # silent — its "0/0 annotated" line would only confuse the operator.
    
    for report in (ext_report, sum_report):
        if report.n_annotated or report.n_failed:
            for line in report.format_lines():
                print(line)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    print(f"OK -> {out}")
    total_annotated = ext_report.n_annotated + sum_report.n_annotated
    total_failed = ext_report.n_failed + sum_report.n_failed
    if total_failed:
        # A partially annotated pool is usable (failed rows score N/A), but
        # the operator should see it; a fully failed run is a hard error.
        return 0 if total_annotated > 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
