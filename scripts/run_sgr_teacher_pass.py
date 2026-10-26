"Teacher pilot: run GLM-5.3 over the frozen agent suite (stage 3 gate).\n\n\nStage 3 of ``specifications/2026-10-17-agent-suite-glm53-validation.md``:\nthe big-model teacher solves suite tasks through the SGR protocol\n(reasoning -> ``searchdocs`` over the FROZEN index -> work -> final\n``finalanswertool``), the tool executes LOCALLY on this machine against the\nfrozen index (only text crosses the network: Z.AI for the teacher, the\nembeddings tunnel for ``searchdocs``). The pilot's gates:\n\n* envelope acceptance — share of tasks whose run ends with a valid\n  ``finalanswertool`` (>= 0.80 to pass, spec section 6);\n* schedule adherence — every finished run made at least one ``searchdocs``\n  call;\n* per-family latency/tokens for the full-run budget decision (section 13;\n  the FULL 423-task run awaits the operator's budget approval).\n\n\nUsage\n\n    python scripts/run_sgr_teacher_pass.py         --suite src_data/agent_suite/suite.jsonl         --index-dir work/agent_suite/index         --embed-base-url http://localhost:8025/v1 --embed-model-id bge-m3         --teacher-base-url https://api.z.ai/api/paas/v4/         --teacher-model glm-5.3 --limit 20         --out work/agent_suite/teacher_pilot.jsonl"

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llm_training.extraction.retrieval import (
    HybridIndex,
    OpenAIEmbeddingsEncoder,
)
from llm_training.runtime.agent_suite import extract_tool_calls

_MAX_TURNS = 8
_TOOL_RESULT_PREFIX = "[tool] searchdocs result:\n"


def run_searchdocs(
    index: HybridIndex, arguments: dict[str, Any], top_k_cap: int = 10
) -> list[str]:
    'Execute the searchdocs tool: the top-k chunk ids from the frozen index.\n\n    rejected — the pilot measures the teacher, not the validator. Chunk\n    TEXTS are joined by the caller from the corpus map (the index does not\n    store texts).'
    query = str(arguments.get("query") or "").strip()
    try:
        top_k = int(arguments.get("top_k") or 5)
    except (TypeError, ValueError):
        top_k = 5
    top_k = max(1, min(top_k, top_k_cap))
    if not query:
        return []
    hits = index.search(query, top_k=top_k)
    return [hit.doc_id for hit in hits]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Teacher pilot: GLM-5.3 over the frozen agent suite (stage 3 gate).'
        )
    )
    parser.add_argument("--suite", required=True,
                        help='Frozen suite JSONL.')
    parser.add_argument("--index-dir", required=True,
                        help='Frozen index snapshot.')
    parser.add_argument("--embed-base-url", default=None,
                        help='Embeddings endpoint root.')
    parser.add_argument("--embed-model-id", default=None,
                        help='Served embeddings model id.')
    parser.add_argument("--embed-api-key", default="EMPTY")
    parser.add_argument("--embed-batch", type=int, default=64)
    parser.add_argument(
        "--local-embed-model", default=None,
        help=(
            "sentence-transformers model id INSTEAD of the endpoint: queries are embedded in-process on this machine's GPU (the heavy index vectors are precomputed in the snapshot)."
        ),
    )
    parser.add_argument("--teacher-base-url",
                        default="https://api.z.ai/api/paas/v4/")
    parser.add_argument("--teacher-model", default="glm-5.3")
    parser.add_argument("--teacher-api-key", default=None,
                        help='Or ZAI_API_KEY env.')
    parser.add_argument("--max-tokens", type=int, default=16384,
                        help="Per-turn cap (glm-5.3 thinking counts here).")
    parser.add_argument(
        "--thinking-budget", type=int, default=None,
        help=(
            "glm-5.3: bound the reasoning with thinking={type: enabled, budget_tokens: N}. Default: no parameter (unbounded thinking — the 2026-10-17 pilot showed turn-level 'length' cuts)."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=20,
                        help='Pilot size (stratified round-robin).')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    return parser


def _load_corpus_texts(paths: list[str]) -> dict[str, str]:
    'Chunk texts by row id (blocking; called from the sync main only).\n\n    main).'
    texts_by_id: dict[str, str] = {}
    for path in paths:
        corpus = Path(path)
        if not corpus.exists():
            continue
        with corpus.open(encoding="utf-8") as fh:
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
    return texts_by_id


def _load_suite(path: str) -> list[dict[str, Any]]:
    'Read the frozen suite (blocking; sync main only).'
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _local_encoder(model_id: str) -> Callable[[list[str]], Any]:
    'sentence-transformers fallback (no service; queries are light).'
    from sentence_transformers import SentenceTransformer  # lazy, heavy.

    model = SentenceTransformer(model_id)

    def encode(texts: list[str]) -> Any:
        return model.encode(list(texts), batch_size=64, show_progress_bar=False)

    return encode


async def _amain(
    args: argparse.Namespace,
    suite_rows: list[dict[str, Any]],
    texts_by_id: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import os

    from openai import AsyncOpenAI

    api_key = args.teacher_api_key or os.environ.get("ZAI_API_KEY")
    if not api_key:
        raise SystemExit(
            'a teacher API key is required: --teacher-api-key or ZAI_API_KEY.'
        )
    if args.local_embed_model:
        encode = _local_encoder(args.local_embed_model)
    else:
        if not (args.embed_base_url and args.embed_model_id):
            raise SystemExit(
                'either --local-embed-model or --embed-base-url (+ --embed-model-id) is required.'
            )
        encode = OpenAIEmbeddingsEncoder(
            base_url=args.embed_base_url,
            model_id=args.embed_model_id,
            api_key=args.embed_api_key,
            batch_size=args.embed_batch,
        )
    index = HybridIndex.load(args.index_dir, encode=encode)

    rng = random.Random(args.seed)
    order = list(range(len(suite_rows)))
    rng.shuffle(order)
    # Stratified round-robin: families interleaved (the pilot must touch all
    # three even at small limits).
    
    
    by_family: dict[str, list[int]] = {}
    for i in order:
        by_family.setdefault(str(suite_rows[i]["family"]), []).append(i)
    picked: list[int] = []
    families = sorted(by_family)
    level = 0
    while len(picked) < min(args.limit, len(suite_rows)):
        progressed = False
        for family in families:
            queue = by_family[family]
            if level < len(queue):
                picked.append(queue[level])
                progressed = True
                if len(picked) >= min(args.limit, len(suite_rows)):
                    break
        if not progressed:
            break
        level += 1

    client = AsyncOpenAI(
        base_url=args.teacher_base_url,
        api_key=api_key,
        timeout=300.0,
        max_retries=2,
    )
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def solve(row: dict[str, Any]) -> dict[str, Any]:
        messages: list[dict[str, str]] = [
            {"role": m["role"], "content": m["content"]} for m in row["prompt"]
        ]
        started = time.monotonic()
        turns = 0
        search_calls = 0
        final: dict[str, Any] | None = None
        usage_total = 0
        error: str | None = None
        forced = False

        async def create_with_retry(**kwargs: Any):
            delay = 5.0
            if args.thinking_budget:
                kwargs.setdefault("extra_body", {})
                kwargs["extra_body"] = {
                    **kwargs["extra_body"],
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": args.thinking_budget,
                    },
                }
            for attempt in range(3):
                try:
                    return await client.chat.completions.create(**kwargs)
                except Exception as exc:
                    if attempt == 2:
                        raise
                    print(
                        f"WARNING: {row['id']} attempt {attempt + 1} failed "
                        f"({type(exc).__name__}); retry in {delay:.0f}s",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                    delay *= 2

        async with semaphore:
            while turns < _MAX_TURNS and final is None:
                turns += 1
                try:
                    response = await create_with_retry(
                        model=args.teacher_model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=args.max_tokens,
                    )
                except Exception as exc:  # noqa: BLE001 - isolate the task
                    error = f"turn {turns}: {type(exc).__name__}: {exc}"
                    print(f"ERROR: {row['id']} {error}", flush=True)
                    break
                usage = getattr(response, "usage", None)
                usage_total += int(getattr(usage, "total_tokens", 0) or 0)
                choice = response.choices[0]
                if choice.finish_reason != "stop":
                    print(
                        f"WARNING: {row['id']} turn {turns}: "
                        f"finish_reason={choice.finish_reason!r}",
                        flush=True,
                    )
                content = choice.message.content or ""
                calls = extract_tool_calls(content)
                if not calls:
                    # No tool call and no final: one nudge, then give up.
                    
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Emit a <tool_call> with name searchdocs, "
                                "or the finalanswertool envelope."
                            ),
                        }
                    )
                    continue
                executed = False
                for call in calls:
                    if call["name"] == "searchdocs":
                        search_calls += 1
                        doc_ids = run_searchdocs(index, call["arguments"])
                        enriched = [
                            {
                                "doc_id": doc_id,
                                "text": texts_by_id.get(doc_id, "")[:1200],
                            }
                            for doc_id in doc_ids
                        ]
                        messages.append({"role": "assistant", "content": content})
                        messages.append(
                            {
                                "role": "user",
                                "content": _TOOL_RESULT_PREFIX
                                + json.dumps(enriched, ensure_ascii=False),
                            }
                        )
                        executed = True
                        break
                    if call["name"] == "finalanswertool":
                        arguments = call["arguments"]
                        final = {
                            "reasoning": str(arguments.get("reasoning") or ""),
                            "completed_steps": arguments.get("completed_steps") or [],
                            "answer": arguments.get("answer"),
                            "status": str(arguments.get("status") or ""),
                        }
                        break
                if final is None and not executed:
                    # A tool call we do not serve: nudge as above.
                    
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Only searchdocs and finalanswertool are "
                                "available. Emit one of them."
                            ),
                        }
                    )
                if (
                    final is None
                    and not forced
                    and turns >= _MAX_TURNS - 2
                ):
                    # Force-final: near the turn cap the task MUST close —
                    # the 2026-10-17 pilot showed search-loops that never
                    # reached finalanswertool. The answer may be partial;
                    # the anchors score quality, the pilot counts envelopes.
                    
                    
                    
                    
                    forced = True
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "STOP searching. Emit the finalanswertool "
                                "envelope NOW with the material you have "
                                "already collected. Do not call searchdocs "
                                "again."
                            ),
                        }
                    )
        accepted = final is not None and bool(final["answer"]) and final["status"] in {
            "completed",
            "failed",
        }
        return {
            "suite_id": row["id"],
            "family": row["family"],
            "source_row_id": row["source_row_id"],
            "task_class": row.get("task_class"),
            "accepted_envelope": accepted,
            "schedule_ok": search_calls >= 1,
            "turns": turns,
            "search_calls": search_calls,
            "total_tokens": usage_total,
            "latency_s": round(time.monotonic() - started, 1),
            "error": error,
            "final": final,
            "messages": messages,
        }

    results: list[dict[str, Any]] = []
    weights = await asyncio.gather(
        *(solve(suite_rows[i]) for i in picked)
    )
    results.extend(weights)

    n = len(results)
    accepted = sum(1 for r in results if r["accepted_envelope"])
    scheduled = sum(1 for r in results if r["schedule_ok"])
    report = {
        "n": n,
        "envelope_acceptance": round(accepted / n, 4) if n else None,
        "schedule_adherence": round(scheduled / n, 4) if n else None,
        "gate_pass": bool(n and accepted / n >= 0.80),
        "by_family": {
            family: {
                "n": sum(1 for r in results if r["family"] == family),
                "accepted": sum(
                    1 for r in results if r["family"] == family and r["accepted_envelope"]
                ),
                "mean_latency_s": round(
                    sum(r["latency_s"] for r in results if r["family"] == family)
                    / max(1, sum(1 for r in results if r["family"] == family)),
                    1,
                ),
                "mean_tokens": round(
                    sum(r["total_tokens"] for r in results if r["family"] == family)
                    / max(1, sum(1 for r in results if r["family"] == family)),
                ),
            }
            for family in sorted({r["family"] for r in results})
        },
    }
    return results, report


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    # All blocking I/O stays in the sync main; the async part only talks to
    # the API endpoints and the in-memory index.
    
    
    suite_rows = _load_suite(args.suite)
    texts_by_id = _load_corpus_texts(
        [
            "src_data/extraction/train_extract.jsonl",
            "src_data/extraction/val_extract.jsonl",
        ]
    )
    report_and_results = asyncio.run(_amain(args, suite_rows, texts_by_id))
    results, report = report_and_results

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in results:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    report_path = out.parent / (out.stem + "_report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=1),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))
    gate = "PASS" if report["gate_pass"] else "FAIL"
    print(f"PILOT GATE (envelope >= 0.80): {gate}")
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
