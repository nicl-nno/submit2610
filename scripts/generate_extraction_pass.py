"Run one generative pass over corpus rows -> ``{row_id, response}`` JSONL.\n\n``{row_id, response}``.\n\nThe DATA-PHASE driver of the GRPO-extraction line: all base-model generations\nthat feed the offline corpus composition run HERE, before any training (the\nplan's rule — the data phases finish first, the anchors freeze once):\n\n* **pass A** (stage-1 rows) -> ``stage1_gens.jsonl`` feeds the ``stage2``\n  subcommand (entity verification) AND ``summaries --stage1-generations``;\n* **pass B** (stage-2 rows) -> ``stage2_gens.jsonl`` feeds\n  ``summaries --stage2-generations``;\n* any other row kind (consolidation rows, QA rows for a variance probe) — the\n  script is generic: ``--task-class`` selects the rows, everything else stays.\n\n\nThe output honours the ``_load_generations`` contract of\n``scripts/build_extraction_corpus.py`` (``row_id``/``id`` + ``response``/\n``completion``); the file is COMPACTED at the end of every run (valid records\nonly, one per row), so a torn write from an interrupted run can never poison\nthe consumers. Generation is greedy + seeded (vLLM ``SamplingParams`` pins\n``skip_special_tokens=False`` — the Meno-Lite special-token envelope must\nsurvive, the ``PolicyGenerator`` lesson); empty responses are NOT written (the\nconsumers skip them anyway) and are counted, so a re-run retries them.\n\n\nUsage\n\n    # Pass A: stage-1 generations -> stage2 verification + entity summaries.\n    python scripts/generate_extraction_pass.py         --rows src_data/extraction/train.jsonl         --task-class ragu_entity_extraction         --model /path/to/storage/models/meno-lite-0.2         --out src_data/extraction/stage1_gens.jsonl\n\n    # Pass B: stage-2 generations -> relation summaries.\n    python scripts/generate_extraction_pass.py         --rows src_data/extraction/stage2.jsonl         --model /path/to/storage/models/meno-lite-0.2         --out src_data/extraction/stage2_gens.jsonl"

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: Rows per generation batch (an interrupted run loses at most one batch of
#: NEW records; the file is appended after each batch and compacted at the end).

DEFAULT_BATCH = 2048


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def load_rows(path: str) -> list[dict[str, Any]]:
    'Read corpus rows (JSONL); rows without a prompt are a hard error.'
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    bad = [r.get("id") for r in rows if not r.get("prompt")]
    if bad:
        raise SystemExit(
            f'{path}: rows without a prompt: {bad[:5]}...\n{path}'
        )
    if not rows:
        raise SystemExit(f"{path}: no rows.")
    return rows


def select_rows(
    rows: list[dict[str, Any]],
    task_classes: list[str] | None = None,
) -> list[dict[str, Any]]:
    'Filter rows to the given task classes (``None``/empty = keep all).'
    if not task_classes:
        return list(rows)
    wanted = set(task_classes)
    return [r for r in rows if r.get("task_class") in wanted]


def load_done(path: str) -> dict[str, str]:
    "Read the existing output for RESUME: ``row_id -> response``.\n\n    Tolerant by design: malformed lines (a torn write of an interrupted run)\n    are SKIPPED, not fatal — the row is simply regenerated. Mirrors the\n    builder's ``_load_generations`` key conventions (``row_id``/``id``,\n    ``response``/``completion``, empty responses skipped)."
    p = Path(path)
    if not p.exists():
        return {}
    gens: dict[str, str] = {}
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            row_id = rec.get("row_id") or rec.get("id")
            response = rec.get("response") or rec.get("completion") or ""
            if row_id and response:
                gens[str(row_id)] = str(response)
    return gens


def write_records(path: str, records: list[dict[str, Any]]) -> int:
    'APPEND JSONL records (one per line, UTF-8); returns the count.\n\n    Guards the boundary: if the existing file does NOT end with a newline (a\n    torn write of an interrupted run), a ``\n`` is emitted FIRST — otherwise\n    the new record would be glued onto the torn fragment and BOTH would be\n    lost to the parser.'
    if not records:
        return 0
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        with out.open("rb") as fh:
            fh.seek(-1, 2)  # last byte
            if fh.read(1) != b"\n":
                with out.open("ab") as fh2:
                    fh2.write(b"\n")
    with out.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def compact(path: str) -> dict[str, str]:
    'Rewrite the output as clean ``{row_id, response}`` JSONL (last wins).\n\n    Runs at the END of every pass: drops torn/malformed lines and duplicates,\n    so the consumers (``stage2``/``summaries``) always see a valid file. The\n    record order of first appearance is preserved (deterministic output).'
    gens = load_done(path)
    if not gens:
        # Nothing valid: remove a fully-corrupt file so consumers fail loudly
        # instead of silently seeing zero generations.
        
        p = Path(path)
        if p.exists():
            p.unlink()
        return {}
    # Rewrite atomically from scratch (records only in ``gens`` — torn lines
    # and duplicates are gone
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for k, v in gens.items():
            fh.write(json.dumps({"row_id": k, "response": v},
                                ensure_ascii=False) + "\n")
    tmp.replace(p)
    return gens


def run_pass(
    rows: list[dict[str, Any]],
    out_path: str,
    generate_fn: Callable[[list[dict[str, Any]]], list[str]],
    *,
    resume: bool = True,
    batch: int = DEFAULT_BATCH,
    limit: int = 0,
    skip: int = 0,
    task_classes: list[str] | None = None,
) -> dict[str, Any]:
    'Run one pass: filter -> resume-skip -> batched generate -> append -> compact.\n\n    list of ROWS and returns ONE response per row (aligned); the vLLM-backed\n    implementation lives in :func:`make_vllm_generate_fn`.'
    if batch < 1:
        raise ValueError(
            f'batch must be >= 1, got {batch}'
        )
    selected = select_rows(rows, task_classes)
    if limit and limit > 0:
        selected = selected[:limit]
    if skip and skip > 0:
        # Sharding: skip the first N rows AFTER class filtering (before the
        # resume-skip) — a shard owns a disjoint id range of the corpus.
        
        
        selected = selected[skip:]
    done = load_done(out_path) if resume else {}
    if not resume:
        # Fresh start: the old file is discarded (its records would otherwise
        # mix with the re-generated ones and confuse the consumers).
        
        p = Path(out_path)
        if p.exists():
            p.unlink()
        done = {}
    pending = [r for r in selected if str(r.get("id")) not in done]
    report: dict[str, Any] = {
        "n_selected": len(selected),
        "n_done": len(selected) - len(pending),
        "n_generated": 0,
        "n_empty": 0,
        "n_batches": 0,
    }
    for start in range(0, len(pending), batch):
        chunk = pending[start : start + batch]
        responses = generate_fn(chunk)
        if len(responses) != len(chunk):
            raise ValueError(
                f'generate_fn returned {len(responses)} responses for {len(chunk)}'
            )
        records: list[dict[str, Any]] = []
        for row, response in zip(chunk, responses):
            response = (response or "").strip()
            if not response:
                report["n_empty"] += 1
                continue
            records.append({"row_id": str(row["id"]), "response": response})
        write_records(out_path, records)
        report["n_generated"] += len(records)
        report["n_batches"] += 1
    gens = compact(out_path)
    report["n_total_in_file"] = len(gens)
    return report


# ---------------------------------------------------------------------------
# The vLLM-backed generator (heavy: lazily imported, NOT unit-tested here)

# ---------------------------------------------------------------------------
def make_vllm_generate_fn(
    model: str,
    gpu_ids: list[int],
    tp: int,
    gpu_memory_utilization: float,
    max_tokens: int,
    seed: int,
) -> Callable[[list[dict[str, Any]]], list[str]]:
    "Build a greedy, seeded in-process vLLM generator over corpus rows.\n\n    The engine is loaded ONCE per pass; every call renders the rows' message\n    prompts with the model's chat template and returns one response per row.\n    ``skip_special_tokens=False`` / ``spaces_between_special_tokens=False``\n    keep the special-token envelope (the ``PolicyGenerator`` invariant);\n    greedy + explicit ``seed`` make the pass reproducible."
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES",
                          ",".join(str(g) for g in gpu_ids))
    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=None,  # derive from the model config
    )
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        seed=seed,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )

    def generate(rows: list[dict[str, Any]]) -> list[str]:
        prompts = [
            tokenizer.apply_chat_template(
                row["prompt"], tokenize=False, add_generation_prompt=True
            )
            for row in rows
        ]
        outputs = llm.generate(prompts, params)
        # ``llm.generate`` returns outputs aligned with the prompts (vLLM
        # documents this), so per-row alignment holds by construction.
        first = [out.outputs[0].text for out in outputs]
        return first

    return generate


def make_api_generate_fn(
    base_url: str,
    model_id: str,
    api_key: str,
    concurrency: int,
    max_tokens: int,
    seed: int,
    timeout_s: float = 600.0,
    max_retries: int = 2,
    client: Any = None,
) -> Callable[[list[dict[str, Any]]], list[str]]:
    'Build a greedy generator over a served OpenAI-compatible endpoint.\n\n    the cluster as ``vllm serve``, the client runs on the workstation, only\n    text crosses the tunnel). The rows\' message prompts go to\n    ``/v1/chat/completions`` AS-IS — the server applies the chat template;\n    temperature 0 + ``seed`` keep the pass reproducible; the special-token\n    flags ride ``extra_body`` (the ``PolicyGenerator`` invariant — the\n    Meno-Lite envelope must survive). Acceptance is strictly\n    ``finish_reason == "stop"`` (the cascade lesson): truncated responses\n    are returned but flagged on stderr, so a re-run can retry them.\n    ``vllm serve``'
    import asyncio

    from openai import AsyncOpenAI  # lazy: train-extra dependency.

    # ONE persistent event loop for the whole pass: the AsyncOpenAI client
    # binds its httpx pool to the first loop it serves — a second
    # ``asyncio.run`` would re-bind to a dead loop (the documented
    # judge-service lesson). The concurrency semaphore is PER BATCH (created
    # inside the running coroutine): an asyncio primitive constructed at
    # factory time binds to whatever loop first awaits it and crashes the
    # second batch ("bound to a different event loop" — the 2026-10-17
    # pass-A crash).
    
    
    
    
    
    
    
    loop = asyncio.new_event_loop()
    client = client or AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout_s,
        max_retries=max_retries,
    )

    async def generate_one(row: dict[str, Any], semaphore: asyncio.Semaphore) -> str:
        async with semaphore:
            response = await client.chat.completions.create(
                model=model_id,
                messages=row["prompt"],
                temperature=0.0,
                max_tokens=max_tokens,
                seed=seed,
                extra_body={
                    "skip_special_tokens": False,
                    "spaces_between_special_tokens": False,
                },
            )
        choice = response.choices[0]
        finish = getattr(choice, "finish_reason", None)
        if finish != "stop":
            print(
                f"WARNING: row {row.get('id')}: finish_reason={finish!r}",
                flush=True,
            )
        return choice.message.content or ""

    def generate(rows: list[dict[str, Any]]) -> list[str]:
        async def run_all() -> list[str]:
            semaphore = asyncio.Semaphore(max(1, concurrency))
            return await asyncio.gather(
                *(generate_one(row, semaphore) for row in rows)
            )

        return loop.run_until_complete(run_all())

    return generate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            'One generative pass over corpus rows -> {row_id, response} JSONL (data phase of the GRPO-extraction line).'
        )
    )
    p.add_argument("--rows", required=True,
                   help='Corpus JSONL (in).')
    p.add_argument("--out", required=True,
                   help='Generations JSONL (out).')
    p.add_argument("--model", default=None,
                   help='Model path/id for the pass (in-process vLLM mode).')
    p.add_argument("--base-url", default=None,
                   help='OpenAI-compatible endpoint of a SERVED model (the served-model mode; overrides --model).')
    p.add_argument("--model-id", default=None,
                   help='Served model id for --base-url.')
    p.add_argument("--api-key", default="EMPTY",
                   help='API key for --base-url (local vLLM: EMPTY).')
    p.add_argument("--concurrency", type=int, default=8,
                   help='In-flight requests in the served-model mode.')
    p.add_argument("--request-timeout", type=float, default=600.0,
                   help='Per-request timeout, seconds.')
    p.add_argument("--task-class", action="append", default=None,
                   help='Repeatable row filter.')
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                   help='Rows per generation batch.')
    p.add_argument("--limit", type=int, default=0,
                   help='>0 = a smoke pass over the first N rows.')
    p.add_argument("--skip", type=int, default=0,
                   help='Skip the first N rows after filtering (sharding: each shard owns a disjoint id range and writes its own --out).')
    p.add_argument("--no-resume", action="store_true",
                   help='Rewrite --out from scratch (ignore existing).')
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--gpu-ids", type=int, nargs="+", default=[0])
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = load_rows(args.rows)
    selected = select_rows(rows, args.task_class)
    if not selected:
        raise SystemExit(
            f'{args.rows}: no rows match --task-class {args.task_class}.\n{args.rows}'
        )
    # Seed BEFORE the engine build (vLLM reads the env at init) — mirrors the
    # project-wide "seed everything, log the seed" rule.
    
    random.seed(args.seed)
    try:
        import numpy as np

        np.random.seed(args.seed)
    except ImportError:
        pass
    if args.base_url:
        # The served-model mode (spec section 16): the model lives on the
        # cluster, the client runs on the workstation.
        
        
        if not args.model_id:
            raise SystemExit(
                '--base-url requires --model-id (the served model name).\n--base-url'
            )
        generate_fn = make_api_generate_fn(
            args.base_url,
            args.model_id,
            args.api_key,
            args.concurrency,
            args.max_tokens,
            args.seed,
            args.request_timeout,
        )
        # One append per API batch: a coarse batch delays the resume point,
        # a fine one wastes nothing — 256 keeps the appends frequent without
        # hammering the file.
        
        
        batch = min(args.batch, 256)
        print(
            f"mode: served model base_url={args.base_url} "
            f"model_id={args.model_id} concurrency={args.concurrency}"
        )
    else:
        if not args.model:
            raise SystemExit(
                'either --base-url (+ --model-id) or --model is required.'
            )
        generate_fn = make_vllm_generate_fn(
            args.model,
            args.gpu_ids,
            args.tp,
            args.gpu_memory_utilization,
            args.max_tokens,
            args.seed,
        )
        batch = args.batch
    report = run_pass(
        rows,
        args.out,
        generate_fn,
        resume=not args.no_resume,
        batch=batch,
        limit=args.limit,
        skip=args.skip,
        task_classes=args.task_class,
    )
    print(
        f"OK: pass over {report['n_selected']} rows "
        f"(done {report['n_done']}, generated {report['n_generated']}, "
        f"empty {report['n_empty']}, batches {report['n_batches']}) "
        f"-> {args.out} (total {report['n_total_in_file']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
