#!/usr/bin/env python3
'websearchtool, extractpagecontenttool, finalanswertool\n\n\n    SGR_ROOT\n    LLM_API_KEY\n    LLM_BASE_URL\n    LLM_MODEL\n    LLM_TEMPERATURE\n    LLM_JUDGE_MODEL\n    TAVILY_API_KEY\n\n\nprepare\nprobe\nrun'
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QUESTIONS_JSON = REPO / "scripts" / "distill_questions.json"
DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4/"
DEFAULT_AGENT = "sgr_tool_calling_agent"

FINAL_TOOL = "finalanswertool"
REASONING_TOOL = "reasoningtool"
SEARCH_TOOLS = ("websearchtool", "extractpagecontenttool")





_DEGENERATE_ATTEMPT = "degenerate fallback to FinalAnswerTool"
DEGENERATE_FINAL_MARKER = "degenerate fallback FINAL (retries exhausted)"

QHASH = lambda q: hashlib.blake2b(q.strip().lower().encode(), digest_size=16).hexdigest()




def load_questions(path) -> list[dict]:
    'Internal implementation detail.'
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    recs = []
    for r in raw:
        q = (r.get("question") or "").strip()
        if not q:
            continue
        recs.append({"qid": r["qid"], "lang": r.get("lang", "en"),
                     "trap": bool(r.get("trap")), "question": q})
    return recs


def dedup_report(recs: list[dict], simpleqa_questions) -> tuple[list[dict], dict]:
    'Internal implementation detail.'
    banned = {QHASH(q) for q in simpleqa_questions}
    kept = [r for r in recs if QHASH(r["question"]) not in banned]
    report = {"total": len(recs), "kept": len(kept),
              "dropped_simpleqa": len(recs) - len(kept)}
    return kept, report


def spotcheck_selected(qid: str, every: int = 4) -> bool:
    'Internal implementation detail.'
    if every <= 0:
        return False
    h = int.from_bytes(hashlib.blake2b(qid.encode(), digest_size=8).digest(), "big")
    return h % every == 0


def has_degenerate_final(retry_log: list) -> bool:
    'with the degenerate fallback final after exhausting retries.'
    return any(r.get("error") == DEGENERATE_FINAL_MARKER for r in retry_log)


def shape_keep(row: dict) -> bool:
    'Whether an episode is retained for training.'
    names = [t for t in (row.get("tools") or "").split(",") if t]
    return (row.get("status") == "ok"
            and bool(row.get("predicted_answer"))
            and FINAL_TOOL in names
            and names.count(REASONING_TOOL) >= 1
            and any(t in names for t in SEARCH_TOOLS)
            and not row.get("degenerate_final"))


def parse_judge_verdict(text: str) -> dict | None:
    'Internal implementation detail.'
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    cleaned = re.sub(r"```[a-z]*", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
    verdict = str(data.get("verdict", "")).lower().strip()
    if verdict not in ("grounded", "ungrounded", "unclear"):
        return None
    return {"verdict": verdict, "comment": str(data.get("comment", ""))[:500]}


def run_summary(rows: list[dict], credits: float) -> str:
    'Internal implementation detail.'
    n = len(rows)
    if not n:
        return "нет завершённых эпизодов"
    keep = sum(1 for r in rows if r.get("keep"))
    errs = sum(1 for r in rows if r.get("status") == "error")
    retr = sum(int(r.get("retries") or 0) for r in rows)
    degen = sum(int(r.get("degenerates") or 0) for r in rows)
    checked = sum(1 for r in rows if r.get("grounded"))
    ground = sum(1 for r in rows if r.get("grounded") == "grounded")
    return (f"episodes={n} keep={keep} ({keep / n:.0%}) errors={errs} "
            f"retries={retr} degenerates={degen} "
            f"grounded={ground}/{checked} credits={credits:.1f} "
            f"per_episode={credits / n:.2f}")




def _require_env(*names: str) -> dict:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        sys.exit("Нет переменных окружения: " + ", ".join(missing)
                 + "\nMissing env vars; задайте их перед запуском (ключи в "
                   "командной строке и репозитории не хранятся).")
    return {n: os.environ[n] for n in names}


class RateLimiter:
    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm
        self.next_at = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            now = time.monotonic()
            delay = self.next_at - now
            self.next_at = max(now, self.next_at) + self.interval
        if delay > 0:
            await asyncio.sleep(delay)


class CreditGuard(RuntimeError):
    pass



EXTRACT_URLS_PER_CREDIT = 5


def _patch_tavily(credits_state: dict, cap: float, rpm: float):
    'Meter-only wrapper: search and extract both hit the real Tavily API.'
    from tavily import AsyncTavilyClient

    limiter = RateLimiter(rpm)
    real_search = AsyncTavilyClient.search
    real_extract = AsyncTavilyClient.extract

    def spend(amount: float):
        credits_state["used"] += amount
        if cap and credits_state["used"] >= cap:
            raise CreditGuard(
                f"Исчерпан лимит кредитов Tavily: {credits_state['used']:.0f} >= {cap}")

    async def _call_with_retry(make_call):
        resp = None
        for attempt in range(1, 13):
            await limiter.wait()
            try:
                resp = await make_call()
                break
            except Exception as e:
                if not re.search(r"429|excessive|rate", str(e), re.I):
                    raise
                await asyncio.sleep(min(15 * attempt, 120))
        return resp

    async def metered_search(self, query, **kw):
        resp = await _call_with_retry(lambda: real_search(self, query, **kw))
        credits_state["search_calls"] = (credits_state.get("search_calls", 0) + 1)
        spend(1.0)
        return resp

    async def metered_extract(self, urls, **kw):
        url_list = [urls] if isinstance(urls, str) else list(urls or [])
        resp = await _call_with_retry(lambda: real_extract(self, urls, **kw))
        credits_state["extract_calls"] = (credits_state.get("extract_calls", 0) + 1)
        spend(max(1.0, math.ceil(len(url_list) / EXTRACT_URLS_PER_CREDIT)))
        return resp

    AsyncTavilyClient.search = metered_search
    AsyncTavilyClient.extract = metered_extract


def canonicalize_messages(msgs: list[dict]) -> list[dict]:
    '«Agent {id} started»\n    Relabel non-leading system messages to user for strict backends.'
    if not msgs or all(m.get("role") != "system" for m in msgs[1:]):
        return msgs
    return [msgs[0]] + [
        m if m.get("role") != "system" else {**m, "role": "user"}
        for m in msgs[1:]
    ]


def install_retry_and_trace(agent, retry_log: list, tools_log: list,
                            max_retries: int = 2) -> None:
    '* `_reasoning_phase`\n* `_select_action_phase`\n* `_prepare_context`\n      (`canonicalize_messages`\n(reasoningtool\n    Retry phase wrappers for schema violations and empty SSE streams.'
    from pydantic import ValidationError

    import httpx
    import openai

    from sgr_agent_core.tools import FinalAnswerTool

    orig_reasoning = agent._reasoning_phase
    orig_action = agent._select_action_phase
    orig_ctx = agent._prepare_context

    
    
    
    
    
    
    
    
    _PHASE_ERRORS = (ValidationError, AssertionError,
                     openai.APIConnectionError, httpx.TransportError)

    def _note(phase: str, attempt: int, error: str) -> None:
        retry_log.append({"phase": phase, "attempt": attempt,
                          "error": str(error)[:400]})

    async def reasoning_with_retry():
        last: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                out = await orig_reasoning()
                tools_log.append(REASONING_TOOL)
                return out
            except _PHASE_ERRORS as exc:
                last = exc
                _note("reasoning", attempt, f"{type(exc).__name__}: {exc}")
        assert last is not None
        raise last

    def _degenerate(tool) -> bool:
        return (isinstance(tool, FinalAnswerTool)
                and list(getattr(tool, "completed_steps", None) or [])
                == ["Response synthesized without a tool call"])

    async def action_with_retry(reasoning):
        last_exc: Exception | None = None
        tool = None
        for attempt in range(max_retries + 1):
            snapshot = list(agent.conversation)
            try:
                tool = await orig_action(reasoning)
            except _PHASE_ERRORS as exc:
                last_exc = exc
                agent.conversation[:] = snapshot
                _note("action", attempt, f"{type(exc).__name__}: {exc}")
                continue
            last_exc = None
            if not _degenerate(tool):
                tools_log.append(tool.tool_name)
                return tool
            agent.conversation[:] = snapshot
            _note("action", attempt, _DEGENERATE_ATTEMPT)
        if last_exc is not None:
            raise last_exc
        
        
        _note("action", max_retries, DEGENERATE_FINAL_MARKER)
        return tool

    async def prepare_context_canonical():
        return canonicalize_messages(await orig_ctx())

    agent._reasoning_phase = reasoning_with_retry
    agent._select_action_phase = action_with_retry
    agent._prepare_context = prepare_context_canonical


def _observations(agent, budget: int = 24000) -> str:
    'Internal implementation detail.'
    parts: list[str] = []
    total = 0
    for msg in getattr(agent, "conversation", []):
        if msg.get("role") != "tool":
            continue
        text = str(msg.get("content") or "")
        if not text:
            continue
        parts.append(text)
        total += len(text) + 2
        if total >= budget:
            break
    return "\n\n".join(parts)[:budget]


GROUNDING_PROMPT = """Ты проверяешь заземлённость финального ответа исследовательского агента. Дан вопрос агента, его финальный ответ и наблюдения, собранные агентом из интернета (результаты поисков и выгрузок страниц). Оцени, согласуется ли финальный ответ с наблюдениями: не содержит ли он утверждений, противоречащих наблюдениям или никак не подкреплённых ими (выдуманные факты, источники, числа).

Ответь ТОЛЬКО JSON-объектом:
{{"verdict": "grounded" | "ungrounded" | "unclear",
  "comment": "краткое пояснение решения (1-3 предложения)"}}

Если наблюдений мало или ответ подкреплён лишь частично — верни "unclear".

Вопрос агента:
{question}

Финальный ответ агента:
{answer}

Наблюдения агента:
{observations}"""


async def groundedness_check(client, model: str, question: str, answer: str,
                             observations: str, timeout: float = 300.0) -> dict | None:
    "Reference-free groundedness spot-check of the teacher's final answer."
    if not observations.strip():
        return {"verdict": "unclear", "comment": "нет наблюдений в разговоре"}
    try:
        resp = await asyncio.wait_for(client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=2048,
            messages=[{"role": "user", "content": GROUNDING_PROMPT.format(
                question=question[:4000], answer=answer[:8000],
                observations=observations)}]), timeout=timeout)
    except Exception as e:
        print(f"groundedness_check failed: {type(e).__name__}: {str(e)[:200]}",
              file=sys.stderr)
        return None
    return parse_judge_verdict(resp.choices[0].message.content or "")


async def run_episodes(args, pool: list[dict], done: set[str]) -> None:
    env = _require_env("SGR_ROOT", "LLM_API_KEY", "TAVILY_API_KEY")
    
    
    out_dir = Path(args.out).resolve()
    sgr_root = Path(env["SGR_ROOT"]).resolve()
    os.chdir(sgr_root)                      
    sys.path.insert(0, ".")

    import openai  # noqa: E402
    import httpx  # noqa: E402
    from sgr_agent_core.agent_config import GlobalConfig  # noqa: E402

    if args.loop == "benchmark":
        from benchmark.benchmark_agent import BenchmarkAgent  # noqa: E402
        from sgr_agent_core.tools import (  # noqa: E402
            ExtractPageContentTool, FinalAnswerTool, ReasoningTool, WebSearchTool)
    else:
        from sgr_agent_core.agent_factory import AgentFactory  # noqa: E402

    cfg_path = (Path(args.config) if args.config else
                sgr_root / "examples" / "sgr_deep_research" / "config.yaml")
    config = GlobalConfig.from_yaml(str(cfg_path))

    
    env_temp = os.environ.get("LLM_TEMPERATURE")
    resolved_model = (args.model or os.environ.get("LLM_MODEL")
                      or config.llm.model)
    resolved_temp = (args.temperature if args.temperature is not None
                     else float(env_temp) if env_temp
                     else config.llm.temperature)

    
    
    agent_def = (config.agents.get(args.agent)
                 if args.loop == "research" else None)
    if args.loop == "research" and agent_def is None:
        sys.exit(f"Агент '{args.agent}' не найден в {cfg_path}")
    for target in (config, agent_def):
        llm = getattr(target, "llm", None)
        if llm is None:
            continue
        if os.environ.get("LLM_API_KEY"):
            llm.api_key = env["LLM_API_KEY"]
        if os.environ.get("LLM_BASE_URL"):
            llm.base_url = os.environ["LLM_BASE_URL"]
        llm.model = resolved_model
        llm.temperature = resolved_temp
    
    
    for target in (config, agent_def):
        if hasattr(target, "reports_dir"):
            target.reports_dir = str(out_dir / "reports")
        if hasattr(target, "logs_dir"):
            target.logs_dir = str(out_dir / "agent_logs")
    
    
    
    
    for target in (config, agent_def):
        llm = getattr(target, "llm", None)
        if llm is not None:
            llm.max_tokens = args.max_completion_tokens
    if args.max_steps:
        for target in (getattr(config, "execution", None),
                       getattr(agent_def, "execution", None) if agent_def
                       else None):
            if target is not None:
                target.max_iterations = args.max_steps

    
    
    
    
    tool_configs = None
    if args.loop == "benchmark":
        tool_configs = {
            "websearchtool": {"engine": "tavily",
                              "api_key": env["TAVILY_API_KEY"]},
            "extractpagecontenttool": {"tavily_api_key": env["TAVILY_API_KEY"]},
        }

    credits: dict = {"used": 0.0, "search_calls": 0, "extract_calls": 0}
    _patch_tavily(credits, args.credit_cap, args.rpm)

    
    
    
    # max_retries=0
    transport_timeout = httpx.Timeout(connect=30.0, read=180.0, write=60.0,
                                      pool=30.0)
    client = openai.AsyncOpenAI(        
        api_key=env["LLM_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
        timeout=transport_timeout, max_retries=0)
    judge_model = os.environ.get("LLM_JUDGE_MODEL", resolved_model)

    sem = asyncio.Semaphore(args.concurrency)
    out_path = out_dir / "episodes.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in pool if r["qid"] not in done][:args.limit or None]

    async def one(rec: dict) -> dict:
        async with sem:
            if args.loop == "benchmark":
                agent = BenchmarkAgent(
                    task_messages=[{"role": "user", "content": rec["question"]}],
                    openai_client=client, agent_config=config,
                    toolkit=[ReasoningTool, WebSearchTool,
                             ExtractPageContentTool, FinalAnswerTool],
                    tool_configs=tool_configs)
                agent.max_iterations = args.max_steps
                agent.max_searches = args.max_searches
            else:
                agent = await AgentFactory.create(
                    agent_def, [{"role": "user", "content": rec["question"]}])
            retry_log: list = []
            tools_log: list = []
            install_retry_and_trace(agent, retry_log, tools_log,
                                    max_retries=args.max_retries)
            status, error, predicted = "ok", None, None
            t0 = time.monotonic()
            try:
                await agent.execute()
                predicted = agent._context.execution_result
                if predicted is None:
                    status = "no_final"
            except Exception as e:  
                status, error = "error", f"{type(e).__name__}: {e}"[:500]
            row = {
                "qid": rec["qid"], "lang": rec["lang"],
                "trap": rec.get("trap", False), "question": rec["question"],
                "loop": args.loop, "model": resolved_model,
                "temperature": resolved_temp,
                "status": status, "predicted_answer": predicted,
                "tools": ",".join(tools_log),
                "n_search": tools_log.count("websearchtool"),
                "n_extract": tools_log.count("extractpagecontenttool"),
                "retries": len(retry_log), "retry_log": retry_log,
                "degenerates": sum(1 for r in retry_log
                                   if r["error"] == _DEGENERATE_ATTEMPT),
                "degenerate_final": has_degenerate_final(retry_log),
                "grounded": None, "grounding_comment": None,
                
                
                "observations": _observations(agent),
                "agent_id": getattr(agent, "id", ""), "error": error,
                "credits_used": round(credits["used"], 2),
                "seconds": round(time.monotonic() - t0, 1),
            }
            if (status == "ok" and predicted
                    and spotcheck_selected(rec["qid"], args.spotcheck_every)):
                verdict = await groundedness_check(
                    client, judge_model, rec["question"], str(predicted),
                    _observations(agent))
                row["grounded"] = verdict["verdict"] if verdict else "check_failed"
                row["grounding_comment"] = (verdict or {}).get("comment")
            row["keep"] = shape_keep(row)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            print(f"[{row['qid']}] {row['status']} keep={row['keep']} "
                  f"retries={row['retries']} tools={row['tools'][:90]}",
                  flush=True)
            return row

    async def guarded(rec):
        try:
            return await one(rec)
        except CreditGuard as e:
            print(f"STOP: {e}", file=sys.stderr)
            return None

    rows: list[dict] = []
    tasks = [asyncio.create_task(guarded(r)) for r in todo]
    for t in asyncio.as_completed(tasks):
        r = await t
        if r is None:
            for task in tasks:
                task.cancel()
            break
        rows.append(r)
        if args.pause_after and len(rows) == args.pause_after:
            print("PAUSE", run_summary(rows, credits["used"]))
            for task in tasks:
                task.cancel()
            break
    if not args.pause_after or len(rows) < args.pause_after:
        print("DONE", run_summary(rows, credits["used"]))




def cmd_prepare(args) -> int:
    recs = load_questions(args.questions_json)
    report: dict = {"total": len(recs),
                    "ru": sum(1 for r in recs if r["lang"] == "ru"),
                    "en": sum(1 for r in recs if r["lang"] == "en"),
                    "traps": sum(1 for r in recs if r["trap"])}
    if args.simpleqa_results:
        with open(args.simpleqa_results, encoding="utf-8") as f:
            simpleqa = [json.loads(l)["question"] for l in f if l.strip()]
        recs, dedup = dedup_report(recs, simpleqa)
        report.update(dedup)
        if dedup["dropped_simpleqa"]:
            print("WARNING: approved questions overlap SimpleQA; remove them "
                  "from distill_questions.json and rebuild the pool.",
                  file=sys.stderr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "questions.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


def cmd_probe(args) -> int:
    import openai
    env = _require_env("LLM_API_KEY")
    client = openai.OpenAI(
        api_key=env["LLM_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
        timeout=600.0)
    tools = [{"type": "function", "function": {
        "name": "echo", "parameters": {"type": "object",
                                       "properties": {"x": {"type": "string"}}}}}]
    dump = {}
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "glm-5.3"),
            messages=[{"role": "user", "content": "Вызови echo с x=1"}],
            tools=tools, tool_choice="required", max_tokens=16384)
        dump = resp.choices[0].message.model_dump()
        print("PROBE tool_choice=required OK:",
              json.dumps(dump, ensure_ascii=False)[:400])
    except Exception as e:
        print(f"PROBE FAILED: {type(e).__name__}: {e}")
    out = Path(args.out) / "probe_dump.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Dump saved:", out)
    if not dump.get("tool_calls"):
        print("WARNING: tool_calls is empty; check required-tool support and reasoning content.")
        return 2
    return 0


def cmd_run(args) -> int:
    pool = [json.loads(l) for l in
            Path(args.questions).read_text(encoding="utf-8").splitlines() if l.strip()]
    done: set[str] = set()
    out_path = Path(args.out) / "episodes.jsonl"
    if out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            row = json.loads(l)
            
            # no_final/error
            
            if row.get("status") == "ok":
                done.add(row["qid"])
    print(f"start pool={args.questions} total={len(pool)} done={len(done)}")
    asyncio.run(run_episodes(args, pool, done))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="publish the approved question pool")
    pp.add_argument("--out", default="distill_pools")
    pp.add_argument("--questions-json", default=str(QUESTIONS_JSON),
                    help="Approved list (defaults to scripts/distill_questions.json).")
    pp.add_argument("--simpleqa-results",
                    help="SimpleQA results.jsonl used for an overlap check.")
    pp.set_defaults(fn=cmd_prepare)

    pr = sub.add_parser("probe", help="one request with tool_choice=required")
    pr.add_argument("--out", default="distill_out")
    pr.set_defaults(fn=cmd_probe)

    rn = sub.add_parser("run", help="main resumable run")
    rn.add_argument("--questions", required=True,
                    help="questions.jsonl produced by prepare")
    rn.add_argument("--out", required=True)
    rn.add_argument("--loop", choices=("benchmark", "research"),
                    default="benchmark",
                    help="Distillation loop: benchmark tools or research config fallback.")
    rn.add_argument("--temperature", type=float, default=1.0,
                    help="Teacher sampling temperature.")
    rn.add_argument("--model",
                    help="Teacher model (defaults to LLM_MODEL or the configured model).")
    rn.add_argument("--config",
                    help="Global checkout YAML configuration.")
    rn.add_argument("--agent", default=DEFAULT_AGENT,
                    help="Agent name in the configuration (research loop only).")
    rn.add_argument("--limit", type=int, default=0, help="0 means the full pool.")
    rn.add_argument("--pause-after", type=int, default=0,
                    help="Stop after N episodes and print a summary.")
    rn.add_argument("--concurrency", type=int, default=6)
    rn.add_argument("--max-steps", type=int, default=12,
                    help="Maximum loop iterations.")
    rn.add_argument("--max-searches", type=int, default=4,
                    help="Maximum searches per episode.")
    rn.add_argument("--max-completion-tokens", type=int, default=8192,
                    help="Teacher completion-token budget per request.")
    rn.add_argument("--max-retries", type=int, default=2,
                    help="Retries per phase after schema or empty-stream failures.")
    rn.add_argument("--spotcheck-every", type=int, default=4,
                    help="Run a deterministic grounding check every N episodes; 0 disables.")
    rn.add_argument("--credit-cap", type=float, default=1000.0,
                    help="Hard stop for external-search credits.")
    rn.add_argument("--rpm", type=float, default=30.0)
    rn.set_defaults(fn=cmd_run)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
