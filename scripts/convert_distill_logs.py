#!/usr/bin/env python3
'sgr-agent-core (`logs/*-log.json`\n`src_data/sgr_agent_loop/`\n\n\n      reasoning      → assistant(reasoningtool(args=agent_reasoning))\n      tool_execution → assistant(<tool>(args=agent_tool_context))\n\n\n\nConverts distilled episodes + agent logs into loop-format records.'
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REASONING_TOOL = "reasoningtool"



TOOLKIT_ORDER = ("reasoningtool", "websearchtool",
                 "extractpagecontenttool", "finalanswertool")




def build_messages(task_user: str, initial_user: str, system_prompt: str,
                   log_entries: list[dict]) -> list[dict]:
    'Build the episode conversation from the agent log entries.'
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_user},
        {"role": "user", "content": initial_user},
    ]
    for e in log_entries:
        n = e["step_number"]
        if e["step_type"] == "reasoning":
            msgs.append({
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "type": "function", "id": f"{n}-reasoning",
                    "function": {"name": REASONING_TOOL,
                                 "arguments": json.dumps(
                                     e["agent_reasoning"], ensure_ascii=False)},
                }],
            })
            
            
            msgs.append({"role": "tool", "tool_call_id": f"{n}-reasoning",
                         "content": e.get("agent_tool_execution_result") or ""})
        elif e["step_type"] == "tool_execution":
            msgs.append({
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "type": "function", "id": f"{n}-action",
                    "function": {"name": e["tool_name"],
                                 "arguments": json.dumps(
                                     e["agent_tool_context"],
                                     ensure_ascii=False)},
                }],
            })
            msgs.append({"role": "tool", "tool_call_id": f"{n}-action",
                         "content": e.get("agent_tool_execution_result") or ""})
        else:
            raise ValueError(f"unknown step_type: {e['step_type']!r}")
    return msgs


def tool_sequence(log_entries: list[dict]) -> list[str]:
    'Internal implementation detail.'
    seq = []
    for e in log_entries:
        if e["step_type"] == "reasoning":
            seq.append(REASONING_TOOL)
        elif e["step_type"] == "tool_execution":
            seq.append(e["tool_name"])
    return seq


def validate_record(record: dict, row: dict) -> list[str]:
    'Internal implementation detail.'
    warns = []
    names = [tc["function"]["name"] for m in record["messages"]
             for tc in m.get("tool_calls") or []]
    if names != [t for t in row["tools"].split(",") if t]:
        warns.append("последовательность инструментов не совпала с episodes.jsonl")
    finals = [tc for m in record["messages"] if m.get("role") == "assistant"
              for tc in m.get("tool_calls") or []
              if tc["function"]["name"] == "finalanswertool"]
    if len(finals) != 1:
        warns.append(f"финальных вызовов finalanswertool: {len(finals)}")
    else:
        answer = json.loads(finals[0]["function"]["arguments"]).get("answer", "")
        if answer != row.get("predicted_answer"):
            warns.append("answer финала не совпал с predicted_answer")
    return warns




def build_static_context(sgr_root: str) -> dict:
    'Internal implementation detail.'
    import httpx

    sys.path.insert(0, ".")
    from openai import AsyncOpenAI

    from sgr_agent_core.agent_config import GlobalConfig
    from sgr_agent_core.tools import (ExtractPageContentTool, FinalAnswerTool,
                                      ReasoningTool, WebSearchTool)
    from benchmark.benchmark_agent import BenchmarkAgent

    prev = Path.cwd()
    os.chdir(sgr_root)
    try:
        sys.path.insert(0, ".")
        config = GlobalConfig.from_yaml(
            str(Path(sgr_root) / "examples" / "sgr_deep_research" / "config.yaml"))
        client = AsyncOpenAI(api_key="dummy", base_url="http://localhost:9",
                             timeout=httpx.Timeout(10))
        agent = BenchmarkAgent(
            task_messages=[{"role": "user", "content": "ВОПРОС"}],
            openai_client=client, agent_config=config,
            toolkit=[ReasoningTool, WebSearchTool, ExtractPageContentTool,
                     FinalAnswerTool],
            tool_configs={"websearchtool": {"engine": "tavily", "api_key": "x"},
                          "extractpagecontenttool": {"tavily_api_key": "x"}})
        
        
        
        agent.max_iterations = 12
        agent.max_searches = 4
        tools = asyncio_run(agent._prepare_tools())
        ctx = asyncio_run(agent._prepare_context())
    finally:
        os.chdir(prev)
    return {
        "system_prompt": ctx[0]["content"],
        
        
        "initial_user_template": config.prompts.initial_user_request,
        "tools": sorted(tools, key=lambda t: TOOLKIT_ORDER.index(
            t["function"]["name"])),
    }


def asyncio_run(coro):
    import asyncio
    return asyncio.run(coro)


def render_initial_user(template: str, current_date: str) -> str:
    'Internal implementation detail.'
    return template.format(current_date=current_date)




def convert(episodes_path: str, logs_dir: str, static: dict,
            out_path: str) -> list[dict]:
    rows = [json.loads(l) for l in
            Path(episodes_path).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    kept = [r for r in rows
            if r.get("status") == "ok" and r.get("keep")
            and r.get("grounded") != "ungrounded"]
    records = []
    for row in kept:
        
        matches = sorted(Path(logs_dir).glob(f"*{row['agent_id']}-log.json"))
        if not matches:
            print(f"SKIP {row['qid']}: no log for {row['agent_id']}",
                  file=sys.stderr)
            continue
        log_path = matches[-1]  
        agent_log = json.loads(log_path.read_text(encoding="utf-8"))
        first_ts = agent_log["log"][0]["timestamp"].replace("T", " ").split(".")[0]
        initial_user = render_initial_user(static["initial_user_template"],
                                           first_ts)
        task_user = agent_log["task_messages"][0]["content"]
        record = {
            "id": f"distill-b-{row['qid']}",
            "task_class": "distill_b_loop",
            "lang": row["lang"],
            "question": row["question"],
            "messages": build_messages(task_user, initial_user,
                                       static["system_prompt"],
                                       agent_log["log"]),
            "tools": static["tools"],
            "extra": {
                "source": "distill_b",
                "agent_id": row["agent_id"],
                "n_search": row.get("n_search"),
                "n_extract": row.get("n_extract"),
                "grounded": row.get("grounded"),
                "trap": row.get("trap", False),
            },
        }
        warns = validate_record(record, row)
        for w in warns:
            print(f"WARNING {row['qid']}: {w}", file=sys.stderr)
        records.append(record)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", default="distill_out/episodes.jsonl")
    p.add_argument("--logs-dir",
                   default="/path/to/user/sgr-agent-core/logs")
    p.add_argument("--sgr-root",
                   default="/path/to/user/sgr-agent-core")
    p.add_argument("--static-cache",
                   help="JSON cache for the static agent context.")
    p.add_argument("--out",
                   default=str(REPO / "src_data" / "distill_loop"
                               / "train_distill_b_loop.jsonl"))
    args = p.parse_args(argv)

    static = None
    if args.static_cache and Path(args.static_cache).exists():
        static = json.loads(Path(args.static_cache).read_text(encoding="utf-8"))
        print(f"static context loaded from cache: {args.static_cache}")
    else:
        static = build_static_context(args.sgr_root)
        if args.static_cache:
            Path(args.static_cache).write_text(
                json.dumps(static, ensure_ascii=False), encoding="utf-8")
            print(f"static context saved: {args.static_cache}")

    records = convert(args.episodes, args.logs_dir, static, args.out)
    langs = {}
    for r in records:
        langs[r["lang"]] = langs.get(r["lang"], 0) + 1
    print(f"records: {len(records)} | {langs} | {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
