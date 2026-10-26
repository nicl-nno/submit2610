#!/usr/bin/env python3
'``src_data/sgr_agent_loop/``\n\n  ``assistant(tool_calls=[reasoningtool(...)])`` +\n  ``role:"tool"``\n* ``enough_data``/``task_completed``\n  (current_situation ≤300, plan_status ≤150\n\n\n(data_analyst, deep_research, factual_qa\n``--termination-take``\n\nSgr loop conversion: reasoning-as-plain-text becomes reasoningtool calls.'
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


# llm_training/reward/scoring.py::_SGR_REASONING_MARKERS

MARKER_REASONING_STEPS = "Reasoning steps:"
MARKER_CURRENT_SITUATION = "Current situation:"
MARKER_PLAN_STATUS = "Plan status:"
MARKER_REMAINING_STEPS = "Remaining steps:"
_MARKERS = (
    MARKER_REASONING_STEPS,
    MARKER_CURRENT_SITUATION,
    MARKER_PLAN_STATUS,
    MARKER_REMAINING_STEPS,
)

FINAL_ANSWER_TOOL = "finalanswertool"
REASONING_TOOL = "reasoningtool"


# Trainable classes of the 0.2 series (val hold-outs excluded).
DEFAULT_TRAINABLE = (
    "coder",
    "creative_writer",
    "data_analyst",
    "deep_research",
    "factual_qa",
    "instruction_following",
    "rewriter",
)

MULTISTEP_TRAINABLE = ("data_analyst", "deep_research", "factual_qa")

# JSON-schema
# sgr-agent-core (sgr_agent_core/tools/reasoning_tool.py, pydantic
# ReasoningTool): minItems/maxItems 2–3


REASONINGTOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": REASONING_TOOL,
        "description": "",
        "parameters": {
            "properties": {
                "reasoning_steps": {
                    "description": "Step-by-step reasoning (brief, 1 sentence each)",
                    "items": {"type": "string"},
                    "maxItems": 3,
                    "minItems": 2,
                    "title": "Reasoning Steps",
                    "type": "array",
                },
                "current_situation": {
                    "description": "Current research situation (2-3 sentences MAX)",
                    "maxLength": 300,
                    "title": "Current Situation",
                    "type": "string",
                },
                "plan_status": {
                    "description": "Status of current plan (1 sentence)",
                    "maxLength": 150,
                    "title": "Plan Status",
                    "type": "string",
                },
                "enough_data": {
                    "default": False,
                    "description": "Sufficient data collected for comprehensive report?",
                    "title": "Enough Data",
                    "type": "boolean",
                },
                "remaining_steps": {
                    "description": "0-3 remaining steps (brief, action-oriented)",
                    "items": {"type": "string"},
                    "maxItems": 3,
                    "title": "Remaining Steps",
                    "type": "array",
                },
                "task_completed": {
                    "description": "Is the research task finished?",
                    "title": "Task Completed",
                    "type": "boolean",
                },
            },
            "required": [
                "reasoning_steps",
                "current_situation",
                "plan_status",
                "enough_data",
                "remaining_steps",
                "task_completed",
            ],
            "type": "object",
        },
    },
}



_STRING_LIMITS = (("current_situation", 300), ("plan_status", 150))
_MAX_STEPS_ITEMS = 3
_MIN_REASONING_ITEMS = 2
_SENTENCE_END = re.compile(r"[.!?…](?=\s|$)")


def truncate_at_sentence(text: str, limit: int) -> str:
    'Truncate at the last sentence boundary fitting the limit.'
    if len(text) <= limit:
        return text
    cut = None
    for m in _SENTENCE_END.finditer(text, 0, limit):
        cut = m.end()
    if cut and cut >= limit // 3:
        return text[:cut].strip()
    return text[:limit].rstrip()


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def normalize_reasoning_args(args: dict) -> tuple[dict, bool]:
    'reasoning_steps\n    Normalize the parsed arguments to the canonical tool schema.'
    out = dict(args)
    changed = False
    steps = [s for s in (out.get("reasoning_steps") or []) if str(s).strip()]
    if len(steps) > _MAX_STEPS_ITEMS:
        steps = steps[:_MAX_STEPS_ITEMS]
        changed = True
    elif len(steps) == 1:
        parts = _split_sentences(steps[0])
        steps = (parts[:2] if len(parts) >= _MIN_REASONING_ITEMS
                 else [steps[0], steps[0]])
        changed = True
    out["reasoning_steps"] = steps
    for key, limit in _STRING_LIMITS:
        val = str(out.get(key) or "")
        new = truncate_at_sentence(val, limit)
        if new != val:
            out[key] = new
            changed = True
    remaining = [s for s in (out.get("remaining_steps") or []) if str(s).strip()]
    if len(remaining) > _MAX_STEPS_ITEMS:
        out["remaining_steps"] = remaining[:_MAX_STEPS_ITEMS]
        changed = True
    return out, changed


def parse_reasoning(text: str) -> dict | None:
    'Parse the plain-text reasoning into reasoningtool arguments.'
    if not text:
        return None
    lower = text.lower()
    positions = []
    for marker in _MARKERS:
        idx = lower.find(marker.lower())
        if idx < 0:
            return None
        positions.append((idx, marker))
    positions.sort()
    sections: dict[str, str] = {}
    for k, (idx, marker) in enumerate(positions):
        end = positions[k + 1][0] if k + 1 < len(positions) else len(text)
        sections[marker] = text[idx + len(marker):end].strip()

    def bullets(block: str) -> list[str]:
        out = []
        for line in block.splitlines():
            line = line.strip().lstrip("-*•").strip()
            if line:
                out.append(line)
        return out

    reasoning_steps = bullets(sections[MARKER_REASONING_STEPS])
    remaining_steps = bullets(sections[MARKER_REMAINING_STEPS])
    current_situation = " ".join(sections[MARKER_CURRENT_SITUATION].split())
    plan_status = " ".join(sections[MARKER_PLAN_STATUS].split())
    if not reasoning_steps or not current_situation or not plan_status:
        return None
    return {
        "reasoning_steps": reasoning_steps,
        "current_situation": current_situation,
        "plan_status": plan_status,
        "remaining_steps": remaining_steps,
    }


def _is_final_turn(msg: dict) -> bool:
    for tc in msg.get("tool_calls") or []:
        if (tc.get("function") or {}).get("name") == FINAL_ANSWER_TOOL:
            return True
    return False


def _next_assistant_is_final(messages: list[dict], from_idx: int) -> bool:
    'finalanswertool.\n\n    Whether the next assistant turn finalizes the trajectory.'
    for msg in messages[from_idx + 1:]:
        if msg.get("role") == "assistant":
            return _is_final_turn(msg)
    return False


def _reasoning_call(call_id: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "id": call_id,
                "function": {
                    "name": REASONING_TOOL,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        ],
    }


def _echo_result(call_id: str, args: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(args, ensure_ascii=False, indent=2),
    }


def convert_record(rec: dict) -> tuple[dict, dict]:
    'Convert one trajectory; original record is not mutated.'
    stats = {"reasoning": 0, "converted": 0, "normalized": 0, "folded": 0,
             "action": 0, "final": 0}
    messages = rec.get("messages") or []
    out: list[dict] = []
    reasoning_idx = 0
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            out.append(msg)
            continue
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            stats["final" if _is_final_turn(msg) else "action"] += 1
            out.append(msg)
            continue
        stats["reasoning"] += 1
        parsed = parse_reasoning(msg.get("content") or "")
        if parsed is None:
            
            stats["folded"] += 1
            out.append(msg)
            continue
        final_next = _next_assistant_is_final(messages, i)
        args = dict(parsed)
        args, changed = normalize_reasoning_args(args)
        if changed:
            stats["normalized"] += 1
        args["enough_data"] = final_next
        args["task_completed"] = final_next
        reasoning_idx += 1
        call_id = f"{reasoning_idx}-reasoning"
        out.append(_reasoning_call(call_id, args))
        out.append(_echo_result(call_id, args))
        stats["converted"] += 1
    tools = list(rec.get("tools") or [])
    if tools and not any(
        (t.get("function") or {}).get("name") == REASONING_TOOL for t in tools
    ):
        tools = [REASONINGTOOL_SCHEMA, *tools]
    new_rec = dict(rec)
    new_rec["messages"] = out
    new_rec["tools"] = tools
    return new_rec, stats


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _has_convertible_reasoning(rec: dict) -> bool:
    return any(
        (m.get("role") == "assistant" and not (m.get("tool_calls") or [])
         and parse_reasoning(m.get("content") or "") is not None)
        for m in rec.get("messages") or []
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, default=REPO / "src_data" / "sgr_agent")
    p.add_argument("--dst", type=Path, default=REPO / "src_data" / "sgr_agent_loop")
    p.add_argument("--task", action="append",
                   help="Trainable task class; repeatable (defaults to all DEFAULT_TRAINABLE classes).")
    p.add_argument("--termination-take", type=int, default=10,
                   help="Validation trajectories per class moved to termination_val (0 disables).")
    args = p.parse_args(argv)

    tasks = args.task or list(DEFAULT_TRAINABLE)
    dst = args.dst
    dst.mkdir(parents=True, exist_ok=True)
    manifest = {"converter": "convert_sgr_to_loop", "inputs": {}, "classes": {}}
    for task in tasks:
        total = {"records": 0, "reasoning": 0, "converted": 0, "normalized": 0,
                 "folded": 0, "action": 0, "final": 0, "termination_carved": 0}
        for split in ("train", "val"):
            src_file = args.src / f"{split}_{task}.jsonl"
            if not src_file.exists():
                continue
            recs = [json.loads(l) for l in src_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            manifest["inputs"][str(src_file)] = _sha256(src_file)
            carved = 0
            if (split == "val" and args.termination_take > 0
                    and task in MULTISTEP_TRAINABLE):
                keep, taken = [], []
                for rec in recs:  
                    keep.append(rec)
                for rec in reversed(recs):
                    if len(taken) >= args.termination_take:
                        break
                    if _has_convertible_reasoning(rec):
                        taken.append(rec)
                        keep.remove(rec)
                taken.reverse()  
                carved = len(taken)
                tv_dir = dst / "termination_val"
                tv_dir.mkdir(exist_ok=True)
                with open(tv_dir / f"{task}.jsonl", "w", encoding="utf-8") as f:
                    for rec in taken:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                recs = keep
            out_file = dst / f"{split}_{task}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for rec in recs:
                    new_rec, stats = convert_record(rec)
                    f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                    for k in stats:
                        total[k] += stats[k]
            total["records"] += len(recs)
            if split == "val":
                total["termination_carved"] += carved
        manifest["classes"][task] = total
        print(f"{task:22s} records={total['records']:3d} "
              f"reasoning={total['reasoning']:3d} converted={total['converted']:3d} "
              f"normalized={total['normalized']:3d} folded={total['folded']:3d} "
              f"action={total['action']:3d} final={total['final']:3d} "
              f"termination_carved={total['termination_carved']}")
    with open(dst / "MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    grand = {k: sum(c[k] for c in manifest["classes"].values())
             for k in ("records", "reasoning", "converted", "normalized",
                       "folded", "action", "final", "termination_carved")}
    print("TOTAL", json.dumps(grand))
    return 0


if __name__ == "__main__":
    sys.exit(main())
