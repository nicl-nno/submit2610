'Internal implementation detail.'
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from convert_sgr_to_loop import (  # noqa: E402
    DEFAULT_TRAINABLE,
    REASONINGTOOL_SCHEMA,
    REASONING_TOOL,
    convert_record,
    main,
    normalize_reasoning_args,
    parse_reasoning,
    truncate_at_sentence,
)

REASONING_TEXT = (
    "Reasoning steps:\n- First step\n- Second step\n"
    "Current situation: Started the task, nothing found yet.\n"
    "Plan status: Searching for sources.\n"
    "Remaining steps:\n- Search\n- Extract pages"
)
REASONING_TEXT_FINAL = (
    "Reasoning steps:\n- All found\n"
    "Current situation: Enough evidence collected.\n"
    "Plan status: Ready to finalize.\n"
    "Remaining steps:\n- Give final answer"
)
PARTIAL_TEXT = "Reasoning steps:\n- Only steps here, other markers missing."


def _tool_call(idx: int, name: str, args: dict) -> dict:
    return {
        "type": "function",
        "id": f"{idx}-action",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _record() -> dict:
    return {
        "tools": [
            {"type": "function", "function": {"name": "websearchtool",
                                              "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "finalanswertool",
                                              "parameters": {"type": "object"}}},
        ],
        "messages": [
            {"role": "system", "content": "task"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": REASONING_TEXT},
            {"role": "assistant", "content": "search", "tool_calls": [
                _tool_call(1, "websearchtool", {"query": "q"})]},
            {"role": "tool", "tool_call_id": "1-action", "content": "results"},
            {"role": "assistant", "content": REASONING_TEXT_FINAL},
            {"role": "assistant", "content": "done",
             "tool_calls": [_tool_call(2, "finalanswertool", {"answer": "a"})]},
        ],
    }


def test_parse_reasoning_full() -> None:
    parsed = parse_reasoning(REASONING_TEXT)
    assert parsed is not None
    assert parsed["reasoning_steps"] == ["First step", "Second step"]
    assert parsed["current_situation"] == "Started the task, nothing found yet."
    assert parsed["plan_status"] == "Searching for sources."
    assert parsed["remaining_steps"] == ["Search", "Extract pages"]


def test_parse_reasoning_rejects_incomplete() -> None:
    assert parse_reasoning(PARTIAL_TEXT) is None
    assert parse_reasoning("") is None
    no_steps = REASONING_TEXT.replace("Reasoning steps:\n- First step\n- Second step\n", "")
    assert parse_reasoning(no_steps) is None


def test_canonical_schema() -> None:
    'Internal implementation detail.'
    fn = REASONINGTOOL_SCHEMA["function"]
    assert fn["name"] == REASONING_TOOL
    assert fn.get("description") == ""            
    assert "strict" not in fn
    props = fn["parameters"]["properties"]
    steps = props["reasoning_steps"]
    assert steps["minItems"] == 2 and steps["maxItems"] == 3
    assert props["current_situation"]["maxLength"] == 300
    assert props["plan_status"]["maxLength"] == 150
    assert props["remaining_steps"]["maxItems"] == 3
    assert fn["parameters"]["required"] == [
        "reasoning_steps", "current_situation", "plan_status",
        "enough_data", "remaining_steps", "task_completed"]


def test_truncate_at_sentence() -> None:
    assert truncate_at_sentence("Коротко.", 300) == "Коротко."
    long_text = ("Первое предложение раскрывает суть вопроса. " * 4).strip()
    cut = truncate_at_sentence(long_text, 60)
    assert len(cut) <= 60 and cut.endswith(".")
    
    hard = truncate_at_sentence("а" * 40 + ". " + "б" * 100, 30)
    assert len(hard) <= 30


def test_normalize_reasoning_args() -> None:
    
    out, changed = normalize_reasoning_args({
        "reasoning_steps": ["a", "b", "c", "d", "e"],
        "current_situation": "ок", "plan_status": "ок",
        "remaining_steps": ["r1", "r2", "r3", "r4"]})
    assert changed and out["reasoning_steps"] == ["a", "b", "c"]
    assert out["remaining_steps"] == ["r1", "r2", "r3"]
    
    out, changed = normalize_reasoning_args({
        "reasoning_steps": ["Нужно найти данные. Затем сверить факты."],
        "current_situation": "ок", "plan_status": "ок", "remaining_steps": []})
    assert changed and len(out["reasoning_steps"]) == 2
    
    long = ("Предложение о ситуации номер один. " * 30).strip()
    out, changed = normalize_reasoning_args({
        "reasoning_steps": ["a", "b"], "current_situation": long,
        "plan_status": "s" * 200, "remaining_steps": []})
    assert changed
    assert len(out["current_situation"]) <= 300 and out["current_situation"].endswith(".")
    assert len(out["plan_status"]) <= 150
    
    out, changed = normalize_reasoning_args({
        "reasoning_steps": ["a", "b"], "current_situation": "ок",
        "plan_status": "ок", "remaining_steps": ["r"]})
    assert not changed


def test_convert_record_flags_and_echo() -> None:
    rec = _record()
    out, stats = convert_record(rec)
    msgs = out["messages"]
    reasoning_calls = [m for m in msgs if m.get("role") == "assistant"
                       and m.get("tool_calls")
                       and m["tool_calls"][0]["function"]["name"] == REASONING_TOOL]
    assert len(reasoning_calls) == 2
    args0 = json.loads(reasoning_calls[0]["tool_calls"][0]["function"]["arguments"])
    args1 = json.loads(reasoning_calls[1]["tool_calls"][0]["function"]["arguments"])
    assert (args0["task_completed"], args0["enough_data"]) == (False, False)
    assert (args1["task_completed"], args1["enough_data"]) == (True, True)
    ids = [c["tool_calls"][0]["id"] for c in reasoning_calls]
    assert ids == ["1-reasoning", "2-reasoning"]
    echoes = [m for m in msgs if m.get("role") == "tool"
              and str(m.get("tool_call_id", "")).endswith("-reasoning")]
    assert sorted(e["tool_call_id"] for e in echoes) == sorted(ids)
    for e in echoes:
        assert json.loads(e["content"])["reasoning_steps"]
    # action/final
    by_call_id = {m.get("tool_call_id"): m for m in msgs if m.get("role") == "tool"}
    assert by_call_id["1-action"] == rec["messages"][4]
    final_out = [m for m in msgs if m.get("role") == "assistant"
                 and m.get("tool_calls")
                 and m["tool_calls"][0]["function"]["name"] == "finalanswertool"][0]
    assert final_out == rec["messages"][-1]
    
    assert out["tools"][0]["function"]["name"] == REASONING_TOOL
    assert out["tools"][1:] == rec["tools"]
    
    # ("All found")
    assert stats == {"reasoning": 2, "converted": 2, "normalized": 1,
                     "folded": 0, "action": 1, "final": 1}
    
    assert "tool_calls" not in rec["messages"][2]


def test_convert_record_normalizes_long_args() -> None:
    rec = _record()
    long_situation = ("Предложение о текущей ситуации. " * 30).strip()
    rec["messages"][2] = {"role": "assistant", "content": (
        "Reasoning steps:\n- a\n- b\n- c\n- d\n"
        f"Current situation: {long_situation}\n"
        "Plan status: ok.\n"
        "Remaining steps:\n- r1\n- r2\n- r3\n- r4")}
    out, stats = convert_record(rec)
    # 1
    assert stats["normalized"] == 2
    call = [m for m in out["messages"] if m.get("role") == "assistant"
            and m.get("tool_calls")
            and m["tool_calls"][0]["function"]["name"] == REASONING_TOOL][0]
    args = json.loads(call["tool_calls"][0]["function"]["arguments"])
    assert args["reasoning_steps"] == ["a", "b", "c"]
    assert len(args["current_situation"]) <= 300
    assert args["remaining_steps"] == ["r1", "r2", "r3"]


def test_convert_record_folds_partial() -> None:
    rec = _record()
    rec["messages"][2] = {"role": "assistant", "content": PARTIAL_TEXT}
    out, stats = convert_record(rec)
    assert stats["folded"] == 1 and stats["converted"] == 1
    assert any(m.get("role") == "assistant" and m.get("content") == PARTIAL_TEXT
               and not m.get("tool_calls") for m in out["messages"])


def test_convert_idempotent() -> None:
    rec = _record()
    once, _ = convert_record(rec)
    twice, stats2 = convert_record(once)
    assert once == twice
    assert stats2["reasoning"] == 0  


def test_termination_split_and_manifest(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "train_deep_research.jsonl").write_text(
        json.dumps(_record(), ensure_ascii=False) + "\n", encoding="utf-8")
    val_recs = [_record() for _ in range(3)]
    (src / "val_deep_research.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in val_recs),
        encoding="utf-8")
    dst = tmp_path / "dst"
    rc = main(["--src", str(src), "--dst", str(dst),
               "--task", "deep_research", "--termination-take", "2"])
    assert rc == 0
    carved = [json.loads(l) for l in
              (dst / "termination_val" / "deep_research.jsonl").read_text().splitlines()]
    assert len(carved) == 2
    val_out = [json.loads(l) for l in
               (dst / "val_deep_research.jsonl").read_text().splitlines()]
    assert len(val_out) == 1
    manifest = json.loads((dst / "MANIFEST.json").read_text())
    assert manifest["classes"]["deep_research"]["termination_carved"] == 2
    assert manifest["inputs"]


@pytest.mark.parametrize("cls", DEFAULT_TRAINABLE)
def test_real_catalog_counts(cls: str, tmp_path: Path) -> None:
    'Internal implementation detail.'
    dst = tmp_path / "loop"
    rc = main(["--dst", str(dst)])
    assert rc == 0
    manifest = json.loads((dst / "MANIFEST.json").read_text())
    total = manifest["classes"][cls]
    assert total["folded"] + total["converted"] == total["reasoning"]
    
    assert total["final"] == total["records"]


def test_real_catalog_totals(tmp_path: Path) -> None:
    'Internal implementation detail.'
    dst = tmp_path / "loop"
    main(["--dst", str(dst)])
    manifest = json.loads((dst / "MANIFEST.json").read_text())
    grand = {k: sum(c[k] for c in manifest["classes"].values())
             for k in ("records", "reasoning", "converted", "normalized",
                       "folded", "action", "final", "termination_carved")}
    assert grand == {"records": 110, "reasoning": 175, "converted": 170,
                     "normalized": 8, "folded": 5, "action": 139,
                     "final": 110, "termination_carved": 3}
