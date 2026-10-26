'Internal implementation detail.'
from __future__ import annotations

import json

import pytest

from llm_training.config import DataConfig
from llm_training.loaders import sgr_agent

LOOP_SUBDIR = "sgr_agent_loop"


def _examples(cls, data_dir, loop: bool):
    cfg = DataConfig(
        data_dir=data_dir,
        sgr_loop_subdir=LOOP_SUBDIR if loop else None,
    )
    return list(sgr_agent.load_class(cls, None, cfg))


def test_loop_subdir_preferred(tokenizer, data_dir):
    'Internal implementation detail.'
    examples = _examples("deep_research", data_dir, loop=True)
    assert examples
    assert all(ex.source.startswith(f"{LOOP_SUBDIR}/") for ex in examples)
    
    
    
    manifest = json.loads(
        (data_dir / LOOP_SUBDIR / "MANIFEST.json").read_text(encoding="utf-8"))
    folded = manifest["classes"]["deep_research"]["folded"]
    textual = sum(1 for ex in examples if (ex.extra or {}).get("is_reasoning_turn"))
    assert textual == folded
    
    for ex in examples:
        names = [(t.get("function") or {}).get("name") for t in (ex.tools or [])]
        assert "reasoningtool" in names or not ex.tools


def test_holdout_falls_back(tokenizer, data_dir):
    'Internal implementation detail.'
    examples = _examples("summarizer", data_dir, loop=True)
    assert examples
    assert all(ex.source.startswith("sgr_agent/") for ex in examples)
    assert any((ex.extra or {}).get("is_reasoning_turn") for ex in examples)


def test_knob_off_reads_original(tokenizer, data_dir):
    examples = _examples("deep_research", data_dir, loop=False)
    assert examples
    assert all(ex.source.startswith("sgr_agent/") for ex in examples)
    assert any((ex.extra or {}).get("is_reasoning_turn") for ex in examples)


def test_missing_loop_file_falls_back(tmp_path):
    'Internal implementation detail.'
    src = tmp_path / "src_data"
    (src / "sgr_agent").mkdir(parents=True)
    (src / "sgr_agent_loop").mkdir()
    rec = {
        "tools": [{"type": "function", "function": {
            "name": "finalanswertool", "parameters": {"type": "object"}}}],
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "type": "function", "id": "1-action",
                "function": {"name": "finalanswertool",
                             "arguments": json.dumps({"answer": "a"})}}]},
        ],
    }
    (src / "sgr_agent" / "train_deep_research.jsonl").write_text(
        json.dumps(rec) + "\n", encoding="utf-8")
    cfg = DataConfig(data_dir=src, sgr_loop_subdir=LOOP_SUBDIR)
    examples = list(sgr_agent.load_class("deep_research", None, cfg))
    assert len(examples) == 1
    assert examples[0].source.startswith("sgr_agent/")
