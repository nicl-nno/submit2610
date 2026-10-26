'(Current Date'
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "llm_training"))
sys.path.insert(0, str(REPO))

from llm_training.config import DataConfig  # noqa: E402
from llm_training.loaders import LOADERS  # noqa: E402
from llm_training.loaders.distill_loop import CLASS, load_class  # noqa: E402


def _record():
    'Internal implementation detail.'
    def call(rid, name, args):
        return {"type": "function", "id": rid,
                "function": {"name": name,
                             "arguments": json.dumps(args, ensure_ascii=False)}}
    return {
        "id": "distill-b-q45", "task_class": "distill_b_loop", "lang": "ru",
        "question": "Расскажи о статье про RAGU на конференции EMNLP-2026.",
        "messages": [
            {"role": "system", "content": "СИСТЕМА"},
            {"role": "user", "content": "Расскажи о статье про RAGU…"},
            {"role": "user",
             "content": "Current Date: 2026-10-26 09:39:01 (Year-Month-Day)"},
            {"role": "assistant", "content": "", "tool_calls": [call(
                "1-reasoning", "reasoningtool",
                {"reasoning_steps": ["Ищу", "Сверяю"],
                 "current_situation": "Начинаю.", "plan_status": "Ищу.",
                 "enough_data": False, "remaining_steps": ["Поиск"],
                 "task_completed": False})]},
            {"role": "tool", "tool_call_id": "1-reasoning", "content": ""},
            {"role": "assistant", "content": "", "tool_calls": [call(
                "1-action", "websearchtool",
                {"reasoning": "Ищу статью", "query": "RAGU EMNLP 2026",
                 "max_results": 5})]},
            {"role": "tool", "tool_call_id": "1-action",
             "content": "Search Results: [1] …"},
            {"role": "assistant", "content": "", "tool_calls": [call(
                "2-reasoning", "reasoningtool",
                {"reasoning_steps": ["Не найдено"],
                 "current_situation": "Статьи нет.", "plan_status": "Завершаю.",
                 "enough_data": True, "remaining_steps": [],
                 "task_completed": True})]},
            {"role": "tool", "tool_call_id": "2-reasoning", "content": ""},
            {"role": "assistant", "content": "", "tool_calls": [call(
                "2-action", "finalanswertool",
                {"reasoning": "Данных нет", "completed_steps": ["Поиск"],
                 "answer": "Статья не найдена",
                 "status": "completed"})]},
            {"role": "tool", "tool_call_id": "2-action",
             "content": "Статья не найдена"},
        ],
        "tools": [{"type": "function", "function": {"name": "reasoningtool"}}],
        "extra": {"trap": True},
    }


def _load_all(tmp_path):
    (tmp_path / "distill_loop").mkdir(exist_ok=True)
    (tmp_path / "distill_loop" / "train_distill_b_loop.jsonl").write_text(
        json.dumps(_record(), ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = DataConfig(data_dir=str(tmp_path))
    return list(load_class(CLASS, None, cfg))


def test_registry_wires_the_class():
    assert CLASS == "distill_b_loop"
    assert LOADERS[CLASS] is not None


def test_turn_split_and_flags(tmp_path):
    exs = _load_all(tmp_path)
    
    assert len(exs) == 4
    
    assert exs[0].extra["is_call_turn"] is True
    assert "is_final_answer" not in exs[0].extra
    assert json.loads(json.dumps(
        {"name": "reasoningtool", "arguments": json.loads(
            exs[0].reference)}))["name"] == "reasoningtool"
    
    assert exs[1].extra["is_call_turn"] is True
    assert "websearchtool" in exs[1].reference
    
    assert exs[3].extra["is_final_answer"] is True
    assert exs[3].reference == "Статья не найдена"
    
    for ex in exs:
        assert ex.extra["lang"] == "ru" and ex.extra["is_trap"] is True
        assert ex.source.startswith("distill_loop/train_distill_b_loop.jsonl#")


def test_current_date_folded_into_prompt(tmp_path):
    'Current Date'
    exs = _load_all(tmp_path)
    for ex in exs:
        assert any("Current Date" in m.get("content", "")
                   for m in ex.prompt if m.get("role") == "user")


def test_tools_passthrough(tmp_path):
    exs = _load_all(tmp_path)
    assert exs[0].tools == _record()["tools"]


def test_empty_file_yields_nothing(tmp_path):
    (tmp_path / "distill_loop").mkdir(exist_ok=True)
    (tmp_path / "distill_loop" / "train_distill_b_loop.jsonl").write_text(
        "", encoding="utf-8")
    cfg = DataConfig(data_dir=str(tmp_path))
    assert list(load_class(CLASS, None, cfg)) == []
