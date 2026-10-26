'Internal implementation detail.'
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from convert_distill_logs import (  # noqa: E402
    TOOLKIT_ORDER,
    REASONING_TOOL,
    build_messages,
    render_initial_user,
    tool_sequence,
    validate_record,
)


def _log_entries():
    'Internal implementation detail.'
    return [
        {"step_number": 1, "timestamp": "2026-10-26T09:39:01.1",
         "step_type": "reasoning",
         "agent_reasoning": {"reasoning_steps": ["Найти высоту Эвереста",
                                                 "Сверить источник"],
                             "current_situation": "Начинаю исследование.",
                             "plan_status": "Ищу первоисточник.",
                             "enough_data": False,
                             "remaining_steps": ["Поиск"],
                             "task_completed": False}},
        {"step_number": 1, "timestamp": "2026-10-26T09:39:30.1",
         "step_type": "tool_execution", "tool_name": "websearchtool",
         "agent_tool_context": {"reasoning": "Нужна высота",
                                "query": "Эверест высота",
                                "max_results": 5},
         "agent_tool_execution_result": "Search Results: [1] ..."},
        {"step_number": 2, "timestamp": "2026-10-26T09:40:02.1",
         "step_type": "reasoning",
         "agent_reasoning": {"reasoning_steps": ["Данных достаточно"],
                             "current_situation": "Ответ найден.",
                             "plan_status": "Завершаю.",
                             "enough_data": True,
                             "remaining_steps": [],
                             "task_completed": True}},
        {"step_number": 2, "timestamp": "2026-10-26T09:40:30.1",
         "step_type": "tool_execution", "tool_name": "finalanswertool",
         "agent_tool_context": {"reasoning": "Данные подтверждены",
                                "completed_steps": ["Поиск", "Сверка"],
                                "answer": "8848 м",
                                "status": "completed"},
         "agent_tool_execution_result": "8848 м"},
    ]


def _row():
    return {"qid": "q01", "status": "ok", "keep": True,
            "tools": "reasoningtool,websearchtool,reasoningtool,finalanswertool",
            "predicted_answer": "8848 м",
            "agent_id": "abc"}


def test_build_messages_structure():
    msgs = build_messages("ВОПРОС", "Current Date: 2026-10-26 09:00:00\n",
                          "СИСТЕМА", _log_entries())
    assert [m["role"] for m in msgs] == ["system", "user", "user",
                                         "assistant", "tool",
                                         "assistant", "tool",
                                         "assistant", "tool",
                                         "assistant", "tool"]
    assert msgs[0]["content"] == "СИСТЕМА"
    assert msgs[1]["content"] == "ВОПРОС"
    assert "Current Date: 2026-10-26 09:00:00" in msgs[2]["content"]
    
    r_call = msgs[3]["tool_calls"][0]
    assert r_call["function"]["name"] == REASONING_TOOL
    assert r_call["id"] == "1-reasoning"
    args = json.loads(r_call["function"]["arguments"])
    assert args["task_completed"] is False
    assert msgs[4] == {"role": "tool", "tool_call_id": "1-reasoning", "content": ""}
    
    a_call = msgs[5]["tool_calls"][0]
    assert a_call["id"] == "1-action"
    assert a_call["function"]["name"] == "websearchtool"
    assert msgs[6]["content"] == "Search Results: [1] ..."
    r2 = json.loads(msgs[7]["tool_calls"][0]["function"]["arguments"])
    assert r2["task_completed"] is True
    f_call = msgs[9]["tool_calls"][0]
    assert f_call["id"] == "2-action"
    assert json.loads(f_call["function"]["arguments"])["answer"] == "8848 м"


def test_tool_sequence_matches_episodes_column():
    assert tool_sequence(_log_entries()) == _row()["tools"].split(",")


def test_validate_record_clean_and_warns():
    record = {"messages": build_messages("В", "Current Date: x\n", "С",
                                         _log_entries())}
    assert validate_record(record, _row()) == []
    broken = {"messages": record["messages"][:-2]}  
    warns = validate_record(broken, _row())
    assert warns
    assert any("finalanswertool" in w for w in warns)


def test_render_initial_user():
    tpl = "Current Date: {current_date} (Year-Month-Day ISO format: YYYY-MM-DD HH:MM:SS)\n"
    out = render_initial_user(tpl, "2026-10-26 09:39:01")
    assert out.startswith("Current Date: 2026-10-26 09:39:01 (")
    assert out.endswith("HH:MM:SS)\n")


def test_toolkit_order_fixed():
    assert TOOLKIT_ORDER == ("reasoningtool", "websearchtool",
                             "extractpagecontenttool", "finalanswertool")
