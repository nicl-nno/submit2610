'Offline tests of the QA prompt builder (no network, no GPU).'

from __future__ import annotations

import pytest

from llm_training.extraction.qa_prompts import (
    QA_SYSTEM,
    QA_TASK_CLASS,
    build_qa_prompt,
)


def test_closed_book_prompt() -> None:
    prompt = build_qa_prompt("Столица Франции?", None, language="Russian")
    assert [m["role"] for m in prompt] == ["system", "user"]
    system, user = prompt[0]["content"], prompt[1]["content"]
    assert "Russian" in system
    assert "Context document" not in user
    assert user == "Question: Столица Франции?"


def test_open_book_prompt_prepends_context() -> None:
    prompt = build_qa_prompt(
        "Столица Франции?", "Франция — государство в Европе.",
        language="Russian",
    )
    user = prompt[1]["content"]
    assert user.startswith("Context document:\n")
    assert "Франция — государство" in user
    assert user.endswith("Question: Столица Франции?")
    # The open-book system prompt carries the context rule.
    assert "context document" in prompt[0]["content"].lower()


def test_compact_variant_differs() -> None:
    full = build_qa_prompt("q", None, language="English", variant=0)
    compact = build_qa_prompt("q", None, language="English", variant=1)
    assert full[0]["content"] != compact[0]["content"]
    assert "Answer in English" in compact[0]["content"]


def test_empty_question_rejected() -> None:
    with pytest.raises(ValueError):
        build_qa_prompt("   ", "text", language="Russian")


def test_task_class_constant() -> None:
    assert QA_TASK_CLASS == "fineweb_qa"
    # The system template promises a SHORT answer (the reward's compression
    # band and the reference-based scoring both rely on it).
    assert "SHORT" in QA_SYSTEM
