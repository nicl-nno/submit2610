'Tests for warmup/metrics.py — harmonic(format-side, chrF++) scoring.\n\n\nAll offline: canned generations + references, no GPU, no model.'

from __future__ import annotations

import json

import pytest

from llm_training.warmup.metrics import (
    aggregate_warmup_scores,
    chrf_plus_plus,
    warmup_turn_score,
)

TOOLS = frozenset({"finalanswertool", "pandas_analysis_tool"})


def _final_extra():
    return {"is_call_turn": True, "is_final_answer": True, "turn_index": 0}


def _action_extra():
    return {"is_call_turn": True, "turn_index": 1}


def _reasoning_extra():
    return {"is_call_turn": False, "is_reasoning_turn": True, "turn_index": 0}


def _envelope(answer: str, name: str = "finalanswertool") -> str:
    payload = json.dumps(
        {"name": name, "arguments": {"answer": answer, "status": "completed"}},
        ensure_ascii=False,
    )
    return f"<tool_call>\n{payload}\n</tool_call>"


class TestChrfPlusPlus:
    def test_identical_is_one(self):
        assert chrf_plus_plus("мама мыла раму", "мама мыла раму") == pytest.approx(1.0)

    def test_disjoint_is_low(self):
        assert chrf_plus_plus("абсолютно другой текст", "совсем иные слова") < 0.3

    def test_empty_is_zero(self):
        assert chrf_plus_plus("", "референс") == 0.0
        assert chrf_plus_plus("гипотеза", "") == 0.0

    def test_in_unit_interval(self):
        v = chrf_plus_plus("частично совпадающий текст", "частично другой текст")
        assert 0.0 <= v <= 1.0


class TestWarmupTurnScoreFinal:
    def test_perfect_generation_is_one(self):
        # The reference answer in a valid finalanswertool envelope + a good
        # chrF++ on the extracted answer → harmonic ≈ 1.
        ref = "Ответ на вопрос задачи с деталями."
        gen = _envelope(ref)
        s = warmup_turn_score(gen, ref, "coder", _final_extra(), TOOLS)
        assert s == pytest.approx(1.0)

    def test_no_envelope_is_zero(self):
        # A near-miss (bare JSON payload, no <tool_call> envelope) — the exact
        # failure mode of the cold start — must score 0 via the strict
        # harmonic mean (format half = 0).
        ref = "Ответ на вопрос задачи."
        gen = json.dumps(
            {"name": "finalanswertool",
             "arguments": {"answer": ref, "status": "completed"}},
            ensure_ascii=False,
        )
        assert warmup_turn_score(gen, ref, "coder", _final_extra(), TOOLS) == 0.0

    def test_valid_envelope_wrong_content_is_low(self):
        # Envelope valid but the answer is wrong → format = 1.0, chrF small
        # (chrF++ has an inherent ~0.3 floor on disjoint Russian texts —
        # characters overlap) → the harmonic is well below the perfect 1.0.
        gen = _envelope("Полностью нерелевантный ответ про что-то другое.")
        perfect = warmup_turn_score(
            _envelope("Правильный ответ про Rust и владение."),
            "Правильный ответ про Rust и владение.",
            "coder", _final_extra(), TOOLS,
        )
        s = warmup_turn_score(gen, "Правильный ответ про Rust и владение.", "coder", _final_extra(), TOOLS)
        assert perfect == pytest.approx(1.0)
        assert s < 0.6  # well below perfect; both halves must be good

    def test_unknown_tool_name_gives_zero_answer(self):
        # A call to a wrong tool → extract_final_answer finds no finalanswertool
        # → the content half is empty → the strict harmonic zeroes the turn.
        # (By design: a wrong-tool call has no answer to compare.)
        gen = _envelope("Ответ на вопрос задачи с деталями.", name="nonexistent_tool")
        s = warmup_turn_score(gen, "Ответ на вопрос задачи с деталями.", "coder", _final_extra(), TOOLS)
        assert s == 0.0


class TestWarmupTurnScoreAction:
    def test_valid_action_call(self):
        payload = json.dumps(
            {"name": "pandas_analysis_tool",
             "arguments": {"csv_path": "population.csv", "operation": "head"}},
            ensure_ascii=False,
        )
        gen = f"<tool_call>\n{payload}\n</tool_call>"
        # The action reference is the rendered turn; use the same text as the
        # reference so chrF++ is high and the format side is what we test.
        s = warmup_turn_score(gen, gen, "data_analyst", _action_extra(), TOOLS)
        assert s > 0.9

    def test_plain_text_action_is_zero(self):
        # No envelope at all → format 0 → harmonic 0.
        assert warmup_turn_score(
            "Просто текст без вызова", "ref", "data_analyst", _action_extra(), TOOLS
        ) == 0.0


class TestWarmupTurnScoreReasoning:
    def test_full_sgr_markers(self):
        text = (
            "Reasoning steps:\n- a\nCurrent situation: x\n"
            "Plan status: y\nRemaining steps:\n- b"
        )
        s = warmup_turn_score(text, text, "math_reasoner", _reasoning_extra(), TOOLS)
        assert s == pytest.approx(1.0)

    def test_unstructured_reasoning_is_low(self):
        text = "Просто рассуждение без всякой структуры SGR вообще."
        s = warmup_turn_score(text, text, "math_reasoner", _reasoning_extra(), TOOLS)
        # reasoning_check=0 → harmonic 0 even with a perfect chrF++.
        assert s == 0.0


class TestAggregate:
    def test_mean(self):
        assert aggregate_warmup_scores([1.0, 0.0, 0.5]) == pytest.approx(0.5)

    def test_empty_is_zero(self):
        assert aggregate_warmup_scores([]) == 0.0
