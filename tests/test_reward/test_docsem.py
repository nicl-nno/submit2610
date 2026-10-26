'Tests for the deterministic DocSem check (executable PoT scoring).\n\n\nPure functions, no mocks: the JSON parsing (fence tolerance), the\nanswer canonicalization (portal rules), the evidence set-F1, the\nformat/exec/answer/evidence dimensions and the strict harmonic combine.'

from __future__ import annotations

import json

import pytest

from llm_training.reward.docsem import (
    canonical_answer,
    clean_stated_answer,
    docsem_answer_text,
    docsem_check,
    docsem_check_applies,
    docsem_check_detailed,
    evidence_f1,
    parse_docsem_response,
)

GOLD = ("58", ["b07"])


def _resp(**over) -> str:
    base = {
        "reasoning": "Block b07: 98 total minus 40 donated.",
        "program": "total = 98\ndonated = 40\nanswer = total - donated",
        "answer": "58",
        "evidence": ["b07"],
    }
    base.update(over)
    return json.dumps(base, ensure_ascii=False)


# ----------------------------------------------------------------- applies
def test_applies_only_to_docsem_quant():
    assert docsem_check_applies("docsem_quant")
    assert not docsem_check_applies("menon_qa")
    assert not docsem_check_applies("coder")


# ----------------------------------------------------------------- parsing
def test_parse_plain_and_fenced_json():
    assert parse_docsem_response(_resp()) is not None
    fenced = "```json\n" + _resp() + "\n```"
    assert parse_docsem_response(fenced) is not None
    assert parse_docsem_response("```\n" + _resp() + "\n```") is not None


def test_parse_rejects_garbage():
    assert parse_docsem_response("The answer is 58.") is None
    assert parse_docsem_response("") is None
    assert parse_docsem_response('["not", "an", "object"]') is None


# --------------------------------------------------------- canonicalization
def test_canonical_answer_portal_rules():
    assert canonical_answer("58") == "58"
    assert canonical_answer("58.0") == "58"
    assert canonical_answer("1,067") == "1067"
    assert canonical_answer("10,5") == "10.5"
    assert canonical_answer(" 58 ") == "58"
    assert canonical_answer("answer: 58") == "58"
    assert canonical_answer("Final Answer: -3.50") == "-3.5"
    assert canonical_answer("58 books") is None


def test_clean_stated_answer():
    assert clean_stated_answer("`48`") == "48"
    assert clean_stated_answer("48 books") == "48"
    assert clean_stated_answer("none") == "none"


def test_evidence_f1():
    assert evidence_f1(["b07"], ["b07"]) == 1.0
    assert evidence_f1(["b12"], ["b07"]) == 0.0
    assert evidence_f1(["b07", "b12"], ["b07"]) == pytest.approx(2 / 3)
    assert evidence_f1([], []) == 1.0


# ------------------------------------------------------------ check itself
def test_perfect_response_scores_one():
    assert docsem_check(_resp(), *GOLD) == 1.0


def test_fenced_response_scores_one():
    fenced = "```json\n" + _resp() + "\n```"
    assert docsem_check(fenced, *GOLD) == 1.0


def test_unparseable_response_zeroes():
    d = docsem_check_detailed("I think it is 58.", *GOLD)
    assert d.format_score == 0.0
    assert d.score == 0.0


def test_wrong_answer_zeroes():
    # BOTH the program and the stated field carry the wrong value — the
    # effective answer (executed) is wrong → 0.
    
    d = docsem_check_detailed(_resp(answer="40", program="answer = 40"), *GOLD)
    assert d.answer_score == 0.0
    assert d.score == 0.0


def test_wrong_evidence_zeroes():
    d = docsem_check_detailed(_resp(evidence=["b12"]), *GOLD)
    assert d.evidence_f1 == 0.0
    assert d.score == 0.0


def test_non_executing_program_zeroes_despite_stated_answer():
    # The champion fallback lets the STATED answer carry the answer dimension,
    # but the strict harmonic still zeroes the whole check while the program is
    # broken — the strong PoT gradient (write WORKING programs).
    
    
    d = docsem_check_detailed(_resp(program="answer = 58 /"), *GOLD)
    assert d.exec_score == 0.0
    assert d.answer_score == 1.0
    assert d.score == 0.0


def test_list_final_program_falls_back_to_stated_answer():
    # Regression for the training-time warning ``Scorer failed for
    # task_class=docsem_quant: type list doesn't define __round__ method``:
    # a program whose final assignment is a list is a FAILED execution
    # (ProgramError → stated-answer fallback), never a dispatcher crash.
    
    
    program = "values = [98, 40]\nanswer = values"
    d = docsem_check_detailed(_resp(program=program), *GOLD)
    assert d.exec_score == 0.0
    assert d.answer_score == 1.0
    assert d.format_score == pytest.approx(1.0)
    assert d.evidence_f1 == 1.0
    assert d.score == 0.0


def test_empty_program_format_penalty():
    d = docsem_check_detailed(_resp(program=""), *GOLD)
    assert d.exec_score == 0.0
    assert d.format_score == pytest.approx(0.4 + 0.3 + 0.2)
    assert d.score == 0.0


def test_missing_keys_format_penalty():
    d = docsem_check_detailed(
        json.dumps({"answer": "58", "evidence": ["b07"]}), *GOLD
    )
    assert d.format_score == pytest.approx(0.4 + 0.2)
    assert d.score == 0.0


def test_executed_value_outranks_stated():
    # The program computes 58 while the stated field says 40: the effective
    # answer is the EXECUTED value (champion semantics).
    
    assert docsem_check(_resp(answer="40"), *GOLD) == 1.0


def test_answer_text_extraction():
    assert docsem_answer_text(_resp()) == "58"
    assert docsem_answer_text(_resp(program="answer = 58 /")) == "58"
    assert docsem_answer_text("garbage") is None


def test_answer_text_list_final_program():
    # The list-final program raises ProgramError inside the interpreter, so
    # the effective answer falls back to the stated field (the AC path at
    # the anchors uses this extraction).
    
    
    program = "values = [98, 40]\nanswer = values"
    assert docsem_answer_text(_resp(program=program)) == "58"


def test_missing_gold_raises():
    with pytest.raises(ValueError):
        docsem_check(_resp(), "", ["b07"])
    with pytest.raises(ValueError):
        docsem_check(_resp(), "58", None)
    with pytest.raises(ValueError):
        docsem_check(_resp(), None, ["b07"])
