'Internal implementation detail.'
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from distill_loop_corpus import (  # noqa: E402
    QHASH,
    QUESTIONS_JSON,
    canonicalize_messages,
    dedup_report,
    has_degenerate_final,
    load_questions,
    parse_judge_verdict,
    run_summary,
    shape_keep,
    spotcheck_selected,
)


def test_approved_questions_file():
    'Internal implementation detail.'
    recs = load_questions(QUESTIONS_JSON)
    assert len(recs) == 50
    assert sum(1 for r in recs if r["lang"] == "ru") == 25
    assert sum(1 for r in recs if r["lang"] == "en") == 25
    assert sum(1 for r in recs if r["trap"]) == 6
    assert len({r["qid"] for r in recs}) == 50
    assert all(r["question"].strip() for r in recs)
    
    questions = {QHASH(r["question"]) for r in recs}
    assert QHASH("Расскажи об Anonymous Author из АУ.") in questions
    assert QHASH("Расскажи о статье про RAGU на конференции EMNLP-2026.") in questions


def test_dedup_report_drops_simpleqa_collisions():
    recs = [{"qid": "x", "question": "Same question?", "lang": "en", "trap": False},
            {"qid": "y", "question": "Unique?", "lang": "en", "trap": False}]
    kept, report = dedup_report(recs, ["  Same QUESTION? "])
    assert report == {"total": 2, "kept": 1, "dropped_simpleqa": 1}
    assert [r["qid"] for r in kept] == ["y"]
    
    assert QHASH("Same question?") == QHASH("  same question? ")


def test_spotcheck_deterministic_quarter():
    picked = [f"q{i:02d}" for i in range(1, 51) if spotcheck_selected(f"q{i:02d}")]
    
    assert 6 <= len(picked) <= 20
    assert picked == [f"q{i:02d}" for i in range(1, 51)
                      if spotcheck_selected(f"q{i:02d}")]
    assert not spotcheck_selected("q01", every=0)


def _row(**kw):
    base = {"status": "ok", "predicted_answer": "ответ",
            "tools": "reasoningtool,websearchtool,extractpagecontenttool,"
                     "reasoningtool,finalanswertool",
            "degenerates": 0, "degenerate_final": False}
    base.update(kw)
    return base


def test_shape_keep_positive():
    assert shape_keep(_row())
    
    assert shape_keep(_row(tools="reasoningtool,websearchtool,finalanswertool"))


@pytest.mark.parametrize("kw", [
    {"status": "error"},
    {"status": "no_final"},
    {"predicted_answer": None},
    {"predicted_answer": ""},
    {"tools": "reasoningtool,websearchtool"},                   
    {"tools": "websearchtool,extractpagecontenttool,finalanswertool"},  
    {"tools": "reasoningtool,finalanswertool"},                 
    {"degenerate_final": True},                                 # fallback
])
def test_shape_keep_rejections(kw):
    assert not shape_keep(_row(**kw))


def test_shape_keep_recovered_degenerate_attempt_kept():
    'Internal implementation detail.'
    assert shape_keep(_row(degenerates=2))
    assert not shape_keep(_row(degenerates=2, degenerate_final=True))


def test_canonicalize_messages_relabels_nonleading_system():
    msgs = [
        {"role": "system", "content": "Настоящий системный промпт"},
        {"role": "user", "content": "вопрос"},
        {"role": "system", "content": "Agent abc started\n"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t"}]},
        {"role": "tool", "content": "результат", "tool_call_id": "t"},
    ]
    out = canonicalize_messages(msgs)
    assert out[0]["role"] == "system"                       
    assert out[2]["role"] == "user" and out[2]["content"] == "Agent abc started\n"
    assert out[4] == msgs[4]                                
    
    plain = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert canonicalize_messages(plain) is plain
    assert canonicalize_messages([]) == []


def test_has_degenerate_final_marker():
    attempts = [{"error": "degenerate fallback to FinalAnswerTool"},
                {"error": "ValidationError: ..."}]
    assert not has_degenerate_final(attempts)
    assert has_degenerate_final(
        attempts + [{"error": "degenerate fallback FINAL (retries exhausted)"}])
    assert not has_degenerate_final([])


def test_parse_judge_verdict_tolerates_fences_and_think():
    good = '{"verdict": "grounded", "comment": "все факты из наблюдений"}'
    assert parse_judge_verdict(good) == {
        "verdict": "grounded", "comment": "все факты из наблюдений"}
    fenced = f"Вот оценка:\n```json\n{good}\n```\n"
    assert parse_judge_verdict(fenced)["verdict"] == "grounded"
    thinking = f"<think>размышляю…</think>\n{good}"
    assert parse_judge_verdict(thinking)["verdict"] == "grounded"
    assert parse_judge_verdict('{"verdict": "wrong-value"}') is None
    assert parse_judge_verdict("никакого json здесь нет") is None
    assert parse_judge_verdict("") is None


def test_run_summary_math():
    rows = [_row(keep=True) for _ in range(6)] + [
        _row(status="error", keep=False), _row(degenerates=1, keep=False)]
    s = run_summary(rows, credits=12.0)
    assert "episodes=8" in s and "keep=6 (75%)" in s
    assert "errors=1" in s and "degenerates=1" in s
    assert "per_episode=1.50" in s
    empty = run_summary([], credits=0.0)
    assert empty == "нет завершённых эпизодов"


def test_prepare_command_end_to_end(tmp_path):
    import distill_loop_corpus as d
    simpleqa = tmp_path / "results.jsonl"
    simpleqa.write_text(
        json.dumps({"question": "Полностью посторонний вопрос SimpleQA?"}) + "\n",
        encoding="utf-8")
    out = tmp_path / "pools"
    rc = d.main(["prepare", "--out", str(out),
                 "--simpleqa-results", str(simpleqa)])
    assert rc == 0
    pool = [json.loads(l) for l in
            (out / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(pool) == 50
    assert {r["qid"] for r in pool} == {f"q{i:02d}" for i in range(1, 51)}
