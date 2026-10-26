"Offline tests for scripts/rerank_block_eval.py (no network, no GPU).\n\n\nThe script lives outside the package — loaded from its path via ``importlib``\n(the test_scripts convention). The endpoint path is exercised against a FAKE\nOpenAI-style client; the metric itself is the repo's ``rerank_em``.\ntest_scripts"

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "rerank_block_eval.py"

RERANK_SYSTEM = (
    "Вы — модуль оценки пригодности документа. Оцените, насколько документ "
    "пригоден как фактическая основа для ответа. Ответьте JSON {\"label\": N}."
)
REWRITE_SYSTEM = "Вы — Query Rewriter в RAG-пайплайне."


@pytest.fixture(scope="module")
def rbe():
    spec = importlib.util.spec_from_file_location("_rerank_block_eval", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rerank_block_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_pool(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "test.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


class TestLoadRerankRows:
    def test_filters_and_gold_parsing(self, rbe, tmp_path):
        rows = [
            # Rerank, string-spelling gold (the distilled references' habit):
            
            {"system": RERANK_SYSTEM, "query": "Q1", "response": '{"label": "2"}',
             "history": []},
            # Rerank, integer gold (the system prompt's instruction):
            
            {"system": RERANK_SYSTEM, "query": "Q2", "response": '{"label":0}',
             "history": []},
            # Rerank with an unparseable reference → skipped, counted:
            
            {"system": RERANK_SYSTEM, "query": "Q3", "response": "не JSON",
             "history": []},
            # Foreign classes → skipped, counted:
            
            {"system": REWRITE_SYSTEM, "query": "Q4", "response": '{"label": "1"}',
             "history": []},
        ]
        path = _write_pool(tmp_path, rows)
        got, n_foreign, n_gold = rbe.load_rerank_rows(path)
        assert n_foreign == 1
        assert n_gold == 1
        assert [r["gold"] for r in got] == [2, 0]
        assert [r["id"] for r in got] == [0, 1]
        assert got[0]["query"] == "Q1"

    def test_empty_pool_fails_later(self, rbe, tmp_path):
        path = _write_pool(tmp_path, [
            {"system": REWRITE_SYSTEM, "query": "Q", "response": "{}", "history": []},
        ])
        got, n_foreign, n_gold = rbe.load_rerank_rows(path)
        assert got == [] and n_foreign == 1 and n_gold == 0


class TestExtractLabelLogprobs:
    def test_maps_all_three_labels(self, rbe):
        payload = [
            {"token": "1", "logprob": -0.7},
            {"token": " 2", "logprob": -1.1},   # whitespace stripped
            {"token": "0", "logprob": -3.2},
            {"token": "hello", "logprob": -0.1},  # junk ignored
        ]
        got = rbe.extract_label_logprobs(payload)
        assert got == {1: -0.7, 2: -1.1, 0: -3.2}

    def test_first_occurrence_wins(self, rbe):
        payload = [{"token": "2", "logprob": -0.5}, {"token": "2", "logprob": -9.0}]
        assert rbe.extract_label_logprobs(payload) == {2: -0.5}

    def test_missing_and_none(self, rbe):
        assert rbe.extract_label_logprobs([]) == {}
        assert rbe.extract_label_logprobs(None) == {}
        assert rbe.extract_label_logprobs([{"token": "x", "logprob": -0.1}]) == {}


class TestSoftScores:
    def test_uniform(self, rbe):
        lp = {0: -1.0, 1: -1.0, 2: -1.0}
        soft = rbe.soft_scores(lp)
        assert soft is not None
        assert all(abs(v - 1 / 3) < 1e-9 for v in soft.values())

    def test_missing_label_gets_zero(self, rbe):
        # Only labels 1 and 2 in the top-k → label 0 gets probability 0.
        
        lp = {1: -0.7, 2: -1.1}
        soft = rbe.soft_scores(lp)
        assert soft is not None
        assert soft[0] == 0.0
        assert soft[1] > soft[2] > 0.0
        assert abs(soft[1] + soft[2] - 1.0) < 1e-9

    def test_no_labels_undefined(self, rbe):
        assert rbe.soft_scores({}) is None
        # Keys outside the valid labels never produce a mass:
        
        assert rbe.soft_scores({5: -0.1}) is None


class _FakeChoice:
    'Mimics the OpenAI choice surface used by the script.'

    def __init__(self, content: str, top_logprobs: list[dict] | None):
        self.message = SimpleNamespace(content=content)
        self.logprobs = SimpleNamespace(
            content=[SimpleNamespace(top_logprobs=top_logprobs)]
            if top_logprobs is not None else []
        )


class _FakeClient:
    def __init__(self, choice: _FakeChoice):
        self._choice = choice
        self.calls = 0

        class _Completions:
            def create(inner_self, **kwargs):  # noqa: N805 — SimpleNamespace pattern
                self.calls += 1
                return SimpleNamespace(choices=[self._choice])

        self.chat = SimpleNamespace(completions=_Completions())


class TestScoreOne:
    def test_exact_verdict_and_soft_score(self, rbe):
        choice = _FakeChoice(
            content='{"label": "2"}',
            top_logprobs=[{"token": "1", "logprob": -1.0},
                          {"token": "2", "logprob": -1.0},
                          {"token": "0", "logprob": -1.0}],
        )
        row = {"id": 7, "system": RERANK_SYSTEM, "query": "Q", "gold": 2}
        result = rbe._score_one(_FakeClient(choice), "m", row, max_tokens=64, retries=0)
        assert result["parsed_label"] == 2
        assert result["rerank_em"] == pytest.approx(1.0)
        assert result["soft_gold"] == pytest.approx(1 / 3)
        assert result["top1_correct"] is True

    def test_adjacent_verdict_scores_065(self, rbe):
        choice = _FakeChoice(content='{"label": 0}', top_logprobs=None)
        row = {"id": 1, "system": RERANK_SYSTEM, "query": "Q", "gold": 1}
        result = rbe._score_one(_FakeClient(choice), "m", row, max_tokens=64, retries=0)
        assert result["rerank_em"] == pytest.approx(0.65)
        assert result["soft_gold"] is None
        assert result["top1_correct"] is False

    def test_unparseable_response_scores_zero(self, rbe):
        choice = _FakeChoice(content="просто текст", top_logprobs=None)
        row = {"id": 2, "system": RERANK_SYSTEM, "query": "Q", "gold": 1}
        result = rbe._score_one(_FakeClient(choice), "m", row, max_tokens=64, retries=0)
        assert result["rerank_em"] == 0.0
        assert result["parsed_label"] is None
