'Tests for reward/rerank_em.py — the Meno-stand rerank verdict grading.\n\n\nPure offline logic: the tolerant JSON parse with BOTH label spellings\nnormalized to integers, the graded 0.3 + 0.7 ordinal scoring and the\napplicability rules.'

from __future__ import annotations

import pytest

from llm_training.reward.rerank_em import (
    RERANK_EM_CLASS,
    parse_label,
    rerank_em,
    rerank_em_applies,
)


class TestParseLabel:
    @pytest.mark.parametrize(
        "response,expected",
        [
            # The distilled references' string habit.
            ('{"label":"0"}', 0),
            ('{"label": "2"}', 2),  # the space-after-colon reference quirk
            ('  {"label":"1"}  ', 1),
            ('```json\n{"label": "1"}\n```', 1),  # markdown fences
            # The system prompt's integer instruction (the 2026-09-24
            # mismatch: the prompt asks for a digit — now accepted).
            ('{"label": 2}', 2),
            ('{"label":0}', 0),
            # An integral JSON number and a padded string normalize too.
            ('{"label": 1.0}', 1),
            ('{"label": " 2 "}', 2),
        ],
    )
    def test_tolerant_parse(self, response, expected):
        assert parse_label(response) == expected

    @pytest.mark.parametrize(
        "response",
        [
            "",
            "не json",
            '{"label": "3"}',  # out of range
            '{"label": 3}',  # out of range, integer form
            '{"label": 1.5}',  # a non-integral number
            '{"label": true}',  # bool is an int subclass — still junk
            '{"label": null}',
            '{"label": "да"}',
            '"label"',  # valid JSON, not an object
            "[]",
            "{}",  # no label key at all
        ],
    )
    def test_malformed(self, response):
        assert parse_label(response) is None


class TestRerankEm:
    def test_applies_only_to_rerank(self):
        assert rerank_em_applies("meno_rerank")
        assert not rerank_em_applies("menon_qa")
        assert not rerank_em_applies("rummlu_pro")
        assert RERANK_EM_CLASS == "meno_rerank"

    def test_exact_match_full_credit(self):
        # Both spellings reach the full 1.0 against an int gold (the
        # loader stores gold_label as int).
        
        assert rerank_em('{"label":"2"}', "meno_rerank", {"gold_label": 2}) == 1.0
        assert rerank_em('{"label": "0"}', "meno_rerank", {"gold_label": 0}) == 1.0
        assert rerank_em('{"label": 2}', "meno_rerank", {"gold_label": 2}) == 1.0
        assert rerank_em('{"label": 2.0}', "meno_rerank", {"gold_label": 2}) == 1.0

    @pytest.mark.parametrize(
        "response,gold",
        [
            ('{"label":"1"}', 0),
            ('{"label":"1"}', 2),
            ('{"label": 0}', 1),
            ('{"label": 2}', 1),
        ],
    )
    def test_adjacent_grade_partial_credit(self, response, gold):
        # The label is ORDINAL: mixing up ADJACENT grades (|Δ| = 1) keeps
        # half of the graded component — 0.3 + 0.7·0.5 = 0.65.
        
        # 0.3 + 0.7·0.5 = 0.65.
        assert rerank_em(response, "meno_rerank", {"gold_label": gold}) == pytest.approx(0.65)

    @pytest.mark.parametrize(
        "response,gold",
        [
            ('{"label":"0"}', 2),
            ('{"label": 2}', 0),
            ('{"label":"2"}', 0),
        ],
    )
    def test_extreme_grades_only_format_credit(self, response, gold):
        # Mixing up the EXTREME grades 0 ↔ 2 inverts the verdict — only the
        # envelope's 0.3 survives.
        
        
        assert rerank_em(response, "meno_rerank", {"gold_label": gold}) == pytest.approx(0.3)

    def test_unparseable_scores_zero(self):
        assert rerank_em("документ не подходит", "meno_rerank", {"gold_label": 0}) == 0.0
        assert rerank_em('{"label": 3}', "meno_rerank", {"gold_label": 2}) == 0.0

    def test_string_gold_normalized(self):
        # Robustness: a string gold (e.g. a hand-built extra) normalizes to
        # the integer comparison too.
        
        assert rerank_em('{"label": 2}', "meno_rerank", {"gold_label": "2"}) == 1.0
        assert rerank_em('{"label":"1"}', "meno_rerank", {"gold_label": "2"}) == pytest.approx(0.65)

    def test_other_class_none(self):
        assert rerank_em('{"label":"2"}', "menon_qa", {"gold_label": 2}) is None

    def test_missing_or_bad_gold_none(self):
        # No normalizable gold verdict → undefined → N/A (the loader
        # guarantees an int verdict for every rerank row; reaching here is a
        # data bug elsewhere).
        
        assert rerank_em('{"label":"2"}', "meno_rerank", None) is None
        assert rerank_em('{"label":"2"}', "meno_rerank", {}) is None
        assert rerank_em('{"label":"2"}', "meno_rerank", {"gold_label": 3}) is None
        assert rerank_em('{"label":"2"}', "meno_rerank", {"gold_label": "да"}) is None
