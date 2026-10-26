'Tests for reward/nerel_f1.py — deterministic NEREL set-F1 (EleutherAI port).\n\n\nExercises the pure normalization + set-F1 logic offline. ``nltk`` is required\n(the ``train`` extra); ``wordpunct_tokenize`` and ``RussianStemmer`` need no\ndata download'

from __future__ import annotations

import pytest

from llm_training.reward.nerel_f1 import (
    NEREL_F1_CONFIGS,
    nerel_f1,
    nerel_f1_applies,
    normalize_answer,
)


# ---------------------------------------------------------------------------
# normalize_answer
# ---------------------------------------------------------------------------
class TestNormalizeAnswer:
    def test_lowercases(self):
        assert normalize_answer("ПАРИЖ") == normalize_answer("париж")

    def test_stemming_collapses_inflection(self):
        # Snowball Russian stems "
        # one-token normalization matches across inflection.
        
        assert normalize_answer("Париж") == normalize_answer("Парижа")

    def test_punctuation_dropped(self):
        assert normalize_answer("Привет, мир!!!") == normalize_answer("привет мир")

    def test_empty_is_empty(self):
        assert normalize_answer("") == ""
        assert normalize_answer("!!!") == ""

    def test_no_stemmer(self):
        # With apply_stemmer=False the raw lowercased alnum tokens are joined.
        
        assert normalize_answer("Париж, Москва", apply_stemmer=False) == "париж москва"


# ---------------------------------------------------------------------------
# nerel_f1_applies
# ---------------------------------------------------------------------------
class TestNerelF1Applies:
    def test_extraction_config_applies(self):
        for cfg in NEREL_F1_CONFIGS:
            assert nerel_f1_applies("nerel_instruct", {"config": cfg}) is True

    def test_definition_config_does_not_apply(self):
        assert nerel_f1_applies(
            "nerel_instruct", {"config": "entity_definition"}
        ) is False

    def test_normalization_config_does_not_apply(self):
        assert nerel_f1_applies(
            "nerel_instruct", {"config": "entity_normalization"}
        ) is False

    def test_non_nerel_class_does_not_apply(self):
        assert nerel_f1_applies("menon_qa", {"config": "entity_recognition"}) is False

    def test_missing_config_does_not_apply(self):
        assert nerel_f1_applies("nerel_instruct", None) is False
        assert nerel_f1_applies("nerel_instruct", {}) is False


# ---------------------------------------------------------------------------
# nerel_f1
# ---------------------------------------------------------------------------
class TestNerelF1:
    def test_non_applicable_returns_none(self):
        assert nerel_f1("a\nb", "a\nb", "menon_qa") is None
        assert nerel_f1(
            "a\nb", "a\nb", "nerel_instruct", {"config": "entity_definition"}
        ) is None

    def test_perfect_match_is_one(self):
        ref = "смерть\nЛетов\nсоветский"
        assert nerel_f1(ref, ref, "nerel_instruct", {"config": "entity_recognition"}) == pytest.approx(1.0)

    def test_partial_match(self):
        # predicted 2 of 3 reference entities → precision 1.0, recall 2/3.
        
        pred = "смерть\nЛетов"
        ref = "смерть\nЛетов\nсоветский"
        val = nerel_f1(pred, ref, "nerel_instruct", {"config": "entity_recognition"})
        assert val == pytest.approx(0.8)

    def test_no_match_is_zero(self):
        pred = "один\nдва"
        ref = "три\nчетыре"
        val = nerel_f1(pred, ref, "nerel_instruct", {"config": "relation_extraction"})
        assert val == 0.0

    def test_empty_prediction_vs_reference_is_zero(self):
        # An empty / garbage prediction against a real reference → 0.0 (strong
        # gradient toward the correct entity set).
        
        val = nerel_f1("", "смерть\nЛетов", "nerel_instruct", {"config": "entity_recognition"})
        assert val == 0.0

    def test_empty_reference_is_none(self):
        # Nothing to match against → N/A (excluded from the harmonic combine).
        
        val = nerel_f1("смерть", "   ", "nerel_instruct", {"config": "entity_recognition"})
        assert val is None

    def test_order_independent(self):
        # set-F1 is insensitive to line order.
        
        a = "смерть\nЛетов\nсоветский"
        b = "советский\nЛетов\nсмерть"
        assert nerel_f1(a, b, "nerel_instruct", {"config": "entity_recognition"}) == pytest.approx(1.0)

    def test_duplicates_collapsed(self):
        # Repeated lines collapse (set semantics); no double credit.
        
        a = "смерть\nсмерть\nЛетов"
        b = "смерть\nЛетов"
        assert nerel_f1(a, b, "nerel_instruct", {"config": "entity_recognition"}) == pytest.approx(1.0)

    def test_inflection_matches(self):
        # Stemming lets "
        
        pred = "Париж"
        ref = "Парижа"
        val = nerel_f1(pred, ref, "nerel_instruct", {"config": "entity_recognition"})
        assert val == pytest.approx(1.0)

    def test_all_three_extraction_configs(self):
        ref = "смерть\nЛетов"
        for cfg in NEREL_F1_CONFIGS:
            assert nerel_f1(ref, ref, "nerel_instruct", {"config": cfg}) == pytest.approx(1.0)
