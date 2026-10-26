'Tests for reward/rummlu_em.py — the deterministic ruMMLU-pro letter EM.\n\n\nPure offline logic: extraction robustness, grading, applicability.'

from __future__ import annotations

import pytest

from llm_training.reward.rummlu_em import (
    RUMLU_EM_CLASS,
    extract_answer_letter,
    rummlu_em,
    rummlu_em_applies,
)


class TestExtractAnswerLetter:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("з", "з"),
            ("З", "з"),  # case-insensitive
            ("  з  ", "з"),  # whitespace
            ("з.", "з"),  # trailing punctuation
            ("з)", "з"),
            ("**з**", "з"),  # markdown emphasis
            ("Ответ: з", "з"),  # RU label
            ("ответ — з", "з"),
            ("Answer: з", "з"),  # EN label
            ("answer: б.", "б"),
            ("ё", "ё"),
        ],
    )
    def test_extraction(self, text, expected):
        assert extract_answer_letter(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "Безопасные практики, Дистресс",  # the option TEXT, not a letter
            "Правильный ответ — вариант з",  # prose
            "з или ж",  # two letters
            "***",
        ],
    )
    def test_extraction_none(self, text):
        assert extract_answer_letter(text) is None


class TestRummluEm:
    def test_applies_only_to_rummlu(self):
        assert rummlu_em_applies("rummlu_pro")
        assert not rummlu_em_applies("menon_qa")
        assert not rummlu_em_applies("")

    def test_exact_match(self):
        assert rummlu_em("з", "з", "rummlu_pro") == 1.0
        assert rummlu_em(" Ответ: ж.", "ж", "rummlu_pro") == 1.0

    def test_wrong_letter(self):
        assert rummlu_em("ж", "з", "rummlu_pro") == 0.0

    def test_prose_response_scores_zero(self):
        # A response long enough that no single letter extracts is a FORMAT
        # violation of the single-letter contract — a strong zero gradient.
        
        
        assert rummlu_em("Вариант з кажется верным", "з", "rummlu_pro") == 0.0

    def test_latin_lookalike_not_folded(self):
        
        # would blur the format discipline.
        
        
        assert rummlu_em("z", "з", "rummlu_pro") == 0.0

    def test_other_class_none(self):
        assert rummlu_em("з", "з", "menon_qa") is None

    def test_malformed_reference_none(self):
        # A reference that is not a single letter is a data anomaly → N/A
        # (the class falls back to SemSim — the nerel_f1 pattern).
        
        assert rummlu_em("з", "вариант з", "rummlu_pro") is None
        assert rummlu_em("з", "", "rummlu_pro") is None

    def test_extra_ignored(self):
        assert rummlu_em("з", "з", "rummlu_pro", extra={"config": "x"}) == 1.0

    def test_ruml_class_constant(self):
        assert RUMLU_EM_CLASS == "rummlu_pro"
