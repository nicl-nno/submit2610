'Tests for reward/language.py — target-language detection + fluency guard.\n\n\nOffline: the real lingua detector is never imported — every test injects a fake\n:class:`LanguageDetector` (the DI seam), mirroring the fluency-scorer test'

from __future__ import annotations

import pytest

from llm_training.reward.language import (
    LANG_EN,
    LANG_RU,
    detect_target_language,
    language_match_factor,
)


class _FakeDetector:
    'Returns RU for Cyrillic-bearing text, EN otherwise. Records calls.'

    def __init__(self, ru_conf: float = 0.99, en_conf: float = 0.99) -> None:
        self.ru_conf = ru_conf
        self.en_conf = en_conf
        self.calls: list[str] = []

    def confidence_values(self, text: str) -> list[tuple[str, float]]:
        self.calls.append(text)
        if any("а" <= c.lower() <= "я" for c in text):
            return [("ru", self.ru_conf), ("en", 1.0 - self.ru_conf)]
        return [("en", self.en_conf), ("ru", 1.0 - self.en_conf)]


class _FixedDetector:
    'Returns a caller-supplied confidence list (no language logic).'

    def __init__(self, values: list[tuple[str, float]]) -> None:
        self._values = values

    def confidence_values(self, text: str) -> list[tuple[str, float]]:
        return list(self._values)


def _prompt(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# detect_target_language
# ---------------------------------------------------------------------------
class TestDetectTargetLanguage:
    def test_ru_question_detected(self):
        det = _FakeDetector()
        prompt = _prompt("Ты помощник.", "Объясни устройство атома.")
        assert detect_target_language(prompt, "menon_qa", {}, detector=det) == LANG_RU

    def test_en_question_detected(self):
        det = _FakeDetector()
        prompt = _prompt("You are a helpful assistant.", "Explain the atom.")
        assert detect_target_language(prompt, "multihop_rag", {}, detector=det) == LANG_EN

    def test_override_ru_in_english_instruction(self):
        # An English instruction that explicitly asks for Russian wins.
        
        det = _FakeDetector()
        prompt = _prompt("You are a helpful assistant.", "Explain the atom. Answer in Russian.")
        assert detect_target_language(prompt, "menon_qa", {}, detector=det) == LANG_RU

    def test_override_en_in_russian_instruction(self):
        
        
        det = _FakeDetector()
        prompt = _prompt("Ты помощник.", "Объясни устройство атома. Отвечай на английском.")
        assert detect_target_language(prompt, "menon_qa", {}, detector=det) == LANG_EN

    def test_negated_override_inverts(self):
        # "not in Russian" forces English, not Russian.
        
        det = _FakeDetector()
        prompt = _prompt("You are an assistant.", "Do not answer in Russian.")
        assert detect_target_language(prompt, "menon_qa", {}, detector=det) == LANG_EN

    def test_short_question_falls_back_to_system(self):
        # A terse user message ("Hi") is too short to detect; the system
        # message supplies the language.
        
        det = _FakeDetector()
        prompt = _prompt("Ты полезный помощник, отвечай подробно.", "Hi")
        assert detect_target_language(prompt, "menon_qa", {}, detector=det) == LANG_RU

    def test_inconclusive_returns_none(self):
        # Low-confidence detector → no target language (guard becomes no-op).
        
        det = _FixedDetector([("ru", 0.4), ("en", 0.4)])
        prompt = _prompt("Ты помощник.", "Объясни устройство атома.")
        assert detect_target_language(prompt, "menon_qa", {}, detector=det) is None

    def test_empty_prompt_returns_none(self):
        det = _FakeDetector()
        assert detect_target_language([], "menon_qa", {}, detector=det) is None


# ---------------------------------------------------------------------------
# language_match_factor
# ---------------------------------------------------------------------------
class TestLanguageMatchFactor:
    def test_unknown_target_is_noop(self):
        assert language_match_factor("Какой-то ответ.", None, detector=_FakeDetector()) == 1.0
        assert language_match_factor("Какой-то ответ.", "zz", detector=_FakeDetector()) == 1.0

    def test_short_text_is_noop(self):
        # "Hi" is below MIN_LANG_DETECT_WORDS → skip (no false positive).
        
        assert language_match_factor("Hi", LANG_EN, detector=_FakeDetector()) == 1.0

    def test_correct_language_passes_through(self):
        det = _FakeDetector()
        assert language_match_factor("Мама мыла раму.", LANG_RU, detector=det) == pytest.approx(0.99)

    def test_wrong_language_zeroes(self):
        # An English answer against a Russian target → factor ≈ 0.
        
        det = _FakeDetector()
        assert language_match_factor("The quick brown fox.", LANG_RU, detector=det) == pytest.approx(0.01)

    def test_mixed_confidence_is_partial(self):
        det = _FixedDetector([("en", 0.6), ("ru", 0.4)])
        assert language_match_factor("some long enough text", LANG_RU, detector=det) == pytest.approx(0.4)

    def test_detector_failure_is_noop(self):
        class _Boom:
            def confidence_values(self, text: str):
                raise RuntimeError("boom")

        assert language_match_factor("Some long enough text.", LANG_RU, detector=_Boom()) == 1.0
