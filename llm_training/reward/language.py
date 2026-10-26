'Language-aware guard for the fluency signal (reward-hacking defence).\n\n\nThe fluency classifier is a grammar model trained on English C4. It transfers to\nRussian well, but it can still score an English answer slightly higher than an\nequally good Russian one — which opens a reward-hacking window: the policy could\ndrift to answering every bilingual task in English to inflate ``fluency``. This\nmodule closes that window by comparing the language of the generated answer\nagainst the **target language** of the task and down-weighting the fluency score\nwhen they mismatch.\n\n\nTwo responsibilities\n\n1. :func:`detect_target_language` — determine the target language of a task from\n   its original message-list prompt (build time): the extracted user question\n   (retrieved context stripped) + the ``system`` message, with an explicit\noverride scan ("in Russian"\n   or ``None`` (inconclusive → the guard becomes a no-op).\n2. :func:`language_match_factor` — at reward time, compute the target language\'s\n   confidence over the generated prose (code stripped); the fluency score is\n   multiplied by it. ``1.0`` when the answer is confidently in the target\n   language, ``≈0`` when it is confidently in the wrong one (the strict harmonic\n   mean then zeroes the example), a fraction for mixed text.\n\nThe detector is ``lingua-language-detector`` (pure Python), limited to the two\ndataset languages (RU + EN) for speed and footprint. It is imported lazily and\nbuilt once as a module-level singleton; both helpers accept a DI ``detector``\nseam (a :class:`LanguageDetector`-like object) so offline tests inject a fake.'

from __future__ import annotations

import logging
import re
from typing import Any, Protocol, runtime_checkable

from llm_training.reward.inputs import extract_user_question

logger = logging.getLogger(__name__)

# : Internal language codes.
LANG_RU = "ru"
LANG_EN = "en"
_KNOWN_LANGS = (LANG_RU, LANG_EN)

#: Minimum alphabetic-token count below which detection is skipped (avoid
#: false positives on terse / numeric / code-only text). Lingua itself is
#: unreliable under ~3 words.

MIN_LANG_DETECT_WORDS = 3

#: Minimum top confidence to accept a detection (below → ``None``).

_MIN_DETECT_CONFIDENCE = 0.5

_ALPHA_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")

# Explicit language-override phrases (case-insensitive). The override wins over
# plain detection: an English instruction that says "answer in Russian" targets
# Russian, not the instruction language. Negated forms ("not in Russian",

# the positive phrase.



_OVERRIDE_RU_POSITIVE = (
    r"answer\s+in\s+russian",
    r"respond\s+in\s+russian",
    r"reply\s+in\s+russian",
    r"write\s+in\s+russian",
    r"\bin\s+russian\b",
    r"на\s+русском(?:\s+языке)?",
    r"по[- ]русски",
    r"ответь(?:те)?\s+на\s+русском",
    r"отвечай(?:те)?\s+на\s+русском",
    r"пиши(?:те)?\s+на\s+русском",
)
_OVERRIDE_EN_POSITIVE = (
    r"answer\s+in\s+english",
    r"respond\s+in\s+english",
    r"reply\s+in\s+english",
    r"write\s+in\s+english",
    r"\bin\s+english\b",
    r"на\s+английском(?:\s+языке)?",
    r"по[- ]английски",
    r"ответь(?:те)?\s+на\s+английском",
    r"отвечай(?:те)?\s+на\s+английском",
    r"пиши(?:те)?\s+на\s+английском",
)
_OVERRIDE_NEGATION = (
    r"\bnot\s+",
    r"\bdo\s+not\s+",
    r"\bdon'?t\s+",
    r"\bnever\s+",
    r"\bno\s+",
    r"не\s+",
    r"ни\s+в\s+коем\s+случае\s+",
)


def _compile(patterns) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


_OVERRIDE_RU_RE = _compile(_OVERRIDE_RU_POSITIVE)
_OVERRIDE_EN_RE = _compile(_OVERRIDE_EN_POSITIVE)
_OVERRIDE_NEG_RE = _compile(_OVERRIDE_NEGATION)


@runtime_checkable
class LanguageDetector(Protocol):
    'A lingua-like detector: ISO code + confidence per text.'

    def confidence_values(self, text: str) -> list[tuple[str, float]]:
        """Return ``(iso_code, confidence)`` pairs, sorted by confidence desc."""
        ...


class _LinguaDetector:
    'Thin facade over lingua, returning ``(iso_code, confidence)`` pairs.\n\n\n    Built over RU + EN only (the two dataset languages) — two small language\n    models instead of the full ~75-language catalog, so the footprint and the\n    per-call latency stay negligible alongside the grammar classifier.'

    def __init__(self) -> None:
        from lingua import Language, LanguageDetectorBuilder

        self._det = LanguageDetectorBuilder.from_languages(
            Language.RUSSIAN, Language.ENGLISH
        ).build()

    def confidence_values(self, text: str) -> list[tuple[str, float]]:
        values = self._det.compute_language_confidence_values(text)
        return [
            (v.language.iso_code_639_1.name.lower(), float(v.value))
            for v in values
        ]


_detector_singleton: _LinguaDetector | None = None
_detector_failed: bool = False


def _get_detector() -> _LinguaDetector | None:
    'Build (once) and return the shared lingua detector, or ``None`` on failure.'
    global _detector_singleton, _detector_failed
    if _detector_singleton is not None:
        return _detector_singleton
    if _detector_failed:
        return None
    try:
        _detector_singleton = _LinguaDetector()
    except Exception as e:  # noqa: BLE001 - degrade to a no-op guard
        _detector_failed = True
        logger.warning(
            "lingua language detector unavailable; the fluency language guard "
            "is disabled (%s)", e,
        )
        return None
    return _detector_singleton


def _word_count(text: str) -> int:
    'Count alphabetic (Cyrillic/Latin) tokens in ``text``.'
    return len(_ALPHA_TOKEN_RE.findall(text or ""))


def _instruction_text(
    prompt: list[dict[str, Any]] | None,
    task_class: str,
    extra: dict[str, Any] | None,
) -> str:
    'Join the system message + the extracted (context-stripped) user question.'
    system_parts: list[str] = []
    for m in prompt or []:
        content = m.get("content") or ""
        if m.get("role") == "system" and isinstance(content, str) and content.strip():
            system_parts.append(content)
    question = extract_user_question(task_class, prompt or [], extra or {})
    parts = [*system_parts, question] if question.strip() else system_parts
    return "\n".join(parts).strip()


def _find_override(text: str) -> str | None:
    'Return the target language forced by an explicit override phrase, if any.\n\n\n    Positive phrases set the language; a nearby negation ("not in Russian",\n    un-negated RU+EN overrides resolve to ``None`` (fall back to detection).'
    low = text.lower()
    ru = False
    en = False
    for p in _OVERRIDE_RU_RE:
        if p.search(low):
            ru = True
            break
    for p in _OVERRIDE_EN_RE:
        if p.search(low):
            en = True
            break
    if ru or en:
        # Negation: "not in Russian" → force English; «
        for p in _OVERRIDE_NEG_RE:
            if p.search(low):
                return LANG_EN if ru and not en else LANG_RU if en and not ru else None
    if ru and not en:
        return LANG_RU
    if en and not ru:
        return LANG_EN
    return None


def _detect_lang(detector: LanguageDetector, text: str) -> str | None:
    'Return the dominant language of ``text``, or ``None`` when inconclusive.'
    text = (text or "").strip()
    if _word_count(text) < MIN_LANG_DETECT_WORDS:
        return None
    try:
        values = detector.confidence_values(text)
    except Exception as e:  # noqa: BLE001 - degrade to None
        logger.warning("language detection failed: %s", e)
        return None
    if not values:
        return None
    top_iso, top_conf = values[0]
    if top_conf < _MIN_DETECT_CONFIDENCE:
        return None
    return top_iso if top_iso in _KNOWN_LANGS else None


def detect_target_language(
    prompt: list[dict[str, Any]] | None,
    task_class: str,
    extra: dict[str, Any] | None = None,
    *,
    detector: LanguageDetector | None = None,
) -> str | None:
    'Determine the target answer language of a task (``"ru"`` / ``"en"`` / ``None``).\n\n``None``).\n\nOrder of precedence\n\n    1. An explicit language-override phrase in the instruction (system + user\n       question) — e.g. "answer in Russian" inside an English instruction.\n    2. The dominant language of the extracted user question (context stripped);\n       if it is too short / empty, the system message.\n    3. ``None`` when nothing is conclusive — the language guard then becomes a\n       no-op for the example (conservative: never falsely penalise).\n\n    ``detector`` is a DI seam for tests; ``None`` → the shared lingua detector.'
    instruction = _instruction_text(prompt, task_class, extra)
    override = _find_override(instruction)
    if override:
        return override
    det = detector if detector is not None else _get_detector()
    if det is None:
        return None
    question = extract_user_question(task_class, prompt or [], extra or {})
    lang = _detect_lang(det, question)
    if lang is not None:
        return lang
    return _detect_lang(det, instruction)


def language_match_factor(
    text: str,
    target: str | None,
    *,
    detector: LanguageDetector | None = None,
) -> float:
    "Return the multiplier for the fluency score of ``text`` given ``target``.\n\n\n    ``target`` ``None`` (unknown target) or a terse/non-lingual ``text`` → ``1.0``\n    (the guard is a no-op — never falsely penalise). Otherwise the target\n    language's confidence over ``text``: ``≈1`` when the answer is confidently in\n    the target language, ``≈0`` when it is confidently in another language (the\n    strict harmonic mean then zeroes the example), a fraction for mixed text."
    if target not in _KNOWN_LANGS:
        return 1.0
    text = (text or "").strip()
    if _word_count(text) < MIN_LANG_DETECT_WORDS:
        return 1.0
    det = detector if detector is not None else _get_detector()
    if det is None:
        return 1.0
    try:
        values = det.confidence_values(text)
    except Exception as e:  # noqa: BLE001 - degrade to a no-op
        logger.warning("language detection failed: %s", e)
        return 1.0
    conf = dict(values)
    return max(0.0, min(1.0, conf.get(target, 0.0)))
