"Warm-up validation metric: harmonic(format_check-side, chrF++).\n\n\nPer generated turn the score is the **harmonic mean** of two halves — the same\nstrict-combination convention the reward layer uses:\n\n* the **format side** depends on the turn kind:\n  - ``final`` / ``action`` turns → :func:`llm_training.reward.scoring.format_check`\n    on the RAW generation (validates the ``<tool_call>`` envelope: JSON parses,\n    tool name in the registry, ``name``/``arguments`` fields);\n  - ``reasoning`` turns → :func:`llm_training.reward.scoring.reasoning_check`\n    (the SGR schema markers ARE the format of a reasoning turn;\n    ``format_check`` is N/A there by design);\n* the **content side** is ``chrF++`` (``sacrebleu`` ``CHRF(word_order=2)``,\n  normalised to [0, 1]) between the generation's comparable content and the\n  reference:\n  - ``final`` → the extracted ``finalanswertool`` ``answer`` field vs the\n    reference (a failed extraction → chrF++ of an empty string — and the\n    harmonic mean punishes it further);\n  - ``action`` / ``reasoning`` → the generation text vs the reference text.\n\nA per-example zero on either half zeroes the whole score (strict harmonic\nmean), the aggregate is the mean over examples, and the best checkpoint is the\none maximising the aggregate. chrF++ is character-level, so it is robust to\nRussian inflection without any stemmer."

from __future__ import annotations

from typing import Any

from llm_training.reward.scoring import (
    extract_final_answer,
    format_check,
    harmonic_mean,
    reasoning_check,
    safe_score,
)

#: Default chrF++ word order (chrF++ = chrF with word n-grams up to this order).

CHRF_WORD_ORDER = 2

_chrf: Any = None


def _chrf_scorer() -> Any:
    'Lazily build the sacrebleu CHRF (chrF++) scorer.'
    global _chrf
    if _chrf is None:
        from sacrebleu.metrics import CHRF  # lazy: a train-extra dep.

        _chrf = CHRF(word_order=CHRF_WORD_ORDER)
    return _chrf


def chrf_plus_plus(hypothesis: str, reference: str) -> float:
    'chrF++ ∈ [0, 1] between the hypothesis and the reference.\n\n\n    ``sacrebleu`` returns a 0–100 score; normalise to [0, 1] so it combines\n    with the format side on one scale. An empty hypothesis scores 0 (sacrebleu\n    returns 0 for empty input; ``safe_score`` clamps defensively).'
    if not hypothesis or not reference:
        return 0.0
    score = _chrf_scorer().sentence_score(hypothesis, [reference]).score
    return safe_score(score / 100.0)


def warmup_turn_score(
    generation: str,
    reference: str,
    task_class: str,
    extra: dict[str, Any] | None,
    known_tool_names: frozenset[str] = frozenset(),
) -> float:
    "Per-turn harmonic(format-side, chrF++) ∈ [0, 1].\n\n\n    The format side is chosen by the turn kind carried in ``extra`` (the same\n    loader metadata the reward dispatcher uses): ``is_reasoning_turn`` →\n    ``reasoning_check``; otherwise (final / action turns) → ``format_check`` on\n    the raw generation. The content side compares the generation's comparable\n    content with ``reference`` — for ``is_final_answer`` turns both sides are\n    the extracted / reference ``answer`` field; for other turns the texts.\n    A zero on either half zeroes the turn (strict harmonic mean)."
    _extra = extra or {}
    if _extra.get("is_reasoning_turn"):
        fmt = reasoning_check(generation)
        content_gen = generation
    else:
        fmt = format_check(generation, task_class, known_tool_names)
        if _extra.get("is_final_answer"):
            extracted = extract_final_answer(generation)
            content_gen = extracted if extracted is not None else ""
        else:
            content_gen = generation
    fmt_val = fmt if fmt is not None else 0.0  # N/A format → treat as 0 (unreachable for SGR kinds)
    chrf_val = chrf_plus_plus(content_gen, reference)
    return harmonic_mean([fmt_val, chrf_val])


def aggregate_warmup_scores(per_turn: list[float]) -> float:
    'Mean of per-turn scores (the checkpoint-selection aggregate).'
    if not per_turn:
        return 0.0
    return sum(per_turn) / len(per_turn)
