'Deterministic Meno-stand rerank grading — the JSON label verdict check.\n\n\nThe ``meno_rerank`` class (the GPT-4o distillation through the AU "Menon"\nstand) grades a candidate RAG document as a factual basis for the question on\nthe 0/1/2 scale; the required response is a compact JSON object\n``{"label": N}``. The label is read in BOTH spellings and NORMALIZED TO AN\nINTEGER before the comparison: the integer the system prompt asks for\nreferences actually carry (``{"label": "N"}`` — GPT-4o\'s habit). The\n2026-09-24 anchor attribution: the warmed model dutifully answered with the\nprompt\'s integer form and scored 0.0 under the string-only metric, freezing\n``val_by_subtype[\'meno_rerank\']`` at exactly 0.0 in every anchor of the r6\nrun.\n\nGrading (the operator\'s spec, updated 2026-09-25):\n\n* ``0.3`` — the response parses as JSON (markdown fences and whitespace\n  tolerated) into an object whose label — integer or string — is one of\n  ``0``/``1``/``2`` (an out-of-range value, a non-integral number, a boolean\n  or any other junk fails this half);\n* ``+0.7 × (2 − |gold − got|) / 2`` — the label is an ORDINAL quantity\n  (0 — not suitable, 1 — partially, 2 — suitable), so the graded half is\n  scaled by the ordinal distance: an exact match adds the full 0.7\n  (total 1.0), mixing up ADJACENT grades adds 0.35 (total 0.65), mixing up\n  the extreme grades 0 ↔ 2 (an inverted verdict) adds nothing (total 0.3).\n\nAn unparseable response scores ``0.0`` (a strong gradient toward the JSON\nenvelope). ``None`` (not applicable, excluded from the harmonic combine) for\nother task classes and when ``extra["gold_label"]`` cannot be normalized\n(the loader guarantees an int verdict for every rerank row).'

from __future__ import annotations

import json
from typing import Any

from llm_training.reward.scoring import safe_score

#: The only task class scored by the deterministic rerank grading.

RERANK_EM_CLASS: str = "meno_rerank"

#: The valid verdict labels (as integers — both spellings normalize to them).

_VALID_LABELS: frozenset[int] = frozenset({0, 1, 2})

#: The maximum ordinal distance between two verdicts on the 0/1/2 scale.

_MAX_LABEL_DISTANCE: int = 2

#: The format-half weight of the score (the envelope gradient).

_FORMAT_SCORE: float = 0.3

#: The label-half weight of the score (the ordinal-closeness gradient).

_LABEL_SCORE: float = 0.7


def _strip_json_fences(text: str) -> str:
    'Strip markdown fences around a JSON payload (a common LLM habit).'
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1 :]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    return cleaned.strip()


def _coerce_label(value: Any) -> int | None:
    'Normalize a verdict label to an integer 0/1/2; ``None`` when invalid.\n\n\n    Accepts the integer the system prompt asks for, the string the distilled\n    references carry (outer whitespace tolerated), and an integral JSON\n    number (``2.0``). Rejects booleans (``bool`` is an ``int`` subclass), any\n    non-integral number, and out-of-range values.'
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in _VALID_LABELS else None
    if isinstance(value, float):
        return int(value) if value.is_integer() and int(value) in _VALID_LABELS else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in ("0", "1", "2"):
            return int(stripped)
    return None


def parse_label(response: str) -> int | None:
    'Extract the normalized verdict label; ``None`` when malformed.\n\n\n    Tolerant to markdown fences and whitespace; accepts BOTH the integer\n    form ``{"label": 2}`` (the system prompt\'s instruction) and the string\n    form ``{"label": "2"}`` (the distilled references\' habit), returning the\n    integer 0/1/2 either way. An out-of-range label returns ``None``.'
    try:
        payload = json.loads(_strip_json_fences(response or ""))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return _coerce_label(payload.get("label"))


def rerank_em_applies(task_class: str) -> bool:
    'True iff this example should be scored by the deterministic grading.'
    return task_class == RERANK_EM_CLASS


def rerank_em(
    response: str,
    task_class: str,
    extra: dict[str, Any] | None = None,
) -> float | None:
    'Graded ordinal rerank verdict score; ``None`` when N/A.\n\n\n    Returns ``None`` (not applicable — excluded from the harmonic combine,\n    the class falls back to Semantic Similarity) when ``task_class`` is not\n    ``meno_rerank`` or ``extra["gold_label"]`` cannot be normalized to a\n    valid verdict (the loader guarantees it for every rerank row; reaching\n    here is a data bug elsewhere). Otherwise: ``1.0`` — the normalized label\n    equals the gold verdict; ``0.65`` — an ADJACENT grade (|Δ| = 1);\n    ``0.3`` — the extreme grades mixed up (|Δ| = 2) or any other parseable\n    label mismatch; ``0.0`` — unparseable / not a verdict object at all.'
    if not rerank_em_applies(task_class):
        return None
    gold = _coerce_label((extra or {}).get("gold_label"))
    if gold is None:
        # No normalizable gold verdict → undefined → N/A.
        
        return None
    got = parse_label(response or "")
    if got is None:
        return safe_score(0.0)
    # The label is ORDINAL: mixing up adjacent grades (0↔1, 1↔2) is less
    # wrong than inverting the verdict (0↔2), so the graded half decays
    # linearly with the distance.
    
    
    
    distance = abs(got - gold)
    label_part = _LABEL_SCORE * (_MAX_LABEL_DISTANCE - distance) / _MAX_LABEL_DISTANCE
    return safe_score(_FORMAT_SCORE + label_part)
