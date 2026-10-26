'Deterministic ruMMLU-pro exact match — the option letter comparison.\n\n\n``rummlu_pro`` is a multiple-choice task whose system prompt mandates a\nsingle-character answer (the letter of the chosen option, ``\none-letter answer is meaningless — cosine similarity between a single letter\nand the reference letter barely separates the correct choice from the wrong\nones, and an embedding of "\nreward therefore REPLACES SemSim for this class with a deterministic exact\nmatch (the NEREL-F1 / docsem pattern): extract the answer letter from the\nresponse and compare it with the reference letter.\nNEREL-F1 / docsem).\n\nExtraction (tolerant to typical format deviations, strict in substance):\n\n* surrounding whitespace / markdown emphasis (``*``/``_``) and trailing\n  punctuation (``.``, ``,``, ``)``, ``:``, ``!``, ``?``, ``;``) are stripped;\n``Answer:``\n  (case-insensitive, optional punctuation) is stripped before re-checking;\n* if the remainder is exactly ONE letter (Cyrillic or Latin) it is the\n  extracted answer; anything longer (a word, a sentence, the option TEXT)\n  extracts to ``None`` → the metric scores ``0.0`` — the model violated the\n  single-letter contract, a strong format gradient (the same philosophy as\n  the strict harmonic mean: a tool-call turn without a ``<tool_call>`` block\n  scores ``format_score=0``).\n\nThe comparison itself is case-insensitive but does NOT fold Latin lookalikes\nformat discipline the metric enforces.'

from __future__ import annotations

import re
from typing import Any

from llm_training.reward.scoring import safe_score

#: The only task class scored by the deterministic letter exact match.

RUMLU_EM_CLASS: str = "rummlu_pro"

# Latin), case-insensitive.

_LETTER_RE = re.compile(r"^[А-ЯЁа-яёA-Za-z]$")

# : An optional answer label before the letter: "

_LABEL_RE = re.compile(
    r"^(?:ответ|answer)\s*[):\-.—–]?\s*", re.IGNORECASE
)

#: Characters stripped around the answer: markdown emphasis + punctuation.

_STRIP_CHARS = " \t\r\n*_`'\"#>:.!,);(?¡¿…"


def extract_answer_letter(text: str) -> str | None:
    'Extract the single answer letter from a response; ``None`` if not one.\n\n\n    Tolerant to whitespace / markdown emphasis / trailing punctuation and an\n"Answer:" label; a longer answer (prose, the\n    option text) returns ``None`` — the caller maps that to ``0.0``.'
    if not text:
        return None
    cleaned = text.strip(_STRIP_CHARS)
    if not cleaned:
        return None
    if _LETTER_RE.match(cleaned):
        return cleaned.lower()
    stripped = _LABEL_RE.sub("", cleaned, count=1)
    if stripped != cleaned:
        stripped = stripped.strip(_STRIP_CHARS)
        if _LETTER_RE.match(stripped):
            return stripped.lower()
    return None


def rummlu_em_applies(task_class: str) -> bool:
    'True iff this example should be scored by the deterministic letter EM.'
    return task_class == RUMLU_EM_CLASS


def rummlu_em(
    response: str,
    reference: str,
    task_class: str,
    extra: dict[str, Any] | None = None,
) -> float | None:
    "Deterministic option-letter exact match ∈ {0.0, 1.0}; ``None`` if N/A.\n\n\n    Returns ``None`` (not applicable — excluded from the harmonic combine,\n    the class falls back to Semantic Similarity) when ``task_class`` is not\n    ``rummlu_pro`` or the reference is not a single letter (a data anomaly —\n    the loader's references are exactly one letter). Otherwise returns\n    ``1.0`` when the response's extracted letter equals the reference letter\n    and ``0.0`` otherwise — including a response long enough that no single\n    letter could be extracted (a format violation of the single-letter\n    contract, a strong gradient toward the mandated output format)."
    if not rummlu_em_applies(task_class):
        return None
    gold = extract_answer_letter(reference or "")
    if gold is None:
        # Malformed reference (not a single letter) — undefined → N/A.
        
        return None
    got = extract_answer_letter(response or "")
    return safe_score(1.0 if got == gold else 0.0)
