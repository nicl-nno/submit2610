'Deterministic DocSem check — executable PoT scoring against the gold.\n\n\nThe ``docsem_quant`` task class (document-grounded quantitative reasoning,\ndistilled from the DocSem shared task — see\n``scripts/build_docsem_quant.py``) is a deterministic task: the gold is a\ncanonical numeric answer plus a set of evidence block ids, and the model\'s\nresponse is a JSON object ``{"reasoning", "program", "answer", "evidence"}``\n(Program-of-Thought: the model writes a short Python program, the reward\nEXECUTES it in the safe interpreter —\n:mod:`llm_training.reward.pot_interpreter`, ported from the champion DocSem\npipeline). Semantic Similarity over a JSON envelope is meaningless (it would\nreward lexical overlap of program text, not numeric correctness), so for\nthis class the reward replaces SemSim with the deterministic\n:func:`docsem_check` — the same pattern as\n:func:`llm_training.reward.nerel_f1.nerel_f1` for the NEREL extraction\nsub-configs: no embeddings, no network, zero reward-hacking surface.\n\n``{"reasoning", "program", "answer", "evidence"}`` (PoT\n``docsem_check``\n\nScoring dimensions (strict harmonic mean\n\n* **format** (graded 0.4 + 0.3 + 0.2 + 0.1, mirroring ``format_check``): the\n  response parses as JSON (markdown fences tolerated) → 0.4; every required\n  key is present with the right type (``reasoning``/``program``/``answer``\n  str, ``evidence`` list of str) → 0.3; ``answer`` and ``evidence`` are\n  non-empty → 0.2; the ``program`` is non-empty (the PoT envelope: a docsem\n  answer without a program is structurally wrong at the format level too)\n  → 0.1. An unparseable response scores 0 → the whole check is 0 (a strong\n  format gradient, the warm-up philosophy without the warm-up).\n* **exec** (0/1): the ``program`` executes under the safe interpreter\'s AST\n  whitelist.\n* **answer_exact** (0/1): the EFFECTIVE answer (the executed program value;\n  the stated ``answer`` field with light numeric cleaning as the fallback —\n  the champion\'s semantics) canonically equals the gold answer (numeric\n  canonicalization ported from the DocSem portal scorer: thousands\n  separators, comma decimals, integer-valued floats).\n* **evidence_f1**: set-level F1 over the evidence block ids (exact strings;\n  ported from the champion scorer — single-element gold sets degenerate to\n  an exact 0/1).'

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from llm_training.reward.pot_interpreter import (
    ProgramError,
    execute_program,
    format_result,
)
from llm_training.reward.scoring import harmonic_mean, safe_score

# : The task class scored by :func:`docsem_check`.
DOCSEM_QUANT_CLASS: str = "docsem_quant"

#: Sub-score weights of the format dimension (sum 1.0, mirroring
#: ``format_check``): JSON parse / key types / non-empty answer+evidence /
# : non-empty program.
_FORMAT_WEIGHT_JSON = 0.4     # parses as JSON (fences tolerated).
_FORMAT_WEIGHT_TYPES = 0.3    # every required key has the right type.
_FORMAT_WEIGHT_NONEMPTY = 0.2  # answer / evidence non-empty.

#: Markdown fences around the JSON (models emit them despite the prompt;
# : the judge-side sanitizer strips the same thing).
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+.-]*[ \t]*\r?\n?(.*?)\r?\n?```[ \t]*$", re.DOTALL)

#: Numeric extraction from a stated answer ("`48`", "48 books", "1,067 items").

_NUMBER_RE = re.compile(r"-?\d[\d\s]*(?:[.,]\d+)?")

#: Thousands separators: 1,067 / 1.234.567 (dot-grouping unsupported — the
# : dot is reserved as the decimal point).
_THOUSANDS_RE = re.compile(r"^[+-]?\d{1,3}(,\d{3})+$")

# : Comma decimal: 10,5.
_COMMA_DECIMAL_RE = re.compile(r"^[+-]?\d+,\d+$")

#: Leading answer markers stripped before parsing ("answer:", "##").

_ANSWER_MARKER_RE = re.compile(r"^(?:#+\s*|(?:final\s+)?answer\s*[:\-]\s*|ответ\s*[:\-]\s*)+", re.IGNORECASE)


@dataclass(frozen=True)
class DocsemCheck:
    'Per-dimension breakdown of :func:`docsem_check`.'

    format_score: float
    exec_score: float
    answer_score: float
    evidence_f1: float
    score: float


def docsem_check_applies(task_class: str) -> bool:
    'True iff the class is scored by :func:`docsem_check`.'
    return task_class == DOCSEM_QUANT_CLASS


def strip_docsem_fences(text: str) -> str:
    'Strip a single wrapping markdown fence, if present.'
    stripped = (text or "").strip()
    match = _FENCE_RE.match(stripped)
    return match.group(1).strip() if match else stripped


def parse_docsem_response(response: str) -> dict[str, Any] | None:
    "Parse the model's JSON response (fence-tolerant); ``None`` if invalid."
    if not response or not response.strip():
        return None
    text = strip_docsem_fences(response)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _canonical_number(value: float) -> str:
    'Canonical numeric string (integer without a dot, decimal without\nthe DocSem portal rule.'
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.12f}".rstrip("0").rstrip(".")


def canonical_answer(raw: str) -> str | None:
    'Canonicalize a numeric answer string; ``None`` if not numeric.\n\n\n    Accepts plain decimals, thousands separators (``1,067``) and comma\n    decimals (``10,5``); strips leading ``answer:`` / ``#`` markers and\n    surrounding whitespace. Non-numeric text canonicalizes to ``None``\n    (never compared loosely).'
    text = _ANSWER_MARKER_RE.sub("", str(raw or "").strip()).strip()
    if not text:
        return None
    try:
        return _canonical_number(float(text))
    except ValueError:
        pass
    compact = text.replace(" ", "").replace("_", "")
    if _THOUSANDS_RE.match(compact):
        return _canonical_number(float(compact.replace(",", "")))
    if _COMMA_DECIMAL_RE.match(compact):
        return _canonical_number(float(compact.replace(",", ".")))
    try:
        return _canonical_number(float(compact))
    except ValueError:
        return None


def clean_stated_answer(raw: str) -> str:
    'Extract the numeric value from a stated answer ("`48`", "48 books").\n\n\n    Ported from the champion\'s ``_clean_stated_answer``: the first numeric\n    run wins; a non-numeric string is returned as-is (the canonical\n    comparison then fails honestly).'
    match = _NUMBER_RE.search(str(raw or ""))
    if match is None:
        return str(raw or "").strip()
    return match.group(0).replace(" ", "")


def evidence_f1(predicted: Sequence[str], gold: Sequence[str]) -> float:
    'Set-level F1 over evidence block ids (exact strings, stripped).'
    pred_set = {str(e).strip() for e in predicted or [] if str(e).strip()}
    gold_set = {str(e).strip() for e in gold or [] if str(e).strip()}
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    common = len(pred_set & gold_set)
    return (2.0 * common) / (len(pred_set) + len(gold_set))


def docsem_answer_text(response: str) -> str | None:
    "Extract the model's effective answer string from its JSON response.\n\n\n    Used by the content metrics (AC at the anchors, offline Phase-0\n    analysis): the executed program value when the program runs, otherwise\n    the stated ``answer`` field; ``None`` when the JSON is unparseable (no\n    scoreable content — mirrors ``extract_final_answer`` for SGR finals)."
    data = parse_docsem_response(response)
    if data is None:
        return None
    program = data.get("program")
    if isinstance(program, str) and program.strip():
        try:
            return format_result(execute_program(program))
        except ProgramError:
            pass
    answer = data.get("answer")
    if isinstance(answer, str) and answer.strip():
        return clean_stated_answer(answer)
    return None


def docsem_check(
    response: str,
    gold_answer: Any,
    gold_evidence: Any,
) -> float:
    'Deterministic score ∈ [0, 1] for a ``docsem_quant`` response.\n\n\n    Strict harmonic mean of the four dimensions (format / exec /\n    answer_exact / evidence_f1 — see the module docstring). Raises\n    :class:`ValueError` when the gold metadata is missing (a data bug —\n    the caller maps that to a failed metric, never to a silent skip).'
    return safe_score(docsem_check_detailed(response, gold_answer, gold_evidence).score)


def docsem_check_detailed(
    response: str,
    gold_answer: Any,
    gold_evidence: Any,
) -> DocsemCheck:
    'Per-dimension breakdown; see :func:`docsem_check`.'
    if not isinstance(gold_answer, str) or not gold_answer:
        raise ValueError(
            "docsem_check requires a non-empty string gold answer (extra['gold_answer']).\ndocsem_check"
        )
    if not isinstance(gold_evidence, (list, tuple)):
        raise ValueError(
            "docsem_check requires a list gold evidence (extra['gold_evidence']).\ndocsem_check"
        )

    data = parse_docsem_response(response)
    if data is None:
        return DocsemCheck(
            format_score=0.0, exec_score=0.0, answer_score=0.0,
            evidence_f1=0.0, score=0.0,
        )

    # --- format: JSON parsed (0.4) + key types (0.3) + non-empty (0.2) +
    # non-empty program (0.1).
    types_ok = (
        isinstance(data.get("reasoning"), str)
        and isinstance(data.get("program"), str)
        and isinstance(data.get("answer"), str)
        and isinstance(data.get("evidence"), list)
        and all(isinstance(e, str) for e in data["evidence"])
    )
    answer_field = data.get("answer")
    evidence_field = data.get("evidence")
    nonempty_ok = (
        isinstance(answer_field, str) and bool(answer_field.strip())
        and isinstance(evidence_field, list) and len(evidence_field) > 0
    )
    program_field = data.get("program")
    program_nonempty = isinstance(program_field, str) and bool(program_field.strip())
    fmt = _FORMAT_WEIGHT_JSON + (
        _FORMAT_WEIGHT_TYPES if types_ok else 0.0
    ) + (_FORMAT_WEIGHT_NONEMPTY if nonempty_ok else 0.0) + (0.1 if program_nonempty else 0.0)

    # --- exec: the program runs under the AST whitelist.
    executed: str | None = None
    if program_nonempty:
        try:
            executed = format_result(execute_program(program_field))
        except ProgramError:
            executed = None
    exec_score = 1.0 if executed is not None else 0.0

    # --- answer_exact: effective answer (executed; stated fallback) vs
    # gold, canonically.
    gold_canonical = canonical_answer(gold_answer)
    if executed is not None:
        effective = executed
    elif isinstance(answer_field, str) and answer_field.strip():
        effective = clean_stated_answer(answer_field)
    else:
        effective = ""
    effective_canonical = canonical_answer(effective)
    answer_score = (
        1.0
        if gold_canonical is not None
        and effective_canonical is not None
        and effective_canonical == gold_canonical
        else 0.0
    )

    # --- evidence set-F1.
    ev_f1 = evidence_f1(
        evidence_field if isinstance(evidence_field, list) else [],
        [str(e) for e in gold_evidence],
    )

    score = harmonic_mean([fmt, exec_score, answer_score, ev_f1])
    return DocsemCheck(
        format_score=safe_score(fmt),
        exec_score=exec_score,
        answer_score=answer_score,
        evidence_f1=safe_score(ev_f1),
        score=safe_score(score),
    )
