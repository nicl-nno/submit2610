'Deterministic summarization checks for the GRPO-extraction line.\n\n\nDispatch: ``task_class`` ∈ {``ragu_entity_summary``, ``ragu_relation_summary``,\n``ragu_community_summary``} AND ``extra["summary"]`` present (the same\nextra-keyed replacement pattern as :mod:`llm_training.reward.extraction`).\nThe row payload\n\n* entity:   ``{"kind": "entity", "entity_name": ..., "entity_type": ...,\n              "descriptions": [...]}``\n* relation: ``{"kind": "relation", "subject": ..., "object": ...,\n              "relation_type": ..., "descriptions": [...]}``\n* community: ``{"kind": "community", "entities": [...], "relations": [...]}``\n\nThe task is CONSOLIDATION: merge multiple primary descriptions into one, or\nreport a local graph community. The "reference" is implicit in the input, so\nevery component is reference-free and deterministic (the line\'s core\ninvariant)\n\n* **json_score** — the output parses into the promised envelope (tolerant to\n  fences/prose, the extraction convention) with the right key types and a\n  non-empty core (``description`` / ``title``+``summary``+``findings``);\n* **coverage** (recall) — the share of each primary description\'s\n  content-word stems that survive into the output (nothing essential lost);\n* **precision** (faithfulness) — the share of output content-word stems\n  grounded in the inputs (anti-hallucination);\n* **membership** (community only) — the share of findings mentioning at\n  least one entity of the PROVIDED subgraph (anti-invention of actors);\n* **compression** — the anti-concatenation guard of the CONSOLIDATION\n  kinds (entity/relation merging must compress the inputs; concatenating\n  them would otherwise be the degenerate optimum). A community REPORT is\n  N/A here: a report legitimately ELABORATES on its material (5–10\n  findings with explanations), so the band would punish every good report;\n* **repetition** — n-gram loop penalty inside the output;\n* **fluency** — OPTIONAL (a callable wired by the trainer when the grammar\n  classifier is resident; ``None`` → component N/A, mirrors the graceful\n  spaCy degradation).\n\nCombined by the STRICT harmonic mean of the active components (a zero on\nany active component zeroes the reward — the line\'s convention).'

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from llm_training.reward.extraction import (
    _loads_tolerant,
    _stem_list,
    _strip_fences,
    detect_lang,
)
from llm_training.reward.scoring import harmonic_mean, safe_score

logger = logging.getLogger(__name__)

# : Task classes served by this check
SUMMARY_TASK_CLASSES: frozenset[str] = frozenset(
    {
        "ragu_entity_summary",
        "ragu_relation_summary",
        "ragu_community_summary",
    }
)

# : Payload key in ``extra``
SUMMARY_EXTRA_KEY = "summary"

#: The judge-written reference cache inside ``extra["summary"]`` — the
#: summary twin of ``llm_training.runtime.pseudo_gold.PSEUDO_GOLD_KEY``.
#: Written ONCE by ``scripts/annotate_pseudo_gold.py`` (summary mode) as
#: ``{"response": <raw judge envelope>, "model": <judge id>}``; the anchor
#: compares the policy's prose against it with plain clipped-cosine SemSim
#: (judge-free at anchor time — the pseudo-gold precedent).


# : ``{"response": ..., "model": ...}``

SUMMARY_REFERENCE_KEY = "reference"

#: Content words are lowercase alphabetic tokens of length > 2 (function
#: words and 1-2-letter abbreviations carry no consolidation signal).

_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")

#: Output length cap as a share of the input length (the compression band).
#: The zero point is EXACTLY the concatenation of the inputs (ratio ≈ 1.0):
#: a legitimate merge sits well below, a concat is killed outright.

COMPRESSION_CAP: float = 0.75
COMPRESSION_FAIL_AT: float = 1.0

#: Community findings must number 1..10 (RAGU asks 5–10; the graded floor
#: is 1 so a single-finding answer still parses but scores the format
#: share, and the coverage/precision components still apply).
FINDINGS_MIN, FINDINGS_MAX = 1, 10

FluencyFn = Callable[[str], "float | None"]


@dataclass
class SummaryScored:
    'Per-component scores of one summarization response.\n\n    component drops from the harmonic, the extraction convention).'

    kind: str
    json_score: float
    coverage: float | None = None
    precision: float | None = None
    membership: float | None = None
    compression: float | None = None
    repetition: float | None = None
    fluency: float | None = None
    failed_metrics: list[str] = field(default_factory=list)
    combined: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """All components as a flat dict (the eval report shape)."""
        return {
            "kind": self.kind,
            "json_score": self.json_score,
            "coverage": self.coverage,
            "precision": self.precision,
            "membership": self.membership,
            "compression": self.compression,
            "repetition": self.repetition,
            "fluency": self.fluency,
            "combined": self.combined,
        }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_summary_output(response: str, kind: str) -> dict[str, Any] | None:
    "Parse a response into the kind's envelope; ``None`` when unusable.\n\n    markdown fences, surrounding prose and Python-literal quoting (the same\n    loader family the extraction parser uses)."
    if not response or not response.strip():
        return None
    data = _loads_tolerant(_strip_fences(response))
    if not isinstance(data, dict):
        return None
    if kind == "community":
        findings = data.get("findings")
        if not isinstance(findings, list):
            return None
        clean: list[dict[str, str]] = []
        for f in findings:
            if not isinstance(f, dict):
                continue
            summary = str(f.get("summary") or "").strip()
            explanation = str(f.get("explanation") or "").strip()
            if summary or explanation:
                clean.append({"summary": summary, "explanation": explanation})
        if not clean:
            return None
        data["findings"] = clean
        return data
    description = str(data.get("description") or "").strip()
    if not description:
        return None
    return data


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def summary_json_score(response: str, kind: str) -> float:
    'Graded envelope validity [0, 1] (the extraction ``json`` convention).\n\n    ``0.4`` parse + ``0.3`` key types + ``0.3`` non-empty core; for\n    community the core includes the findings list within\n    ``[FINDINGS_MIN, FINDINGS_MAX]`` and a rating in ``[0, 10]``.'
    data = parse_summary_output(response, kind)
    if data is None:
        return 0.0
    score = 0.4
    if kind == "community":
        keys_ok = (
            isinstance(data.get("title"), str)
            and isinstance(data.get("summary"), str)
            and isinstance(data.get("rating"), (int, float))
            and not isinstance(data.get("rating"), bool)
            and isinstance(data.get("rating_explanation"), str)
            and isinstance(data.get("findings"), list)
        )
        rating = data.get("rating")
        rating_ok = isinstance(rating, (int, float)) and 0 <= float(rating) <= 10
        findings: list[Any] = data.get("findings") or []
        core_ok = (
            bool(str(data.get("title") or "").strip())
            and bool(str(data.get("summary") or "").strip())
            and bool(str(data.get("rating_explanation") or "").strip())
            and FINDINGS_MIN <= len(findings) <= FINDINGS_MAX
        )
        # Weights sum to 1.0: parse 0.4 + keys 0.2 + rating 0.1 + core 0.3.
        if keys_ok:
            score += 0.2
        if rating_ok:
            score += 0.1
        if core_ok:
            score += 0.3
        return safe_score(min(1.0, score))
    name_key = "entity_name" if kind == "entity" else "subject_name"
    keys_ok = isinstance(data.get(name_key), str) and isinstance(
        data.get("description"), str
    )
    core_ok = bool(str(data.get("description") or "").strip())
    if keys_ok:
        score += 0.3
    if core_ok:
        score += 0.3
    return safe_score(min(1.0, score))


def _content_stems(text: str, lang: str) -> set[str]:
    """Content-word stems of a text (len > 2 before stemming)."""
    words = [w.lower() for w in _WORD_RE.findall(text or "") if len(w) > 2]
    return {s for s in _stem_list(" ".join(words), lang) if len(s) > 1}


def summary_coverage_score(
    descriptions: list[str], output: str, lang: str
) -> float:
    "Mean share of each primary description's stems surviving the output."
    out_stems = _content_stems(output, lang)
    if not out_stems:
        return 0.0
    shares: list[float] = []
    for desc in descriptions:
        stems = _content_stems(desc, lang)
        if not stems:
            continue
        shares.append(len(stems & out_stems) / len(stems))
    return safe_score(sum(shares) / len(shares)) if shares else 0.0


def summary_precision_score(
    descriptions: list[str], output: str, lang: str
) -> float:
    'Share of output stems grounded in the inputs (anti-hallucination).'
    in_stems: set[str] = set()
    for desc in descriptions:
        in_stems |= _content_stems(desc, lang)
    out_stems = _content_stems(output, lang)
    if not out_stems:
        return 0.0
    return safe_score(len(in_stems & out_stems) / len(out_stems))


def summary_compression_score(
    descriptions: list[str],
    output: str,
    cap: float = COMPRESSION_CAP,
    fail_at: float = COMPRESSION_FAIL_AT,
) -> float:
    'Anti-concatenation band: ``1.0`` up to ``cap`` of the input length,\n    linear decay to 0 at ``fail_at`` (the concatenation of the inputs).\n\n    ``fail_at`` (≈'
    out_n = len(_WORD_RE.findall(output or ""))
    in_n = sum(len(_WORD_RE.findall(d)) for d in descriptions)
    if out_n == 0 or in_n == 0:
        return 0.0
    ratio = out_n / in_n
    if ratio <= cap:
        return 1.0
    if ratio >= fail_at:
        return 0.0
    return safe_score((fail_at - ratio) / (fail_at - cap))


def summary_repetition_score(output: str, n: int = 3) -> float:
    '``1 − duplicated n-gram share`` inside the output (loop penalty).'
    words = [w.lower() for w in _WORD_RE.findall(output or "")]
    if len(words) < n * 2:
        return 1.0
    grams: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - n + 1):
        key = tuple(words[i : i + n])
        grams[key] = grams.get(key, 0) + 1
    duplicated = sum((c - 1) * n for c in grams.values() if c > 1)
    return safe_score(1.0 - duplicated / len(words))


def community_membership_score(
    entities: list[dict[str, Any]], output: str, lang: str
) -> float:
    'Share of findings mentioning ≥1 entity of the PROVIDED subgraph.\n\n    (inflected mentions count); an entity with no usable stems matches by\n    its casefolded name containment.'
    data = parse_summary_output(output, "community")
    if data is None:
        return 0.0
    findings: list[dict[str, str]] = data.get("findings") or []
    if not findings:
        return 0.0
    names: list[str] = []
    for e in entities:
        name = str(e.get("entity_name") or "").strip()
        if name:
            names.append(name)
    if not names:
        return 0.0
    hits = 0
    for f in findings:
        # Containment is checked against THIS finding's text only — matching
        # against the whole response would let one global mention validate
        # every finding (the anti-invention hole).
        
        
        f_text = f"{f.get('summary', '')} {f.get('explanation', '')}"
        f_stems = _content_stems(f_text, lang)
        f_lower = f_text.casefold()
        hit = False
        for name in names:
            stems = _content_stems(name, lang)
            if stems and (stems & f_stems):
                hit = True
                break
            if name.casefold() in f_lower:
                hit = True
                break
        if hit:
            hits += 1
    return safe_score(hits / len(findings))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def summarization_applies(task_class: str, extra: dict[str, Any] | None) -> bool:
    """True for the summary classes carrying ``extra["summary"]``."""
    return (
        task_class in SUMMARY_TASK_CLASSES
        and isinstance(extra, dict)
        and isinstance(extra.get(SUMMARY_EXTRA_KEY), dict)
    )


def _output_text(data: dict[str, Any], kind: str) -> str:
    """The prose payload of a parsed response (what coverage/precision see)."""
    if kind == "community":
        parts = [
            str(data.get("title") or ""),
            str(data.get("summary") or ""),
            str(data.get("rating_explanation") or ""),
        ]
        for f in data.get("findings") or []:
            parts.append(str(f.get("summary") or ""))
            parts.append(str(f.get("explanation") or ""))
        return " ".join(parts)
    return str(data.get("description") or "")


def summary_output_text(response: str, kind: str) -> str | None:
    "The prose payload of a RAW summary response; ``None`` if unusable.\n\n    Parses the envelope (fence/prose tolerant, the shared loader family) and\n    returns the SAME text the coverage/precision components see\n    (:func:`_output_text`); an unparseable envelope or an empty payload\n    yields ``None``. This is the text the anchor SemSim compares — the\n    policy's response prose against the cached judge reference prose."
    data = parse_summary_output(response, kind)
    if data is None:
        return None
    text = _output_text(data, kind).strip()
    return text or None


def _descriptions_of(payload: dict[str, Any]) -> list[str]:
    kind = payload.get("kind")
    if kind == "entity":
        return [str(d) for d in payload.get("descriptions") or []]
    if kind == "relation":
        return [str(d) for d in payload.get("descriptions") or []]
    # community: the primary material is entity descriptions + relation
    # descriptions (the same texts the prompt rendered).
    descs: list[str] = []
    for e in payload.get("entities") or []:
        d = str(e.get("description") or "").strip()
        if d:
            descs.append(d)
    for r in payload.get("relations") or []:
        d = str(r.get("description") or "").strip()
        if d:
            descs.append(d)
    return descs


def summarization_check(
    task_class: str,
    extra: dict[str, Any] | None,
    response: str,
    fluency: FluencyFn | None = None,
) -> SummaryScored | None:
    'Score one summarization response; ``None`` when the check is N/A.\n\n    Mirrors :func:`llm_training.reward.extraction.extraction_check`: keyed by\n    ``task_class`` + ``extra["summary"]``, strictly harmonic combine, a\n    failed optional component (fluency) records into ``failed_metrics``\n    instead of zeroing.'
    if not summarization_applies(task_class, extra):
        return None
    payload: dict[str, Any] = (extra or {})[SUMMARY_EXTRA_KEY]
    kind = str(payload.get("kind") or "")
    if kind not in ("entity", "relation", "community"):
        return None

    json_score = summary_json_score(response, kind)
    data = parse_summary_output(response, kind)
    descriptions = _descriptions_of(payload)
    source_text = payload.get("text")
    lang = detect_lang(
        source_text if source_text else " ".join(descriptions)
    )

    coverage = precision = membership = None
    compression = repetition = fluency_score = None
    failed: list[str] = []
    if data is not None:
        output_text = _output_text(data, kind)
        if descriptions:
            coverage = summary_coverage_score(descriptions, output_text, lang)
            precision = summary_precision_score(descriptions, output_text, lang)
            # Compression is a CONSOLIDATION guard only (entity/relation
            # merging must compress the inputs). A community REPORT
            # legitimately ELABORATES on its material (5-10 findings with
            # explanations) — the band would punish every good report.
            # Compression
            
            if kind in ("entity", "relation"):
                compression = summary_compression_score(
                    descriptions, output_text
                )
        repetition = summary_repetition_score(output_text)
        if kind == "community":
            membership = community_membership_score(
                payload.get("entities") or [], response, lang
            )
        if fluency is not None:
            try:
                fluency_score = fluency(output_text)
            except Exception:  # noqa: BLE001 - a failed metric never aborts
                fluency_score = None
                failed.append("fluency")
    # A broken envelope keeps the structural components active at their
    # zero-defaults — the strict harmonic zeroes the reward either way.

    active: list[float] = [json_score]
    for comp in (coverage, precision, membership, compression, repetition,
                 fluency_score):
        if comp is not None:
            active.append(comp)
    return SummaryScored(
        kind=kind,
        json_score=json_score,
        coverage=coverage,
        precision=precision,
        membership=membership,
        compression=compression,
        repetition=repetition,
        fluency=fluency_score,
        failed_metrics=failed,
        combined=harmonic_mean(active) if active else 0.0,
    )


# ---------------------------------------------------------------------------
# Anchor SemSim
# ---------------------------------------------------------------------------
def summary_reference_semsim(
    semsim_fn: Callable[[str, str], float],
    payload: dict[str, Any],
    response: str,
) -> float | None:
    'Clipped-cosine SemSim of the response prose vs the cached reference.\n\n    (``annotate_pseudo_gold.py`` summary mode), either prose side is\n    unusable (an unparseable response — the structural combined already\n    punishes it), or the embedder call fails: a measurement failure degrades\n    the row to the structural combined alone, never aborts the anchor/eval.\n    The cosine is clipped to ``[0, 1]`` for the strict harmonic combine.\n``None``'
    ref = payload.get(SUMMARY_REFERENCE_KEY)
    ref_raw = ref.get("response") if isinstance(ref, dict) else ref
    kind = str(payload.get("kind") or "")
    if not isinstance(ref_raw, str) or not ref_raw.strip() or not kind:
        return None
    try:
        ref_prose = summary_output_text(ref_raw, kind)
        out_prose = summary_output_text(response, kind)
        if not ref_prose or not out_prose:
            return None
        value = float(semsim_fn(out_prose, ref_prose))
    except Exception as exc:  # noqa: BLE001 - observability, never fatal
        logger.warning("Summary reference SemSim failed: %s", type(exc).__name__)
        return None
    if value != value:  # NaN guard
        return None
    return max(0.0, min(1.0, value))


def make_summary_semsim(
    embeddings: Any,
) -> Callable[[str, str], float] | None:
    'Plain clipped-cosine SemSim over a synchronous embeddings object.\n\n    the closure, so the frozen reference texts stay embedded across anchors)\n    and injected into the anchor as ``summary_semsim``.\n\n    ``None`` when the object exposes no synchronous ``embed_query`` /\n    ``embed_documents`` pair — the anchor then falls back to the structural\n    ``combined`` alone (a missing embedder is a degraded measurement, never\n    an abort). The cosine of two vectors is clipped to ``[0, 1]``: a negative\n    cosine (no shared content) floors at 0, keeping the value on the reward\n    scale for the strict harmonic combine. Embeddings are cached per text\n    (bounded — policy responses change every anchor; references do not).\n``embed_documents``'
    query_fn = getattr(embeddings, "embed_query", None)
    docs_fn = getattr(embeddings, "embed_documents", None)
    if not callable(query_fn) and not callable(docs_fn):
        return None

    cache: dict[str, list[float]] = {}

    def _embed(text: str) -> list[float]:
        if callable(query_fn):
            return list(query_fn(text))
        return list(docs_fn([text])[0])  # type: ignore[index]

    def semsim(text_a: str, text_b: str) -> float:
        vectors: list[list[float]] = []
        for text in (text_a, text_b):
            vec = cache.get(text)
            if vec is None:
                vec = _embed(text)
                if len(cache) >= 8192:
                    # Bounded: responses churn per anchor — drop everything
                    # (the next anchor re-embeds the references once).
                    
                    cache.clear()
                cache[text] = vec
            vectors.append(vec)
        va, vb = vectors
        if not va or len(va) != len(vb):
            return 0.0
        dot = math.fsum(x * y for x, y in zip(va, vb))
        na = math.sqrt(math.fsum(x * x for x in va))
        nb = math.sqrt(math.fsum(y * y for y in vb))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (na * nb)))

    return semsim
