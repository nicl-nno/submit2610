'Deterministic extraction check — the label-free reward for GRPO-extraction.\n\n\nThe GRPO-extraction line trains entity/relation extraction in the RAGU style\non **unlabeled** corpora: the prompt promises an ontology schema, the model\nreturns a JSON of entities and relations, and the reward verifies the response\n**structurally** — no gold annotation is needed (the reward-hacking analysis\nand the anti-collapse consensus component live in this module too).\n\nComponents (combined by the strict harmonic mean; ``None`` = N/A, excluded) /\n\n1. ``json_format`` — the response parses into the stage envelope\n   (0.4 parse + 0.3 top-level shape + 0.3 per-item fields).\n2. ``schema_conformance`` — entity/relation types are members of the promised\n   ontology subset and relation triples satisfy the ``DOMAIN -> RANGE``\n   signatures (0.5 membership + 0.5 signatures, graded per item).\n3. ``entity_grounding`` — every extracted entity is attributable to the source\n   text: Snowball-stemmed (ru/en, ``nltk`` — no downloads needed) token\n   matching with a light fuzzy fallback (:mod:`difflib`), both sides stemmed\n   by the same function so stemming errors cancel; the head (last) stem must\n   match, the remaining stems earn the graded share.\n4. ``relation_closure`` — both endpoints of every relation resolve to actual\n   extracted entities (one-stage) / to the PROVIDED verified stage-1 entity\n   list (stage 2) — exactly what RAGU\'s consumer does\n   (``entity_by_name.get(source_entity)``).\n5. ``normalization`` — nominal case + singular number of the entity head form\n   via spaCy morphology (``ru_core_news_*`` / ``en_core_web_*``, lazily\n   imported; ``None`` when spaCy is unavailable — the component degrades\n   gracefully). Literal types (DATE/NUMBER/MONEY/...) are skipped.\n6. ``coverage`` — the anti-collapse consensus component (GRPO generates G\n   completions per prompt anyway): entity names present in ≥50 % of the\n   group\'s completions form a pseudo-reference; each completion is scored by\n   set-F1 against it. Without it, an empty extraction\n   ``{"entities": [], "relations": []}`` would be vacuously perfect on every\n   other component. Group-aware (non-i.i.d. within the group) — computed by\n   the dispatcher over the whole group and combined here.\n\nEverything is pure Python + ``nltk`` stemmers + optional spaCy — no network,\nno embeddings, no judge in the hot loop.'

from __future__ import annotations

import ast
import difflib
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from llm_training.extraction.ontology import (
    ExtractionSchema,
    get_ontology,
)
from llm_training.reward.scoring import harmonic_mean, safe_score

#: Task classes served by this check. They are the SAME classes the labeled
#: RAGU ICL data uses; the unlabeled corpus rows carry ``extra["extraction"]``
#: which switches the reward from the SemSim path to this deterministic check
#: (the ``nerel_f1`` / ``docsem_check`` replacement pattern, keyed by extra).

EXTRACTION_CLASSES: frozenset[str] = frozenset(
    {
        "ragu_artifact_extraction",  # one-stage
        "ragu_entity_extraction",    # two-stage stage 1
        "ragu_relation_extraction",  # two-stage stage 2
    }
)

#: The ``extra`` key carrying the extraction payload (stage/language/schema/
#: text/provided entities), produced by ``scripts/build_extraction_corpus.py``.
EXTRA_EXTRA_KEY: str = "extraction"

#: Weights of the ``json_format`` dimensions (sum 1.0, mirrors format_check).
_JSON_W_PARSE = 0.4
_JSON_W_SHAPE = 0.3
_JSON_W_FIELDS = 0.3

#: Stem-level fuzzy match threshold (difflib ratio) for grounding.

_FUZZY_STEM_THRESHOLD = 0.8

#: Share of group completions an entity must appear in to enter the consensus
# : pseudo-reference.
_CONSENSUS_SHARE = 0.5

#: spaCy models for the normalization component (operator installs them via
#: ``python -m spacy download ...``; absence degrades the component to None).
_SPACY_MODELS: dict[str, str] = {
    "ru": "ru_core_news_md",
    "en": "en_core_web_md",
}

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _strip_fences(response: str) -> str:
    'Strip markdown fences / prose around the outermost JSON object.'
    m = _FENCE_RE.search(response)
    if m and m.group(1).strip().startswith("{"):
        return m.group(1).strip()
    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end > start:
        return response[start : end + 1]
    return response.strip()


def _loads_tolerant(payload: str) -> Any:
    """``json.loads`` → ``ast.literal_eval`` fallback (repo convention)."""
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        try:
            return ast.literal_eval(payload)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None


def parse_extraction_output(response: str) -> dict[str, Any] | None:
    'Parse a response into ``{"entities": [...], "relations": [...]}``.\n\n\n    Tolerant to markdown fences, surrounding prose and Python-literal quoting\n    (the Qwen2.5 template sometimes emits Python repr). Items are normalized\n    to ``{entity_name, entity_type, description}`` /\n    ``{source_entity, target_entity, relation_type, description,\n    relationship_strength}``; malformed items are dropped, not fatal.\n    Returns ``None`` when nothing parses into a dict.'
    if not response or not response.strip():
        return None
    data = _loads_tolerant(_strip_fences(response))
    if not isinstance(data, dict):
        return None
    entities: list[dict[str, Any]] = []
    raw_entities = data.get("entities")
    if isinstance(raw_entities, list):
        for e in raw_entities:
            if not isinstance(e, dict):
                continue
            name = str(e.get("entity_name") or e.get("name") or "").strip()
            if not name:
                continue
            entities.append(
                {
                    "entity_name": name,
                    "entity_type": str(e.get("entity_type") or e.get("type") or "").strip(),
                    "description": str(e.get("description") or "").strip(),
                }
            )
    relations: list[dict[str, Any]] = []
    raw_relations = data.get("relations")
    if isinstance(raw_relations, list):
        for r in raw_relations:
            if not isinstance(r, dict):
                continue
            src = str(r.get("source_entity") or "").strip()
            tgt = str(r.get("target_entity") or "").strip()
            if not src or not tgt:
                continue
            relations.append(
                {
                    "source_entity": src,
                    "target_entity": tgt,
                    "relation_type": str(r.get("relation_type") or r.get("type") or "").strip(),
                    "description": str(r.get("description") or "").strip(),
                    "relationship_strength": r.get("relationship_strength"),
                }
            )
    return {"entities": entities, "relations": relations}


def extraction_json_score(response: str, stage: str) -> float:
    'Envelope score [0, 1]: 0.4 parse + 0.3 shape + 0.3 item fields.\n\n\n    The stage determines the expected top-level shape (``one_stage`` needs\n    both ``entities`` and ``relations`` lists; ``stage1`` / ``stage2`` need\n    their own). Empty lists are structurally valid — the anti-collapse\n    pressure lives in ``coverage``, not here. Empty item lists earn the\n    item-fields share neutrally.'
    parsed = parse_extraction_output(response)
    if parsed is None:
        return 0.0
    score = _JSON_W_PARSE
    raw = _loads_tolerant(_strip_fences(response))
    shape_ok = isinstance(raw, dict)
    if shape_ok:
        if stage == "one_stage":
            shape_ok = isinstance(raw.get("entities"), list) and isinstance(
                raw.get("relations"), list
            )
        elif stage == "stage1":
            shape_ok = isinstance(raw.get("entities"), list)
        elif stage == "stage2":
            shape_ok = isinstance(raw.get("relations"), list)
    if shape_ok:
        score += _JSON_W_SHAPE
    n_items = len(parsed["entities"]) + len(parsed["relations"])
    if n_items == 0:
        score += _JSON_W_FIELDS  # neutral: empty lists are valid JSON
    else:
        ok = sum(
            1
            for e in parsed["entities"]
            if e["entity_name"] and e["entity_type"] and e["description"]
        ) + sum(
            1
            for r in parsed["relations"]
            if r["source_entity"] and r["target_entity"] and r["relation_type"]
        )
        score += _JSON_W_FIELDS * (ok / n_items)
    return safe_score(score)


# ---------------------------------------------------------------------------
# Stemming helpers (Snowball, language-aware)
# ---------------------------------------------------------------------------
_stemmers: dict[str, Any] = {}


def _stemmer(lang: str) -> Any:
    """Lazily build + cache the Snowball stemmer of the language."""
    if lang not in _stemmers:
        from nltk.stem.snowball import EnglishStemmer, RussianStemmer  # no data needed

        _stemmers[lang] = RussianStemmer() if lang == "ru" else EnglishStemmer()
    return _stemmers[lang]


def detect_lang(text: str) -> str:
    """Cheap language pick for the stemmer: Cyrillic share > 20 % → ``ru``."""
    if not text:
        return "en"
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "en"
    cyr = sum(1 for ch in letters if "\u0400" <= ch <= "\u04FF")
    return "ru" if cyr / len(letters) > 0.2 else "en"


@lru_cache(maxsize=4096)
def _stem_tokens_cached(text: str, lang: str) -> tuple[tuple[str, int], ...]:
    """Tokenize + stem, returning ``((stem, token_len), ...)`` (hashable)."""
    from nltk import wordpunct_tokenize  # lazy: nltk is a train-extra dep.

    tokens = [
        t for t in (w.strip().lower() for w in wordpunct_tokenize(text or ""))
        if t and t.isalnum()
    ]
    stemmer = _stemmer(lang)
    return tuple((stemmer.stem(t), len(t)) for t in tokens)


def _stem_list(text: str, lang: str) -> list[str]:
    return [s for s, _ in _stem_tokens_cached(text, lang)]


def _canonical_name(name: str, lang: str) -> str:
    """Canonical (stem-joined, lowercased) form of an entity name for set ops."""
    return " ".join(_stem_list(name, lang))


def _fuzzy_member(stem: str, by_first: dict[str, set[str]]) -> bool:
    """True iff a text stem fuzzy-matches ``stem`` (same first char, ratio ≥ thr)."""
    candidates = by_first.get(stem[0], ()) if stem else ()
    for cand in candidates:
        if abs(len(cand) - len(stem)) <= 3 and difflib.SequenceMatcher(
            None, stem, cand
        ).ratio() >= _FUZZY_STEM_THRESHOLD:
            return True
    return False


def _entity_grounded(name: str, text_stems: list[str], by_first: dict[str, set[str]]) -> float:
    "Grounding score [0, 1] of ONE entity against the stemmed text.\n\n    ``0.4`` for the head (last) stem matching (exact or fuzzy) +\n    ``0.6`` × (share of the entity's stems found in the text). Both sides are\n    stemmed by the same function, so systematic stemming errors cancel;\n    the fuzzy fallback covers residual inflection/typo mismatches."
    stems = _stem_list(name, detect_lang(name))
    if not stems:
        return 0.0
    stem_set = set(text_stems)
    matched = sum(
        1
        for s in stems
        if s in stem_set or _fuzzy_member(s, by_first)
    )
    head_ok = 1.0 if (stems[-1] in stem_set or _fuzzy_member(stems[-1], by_first)) else 0.0
    return safe_score(0.4 * head_ok + 0.6 * matched / len(stems))

def entity_grounding_score(
    parsed: dict[str, Any] | None, source_text: str, language: str | None = None
) -> float | None:
    'Mean grounding of all extracted entities; ``None`` when N/A.\n\n    (nothing parsed / no entities — e.g. the stage-2 response carries only\n    relations, whose entities were verified at stage 1 already).'
    if not parsed or not parsed["entities"]:
        return None
    lang = language or detect_lang(source_text)
    text_stems = _stem_list(source_text, lang)
    by_first: dict[str, set[str]] = {}
    for s in text_stems:
        by_first.setdefault(s[0] if s else "", set()).add(s)
    scores = [
        _entity_grounded(e["entity_name"], text_stems, by_first)
        for e in parsed["entities"]
    ]
    return safe_score(sum(scores) / len(scores))


# ---------------------------------------------------------------------------
# Schema conformance
# ---------------------------------------------------------------------------
def extraction_schema_score(
    parsed: dict[str, Any] | None, schema: ExtractionSchema
) -> float | None:
    'Ontology membership + ``DOMAIN -> RANGE`` conformance; ``None`` when N/A.\n\n\n    Graded per item: ``0.5`` × the mean type-membership share (entity types\n    and relation types against the promised subset) + ``0.5`` × the mean\n    signature conformance of relations (a relation whose endpoint entity is\n    absent from the entity list is NOT signature-checked — ``closure``\n    already penalizes it). When the schema carries no signatures the\n    signature share is awarded neutrally. ``None`` when there is nothing to\n    check (no entities and no relations) or the schema is empty.'
    if not parsed:
        return None
    if not schema.entity_types and not schema.relation_types:
        return None
    if not parsed["entities"] and not parsed["relations"]:
        return None
    entity_set = set(schema.entity_types)
    relation_set = set(schema.relation_types)
    membership: list[float] = []
    for e in parsed["entities"]:
        membership.append(1.0 if e["entity_type"] in entity_set else 0.0)
    for r in parsed["relations"]:
        membership.append(1.0 if r["relation_type"] in relation_set else 0.0)
    member_share = sum(membership) / len(membership)
    has_signatures = bool(schema.signatures)
    if not has_signatures or not parsed["relations"]:
        sig_share = 1.0  # neutral: nothing to enforce
    else:
        ontology = get_ontology(schema.ontology)
        type_of = {e["entity_name"].casefold(): e["entity_type"] for e in parsed["entities"]}
        sig_scores: list[float] = []
        for r in parsed["relations"]:
            sig = schema.signatures.get(r["relation_type"])
            if sig is None:
                # Unknown type: membership already penalized; skip here.
                sig_scores.append(1.0)
                continue
            src_t = type_of.get(r["source_entity"].casefold())
            tgt_t = type_of.get(r["target_entity"].casefold())
            if src_t is None or tgt_t is None:
                # Endpoint not in the entity list — closure's concern.
                sig_scores.append(1.0)
                continue
            sig_scores.append(
                1.0
                if ontology.allows(r["relation_type"], src_t, tgt_t)
                else 0.0
            )
        sig_share = sum(sig_scores) / len(sig_scores)
    return safe_score(0.5 * member_share + 0.5 * sig_share)


# ---------------------------------------------------------------------------
# Relation closure
# ---------------------------------------------------------------------------
def relation_closure_score(
    parsed: dict[str, Any] | None,
    provided_entities: list[dict[str, Any]] | None = None,
) -> float | None:
    "Fraction of relations whose both endpoints resolve; ``None`` when N/A.\n\n    an empty relation list is legitimate for a short text; the anti-collapse\n    pressure on relations lives in the consensus ``coverage``). Endpoint\n    names are compared case-insensitively against the extracted entity names\n    (one-stage) or against the PROVIDED stage-1 entity list (stage 2) —\n    exactly RAGU's ``entity_by_name`` resolution."
    if not parsed or not parsed["relations"]:
        return None
    if provided_entities is not None:
        allowed = {
            str(e.get("entity_name") or "").strip().casefold()
            for e in provided_entities
            if isinstance(e, dict) and str(e.get("entity_name") or "").strip()
        }
    else:
        allowed = {e["entity_name"].casefold() for e in parsed["entities"]}
    if not allowed:
        return 0.0
    ok = sum(
        1
        for r in parsed["relations"]
        if r["source_entity"].casefold() in allowed
        and r["target_entity"].casefold() in allowed
    )
    return safe_score(ok / len(parsed["relations"]))


# ---------------------------------------------------------------------------
# Normalization (spaCy, optional)
# ---------------------------------------------------------------------------
#: spaCy components the normalization path actually needs. The morph
#: features (``Case``/``Number``) come from the ``tagger`` (+ the
#: ``attribute_ruler``), and ``t.is_alpha`` is tokenizer-level — so the
#: heavyweight ``parser``, ``ner`` and ``lemmatizer`` are dropped at load
#: time (≈ halves both the load latency and the resident size).


_SPACY_EXCLUDE: tuple[str, ...] = ("parser", "ner", "lemmatizer")

_spacy_pipelines: dict[str, Any] = {}


def _spacy_nlp(lang: str) -> Any:
    'Lazily load + cache the spaCy pipeline; raises when spaCy is missing.\n\n    install (``python -m spacy download ru_core_news_md en_core_web_md``);\n    callers map the failure to ``None`` (component N/A, never an abort).\n    Only the components the normalization needs are kept (``_SPACY_EXCLUDE``\n    drops parser/ner/lemmatizer).'
    if lang not in _spacy_pipelines:
        import spacy  # lazy: optional dependency of the [train] extra

        _spacy_pipelines[lang] = spacy.load(
            _SPACY_MODELS.get(lang, _SPACY_MODELS["en"]),
            exclude=list(_SPACY_EXCLUDE),
        )
    return _spacy_pipelines[lang]


def _is_literal_doc(doc: Any) -> bool:
    'True when the parsed name is a LITERAL value (numeric/temporal).\n\n    Content-driven detection (``token.like_num`` / ``is_digit`` / a ``NUM``\n    part of speech) CATCHES literals that slip into a nominal-typed slot\n    (a DATE phrase tagged ``PERSON``): such a value has no dictionary head\n    form, so nominal normalization is inapplicable and the component is\n    vacuously satisfied (``1.0``). Verified on ru_core_news_md 3.8.16:\nwithout this the trailing NOUN in Genitive'
    tokens = [t for t in doc if t.is_alpha or t.like_num or t.is_digit]
    if not tokens:
        return True
    # Any numeric token ⇒ literal value (dates/numbers/money carry digits).
    if any(t.like_num or t.is_digit or t.pos_ == "NUM" for t in tokens):
        return True
    return tokens[-1].pos_ == "NUM"


def _normalize_with_doc(doc: Any) -> float:
    'Morph normalization score [0, 1] over an ALREADY-parsed name.\n\n    and singular (``Number=Sing``); each half earns 0.5, a missing feature is\n    vacuously satisfied (English ``Case`` is sparse; PROPN may lack\n    ``Number``). A literal value (``_is_literal_doc``) is vacuously 1.0.'
    if _is_literal_doc(doc):
        return 1.0
    tokens = [t for t in doc if t.is_alpha]
    if not tokens:
        return 1.0
    head = tokens[-1]
    morph = head.morph
    case = morph.get("Case")
    number = morph.get("Number")
    nom_ok = (not case) or ("Nom" in case)
    sing_ok = (not number) or ("Sing" in number)
    return safe_score(0.5 * float(nom_ok) + 0.5 * float(sing_ok))


def _entity_normalized(name: str, lang: str) -> float:
    'Morphological normalization score [0, 1] of one entity name.\n\n    (last) token must be nominative (``Case=Nom``) and singular\n    (``Number=Sing``); each half earns 0.5, a missing morph feature is\n    vacuously satisfied (English ``Case`` is sparse; proper nouns may lack\n    ``Number``). Disambiguation is context-based inside the entity phrase —\n    the reason spaCy is used over dictionary analyzers. A literal value\n    (``_is_literal_doc``) is vacuously normalized (``1.0``): there is no\n    dictionary head form of a date/number.'
    doc = _spacy_nlp(lang)(name)
    return _normalize_with_doc(doc)


def entity_normalization_score(
    parsed: dict[str, Any] | None,
    language: str | None = None,
    ontology: str | None = None,
) -> float | None:
    'Mean normalization score over applicable entities; ``None`` when N/A.\n\n    nothing parsed, no applicable entities (LITERAL types of the row\'s\n    vocabulary — DATE/NUMBER/... — are skipped; there is no dictionary head\n    form of "17\n    degradation, never an abort). The literal set is per-vocabulary\n    (``nerel`` vs ``nerel_bio`` — the latter has no LANGUAGE).'
    if not parsed or not parsed["entities"]:
        return None
    literal_types = get_ontology(ontology).literal_types
    applicable = [
        e for e in parsed["entities"] if e["entity_type"] not in literal_types
    ]
    # Types outside the ontology's literal set are still nominal — checked.
    if not applicable:
        return None
    lang = language or detect_lang(parsed["entities"][0]["entity_name"])
    try:
        scores = [_entity_normalized(e["entity_name"], lang) for e in applicable]
    except Exception:
        # spaCy / models unavailable → component N/A (never an abort).
        return None
    return safe_score(sum(scores) / len(scores))


# ---------------------------------------------------------------------------
# Consensus coverage (anti-collapse)
# ---------------------------------------------------------------------------
def group_consensus_entities(
    parsed_group: list[dict[str, Any] | None],
    language: str | None = None,
    source_text: str | None = None,
    share: float = _CONSENSUS_SHARE,
) -> list[str]:
    "Canonical entity names present in ≥ ``share`` of the group's responses.\n\n    reference: mentions surviving across independently sampled completions\n    are near-certainly real. Names are canonicalized by language-aware\n    stemming, so inflection variants of one entity merge."
    group = [p for p in parsed_group if p and p["entities"]]
    if len(group) < 2:
        return []
    lang = language or detect_lang(source_text or "")
    counts: dict[str, int] = {}
    for parsed in group:
        seen: set[str] = set()
        for e in parsed["entities"]:
            seen.add(_canonical_name(e["entity_name"], lang))
        for key in seen:
            counts[key] = counts.get(key, 0) + 1
    need = max(2, math.ceil(share * len(group)))
    return [key for key, n in counts.items() if n >= need]


def coverage_f1(
    parsed: dict[str, Any] | None,
    consensus: list[str],
    language: str | None = None,
    source_text: str | None = None,
) -> float | None:
    "Set-F1 of the response's entity names against the group consensus.\n\n    missed ones: the empty-extraction degenerate optimum scores 0."
    if not consensus:
        return None
    lang = language or detect_lang(source_text or "")
    predicted = {
        _canonical_name(e["entity_name"], lang)
        for e in (parsed or {}).get("entities", [])
    }
    predicted.discard("")
    consensus_set = set(consensus)
    common = predicted & consensus_set
    if not common:
        return 0.0
    precision = len(common) / len(predicted)
    recall = len(common) / len(consensus_set)
    return safe_score(2 * precision * recall / (precision + recall))


# ---------------------------------------------------------------------------
# Pseudo-gold scoring (ctrl-points, judge-annotated once) /

# ---------------------------------------------------------------------------
def _canonical_triple(rel: dict[str, Any], lang: str) -> tuple[str, str, str] | None:
    src = _canonical_name(rel.get("source_entity") or "", lang)
    tgt = _canonical_name(rel.get("target_entity") or "", lang)
    rtype = str(rel.get("relation_type") or "").strip()
    if not src or not tgt or not rtype:
        return None
    return (src, tgt, rtype)


@dataclass
class PseudoGoldScored:
    "F1 breakdown of a response against the judge's pseudo-gold annotation."

    entity_f1: float | None
    relation_f1: float | None
    combined: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_f1": self.entity_f1,
            "relation_f1": self.relation_f1,
            "combined": self.combined,
        }


def pseudo_gold_score(
    parsed: dict[str, Any] | None,
    pseudo_gold: dict[str, Any] | None,
    language: str | None = None,
    source_text: str | None = None,
) -> PseudoGoldScored | None:
    "Score a parsed response against the cached judge annotation; ``None`` N/A.\n\n    component): the judge annotated the frozen val pool ONCE (see\n    :mod:`llm_training.runtime.pseudo_gold`), and every anchor compares the\n    model's extraction against the cached annotation — cheap, consistent\n    across steps (the measuring instrument stays frozen), methodology\n    different from the in-loop reward. Entities are compared as canonical\n    (stemmed) name sets, relations as canonical ``(source, relation, target)``\n    triples; per kind precision/recall F1, the pair combined by the strict\n    harmonic mean."
    if not isinstance(pseudo_gold, dict):
        return None
    gold_entities = pseudo_gold.get("entities")
    gold_relations = pseudo_gold.get("relations")
    if not isinstance(gold_entities, list) or not gold_entities:
        return None
    lang = language or detect_lang(source_text or "")
    gold_names = {
        _canonical_name(
            str(e.get("entity_name") or ""), lang
        )
        for e in gold_entities
        if isinstance(e, dict) and str(e.get("entity_name") or "").strip()
    }
    gold_names.discard("")
    pred_names = {
        _canonical_name(e["entity_name"], lang)
        for e in (parsed or {}).get("entities", [])
    }
    pred_names.discard("")

    def _f1(pred: set[str], gold: set[str]) -> float | None:
        if not gold:
            return None
        if not pred:
            return 0.0
        common = pred & gold
        if not common:
            return 0.0
        precision = len(common) / len(pred)
        recall = len(common) / len(gold)
        return safe_score(2 * precision * recall / (precision + recall))

    entity_f1 = _f1(pred_names, gold_names)
    relation_f1: float | None = None
    if isinstance(gold_relations, list) and gold_relations:
        gold_triples = {
            t
            for t in (
                _canonical_triple(r, lang)
                for r in gold_relations
                if isinstance(r, dict)
            )
            if t is not None
        }
        pred_triples = {
            t
            for t in (
                _canonical_triple(r, lang)
                for r in (parsed or {}).get("relations", [])
            )
            if t is not None
        }
        relation_f1 = _f1(pred_triples, gold_triples)
    active = [v for v in (entity_f1, relation_f1) if v is not None]
    if not active:
        return None
    return PseudoGoldScored(
        entity_f1=entity_f1,
        relation_f1=relation_f1,
        combined=harmonic_mean(active),
    )


# ---------------------------------------------------------------------------
# The combined check
# ---------------------------------------------------------------------------
@dataclass
class ExtractionScored:
    'Per-response component breakdown of the extraction check.'

    json_score: float
    combined: float
    schema_score: float | None = None
    grounding: float | None = None
    closure: float | None = None
    normalization: float | None = None
    coverage: float | None = None
    n_entities: int = 0
    n_relations: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "json_score": self.json_score,
            "schema_score": self.schema_score,
            "grounding": self.grounding,
            "closure": self.closure,
            "normalization": self.normalization,
            "coverage": self.coverage,
            "n_entities": self.n_entities,
            "n_relations": self.n_relations,
            "combined": self.combined,
        }


def extraction_check_applies(task_class: str, extra: dict[str, Any] | None) -> bool:
    """True iff this example is an unlabeled-corpus extraction row."""
    if task_class not in EXTRACTION_CLASSES:
        return False
    return isinstance((extra or {}).get(EXTRA_EXTRA_KEY), dict)


def extraction_check(
    response: str,
    task_class: str,
    extra: dict[str, Any] | None = None,
    group_responses: list[str] | None = None,
) -> ExtractionScored | None:
    'Score one extraction response; ``None`` when the check does not apply.\n\n    :data:`EXTRACTION_CLASSES`\n\n    Args:\n        response: the model\'s raw response text.\n        task_class: one of :data:`EXTRACTION_CLASSES`.\n        extra: the corpus-row payload — ``extraction.stage`` (``one_stage`` /\n            ``stage1`` / ``stage2``), ``extraction.language`` (``"ru"`` /\n            ``"en"`` / ``None``), ``extraction.text`` (the source chunk),\n            ``extraction.provided_entities`` (the verified stage-1 list) and\n            the :class:`ExtractionSchema` fields.\n        group_responses: ALL completions of the GRPO group (this response\n            included) for the consensus ``coverage``; ``None`` / a singleton\n            group skips the component (N/A). The reward is therefore\n            group-aware (non-i.i.d. within the group) — by design.'
    if not extraction_check_applies(task_class, extra):
        return None
    payload: dict[str, Any] = (extra or {})[EXTRA_EXTRA_KEY]
    stage = str(payload.get("stage") or "one_stage")
    source_text = str(payload.get("text") or "")
    language = payload.get("language")
    if language not in ("ru", "en", None):
        language = None
    schema = ExtractionSchema.from_extra(payload)
    provided = payload.get("provided_entities")
    provided_entities = provided if isinstance(provided, list) else None

    parsed = parse_extraction_output(response)
    json_score = extraction_json_score(response, stage)
    schema_score = (
        extraction_schema_score(parsed, schema) if schema is not None else None
    )
    grounding = (
        entity_grounding_score(parsed, source_text, language)
        if stage != "stage2"
        else None
    )
    closure = relation_closure_score(
        parsed,
        provided_entities=provided_entities if stage == "stage2" else None,
    )
    normalization = entity_normalization_score(
        parsed, language, ontology=payload.get("ontology")
    )
    coverage: float | None = None
    if group_responses and len(group_responses) >= 2:
        parsed_group = [parse_extraction_output(r) for r in group_responses]
        consensus = group_consensus_entities(parsed_group, language, source_text)
        coverage = coverage_f1(parsed, consensus, language, source_text)

    active: list[float] = [json_score]
    for comp in (schema_score, grounding, closure, normalization, coverage):
        if comp is not None:
            active.append(comp)
    return ExtractionScored(
        json_score=json_score,
        schema_score=schema_score,
        grounding=grounding,
        closure=closure,
        normalization=normalization,
        coverage=coverage,
        n_entities=len((parsed or {}).get("entities", [])),
        n_relations=len((parsed or {}).get("relations", [])),
        combined=harmonic_mean(active),
    )
