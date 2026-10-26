'Build the unlabeled GRPO-extraction corpus (RAGU-style prompts), offline.\n\n\nThe producer of the GRPO-extraction training data: takes a style-stratified\ncorpus of plain-text chunks (open corpora — literary / scientific /\njournalistic / conversational, RU + EN) and emits prompt rows for the three\nextraction stages. NO references are produced — the whole point of the line is\nthat the reward (``llm_training.reward.extraction``) is label-free; the row\'s\n``extra["extraction"]`` payload carries the SAME schema the prompt promises,\nso the prompt and the reward cannot drift apart.\n\nRow format (deliberately NOT :class:`UnifiedExample` — that schema requires a\nnon-empty reference; the corpus is reference-free by design)\n\n    {"id": "ext-000123",\n     "task_class": "ragu_artifact_extraction" | "ragu_entity_extraction"\n                                 | "ragu_relation_extraction",\n     "prompt": [{"role": "system", ...}, {"role": "user", ...}],\n     "extra": {"style": ..., "source": ..., "target_language": "ru"|"en"|null,\n               "extraction": {stage, language, text, entity_types,\n                              relation_types, signatures,\n                              provided_entities?}}}\n\nTwo subcommands\n\n* ``build`` — one-stage + stage-1 prompts from raw chunks (pure offline).\n* ``stage2`` — stage-2 prompts conditioned on the VERIFIED stage-1 entities:\n  reads the stage-1 generations (a vLLM batch over the stage-1 rows) and\n  keeps only entities that pass the deterministic verification (grounding +\n  schema membership), mirroring the reward itself — the same functions from\n  ``llm_training.reward.extraction`` are the single source of truth.\n\nEverything is deterministic under ``--seed`` (re-runs reproduce the output\nbyte-for-byte).\n\nUsage\n\n    python scripts/build_extraction_corpus.py build         --input-jsonl chunks.jsonl --out src_data/extraction/train.jsonl         --seed 42 --val-out src_data/extraction/val.jsonl --val-per-style 8\n\n    python scripts/build_extraction_corpus.py stage2         --corpus src_data/extraction/train.jsonl         --generations stage1_gens.jsonl         --out src_data/extraction/stage2.jsonl --seed 42'

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from llm_training.extraction.ontology import (
    ExtractionSchema,
    get_ontology,
)
from llm_training.extraction.prompts import (
    LANGUAGE_VALUES,
    STAGE_ENTITIES,
    STAGE_ONE,
    STAGE_RELATIONS,
    build_extraction_prompt,
)
from llm_training.extraction.qa_prompts import build_qa_prompt
from llm_training.extraction.summary_prompts import (
    build_community_summary_prompt,
    build_entity_summary_prompt,
    build_relation_summary_prompt,
)
from llm_training.reward import extraction as exm

#: Known style strata (anything else is accepted but reported).

STYLES: tuple[str, ...] = (
    "literary", "scientific", "journalistic", "conversational", "qa",
)

# : task_class per stage.
_STAGE_CLASS: dict[str, str] = {
    STAGE_ONE: "ragu_artifact_extraction",
    STAGE_ENTITIES: "ragu_entity_extraction",
    STAGE_RELATIONS: "ragu_relation_extraction",
}

_WORD_RE = re.compile(r"\S+")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Input reading
# ---------------------------------------------------------------------------
def iter_input_records(
    input_jsonl: str | list[str] | None = None,
    corpus_dir: str | None = None,
) -> Iterator[dict[str, Any]]:
    'Yield ``{text, style, lang, source, doc_id, ontology, code_switched}``.\n\n    code_switched}``\n    ``<style>/<lang>/*.txt``, one document per file). ``input_jsonl`` accepts\n    ONE PATH OR A LIST (repeatable CLI flag — the collector writes one file\n    per source). JSONL fields: ``text`` (required), ``style`` (required),\n    ``lang`` — ``ru``/``en`` (default ``ru``), ``source`` (optional\n    provenance string), ``doc_id`` (optional — the DOCUMENT identity carried\n    through chunking; auto-derived from the file position when absent),\n    ``ontology`` (optional — ``"nerel"`` | ``"nerel_bio"``; rows built from\n    the record promise THAT vocabulary), ``code_switched`` (optional bool —\n    the RU-EN mixed register, surfaced in the row\'s ``extra``).'
    files = [input_jsonl] if isinstance(input_jsonl, str) else list(input_jsonl or ())
    for input_jsonl_path in files:
        yield from _iter_jsonl_records(input_jsonl_path)
    if corpus_dir:
        root = Path(corpus_dir)
        if not root.is_dir():
            raise SystemExit(
                f'--corpus-dir {corpus_dir!r} is not a directory.\n--corpus-dir {corpus_dir!r}'
            )
        for style_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for lang_dir in sorted(p for p in style_dir.iterdir() if p.is_dir()):
                lang = lang_dir.name if lang_dir.name in ("ru", "en") else "ru"
                for file_path in sorted(lang_dir.glob("*.txt")):
                    text = file_path.read_text(encoding="utf-8", errors="replace").strip()
                    if text:
                        yield {
                            "text": text,
                            "style": style_dir.name,
                            "lang": lang,
                            "source": str(file_path),
                            "doc_id": str(file_path),
                            "ontology": None,
                            "code_switched": False,
                        }


def _iter_jsonl_records(input_jsonl: str) -> Iterator[dict[str, Any]]:
    """Yield records from ONE JSONL file (see :func:`iter_input_records`)."""
    with open(input_jsonl, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f'{input_jsonl}:{line_no}: not valid JSON ({exc}).\n{input_jsonl}:{line_no}'
                ) from exc
            text = str(rec.get("text") or "").strip()
            style = str(rec.get("style") or "").strip()
            if not text or not style:
                continue
            yield {
                "text": text,
                "style": style,
                "lang": rec.get("lang") if rec.get("lang") in ("ru", "en") else "ru",
                "source": str(rec.get("source") or f"{input_jsonl}#{line_no}"),
                "doc_id": str(rec.get("doc_id") or f"{input_jsonl}#{line_no}"),
                "ontology": str(rec["ontology"]) if rec.get("ontology") else None,
                "code_switched": bool(rec.get("code_switched")),
            }


def chunk_text(text: str, chunk_words: int) -> list[str]:
    'Split a document into deterministic word windows (no overlap).'
    words = _WORD_RE.findall(text)
    if len(words) <= chunk_words:
        return [_WS_RE.sub(" ", text).strip()]
    # Drop a tiny tail fragment (a quarter of the window) — it carries no
    # extraction signal.
    min_window = max(1, chunk_words // 4)
    chunks: list[str] = []
    for start in range(0, len(words), chunk_words):
        window = words[start : start + chunk_words]
        if len(window) >= min_window:
            chunks.append(" ".join(window))
    return chunks


_CHUNKER_TOKENIZER: dict[str, Any] = {}


def _chunker_tokenizer(model_id: str) -> Any:
    'Lazily load + cache the chunker tokenizer (bge-m3 from the local cache).'
    if model_id not in _CHUNKER_TOKENIZER:
        from transformers import AutoTokenizer

        _CHUNKER_TOKENIZER[model_id] = AutoTokenizer.from_pretrained(model_id)
    return _CHUNKER_TOKENIZER[model_id]


def chunk_text_tokens(
    text: str,
    tokenizer: Any,
    tokens_min: int,
    tokens_max: int,
    rng: random.Random,
) -> list[str]:
    "Token-window chunking with a SEEDED per-chunk target size.\n\n    a window sampled from ``[tokens_min, tokens_max]`` (the operator's\n    200–700 spec), producing a natural size mix instead of uniform windows.\n    Windows are cut on token ids and decoded back to text; a re-tokenized\n    chunk stays within ``[tokens_min - 2, tokens_max + 2]`` (decoder drift\n    of a couple of tokens is expected and bounded)."
    if tokens_min < 1 or tokens_max < tokens_min:
        raise ValueError(
            f"invalid token window: min={tokens_min}, max={tokens_max}"
        )
    ids = tokenizer(text)["input_ids"]
    if len(ids) <= tokens_max:
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        return [_WS_RE.sub(" ", decoded).strip()]
    # Drop a tiny tail (half of the minimum window) — no extraction signal.
    
    min_tail = max(1, tokens_min // 2)
    chunks: list[str] = []
    position = 0
    while position < len(ids):
        target = rng.randint(tokens_min, tokens_max)
        window = ids[position : position + target]
        if len(window) < min_tail and chunks:
            break
        decoded = tokenizer.decode(window, skip_special_tokens=True)
        cleaned = _WS_RE.sub(" ", decoded).strip()
        if cleaned:
            chunks.append(cleaned)
        position += len(window)
    return chunks


_CHUNKER_TOKENIZER: dict[str, Any] = {}


def _chunker_tokenizer(model_id: str) -> Any:
    'Lazily load + cache the chunker tokenizer (bge-m3 from the local cache).'
    if model_id not in _CHUNKER_TOKENIZER:
        from transformers import AutoTokenizer

        _CHUNKER_TOKENIZER[model_id] = AutoTokenizer.from_pretrained(model_id)
    return _CHUNKER_TOKENIZER[model_id]


def _punct_ratio(text: str) -> float:
    letters = sum(1 for ch in text if ch.isalpha())
    punct = sum(1 for ch in text if ch in ".,!?;:\"'()—–")
    if not letters:
        return 0.0
    return punct / letters


def looks_human_made(text: str, min_punct_ratio: float) -> bool:
    'Cheap proxy for human-made conversational text (vs ASR auto-subs).\n\n    minimal punctuation density AND a share of sentence-initial capitals.'
    if _punct_ratio(text) < min_punct_ratio:
        return False
    sentences = [s for s in re.split(r"[.!?]\s+", text) if s.strip()]
    if not sentences:
        return False
    capitalized = sum(1 for s in sentences if s[0].isupper())
    return capitalized / len(sentences) >= 0.5


# ---------------------------------------------------------------------------
# build subcommand
# ---------------------------------------------------------------------------
def build_rows(
    records: list[dict[str, Any]],
    seed: int,
    schema_mode: str = "mixed",
    n_entity_types: int = 10,
    n_entity_types_max: int | None = None,
    stage1_share: float = 0.5,
    compact_variant_share: float = 0.3,
    no_target_language_share: float = 0.2,
    with_signatures_share: float = 0.5,
    chunk_words: int = 350,
    chunk_tokens_min: int | None = None,
    chunk_tokens_max: int | None = None,
    chunker_tokenizer: str = "BAAI/bge-m3",
    min_words: int = 40,
    max_words: int = 1200,
    min_punct_ratio: float = 0.02,
    max_rows: int | None = None,
    per_style: int | None = None,
) -> list[dict[str, Any]]:
    "Build one-stage + stage-1 prompt rows from raw chunk records.\n\n    Deterministic under ``seed``; the RNG stream is consumed in a fixed order\n    (chunk splitting and filtering happen BEFORE any sampling, so the corpus\n    composition is stable regardless of the sampling knobs). Per-chunk knobs:\n    the record's ``ontology`` selects the vocabulary (``nerel`` default,\n    ``nerel_bio`` for biomedical documents — never merged, see\n    :mod:`llm_training.extraction.ontology`); ``n_entity_types_max`` > ``n\n   _entity_types`` makes the schema size K vary per row (seeded uniform) —\n    an additional task-difficulty axis. ``doc_id`` rides into ``extra`` and\n    the whole document stays on ONE side of the train/val split."
    rng = random.Random(seed)
    chunk_rng = random.Random(f"{seed}:chunk-tokens")
    tokenizer = None
    tokens_min = chunk_tokens_min if chunk_tokens_min is not None else 200
    tokens_max = chunk_tokens_max if chunk_tokens_max is not None else 700
    if chunk_tokens_min is not None or chunk_tokens_max is not None:
        tokenizer = _chunker_tokenizer(chunker_tokenizer)
    # 1) Chunk + filter + dedup (deterministic, RNG-free).
    seen_hashes: set[str] = set()
    chunks: list[dict[str, Any]] = []
    dropped = {"short": 0, "long": 0, "dup": 0, "not_human": 0}
    for rec in records:
        if tokenizer is not None:
            pieces = chunk_text_tokens(
                rec["text"], tokenizer, tokens_min, tokens_max, chunk_rng
            )
        else:
            pieces = chunk_text(rec["text"], chunk_words)
        for piece in pieces:
            n_words = len(_WORD_RE.findall(piece))
            if n_words < min_words:
                dropped["short"] += 1
                continue
            if n_words > max_words:
                dropped["long"] += 1
                continue
            if rec["style"] == "conversational" and not looks_human_made(
                piece, min_punct_ratio
            ):
                dropped["not_human"] += 1
                continue
            h = hashlib.blake2b(
                _WS_RE.sub(" ", piece.lower()).encode("utf-8"), digest_size=16
            ).hexdigest()
            if h in seen_hashes:
                dropped["dup"] += 1
                continue
            seen_hashes.add(h)
            chunks.append({**rec, "text": piece})
    # 2) Per-style cap.
    if per_style is not None:
        per_style_counts: dict[str, int] = {}
        capped: list[dict[str, Any]] = []
        for chunk in chunks:
            used = per_style_counts.get(chunk["style"], 0)
            if used >= per_style:
                continue
            per_style_counts[chunk["style"]] = used + 1
            capped.append(chunk)
        chunks = capped
    # 3) Prompt assembly (seeded).
    rows: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, 1):
        if max_rows is not None and len(rows) >= max_rows:
            break
        stage = STAGE_ENTITIES if rng.random() < stage1_share else STAGE_ONE
        ontology = get_ontology(chunk.get("ontology"))
        if n_entity_types_max is not None and n_entity_types_max > n_entity_types:
            k_types = rng.randint(n_entity_types, n_entity_types_max)
        else:
            k_types = n_entity_types
        if schema_mode == "full":
            schema = ontology.full_schema()
        elif schema_mode == "subset":
            schema = ontology.subset(rng, n_entity_types=k_types)
        else:  # mixed
            schema = (
                ontology.full_schema()
                if rng.random() < 0.3
                else ontology.subset(rng, n_entity_types=k_types)
            )
        target_language = (
            None if rng.random() < no_target_language_share else chunk["lang"]
        )
        language_value = (
            "the same language as the source text"
            if target_language is None
            else LANGUAGE_VALUES[target_language]
        )
        variant = 1 if rng.random() < compact_variant_share else 0
        with_signatures = rng.random() < with_signatures_share
        prompt = build_extraction_prompt(
            stage,
            chunk["text"],
            schema,
            language=language_value,
            variant=variant,
            with_signatures=with_signatures,
        )
        extraction_payload = schema.to_extra()
        extraction_payload.update(
            {
                "stage": stage,
                "language": target_language,
                "text": chunk["text"],
            }
        )
        rows.append(
            {
                "id": f"ext-{idx:06d}",
                "task_class": _STAGE_CLASS[stage],
                "prompt": prompt,
                "extra": {
                    "style": chunk["style"],
                    "source": chunk["source"],
                    "doc_id": chunk.get("doc_id") or chunk["source"],
                    "target_language": target_language,
                    "code_switched": bool(chunk.get("code_switched")),
                    "extraction": extraction_payload,
                },
            }
        )
    return rows


def split_val(
    rows: list[dict[str, Any]], val_per_style: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    'Carve a frozen stratified anchor sample out of the built rows.\n\n    land on the same side, so no val chunk comes from a document whose other\n    chunks trained (the anchor measures transfer, not memorization of the\n    same document). Returns ``(train_rows, val_rows)``; val ids are\n    re-prefixed.'
    rng = random.Random(f"{seed}:val")
    by_style: dict[str, list[str]] = {}
    doc_rows: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        doc = row["extra"].get("doc_id") or row["extra"].get("source") or f"row#{i}"
        doc_rows.setdefault(doc, []).append(i)
        by_style.setdefault(row["extra"]["style"], []).append(doc)
    val_docs: set[str] = set()
    for style in sorted(by_style):
        docs = sorted(set(by_style[style]))
        rng.shuffle(docs)
        kept = 0
        for doc in docs:
            if kept >= val_per_style:
                break
            val_docs.add(doc)
            kept += 1
    val_rows = []
    train_rows = []
    n_val = 0
    for i, row in enumerate(rows):
        doc = row["extra"].get("doc_id") or row["extra"].get("source") or f"row#{i}"
        if doc in val_docs:
            n_val += 1
            val_rows.append({**row, "id": f"ext-val-{n_val:06d}"})
        else:
            train_rows.append(row)
    return train_rows, val_rows


# ---------------------------------------------------------------------------
# stage2 subcommand
# ---------------------------------------------------------------------------
#: Grounding threshold for a stage-1 entity to enter the stage-2 list.

_DEFAULT_VERIFY_GROUNDING = 0.6


def verify_stage1_response(
    response: str, payload: dict[str, Any]
) -> list[dict[str, Any]] | None:
    'Verify a stage-1 response; return the verified entity list (or ``None``).\n\n    JSON parses; an ENTITY is verified when its type is a schema member and\n    its grounding against the source text clears the threshold — the SAME\n    functions the reward uses (single source of truth).'
    parsed = exm.parse_extraction_output(response)
    if parsed is None or not parsed["entities"]:
        return None
    schema = ExtractionSchema.from_extra(payload)
    if schema is None:
        return None
    entity_set = set(schema.entity_types)
    language = payload.get("language")
    text = str(payload.get("text") or "")
    text_stems = exm._stem_list(text, language or exm.detect_lang(text))
    by_first: dict[str, set[str]] = {}
    for s in text_stems:
        by_first.setdefault(s[0] if s else "", set()).add(s)
    verified: list[dict[str, Any]] = []
    for e in parsed["entities"]:
        if e["entity_type"] not in entity_set:
            continue
        grounded = exm._entity_grounded(e["entity_name"], text_stems, by_first)
        if grounded >= _DEFAULT_VERIFY_GROUNDING:
            verified.append(e)
    return verified


def build_stage2_rows(
    corpus_rows: list[dict[str, Any]],
    generations: dict[str, str],
    min_entities: int = 2,
    with_signatures_share: float = 0.5,
    compact_variant_share: float = 0.3,
    seed: int = 42,
) -> list[dict[str, Any]]:
    'Build stage-2 rows over verified stage-1 entities.'
    rng = random.Random(f"{seed}:stage2")
    rows: list[dict[str, Any]] = []
    n_no_gen, n_bad, n_few = 0, 0, 0
    for row in corpus_rows:
        if row["task_class"] != _STAGE_CLASS[STAGE_ENTITIES]:
            continue
        response = generations.get(row["id"])
        if response is None:
            n_no_gen += 1
            continue
        payload = row["extra"]["extraction"]
        verified = verify_stage1_response(response, payload)
        if verified is None:
            n_bad += 1
            continue
        if len(verified) < min_entities:
            n_few += 1
            continue
        schema = ExtractionSchema.from_extra(payload)
        assert schema is not None
        target_language = row["extra"].get("target_language")
        language_value = (
            "the same language as the source text"
            if target_language is None
            else LANGUAGE_VALUES[target_language]
        )
        variant = 1 if rng.random() < compact_variant_share else 0
        with_signatures = rng.random() < with_signatures_share
        prompt = build_extraction_prompt(
            STAGE_RELATIONS,
            payload["text"],
            schema,
            language=language_value,
            variant=variant,
            with_signatures=with_signatures,
            provided_entities=verified,
        )
        extraction_payload = schema.to_extra()
        extraction_payload.update(
            {
                "stage": STAGE_RELATIONS,
                "language": target_language,
                "text": payload["text"],
                "provided_entities": verified,
            }
        )
        rows.append(
            {
                "id": row["id"].replace("ext-", "ext-s2-", 1),
                "task_class": _STAGE_CLASS[STAGE_RELATIONS],
                "prompt": prompt,
                "extra": {
                    "style": row["extra"]["style"],
                    "source": row["extra"]["source"],
                    # The document id MUST survive into stage-2 rows: the
                    # summary builders group relation mentions across the
                    # chunks of ONE document by doc_id (the chunk sources
                    # differ — wiki/1, wiki/2, ...).
                    # doc_id
                    
                    
                    "doc_id": row["extra"].get("doc_id")
                    or row["extra"].get("source"),
                    "target_language": target_language,
                    "extraction": extraction_payload,
                },
            }
        )
    print(
        f'stage2: {len(rows)} rows (no generation: {n_no_gen}, unusable response: {n_bad}, too few entities: {n_few}).\nstage2: {len(rows)}'
    )
    return rows


def _load_corpus_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# summaries subcommand
# ---------------------------------------------------------------------------
def _canonical_key(name: str) -> str:
    'Casefolded, whitespace-normalized grouping key of an entity name.'
    return " ".join(re.split(r"\s+", name.strip().casefold())).strip()


def _verified_stage1_entities(
    corpus_rows: list[dict[str, Any]], generations: dict[str, str]
) -> list[dict[str, Any]]:
    'Verified stage-1 mentions: ``{doc_id, style, lang, name, type, desc}``.'
    mentions: list[dict[str, Any]] = []
    for row in corpus_rows:
        if row.get("task_class") != _STAGE_CLASS[STAGE_ENTITIES]:
            continue
        response = generations.get(row["id"])
        if not response:
            continue
        payload = row["extra"]["extraction"]
        verified = verify_stage1_response(response, payload)
        if not verified:
            continue
        mentions.append(
            {
                "doc_id": row["extra"].get("doc_id") or row["extra"].get("source"),
                "style": row["extra"]["style"],
                "lang": row["extra"].get("target_language"),
                "source": row["extra"].get("source"),
                "entities": verified,
            }
        )
    return mentions


def build_entity_summary_rows(
    corpus_rows: list[dict[str, Any]],
    generations: dict[str, str],
    min_descriptions: int = 2,
    max_descriptions: int = 8,
    seed: int = 42,
) -> list[dict[str, Any]]:
    'Entity-consolidation rows: group verified mentions by (name, type)\n    ACROSS chunks of one document.\n\n    ``>= min_descriptions``\n    ``extra["summary"]`` (kind ``entity``).'
    rng = random.Random(f"{seed}:entity-summaries")
    mentions = _verified_stage1_entities(corpus_rows, generations)
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for m in mentions:
        for e in m["entities"]:
            key = (_canonical_key(e["entity_name"]), e["entity_type"], m["doc_id"])
            g = groups.setdefault(
                key,
                {
                    "name": e["entity_name"],
                    "type": e["entity_type"],
                    "descriptions": [],
                    "doc_id": m["doc_id"],
                    "style": m["style"],
                    "lang": m["lang"],
                    "source": m["source"],
                },
            )
            desc = str(e.get("description") or "").strip()
            if desc:
                g["descriptions"].append(desc)
    rows: list[dict[str, Any]] = []
    idx = 0
    for key in sorted(groups):
        g = groups[key]
        descs = g["descriptions"]
        if len(descs) < min_descriptions:
            continue
        if len(descs) > max_descriptions:
            descs = rng.sample(descs, max_descriptions)
        idx += 1
        language = LANGUAGE_VALUES.get(g["lang"], LANGUAGE_VALUES["ru"])
        rows.append(
            {
                "id": f"ext-sum-e-{idx:06d}",
                "task_class": "ragu_entity_summary",
                "prompt": build_entity_summary_prompt(
                    g["name"], g["type"], descs, language
                ),
                "extra": {
                    "style": g["style"],
                    "source": g["source"],
                    "doc_id": g["doc_id"],
                    "summary": {
                        "kind": "entity",
                        "entity_name": g["name"],
                        "entity_type": g["type"],
                        "descriptions": descs,
                    },
                },
            }
        )
    return rows


def build_relation_summary_rows(
    corpus_rows: list[dict[str, Any]],
    stage2_generations: dict[str, str],
    min_descriptions: int = 2,
    max_descriptions: int = 8,
    seed: int = 42,
) -> list[dict[str, Any]]:
    "Relation-consolidation rows: group stage-2 relation mentions by\n    (subject, relation, object) across chunks of one document.\n\n    names resolve through the row's ``provided_entities`` (the stage-2\n    contract: relations reference the provided list)."
    rng = random.Random(f"{seed}:relation-summaries")
    groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in corpus_rows:
        if row.get("task_class") != _STAGE_CLASS[STAGE_RELATIONS]:
            continue
        response = stage2_generations.get(row["id"])
        if not response:
            continue
        payload = row["extra"]["extraction"]
        parsed = exm.parse_extraction_output(response)
        if not parsed or not parsed["relations"]:
            continue
        schema = ExtractionSchema.from_extra(payload)
        allowed = set(schema.relation_types) if schema else set()
        name_type = {
            _canonical_key(e["entity_name"]): e["entity_type"]
            for e in payload.get("provided_entities") or []
        }
        doc_id = row["extra"].get("doc_id") or row["extra"].get("source")
        for r in parsed["relations"]:
            rel = r["relation_type"]
            if allowed and rel not in allowed:
                continue
            subject = r["source_entity"]
            obj = r["target_entity"]
            desc = str(r.get("description") or "").strip()
            if not subject or not obj or not desc:
                continue
            key = (
                _canonical_key(subject), rel, _canonical_key(obj), doc_id
            )
            g = groups.setdefault(
                key,
                {
                    "subject": subject,
                    "subject_type": name_type.get(_canonical_key(subject), ""),
                    "relation_type": rel,
                    "object": obj,
                    "object_type": name_type.get(_canonical_key(obj), ""),
                    "descriptions": [],
                    "doc_id": doc_id,
                    "style": row["extra"]["style"],
                    "lang": row["extra"].get("target_language"),
                    "source": row["extra"].get("source"),
                },
            )
            g["descriptions"].append(desc)
    rows: list[dict[str, Any]] = []
    idx = 0
    for key in sorted(groups):
        g = groups[key]
        descs = g["descriptions"]
        if len(descs) < min_descriptions:
            continue
        if len(descs) > max_descriptions:
            descs = rng.sample(descs, max_descriptions)
        idx += 1
        language = LANGUAGE_VALUES.get(g["lang"], LANGUAGE_VALUES["ru"])
        rows.append(
            {
                "id": f"ext-sum-r-{idx:06d}",
                "task_class": "ragu_relation_summary",
                "prompt": build_relation_summary_prompt(
                    g["subject"], g["subject_type"], g["relation_type"],
                    g["object"], g["object_type"], descs, language,
                ),
                "extra": {
                    "style": g["style"],
                    "source": g["source"],
                    "doc_id": g["doc_id"],
                    "summary": {
                        "kind": "relation",
                        "subject": g["subject"],
                        "subject_type": g["subject_type"],
                        "relation_type": g["relation_type"],
                        "object": g["object"],
                        "object_type": g["object_type"],
                        "descriptions": descs,
                    },
                },
            }
        )
    return rows


def build_qa_rows(
    records: list[dict[str, Any]],
    seed: int,
    with_context_share: float = 0.5,
    min_question_words: int = 2,
    max_question_words: int = 80,
    min_answer_words: int = 2,
    max_answer_words: int = 400,
    compact_variant_share: float = 0.3,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    "Build reference-bearing QA rows from FineWeb2-HQ-QA records.\n\n    These are the ONE reference-bearing class of this line (everything else\n    is label-free): ``reference`` = the collector's answer, scored by the\n    standard SemSim-vs-reference + fluency path (the QA decision of the\n    plan). Deterministic under ``seed``; the context flavour is a seeded\n    per-row coin (``with_context_share`` = P(open-book)).\n\n    Each record: ``{question, answer, text?, source, url, doc_id, lang}``."
    rng = random.Random(f"{seed}:qa")
    rows: list[dict[str, Any]] = []
    idx = 0
    for rec in records:
        q = str(rec.get("question") or "").strip()
        a = str(rec.get("answer") or "").strip()
        if not q or not a:
            continue
        nq = len(q.split())
        na = len(a.split())
        if not (min_question_words <= nq <= max_question_words):
            continue
        if not (min_answer_words <= na <= max_answer_words):
            continue
        text = str(rec.get("text") or "").strip()
        with_context = bool(text) and rng.random() < with_context_share
        variant = 1 if rng.random() < compact_variant_share else 0
        lang = str(rec.get("lang") or "en").strip() or "en"
        if lang not in LANGUAGE_VALUES:
            lang = "en"
        prompt = build_qa_prompt(
            q, text if with_context else None,
            language=LANGUAGE_VALUES[lang], variant=variant,
        )
        idx += 1
        rows.append(
            {
                "id": f"qa-{idx:06d}",
                "task_class": "fineweb_qa",
                "prompt": prompt,
                "reference": a,
                "extra": {
                    "style": "qa",
                    "source": rec.get("source") or "fineweb2",
                    "url": rec.get("url"),
                    "doc_id": rec.get("doc_id"),
                    "target_language": lang,
                    "with_context": with_context,
                    # The reward/anchor SemSim user_input (the question, not the
                    # rendered prompt with the context block).
                    # user_input
                    "qa": {"question": q, "answer": a, "with_context": with_context},
                },
            }
        )
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def detect_communities(
    nodes: list[str],
    edges: list[tuple[str, str]],
    min_size: int = 4,
    max_size: int = 24,
    max_iters: int = 20,
) -> list[list[str]]:
    'Deterministic label propagation over the undirected document graph.\n\n    each adopts the most frequent neighbour label (ties → the smallest\n    label), iterated to stability. Communities outside\n    ``[min_size, max_size]`` are dropped / trimmed (trimmed by sorted order\n    — deterministic).'
    if not nodes:
        return []
    order = sorted(set(nodes))
    index = {n: i for i, n in enumerate(order)}
    labels = {n: index[n] for n in order}
    neighbors: dict[str, list[str]] = {n: [] for n in order}
    for src, tgt in edges:
        if src in index and tgt in index and src != tgt:
            neighbors[src].append(tgt)
            neighbors[tgt].append(src)
    for _ in range(max_iters):
        changed = False
        for node in order:
            counts: dict[int, int] = {}
            for nb in neighbors[node]:
                counts[labels[nb]] = counts.get(labels[nb], 0) + 1
            if not counts:
                continue
            best = min(
                (-(c), l) for l, c in counts.items()
            )[1]
            if labels[node] != best:
                labels[node] = best
                changed = True
        if not changed:
            break
    communities: dict[int, list[str]] = {}
    for node in order:
        communities.setdefault(labels[node], []).append(node)
    result: list[list[str]] = []
    for label in sorted(communities):
        members = sorted(communities[label])
        if len(members) < min_size:
            continue
        if len(members) > max_size:
            members = members[:max_size]
        result.append(members)
    return result


def build_community_summary_rows(
    corpus_rows: list[dict[str, Any]],
    stage1_generations: dict[str, str],
    stage2_generations: dict[str, str],
    min_size: int = 4,
    max_size: int = 24,
    seed: int = 42,
) -> list[dict[str, Any]]:
    'Community-report rows: per-document graph → label propagation →\n    report prompts over the kept communities.'
    del seed  # deterministic: sorted orders everywhere; kept for API symmetry
    mentions = _verified_stage1_entities(corpus_rows, stage1_generations)
    node_info: dict[str, dict[str, str]] = {}  # doc:canonical -> info
    doc_nodes: dict[str, set[str]] = {}
    for m in mentions:
        for e in m["entities"]:
            key = _canonical_key(e["entity_name"])
            node_key = (m["doc_id"], key)
            info = node_info.setdefault(
                node_key,
                {
                    "entity_name": e["entity_name"],
                    "entity_type": e["entity_type"],
                    "description": str(e.get("description") or "").strip(),
                },
            )
            if not info["description"] and e.get("description"):
                info["description"] = str(e["description"]).strip()
            doc_nodes.setdefault(m["doc_id"], set()).add(key)

    doc_edges: dict[str, dict[tuple[str, str, str], str]] = {}
    for row in corpus_rows:
        if row.get("task_class") != _STAGE_CLASS[STAGE_RELATIONS]:
            continue
        response = stage2_generations.get(row["id"])
        if not response:
            continue
        payload = row["extra"]["extraction"]
        parsed = exm.parse_extraction_output(response)
        if not parsed or not parsed["relations"]:
            continue
        schema = ExtractionSchema.from_extra(payload)
        allowed = set(schema.relation_types) if schema else set()
        doc_id = row["extra"].get("doc_id") or row["extra"].get("source")
        provided = {
            _canonical_key(e["entity_name"]): e
            for e in payload.get("provided_entities") or []
        }
        for r in parsed["relations"]:
            rel = r["relation_type"]
            if allowed and rel not in allowed:
                continue
            src = _canonical_key(r["source_entity"])
            tgt = _canonical_key(r["target_entity"])
            if src not in provided or tgt not in provided:
                continue
            key = (src, rel, tgt)
            desc = str(r.get("description") or "").strip()
            edges = doc_edges.setdefault(doc_id, {})
            if key not in edges or (desc and not edges[key]):
                edges[key] = desc

    rows: list[dict[str, Any]] = []
    idx = 0
    for doc_id in sorted(set(doc_nodes) | set(doc_edges)):
        nodes = sorted(doc_nodes.get(doc_id, set()))
        triples = sorted(doc_edges.get(doc_id, {}))
        communities = detect_communities(
            nodes, [(s, t) for s, _rel, t in triples], min_size, max_size
        )
        for members in communities:
            member_set = set(members)
            entities = [
                {
                    "entity_name": node_info[(doc_id, k)]["entity_name"],
                    "entity_type": node_info[(doc_id, k)]["entity_type"],
                    "description": node_info[(doc_id, k)]["description"],
                }
                for k in members
            ]
            relations = [
                {
                    "source_entity": node_info[(doc_id, s)]["entity_name"],
                    "relation_type": rel,
                    "target_entity": node_info[(doc_id, t)]["entity_name"],
                    "description": doc_edges[doc_id].get((s, rel, t), ""),
                }
                for s, rel, t in triples
                if s in member_set and t in member_set
            ]
            idx += 1
            first_row = next(
                (r for r in corpus_rows
                 if (r["extra"].get("doc_id") or r["extra"].get("source")) == doc_id),
                None,
            )
            style = first_row["extra"]["style"] if first_row else "scientific"
            source = first_row["extra"].get("source") if first_row else doc_id
            rows.append(
                {
                    "id": f"ext-sum-c-{idx:06d}",
                    "task_class": "ragu_community_summary",
                    "prompt": build_community_summary_prompt(
                        entities, relations, LANGUAGE_VALUES["ru"]
                    ),
                    "extra": {
                        "style": style,
                        "source": source,
                        "doc_id": doc_id,
                        "summary": {
                            "kind": "community",
                            "entities": entities,
                            "relations": relations,
                        },
                    },
                }
            )
    return rows


def _load_generations(path: str) -> dict[str, str]:
    """Read ``{row_id, response}`` (or ``{id, response}``) JSONL records."""
    gens: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            row_id = rec.get("row_id") or rec.get("id")
            response = rec.get("response") or rec.get("completion") or ""
            if row_id and response:
                gens[str(row_id)] = str(response)
    return gens


def _write_rows(rows: list[dict[str, Any]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def _report(rows: list[dict[str, Any]], label: str, path: str) -> None:
    per_style: dict[str, int] = {}
    per_stage: dict[str, int] = {}
    per_lang: dict[str, int] = {}
    per_ontology: dict[str, int] = {}
    n_with_ref = 0
    for row in rows:
        style = row["extra"]["style"]
        lang = row["extra"].get("target_language") or "auto"
        per_style[style] = per_style.get(style, 0) + 1
        per_lang[lang] = per_lang.get(lang, 0) + 1
        if str(row.get("reference") or "").strip():
            n_with_ref += 1
        # extraction/summary rows carry a schema; QA rows do not.
        ext = row["extra"].get("extraction")
        if isinstance(ext, dict):
            stage = ext.get("stage") or "?"
            onto = ext.get("ontology") or "nerel"
            per_stage[stage] = per_stage.get(stage, 0) + 1
            per_ontology[onto] = per_ontology.get(onto, 0) + 1
    print(f"{label}: {len(rows)} rows -> {path}")
    print(f'per style')
    if per_stage:
        print(f'per stage')
    print(f'per target language')
    if per_ontology:
        print(f'per ontology')
    if n_with_ref:
        print(f'with reference')
    unknown = set(per_style) - set(STYLES)
    if unknown:
        print(f"  WARNING: unknown style strata: {sorted(unknown)}")


# ---------------------------------------------------------------------------
# CLI / CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Build the unlabeled GRPO-extraction corpus (deterministic).'
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser(
        "build",
        help="one-stage + stage-1 prompts from raw chunks / промпты из чанков",
    )
    p_build.add_argument(
        "--input-jsonl",
        action="append",
        help=(
            'JSONL with {text, style, lang, source, doc_id, ontology} records; repeatable.'
        ),
    )
    p_build.add_argument(
        "--corpus-dir",
        help='Directory tree <style>/<lang>/*.txt.',
    )
    p_build.add_argument(
        "--out", required=True, help='Output train JSONL.'
    )
    p_build.add_argument("--val-out", help='Optional frozen anchor JSONL.')
    p_build.add_argument("--val-per-style", type=int, default=8)
    p_build.add_argument("--seed", type=int, default=42)
    p_build.add_argument(
        "--schema-mode", choices=("mixed", "full", "subset"), default="mixed"
    )
    p_build.add_argument("--n-entity-types", type=int, default=10)
    p_build.add_argument(
        "--n-entity-types-max",
        type=int,
        default=None,
        help=(
            'Upper bound of the per-row schema size K (seeded uniform n..max; unset = fixed --n-entity-types).'
        ),
    )
    p_build.add_argument("--stage1-share", type=float, default=0.5)
    p_build.add_argument("--compact-variant-share", type=float, default=0.3)
    p_build.add_argument("--no-target-language-share", type=float, default=0.2)
    p_build.add_argument("--with-signatures-share", type=float, default=0.5)
    p_build.add_argument("--chunk-words", type=int, default=350)
    p_build.add_argument(
        "--chunk-tokens-min", type=int, default=None,
        help=(
            'Token-based chunking: minimum window in bge-m3 tokens (with --chunk-tokens-max replaces --chunk-words; the per-chunk window is seeded-sampled in [min, max]).'
        ),
    )
    p_build.add_argument(
        "--chunk-tokens-max", type=int, default=None,
        help='Token-based chunking: maximum window.'
    )
    p_build.add_argument(
        "--chunker-tokenizer", default="BAAI/bge-m3",
        help='Tokenizer id for token chunking.'
    )
    p_build.add_argument("--min-words", type=int, default=40)
    p_build.add_argument("--max-words", type=int, default=1200)
    p_build.add_argument("--min-punct-ratio", type=float, default=0.02)
    p_build.add_argument("--max-rows", type=int, default=None)
    p_build.add_argument("--per-style", type=int, default=None)

    p_s2 = sub.add_parser(
        "stage2",
        help="stage-2 prompts over verified stage-1 entities / стадия 2",
    )
    p_s2.add_argument("--corpus", required=True)
    p_s2.add_argument("--generations", required=True)
    p_s2.add_argument("--out", required=True)
    p_s2.add_argument("--min-entities", type=int, default=2)
    p_s2.add_argument("--seed", type=int, default=42)

    p_sum = sub.add_parser(
        "summaries",
        help=(
            "entity/relation/community summary rows over stage-1 + stage-2"
            " generations / строки саммаризации по генерациям стадий 1 и 2"
        ),
    )
    p_sum.add_argument("--corpus", required=True)
    p_sum.add_argument(
        "--stage1-generations",
        required=True,
        help='{row_id, response} JSONL over the stage1 rows.',
    )
    p_sum.add_argument(
        "--stage2-generations",
        required=True,
        help='{row_id, response} JSONL over the stage2 rows.',
    )
    p_sum.add_argument("--out", required=True)
    p_sum.add_argument("--min-descriptions", type=int, default=2)
    p_sum.add_argument("--max-descriptions", type=int, default=8)
    p_sum.add_argument("--min-community-size", type=int, default=4)
    p_sum.add_argument("--max-community-nodes", type=int, default=24)
    p_sum.add_argument("--seed", type=int, default=42)

    p_qa = sub.add_parser(
        "qa",
        help=(
            "reference-bearing QA rows from FineWeb2-HQ-QA records"
            " / строки вопросо-ответа с референсом из FineWeb2-HQ-QA"
        ),
    )
    p_qa.add_argument(
        "--qa-jsonl",
        action="append",
        required=True,
        help=(
            '{question, answer, text, source, url, doc_id, lang} JSONL; repeatable.'
        ),
    )
    p_qa.add_argument("--out", required=True)
    p_qa.add_argument(
        "--with-context-share",
        type=float,
        default=0.5,
        help='P(open-book row): the source text is prepended.',
    )
    p_qa.add_argument("--min-question-words", type=int, default=2)
    p_qa.add_argument("--max-question-words", type=int, default=80)
    p_qa.add_argument("--min-answer-words", type=int, default=2)
    p_qa.add_argument("--max-answer-words", type=int, default=400)
    p_qa.add_argument("--compact-variant-share", type=float, default=0.3)
    p_qa.add_argument("--max-rows", type=int, default=None)
    p_qa.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    if args.command == "build":
        if not args.input_jsonl and not args.corpus_dir:
            parser.error('build requires --input-jsonl and/or --corpus-dir.')
        records = list(
            iter_input_records(args.input_jsonl, args.corpus_dir)
        )
        rows = build_rows(
            records,
            seed=args.seed,
            schema_mode=args.schema_mode,
            n_entity_types=args.n_entity_types,
            n_entity_types_max=args.n_entity_types_max,
            stage1_share=args.stage1_share,
            compact_variant_share=args.compact_variant_share,
            no_target_language_share=args.no_target_language_share,
            with_signatures_share=args.with_signatures_share,
            chunk_words=args.chunk_words,
            chunk_tokens_min=args.chunk_tokens_min,
            chunk_tokens_max=args.chunk_tokens_max,
            chunker_tokenizer=args.chunker_tokenizer,
            min_words=args.min_words,
            max_words=args.max_words,
            min_punct_ratio=args.min_punct_ratio,
            max_rows=args.max_rows,
            per_style=args.per_style,
        )
        if args.val_out:
            rows, val_rows = split_val(rows, args.val_per_style, args.seed)
            _write_rows(val_rows, args.val_out)
            _report(val_rows, "val", args.val_out)
        _write_rows(rows, args.out)
        _report(rows, "train", args.out)
        return 0

    if args.command == "summaries":
        corpus_rows = _load_corpus_rows(args.corpus)
        stage1_gens = _load_generations(args.stage1_generations)
        stage2_gens = _load_generations(args.stage2_generations)
        rows = []
        rows.extend(
            build_entity_summary_rows(
                corpus_rows, stage1_gens,
                min_descriptions=args.min_descriptions,
                max_descriptions=args.max_descriptions,
                seed=args.seed,
            )
        )
        rows.extend(
            build_relation_summary_rows(
                corpus_rows, stage2_gens,
                min_descriptions=args.min_descriptions,
                max_descriptions=args.max_descriptions,
                seed=args.seed,
            )
        )
        rows.extend(
            build_community_summary_rows(
                corpus_rows, stage1_gens, stage2_gens,
                min_size=args.min_community_size,
                max_size=args.max_community_nodes,
                seed=args.seed,
            )
        )
        _write_rows(rows, args.out)
        _report(rows, "summaries", args.out)
        return 0

    if args.command == "qa":
        qa_records: list[dict[str, Any]] = []
        for path in args.qa_jsonl:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        qa_records.append(json.loads(line))
        rows = build_qa_rows(
            qa_records,
            seed=args.seed,
            with_context_share=args.with_context_share,
            min_question_words=args.min_question_words,
            max_question_words=args.max_question_words,
            min_answer_words=args.min_answer_words,
            max_answer_words=args.max_answer_words,
            compact_variant_share=args.compact_variant_share,
            max_rows=args.max_rows,
        )
        _write_rows(rows, args.out)
        _report(rows, "qa", args.out)
        return 0

    # stage2
    corpus_rows = _load_corpus_rows(args.corpus)
    generations = _load_generations(args.generations)
    rows = build_stage2_rows(
        corpus_rows,
        generations,
        min_entities=args.min_entities,
        seed=args.seed,
    )
    _write_rows(rows, args.out)
    _report(rows, "stage2", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
