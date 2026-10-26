'Assemble the frozen agent suite (three families, retrieval mode).\n\n\nStage 2 of ``specifications/2026-10-17-agent-suite-glm53-validation.md``:\n150 tasks per family (extraction / summary / qa) selected from the\nANNOTATED val pool; the source text NEVER enters the prompt (the model must\nfind it via the ``searchdocs`` tool over the frozen index). Gold rides in\n``gold`` per family: ``pseudo_gold`` (extraction), judge ``reference``\n(summary), the native answer (qa). A family with fewer gold-bearing\ncandidates than ``--per-family`` is a HARD refusal (the wait-for-full-data\ndecision): a partial suite would silently shift the measuring instrument.\n\n\nUsage\n\n    python scripts/build_agent_suite.py         --extraction-jsonl work/agent_suite/val_annotated.jsonl         --summary-jsonl work/agent_suite/val_summaries_annotated.jsonl         --qa-jsonl work/agent_suite/val_qa.jsonl         --index-digest-file work/agent_suite/index/digest.txt         --out-dir work/agent_suite'

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_training.extraction.ontology import ExtractionSchema
from llm_training.extraction.prompts import (
    JSON_SCHEMAS,
    LANGUAGE_VALUES,
    STAGE_ENTITIES,
    STAGE_ONE,
    render_system,
)
from llm_training.extraction.summary_prompts import (
    SUMMARY_COMMUNITY,
    SUMMARY_ENTITY,
    SUMMARY_JSON_SCHEMAS,
    SUMMARY_RELATION,
)

#: Suite row format version (bump on incompatible changes).

FORMAT_VERSION: int = 1

#: The searchdocs tool schema pinned into every suite row (the `tools` field
#: of the SGR-catalog format; arguments validated by ``schema_check``).


SEARCHDOCS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "searchdocs",
        "description": (
            "Semantic search over the local document base: returns the "
            "top_k most relevant document chunks (ids + texts)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (8-256 chars).",
                    "minLength": 8,
                    "maxLength": 256,
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to return (1-10).",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query", "top_k"],
        },
    },
}

#: Shared system template (the SGR protocol + the searchdocs contract + the
#: finalanswertool envelope; {payload} is the per-family answer contract).

# : finalanswertool; {payload}
SYSTEM_TEMPLATE = """You are a research agent working with a LOCAL DOCUMENT BASE (a large set of text chunks). Your only information source is the searchdocs tool; never invent document content.

Tool call format:
<tool_call>
{{"name": "searchdocs", "arguments": {{"query": "<search query>", "top_k": 5}}}}
</tool_call>

Protocol:
1. Form 1-3 search queries from the task and call searchdocs (top_k between 1 and 10).
2. Use ONLY the returned chunks; if the chunks do not cover the task, refine the query and search again.
3. Emit the FINAL ANSWER as:
<tool_call>
{{"name": "finalanswertool", "arguments": {{"reasoning": "<why the task is complete and how the answer was verified>", "completed_steps": ["<completed step>", "..."], "answer": <PAYLOAD>, "status": "completed"}}}}
</tool_call>

The `answer` payload for THIS task: {payload}

Answer in {language}."""

_EXTRA_PAYLOAD_ONE = JSON_SCHEMAS[STAGE_ONE]
_EXTRA_PAYLOAD_ENTITIES = JSON_SCHEMAS[STAGE_ENTITIES]
_QA_PAYLOAD = (
    "the answer text itself — a plain, self-contained answer "
    "(no JSON, no markdown fences)"
)

#: Row-kind -> envelope schema. The builder writes ``kind`` as
#: ``"entity"``/``"relation"``/``"community"`` (see
#: ``build_extraction_corpus.py``), while the envelope schemas are keyed by
#: the ``entity_summary``/``relation_summary``/``community_summary``
#: stage constants — THIS table is the bridge.

# : ``"entity"``/``"relation"``/``"community"``
# : ``entity_summary``/...
_KIND_PAYLOADS: dict[str, str] = {
    "entity": SUMMARY_JSON_SCHEMAS[SUMMARY_ENTITY],
    "relation": SUMMARY_JSON_SCHEMAS[SUMMARY_RELATION],
    "community": SUMMARY_JSON_SCHEMAS[SUMMARY_COMMUNITY],
}
_SUMMARY_PAYLOADS = _KIND_PAYLOADS

_SENTENCE_END = re.compile(r"[.!?…](?:\s|$|\n)")


def seed_fragment(text: str, limit: int = 220) -> str:
    'Deterministic excerpt anchor: first sentence(s) up to ``limit`` chars.'
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    best = 0
    for match in _SENTENCE_END.finditer(cut):
        best = match.end()
    if best >= 40:
        return cut[:best].strip()
    return cut.rstrip() + "…"


def load_rows(path: str) -> list[dict[str, Any]]:
    'Read corpus rows (JSONL, non-empty lines).'
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: str) -> str:
    'sha256 of a file (hex).'
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def language_value(row: dict[str, Any]) -> str:
    "The row's answer language value (``LANGUAGE_VALUES``; RU fallback)."
    extra = row.get("extra") or {}
    lang = extra.get("target_language") or extra.get("lang")
    return LANGUAGE_VALUES.get(lang, LANGUAGE_VALUES["ru"])


def _final_answer_payload(stage: str) -> str:
    "The extraction answer contract for the row's stage."
    if stage == STAGE_ONE:
        return _EXTRA_PAYLOAD_ONE
    if stage == STAGE_ENTITIES:
        return _EXTRA_PAYLOAD_ENTITIES
    raise ValueError(f'unsupported extraction stage {stage!r}')


def build_extraction_row(
    row: dict[str, Any], fragment_chars: int, index_digest: str
) -> dict[str, Any]:
    'One retrieval-mode extraction task from an annotated stage-1/one row.'
    extra = row["extra"]
    payload = extra["extraction"]
    schema = ExtractionSchema.from_extra(payload)
    if schema is None:
        raise ValueError(f"{row.get('id')}: extraction schema is empty")
    stage = str(payload.get("stage") or STAGE_ONE)
    text = str(payload.get("text") or "")
    fragment = seed_fragment(text, fragment_chars)
    system = render_system(
        stage,
        schema,
        language_value(row),
        with_signatures=bool(payload.get("signatures")),
    )
    user = (
        "Task: find the document fragment that contains or continues the "
        "excerpt below, then run the extraction over the FULL found "
        "fragment per the schema from the system message.\n\n"
        f'Excerpt:\n"""\n{fragment}\n"""\n\n'
        "Steps: call searchdocs with a query derived from the excerpt; "
        "check that the found chunk really continues it (matching wording); "
        "extract over that chunk.\n"
        "Keep each entity/relation description concise — no longer than "
        "~15 words."
    )
    return {
        "id": f"suite-ext-{row['id']}",
        "family": "extraction",
        "source_row_id": row["id"],
        "task_class": row.get("task_class"),
        "prompt": [
            {"role": "system", "content": system + "\n\n" + SYSTEM_TEMPLATE.format(
                payload=_final_answer_payload(stage), language=language_value(row)
            )},
            {"role": "user", "content": user},
        ],
        "tools": [SEARCHDOCS_TOOL],
        "gold": {
            "pseudo_gold": payload["pseudo_gold"],
            "text": text,
        },
        "extra": {
            "family": "extraction",
            "source_row_id": row["id"],
            "style": extra.get("style"),
            "source": extra.get("source"),
            "target_language": extra.get("target_language"),
            "seed_fragment": fragment,
            "index_digest": index_digest,
        },
    }


def _summary_subjects(extra_summary: dict[str, Any]) -> str:
    "A short retrieval anchor string for a summary row's kind."
    kind = extra_summary.get("kind")
    if kind == "entity":
        return f"{extra_summary.get('entity_name', '')} ({extra_summary.get('entity_type', '')})"
    if kind == "relation":
        return (
            f"{extra_summary.get('subject', '')} "
            f"({extra_summary.get('subject_type', '')}) -> "
            f"{extra_summary.get('relation_type', '')} -> "
            f"{extra_summary.get('object', '')} "
            f"({extra_summary.get('object_type', '')})"
        )
    entities = extra_summary.get("entities") or []
    names: list[str] = []
    for item in entities:
        if isinstance(item, dict):
            name = item.get("name") or item.get("entity_name")
        else:
            name = str(item)
        if name:
            names.append(str(name))
    return ", ".join(names[:6])


def build_summary_row(
    row: dict[str, Any], index_digest: str
) -> dict[str, Any]:
    'One retrieval-mode consolidation task from an annotated summary row.\n\n    The consolidation material (descriptions / findings) stays in ``gold`` —\n    the model must re-derive it from the found chunks.'
    extra = row["extra"]
    payload = extra["summary"]
    kind = str(payload.get("kind") or "entity")
    if kind not in _SUMMARY_PAYLOADS:
        raise ValueError(f"{row.get('id')}: unknown summary kind {kind!r}")
    anchor = _summary_subjects(payload)
    system = SYSTEM_TEMPLATE.format(
        payload=_SUMMARY_PAYLOADS[kind], language=language_value(row)
    )
    user = (
        "Task: find in the document base the fragments about "
        f"{anchor} and write the consolidated summary from the found "
        "material following the schema from the system message.\n\n"
        "Steps: call searchdocs with queries derived from the anchor; "
        "collect the relevant chunks; consolidate."
    )
    return {
        "id": f"suite-sum-{row['id']}",
        "family": "summary",
        "source_row_id": row["id"],
        "task_class": row.get("task_class"),
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "tools": [SEARCHDOCS_TOOL],
        "gold": {
            "reference": payload["reference"],
            "material": payload,
        },
        "extra": {
            "family": "summary",
            "source_row_id": row["id"],
            "style": extra.get("style"),
            "source": extra.get("source"),
            "target_language": extra.get("target_language"),
            "index_digest": index_digest,
        },
    }


def build_qa_row(row: dict[str, Any], index_digest: str) -> dict[str, Any]:
    'One retrieval-mode QA task (forced closed-book: find, then answer).\n\n    originally open-book rows — retrieval is the point of the instrument.'
    qa = (row.get("extra") or {}).get("qa") or {}
    question = str(qa.get("question") or "").strip()
    if not question:
        raise ValueError(f"{row.get('id')}: qa question is empty")
    system = SYSTEM_TEMPLATE.format(
        payload=_QA_PAYLOAD, language=language_value(row)
    )
    user = (
        "Task: find in the document base the fragment that answers the "
        "question, then answer it.\n\n"
        f"Question: {question}"
    )
    return {
        "id": f"suite-qa-{row['id']}",
        "family": "qa",
        "source_row_id": row["id"],
        "task_class": row.get("task_class"),
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "tools": [SEARCHDOCS_TOOL],
        "gold": {
            "reference": row["reference"],
            "question": question,
        },
        "extra": {
            "family": "qa",
            "source_row_id": row["id"],
            "style": (row.get("extra") or {}).get("style"),
            "source": (row.get("extra") or {}).get("source"),
            "target_language": (row.get("extra") or {}).get("target_language"),
            "index_digest": index_digest,
        },
    }


def select_family(
    candidates: list[dict[str, Any]], per_family: int, seed: int, family: str
) -> list[dict[str, Any]]:
    'Deterministic per-family sample (sorted ids, seeded RNG).'
    ordered = sorted(candidates, key=lambda row: str(row.get("id")))
    if len(ordered) < per_family:
        raise SystemExit(
            f'family {family!r}: {len(ordered)} gold-bearing candidates < --per-family {per_family}'
        )
    return random.Random(f"{seed}:{family}").sample(ordered, per_family)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Assemble the frozen agent suite (3 families x 150, retrieval mode; stage 2).'
        )
    )
    parser.add_argument(
        "--extraction-jsonl", required=True,
        help='ANNOTATED stage-1/one val rows.',
    )
    parser.add_argument(
        "--summary-jsonl", required=True,
        help='ANNOTATED summary rows.',
    )
    parser.add_argument(
        "--qa-jsonl", required=True,
        help='QA rows with native references.',
    )
    parser.add_argument("--per-family", type=int, default=150)
    parser.add_argument(
        "--max-gold-entities", type=int, default=None,
        help=(
            "OPTIONAL instrument boundary: exclude extraction rows whose pseudo-gold carries more entities (the reference output would not fit a single response turn). Default: OFF — the full entity density is in scope (2026-10-18 operator decision; the cost driver was the teacher's verbose descriptions, fixed by the compact-description contract, not by dropping rows)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--index-digest-file", required=True,
        help='digest.txt of the frozen index.',
    )
    parser.add_argument("--fragment-chars", type=int, default=220)
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    index_digest = (
        Path(args.index_digest_file).read_text(encoding="utf-8").strip()
    )
    if not index_digest.startswith("sha256:"):
        raise SystemExit(
            f'{args.index_digest_file}: not a sha256 digest line.\n{args.index_digest_file}'
        )

    extraction_rows = [
        row
        for row in load_rows(args.extraction_jsonl)
        if row.get("task_class")
        in {"ragu_artifact_extraction", "ragu_entity_extraction"}
        and isinstance((row.get("extra") or {}).get("extraction"), dict)
        and isinstance(
            row["extra"]["extraction"].get("pseudo_gold"), dict
        )
        and (
            not args.max_gold_entities
            or len(row["extra"]["extraction"]["pseudo_gold"].get("entities") or [])
            <= args.max_gold_entities
        )
    ]
    summary_rows = [
        row
        for row in load_rows(args.summary_jsonl)
        if isinstance((row.get("extra") or {}).get("summary"), dict)
        and isinstance(row["extra"]["summary"].get("reference"), dict)
    ]
    qa_rows = [
        row
        for row in load_rows(args.qa_jsonl)
        if row.get("task_class") == "fineweb_qa"
        and str(row.get("reference") or "").strip()
    ]

    builders = (
        (
            "extraction",
            extraction_rows,
            lambda row: build_extraction_row(
                row, args.fragment_chars, index_digest
            ),
        ),
        ("summary", summary_rows, lambda row: build_summary_row(row, index_digest)),
        ("qa", qa_rows, lambda row: build_qa_row(row, index_digest)),
    )
    suite_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for family, candidates, builder in builders:
        picked = select_family(candidates, args.per_family, args.seed, family)
        built = [builder(row) for row in picked]
        suite_rows.extend(built)
        counts[family] = len(built)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suite_path = out_dir / "suite.jsonl"
    with suite_path.open("w", encoding="utf-8") as fh:
        for row in suite_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "per_family": args.per_family,
        "counts": counts,
        "total": len(suite_rows),
        "inputs": {
            args.extraction_jsonl: sha256_file(args.extraction_jsonl),
            args.summary_jsonl: sha256_file(args.summary_jsonl),
            args.qa_jsonl: sha256_file(args.qa_jsonl),
        },
        "index_digest": index_digest,
        "tool": "searchdocs",
        "fragment_chars": args.fragment_chars,
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=1),
        encoding="utf-8",
    )
    print(f"OK: suite -> {suite_path} ({counts}) | meta -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
