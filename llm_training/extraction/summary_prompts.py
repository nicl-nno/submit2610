'Prompt builders for the RAGU-style summarization tasks (GRPO-extraction).\n\n\nThree consolidation stages mirroring RAGU\'s graph-assembly summarizers\n(:class:`ragu.graph.artifacts_summarizer.EntitySummarizer` /\n``RelationSummarizer`` and :class:`ragu.graph.community_summarizer.\nCommunitySummarizer`):\n\n* ``entity_summary`` — merge the MULTIPLE primary descriptions of one\n  entity (extracted from different chunks) into one concise, consistent\n  description (RAGU\'s ``entity_summarizer``);\n* ``relation_summary`` — the same for one (subject, relation, object) pair\n  (RAGU\'s ``relation_summarizer``\n* ``community_summary`` — a structured report over a local graph community:\n  title, summary, impact rating 0–10 with justification, 5–10 findings\n  (RAGU\'s ``community_report`` / ``CommunityReportModel``);\n\nTwo deliberate deviations from upstream, the SAME two as the extraction\nprompts\n\n1. The JSON output schema is SPELLED OUT INLINE in every system prompt\n   (upstream relies on server-side structured decoding).\n   structured decoding).\n2. A compact variant per stage (``variant=1``) for instruction variability.\n\nThe consolidation contract (prompt ↔ reward)\n:mod:`llm_training.reward.summarization` against the SAME payload the row\ncarries in ``extra["summary"]`` — the two cannot drift apart.'

from __future__ import annotations

from typing import Any

from jinja2 import Environment, StrictUndefined

# Single source of truth for the target-language prompt values

from llm_training.extraction.prompts import LANGUAGE_VALUES

# Jinja2 environment exactly as in RAGU (StrictUndefined, trim/lstrip) —
# the same rendering contract the extraction prompts use.
_ENV = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)

#: Stage identifiers used across the corpus builder, the reward and anchors.

SUMMARY_ENTITY = "entity_summary"
SUMMARY_RELATION = "relation_summary"
SUMMARY_COMMUNITY = "community_summary"
SUMMARY_STAGES: tuple[str, ...] = (
    SUMMARY_ENTITY,
    SUMMARY_RELATION,
    SUMMARY_COMMUNITY,
)

#: Target-language prompt values — re-exported from the single source of
#: truth (:data:`llm_training.extraction.prompts.LANGUAGE_VALUES`); see the
#: import at the top of the module.



# --- System templates (variant 0)
ENTITY_SUMMARY_SYSTEM = """
You are an expert knowledge-graph consolidation system.
You are given ONE entity and MULTIPLE primary descriptions of it that were
extracted from different fragments of a larger document. Your task is to
merge them into ONE concise, consistent entity description.

## TASK
Produce a single description that:
1. States what the entity IS (its most general established sense).
2. Merges ALL complementary facts from the primary descriptions — nothing
   essential may be lost.
3. Resolves contradictions in favour of the majority/most specific phrasing.
4. Introduces NO new facts: every claim MUST be supported by at least one
   primary description.

## RULES
- Keep the entity name exactly as given; do not rename it.
- Length: 1–3 sentences. No enumerations of the inputs, no meta-commentary
  ("the descriptions say..."), no placeholders.
- Grounded merging only: do not hallucinate, do not infer beyond the inputs.

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

RELATION_SUMMARY_SYSTEM = """
You are an expert knowledge-graph consolidation system.
You are given ONE directed relation (subject — relation — object) and
MULTIPLE primary descriptions of it that were extracted from different
fragments of a larger document. Your task is to merge them into ONE
concise, consistent relation description.

## TASK
Produce a single description that:
1. States the relationship between the subject and the object in one clear
   sentence.
2. Merges ALL complementary facts (roles, conditions, temporality,
   causality, strength qualifiers) from the primary descriptions.
3. Resolves contradictions in favour of the majority/most specific phrasing.
4. Introduces NO new facts: every claim MUST be supported by at least one
   primary description.

## RULES
- Keep the subject and object names exactly as given; do not rename them.
- Length: 1–3 sentences. No meta-commentary, no placeholders.
- Grounded merging only: do not hallucinate, do not infer beyond the inputs.

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

COMMUNITY_SUMMARY_SYSTEM = """
You are an expert knowledge-graph analysis system.
You are given a LOCAL COMMUNITY of a knowledge graph: its entities (with
descriptions) and the relations between them. Your task is to write a
structured report about this community.

## TASK
Produce a report with:
1. **title** — a short informative title of the community.
2. **summary** — 2–4 sentences: what this community is about, its core
   theme, the key entities and how they connect.
3. **rating** — an impact rating from 0 to 10 (integer): how central and
   informational this community is for the document.
4. **rating_explanation** — 1–2 sentences justifying the rating.
5. **findings** — 5 to 10 key findings; each finding has a short **summary**
   and a detailed **explanation** grounded in the provided entities and
   relations.

## RULES
- Every entity name and every claim MUST come from the provided community —
  do not invent entities, relations, or facts.
- Grounded writing only: no hallucination, no external knowledge.

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

# --- Compact variants (variant 1)
COMPACT_ENTITY_SUMMARY_SYSTEM = """
Merge the multiple primary descriptions of one entity into a single
consistent description: what the entity is, all complementary facts kept,
no new facts, no meta-commentary, 1-3 sentences.
Entity name: {{ entity_name }}.
Answer in {{ language }}. Return ONLY valid JSON matching the schema.
"""

COMPACT_RELATION_SUMMARY_SYSTEM = """
Merge the multiple primary descriptions of one relation into a single
consistent description: one clear sentence about how the subject relates to
the object, all complementary facts kept, no new facts, 1-3 sentences.
Subject: {{ subject_name }}. Object: {{ object_name }}.
Answer in {{ language }}. Return ONLY valid JSON matching the schema.
"""

COMPACT_COMMUNITY_SUMMARY_SYSTEM = """
Write a structured report about the given knowledge-graph community:
title, 2-4 sentence summary, impact rating 0-10 with a 1-2 sentence
justification, and 5-10 findings (short summary + detailed explanation
each). Every entity and claim MUST come from the provided community.
Answer in {{ language }}. Return ONLY valid JSON matching the schema.
"""

#: Inline JSON schemas per stage (the envelope the policy must learn from
#: the prompt — the docsem/extract deviation from server-side decoding).

_SUMMARY_JSON_SCHEMAS: dict[str, str] = {
    SUMMARY_ENTITY: '{"entity_name": str, "description": str}',
    SUMMARY_RELATION: (
        '{"subject_name": str, "object_name": str, "description": str}'
    ),
    SUMMARY_COMMUNITY: (
        '{"title": str, "summary": str, "rating": int, '
        '"rating_explanation": str, "findings": [{"summary": str, '
        '"explanation": str}]}'
    ),
}


#: Public alias of the inline envelope schemas (the agent-suite builder and
#: other consumers outside this module read THEM, not private names).


SUMMARY_JSON_SCHEMAS = _SUMMARY_JSON_SCHEMAS


def _render(stage: str, template: str, language: str, **ctx: Any) -> str:
    rendered = _ENV.from_string(template).render(language=language, **ctx)
    schema_block = (
        "\nOutput JSON schema (return exactly this envelope):\n"
        f"{_SUMMARY_JSON_SCHEMAS[stage]}\nReturn ONLY the JSON object, "
        "no markdown fences, no extra text."
    )
    return rendered.rstrip() + "\n" + schema_block


def build_entity_summary_prompt(
    entity_name: str,
    entity_type: str,
    descriptions: list[str],
    language: str,
    variant: int = 0,
) -> list[dict[str, str]]:
    'Build the entity-description consolidation prompt.\n\n    the VERIFIED stage-1 outputs grouped by canonical name across chunks;\n    the prompt lists them verbatim and numbered (merge targets).'
    if not descriptions:
        raise ValueError(
            'entity summary requires ≥1 primary description.'
        )
    template = (
        ENTITY_SUMMARY_SYSTEM if variant == 0
        else COMPACT_ENTITY_SUMMARY_SYSTEM
    )
    system = _render(
        SUMMARY_ENTITY,
        template,
        language,
        entity_name=entity_name,
    )
    lines = "\n".join(f"{i}. {d.strip()}" for i, d in enumerate(descriptions, 1))
    type_part = f" ({entity_type})" if entity_type else ""
    user = (
        f"Entity name: {entity_name}{type_part}\n"
        f"Primary descriptions:\n{lines}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_relation_summary_prompt(
    subject: str,
    subject_type: str,
    relation_type: str,
    obj: str,
    object_type: str,
    descriptions: list[str],
    language: str,
    variant: int = 0,
) -> list[dict[str, str]]:
    'Build the relation-description consolidation prompt.'
    if not descriptions:
        raise ValueError(
            'relation summary requires ≥1 primary description.'
        )
    template = (
        RELATION_SUMMARY_SYSTEM if variant == 0
        else COMPACT_RELATION_SUMMARY_SYSTEM
    )
    system = _render(
        SUMMARY_RELATION,
        template,
        language,
        subject_name=subject,
        object_name=obj,
    )
    lines = "\n".join(f"{i}. {d.strip()}" for i, d in enumerate(descriptions, 1))
    user = (
        f"Subject: {subject} ({subject_type})\n"
        f"Relation: {relation_type}\n"
        f"Object: {obj} ({object_type})\n"
        f"Primary descriptions:\n{lines}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_community_summary_prompt(
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    language: str,
    variant: int = 0,
) -> list[dict[str, str]]:
    'Build the community-report prompt.\n\n    ``{entity_name, entity_type, description}``; ``relations`` items:\n    ``{source_entity, relation_type, target_entity, description}`` — the\n    SAME item shapes the extraction line already produces.'
    if not entities:
        raise ValueError(
            'community summary requires ≥1 entity.'
        )
    template = (
        COMMUNITY_SUMMARY_SYSTEM if variant == 0
        else COMPACT_COMMUNITY_SUMMARY_SYSTEM
    )
    system = _render(SUMMARY_COMMUNITY, template, language)
    entity_lines = []
    for e in entities:
        name = str(e.get("entity_name", "")).strip()
        etype = str(e.get("entity_type", "")).strip()
        desc = str(e.get("description", "")).strip()
        head = f"- {name}" + (f" ({etype})" if etype else "")
        entity_lines.append(f"{head}: {desc}" if desc else head)
    relation_lines = []
    for r in relations:
        src = str(r.get("source_entity", "")).strip()
        tgt = str(r.get("target_entity", "")).strip()
        rel = str(r.get("relation_type", "")).strip()
        desc = str(r.get("description", "")).strip()
        if not src or not tgt:
            continue
        tail = f", description: {desc}" if desc else ""
        relation_lines.append(f"- {src} -> {tgt} ({rel}){tail}")
    user = (
        "Entities:\n" + ("\n".join(entity_lines) or "(none)") + "\n\n"
        "Relations:\n" + ("\n".join(relation_lines) or "(none)")
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
