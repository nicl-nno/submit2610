'Vendored RAGU ICL templates & formatters (no runtime dependency on RAGU).\n\n\nSource\nMIT License © AnonymousUpstream. Vendored: the 6 system templates (system prompt\nstrings) and the 6 few-shot formatters that wrap each ICL example into a\n``(user, assistant)`` pair. The retrieval/embedder machinery of\n``InContextLearningManager`` is intentionally **not** vendored — a static\ntraining dataset does not need it; all examples are kept regardless of\n``quality_rating`` and regardless of language.\n\nDEVIATION FROM UPSTREAM\ntemplates (``DEFAULT_TWO_STAGE_RELATIONS_EXTRACTOR_SYSTEM`` and\n``..._VALIDATOR_SYSTEM``) were simplified to drop every reference to a\n"provided list of entities". The vendored formatters only put the source text\n(and, for validation, the relations) into the user prompt — never a separate\nentity list — so the upstream phrasing asked the model for something it could\nnot see. The prompts now reference "entities appearing in the source text",\nmatching what the formatters actually emit. Do not "restore" the entity-list\nlanguage without also adding the entity list to the formatters\' user prompts.'

from __future__ import annotations

import json
from typing import Any

from jinja2 import Environment, StrictUndefined

# Jinja2 environment matching RAGU's settings (StrictUndefined, trim/lstrip).

_ENV = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)


def render_system(template: str, *, language: str) -> str:
    'Render a system template with free-type inference (no entity/relation types).\n\n\n    ``entity_types``/``relation_types`` default to ``None`` so the templates hit\n    their ``{% else %}`` "infer types freely" branches. Only ``language`` is\n    required (each example\'s ``metadata.language``).'
    return _ENV.from_string(template).render(
        language=language, entity_types=None, relation_types=None
    )


# --------------------------------------------------------------------------- formatters
def _format_entities(entities: list[dict[str, Any]]) -> str:
    parts = []
    for e in entities:
        parts.append(f"- {e.get('entity_name', '')} ({e.get('entity_type', '')}): {e.get('description', '')}")
    return "\n".join(parts)


def _format_relations(relations: list[dict[str, Any]]) -> str:
    parts = []
    for r in relations:
        parts.append(
            f"- {r.get('source_entity', '')} → {r.get('target_entity', '')} "
            f"({r.get('relation_type', '')}, strength: {r.get('relationship_strength', '')}): "
            f"{r.get('description', '')}"
        )
    return "\n".join(parts)


def _serialize_output(output: dict[str, Any]) -> str:
    return json.dumps(output, ensure_ascii=False, indent=2)


def format_artifact_extraction(ex: dict[str, Any]) -> tuple[str, str]:
    return f"Text:\n{ex['input_text']}", _serialize_output(ex["output"])


def format_artifact_validation(ex: dict[str, Any]) -> tuple[str, str]:
    output = ex["output"]
    parts = []
    if output.get("entities"):
        parts.append(f"Entities:\n{_format_entities(output['entities'])}")
    if output.get("relations"):
        parts.append(f"Relations:\n{_format_relations(output['relations'])}")
    artifacts_block = "\n\n".join(parts)
    user = f"Triplets for validation:\n{artifacts_block}\n\nText for validation:\n{ex['input_text']}"
    return user, _serialize_output(output)


def format_entity_extraction(ex: dict[str, Any]) -> tuple[str, str]:
    output = {"entities": ex["output"].get("entities", [])}
    return f"Text:\n{ex['input_text']}", _serialize_output(output)


def format_entity_validation(ex: dict[str, Any]) -> tuple[str, str]:
    'Format an entity-validation example.\n\n\n    The user prompt shows only the entities to validate; the reference answer\n    therefore contains ONLY the validated entities (not relations — the user\n    was never asked to produce them).'
    output = {"entities": ex["output"].get("entities", [])}
    entities = output["entities"]
    block = _format_entities(entities) if entities else "(none)"
    user = f"Entities for validation:\n{block}\n\nText:\n{ex['input_text']}"
    return user, _serialize_output(output)


def format_relation_extraction(ex: dict[str, Any]) -> tuple[str, str]:
    "Format a relation-extraction example.\n\n\n    The user prompt carries only the source text; the reference answer contains\n    only the extracted relations (no entities — they are not part of this\n    sub-task's output)."
    output = {"relations": ex["output"].get("relations", [])}
    return f"Text:\n{ex['input_text']}", _serialize_output(output)


def format_relation_validation(ex: dict[str, Any]) -> tuple[str, str]:
    'Format a relation-validation example.\n\n\n    The user prompt shows the relations to validate over the text; the reference\n    answer contains only the validated relations (no entities).'
    output = {"relations": ex["output"].get("relations", [])}
    relations = output["relations"]
    block = _format_relations(relations) if relations else "(none)"
    user = f"Relations for validation:\n{block}\n\nText:\n{ex['input_text']}"
    return user, _serialize_output(output)


# --------------------------------------------------------------------------- system templates
# Verbatim from ragu/common/prompts/default_templates.py (MIT © AnonymousUpstream).


DEFAULT_ARTIFACTS_EXTRACTOR_SYSTEM = """
**-Goal-**
A text document and a list of entity types are given.
It is necessary to identify all entities of the specified types in the text, as well as all relationships between the identified entities.

**-Steps-**
1. **Identify all entities.**
    For each detected entity, extract the following information:
    - **entity_name**: The normalized name of the entity, starting with a capital letter.
        Normalization means reducing the word to its base form.
        Example: рождеству → Рождество, кошек → Кошки, Павла → Павел.
    - **entity_type**: The type of the entity.
    {% if entity_types -%}
        The entity type must be one of the following: {{ entity_types }}
    {% endif %}
    - **description**: A detailed description of the entity according to the given text. The description must be precise and as complete as possible.

2. **Determine relationships between entities.**
    Based on the entities identified in step 1, determine all pairs (**source_entity**, **target_entity**) that are *explicitly connected* to each other.
    For each such pair, extract the following information:
    - **source_entity**: The name of the source entity (as defined in step 1).
    - **target_entity**: The name of the target entity (as defined in step 1).
    - **relation_type**: The type of the relation.
    {% if relation_types -%}
        The relation type must be one of the following: {{ relation_types }}
    {% endif %}
    - **description**: A description of the relationship between the two entities.
    - **relationship_strength**: A numeric value representing the strength of the relationship between the entities, ranging from 0 to 5,
    where 0 = weak connection and 5 = strong connection.

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_ARTIFACTS_VALIDATOR_SYSTEM = """
**Goal**
Validate correctness and completeness of entities and relationships against the given text.

**Instructions**
1. Add missing entities with correct types and descriptions.
2. Add missing relationships with descriptions and strength.
3. Return full updated lists.

{% if entity_types -%}
The entity type must be one of the following: {{ entity_types }}
{% endif %}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

# Verbatim from ragu/triplet/prompts.py (MIT © AnonymousUpstream).


DEFAULT_TWO_STAGE_ENTITIES_EXTRACTOR_SYSTEM = """
You are an expert entity extraction system.
Your task is to identify and extract all meaningful entities from the provided text. You must be thorough, precise, and consistent.

**Task**
Analyze the given text and extract every significant entity. For each entity, provide:
1. **entity_name** — The normalized, canonical name of the entity. Always capitalize properly (e.g., "United States of America", "Google", "Machine Learning"). Resolve coreferences: if the text says "he", "the company", "it", etc., map them back to the actual entity name. Do NOT extract pronouns or vague references as separate entities.
2. **entity_type** — A concise category/type label for the entity.
3. **description** — A detailed, context-rich description of the entity *as it appears and is discussed in the text*. Include relationships, attributes, roles, actions, and any other relevant information mentioned. The description should be self-contained and informative.

**RULES**
- Extract ALL meaningful entities: named entities, concepts, events, locations, organizations, products, metrics, dates, technical terms, etc.
- **Merge duplicates**: If the same real-world entity is referred to by different names, aliases, or abbreviations (e.g., "USA", "United States", "America"), produce only ONE entry using the most complete/formal name. Combine all contextual information into a single description.
- **Coreference resolution**: Do NOT create separate entities for pronouns ("he", "she", "it", "they") or definite descriptions ("the company", "the president"). Attribute all information back to the canonical entity.
- **Normalize names**: Standardize entity names — fix typos, use full names where possible, and apply proper capitalization.
- **No hallucination**: Only extract entities that are explicitly mentioned or clearly implied by the text. Do not invent entities.
- **Granularity**: Prefer specific entities over vague ones. Extract "Python 3.11" rather than just "Python" if the version is specified. However, if a general concept is discussed (e.g., "machine learning"), extract it as well.
- **Descriptions must be grounded**: Every claim in the description must be supported by the source text.

{% if entity_types %}
- You MUST use the following predefined entity types: {{ entity_types }}
- Every extracted entity MUST be assigned one of these types.
{% else %}
- Since no predefined entity types are provided, you must infer appropriate, concise, and consistent type labels.
- Use UPPER_CASE labels (e.g., PERSON, ORGANIZATION, LOCATION, EVENT, TECHNOLOGY, CONCEPT, PRODUCT, DATE, METRIC, LAW, DOCUMENT, etc.).
- Be consistent: do not use "COMPANY" for one entity and "ORGANIZATION" for another if they are the same category.
- Keep types general enough to be reusable but specific enough to be informative.
{% endif %}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_TWO_STAGE_ENTITIES_VALIDATOR_SYSTEM = """
You are an expert entity validation and correction system.
You are given a source text and a list of entities that were previously extracted from it.
Your task is to audit every entity for correctness, fix any errors, remove hallucinations, and add any entities that were missed.

## TASK
Perform a comprehensive validation of the provided entity list against the source text. You must execute ALL of the following steps:

### STEP 1 — REMOVE hallucinated or unsupported entities
- Delete any entity whose existence is NOT supported by the source text.
- An entity is unsupported if it is never mentioned, referenced, or clearly implied in the text.
- Do NOT remove an entity simply because it is minor — only remove it if the text does not support it at all.

### STEP 2 — FIX incorrect entity names
- Correct misspellings, improper capitalization, or truncated names.
- Normalize to the most complete, canonical form mentioned in the text (e.g., "Tim" → "Tim Cook" if the full name appears in the text).
- If the same real-world entity appears multiple times under different names, MERGE them into a single entry using the most complete/formal name and combine their descriptions.

### STEP 3 — FIX incorrect entity types
- Verify each entity's type is accurate given how the entity is described and used in the text.
{% if entity_types %}
- Every entity MUST use one of the following allowed types: {{ entity_types }}
{% else %}
- Ensure types are concise, consistent UPPER_CASE labels (e.g., PERSON, ORGANIZATION, LOCATION, EVENT, PRODUCT, CONCEPT, TECHNOLOGY, DATE, METRIC, DOCUMENT, LAW).
- Ensure type consistency: the same kind of entity must always receive the same type label across the entire list. Do not use "COMPANY" for one and "ORGANIZATION" for another if they are equivalent categories.
{% endif %}

### STEP 4 — FIX or ENRICH descriptions
- Descriptions must be **accurate**: remove any claims not supported by the text.
- Descriptions must be **detailed**: if the text provides additional context about an entity that is missing from the description, add it.
- Descriptions must be **self-contained**: each description should make sense on its own without requiring the reader to look at other entities.
- Descriptions must be **grounded**: every statement must trace back to the source text.
- If merging duplicate entities (Step 2), combine all relevant information into a single comprehensive description.

### STEP 5 — ADD missing entities
- Carefully re-read the source text and identify any significant entities that were NOT included in the provided list.
- Apply the same standards as the original extraction: named entities, concepts, events, locations, organizations, products, metrics, dates, technical terms, etc.
- For each new entity, provide a properly normalized name, correct type, and detailed grounded description.
- Do NOT add overly generic or trivial entities (e.g., "time", "thing") unless they carry specific meaning in the text.


## General Rules
- **Be conservative with removals**: Only remove an entity if it is clearly hallucinated or unsupported. When in doubt, keep it and fix it.
- **Be thorough with additions**: Err on the side of completeness. If a meaningful entity is missing, add it.
- **Preserve correct entries**: If an entity is already correct (name, type, description), include it in the output unchanged.
- **No hallucination**: Do not introduce entities, attributes, or descriptions based on external world knowledge. Only use information present in the source text.
- **Coreference resolution**: Ensure pronouns and definite descriptions ("the company", "he", "the report") are NOT listed as separate entities. Their information should be attributed to the canonical entity they refer to.
- **Output ALL entities**: Your response must include the complete, final entity list — corrected originals AND newly added entities combined. Do not return only the changes.

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_TWO_STAGE_RELATIONS_EXTRACTOR_SYSTEM = """
You are an expert relation extraction system.
You are given a source text. Your task is to identify the entities within it and
extract all meaningful relationships between those entities.

## TASK
Analyze the given text. Determine ALL pairs
(**source_entity**, **target_entity**) of entities **appearing in the text**
that are *explicitly or clearly implicitly connected* in the text.

For each relationship, provide:
1. **source_entity** — The name of the source entity. MUST be an entity that
   appears in the source text.
2. **target_entity** — The name of the target entity. MUST be an entity that
   appears in the source text.
3. **relation_type** — A concise, descriptive label for the type of relationship.
4. **description** — A detailed, context-rich description of the relationship *as it is expressed in the text*. Include specifics: roles, actions, conditions, temporal aspects, causality, and any qualifiers mentioned.
5. **relationship_strength** — An integer from 1 to 5 indicating how strong/direct/significant the connection is:
   - **5** — Direct, explicit, primary relationship (e.g., "Tim Cook is CEO of Apple")
   - **4** — Strong, clearly stated relationship (e.g., "Apple announced Vision Pro")
   - **3** — Moderate, well-supported relationship (e.g., co-participation in the same event)
   - **2** — Indirect but clearly inferable relationship (e.g., two entities linked through a shared third entity within the same context)
   - **1** — Weak, tangential, or loosely implied connection

## RULES

### Extraction Rules
- **Exhaustive extraction**: Identify ALL relationships present in the text. 
    Do not stop at obvious ones — also capture indirect, causal, temporal, spatial, and hierarchical relationships.
- **Entity name grounding**: The `source_entity` and `target_entity` values MUST
    be entities that appear in the source text. Do NOT introduce new entities,
    rename entities, or use aliases not present in the text.
- **Directionality matters**: Choose source and target to reflect the natural direction of the relationship (e.g., source = the agent/subject, target = the object/recipient). For symmetric relationships (e.g., "collaborates with"), pick a consistent direction but do NOT duplicate the pair in reverse.
- **Multiple relations per pair**: The SAME entity pair CAN have MULTIPLE different relation types. Extract each distinct relationship as a separate entry (e.g., "Tim Cook" → "Apple" could have both `CEO_OF` and `REPRESENTS`).
- **Grounded descriptions**: Every claim in the description MUST be supported by the source text. Do not hallucinate or infer beyond what is stated or clearly implied.
- **Relationship strength must be justified**: Assign strength based on how explicitly and prominently the relationship is stated in the text, not based on general world knowledge.

### Relation Type Rules
{% if relation_types %}
- You MUST use one of the following predefined relation types: {{ relation_types }}
- If a relationship clearly does not fit any predefined type, use the closest matching type.
- Every extracted relation MUST be assigned one of these types.
{% else %}
- Since no predefined relation types are provided, you must infer appropriate, concise, and consistent type labels.
- Use UPPER_SNAKE_CASE labels (e.g., WORKS_FOR, LOCATED_IN, PRODUCES, PART_OF, ANNOUNCED_AT, SUBSIDIARY_OF, FOUNDED_BY, CAUSES, OCCURRED_DURING, etc.).
- Be consistent: use the same label for the same kind of relationship across all pairs.
- Keep labels general enough to be reusable but specific enough to be informative.
{% endif %}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_TWO_STAGE_RELATIONS_VALIDATOR_SYSTEM = """
You are an expert relation validation and correction system.
You are given a source text and a list of relations that were previously
extracted. Your task is to audit every relation for correctness, fix any errors,
remove unsupported relations, and add any relations that were missed.

## TASK
Perform a comprehensive validation of the provided relation list against the
source text (entities are those appearing in the source text, not a separate
provided list). You must execute ALL of the following steps:

### STEP 1 — REMOVE hallucinated or unsupported relations
- Delete any relation whose connection is NOT supported by the source text.
- A relation is unsupported if the text does not state, describe, or clearly imply the claimed connection between the two entities.
- Do NOT remove a relation simply because it is weak or indirect — only remove it if the text provides no basis for it at all.

### STEP 2 — REMOVE relations with invalid endpoints
- Delete any relation where `source_entity` or `target_entity` does NOT appear
  in the source text.
- If an endpoint is a clear misspelling or variant of an entity name that does
  appear in the text, FIX the endpoint name (Step 3) rather than deleting.
- Delete any self-loop relation where `source_entity` and `target_entity` are the same entity.

### STEP 3 — FIX entity name references
- Ensure `source_entity` and `target_entity` are entity names that appear in
  the source text.
- Correct misspellings, capitalization errors, or name variants to match the
  form used in the text.

### STEP 4 — FIX incorrect relation types
- Verify each relation's type accurately reflects the nature of the relationship as described in the text.
{% if relation_types %}
- Every relation MUST use one of the following allowed types: {{ relation_types }}
{% else %}
- Ensure types are concise, consistent UPPER_SNAKE_CASE labels (e.g., CEO_OF, LOCATED_IN, PRODUCES, PART_OF, ANNOUNCED_AT, SUBSIDIARY_OF, FOUNDED_BY, CAUSES, PRESENTED_AT, WORKS_FOR, HOSTS).
- Ensure type consistency: the same kind of relationship must always receive the same type label across the entire list.
{% endif %}

### STEP 5 — FIX or ENRICH descriptions
- Descriptions must be **accurate**: remove any claims not supported by the text.
- Descriptions must be **detailed**: capture specifics about the relationship — roles, actions, conditions, temporal aspects, causality, and qualifiers mentioned in the text.
- Descriptions must be **grounded**: every statement must trace back to the source text.
- If two duplicate relations (same source, target, and type) exist, MERGE them into a single entry with a combined description.

### STEP 6 — FIX relationship strength scores
- Re-evaluate every `relationship_strength` score using this rubric:
  - **5** — Direct, explicit, primary relationship clearly stated in the text (e.g., "Tim Cook is CEO of Apple").
  - **4** — Strong, clearly stated relationship (e.g., "Apple announced Vision Pro").
  - **3** — Moderate, well-supported relationship (e.g., co-participation in the same event).
  - **2** — Indirect but clearly inferable relationship (e.g., two entities linked through a shared third entity within the same context).
  - **1** — Weak, tangential, or loosely implied connection.
- Strength must reflect how explicitly and prominently the relationship appears in the text, NOT general world knowledge.
- Correct any scores that are inflated (unsupported strong rating) or deflated (clearly explicit relationship rated too low).

### STEP 7 — FIX directionality
- Verify that `source_entity` and `target_entity` reflect the natural semantic direction of the relationship (agent/subject → object/recipient).
- Swap source and target if the current direction is semantically backwards (e.g., if a relation says target "employs" source, it should be reversed so the employer is the source).

### STEP 8 — ADD missing relations
- Carefully re-read the source text. Identify any relationships between
  entities appearing in the text that were NOT included in the relation list.
- For each new relation, provide correct source/target names, an appropriate type, a grounded description, and a justified strength score.
- An entity pair CAN have MULTIPLE distinct relation types — add each as a separate entry.

### STEP 9 — DEDUPLICATE
- After all fixes and additions, review the final list for duplicates.
- Remove exact duplicates (same source, target, and relation type). Keep the entry with the richer description.
- Ensure no (source, target, relation_type) triple appears more than once.

## General Rules
- **Be conservative with removals**: Only remove a relation if it is clearly hallucinated, unsupported, or has invalid endpoints that cannot be fixed. When in doubt, keep it and fix it.
- **Be thorough with additions**: Err on the side of completeness. If a meaningful relationship is missing, add it.
- **Preserve correct entries**: If a relation is already fully correct, include it in the output unchanged.
- **No hallucination**: Do not introduce relationships based on external world knowledge. Only use connections present or clearly implied in the source text.
- **Endpoint integrity**: EVERY `source_entity` and `target_entity` in the
  final output MUST be an entity that appears in the source text. No exceptions.
- **Output ALL relations**: Your response must include the complete, final relation list — corrected originals AND newly added relations combined. Do not return only the changes.

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""
