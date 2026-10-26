'Offline tests of the extraction prompt builders (no network, no GPU).'

from __future__ import annotations

import random

from llm_training.extraction.ontology import NEREL_ONTOLOGY
from llm_training.extraction.prompts import (
    LANGUAGE_VALUES,
    STAGE_ONE,
    STAGE_RELATIONS,
    build_extraction_prompt,
    build_user_message,
    render_system,
)

_SCHEMA = NEREL_ONTOLOGY.subset(random.Random(3), n_entity_types=6)


def test_one_stage_prompt_shape() -> None:
    messages = build_extraction_prompt(
        STAGE_ONE, "Some source text.", _SCHEMA, language="Russian"
    )
    assert [m["role"] for m in messages] == ["system", "user"]
    system, user = messages[0]["content"], messages[1]["content"]
    # The promised types come from the SAME schema the reward will enforce.
    assert _SCHEMA.entity_types[0] in system
    assert _SCHEMA.relation_types[0] in system
    # The JSON envelope is spelled out inline (the docsem pattern).
    assert '"entities"' in system and '"relations"' in system
    assert "Return ONLY the JSON object" in system
    assert user == "Text:\nSome source text."


def test_stage2_prompt_carries_entity_list() -> None:
    entities = [
        {"entity_name": "Павел Дуров", "entity_type": "PERSON"},
        {"entity_name": "ВКонтакте", "entity_type": "ORGANIZATION"},
    ]
    messages = build_extraction_prompt(
        STAGE_RELATIONS,
        "Текст про Дурова.",
        _SCHEMA,
        language="Russian",
        provided_entities=entities,
    )
    system, user = messages[0]["content"], messages[1]["content"]
    assert "PROVIDED ENTITIES" in system
    assert "Entities extracted from the text:" in user
    assert "- Павел Дуров (PERSON)" in user
    assert "- ВКонтакте (ORGANIZATION)" in user
    assert user.endswith("Text:\nТекст про Дурова.")
    # The stage-2 envelope asks for relations only.
    assert '"relations"' in system
    assert '"entities": [' not in system.split("Output JSON schema")[-1]


def test_stage2_requires_entities() -> None:
    try:
        build_user_message(STAGE_RELATIONS, "text")
    except ValueError:
        return
    raise AssertionError("stage2 without provided_entities must raise")


def test_variants_and_signatures() -> None:
    base = render_system(STAGE_ONE, _SCHEMA, "Russian", variant=0)
    compact = render_system(STAGE_ONE, _SCHEMA, "Russian", variant=1)
    assert base != compact
    assert "Extract named entities" in compact
    signed = render_system(
        STAGE_ONE, _SCHEMA, "Russian", variant=0, with_signatures=True
    )
    assert " [" in signed  # [DOMAIN -> RANGE] blocks rendered


def test_language_values() -> None:
    assert LANGUAGE_VALUES["ru"] == "Russian"
    assert LANGUAGE_VALUES["en"] == "English"


def test_unknown_stage_raises() -> None:
    try:
        render_system("stage9", _SCHEMA, "Russian")
    except ValueError:
        return
    raise AssertionError("unknown stage must raise")
