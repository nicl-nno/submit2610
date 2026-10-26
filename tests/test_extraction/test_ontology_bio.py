'Offline tests of the vendored NEREL-BIO ontology (no network, no GPU).\n\n\nCovers: the vocabulary registry, the multi-line (alternative-constraint)\nsignature semantics of the brat config, the per-vocabulary literal sets,\nschema serialization round-trips (legacy single-dict form vs the\nalternative-lines list) and the vocabulary-aware reward dispatch.'

from __future__ import annotations

import random

from llm_training.extraction._nerel_bio_tables import (
    BIO_ENTITY_TYPES,
    BIO_RELATION_TYPES,
)
from llm_training.extraction.ontology import (
    NEREL_BIO_LITERAL_ENTITY_TYPES,
    NEREL_BIO_ONTOLOGY,
    NEREL_ONTOLOGY,
    ONTOLOGIES,
    ExtractionSchema,
    get_ontology,
    signature_lines,
)
from llm_training.reward.extraction import extraction_schema_score


# --------------------------------------------------------------------- registry
def test_registry_resolves_both_vocabularies() -> None:
    assert sorted(ONTOLOGIES) == ["nerel", "nerel_bio"]
    assert get_ontology(None) is NEREL_ONTOLOGY
    assert get_ontology("nerel") is NEREL_ONTOLOGY
    assert get_ontology("nerel_bio") is NEREL_BIO_ONTOLOGY
    assert NEREL_BIO_ONTOLOGY.name == "nerel_bio"


def test_registry_rejects_unknown_loudly() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown extraction ontology"):
        get_ontology("nerel_bioo")  # a typo must never silently fall back


def test_bio_type_counts_match_annotation_conf() -> None:
    # 45 entity types (29 general + 16 biomedical), 66 relation types —
    # the exact counts of nerel-bio-v1.0/annotation.conf.
    assert len(BIO_ENTITY_TYPES) == 45
    assert len(BIO_RELATION_TYPES) == 66
    assert NEREL_BIO_ONTOLOGY.knows_entity_type("DISO")
    assert NEREL_BIO_ONTOLOGY.knows_entity_type("STATE_OR_PROVINCE")
    # The NEREL twin names are NOT in the BIO vocabulary — the schemes are
    # deliberately separate, never merged.
    assert not NEREL_BIO_ONTOLOGY.knows_entity_type("DISEASE")
    assert not NEREL_BIO_ONTOLOGY.knows_entity_type("STATE_OR_PROV")
    assert not NEREL_ONTOLOGY.knows_entity_type("DISO")


# ------------------------------------------------- multi-line signature checks
def test_allows_any_line_semantics() -> None:
    bio = NEREL_BIO_ONTOLOGY
    # Line 1: PERSON|LIVB|PROFESSION|NATIONALITY|DISO|INJURY_POISONING -> MEDPROC|CHEM
    assert bio.allows("TREATED_USING", "PERSON", "MEDPROC")
    assert bio.allows("TREATED_USING", "DISO", "CHEM")
    # Line 2: PHYS|MENTALPROC -> CHEM
    assert bio.allows("TREATED_USING", "PHYS", "CHEM")
    # Cross-line endpoint pairs satisfy NO single line (union would pass!).
    assert not bio.allows("TREATED_USING", "PHYS", "MEDPROC")
    assert not bio.allows("TREATED_USING", "PERSON", "ANATOMY")


def test_allows_located_in_three_lines() -> None:
    bio = NEREL_BIO_ONTOLOGY
    assert bio.allows("LOCATED_IN", "PERSON", "CITY")  # general line
    assert bio.allows("LOCATED_IN", "DISO", "ANATOMY")  # disease -> anatomy
    assert bio.allows("LOCATED_IN", "DISO", "COUNTRY")  # disease -> geography
    assert not bio.allows("LOCATED_IN", "PERSON", "ANATOMY")


def test_allows_bio_ancestors() -> None:
    bio = NEREL_BIO_ONTOLOGY
    # STATE_OR_PROVINCE is-a LOCATION through the mirrored BIO hierarchy.
    assert bio.ancestors("STATE_OR_PROVINCE") == frozenset(
        {"STATE_OR_PROVINCE", "LOCATION"}
    )
    assert bio.ancestors("PERCENT") == frozenset({"PERCENT", "NUMBER"})
    assert bio.allows("HEADQUARTERED_IN", "ORGANIZATION", "STATE_OR_PROVINCE")


def test_allows_single_line_form_unchanged() -> None:
    # The NEREL single-constraint form keeps its semantics.
    assert NEREL_ONTOLOGY.allows("WORKPLACE", "PERSON", "ORGANIZATION")
    assert not NEREL_ONTOLOGY.allows("WORKPLACE", "ORGANIZATION", "PERSON")


def test_relation_lines_normalized() -> None:
    bio = NEREL_BIO_ONTOLOGY
    entry = bio._relations["TREATED_USING"]
    lines = bio._relation_lines(entry)
    assert len(lines) == 2
    assert lines[0][0] == (
        "PERSON", "LIVB", "PROFESSION", "NATIONALITY", "DISO", "INJURY_POISONING"
    )
    # An empty single constraint still yields exactly one line.
    nerel_empty = NEREL_ONTOLOGY._relations["PART_OF"]
    assert NEREL_ONTOLOGY._relation_lines(nerel_empty) == (((), ()),)


def test_signature_lines_helper() -> None:
    single = {"domain": ["PERSON"], "range": ["AGE"]}
    assert signature_lines(single) == [single]
    two = [single, {"domain": [], "range": []}]
    assert signature_lines(two) is two


# ------------------------------------------------------------------- literals
def test_literal_sets_per_vocabulary() -> None:
    assert "LANGUAGE" in NEREL_ONTOLOGY.literal_types
    # LANGUAGE is absent from the BIO scheme entirely.
    assert "LANGUAGE" not in NEREL_BIO_ONTOLOGY.literal_types
    assert NEREL_BIO_LITERAL_ENTITY_TYPES == frozenset(
        {"DATE", "TIME", "AGE", "MONEY", "NUMBER", "PERCENT", "ORDINAL"}
    )
    # DISO is nominal in BIO: it must NOT be skipped by normalization.
    assert "DISO" not in NEREL_BIO_ONTOLOGY.literal_types
    assert "DISEASE" not in NEREL_ONTOLOGY.literal_types


# --------------------------------------------------------------------- subsets
def test_bio_subset_deterministic_and_usable() -> None:
    rng1 = random.Random(7)
    rng2 = random.Random(7)
    a = NEREL_BIO_ONTOLOGY.subset(rng1, n_entity_types=10)
    b = NEREL_BIO_ONTOLOGY.subset(rng2, n_entity_types=10)
    assert a == b
    assert a.ontology == "nerel_bio"
    assert 2 <= len(a.entity_types) <= 10
    # Every kept relation is usable: SOME line's domain and range each
    # accept at least one chosen type.
    for rel in a.relation_types:
        entry = NEREL_BIO_ONTOLOGY._relations[rel]
        assert any(
            (not domain or any(NEREL_BIO_ONTOLOGY.ancestors(t) & set(domain) for t in a.entity_types))
            and (not rng_types or any(NEREL_BIO_ONTOLOGY.ancestors(t) & set(rng_types) for t in a.entity_types))
            for domain, rng_types in NEREL_BIO_ONTOLOGY._relation_lines(entry)
        ), rel


def test_bio_subset_serialization_forms() -> None:
    schema = NEREL_BIO_ONTOLOGY.full_schema()
    assert schema.ontology == "nerel_bio"
    # Single-line relations keep the legacy dict form.
    assert isinstance(schema.signatures["WORKS_AS"], dict)
    # Multi-line relations serialize as the alternative-lines list.
    assert isinstance(schema.signatures["TREATED_USING"], list)
    assert len(schema.signatures["TREATED_USING"]) == 2


def test_schema_round_trip_bio_and_legacy() -> None:
    schema = NEREL_BIO_ONTOLOGY.subset(random.Random(11), n_entity_types=8)
    payload = schema.to_extra()
    assert payload["ontology"] == "nerel_bio"
    rebuilt = ExtractionSchema.from_extra(payload)
    assert rebuilt == schema
    # Legacy payload (no ontology key, dict signature) still resolves to NEREL.
    legacy = {
        "entity_types": ["PERSON", "AGE"],
        "relation_types": ["AGE_IS"],
        "signatures": {"AGE_IS": {"domain": ["PERSON"], "range": ["AGE"]}},
    }
    rebuilt_legacy = ExtractionSchema.from_extra(legacy)
    assert rebuilt_legacy is not None
    assert rebuilt_legacy.ontology == "nerel"
    # A list-form signature parses back into the same list.
    bio_payload = {
        "ontology": "nerel_bio",
        "entity_types": ["PHYS", "CHEM", "MEDPROC", "PERSON"],
        "relation_types": ["TREATED_USING"],
        "signatures": {
            "TREATED_USING": [
                {"domain": ["PERSON", "DISO"], "range": ["MEDPROC", "CHEM"]},
                {"domain": ["PHYS"], "range": ["CHEM"]},
            ]
        },
    }
    rebuilt_bio = ExtractionSchema.from_extra(bio_payload)
    assert rebuilt_bio is not None
    assert rebuilt_bio.signatures["TREATED_USING"] == [
        {"domain": ["PERSON", "DISO"], "range": ["MEDPROC", "CHEM"]},
        {"domain": ["PHYS"], "range": ["CHEM"]},
    ]


# ------------------------------------------------------------------- rendering
def test_render_multi_line_signatures() -> None:
    rendered = NEREL_BIO_ONTOLOGY.render_relation_types(
        ["TREATED_USING"], with_signatures=True
    )
    assert rendered.startswith("TREATED_USING [")
    assert "-> MEDPROC|CHEM]" in rendered
    assert "[PHYS|MENTALPROC -> CHEM]" in rendered
    # Single-line NEREL rendering is byte-identical to the legacy format.
    nerel = NEREL_ONTOLOGY.render_relation_types(
        ["WORKPLACE"], with_signatures=True
    )
    assert nerel.startswith("WORKPLACE [PERSON -> ORGANIZATION|FACILITY]")


# ------------------------------------------------------- vocabulary-aware reward
def _parsed(entities: list[tuple[str, str]], relations: list[tuple[str, str, str, str]]):
    return {
        "entities": [
            {"entity_name": n, "entity_type": t, "description": ""}
            for n, t in entities
        ],
        "relations": [
            {
                "source_entity": s,
                "target_entity": o,
                "relation_type": r,
                "description": "",
            }
            for s, r, o, _ in relations
        ],
    }


def test_reward_schema_score_dispatches_by_vocabulary() -> None:
    payload = {
        "ontology": "nerel_bio",
        "entity_types": ["PERSON", "PHYS", "MEDPROC", "CHEM"],
        "relation_types": ["TREATED_USING"],
        "signatures": {
            "TREATED_USING": [
                {"domain": ["PERSON"], "range": ["MEDPROC", "CHEM"]},
                {"domain": ["PHYS"], "range": ["CHEM"]},
            ]
        },
    }
    schema = ExtractionSchema.from_extra(payload)
    assert schema is not None
    parsed = _parsed(
        [("Иванов", "PERSON"), ("физиология", "PHYS"), ("процедура", "MEDPROC")],
        [
            ("Иванов", "TREATED_USING", "процедура", "ok"),      # line 1 ✓
            ("физиология", "TREATED_USING", "процедура", "cross"),  # no line ✗
        ],
    )
    score = extraction_schema_score(parsed, schema)
    assert score is not None
    # Membership share = 1.0; signature share = 0.5 → combined 0.75.
    assert abs(score - 0.75) < 1e-9


def test_reward_schema_score_nerel_bio_membership() -> None:
    # A DISO entity is a VALID member under nerel_bio but NOT under nerel —
    # the same response scores differently per the row's vocabulary.
    parsed = _parsed([("грипп", "DISO")], [])
    bio_schema = ExtractionSchema.from_extra(
        {
            "ontology": "nerel_bio",
            "entity_types": ["DISO"],
            "relation_types": [],
            "signatures": {},
        }
    )
    nerel_schema = ExtractionSchema.from_extra(
        {
            "ontology": "nerel",
            "entity_types": ["DISEASE"],
            "relation_types": [],
            "signatures": {},
        }
    )
    assert bio_schema is not None and nerel_schema is not None
    # Membership share: 1.0 under BIO, 0.0 under NEREL; the signature share
    # is awarded neutrally (no relations) → 1.0 / 0.5 respectively.
    assert extraction_schema_score(parsed, bio_schema) == 1.0
    assert extraction_schema_score(parsed, nerel_schema) == 0.5
