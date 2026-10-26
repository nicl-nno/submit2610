'Offline tests of the vendored NEREL ontology (no network, no GPU).'

from __future__ import annotations

import random

from llm_training.extraction.ontology import (
    ExtractionSchema,
    LITERAL_ENTITY_TYPES,
    NEREL_ONTOLOGY,
)


def test_type_counts_match_upstream() -> None:
    assert len(NEREL_ONTOLOGY._entities) == 29
    assert len(NEREL_ONTOLOGY._relations) == 49


def test_ancestor_closure() -> None:
    assert NEREL_ONTOLOGY.ancestors("CITY") == frozenset({"CITY", "LOCATION"})
    assert NEREL_ONTOLOGY.ancestors("PERCENT") == frozenset({"PERCENT", "NUMBER"})
    assert NEREL_ONTOLOGY.ancestors("PERSON") == frozenset({"PERSON"})
    assert NEREL_ONTOLOGY.ancestors("NOT_A_TYPE") == frozenset()


def test_allows_respects_signatures() -> None:
    assert NEREL_ONTOLOGY.allows("WORKPLACE", "PERSON", "ORGANIZATION")
    # Reversed direction violates the signature.
    assert not NEREL_ONTOLOGY.allows("WORKPLACE", "ORGANIZATION", "PERSON")
    # A subtype satisfies a constraint accepting its ancestor.
    assert NEREL_ONTOLOGY.allows("HEADQUARTERED_IN", "ORGANIZATION", "CITY")
    # Unknown endpoint types reject.
    assert not NEREL_ONTOLOGY.allows("WORKPLACE", "PERSON", "WAT")
    # Empty domain/range accept any known pair.
    assert NEREL_ONTOLOGY.allows("PART_OF", "DATE", "PERSON")
    # Unknown relation rejects.
    assert not NEREL_ONTOLOGY.allows("NOT_A_RELATION", "PERSON", "PERSON")


def test_render_entity_types() -> None:
    rendered = NEREL_ONTOLOGY.render_entity_types(["PERSON", "CITY"])
    assert rendered.startswith("PERSON (a human being")
    assert "CITY (a city or town)" in rendered


def test_render_relation_types_signatures() -> None:
    plain = NEREL_ONTOLOGY.render_relation_types(["WORKPLACE"])
    assert plain.startswith("WORKPLACE (")
    signed = NEREL_ONTOLOGY.render_relation_types(["WORKPLACE"], with_signatures=True)
    assert "WORKPLACE [PERSON -> ORGANIZATION|FACILITY]" in signed
    # Empty sides render as ANY.
    part = NEREL_ONTOLOGY.render_relation_types(["PART_OF"], with_signatures=True)
    assert "PART_OF [ANY -> ANY]" in part


def test_subset_deterministic_and_usable() -> None:
    a = NEREL_ONTOLOGY.subset(random.Random(42), n_entity_types=8)
    b = NEREL_ONTOLOGY.subset(random.Random(42), n_entity_types=8)
    assert a == b
    assert 2 <= len(a.entity_types) <= 8
    assert len(a.relation_types) >= 1
    # Usability invariant: every kept relation is applicable inside the subset
    # (domain and range each accept ≥ one chosen type through the closure).
    for rel in a.relation_types:
        sig = a.signatures[rel]
        domain = sig["domain"]
        rng = sig["range"]
        if domain:
            assert any(
                NEREL_ONTOLOGY.ancestors(t) & set(domain) for t in a.entity_types
            ), rel
        if rng:
            assert any(
                NEREL_ONTOLOGY.ancestors(t) & set(rng) for t in a.entity_types
            ), rel


def test_subset_relation_cap() -> None:
    schema = NEREL_ONTOLOGY.subset(
        random.Random(7), n_entity_types=15, n_relation_types=5
    )
    assert len(schema.relation_types) == 5
    assert set(schema.signatures) == set(schema.relation_types)


def test_full_schema() -> None:
    full = NEREL_ONTOLOGY.full_schema()
    assert len(full.entity_types) == 29
    assert len(full.relation_types) == 49
    assert "WORKPLACE" in full.signatures


def test_schema_extra_round_trip() -> None:
    schema = NEREL_ONTOLOGY.subset(random.Random(1), n_entity_types=6)
    restored = ExtractionSchema.from_extra(schema.to_extra())
    assert restored == schema
    # Malformed payloads degrade to None, never raise.
    assert ExtractionSchema.from_extra(None) is None
    assert ExtractionSchema.from_extra({"entity_types": "PERSON"}) is None


def test_literal_entity_types_present() -> None:
    assert {
        "DATE", "TIME", "AGE", "MONEY", "NUMBER", "PERCENT", "ORDINAL"
    } <= LITERAL_ENTITY_TYPES
