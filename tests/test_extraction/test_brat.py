'Offline tests of the brat .ann parser (no network, no GPU).'

from __future__ import annotations

from llm_training.extraction.brat import parse_brat_ann, parse_brat_file


REAL_SAMPLE = """T1\tNUMBER 135 138\t194
T2\tNUMBER 175 177\t65
T3\tORGANIZATION 1027 1030\tООН
T4\tDISO 1204 1218\tсахарный диабет
T5\tDISO 1204 1212\tсахарный
T6\tDATE 1220 1235\tв 2013 году
R1\tABBREVIATION Arg1:T7 Arg2:T3
T7\tORGANIZATION 1400 1420\tВсемирная организация здравоохранения
R2\tMEDICAL_CONDITION Arg1:T3 Arg2:T4
R3\tMEDICAL_CONDITION Arg1:T3 Arg2:T4
#1\tAnnotatorNotes T4\tdouble-check the span
E1\tPLACEHOLDER Arg1:T4
A1\tCertainty T4
R4\tDANGLING Arg1:T3 Arg2:T99
R5\tMALFORMED-NO-ARGS"""


def test_parses_entities_with_dedup() -> None:
    doc = parse_brat_ann(REAL_SAMPLE)
    # 7 T-lines parsed (one is a nested DISO inside another DISO).
    assert doc.n_mentions == 7
    pairs = {(e["entity_name"], e["entity_type"]) for e in doc.entities}
    assert ("ООН", "ORGANIZATION") in pairs
    assert ("сахарный диабет", "DISO") in pairs
    # Nested mentions stay separate list items (offset-free semantics).
    assert ("сахарный", "DISO") in pairs
    # No duplicate items after dedup.
    assert len(doc.entities) == len({(e["entity_name"], e["entity_type"]) for e in doc.entities})


def test_parses_relations_resolving_t_ids() -> None:
    doc = parse_brat_ann(REAL_SAMPLE)
    triples = {
        (r["source_entity"], r["relation_type"], r["target_entity"])
        for r in doc.relations
    }
    assert ("ООН", "MEDICAL_CONDITION", "сахарный диабет") in triples
    # R3 duplicates R2 -> collapsed.
    assert len([r for r in doc.relations if r["relation_type"] == "MEDICAL_CONDITION"]) == 1
    # R1 resolves through the forward-declared T7 (declaration order free).
    assert ("Всемирная организация здравоохранения", "ABBREVIATION", "ООН") in triples


def test_skips_notes_events_attributes_and_dangling() -> None:
    doc = parse_brat_ann(REAL_SAMPLE)
    assert doc.skipped.get("#") == 1
    assert doc.skipped.get("E") == 1
    assert doc.skipped.get("A") == 1
    # R4 references a missing T99 -> dangling, dropped from relations.
    assert doc.skipped.get("R_dangling") == 1
    assert all(r["relation_type"] != "DANGLING" for r in doc.relations)
    # R5 lacks Arg1/Arg2 -> malformed.
    assert doc.skipped.get("R_malformed") == 1


def test_fragmented_span_text_is_normalized() -> None:
    # In brat the TEXT field carries the FULL mention (fragments joined);
    # only the offsets are fragmented. Whitespace collapses to single spaces.
    doc = parse_brat_ann("T1\tDATE 0 5;7 12\tв 2013  году")
    assert doc.entities[0]["entity_name"] == "в 2013 году"


def test_empty_and_garbage_input_is_safe() -> None:
    doc = parse_brat_ann("")
    assert doc.entities == [] and doc.relations == [] and doc.n_mentions == 0
    doc2 = parse_brat_ann("random text without tabs\n\nT1\tTOOSHORT\t")
    # n_mentions counts RAW T-lines (the malformed one has no offsets and
    # lands in no entity item).
    assert doc2.n_mentions == 1
    assert doc2.entities == []


def test_parse_brat_file_roundtrip(tmp_path) -> None:
    path = tmp_path / "doc.ann"
    path.write_text(REAL_SAMPLE, encoding="utf-8")
    doc = parse_brat_file(path)
    assert doc.n_mentions == 7
