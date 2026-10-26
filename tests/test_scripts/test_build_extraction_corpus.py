'Offline tests of the extraction-corpus builder script (no network, no GPU).'

from __future__ import annotations

import json

from llm_training.extraction._nerel_bio_tables import BIO_ENTITY_TYPES
from llm_training.extraction.ontology import ExtractionSchema
from scripts.build_extraction_corpus import (
    build_community_summary_rows,
    build_entity_summary_rows,
    build_qa_rows,
    build_relation_summary_rows,
    build_rows,
    build_stage2_rows,
    chunk_text,
    detect_communities,
    looks_human_made,
    split_val,
    verify_stage1_response,
)

_TEXT_RU = (
    "Павел Дуров основал социальную сеть ВКонтакте в Санкт-Петербурге "
    "в сентябре 2006 года. Сеть быстро стала популярной среди студентов."
)

_RECORDS = [
    {
        "text": _TEXT_RU,
        "style": "journalistic",
        "lang": "ru",
        "source": "wiki/ru/1",
    },
    {
        "text": (
            "Tim Cook is the CEO of Apple. The company is headquartered in "
            "Cupertino, California, and employs over a hundred thousand people."
        ),
        "style": "journalistic",
        "lang": "en",
        "source": "wiki/en/2",
    },
    {
        "text": "ну короче мы типа поехали туда а там как обычно никого нет",
        "style": "conversational",
        "lang": "ru",
        "source": "social/3",  # fails the human-made proxy (no punctuation)
    },
    {
        "text": _TEXT_RU,  # duplicate -> dropped
        "style": "journalistic",
        "lang": "ru",
        "source": "wiki/ru/1-dup",
    },
    {
        "text": "Слишком коротко.",
        "style": "literary",
        "lang": "ru",
        "source": "books/4",  # fails min_words
    },
]


def test_chunk_text_passthrough_and_split() -> None:
    assert chunk_text("one two three", 350) == ["one two three"]
    words = " ".join(f"w{i}" for i in range(10))
    chunks = chunk_text(words, 4)
    assert len(chunks) == 3
    assert chunks[0] == "w0 w1 w2 w3"


def test_looks_human_made() -> None:
    assert looks_human_made(_TEXT_RU, min_punct_ratio=0.01)
    assert not looks_human_made(
        "ну короче мы типа поехали туда а там никого нет и вот так вот",
        min_punct_ratio=0.01,
    )


def test_build_rows_filters_and_determinism() -> None:
    rows_a = build_rows(_RECORDS, seed=42, min_words=10)
    rows_b = build_rows(_RECORDS, seed=42, min_words=10)
    assert rows_a == rows_b  # byte-for-byte deterministic under the seed

    # Dedup + min-words + conversational punctuation filter removed 3 records.
    assert len(rows_a) == 2

    for row in rows_a:
        assert row["task_class"] in (
            "ragu_artifact_extraction",
            "ragu_entity_extraction",
        )
        assert [m["role"] for m in row["prompt"]] == ["system", "user"]
        payload = row["extra"]["extraction"]
        assert payload["stage"] in ("one_stage", "stage1")
        assert payload["text"]
        # The payload schema round-trips (the reward reads the SAME schema).
        schema = ExtractionSchema.from_extra(payload)
        assert schema is not None
        assert schema.entity_types and schema.relation_types
        assert row["extra"]["target_language"] in ("ru", "en", None)


def test_build_rows_rng_knobs() -> None:
    rows = build_rows(
        _RECORDS,
        seed=42,
        stage1_share=1.0,
        no_target_language_share=1.0,
        schema_mode="subset",
        n_entity_types=4,
    )
    assert all(
        row["extra"]["extraction"]["stage"] == "stage1" for row in rows
    )
    assert all(row["extra"]["target_language"] is None for row in rows)
    for row in rows:
        assert len(row["extra"]["extraction"]["entity_types"]) <= 4


def test_split_val_stratified() -> None:
    rows = build_rows(_RECORDS, seed=1, min_words=10)
    train, val = split_val(rows, val_per_style=1, seed=1)
    assert len(val) == 1  # one style survived filtering (journalistic)
    assert len(train) == len(rows) - 1
    assert val[0]["id"].startswith("ext-val-")
    styles = {row["extra"]["style"] for row in val}
    assert styles <= {"journalistic"}


def _stage1_row() -> dict:
    rows = build_rows(
        [
            {
                "text": _TEXT_RU,
                "style": "journalistic",
                "lang": "ru",
                "source": "wiki/ru/1",
            }
        ],
        seed=3,
        stage1_share=1.0,
        min_words=10,
    )
    return rows[0]


def _stage1_response() -> str:
    return json.dumps(
        {
            "entities": [
                {
                    "entity_name": "Павел Дуров",
                    "entity_type": "PERSON",
                    "description": "Основатель сети.",
                },
                {
                    "entity_name": "ВКонтакте",
                    "entity_type": "ORGANIZATION",
                    "description": "Социальная сеть.",
                },
                {
                    "entity_name": "Эйфелева башня",
                    "entity_type": "PERSON",  # schema member but ungrounded
                    "description": "Галлюцинация.",
                },
                {
                    "entity_name": "Что-то",
                    "entity_type": "WAT",  # grounded-ish but not a member
                    "description": "Тип вне схемы.",
                },
            ]
        },
        ensure_ascii=False,
    )


def test_verify_stage1_response() -> None:
    row = _stage1_row()
    payload = row["extra"]["extraction"]
    # Make sure the schema actually contains the types used below.
    payload["entity_types"] = ["PERSON", "ORGANIZATION", "CITY", "DATE"]

    verified = verify_stage1_response(_stage1_response(), payload)
    assert verified is not None
    names = [e["entity_name"] for e in verified]
    # Grounded schema members survive; hallucinated / off-schema drop.
    assert "Павел Дуров" in names
    assert "ВКонтакте" in names
    assert "Эйфелева башня" not in names
    assert "Что-то" not in names

    # Unparseable response → None.
    assert verify_stage1_response("мусор", payload) is None


def test_build_stage2_rows() -> None:
    row = _stage1_row()
    row["extra"]["extraction"]["entity_types"] = [
        "PERSON", "ORGANIZATION", "CITY", "DATE",
    ]
    gens = {row["id"]: _stage1_response()}
    rows = build_stage2_rows([row], gens, min_entities=2, seed=5)
    assert len(rows) == 1
    s2 = rows[0]
    assert s2["task_class"] == "ragu_relation_extraction"
    assert s2["id"].startswith("ext-s2-")
    payload = s2["extra"]["extraction"]
    assert payload["stage"] == "stage2"
    provided = payload["provided_entities"]
    assert {e["entity_name"] for e in provided} == {"Павел Дуров", "ВКонтакте"}
    # The user prompt carries the entity list; the system references it.
    user = s2["prompt"][1]["content"]
    assert "- Павел Дуров (PERSON)" in user
    # Too few verified entities → the row is skipped.
    few = build_stage2_rows(
        [row], {row["id"]: json.dumps({"entities": []})}, min_entities=2
    )
    assert few == []


# ---------------------------------------------------------------------------
# Vocabulary / doc_id / variable-K extensions (the NEREL-BIO plan)
# doc_id
# ---------------------------------------------------------------------------
def test_build_rows_carries_doc_id_and_default_ontology() -> None:
    rows = build_rows(
        [{"text": _TEXT_RU, "style": "journalistic", "lang": "ru",
          "source": "wiki/ru/1", "doc_id": "wiki/ru/1"}],
        seed=42,
        schema_mode="subset",
        min_words=5,
    )
    assert rows
    for row in rows:
        assert row["extra"]["doc_id"] == "wiki/ru/1"
        assert row["extra"]["extraction"]["ontology"] == "nerel"


def test_build_rows_resolves_bio_ontology() -> None:
    rows = build_rows(
        [{"text": _TEXT_RU, "style": "scientific", "lang": "ru",
          "source": "nerel-bio/train/1", "doc_id": "bio-1",
          "ontology": "nerel_bio"}],
        seed=42,
        schema_mode="subset",
        n_entity_types=6,
        min_words=5,
    )
    assert rows
    payload = rows[0]["extra"]["extraction"]
    assert payload["ontology"] == "nerel_bio"
    # The promised types are members of the BIO vocabulary (DISO possible,
    # DISEASE impossible) and the subset is bounded by K.
    assert len(payload["entity_types"]) <= 6
    assert set(payload["entity_types"]) <= set(BIO_ENTITY_TYPES)
    assert "DISEASE" not in payload["entity_types"]
    # The prompt is rendered through the BIO vocabulary as well.
    system = rows[0]["prompt"][0]["content"]
    assert any(t in system for t in payload["entity_types"])


def test_build_rows_rejects_unknown_ontology_loudly() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown extraction ontology"):
        build_rows(
            [{"text": _TEXT_RU, "style": "journalistic", "lang": "ru",
              "source": "x", "doc_id": "x", "ontology": "nurel_bio"}],
            seed=42,
            min_words=5,
        )


def test_variable_k_produces_schema_size_diversity() -> None:
    records = [
        {"text": _TEXT_RU + f" Документ номер {i}.", "style": "journalistic",
         "lang": "ru", "source": f"src/{i}", "doc_id": f"doc-{i}"}
        for i in range(40)
    ]
    rows = build_rows(
        records, seed=9, schema_mode="subset",
        n_entity_types=6, n_entity_types_max=14, min_words=5,
    )
    sizes = {len(r["extra"]["extraction"]["entity_types"]) for r in rows}
    assert min(sizes) >= 2
    assert len(sizes) > 1  # the difficulty axis varies
    assert max(sizes) <= 14
    # Determinism: same seed → same sizes stream.
    rows2 = build_rows(
        records, seed=9, schema_mode="subset",
        n_entity_types=6, n_entity_types_max=14, min_words=5,
    )
    assert [r["extra"]["extraction"]["entity_types"] for r in rows] == [
        r["extra"]["extraction"]["entity_types"] for r in rows2
    ]


def test_split_val_keeps_whole_document_on_one_side() -> None:
    rows = []
    for doc in range(6):
        for part in range(3):
            rows.append(
                {
                    "id": f"r-{doc}-{part}",
                    "task_class": "ragu_artifact_extraction",
                    "prompt": [],
                    "extra": {
                        "style": "journalistic",
                        "source": f"src/{doc}",
                        "doc_id": f"doc-{doc}",
                        "extraction": {"stage": "one_stage"},
                    },
                }
            )
    train, val = split_val(rows, val_per_style=2, seed=42)
    train_docs = {r["extra"]["doc_id"] for r in train}
    val_docs = {r["extra"]["doc_id"] for r in val}
    assert train_docs & val_docs == set()
    # val_per_style counts DOCUMENTS now, not rows.
    assert len(val_docs) == 2


def test_iter_input_records_accepts_multiple_files(tmp_path) -> None:
    from scripts.build_extraction_corpus import iter_input_records

    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(
        json.dumps(
            {"text": _TEXT_RU, "style": "journalistic", "lang": "ru",
             "source": "s1", "doc_id": "d1", "ontology": "nerel_bio"}
        ) + "\n",
        encoding="utf-8",
    )
    b.write_text(
        json.dumps({"text": "Tim Cook runs Apple from Cupertino.", "style":
                    "scientific", "lang": "en", "source": "s2", "doc_id": "d2"})
        + "\n",
        encoding="utf-8",
    )
    records = list(iter_input_records([str(a), str(b)]))
    assert [r["doc_id"] for r in records] == ["d1", "d2"]
    assert records[0]["ontology"] == "nerel_bio"
    assert records[1]["ontology"] is None
    # A single path still works (the legacy call shape).
    records_one = list(iter_input_records(str(a)))
    assert len(records_one) == 1


# ---------------------------------------------------------------------------
# summaries subcommand
# ---------------------------------------------------------------------------
def _summary_stage1_corpus() -> tuple[list[dict], dict[str, str]]:
    'Two chunks of ONE document (doc_id ``d1``) + grounded stage-1 gens.'
    records = [
        {"text": _TEXT_RU, "style": "journalistic", "lang": "ru",
         "source": "wiki/1", "doc_id": "d1"},
        {"text": ("Павел Дуров развивал ВКонтакте в Санкт-Петербурге. "
                  "Сеть росла стремительно среди студентов."),
         "style": "journalistic", "lang": "ru", "source": "wiki/2",
         "doc_id": "d1"},
    ]
    rows = build_rows(records, seed=7, stage1_share=1.0, min_words=5)
    for row in rows:
        row["extra"]["extraction"]["entity_types"] = [
            "PERSON", "ORGANIZATION",
        ]
    gens = {
        rows[0]["id"]: json.dumps(
            {"entities": [
                {"entity_name": "Павел Дуров", "entity_type": "PERSON",
                 "description": "основатель социальной сети ВКонтакте"},
                {"entity_name": "ВКонтакте", "entity_type": "ORGANIZATION",
                 "description": "социальная сеть, запущенная в 2006 году"},
            ]}, ensure_ascii=False),
        # Case-variant mention: the casefolded key must still group.
        rows[1]["id"]: json.dumps(
            {"entities": [
                {"entity_name": "ПАВЕЛ ДУРОВ", "entity_type": "PERSON",
                 "description": "российский предприниматель, создатель "
                                "мессенджера Telegram"},
                {"entity_name": "Вконтакте", "entity_type": "ORGANIZATION",
                 "description": "крупнейшая русскоязычная соцсеть"},
            ]}, ensure_ascii=False),
    }
    return rows, gens


def test_build_entity_summary_rows_grouping() -> None:
    rows, gens = _summary_stage1_corpus()
    out = build_entity_summary_rows(rows, gens, seed=1)
    # Two entities, each consolidated across the TWO chunks of doc d1;
    # the case-variant mention
    assert len(out) == 2
    by_name = {r["extra"]["summary"]["entity_name"]: r for r in out}
    durov = by_name["Павел Дуров"]
    assert durov["task_class"] == "ragu_entity_summary"
    assert durov["id"].startswith("ext-sum-e-")
    payload = durov["extra"]["summary"]
    assert payload["kind"] == "entity"
    assert len(payload["descriptions"]) == 2
    assert durov["extra"]["doc_id"] == "d1"
    # The prompt lists both merged descriptions (the contract the reward sees).
    user = durov["prompt"][1]["content"]
    assert "1. " in user and "2. " in user
    # Deterministic under the seed.
    assert out == build_entity_summary_rows(rows, gens, seed=1)


def test_build_entity_summary_rows_thresholds() -> None:
    rows, gens = _summary_stage1_corpus()
    # min_descriptions above the group size → nothing survives.
    assert build_entity_summary_rows(rows, gens, min_descriptions=5) == []
    # Missing generations → nothing to verify.
    assert build_entity_summary_rows(rows, {}) == []


def _summary_stage2_rows(
    rows: list[dict], gens: dict[str, str]
) -> tuple[list[dict], dict[str, str]]:
    stage2 = build_stage2_rows(rows, gens, min_entities=2, seed=5)
    for row in stage2:
        row["extra"]["extraction"]["relation_types"] = ["founded"]
    s2_gens = {
        row["id"]: json.dumps(
            {"entities": [], "relations": [
                {"source_entity": "Павел Дуров",
                 "target_entity": "ВКонтакте",
                 "relation_type": "founded",
                 "description": "Дуров создал социальную сеть ВКонтакте"},
            ]}, ensure_ascii=False)
        for row in stage2
    }
    return stage2, s2_gens


def test_build_relation_summary_rows_grouping() -> None:
    rows, gens = _summary_stage1_corpus()
    stage2, s2_gens = _summary_stage2_rows(rows, gens)
    out = build_relation_summary_rows(stage2, s2_gens, seed=1)
    # The same (subject, relation, object) across two chunks → ONE row.
    assert len(out) == 1
    rel = out[0]
    assert rel["task_class"] == "ragu_relation_summary"
    assert rel["id"].startswith("ext-sum-r-")
    payload = rel["extra"]["summary"]
    assert payload["kind"] == "relation"
    assert payload["subject"] == "Павел Дуров"
    assert payload["object"] == "ВКонтакте"
    assert len(payload["descriptions"]) == 2
    assert rel["extra"]["doc_id"] == "d1"
    assert out == build_relation_summary_rows(stage2, s2_gens, seed=1)


def test_build_community_summary_rows() -> None:
    rows, gens = _summary_stage1_corpus()
    stage2, s2_gens = _summary_stage2_rows(rows, gens)
    out = build_community_summary_rows(
        rows + stage2, gens, s2_gens, min_size=2, max_size=10
    )
    # One document, one connected community {
    assert len(out) == 1
    comm = out[0]
    assert comm["task_class"] == "ragu_community_summary"
    assert comm["id"].startswith("ext-sum-c-")
    payload = comm["extra"]["summary"]
    assert payload["kind"] == "community"
    names = {e["entity_name"] for e in payload["entities"]}
    assert names == {"Павел Дуров", "ВКонтакте"}
    # The single internal edge carries its consolidated description.
    assert len(payload["relations"]) == 1
    assert payload["relations"][0]["relation_type"] == "founded"
    assert "Дуров создал" in payload["relations"][0]["description"]
    # The community prompt renders entities + relations.
    user = comm["prompt"][1]["content"]
    assert "Павел Дуров (PERSON)" in user
    assert "Павел Дуров -> ВКонтакте (founded)" in user
    assert out == build_community_summary_rows(
        rows + stage2, gens, s2_gens, min_size=2, max_size=10
    )


def test_detect_communities_deterministic() -> None:
    nodes = ["a", "b", "c", "d", "e"]
    edges = [("a", "b"), ("b", "c"), ("c", "a"), ("d", "e")]
    comms = detect_communities(nodes, edges, min_size=2, max_size=10)
    # A triangle {a,b,c} + a pair {d,e}, sorted deterministically.
    assert comms == [["a", "b", "c"], ["d", "e"]]
    assert comms == detect_communities(nodes, edges, min_size=2, max_size=10)
    # Communities below min_size drop out.
    assert detect_communities(nodes, edges, min_size=3) == [["a", "b", "c"]]
    assert detect_communities([], []) == []


# ---------------------------------------------------------------------------
# qa subcommand
# ---------------------------------------------------------------------------
_QA_RECORDS = [
    {
        "question": "Столица Франции?",
        "answer": "Париж — столица Франции.",
        "text": "Франция — государство в Западной Европе. Столица — Париж.",
        "source": "fineweb2/d1", "url": "http://x", "doc_id": "d1",
        "lang": "ru",
    },
    {
        "question": "What is the capital of Japan?",
        "answer": "Tokyo is the capital of Japan.",
        "text": "Japan is an island country. Its capital is Tokyo.",
        "source": "fineweb2/d2", "url": "http://y", "doc_id": "d2",
        "lang": "en",
    },
    # Empty question → dropped.
    {"question": "", "answer": "x", "source": "s", "doc_id": "d3",
     "lang": "ru"},
    # One-word answer → below the answer floor → dropped.
    {"question": "Сколько будет два плюс два?", "answer": "Четыре.",
     "source": "s", "doc_id": "d4", "lang": "ru"},
]


def test_build_qa_rows_reference_and_determinism() -> None:
    rows = build_qa_rows(_QA_RECORDS, seed=42, with_context_share=1.0)
    # Two valid records survive the empty-question / short-answer filters.
    assert len(rows) == 2
    assert rows == build_qa_rows(_QA_RECORDS, seed=42, with_context_share=1.0)
    for row in rows:
        assert row["task_class"] == "fineweb_qa"
        assert row["id"].startswith("qa-")
        assert row["reference"]  # the ONE reference-bearing class
        assert [m["role"] for m in row["prompt"]] == ["system", "user"]
        assert row["extra"]["with_context"] is True
        assert row["extra"]["target_language"] in ("ru", "en")
    # Open-book rows render the context into the user message.
    ru_row = next(r for r in rows if r["extra"]["target_language"] == "ru")
    assert "Context document:" in ru_row["prompt"][1]["content"]
    assert ru_row["prompt"][1]["content"].endswith("Question: Столица Франции?")


def test_build_qa_rows_closed_book_and_language_fallback() -> None:
    rows = build_qa_rows(
        _QA_RECORDS, seed=42, with_context_share=0.0,
        compact_variant_share=1.0,
    )
    assert len(rows) == 2
    for row in rows:
        assert row["extra"]["with_context"] is False
        user = row["prompt"][1]["content"]
        assert user.startswith("Question: ")
        assert "Context document" not in user
    # Unknown lang falls back to English.
    odd = build_qa_rows(
        [{"question": "q q q", "answer": "a a a", "lang": "de"}], seed=1,
    )
    assert odd[0]["extra"]["target_language"] == "en"
    assert "English" in odd[0]["prompt"][0]["content"]


def test_build_qa_rows_max_rows() -> None:
    rows = build_qa_rows(_QA_RECORDS, seed=42, max_rows=1)
    assert len(rows) == 1
    assert rows[0]["id"] == "qa-000001"
