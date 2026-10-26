'Offline tests of the summarization prompts + reward (no network, no GPU).'

from __future__ import annotations

import pytest

from llm_training.extraction.summary_prompts import (
    LANGUAGE_VALUES,
    SUMMARY_COMMUNITY,
    SUMMARY_ENTITY,
    SUMMARY_RELATION,
    SUMMARY_STAGES,
    build_community_summary_prompt,
    build_entity_summary_prompt,
    build_relation_summary_prompt,
)
from llm_training.reward.summarization import (
    SUMMARY_REFERENCE_KEY,
    SUMMARY_TASK_CLASSES,
    community_membership_score,
    make_summary_semsim,
    parse_summary_output,
    summarization_applies,
    summarization_check,
    summary_compression_score,
    summary_coverage_score,
    summary_json_score,
    summary_output_text,
    summary_precision_score,
    summary_repetition_score,
)

_ENTITY_PAYLOAD = {
    "kind": "entity",
    "entity_name": "метформин",
    "entity_type": "CHEM",
    "descriptions": [
        "метформин — пероральный гипогликемический препарат из группы "
        "бигуанидов, применяемый для лечения сахарного диабета второго типа",
        "препарат снижает уровень глюкозы в крови у пациентов с диабетом "
        "и повышает чувствительность тканей к инсулину",
        "назначается при инсулинорезистентности и ожирении у больных "
        "диабетом, хорошо переносится пациентами",
    ],
    "text": "Метформин снижает уровень глюкозы в крови у пациентов с "
            "диабетом второго типа.",
}

_RELATION_PAYLOAD = {
    "kind": "relation",
    "subject": "метформин",
    "object": "диабет",
    "relation_type": "TREATED_USING",
    "descriptions": [
        "метформин применяется для лечения диабета",
        "препарат назначают пациентам с диабетом второго типа",
    ],
    "text": "Метформин применяется для лечения диабета.",
}

_COMMUNITY_PAYLOAD = {
    "kind": "community",
    "entities": [
        {"entity_name": "метформин", "entity_type": "CHEM",
         "description": "препарат против диабета"},
        {"entity_name": "диабет", "entity_type": "DISO",
         "description": "заболевание обмена веществ"},
        {"entity_name": "инсулин", "entity_type": "CHEM",
         "description": "гормон поджелудочной железы"},
    ],
    "relations": [
        {"source_entity": "метформин", "relation_type": "TREATED_USING",
         "target_entity": "диабет", "description": "лечение метформином"},
    ],
    "text": "Метформин применяют при диабете, инсулин вводят инъекционно.",
}


# --------------------------------------------------------------------- prompts
def test_entity_summary_prompt_structure() -> None:
    prompt = build_entity_summary_prompt(
        "метформин", "CHEM", ["описание один", "описание два"], "Russian"
    )
    assert [m["role"] for m in prompt] == ["system", "user"]
    system, user = prompt[0]["content"], prompt[1]["content"]
    # The inline JSON schema is spelled out (the no-server-decoding rule).
    assert '{"entity_name": str, "description": str}' in system
    assert "Russian" in system
    # The primary descriptions are listed and numbered.
    assert "1. описание один" in user
    assert "2. описание два" in user
    assert "метформин (CHEM)" in user


def test_relation_and_community_prompts() -> None:
    rel = build_relation_summary_prompt(
        "метформин", "CHEM", "TREATED_USING", "диабет", "DISO",
        ["описание"], "English",
    )
    assert '{"subject_name": str, "object_name": str, "description": str}' in rel[0]["content"]
    assert "TREATED_USING" in rel[1]["content"]
    comm = build_community_summary_prompt(
        _COMMUNITY_PAYLOAD["entities"],
        _COMMUNITY_PAYLOAD["relations"],
        "Russian",
    )
    system, user = comm[0]["content"], comm[1]["content"]
    assert '"rating": int' in system
    assert "5 to 10 key findings" in system
    assert "метформин (CHEM): препарат против диабета" in user
    assert "метформин -> диабет (TREATED_USING)" in user


def test_prompt_compact_variants_and_validation() -> None:
    full = build_entity_summary_prompt(
        "x", "", ["d"], "Russian", variant=0
    )[0]["content"]
    compact = build_entity_summary_prompt(
        "x", "", ["d"], "Russian", variant=1
    )[0]["content"]
    assert full != compact
    assert "Return ONLY valid JSON" in compact
    with pytest.raises(ValueError):
        build_entity_summary_prompt("x", "", [], "Russian")
    with pytest.raises(ValueError):
        build_community_summary_prompt([], [], "Russian")


# ----------------------------------------------------------------- parsing
def test_parse_summary_output_tolerant() -> None:
    raw = '```json\n{"entity_name": "x", "description": "y"}\n```'
    assert parse_summary_output(raw, "entity") == {
        "entity_name": "x", "description": "y"
    }
    # Prose around the outermost JSON is stripped (the Qwen fence habit).
    assert parse_summary_output('Ответ: {"description": "z"}', "entity") == {
        "description": "z"
    }
    # Empty description -> unusable.
    assert parse_summary_output('{"description": ""}', "entity") is None
    # Community without a findings list -> unusable; findings without
    # usable entries -> unusable.
    assert parse_summary_output('{"title": "t"}', "community") is None
    assert parse_summary_output(
        '{"title": "t", "findings": [{}]}', "community"
    ) is None
    # Python-literal quoting (single quotes) tolerated.
    assert parse_summary_output(
        "{'entity_name': 'x', 'description': 'y'}", "entity"
    ) is not None


# ---------------------------------------------------------------- components
def test_json_score_grading() -> None:
    good = '{"entity_name": "x", "description": "y"}'
    assert summary_json_score(good, "entity") == 1.0
    # The parse gate already requires a non-empty description: a response
    # without it is fully unusable (0.0, the extraction convention).
    assert summary_json_score('{"entity_name": "x"}', "entity") == 0.0
    # Missing the name key but a usable description: parse 0.4 + core 0.3,
    # the type share is lost → 0.7.
    assert summary_json_score('{"description": "y"}', "entity") == 0.7
    assert summary_json_score("мусор", "entity") == 0.0
    community = (
        '{"title": "t", "summary": "s", "rating": 7, '
        '"rating_explanation": "e", "findings": [{"summary": "a", '
        '"explanation": "b"}]}'
    )
    assert summary_json_score(community, "community") == 1.0
    # rating out of range → the rating share is lost.
    bad_rating = community.replace('"rating": 7', '"rating": 42')
    assert summary_json_score(bad_rating, "community") < 1.0
    assert summary_json_score(bad_rating, "community") >= 0.9


def test_coverage_precision_compression_repetition() -> None:
    descs = [
        "препарат снижает уровень глюкозы в крови у пациентов с диабетом",
        "применяется при диабете второго типа у взрослых больных",
        "хорошо переносится, назначается при инсулинорезистентности",
    ]
    merged = (
        "Препарат снижает уровень глюкозы в крови у пациентов с диабетом "
        "второго типа и назначается при инсулинорезистентности."
    )
    # A good merge keeps most input stems (coverage high)...
    assert summary_coverage_score(descs, merged, "ru") > 0.5
    # ...and stays grounded (precision high).
    assert summary_precision_score(descs, merged, "ru") > 0.6
    # Compression: well within the band.
    assert summary_compression_score(descs, merged) == 1.0
    # Full concatenation is the zero point of the band.
    concat = " ".join(descs)
    assert summary_compression_score(descs, concat) == 0.0
    # Degenerate loop → repetition penalty bites.
    looped = " ".join(["снижает уровень глюкозы в крови у пациентов"] * 6)
    assert summary_repetition_score(looped) < 0.6
    assert summary_repetition_score(merged) == 1.0


def test_membership_anti_invention() -> None:
    output = (
        '{"title": "Диабет", "summary": "Сообщество о лечении диабета.",'
        ' "rating": 7, "rating_explanation": "Ключевая тема документа.",'
        ' "findings": ['
        '{"summary": "Метформин лечит диабет", "explanation": "связь метформин-диабет"},'
        '{"summary": "Инсулин важен", "explanation": "инсулин как гормон"},'
        '{"summary": "Придуманный препарат", "explanation": "никак не связан"}'
        "]}"
    )
    # 2 of 3 findings mention provided entities
    # containment is PER-FINDING — a global mention elsewhere in the
    # response does not validate the invented finding.
    score = community_membership_score(_COMMUNITY_PAYLOAD["entities"], output, "ru")
    assert abs(score - 2 / 3) < 1e-6


def test_compression_is_consolidation_only() -> None:
    'A community REPORT legitimately elaborates on its material — the\n    compression band applies to entity/relation consolidation ONLY.'
    long_report = (
        '{"title": "Диабет", "summary": "Сообщество про метформин, инсулин '
        'и диабет.", "rating": 8, "rating_explanation": "Ядро документа.", '
        '"findings": ['
        '{"summary": "f1", "explanation": "метформин лечит диабет"},'
        '{"summary": "f2", "explanation": "инсулин — гормон"},'
        '{"summary": "f3", "explanation": "диабет — болезнь обмена"},'
        '{"summary": "f4", "explanation": "метформин назначают пациентам"},'
        '{"summary": "f5", "explanation": "инсулин вводят инъекционно"}]}'
    )
    scored = summarization_check(
        "ragu_community_summary", {"summary": _COMMUNITY_PAYLOAD}, long_report
    )
    assert scored is not None
    assert scored.compression is None  # N/A for the report kind
    assert scored.combined > 0.0
    # Consolidation kinds keep the guard.
    consolidation = summarization_check(
        "ragu_entity_summary", {"summary": _ENTITY_PAYLOAD},
        '{"entity_name": "метформин", "description": "' + "метформин препарат. " * 40 + '"}',
    )
    assert consolidation is not None
    assert consolidation.compression is not None
    assert consolidation.compression == 0.0  # far beyond the band


# ------------------------------------------------------------------- dispatch
def test_applies_and_check_dispatch() -> None:
    assert summarization_applies(
        "ragu_entity_summary", {"summary": _ENTITY_PAYLOAD}
    )
    assert not summarization_applies("ragu_entity_summary", {})
    assert not summarization_applies("ragu_artifact_extraction", {"summary": _ENTITY_PAYLOAD})
    assert SUMMARY_TASK_CLASSES == frozenset(
        {
            "ragu_entity_summary",
            "ragu_relation_summary",
            "ragu_community_summary",
        }
    )
    # Not a summary class → None.
    assert summarization_check(
        "ragu_artifact_extraction", {"summary": _ENTITY_PAYLOAD}, "{}"
    ) is None
    # Summary class without the payload key → None.
    assert summarization_check("ragu_entity_summary", {}, "{}") is None


def test_check_good_and_broken_responses() -> None:
    good = (
        '{"entity_name": "метформин", "description": "Препарат для лечения '
        'диабета второго типа, снижающий уровень глюкозы в крови."}'
    )
    scored = summarization_check(
        "ragu_entity_summary", {"summary": _ENTITY_PAYLOAD}, good
    )
    assert scored is not None
    assert scored.kind == "entity"
    assert scored.json_score == 1.0
    assert scored.coverage is not None and scored.coverage > 0.3
    assert scored.precision is not None and scored.precision > 0.5
    assert scored.compression == 1.0
    assert 0.0 < scored.combined <= 1.0

    broken = summarization_check(
        "ragu_entity_summary", {"summary": _ENTITY_PAYLOAD}, "не JSON вовсе"
    )
    assert broken is not None
    # The strict harmonic zeroes a broken envelope.
    assert broken.combined == 0.0


def test_check_community_and_fluency_wiring() -> None:
    community = (
        '{"title": "Лечение диабета", "summary": "Сообщество про метформин, '
        'инсулин и диабет.", "rating": 8, "rating_explanation": "Ядро '
        'документа.", "findings": ['
        '{"summary": "Метформин лечит диабет", "explanation": "прямая связь '
        'метформин-диабет из текста"},'
        '{"summary": "Инсулин — гормон", "explanation": "инсулин упомянут как '
        'гормон поджелудочной железы"},'
        '{"summary": "Диабет — болезнь обмена", "explanation": "диабет описан '
        'как заболевание обмена веществ"},'
        '{"summary": "Терапия метформином", "explanation": "метформин '
        'назначается пациентам"},'
        '{"summary": "Роль инсулина", "explanation": "инсулин вводят '
        'инъекционно"}]}'
    )
    scored = summarization_check(
        "ragu_community_summary", {"summary": _COMMUNITY_PAYLOAD}, community
    )
    assert scored is not None
    assert scored.membership is not None and scored.membership == 1.0
    assert scored.coverage is not None and scored.coverage > 0.3
    assert scored.combined > 0.0

    # A wired fluency callable joins the harmonic; a failing one is recorded.
    scored_f = summarization_check(
        "ragu_entity_summary", {"summary": _ENTITY_PAYLOAD},
        '{"entity_name": "метформин", "description": "Препарат от диабета."}',
        fluency=lambda text: 0.9,
    )
    assert scored_f is not None and scored_f.fluency == 0.9
    scored_bad = summarization_check(
        "ragu_entity_summary", {"summary": _ENTITY_PAYLOAD},
        '{"entity_name": "метформин", "description": "Препарат от диабета."}',
        fluency=lambda text: (_ for _ in ()).throw(RuntimeError("model down")),
    )
    assert scored_bad is not None
    assert scored_bad.fluency is None
    assert "fluency" in scored_bad.failed_metrics


# ---------------------------------------------------------------------------
# Anchor SemSim primitives
# ---------------------------------------------------------------------------
def test_summary_output_text_public() -> None:
    """The prose extractor parses the envelope (fences tolerated)."""
    good = '{"entity_name": "метформин", "description": "Препарат от диабета."}'
    assert summary_output_text(good, "entity") == "Препарат от диабета."
    fenced = f"```json\n{good}\n```"
    assert summary_output_text(fenced, "entity") == "Препарат от диабета."
    # Unparseable / empty prose → None (the anchor then skips the SemSim).
    assert summary_output_text("просто проза без конверта", "entity") is None
    empty = '{"entity_name": "метформин", "description": ""}'
    assert summary_output_text(empty, "entity") is None


def test_summary_reference_key_is_the_documented_cache_slot() -> None:
    """The reference cache key is the frozen ``extra["summary"]`` slot."""
    assert SUMMARY_REFERENCE_KEY == "reference"


def test_make_summary_semsim_clips_and_caches() -> None:
    class _Emb:
        def __init__(self) -> None:
            self.calls = 0
            self._vecs = {
                "a": [1.0, 0.0],
                "b": [1.0, 0.0],    # identical direction → 1.0
                "c": [0.0, 1.0],    # orthogonal → 0.0
                "d": [-1.0, 0.0],   # opposite → clipped to 0.0
                "e": [2.0, 0.0],    # same direction, unnormalized → 1.0
                "f": [1.0, 1.0],    # 45° → 1/√2
            }

        def embed_query(self, text: str) -> list[float]:
            self.calls += 1
            return self._vecs[text]

    emb = _Emb()
    semsim = make_summary_semsim(emb)
    assert semsim is not None
    assert semsim("a", "b") == pytest.approx(1.0)
    assert semsim("a", "c") == pytest.approx(0.0)
    assert semsim("a", "d") == pytest.approx(0.0)   # negative clipped
    assert semsim("a", "e") == pytest.approx(1.0)   # scale-invariant
    assert semsim("a", "f") == pytest.approx(1.0 / 2 ** 0.5)
    # The cache: six distinct texts embedded once each despite repeats.
    assert emb.calls == 6

    # Only async methods → no sync pair → None (the anchor degrades).
    class _AsyncOnly:
        async def aembed_query(self, text: str) -> list[float]:
            return [0.0]

    assert make_summary_semsim(_AsyncOnly()) is None
    assert make_summary_semsim(object()) is None


def test_make_summary_semsim_embed_documents_fallback() -> None:
    class _Docs:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    semsim = make_summary_semsim(_Docs())
    assert semsim is not None
    assert semsim("x", "y") == pytest.approx(1.0)
    # A zero vector cannot form a cosine → 0.0, never a crash.

    class _Zero:
        def embed_query(self, text: str) -> list[float]:
            return [0.0, 0.0]

    semsim_zero = make_summary_semsim(_Zero())
    assert semsim_zero is not None
    assert semsim_zero("x", "y") == pytest.approx(0.0)
