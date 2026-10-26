'Offline tests of the deterministic extraction reward (no network, no GPU).'

from __future__ import annotations

import json

from llm_training.reward import extraction as ex

_TEXT_RU = (
    "Павел Дуров основал социальную сеть ВКонтакте в Санкт-Петербурге "
    "в сентябре 2006 года."
)
_TEXT_EN = "Tim Cook is the CEO of Apple, based in Cupertino."

_EXTRA_RU = {
    "stage": "one_stage",
    "language": "ru",
    "text": _TEXT_RU,
    "entity_types": ["PERSON", "ORGANIZATION", "CITY", "DATE"],
    "relation_types": ["FOUNDED_BY", "TAKES_PLACE_IN"],
    "signatures": {
        "FOUNDED_BY": {
            "domain": ["ORGANIZATION", "FACILITY", "LOCATION"],
            "range": ["PERSON", "ORGANIZATION"],
        },
        "TAKES_PLACE_IN": {"domain": ["EVENT"], "range": ["LOCATION"]},
    },
}


def _one_stage_response() -> str:
    return json.dumps(
        {
            "entities": [
                {
                    "entity_name": "Павел Дуров",
                    "entity_type": "PERSON",
                    "description": "Основатель социальной сети.",
                },
                {
                    "entity_name": "ВКонтакте",
                    "entity_type": "ORGANIZATION",
                    "description": "Социальная сеть.",
                },
                {
                    "entity_name": "Санкт-Петербург",
                    "entity_type": "CITY",
                    "description": "Город основания.",
                },
            ],
            "relations": [
                {
                    "source_entity": "ВКонтакте",
                    "target_entity": "Павел Дуров",
                    "relation_type": "FOUNDED_BY",
                    "description": "Дуров основал ВКонтакте.",
                    "relationship_strength": 5,
                },
                {
                    "source_entity": "ВКонтакте",
                    "target_entity": "Санкт-Петербург",
                    "relation_type": "FOUNDED_BY",
                    "description": "Город, где основана сеть.",
                    "relationship_strength": 4,
                },
            ],
        },
        ensure_ascii=False,
    )


# --------------------------------------------------------------------- parse
def test_parse_plain_fenced_and_literal() -> None:
    payload = _one_stage_response()
    parsed = ex.parse_extraction_output(payload)
    assert parsed is not None and len(parsed["entities"]) == 3

    fenced = f"Вот ответ:\n```json\n{payload}\n```\nКонец."
    assert ex.parse_extraction_output(fenced) == parsed

    # Python-repr quoting (the Qwen2.5 template occasionally emits it).
    literal = payload.replace('"', "'")
    parsed_lit = ex.parse_extraction_output(literal)
    assert parsed_lit is not None and parsed_lit["entities"][0][
        "entity_name"
    ] == "Павел Дуров"

    assert ex.parse_extraction_output("no json here") is None
    assert ex.parse_extraction_output("") is None


def test_parse_drops_malformed_items() -> None:
    parsed = ex.parse_extraction_output(
        json.dumps(
            {
                "entities": [
                    {"entity_name": "X", "entity_type": "PERSON"},
                    {"entity_type": "PERSON"},  # no name → dropped
                    "junk",
                ],
                "relations": [
                    {"source_entity": "X", "target_entity": ""},  # dropped
                ],
            }
        )
    )
    assert parsed is not None
    assert len(parsed["entities"]) == 1
    assert parsed["relations"] == []


# ----------------------------------------------------------------- json score
def test_json_score_perfect_and_broken() -> None:
    assert ex.extraction_json_score(_one_stage_response(), "one_stage") == 1.0
    # Empty lists are valid shape; item share is neutral.
    empty = json.dumps({"entities": [], "relations": []})
    assert ex.extraction_json_score(empty, "one_stage") == 1.0
    assert ex.extraction_json_score("garbage", "one_stage") == 0.0
    # Wrong-stage shape: entities-only response on the one-stage task keeps
    # the parse share but loses the shape share.
    entities_only = json.dumps({"entities": []})
    assert 0.0 < ex.extraction_json_score(entities_only, "one_stage") < 1.0
    assert ex.extraction_json_score(entities_only, "stage1") == 1.0


# ----------------------------------------------------------------- grounding
def test_grounding_inflected_ru() -> None:
    parsed = ex.parse_extraction_output(_one_stage_response())
    score = ex.entity_grounding_score(parsed, _TEXT_RU, "ru")
    assert score is not None and score > 0.85
    # Hallucinated entity scores low.
    bad = {"entities": [{"entity_name": "Эйфелева башня", "entity_type": "FACILITY"}]}
    bad_score = ex.entity_grounding_score(bad, _TEXT_RU, "ru")
    assert bad_score is not None and bad_score < 0.35


def test_grounding_english() -> None:
    parsed = {
        "entities": [
            {"entity_name": "Tim Cook", "entity_type": "PERSON"},
            {"entity_name": "Apple", "entity_type": "ORGANIZATION"},
        ]
    }
    score = ex.entity_grounding_score(parsed, _TEXT_EN, "en")
    assert score == 1.0


def test_detect_lang() -> None:
    assert ex.detect_lang("Привет мир") == "ru"
    assert ex.detect_lang("Hello world") == "en"


# -------------------------------------------------------------------- schema
def test_schema_membership_and_signatures() -> None:
    parsed = ex.parse_extraction_output(_one_stage_response())
    schema = ex.ExtractionSchema.from_extra(_EXTRA_RU)
    assert schema is not None
    # All types are members; FOUNDED_BY(ORG -> PERSON) is valid, but
    # FOUNDED_BY(ORG -> CITY) violates the range → sig share = 1/2.
    score = ex.extraction_schema_score(parsed, schema)
    assert score is not None
    assert abs(score - (0.5 * 1.0 + 0.5 * 0.5)) < 1e-6

    unknown_type = {
        "entities": [
            {"entity_name": "Павел Дуров", "entity_type": "WAT", "description": "d"}
        ],
        "relations": [],
    }
    sc = ex.extraction_schema_score(unknown_type, schema)
    assert sc is not None and abs(sc - 0.5) < 1e-6  # membership 0, sig neutral


# ------------------------------------------------------------------- closure
def test_relation_closure_one_stage() -> None:
    parsed = ex.parse_extraction_output(_one_stage_response())
    assert ex.relation_closure_score(parsed) == 1.0
    parsed["relations"][1]["target_entity"] = "Москва"  # not extracted
    assert ex.relation_closure_score(parsed) == 0.5
    # No relations → N/A.
    assert ex.relation_closure_score({"entities": [], "relations": []}) is None


def test_relation_closure_stage2_provided() -> None:
    provided = [
        {"entity_name": "ВКонтакте", "entity_type": "ORGANIZATION"},
        {"entity_name": "Павел Дуров", "entity_type": "PERSON"},
    ]
    resp = json.dumps(
        {
            "relations": [
                {
                    "source_entity": "вконтакте",  # case-insensitive match
                    "target_entity": "Павел Дуров",
                    "relation_type": "FOUNDED_BY",
                    "description": "d",
                    "relationship_strength": 5,
                }
            ]
        },
        ensure_ascii=False,
    )
    parsed = ex.parse_extraction_output(resp)
    assert ex.relation_closure_score(parsed, provided_entities=provided) == 1.0
    assert ex.relation_closure_score(parsed, provided_entities=None) == 0.0


# -------------------------------------------------------------- normalization
class _FakeMorph:
    def __init__(self, features: dict[str, list[str]]) -> None:
        self._features = features

    def get(self, key: str) -> list[str]:
        return self._features.get(key, [])


class _FakeToken:
    def __init__(
        self,
        features: dict[str, list[str]],
        *,
        is_alpha: bool = True,
        like_num: bool = False,
        is_digit: bool = False,
        pos: str = "",
    ) -> None:
        self.is_alpha = is_alpha
        self.like_num = like_num
        self.is_digit = is_digit
        self.pos_ = pos
        self.morph = _FakeMorph(features)


def _install_fake_spacy(monkeypatch, head_features: dict[str, list[str]]):
    """Install a fake spaCy pipeline that tags ONLY the last token."""

    def fake_nlp(name: str):
        tokens = [
            _FakeToken({}) for _ in name.split()[:-1]
        ] + [_FakeToken(head_features)]
        return tokens

    monkeypatch.setitem(ex._spacy_pipelines, "ru", fake_nlp)
    return fake_nlp


def test_normalization_morph_rules(monkeypatch) -> None:
    parsed = {
        "entities": [
            {"entity_name": "Павел Дуров", "entity_type": "PERSON"}
        ]
    }
    _install_fake_spacy(monkeypatch, {"Case": ["Nom"], "Number": ["Sing"]})
    assert ex.entity_normalization_score(parsed, "ru") == 1.0

    _install_fake_spacy(monkeypatch, {"Case": ["Loc"], "Number": ["Sing"]})
    assert ex.entity_normalization_score(parsed, "ru") == 0.5

    # Missing features are vacuously satisfied.
    _install_fake_spacy(monkeypatch, {})
    assert ex.entity_normalization_score(parsed, "ru") == 1.0


def test_normalization_skips_literals_and_missing_spacy(monkeypatch) -> None:
    literal_only = {
        "entities": [
            {"entity_name": "17 мая 1998 года", "entity_type": "DATE"}
        ]
    }
    assert ex.entity_normalization_score(literal_only, "ru") is None

    # Content-driven literal detection: a NUMERAL-tagged / numeric token
    # value is vacuously normalized even under a nominal-typed slot.
    def numeral_nlp(name: str):
        return [
            _FakeToken({}, is_alpha=False, like_num=True, pos="NUM")
            for _ in name.split()
        ]

    monkeypatch.setitem(ex._spacy_pipelines, "ru", numeral_nlp)
    nominal_slot = {
        "entities": [
            {"entity_name": "17", "entity_type": "PERSON"}
        ]
    }
    assert ex.entity_normalization_score(nominal_slot, "ru") == 1.0

    # spaCy unavailable → graceful None (never an abort).
    def _boom():
        raise RuntimeError("no spacy")

    import builtins

    real_import = builtins.__import__

    def no_spacy(name, *args, **kwargs):
        if name == "spacy":
            raise ImportError("spacy missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_spacy)
    monkeypatch.delitem(ex._spacy_pipelines, "ru", raising=False)
    assert ex.entity_normalization_score(
        {"entities": [{"entity_name": "Париж", "entity_type": "CITY"}]}, "ru"
    ) is None


# ------------------------------------------------------------------ consensus
def test_group_consensus_and_coverage() -> None:
    def resp(names: list[str]) -> str:
        return json.dumps(
            {
                "entities": [
                    {"entity_name": n, "entity_type": "PERSON", "description": "d"}
                    for n in names
                ],
                "relations": [],
            },
            ensure_ascii=False,
        )

    group = [
        ex.parse_extraction_output(resp(["Париж", "Москва"])),
        ex.parse_extraction_output(resp(["Париже", "Москва"])),  # inflection merges
        ex.parse_extraction_output(resp(["Париж"])),
        ex.parse_extraction_output(resp(["Лондон"])),  # minority → no consensus
    ]
    consensus = ex.group_consensus_entities(group, "ru")
    assert len(consensus) == 2  
    # Minority mention is NOT in the consensus.
    assert not any("лондон" in c for c in consensus)

    full = ex.coverage_f1(group[0], consensus, "ru")
    assert full == 1.0
    # One hallucinated extra: precision 2/3, recall 1 → F1 = 0.8.
    with_extra = ex.parse_extraction_output(resp(["Париж", "Москва", "Токио"]))
    assert abs(ex.coverage_f1(with_extra, consensus, "ru") - 0.8) < 1e-6
    # The empty-extraction degenerate optimum scores 0.
    empty = ex.parse_extraction_output(resp([]))
    assert ex.coverage_f1(empty, consensus, "ru") == 0.0
    # Degenerate groups (singleton / no agreement) → N/A.
    assert ex.group_consensus_entities(group[:1], "ru") == []
    assert ex.coverage_f1(group[0], [], "ru") is None


# ------------------------------------------------------------------- combined
def test_extraction_check_end_to_end() -> None:
    response = _one_stage_response()
    scored = ex.extraction_check(
        response, "ragu_artifact_extraction", {"extraction": _EXTRA_RU}
    )
    assert scored is not None
    assert scored.json_score == 1.0
    assert scored.schema_score is not None and scored.schema_score == 0.75
    assert scored.grounding is not None and scored.grounding > 0.85
    assert scored.closure == 1.0
    assert scored.combined > 0.0
    assert scored.n_entities == 3 and scored.n_relations == 2
    assert set(scored.as_dict()) == {
        "json_score", "schema_score", "grounding", "closure", "normalization",
        "coverage", "n_entities", "n_relations", "combined",
    }


def test_extraction_check_group_coverage() -> None:
    response = _one_stage_response()
    shorter = json.dumps(
        {
            "entities": [
                {"entity_name": "Павел Дуров", "entity_type": "PERSON",
                 "description": "d"},
            ],
            "relations": [],
        },
        ensure_ascii=False,
    )
    group = [response, response, shorter]
    scored = ex.extraction_check(
        response,
        "ragu_artifact_extraction",
        {"extraction": _EXTRA_RU},
        group_responses=group,
    )
    assert scored is not None and scored.coverage is not None
    assert scored.coverage == 1.0  # the full response covers the consensus
    # The partial sibling misses 2 of the 3 consensus entities
    # (precision 1, recall 1/3 → F1 = 0.5).
    partial = ex.extraction_check(
        shorter,
        "ragu_artifact_extraction",
        {"extraction": _EXTRA_RU},
        group_responses=group,
    )
    assert partial is not None and partial.coverage is not None
    assert abs(partial.coverage - 0.5) < 1e-6
    # Singleton group → coverage N/A.
    single = ex.extraction_check(
        response, "ragu_artifact_extraction",
        {"extraction": _EXTRA_RU}, group_responses=[response],
    )
    assert single is not None and single.coverage is None


def test_extraction_check_gating() -> None:
    response = _one_stage_response()
    # Wrong task class → not applicable.
    assert (
        ex.extraction_check(response, "menon_qa", {"extraction": _EXTRA_RU}) is None
    )
    # Extraction class WITHOUT the corpus payload (a labeled ICL row) →
    # stays on the SemSim path.
    assert ex.extraction_check(response, "ragu_artifact_extraction", None) is None
    assert not ex.extraction_check_applies("ragu_artifact_extraction", None)
    assert ex.extraction_check_applies(
        "ragu_entity_extraction", {"extraction": {"stage": "stage1"}}
    )


def test_extraction_check_empty_response_is_not_perfect() -> None:
    empty = json.dumps({"entities": [], "relations": []})
    scored = ex.extraction_check(
        empty, "ragu_artifact_extraction", {"extraction": _EXTRA_RU}
    )
    # Without a group the empty response is not yet punished (coverage N/A),
    # but inside a group where siblings agree it scores a hard zero.
    assert scored is not None
    consensus_sibling = _one_stage_response()
    zeroed = ex.extraction_check(
        empty,
        "ragu_artifact_extraction",
        {"extraction": _EXTRA_RU},
        group_responses=[empty, consensus_sibling, consensus_sibling],
    )
    assert zeroed is not None and zeroed.coverage == 0.0
    assert zeroed.combined == 0.0  # the strict harmonic mean zeroes it


def test_extraction_check_stage2_uses_provided_entities() -> None:
    extra = {
        "stage": "stage2",
        "language": "ru",
        "text": _TEXT_RU,
        "entity_types": ["PERSON", "ORGANIZATION"],
        "relation_types": ["FOUNDED_BY"],
        "signatures": {
            "FOUNDED_BY": {
                "domain": ["ORGANIZATION", "FACILITY", "LOCATION"],
                "range": ["PERSON", "ORGANIZATION"],
            }
        },
        "provided_entities": [
            {"entity_name": "ВКонтакте", "entity_type": "ORGANIZATION"},
            {"entity_name": "Павел Дуров", "entity_type": "PERSON"},
        ],
    }
    response = json.dumps(
        {
            "relations": [
                {
                    "source_entity": "ВКонтакте",
                    "target_entity": "Павел Дуров",
                    "relation_type": "FOUNDED_BY",
                    "description": "Дуров основал ВКонтакте.",
                    "relationship_strength": 5,
                }
            ]
        },
        ensure_ascii=False,
    )
    scored = ex.extraction_check(
        response, "ragu_relation_extraction", {"extraction": extra}
    )
    assert scored is not None
    assert scored.closure == 1.0
    assert scored.grounding is None  # stage-2 entities were verified upstream
    assert scored.n_entities == 0


# ------------------------------------------------- real-spaCy integration
def _real_spacy_nlp_or_skip(lang: str):
    'Load the REAL spaCy model or skip the test when unavailable.'
    import importlib.util

    if importlib.util.find_spec("spacy") is None:
        import pytest

        pytest.skip("spacy not installed / spacy не установлен")
    model = ex._SPACY_MODELS.get(lang, ex._SPACY_MODELS["en"])
    try:
        nlp = ex._spacy_nlp(lang)
    except Exception as exc:  # pragma: no cover - operator env dependent
        import pytest

        pytest.skip(f"spaCy model {model} unavailable: {exc}")
    return nlp


def test_normalization_real_spacy_literal_and_nominal() -> None:
    'The literal guard + morph rules on the REAL ru/en pipelines.\n\n    an operator install; skipped when absent). Locks the two findings\n    verified by hand on ru_core_news_md 3.8.16:\n      the trailing "'
    _real_spacy_nlp_or_skip("ru")

    # Literal value under a nominal-typed slot → vacuous 1.0.
    literal_slot = {
        "entities": [
            {"entity_name": "17 мая 1998 года", "entity_type": "PERSON"}
        ]
    }
    assert ex.entity_normalization_score(literal_slot, "ru") == 1.0

    nom = {
        "entities": [
            {"entity_name": "Павел Дуров", "entity_type": "PERSON"},
            {"entity_name": "Москва", "entity_type": "CITY"},
        ]
    }
    assert ex.entity_normalization_score(nom, "ru") == 1.0

    # The Genitive head of an inflected multi-token name costs 0.5.
    gen = {
        "entities": [
            {"entity_name": "Павла Дурова", "entity_type": "PERSON"}
        ]
    }
    score = ex.entity_normalization_score(gen, "ru")
    assert score is not None and score == 0.5
