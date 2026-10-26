"Tests for runtime/drift_probe.py + scripts/build_drift_probe.py.\n\n\nOffline: the deterministic checks, the response scoring, the JSONL round-trip\nand the producer's determinism (the probe set must stay byte-stable across\nregenerations — otherwise the ctrl-point drift column measures the set\nchange, not the model change). No model is touched (run_drift_probe itself\nneeds a GPU model — exercised on the cluster)."

from __future__ import annotations

import json

import pytest

from llm_training.runtime.drift_probe import (
    ProbeItem,
    check_response,
    load_probe,
    score_responses,
)
from scripts.build_drift_probe import _recipes


@pytest.mark.parametrize(
    "check,text,expected",
    [
        # word counts
        ({"type": "min_words", "n": 3}, "раз два три четыре", True),
        ({"type": "min_words", "n": 5}, "раз два три", False),
        ({"type": "max_words", "n": 2}, "раз два", True),
        ({"type": "max_words", "n": 1}, "раз два", False),
        ({"type": "exact_words", "n": 2}, "раз, два!", True),
        ({"type": "exact_words", "n": 2}, "раз", False),
        # punctuation-free
        ({"type": "no_commas"}, "текст без запятых", True),
        ({"type": "no_commas"}, "текст, с запятой", False),
        # JSON keys
        ({"type": "json_keys", "keys": ["name", "age"]}, '{"name": "a", "age": 1}', True),
        ({"type": "json_keys", "keys": ["name"]}, "```json\n{\"name\": \"x\"}\n```", True),
        ({"type": "json_keys", "keys": ["name"]}, '{"age": 1}', False),
        ({"type": "json_keys", "keys": ["name"]}, "не json", False),
        ({"type": "json_keys", "keys": ["name"]}, '[1, 2]', False),
        # keyword frequency
        ({"type": "keyword_min", "keyword": "модель", "n": 2}, "Модель и модель", True),
        ({"type": "keyword_min", "keyword": "data", "n": 2}, "data once", False),
        ({"type": "keyword_max", "keyword": "cloud", "n": 0}, "no word here", True),
        ({"type": "keyword_max", "keyword": "cloud", "n": 0}, "a cloud", False),
        # answer language
        ({"type": "language_ru"}, "привет мир как дела", True),
        ({"type": "language_ru"}, "hello world", False),
        ({"type": "language_ru"}, "", False),
        ({"type": "language_en"}, "hello world", True),
        ({"type": "language_en"}, "привет мир", False),
    ],
)
def test_check_response(check, text, expected):
    assert check_response(check, text) is expected


def test_check_response_unknown_type_fails_closed():
    # A probe record the runtime cannot interpret must never silently pass.
    
    assert check_response({"type": "bogus"}, "любой текст") is False


def test_score_responses_aggregates():
    items = [
        ProbeItem(id="a", prompt="p", check={"type": "no_commas"}),
        ProbeItem(id="b", prompt="p", check={"type": "min_words", "n": 3}),
        ProbeItem(id="c", prompt="p", check={"type": "min_words", "n": 3}),
        ProbeItem(id="d", prompt="p", check={"type": "no_commas"}),
    ]
    result = score_responses(items, ["без запятых", "мало", "слов тут хватает", "с, запятой"])
    assert result["n"] == 4
    assert result["score"] == pytest.approx(0.5)
    assert result["per_check"]["no_commas"] == {"pass_rate": 0.5, "n": 2}
    assert result["per_check"]["min_words"] == {"pass_rate": 0.5, "n": 2}
    assert set(result["failed_ids"]) == {"b", "d"}


def test_score_responses_truncates_to_shorter_input():
    items = [ProbeItem(id="a", prompt="p", check={"type": "no_commas"})] * 3
    result = score_responses(items, ["ок"])
    assert result["n"] == 1


def test_load_probe_round_trip(tmp_path):
    records = [
        {"id": "probe-0001", "lang": "ru", "check": {"type": "min_words", "n": 5},
         "prompt": "Напиши не менее 5 слов."},
        {"id": "probe-0002", "lang": "en", "check": {"type": "json_keys", "keys": ["a"]},
         "prompt": "Answer with JSON keys a."},
    ]
    path = tmp_path / "drift_probe.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    items = load_probe(path)
    assert [it.id for it in items] == ["probe-0001", "probe-0002"]
    assert items[0].check["n"] == 5
    assert items[1].lang == "en"


def test_producer_deterministic_and_covering():
    # Byte-stability across regenerations + every check type is covered.
    
    a, b = _recipes(), _recipes()
    assert a == b
    ids = [rec["id"] for rec in a]
    assert len(ids) == len(set(ids))
    types = {rec["check"]["type"] for rec in a}
    assert types == {
        "min_words", "max_words", "exact_words", "no_commas",
        "json_keys", "keyword_min", "keyword_max",
        "language_ru", "language_en",
    }
    assert len(a) >= 60  # a meaningful probe sample


def test_committed_probe_file_matches_producer(tmp_path):
    # The committed src_data/drift_probe.jsonl must equal what the producer
    # builds — regenerating accidentally must not drift the ctrl-point set.
    
    
    from pathlib import Path

    committed = Path(__file__).resolve().parents[2] / "src_data" / "drift_probe.jsonl"
    if not committed.is_file():  # pragma: no cover - committed with the repo
        pytest.skip("src_data/drift_probe.jsonl not generated yet")
    out = tmp_path / "regen.jsonl"
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in _recipes()),
        encoding="utf-8",
    )
    assert committed.read_text(encoding="utf-8") == out.read_text(encoding="utf-8")
