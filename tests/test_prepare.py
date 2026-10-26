'Tests for llm_training.prepare — cache serialization + save/load round-trips.'

from __future__ import annotations

import json

import pytest

from llm_training.prepare import (
    example_from_dict,
    example_to_dict,
    load_grpo_cache,
    load_warmup_cache,
    read_examples_jsonl,
    save_grpo_cache,
    save_warmup_cache,
    write_examples_jsonl,
)
from llm_training.schema import UnifiedExample


def _example(task_class: str = "coder", *, tools=True, extra=None) -> UnifiedExample:
    return UnifiedExample(
        task_class=task_class,
        prompt=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "вопрос пользователя"},
        ],
        reference="эталон",
        tools=(
            [{"type": "function", "function": {"name": "t",
                                               "parameters": {"type": "object"}}}]
            if tools
            else None
        ),
        reference_messages=[{"role": "assistant", "content": "эталон"}],
        source="src#0",
        extra=dict(extra or {}),
    )


def test_example_round_trip():
    ex = _example(extra={"is_call_turn": True, "is_final_answer": True,
                         "turn_index": 0, "gold_label": 1})
    d = example_to_dict(ex)
    json.dumps(d, ensure_ascii=False)  # must be JSON-native
    ex2 = example_from_dict(d)
    assert ex2.task_class == ex.task_class
    assert ex2.prompt == ex.prompt
    assert ex2.reference == ex.reference
    assert ex2.tools == ex.tools
    assert ex2.reference_messages == ex.reference_messages
    assert ex2.source == ex.source
    assert ex2.extra == ex.extra


def test_example_round_trip_no_tools():
    ex = _example(tools=False)
    d = example_to_dict(ex)
    assert d["tools"] is None
    ex2 = example_from_dict(d)
    assert ex2.tools is None


def test_jsonl_round_trip(tmp_path):
    examples = [
        _example("coder", extra={"is_call_turn": True, "is_final_answer": True}),
        _example("menon_qa", tools=False, extra={}),
    ]
    write_examples_jsonl(tmp_path / "e.jsonl", examples)
    loaded = read_examples_jsonl(tmp_path / "e.jsonl")
    assert [ex.source for ex in loaded] == ["src#0", "src#0"]
    assert [ex.task_class for ex in loaded] == ["coder", "menon_qa"]
    assert loaded[0].extra["is_call_turn"] is True
    assert loaded[1].tools is None


def test_grpo_cache_round_trip(tmp_path):
    pools = {"coder": [_example("coder")], "menon_qa": [_example("menon_qa")]}
    val = [_example("summarizer", extra={"is_call_turn": True,
                                         "is_final_answer": True})]
    scores = {"src#0": 1.5, "src#1": 2.0}
    save_grpo_cache(
        tmp_path,
        train_pools=pools,
        val=val,
        scores=scores,
        seed=24,
        val_tasks=["summarizer"],
        include_network=False,
        max_per_class=200,
        difficulty_model="anonymous/base-model",
        tool_names=frozenset({"t"}),
        tool_schemas={"t": {"type": "object"}},
        stats={"foo": 1},
    )
    cache = load_grpo_cache(tmp_path)
    assert cache.seed == 24
    assert set(cache.train_pools) == {"coder", "menon_qa"}
    assert len(cache.train_pools["coder"]) == 1
    assert cache.val[0].task_class == "summarizer"
    assert cache.scores == {"src#0": 1.5, "src#1": 2.0}
    assert cache.tool_names == frozenset({"t"})
    assert cache.tool_schemas == {"t": {"type": "object"}}
    assert cache.val_tasks == ("summarizer",)
    assert cache.include_network is False
    assert cache.max_per_class == 200
    assert cache.difficulty_model == "anonymous/base-model"


def test_warmup_cache_round_trip(tmp_path):
    train = [_example("coder")]
    val = [_example("coder", extra={"is_call_turn": True, "is_final_answer": True})]
    save_warmup_cache(
        tmp_path,
        train=train,
        val=val,
        scores={"src#0": 2.0},
        seed=42,
        difficulty_model="anonymous/base-model",
        tool_names=frozenset({"t"}),
        stats={"n_train": 1},
    )
    cache = load_warmup_cache(tmp_path)
    assert cache.seed == 42
    assert len(cache.train) == 1
    assert len(cache.val) == 1
    assert cache.scores == {"src#0": 2.0}
    assert cache.tool_names == frozenset({"t"})


def test_wrong_pipeline_rejected(tmp_path):
    save_grpo_cache(
        tmp_path, train_pools={"coder": [_example("coder")]}, val=[],
        scores={}, seed=1, val_tasks=[], include_network=False,
        max_per_class=200, difficulty_model="", tool_names=frozenset(),
        tool_schemas={}, stats={},
    )
    with pytest.raises(ValueError):
        load_warmup_cache(tmp_path)


def test_version_guard(tmp_path):
    save_grpo_cache(
        tmp_path, train_pools={"coder": [_example("coder")]}, val=[],
        scores={}, seed=1, val_tasks=[], include_network=False,
        max_per_class=200, difficulty_model="", tool_names=frozenset(),
        tool_schemas={}, stats={},
    )
    meta_path = tmp_path / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["format_version"] = 999
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError):
        load_grpo_cache(tmp_path)
