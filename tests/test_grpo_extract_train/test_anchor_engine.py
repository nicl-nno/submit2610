'Offline tests of the anchor vLLM path (colocate engine + rank sharding).'

from __future__ import annotations

from typing import Any

import pytest

import grpo_extract_train as get_train
from llm_training.runtime.agent_suite import AgentSuiteScorer


class _FakeVGen:
    """TRL-1.10-like colocate engine: instance-state sampling + generate."""

    def __init__(self) -> None:
        self.temperature = 1.0
        self.top_p = 1.0
        self.top_k = 0
        self.min_p = None
        self.max_completion_length = 4096
        self.sync_calls = 0
        self.calls: list[list[list[int]]] = []
        self.state_at_call: list[tuple[float, int]] = []

    def sync_weights(self) -> None:
        self.sync_calls += 1

    def generate(self, prompts, images=None, num_generations=None, *args, **kwargs):
        self.calls.append([list(p) for p in prompts])
        self.state_at_call.append((self.temperature, self.top_k))
        completions = [[100 + i] for i in range(len(prompts))]
        return prompts, completions, None, None


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert add_generation_prompt is True
        return "P:" + messages[-1]["content"]

    def __call__(self, text, add_special_tokens=False, **kwargs):
        assert add_special_tokens is False
        return {"input_ids": [abs(hash(text)) % 1000]}

    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        return f"text-{ids[0]}"


class _Model:
    training = False


_EXTRA = {
    "extraction": {
        "entity_types": ["PERSON"],
        "language": "ru",
        "ontology": "nerel",
        "relation_types": [],
        "signatures": {},
    }
}


def _val_rows() -> list[dict[str, Any]]:
    return [
        {
            "id": f"e{i}",
            "task_class": "ragu_entity_extraction",
            "prompt": [{"role": "user", "content": f"чанк {i}"}],
            "extra": {"extraction": dict(_EXTRA)},
        }
        for i in range(5)
    ]


def test_colocate_engine_greedy_override_and_restore() -> None:
    vgen = _FakeVGen()
    tok = _FakeTok()
    engine = get_train._ColocateAnchorEngine(vgen, tok, max_new_tokens=2048)
    with engine:
        # greedy state during the anchor window
        assert (vgen.temperature, vgen.top_p, vgen.top_k, vgen.min_p) == (
            0.0,
            1.0,
            0,
            0.0,
        )
        assert vgen.max_completion_length == 2048
        out = engine.generate_texts(["P:a", "P:b", "P:c"], batch_size=2)
    # batching: ceil(3/2) = 2 engine calls, order preserved (the fake numbers
    # completions per CALL, so the second batch restarts at 100)
    assert [len(c) for c in vgen.calls] == [2, 1]
    # greedy sampling was active inside generate()
    assert vgen.state_at_call == [(0.0, 0), (0.0, 0)]
    # the training sampling state is restored on exit
    assert (vgen.temperature, vgen.top_p, vgen.top_k, vgen.min_p) == (
        1.0,
        1.0,
        0,
        None,
    )
    assert vgen.max_completion_length == 4096
    assert vgen.sync_calls == 1
    assert out == ["text-100", "text-101", "text-100"]


def test_anchor_shard_covers_every_index_once() -> None:
    for n, world in ((7, 3), (8, 8), (5, 1)):
        seen: list[int] = []
        for r in range(world):
            seen.extend(get_train._anchor_shard(n, r, world))
        assert sorted(seen) == list(range(n))


def test_run_extraction_anchor_engine_path_scores_whole_pool() -> None:
    vgen = _FakeVGen()
    tok = _FakeTok()
    engine = get_train._ColocateAnchorEngine(vgen, tok, max_new_tokens=2048)
    rows = _val_rows()
    breakdown = get_train.run_extraction_anchor(
        _Model(),
        tok,
        rows,
        max_new_tokens=2048,
        engine=engine,
        rank=0,
        world_size=1,
    )
    assert breakdown is not None
    # every row scored exactly once (the "n" counts val entries)
    assert breakdown["n"] == len(rows)
    # the extraction envelope failed (canned text is not JSON) → combined 0.0,
    # but the pseudo-gold component still recorded its own accumulator key
    assert "pseudo_gold" not in breakdown or breakdown["pseudo_gold"] >= 0.0
    assert breakdown["val"] == 0.0  # strict harmonic: broken JSON → 0.0
    # the HF path was NOT taken (the model was never used for generation)
    assert len(vgen.calls) >= 1


def test_run_extraction_anchor_none_for_nonzero_rank(monkeypatch) -> None:
    vgen = _FakeVGen()
    tok = _FakeTok()
    engine = get_train._ColocateAnchorEngine(vgen, tok, max_new_tokens=2048)
    rows = _val_rows()
    # canned per-rank decoded responses: 5 rows over 4 ranks → [0,4], [1], [2], [3]
    shards = [["t0", "t4"], ["t1"], ["t2"], ["t3"]]

    def fake_gather(obj, world_size):
        assert obj == ["text-100"]  # rank 2 gathers its own decoded responses
        return shards

    monkeypatch.setattr(get_train, "_gather_anchor_objects", fake_gather)
    breakdown = get_train.run_extraction_anchor(
        _Model(),
        tok,
        rows,
        max_new_tokens=2048,
        engine=engine,
        rank=2,
        world_size=4,
    )
    assert breakdown is None  # rank 0 owns the breakdown
    # the rank generated only its interleaved shard (index 2 → one prompt),
    # tokenized exactly like the engine contract expects
    flat = [p for call in vgen.calls for p in call]
    expected_ids = tok("P:чанк 2", add_special_tokens=False)["input_ids"]
    assert flat == [expected_ids]


class _DuckSuite:
    """Duck-typed scorer: only what ``evaluate`` touches in engine mode."""

    max_new_tokens = 8
    rows = [
        {"prompt": [[{"role": "user", "content": "t0"}]]},
        {"prompt": [[{"role": "user", "content": "t1"}]]},
        {"prompt": [[{"role": "user", "content": "t2"}]]},
        {"prompt": [[{"role": "user", "content": "t3"}]]},
    ]

    def __init__(self) -> None:
        self.seen: list[int] | None = None

    def rollout(self, generate_batch, rows=None):
        rows = self.rows if rows is None else rows
        self.seen = [id(r) for r in rows]
        return [{"task": i} for i in range(len(rows))]

    def score(self, records):
        return {"agent_val": 0.9, "n": len(records)}


def test_anchor_engine_bypasses_the_cuts_wrapper() -> None:
    'Mixed-CUTS patches vgen.generate; the anchor must measure through the\n    SAVED ORIGINAL (no standard/CUTS split, no CUTS sampling), and vLLM 0.27\n    would reject the CUTS temperature outright (must be in [0, 2]).\n\n    Mixed-CUTS'
    calls: list[tuple[str, list[list[int]]]] = []

    def original(ids, images=None, num_generations=None, *a, **k):
        calls.append(("original", [list(p) for p in ids]))
        return ids, [[200 + i] for i in range(len(ids))], None, None

    def wrapped(ids, images=None, num_generations=None, *a, **k):
        calls.append(("wrapped", [list(p) for p in ids]))
        raise AssertionError("the anchor must not route through Mixed-CUTS")

    class _WrappedVGen(_FakeVGen):
        pass

    vgen = _WrappedVGen()
    vgen.generate = wrapped
    vgen.generate_original = original  # the mixed_cuts unwrap seam
    tok = _FakeTok()
    engine = get_train._ColocateAnchorEngine(vgen, tok, max_new_tokens=2048)
    assert engine._generate is original  # the seam is picked up at init
    breakdown = get_train.run_extraction_anchor(
        _Model(),
        tok,
        _val_rows(),
        max_new_tokens=2048,
        engine=engine,
        rank=0,
        world_size=1,
    )
    assert breakdown is not None
    # exactly ONE original-call (5 prompts in one batch), zero wrapped calls
    assert [name for name, _ in calls] == ["original"]


def test_agent_suite_evaluate_engine_path_world1() -> None:
    suite = _DuckSuite()
    breakdown, records = AgentSuiteScorer.evaluate(
        suite,
        model=None,
        tokenizer=None,
        generate_batch_fn=lambda msgs: ["resp"] * len(msgs),
        rank=0,
        world_size=1,
    )
    assert breakdown == {"agent_val": 0.9, "n": 4}
    assert records is not None and len(records) == 4
    # world_size == 1 → the full pool, no shard
    assert suite.seen == [id(r) for r in suite.rows]


def test_agent_suite_evaluate_shards_rows_by_rank(monkeypatch) -> None:
    suite = _DuckSuite()
    monkeypatch.setattr(
        "llm_training.runtime.agent_suite._gather_objects",
        lambda obj, world_size: [
            [{"task": i} for i in range(r_, r_ + 1)] for r_ in range(world_size)
        ],
    )
    result = AgentSuiteScorer.evaluate(
        suite,
        model=None,
        tokenizer=None,
        generate_batch_fn=lambda msgs: ["resp"] * len(msgs),
        rank=1,
        world_size=4,
    )
    assert result == (None, None)  # non-zero rank gets nothing back
    # rank 1 of 4 over 4 rows → exactly rows[1]
    assert suite.seen is not None and len(suite.seen) == 1
