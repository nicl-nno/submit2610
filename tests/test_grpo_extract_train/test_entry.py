'Offline tests of the GRPO-extraction entry (no network, no GPU).'

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import grpo_extract_train as get_train


class _Control:
    def __init__(self) -> None:
        self.should_training_stop = False


class _FakeTrainer:
    def __init__(self) -> None:
        self.saved: list[str] = []
        self.tokenizers_saved: list[str] = []
        self.control = _Control()
        self.model = object()

    def save_model(self, path: str) -> None:
        self.saved.append(path)


class _FakeTokenizer:
    def save_pretrained(self, path: str) -> None:
        pass


_TEXT_RU = "Павел Дуров основал социальную сеть ВКонтакте в Санкт-Петербурге."

# A realistic consolidation payload for the summary-composite tests.

_ENTITY_PAYLOAD = {
    "kind": "entity",
    "entity_name": "метформин",
    "entity_type": "CHEM",
    "descriptions": [
        "метформин — пероральный гипогликемический препарат из группы "
        "бигуанидов, применяемый для лечения сахарного диабета второго типа",
        "препарат снижает уровень глюкозы в крови у пациентов с диабетом "
        "и повышает чувствительность тканей к инсулину",
    ],
    "text": "Метформин снижает уровень глюкозы в крови у пациентов с "
            "диабетом второго типа.",
}

_EXTRA = {
    "stage": "one_stage",
    "language": "ru",
    "text": _TEXT_RU,
    "entity_types": ["PERSON", "ORGANIZATION", "CITY"],
    "relation_types": ["FOUNDED_BY"],
    "signatures": {
        "FOUNDED_BY": {
            "domain": ["ORGANIZATION", "FACILITY", "LOCATION"],
            "range": ["PERSON", "ORGANIZATION"],
        }
    },
}


def _good_response() -> str:
    return json.dumps(
        {
            "entities": [
                {"entity_name": "Павел Дуров", "entity_type": "PERSON",
                 "description": "Основатель."},
                {"entity_name": "ВКонтакте", "entity_type": "ORGANIZATION",
                 "description": "Соцсеть."},
            ],
            "relations": [
                {"source_entity": "ВКонтакте", "target_entity": "Павел Дуров",
                 "relation_type": "FOUNDED_BY", "description": "Основал.",
                 "relationship_strength": 5},
            ],
        },
        ensure_ascii=False,
    )


# ------------------------------------------------------------ reward function
def test_extraction_reward_group_coverage() -> None:
    reward = get_train.make_extraction_reward_fn(group_size=4)
    good = _good_response()
    empty = json.dumps({"entities": [], "relations": []})
    completions = [good, good, good, empty]
    extras = [{"extraction": _EXTRA}] * 4
    classes = ["ragu_artifact_extraction"] * 4
    rewards = asyncio.run(
        reward(
            prompts=[None] * 4,
            completions=completions,
            extra=extras,
            task_class=classes,
        )
    )
    assert len(rewards) == 4
    # The empty degenerate completion is punished by the consensus coverage.
    assert rewards[3] == 0.0
    assert all(r > 0.5 for r in rewards[:3])


def test_extraction_reward_unaligned_degrades_gracefully() -> None:
    reward = get_train.make_extraction_reward_fn(group_size=4)
    good = _good_response()
    # 3 completions with a group size of 4 → per-completion fallback.
    rewards = asyncio.run(
        reward(
            prompts=[None] * 3,
            completions=[good, good, good],
            extra=[{"extraction": _EXTRA}] * 3,
            task_class=["ragu_artifact_extraction"] * 3,
        )
    )
    assert len(rewards) == 3
    assert all(r > 0.0 for r in rewards)


# ------------------------------------------------- composite reward: summary
def test_composite_reward_summary_row_scores_structurally() -> None:
    'A consolidation row routes to summarization_check (not extraction).'
    from llm_training.reward.summarization import SUMMARY_TASK_CLASSES

    reward = get_train.make_extraction_reward_fn(group_size=1)
    tc = sorted(SUMMARY_TASK_CLASSES)[0]
    good = json.dumps(
        {"entity_name": "метформин",
         "description": "метформин — пероральный гипогликемический препарат "
                        "из группы бигуанидов для лечения диабета второго типа"},
        ensure_ascii=False,
    )
    bad = "не json вовсе"
    rewards = asyncio.run(
        reward(
            prompts=[None] * 2,
            completions=[good, bad],
            extra=[{"summary": _ENTITY_PAYLOAD}] * 2,
            task_class=[tc] * 2,
        )
    )
    assert rewards[0] > rewards[1]
    assert rewards[1] == 0.0  # a broken envelope zeroes the structural reward


def test_composite_reward_summary_fluency_failure_never_zeroes() -> None:
    'A raising fluency callable records but does not zero the reward.'
    from llm_training.reward.summarization import SUMMARY_TASK_CLASSES

    def boom(text: str, task_class: str) -> float | None:
        raise RuntimeError("fluency model down")

    reward = get_train.make_extraction_reward_fn(group_size=1, fluency_fn=boom)
    tc = sorted(SUMMARY_TASK_CLASSES)[0]
    good = json.dumps(
        {"entity_name": "метформин",
         "description": "метформин — пероральный гипогликемический препарат "
                        "из группы бигуанидов для лечения диабета второго типа"},
        ensure_ascii=False,
    )
    rewards = asyncio.run(
        reward(
            prompts=[None],
            completions=[good],
            extra=[{"summary": _ENTITY_PAYLOAD}],
            task_class=[tc],
        )
    )
    # Same as the no-fluency path: structural only, still > 0.
    assert rewards[0] > 0.0


# ---------------------------------------------------- composite reward: QA
class _FakeQaScorer:
    """Async scorer stub returning a fixed ``ScoredMetrics``-like object."""

    def __init__(self, combined: float) -> None:
        self.combined = combined
        self.calls: list[dict] = []

    async def score(self, **kwargs):
        self.calls.append(kwargs)

        class _M:
            pass

        m = _M()
        m.combined = self.combined
        return m


def test_composite_reward_qa_row_uses_scorer_and_reference() -> None:
    'A QA row routes to the reference scorer (SemSim + fluency).'
    scorer = _FakeQaScorer(combined=0.83)
    reward = get_train.make_extraction_reward_fn(group_size=1, scorer=scorer)
    rewards = asyncio.run(
        reward(
            prompts=[None],
            completions=["Париж — столица Франции."],
            extra=[{"qa": {"question": "Столица Франции?"}}],
            task_class=["fineweb_qa"],
            reference=["Париж является столицей Франции."],
        )
    )
    assert rewards == [0.83]
    call = scorer.calls[0]
    assert call["task_class"] == "fineweb_qa"
    assert call["reference"] == "Париж является столицей Франции."
    # user_input is the bare question, never the open-book context block.
    assert call["user_input"] == "Столица Франции?"


def test_composite_reward_qa_scorer_crash_maps_to_zero() -> None:
    class _Boom:
        async def score(self, **kwargs):
            raise RuntimeError("embedder down")

    reward = get_train.make_extraction_reward_fn(group_size=1, scorer=_Boom())
    rewards = asyncio.run(
        reward(
            prompts=[None],
            completions=["ok"],
            extra=[{"qa": {"question": "q"}}],
            task_class=["fineweb_qa"],
            reference=["a"],
        )
    )
    assert rewards == [0.0]


# ------------------------------------------------------- anchor: QA batching
class _FakeGenModel:
    'Minimal HF-style model whose ``generate`` echoes a canned response.'

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._i = 0
        self.training = False
        self.config = type("C", (), {"use_cache": True})()
        self.device = "cpu"

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def generate(self, input_ids=None, **kwargs):
        import torch

        prompt_len = input_ids.shape[1]
        # Encode the canned response as prompt_len dummy ids + 1 (decoded back
        # to the canned text via the fake tokenizer's decode).
        self._last_prompt_len = prompt_len
        self._last_response = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return torch.zeros((1, prompt_len + 1), dtype=torch.long)


class _FakeGenTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "PROMPT"

    def __call__(self, text, return_tensors=None, add_special_tokens=None):
        import torch

        class _Out(dict):
            @property
            def shape(self):
                return (1, 3)

            def to(self, device):
                return self

        return _Out({"input_ids": torch.zeros((1, 3), dtype=torch.long)})

    def decode(self, ids, skip_special_tokens=False):
        model = self._model
        return model._last_response


def test_anchor_qa_rows_batched_and_aligned() -> None:
    'QA rows are scored in one batch; ``vals`` stays index-aligned.\n\nthe\n    composite ``val`` must average all three in ORIGINAL row order.'
    model = _FakeGenModel(["entity answer", "Париж", _good_response()])
    tok = _FakeGenTokenizer()
    tok._model = model

    class _M:
        pass

    def _metrics(combined: float):
        m = _M()
        m.combined = combined
        return m

    class _FakeBatchScorer:
        def __init__(self) -> None:
            self.requests = []

        def score_batch(self, requests):
            self.requests = list(requests)
            return [_metrics(0.6)]

    scorer = _FakeBatchScorer()
    val_rows = [
        {"id": "s1", "task_class": "ragu_entity_summary",
         "prompt": [], "extra": {"summary": _ENTITY_PAYLOAD}},
        {"id": "q1", "task_class": "fineweb_qa", "prompt": [],
         "extra": {"qa": {"question": "Столица Франции?"}},
         "reference": "Париж — столица Франции."},
        {"id": "e1", "task_class": "ragu_artifact_extraction",
         "prompt": [], "extra": {"extraction": _EXTRA}},
    ]
    breakdown = get_train.run_extraction_anchor(
        model, tok, val_rows, max_new_tokens=8, qa_scorer=scorer
    )
    # The single QA row was scored via the batch scorer, in ONE call.
    assert len(scorer.requests) == 1
    assert scorer.requests[0].task_class == "fineweb_qa"
    assert scorer.requests[0].reference == "Париж — столица Франции."
    # n counts all three rows (summary + QA + extraction).
    assert breakdown["n"] == 3
    assert breakdown["qa"] == pytest.approx(0.6)
    # The composite val is the mean over the three rows, in original order.
    assert 0.0 < breakdown["val"] <= 1.0


def test_anchor_qa_without_scorer_zeroes_and_warns() -> None:
    model = _FakeGenModel(["Париж"])
    tok = _FakeGenTokenizer()
    tok._model = model
    val_rows = [
        {"id": "q1", "task_class": "fineweb_qa", "prompt": [],
         "extra": {"qa": {"question": "q"}}, "reference": "a"},
    ]
    breakdown = get_train.run_extraction_anchor(
        model, tok, val_rows, max_new_tokens=8, qa_scorer=None
    )
    assert breakdown["n"] == 1
    assert breakdown["val"] == 0.0


# ------------------------------------ anchor: summary reference SemSim
_SUMMARY_RESPONSE = (
    '{"entity_name": "метформин", '
    '"description": "Метформин — препарат от диабета."}'
)
_SUMMARY_REFERENCE = (
    '{"entity_name": "метформин", '
    '"description": "Метформин применяется при диабете."}'
)


def _summary_val_row(with_reference: bool) -> dict:
    payload = dict(_ENTITY_PAYLOAD)
    if with_reference:
        payload["reference"] = {
            "response": _SUMMARY_REFERENCE,
            "model": "fake-judge",
        }
    return {"id": "s1", "task_class": "ragu_entity_summary",
            "prompt": [], "extra": {"summary": payload}}


def test_anchor_summary_reference_semsim_harmonic() -> None:
    'A cached judge reference joins the summary val as clipped cosine.'
    from llm_training.reward.scoring import harmonic_mean
    from llm_training.reward.summarization import summarization_check

    calls: list[tuple[str, str]] = []

    def semsim(a: str, b: str) -> float:
        calls.append((a, b))
        return 0.5

    model = _FakeGenModel([_SUMMARY_RESPONSE])
    tok = _FakeGenTokenizer()
    tok._model = model
    row = _summary_val_row(with_reference=True)
    breakdown = get_train.run_extraction_anchor(
        model, tok, [row], max_new_tokens=8, summary_semsim=semsim
    )
    structural = summarization_check(
        "ragu_entity_summary", {"summary": row["extra"]["summary"]},
        _SUMMARY_RESPONSE,
    )
    assert structural is not None
    assert breakdown["val"] == pytest.approx(
        harmonic_mean([structural.combined, 0.5])
    )
    assert breakdown["summary_semsim"] == pytest.approx(0.5)
    # The SemSim compares the PROSE of both sides (no JSON scaffolding).
    assert calls == [
        ("Метформин — препарат от диабета.",
         "Метформин применяется при диабете."),
    ]


def test_anchor_summary_without_reference_falls_back() -> None:
    """No reference → the structural combined alone; SemSim never called."""
    from llm_training.reward.summarization import summarization_check

    called: list[bool] = []

    def semsim(a: str, b: str) -> float:
        called.append(True)
        return 0.5

    model = _FakeGenModel([_SUMMARY_RESPONSE])
    tok = _FakeGenTokenizer()
    tok._model = model
    row = _summary_val_row(with_reference=False)
    breakdown = get_train.run_extraction_anchor(
        model, tok, [row], max_new_tokens=8, summary_semsim=semsim
    )
    structural = summarization_check(
        "ragu_entity_summary", {"summary": row["extra"]["summary"]},
        _SUMMARY_RESPONSE,
    )
    assert structural is not None
    assert breakdown["val"] == pytest.approx(structural.combined)
    assert "summary_semsim" not in breakdown
    assert called == []  # the SemSim callable was never invoked


def test_anchor_summary_semsim_failure_degrades() -> None:
    """A crashing embedder degrades to the structural combined, no abort."""
    from llm_training.reward.summarization import summarization_check

    model = _FakeGenModel([_SUMMARY_RESPONSE])
    tok = _FakeGenTokenizer()
    tok._model = model
    row = _summary_val_row(with_reference=True)
    breakdown = get_train.run_extraction_anchor(
        model, tok, [row], max_new_tokens=8,
        summary_semsim=lambda a, b: (_ for _ in ()).throw(
            RuntimeError("gpu down")
        ),
    )
    structural = summarization_check(
        "ragu_entity_summary", {"summary": row["extra"]["summary"]},
        _SUMMARY_RESPONSE,
    )
    assert structural is not None
    assert breakdown["val"] == pytest.approx(structural.combined)
    assert "summary_semsim" not in breakdown


def test_summary_reference_semsim_helper_contract() -> None:
    """The shared helper's edge cases: ref shapes, bad prose, NaN, clipping."""
    ref = {"kind": "entity", "reference": {"response": _SUMMARY_REFERENCE}}
    got = get_train.summary_reference_semsim(
        lambda a, b: 0.7, ref, _SUMMARY_RESPONSE
    )
    assert got == pytest.approx(0.7)
    # A plain-string reference is tolerated.
    flat = {"kind": "entity", "reference": _SUMMARY_REFERENCE}
    assert get_train.summary_reference_semsim(
        lambda a, b: 0.7, flat, _SUMMARY_RESPONSE
    ) == pytest.approx(0.7)
    # No reference / unknown kind → None (no SemSim component).
    assert get_train.summary_reference_semsim(
        lambda a, b: 0.7, {"kind": "entity"}, _SUMMARY_RESPONSE
    ) is None
    assert get_train.summary_reference_semsim(
        lambda a, b: 0.7, {"reference": _SUMMARY_REFERENCE}, _SUMMARY_RESPONSE
    ) is None
    # An unparseable response → None (the structural side punishes it).
    assert get_train.summary_reference_semsim(
        lambda a, b: 0.7, ref, "просто текст без конверта"
    ) is None
    # NaN → None; out-of-range cosine → clipped.
    import math

    assert get_train.summary_reference_semsim(
        lambda a, b: math.nan, ref, _SUMMARY_RESPONSE
    ) is None
    assert get_train.summary_reference_semsim(
        lambda a, b: 2.0, ref, _SUMMARY_RESPONSE
    ) == 1.0
    assert get_train.summary_reference_semsim(
        lambda a, b: -1.0, ref, _SUMMARY_RESPONSE
    ) == 0.0


# ---------------------------------------------------------------- data loading
def test_load_extraction_rows(tmp_path: Path) -> None:
    good = {
        "id": "ext-000001",
        "task_class": "ragu_artifact_extraction",
        "prompt": [{"role": "user", "content": "hi"}],
        "extra": {"extraction": {"stage": "one_stage"}},
    }
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps(good, ensure_ascii=False) + "\n"
        + json.dumps({"id": "bad", "prompt": [], "extra": {}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        get_train.load_extraction_rows(str(path))


# --------------------------------------------------------------- anchor logic
def _make_callback(monkeypatch, tmp_path: Path, probe_scores: list[float]):
    """Wire the anchor callback with patched anchor/probe/generation."""
    breakdowns = [
        {"val": 0.5, "n": 2, "json_score": 1.0, "grounding": 1.0,
         "closure": 1.0, "pseudo_gold": 0.5},
        {"val": 0.7, "n": 2, "json_score": 1.0, "grounding": 1.0,
         "closure": 1.0, "pseudo_gold": 0.7},
        {"val": 0.8, "n": 2, "json_score": 1.0, "grounding": 1.0,
         "closure": 1.0, "pseudo_gold": 0.8},
        {"val": 0.9, "n": 2, "json_score": 1.0, "grounding": 1.0,
         "closure": 1.0, "pseudo_gold": 0.9},
    ]
    it = iter(breakdowns)
    monkeypatch.setattr(
        get_train, "run_extraction_anchor", lambda *a, **k: next(it)
    )
    scores = iter(probe_scores)

    def fake_probe(model, tokenizer, items, max_new_tokens=256):
        return {"score": next(scores)}

    import llm_training.runtime.drift_probe as dp

    monkeypatch.setattr(dp, "run_drift_probe", fake_probe)

    import llm_training.runtime.checkpointing as ckpt

    from contextlib import contextmanager

    @contextmanager
    def fake_saves(trainer, max_shard_size):
        yield

    monkeypatch.setattr(ckpt, "sharded_checkpoint_saves", fake_saves)

    callback, state, run_anchor = get_train.make_extraction_anchor_callback(
        tokenizer=_FakeTokenizer(),
        val_rows=[{"id": "v1", "task_class": "ragu_artifact_extraction",
                   "prompt": [], "extra": {"extraction": _EXTRA}}],
        anchor_every=10,
        patience=2,
        anchor_max_new_tokens=64,
        output_dir=tmp_path,
        max_shard_size="5GB",
        probe_items=[object()],
        probe_gate_tol=0.10,
        probe_warn_tol=0.05,
    )
    return state, run_anchor


def test_anchor_best_save_and_patience(monkeypatch, tmp_path: Path) -> None:
    state, run_anchor = _make_callback(
        monkeypatch, tmp_path, probe_scores=[0.9, 0.9, 0.9, 0.9]
    )
    trainer = _FakeTrainer()
    run_anchor(trainer, 0, is_baseline=True)
    assert state["best_val"] == pytest.approx(0.5)
    assert trainer.saved == []  # the baseline never saves
    run_anchor(trainer, 10)  # 0.7 > 0.5 → best-10
    assert trainer.saved == [str(tmp_path / "best-10")]
    run_anchor(trainer, 20)  # 0.8 → best-20
    assert trainer.saved[-1] == str(tmp_path / "best-20")
    assert state["no_improve"] == 0


def test_anchor_probe_gate_blocks_and_stops(monkeypatch, tmp_path: Path) -> None:
    # Baseline probe 0.9; then a probe collapse to 0.5 (< 0.9 − 0.10).
    state, run_anchor = _make_callback(
        monkeypatch, tmp_path, probe_scores=[0.9, 0.5, 0.5, 0.5]
    )
    trainer = _FakeTrainer()
    run_anchor(trainer, 0, is_baseline=True)
    run_anchor(trainer, 10)  # val improves but the gate blocks
    assert trainer.saved == []
    assert state["no_improve"] == 1
    assert not trainer.control.should_training_stop
    run_anchor(trainer, 20)  # second block → patience reached → early stop
    assert trainer.saved == []
    assert trainer.control.should_training_stop


# ------------------------------------------------- pyarrow struct-trap regression
def test_build_extraction_dataset_hetero_extra() -> None:
    "The 8-class corpus in ONE Dataset: per-class ``extra`` shapes must not\n    trip pyarrow's struct unification (the real SMOKE failure: ``ArrowInvalid:\n    cannot mix list and non-list, non-null values``). The column rides as JSON\n    strings; ``reference`` stays a uniform string column."
    pytest.importorskip("datasets")
    rows = [
        {
            "id": "ext-1",
            "task_class": "ragu_entity_extraction",
            "prompt": [{"role": "user", "content": "извлеки сущности"}],
            "extra": {
                "extraction": {
                    "entity_types": ["PERSON"],
                    "language": None,
                    "ontology": "nerel",
                    "relation_types": [],
                    "signatures": {},
                }
            },
        },
        {
            "id": "ext-sum-e-1",
            "task_class": "ragu_entity_summary",
            "prompt": [{"role": "user", "content": "обобщи сущность"}],
            "extra": {
                "summary": {
                    "kind": "entity",
                    "entity_name": "Дуров",
                    "entity_type": "PERSON",
                    "descriptions": ["основатель VK"],
                }
            },
        },
        {
            "id": "ext-sum-c-1",
            "task_class": "ragu_community_summary",
            "prompt": [{"role": "user", "content": "отчёт по документу"}],
            "extra": {
                "summary": {
                    "kind": "community",
                    "entities": [
                        {"entity_name": "Дуров", "entity_type": "PERSON", "description": "основатель"}
                    ],
                    "relations": [],
                }
            },
        },
        {
            "id": "qa-1",
            "task_class": "fineweb_qa",
            "prompt": [{"role": "user", "content": "кто основал VK?"}],
            "extra": {
                "qa": {"question": "кто основал VK?", "answer": "Павел Дуров", "with_context": False}
            },
            "reference": "Павел Дуров.",
        },
    ]
    ds = get_train.build_extraction_dataset(rows)
    assert ds.column_names == ["prompt", "task_class", "extra", "reference"]
    extras = ds["extra"]
    assert all(isinstance(e, str) for e in extras)
    assert json.loads(extras[0])["extraction"]["entity_types"] == ["PERSON"]
    assert json.loads(extras[2])["summary"]["kind"] == "community"
    assert ds["reference"] == ["", "", "", "Павел Дуров."]
    # reward-side coercion: strings parse, dicts pass through, None -> {}
    assert get_train._coerce_extra(extras[1])["summary"]["entity_name"] == "Дуров"
    assert get_train._coerce_extra({"a": 1}) == {"a": 1}
    assert get_train._coerce_extra(None) == {}


def test_extraction_reward_accepts_json_string_extra() -> None:
    'The reward fn sees dataset columns verbatim — the JSON-string ``extra``\n    encoding must dispatch exactly like the plain dict (same numbers).'
    reward = get_train.make_extraction_reward_fn(group_size=4)
    good = _good_response()
    completions = [good, good, good, json.dumps({"entities": [], "relations": []})]
    dict_rewards = asyncio.run(
        reward(
            prompts=[None] * 4,
            completions=completions,
            extra=[{"extraction": _EXTRA}] * 4,
            task_class=["ragu_artifact_extraction"] * 4,
        )
    )
    string_rewards = asyncio.run(
        reward(
            prompts=[None] * 4,
            completions=completions,
            extra=[json.dumps({"extraction": _EXTRA}, ensure_ascii=False)] * 4,
            task_class=["ragu_artifact_extraction"] * 4,
        )
    )
    assert string_rewards == pytest.approx(dict_rewards)
    assert string_rewards[3] == 0.0
    assert all(r > 0.5 for r in string_rewards[:3])


# ------------------------------------------------- text-prompt stream (TRL 1.10)
class _RecordingTokenizer:
    """Fake mirroring the anchor path's apply_chat_template contract."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert add_generation_prompt is True
        return "RENDERED:" + messages[-1]["content"]


def test_render_prompt_uses_generation_prompt() -> None:
    tok = _RecordingTokenizer()
    out = get_train._render_prompt(tok, [{"role": "user", "content": "извлеки"}])
    assert out == "RENDERED:извлеки"


def _stream_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in (("big", 1000), ("mid", 200), ("tiny", 50)):
        name, n = spec
        rows.extend(
            {"id": f"{name}-{i}", "task_class": name, "prompt": [{"role": "user", "content": f"{name} {i}"}], "extra": {}}
            for i in range(n)
        )
    return rows


def test_train_stream_balanced_and_deterministic() -> None:
    rows = _stream_rows()
    stream, stats = get_train.build_extraction_train_stream(
        rows, train_examples=600, max_upsample_ratio=8, seed=42
    )
    assert len(stream) == 600  # share 200 × 3 classes
    assert stats["big"]["total"] == 200  # subsampled fresh
    assert stats["mid"]["total"] == 200  # whole pool
    assert stats["tiny"]["total"] == 200  # 50 fresh + 150 duplicates (cap 8×)
    assert stats["tiny"]["duplicated"] == 150
    # determinism: the same seed rebuilds the identical stream
    again, _ = get_train.build_extraction_train_stream(
        rows, train_examples=600, max_upsample_ratio=8, seed=42
    )
    assert [r["id"] for r in again] == [r["id"] for r in stream]


def test_train_stream_deficit_redistribution() -> None:
    rows = _stream_rows()  # big=1000, tiny=50
    rows = [r for r in rows if r["task_class"] in ("big", "tiny")]
    # ratio=1 caps tiny at its pool (50) → deficit 250 flows to big as a donor
    stream, stats = get_train.build_extraction_train_stream(
        rows, train_examples=600, max_upsample_ratio=1, seed=42
    )
    assert len(stream) == 600
    assert stats["tiny"]["total"] == 50
    assert stats["big"]["total"] == 550


def test_train_stream_zero_returns_whole_corpus() -> None:
    rows = _stream_rows()
    stream, stats = get_train.build_extraction_train_stream(
        rows, train_examples=0, max_upsample_ratio=8, seed=42
    )
    assert stream == rows
    assert stats == {}


def test_build_extraction_dataset_rendered_prompts() -> None:
    pytest.importorskip("datasets")
    rows = [
        {
            "id": "r1",
            "task_class": "fineweb_qa",
            "prompt": [{"role": "user", "content": "кто основал VK?"}],
            "extra": {"qa": {"question": "кто основал VK?", "answer": "Дуров"}},
            "reference": "Павел Дуров.",
        },
        {
            "id": "r2",
            "task_class": "ragu_entity_extraction",
            "prompt": [{"role": "user", "content": "извлеки"}],
            "extra": {"extraction": {"entity_types": ["PERSON"]}},
        },
    ]
    ds = get_train.build_extraction_dataset(rows, tokenizer=_RecordingTokenizer())
    # TEXT prompts: plain strings (TRL standard format), not message lists
    assert ds["prompt"] == [
        "RENDERED:кто основал VK?",
        "RENDERED:извлеки",
    ]
    assert ds["reference"] == ["Павел Дуров.", ""]


def test_dataset_shuffle_fingerprint_safe_at_scale() -> None:
    "The failure mode of the full corpus: dill pickling the arrow table\n    (dataset fingerprinting on .shuffle/.select) runs ``combine_chunks`` and\n    overflows pyarrow's 2 GiB int32 offsets on a multi-GB column. The\n    balanced stream keeps the rendered column far below the limit — a\n    moderate-scale shuffle must pass."
    pytest.importorskip("datasets")
    rows = [
        {
            "id": f"r{i}",
            "task_class": f"c{i % 3}",
            "prompt": ("извлеки сущности из чанка " + str(i)) * 60,  # ~4 KB
            "extra": {"extraction": {"entity_types": ["PERSON", "ORG"]}},
        }
        for i in range(20000)
    ]
    ds = get_train.build_extraction_dataset(rows)  # prompts already strings
    assert len(ds) == 20000
    shuffled = ds.shuffle(seed=0)  # the fingerprinting path that died at 515k
    assert len(shuffled) == 20000
