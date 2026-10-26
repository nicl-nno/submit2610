'Tests for ``run_anchor_scoring`` (anchor breakdown counters).\n\n\nOffline with fakes: a fake model/tokenizer with scripted greedy generations\n(the BATCHED path — prompts of unique lengths, generations keyed by prompt\nlength, real left-padding semantics) and a fake RewardService returning\nscripted ``ScoredMetrics``. Covers the refined AC-failure classification\n(real judge failures / dispatcher crashes vs structural N/A skips for SGR\nreasoning turns), the ``final_unparsed``/``n_final`` counters for\nfinal-answer turns without a valid ``finalanswertool`` call, and the\nbatched anchor generation (multi-package packing under a small KV budget\nre-assembles the requests in the ORIGINAL order).'

from __future__ import annotations

import importlib.util
import sys

import pytest
from pathlib import Path
from types import SimpleNamespace

from llm_training.reward.config import ScoredMetrics
from llm_training.reward.dispatch import fallback_scored_metrics

_SCRIPT = Path(__file__).resolve().parents[2] / "grpo_train.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_train_anchor_run", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tr = _load_module()

# A valid finalanswertool call (extract_final_answer finds arguments.answer).

_FINAL_OK = (
    '<tool_call>\n{"name": "finalanswertool", "arguments": '
    '{"reasoning": "r", "answer": "ok final answer"}}\n</tool_call>'
)


class _FakeIds:
    'Indexable (B, L) char-id rows; iteration yields rows (legacy protocol).'

    def __init__(self, rows: list[list[str]]) -> None:
        self.rows = rows

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def __getitem__(self, i: int) -> list[str]:
        return self.rows[i]


class _FakeBatch(dict):
    'Tokenizer batch: a mapping (for ``**inputs``) with a no-op ``.to()``.'

    def __init__(self, input_ids: _FakeIds) -> None:
        super().__init__(input_ids=input_ids)

    def to(self, device: str) -> "_FakeBatch":
        return self


class _FakeTokenizer:
    'Char-tokenizes texts, pads by ``padding_side``, scripts generations.\n\n    ``<tool_call>``'

    pad_token_id = "\x00"
    eos_token_id = "\x01"

    def __init__(self, gens_by_len: dict[int, str] | None = None) -> None:
        self.padding_side = "right"
        self.padding_sides_seen: list[str] = []
        self._gens_by_len = gens_by_len or {}

    def __call__(self, texts, return_tensors: str | None = None,
                 padding: bool = False, add_special_tokens: bool = False,
                 ) -> _FakeBatch:
        if isinstance(texts, str):
            texts = [texts]
        rows = [list(t) for t in texts]
        if padding:
            self.padding_sides_seen.append(self.padding_side)
            width = max(len(r) for r in rows)
            if self.padding_side == "left":
                rows = [
                    [self.pad_token_id] * (width - len(r)) + r for r in rows
                ]
            else:
                rows = [
                    r + [self.pad_token_id] * (width - len(r)) for r in rows
                ]
        return _FakeBatch(_FakeIds(rows))

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(ids)

    def gen_for(self, prompt_len: int) -> list[str]:
        return list(self._gens_by_len[prompt_len])


class _FakeOutput:
    def __init__(self, rows: list[list[str]]) -> None:
        self._rows = rows

    def __getitem__(self, i: int) -> list[str]:
        return self._rows[i]


class _FakeModel:
    def __init__(self, tokenizer: _FakeTokenizer) -> None:
        self._tokenizer = tokenizer
        self.config = SimpleNamespace(use_cache=None)
        self.device = "cpu"
        self.modes: list[str] = []
        self.batch_sizes: list[int] = []

    def eval(self) -> None:
        self.modes.append("eval")

    def train(self) -> None:
        self.modes.append("train")

    def generate(self, input_ids: _FakeIds | None = None, **kwargs
                 ) -> _FakeOutput:
        pad = kwargs.get("pad_token_id")
        rows = input_ids.rows if input_ids is not None else []
        self.batch_sizes.append(len(rows))
        out_rows: list[list[str]] = []
        for row in rows:
            real_len = sum(1 for t in row if t != pad)
            out_rows.append(list(row) + self._tokenizer.gen_for(real_len))
        return _FakeOutput(out_rows)


class _FakeRewardService:
    'Scores each request by a scripted rule over its ``extra`` metadata.'

    def __init__(self, rule) -> None:
        self._rule = rule
        self.requests: list = []

    def score_batch(self, requests: list) -> list[ScoredMetrics]:
        self.requests = list(requests)
        return [self._rule(r) for r in requests]


def _example(extra: dict, source: str) -> SimpleNamespace:
    return SimpleNamespace(
        task_class="summarizer",
        prompt=[{"role": "user", "content": "task"}],
        reference="ref",
        extra=extra,
        source=source,
    )


def _wire(examples, gens) -> tuple[_FakeTokenizer, _FakeModel, dict[str, str]]:
    'Unique prompt length per example; generations keyed by that length.'
    prompt_of = {ex.source: "p" * (3 + i) for i, ex in enumerate(examples)}
    gens_by_len = {
        len(prompt_of[ex.source]): g for ex, g in zip(examples, gens)
    }
    tokenizer = _FakeTokenizer(gens_by_len)
    return tokenizer, _FakeModel(tokenizer), prompt_of


class TestRunAnchorScoring:
    def test_counters_classify_failures_skips_and_unparsed_finals(
        self, monkeypatch, caplog,
    ):
        # 5 examples: reasoning skip / action ok / final ok / final unparsed /
        # action with a dispatcher crash. The old code counted BOTH the
        # reasoning skip and — via None — nothing for the crash correctly,
        # but reported the structural skip as a "failure"; the refined
        # counters must separate them.
        
        
        
        examples = [
            _example({"is_reasoning_turn": True, "is_call_turn": False}, "s0"),
            _example({"is_call_turn": True}, "s1"),
            _example({"is_call_turn": True, "is_final_answer": True}, "s2"),
            _example({"is_call_turn": True, "is_final_answer": True}, "s3"),
            _example({"is_call_turn": True}, "s4"),
        ]
        gens = [
            "Reasoning steps: ...",
            '<tool_call>\n{"name": "websearchtool", "arguments": {}}\n</tool_call>',
            _FINAL_OK,
            "truncated generation without any tool call",
            '<tool_call>\n{"name": "websearchtool", "arguments": {}}\n</tool_call>',
        ]

        def _rule(req) -> ScoredMetrics:
            e = req.extra or {}
            if e.get("is_reasoning_turn"):
                # Structurally N/A: AC never attempted, no failure marker.
                
                return ScoredMetrics(
                    semantic_similarity=None, combined=0.5,
                    failed_metrics=(),
                )
            if req.source == "s4":
                return fallback_scored_metrics()  # dispatcher crash
            if req.source == "s3":
                return ScoredMetrics(
                    semantic_similarity=0.1, combined=0.1,
                    answer_correctness=0.1,
                )
            return ScoredMetrics(
                semantic_similarity=0.9, combined=0.9,
                answer_correctness=0.9 if req.source == "s1" else 0.8,
            )

        tokenizer, model, prompt_of = _wire(examples, gens)
        service = _FakeRewardService(_rule)
        monkeypatch.setattr(
            tr, "render_prompt", lambda ex, tok: prompt_of[ex.source],
        )

        with caplog.at_level("WARNING"):
            breakdown = tr.run_anchor_scoring(
                model, tokenizer, examples, object(), service,
                max_new_tokens=64,
            )

        assert breakdown["n"] == 5.0
        assert breakdown["ac_failed"] == 1.0  # the dispatcher crash only
        assert breakdown["ac_skipped"] == 1.0  # the SGR reasoning turn
        assert breakdown["n_final"] == 2.0
        assert breakdown["final_unparsed"] == 1.0  # s3 has no tool call
        # AC mean over the scored examples only: (0.9 + 0.8 + 0.1) / 3.
        assert abs(breakdown["ac"] - 0.6) < 1e-9
        # The envelope-preserving decode forwards the scripted generations.
        
        assert [r.completion for r in service.requests] == gens
        assert model.modes == ["eval", "train"]
        # The default budget admits all 5 equal-ish prompts into ONE
        # LEFT-padded package.
        
        
        assert model.batch_sizes == [5]
        assert tokenizer.padding_sides_seen == ["left"]
        assert tokenizer.padding_side == "right"  # restored in finally
        # No ALL-failed / ALL-unparsed warnings in this mixed scenario.
        
        assert not [r for r in caplog.records if "suspect" in r.message]
        assert not [
            r for r in caplog.records if "finalanswertool call" in r.message
        ]

    def test_warns_when_all_finals_unparsed(self, monkeypatch, caplog):
        # A systematically truncated budget (every final turn unparsed) must
        # warn: their AC is scored against an EMPTY answer.
        
        
        examples = [
            _example({"is_call_turn": True, "is_final_answer": True}, "s0"),
        ]
        gens = ["cut off mid-json..."]
        tokenizer, model, prompt_of = _wire(examples, gens)
        service = _FakeRewardService(
            lambda req: ScoredMetrics(
                semantic_similarity=0.0, combined=0.0, answer_correctness=0.0,
            )
        )
        monkeypatch.setattr(
            tr, "render_prompt", lambda ex, tok: prompt_of[ex.source],
        )
        with caplog.at_level("WARNING"):
            breakdown = tr.run_anchor_scoring(
                model, tokenizer, examples, object(), service,
                max_new_tokens=64,
            )
        assert breakdown["final_unparsed"] == 1.0
        assert breakdown["n_final"] == 1.0
        assert any(
            "--anchor-max-new-tokens" in r.message for r in caplog.records
        )

    def test_val_measure_full_coverage_without_semsim_fluency(
        self, monkeypatch,
    ):
        # The ctrl-point selection measure `val`: every example contributes
        # via its judge-side/structural metrics ONLY (rubrics for reasoning
        # turns, AC + format_check + code_check for action/final turns);
        # SemSim and small-fluency are present but must stay OUT of val.
        
        # reasoning; AC + format + code
        # action/final); SemSim
        
        examples = [
            _example({"is_reasoning_turn": True, "is_call_turn": False}, "r0"),
            _example({"is_call_turn": True}, "a0"),
            _example({"is_call_turn": True, "is_final_answer": True}, "f0"),
            _example({"is_call_turn": True, "is_final_answer": True}, "f1"),
            _example({"is_reasoning_turn": True, "is_call_turn": False}, "r1"),
        ]
        gens = [
            "Reasoning steps: ...",
            '<tool_call>\n{"name": "websearchtool", "arguments": {}}\n</tool_call>',
            _FINAL_OK,
            "truncated generation without any tool call",
            "Reasoning steps: ...",
        ]

        def _rule(req) -> ScoredMetrics:
            s = req.source
            if s == "r0":  # reasoning, rubric ok
                return ScoredMetrics(
                    semantic_similarity=None, combined=0.8,
                    rubric_score=0.8, fluency=0.9,
                )
            if s == "a0":  # action: AC + format; SemSim must NOT enter val
                return ScoredMetrics(
                    semantic_similarity=0.99, combined=0.7,
                    answer_correctness=0.6, format_score=0.9, fluency=0.95,
                )
            if s == "f0":  # final: AC + format
                return ScoredMetrics(
                    semantic_similarity=0.8, combined=0.8,
                    answer_correctness=0.8, format_score=1.0,
                )
            if s == "f1":  # broken final: format 0 → val_i = 0 (strict)
                return ScoredMetrics(
                    semantic_similarity=0.1, combined=0.0,
                    answer_correctness=0.1, format_score=0.0,
                )
            # r1: rubric attempted and failed → drops out of the val sample
            return ScoredMetrics(
                semantic_similarity=None, combined=0.0,
                failed_metrics=("rubrics",),
            )

        tokenizer, model, prompt_of = _wire(examples, gens)
        service = _FakeRewardService(_rule)
        monkeypatch.setattr(
            tr, "render_prompt", lambda ex, tok: prompt_of[ex.source],
        )
        breakdown = tr.run_anchor_scoring(
            model, tokenizer, examples, object(), service,
            max_new_tokens=64,
        )
        # Per-example val: harmonic(rubric=0.8)=0.8; harmonic(0.6,0.9)≈0.72;
        # harmonic(0.8,1.0)=0.8̄; harmonic(0.1,0.0)=0; r1 dropped.
        h_action = 2.0 / (1.0 / 0.6 + 1.0 / 0.9)
        h_final = 2.0 / (1.0 / 0.8 + 1.0 / 1.0)
        expected = (0.8 + h_action + h_final + 0.0) / 4.0
        assert breakdown["n_val"] == 4.0  # the failed-rubric example dropped
        assert breakdown["rubric_failed"] == 1.0
        assert breakdown["rubric"] == pytest.approx(0.8)
        assert breakdown["val"] == pytest.approx(expected)
        # SemSim/fluency means still reported as diagnostics.
        
        assert breakdown["semsim"] == pytest.approx((0.99 + 0.8 + 0.1) / 3.0)
        assert breakdown["ac"] == pytest.approx((0.6 + 0.8 + 0.1) / 3.0)

    def test_small_budget_splits_packages_keeps_request_order(
        self, monkeypatch,
    ):
        # A KV budget admitting at most 2 rows per package must still
        # re-assemble the requests in the ORIGINAL example order (the
        # generation inside sorts by prompt length — the anchor aggregates
        # must never see that order).
        
        
        
        
        examples = [_example({"is_call_turn": True}, f"s{i}") for i in range(4)]
        gens = [f"answer #{i}" for i in range(4)]
        tokenizer, model, prompt_of = _wire(examples, gens)
        service = _FakeRewardService(
            lambda req: ScoredMetrics(
                semantic_similarity=0.5, combined=0.5, answer_correctness=0.5,
            )
        )
        monkeypatch.setattr(
            tr, "render_prompt", lambda ex, tok: prompt_of[ex.source],
        )
        # Prompt lengths 3..6, budget 64: (6 + 8) * 2 = 28 <= 64 but
        # (6 + 8) * 3 = 42 <= 64 too... pick a budget that caps rows at 2:
        # (6 + 8) * 3 = 42 → need budget < 42 and >= 28: 32.
        breakdown = tr.run_anchor_scoring(
            model, tokenizer, examples, object(), service,
            max_new_tokens=8, gen_batch_tokens=32, gen_max_batch=8,
        )
        assert breakdown["n"] == 4.0
        assert model.batch_sizes == [2, 2]
        assert [r.completion for r in service.requests] == gens
        assert [r.source for r in service.requests] == [
            "s0", "s1", "s2", "s3",
        ]

    def test_legacy_budget_one_example_per_package(self, monkeypatch):
        # gen_batch_tokens=0 = the legacy escape: per-example packages in
        # the ORIGINAL order.
        # gen_batch_tokens=0
        examples = [_example({"is_call_turn": True}, f"s{i}") for i in range(3)]
        gens = [f"answer #{i}" for i in range(3)]
        tokenizer, model, prompt_of = _wire(examples, gens)
        service = _FakeRewardService(
            lambda req: ScoredMetrics(
                semantic_similarity=0.5, combined=0.5, answer_correctness=0.5,
            )
        )
        monkeypatch.setattr(
            tr, "render_prompt", lambda ex, tok: prompt_of[ex.source],
        )
        breakdown = tr.run_anchor_scoring(
            model, tokenizer, examples, object(), service,
            max_new_tokens=8, gen_batch_tokens=0,
        )
        assert breakdown["n"] == 3.0
        assert model.batch_sizes == [1, 1, 1]
        assert [r.completion for r in service.requests] == gens
