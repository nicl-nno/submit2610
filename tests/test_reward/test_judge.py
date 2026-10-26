'Tests for reward/judge.py — RAGASScorer retry, failure recording, wiring.\n\n\nThese tests exercise the **real** :class:`RAGASScorer` logic offline (no RAGAS\nimported, no GPU, no network) by injecting fake metrics via the ``metrics``\nDI-seam of :meth:`RAGASScorer.__init__`. The wiring tests\n(:func:`_build_metrics_from_ragas`) monkeypatch the lazy-import helpers.'

from __future__ import annotations

import asyncio
import json

import pytest

from llm_training.reward import judge as judge_mod
from llm_training.reward.config import RewardConfig
from llm_training.reward.judge import (
    CachingEmbeddings,
    RAGASScorer,
    _build_metrics_from_ragas,
    _clean_judge_response_text,
    _maybe_repair_json_escapes,
    _repair_invalid_json_escapes,
    extract_json_payload,
    make_ragas_scorer,
    sanitize_judge_client_json_fences,
    strip_json_code_fences,
    strip_think_blocks,
)
from llm_training.runtime.endpoints import OpenAIEndpoint

# pytest-asyncio is in strict mode (see pyproject). Only the two classes that
# hold async tests are marked at class level; the sync wiring class stays bare.
# pytest-asyncio
pytestmark_async = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _Result:
    'Stand-in for a RAGAS result object exposing ``.value``.'

    def __init__(self, value: float) -> None:
        self.value = value


class _FakeMetric:
    'Fake RAGAS metric: a configurable queue of outcomes per ``ascore`` call.\n\n\n    Each ``ascore`` call pops the next item from ``outcomes``: a ``float``\n    yields ``_Result(value)``; an ``Exception`` instance is raised. Tracks\n    ``call_count`` so tests can assert how many attempts were made, and\n    ``calls_kwargs`` so tests can inspect what each call received.\n    ``_Result(value)``; ``Exception``'

    def __init__(self, outcomes: list) -> None:
        self._outcomes = list(outcomes)
        self.call_count = 0
        self.calls_kwargs: list[dict] = []

    async def ascore(self, **kwargs):
        self.call_count += 1
        self.calls_kwargs.append(kwargs)
        if not self._outcomes:
            raise RuntimeError("no more outcomes configured")
        item = self._outcomes.pop(0)
        if isinstance(item, BaseException):
            raise item
        return _Result(item)


class _FakeCompletions:
    async def create(self, **kwargs):  # noqa: D102
        return None


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeJudgeClient:
    'Minimal AsyncOpenAI-shaped client (``chat.completions.create``).\n\n    ``sanitize_judge_client_json_fences``'

    def __init__(self) -> None:
        self.chat = _FakeChat()


def _judge() -> OpenAIEndpoint:
    return OpenAIEndpoint(base_url="http://judge/v1", model="j")


def _cfg(
    *,
    max_retries: int = 2,
    max_tokens: int = 4096,
    with_judge: bool = True,
    known_tool_names: frozenset[str] = frozenset(),
) -> RewardConfig:
    'Minimal RewardConfig (in-process embeddings default to bge-m3).'
    return RewardConfig(
        judge=_judge() if with_judge else None,
        judge_max_retries=max_retries,
        judge_max_tokens=max_tokens,
        known_tool_names=known_tool_names,
    )


def _scorer(
    semsim: _FakeMetric,
    ac: _FakeMetric | None,
    *,
    max_retries: int = 2,
    known_tool_names: frozenset[str] = frozenset(),
) -> RAGASScorer:
    'Build a RAGASScorer with the metrics injected (DI-seam).\n\n\n    The DI tuple is ``(ac, semsim, rubrics)`` — ac may be ``None`` to mirror\n    ``compute_answer_correctness=False``; rubrics default to ``None``\n    (``enable_judge_rubrics=False``).'
    return RAGASScorer(
        _cfg(max_retries=max_retries, known_tool_names=known_tool_names),
        metrics=(ac, semsim, None),
    )


# ---------------------------------------------------------------------------
# _ascore retry behavior
# ---------------------------------------------------------------------------
class TestAscoreRetry:
    pytestmark = pytestmark_async

    async def test_success_first_try(self):
        # Metric returns 0.7 on the first call → no retries, no sleep.
        
        metric = _FakeMetric([0.7])
        scorer = _scorer(_FakeMetric([0.5]), None)
        result = await scorer._ascore(
            metric, user_input="q", response="a", reference="r"
        )
        assert result == 0.7
        assert metric.call_count == 1

    async def test_retries_then_succeeds(self):
        # Two failures then success; max_retries=3 → 3 calls total.
        
        metric = _FakeMetric([
            RuntimeError("boom"), RuntimeError("boom"), 0.9,
        ])
        scorer = _scorer(_FakeMetric([0.5]), None, max_retries=3)
        result = await scorer._ascore(
            metric, user_input="q", response="a", reference="r"
        )
        assert result == 0.9
        assert metric.call_count == 3

    async def test_exhausts_retries_returns_none(self):
        # Always fails; max_retries=2 → 3 attempts (1 + 2 retries), None.
        
        metric = _FakeMetric([
            RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom"),
        ])
        scorer = _scorer(_FakeMetric([0.5]), None, max_retries=2)
        result = await scorer._ascore(
            metric, user_input="q", response="a", reference="r"
        )
        assert result is None
        assert metric.call_count == 3

    async def test_zero_retries_single_attempt(self):
        # max_retries=0 → exactly 1 attempt, no sleep, None on failure.
        # max_retries=0 →
        metric = _FakeMetric([RuntimeError("boom")])
        scorer = _scorer(_FakeMetric([0.5]), None, max_retries=0)
        result = await scorer._ascore(
            metric, user_input="q", response="a", reference="r"
        )
        assert result is None
        assert metric.call_count == 1

    async def test_backoff_is_exponential(self, monkeypatch):
        # Verify asyncio.sleep is called with 2^0, 2^1, 2^2 between attempts.
        
        sleep_calls: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        metric = _FakeMetric([
            RuntimeError("boom"), RuntimeError("boom"),
            RuntimeError("boom"), RuntimeError("boom"),
        ])
        scorer = _scorer(_FakeMetric([0.5]), None, max_retries=3)
        await scorer._ascore(metric, user_input="q", response="a", reference="r")
        # 4 attempts (max_retries=3); 3 sleeps before attempts 2, 3, 4.
        
        assert sleep_calls == [1, 2, 4]
        assert metric.call_count == 4

    async def test_no_sleep_on_success(self, monkeypatch):
        # A first-try success must not trigger any sleep.
        
        sleep_calls: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        metric = _FakeMetric([0.5])
        scorer = _scorer(_FakeMetric([0.5]), None, max_retries=3)
        await scorer._ascore(metric, user_input="q", response="a", reference="r")
        assert sleep_calls == []


# ---------------------------------------------------------------------------
# score() behavior
# ---------------------------------------------------------------------------
class TestScoreBehavior:
    pytestmark = pytestmark_async

    async def test_semsim_always_called(self):
        # Even with ac=None (no judge), SemSim is computed.
        
        semsim = _FakeMetric([0.8])
        scorer = _scorer(semsim, None)
        m = await scorer.score(
            user_input="q", response="a", reference="r", task_class="menon_qa"
        )
        assert m.semantic_similarity == 0.8
        assert m.answer_correctness is None
        assert m.format_score is None  # menon_qa is non-tool-use
        assert m.combined == pytest.approx(0.8)  # only SemSim active → harmonic of one
        assert m.failed_metrics == ()
        assert semsim.call_count == 1

    async def test_semsim_failure_recorded(self):
        semsim = _FakeMetric([RuntimeError("boom")])
        ac = _FakeMetric([0.5])
        scorer = _scorer(semsim, ac)
        m = await scorer.score(
            user_input="q", response="a", reference="r", task_class="menon_qa"
        )
        assert m.semantic_similarity == 0.0  # safe_score(None)
        assert m.failed_metrics == ("semantic_similarity",)
        # Combined → harmonic of [0.0, 0.5] → 0 (zero dominates).
        
        assert m.combined == 0.0

    async def test_ac_called_when_judge_configured(self):
        # With ac injected (non-None), both metrics run.
        
        semsim = _FakeMetric([0.7])
        ac = _FakeMetric([0.9])
        scorer = _scorer(semsim, ac)
        m = await scorer.score(
            user_input="q", response="a", reference="r", task_class="menon_qa"
        )
        assert m.semantic_similarity == 0.7
        assert m.answer_correctness == 0.9
        # Harmonic mean of [0.7, 0.9] = 2/(1/0.7 + 1/0.9) ≈ 0.7875
        
        assert m.combined == pytest.approx(2 / (1 / 0.7 + 1 / 0.9))
        assert ac.call_count == 1

    async def test_ac_failure_recorded(self):
        semsim = _FakeMetric([0.7])
        ac = _FakeMetric([RuntimeError("ac boom")])
        scorer = _scorer(semsim, ac)
        m = await scorer.score(
            user_input="q", response="a", reference="r", task_class="menon_qa"
        )
        assert m.failed_metrics == ("answer_correctness",)
        assert m.answer_correctness is None
        # SemSim succeeded → combined = harmonic of [0.7] = 0.7.
        
        assert m.combined == pytest.approx(0.7)

    async def test_format_check_runs_for_tool_use(self):
        # tool_calls_multiturn with a well-formed tool_call → format_score 1.0.
        # tool_calls_multiturn
        semsim = _FakeMetric([0.7])
        ac = _FakeMetric([0.9])
        scorer = _scorer(
            semsim, ac, known_tool_names=frozenset({"read"}),
        )
        response = '<tool_call>{"name": "read", "arguments": {"path": "x"}}</tool_call>'
        m = await scorer.score(
            user_input="q", response=response, reference="r",
            task_class="tool_calls_multiturn",
        )
        assert m.format_score == pytest.approx(1.0)
        # Harmonic mean of [0.7, 0.9, 1.0].
        
        assert m.combined == pytest.approx(3 / (1 / 0.7 + 1 / 0.9 + 1 / 1.0))

    async def test_invalid_tool_call_zeroes_combined(self):
        # An invalid tool_call → format_score = 0 → harmonic mean = 0.
        
        semsim = _FakeMetric([0.9])
        ac = _FakeMetric([0.9])
        scorer = _scorer(semsim, ac)
        response = "no tool call here"  # no <tool_call> block
        m = await scorer.score(
            user_input="q", response=response, reference="r",
            task_class="opencode_gen",
        )
        assert m.format_score == 0.0
        assert m.combined == 0.0

    async def test_both_metrics_fail_recorded(self):
        semsim = _FakeMetric([RuntimeError("semsim boom")])
        ac = _FakeMetric([RuntimeError("ac boom")])
        scorer = _scorer(semsim, ac)
        m = await scorer.score(
            user_input="q", response="a", reference="r", task_class="coder"
        )
        assert set(m.failed_metrics) == {"semantic_similarity", "answer_correctness"}
        assert m.combined == 0.0

    async def test_final_answer_turn_skips_format_check(self):
        # A tool-use final-answer turn (text-only, no <tool_call>) must get
        # format_score=None (not 0.0) so the harmonic mean doesn't zero it.
        
        semsim = _FakeMetric([0.7])
        scorer = _scorer(semsim, None)
        m = await scorer.score(
            user_input="q", response="The answer is 42.",
            reference="42", task_class="opencode_gen",
            extra={"is_call_turn": False},
        )
        assert m.format_score is None  # not 0.0 — format_check skipped
        assert m.combined == pytest.approx(0.7)  # SemSim only

    async def test_final_turn_via_is_final_flag(self):
        # Same as above but using the is_final_turn flag (tool_calls_multiturn).
        semsim = _FakeMetric([0.7])
        scorer = _scorer(semsim, None)
        m = await scorer.score(
            user_input="q", response="Final text answer",
            reference="ref", task_class="tool_calls_multiturn",
            extra={"is_final_turn": True},
        )
        assert m.format_score is None
        assert m.combined == pytest.approx(0.7)

    async def test_call_turn_still_runs_format_check(self):
        # A tool-call turn must still run format_check (backward-compatible).
        # tool-call
        semsim = _FakeMetric([0.9])
        scorer = _scorer(semsim, None)
        response = '<tool_call>{"name": "read", "arguments": {"path": "x"}}</tool_call>'
        m = await scorer.score(
            user_input="q", response=response, reference="r",
            task_class="opencode_gen",
            extra={"is_call_turn": True},
        )
        assert m.format_score is not None  # format_check ran

    async def test_extra_none_backward_compatible(self):
        # When extra=None (not passed), format_check runs unconditionally.
        
        semsim = _FakeMetric([0.9])
        scorer = _scorer(semsim, None)
        m = await scorer.score(
            user_input="q", response="no tool call", reference="r",
            task_class="opencode_gen",
        )
        assert m.format_score == 0.0  # format_check ran, no tool_call → 0


# ---------------------------------------------------------------------------
# Fluency + code_check wiring in score()
# ---------------------------------------------------------------------------
class _FakeFluency:
    'Fake FluencyScorer: a configurable async ``score(response, task_class)``.\n\n\n    ``value`` is returned as-is (``None`` ⇒ N/A); ``raises`` makes it throw so\n    the failure-recording path can be exercised. Records every call.'

    def __init__(self, value: float | None = None, *, raises: bool = False) -> None:
        self.value = value
        self.raises = raises
        self.calls: list[tuple[str, str]] = []

    async def score(
        self, response: str, task_class: str, *, extra: dict | None = None,
    ) -> float | None:
        self.calls.append((response, task_class))
        if self.raises:
            raise RuntimeError("fluency boom")
        return self.value


def _scorer_with(
    semsim: _FakeMetric,
    ac: _FakeMetric | None,
    fluency: _FakeFluency | None,
    *,
    enable_fluency: bool = True,
    enable_code_check: bool = True,
    known_tool_names: frozenset[str] = frozenset(),
    rubrics: tuple[_FakeMetric, _FakeMetric] | None = None,
) -> RAGASScorer:
    'Build a RAGASScorer with injected metrics AND a fake fluency scorer.'
    cfg = RewardConfig(
        judge=None,
        enable_fluency=enable_fluency,
        enable_code_check=enable_code_check,
        known_tool_names=known_tool_names,
    )
    return RAGASScorer(
        cfg, metrics=(ac, semsim, rubrics), fluency_scorer=fluency,
    )


class TestStructuralSignalsWiring:
    pytestmark = pytestmark_async

    async def test_code_check_coder_valid_python(self):
        # coder answer with a clean Python block → code_check=1.0 active.
        # is_call_turn=False isolates code_check (skips format_check, which would
        # otherwise expect a <tool_call> for the now-tool-use coder class).
        
        # code_check (format_check
        scorer = _scorer_with(_FakeMetric([0.7]), None, _FakeFluency(value=None))
        response = "Решение:\n```python\ndef f():\n    return 1\n```"
        m = await scorer.score(
            user_input="q", response=response, reference="r", task_class="coder",
            extra={"is_call_turn": False},
        )
        assert m.code_check == pytest.approx(1.0)
        assert m.fluency is None
        assert m.format_score is None  # format_check skipped
        # combined = harmonic([0.7, 1.0]).
        assert m.combined == pytest.approx(2 / (1 / 0.7 + 1 / 1.0))

    async def test_code_check_broken_python_zeroes_combined(self):
        scorer = _scorer_with(_FakeMetric([0.9]), None, _FakeFluency(value=None))
        response = "```python\ndef (\n```"
        m = await scorer.score(
            user_input="q", response=response, reference="r", task_class="coder",
            extra={"is_call_turn": False},
        )
        assert m.code_check == 0.0
        assert m.combined == 0.0  # harmonic with a zero → zero

    async def test_code_check_non_python_coder_is_none(self):
        # A Rust-only coder answer: code_check=N/A, combined = SemSim only.
        
        scorer = _scorer_with(_FakeMetric([0.7]), None, _FakeFluency(value=None))
        response = "```rust\nfn main() {}\n```"
        m = await scorer.score(
            user_input="q", response=response, reference="r", task_class="coder",
            extra={"is_call_turn": False},
        )
        assert m.code_check is None
        assert m.combined == pytest.approx(0.7)

    async def test_code_check_disabled(self):
        scorer = _scorer_with(
            _FakeMetric([0.7]), None, _FakeFluency(value=None),
            enable_code_check=False,
        )
        response = "```python\ndef f():\n    return 1\n```"
        m = await scorer.score(
            user_input="q", response=response, reference="r", task_class="coder",
            extra={"is_call_turn": False},
        )
        assert m.code_check is None

    async def test_fluency_active_for_nl_class(self):
        # menon_qa (NL allowlist): fake fluency=0.8 → active in harmonic.
        scorer = _scorer_with(_FakeMetric([0.7]), None, _FakeFluency(value=0.8))
        m = await scorer.score(
            user_input="q", response="Ответ из двух предложений.",
            reference="r", task_class="menon_qa",
        )
        assert m.fluency == pytest.approx(0.8)
        assert m.code_check is None  # not coder
        assert m.combined == pytest.approx(2 / (1 / 0.7 + 1 / 0.8))

    async def test_fluency_failure_recorded(self):
        # A model failure → "fluency" in failed_metrics, fluency=None, combined
        # falls back to SemSim only.
        
        scorer = _scorer_with(_FakeMetric([0.7]), None, _FakeFluency(raises=True))
        m = await scorer.score(
            user_input="q", response="Ответ.", reference="r", task_class="menon_qa"
        )
        assert m.fluency is None
        assert "fluency" in m.failed_metrics
        assert m.combined == pytest.approx(0.7)

    async def test_fluency_disabled_skips_signal(self):
        scorer = _scorer_with(
            _FakeMetric([0.7]), None, None, enable_fluency=False,
        )
        m = await scorer.score(
            user_input="q", response="Ответ.", reference="r", task_class="menon_qa"
        )
        assert m.fluency is None
        assert scorer._fluency is None  # no scorer built at all

    async def test_combined_with_semsim_ac_code_and_fluency(self):
        # A coder answer: SemSim + AC + code_check + fluency all active.
        # coder: SemSim + AC + code_check + fluency
        scorer = _scorer_with(
            _FakeMetric([0.7]), _FakeMetric([0.8]), _FakeFluency(value=0.9),
        )
        response = (
            "Объяснение. Ещё предложение.\n```python\ndef f():\n    return 1\n```"
        )
        m = await scorer.score(
            user_input="q", response=response, reference="r", task_class="coder",
            extra={"is_call_turn": False},
        )
        assert m.code_check == pytest.approx(1.0)
        assert m.fluency == pytest.approx(0.9)
        assert m.answer_correctness == pytest.approx(0.8)
        # harmonic of [0.7, 0.8, 1.0, 0.9].
        assert m.combined == pytest.approx(
            4 / (1 / 0.7 + 1 / 0.8 + 1 / 1.0 + 1 / 0.9)
        )


# ---------------------------------------------------------------------------
# NEREL F1 wiring (F1 replaces SemSim for extraction sub-configs) /

# ---------------------------------------------------------------------------
class TestNerelF1Wiring:
    pytestmark = pytestmark_async

    async def test_f1_replaces_semsim_for_extraction(self):
        # For entity_recognition SemSim is NOT computed at all (call_count==0);
        # F1 is the workhorse and combined == F1.
        
        # F1, combined == F1.
        semsim = _FakeMetric([0.7])
        scorer = _scorer_with(semsim, None, _FakeFluency(value=None))
        ref = "смерть\nЛетов\nсоветский"
        m = await scorer.score(
            user_input="q", response=ref, reference=ref,
            task_class="nerel_instruct", extra={"config": "entity_recognition"},
        )
        assert m.semantic_similarity is None
        assert semsim.call_count == 0  # SemSim skipped entirely
        assert m.nerel_f1 == pytest.approx(1.0)
        assert m.combined == pytest.approx(1.0)

    async def test_f1_partial_combined(self):
        semsim = _FakeMetric([0.7])
        scorer = _scorer_with(semsim, None, _FakeFluency(value=None))
        m = await scorer.score(
            user_input="q",
            response="смерть\nЛетов",
            reference="смерть\nЛетов\nсоветский",
            task_class="nerel_instruct", extra={"config": "relation_extraction"},
        )
        assert m.semantic_similarity is None
        assert m.nerel_f1 == pytest.approx(0.8)
        assert m.combined == pytest.approx(0.8)

    async def test_definition_subconfig_uses_semsim(self):
        # entity_definition is NOT an extraction config → SemSim runs, no F1.
        # entity_definition
        semsim = _FakeMetric([0.7])
        scorer = _scorer_with(semsim, None, _FakeFluency(value=None))
        m = await scorer.score(
            user_input="q", response="некоторый ответ", reference="эталон",
            task_class="nerel_instruct", extra={"config": "entity_definition"},
        )
        assert m.semantic_similarity == pytest.approx(0.7)
        assert m.nerel_f1 is None
        assert semsim.call_count == 1

    async def test_f1_disabled_falls_back_to_semsim(self):
        # With enable_nerel_f1=False the extraction config falls back to SemSim.
        
        semsim = _FakeMetric([0.7])
        scorer = _scorer_with(semsim, None, _FakeFluency(value=None))
        scorer._cfg = RewardConfig(
            judge=None, enable_fluency=True, enable_code_check=True,
            enable_nerel_f1=False,
        )
        m = await scorer.score(
            user_input="q", response="смерть\nЛетов", reference="смерть\nЛетов",
            task_class="nerel_instruct", extra={"config": "entity_recognition"},
        )
        assert m.semantic_similarity == pytest.approx(0.7)
        assert m.nerel_f1 is None


# ---------------------------------------------------------------------------
# DocSem check wiring (docsem_check replaces SemSim for docsem_quant) /

# ---------------------------------------------------------------------------
_DOCSEM_GOLD = {"gold_answer": "58", "gold_evidence": ["b07"]}
_DOCSEM_RESPONSE = (
    '{"reasoning": "Block b07: 98 total minus 40 donated.", '
    '"program": "total = 98\\ndonated = 40\\nanswer = total - donated", '
    '"answer": "58", "evidence": ["b07"]}'
)


class TestDocsemWiring:
    pytestmark = pytestmark_async

    async def test_docsem_check_replaces_semsim(self):
        # A perfect docsem response: SemSim is NOT computed at all
        # (call_count==0); docsem_check is the workhorse, combined == 1.0.
        
        # — docsem_check, combined == 1.0.
        semsim = _FakeMetric([0.7])
        scorer = _scorer_with(semsim, None, _FakeFluency(value=None))
        m = await scorer.score(
            user_input="q", response=_DOCSEM_RESPONSE, reference="ref",
            task_class="docsem_quant", extra=dict(_DOCSEM_GOLD),
        )
        assert semsim.call_count == 0  # SemSim skipped entirely
        assert m.semantic_similarity is None
        assert m.docsem_score == pytest.approx(1.0)
        assert m.combined == pytest.approx(1.0)
        assert m.format_score is None  # not a tool-use class
        assert m.code_check is None    # not coder
        assert m.fluency is None       # outside the NL allowlist

    async def test_docsem_wrong_answer_zeroes(self):
        scorer = _scorer_with(_FakeMetric([0.7]), None, _FakeFluency(value=None))
        m = await scorer.score(
            user_input="q", response=_DOCSEM_RESPONSE, reference="ref",
            task_class="docsem_quant",
            extra={"gold_answer": "40", "gold_evidence": ["b07"]},
        )
        assert m.docsem_score == 0.0
        assert m.combined == 0.0

    async def test_docsem_ac_sees_extracted_answer_and_gold(self):
        # With a judge: AC receives the EXTRACTED effective answer as the
        # response and the GOLD answer as the reference (not the JSON trace).
        
        
        ac = _FakeMetric([0.8])
        scorer = _scorer_with(_FakeMetric([0.7]), ac, _FakeFluency(value=None))
        m = await scorer.score(
            user_input="q", response=_DOCSEM_RESPONSE, reference="ref",
            task_class="docsem_quant", extra=dict(_DOCSEM_GOLD),
        )
        assert ac.call_count == 1
        assert ac.calls_kwargs[0]["response"] == "58"
        assert ac.calls_kwargs[0]["reference"] == "58"
        assert m.answer_correctness == pytest.approx(0.8)
        # harmonic of [docsem=1.0, ac=0.8].
        assert m.combined == pytest.approx(2 / (1 / 1.0 + 1 / 0.8))

    async def test_docsem_missing_gold_recorded_as_failure(self):
        # Missing gold metadata is a data bug → a failed metric (never a
        # silent skip): docsem_score=0.0 + "docsem_check" in failed_metrics.
        
        scorer = _scorer_with(_FakeMetric([0.7]), None, _FakeFluency(value=None))
        m = await scorer.score(
            user_input="q", response=_DOCSEM_RESPONSE, reference="ref",
            task_class="docsem_quant", extra={},
        )
        assert m.docsem_score == 0.0
        assert "docsem_check" in m.failed_metrics

    async def test_docsem_disabled_falls_back_to_semsim(self):
        # With enable_docsem_check=False the class falls back to SemSim; the
        # SemSim call receives the EXTRACTED answer vs the gold reference.
        
        semsim = _FakeMetric([0.7])
        scorer = _scorer_with(semsim, None, _FakeFluency(value=None))
        scorer._cfg = RewardConfig(
            judge=None, enable_fluency=True, enable_code_check=True,
            enable_docsem_check=False,
        )
        m = await scorer.score(
            user_input="q", response=_DOCSEM_RESPONSE, reference="ref",
            task_class="docsem_quant", extra=dict(_DOCSEM_GOLD),
        )
        assert semsim.call_count == 1
        assert semsim.calls_kwargs[0]["response"] == "58"
        assert semsim.calls_kwargs[0]["reference"] == "58"
        assert m.semantic_similarity == pytest.approx(0.7)
        assert m.docsem_score is None


# ---------------------------------------------------------------------------
# SGR-agent turn wiring (reasoning / action / final) /
# action
# ---------------------------------------------------------------------------
class TestSGRAgentWiring:
    pytestmark = pytestmark_async

    async def test_reasoning_turn_skips_semsim(self):
        # A reasoning turn: no SemSim (reasoning branch → reasoning_check),
        # format_check skipped (is_call_turn=False).
        
        # format_check
        semsim = _FakeMetric([0.9])
        scorer = _scorer_with(semsim, None, _FakeFluency(value=None))
        response = (
            "Reasoning steps:\n- a\nCurrent situation: x\n"
            "Plan status: y\nRemaining steps:\n- b"
        )
        m = await scorer.score(
            user_input="q", response=response, reference="ref",
            task_class="math_reasoner",
            extra={"is_reasoning_turn": True, "is_call_turn": False},
        )
        assert semsim.call_count == 0  # SemSim skipped for reasoning
        assert m.semantic_similarity is None
        assert m.format_score is None  # format_check skipped
        assert m.combined == pytest.approx(1.0)  # reasoning_check=1.0 (4/4 markers)

    async def test_action_turn_uses_semsim_and_format(self):
        semsim = _FakeMetric([0.7])
        scorer = _scorer_with(
            semsim, None, _FakeFluency(value=None),
            known_tool_names=frozenset({"pandas_analysis_tool"}),
        )
        response = (
            '<tool_call>\n{"name": "pandas_analysis_tool", '
            '"arguments": {"csv_path": "x.csv"}}</tool_call>'
        )
        m = await scorer.score(
            user_input="q", response=response, reference="ref",
            task_class="data_analyst", extra={"is_call_turn": True},
        )
        assert semsim.call_count == 1
        assert m.semantic_similarity == pytest.approx(0.7)
        # A well-formed tool_call with a known name → format_score > 0.
        
        assert m.format_score is not None and m.format_score > 0

    async def test_final_turn_extracts_answer(self):
        # Final turn: format_check validates finalanswertool on RAW response;
        # SemSim runs on the EXTRACTED answer, not the JSON wrapper.
        
        semsim = _FakeMetric([0.8])
        scorer = _scorer_with(
            semsim, None, _FakeFluency(value=None),
            known_tool_names=frozenset({"finalanswertool"}),
        )
        response = (
            '<tool_call>\n{"name": "finalanswertool", "arguments": '
            '{"answer": "ответ модели", "status": "completed"}}</tool_call>'
        )
        m = await scorer.score(
            user_input="q", response=response, reference="ответ",
            task_class="coder",
            extra={"is_call_turn": True, "is_final_answer": True},
        )
        assert m.format_score is not None and m.format_score > 0
        assert semsim.call_count == 1
        assert m.semantic_similarity == pytest.approx(0.8)

    async def test_final_turn_no_valid_call_zeroes_combined(self):
        # No valid finalanswertool → empty extracted answer + format_check=0
        # (a tool-use turn that emitted no <tool_call> is structurally broken).
        
        semsim = _FakeMetric([0.8])
        scorer = _scorer_with(
            semsim, None, _FakeFluency(value=None),
            known_tool_names=frozenset({"finalanswertool"}),
        )
        m = await scorer.score(
            user_input="q", response="мусор без tool-call", reference="ответ",
            task_class="coder",
            extra={"is_call_turn": True, "is_final_answer": True},
        )
        assert m.format_score == 0.0
        assert m.combined == 0.0  # harmonic with a zero

    async def test_reasoning_semsim_flag_enables_semsim(self):
        # With enable_reasoning_semsim=True the reasoning branch falls back to SemSim.
        
        semsim = _FakeMetric([0.6])
        scorer = _scorer_with(semsim, None, _FakeFluency(value=None))
        scorer._cfg = RewardConfig(
            judge=None, enable_fluency=True, enable_code_check=True,
            enable_reasoning_semsim=True,
        )
        m = await scorer.score(
            user_input="q", response="reasoning text", reference="ref",
            task_class="math_reasoner",
            extra={"is_reasoning_turn": True, "is_call_turn": False},
        )
        assert semsim.call_count == 1
        assert m.semantic_similarity == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Judge rubrics for SGR reasoning turns
# ---------------------------------------------------------------------------
class TestJudgeRubrics:
    'The judge-side rubric pair (literacy + context consistency).'

    pytestmark = pytestmark_async

    async def test_reasoning_turn_scores_both_criteria(self):
        # Likert 5 → 1.0, Likert 3 → 0.5; harmonic(1.0, 0.5) = 2/3. The
        # rubric metrics receive the rebuilt context as user_input.
        # Likert 5 → 1.0, Likert 3 → 0.5
        
        lit = _FakeMetric([5])
        ctx = _FakeMetric([3])
        scorer = _scorer_with(
            _FakeMetric([0.5]), None, _FakeFluency(value=None),
            rubrics=(lit, ctx),
        )
        prompt = [
            {"role": "system", "content": "Задача: суммаризировать статью."},
            {"role": "user", "content": "Current Date: 2026-06-29"},
            {"role": "tool", "content": "результат инструмента"},
        ]
        m = await scorer.score(
            user_input="Current Date: 2026-06-29",
            response=(
                "Reasoning steps:\n1. анализ\nCurrent situation: ясна\n"
                "Plan status: в работе\nRemaining steps:\n- финал"
            ),
            reference="ref",
            task_class="summarizer",
            extra={"is_reasoning_turn": True, "is_call_turn": False},
            prompt=prompt,
        )
        assert m.rubric_score == pytest.approx(2.0 / 3.0)
        assert lit.call_count == 1 and ctx.call_count == 1
        # The rubric user_input is the REBUILT context (system + user + tool
        # tail), not the thin extracted question.
        # user_input
        assert "Задача: суммаризировать статью." in lit.calls_kwargs[0]["user_input"]
        assert "результат инструмента" in lit.calls_kwargs[0]["user_input"]
        # Rubrics join the harmonic combine of the example: the response
        # carries all four SGR markers (reasoning_check=1.0), so combined is
        # harmonic(2/3, 1.0) = 0.8.
        
        # (reasoning_check=1.0), combined = harmonic(2/3, 1.0) = 0.8.
        assert m.combined == pytest.approx(0.8)

    async def test_non_reasoning_turn_skips_rubrics(self):
        # An action/final turn must NOT call the rubric metrics (AC is the
        # applicable judge metric there).
        # Action/final
        lit = _FakeMetric([5])
        ctx = _FakeMetric([5])
        scorer = _scorer_with(
            _FakeMetric([0.8]), None, _FakeFluency(value=None),
            rubrics=(lit, ctx),
        )
        m = await scorer.score(
            user_input="q", response="обычный ответ", reference="ref",
            task_class="menon_qa",
        )
        assert m.rubric_score is None
        assert lit.call_count == 0 and ctx.call_count == 0

    async def test_rubrics_none_disables_signal(self):
        # rubrics=None (enable_judge_rubrics=False) — no rubric on a
        # reasoning turn, no failure recorded (not applicable).
        # rubrics=None
        scorer = _scorer_with(
            _FakeMetric([0.5]), None, _FakeFluency(value=None),
        )
        m = await scorer.score(
            user_input="q", response="Reasoning steps: ...", reference="ref",
            task_class="summarizer",
            extra={"is_reasoning_turn": True, "is_call_turn": False},
        )
        assert m.rubric_score is None
        assert "rubrics" not in m.failed_metrics

    async def test_hard_one_likert_zeroes_the_harmonic(self):
        # Likert 1 → 0.0 — the strict harmonic mean zeroes the example (the
        # strong ctrl-point signal against illiterate/off-context reasoning).
        # Likert 1 → 0.0
        lit = _FakeMetric([1])
        ctx = _FakeMetric([5])
        scorer = _scorer_with(
            _FakeMetric([0.5]), None, _FakeFluency(value=None),
            rubrics=(lit, ctx),
        )
        m = await scorer.score(
            user_input="q", response="Reasoning steps: ...", reference="ref",
            task_class="summarizer",
            extra={"is_reasoning_turn": True, "is_call_turn": False},
        )
        assert m.rubric_score == 0.0
        assert m.combined == 0.0

    async def test_rubric_failure_recorded_not_fatal(self):
        # A failing rubric criterion → failed_metrics="rubrics",
        # rubric_score=None (excluded from the combine), scoring continues.
        
        lit = _FakeMetric([RuntimeError("judge down")])
        ctx = _FakeMetric([4])
        scorer = _scorer_with(
            _FakeMetric([0.5]), None, _FakeFluency(value=None),
            rubrics=(lit, ctx),
            # One retry then give up: keep the test fast.
            
        )
        scorer._max_retries = 0
        m = await scorer.score(
            user_input="q", response="Reasoning steps: ...", reference="ref",
            task_class="summarizer",
            extra={"is_reasoning_turn": True, "is_call_turn": False},
        )
        assert m.rubric_score is None
        assert "rubrics" in m.failed_metrics
        assert ctx.call_count == 1  # the second criterion still attempted


# ---------------------------------------------------------------------------
# _build_metrics_from_ragas wiring / wiring _build_metrics_from_ragas
# ---------------------------------------------------------------------------
class TestBuildMetricsWiring:
    "Verify _build_metrics_from_ragas forwards cfg settings to RAGAS/openai.\n\n    RAGAS/openai. Monkeypatch'"

    @staticmethod
    def _patch_ragas_imports(monkeypatch, *, llm_spy=None, embed_spy=None):
        'Replace the RAGAS lazy imports with no-op fakes.'
        class _FakeMetricCls:
            def __init__(self, **kwargs):
                pass

        def _fake_llm_factory(model, client=None, **kwargs):
            if llm_spy is not None:
                llm_spy(kwargs)
            return object()

        def _fake_embedding_factory(provider, model=None, client=None, **kwargs):
            if embed_spy is not None:
                embed_spy(kwargs)
            return object()

        monkeypatch.setattr(
            judge_mod, "_import_ragas_metrics",
            lambda: (_FakeMetricCls, _FakeMetricCls, _FakeMetricCls),
        )
        monkeypatch.setattr(
            judge_mod, "_import_ragas_factories",
            lambda: (_fake_llm_factory, _fake_embedding_factory),
        )

    def test_max_tokens_forwarded_to_llm_factory(self, monkeypatch):
        # llm_factory must receive max_tokens=cfg.judge_max_tokens.
        # llm_factory
        captured: dict = {}
        self._patch_ragas_imports(
            monkeypatch, llm_spy=lambda kw: captured.update(kw),
        )
        cfg = _cfg(max_tokens=8192)
        _build_metrics_from_ragas(cfg)
        assert captured.get("max_tokens") == 8192

    def test_max_retries_forwarded_to_judge_client(self, monkeypatch):
        # The judge async_client must be called with max_retries from cfg.
        # (Embeddings are in-process — no embedding HTTP client is built.)
        
        client_calls: list[tuple[str, int]] = []

        def _fake_async_client(self, *, max_retries: int = 2, timeout=None):
            client_calls.append((self.kind, max_retries))
            return _FakeJudgeClient()  # chat.completions.create-shaped

        monkeypatch.setattr(OpenAIEndpoint, "async_client", _fake_async_client)
        self._patch_ragas_imports(monkeypatch)

        cfg = _cfg(max_retries=5)
        _build_metrics_from_ragas(cfg)

        assert ("judge", 5) in client_calls
        assert len(client_calls) == 1  # only the judge client (embeddings are local)

    def test_timeout_forwarded_to_judge_client(self, monkeypatch):
        # The judge async_client must receive timeout from cfg.judge_timeout_s.
        
        client_calls: list[tuple[str, float | None]] = []

        def _fake_async_client(self, *, max_retries: int = 2, timeout=None):
            client_calls.append((self.kind, timeout))
            return _FakeJudgeClient()

        monkeypatch.setattr(OpenAIEndpoint, "async_client", _fake_async_client)
        self._patch_ragas_imports(monkeypatch)

        cfg = RewardConfig(
            judge=OpenAIEndpoint(base_url="http://judge/v1", model="j"),
            judge_timeout_s=300.0,
        )
        _build_metrics_from_ragas(cfg)

        assert ("judge", 300.0) in client_calls
        assert len(client_calls) == 1

    def test_no_judge_skips_llm_client(self, monkeypatch):
        # With no judge, no HTTP client is built at all (embeddings are
        # in-process; the judge is the only HTTP endpoint).
        
        client_calls: list[str] = []

        def _fake_async_client(self, *, max_retries: int = 2, timeout=None):
            client_calls.append(self.kind)
            return object()

        monkeypatch.setattr(OpenAIEndpoint, "async_client", _fake_async_client)
        self._patch_ragas_imports(monkeypatch)

        cfg = _cfg(with_judge=False)
        ac, _semsim, _rubrics = _build_metrics_from_ragas(cfg)
        assert ac is None  # no AC metric without a judge
        assert client_calls == []  # no HTTP clients (embeddings are local)

    def test_local_embedding_model_uses_huggingface_factory(self, monkeypatch):
        # _build_metrics_from_ragas must call embedding_factory("huggingface",
        # ...) — embeddings are always in-process now (this is the only path).
        
        factory_calls: list[tuple] = []

        def _fake_embedding_factory(provider, model=None, client=None, **kw):
            factory_calls.append((provider, model, kw))
            return object()

        def _fake_llm_factory(model, client=None, **kw):
            return object()

        class _FakeMetricCls:
            def __init__(self, **kwargs):
                pass

        monkeypatch.setattr(
            judge_mod, "_import_ragas_metrics",
            lambda: (_FakeMetricCls, _FakeMetricCls, _FakeMetricCls),
        )
        monkeypatch.setattr(
            judge_mod, "_import_ragas_factories",
            lambda: (_fake_llm_factory, _fake_embedding_factory),
        )

        cfg = RewardConfig(
            local_embedding_model="BAAI/bge-m3",
            local_embedding_device="cuda",
            judge=OpenAIEndpoint(base_url="http://judge/v1", model="j"),
        )
        _build_metrics_from_ragas(cfg)

        # Exactly one embedding_factory call, with provider="huggingface".
        
        assert len(factory_calls) == 1
        assert factory_calls[0][0] == "huggingface"
        assert factory_calls[0][1] == "BAAI/bge-m3"
        assert factory_calls[0][2]["device"] == "cuda"

    def test_default_model_used_when_unspecified(self, monkeypatch):
        # The default embedding model (BAAI/bge-m3) is forwarded to the factory.
        
        factory_calls: list[tuple] = []

        def _fake_embedding_factory(provider, model=None, client=None, **kw):
            factory_calls.append((provider, model))
            return object()

        def _fake_llm_factory(model, client=None, **kw):
            return object()

        class _FakeMetricCls:
            def __init__(self, **kwargs):
                pass

        monkeypatch.setattr(
            judge_mod, "_import_ragas_metrics",
            lambda: (_FakeMetricCls, _FakeMetricCls, _FakeMetricCls),
        )
        monkeypatch.setattr(
            judge_mod, "_import_ragas_factories",
            lambda: (_fake_llm_factory, _fake_embedding_factory),
        )

        cfg = RewardConfig()  # no judge, no explicit embedding model
        _build_metrics_from_ragas(cfg)
        assert factory_calls == [("huggingface", "BAAI/bge-m3")]


# ---------------------------------------------------------------------------
# CachingEmbeddings
# ---------------------------------------------------------------------------
class TestCachingEmbeddings:
    pytestmark = pytestmark_async

    async def test_cache_hit_avoids_recompute(self):
        # Second call with the same text returns the cached result; the base
        # object is hit only once.
        
        calls: list[str] = []

        class _FakeBase:
            async def aembed_text(self, text, **kw):
                calls.append(text)
                return [0.1, 0.2, 0.3]

        cache = CachingEmbeddings(_FakeBase())
        r1 = await cache.aembed_text("hello")
        r2 = await cache.aembed_text("hello")
        assert r1 == r2 == [0.1, 0.2, 0.3]
        assert len(calls) == 1

    async def test_different_texts_not_cross_cached(self):
        calls: list[str] = []

        class _FakeBase:
            async def aembed_text(self, text, **kw):
                calls.append(text)
                # Return a value derived from the text content itself (not from
                # hash(text), whose %100 can collide across short words — which
                # previously made this assertion flaky).
                
                return [float(ord(text[0]))]

        cache = CachingEmbeddings(_FakeBase())
        r1 = await cache.aembed_text("hello")  # 'h' = 104.0
        r2 = await cache.aembed_text("world")  # 'w' = 119.0
        assert r1 != r2
        assert len(calls) == 2

    async def test_concurrent_calls_deduplicated(self):
        # Two concurrent calls for the same text share one Future — the base
        # object is called only once.
        
        calls: list[str] = []

        class _FakeBase:
            async def aembed_text(self, text, **kw):
                calls.append(text)
                await asyncio.sleep(0.05)  # simulate latency
                return [0.5]

        cache = CachingEmbeddings(_FakeBase())
        results = await asyncio.gather(
            cache.aembed_text("same"),
            cache.aembed_text("same"),
        )
        assert results[0] == results[1] == [0.5]
        assert len(calls) == 1

    async def test_getattr_delegates_to_base(self):
        class _FakeBase:
            model = "bge-m3"
            async def aembed_text(self, text, **kw):
                return [0.0]

        cache = CachingEmbeddings(_FakeBase())
        assert cache.model == "bge-m3"

    async def test_concurrent_failure_propagates_exception_not_cancelled(self):
        # CRITICAL: when the base call fails, concurrent waiters must receive
        # the SAME exception (RuntimeError → Exception), NOT CancelledError
        # (BaseException). CancelledError would escape _ascore's
        # ``except Exception`` and abort the entire asyncio.gather batch.
        
        
        calls: list[str] = []

        class _FakeBase:
            async def aembed_text(self, text, **kw):
                calls.append(text)
                await asyncio.sleep(0.05)  # let the waiter arrive
                raise RuntimeError("connection error")

        cache = CachingEmbeddings(_FakeBase())
        results = await asyncio.gather(
            cache.aembed_text("same"),
            cache.aembed_text("same"),
            return_exceptions=True,
        )
        assert len(calls) == 1  # dedup: only one base call
        for r in results:
            # Both coroutines must receive RuntimeError (Exception), which
            # _ascore can catch and retry. NOT CancelledError (BaseException).
            
            
            assert isinstance(r, RuntimeError)
            assert str(r) == "connection error"
            assert not isinstance(r, asyncio.CancelledError)


# ---------------------------------------------------------------------------
# Semaphore behavior
# ---------------------------------------------------------------------------
class TestSemaphoreBehavior:
    pytestmark = pytestmark_async

    async def test_caching_embeddings_sem_gates_real_compute(self):
        # With sem=Semaphore(1), two concurrent calls for DIFFERENT texts (both
        # cache misses) must not overlap on the base computation. This is the
        # actual gate that protects in-process embedding GPU contention.
        
        concurrent = 0
        max_seen = 0

        class _TrackingBase:
            async def aembed_text(self, text, **kw):
                nonlocal concurrent, max_seen
                concurrent += 1
                max_seen = max(max_seen, concurrent)
                await asyncio.sleep(0.05)
                concurrent -= 1
                return [0.1, 0.2]

        cache = CachingEmbeddings(_TrackingBase(), sem=asyncio.Semaphore(1))
        await asyncio.gather(
            cache.aembed_text("alpha"),
            cache.aembed_text("beta"),  # different key → cache miss → gated
        )
        assert max_seen == 1

    async def test_caching_embeddings_cache_hit_skips_sem(self):
        # A cache hit returns instantly and never touches the semaphore — so a
        # hit + a concurrent miss do not contend (max_seen for the base is 1
        # because only one real call happens).
        
        concurrent = 0
        max_seen = 0

        class _TrackingBase:
            async def aembed_text(self, text, **kw):
                nonlocal concurrent, max_seen
                concurrent += 1
                max_seen = max(max_seen, concurrent)
                await asyncio.sleep(0.02)
                concurrent -= 1
                return [0.1]

        cache = CachingEmbeddings(_TrackingBase(), sem=asyncio.Semaphore(1))
        await cache.aembed_text("same")  # warm the cache
        await asyncio.gather(
            cache.aembed_text("same"),  # cache hit — no base call
            cache.aembed_text("other"),  # miss — one base call
        )
        assert max_seen == 1

    async def test_no_deadlock_when_semsim_and_ac_share_embed_sem(self):
        # With embedding_concurrency=1 and judge_concurrency=4, four concurrent
        # score() calls running SemSim+AC must not deadlock (AC's internal
        # embedding calls and SemSim's both gate on _embed_sem inside the shared
        # cache, AC's LLM call gates on _judge_sem).
        # emb=1, judge=4
        class _SlowSemSim:
            async def ascore(self, **kwargs):
                await asyncio.sleep(0.01)
                return _Result(0.7)

        class _FastAC:
            async def ascore(self, **kwargs):
                await asyncio.sleep(0.01)
                return _Result(0.9)

        cfg = RewardConfig(
            judge=OpenAIEndpoint(base_url="http://j/v1", model="j"),
            embedding_concurrency=1,
            judge_concurrency=4,
        )
        scorer = RAGASScorer(cfg, metrics=(_FastAC(), _SlowSemSim(), None))
        results = await asyncio.gather(*[
            scorer.score(
                user_input="q", response=f"r{i}", reference=f"ref{i}",
                task_class="menon_qa",
            )
            for i in range(4)
        ])
        assert all(r.combined > 0 for r in results)  # no deadlock

    async def test_semsim_and_ac_share_one_gated_cache(self):
        # In the production wiring path, SemSim and AC must share the SAME
        # CachingEmbeddings instance, and that instance must carry _embed_sem
        # (so AC's internal embedding calls are gated + cached too).
        
        class _MetricWithEmbeddings:
            def __init__(self, emb):
                self.embeddings = emb

            async def ascore(self, **kwargs):
                return _Result(0.5)

        cfg = RewardConfig(judge=OpenAIEndpoint(base_url="http://j/v1", model="j"))
        shared_base = object()
        scorer = RAGASScorer(
            cfg,
            metrics=(
                _MetricWithEmbeddings(shared_base),  # AC
                _MetricWithEmbeddings(shared_base),  # SemSim
                None,  # rubrics off
            ),
        )
        # Both metrics now point at the same wrapped instance.
        assert scorer._semsim.embeddings is scorer._ac.embeddings
        assert isinstance(scorer._semsim.embeddings, CachingEmbeddings)
        # The wrap carries the embedding semaphore.
        assert scorer._semsim.embeddings._sem is scorer._embed_sem

    async def test_injected_separate_embeddings_get_separate_wraps(self):
        # Test path where AC and SemSim were injected with DIFFERENT base
        # objects: each gets its own wrap (no silent cross-sharing), and only
        # the SemSim wrap carries the sem (production-path wiring is not
        # assumed for fakes).
        
        class _MetricWithEmbeddings:
            def __init__(self, emb):
                self.embeddings = emb

            async def ascore(self, **kwargs):
                return _Result(0.5)

        cfg = RewardConfig()
        scorer = RAGASScorer(
            cfg,
            metrics=(
                _MetricWithEmbeddings(object()),  # AC, distinct base
                _MetricWithEmbeddings(object()),  # SemSim, distinct base
                None,  # rubrics off
            ),
        )
        assert scorer._semsim.embeddings is not scorer._ac.embeddings
        assert isinstance(scorer._semsim.embeddings, CachingEmbeddings)
        assert isinstance(scorer._ac.embeddings, CachingEmbeddings)


# ---------------------------------------------------------------------------
# Real-import guard
# ---------------------------------------------------------------------------
class TestRagasImportContract:
    'Guard against the ragas 0.4.3 export-table drift (regression plomb).\n\n\n    In ragas 0.4.3 ``AnswerSimilarity`` is NOT exported from\n    ``ragas.metrics.collections`` (despite the DeprecationWarning advising\n    exactly that path). We switched to ``SemanticSimilarity`` (the base class,\n    behaviourally identical). This test calls the real ``_import_ragas_metrics``\n    — no monkeypatching — so any future drift in either direction (a ragas\n    upgrade removing ``SemanticSimilarity`` or this codebase reverting to\n    ``AnswerSimilarity``) fails here, in a quick offline test, instead of only\n    surfacing in the @gpu-marked e2e test that needs real judge/embedding\n    endpoints.\n\n    ``ragas.metrics.collections``\n    ``_import_ragas_metrics``'

    def test_ragas_metrics_imports_resolve(self):
        # Skip cleanly when ragas is not installed (e.g. CI without the `train`
        # extra). When ragas IS installed (as on developer machines and the
        # production env), the import must succeed and return the two classes
        # we wire up in _build_metrics_from_ragas.
        
        pytest.importorskip("ragas")
        from llm_training.reward.judge import _import_ragas_metrics

        ac_cls, semsim_cls, rubrics_cls = _import_ragas_metrics()
        # We rely on AnswerCorrectness by exact name.
        
        assert ac_cls.__name__ == "AnswerCorrectness"
        # The reference-free rubric metric (the legacy ``RubricsScore`` alias
        # in ``ragas.metrics`` is deprecated in 0.4.x).
        
        assert rubrics_cls.__name__ == "RubricsScoreWithoutReference"
        # SemanticSimilarity is the base of AnswerSimilarity; both are exported
        # from ragas.metrics.collections in 0.4.x. We accept either name so a
        # future ragas version that re-adds AnswerSimilarity to the export table
        # doesn't break this test as long as our import path still works.
        
        assert semsim_cls.__name__ in {"SemanticSimilarity", "AnswerSimilarity"}


# ---------------------------------------------------------------------------
# Shared instances (cross-scorer weight sharing)
# ---------------------------------------------------------------------------
class TestSharedInstances:
    'The ``base_embeddings`` / ``fluency_classifier`` sharing DI seams.'

    def test_base_embeddings_forwarded_to_wiring(self, monkeypatch):
        # The production path must pass base_embeddings through to
        # _build_metrics_from_ragas instead of building a second ST model;
        # the accessor unwraps the CachingEmbeddings wrapper back to the base.
        
        
        captured: dict = {}

        def _fake_build(cfg, *, base_embeddings=None):
            captured["base"] = base_embeddings
            semsim = _FakeMetric([0.5])
            semsim.embeddings = base_embeddings
            return None, semsim, None

        monkeypatch.setattr(judge_mod, "_build_metrics_from_ragas", _fake_build)
        fake_base = object()
        scorer = RAGASScorer(_cfg(), base_embeddings=fake_base)
        assert captured["base"] is fake_base
        assert scorer.base_embeddings is fake_base

    def test_fluency_classifier_forwarded(self):
        # A shared classifier instance must land inside the scorer's
        # FluencyScorer instead of triggering a second model load.
        
        
        semsim = _FakeMetric([0.5])
        shared = object()
        scorer = RAGASScorer(
            _cfg(), metrics=(None, semsim, None), fluency_classifier=shared,
        )
        assert scorer.fluency_classifier is shared

    def test_fluency_classifier_none_when_disabled(self):
        # No fluency scorer at all → the accessor is None (and must not crash).
        
        scorer = RAGASScorer(
            RewardConfig(judge=None, enable_fluency=False),
            metrics=(None, _FakeMetric([0.5]), None),
        )
        assert scorer.fluency_classifier is None

    def test_make_ragas_scorer_forwards_shared_instances(self, monkeypatch):
        # make_ragas_scorer must forward both sharing kwargs to RAGASScorer.
        # make_ragas_scorer
        def _fake_build(cfg, *, base_embeddings=None):
            semsim = _FakeMetric([0.5])
            semsim.embeddings = base_embeddings
            return None, semsim, None

        monkeypatch.setattr(judge_mod, "_build_metrics_from_ragas", _fake_build)
        fake_base = object()
        shared_clf = object()
        scorer = make_ragas_scorer(
            _cfg(), base_embeddings=fake_base, fluency_classifier=shared_clf,
        )
        assert scorer.base_embeddings is fake_base
        assert scorer.fluency_classifier is shared_clf


# ---------------------------------------------------------------------------
# Judge JSON-fence sanitizer
# ---------------------------------------------------------------------------
class TestFenceSanitizer:
    'The client-side markdown-fence stripper for judge JSON responses.'

    def test_strips_json_fenced_payload(self):
        # ```json { ... } ``` → { ... }
        fenced = '```json\n{"TP": [1], "FP": [], "FN": []}\n```'
        assert strip_json_code_fences(fenced) == '{"TP": [1], "FP": [], "FN": []}'

    def test_strips_bare_fence(self):
        fenced = '```\n{"a": 1}\n```'
        assert strip_json_code_fences(fenced) == '{"a": 1}'

    def test_plain_json_untouched(self):
        plain = '{"a": 1}'
        assert strip_json_code_fences(plain) is plain

    def test_fence_without_newline_untouched(self):
        # A degenerate single-line fence is left as-is (nothing to unwrap).
        
        assert strip_json_code_fences("```json{}") == "```json{}"

    def test_think_blocks_removed(self):
        # The DeepSeek-R1 style inline reasoning: <think>…</think> + JSON.
        
        assert (
            strip_think_blocks('<think>chain of thought</think>{"a": 1}')
            == '{"a": 1}'
        )

    def test_think_blocks_multiple_removed(self):
        assert (
            strip_think_blocks('<think>a</think>mid<think>b</think>{"a": 1}')
            == 'mid{"a": 1}'
        )

    def test_think_multiline_block_removed(self):
        assert (
            strip_think_blocks("<think>line1\nline2</think>{\"a\": 1}")
            == '{"a": 1}'
        )

    def test_think_unclosed_tail_dropped(self):
        # An unclosed <think> means everything after it is reasoning.
        
        assert strip_think_blocks('{"a": 1}<think>unfinished') == '{"a": 1}'

    def test_think_no_tags_untouched(self):
        assert strip_think_blocks('{"a": 1}') == '{"a": 1}'
        assert strip_think_blocks("no tags here") == "no tags here"

    def test_payload_from_prose_prefix(self):
        assert extract_json_payload('Вот JSON: {"a": 1}') == '{"a": 1}'

    def test_payload_array_from_prose(self):
        assert extract_json_payload("Список: [1, 2] и всё") == "[1, 2]"

    def test_payload_already_json_untouched(self):
        # Text already starting with a JSON bracket is not touched.
        
        text = '{"a": 1}'
        assert extract_json_payload(text) is text
        spaced = '  [1]'
        assert extract_json_payload(spaced) is spaced

    def test_payload_no_brackets_untouched(self):
        text = "никаких скобок"
        assert extract_json_payload(text) is text

    def test_payload_earliest_bracket_wins(self):
        # When both an object and an array are present, the earlier opening
        # bracket defines the payload.
        
        assert extract_json_payload('x {"a": 1} mid [2]') == '{"a": 1}'
        assert extract_json_payload("see [1] then {\"a\": 2}") == "[1]"

    def test_pipeline_think_fence_payload(self):
        # The full pipeline: think blocks → fences → payload extraction.
        
        assert (
            _clean_judge_response_text('<think>r</think>```json\n{"k": 1}\n```')
            == '{"k": 1}'
        )
        assert (
            _clean_judge_response_text('Думаю: <think>r</think> Ответ: {"k": 1}')
            == '{"k": 1}'
        )
        plain = '{"k": 1}'
        assert _clean_judge_response_text(plain) is plain

    def test_wrapper_strips_response_content(self):
        # The wrapped create() sanitizes every non-streaming response.
        
        import asyncio

        class _Message:
            def __init__(self, content):
                self.content = content

        class _Choice:
            def __init__(self, content):
                self.message = _Message(content)

        class _Response:
            def __init__(self, contents):
                self.choices = [_Choice(c) for c in contents]

        class _Completions:
            async def create(self, **kwargs):
                if kwargs.get("stream"):
                    return "STREAM_SENTINEL"
                return _Response(
                    ['```json\n{"x": 1}\n```', '{"y": 2}']
                )

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        class _Client:
            def __init__(self):
                self.chat = _Chat()

        client = sanitize_judge_client_json_fences(_Client())
        resp = asyncio.run(client.chat.completions.create())
        assert resp.choices[0].message.content == '{"x": 1}'
        assert resp.choices[1].message.content == '{"y": 2}'
        # Streaming calls pass through untouched.
        
        assert asyncio.run(client.chat.completions.create(stream=True)) == "STREAM_SENTINEL"

    def test_wrapper_falls_back_to_reasoning_content(self):
        # Thinking models (GLM on Z.AI) may leave `content` empty and return
        # the answer only in `reasoning_content` — the wrapper must promote it
        # (and strip its fences) instead of letting every AC parse fail.
        
        
        
        import asyncio

        class _Message:
            def __init__(self, content, reasoning_content=None):
                self.content = content
                self.reasoning_content = reasoning_content

        class _Choice:
            def __init__(self, message):
                self.message = message

        class _Response:
            def __init__(self, messages):
                self.choices = [_Choice(m) for m in messages]

        class _Completions:
            async def create(self, **kwargs):
                return _Response(
                    [
                        _Message(
                            None,
                            '```json\n{"a": 1}\n```',
                        ),
                        _Message(
                            "",
                            'Модель думала… {"b": 2}',
                        ),
                    ]
                )

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        class _Client:
            def __init__(self):
                self.chat = _Chat()

        client = sanitize_judge_client_json_fences(_Client())
        resp = asyncio.run(client.chat.completions.create())
        assert resp.choices[0].message.content == '{"a": 1}'
        # The prose preamble is stripped by extract_json_payload — the
        # reasoning fallback goes through the full cleaning pipeline.
        
        
        assert resp.choices[1].message.content == '{"b": 2}'

    def test_wrapper_prefers_content_over_reasoning(self):
        # Non-empty `content` wins; `reasoning_content` is ignored.
        
        import asyncio

        class _Message:
            def __init__(self, content, reasoning_content):
                self.content = content
                self.reasoning_content = reasoning_content

        class _Choice:
            def __init__(self, message):
                self.message = message

        class _Response:
            def __init__(self, message):
                self.choices = [_Choice(message)]

        class _Completions:
            async def create(self, **kwargs):
                return _Response(_Message('{"x": 1}', "chain of thought"))

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        class _Client:
            def __init__(self):
                self.chat = _Chat()

        client = sanitize_judge_client_json_fences(_Client())
        resp = asyncio.run(client.chat.completions.create())
        assert resp.choices[0].message.content == '{"x": 1}'

    def test_wrapper_both_empty_untouched(self):
        # Empty `content` with no usable `reasoning_content` stays as-is.
        
        import asyncio

        class _Message:
            def __init__(self, content):
                self.content = content

        class _Choice:
            def __init__(self, message):
                self.message = message

        class _Response:
            def __init__(self, message):
                self.choices = [_Choice(message)]

        class _Completions:
            async def create(self, **kwargs):
                return _Response(_Message("   "))

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        class _Client:
            def __init__(self):
                self.chat = _Chat()

        client = sanitize_judge_client_json_fences(_Client())
        resp = asyncio.run(client.chat.completions.create())
        assert resp.choices[0].message.content == "   "

    def test_wrapper_full_cleaning_pipeline(self):
        # End-to-end: think blocks, fences and prose prefixes are all
        # removed from `content` before instructor/RAGAS parse it.
        
        
        import asyncio

        class _Message:
            def __init__(self, content, reasoning_content=None):
                self.content = content
                self.reasoning_content = reasoning_content

        class _Choice:
            def __init__(self, message):
                self.message = message

        class _Response:
            def __init__(self, messages):
                self.choices = [_Choice(m) for m in messages]

        class _Completions:
            async def create(self, **kwargs):
                return _Response(
                    [
                        _Message('<think>reasoning…</think>```json\n{"x": 1}\n```'),
                        _Message("Вот ответ: {\"y\": 2}"),
                        _Message('Думаю… <think>…</think> Итого: {"z": 3}'),
                        _Message('{"w": 4}'),
                    ]
                )

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        class _Client:
            def __init__(self):
                self.chat = _Chat()

        client = sanitize_judge_client_json_fences(_Client())
        resp = asyncio.run(client.chat.completions.create())
        assert resp.choices[0].message.content == '{"x": 1}'
        assert resp.choices[1].message.content == '{"y": 2}'
        assert resp.choices[2].message.content == '{"z": 3}'
        # Clean JSON must survive the pipeline byte-identically.
        
        assert resp.choices[3].message.content == '{"w": 4}'

    def test_wrapper_fallback_reasoning_with_think(self):
        # The reasoning_content fallback goes through the SAME pipeline:
        # <think>…</think> + trailing JSON becomes clean JSON.
        
        # <think>…</think> + JSON
        import asyncio

        class _Message:
            def __init__(self, content, reasoning_content):
                self.content = content
                self.reasoning_content = reasoning_content

        class _Choice:
            def __init__(self, message):
                self.message = message

        class _Response:
            def __init__(self, message):
                self.choices = [_Choice(message)]

        class _Completions:
            async def create(self, **kwargs):
                return _Response(
                    _Message(None, '<think>chain</think>{"b": 2}')
                )

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        class _Client:
            def __init__(self):
                self.chat = _Chat()

        client = sanitize_judge_client_json_fences(_Client())
        resp = asyncio.run(client.chat.completions.create())
        assert resp.choices[0].message.content == '{"b": 2}'

    def test_wiring_applies_sanitizer_to_judge_client(self, monkeypatch):
        # _build_metrics_from_ragas must wrap the judge client BEFORE handing
        # it to llm_factory (instructor captures create at wiring time).
        # _build_metrics_from_ragas
        
        class _Completions:
            async def create(self, **kwargs):
                return None

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        class _Client:
            def __init__(self):
                self.chat = _Chat()

        fake_client = _Client()

        def _fake_async_client(self, *, max_retries: int = 2, timeout=None):
            return fake_client

        monkeypatch.setattr(OpenAIEndpoint, "async_client", _fake_async_client)
        TestBuildMetricsWiring._patch_ragas_imports(monkeypatch)
        _build_metrics_from_ragas(_cfg())
        assert (
            getattr(fake_client.chat.completions.create, "__name__", None)
            == "_create_with_fence_sanitizer"
        )


# ---------------------------------------------------------------------------
# Judge JSON escape repair
# ---------------------------------------------------------------------------
class TestEscapeRepair:
    'The conservative invalid-escape rescue for judge JSON replies.\n\n\n    Observed on the 2026-09-25 vllm_env SFT anchors: a free-form judge (the\n    ragas 0.2.15 compat path carries no ``response_format``) copies LaTeX\n    notation verbatim from the judged math answers — ``\\(x^4 - 5x^2 + 4 =\n    0\\)`` — and ``\\(`` is not a valid JSON escape, so ``json.loads`` fails\n    deterministically at near-greedy temperature and every retry reproduces\n    the same broken output (the "Invalid json output" AC failures).\n    JSON-escape, ``json.loads``'

    def test_observed_latex_payload_rescued(self):
        # The exact failure shape from the anchor log: a statements list
        # mixing invalid \( escapes with a VALID \\pm escape.
        
        
        payload = (
            '{\n    "statements": [\n'
            '        "To solve the equation \\(x^4 - 5x^2 + 4 = 0\\), '
            'substitution is used.",\n'
            '        "For \\(y = 4\\), \\(x^2 = 4\\) implies '
            '\\(x = \\\\pm 2\\)."\n    ]\n}'
        )
        with pytest.raises(json.JSONDecodeError):
            json.loads(payload)  # the raw reply is unparseable

        cleaned = _clean_judge_response_text(payload)
        parsed = json.loads(cleaned)
        # The repaired statements keep the LITERAL LaTeX the judge meant.
        
        assert parsed["statements"][0] == (
            "To solve the equation \\(x^4 - 5x^2 + 4 = 0\\), "
            "substitution is used."
        )
        assert parsed["statements"][1] == (
            "For \\(y = 4\\), \\(x^2 = 4\\) implies \\(x = \\pm 2\\)."
        )

    def test_repair_doubles_only_invalid_escapes(self):
        # Raw walk: \( is doubled; a valid \\ pair is consumed as a whole
        # (the second backslash must never be re-doubled in front of "pm").
        
        
        assert _repair_invalid_json_escapes("a\\(b") == "a\\\\(b"
        assert _repair_invalid_json_escapes("a\\\\pm b") == "a\\\\pm b"
        assert _repair_invalid_json_escapes("no escapes") == "no escapes"

    def test_valid_json_returned_as_is(self):
        # Already-parseable JSON (even with an escaped-backslash-then-paren
        # ``\\(`` pair, the exact case a naive regex corrupts) is identity.
        
        
        valid = '["a\\\\(b", "c\\/d", "e\\u0041"]'
        assert json.loads(valid)  # sanity: really valid
        assert _maybe_repair_json_escapes(valid) is valid

    def test_free_text_unchanged(self):
        # The judge-preflight greeting is not JSON — it passes through.
        
        free = "Привет! Как дела?"
        assert _maybe_repair_json_escapes(free) is free

    def test_truncated_json_not_rescued(self):
        # Truncation (the max_tokens failure class) cannot be repaired —
        # the original is returned so the caller sees the real finish state.
        
        
        truncated = '{"statements": ["\\(x'
        assert _maybe_repair_json_escapes(truncated) is truncated

    def test_pipeline_repairs_fenced_latex(self):
        # Fences are stripped first, then the escapes are repaired.
        
        fenced = '```json\n{"statements": ["\\(x^2\\)"]}\n```'
        out = _clean_judge_response_text(fenced)
        assert json.loads(out)["statements"] == ["\\(x^2\\)"]

    def test_wrapper_repairs_latex_escapes(self):
        # End-to-end through the client wrapper: the judge reply with LaTeX
        # escapes lands in message.content as parseable JSON before
        # instructor/RAGAS ever see it.
        
        
        import asyncio

        class _Message:
            def __init__(self, content):
                self.content = content

        class _Choice:
            def __init__(self, message):
                self.message = message

        class _Response:
            def __init__(self, message):
                self.choices = [_Choice(message)]

        class _Completions:
            async def create(self, **kwargs):
                return _Response(
                    _Message('{"statements": ["\\(y = x^2\\)"]}')
                )

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        class _Client:
            def __init__(self):
                self.chat = _Chat()

        client = sanitize_judge_client_json_fences(_Client())
        resp = asyncio.run(client.chat.completions.create())
        assert json.loads(resp.choices[0].message.content) == {
            "statements": ["\\(y = x^2\\)"]
        }
