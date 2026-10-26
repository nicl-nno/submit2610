'Tests for the fluency guard inside ``RAGASScorer.score`` (judge.py wiring).\n\n\nThe guard transform ``min(1, fluency / T)`` must: be exactly 1.0 while prose\nis good (zero influence on the reward), drop proportionally below the\nthreshold, and interact with the strict harmonic mean as a restoring\ngradient. Offline: fake metrics + fake fluency scorer.'

from __future__ import annotations

import pytest

from llm_training.reward.config import RewardConfig
from llm_training.reward.judge import RAGASScorer


class _Result:
    def __init__(self, value: float) -> None:
        self.value = value


class _FakeMetric:
    def __init__(self, value: float) -> None:
        self.value = value

    async def ascore(self, **kwargs):
        return _Result(self.value)


class _FakeFluency:
    """Fake fluency scorer returning a preset value; records calls."""

    def __init__(self, value: float | None) -> None:
        self.value = value
        self.calls: list[str] = []

    async def score(self, response: str, task_class: str, *, extra=None):
        self.calls.append(response)
        return self.value


def _scorer(fluency_value: float, guard: float | None):
    cfg = RewardConfig(
        judge=None,
        enable_fluency=True,
        enable_code_check=False,
        fluency_classes=frozenset({"menon_qa"}),
        fluency_subtasks={},
        fluency_guard_threshold=guard,
    )
    return RAGASScorer(
        cfg,
        metrics=(None, _FakeMetric(0.7), None),
        fluency_scorer=_FakeFluency(fluency_value),
    )


pytestmark = pytest.mark.asyncio


class TestGuardInScore:
    async def test_good_prose_guard_is_exactly_one(self):
        # fluency 0.95 ≥ T=0.9 → guard = 1.0 → harmonic([0.7, 1.0]).
        
        m = await _scorer(0.95, 0.9).score(
            user_input="q", response="Ответ из нескольких связных предложений.",
            reference="r", task_class="menon_qa",
        )
        assert m.fluency == pytest.approx(1.0)
        assert m.combined == pytest.approx(2 / (1 / 0.7 + 1 / 1.0))

    async def test_degraded_prose_drops_proportionally(self):
        # fluency 0.63 < T=0.9 → guard = 0.63/0.9 = 0.7 → harmonic([0.7, 0.7]).
        
        m = await _scorer(0.63, 0.9).score(
            user_input="q", response="плохой текст", reference="r",
            task_class="menon_qa",
        )
        assert m.fluency == pytest.approx(0.7)
        assert m.combined == pytest.approx(0.7)

    async def test_continuous_at_threshold(self):
        # At exactly T the guard is 1.0 (no discontinuity: T/T = 1).
        
        m = await _scorer(0.9, 0.9).score(
            user_input="q", response="текст", reference="r", task_class="menon_qa",
        )
        assert m.fluency == pytest.approx(1.0)

    async def test_guard_off_reports_raw(self):
        # None (run_eval.py default) → the raw fluency value is reported.
        
        m = await _scorer(0.95, None).score(
            user_input="q", response="текст", reference="r", task_class="menon_qa",
        )
        assert m.fluency == pytest.approx(0.95)
        assert m.combined == pytest.approx(2 / (1 / 0.7 + 1 / 0.95))

    async def test_severe_degradation_heavily_punished(self):
        # fluency 0.27 → guard 0.3 → harmonic([0.7, 0.3]) ≈ 0.42 — the
        # restoring gradient the brake exists for.
        
        m = await _scorer(0.27, 0.9).score(
            user_input="q", response="очень плохой текст", reference="r",
            task_class="menon_qa",
        )
        assert m.fluency == pytest.approx(0.3)
        assert m.combined == pytest.approx(2 / (1 / 0.7 + 1 / 0.3))

    async def test_fluency_none_unaffected_by_guard(self):
        # N/A fluency (short answer) stays None — the guard never fabricates a
        # value for a metric that did not run.
        
        cfg = RewardConfig(
            judge=None, enable_fluency=True, enable_code_check=False,
            fluency_classes=frozenset({"menon_qa"}), fluency_subtasks={},
            fluency_guard_threshold=0.9,
        )
        scorer = RAGASScorer(
            cfg, metrics=(None, _FakeMetric(0.7), None), fluency_scorer=_FakeFluency(None),
        )
        m = await scorer.score(
            user_input="q", response="Да.", reference="r", task_class="menon_qa",
        )
        assert m.fluency is None
