'Tests for the fluency guard — the degradation-brake threshold transform.\n\nT)).\n\nCovers: RewardConfig validation, the judge.score() transform, the interaction\nwith the strict harmonic mean, and the run_eval.py CLI flag plumbing.'

from __future__ import annotations

import pytest

from llm_training.reward.config import RewardConfig, ScoredMetrics


class TestGuardConfig:
    def test_default_is_off(self):
        assert RewardConfig().fluency_guard_threshold is None

    def test_valid_threshold_accepted(self):
        assert RewardConfig(fluency_guard_threshold=0.9).fluency_guard_threshold == 0.9
        assert RewardConfig(fluency_guard_threshold=1.0).fluency_guard_threshold == 1.0
        assert RewardConfig(fluency_guard_threshold=0.05).fluency_guard_threshold == 0.05

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 42.0])
    def test_invalid_threshold_rejected(self, bad):
        with pytest.raises(ValueError, match="fluency_guard_threshold"):
            RewardConfig(fluency_guard_threshold=bad)

    def test_describe_includes_guard(self):
        d = RewardConfig(fluency_guard_threshold=0.9).describe()
        assert d["fluency"]["fluency_guard_threshold"] == 0.9
