'Offline tests of the Mixed-CUTS generation wrapper (no network, no GPU).'

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from llm_training.runtime.mixed_cuts import (
    apply_mixed_cuts,
    mixed_cuts_splits,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
@dataclass
class _FakeSamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    n: int = 1
    max_tokens: int = 128


@dataclass
class _FakeOutput:
    text: str


@dataclass
class _FakeRequestOutput:
    prompt: str
    outputs: list[_FakeOutput] = field(default_factory=list)


class _FakeLLM:
    """Records (prompts, params) per call; returns prompt-tagged outputs."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], _FakeSamplingParams]] = []

    def generate(self, prompts, sampling_params=None, *args, **kwargs):
        prompts = list(prompts)
        self.calls.append((prompts, sampling_params))
        return [
            _FakeRequestOutput(p, [_FakeOutput(p)]) for p in prompts
        ]


class _FakeTrainer:
    def __init__(self, llm, num_generations: int) -> None:
        self.llm = llm
        self.num_generations = num_generations


# ---------------------------------------------------------------------------
# The pure split
# ---------------------------------------------------------------------------
def test_splits_by_group_position() -> None:
    std, cuts = mixed_cuts_splits(n_rows=6, group_size=3, n_standard=2)
    # Groups (0,1,2) and (3,4,5): first two rows of each stay standard.
    assert std == [0, 1, 3, 4]
    assert cuts == [2, 5]
    std, cuts = mixed_cuts_splits(n_rows=4, group_size=2, n_standard=1)
    assert std == [0, 2]
    assert cuts == [1, 3]


def test_splits_validation() -> None:
    with pytest.raises(ValueError):
        mixed_cuts_splits(4, group_size=1, n_standard=1)
    with pytest.raises(ValueError):
        mixed_cuts_splits(4, group_size=4, n_standard=4)  # no CUTS rows
    with pytest.raises(ValueError):
        mixed_cuts_splits(4, group_size=4, n_standard=0)  # no standard rows


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------
def test_apply_mixed_cuts_routes_and_preserves_order() -> None:
    llm = _FakeLLM()
    trainer = _FakeTrainer(llm, num_generations=4)
    info = apply_mixed_cuts(trainer, cuts_k=32, cuts_share=0.5)
    assert (info.n_standard, info.n_cuts) == (2, 2)
    assert "mixed-CUTS" in info.describe()

    prompts = [f"p{i}" for i in range(8)]  # two groups of 4
    params = _FakeSamplingParams(temperature=1.0, top_k=-1)
    out = llm.generate(prompts, sampling_params=params)

    # Order is preserved bit-for-bit at the merge seam.
    assert [o.prompt for o in out] == prompts
    assert [o.outputs[0].text for o in out] == prompts

    # Two calls: 4 standard rows (0,1,4,5) with the ORIGINAL params object,
    # 4 CUTS rows (2,3,6,7) with the patched params.
    assert len(llm.calls) == 2
    std_prompts, std_params = llm.calls[0]
    cuts_prompts, cuts_params = llm.calls[1]
    assert std_prompts == ["p0", "p1", "p4", "p5"]
    assert cuts_prompts == ["p2", "p3", "p6", "p7"]
    assert std_params is params
    assert cuts_params is not params
    assert cuts_params.top_k == 32
    # the default CUTS temperature is 2.0: vLLM 0.27+ validates temperature
    # to [0, 2] — 2.0 is the flattest legal within-top-K scaling.
    
    
    assert cuts_params.temperature == 2.0
    assert cuts_params.top_p == 1.0
    assert cuts_params.min_p == 0.0
    assert cuts_params.max_tokens == params.max_tokens  # carried over


def test_apply_mixed_cuts_guards() -> None:
    # Server mode (no llm) → a clear error.
    with pytest.raises(RuntimeError, match="colocate"):
        apply_mixed_cuts(_FakeTrainer(None, 4))

    # Double wrap → error.
    llm = _FakeLLM()
    trainer = _FakeTrainer(llm, 4)
    apply_mixed_cuts(trainer)
    with pytest.raises(RuntimeError, match="already wrapped"):
        apply_mixed_cuts(trainer)

    # Degenerate share / group → error.
    with pytest.raises(ValueError):
        apply_mixed_cuts(_FakeTrainer(_FakeLLM(), 4), cuts_share=0.0)
    with pytest.raises(ValueError):
        apply_mixed_cuts(_FakeTrainer(_FakeLLM(), 1))

    # The n>1 layout is refused loudly, not mis-split.
    llm2 = _FakeLLM()
    trainer2 = _FakeTrainer(llm2, 4)
    apply_mixed_cuts(trainer2)
    with pytest.raises(RuntimeError, match="n=1"):
        llm2.generate(["p"], sampling_params=_FakeSamplingParams(n=4))
