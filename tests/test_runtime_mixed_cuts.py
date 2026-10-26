'Offline tests of Mixed-CUTS wiring for both TRL colocate layouts.\n\n``trainer.llm`` (TRL 0.20'

from __future__ import annotations

import pytest

from llm_training.runtime.mixed_cuts import (
    MixedCutsInfo,
    apply_mixed_cuts,
    mixed_cuts_splits,
)


class _LegacyParams:
    def __init__(self, n: int = 1, top_k: int = 0, temperature: float = 0.7) -> None:
        self.n = n
        self.top_k = top_k
        self.temperature = temperature


class _LegacyLLM:
    """TRL 0.20 fake: generate(prompts, sampling_params) -> one row each."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], _LegacyParams]] = []

    def generate(self, prompts, sampling_params=None, *args, **kwargs):
        prompts = list(prompts)
        self.calls.append((prompts, sampling_params))
        return [f"gen({p})" for p in prompts]


class _LegacyTrainer:
    num_generations = 4

    def __init__(self) -> None:
        self.llm = _LegacyLLM()


class _VGen:
    """TRL 1.10 fake: sampling knobs on the instance, 4-tuple return."""

    def __init__(self) -> None:
        self.temperature = 0.7
        self.top_p = 1.0
        self.top_k = 0
        self.min_p = None
        self.calls: list[tuple[list[str], tuple[float, int, float]]] = []

    def generate(self, prompts, images=None, num_generations=None, *args, **kwargs):
        prompts = list(prompts)
        state = (self.temperature, self.top_k, self.top_p)
        self.calls.append((prompts, state))
        n = len(prompts)
        prompt_ids = [[i] for i in range(100, 100 + n)]
        completion_ids = [[f"{p}|k{self.top_k}t{self.temperature}"] for p in prompts]
        logprobs = [[-0.1] for _ in range(n)]
        return prompt_ids, completion_ids, logprobs, None


class _VGenTrainer:
    num_generations = 4

    def __init__(self) -> None:
        self.vllm_generation = _VGen()


class _BareTrainer:
    num_generations = 4


def test_splits_by_group_position() -> None:
    std, cuts = mixed_cuts_splits(8, 4, 2)
    assert std == [0, 1, 4, 5]
    assert cuts == [2, 3, 6, 7]


def test_legacy_llm_split_order_and_params() -> None:
    trainer = _LegacyTrainer()
    info = apply_mixed_cuts(trainer, cuts_k=32, cuts_share=0.5, cuts_temperature=5.0)
    assert isinstance(info, MixedCutsInfo)
    assert (info.n_standard, info.n_cuts) == (2, 2)
    prompts = [f"p{i}" for i in range(8)]
    params = _LegacyParams(top_k=0, temperature=0.7)
    out = trainer.llm.generate(prompts, params)
    assert out == [f"gen({p})" for p in prompts]  # original order restored
    std_rows, std_params = trainer.llm.calls[0]
    cuts_rows, cuts_params = trainer.llm.calls[1]
    assert std_rows == ["p0", "p1", "p4", "p5"]
    assert cuts_rows == ["p2", "p3", "p6", "p7"]
    # the standard call keeps the caller's params object untouched
    assert std_params is params
    # the CUTS call goes through a clone with the flattened top-K knobs
    assert cuts_params is not params
    assert cuts_params.top_k == 32
    assert cuts_params.temperature == 5.0
    assert cuts_params.n == 1


def test_legacy_n_gt1_refused() -> None:
    trainer = _LegacyTrainer()
    apply_mixed_cuts(trainer)
    with pytest.raises(RuntimeError, match="n=1 per row"):
        trainer.llm.generate(["a", "b", "c", "d"], _LegacyParams(n=2))


def test_vllm_generation_split_state_and_restore() -> None:
    trainer = _VGenTrainer()
    info = apply_mixed_cuts(trainer, cuts_k=32, cuts_share=0.5, cuts_temperature=5.0)
    assert (info.n_standard, info.n_cuts) == (2, 2)
    vgen = trainer.vllm_generation
    prompts = [f"p{i}" for i in range(8)]
    prompt_ids, completion_ids, logprobs, token_ids = vgen.generate(prompts, None, 4)

    assert len(vgen.calls) == 2
    std_rows, std_state = vgen.calls[0]
    cuts_rows, cuts_state = vgen.calls[1]
    assert std_rows == ["p0", "p1", "p4", "p5"]
    assert cuts_rows == ["p2", "p3", "p6", "p7"]
    # standard rows keep the instance sampling, CUTS rows see the flat top-K
    assert std_state == (0.7, 0, 1.0)
    assert cuts_state == (5.0, 32, 1.0)

    # per-row outputs reassembled in the original order, each reflecting the
    # sampling state of its own sub-call
    for i, p in enumerate(prompts):
        expected_k, expected_t = (0, 0.7) if i % 4 < 2 else (32, 5.0)
        assert completion_ids[i] == [f"{p}|k{expected_k}t{expected_t}"]
    assert prompt_ids[i] is not None
    assert all(lp == [-0.1] for lp in logprobs)
    # a None component merges into an all-None column (the trainer discards it)
    assert token_ids == [None] * len(prompts)

    # the instance state is restored after the wrapped call
    assert (vgen.temperature, vgen.top_k, vgen.top_p, vgen.min_p) == (
        0.7,
        0,
        1.0,
        None,
    )


def test_vllm_generation_pure_standard_passthrough() -> None:
    trainer = _VGenTrainer()
    apply_mixed_cuts(trainer, cuts_share=0.5)
    vgen = trainer.vllm_generation
    vgen.generate(["only"], None, 1)  # 1 row, group_size=4 → all standard
    # a single-row batch cannot contain CUTS rows → no extra call
    assert len(vgen.calls) == 1


def test_no_engine_raises() -> None:
    with pytest.raises(RuntimeError, match="colocate"):
        apply_mixed_cuts(_BareTrainer())
