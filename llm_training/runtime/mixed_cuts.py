'Mixed-CUTS decoding for GRPO (Liang et al., ACL 2026 — "Too Correct to\nLearn: Reinforcement Learning on Saturated Reasoning Data").\n\n«Too Correct\nto Learn: Reinforcement Learning on Saturated Reasoning Data»).\n\nTHE PROBLEM\nsolves the task) every completion in a GRPO group is correct and alike → the\ngroup-relative advantage vanishes → no gradient → mode collapse. The\nGRPO-extraction line hits this by construction: after SFT the model emits\nvalid JSON in the vast majority of samples, so the ``json_format`` /\n``schema_conformance`` components saturate exactly the way MATH saturated in\nthe paper. The repo already fought the zero-variance failure once with\n``temperature=1.0 + num_generations=16`` (the r3 escalation); CUTS is the\nsurgical version of that lever.\n(temperature=1.0 + num_generations=16); CUTS\n\nTHE METHOD\nuniformly from the top-K highest-probability candidates instead of following\nthe model\'s softmax — flattening the local optimization landscape while\nstaying inside the high-confidence (structure-preserving) manifold. **Mixed-\nCUTS**: split every GRPO group — half the rollouts standard sampling\n(exploitation), half CUTS (exploration) — restoring intra-group advantage\nvariance on saturated prompts. For extraction this is especially apt: the\nJSON grammar keeps the envelope valid while the CONTENT varies.\n\nIMPLEMENTATION\n``num_generations`` times (a ``RepeatSampler`` in the dataloader) with a\nsingle ``n=1`` :class:`~vllm.SamplingParams`, so the wrapper patches\n``trainer.llm.generate`` and routes a deterministic share of ROWS (by\nposition inside each group of G consecutive prompts) to a CUTS params object\n— no trainer subclassing, no TRL internals beyond the call.\nTRL 1.10 colocate keeps the SAME repeated-prompt ``n=1`` layout but moves the\nengine to ``trainer.vllm_generation`` (``VLLMGeneration.generate(prompts,\nimages, num_generations, profiler)``) and reads sampling knobs from INSTANCE\nstate (``self.temperature/top_p/top_k/min_p``) at every call — so the 1.10\nwrapper routes row subsets the same way and applies the CUTS knobs by\ntemporarily overriding those instance fields around the CUTS sub-call\n(restored in ``finally``).\nUniform-over-top-K is approximated by ``top_k=K`` + a high ``temperature``\n(positive temperature scaling is monotone, so the top-K SET is unchanged\nwhile the within-K softmax flattens towards uniform; ``cuts_temperature=5``\nby default). Group order within the batch is preserved, so TRL\'s completion\nflattening is unaffected (advantages are group statistics — member order is\nirrelevant).\nTRL 0.20 colocate\n``trainer.llm.generate``\nTRL 1.10 colocate\n``finally``).\n\nCAVEATS\n\n* **Keep ``num_iterations=1``** (the repo default): with PPO-style repeated\n  updates per batch the missing importance-sampling correction for the\n  changed behaviour policy would introduce bias. With a single update per\n  batch GRPO\'s ratios are 1 regardless — CUTS is then exactly as on-policy\n  as any temperature change.\n* The ``n>1`` sampling layout (one call per prompt with several samples) is\n  NOT wrapped — a clear error fires instead of a silent mis-split (checked\n  on the 0.20 path; the 1.10 colocate branch hard-codes ``n=1``).\n* Server mode is not covered: the generation service would need the same\n  split — raise (no ``trainer.llm`` / ``trainer.vllm_generation`` at all).'

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

# : Attribute marker preventing a double wrap.
_WRAP_MARKER = "_mixed_cuts_wrapped"

#: Default CUTS strength knobs (see the paper's sensitivity analysis: the
#: effect is robust across K; the temperature just needs to flatten the
# : within-top-K distribution).
DEFAULT_CUTS_K = 32
# vLLM 0.27+ validates SamplingParams.temperature to [0, 2]; 2.0 is the
# flattest legal within-top-K scaling (the paper's 'high temperature' only
# needs to flatten the top-K softmax — monotone positive scaling).


DEFAULT_CUTS_TEMPERATURE = 2.0
DEFAULT_CUTS_SHARE = 0.5


@dataclass(frozen=True)
class MixedCutsInfo:
    'What was wired.'

    group_size: int
    n_standard: int
    n_cuts: int
    cuts_k: int
    cuts_temperature: float

    def describe(self) -> str:
        return (
            f"mixed-CUTS: {self.n_standard} standard + {self.n_cuts} CUTS "
            f"rollouts per group of {self.group_size} "
            f"(top_k={self.cuts_k}, temperature={self.cuts_temperature})"
        )


def mixed_cuts_splits(
    n_rows: int, group_size: int, n_standard: int
) -> tuple[list[int], list[int]]:
    'Pure split of row indices into (standard, cuts) by group position.\n\n    ``[0, n_standard)`` keep the original sampling params, the rest go CUTS.'
    if group_size < 2:
        raise ValueError(
            'Mixed-CUTS needs num_generations >= 2 (a one-member group has no intra-group variance to restore).\nMixed-CUTS'
        )
    if not 1 <= n_standard < group_size:
        raise ValueError(
            f'n_standard must be in [1, {group_size - 1}] for group_size={group_size}, got {n_standard}'
        )
    standard: list[int] = []
    cuts: list[int] = []
    for i in range(n_rows):
        (standard if (i % group_size) < n_standard else cuts).append(i)
    return standard, cuts


def _make_cuts_params(sampling_params: Any, cuts_k: int, cuts_temperature: float) -> Any:
    "Clone the trainer's SamplingParams into the CUTS variant."
    cuts = copy.copy(sampling_params)
    cuts.top_k = int(cuts_k)
    cuts.temperature = float(cuts_temperature)
    # No double truncation on top of top_k.
    
    if getattr(cuts, "top_p", None) is not None:
        cuts.top_p = 1.0
    if getattr(cuts, "min_p", None):
        cuts.min_p = 0.0
    return cuts


def apply_mixed_cuts(
    trainer: Any,
    cuts_k: int = DEFAULT_CUTS_K,
    cuts_share: float = DEFAULT_CUTS_SHARE,
    cuts_temperature: float = DEFAULT_CUTS_TEMPERATURE,
) -> MixedCutsInfo:
    'Wire Mixed-CUTS into a colocate-mode ``GRPOTrainer`` (idempotent-safe).\n\n    trainer is constructed; resolves the colocate engine by TRL generation —\n    ``trainer.llm`` (TRL 0.20) or ``trainer.vllm_generation`` (TRL 1.10);\n    returns :class:`MixedCutsInfo`. Raises bilingual errors on server mode,\n    a double wrap, an unsupported sampling layout or a degenerate share.'
    llm = getattr(trainer, "llm", None)
    vgen = getattr(trainer, "vllm_generation", None)
    group_size = int(getattr(trainer, "num_generations", 1))
    if not 0.0 < cuts_share < 1.0:
        raise ValueError(
            f'cuts_share must be in (0, 1), got {cuts_share}'
        )
    n_standard = max(1, round(group_size * (1.0 - cuts_share)))
    # Validate the split eagerly (a degenerate split raises here, not at the
    # first generation call).
    mixed_cuts_splits(group_size, group_size, n_standard)

    if llm is not None:
        return _wrap_legacy_llm(llm, group_size, n_standard, cuts_k, cuts_temperature)
    if vgen is not None:
        return _wrap_vllm_generation(
            vgen, group_size, n_standard, cuts_k, cuts_temperature
        )
    raise RuntimeError(
        'Mixed-CUTS supports colocate mode only (neither trainer.llm (TRL 0.20) nor trainer.vllm_generation (TRL 1.10) is present — server mode would need the same split inside the generation service).\nMixed-CUTS'
    )


def _wrap_legacy_llm(
    llm: Any, group_size: int, n_standard: int, cuts_k: int, cuts_temperature: float
) -> MixedCutsInfo:
    'TRL 0.20 wiring: patch ``llm.generate(prompts, sampling_params, ...)``.'
    original = llm.generate
    if getattr(original, _WRAP_MARKER, False):
        raise RuntimeError(
            'trainer.llm.generate is already wrapped with Mixed-CUTS.\ntrainer.llm.generate'
        )

    def generate(prompts: Any, sampling_params: Any = None, *args: Any, **kwargs: Any) -> Any:
        n = getattr(sampling_params, "n", 1)
        if n and int(n) > 1:
            raise RuntimeError(
                f'Mixed-CUTS wraps the repeated-prompt layout (n=1 per row); got SamplingParams.n={n}'
            )
        prompts_list = list(prompts)
        idx_std, idx_cuts = mixed_cuts_splits(len(prompts_list), group_size, n_standard)
        merged: list[Any] = [None] * len(prompts_list)
        if idx_std:
            outs_std = original(
                [prompts_list[i] for i in idx_std], sampling_params, *args, **kwargs
            )
            for pos, i in enumerate(idx_std):
                merged[i] = outs_std[pos]
        if idx_cuts:
            cuts_params = _make_cuts_params(sampling_params, cuts_k, cuts_temperature)
            outs_cuts = original(
                [prompts_list[i] for i in idx_cuts], cuts_params, *args, **kwargs
            )
            for pos, i in enumerate(idx_cuts):
                merged[i] = outs_cuts[pos]
        return merged

    setattr(generate, _WRAP_MARKER, True)
    llm.generate = generate
    # The unwrap seam: the anchor path generates through the ORIGINAL
    # generate (greedy, no CUTS split — anchors measure, they don't explore).
    
    
    llm.generate_original = original
    return MixedCutsInfo(
        group_size=group_size,
        n_standard=n_standard,
        n_cuts=group_size - n_standard,
        cuts_k=int(cuts_k),
        cuts_temperature=float(cuts_temperature),
    )


def _wrap_vllm_generation(
    vgen: Any, group_size: int, n_standard: int, cuts_k: int, cuts_temperature: float
) -> MixedCutsInfo:
    'TRL 1.10 wiring: patch ``VLLMGeneration.generate`` (colocate).\n\n    num_generations, profiler)``. The colocate branch builds ``n=1``\n    SamplingParams from INSTANCE state at every call, so the CUTS sub-call\n    temporarily overrides ``top_k/temperature/top_p/min_p`` and restores them\n    in ``finally``.'
    original = getattr(vgen, "generate", None)
    if original is None:
        raise RuntimeError(
            'trainer.vllm_generation has no .generate — unsupported TRL build.'
        )
    if getattr(original, _WRAP_MARKER, False):
        raise RuntimeError(
            'trainer.vllm_generation.generate is already wrapped with Mixed-CUTS.\ntrainer.vllm_generation.generate'
        )
    # Instance fields the colocate branch reads at every generate() call; the
    # CUTS values approximate uniform-over-top-K (top_p=1.0: no double
    # truncation on top of top_k; min_p=0.0: no second threshold).
    cuts_fields: tuple[tuple[str, Any], ...] = (
        ("top_k", int(cuts_k)),
        ("temperature", float(cuts_temperature)),
        ("top_p", 1.0),
        ("min_p", 0.0),
    )

    def _subset(seq: Any, idx: list[int]) -> Any:
        if seq is None:
            return None
        return [seq[i] for i in idx]

    def generate(
        prompts: Any, images: Any = None, num_generations: Any = None, *args: Any, **kwargs: Any
    ) -> Any:
        prompts_list = list(prompts)
        idx_std, idx_cuts = mixed_cuts_splits(len(prompts_list), group_size, n_standard)
        if not idx_cuts:
            # A pure-standard batch (degenerate share configs) — untouched.
            return original(prompts, images, num_generations, *args, **kwargs)
        merged: list[Any] = [None] * len(prompts_list)
        out_std = None
        if idx_std:
            out_std = original(
                _subset(prompts_list, idx_std),
                _subset(images, idx_std),
                num_generations,
                *args,
                **kwargs,
            )
        saved = {f: getattr(vgen, f) for f, _ in cuts_fields}
        try:
            for f, v in cuts_fields:
                setattr(vgen, f, v)
            out_cuts = original(
                _subset(prompts_list, idx_cuts),
                _subset(images, idx_cuts),
                num_generations,
                *args,
                **kwargs,
            )
        finally:
            for f, v in saved.items():
                setattr(vgen, f, v)
        # Both calls return a (prompt_ids, completion_ids, logprobs,
        # logprob_token_ids) tuple of per-row lists (a component may be None).
        # Reassemble into the original row order.
        for pos, i in enumerate(idx_std):
            merged[i] = tuple(None if comp is None else comp[pos] for comp in out_std)
        for pos, i in enumerate(idx_cuts):
            merged[i] = tuple(None if comp is None else comp[pos] for comp in out_cuts)
        cols = list(zip(*merged))
        return tuple(list(col) for col in cols)

    setattr(generate, _WRAP_MARKER, True)
    vgen.generate = generate
    # The unwrap seam for the anchor path (see _wrap_legacy_llm).
    
    vgen.generate_original = original
    return MixedCutsInfo(
        group_size=group_size,
        n_standard=n_standard,
        n_cuts=group_size - n_standard,
        cuts_k=int(cuts_k),
        cuts_temperature=float(cuts_temperature),
    )
