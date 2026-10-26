'Build configuration for the unified dataset.\n\n\nEncapsulates the knobs from AGENTS.md: the data directory, the fixed RNG seed,\nthe per-class cap, and the length-discard thresholds (32 000 total tokens and\n8 000 reference tokens, measured on the Meno-Lite chat-template rendering).'

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: The HuggingFace id of the policy / target model whose tokenizer is the
#: reference for length accounting and chat rendering.
# : HF-id
MENO_LITE_MODEL_ID: str = "anonymous/base-model"

#: Total-length discard threshold (AGENTS.md: 32 000 tokens).

MAX_TOTAL_TOKENS: int = 32_000

#: Reference-length discard threshold (AGENTS.md: 8 000 tokens).

MAX_REFERENCE_TOKENS: int = 8_000

#: Per-class cap (AGENTS.md: at most 200 examples per class).

DEFAULT_MAX_PER_CLASS: int = 200


@dataclass(frozen=True)
class DataConfig:
    'Configuration for :func:`llm_training.build.build_unified_dataset`.\n\n\n    The seed governs RNG only (Python ``random``) — it does NOT pick the split;\n    the split is chosen manually by passing ``val_task`` / ``test_task``.\n\n    ``final_turn_share`` (optional, ``None`` = off): the minimum share of SGR\n    final-answer turns in each **trainable** SGR class pool, reached by\n    deterministic round-robin duplication AFTER the per-class cap. Held-out\n    classes (``val_task`` / ``test_task``) are never boosted — the anchor\n    measuring instrument must keep its composition. SGR trajectories end with\n    exactly one ``finalanswertool`` turn, so after the per-turn split finals\n    are a small minority of the pool; the boost raises the in-loop envelope\n    gradient so the ``finalanswertool`` format generalizes to the held-out\n    classes (the r2 anchors measured ``final_unparsed`` 3–5/10 there).\n    ``reasoning_keep_share`` (optional, ``None`` / ``1.0`` = off): the share of\n    SGR reasoning turns KEPT in each **trainable** SGR class pool — a\n    deterministic prefix downsample, the mirror of ``final_turn_share`` (see\n    ``build._downsample_reasoning``). The r3 rationale: reasoning turns carry\n    no quality gradient (the structural markers are saturated, the judge\n    rubrics measured flat ~0.48) yet consume the GRPO generation budget like\n    any other prompt; thinning them shifts the budget onto the action/final\n    turns with structural signals. Held-out classes are never downsampled.\n    ``final_turn_share``\n    ``reasoning_keep_share``'

    data_dir: Path = Path("src_data")
    model_id: str = MENO_LITE_MODEL_ID
    seed: int = 42
    max_per_class: int = DEFAULT_MAX_PER_CLASS
    max_total_tokens: int = MAX_TOTAL_TOKENS
    max_reference_tokens: int = MAX_REFERENCE_TOKENS
    final_turn_share: float | None = None
    reasoning_keep_share: float | None = None
    #: Which HuggingFace split the NEREL_Instruct loader reads. The GRPO
    #: training path sets ``"train"`` (the practically inexhaustible split);
    #: the eval/warm-up paths keep the default ``"validation"``.
    
    
    
    nerel_split: str = "validation"
    #: Optional subdir with the loop-format SGR catalog (reasoningtool calls;
    #: produced by ``scripts/convert_sgr_to_loop.py``). When set, the SGR
    #: loader prefers per-file records from there; classes without files in
    #: the subdir (the val hold-outs, never converted) automatically fall
    #: back to the original ``sgr_agent/`` — the anchor val pool keeps its
    #: frozen format. "Format by pool role": trainable train pools read the
    #: loop representation, held-out val pools read the original one.
    
    # : reasoningtool; ``scripts/convert_sgr_to_loop.py``
    
    
    
    
    
    sgr_loop_subdir: str | None = None

    def __post_init__(self) -> None:
        if self.final_turn_share is None:
            pass
        elif not 0.0 < self.final_turn_share < 1.0:
            raise ValueError(
                f'final_turn_share must be in (0, 1) — 1.0 is unsatisfiable for a pool with non-final turns, got {self.final_turn_share}'
            )
        # The reasoning downsample knob: None / 1.0 = off, otherwise a proper
        # fraction of the reasoning turns to KEEP (the mirror of
        # final_turn_share — see build._downsample_reasoning).
        
        
        if self.reasoning_keep_share is None or self.reasoning_keep_share == 1.0:
            pass
        elif not 0.0 < self.reasoning_keep_share < 1.0:
            raise ValueError(
                f'reasoning_keep_share must be in (0, 1] or None, got {self.reasoning_keep_share}'
            )
