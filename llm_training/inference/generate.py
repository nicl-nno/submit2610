"Single-shot policy generator over in-process vLLM (run_eval.py only).\n\n\n``PolicyGenerator`` loads the model under evaluation as an in-process\n:class:`vllm.LLM` (on the policy GPU pool, separate from the judge's), renders\neach unified-dataset prompt with the Meno-Lite chat template (tools pre-baked\ninto the system message), and generates one or N completions. It supports two\nmodes selected by ``run_eval.py``'s ``--generation-mode`` flag:\n\n\n* ``greedy`` — ``temperature=0``, one deterministic completion per prompt;\n* ``sampling`` — ``temperature/top_p`` sampling with ``n=num_samples`` completions.\n\nvLLM is imported lazily so this module imports cleanly in dev environments\nwithout the ``train`` extra; tests inject a fake LLM.  vLLM"

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from llm_training.schema import UnifiedExample
from llm_training.tokenizer_utils import prepare_messages, render

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

#: Ceiling for the per-example auto generation budget (``max_tokens=None``).
#: The build's length-discard rule allows references up to 8 000 tokens, and a
#: tool-call reference wraps the answer + reasoning + completed_steps fields, so
#: 8 192 comfortably covers every realistic completion while never crowding the
#: context of a 32 768-token model.


AUTO_MAX_NEW_TOKENS = 8192


@dataclass(frozen=True)
class GenerationConfig:
    'How the policy should generate (mirrors run_eval.py\'s CLI knobs).\n\n\nAttributes\n        mode: ``"greedy"`` or ``"sampling"``\n        num_samples: number of completions per prompt in ``sampling`` mode\n            (ignored in ``greedy`` where it is effectively 1\n        temperature: sampling temperature\n        top_p: nucleus-sampling probability\n        max_tokens: per-completion generation cap. ``None`` (default) → the\n            **per-example auto budget** ``min(AUTO_MAX_NEW_TOKENS,\n            max_model_len − len(prompt))`` — never vLLM\'s own default (which is\n            a tiny 16 tokens in vLLM ≥0.22 and silently truncates every\n            completion). An explicit value acts as an upper bound, still\n            combined with the remaining context (``min(value, remaining)``).\n        seed: vLLM ``SamplingParams.seed`` for reproducible sampling.'

    mode: str = "greedy"
    num_samples: int = 1
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.mode not in ("greedy", "sampling"):
            raise ValueError(
                f"mode must be 'greedy' or 'sampling', got {self.mode!r}"
            )
        if self.num_samples < 1:
            raise ValueError('num_samples must be >= 1.\nnum_samples')
        if self.mode == "greedy" and self.num_samples != 1:
            raise ValueError(
                'greedy mode produces exactly 1 sample; set num_samples=1 or use sampling.'
            )
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(
                'max_tokens must be >= 1 or None (auto per-example budget).\nmax_tokens'
            )
        if self.num_samples < 1:
            raise ValueError('num_samples must be >= 1.\nnum_samples')
        if self.mode == "greedy" and self.num_samples != 1:
            raise ValueError(
                'greedy mode produces exactly 1 sample; set num_samples=1 or use sampling.'
            )


def render_prompt(
    example: UnifiedExample, tokenizer: "PreTrainedTokenizer"
) -> str:
    'Render a unified example to a prompt string ready for vLLM generation.\n\n\n    Tool schemas are baked into the system message (see\n    :func:`llm_training.tokenizer_utils.prepare_messages`) and the chat template\n    is applied with ``add_generation_prompt=True`` and ``tools=None`` (tools are\n    already in the system text), exactly matching how ``mask_example`` renders.'
    prompt_messages, _full = prepare_messages(example, tokenizer)
    return render(tokenizer, prompt_messages, add_generation_prompt=True)


class PolicyGenerator:
    'In-process vLLM policy generator (the model under evaluation).\n\n    In-process vLLM\n\n    For tests, pass ``llm=`` and ``tokenizer=`` directly to skip vLLM loading.'

    def __init__(
        self,
        *,
        model: str | None = None,
        llm: Any = None,
        tokenizer: "PreTrainedTokenizer | None" = None,
        gpu_ids: tuple[int, ...] | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        dtype: str = "auto",
    ) -> None:
        if llm is not None:
            # Test
            self._llm = llm
            self._tokenizer = tokenizer if tokenizer is not None else self._get_tokenizer(llm)
            self._owns_llm = False
        else:
            if model is None:
                raise ValueError(
                    "Either 'model' or 'llm' must be provided."
                )
            self._llm = self._load_vllm(
                model=model,
                gpu_ids=gpu_ids,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                dtype=dtype,
            )
            self._tokenizer = self._get_tokenizer(self._llm)
            self._owns_llm = True
        # The per-example auto budget needs the model's context window. Prefer
        # an explicit value, then resolve best-effort from the LLM object
        # (vLLM 0.22: llm.llm_engine.model_config.max_model_len; older/newer
        # layouts fall back to attribute probes).
        
        
        self._max_model_len: int | None = max_model_len or self._resolve_max_model_len(
            self._llm
        )

    @staticmethod
    def _get_tokenizer(llm: Any) -> "PreTrainedTokenizer":
        # vLLM exposes the tokenizer either via get_tokenizer() or .tokenizer.
        
        get_tok = getattr(llm, "get_tokenizer", None)
        if callable(get_tok):
            return get_tok()
        return getattr(llm, "tokenizer")

    @staticmethod
    def _load_vllm(
        *,
        model: str,
        gpu_ids: tuple[int, ...] | None,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int | None,
        dtype: str,
    ) -> Any:
        'Import vLLM lazily and build the in-process LLM on the policy GPU pool.'
        # Pin the policy GPU pool BEFORE vLLM initializes CUDA. This is process-
        # global, which is fine: the judge runs in a separate process.
        
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        # Force the 'spawn' multiprocessing start method for the vLLM EngineCore
        # subprocess. CUDA may already be initialized in the parent process
        # (e.g. by torch.manual_seed or any torch import that triggers lazy CUDA
        # init), and a forked subprocess cannot re-initialize CUDA — vLLM dies
        # with "Cannot re-initialize CUDA in forked subprocess". 'spawn' creates
        # a fresh interpreter with its own CUDA context. setdefault lets the
        # operator override via the environment if needed.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        try:
            from vllm import LLM  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'vllm' package is required to generate with the policy. "
                "Install the train extra: pip install -e '.[train]'."
                "\nДля генерации policy нужен 'vllm': pip install -e '.[train]'."
            ) from e
        kwargs: dict[str, Any] = dict(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
        )
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        logger.info("Loading policy vLLM model %s (tp=%d)", model, tensor_parallel_size)
        return LLM(**kwargs)

    @staticmethod
    def _resolve_max_model_len(llm: Any) -> int | None:
        "Best-effort extraction of the model's context window from an LLM.\n\n        Best-effort"
        engine = getattr(llm, "llm_engine", None)
        model_config = getattr(engine, "model_config", None) if engine is not None else None
        for holder in (model_config, engine, llm):
            value = getattr(holder, "max_model_len", None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    def _build_sampling_params(
        self, cfg: GenerationConfig, prompt_token_counts: list[int]
    ) -> list[Any]:
        "Build a per-example ``SamplingParams`` list (one per prompt).\n\n\n        vLLM's own default ``max_tokens`` is a tiny 16 tokens (vLLM ≥ 0.22),\n        which silently truncates every completion mid-sentence — a bug that\n        once zeroed an entire eval. We therefore ALWAYS set an explicit\n        ``max_tokens`` per prompt: the user's cap when given, otherwise the\n        auto ceiling :data:`AUTO_MAX_NEW_TOKENS`, in both cases clamped to the\n        remaining context (``max_model_len − prompt_tokens``) with a floor of 1.\n        vLLM ``LLM.generate`` natively accepts a list of ``SamplingParams``\n        aligned with the prompts.\n\n        Both modes also pin ``skip_special_tokens=False`` and\n        ``spaces_between_special_tokens=False`` (vLLM defaults to stripping\n        special tokens): ``<tool_call>`` / ``</tool_call>`` are added special\n        tokens of the Meno-Lite tokenizer, so the default stripping deletes\n        the envelope from ``out.text`` — every tool-use generation would then\n        fail ``format_check`` regardless of model quality (this exact bug\n        zeroed the first warm-up's val report and would have zeroed the\n        post-warm-up ≥80 %-valid-calls gate). The spacing flag keeps vLLM\n        from inserting spaces between adjacent special tokens (it would\n        corrupt the envelope layout).\n        ``spaces_between_special_tokens=False``\n``</tool_call>``\n        ``format_check``"
        from vllm import SamplingParams  # type: ignore[import-not-found]

        cap = cfg.max_tokens if cfg.max_tokens is not None else AUTO_MAX_NEW_TOKENS
        if self._max_model_len is None:
            raise RuntimeError(
                "Cannot compute the per-example generation budget: the model's max_model_len is unknown (the LLM object does not expose it and no explicit value was given). Pass max_model_len= to PolicyGenerator or set an explicit GenerationConfig.max_tokens."
            )
        def _common_max(prompt_tokens: int) -> int:
            remaining = self._max_model_len - prompt_tokens  # type: ignore[operator]
            if remaining < 1:
                # The build guarantees total ≤ 32 000 for a 32 768 model, so a
                # non-positive remainder should be impossible; clamp + warn.
                
                
                logger.warning(
                    "Prompt of %d tokens leaves no room for generation "
                    "(max_model_len=%d); clamping max_tokens to 1.",
                    prompt_tokens, self._max_model_len,
                )
                remaining = 1
            return min(cap, remaining)

        params = []
        for prompt_tokens in prompt_token_counts:
            common: dict[str, Any] = {
                "max_tokens": _common_max(prompt_tokens),
                # The <tool_call> envelope must survive into out.text.
                
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
            }
            if cfg.mode == "greedy":
                params.append(SamplingParams(temperature=0.0, seed=cfg.seed, **common))
            else:
                params.append(
                    SamplingParams(
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        n=cfg.num_samples,
                        seed=cfg.seed,
                        **common,
                    )
                )
        caps = [p.max_tokens for p in params]
        logger.info(
            "Per-example max_tokens: min=%d, max=%d (budget=%s, max_model_len=%d)",
            min(caps), max(caps),
            f"user={cfg.max_tokens}" if cfg.max_tokens is not None else f"auto={AUTO_MAX_NEW_TOKENS}",
            self._max_model_len,
        )
        return params

    def generate(
        self,
        examples: Iterable[UnifiedExample],
        cfg: GenerationConfig | None = None,
    ) -> list[list[str]]:
        'Generate completions for each example.\n\n\n        Returns one list-of-strings per example: length 1 in greedy mode, length\n        ``cfg.num_samples`` in sampling mode. The caller (run_eval.py) pairs\n        them back with the examples for scoring.'
        cfg = cfg or GenerationConfig()
        examples = list(examples)
        prompts = [render_prompt(ex, self._tokenizer) for ex in examples]
        if not prompts:
            return []
        # Token counts for the per-example max_tokens budget. add_special_tokens
        # matches how vLLM tokenizes string prompts (Qwen2-style tokenizers
        # have no BOS, so the counts agree with the engine's own).
        
        
        encoded = self._tokenizer(prompts, add_special_tokens=False)
        prompt_token_counts = [len(ids) for ids in encoded["input_ids"]]
        params = self._build_sampling_params(cfg, prompt_token_counts)
        # vLLM LLM.generate accepts a list of SamplingParams aligned with prompts.
        # vLLM LLM.generate
        outputs = self._llm.generate(prompts, params)
        return [[o.text for o in out.outputs] for out in outputs]
