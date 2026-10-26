"run_eval.py — metrics-only evaluation of an LLM on the unified dataset.\n\nrun_eval.py\n\nThis script scores a **policy** model against human reference answers. The reward\ncombines (a) **Semantic Similarity** (RAGAS ``AnswerSimilarity``, embeddings\nonly — always), (b) **Answer Correctness** (RAGAS ``AnswerCorrectness``, via an\nLLM-as-Judge — when ``--judge-*`` is provided), (c) a **format check** for\ntool-use classes, (d) **code_check** (Python ``ast``) for the ``coder`` class,\nand (e) **fluency** (in-process grammar classifier + ``razdel``) for the\nnatural-language classes. The same scorer is reused by the future ``grpo_train.py``\nfor the GRPO loop's reward. **No training** happens here.\n\n\nArchitecture (parallel GPU partition)\n\n* the **policy** runs as an in-process :class:`vllm.LLM` on the policy GPU pool;\n* the **embeddings** are computed in-process via ``sentence-transformers``\n  (default ``BAAI/bge-m3``), sharing the policy GPU — no HTTP server needed;\n* the **fluency classifier** is likewise in-process via ``transformers``\n  (default ``agentlans/snowflake-arctic-xs-grammar-classifier``), also sharing\n  the policy GPU;\n* the **judge** is an external OpenAI-compatible HTTP endpoint (a local\n  ``vllm serve`` the operator started, or a cloud provider) — this script never\n  launches or owns it;"

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

from llm_training.build import build_unified_dataset
from llm_training.config import (
    DEFAULT_MAX_PER_CLASS,
    MAX_REFERENCE_TOKENS,
    MAX_TOTAL_TOKENS,
    DataConfig,
)
from llm_training.inference.generate import GenerationConfig, PolicyGenerator
from llm_training.reward.config import (
    DEFAULT_FLUENCY_CLASSES,
    DEFAULT_LOCAL_FLUENCY_MODEL,
    RewardConfig,
)
from llm_training.reward.judge import make_ragas_scorer
from llm_training.runtime.endpoints import OpenAIEndpoint
from llm_training.runtime.reward_service import RewardService, ScoreRequest
from llm_training.schema import TASK_CLASSES, UnifiedExample
from llm_training.tokenizer_utils import load_tokenizer

logger = logging.getLogger("run_eval")

#: Default examples-per-batch for the generate+score loop.

DEFAULT_BATCH_SIZE = 16


def set_seed(seed: int) -> None:
    'Seed every RNG the evaluation touches (Python, NumPy, torch, HF, vLLM).\n\n\n    Per AGENTS.md: ``--seed`` governs RNG reproducibility on CPU and CUDA; it\n    does **not** pick the data split (the operator selects task classes via CLI).\n    The vLLM seed is applied later via :class:`GenerationConfig`.  ``--seed``'
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch
        torch.manual_seed(seed)
        # NOTE: torch.cuda.manual_seed_all is intentionally omitted here.
        # Calling it initializes the CUDA runtime in the parent process, which
        # breaks vLLM's forked EngineCore subprocess ("Cannot re-initialize
        # CUDA in forked subprocess"). vLLM generation reproducibility is
        # governed by SamplingParams.seed (see GenerationConfig.seed), not by
        # the PyTorch CUDA RNG. The future grpo_train.py will seed CUDA in the
        # GRPOTrainer context where vLLM runs as a separate server process.
    except ImportError:  # pragma: no cover
        pass
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except ImportError:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _stats(values: Sequence[float]) -> dict[str, float | int]:
    'Mean/std/median/min/max/n for a list of floats (empty → zeros).'
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "n": n,
    }


#: Metric names carried through to the report's per-class block. ``combined``
#: and (usually) ``semantic_similarity`` are always populated; ``nerel_f1``
#: replaces ``semantic_similarity`` for the NEREL extraction sub-configs and
#: ``docsem_score`` for ``docsem_quant`` (where SemSim is not computed);
#: ``answer_correctness`` and ``format_score`` may be
#: ``None`` when not computed (no judge; non-tool class); ``rubric_score``
#: appears only with ``--judge-rubrics`` (SGR reasoning turns).

_REPORT_METRICS: tuple[str, ...] = (
    "combined",
    "semantic_similarity",
    "answer_correctness",
    "rubric_score",
    "format_score",
    "fluency",
    "code_check",
    "nerel_f1",
    "docsem_score",
)

#: Metric names tracked in the ``metric_failures`` counter. Only metrics that
#: were expected (computed at all) can fail; non-applicable ones never count.
#: ``code_check`` is a pure function (never fails) so it is not listed;
#: ``docsem_check`` fails only on missing gold metadata (a data bug).

_FAILURE_METRICS: tuple[str, ...] = (
    "semantic_similarity",
    "answer_correctness",
    "rubrics",
    "fluency",
    "docsem_check",
)


def _aggregate_examples(per_example: list[dict]) -> dict[str, Any]:
    'Aggregate per-example metric dicts into per-class + overall summaries.'
    by_class: dict[str, list[dict]] = {}
    for row in per_example:
        by_class.setdefault(row["task_class"], []).append(row)

    def _class_block(rows: list[dict]) -> dict[str, Any]:
        block: dict[str, Any] = {"n": len(rows)}
        for metric in _REPORT_METRICS:
            vals = [
                r["metrics"].get(metric) for r in rows
                if r["metrics"].get(metric) is not None
            ]
            if vals:
                block[metric] = _stats(vals)
            else:
                block[metric] = None
        # Per-metric failure counts: how many examples in this class had each
        # metric expected but failed due to a judge/embedding error. Helps the
        # operator interpret a deflated ``combined`` (a failed metric maps to
        # ``None`` → ``safe_score(None) = 0.0`` → drags the harmonic mean down).
        
        
        block["metric_failures"] = {
            m: sum(
                1 for r in rows
                if m in r["metrics"].get("failed_metrics", [])
            )
            for m in _FAILURE_METRICS
        }
        return block

    per_class = {cls: _class_block(rows) for cls, rows in sorted(by_class.items())}
    overall = _class_block(per_example) if per_example else {"n": 0}
    return {"per_class": per_class, "overall": overall}


# ---------------------------------------------------------------------------
# Core evaluation pipeline (testable with injected fakes)

# ---------------------------------------------------------------------------
def run_evaluation(
    generator: PolicyGenerator,
    service: RewardService,
    examples: Sequence[UnifiedExample],
    gen_cfg: GenerationConfig,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    detailed: bool = False,
    on_progress: Any = None,
) -> dict[str, Any]:
    'Generate + score every example and return the metrics report dict.\n\n\n    This is split out of :func:`main` so the pipeline can be tested end-to-end\n    with a fake generator + service (no vLLM, no judge network\n    :func:`main`'
    examples = list(examples)
    per_example: list[dict] = []
    n = len(examples)
    for start in range(0, n, batch_size):
        batch = examples[start : start + batch_size]
        # Generate returns list[list[str]] aligned with `batch` (1 sample in
        # greedy mode, N in sampling mode).
        
        samples_per_ex = generator.generate(batch, gen_cfg)
        # Build one ScoreRequest per (example, sample) and score them together
        # so the judge batches across both examples and samples.
        
        flat_reqs: list[ScoreRequest] = []
        owner_idx: list[tuple[int, int]] = []  # (example_index, sample_index)
        for ex_i, (ex, samples) in enumerate(zip(batch, samples_per_ex)):
            for s_i, sample_text in enumerate(samples):
                flat_reqs.append(
                    ScoreRequest(
                        prompt=ex.prompt,
                        completion=sample_text,
                        task_class=ex.task_class,
                        reference=ex.reference,
                        extra=dict(ex.extra),
                        source=ex.source,
                    )
                )
                owner_idx.append((start + ex_i, s_i))
        metrics_flat = service.score_batch(flat_reqs)

        # Group metrics back by example, averaging across samples per metric.
        
        by_ex: dict[int, list[Any]] = {}
        for (ex_i, _s_i), m in zip(owner_idx, metrics_flat):
            by_ex.setdefault(ex_i, []).append(m)
        for ex_i, ex in enumerate(batch):
            ms = by_ex.get(start + ex_i, [])
            if not ms:
                continue
            def _maybe_mean(field: str) -> float | None:
                vals = [getattr(m, field) for m in ms if getattr(m, field) is not None]
                if not vals:
                    return None
                return statistics.fmean(vals)
            agg_metrics: dict[str, Any] = {
                "combined": statistics.fmean([m.combined for m in ms]),
                "semantic_similarity": _maybe_mean("semantic_similarity"),
                "answer_correctness": _maybe_mean("answer_correctness"),
                "format_score": _maybe_mean("format_score"),
                "fluency": _maybe_mean("fluency"),
                "code_check": _maybe_mean("code_check"),
                "nerel_f1": _maybe_mean("nerel_f1"),
            }
            # Union of failed metric names across samples (for the per-class
            # failure counter in the report). For N>1 samples, mark a metric
            # failed only if it failed in all samples; otherwise at least one
            # sample produced a real value.
            
            if len(ms) == 1:
                agg_metrics["failed_metrics"] = list(ms[0].failed_metrics)
            else:
                agg_metrics["failed_metrics"] = [
                    name for name in _FAILURE_METRICS
                    if all(name in m.failed_metrics for m in ms)
                ]
            row = {
                "task_class": ex.task_class,
                "source": ex.source,
                "extra": dict(ex.extra),
                "n_samples": len(ms),
                "metrics": agg_metrics,
            }
            if detailed:
                row["samples"] = [
                    {
                        "text": samples_per_ex[ex_i][s_i],
                        "metrics": ms[s_i].as_dict(),
                    }
                    for s_i in range(len(ms))
                ]
            per_example.append(row)
        if on_progress:
            on_progress(start + len(batch), n)

    agg = _aggregate_examples(per_example)
    return {
        "per_class": agg["per_class"],
        "overall": agg["overall"],
        "examples": per_example if detailed else [],
        "n_examples": len(per_example),
    }


# ---------------------------------------------------------------------------
# CLI / CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_eval.py",
        description=(
            'Metrics-only LLM evaluation on the unified dataset (Semantic Similarity + Answer Correctness via an LLM-as-Judge + tool-use format check + code_check for `coder` + fluency for NL classes). No training.'
        ),
    )
    # --- Policy (in-process vLLM) / Policy (in-process vLLM) ---
    p.add_argument("--model", required=True, help='Policy model id or local path')
    p.add_argument("--policy-gpu-ids", type=int, nargs="+", default=None,
                   help='CUDA GPU ids for the policy pool')
    p.add_argument("--policy-tp", type=int, default=1, help="Policy tensor-parallel size / TP policy.")
    p.add_argument("--policy-gpu-memory-utilization", type=float, default=0.9,
                   help='vLLM gpu_memory_utilization for the policy')
    p.add_argument("--policy-max-model-len", type=int, default=None,
                   help="vLLM max_model_len for the policy / max_model_len policy.")
    p.add_argument("--policy-dtype", type=str, default="auto", help="vLLM dtype / dtype.")

    # --- Judge (OpenAI-compatible endpoint)
    p.add_argument("--judge-base-url", default=None,
                   help='Judge base_url, e.g. http://localhost:8001/v1 or https://vsellm.ru/v1. Env: JUDGE_BASE_URL')
    p.add_argument("--judge-model", default=None,
                   help='Judge model name on the endpoint. Env: JUDGE_MODEL')
    p.add_argument("--judge-api-key", default=None,
                   help='Judge API key (optional for local vLLM; required for cloud). Env: JUDGE_API_KEY')

    # --- Embeddings (in-process via sentence-transformers)
    p.add_argument("--local-embedding-model", default="BAAI/bge-m3",
                   help='HuggingFace model id for in-process embeddings via sentence-transformers (default: BAAI/bge-m3, multilingual, 1024-dim). The model is downloaded from HuggingFace on first use and shares the policy GPU. Env: LOCAL_EMBEDDING_MODEL.')
    p.add_argument("--local-embedding-device", default=None,
                   help="Device for the local embedding model (default: auto-detect CUDA). Use 'cpu' to force CPU, 'cuda' for GPU.")

    # --- Fluency reward (grammar classifier) + code_check (Python ast) / Fluency + code_check ---
    p.add_argument("--fluency-model", default=None,
                   help='HuggingFace model id for the in-process fluency classifier (default: agentlans/snowflake-arctic-xs-grammar-classifier). Env: LOCAL_FLUENCY_MODEL.')
    p.add_argument("--fluency-device", default=None,
                   help='Device for the fluency model (default: auto-detect).')
    p.add_argument("--fluency-concurrency", type=int, default=None,
                   help='Max concurrent in-process fluency forward passes (default: same as --embedding-concurrency). The fluency model shares the policy GPU.')
    p.add_argument("--fluency-calibrator", default=None,
                   help='Path to the JSON calibrator artifact produced once by scripts/calibrate_fluency.py. None = no calibration (raw P(grammatical); the default')
    p.add_argument("--no-fluency", action="store_true",
                   help='Disable the fluency signal entirely (no model load).')
    p.add_argument("--no-code-check", action="store_true",
                   help='Disable the code_check signal (Python ast validation of coder answers).')
    p.add_argument("--no-nerel-f1", action="store_true",
                   help='Disable the deterministic NEREL F1 signal (for the NEREL extraction sub-configs it otherwise replaces Semantic Similarity).')
    p.add_argument("--no-docsem-check", action="store_true",
                   help='Disable the deterministic DocSem check (for docsem_quant it otherwise replaces Semantic Similarity with the executable PoT scoring).')
    p.add_argument("--no-rummlu-em", action="store_true",
                   help='Disable the deterministic rummlu_pro letter exact match (for the class it otherwise replaces Semantic Similarity).')
    p.add_argument("--no-rerank-em", action="store_true",
                   help='Disable the deterministic meno_rerank verdict grading (0.3 valid JSON + 0.7 graded by the ordinal label distance; for the class it otherwise replaces Semantic Similarity and skips Answer Correctness).')
    p.add_argument("--schema-check", action="store_true",
                   help='Enable the deterministic tool-call schema_check (argument validation against the prompt-declared JSON schemas; default OFF')
    p.add_argument("--language-guard", action="store_true",
                   help="Enable the fluency language guard: multiply the fluency score by the confidence of the example's target language (extra['target_language'], computed at build time)")
    p.add_argument("--reasoning-semsim", action="store_true",
                   help='Score SGR-agent reasoning turns with Semantic Similarity (default OFF: reasoning turns use fluency + the SGR-schema structural check, since reasoning is open-ended and SemSim would reward paraphrase-imitation).')
    p.add_argument("--judge-rubrics", action="store_true",
                   help='Score SGR reasoning turns (where Answer Correctness is not applicable) with judge rubrics: Russian-language literacy + consistency with the input context; adds rubric_score to the report (needs a judge endpoint).')
    p.add_argument("--fluency-guard", type=float, default=None, metavar="THRESHOLD",
                   help='an optional degradation brake (grpo_train.py no longer enables it by default). Default here: OFF, so eval reports the raw P(grammatical) unchanged.')
    p.add_argument("--fluency-class", action="append", default=None, dest="fluency_class",
                   help='Add a task class to the fluency allowlist (repeatable).')
    p.add_argument("--no-fluency-class", action="append", default=None, dest="no_fluency_class",
                   help='Remove a task class from the fluency allowlist (repeatable).')

    # --- Dataset
    p.add_argument("--task", action="append", default=None, dest="task",
                   help='Task class to evaluate (repeatable; default: all non-network).')
    p.add_argument("--seed", type=int, default=42, help='RNG seed (logged).')
    p.add_argument("--max-per-class", type=int, default=DEFAULT_MAX_PER_CLASS,
                   help='Per-class cap after length discard')
    p.add_argument("--max-total-tokens", type=int, default=MAX_TOTAL_TOKENS)
    p.add_argument("--max-reference-tokens", type=int, default=MAX_REFERENCE_TOKENS)
    p.add_argument("--include-network", action="store_true",
                   help='Include nerel_instruct (needs HuggingFace access).')
    p.add_argument("--generation-mode", choices=("greedy", "sampling"), default="greedy",
                   help='Policy generation mode')
    p.add_argument("--num-samples", type=int, default=1,
                   help='Samples per prompt in sampling mode')
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument(
        "--max-tokens", type=int, default=None,
        help="never vLLM's own 16-token default, which silently truncates completions. An explicit value acts as an upper bound, still clamped to the remaining context.",
    )
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help='Examples per generate+score batch')
    p.add_argument("--judge-concurrency", type=int, default=8,
                   help='Max in-flight judge requests')
    p.add_argument("--embedding-concurrency", type=int, default=None,
                   help='Max concurrent in-process embedding computations (default: same as --judge-concurrency). The embeddings share the policy GPU, so this caps GPU contention. Independent from --judge-concurrency.')
    p.add_argument("--judge-max-tokens", type=int, default=4096,
                   help='Max tokens for the judge LLM structured output. RAGAS defaults to 1024 which truncates the verbose CorrectnessClassifier JSON; with a thinking judge (GLM on Z.AI) the reasoning counts toward the same cap')
    p.add_argument("--judge-max-retries", type=int, default=2,
                   help='Extra attempts (with backoff) on transient judge errors (HTTP 429/5xx, proxy 400).')
    p.add_argument("--judge-timeout", type=float, default=None,
                   help="e.g. thinking models whose reasoning cannot be disabled (GLM on Z.AI: thinking.type supports only 'enabled'); scale it with --judge-max-tokens, e.g. 1800 for 16384 tokens at ~15-60 tok/s.")
    p.add_argument("--no-answer-correctness", action="store_true",
                   help='Disable the LLM judge (Semantic Similarity + format + code_check + fluency still run; AC will be None in the report). Note: fluency still loads the grammar classifier; pass --no-fluency for the lightest mode.')
    p.add_argument("--output", default="evaluate_report.json", help='JSON report path')
    p.add_argument("--detailed", action="store_true", help="Include per-example + per-sample dump in the report.")
    return p


def _normalize_task_classes(requested: list[str] | None) -> list[str]:
    'Resolve the ``--task`` list (None → all non-network classes).'
    if requested:
        for cls in requested:
            if cls not in TASK_CLASSES:
                raise SystemExit(
                    f'Unknown task class {cls!r}. Known: {TASK_CLASSES}'
                )
        return requested
    return [c for c in TASK_CLASSES if c != "nerel_instruct"]


def _build_report(
    args: argparse.Namespace,
    reward_cfg: RewardConfig,
    examples_count: int,
    build_stats: dict,
    results: dict[str, Any],
) -> dict[str, Any]:
    _cfg_desc = reward_cfg.describe()
    return {
        "config": {
            "model": args.model,
            "judge": _cfg_desc["judge"],
            "embeddings": _cfg_desc["embeddings"],
            "fluency": _cfg_desc["fluency"],
            "compute_answer_correctness": reward_cfg.compute_answer_correctness,
            "known_tool_names_count": len(reward_cfg.known_tool_names),
            "seed": args.seed,
            "generation_mode": args.generation_mode,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "task_classes_evaluated": args.task or "all-non-network",
            "max_per_class": args.max_per_class,
            "include_network": args.include_network,
            "judge_max_tokens": reward_cfg.judge_max_tokens,
            "judge_max_retries": reward_cfg.judge_max_retries,
            "judge_timeout_s": reward_cfg.judge_timeout_s,
        },
        "build_stats": build_stats,
        "n_examples": examples_count,
        "metrics": results,
    }


def _resolve_endpoint_settings(
    args: argparse.Namespace, env: dict[str, str], *, require_judge: bool,
) -> tuple[str | None, str | None]:
    'Resolve judge base_url + model: CLI flag > env var.\n\n\n    Per AGENTS.md, a CLI flag paired with an env var is never ``required``; the\n    env var is the runtime default, and an explicit flag always wins.\n    ``--no-answer-correctness`` makes the judge optional\n    (``require_judge=False``). Raises :class:`SystemExit` with a bilingual\n    message if a required value is still missing. API keys are resolved later\n    inside :class:`OpenAIEndpoint` (``_resolve_key``). (Embeddings are computed\n    in-process and have no endpoint to resolve here.)'
    judge_base_url = args.judge_base_url or env.get("JUDGE_BASE_URL")
    judge_model = args.judge_model or env.get("JUDGE_MODEL")
    missing: list[str] = []
    if require_judge:
        if not judge_base_url:
            missing.append("--judge-base-url (env JUDGE_BASE_URL)")
        if not judge_model:
            missing.append("--judge-model (env JUDGE_MODEL)")
    if missing:
        raise SystemExit(
            "Missing required endpoint setting(s): " + ", ".join(missing)
            + 'Operation details unavailable.' + ", ".join(missing)
        )
    return judge_base_url, judge_model


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = build_arg_parser().parse_args(argv)

    logger.info("run_eval.py: seed=%d", args.seed)
    set_seed(args.seed)

    task_classes = _normalize_task_classes(args.task)
    args.task = task_classes
    logger.info("Task classes to evaluate: %s", task_classes)

    # Resolve endpoint settings (CLI flag > env var) before building the
    # OpenAIEndpoint descriptor. API keys stay resolved inside
    # OpenAIEndpoint._resolve_key. Embeddings are in-process (no endpoint).
    
    require_judge = not args.no_answer_correctness
    local_embedding_model = (
        args.local_embedding_model or os.environ.get("LOCAL_EMBEDDING_MODEL")
    )
    local_embedding_device = args.local_embedding_device
    local_fluency_model = (
        args.fluency_model or os.environ.get("LOCAL_FLUENCY_MODEL")
    ) or DEFAULT_LOCAL_FLUENCY_MODEL
    # Build the fluency allowlist from the default ± CLI edits.
    
    fluency_classes = set(DEFAULT_FLUENCY_CLASSES)
    if args.fluency_class:
        fluency_classes |= set(args.fluency_class)
    if args.no_fluency_class:
        fluency_classes -= set(args.no_fluency_class)
    judge_base_url, judge_model = _resolve_endpoint_settings(
        args, os.environ, require_judge=require_judge,
    )

    # Build dataset with the policy's tokenizer FIRST so we can collect the
    # tool-name registry (used by format_check) before wiring the reward cfg.
    
    tokenizer = load_tokenizer(args.model)
    data_cfg = DataConfig(
        seed=args.seed,
        max_per_class=args.max_per_class,
        max_total_tokens=args.max_total_tokens,
        max_reference_tokens=args.max_reference_tokens,
    )
    build_result = build_unified_dataset(
        tokenizer,
        data_cfg,
        val_task=None,
        test_task=None,
        task_classes=task_classes,
        include_network=args.include_network,
    )
    examples = build_result.train
    logger.info(
        "Built %d examples for evaluation (held-out split not used).", len(examples)
    )
    logger.info(
        "Collected %d known tool names for format_check.",
        len(build_result.tool_names),
    )

    # Reward wiring / Reward.
    # Embeddings are always in-process (sentence-transformers on the chosen
    # device, sharing the policy GPU); the judge is an external HTTP endpoint.
    
    logger.info(
        "Using in-process embeddings: model=%s, device=%s",
        local_embedding_model, local_embedding_device or "auto",
    )
    if require_judge and judge_base_url and judge_model:
        judge_ep = OpenAIEndpoint(
            base_url=judge_base_url, model=judge_model,
            api_key=args.judge_api_key, kind="judge",
        )
        compute_ac = True
    else:
        judge_ep = None
        compute_ac = False
    # --judge-timeout: None → the RewardConfig default (120 s). An explicit
    # value covers slow cloud judges (thinking models with non-disableable
    # reasoning, GLM on Z.AI). Passed via kwargs to avoid duplicating the
    # default here.
    # --judge-timeout: None →
    
    
    extra_reward_kwargs: dict[str, float] = {}
    if args.judge_timeout is not None:
        extra_reward_kwargs["judge_timeout_s"] = args.judge_timeout
    reward_cfg = RewardConfig(
        local_embedding_model=local_embedding_model,
        local_embedding_device=local_embedding_device,
        judge=judge_ep,
        compute_answer_correctness=compute_ac,
        known_tool_names=build_result.tool_names,
        known_tool_schemas=build_result.tool_schemas,
        judge_concurrency=args.judge_concurrency,
        embedding_concurrency=args.embedding_concurrency,
        judge_max_tokens=args.judge_max_tokens,
        judge_max_retries=args.judge_max_retries,
        local_fluency_model=local_fluency_model,
        local_fluency_device=args.fluency_device,
        fluency_concurrency=args.fluency_concurrency,
        fluency_classes=frozenset(fluency_classes),
        fluency_calibrator_path=args.fluency_calibrator,
        enable_fluency=not args.no_fluency,
        enable_code_check=not args.no_code_check,
        enable_nerel_f1=not args.no_nerel_f1,
        enable_docsem_check=not args.no_docsem_check,
        enable_reasoning_semsim=args.reasoning_semsim,
        enable_judge_rubrics=args.judge_rubrics,
        enable_schema_check=args.schema_check,
        enable_language_guard=args.language_guard,
        enable_rummlu_em=not args.no_rummlu_em,
        enable_rerank_em=not args.no_rerank_em,
        fluency_guard_threshold=args.fluency_guard,
        **extra_reward_kwargs,
    )
    if judge_ep is not None:
        logger.info("Judge endpoint: %s", judge_ep.describe())
    else:
        logger.info("Judge disabled (--no-answer-correctness); AC will be None.")
    logger.info("Embeddings: %s", reward_cfg.describe()["embeddings"])
    logger.info("Fluency: %s", reward_cfg.describe()["fluency"])

    gen_cfg = GenerationConfig(
        mode=args.generation_mode,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    scorer = make_ragas_scorer(reward_cfg)
    generator = PolicyGenerator(
        model=args.model,
        gpu_ids=tuple(args.policy_gpu_ids) if args.policy_gpu_ids else None,
        tensor_parallel_size=args.policy_tp,
        gpu_memory_utilization=args.policy_gpu_memory_utilization,
        max_model_len=args.policy_max_model_len,
        dtype=args.policy_dtype,
    )

    def _progress(done: int, total: int) -> None:
        logger.info("Scored %d / %d examples", done, total)

    # The RewardService semaphore limits concurrent scorer calls (overall).
    # The per-endpoint embedding/judge semaphores inside RAGASScorer provide
    # finer-grained control. Setting the service-level limit to the max of the
    # two ensures neither endpoint is artificially throttled by the service.
    
    # per-endpoint
    _overall_concurrency = max(
        reward_cfg.embedding_concurrency,
        reward_cfg.judge_concurrency,
        reward_cfg.fluency_concurrency,
    )
    with RewardService(
        scorer, concurrency=_overall_concurrency
    ) as service:
        results = run_evaluation(
            generator,
            service,
            examples,
            gen_cfg,
            batch_size=args.batch_size,
            detailed=args.detailed,
            on_progress=_progress,
        )

    report = _build_report(args, reward_cfg, len(examples), build_result.stats, results)
    out_path = Path(args.output)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote report to %s", out_path)

    # Brief stdout summary
    overall = results["overall"]
    if "combined" not in overall:
        # Empty dataset (zero examples after build): ``_aggregate_examples``
        # returns a degenerate ``{"n": 0}`` block with no metric stats — print a
        # friendly message instead of crashing on ``overall['combined']``.
        
        print(
            f'No examples to evaluate (n=0). Check --task / --max-per-class / data filters.',
            file=sys.stderr,
        )
        return 0
    print(
        f"Overall combined reward: {overall['combined']['mean']:.4f} "
        f"(n={overall['n']}, median={overall['combined']['median']:.4f})",
        file=sys.stderr,
    )
    for cls, block in sorted(results["per_class"].items()):
        if "combined" not in block:
            continue
        print(
            f"  {cls:30s} n={block['n']:4d} combined={block['combined']['mean']:.4f}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
