'Multi-task reward function — TRL-compatible, uniform across all task classes.\n\n\nThis is the **reuse point** between ``run_eval.py`` and the future ``grpo_train.py``.\n:func:`make_reward_function` returns a callable with the exact signature TRL\'s\n``GRPOTrainer`` expects (see TRL GRPOTrainer docs, "Custom reward function"):\n\n:func:`make_reward_function`\n\n* async (TRL awaits async reward functions concurrently);\n* receives ``prompts``, ``completions``, ``completion_ids`` plus every dataset\n  column as a kwarg — so ``task_class``, ``reference`` and ``extra`` arrive\n  automatically (the unified dataset keeps them as columns);\n* returns ``list[float]`` (one reward per completion).\n\nThe function is now **uniform** across all 25 task classes: the per-class logic\n(SemSim always; AC if the scorer\'s judge is configured; format-check for\ntool-use) lives inside the scorer itself (Variant B of the protocol). The\ndispatcher\'s only job is to extract the user question for the judge\'s\n``user_input`` field (via :func:`extract_user_question`) and call the scorer.\n\n``user_input``'

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from llm_training.reward.config import RewardConfig, ScoredMetrics, Scorer
from llm_training.reward.inputs import extract_user_question
from llm_training.tokenizer_utils import assistant_content_str

logger = logging.getLogger(__name__)

#: A TRL-compatible reward callable built by :func:`make_reward_function`.

RewardFunc = Callable[..., Any]

#: Sentinel ``failed_metrics`` value used when the dispatcher itself (not a
#: metric) crashed — e.g. an ``extract_user_question`` bug. Kept distinct from
#: the metric names ("semantic_similarity", "answer_correctness") so that
#: run_eval.py's per-class failure counter reports dispatcher failures
#: honestly instead of mis-attributing them to SemSim.

# : run_eval.py
DISPATCHER_FAILURE = "__dispatcher__"


def completion_to_text(completion: Any) -> str:
    'Normalize a TRL completion (string or conversational messages) to text.\n\n\n    TRL passes either a list of message dicts (conversational format — our case)\n    or a plain string (standard format). Tool-call blocks are preserved by\n    :func:`assistant_content_str` so tool-use references are comparable verbatim\n    (and :func:`format_check` can still find ``<tool_call>`` blocks in the\n    rendered text). TRL'
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return assistant_content_str(completion)
    # Defensive: some backends may pass a single message dict.
    
    if isinstance(completion, dict):
        return str(completion.get("content", "") or "")
    return str(completion)


def fallback_scored_metrics(reason: str = DISPATCHER_FAILURE) -> ScoredMetrics:
    'Conservative ScoredMetrics returned when the scorer/dispatcher crashes.\n\n\n    A failure here must never abort the batch (it would cancel the whole\n    :func:`asyncio.gather`). The combined reward is ``0.0`` and\n    ``failed_metrics`` records ``reason`` (default :data:`DISPATCHER_FAILURE`)\n    so consumers can tell dispatcher crashes apart from genuine metric\n    failures. Used by both :func:`make_reward_function` (grpo_train.py path) and\n    :class:`~llm_training.runtime.reward_service.RewardService` (run_eval.py\n    path) — a single source of truth so the two stay in sync.'
    return ScoredMetrics(
        semantic_similarity=0.0,
        combined=0.0,
        answer_correctness=None,
        format_score=None,
        failed_metrics=(reason,),
    )


def make_reward_function(
    scorer: Scorer,
    *,
    reward_config: RewardConfig | None = None,
    concurrency: int | None = None,
) -> RewardFunc:
    'Build a TRL-compatible async reward function over ``scorer``.\n\n\nArgs\n        scorer: anything satisfying :class:`~llm_training.reward.config.Scorer`\n            (the real RAGAS scorer, or a fake in tests\n        reward_config: the reward configuration. When ``concurrency`` is\n            ``None``, the service-level semaphore is derived from it as\n            ``max(embedding_concurrency, judge_concurrency)`` (the Level-1\n            service limit mandated by AGENTS.md\n        concurrency: explicit override for the max simultaneous scorer requests\n            inside one reward call. Takes precedence over ``reward_config``.\n            ``reward_config``).\n\nRaises\n        ValueError: if neither ``concurrency`` nor ``reward_config`` is supplied\n            (one of them is required to size the service semaphore), or if the\n            resolved value is < 1.\n\nReturns\n        An ``async def reward_func(prompts, completions, completion_ids=None,\n        **kwargs) -> list[float]`` ready for ``GRPOTrainer(reward_funcs=...)``.\n\nNote\n        TRL invokes async reward functions and awaits them; a failed scorer call\n        on a single example maps to a conservative ``0.0`` via\n        :func:`fallback_scored_metrics`, never aborting the batch. TRL\n        async reward'
    if concurrency is None:
        if reward_config is None:
            raise ValueError(
                'make_reward_function requires either `concurrency` or `reward_config` to size the service semaphore.\nmake_reward_function'
            )
        concurrency = max(
            reward_config.embedding_concurrency,
            reward_config.judge_concurrency,
            reward_config.fluency_concurrency,
        )
    if concurrency < 1:
        raise ValueError('concurrency must be >= 1.')

    sem = asyncio.Semaphore(concurrency)

    async def _score_one(
        prompt: Any,
        completion: Any,
        task_class: str,
        reference: str,
        extra: Any,
        question: Any = None,
    ) -> ScoredMetrics:
        async with sem:
            try:
                response = completion_to_text(completion)
                # Prefer the precomputed question column (grpo_train.py passes
                # pre-rendered TEXT prompts, so extraction from the prompt is
                # impossible); fall back to extracting from the prompt
                # (run_eval.py path with message-list prompts).
                
                
                if isinstance(question, str) and question.strip():
                    user_input = question
                else:
                    user_input = extract_user_question(task_class, prompt, extra)
                return await scorer.score(
                    user_input=user_input,
                    response=response,
                    reference=reference,
                    task_class=task_class,
                    extra=extra if isinstance(extra, dict) else None,
                )
            except Exception as e:  # noqa: BLE001 - never abort the batch
                logger.warning(
                    "Scorer failed for task_class=%s: %s", task_class, e
                )
                return fallback_scored_metrics()

    async def reward_func(
        prompts: list[Any],
        completions: list[Any],
        completion_ids: list[Any] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        # TRL forwards dataset columns as parallel lists aligned with
        # ``completions``. ``task_class`` and ``reference`` are always present
        # (unified-dataset columns); ``extra`` usually is too.
        
        task_classes: list[str] = list(kwargs.get("task_class") or [])
        references: list[str] = list(kwargs.get("reference") or [])
        extras: list[Any] = list(kwargs.get("extra") or [])
        # grpo_train.py supplies a precomputed ``question`` column (extracted from
        # the ORIGINAL message list at dataset-build time) because its prompt
        # column is a pre-rendered TEXT — extracting the question from raw
        # rendered text would be fragile. When absent (run_eval.py path, which
        # passes message-list prompts), the question is extracted from the
        # prompt inside ``_score_one`` as before.
        # grpo_train.py
        
        
        questions: list[Any] = list(kwargs.get("question") or [])

        n = len(completions)
        # Defensive tiling if TRL did not expand a column to the completions
        # length (should not happen, but never index out of range).
        
        if len(task_classes) < n:
            if not task_classes:
                raise ValueError(
                    "reward_func received no 'task_class' column; the unified dataset must carry it.\nreward_func"
                )
            base_tc = task_classes[0]
            task_classes = [task_classes[i] if i < len(task_classes) else base_tc for i in range(n)]
        if len(references) < n:
            if not references:
                raise ValueError(
                    "reward_func received no 'reference' column; the unified dataset must carry it.\nreward_func"
                )
            references = [references[i] if i < len(references) else "" for i in range(n)]
        if len(extras) < n:
            base_ex = extras[0] if extras else {}
            extras = [extras[i] if i < len(extras) else base_ex for i in range(n)]
        if len(questions) < n:
            base_q = questions[0] if questions else None
            questions = [questions[i] if i < len(questions) else base_q for i in range(n)]

        coros = [
            _score_one(
                prompts[i], completions[i], task_classes[i], references[i],
                extras[i], question=questions[i],
            )
            for i in range(n)
        ]
        results = await asyncio.gather(*coros)
        return [m.combined for m in results]

    reward_func.__name__ = "multi_task_ragas_reward"
    reward_func.__doc__ = (
        "TRL-compatible multi-task reward (Semantic Similarity + optional "
        "Answer Correctness + format-check for tool-use, harmonic-combined). "
        "Built by make_reward_function()."
    )
    return reward_func
