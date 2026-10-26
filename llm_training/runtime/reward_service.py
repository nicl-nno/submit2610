"RewardService — run_eval.py's batch scorer (concurrency + per-metric output).\n\nRewardService\n\n:mod:`llm_training.reward.dispatch` gives ``grpo_train.py`` a minimal TRL-compatible\nreward function returning ``list[float]``. ``run_eval.py`` needs more: the full\n:class:`~llm_training.reward.config.ScoredMetrics` breakdown per example (for the\nJSON report), batched + concurrency-limited scoring, and a clean sync entry point\n(the generation loop is sync vLLM). That is what :class:`RewardService` provides.\n\n:mod:`llm_training.reward.dispatch`\n:class:`ScoredMetrics`\n\nThe service owns a single persistent event loop (in a background thread) so the\njudge's ``AsyncOpenAI`` client stays bound to one loop across batches (creating a\nfresh loop per batch would break httpx transport reuse"

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from llm_training.reward.config import ScoredMetrics, Scorer
from llm_training.reward.inputs import extract_user_question
from llm_training.reward.dispatch import completion_to_text, fallback_scored_metrics

logger = logging.getLogger(__name__)


@dataclass
class ScoreRequest:
    "One example to score, as built by the evaluation pipeline.\n\n\nAttributes\n        prompt: the conversational prompt (``list[Message]``) shown to the policy.\n        completion: the policy's generated answer (string or ``list[Message]``).\n        task_class: the unified-dataset task class (dispatches format-check, and\n            question extraction for the judge\n        reference: the human reference answer string\n        extra: the example metadata dict (e.g. mtrageval ``context_text``).\n        source: optional provenance string, carried through to the report."

    prompt: list[dict]
    completion: Any
    task_class: str
    reference: str
    extra: dict[str, Any] = field(default_factory=dict)
    source: str = ""


class RewardService:
    'Batched, concurrency-limited scorer returning full metric breakdowns.\n\n\n    Use as a context manager so the background event-loop thread is always torn\n    down: ``with RewardService(scorer, ...) as svc: svc.score_batch(reqs)``.'

    def __init__(
        self,
        scorer: Scorer,
        *,
        concurrency: int = 8,
        batch_timeout: float = 3600.0,
    ) -> None:
        if concurrency < 1:
            raise ValueError(
                'concurrency must be >= 1.'
            )
        if batch_timeout <= 0:
            raise ValueError(
                'batch_timeout must be positive.\nbatch_timeout'
            )
        self._scorer = scorer
        self._concurrency = concurrency
        self._batch_timeout = batch_timeout
        self._loop = asyncio.new_event_loop()
        self._sem: asyncio.Semaphore | None = None
        self._ready = threading.Event()
        self._init_error: Exception | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run_loop, name="RewardServiceLoop", daemon=True
        )
        self._thread.start()
        # Wait until the loop is running and the semaphore is created inside it.
        # On any init failure (timeout or _init_error), tear the thread + loop
        # down explicitly via close() so we don't leak a daemon thread and an
        # open event loop with its selector FDs. The early `_closed=True` +
        # re-raise pattern used previously skipped close()'s join (close() early-
        # exits when _closed is True), leaving the thread alive.
        
        
        # event-loop
        try:
            if not self._ready.wait(timeout=30):
                raise RuntimeError(
                    'RewardService background loop did not start within 30 s.'
                )
            if self._init_error is not None:
                raise self._init_error
        except BaseException:
            # close() flips _closed=True (if not already), stops the loop, and
            # joins the thread — preventing the leak. Re-raise the init error.
            
            self.close()
            raise

    def _run_loop(self) -> None:
        'Background-thread target: run the loop and create the semaphore in it.\n\n\n        Exceptions during semaphore creation are captured and signalled to\n        ``__init__`` via ``self._init_error`` (re-raised on the main thread).\n        The loop is always closed in a ``finally`` block to prevent selector/FD'
        try:
            asyncio.set_event_loop(self._loop)
            self._sem = asyncio.Semaphore(self._concurrency)
        except BaseException as e:
            self._init_error = e
            self._ready.set()
            return
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    async def _score_one(self, req: ScoreRequest) -> ScoredMetrics:
        sem = self._sem
        if sem is None:
            raise RuntimeError(
                'RewardService semaphore not initialized.'
            )
        async with sem:
            try:
                response = completion_to_text(req.completion)
                user_input = extract_user_question(
                    req.task_class, req.prompt, req.extra,
                )
                return await self._scorer.score(
                    user_input=user_input,
                    response=response,
                    reference=req.reference,
                    task_class=req.task_class,
                    extra=req.extra,
                    # The original message-list prompt: the judge-rubric
                    # criteria rebuild the input context of a reasoning turn
                    # from it (None-tolerant on the receiving side).
                    
                    
                    prompt=req.prompt,
                )
            except Exception as e:  # noqa: BLE001 - never abort the batch
                logger.warning(
                    "Scorer failed for task_class=%s source=%s: %s",
                    req.task_class,
                    req.source,
                    e,
                )
                return fallback_scored_metrics()

    async def _score_all(self, requests: list[ScoreRequest]) -> list[ScoredMetrics]:
        if not requests:
            return []
        return await asyncio.gather(*[self._score_one(r) for r in requests])

    def score_batch(self, requests: list[ScoreRequest]) -> list[ScoredMetrics]:
        'Score a batch synchronously (blocks until all results are ready).\n\n\n        Runs :meth:`_score_all` on the background loop via\n        :func:`asyncio.run_coroutine_threadsafe`. The generation loop can call\n        this between sync vLLM batches. On timeout the orphaned coroutine is\n        cancelled so it cannot keep holding the service semaphore or in-flight\n        HTTP requests\n        :func:`asyncio.run_coroutine_threadsafe`'
        if self._closed:
            raise RuntimeError(
                'RewardService is closed.'
            )
        future = asyncio.run_coroutine_threadsafe(
            self._score_all(list(requests)), loop=self._loop
        )
        try:
            return future.result(timeout=self._batch_timeout)
        except TimeoutError:
            # Cancel the orphaned coroutine so it does not keep running on the
            # background loop (holding the service semaphore / in-flight HTTP)
            # after we propagate the timeout to the caller.
            
            
            future.cancel()
            raise

    def close(self) -> None:
        'Stop the background loop and join the thread (idempotent).'
        if self._closed:
            return
        self._closed = True
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:
            # Loop already stopped
            pass
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            logger.warning('RewardService background thread did not terminate within 10 s after close(). It will be killed when the process exits.')

    def __enter__(self) -> "RewardService":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
