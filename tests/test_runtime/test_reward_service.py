'Tests for runtime/reward_service.py — batched scoring (FakeScorer, no network).'

from __future__ import annotations

import pytest

from llm_training.reward.config import ScoredMetrics
from llm_training.runtime.reward_service import RewardService, ScoreRequest


class _FakeScorer:
    'Records calls; returns SemSim-only for non-tool, SemSim+format for tool.'

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def score(self, *, user_input, response, reference, task_class, extra=None, prompt=None) -> ScoredMetrics:
        self.calls.append(
            {
                "user_input": user_input,
                "response": response,
                "reference": reference,
                "task_class": task_class,
            }
        )
        if task_class in ("opencode_gen", "tool_calls_multiturn"):
            return ScoredMetrics(
                semantic_similarity=0.5,
                combined=0.4,  # arbitrary lower value to test the format path
                format_score=0.8,
            )
        return ScoredMetrics(semantic_similarity=0.5, combined=0.5)


def _req(task_class="coder", user="q", completion="a", reference="r", extra=None):
    return ScoreRequest(
        prompt=[{"role": "system", "content": "s"}, {"role": "user", "content": user}],
        completion=completion,
        task_class=task_class,
        reference=reference,
        extra=extra or {},
        source=f"{task_class}#0",
    )


class TestScoreBatch:
    def test_empty_batch_returns_empty(self):
        with RewardService(_FakeScorer()) as svc:
            assert svc.score_batch([]) == []

    def test_single_plain_example(self):
        scorer = _FakeScorer()
        with RewardService(scorer) as svc:
            results = svc.score_batch([_req()])
        assert len(results) == 1
        assert results[0].semantic_similarity == 0.5
        assert results[0].combined == 0.5
        assert results[0].answer_correctness is None
        assert results[0].format_score is None

    def test_tool_use_example_returns_format_score(self):
        scorer = _FakeScorer()
        with RewardService(scorer) as svc:
            results = svc.score_batch(
                [_req(task_class="tool_calls_multiturn")]
            )
        assert results[0].format_score == 0.8
        assert results[0].combined == 0.4

    def test_mixed_batch_aligned(self):
        scorer = _FakeScorer()
        with RewardService(scorer) as svc:
            results = svc.score_batch(
                [
                    _req(task_class="coder"),
                    _req(task_class="tool_calls_multiturn"),
                    _req(task_class="math_reasoner"),
                ]
            )
        assert [r.combined for r in results] == [0.5, 0.4, 0.5]

    def test_completion_messages_normalized(self):
        # A conversational completion (list of messages) must be reduced to text
        # before being sent to the scorer.
        
        scorer = _FakeScorer()
        with RewardService(scorer) as svc:
            svc.score_batch(
                [_req(completion=[{"role": "assistant", "content": "Hello!"}])]
            )
        assert scorer.calls[0]["response"] == "Hello!"

    def test_user_input_extracted_from_prompt(self):
        scorer = _FakeScorer()
        with RewardService(scorer) as svc:
            svc.score_batch([_req(user="What is the speed of light?")])
        assert scorer.calls[0]["user_input"] == "What is the speed of light?"

    def test_menon_qa_question_not_full_query(self):
        # The user_input field for the judge must be the question only, not the
        # whole menon_qa query (documents included).
        
        scorer = _FakeScorer()
        query = (
            "==========\nDOCUMENT 1\n==========\n\nКонтекст.\n\n"
            "==========\nCURRENT QUESTION\n==========\n\nВопрос?"
        )
        with RewardService(scorer) as svc:
            svc.score_batch([_req(task_class="menon_qa", user=query)])
        assert "Контекст" not in scorer.calls[0]["user_input"]
        assert "Вопрос?" in scorer.calls[0]["user_input"]


class TestConcurrencyAndResilience:
    def test_large_batch_completes(self):
        scorer = _FakeScorer()
        with RewardService(scorer, concurrency=4) as svc:
            results = svc.score_batch([_req() for _ in range(40)])
        assert len(results) == 40
        assert all(r.combined == 0.5 for r in results)
        assert len(scorer.calls) == 40

    def test_one_failure_maps_to_zero_others_unaffected(self):
        class Flaky:
            def __init__(self):
                self.n = 0

            async def score(self, *, user_input, response, reference, task_class, extra=None, prompt=None):
                self.n += 1
                if response == "boom":
                    raise RuntimeError("judge down")
                return ScoredMetrics(semantic_similarity=0.5, combined=0.5)

        with RewardService(Flaky()) as svc:
            results = svc.score_batch(
                [_req(completion="ok"), _req(completion="boom"), _req(completion="ok")]
            )
        assert [r.combined for r in results] == [0.5, 0.0, 0.5]
        # The failed example's ScoredMetrics must record a dispatcher failure
        # (the scorer crashed before/at SemSim, so the dispatcher sentinel is
        # used — distinct from a genuine metric failure).
        
        assert results[1].failed_metrics == ("__dispatcher__",)
        # Successful examples have no failures
        assert results[0].failed_metrics == ()
        assert results[2].failed_metrics == ()


class TestLifecycle:
    def test_context_manager_closes(self):
        svc = RewardService(_FakeScorer())
        svc.__enter__()
        assert svc.score_batch([_req()])  # works while open
        svc.__exit__(None, None, None)
        with pytest.raises(RuntimeError):
            svc.score_batch([_req()])  # closed

    def test_close_is_idempotent(self):
        svc = RewardService(_FakeScorer())
        svc.close()
        svc.close()  # must not raise

    def test_concurrency_validation(self):
        with pytest.raises(ValueError):
            RewardService(_FakeScorer(), concurrency=0)

    def test_reuse_across_batches(self):
        # The same service must handle many sequential batches (the background
        # loop persists, AsyncOpenAI client reuse is safe).
        
        
        scorer = _FakeScorer()
        with RewardService(scorer) as svc:
            for _ in range(5):
                r = svc.score_batch([_req(), _req()])
                assert len(r) == 2
        assert len(scorer.calls) == 10

    def test_all_batches_share_one_event_loop(self):
        # The anchor ctrl-points in grpo_train.py reuse ONE RewardService for the
        # whole run; the judge's AsyncOpenAI httpx client binds to the first
        # loop it serves, so every batch must run on the SAME loop. (Building
        # a fresh service per anchor re-bound the client to a dead loop and
        # silently zeroed every AC from the second anchor on.)
        
        
        
        import asyncio

        class _LoopRecordingScorer:
            def __init__(self):
                self.loops: list = []

            async def score(self, *, user_input, response, reference, task_class, extra=None, prompt=None):
                self.loops.append(asyncio.get_running_loop())
                return ScoredMetrics(semantic_similarity=0.5, combined=0.5)

        scorer = _LoopRecordingScorer()
        with RewardService(scorer) as svc:
            svc.score_batch([_req()])
            svc.score_batch([_req(), _req()])
            svc.score_batch([_req()])
        assert len(scorer.loops) == 4
        assert all(loop is scorer.loops[0] for loop in scorer.loops)

    def test_init_failure_joins_thread(self, monkeypatch):
        # B2: when __init__ raises on init timeout, close() must still be
        # invoked so the daemon thread + open event loop are torn down (no
        # leak). We simulate the timeout ONLY for RewardService's 30-second
        # readiness wait. A global Event.wait -> False stub also intercepts
        # Thread.start()'s internal ``self._started.wait()`` and lets start()
        # return before the thread is marked started, making close().join()
        # race with bootstrap ("cannot join thread before it is started").
        # Previously __init__ set _closed=True and re-raised WITHOUT joining,
        # leaving the thread + loop leaked.
        
        
        
        import llm_training.runtime.reward_service as rs_mod

        captured_threads: list = []
        real_wait = rs_mod.threading.Event.wait
        real_close = rs_mod.RewardService.close

        def _fast_wait(self, timeout=None):
            # Pretend only RewardService._ready.wait(timeout=30) timed out.
            # Thread.start() calls Event.wait(timeout=None) on its private
            # _started event; delegate that call to preserve Thread's contract.
            
            # _started Event
            if timeout == 30:
                return False
            return real_wait(self, timeout)

        def _capturing_close(self):
            # __init__ never returns on failure, so capture the thread from the
            # close() call itself (an append after the constructor is
            # unreachable).
            
            captured_threads.append(self._thread)
            real_close(self)

        monkeypatch.setattr(rs_mod.threading.Event, "wait", _fast_wait)
        monkeypatch.setattr(rs_mod.RewardService, "close", _capturing_close)

        with pytest.raises(RuntimeError, match="did not start"):
            rs_mod.RewardService(_FakeScorer())
        # The thread reference leaked from the failed constructor must have been
        # joined by close() (within its 10 s join timeout) — i.e. it is no
        # longer alive once __init__ has re-raised.
        
        
        assert len(captured_threads) == 1
        for t in captured_threads:
            assert not t.is_alive()

    def test_score_batch_timeout_cancels_orphan(self):
        # B3: batch_timeout is configurable; a fast batch completes within it.
        # (Verifying the orphan-coroutine cancellation directly requires
        # instrumenting the background loop; here we at least pin that
        # batch_timeout is honoured and the service keeps working.)
        # B3: batch_timeout
        scorer = _FakeScorer()
        with RewardService(scorer, batch_timeout=5.0) as svc:
            r = svc.score_batch([_req()])
            assert len(r) == 1

    def test_batch_timeout_validation(self):
        with pytest.raises(ValueError):
            RewardService(_FakeScorer(), batch_timeout=0)
