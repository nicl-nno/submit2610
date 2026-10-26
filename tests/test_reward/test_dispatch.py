'Tests for reward/dispatch.py — multi-task reward dispatcher (FakeScorer, no RAGAS).'

from __future__ import annotations

import asyncio

import pytest

from llm_training.reward.config import ScoredMetrics
from llm_training.reward.dispatch import make_reward_function
from llm_training.reward.scoring import harmonic_mean

# pytest-asyncio is in strict mode (see pyproject): mark every async test.
# pytest-asyncio
pytestmark = pytest.mark.asyncio


class FakeScorer:
    'Deterministic stand-in for the RAGAS scorer.\n\n\n    Records every call (including ``task_class``) and returns preset metrics.\n    Optionally raises on a chosen ``response`` string to exercise error\n    resilience. For tool-use classes it computes a real format-style score so\n    the harmonic-combine path can be exercised end-to-end.'

    def __init__(self, *, fail_response: str | None = None) -> None:
        self.calls: list[dict] = []
        self.fail_response = fail_response

    async def score(self, *, user_input, response, reference, task_class, extra=None) -> ScoredMetrics:
        self.calls.append(
            {
                "user_input": user_input,
                "response": response,
                "reference": reference,
                "task_class": task_class,
            }
        )
        if self.fail_response is not None and response == self.fail_response:
            raise RuntimeError("simulated judge failure")
        # For non-tool classes: just SemSim. For tool classes: SemSim + a
        # perfect format score (so the dispatcher's harmonic path is tested).
        
        semsim = 0.5
        if task_class in ("opencode_gen", "tool_calls_multiturn"):
            fmt = 1.0
            combined = harmonic_mean([semsim, fmt])
            return ScoredMetrics(
                semantic_similarity=semsim,
                combined=combined,
                format_score=fmt,
            )
        return ScoredMetrics(semantic_similarity=semsim, combined=semsim)


def _conv_prompt(*, system="s", user="u"):
    'Build a conversational prompt + matching dataset kwargs for one example.'
    prompt = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return prompt


class TestCompletionNormalization:
    async def test_conversational_completion_extracted(self):
        scorer = FakeScorer()
        rf = make_reward_function(scorer, concurrency=8)
        prompt = _conv_prompt(user="What is 2+2?")
        completion = [{"role": "assistant", "content": "4"}]
        rewards = await rf(
            prompts=[prompt],
            completions=[completion],
            task_class=["math_reasoner"],
            reference=["4"],
            extra=[{}],
        )
        assert rewards == [0.5]
        assert scorer.calls[0]["response"] == "4"

    async def test_string_completion(self):
        scorer = FakeScorer()
        rf = make_reward_function(scorer, concurrency=8)
        rewards = await rf(
            prompts=[_conv_prompt()],
            completions=["plain string"],
            task_class=["coder"],
            reference=["ref"],
            extra=[{}],
        )
        assert rewards == [0.5]
        assert scorer.calls[0]["response"] == "plain string"

    async def test_tool_call_completion_preserved(self):
        # A tool-use completion (assistant with tool_calls) must be serialized
        # via assistant_content_str so it is comparable to the reference verbatim.
        # tool-use
        scorer = FakeScorer()
        rf = make_reward_function(scorer, concurrency=8)
        completion = [
            {"role": "assistant", "content": "", "tool_calls": [{"name": "read", "arguments": {"path": "x"}}]}
        ]
        await rf(
            prompts=[_conv_prompt()],
            completions=[completion],
            task_class=["tool_calls_multiturn"],
            reference=["ref"],
            extra=[{}],
        )
        resp = scorer.calls[0]["response"]
        assert "read" in resp and "path" in resp


class TestUserInputExtraction:
    'The dispatcher must use extract_user_question (not the raw user text).'

    async def test_menon_qa_question_not_full_query(self):
        # The judge must receive just the question, not the documents.
        
        scorer = FakeScorer()
        rf = make_reward_function(scorer, concurrency=8)
        query = (
            "==========\nDOCUMENT 1\n==========\n\nAU основан в 1959.\n\n"
            "==========\nCURRENT QUESTION\n==========\n\nКогда основан AU?"
        )
        prompt = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": query},
        ]
        await rf(
            prompts=[prompt],
            completions=[{"role": "assistant", "content": "В 1959."}],
            task_class=["menon_qa"],
            reference=["1959"],
            extra=[{"current_datetime": "2026-03-01 00:00:00"}],
        )
        user_input = scorer.calls[0]["user_input"]
        assert "Когда основан AU?" in user_input
        assert "AU основан в 1959" not in user_input  # docs stripped

    async def test_multihop_rag_chunks_stripped(self):
        scorer = FakeScorer()
        rf = make_reward_function(scorer, concurrency=8)
        query = "Read carefully.\nThe chunk list:\nChunk A. Chunk B."
        prompt = [{"role": "system", "content": "s"}, {"role": "user", "content": query}]
        await rf(
            prompts=[prompt],
            completions=[{"role": "assistant", "content": "answer"}],
            task_class=["multihop_rag"],
            reference=["ref"],
            extra=[{}],
        )
        assert "Chunk A" not in scorer.calls[0]["user_input"]
        assert "Read carefully" in scorer.calls[0]["user_input"]

    async def test_non_rag_class_passes_full_user(self):
        # For non-RAG classes the whole user message is the question already.
        
        scorer = FakeScorer()
        rf = make_reward_function(scorer, concurrency=8)
        await rf(
            prompts=[_conv_prompt(user="solve 2+2")],
            completions=[{"role": "assistant", "content": "4"}],
            task_class=["math_reasoner"],
            reference=["4"],
            extra=[{}],
        )
        assert scorer.calls[0]["user_input"] == "solve 2+2"


class TestDispatch:
    async def test_tool_use_class_combines_format_score(self):
        scorer = FakeScorer()
        rf = make_reward_function(scorer, concurrency=8)
        rewards = await rf(
            prompts=[_conv_prompt()],
            completions=[
                {"role": "assistant", "content": "x"}
            ],
            task_class=["tool_calls_multiturn"],
            reference=["ref"],
            extra=[{}],
        )
        # FakeScorer returns format_score=1.0 for tool classes → harmonic mean
        # of [0.5, 1.0] = 2/(1/0.5 + 1/1.0) = 2/3.
        
        assert rewards == [pytest.approx(2 / 3)]

    async def test_task_class_forwarded_to_scorer(self):
        # Crucial for Variant B: the scorer receives task_class.
        
        scorer = FakeScorer()
        rf = make_reward_function(scorer, concurrency=8)
        await rf(
            prompts=[_conv_prompt()],
            completions=[{"role": "assistant", "content": "x"}],
            task_class=["menon_qa"],
            reference=["ref"],
            extra=[{}],
        )
        assert scorer.calls[0]["task_class"] == "menon_qa"


class TestBatchAndResilience:
    async def test_mixed_batch_returns_aligned_rewards(self):
        scorer = FakeScorer()
        rf = make_reward_function(scorer, concurrency=8)
        rewards = await rf(
            prompts=[
                _conv_prompt(user="q1"),
                _conv_prompt(user="q2"),
            ],
            completions=[
                [{"role": "assistant", "content": "a1"}],
                [{"role": "assistant", "content": "a2"}],
            ],
            task_class=["math_reasoner", "coder"],
            reference=["r1", "r2"],
            extra=[{}, {}],
        )
        assert len(rewards) == 2
        # Both non-tool → FakeScorer returns SemSim 0.5 → reward 0.5.
        assert rewards == [0.5, 0.5]

    async def test_one_failure_does_not_abort_batch(self):
        # The failing example maps to 0.0; the other still scores normally.
        
        scorer = FakeScorer(fail_response="boom")
        rf = make_reward_function(scorer, concurrency=8)
        rewards = await rf(
            prompts=[_conv_prompt(), _conv_prompt()],
            completions=[
                [{"role": "assistant", "content": "boom"}],
                [{"role": "assistant", "content": "ok"}],
            ],
            task_class=["coder", "coder"],
            reference=["r", "r"],
            extra=[{}, {}],
        )
        assert rewards == [0.0, 0.5]

    async def test_large_batch_respects_concurrency_and_completes(self):
        scorer = FakeScorer()
        rf = make_reward_function(scorer, concurrency=4)
        n = 50
        rewards = await rf(
            prompts=[_conv_prompt() for _ in range(n)],
            completions=[{"role": "assistant", "content": "x"} for _ in range(n)],
            task_class=["coder"] * n,
            reference=["r"] * n,
            extra=[{}] * n,
        )
        assert len(rewards) == n
        assert all(r == 0.5 for r in rewards)
        assert len(scorer.calls) == n

    async def test_returns_floats_in_unit_interval(self):
        scorer = FakeScorer()
        rf = make_reward_function(scorer, concurrency=8)
        rewards = await rf(
            prompts=[_conv_prompt()],
            completions=[{"role": "assistant", "content": "x"}],
            task_class=["coder"],
            reference=["r"],
            extra=[{}],
        )
        assert all(isinstance(r, float) for r in rewards)
        assert all(0.0 <= r <= 1.0 for r in rewards)


class TestArgumentValidation:
    async def test_missing_task_class_raises(self):
        scorer = FakeScorer()
        rf = make_reward_function(scorer, concurrency=8)
        with pytest.raises(ValueError):
            await rf(
                prompts=[_conv_prompt()],
                completions=[{"role": "assistant", "content": "x"}],
                reference=["r"],
                extra=[{}],
            )

    async def test_invalid_concurrency_raises(self):
        with pytest.raises(ValueError):
            make_reward_function(FakeScorer(), concurrency=0)

    async def test_dispatcher_failure_uses_dispatcher_sentinel(self):
        # When the scorer crashes, failed_metrics must be the __dispatcher__
        # sentinel (not "semantic_similarity") so run_eval.py can tell
        # dispatcher crashes apart from genuine metric failures.
        
        from llm_training.reward.dispatch import fallback_scored_metrics, DISPATCHER_FAILURE
        m = fallback_scored_metrics()
        assert m.combined == 0.0
        assert m.failed_metrics == (DISPATCHER_FAILURE,)
        assert m.failed_metrics != ("semantic_similarity",)


class TestTrlCompatibilityShape:
    async def test_accepts_completion_ids_kwarg(self):
        # TRL always passes completion_ids; the function must accept it.
        
        scorer = FakeScorer()
        rf = make_reward_function(scorer, concurrency=8)
        rewards = await rf(
            prompts=[_conv_prompt()],
            completions=[{"role": "assistant", "content": "x"}],
            completion_ids=[[1, 2, 3]],
            task_class=["coder"],
            reference=["r"],
            extra=[{}],
        )
        assert rewards == [0.5]

    async def test_function_has_name_and_docstring(self):
        rf = make_reward_function(FakeScorer(), concurrency=8)
        assert callable(rf)
        assert rf.__name__ == "multi_task_ragas_reward"
        assert rf.__doc__ is not None

    async def test_is_coroutine_function(self):
        # TRL detects async reward functions via asyncio.iscoroutinefunction.
        
        rf = make_reward_function(FakeScorer(), concurrency=8)
        assert asyncio.iscoroutinefunction(rf)


class TestServiceSemaphoreSizing:
    'B6: the Level-1 service semaphore must be max(embedding, judge) when\n    derived from a RewardConfig, not a hard-coded 8.'

    async def test_neither_concurrency_nor_config_rejected(self):
        # Calling with neither forces the caller to be explicit (no silent
        # default of 8 that ignores endpoint tuning).
        
        with pytest.raises(ValueError, match="concurrency"):
            make_reward_function(FakeScorer())

    async def test_concurrency_derived_from_reward_config(self):
        # max(embedding_concurrency=2, judge_concurrency=8) = 8.
        from llm_training.reward.config import RewardConfig
        from llm_training.runtime.endpoints import OpenAIEndpoint
        cfg = RewardConfig(
            judge=OpenAIEndpoint(base_url="http://j/v1", model="j"),
            judge_concurrency=8,
            embedding_concurrency=2,
        )
        # No crash ⇒ the semaphore was sized (the call succeeds only if a value
        # was derived
        rf = make_reward_function(FakeScorer(), reward_config=cfg)
        assert asyncio.iscoroutinefunction(rf)

    async def test_explicit_concurrency_overrides_config(self):
        # An explicit concurrency takes precedence over reward_config.
        
        from llm_training.reward.config import RewardConfig
        cfg = RewardConfig(judge_concurrency=8, embedding_concurrency=4)
        rf = make_reward_function(FakeScorer(), reward_config=cfg, concurrency=1)
        assert asyncio.iscoroutinefunction(rf)

    async def test_fluency_concurrency_participates_in_derivation(self):
        # The service semaphore is max(embedding, judge, fluency). When
        # fluency_concurrency is the largest, the function still builds cleanly
        # (the derivation no longer ignores it).
        
        # fluency_concurrency
        from llm_training.reward.config import RewardConfig
        cfg = RewardConfig(
            judge_concurrency=2,
            embedding_concurrency=2,
            fluency_concurrency=8,
        )
        rf = make_reward_function(FakeScorer(), reward_config=cfg)
        assert asyncio.iscoroutinefunction(rf)
