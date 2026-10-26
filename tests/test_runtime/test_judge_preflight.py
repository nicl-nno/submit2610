'Tests for runtime/judge_preflight.py — the pre-flight judge ping.\n\n\nThe real OpenAI client is replaced with a fake via\n``OpenAIEndpoint.async_client`` (the same pattern as\n``tests/test_reward/test_judge.py::TestFenceSanitizer``); the sanitizer\nwrapper is applied on top of the fake, so the ``reasoning_content``\nfallback is exercised end-to-end.\n``OpenAIEndpoint.async_client``\n``reasoning_content``'

from __future__ import annotations

import asyncio

import pytest

from llm_training.runtime import judge_preflight
from llm_training.runtime.endpoints import OpenAIEndpoint
from llm_training.runtime.judge_preflight import run_judge_preflight


class _Message:
    def __init__(self, content, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content


class _Choice:
    def __init__(self, message):
        self.message = message


class _Response:
    def __init__(self, message):
        self.choices = [_Choice(message)]


class _FakeCompletions:
    def __init__(self, behavior):
        self._behavior = behavior
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        behavior = self._behavior
        if isinstance(behavior, Exception):
            raise behavior
        if isinstance(behavior, float):  # hang for N seconds
            await asyncio.sleep(behavior)
            behavior = _Message("late")
        return behavior


class _FakeChat:
    def __init__(self, behavior):
        self.completions = _FakeCompletions(behavior)


class _FakeClient:
    def __init__(self, behavior):
        self.chat = _FakeChat(behavior)


def _install_fake(monkeypatch, behavior) -> _FakeClient:
    fake = _FakeClient(behavior)

    def _fake_async_client(self, *, max_retries=2, timeout=None):
        return fake

    monkeypatch.setattr(OpenAIEndpoint, "async_client", _fake_async_client)
    return fake


def _set_env(monkeypatch, *, base_url="http://judge:8001/v1", model="judge-m"):
    monkeypatch.setenv("JUDGE_BASE_URL", base_url)
    monkeypatch.setenv("JUDGE_MODEL", model)
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)


class TestRunJudgePreflight:
    def test_success_returns_reply(self, monkeypatch):
        # A non-empty content → the reply text is returned for the log line.
        
        _set_env(monkeypatch)
        fake = _install_fake(monkeypatch, _Response(_Message("Привет! Отлично.")))
        reply = run_judge_preflight(timeout_s=5.0)
        assert reply == "Привет! Отлично."
        assert fake.chat.completions.calls == 1

    def test_thinking_judge_reasoning_fallback(self, monkeypatch):
        # Empty content + non-empty reasoning_content → the sanitizer
        # promotes the reasoning text; the ping passes.
        
        
        _set_env(monkeypatch)
        _install_fake(
            monkeypatch, _Response(_Message(None, "Думаю… отвечаю: всё хорошо"))
        )
        reply = run_judge_preflight(timeout_s=5.0)
        assert reply == "Думаю… отвечаю: всё хорошо"

    def test_transport_error_exits_with_hint(self, monkeypatch):
        # Any exception from the endpoint → SystemExit naming the env vars
        # and the --no-judge-preflight escape hatch.
        
        
        _set_env(monkeypatch, base_url="https://api.z.ai/api/paas/v4/")
        _install_fake(monkeypatch, RuntimeError("401 unauthorized"))
        with pytest.raises(SystemExit) as exc_info:
            run_judge_preflight(timeout_s=5.0)
        message = str(exc_info.value)
        assert "JUDGE_BASE_URL" in message
        assert "https://api.z.ai/api/paas/v4/" in message
        assert "--no-judge-preflight" in message
        assert "401 unauthorized" in message

    def test_empty_reply_exits(self, monkeypatch):
        # Neither content nor reasoning_content → a hard failure.
        
        _set_env(monkeypatch)
        _install_fake(monkeypatch, _Response(_Message("   ")))
        with pytest.raises(SystemExit) as exc_info:
            run_judge_preflight(timeout_s=5.0)
        assert "EMPTY" in str(exc_info.value) or "ПУСТЫМ" in str(exc_info.value)

    def test_hung_server_cut_by_outer_timeout(self, monkeypatch):
        # A server that never answers is cut by asyncio.wait_for, not by the
        # wall clock: the margin constant is shrunk to keep the test fast.
        
        
        _set_env(monkeypatch)
        _install_fake(monkeypatch, 5.0)  # sleep 5 s
        monkeypatch.setattr(judge_preflight, "JUDGE_PING_MARGIN_S", 0.05)
        with pytest.raises(SystemExit) as exc_info:
            run_judge_preflight(timeout_s=0.05)
        assert "preflight" in str(exc_info.value).lower() or "проверка" in str(
            exc_info.value
        )

    def test_missing_env_warns_and_skips(self, monkeypatch, caplog):
        # No JUDGE_* env → a warning, None returned, no client call —
        # mirrors the anchor skip behaviour (training continues).
        
        
        monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
        monkeypatch.delenv("JUDGE_MODEL", raising=False)
        fake = _install_fake(monkeypatch, _Response(_Message("unused")))
        with caplog.at_level("WARNING"):
            assert run_judge_preflight(timeout_s=5.0) is None
        assert fake.chat.completions.calls == 0
        assert "JUDGE_BASE_URL" in caplog.text

    def test_missing_model_only_also_skips(self, monkeypatch):
        monkeypatch.setenv("JUDGE_BASE_URL", "http://judge:8001/v1")
        monkeypatch.delenv("JUDGE_MODEL", raising=False)
        fake = _install_fake(monkeypatch, _Response(_Message("unused")))
        assert run_judge_preflight(timeout_s=5.0) is None
        assert fake.chat.completions.calls == 0


class TestPingJudge:
    def test_api_root_validation_applies(self):
        # OpenAIEndpoint rejects a base_url with a /chat/completions tail —
        # the pre-flight inherits the check for free.
        
        
        import asyncio

        with pytest.raises(ValueError):
            asyncio.run(
                judge_preflight.ping_judge(
                    "https://api.z.ai/api/paas/v4/chat/completions",
                    "glm-5.3-flash",
                    timeout_s=5.0,
                )
            )
