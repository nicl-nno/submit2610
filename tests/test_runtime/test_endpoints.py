'Tests for runtime/endpoints.py — OpenAI-compatible endpoint descriptor.'

from __future__ import annotations

import pytest

from llm_training.runtime.endpoints import OpenAIEndpoint, mask_api_key


class TestMaskApiKey:
    def test_none_and_empty(self):
        assert mask_api_key(None) == "<none>"
        assert mask_api_key("") == "<none>"

    def test_short_key_fully_masked(self):
        assert mask_api_key("ab") == "***"
        assert mask_api_key("abc") == "***"
        assert mask_api_key("abcd") == "***"

    def test_long_key_shows_edges(self):
        # first + *** + last, middle hidden
        assert mask_api_key("sk-abcdef1234") == "s***4"

    def test_does_not_leak_full_key(self):
        key = "super-secret-key-1234567890"
        masked = mask_api_key(key)
        assert masked != key
        assert key[1:-1] not in masked


class TestOpenAIEndpointConstruction:
    def test_minimal_local_endpoint(self, monkeypatch):
        # Make sure no env key changes the local default for this test.
        
        monkeypatch.delenv("JUDGE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        ep = OpenAIEndpoint(base_url="http://localhost:8001/v1", model="my-judge")
        assert ep.base_url == "http://localhost:8001/v1"
        assert ep.model == "my-judge"
        # No key + no env => local-default sentinel.
        
        assert ep.api_key == "EMPTY"

    def test_explicit_key_preserved(self):
        ep = OpenAIEndpoint(
            base_url="https://vsellm.ru/v1", model="gpt-proxy", api_key="sk-real-12345"
        )
        assert ep.api_key == "sk-real-12345"

    def test_judge_env_var_picked_up(self, monkeypatch):
        monkeypatch.setenv("JUDGE_API_KEY", "sk-from-judge-env")
        # OPENAI_API_KEY also set but kind-specific wins.
        # OPENAI_API_KEY
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-openai-env")
        ep = OpenAIEndpoint(base_url="http://x/v1", model="m", kind="judge")
        assert ep.api_key == "sk-from-judge-env"

    def test_openai_api_key_fallback(self, monkeypatch):
        monkeypatch.delenv("JUDGE_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-generic")
        ep = OpenAIEndpoint(base_url="http://x/v1", model="m", kind="judge")
        assert ep.api_key == "sk-generic"

    def test_embedding_kind_uses_embedding_env(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_API_KEY", "sk-embed-env")
        ep = OpenAIEndpoint(base_url="http://x/v1", model="emb", kind="embedding")
        assert ep.api_key == "sk-embed-env"

    def test_empty_base_url_raises(self):
        with pytest.raises(ValueError):
            OpenAIEndpoint(base_url="", model="m")

    def test_method_url_base_raises(self):
        # base_url must be the API root; a URL already ending in
        # /chat/completions (the form cloud cURL docs show, e.g. Z.AI) sends
        # every request to a 404 because the SDK appends the path again.
        # base_url
        
        
        with pytest.raises(ValueError):
            OpenAIEndpoint(
                base_url="https://api.z.ai/api/paas/v4/chat/completions",
                model="glm-5.3-flash",
            )

    def test_method_url_base_with_trailing_slash_raises(self):
        with pytest.raises(ValueError):
            OpenAIEndpoint(
                base_url="https://api.z.ai/api/paas/v4/chat/completions/",
                model="glm-5.3-flash",
            )

    def test_zai_api_root_accepted(self):
        ep = OpenAIEndpoint(
            base_url="https://api.z.ai/api/paas/v4/",
            model="glm-5.3-flash",
            api_key="sk-zai-test-12345",
        )
        assert ep.base_url == "https://api.z.ai/api/paas/v4/"
        assert ep.model == "glm-5.3-flash"

    def test_empty_model_raises(self):
        with pytest.raises(ValueError):
            OpenAIEndpoint(base_url="http://x/v1", model="")

    def test_frozen(self):
        ep = OpenAIEndpoint(base_url="http://x/v1", model="m")
        with pytest.raises(Exception):
            ep.base_url = "http://y/v1"  # type: ignore[misc]


class TestDescribeAndMasking:
    def test_describe_masks_key(self):
        ep = OpenAIEndpoint(
            base_url="https://vsellm.ru/v1", model="m", api_key="sk-very-secret-12345"
        )
        d = ep.describe()
        assert d["base_url"] == "https://vsellm.ru/v1"
        assert d["model"] == "m"
        assert d["api_key"] == "s***5"
        # The full key must never appear in the describe dict or its repr.
        
        assert "sk-very-secret-12345" not in str(d)

    def test_masked_key_helper(self):
        ep = OpenAIEndpoint(base_url="http://x/v1", model="m", api_key="sk-abcdef1234")
        assert ep.masked_key() == "s***4"

    def test_describe_local_no_key(self, monkeypatch):
        monkeypatch.delenv("JUDGE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        ep = OpenAIEndpoint(base_url="http://localhost:8001/v1", model="m")
        d = ep.describe()
        assert d["api_key"] == mask_api_key("EMPTY")


class TestAsyncClient:
    def test_builds_client_without_calling_network(self, monkeypatch):
        # openai is a dependency of ragas (train extra). If absent, skip — the
        # factory's ImportError path is covered separately.
        # openai
        pytest.importorskip("openai")
        # Ensure no real env key interferes with the local default.
        
        monkeypatch.delenv("JUDGE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        ep = OpenAIEndpoint(base_url="http://localhost:9999/v1", model="m")
        client = ep.async_client()
        assert client is not None
        # The client should target our base_url
        assert "localhost:9999" in str(client.base_url)

    def test_max_retries_forwarded_to_client(self, monkeypatch):
        # max_retries is forwarded to AsyncOpenAI for HTTP-level transients.
        # max_retries
        pytest.importorskip("openai")
        monkeypatch.delenv("JUDGE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        ep = OpenAIEndpoint(base_url="http://localhost:9999/v1", model="m")
        client_default = ep.async_client()
        client_5 = ep.async_client(max_retries=5)
        # openai stores max_retries on the client instance.
        
        assert client_5.max_retries == 5
        assert client_default.max_retries == 2  # default of async_client()

    def test_timeout_forwarded_to_client(self, monkeypatch):
        # timeout is forwarded to AsyncOpenAI (was previously a no-op field).
        
        pytest.importorskip("openai")
        monkeypatch.delenv("JUDGE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        ep = OpenAIEndpoint(base_url="http://localhost:9999/v1", model="m")
        client = ep.async_client(timeout=300.0)
        # openai stores timeout on the client instance (float or httpx.Timeout).
        
        assert client.timeout == 300.0
