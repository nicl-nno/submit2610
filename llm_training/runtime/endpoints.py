'OpenAI-compatible endpoint descriptor (the judge LLM).\n\n\nAGENTS.md requires the judge LLM to be a **different model from the policy**,\naccessed over an OpenAI-compatible HTTP API. The backend is irrelevant to this\ncode: it may be a local ``vllm serve`` process the operator started, or a cloud\nprovider (e.g. ``https://vsellm.ru/v1``). All that matters here is\n``(base_url, api_key, model)``. (Embeddings are computed in-process via\n``sentence-transformers`` and are NOT described by this class.)\n\nAGENTS.md\n\nThe ``openai`` library is imported lazily so the rest of the package stays\nimportable in offline (dev) environments without the ``train`` extra.'

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

#: Default api key for local vLLM servers (which do not validate the key).

_LOCAL_DEFAULT_API_KEY = "EMPTY"

#: Sentinel used in logs / reprs in place of the real key.

_MASKED_KEY = "***"


def mask_api_key(api_key: str | None) -> str:
    'Return a log-safe representation of an API key.\n\n\n    ``None``/empty → ``"<none>"``; otherwise the first and last character with\n    ``***`` between (or just ``***`` for very short keys). Never raises.'
    if not api_key:
        return "<none>"
    if len(api_key) <= 4:
        return _MASKED_KEY
    return f"{api_key[0]}***{api_key[-1]}"


@dataclass(frozen=True)
class OpenAIEndpoint:
    'An OpenAI-compatible endpoint: ``base_url`` + optional ``api_key`` + model.\n\n\nAttributes\n        base_url: the ``/v1``-style root, e.g. ``http://localhost:8001/v1`` or\n            ``https://vsellm.ru/v1``\n        model: the model name to request on the endpoint (e.g. the served vLLM\n            model id or a cloud model id\n        api_key: optional API key. For a local vLLM server the key is not\n            validated and may be left as the default sentinel. Cloud providers\n            require a real key\n        kind: a short label for logging (typically ``"judge"``); purely'

    base_url: str
    model: str
    api_key: str | None = None
    kind: str = "judge"

    def __post_init__(self) -> None:
        if not self.base_url or not self.base_url.strip():
            raise ValueError(
                'OpenAIEndpoint.base_url must be non-empty.\nOpenAIEndpoint.base_url'
            )
        # base_url must be the API ROOT, not the full method URL: the OpenAI
        # SDK appends `chat/completions` itself. Cloud providers' cURL docs
        # (e.g. Z.AI) show the already-expanded path — passing it verbatim
        # makes every request hit a 404. Mirrors the established check in
        # scripts/calibrate_fluency.py.
        # base_url
        
        
        
        # scripts/calibrate_fluency.py.
        if self.base_url.rstrip("/").endswith("/chat/completions"):
            raise ValueError(
                f'OpenAIEndpoint.base_url must be the API root, not the method URL: got {self.base_url!r}'
            )
        if not self.model or not self.model.strip():
            raise ValueError(
                'OpenAIEndpoint.model must be non-empty.\nOpenAIEndpoint.model'
            )
        # Resolve a usable key: explicit > env > local-default sentinel.
        # This never raises on missing env
        resolved = self._resolve_key(self.api_key, self.kind)
        # frozen dataclass → use object.__setattr__ to fill the normalized key.
        object.__setattr__(self, "api_key", resolved)

    @staticmethod
    def _resolve_key(api_key: str | None, kind: str) -> str:
        if api_key:
            return api_key
        env_var = f"{kind.upper()}_API_KEY"
        env_val = os.environ.get(env_var)
        if env_val:
            return env_val
        env_val = os.environ.get("OPENAI_API_KEY")
        if env_val:
            return env_val
        return _LOCAL_DEFAULT_API_KEY

    def masked_key(self) -> str:
        'Log-safe view of the key (see :func:`mask_api_key`).'
        return mask_api_key(self.api_key)

    def describe(self) -> dict[str, str]:
        'Return a log-safe dict (key masked) describing this endpoint.'
        return {
            "base_url": self.base_url,
            "model": self.model,
            "api_key": self.masked_key(),
            "kind": self.kind,
        }

    def async_client(
        self, *, max_retries: int = 2, timeout: float | None = None,
    ) -> Any:
        'Build a fresh ``openai.AsyncOpenAI`` client bound to this endpoint.\n\n\n        Imports ``openai`` lazily; raises a helpful error if the package is not\n        installed (install with ``pip install -e ".[train]"``). ``max_retries``\n        is forwarded to the openai client which retries HTTP-level transients\n        (429/5xx) on its own; structured-output / proxy-level errors are retried\n        separately in :mod:`llm_training.reward.judge`.\n\nArgs\n            max_retries: forwarded to :class:`AsyncOpenAI` for HTTP-level\n                retries (429/5xx, connection errors\n                :class:`AsyncOpenAI`\n            timeout: per-request timeout in seconds. When ``None`` (default),\n                the openai library\'s own default (~600 s) is used. Typically\n                wired from :attr:`RewardConfig.judge_timeout_s`\n\n        The ``http_client`` parameter is intentionally **not** passed to\n        :class:`AsyncOpenAI`. When a custom ``httpx.AsyncClient`` is created\n        outside the event loop (e.g., in the main thread during\n        :func:`_build_metrics_from_ragas`) and then used inside a different\n        event loop (the :class:`RewardService` background loop), httpx\'s\n        internal anyio/httpcore primitives bind to the wrong loop →\n        ``RuntimeError: Event loop is closed``. Letting ``AsyncOpenAI`` create\n        its own ``httpx.AsyncClient`` internally avoids this: the openai\n        library creates the client lazily on first request, within the correct\n        event loop.\n        ``http_client``\n        background loop ``RewardService``\n        is closed``.'
        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required to use an OpenAI-compatible "
                "endpoint. Install the train extra: pip install -e '.[train]'."
                "\nДля OpenAI-совместимого эндпоинта нужен пакет 'openai': "
                "pip install -e '.[train]'."
            ) from e
        kwargs: dict[str, Any] = dict(
            base_url=self.base_url,
            api_key=self.api_key or _LOCAL_DEFAULT_API_KEY,
            max_retries=max_retries,
        )
        if timeout is not None:
            kwargs["timeout"] = timeout
        return AsyncOpenAI(**kwargs)
