'Pre-flight judge availability check (a one-message ping).\n\n\nBefore a long GRPO run starts, ``grpo_train.py`` sends the judge a trivial\nwithin a bounded time. A broken judge (wrong ``JUDGE_BASE_URL`` root, missing\nAPI key, 401/404, a hung server) would otherwise stay invisible until the\nFIRST anchor — hours of setup wasted with checkpoint selection silently\ndead. The check reuses :class:`llm_training.runtime.endpoints.OpenAIEndpoint`\n(so the API-root validation applies) and the judge reply sanitizer\n(:func:`llm_training.reward.judge.sanitize_judge_client_json_fences`, whose\n``reasoning_content`` fallback serves thinking judges for free). A failure\nraises :exc:`SystemExit` with a bilingual message naming the ``JUDGE_*`` env\nvars and the ``--no-judge-preflight`` escape hatch; missing env vars are a\nsoft skip (a warning — mirrors the anchor behaviour: training continues,\nanchors get skipped).'

from __future__ import annotations

import asyncio
import logging
import os

from llm_training.runtime.endpoints import OpenAIEndpoint

logger = logging.getLogger(__name__)

# : The ping message sent to the judge.
JUDGE_PING_MESSAGE = "Привет! Как дела?"

#: Default token cap for the ping: small on purpose (the check only needs
#: ANY reply; thinking judges may spend the whole cap on reasoning — the
#: sanitizer's reasoning_content fallback then supplies the non-empty text).


# : reasoning_content
JUDGE_PING_MAX_TOKENS = 2048

#: Wall-clock margin added on top of the doubled request timeout (connect +
#: one HTTP retry overhead); the outer wait_for cuts a hung server here.


JUDGE_PING_MARGIN_S = 30.0


def preflight_judge_env() -> tuple[str | None, str | None, str | None]:
    'Read the judge env triple (base_url, model, api_key).'
    return (
        os.environ.get("JUDGE_BASE_URL"),
        os.environ.get("JUDGE_MODEL"),
        os.environ.get("JUDGE_API_KEY"),
    )


async def ping_judge(
    base_url: str,
    model: str,
    api_key: str | None = None,
    *,
    timeout_s: float,
    max_tokens: int = JUDGE_PING_MAX_TOKENS,
    ping_message: str = JUDGE_PING_MESSAGE,
) -> str:
    'Send a one-message ping and return the (sanitized) reply text.\n\n\n    Raises on any failure: a transport/API error, an outer timeout (the\n    request must not hang — ``asyncio.wait_for`` bounds even a retrying\n    client), or an empty reply (no content and no reasoning_content).'
    from llm_training.reward.judge import sanitize_judge_client_json_fences

    endpoint = OpenAIEndpoint(base_url=base_url, model=model, api_key=api_key)
    client = endpoint.async_client(max_retries=1, timeout=timeout_s)
    client = sanitize_judge_client_json_fences(client)
    # Outer bound: one request + one HTTP retry + connect overhead must fit;
    # a hung server (or a retry loop) is cut here, not by the wall clock.
    
    
    budget = timeout_s * 2 + JUDGE_PING_MARGIN_S
    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": ping_message}],
            max_tokens=max_tokens,
        ),
        timeout=budget,
    )
    reply = ""
    choices = getattr(response, "choices", None) or []
    if choices:
        message = getattr(choices[0], "message", None)
        reply = (getattr(message, "content", None) or "").strip()
    if not reply:
        raise RuntimeError(
            'The judge replied with an EMPTY message (no content and no reasoning_content).'
        )
    return reply


def run_judge_preflight(
    timeout_s: float,
    *,
    max_tokens: int = JUDGE_PING_MAX_TOKENS,
    ping_message: str = JUDGE_PING_MESSAGE,
) -> str | None:
    "Synchronous pre-flight: ping the judge or exit with a clear error.\n\n\nBehaviour\n    - ``JUDGE_BASE_URL``/``JUDGE_MODEL`` absent → a bilingual warning,\n      ``None`` returned (mirrors the anchors: training continues, anchors\nare skipped).\nSuccess → the reply text (for the caller's log line).\n    - Any failure → :exc:`SystemExit` with a bilingual message naming the\n      env vars, the cause and the ``--no-judge-preflight`` escape hatch."
    base_url, model, api_key = preflight_judge_env()
    if not base_url or not model:
        logger.warning('Judge pre-flight skipped: JUDGE_BASE_URL / JUDGE_MODEL are not set — the judge anchors will be skipped too.\nJUDGE_MODEL')
        return None
    try:
        reply = asyncio.run(
            ping_judge(
                base_url, model, api_key,
                timeout_s=timeout_s, max_tokens=max_tokens,
                ping_message=ping_message,
            )
        )
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - report ANY failure uniformly
        raise SystemExit(
            f'Judge pre-flight FAILED: {e!r}\nCheck JUDGE_BASE_URL={base_url!r}, JUDGE_MODEL={model!r}'
        ) from e
    return reply
