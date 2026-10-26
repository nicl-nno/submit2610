'Loader: tool-calls multiturn → ``tool_calls_multiturn`` (Qwen function-call).\n\n\nThe source ``tool_calls_multiturn.jsonl`` has flat records\n``{system, query, response, history}``. The ``<tools>`` list embedded in\n``system`` uses **Python ``repr``** (single quotes, ``True``/``False``/``None``),\nparsed via :func:`ast.literal_eval`. The ``<tool_call>`` blocks in\n``response``/``history`` arrive in **two formats** from the upstream\n``interstellarninja/tool-calls-multiturn`` source: pure Python ``repr`` (~70 %\nof records) and a **hybrid** form (~30 %) — the block is padded with literal\n``\\n`` (backslash + ``n``, two chars) and mixes a JSON-quoted outer dict with\nsingle-quoted Python strings inside (e.g.\n``\\n{"arguments": {"q": [\'a\', \'b\']}, "name": "Foo"}\\n``). The loader\'s\n:func:`_parse_call_payload` is tolerant to both: it tries\n:func:`ast.literal_eval`, then strips the literal ``\\n`` padding and retries,\nthen falls back to :func:`json.loads`, before giving up and warning. The\nresult is normalized to the Qwen2.5/Meno-Lite ``{tools, messages}`` shape\n(assistant ``tool_calls``, tool results as ``role:"tool"`` which the template\nrenders as ``<tool_response>``).\nMulti-turn is encoded across the fields: ``history`` holds prior\n``[user, assistant]`` pairs; ``query`` may itself be one or more\n``<tool_response>`` turns (then it is a "final answer" record); ``response`` is\neither a ``<tool_call>`` block ("call" record) or the final text answer.'

from __future__ import annotations

import ast
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Iterator

from llm_training.config import DataConfig
from llm_training.loaders.base import read_jsonl
from llm_training.schema import Message, UnifiedExample

logger = logging.getLogger(__name__)
from llm_training.tokenizer_utils import assistant_content_str

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer

CLASSES = ("tool_calls_multiturn",)
FILENAME = "tool_calls_multiturn.jsonl"

_TOOLS_SECTION_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_RESPONSE_BLOCK_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)


def _parse_tools(system_text: str) -> tuple[list[dict[str, Any]] | None, str]:
    'Extract the real ``<tools>`` repr list from ``system_text``.\n\n\n    The system preamble *mentions* ``<tools> </tools>`` in prose, so we cannot\n    just take the first ``<tools>…</tools>`` match — we pick the one whose\n    content actually parses as a Python list (the real tool schemas).'
    for m in _TOOLS_SECTION_RE.finditer(system_text):
        candidate = m.group(1).strip()
        if not candidate.startswith("["):
            continue
        try:
            tools = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
        if isinstance(tools, list):
            # Drop the source's manual <tools> block AND its hand-written
            # postamble; keep only the preamble before this block. The standard
            # schema is re-baked by bake_tools_into_system at render time.
            
            preamble = system_text[: m.start()].strip()
            return tools, preamble
    return None, system_text.strip()


def _parse_call_payload(raw: str) -> dict[str, Any] | None:
    'Parse one ``<tool_call>`` payload, tolerant to two upstream formats.\n\n\n    The upstream ``interstellarninja/tool-calls-multiturn`` ships ``<tool_call>``\n    blocks in two forms: (1) pure Python ``repr`` (single quotes) and (2) a\n    hybrid — the block is padded with literal ``\\n`` (backslash + ``n``) and\n    mixes a JSON-quoted outer dict with single-quoted Python strings inside.\n    A bare :func:`ast.literal_eval` rejects form (2) on the leading backslash\n    ("unexpected character after line continuation character"). We therefore\n    try ``ast.literal_eval`` first, then the same with the literal ``\\n``\n    padding stripped, then :func:`json.loads` as a final fallback. Returns the\n    parsed dict, or ``None`` when nothing parses.'
    candidates: list[str] = [raw]
    stripped = raw.replace("\\n", " ").strip()
    if stripped != raw:
        candidates.append(stripped)
    for candidate in candidates:
        for fn in (ast.literal_eval, json.loads):
            try:
                d = fn(candidate)
            except (SyntaxError, ValueError):
                continue
            if isinstance(d, dict):
                return d
    return None


def _parse_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    'Parse ``<tool_call>{payload}</tool_call>`` blocks into normalized tool calls.\n\n\n    Returns ``(calls, remaining_text)`` where each call is\n    ``{"name": ..., "arguments": {...}}`` (Qwen form) and ``remaining_text`` is\n    the text outside any ``<tool_call>`` blocks. An unparseable block is logged\n    and skipped (not silently dropped).\n\n    A second upstream quirk is normalized here: ~793 hybrid blocks store the\n    tool ``name`` **nested inside** ``arguments`` (e.g.\n    ``{"arguments": {"queries": [...], "name": "ExpertQAExtractor"}}``) instead\n    of as a top-level sibling. When the parsed dict has no top-level ``name``\n    but its ``arguments`` dict does, that ``name`` is hoisted to the top level\n    (and removed from ``arguments``), restoring the standard Qwen form. This\n    only fires when top-level ``name`` is absent, so a legitimate function\n    parameter coincidentally called ``name`` is never touched.'
    calls: list[dict[str, Any]] = []
    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        raw = m.group(1).strip()
        d = _parse_call_payload(raw)
        if d is None:
            logger.warning(
                "tool_calls_multiturn: skipping unparseable <tool_call> block: %s",
                raw[:120],
            )
            continue
        args = d.get("arguments", {})
        name = d.get("name")
        if name is None and isinstance(args, dict) and "name" in args:
            # Hoist the misplaced tool name out of `arguments` (upstream quirk).
            
            name = args.pop("name")
        calls.append(
            {"name": name, "arguments": args if isinstance(args, dict) else {}}
        )
    remaining = _TOOL_CALL_BLOCK_RE.sub("", text).strip()
    return calls, remaining


def _parse_tool_responses(text: str) -> list[str]:
    'Return the inner-text of each ``<tool_response>`` block in ``text``.'
    return [m.group(1).strip() for m in _TOOL_RESPONSE_BLOCK_RE.finditer(text)]


def _assistant_msg(text: str) -> Message:
    'Build an assistant message, parsing any ``<tool_call>`` repr blocks.'
    calls, remaining = _parse_tool_calls(text)
    if calls:
        return {"role": "assistant", "content": remaining, "tool_calls": calls}
    return {"role": "assistant", "content": text.strip()}


def load(tokenizer: "PreTrainedTokenizer", cfg: DataConfig) -> Iterator[UnifiedExample]:
    """Yield ``tool_calls_multiturn`` examples normalized to ``{tools, messages}``."""
    path = cfg.data_dir / FILENAME
    for i, rec in enumerate(read_jsonl(path)):
        system_text = rec.get("system", "") or ""
        query = rec.get("query", "") or ""
        response = rec.get("response", "") or ""
        history = rec.get("history") or []
        if not response:
            continue

        tools, preamble = _parse_tools(system_text)
        prompt: list[Message] = [{"role": "system", "content": preamble}]

        # Prior turns from history
        for pair in history:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            u, a = pair[0], pair[1]
            if u:
                prompt.append({"role": "user", "content": u})
            if a:
                prompt.append(_assistant_msg(a))

        # The query: either a normal user turn or one/more <tool_response> turns.
        
        if "<tool_response>" in query:
            for content in _parse_tool_responses(query):
                prompt.append({"role": "tool", "content": content})
        elif query:
            prompt.append({"role": "user", "content": query})

        reference_message = _assistant_msg(response)
        reference_messages = [reference_message]
        reference = assistant_content_str(reference_messages)
        if not reference:
            continue

        yield UnifiedExample(
            task_class="tool_calls_multiturn",
            prompt=prompt,
            reference=reference,
            tools=tools,
            reference_messages=reference_messages,
            source=f"{FILENAME}#{i}",
            extra={
                "is_call_turn": "<tool_call>" in response,
                "is_final_turn": "<tool_response>" in query and "<tool_call>" not in response,
            },
        )
