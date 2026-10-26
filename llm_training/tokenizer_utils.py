'Meno-Lite tokenizer utilities: loading, rendering, tool-baking.\n\n\nThe Meno-Lite tokenizer is a ``Qwen2TokenizerFast`` with the standard Qwen2.5\nchat template. Boundary markers are ``<|im_start|>{role}\\n`` … ``<|im_end|>``;\n``tools`` go into the system turn as a ``<tools>`` block, ``tool_calls`` render\nas ``<tool_call>``, and ``role:"tool"`` results render as ``<tool_response>``\ninside a ``user`` turn.\n\n\nTRL 0.20 ``GRPOTrainer`` applies ``apply_chat_template`` to the ``prompt``\ncolumn but does **not** forward a ``tools`` argument from the dataset. Therefore\nfor tool-use classes the tool schemas are *baked* into the system message\ncontent at load time (see :func:`bake_tools_into_system`), so the prompt is\nself-contained and TRL can render it with ``tools=None``.'

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from transformers import AutoTokenizer

from llm_training.schema import Message, UnifiedExample

#: Matches a real baked tools block: a ``<tools>`` tag followed (after optional
#: whitespace) by the first tool schema's opening ``{``. This distinguishes the
#: baked artifact from prose mentions like ``<tools></tools> XML tags:`` that
#: appear in some source preambles — those never have a ``{`` right after the
#: tag. Substring checks on ``"# Tools"`` / ``"</tools>"`` (used previously)
#: false-positive on such prose and skipped baking.


_BAKED_TOOLS_RE = re.compile(r"<tools>\s*\{")

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer

#: Chat-template boundary markers (Qwen2.5 / Meno-Lite).
# Meno-Lite).
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"
SYSTEM_PREFIX = f"{IM_START}system\n"
SYSTEM_SUFFIX = f"{IM_END}\n"


def load_tokenizer(model_id: str) -> "PreTrainedTokenizer":
    'Load the Meno-Lite tokenizer (or any Qwen2.5-compatible one).'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.chat_template is None:  # pragma: no cover - defensive
        raise ValueError(
            f'Tokenizer for {model_id!r}'
        )
    return tokenizer


def render(
    tokenizer: "PreTrainedTokenizer",
    messages: list[Message],
    *,
    add_generation_prompt: bool,
) -> str:
    'Render chat messages with the Meno-Lite template (``tools=None``).\n\n\n    ``tools`` is always ``None`` here: tool-use examples must have the tool\n    schemas pre-baked into their system message (see\n    :func:`bake_tools_into_system`). This keeps rendering uniform across all\n    task classes and matches what TRL does internally.'
    return tokenizer.apply_chat_template(
        messages,
        tools=None,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def bake_tools_into_system(
    tokenizer: "PreTrainedTokenizer",
    system_content: str,
    tools: list[Message] | None,
) -> str:
    'Bake tool schemas into a system message content, byte-exact to the template.\n\n\n    Returns ``system_content`` unchanged when ``tools`` is empty/``None``.\n    Otherwise renders a single-system-message conversation with ``tools=tools``\n    and strips the ``<|im_start|>system\\n`` / ``<|im_end|>\\n`` wrappers, so the\n    result is exactly the system *content* (with the ``<tools>`` block appended)\n    that the template would have produced. The result is then stored verbatim as\n    the system message content, and TRL can re-render it with ``tools=None``.'
    if not tools:
        return system_content
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_content}],
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not (rendered.startswith(SYSTEM_PREFIX) and rendered.endswith(SYSTEM_SUFFIX)):
        raise ValueError(
            'Unexpected tool-system rendering; template markers not found.'
        )
    return rendered[len(SYSTEM_PREFIX) : -len(SYSTEM_SUFFIX)]


def prepare_messages(
    example: UnifiedExample,
    tokenizer: "PreTrainedTokenizer",
) -> tuple[list[Message], list[Message]]:
    'Return (prompt_messages, full_messages) with tools baked into system.\n\n\n    - ``prompt_messages``: a copy of ``example.prompt`` where, for tool-use\n      examples, the leading system message has the tool schemas baked into its\n      content.\n    - ``full_messages``: ``prompt_messages + example.reference_messages``\n      (the complete conversation including the reference assistant turn(s)).\n\n    For non-tool examples this is essentially ``(example.prompt, prompt+ref)``.'
    prompt_messages: list[Message] = [dict(m) for m in example.prompt]
    if example.tools:
        if prompt_messages and prompt_messages[0].get("role") == "system":
            content = prompt_messages[0].get("content", "")
            # Idempotency: the bake emits a ``<tools>`` tag immediately
            # followed by the first tool schema's ``{``. Detect that artifact
            # precisely (via _BAKED_TOOLS_RE) rather than loose substrings like
            # "# Tools"/"</tools>", which false-positive on source preambles
            # that mention "<tools></tools> XML tags:" in prose.
            
            
            if _BAKED_TOOLS_RE.search(content):
                pass
            else:
                prompt_messages[0] = {
                    "role": "system",
                    "content": bake_tools_into_system(
                        tokenizer, content, example.tools
                    ),
                }
        else:
            raise ValueError(
                'Tool-use example has no leading system message to bake tools into.'
            )
    full_messages = [*prompt_messages, *[dict(m) for m in example.reference_messages]]
    return prompt_messages, full_messages


def tokenize(tokenizer: "PreTrainedTokenizer", text: str) -> list[int]:
    'Tokenize text to a list of ids (no special tokens added beyond text).'
    return tokenizer.encode(text, add_special_tokens=False)


def count_tokens(tokenizer: "PreTrainedTokenizer", text: str) -> int:
    'Count the number of tokens in ``text``.'
    return len(tokenize(tokenizer, text))


def decode_generation(tokenizer: "PreTrainedTokenizer", generated_ids: Any) -> str:
    "Decode generated token ids WITHOUT stripping the ``<tool_call>`` envelope.\n\n\n    ``<tool_call>`` / ``</tool_call>`` are **added special tokens** of the\n    Meno-Lite tokenizer (ids 145123 / 145124), so the plain\n    ``tokenizer.decode(..., skip_special_tokens=True)`` deletes them from the\n    text — ``format_check`` then finds no blocks and the strict harmonic mean\n    zeroes every final/action turn (the exact bug behind the first cluster\n    run's ``final=0.0000 action=0.0000`` val report, and behind the anchor\n    ctrl-point corruption it later caused in ``grpo_train.py``). Decoding with\n    ``skip_special_tokens=False`` keeps the envelope; the only cleanup needed\n    is the stop token HF ``generate`` appends to the continuation\n    (``<|im_end|>`` / ``<|endoftext|>``): trailing occurrences are stripped,\n    and any residual in-text occurrence of those two control tokens is removed\n    defensively (they cannot legitimately appear inside an assistant turn).\n    ``<tool_call>`` / ``</tool_response>`` are never touched — they are not\n    ``<|…|>``-shaped and never match the removal patterns.\n``</tool_call>``\nMeno-Lite (id 145123\n    ``tokenizer.decode(..., skip_special_tokens=True)``\n    — ``format_check``\n    final/action\n    ``grpo_train.py``"
    text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    stops = ("<|im_end|>", "<|endoftext|>")
    while True:
        for stop in stops:
            if text.endswith(stop):
                text = text[: -len(stop)].rstrip()
                break
        else:
            break
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    return text


def message_is_tool_call(msg: Message) -> bool:
    'True iff ``msg`` is an assistant turn carrying ``tool_calls``.'
    return msg.get("role") == "assistant" and bool(msg.get("tool_calls"))


def last_user_text(prompt: Any) -> str:
    'Return the content of the last ``user`` message in ``prompt`` (``""`` if none).\n\n\n    Accepts a conversational prompt (``list[{role, content}]``) or anything else\n    and never raises: non-list inputs and prompts without a user turn both yield\n    ``""``. This makes it safe for the reward dispatcher, where TRL may pass\n    various prompt shapes'
    if not isinstance(prompt, list):
        return ""
    for msg in reversed(prompt):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", "") or "")
    return ""


def assistant_content_str(messages: list[Message]) -> str:
    'Concatenate every assistant turn\'s content (text + ``tool_call`` blocks).\n\n\n    Per AGENTS.md, the tool-use "reference answer" is *all* assistant-generated\n    content (``<tool_call>`` blocks + final text); ``<tool_response>`` turns are\n    excluded (they come from the tool executor). This helper builds that string.\n    Handles both the Qwen form ``{name, arguments}`` and the OpenAI form\n    ``{function: {name, arguments}}`` of a tool call.\n    (``<tool_call>`` +'
    import json as _json

    parts: list[str] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        if m.get("content"):
            parts.append(str(m["content"]))
        for tc in m.get("tool_calls") or []:
            tc = tc.get("function", tc)
            parts.append(
                _json.dumps(
                    {"name": tc.get("name"), "arguments": tc.get("arguments", {})},
                    ensure_ascii=False,
                    indent=2,
                )
            )
    return "\n".join(parts)
