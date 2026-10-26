'Tests for tokenizer utilities (rendering, tool-baking)'

from __future__ import annotations

import pytest

from llm_training.schema import UnifiedExample
from llm_training.tokenizer_utils import (
    IM_END,
    IM_START,
    SYSTEM_PREFIX,
    SYSTEM_SUFFIX,
    bake_tools_into_system,
    count_tokens,
    decode_generation,
    message_is_tool_call,
    prepare_messages,
    render,
)


def test_render_adds_generation_prompt_for_user_turn(tokenizer):
    text = render(tokenizer, [{"role": "user", "content": "hi"}], add_generation_prompt=True)
    assert text.startswith(f"{IM_START}user\nhi{IM_END}\n")
    assert text.endswith(f"{IM_START}assistant\n")


def test_render_full_closes_assistant(tokenizer):
    text = render(
        tokenizer,
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        add_generation_prompt=False,
    )
    assert text.endswith(f"hello{IM_END}\n")


def test_render_assistant_tool_call(tokenizer):
    msgs = [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "ls", "arguments": {"path": "/tmp"}},
        ]},
    ]
    text = render(tokenizer, msgs, add_generation_prompt=False)
    assert "<tool_call>" in text and "</tool_call>" in text
    assert '"name": "ls"' in text and '"path": "/tmp"' in text


def test_render_tool_role_becomes_tool_response(tokenizer):
    msgs = [
        {"role": "user", "content": "x"},
        {"role": "tool", "content": "result-text"},
    ]
    text = render(tokenizer, msgs, add_generation_prompt=False)
    assert "<tool_response>" in text and "result-text" in text


def test_bake_tools_into_system_is_byte_exact_and_idempotent(tokenizer):
    tools = [{"type": "function", "function": {"name": "ls", "description": "list",
              "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
    baked = bake_tools_into_system(tokenizer, "You are helpful.", tools)
    # The baked content must reconstruct exactly what the template builds for a
    # system message rendered with tools=tools.
    
    full = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are helpful."}],
        tools=tools, tokenize=False, add_generation_prompt=False,
    )
    assert full == f"{SYSTEM_PREFIX}{baked}{SYSTEM_SUFFIX}"
    # Idempotent: no tools -> unchanged
    assert bake_tools_into_system(tokenizer, "You are helpful.", None) == "You are helpful."
    assert bake_tools_into_system(tokenizer, "You are helpful.", []) == "You are helpful."


def test_prepare_messages_bakes_tools_only_for_tool_examples(tokenizer):
    tools = [{"type": "function", "function": {"name": "ls", "description": "list",
              "parameters": {"type": "object"}}}]
    ex = UnifiedExample(
        task_class="opencode_gen",
        prompt=[{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}],
        reference="ref",
        tools=tools,
        reference_messages=[{"role": "assistant", "content": "ok"}],
    )
    prompt_msgs, full_msgs = prepare_messages(ex, tokenizer)
    assert "<tools>" in prompt_msgs[0]["content"]
    assert prompt_msgs[0]["role"] == "system"
    assert len(full_msgs) == len(prompt_msgs) + 1
    # Non-tool example: no baking
    plain = UnifiedExample(
        task_class="menon_qa",
        prompt=[{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}],
        reference="ref",
    )
    p2, _ = prepare_messages(plain, tokenizer)
    assert p2[0]["content"] == "sys"


def test_prepare_messages_requires_leading_system_for_tools(tokenizer):
    ex = UnifiedExample(
        task_class="opencode_gen",
        prompt=[{"role": "user", "content": "u"}],  # no system
        reference="ref",
        tools=[{"type": "function", "function": {"name": "x"}}],
    )
    with pytest.raises(ValueError):
        prepare_messages(ex, tokenizer)


def test_count_tokens_positive(tokenizer):
    assert count_tokens(tokenizer, "hello world") > 0


def test_message_is_tool_call():
    assert message_is_tool_call({"role": "assistant", "tool_calls": [{"name": "x"}]}) is True
    assert message_is_tool_call({"role": "assistant", "content": "hi"}) is False
    assert message_is_tool_call({"role": "user", "content": "hi"}) is False


def test_decode_generation_keeps_tool_call_envelope(tokenizer):
    # <tool_call>/</tool_call> are added special tokens — a plain
    # skip_special_tokens=True decode deletes them (the bug that once zeroed
    # the warm-up val report and later the grpo_train.py anchor AC).
    
    text = '<tool_call>\n{"name": "ls", "arguments": {"path": "/tmp"}}\n</tool_call><|im_end|>'
    ids = tokenizer.encode(text, add_special_tokens=False)
    out = decode_generation(tokenizer, ids)
    assert out.startswith("<tool_call>")
    assert out.endswith("</tool_call>")
    assert '"name": "ls"' in out
    assert "<|im_end|>" not in out


def test_decode_generation_strips_trailing_stop_tokens(tokenizer):
    # HF generate appends <|im_end|> (or <|endoftext|>) — both must be cut.
    # HF generate
    for stop in ("<|im_end|>", "<|endoftext|>"):
        ids = tokenizer.encode(f"Ответ текстом.{stop}", add_special_tokens=False)
        assert decode_generation(tokenizer, ids) == "Ответ текстом."
    ids = tokenizer.encode("Двойной<|im_end|>\n<|endoftext|>", add_special_tokens=False)
    assert decode_generation(tokenizer, ids) == "Двойной"


def test_decode_generation_removes_stray_control_tokens(tokenizer):
    # Residual in-text control tokens cannot legitimately appear inside an
    # assistant turn — removed defensively; the envelope is never touched.
    
    ids = tokenizer.encode("До<|im_end|>после<|endoftext|>", add_special_tokens=False)
    assert decode_generation(tokenizer, ids) == "Допосле"
