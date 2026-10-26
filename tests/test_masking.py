'Tests for masking (prompt/reference boundary + tool-token mask)'

from __future__ import annotations

from llm_training.masking import compute_tool_token_mask, mask_example
from llm_training.schema import UnifiedExample


def _plain_example() -> UnifiedExample:
    return UnifiedExample(
        task_class="menon_qa",
        prompt=[
            {"role": "system", "content": "You are Meno."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        reference="2 + 2 equals 4.",
        reference_messages=[{"role": "assistant", "content": "2 + 2 equals 4."}],
    )


def _tool_example() -> UnifiedExample:
    return UnifiedExample(
        task_class="opencode_gen",
        prompt=[{"role": "system", "content": "You use tools."}, {"role": "user", "content": "list files"}],
        reference="called ls and answered",
        tools=[{"type": "function", "function": {"name": "ls", "description": "list",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}],
        reference_messages=[
            {"role": "assistant", "content": "", "tool_calls": [{"name": "ls", "arguments": {"path": "/tmp"}}]},
            {"role": "tool", "content": "file_a\nfile_b"},
            {"role": "assistant", "content": "There are two files."},
        ],
    )


def test_plain_mask_prefix_ok_and_reference_decodes(tokenizer):
    ex = _plain_example()
    res = mask_example(ex, tokenizer)
    assert res.prefix_ok is True
    assert res.total_tokens == len(res.prompt_ids) + len(res.reference_ids)
    # Reference decodes back to a string containing the answer text.
    decoded = tokenizer.decode(res.reference_ids, skip_special_tokens=False)
    assert "2 + 2 equals 4." in decoded
    # Labels: -100 on prompt, ids on reference
    assert res.labels[: len(res.prompt_ids)] == [-100] * len(res.prompt_ids)
    assert res.labels[len(res.prompt_ids):] == res.reference_ids
    assert res.tool_token_mask is None


def test_tool_mask_prefix_ok_and_decodes(tokenizer):
    ex = _tool_example()
    res = mask_example(ex, tokenizer)
    assert res.prefix_ok is True
    decoded = tokenizer.decode(res.reference_ids, skip_special_tokens=False)
    assert "<tool_call>" in decoded
    assert "file_a" in decoded  # tool_response content
    assert "There are two files." in decoded


def test_tool_token_mask_flags_only_tool_blocks(tokenizer):
    ex = _tool_example()
    res = mask_example(ex, tokenizer)
    assert res.tool_token_mask is not None
    assert len(res.tool_token_mask) == len(res.reference_ids)
    assert any(res.tool_token_mask), "expected some tokens flagged as tool-mechanics"
    # Tokens flagged must decode into tool-block text (tool_call/tool_response).
    flagged = [t for t, m in zip(res.reference_ids, res.tool_token_mask) if m]
    assert "<tool" in tokenizer.decode(flagged, skip_special_tokens=False)
    # Non-flagged tokens must NOT contain tool delimiters.
    unflagged = [t for t, m in zip(res.reference_ids, res.tool_token_mask) if not m]
    decoded_unflagged = tokenizer.decode(unflagged, skip_special_tokens=False)
    assert "<tool_call>" not in decoded_unflagged and "</tool_call>" not in decoded_unflagged


def test_compute_tool_token_mask_directly(tokenizer):
    text = (
        "before <tool_call>\n{\"name\": \"ls\", \"arguments\": {\"path\": \"/tmp\"}}\n</tool_call>"
        " mid <tool_response>\nfile1\n</tool_response> after"
    )
    mask = compute_tool_token_mask(text, tokenizer)
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(mask) == len(ids)
    # "before", "mid", "after" tokens must be unflagged; block tokens flagged.
    flagged = tokenizer.decode([t for t, m in zip(ids, mask) if m], skip_special_tokens=False)
    assert "<tool_call>" in flagged and "</tool_call>" in flagged
    assert "<tool_response>" in flagged and "file1" in flagged
    unflagged = tokenizer.decode([t for t, m in zip(ids, mask) if not m], skip_special_tokens=False)
    for token in ("<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "file1"):
        assert token not in unflagged


def test_mask_multiturn_history_example(tokenizer):
    # Multi-turn: prompt includes prior turns, reference is the LAST assistant.
    ex = UnifiedExample(
        task_class="mtrageval",
        prompt=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ],
        reference="a2",
        reference_messages=[{"role": "assistant", "content": "a2"}],
    )
    res = mask_example(ex, tokenizer)
    assert res.prefix_ok is True
    assert "a2" in tokenizer.decode(res.reference_ids, skip_special_tokens=False)
    assert "a1" in tokenizer.decode(res.prompt_ids, skip_special_tokens=False)
