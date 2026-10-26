'Tests for the tool_calls_multiturn loader (Python-repr parsing & normalization).'

from __future__ import annotations

import pytest

from llm_training.config import DataConfig
from llm_training.loaders import tool_calls_multiturn as tcmt
from llm_training.masking import mask_example


def _cfg(data_dir) -> DataConfig:
    return DataConfig(data_dir=data_dir)


@pytest.fixture(scope="module")
def examples(tokenizer, data_dir):
    return list(tcmt.load(tokenizer, _cfg(data_dir)))


def test_loads_many_records(examples):
    assert len(examples) > 500  # 2655 records


def test_call_and_final_turns_present(examples):
    calls = [ex for ex in examples if ex.extra["is_call_turn"]]
    finals = [ex for ex in examples if ex.extra["is_final_turn"]]
    assert len(calls) > 100, "expected many call-turn records"
    assert len(finals) > 100, "expected many final-answer records"


def test_no_python_repr_remains(examples):
    'After normalization, no raw Python-repr must survive (no single-quoted\n    dicts, no True/False/None outside JSON). The arguments must be JSON objects.'
    bad = []
    for ex in examples[:500]:
        for msg in ex.prompt + ex.reference_messages:
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", tc)
                args = fn.get("arguments")
                if not isinstance(args, dict):
                    bad.append((ex.source, "arguments not dict", repr(args)[:60]))
    assert not bad[:3], f"non-normalized tool_call arguments: {bad[:3]}"


def test_tools_are_parsed_not_repr(examples):
    """The ``tools`` field is a real list of dicts (parsed from repr), not text."""
    ex = next(e for e in examples if e.tools)
    assert isinstance(ex.tools, list)
    assert all(isinstance(t, dict) for t in ex.tools)
    assert all("function" in t or "name" in t for t in ex.tools)


def test_call_turn_reference_has_tool_calls(examples):
    ex = next(e for e in examples if e.extra["is_call_turn"])
    assert ex.reference_messages[0].get("tool_calls")
    assert ex.reference_messages[0]["role"] == "assistant"


def test_all_call_turns_have_parsed_tool_calls(examples):
    'Every call-turn reference must carry parsed ``tool_calls`` (regression for\n    the hybrid ``\\n``-padded ``<tool_call>`` blocks that ``ast.literal_eval``\n    alone rejected). Before the fix ~793 records (~30 %) silently fell back to a\n    raw-markup text reference with no ``tool_calls`` field.'
    call_turns = [e for e in examples if e.extra["is_call_turn"]]
    assert call_turns, "expected call-turn records in the dataset"
    missing = [
        e.source
        for e in call_turns
        if not e.reference_messages[0].get("tool_calls")
    ]
    assert not missing[:5], (
        f"{len(missing)} call-turn records lack parsed tool_calls, e.g. {missing[:5]}"
    )


def test_final_turn_reference_is_text(examples):
    ex = next(e for e in examples if e.extra["is_final_turn"])
    assert not ex.reference_messages[0].get("tool_calls")
    assert ex.reference_messages[0]["content"]


def test_final_turn_prompt_has_tool_response_role(examples):
    ex = next(e for e in examples if e.extra["is_final_turn"])
    roles = [m["role"] for m in ex.prompt]
    assert "tool" in roles, "final-turn prompt must contain tool results"


def test_history_assistant_tool_calls_parsed(examples):
    # At least one example carries a history assistant turn with parsed tool_calls.
    
    found = False
    for ex in examples:
        for m in ex.prompt:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                found = True
                break
        if found:
            break
    assert found, "expected history assistant tool_calls to be parsed"


def test_tokenization_masking_and_tools_baked(tokenizer, examples):
    ex = next(e for e in examples if e.tools)
    res = mask_example(ex, tokenizer)
    assert res.prefix_ok
    assert res.tool_token_mask is not None
    decoded_prompt = tokenizer.decode(res.prompt_ids, skip_special_tokens=False)
    # Tools baked as proper JSON (double quotes), not Python repr.
    
    assert "<tools>" in decoded_prompt
    assert '"name"' in decoded_prompt or '"type": "function"' in decoded_prompt


def test_rendered_tool_call_uses_json_not_repr(tokenizer, examples):
    ex = next(e for e in examples if e.extra["is_call_turn"])
    res = mask_example(ex, tokenizer)
    decoded = tokenizer.decode(res.reference_ids, skip_special_tokens=False)
    assert "<tool_call>" in decoded
    # JSON uses double quotes for keys; Python repr would use single quotes.
    
    block = decoded.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
    assert '"name"' in block and '"arguments"' in block
    # No Python-repr single-quoted form
    assert "{'" not in block


# --- Parser unit tests: two upstream <tool_call> formats ---


def test_parse_call_payload_python_repr():
    # Pure Python repr (single quotes) — the original, dominant format.
    
    raw = "{'arguments': {'q': ['a', 'b']}, 'name': 'Foo'}"
    d = tcmt._parse_call_payload(raw)
    assert d == {"arguments": {"q": ["a", "b"]}, "name": "Foo"}


def test_parse_call_payload_hybrid_backslash_n():
    # Hybrid form (~30% of the upstream dataset): the block is padded with a
    # literal backslash+n and mixes a JSON-quoted outer dict with single-quoted
    # Python strings inside. A bare ast.literal_eval rejects this on the leading
    # backslash ("unexpected character after line continuation character").
    # Note: this returns the *raw* parsed dict (name nested inside arguments,
    # the upstream schema bug); the hoist to top-level happens in
    # _parse_tool_calls (see test_parse_tool_calls_handles_hybrid_block).
    
    
    
    raw = '\\n{"arguments": {"queries": [\'how?\', \'why?\'], "name": "ExpertQAExtractor"}}\\n'
    d = tcmt._parse_call_payload(raw)
    assert d == {"arguments": {"queries": ["how?", "why?"], "name": "ExpertQAExtractor"}}


def test_parse_call_payload_plain_json():
    # Pure JSON (double quotes throughout) — the json.loads fallback path.
    
    raw = '{"arguments": {"q": ["a"]}, "name": "Bar"}'
    d = tcmt._parse_call_payload(raw)
    assert d == {"arguments": {"q": ["a"]}, "name": "Bar"}


def test_parse_call_payload_unparseable_returns_none():
    # Documentation placeholder from a system prompt — not a real call.
    
    raw = "{'arguments': <args-dict>, 'name': <function-name>}"
    assert tcmt._parse_call_payload(raw) is None


def test_parse_tool_calls_handles_hybrid_block():
    # A full response that is exactly one hybrid-format <tool_call> block.
    
    response = '<tool_call>\\n{"arguments": {"x": [\'1\', \'2\']}, "name": "DoX"}\\n</tool_call>'
    calls, remaining = tcmt._parse_tool_calls(response)
    assert len(calls) == 1
    assert calls[0]["name"] == "DoX"
    assert calls[0]["arguments"] == {"x": ["1", "2"]}
    assert remaining == ""


def test_load_emits_no_unparseable_warnings(tokenizer, data_dir, caplog):
    "Loading the real dataset must not emit any 'skipping unparseable'\n    warnings now that both <tool_call> formats are handled."
    import logging as _logging

    with caplog.at_level(_logging.WARNING, logger="llm_training.loaders.tool_calls_multiturn"):
        list(tcmt.load(tokenizer, _cfg(data_dir)))
    skipped = [r for r in caplog.records if "skipping unparseable" in r.message]
    assert not skipped, f"unexpected unparseable-block warnings: {len(skipped)}"
