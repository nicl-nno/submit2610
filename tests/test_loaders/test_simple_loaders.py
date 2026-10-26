'Tests for the simple loaders (parsing + tokenization + masking on real data).'

from __future__ import annotations

from llm_training.config import DataConfig
from llm_training.loaders import asr_summarization, multihop_rag, opencode_gen, rummlu_pro, self_cognition
from llm_training.masking import mask_example


def _cfg(data_dir) -> DataConfig:
    return DataConfig(data_dir=data_dir)


def _check_tokenization_and_masking(tokenizer, ex):
    'Shared invariants: rendering + clean masking boundary.'
    res = mask_example(ex, tokenizer)
    assert res.prefix_ok, f"prefix split failed for {ex.task_class} {ex.source}"
    assert res.total_tokens > 0
    assert len(res.reference_ids) > 0


# --------------------------------------------------------------------------- multihop_rag
def test_multihop_rag_parses(tokenizer, data_dir):
    examples = list(multihop_rag.load(tokenizer, _cfg(data_dir)))
    assert len(examples) > 100
    ex = examples[0]
    assert ex.task_class == "multihop_rag"
    assert ex.reference  # non-empty
    # Meno identity system prompt preserved
    assert any(m["role"] == "system" and "Meno" in m["content"] for m in ex.prompt)


def test_multihop_rag_tokenization_and_masking(tokenizer, data_dir):
    ex = next(iter(multihop_rag.load(tokenizer, _cfg(data_dir))))
    _check_tokenization_and_masking(tokenizer, ex)
    assert ex.tools is None


# --------------------------------------------------------------------------- rummlu_pro
def test_rummlu_pro_parses(tokenizer, data_dir):
    examples = list(rummlu_pro.load(tokenizer, _cfg(data_dir)))
    assert len(examples) > 100
    assert all(ex.task_class == "rummlu_pro" for ex in examples[:50])


def test_rummlu_pro_short_answers(tokenizer, data_dir):
    # ruMMLU-pro answers are short (a single option). Reference must be non-empty
    # but typically very short.
    ex = next(iter(rummlu_pro.load(tokenizer, _cfg(data_dir))))
    _check_tokenization_and_masking(tokenizer, ex)
    assert len(ex.reference) < 200  # short option


# --------------------------------------------------------------------------- self_cognition
def test_self_cognition_parses(tokenizer, data_dir):
    examples = list(self_cognition.load(tokenizer, _cfg(data_dir)))
    assert len(examples) > 50
    # Records with empty `system` have NO leading system message — the first
    # message is `user`. AGENTS: ~95/136 records; keep them system-less.
    
    no_system = [ex for ex in examples if ex.prompt[0]["role"] != "system"]
    assert len(no_system) > 50


def test_self_cognition_identity(tokenizer, data_dir):
    ex = next(iter(self_cognition.load(tokenizer, _cfg(data_dir))))
    assert "Anonymous Author" in ex.reference or "Meno" in ex.reference or ex.reference
    _check_tokenization_and_masking(tokenizer, ex)


def test_self_cognition_with_history(tokenizer, data_dir):
    # At least one record carries a prior [user, assistant] pair in history.
    examples = list(self_cognition.load(tokenizer, _cfg(data_dir)))
    with_history = [ex for ex in examples if len(ex.prompt) > 2]  # system?+user+...
    assert with_history, "expected at least one multi-turn self_cognition example"
    _check_tokenization_and_masking(tokenizer, with_history[0])


# --------------------------------------------------------------------------- opencode_gen
def test_opencode_gen_parses_single_turn(tokenizer, data_dir):
    # Each trajectory is split into one example per assistant turn, so the
    # example count exceeds the v2 trajectory count (34).
    
    
    examples = list(opencode_gen.load(tokenizer, _cfg(data_dir)))
    assert len(examples) > 34
    ex = examples[0]
    assert ex.task_class == "opencode_gen"
    assert ex.tools is not None and len(ex.tools) == 11
    # Each example is single-turn: prompt ends right before an assistant turn,
    # and reference_messages holds exactly one assistant message.
    
    assert ex.prompt[0]["role"] == "system"
    assert len(ex.reference_messages) == 1
    assert ex.reference_messages[0]["role"] == "assistant"


def test_opencode_gen_each_example_has_single_assistant_reference(tokenizer, data_dir):
    examples = list(opencode_gen.load(tokenizer, _cfg(data_dir)))
    for ex in examples:
        # Single-turn invariant: exactly one assistant reference message and no
        # tool message inside reference_messages.
        
        assert len(ex.reference_messages) == 1
        assert ex.reference_messages[0]["role"] == "assistant"
        assert "<tool_response>" not in ex.reference
        assert ex.reference.strip() != ""


def test_opencode_gen_has_call_turns_and_context_turns(tokenizer, data_dir):
    # Across the split: some examples are "call turns" (reference carries
    # tool_calls), and some have prior tool results folded into the prompt
    # (final turns after tool responses).
    
    
    examples = list(opencode_gen.load(tokenizer, _cfg(data_dir)))
    call_turns = [ex for ex in examples if ex.extra.get("is_call_turn")]
    final_turns = [ex for ex in examples if not ex.extra.get("is_call_turn")]
    assert call_turns, "expected at least one tool-call turn"
    assert final_turns, "expected at least one final-text turn"
    # Final turns condition on a prior tool result: their prompt contains a
    # role:"tool" message carried over from the trajectory.
    
    with_tool_ctx = [
        ex for ex in examples if any(m.get("role") == "tool" for m in ex.prompt)
    ]
    assert with_tool_ctx, "expected examples whose prompt carries role:'tool'"


def test_opencode_gen_call_turn_tokenization_masking_and_tools_baked(tokenizer, data_dir):
    # A call-turn reference is a <tool_call> block; masking must mark it.
    
    examples = list(opencode_gen.load(tokenizer, _cfg(data_dir)))
    call_turn = next(ex for ex in examples if ex.extra.get("is_call_turn"))
    res = mask_example(call_turn, tokenizer)
    assert res.prefix_ok
    assert res.tool_token_mask is not None
    assert any(res.tool_token_mask)
    decoded_ref = tokenizer.decode(res.reference_ids, skip_special_tokens=False)
    assert "<tool_call>" in decoded_ref


def test_opencode_gen_final_turn_tokenization_and_masking(tokenizer, data_dir):
    # A final-text turn: reference is plain text, still tokenizes cleanly, tools
    # are still baked into the system message.
    
    examples = list(opencode_gen.load(tokenizer, _cfg(data_dir)))
    final_turn = next(ex for ex in examples if not ex.extra.get("is_call_turn"))
    res = mask_example(final_turn, tokenizer)
    assert res.prefix_ok
    assert len(res.reference_ids) > 0


# --------------------------------------------------------------------------- asr_summarization
def test_asr_summarization_count_and_suffixes(tokenizer, data_dir):
    examples = list(asr_summarization.load(tokenizer, _cfg(data_dir)))
    nns = {ex.extra["nn"] for ex in examples}
    # NN 01–26 expected (26 triplets).
    assert len(examples) == 26
    assert min(nns) == 1 and max(nns) == 26


def test_asr_summarization_system_split(tokenizer, data_dir):
    """NN 01–13 have a systemNN.txt; NN 14–26 do not (empty system)."""
    examples = list(asr_summarization.load(tokenizer, _cfg(data_dir)))
    by_nn = {ex.extra["nn"]: ex for ex in examples}
    # NN 05 has a system; NN 15 does not.
    ex_with = by_nn[5]
    ex_without = by_nn[15]
    assert ex_with.extra["has_system"] is True
    assert ex_without.extra["has_system"] is False
    assert ex_with.prompt[0]["role"] == "system"
    # No system for NN 15 → user is the first message.
    assert ex_without.prompt[0]["role"] == "user"


def test_asr_summarization_tokenization_and_masking(tokenizer, data_dir):
    examples = list(asr_summarization.load(tokenizer, _cfg(data_dir)))
    for ex in examples[:5]:
        _check_tokenization_and_masking(tokenizer, ex)
        assert ex.tools is None
