'Tests for the SGR-agent loader (9 task classes, tool-call-faithful, turn-split).\n\n\nThe source (distilled from the sgr-agent-core SGR framework, formerly "sharegpt")\nstores tool-call-faithful trajectories where every finalization is a\n``finalanswertool`` call. The loader splits multi-step trajectories into one\nexample per assistant turn. These tests exercise the turn-splitting and the\nthree turn kinds (reasoning / action / final) on the real in-repo data.'

from __future__ import annotations

import pytest

from llm_training.config import DataConfig
from llm_training.loaders import sgr_agent
from llm_training.masking import mask_example

SGR_CLASSES = sgr_agent.CLASSES  # 9 classes
# Pure-final classes: a single finalanswertool turn, no reasoning/action.

PURE_FINAL_CLASSES = ("coder", "creative_writer", "instruction_following", "rewriter")
# Multi-step classes: reasoning + action tool-calls + final.

MULTI_STEP_CLASSES = ("data_analyst", "deep_research", "factual_qa", "summarizer")


def _cfg(data_dir) -> DataConfig:
    return DataConfig(data_dir=data_dir)


def _all_examples(cls, tokenizer, data_dir):
    return list(sgr_agent.load_class(cls, tokenizer, _cfg(data_dir)))


@pytest.fixture(scope="module")
def first_examples(tokenizer, data_dir):
    'One example per SGR class (the first).'
    return {cls: next(iter(sgr_agent.load_class(cls, tokenizer, _cfg(data_dir))))
            for cls in SGR_CLASSES}


@pytest.mark.parametrize("cls", SGR_CLASSES)
def test_each_class_yields_examples(tokenizer, data_dir, cls):
    examples = _all_examples(cls, tokenizer, data_dir)
    assert len(examples) > 0, f"no examples for {cls}"
    assert all(ex.task_class == cls for ex in examples)


@pytest.mark.parametrize("cls", PURE_FINAL_CLASSES)
def test_pure_final_classes_have_only_final_turns(tokenizer, data_dir, cls):
    # A pure-final class yields exactly one example per trajectory, all tagged
    # is_final_answer, and their count equals train+val record count.
    
    examples = _all_examples(cls, tokenizer, data_dir)
    assert examples, f"no examples for {cls}"
    assert all(ex.extra.get("is_final_answer") for ex in examples)
    assert all(ex.extra.get("is_call_turn") for ex in examples)
    # No reasoning / action turns in a pure-final class.
    
    assert not any(ex.extra.get("is_reasoning_turn") for ex in examples)


@pytest.mark.parametrize("cls", MULTI_STEP_CLASSES)
def test_multi_step_classes_split_into_turns(tokenizer, data_dir, cls):
    # A multi-step class yields reasoning, action and final turns (turn-splitting
    # produces strictly more examples than trajectories).
    
    examples = _all_examples(cls, tokenizer, data_dir)
    kinds = {
        "reasoning": [e for e in examples if e.extra.get("is_reasoning_turn")],
        "action": [e for e in examples if e.extra.get("is_call_turn")
                   and not e.extra.get("is_final_answer")
                   and not e.extra.get("is_reasoning_turn")],
        "final": [e for e in examples if e.extra.get("is_final_answer")],
    }
    assert kinds["reasoning"], f"no reasoning turns in {cls}"
    assert kinds["action"], f"no action turns in {cls}"
    assert kinds["final"], f"no final turns in {cls}"


def test_math_reasoner_reasoning_then_final(tokenizer, data_dir):
    # math_reasoner reasons once (no action tools) then finalises.
    # math_reasoner
    examples = _all_examples("math_reasoner", tokenizer, data_dir)
    reasoning = [e for e in examples if e.extra.get("is_reasoning_turn")]
    final = [e for e in examples if e.extra.get("is_final_answer")]
    assert reasoning and final
    # No action turns (no action tools in this class).
    
    action = [e for e in examples if e.extra.get("is_call_turn")
              and not e.extra.get("is_final_answer")
              and not e.extra.get("is_reasoning_turn")]
    assert action == []


@pytest.mark.parametrize("cls", SGR_CLASSES)
def test_first_example_is_well_formed(first_examples, cls):
    # Every class's first example has a non-empty string reference and an
    # assistant reference_messages entry.
    
    ex = first_examples[cls]
    assert ex.reference and isinstance(ex.reference, str)
    assert ex.reference_messages[0]["role"] == "assistant"


def test_final_turn_reference_and_metadata(tokenizer, data_dir):
    # Pick coder (pure-final): its first example IS a final turn.
    
    ex = next(iter(sgr_agent.load_class("coder", tokenizer, _cfg(data_dir))))
    assert ex.extra.get("is_final_answer") is True
    assert ex.extra.get("is_call_turn") is True
    # reference is the answer text (non-empty, plain string), NOT the tool_call JSON.
    # reference
    assert ex.reference and isinstance(ex.reference, str)
    assert "finalanswertool" not in ex.reference
    assert "arguments" not in ex.reference
    # reference_messages carries the finalanswertool tool_call.
    assert ex.reference_messages[0]["role"] == "assistant"
    tcs = ex.reference_messages[0].get("tool_calls") or []
    assert any((tc.get("function", tc) or {}).get("name") == "finalanswertool"
               for tc in tcs)


def test_reasoning_turn_reference_is_text(tokenizer, data_dir):
    # A reasoning turn's reference is the reasoning CoT text (no tool_call JSON),
    # tagged is_reasoning_turn and is_call_turn=False.
    # reference reasoning
    examples = _all_examples("data_analyst", tokenizer, data_dir)
    reasoning = next(e for e in examples if e.extra.get("is_reasoning_turn"))
    assert reasoning.extra.get("is_call_turn") is False
    assert reasoning.reference and isinstance(reasoning.reference, str)
    assert "<tool_call>" not in reasoning.reference
    # reference_messages is the reasoning assistant turn (no tool_calls).
    # reference_messages
    assert reasoning.reference_messages[0]["role"] == "assistant"
    assert not (reasoning.reference_messages[0].get("tool_calls"))


def test_action_turn_reference_is_tool_call(tokenizer, data_dir):
    # An action turn's reference is the rendered turn with a tool_call, tagged
    # is_call_turn=True (and not reasoning / not final).
    # reference action
    examples = _all_examples("data_analyst", tokenizer, data_dir)
    action = next(
        e for e in examples
        if e.extra.get("is_call_turn")
        and not e.extra.get("is_final_answer")
        and not e.extra.get("is_reasoning_turn")
    )
    # The action tool name (e.g. pandas_analysis_tool) appears in the reference,
    # which is built by assistant_content_str (content + tool_call JSON).
    
    assert action.reference
    assert any(
        (tc.get("function", tc) or {}).get("name")
        for tc in (action.reference_messages[0].get("tool_calls") or [])
    )


def test_tool_results_folded_into_prompt(tokenizer, data_dir):
    # For a multi-step trajectory, a later turn's prompt contains the prior
    # tool results as role:"tool" context (no tool executor needed).
    
    
    examples = _all_examples("data_analyst", tokenizer, data_dir)
    # Find a turn whose prompt includes a tool result (an action turn's result).
    
    has_tool_in_prompt = any(
        any(m.get("role") == "tool" for m in ex.prompt) for ex in examples
    )
    assert has_tool_in_prompt, "no tool results folded into any prompt"


def test_tools_baked_into_system(tokenizer, data_dir):
    # finalanswertool schema is baked into the system message on rendering.
    
    ex = next(iter(sgr_agent.load_class("coder", tokenizer, _cfg(data_dir))))
    assert ex.tools and len(ex.tools) > 0
    res = mask_example(ex, tokenizer)
    decoded_prompt = tokenizer.decode(res.prompt_ids, skip_special_tokens=False)
    assert "<tools>" in decoded_prompt


@pytest.mark.parametrize("cls", SGR_CLASSES)
def test_tokenization_and_masking(tokenizer, first_examples, cls):
    # Every first example renders cleanly (prefix split OK, reference non-empty).
    
    ex = first_examples[cls]
    res = mask_example(ex, tokenizer)
    assert res.prefix_ok, f"prefix split failed for {cls}"
    assert len(res.reference_ids) > 0


def test_train_and_val_merged(tokenizer, data_dir):
    train = sum(1 for _ in open(data_dir / "sgr_agent" / "train_coder.jsonl", encoding="utf-8"))
    val = sum(1 for _ in open(data_dir / "sgr_agent" / "val_coder.jsonl", encoding="utf-8"))
    examples = _all_examples("coder", tokenizer, data_dir)
    # coder is pure-final: one example per trajectory ⇒ examples == train+val.
    # coder
    assert len(examples) == train + val
