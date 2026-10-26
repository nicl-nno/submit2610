'Tests for the unified schema'

from __future__ import annotations

import pytest

from llm_training.schema import (
    EXTRACTION_LINE_TASK_CLASSES,
    TASK_CLASSES,
    UnifiedExample,
    to_dataset_row,
)


def test_task_classes_count_is_29():
    '28 classes + distill_b_loop (data B, spec 0.2.1 decision №21).'
    assert len(TASK_CLASSES) == 29
    assert len(set(TASK_CLASSES)) == 29  # no duplicates


def test_extraction_line_classes_contract():
    'The GRPO-extraction extra classes keep their own frozen contract.\n\n    modules — a rename must fail here, not at training time).'
    assert len(EXTRACTION_LINE_TASK_CLASSES) == 4
    assert len(set(EXTRACTION_LINE_TASK_CLASSES)) == 4
    # Disjoint from the unified registry (the frozen contract).
    assert not set(EXTRACTION_LINE_TASK_CLASSES) & set(TASK_CLASSES)
    # Covers exactly the consolidation + QA dispatch classes.
    from llm_training.extraction.qa_prompts import QA_TASK_CLASS
    from llm_training.reward.summarization import SUMMARY_TASK_CLASSES

    assert set(EXTRACTION_LINE_TASK_CLASSES) == set(SUMMARY_TASK_CLASSES) | {
        QA_TASK_CLASS
    }


def test_reward_config_accepts_extraction_line_fluency_classes():
    "RewardConfig's fluency allowlist validates against both registries."
    from llm_training.reward.config import RewardConfig

    cfg = RewardConfig(
        judge=None,
        compute_answer_correctness=False,
        fluency_classes=frozenset(EXTRACTION_LINE_TASK_CLASSES),
        # The default subtask map keys nerel_instruct, absent from this narrow
        # allowlist → the dead-config guard would fire (correctly); isolate the
        # class-registry check by dropping it.
        
        
        fluency_subtasks={},
    )
    assert cfg.fluency_classes == frozenset(EXTRACTION_LINE_TASK_CLASSES)
    # An unknown class still fails (the validation stays strict).
    with pytest.raises(ValueError):
        RewardConfig(
            judge=None,
            compute_answer_correctness=False,
            fluency_classes=frozenset(["not_a_real_class"]),
            fluency_subtasks={},
        )


def test_expected_classes_present():
    for c in [
        "coder", "menon_qa", "mtrageval",
        "ragu_artifact_extraction", "ragu_relation_validation",
        "asr_summarization", "tool_calls_multiturn", "opencode_gen",
        "multihop_rag", "rummlu_pro", "self_cognition", "nerel_instruct",
        "docsem_quant",
    ]:
        assert c in TASK_CLASSES


def _ok_example(**over) -> UnifiedExample:
    base = dict(
        task_class="menon_qa",
        prompt=[{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
        reference="answer",
    )
    base.update(over)
    return UnifiedExample(**base)


def test_example_builds_with_defaults():
    ex = _ok_example()
    assert ex.tools is None
    assert ex.reference_messages == []
    assert ex.extra == {}


def test_rejects_unknown_task_class():
    with pytest.raises(ValueError):
        _ok_example(task_class="not_a_real_class")


def test_rejects_empty_reference():
    with pytest.raises(ValueError):
        _ok_example(reference="")


def test_rejects_empty_prompt():
    with pytest.raises(ValueError):
        _ok_example(prompt=[])


def test_to_dataset_row_roundtrip():
    ex = _ok_example(tools=[{"type": "function"}], reference_messages=[{"role": "assistant", "content": "answer"}])
    row = to_dataset_row(ex)
    assert set(row.keys()) == {
        "prompt", "task_class", "reference", "tools",
        "reference_messages", "source", "extra",
    }
    assert row["task_class"] == "menon_qa"
    assert row["tools"] == [{"type": "function"}]
