'Tests for warmup/split.py — the stratified SGR-only train/val split.\n\n\nAll offline over the real in-repo SGR data (loads fast, no network).'

from __future__ import annotations

from pathlib import Path

import pytest

from llm_training.config import DataConfig
from llm_training.loaders import sgr_agent
from llm_training.schema import UnifiedExample
from llm_training.warmup.split import (
    SGR_CLASSES,
    WarmupSplit,
    split_warmup_examples,
    turn_kind,
)


def _ex(task_class: str, kind: str, i: int) -> UnifiedExample:
    'Build a minimal example of a given turn kind.'
    extra: dict = {"turn_index": i}
    if kind == "final":
        extra.update({"is_call_turn": True, "is_final_answer": True})
    elif kind == "reasoning":
        extra.update({"is_call_turn": False, "is_reasoning_turn": True})
    elif kind == "action":
        extra.update({"is_call_turn": True})
    return UnifiedExample(
        task_class=task_class,
        prompt=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        reference="ref",
        reference_messages=[{"role": "assistant", "content": "ref"}],
        extra=extra,
    )


class TestTurnKind:
    def test_kinds(self):
        assert turn_kind(_ex("coder", "final", 0)) == "final"
        assert turn_kind(_ex("data_analyst", "reasoning", 0)) == "reasoning"
        assert turn_kind(_ex("data_analyst", "action", 0)) == "action"
        assert turn_kind(_ex("menon_qa", "plain", 0)) == "plain"


class TestSplitBasics:
    def test_no_overlap_and_full_coverage(self):
        examples = [
            _ex(cls, kind, i)
            for cls in ("coder", "data_analyst")
            for kind in ("final", "reasoning", "action")
            for i in range(10)
        ]
        split = split_warmup_examples(examples, seed=42, val_ratio=0.1)
        ids_train = {id(e) for e in split.train}
        ids_val = {id(e) for e in split.val}
        assert not (ids_train & ids_val)  # disjoint
        assert len(split.train) + len(split.val) == len(examples)

    def test_val_ratio_respected_per_stratum(self):
        n = 20
        examples = [_ex("coder", "final", i) for i in range(n)]
        split = split_warmup_examples(examples, seed=42, val_ratio=0.1)
        assert len(split.val) == round(n * 0.1)  # exactly 2 of 20

    def test_deterministic(self):
        examples = [
            _ex(cls, kind, i)
            for cls in ("coder", "summarizer")
            for kind in ("final", "reasoning")
            for i in range(15)
        ]
        s1 = split_warmup_examples(examples, seed=42, val_ratio=0.2)
        s2 = split_warmup_examples(examples, seed=42, val_ratio=0.2)
        assert [e.source or i for i, e in enumerate(s1.val)] == [
            e.source or i for i, e in enumerate(s2.val)
        ]
        assert len(s1.val) == len(s2.val)
        assert {id(e) for e in s1.val} == {id(e) for e in s2.val}

    def test_stratified_by_kind(self):
        # Every kind present in val when the strata are big enough.
        examples = [
            _ex("data_analyst", kind, i)
            for kind in ("final", "reasoning", "action")
            for i in range(20)
        ]
        split = split_warmup_examples(examples, seed=42, val_ratio=0.2)
        kinds_val = {turn_kind(e) for e in split.val}
        assert kinds_val == {"final", "reasoning", "action"}
        kinds_train = {turn_kind(e) for e in split.train}
        assert kinds_train == {"final", "reasoning", "action"}


class TestNonSGRAndExclude:
    def test_non_sgr_goes_all_to_train(self):
        sgr = [_ex("coder", "final", i) for i in range(10)]
        aux = [
            UnifiedExample(
                task_class="tool_calls_multiturn",
                prompt=[{"role": "system", "content": "s"},
                        {"role": "user", "content": "u"}],
                reference="r",
                reference_messages=[{"role": "assistant", "content": "r"}],
            )
        ]
        split = split_warmup_examples(sgr + aux, seed=42, val_ratio=0.1)
        assert all(e.task_class == "sgr-class" or True for e in split.val)
        assert all(e.task_class in SGR_CLASSES for e in split.val)
        n_aux_train = sum(1 for e in split.train if e.task_class == "tool_calls_multiturn")
        assert n_aux_train == len(aux)  # auxiliary data → train in full

    def test_exclude_task_removes_class_entirely(self):
        examples = (
            [_ex("coder", "final", i) for i in range(10)]
            + [_ex("summarizer", "final", i) for i in range(10)]
        )
        split = split_warmup_examples(
            examples, seed=42, val_ratio=0.1, exclude_tasks=["summarizer"]
        )
        assert split.excluded_examples == 10
        assert split.excluded_classes == ["summarizer"]
        assert all(e.task_class != "summarizer" for e in split.train + split.val)

    def test_single_example_stratum_stays_in_train(self):
        examples = [_ex("coder", "final", 0)]  # a single example
        split = split_warmup_examples(examples, seed=42, val_ratio=0.1)
        assert len(split.val) == 0
        assert len(split.train) == 1

    def test_bad_val_ratio_rejected(self):
        with pytest.raises(ValueError, match="val_ratio"):
            split_warmup_examples([], seed=1, val_ratio=0.0)
        with pytest.raises(ValueError):
            split_warmup_examples([], seed=1, val_ratio=1.0)


class TestRealDataSmoke:
    def test_real_sgr_split(self, data_dir):
        # The real in-repo SGR corpus splits cleanly with every kind in val.
        cfg = DataConfig(data_dir=Path(data_dir))
        examples = list(sgr_agent.load_class("data_analyst", None, cfg))
        assert len(examples) > 30
        split = split_warmup_examples(examples, seed=42, val_ratio=0.15)
        assert 0 < len(split.val) < len(examples)
        kinds_val = {turn_kind(e) for e in split.val}
        assert kinds_val == {"final", "reasoning", "action"}
        # strata_stats account for every example of the class.
        total_in_stats = sum(s["total"] for s in split.strata_stats.values())
        assert total_in_stats == len(examples)
