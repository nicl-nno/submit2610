'Tests for scripts/measure_signal_variance.py — report loading (``_load_pairs``).\n\n\nThe script lives outside the installed package, so it is loaded by path via\n``importlib``. All tests are offline: they exercise only ``_load_pairs`` (pure\nJSON parsing), never the fluency / ``code_check`` computation.\nfluency / ``code_check``.'

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "measure_signal_variance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_measure_signal_variance", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def msv():
    return _load_module()


def _write(tmp_path: Path, body: str) -> Path:
    report = tmp_path / "eval.json"
    report.write_text(body, encoding="utf-8")
    return report


class TestLoadPairs:
    def test_reads_examples_nested_under_metrics(self, tmp_path, msv):
        # run_eval.py's ``_build_report`` nests the detailed dump under
        # ``metrics`` (pinned by test_run_eval.py::TestMainSmoke). The reader
        # must follow that layout, not a hypothetical top-level ``examples``.
        # _build_report
        
        
        report = _write(
            tmp_path,
            '{"metrics": {"examples": ['
            '{"task_class": "menon_qa", "samples": [{"text": "A. B."}]},'
            '{"task_class": "coder", "samples": [{"text": "x = 1"}]}'
            "]}}",
        )
        pairs = msv._load_pairs(report)
        assert pairs == [("menon_qa", "A. B.", {}), ("coder", "x = 1", {})]

    def test_top_level_examples_are_not_read(self, tmp_path, msv):
        # Regression guard: the OLD (buggy) schema put ``examples`` at the top
        # level, which never matched run_eval.py output. A report carrying only
        # a top-level ``examples`` must yield nothing — the real path is
        # ``metrics.examples``.
        
        
        
        
        report = _write(
            tmp_path,
            '{"examples": [{"task_class": "menon_qa", '
            '"samples": [{"text": "A. B."}]}]}',
        )
        assert msv._load_pairs(report) == []

    def test_multiple_samples_per_example(self, tmp_path, msv):
        report = _write(
            tmp_path,
            '{"metrics": {"examples": ['
            '{"task_class": "coder", "samples": ['
            '{"text": "ans1"}, {"text": "ans2"}, {"text": ""}'
            "]}]}}",
        )
        # Empty-text samples are dropped; the rest keep their order.
        
        assert msv._load_pairs(report) == [("coder", "ans1", {}), ("coder", "ans2", {})]

    def test_missing_examples_key_is_empty(self, tmp_path, msv):
        report = _write(tmp_path, '{"config": {}, "metrics": {}}')
        assert msv._load_pairs(report) == []

    def test_missing_metrics_key_is_empty(self, tmp_path, msv):
        report = _write(tmp_path, '{"config": {}}')
        assert msv._load_pairs(report) == []

    def test_final_turns_contribute_extracted_answer(self, tmp_path, msv):
        # The reward-equivalent population (Path A): a final-answer turn must
        # contribute its ``answer`` field extracted from the finalanswertool
        # call, NOT the raw ``<tool_call>`` envelope (which has no scoreable
        # prose). Before this fix final turns silently vanished from the pool.
        
        
        import json as _json
        answer = "Объяснение решения задачи с деталями."
        envelope = (
            "<tool_call>\n"
            + _json.dumps(
                {"name": "finalanswertool",
                 "arguments": {"answer": answer, "status": "completed"}},
                ensure_ascii=False,
            )
            + "\n</tool_call>"
        )
        report = _write(
            tmp_path,
            '{"metrics": {"examples": [{"task_class": "coder", '
            '"extra": {"is_final_answer": true, "is_call_turn": true}, '
            '"samples": [{"text": ' + _json.dumps(envelope) + "}]}]}}",
        )
        pairs = msv._load_pairs(report)
        assert pairs == [("coder", answer, {"is_final_answer": True, "is_call_turn": True})]

    def test_final_turn_without_valid_call_skipped(self, tmp_path, msv):
        # A final turn with no valid finalanswertool call has no scoreable
        # content → the sample is skipped (not fed to the fluency model).
        
        report = _write(
            tmp_path,
            '{"metrics": {"examples": [{"task_class": "coder", '
            '"extra": {"is_final_answer": true, "is_call_turn": true}, '
            '"samples": [{"text": "мусор без tool-call"}, {"text": ""}]}]}}',
        )
        assert msv._load_pairs(report) == []

    def test_reasoning_turns_keep_raw_text(self, tmp_path, msv):
        report = _write(
            tmp_path,
            '{"metrics": {"examples": [{"task_class": "summarizer", '
            '"extra": {"is_reasoning_turn": true, "is_call_turn": false}, '
            '"samples": [{"text": "Reasoning steps:\\n- a"}]}]}}',
        )
        pairs = msv._load_pairs(report)
        assert pairs == [
            ("summarizer", "Reasoning steps:\n- a",
             {"is_reasoning_turn": True, "is_call_turn": False})
        ]
