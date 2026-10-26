"Tests for grpo_train.py's anchor val budget audit (offline).\n\n\nThe audit is OBSERVABILITY ONLY: it reports (never alters) the anchor val\nexamples whose reference span exceeds ``--anchor-max-new-tokens`` or whose\nprompt + budget exceeds the model's position limit. The script lives outside\nthe installed package, so it is loaded by path via ``importlib``."

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

from llm_training.schema import UnifiedExample

_SCRIPT = Path(__file__).resolve().parents[2] / "grpo_train.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_train_audit", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tr = _load_module()


def _ex(ref_len: int = 10) -> UnifiedExample:
    return UnifiedExample(
        task_class="summarizer",
        prompt=[
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ],
        reference="слово " * ref_len,
        reference_messages=[{"role": "assistant", "content": "слово " * ref_len}],
        extra={"is_call_turn": True, "is_final_answer": True, "turn_index": 0},
    )


class TestAuditAnchorValBudget:
    def test_short_references_fit(self, tokenizer):
        report = tr.audit_anchor_val_budget(
            [_ex(10), _ex(20)],
            tokenizer,
            max_new_tokens=4096,
            max_positions=32768,
        )
        assert report["n"] == 2
        assert report["over_budget"] == []
        assert report["over_context"] == []
        assert 0 < report["max_ref_span"] <= 4096
        assert report["max_prompt_tokens"] > 0

    def test_over_budget_reference_reported(self, tokenizer, caplog):
        long_ = _ex(5000)
        with caplog.at_level(logging.WARNING, logger="train"):
            report = tr.audit_anchor_val_budget(
                [_ex(10), long_],
                tokenizer,
                max_new_tokens=4096,
                max_positions=None,
            )
        assert len(report["over_budget"]) == 1
        ex, ref_span, prompt_len = report["over_budget"][0]
        assert ex.reference == long_.reference
        assert ref_span > 4096
        assert prompt_len > 0
        assert report["max_ref_span"] == ref_span
        # The warning names the flag and the offending example.
        
        assert "anchor-max-new-tokens" in caplog.text
        assert "summarizer/final" in caplog.text

    def test_over_context_reported(self, tokenizer, caplog):
        with caplog.at_level(logging.WARNING, logger="train"):
            report = tr.audit_anchor_val_budget(
                [_ex(10), _ex(20)],
                tokenizer,
                max_new_tokens=4096,
                max_positions=100,  # prompt + budget > 100 for both examples
            )
        assert report["over_budget"] == []
        assert len(report["over_context"]) == 2
        ex, prompt_len = report["over_context"][0]
        assert prompt_len > 0
        assert "max_position_embeddings" in caplog.text

    def test_context_check_skipped_without_positions(self, tokenizer):
        report = tr.audit_anchor_val_budget(
            [_ex(10)],
            tokenizer,
            max_new_tokens=4096,
            max_positions=None,
        )
        assert report["over_context"] == []

    def test_info_line_when_everything_fits(self, tokenizer, caplog):
        with caplog.at_level(logging.INFO, logger="train"):
            tr.audit_anchor_val_budget(
                [_ex(10)],
                tokenizer,
                max_new_tokens=4096,
                max_positions=32768,
            )
        assert "Anchor val budget audit" in caplog.text
        assert "all fit" in caplog.text
