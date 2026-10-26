"Tests for sft_train.py's prompt-completion rows (offline, tokenizer).\n\n\nThe row builder is a verbatim copy of sft_warmup.py's — the sync test pins the\ntwo together so the ablation's rendering can never silently drift from the\nwarm-up's."

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


st = _load("_sft_train_rows", _REPO / "sft_train.py")


def _ex(cls: str, ref: str = "тестовый ответ модели") -> object:
    from llm_training.schema import UnifiedExample

    return UnifiedExample(
        task_class=cls,
        prompt=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        reference=ref,
        reference_messages=[{"role": "assistant", "content": ref}],
        extra={"is_call_turn": True, "is_final_answer": True, "turn_index": 0},
    )


class TestToPromptCompletion:
    def test_rows_carry_prompt_and_completion(self, tokenizer):
        rows = st.to_prompt_completion([_ex("coder")], tokenizer)
        assert len(rows) == 1
        assert isinstance(rows[0]["prompt"], str)
        assert isinstance(rows[0]["completion"], str)
        assert "<|im_start|>assistant" in rows[0]["prompt"]
        assert set(rows[0]) == {"prompt", "completion"}

    def test_completion_is_reference_suffix(self, tokenizer):
        # prompt + completion must concatenate to the full render.
        # prompt + completion
        from llm_training.tokenizer_utils import prepare_messages, render

        ex = _ex("summarizer", "краткое содержание")
        rows = st.to_prompt_completion([ex], tokenizer)
        _, full = prepare_messages(ex, tokenizer)
        full_text = render(tokenizer, full, add_generation_prompt=False)
        assert rows[0]["prompt"] + rows[0]["completion"] == full_text

    def test_in_sync_with_sft_warmup(self, tokenizer):
        # The ablation twin MUST render byte-identically to the warm-up.
        
        sw = _load("_sft_warmup_sync", _REPO / "sft_warmup.py")
        examples = [
            _ex("coder", "ответ с <tool_call>"),
            _ex("factual_qa", "фактологический ответ"),
        ]
        assert (
            st.to_prompt_completion(examples, tokenizer)
            == sw.to_prompt_completion(examples, tokenizer)
        )

    def test_render_mismatch_raises(self, tokenizer):
        # A malformed example (reference not renderable as a suffix) must
        # fail loudly, never emit a wrong completion span.
        
        ex = _ex("coder")
        ex.reference_messages = [{"role": "user", "content": "другое"}]
        try:
            st.to_prompt_completion([ex], tokenizer)
        except ValueError as e:
            assert "completion" in str(e)
        else:  # pragma: no cover
            raise AssertionError("ValueError not raised")


class TestTurnKindStats:
    def test_counts_all_four_kinds(self):
        from llm_training.schema import UnifiedExample

        def ex(extra):
            return UnifiedExample(
                task_class="coder",
                prompt=[{"role": "user", "content": "u"}],
                reference="r",
                reference_messages=[{"role": "assistant", "content": "r"}],
                extra=extra,
            )

        examples = [
            ex({"is_final_answer": True, "is_call_turn": True}),
            ex({"is_reasoning_turn": True}),
            ex({"is_call_turn": True}),
            ex({}),
            ex({"is_reasoning_turn": True}),
        ]
        stats = st.turn_kind_stats(examples)
        assert stats == {
            "n_rows": 5,
            "turn_kinds": {"final": 1, "reasoning": 2, "action": 1, "plain": 1},
        }
