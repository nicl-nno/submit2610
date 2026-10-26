"Tests for grpo_train.py's data preparation + the dispatcher ``question`` column.\n\n\nCovers: pre-rendered TEXT prompts with baked tools, the task_class/reference/\nextra/question columns, the held-out classes excluded from the training pool,\nand ``make_reward_function`` preferring the precomputed question."

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "grpo_train.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_train_data", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tr = _load_module()


def _ex(task_class: str = "coder", *, final: bool = True, ref: str = "r"):
    extra = (
        {"is_call_turn": True, "is_final_answer": True, "turn_index": 0}
        if final
        else {"is_call_turn": False, "is_reasoning_turn": True, "turn_index": 0}
    )
    return type(
        "Ex",
        (),
        {
            "task_class": task_class,
            "prompt": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "вопрос пользователя"},
            ],
            "reference": ref,
            "reference_messages": [{"role": "assistant", "content": ref}],
            "tools": None,
            "source": f"{task_class}#0",
            "extra": extra,
        },
    )()


class TestBuildTrainingRows:
    def test_prompt_is_rendered_text(self, tokenizer):
        rows, stats = tr.build_training_rows([_ex("coder")], tokenizer)
        assert isinstance(rows[0]["prompt"], str)
        assert "<|im_start|>assistant" in rows[0]["prompt"]
        assert "<|im_start|>system" in rows[0]["prompt"]

    def test_columns_preserved(self, tokenizer):
        rows, _ = tr.build_training_rows([_ex("coder", ref="эталон")], tokenizer)
        assert rows[0]["task_class"] == "coder"
        assert rows[0]["reference"] == "эталон"
        assert rows[0]["extra"]["is_final_answer"] is True

    def test_question_precomputed_from_messages(self, tokenizer):
        # The judge question is extracted from the ORIGINAL message list —
        # a rendered text prompt cannot be re-parsed reliably.
        
        rows, _ = tr.build_training_rows([_ex("menon_qa")], tokenizer)
        assert rows[0]["question"] == "вопрос пользователя"

    def test_turn_kind_stats(self, tokenizer):
        rows, stats = tr.build_training_rows(
            [_ex("coder"), _ex("coder", final=False)], tokenizer
        )
        assert stats["n_rows"] == 2
        assert stats["turn_kinds"]["final"] == 1
        assert stats["turn_kinds"]["reasoning"] == 1

    def test_real_sgr_build_excludes_holdouts(self, tokenizer, data_dir):
        # The real build with the recommended hold-outs: summarizer and
        # math_reasoner never leak into the training pool — both are held out
        # into the SAME val pool (the merged-validation design).
        
        
        from llm_training.build import build_unified_dataset
        from llm_training.config import DataConfig

        cfg = DataConfig(data_dir=Path(data_dir), seed=42, max_per_class=50)
        result = build_unified_dataset(
            tokenizer, cfg,
            val_task=list(tr.RECOMMENDED_VAL_TASKS),
            include_network=False,
        )
        train_classes = {ex.task_class for ex in result.train}
        val_classes = {ex.task_class for ex in result.val}
        for cls in tr.RECOMMENDED_VAL_TASKS:
            assert cls not in train_classes
            assert cls in val_classes

    def test_rows_are_arrow_safe(self, tokenizer):
        # The regression class: pre-rendered TEXT rows must pass
        # Dataset.from_list where heterogeneous tool_calls messages once
        # raised ArrowInvalid.
        
        from datasets import Dataset

        rows, _ = tr.build_training_rows(
            [_ex("coder"), _ex("summarizer", final=False)], tokenizer
        )
        ds = Dataset.from_list(rows)  # must not raise
        assert len(ds) == 2


class TestRecommendedHoldouts:
    def test_mirrored_pair(self):
        # The pipeline was designed around summarizer/math_reasoner — the
        # SAME pair sft_warmup excluded. Both are held out into the ONE val
        # pool (the merged-validation design: the former separate test split
        # was never scored).
        assert tr.RECOMMENDED_VAL_TASKS == ("summarizer", "math_reasoner")


class TestDispatcherQuestionColumn:
    def _fake_scorer(self):
        from llm_training.reward.config import ScoredMetrics

        class _Scorer:
            def __init__(self):
                self.user_inputs: list[str] = []

            async def score(self, *, user_input, response, reference, task_class,
                            extra=None):
                self.user_inputs.append(user_input)
                return ScoredMetrics(semantic_similarity=0.5, combined=0.5)

        return _Scorer()

    @pytest.mark.asyncio
    async def test_precomputed_question_preferred(self):
        # grpo_train.py passes TEXT prompts; the precomputed question column must
        # reach the judge instead of parsing the rendered text.
        # grpo_train.py
        
        from llm_training.reward.config import RewardConfig
        from llm_training.reward.dispatch import make_reward_function

        scorer = self._fake_scorer()
        fn = make_reward_function(scorer, reward_config=RewardConfig())
        rewards = await fn(
            prompts=["<|im_start|>user\nrendered text<|im_end|>"],
            completions=["ответ"],
            task_class=["menon_qa"],
            reference=["r"],
            question=["готовый вопрос"],
        )
        assert rewards == [0.5]
        assert scorer.user_inputs == ["готовый вопрос"]

    @pytest.mark.asyncio
    async def test_falls_back_to_extraction_without_column(self):
        # run_eval.py path (no question column): extraction from the
        # message-list prompt as before.
        
        from llm_training.reward.config import RewardConfig
        from llm_training.reward.dispatch import make_reward_function

        scorer = self._fake_scorer()
        fn = make_reward_function(scorer, reward_config=RewardConfig())
        await fn(
            prompts=[[{"role": "user", "content": "вопрос из промпта"}]],
            completions=["ответ"],
            task_class=["menon_qa"],
            reference=["r"],
        )
        assert scorer.user_inputs == ["вопрос из промпта"]
