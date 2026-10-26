'Tests for the Menon-QA loader (vendored system prompt + datetime).'

from __future__ import annotations

import datetime

from llm_training.config import DataConfig
from llm_training.loaders import menon_qa
from llm_training.loaders._menon_qa_prompt import (
    QA_SYSTEM_PROMPT,
    format_system_prompt,
    sample_current_datetime,
)
from llm_training.masking import mask_example


def _cfg(data_dir, seed=42) -> DataConfig:
    return DataConfig(data_dir=data_dir, seed=seed)


def test_load_yields_examples(tokenizer, data_dir):
    examples = list(menon_qa.load(tokenizer, _cfg(data_dir)))
    assert len(examples) >= 25  # 31 records
    assert all(ex.task_class == "menon_qa" for ex in examples)


def test_system_prompt_is_vendored_constant(tokenizer, data_dir):
    ex = next(iter(menon_qa.load(tokenizer, _cfg(data_dir))))
    system = ex.prompt[0]["content"]
    # Must NOT contain the unfilled placeholder.
    assert "{current_datetime}" not in system
    # Must be the vendored prompt with the datetime line filled.
    assert system.startswith("Вы — Менон, виртуальный помощник")
    assert "Текущая дата и время:" in system
    assert "Менон" in system and "Anonymous City государственного университета" in system


def test_datetime_in_march_april_2026(tokenizer, data_dir):
    examples = list(menon_qa.load(tokenizer, _cfg(data_dir)))
    for ex in examples:
        dt = datetime.datetime.strptime(ex.extra["current_datetime"], "%Y-%m-%d %H:%M:%S")
        assert datetime.datetime(2026, 3, 1) <= dt <= datetime.datetime(2026, 4, 30, 23, 59, 59)


def test_datetime_is_deterministic_by_seed():
    # Same seed+index -> same datetime.
    assert sample_current_datetime(42, 0) == sample_current_datetime(42, 0)
    # Different seed -> (very likely) different datetime.
    assert sample_current_datetime(42, 0) != sample_current_datetime(7, 0)


def test_format_system_prompt_substitutes_placeholder():
    filled = format_system_prompt("2026-03-15 12:00:00")
    assert "{current_datetime}" not in filled
    assert "2026-03-15 12:00:00" in filled


def test_tokenization_and_masking(tokenizer, data_dir):
    ex = next(iter(menon_qa.load(tokenizer, _cfg(data_dir))))
    res = mask_example(ex, tokenizer)
    assert res.prefix_ok
    decoded_ref = tokenizer.decode(res.reference_ids, skip_special_tokens=False)
    assert ex.reference[:30] in decoded_ref
    assert res.tool_token_mask is None


def test_no_external_dependency_import():
    # The vendored prompt module must not import anything from meno_stand.
    
    import llm_training.loaders._menon_qa_prompt as mod
    assert "Менон" in QA_SYSTEM_PROMPT  # identity preserved
    assert hasattr(mod, "QA_SYSTEM_PROMPT")
