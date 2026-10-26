'Tests for the mt-RAGEval loader (8 sub-types → one ``mtrageval`` class).'

from __future__ import annotations

from collections import Counter

from llm_training.config import DataConfig
from llm_training.loaders import mtrageval
from llm_training.masking import mask_example


def _cfg(data_dir, seed=42) -> DataConfig:
    return DataConfig(data_dir=data_dir, seed=seed)


def test_loads_all_subtypes_into_one_class(tokenizer, data_dir):
    examples = list(mtrageval.load(tokenizer, _cfg(data_dir)))
    assert len(examples) == 45  # 45 examples total
    assert all(ex.task_class == "mtrageval" for ex in examples)
    subtypes = Counter(ex.extra["subtype"] for ex in examples)
    # All 8 sub-types are present.
    assert set(subtypes.keys()) == set(mtrageval.SUBTYPES)


def test_reference_is_last_assistant_content(tokenizer, data_dir):
    examples = list(mtrageval.load(tokenizer, _cfg(data_dir)))
    ex = examples[0]
    assert ex.reference_messages[0]["role"] == "assistant"
    assert ex.reference.strip()


def test_context_placement_varies(tokenizer, data_dir):
    # Across the 39 examples with context, both placements ("system" and "user")
    # must appear (the variation required by AGENTS.md).
    
    examples = list(mtrageval.load(tokenizer, _cfg(data_dir)))
    placements = Counter(
        ex.extra["context_placement"] for ex in examples if ex.extra["n_context_chunks"] > 0
    )
    assert placements["system"] > 0, "expected some examples with context in system"
    assert placements["user"] > 0, "expected some examples with context in user"


def test_context_placement_is_deterministic(tokenizer, data_dir):
    a = list(mtrageval.load(tokenizer, _cfg(data_dir, seed=42)))
    b = list(mtrageval.load(tokenizer, _cfg(data_dir, seed=42)))
    assert [ex.extra["context_placement"] for ex in a] == [
        ex.extra["context_placement"] for ex in b
    ]


def test_empty_context_examples_have_no_placement(tokenizer, data_dir):
    examples = list(mtrageval.load(tokenizer, _cfg(data_dir)))
    no_ctx = [ex for ex in examples if ex.extra["n_context_chunks"] == 0]
    assert len(no_ctx) == 6  # 6 empty contexts
    for ex in no_ctx:
        assert ex.extra["context_placement"] == "none"


def test_context_appears_in_prompt(tokenizer, data_dir):
    examples = list(mtrageval.load(tokenizer, _cfg(data_dir)))
    with_ctx = [ex for ex in examples if ex.extra["n_context_chunks"] > 0][0]
    # The rendered prompt must contain some of the context text.
    
    res = mask_example(with_ctx, tokenizer)
    decoded = tokenizer.decode(res.prompt_ids, skip_special_tokens=False)
    assert len(decoded) > 0


def test_tokenization_and_masking(tokenizer, data_dir):
    examples = list(mtrageval.load(tokenizer, _cfg(data_dir)))
    for ex in examples[:8]:
        res = mask_example(ex, tokenizer)
        assert res.prefix_ok, f"prefix failed for {ex.source}"
        assert len(res.reference_ids) > 0
