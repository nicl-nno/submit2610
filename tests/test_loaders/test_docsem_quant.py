'Tests for the docsem_quant loader (real in-repo data).\n\n\nMirrors ``test_simple_loaders``: parse + tokenize + mask invariants, plus\nthe class-specific ones — the gold metadata in ``extra`` (the deterministic\nreward reads it), the champion user-message layout (blocks + the trailing\n``Question:`` line) and the reference being a valid JSON trace.'

from __future__ import annotations

import json

from llm_training.config import DataConfig
from llm_training.loaders import docsem_quant
from llm_training.masking import mask_example


def _cfg(data_dir) -> DataConfig:
    return DataConfig(data_dir=data_dir)


def test_docsem_quant_parses(tokenizer, data_dir):
    examples = list(docsem_quant.load(tokenizer, _cfg(data_dir)))
    assert len(examples) >= 150  # the check-181 pool
    ex = examples[0]
    assert ex.task_class == "docsem_quant"
    assert [m["role"] for m in ex.prompt] == ["system", "user"]
    assert ex.tools is None


def test_docsem_quant_gold_metadata(tokenizer, data_dir):
    # The deterministic docsem_check reward reads gold from extra — every
    # example must carry a non-empty gold answer and a non-empty evidence list.
    
    for ex in docsem_quant.load(tokenizer, _cfg(data_dir)):
        assert ex.extra["gold_answer"]
        assert isinstance(ex.extra["gold_evidence"], list)
        assert ex.extra["gold_evidence"]
        assert isinstance(ex.extra["trace_verified"], bool)


def test_docsem_quant_reference_is_json_trace(tokenizer, data_dir):
    examples = list(docsem_quant.load(tokenizer, _cfg(data_dir)))
    verified = [e for e in examples if e.extra["trace_verified"]]
    assert verified, "expected verified champion traces / ожидались трассы"
    trace = json.loads(verified[0].reference)
    assert set(trace.keys()) == {"reasoning", "program", "answer", "evidence"}
    assert trace["program"].strip()
    assert trace["answer"] == verified[0].extra["gold_answer"]
    assert trace["evidence"] == verified[0].extra["gold_evidence"]


def test_docsem_quant_user_message_layout(tokenizer, data_dir):
    # The champion layout: blocks first, the question after the LAST
    # ``\nQuestion:`` marker (the reward's question extractor relies on it).
    
    ex = next(iter(docsem_quant.load(tokenizer, _cfg(data_dir))))
    user = ex.prompt[-1]["content"]
    assert user.startswith("Document blocks:\n")
    assert "\nQuestion:" in user


def test_docsem_quant_tokenization_and_masking(tokenizer, data_dir):
    ex = next(iter(docsem_quant.load(tokenizer, _cfg(data_dir))))
    res = mask_example(ex, tokenizer)
    assert res.prefix_ok
    assert res.total_tokens > 0
    assert len(res.reference_ids) > 0
