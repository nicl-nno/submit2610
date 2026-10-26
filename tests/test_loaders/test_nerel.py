'NEREL_Instruct loader test (network-gated).'

from __future__ import annotations

import pytest

from llm_training.config import DataConfig
from llm_training.loaders import nerel
from llm_training.masking import mask_example

pytestmark = pytest.mark.network


def _cfg(data_dir) -> DataConfig:
    return DataConfig(data_dir=data_dir)


def test_nerel_loads_validation_of_all_configs(tokenizer, data_dir):
    examples = list(nerel.load(tokenizer, _cfg(data_dir)))
    assert len(examples) > 100  # 6 configs × ~80 = ~480
    assert all(ex.task_class == "nerel_instruct" for ex in examples)
    configs = {ex.extra["config"] for ex in examples}
    assert configs == set(nerel.CONFIGS)


def test_nerel_tokenization_and_masking(tokenizer, data_dir):
    examples = list(nerel.load(tokenizer, _cfg(data_dir)))
    for ex in examples[:10]:
        res = mask_example(ex, tokenizer)
        assert res.prefix_ok, f"prefix failed for {ex.source}"
        assert len(res.reference_ids) > 0
