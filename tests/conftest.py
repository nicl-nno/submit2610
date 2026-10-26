'Shared pytest fixtures'

from __future__ import annotations

from pathlib import Path

import pytest
from transformers import AutoTokenizer

from llm_training.config import MENO_LITE_MODEL_ID

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "src_data"


@pytest.fixture(scope="session")
def tokenizer():
    'Session-scoped Meno-Lite tokenizer (cached by HF, ~fast on 2nd run).'
    return AutoTokenizer.from_pretrained(MENO_LITE_MODEL_ID)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    'Path to the in-repo ``src_data/`` catalog.'
    return DATA_DIR
