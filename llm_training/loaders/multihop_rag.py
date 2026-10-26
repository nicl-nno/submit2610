'Loader: MultiHopRAG → ``multihop_rag``.'

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from llm_training.config import DataConfig
from llm_training.loaders._history_like import load_history_like
from llm_training.schema import UnifiedExample

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer

CLASSES = ("multihop_rag",)
FILENAME = "multihop_rag.jsonl"


def load(tokenizer: "PreTrainedTokenizer", cfg: DataConfig) -> Iterator[UnifiedExample]:
    return load_history_like("multihop_rag", FILENAME, tokenizer, cfg)
