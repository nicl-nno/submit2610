"Loader: self-cognition (ru) → ``self_cognition``.\n\n\nThe ``system`` field is taken **as-is**: many records (~95/136) have an empty\n``system``; per AGENTS.md we keep it empty and do **not** inject a default\nprompt. This preserves Meno's identity through fine-tuning."

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from llm_training.config import DataConfig
from llm_training.loaders._history_like import load_history_like
from llm_training.schema import UnifiedExample

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer

CLASSES = ("self_cognition",)
FILENAME = "self_cognition.jsonl"


def load(tokenizer: "PreTrainedTokenizer", cfg: DataConfig) -> Iterator[UnifiedExample]:
    return load_history_like("self_cognition", FILENAME, tokenizer, cfg)
