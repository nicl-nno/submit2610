'Loader: Menon QA → ``menon_qa``.\n\n\nThe source ``menon_qa.jsonl`` has records ``{query, response}`` with **no system\nprompt in the file**. Per AGENTS.md, the system prompt is the vendored\n:data:`~llm_training.loaders._menon_qa_prompt.QA_SYSTEM_PROMPT` (originally from\n``meno_stand/code/qa_utils/qa_utils.py``), with ``{current_datetime}`` filled by\na deterministically-sampled March–April 2026 datetime.'

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from llm_training.config import DataConfig
from llm_training.loaders._menon_qa_prompt import format_system_prompt, sample_current_datetime
from llm_training.loaders.base import read_jsonl
from llm_training.schema import UnifiedExample

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer

CLASSES = ("menon_qa",)
FILENAME = "menon_qa.jsonl"


def load(tokenizer: "PreTrainedTokenizer", cfg: DataConfig) -> Iterator[UnifiedExample]:
    """Yield ``menon_qa`` examples with the vendored system prompt injected."""
    path = cfg.data_dir / FILENAME
    for i, rec in enumerate(read_jsonl(path)):
        query = rec.get("query", "") or ""
        response = rec.get("response", "") or ""
        if not response:
            continue
        current_dt = sample_current_datetime(cfg.seed, i)
        system = format_system_prompt(current_dt)
        yield UnifiedExample(
            task_class="menon_qa",
            prompt=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            reference=response,
            reference_messages=[{"role": "assistant", "content": response}],
            source=f"{FILENAME}#{i}",
            extra={"current_datetime": current_dt},
        )
