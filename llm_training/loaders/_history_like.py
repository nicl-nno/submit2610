'Shared loader for the flat ``{system, query, response, history}`` shape.\n\n\nCovers\n``nerel_instruct`` (NEREL builds records in the same shape at runtime).\n``history`` is a list of prior ``[user, assistant]`` pairs; the reference is the\n``response`` field.'

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from llm_training.config import DataConfig
from llm_training.loaders.base import history_to_messages, read_jsonl
from llm_training.schema import UnifiedExample

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer


def load_history_like(
    task_class: str,
    filename: str,
    tokenizer: "PreTrainedTokenizer",
    cfg: DataConfig,
    *,
    data_dir_name: str = "",
) -> Iterator[UnifiedExample]:
    'Yield :class:`UnifiedExample`-s from a JSONL file in the flat shape.'
    base = cfg.data_dir if not data_dir_name else cfg.data_dir / data_dir_name
    path = base / filename
    for i, rec in enumerate(read_jsonl(path)):
        system = rec.get("system", "") or ""
        query = rec.get("query", "") or ""
        response = rec.get("response", "") or ""
        history = rec.get("history") or []
        if not response:
            continue  # malformed; skip
        prompt, ref_msgs = history_to_messages(system, query, response, history)
        yield UnifiedExample(
            task_class=task_class,
            prompt=prompt,
            reference=response,
            reference_messages=ref_msgs,
            source=f"{filename}#{i}",
        )
