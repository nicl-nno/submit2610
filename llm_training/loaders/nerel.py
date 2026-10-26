'Loader: NEREL_Instruct → ``nerel_instruct`` (HuggingFace, runtime only).\n\n\nNEREL_Instruct is the only source **not** stored in ``src_data/`` (too large;\nalready on HuggingFace). It is loaded via ``datasets.load_dataset`` at runtime.\nThe dataset exposes 6 configs — ``entity_definition``, ``entity_normalization``,\n``entity_recognition``, ``relation_definition``, ``relation_extraction``,\n``relation_keywords`` — all of which are **merged into the single class**\n``nerel_instruct`` (per README; the micro-configs would be meaningless as\nseparate leave-task-out splits). The split is selected via\n``DataConfig.nerel_split`` (default ``"validation"``; the GRPO training path\nuses the practically inexhaustible ``"train"`` split). Records are\n``{system, query, response}`` (same shape as the history-like sources).\nNEREL_Instruct\n``DataConfig.nerel_split``'

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from llm_training.config import DataConfig
from llm_training.loaders.base import history_to_messages
from llm_training.schema import UnifiedExample

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer

CLASSES = ("nerel_instruct",)
HF_DATASET_ID = "anonymous/nerel-instruct"
#: The default split (the eval/warm-up paths); the GRPO training path selects
#: the practically inexhaustible ``"train"`` split via ``cfg.nerel_split``.


DEFAULT_SPLIT = "validation"
CONFIGS = (
    "entity_definition",
    "entity_normalization",
    "entity_recognition",
    "relation_definition",
    "relation_extraction",
    "relation_keywords",
)


def load(tokenizer: "PreTrainedTokenizer", cfg: DataConfig) -> Iterator[UnifiedExample]:
    'Yield ``nerel_instruct`` examples (all 6 configs of ``cfg.nerel_split`` merged).\n\n    Requires network access to HuggingFace.'
    from datasets import load_dataset  # local import: not needed at module import.

    global_idx = 0
    split = getattr(cfg, "nerel_split", None) or DEFAULT_SPLIT
    for config in CONFIGS:
        ds = load_dataset(HF_DATASET_ID, config, split=split)
        for rec in ds:
            system = rec.get("system", "") or ""
            query = rec.get("query", "") or ""
            response = rec.get("response", "") or ""
            if not response:
                continue
            prompt, ref_msgs = history_to_messages(system, query, response, None)
            yield UnifiedExample(
                task_class="nerel_instruct",
                prompt=prompt,
                reference=response,
                reference_messages=ref_msgs,
                source=f"{HF_DATASET_ID}/{config}/{split}#{global_idx}",
                extra={"config": config},
            )
            global_idx += 1
