"Loader: RAGU ICL → 6 task classes (vendored templates, no RAGU dependency).\n\n\nEach of the 6 RAGU ICL example files produces one task class. The dataset is\n**not** built by RAGU's ``InContextLearningManager`` (which only *selects*\nexamples via embedder/BM25); instead we reuse RAGU's exact wrapping logic —\nthe system template strings and the few-shot formatters — vendored in\n:mod:`llm_training.loaders._ragu_templates`. All examples are kept regardless of\n``quality_rating`` and regardless of language (both ``english`` and ``russian``)."

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterator

from llm_training.config import DataConfig
from llm_training.loaders._ragu_templates import (
    DEFAULT_ARTIFACTS_EXTRACTOR_SYSTEM,
    DEFAULT_ARTIFACTS_VALIDATOR_SYSTEM,
    DEFAULT_TWO_STAGE_ENTITIES_EXTRACTOR_SYSTEM,
    DEFAULT_TWO_STAGE_ENTITIES_VALIDATOR_SYSTEM,
    DEFAULT_TWO_STAGE_RELATIONS_EXTRACTOR_SYSTEM,
    DEFAULT_TWO_STAGE_RELATIONS_VALIDATOR_SYSTEM,
    format_artifact_extraction,
    format_artifact_validation,
    format_entity_extraction,
    format_entity_validation,
    format_relation_extraction,
    format_relation_validation,
    render_system,
)
from llm_training.loaders.base import read_json
from llm_training.schema import UnifiedExample

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer

CLASSES = (
    "ragu_artifact_extraction",
    "ragu_artifact_validation",
    "ragu_entity_extraction",
    "ragu_entity_validation",
    "ragu_relation_extraction",
    "ragu_relation_validation",
)
SUBDIR = "ragu_icl"

#: task_class -> (json filename, system template, formatter)
# : task_class ->
_SPECS: dict[str, tuple[str, str, Callable[[dict[str, Any]], tuple[str, str]]]] = {
    "ragu_artifact_extraction": (
        "artifact_extraction_examples.json",
        DEFAULT_ARTIFACTS_EXTRACTOR_SYSTEM,
        format_artifact_extraction,
    ),
    "ragu_artifact_validation": (
        "artifact_validation_examples.json",
        DEFAULT_ARTIFACTS_VALIDATOR_SYSTEM,
        format_artifact_validation,
    ),
    "ragu_entity_extraction": (
        "entity_extraction_examples.json",
        DEFAULT_TWO_STAGE_ENTITIES_EXTRACTOR_SYSTEM,
        format_entity_extraction,
    ),
    "ragu_entity_validation": (
        "entity_validation_examples.json",
        DEFAULT_TWO_STAGE_ENTITIES_VALIDATOR_SYSTEM,
        format_entity_validation,
    ),
    "ragu_relation_extraction": (
        "relation_extraction_examples.json",
        DEFAULT_TWO_STAGE_RELATIONS_EXTRACTOR_SYSTEM,
        format_relation_extraction,
    ),
    "ragu_relation_validation": (
        "relation_validation_examples.json",
        DEFAULT_TWO_STAGE_RELATIONS_VALIDATOR_SYSTEM,
        format_relation_validation,
    ),
}


def load_class(
    task_class: str, tokenizer: "PreTrainedTokenizer", cfg: DataConfig
) -> Iterator[UnifiedExample]:
    'Yield examples for one RAGU ICL task class.'
    filename, system_template, formatter = _SPECS[task_class]
    path = cfg.data_dir / SUBDIR / filename
    data = read_json(path)
    examples = data.get("examples", []) or []
    for i, ex in enumerate(examples):
        input_text = ex.get("input_text") or ""
        output = ex.get("output") or {}
        if not input_text or not output:
            continue
        language = (ex.get("metadata") or {}).get("language", "english") or "english"
        system = render_system(system_template, language=language)
        user, assistant = formatter(ex)
        yield UnifiedExample(
            task_class=task_class,
            prompt=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            reference=assistant,
            reference_messages=[{"role": "assistant", "content": assistant}],
            source=f"{SUBDIR}/{filename}#{i}",
            extra={
                "language": language,
                "quality_rating": ex.get("quality_rating"),
                "id": ex.get("id", ""),
            },
        )


def load(tokenizer: "PreTrainedTokenizer", cfg: DataConfig) -> Iterator[UnifiedExample]:
    'Yield examples for all 6 RAGU ICL task classes.'
    for task_class in CLASSES:
        yield from load_class(task_class, tokenizer, cfg)
