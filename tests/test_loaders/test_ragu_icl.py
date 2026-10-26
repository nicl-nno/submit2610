'Tests for the RAGU ICL loader (6 task classes, vendored templates).'

from __future__ import annotations

import json

import pytest

from llm_training.config import DataConfig
from llm_training.loaders import ragu_icl
from llm_training.masking import mask_example

RAGU_CLASSES = ragu_icl.CLASSES  # 6 classes


def _cfg(data_dir) -> DataConfig:
    return DataConfig(data_dir=data_dir)


@pytest.mark.parametrize("cls", RAGU_CLASSES)
def test_each_class_yields_examples(tokenizer, data_dir, cls):
    examples = list(ragu_icl.load_class(cls, tokenizer, _cfg(data_dir)))
    assert len(examples) > 0, f"no examples for {cls}"
    assert all(ex.task_class == cls for ex in examples)


def test_all_examples_keep_both_languages_and_all_ratings(tokenizer, data_dir):
    # AGENTS: keep all examples regardless of quality_rating and language.
    
    all_ex = list(ragu_icl.load(tokenizer, _cfg(data_dir)))
    languages = {ex.extra["language"] for ex in all_ex}
    assert "english" in languages and "russian" in languages
    ratings = {ex.extra["quality_rating"] for ex in all_ex}
    assert None in ratings or any(r is not None for r in ratings)


@pytest.mark.parametrize("cls", RAGU_CLASSES)
def test_reference_is_valid_json(cls, tokenizer, data_dir):
    ex = next(iter(ragu_icl.load_class(cls, tokenizer, _cfg(data_dir))))
    # The assistant reference is the serialized JSON output.
    
    parsed = json.loads(ex.reference)
    assert isinstance(parsed, dict)


@pytest.mark.parametrize("cls", RAGU_CLASSES)
def test_system_prompt_contains_json_instruction(tokenizer, data_dir, cls):
    ex = next(iter(ragu_icl.load_class(cls, tokenizer, _cfg(data_dir))))
    system = ex.prompt[0]["content"]
    assert "Return the result as valid JSON" in system
    # Free-type inference branch: entity/relation type list NOT injected.
    
    assert "You MUST use the following predefined" not in system


@pytest.mark.parametrize("cls", RAGU_CLASSES)
def test_system_prompt_language_token(tokenizer, data_dir, cls):
    ex = next(iter(ragu_icl.load_class(cls, tokenizer, _cfg(data_dir))))
    lang = ex.extra["language"]
    assert f"Provide the answer in the following language: {lang}" in ex.prompt[0]["content"]


@pytest.mark.parametrize("cls", RAGU_CLASSES)
def test_tokenization_and_masking(tokenizer, data_dir, cls):
    ex = next(iter(ragu_icl.load_class(cls, tokenizer, _cfg(data_dir))))
    res = mask_example(ex, tokenizer)
    assert res.prefix_ok, f"prefix failed for {cls}"
    assert len(res.reference_ids) > 0
    # The reference decodes to the JSON answer.
    decoded = tokenizer.decode(res.reference_ids, skip_special_tokens=False)
    assert ex.reference.strip() in decoded or ex.reference[:40] in decoded
    # RAGU classes are not tool-use -> no tool mask.
    
    assert res.tool_token_mask is None


def test_entity_vs_relation_extraction_outputs_differ(tokenizer, data_dir):
    # entity_extraction narrows output to {entities}; relation_extraction to
    # {relations}. Verify the reference JSON reflects that.
    ent = next(iter(ragu_icl.load_class("ragu_entity_extraction", tokenizer, _cfg(data_dir))))
    rel = next(iter(ragu_icl.load_class("ragu_relation_extraction", tokenizer, _cfg(data_dir))))
    ent_out = json.loads(ent.reference)
    rel_out = json.loads(rel.reference)
    assert "entities" in ent_out and "relations" not in ent_out
    assert "relations" in rel_out and "entities" not in rel_out


def test_validation_references_contain_only_their_slice(tokenizer, data_dir):
    # D3: entity_validation's reference must NOT carry relations (the user was
    # only shown entities); relation_validation's must NOT carry entities.
    # The raw source output has both keys — the formatter must narrow it.
    
    ent_val = next(iter(ragu_icl.load_class(
        "ragu_entity_validation", tokenizer, _cfg(data_dir))))
    rel_val = next(iter(ragu_icl.load_class(
        "ragu_relation_validation", tokenizer, _cfg(data_dir))))
    ent_out = json.loads(ent_val.reference)
    rel_out = json.loads(rel_val.reference)
    assert "entities" in ent_out and "relations" not in ent_out
    assert "relations" in rel_out and "entities" not in rel_out


def test_relation_system_prompts_have_no_provided_entity_list(tokenizer, data_dir):
    # D1/D2: the relation system prompts were simplified to drop the "provided
    # list of entities" language (the formatters never put an entity list in the
    # user prompt). Pin this so the upstream phrasing is not "restored".
    
    
    rel_ext = next(iter(ragu_icl.load_class(
        "ragu_relation_extraction", tokenizer, _cfg(data_dir))))
    rel_val = next(iter(ragu_icl.load_class(
        "ragu_relation_validation", tokenizer, _cfg(data_dir))))
    for system in (rel_ext.prompt[0]["content"], rel_val.prompt[0]["content"]):
        assert "provided entity list" not in system.lower()
        assert "list of entities" not in system.lower()
