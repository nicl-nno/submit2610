'Offline tests for lm_eval_tasks/sealqa.py (no network, no GPU).\n\n\nlm_eval\n\nThe module lives outside the package — imported by name after the task\ndirectory is prepended to sys.path (the same convention as\ntests/test_lm_eval_tasks/test_hotpotqa.py: this is also how the ``!function``\nhooks of sealqa_*.yaml see it). No network: the normalization, the\n``contains`` proxy metric and the prompt building are checked on fabricated\ndocuments; the yaml↔module consistency textually, and, when lm_eval is\nimportable, through the TaskManager index (without instantiating the task:\nthat downloads data).'

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

LM_EVAL_TASKS_DIR = Path(__file__).resolve().parents[2] / "lm_eval_tasks"


@pytest.fixture(scope="module")
def sq():
    'The sealqa module imported BY NAME (as the yaml ``!function`` hooks\n    resolve it).'
    if str(LM_EVAL_TASKS_DIR) not in sys.path:
        sys.path.insert(0, str(LM_EVAL_TASKS_DIR))
    return importlib.import_module("sealqa")


def _make_doc() -> dict:
    'A fabricated SealQA row (the parquet fields the scorer reads).'
    return {"question": "How many NBA players scored 60+ points since 2023?", "answer": "12 players"}


class TestNormalizeAnswer:
    def test_lowercases_and_collapses_whitespace(self, sq):
        assert sq.normalize_answer("  12  Players  ") == "12 players"

    def test_strips_punctuation(self, sq):
        assert sq.normalize_answer("SpaceX, Inc.!") == "spacex inc"

    def test_removes_english_articles(self, sq):
        assert sq.normalize_answer("The Beatles") == "beatles"

    def test_number_survives(self, sq):
        assert sq.normalize_answer(" 378 ") == "378"


class TestRelaxedContains:
    def test_exact_answer_is_contained(self, sq):
        assert sq.relaxed_contains_score("12 players", "12 players") == 1.0

    def test_gold_is_substring_of_longer_answer(self, sq):
        
        # the relaxed protocol: the gold inside an explanatory answer counts
        assert sq.relaxed_contains_score("There are 12 players in total.", "12 players") == 1.0

    def test_normalization_applies_before_match(self, sq):
        assert sq.relaxed_contains_score("  SpaceX! ", "spacex") == 1.0

    def test_wrong_answer_is_zero(self, sq):
        assert sq.relaxed_contains_score("Tesla", "SpaceX") == 0.0

    def test_partial_answer_is_zero(self, sq):
        
        # the gold "12 players" is not inside the prediction "12"
        assert sq.relaxed_contains_score("12", "12 players") == 0.0

    def test_empty_prediction_is_zero(self, sq):
        assert sq.relaxed_contains_score("", "12 players") == 0.0

    def test_empty_gold_defensive(self, sq):
        assert sq.relaxed_contains_score("", "") == 1.0
        assert sq.relaxed_contains_score("anything", "") == 0.0


class TestPromptBuilding:
    def test_doc_to_text_layout(self, sq):
        text = sq.doc_to_text(_make_doc())
        assert text.startswith("Answer the question with the exact answer")
        assert text.endswith("Question: How many NBA players scored 60+ points since 2023?")

    def test_doc_to_target_is_the_answer_field(self, sq):
        assert sq.doc_to_target(_make_doc()).strip() == "12 players"


class TestProcessResults:
    def test_contains_one(self, sq):
        out = sq.process_results(_make_doc(), ["12 players"])
        assert out == {"contains": 1.0}

    def test_contains_zero(self, sq):
        out = sq.process_results(_make_doc(), ["9 players"])
        assert out == {"contains": 0.0}

    def test_empty_results_defensive(self, sq):
        out = sq.process_results(_make_doc(), [])
        assert out == {"contains": 0.0}


class TestYamlConsistency:
    @pytest.mark.parametrize(
        "task,dataset_name",
        [("sealqa_0", "seal_0"), ("sealqa_hard", "seal_hard")],
    )
    def test_yaml_fields(self, task, dataset_name):
        text = (LM_EVAL_TASKS_DIR / f"{task}.yaml").read_text(encoding="utf-8")
        for expected in (
            rf"^task:\s*{task}$",
            r"^dataset_path:\s*vtllms/sealqa$",
            rf"^dataset_name:\s*{dataset_name}$",
            r"^test_split:\s*test$",
            r"^output_type:\s*generate_until$",
            r"doc_to_text:\s*!function\s+sealqa\.doc_to_text",
            r"doc_to_target:\s*!function\s+sealqa\.doc_to_target",
            r"process_results:\s*!function\s+sealqa\.process_results",
            r"metric:\s*contains$",
            r"aggregation:\s*mean$",
        ):
            assert re.search(expected, text, re.MULTILINE), expected

    def test_yaml_generation_is_greedy(self):
        for task in ("sealqa_0", "sealqa_hard"):
            text = (LM_EVAL_TASKS_DIR / f"{task}.yaml").read_text(encoding="utf-8")
            assert "do_sample: false" in text
            assert "temperature: 0.0" in text

    def test_yaml_pins_dataset_revision(self):
        
        # SealQA is dynamic — the revision must be pinned.
        for task in ("sealqa_0", "sealqa_hard"):
            text = (LM_EVAL_TASKS_DIR / f"{task}.yaml").read_text(encoding="utf-8")
            assert re.search(r"^\s*revision:\s*[0-9a-f]{40}$", text, re.MULTILINE)

    def test_yaml_has_no_filter_list(self):
        
        # fda/swde
        # Without a filter_list the metric key is contains,none (uniform with
        # fda/swde, which merge_evolve.yaml and the SMOKE gate expect).
        for task in ("sealqa_0", "sealqa_hard"):
            text = (LM_EVAL_TASKS_DIR / f"{task}.yaml").read_text(encoding="utf-8")
            assert "filter_list" not in text

    def test_module_exports_the_hooked_callables(self, sq):
        for name in ("doc_to_text", "doc_to_target", "process_results", "normalize_answer", "relaxed_contains_score"):
            assert callable(getattr(sq, name)), name

    def test_tasks_registered_via_include_path(self):
        pytest.importorskip("lm_eval")
        from lm_eval.tasks import TaskManager

        tm = TaskManager(include_path=str(LM_EVAL_TASKS_DIR))
        assert "sealqa_0" in tm.all_tasks
        assert "sealqa_hard" in tm.all_tasks
