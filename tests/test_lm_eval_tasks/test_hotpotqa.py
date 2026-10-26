'Offline tests for lm_eval_tasks/hotpotqa.py (no network, no GPU).\n\n\n\nThe module lives outside the package — imported by name after the task\ndirectory is prepended to sys.path (the same convention as the scrolls\ntests in ``tests/test_lm_eval_tasks``: this is also how the ``!function``\nhooks of hotpotqa.yaml see it). No network: the EM/F1 metrics and the\nprompt building are checked on fabricated documents, the yaml↔module\nconsistency textually, and, when lm_eval is importable, through the\nTaskManager index (without instantiating the task: that downloads data).'

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

LM_EVAL_TASKS_DIR = Path(__file__).resolve().parents[2] / "lm_eval_tasks"
YAML_PATH = LM_EVAL_TASKS_DIR / "hotpotqa.yaml"


@pytest.fixture(scope="module")
def hq():
    'The hotpotqa module imported BY NAME (as the yaml ``!function`` hooks\n    resolve it).'
    if str(LM_EVAL_TASKS_DIR) not in sys.path:
        sys.path.insert(0, str(LM_EVAL_TASKS_DIR))
    return importlib.import_module("hotpotqa")


def _make_doc() -> dict:
    'A fabricated distractor-mode row (the pyarrow struct arrives as a dict).'
    return {
        "id": "fab-0",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "context": {
            "title": ["France", "Germany"],
            "sentences": [
                ["France is a country. ", "Its capital is Paris."],
                ["Germany is a country. ", "Berlin is its capital."],
            ],
        },
    }


class TestNormalizeAnswer:
    def test_lowercases_and_collapses_whitespace(self, hq):
        assert hq.normalize_answer("  PaRis  ") == "paris"

    def test_strips_punctuation(self, hq):
        assert hq.normalize_answer("Paris, France!") == "paris france"

    def test_removes_english_articles(self, hq):
        assert hq.normalize_answer("The Eiffel Tower") == "eiffel tower"
        assert hq.normalize_answer("an apple a day") == "apple day"

    def test_yes_no_survives_as_plain_words(self, hq):
        assert hq.normalize_answer("Yes") == "yes"


class TestExactMatch:
    def test_equal_modulo_normalization(self, hq):
        assert hq.exact_match_score("  Paris! ", "paris") == 1.0

    def test_different(self, hq):
        assert hq.exact_match_score("London", "Paris") == 0.0

    def test_article_difference_is_not_a_miss(self, hq):
        assert hq.exact_match_score("the Louvre", "Louvre") == 1.0


class TestF1:
    def test_identical_is_one(self, hq):
        assert hq.f1_score("Paris", "Paris") == 1.0

    def test_partial_overlap(self, hq):
        # pred {paris, france} vs gold {paris}: p=1/2, r=1 => f1=2/3
        value = hq.f1_score("Stanford University", "Stanford")
        assert value == pytest.approx(2.0 / 3.0)

    def test_disjoint_is_zero(self, hq):
        assert hq.f1_score("London", "Paris") == 0.0

    def test_empty_prediction_only_matches_empty_gold(self, hq):
        assert hq.f1_score("", "") == 1.0
        assert hq.f1_score("", "Paris") == 0.0
        assert hq.f1_score("Paris", "") == 0.0

    def test_token_bag_counts_duplicates(self, hq):
        # {cat, cat, dog} vs {cat, dog, dog}: common = 2, p = r = 2/3
        value = hq.f1_score("cat cat dog", "cat dog dog")
        assert value == pytest.approx(2.0 / 3.0)


class TestPromptBuilding:
    def test_doc_to_text_layout(self, hq):
        text = hq.doc_to_text(_make_doc())
        assert text.startswith("Read the passages")
        assert "Title: France\nFrance is a country. Its capital is Paris." in text
        assert "Title: Germany\nGermany is a country. Berlin is its capital." in text
        # paragraphs are blank-line separated
        assert "\n\nTitle: Germany" in text
        assert text.endswith("Question: What is the capital of France?")

    def test_doc_to_target_is_the_answer_field(self, hq):
        assert hq.doc_to_target(_make_doc()) == "Paris"


class TestProcessResults:
    def test_exact_answer(self, hq):
        out = hq.process_results(_make_doc(), ["Paris"])
        assert out == {"em": 1.0, "f1": 1.0}

    def test_normalized_exact_answer(self, hq):
        out = hq.process_results(_make_doc(), ["  paris! "])
        assert out == {"em": 1.0, "f1": 1.0}

    def test_partial_answer(self, hq):
        out = hq.process_results(_make_doc(), ["Paris, France"])
        assert out["em"] == 0.0
        assert 0.0 < out["f1"] < 1.0

    def test_empty_results_defensive(self, hq):
        out = hq.process_results(_make_doc(), [])
        assert out == {"em": 0.0, "f1": 0.0}


class TestYamlConsistency:
    def test_yaml_fields(self):
        text = YAML_PATH.read_text(encoding="utf-8")
        for expected in (
            r"^task:\s*hotpotqa$",
            r"^dataset_path:\s*hotpotqa/hotpot_qa$",
            r"^dataset_name:\s*distractor$",
            r"^validation_split:\s*validation$",
            r"doc_to_text:\s*!function\s+hotpotqa\.doc_to_text",
            r"doc_to_target:\s*!function\s+hotpotqa\.doc_to_target",
            r"process_results:\s*!function\s+hotpotqa\.process_results",
            r"function:\s*remove_whitespace$",
            r'function:\s*"take_first"',
            r"metric:\s*em$",
            r"metric:\s*f1$",
        ):
            assert re.search(expected, text, re.MULTILINE), expected

    def test_yaml_generation_is_greedy(self):
        text = YAML_PATH.read_text(encoding="utf-8")
        assert "do_sample: false" in text
        assert "temperature: 0.0" in text

    def test_module_exports_the_hooked_callables(self, hq):
        for name in (
            "doc_to_text",
            "doc_to_target",
            "process_results",
            "normalize_answer",
            "exact_match_score",
            "f1_score",
        ):
            assert callable(getattr(hq, name)), name

    def test_task_registered_via_include_path(self):
        pytest.importorskip("lm_eval")
        from lm_eval.tasks import TaskManager

        tm = TaskManager(include_path=str(LM_EVAL_TASKS_DIR))
        assert "hotpotqa" in tm.all_tasks
