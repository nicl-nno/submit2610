'Offline tests for lm_eval_tasks/sealqa_long.py (no network, no GPU).\n\n\n``doc_to_text``\n``contains``\n\nThe SealQALong class task depends on lm_eval (it imports\nConfigurableTask), so the structural checks run under\n``pytest.importorskip("lm_eval")``; the yaml↔class consistency is checked\ntextually in all cases. No network: ``doc_to_text`` is built from a\nfabricated ``12_docs``, the ``contains`` metric is the shared proxy from\nsealqa.py; ``download()`` (which fetches the dataset) is never called in\nthe tests.'

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

LM_EVAL_TASKS_DIR = Path(__file__).resolve().parents[2] / "lm_eval_tasks"
YAML_PATH = LM_EVAL_TASKS_DIR / "sealqa_long.yaml"


def _make_doc() -> dict:
    'A fabricated longseal row: 12_docs as a list of {date, text} dicts\n    (the pyarrow struct arrives as a dict; date may be null).'
    return {
        "question": "Who holds the all-time Grammy album-of-the-year record?",
        "answer": "Serban Ghenea",
        "12_docs": [
            {"date": "Mar 01, 2006", "text": "Document body A."},
            {"date": None, "text": "Document body B."},
            {"date": "", "text": "Document body C."},
        ],
    }


class TestYamlConsistency:
    def test_yaml_declares_the_class(self):
        text = YAML_PATH.read_text(encoding="utf-8")
        assert re.search(r"^task:\s*sealqa_long$", text, re.MULTILINE)
        assert re.search(r"^class:\s*!function\s+sealqa_long\.SealQALong$", text, re.MULTILINE)


@pytest.fixture(scope="module")
def sq_long():
    'The sealqa_long module imported BY NAME (as the yaml ``class`` hook\n    resolves it).'
    pytest.importorskip("lm_eval")
    if str(LM_EVAL_TASKS_DIR) not in sys.path:
        sys.path.insert(0, str(LM_EVAL_TASKS_DIR))
    return importlib.import_module("sealqa_long")


class TestSealQALong:
    
    
    
    # SealQALong() must not be instantiated in the tests:
    # ConfigurableTask.__init__ immediately calls self.download() (network + the
    # cluster tokenizer path). Hence the self-free methods are called unbound
    # (self=None).

    def test_doc_to_text_layout(self, sq_long):
        text = sq_long.SealQALong.doc_to_text(None, _make_doc())
        assert text.startswith("Answer the question using only the provided documents.")
        assert "Document 1: [Mar 01, 2006]\nDocument body A." in text
        # a null date yields no date header
        assert "Document 2:\nDocument body B." in text
        assert "Document 3:\nDocument body C." in text
        assert text.endswith("Question: Who holds the all-time Grammy album-of-the-year record?")

    def test_doc_to_target_is_the_answer(self, sq_long):
        assert sq_long.SealQALong.doc_to_target(None, _make_doc()).strip() == "Serban Ghenea"

    def test_process_results_contains(self, sq_long):
        assert sq_long.SealQALong.process_results(None, _make_doc(), ["Serban Ghenea"]) == {"contains": 1.0}
        assert sq_long.SealQALong.process_results(None, _make_doc(), ["Taylor Swift"]) == {"contains": 0.0}

    def test_aggregation_and_higher_is_better(self, sq_long):
        assert list(sq_long.SealQALong.aggregation(None).keys()) == ["contains"]
        assert sq_long.SealQALong.higher_is_better(None) == {"contains": True}

    def test_split_routing_is_test_only(self, sq_long):
        assert sq_long.SealQALong.has_training_docs(None) is False
        assert sq_long.SealQALong.has_validation_docs(None) is False
        assert sq_long.SealQALong.has_test_docs(None) is True

    def test_test_docs_returns_the_single_split_dataset(self, sq_long):
        
        
        # test_docs
        
        
        # -> test_docs
        
        
        
        # The regression of the 2026-09-28 cluster crash: download() loads
        # a SINGLE split (split="test" => Dataset, not a DatasetDict)
        # while test_docs() called self.dataset["test"] — on a Dataset
        # that is COLUMN access and no such column exists (KeyError:
        # Column test not in the dataset, crashing at task creation:
        # __init__ -> fewshot_docs -> test_docs). The correct convention:
        # the single split is returned as-is. The instance is built via
        # object.__new__ (bypassing __init__ with its network download()
        # and the cluster tokenizer path), the dataset is a local
        # Dataset.from_list with the longseal columns (no network).
        datasets = pytest.importorskip("datasets")
        task = object.__new__(sq_long.SealQALong)
        task.dataset = datasets.Dataset.from_list(
            [
                {
                    "question": "Who holds the record?",
                    "answer": "Serban Ghenea",
                    "12_docs": [{"date": None, "text": "Document body."}],
                }
            ]
        )
        result = task.test_docs()
        assert result is task.dataset
        assert len(result) == 1
        assert result[0]["answer"] == "Serban Ghenea"

    def test_prune_budget_is_below_the_window(self, sq_long):
        
        # The budget must leave room for the chat template and the generation
        # (max_gen_toks 512).
        assert sq_long._PRUNE_MAX_TOKENS == 32000

    def test_metric_is_the_shared_proxy(self, sq_long):
        
        # The LONGSEAL metric is the same deterministic proxy as seal_0/hard.
        import sealqa

        assert sq_long.relaxed_contains_score is sealqa.relaxed_contains_score

    def test_construct_requests_absorbs_lm_eval_kwargs(self, sq_long):
        # build_all_requests lm-eval
        # chat_template
        
        # drift_probe 2026-09-27; bench
        # --apply_chat_template).
        # lm-eval's build_all_requests ALWAYS passes apply_chat_template /
        # chat_template (even False) + metadata; the override must absorb
        # them — otherwise Instance raises TypeError (the same crash as in
        # drift_probe, 2026-09-27; the bench sealqa group runs with
        # --apply_chat_template).
        for apply_chat_template in (True, False):
            instance = sq_long.SealQALong.construct_requests(
                None,
                _make_doc(),
                "ctx",
                metadata=("sealqa_long", 0, 1),
                apply_chat_template=apply_chat_template,
                chat_template=None,
            )
            assert instance.request_type == "generate_until"
            assert instance.arguments[0] == "ctx"
            assert instance.arguments[1]["max_gen_toks"] == 512
            assert "<|im_end|>" in instance.arguments[1]["until"]
            assert instance.metadata == ("sealqa_long", 0, 1)
