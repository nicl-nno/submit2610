'Offline tests for lm_eval_tasks/drift_probe.py (no network, no GPU).\n\n\n``check_response``\n``ConfigurableTask.__init__``\n\nThe key check is the EQUIVALENCE of the ported instruction checks to the\noriginal implementation ``llm_training/runtime/drift_probe.py``: both\n``check_response`` functions must return identical verdicts on all 76 records\nof the frozen ``src_data/drift_probe.jsonl`` set and on a battery of\nsynthetic responses (including edge cases: an empty response, JSON in\nmarkdown fences, an unknown check type — the closed-world False failure).\nThe task-class checks run only when lm_eval is importable (the\ntest_sealqa_long.py pattern): no instantiation, since\n``ConfigurableTask.__init__`` calls ``download()``. The wiring consistency\n(merge_evolve.yaml, merge_evolve_slurm.sh) is verified textually.'

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LM_EVAL_TASKS_DIR = REPO_ROOT / "lm_eval_tasks"
PROBE_JSONL = REPO_ROOT / "src_data" / "drift_probe.jsonl"
TASK_PY = LM_EVAL_TASKS_DIR / "drift_probe.py"
TASK_YAML = LM_EVAL_TASKS_DIR / "drift_probe.yaml"
MERGE_EVOLVE_YAML = REPO_ROOT / "merge_evolve.yaml"
MERGE_EVOLVE_SH = REPO_ROOT / "merge_evolve_slurm.sh"



#: The response battery: empty, short RU/EN, JSON with and without fences,
#: a long repetitive text.
_RESPONSES = [
    "",
    "Короткий ответ.",
    "Однажды в студёную зимнюю пору я из лесу вышел; был сильный мороз, "
    "и снег хрустел под ногами, а вдаль уходила дорога через поле.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    '{"title": "Отчёт", "pages": 12}',
    '```json\n{"title": "Report", "pages": 7}\n```',
    "слово " * 45,
    "Ответ на русском языке без запятых и прочих особенных символов",
]


def _craft_pass_response(check: dict) -> str:
    'A response satisfying the check by construction (for the positive\n    branch).'
    ctype = check.get("type")
    if ctype == "min_words":
        return "слово " * int(check["n"])
    if ctype == "max_words":
        return "слово"
    if ctype == "exact_words":
        return " ".join(["слово"] * int(check["n"]))
    if ctype == "no_commas":
        return "текст без запятых"
    if ctype == "json_keys":
        return json.dumps({k: 1 for k in check["keys"]}, ensure_ascii=False)
    if ctype == "keyword_min":
        return " ".join([str(check["keyword"])] * int(check["n"]))
    if ctype == "keyword_max":
        return str(check["keyword"]) if int(check["n"]) >= 1 else ""
    if ctype == "language_ru":
        return "Это русский текст о погоде и городах."
    if ctype == "language_en":
        return "This is plain English text about weather."
    return ""


@pytest.fixture(scope="module")
def dp():
    'The task module imported BY NAME (as the yaml ``class`` hook resolves\n    it).'
    if str(LM_EVAL_TASKS_DIR) not in sys.path:
        sys.path.insert(0, str(LM_EVAL_TASKS_DIR))
    return importlib.import_module("drift_probe")


@pytest.fixture(scope="module")
def runtime_probe():
    'The training-side original implementation (the equivalence reference).'
    return pytest.importorskip("llm_training.runtime.drift_probe")


@pytest.fixture(scope="module")
def records() -> list[dict]:
    with open(PROBE_JSONL, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class TestData:
    def test_frozen_set_shape(self, records):
        
        # 76 records of the frozen set (the number pins the criterion noise).
        assert len(records) == 76
        assert len({r["id"] for r in records}) == len(records)

    def test_record_schema(self, records, dp):
        for rec in records:
            assert rec["prompt"].strip()
            assert rec["lang"] in ("ru", "en")
            assert rec["check"]["type"] in dp.CHECK_TYPES

    def test_check_types_match_the_runtime(self, dp, runtime_probe):
        assert dp.CHECK_TYPES == runtime_probe.CHECK_TYPES

    def test_runtime_loader_agrees(self, records, runtime_probe):
        items = runtime_probe.load_probe(PROBE_JSONL)
        assert [i.id for i in items] == [r["id"] for r in records]
        assert [i.prompt for i in items] == [r["prompt"] for r in records]


class TestEquivalence:
    'The ported checks against the original — the core of the\n    self-containment guarantee.'

    def test_battery_over_all_records(self, dp, runtime_probe, records):
        for rec in records:
            for text in _RESPONSES + [_craft_pass_response(rec["check"])]:
                assert dp.check_response(rec["check"], text) == (
                    runtime_probe.check_response(rec["check"], text)
                ), (rec["id"], rec["check"], text[:50])

    def test_unknown_type_fails_closed_on_both(self, dp, runtime_probe):
        check = {"type": "no_such_check", "n": 1}
        assert dp.check_response(check, "любой текст") is False
        assert runtime_probe.check_response(check, "любой текст") is False

    def test_json_fences_stripped(self, dp):
        check = {"type": "json_keys", "keys": ["a", "b"]}
        assert dp.check_response(check, '```json\n{"a": 1, "b": 2}\n```') is True
        assert dp.check_response(check, '{"a": 1}') is False

    def test_language_checks(self, dp):
        ru = {"type": "language_ru"}
        en = {"type": "language_en"}
        assert dp.check_response(ru, "Обычный русский текст") is True
        assert dp.check_response(ru, "plain english text") is False
        assert dp.check_response(en, "plain english text") is True
        assert dp.check_response(en, "русский текст") is False

    def test_word_counting_ignores_punctuation_tokens(self, dp):
        
        # "—" and "…" are not words (≥1 alphanumeric character).
        check = {"type": "exact_words", "n": 3}
        assert dp.check_response(check, "Слово — другое… третье!") is True


class TestSelfContainment:
    def test_no_llm_training_imports(self):
        source = TASK_PY.read_text(encoding="utf-8")
        assert not re.search(
            r"^\s*(import llm_training|from llm_training)",
            source,
            re.MULTILINE,
        ), "Ray-акторы не имеют пакета проекта в sys.path / the Ray actors lack the project package on sys.path"

    def test_data_resolved_relative_to_module(self):
        source = TASK_PY.read_text(encoding="utf-8")
        assert "src_data" in source and "drift_probe.jsonl" in source


class TestTaskClass:
    'The structural class checks (lm_eval required; no instantiation).'

    @pytest.fixture(autouse=True)
    def _require_lm_eval(self, dp):
        pytest.importorskip("lm_eval")
        assert hasattr(dp, "DriftProbeTask")

    def test_yaml_declares_the_class(self):
        text = TASK_YAML.read_text(encoding="utf-8")
        assert re.search(r"^task:\s*drift_probe$", text, re.MULTILINE)
        assert re.search(
            r"^class:\s*!function\s+drift_probe\.DriftProbeTask$", text, re.MULTILINE
        )

    def test_doc_to_text_is_the_prompt_verbatim(self, dp):
        doc = {"prompt": "Напишите текст не короче 20 слов.", "check": {}, "lang": "ru"}
        assert dp.DriftProbeTask.doc_to_text(None, doc) == doc["prompt"]

    def test_doc_to_target_is_empty(self, dp):
        
        # There is no gold string: the checks are parametric.
        assert dp.DriftProbeTask.doc_to_target(None, {"check": {}}) == ""

    def test_process_results_pass_and_fail(self, dp):
        check = {"type": "min_words", "n": 3}
        ok = dp.DriftProbeTask.process_results(None, {"check": check}, ["раз два три"])
        bad = dp.DriftProbeTask.process_results(None, {"check": check}, ["мало"])
        assert ok == {"probe_acc": 1.0}
        assert bad == {"probe_acc": 0.0}

    def test_empty_results_fail(self, dp):
        check = {"type": "min_words", "n": 1}
        out = dp.DriftProbeTask.process_results(None, {"check": check}, [])
        assert out == {"probe_acc": 0.0}

    def test_aggregation_and_higher_is_better(self, dp):
        assert list(dp.DriftProbeTask.aggregation(None).keys()) == ["probe_acc"]
        assert dp.DriftProbeTask.higher_is_better(None) == {"probe_acc": True}

    def test_split_routing_is_test_only(self, dp):
        assert dp.DriftProbeTask.has_training_docs(None) is False
        assert dp.DriftProbeTask.has_validation_docs(None) is False
        assert dp.DriftProbeTask.has_test_docs(None) is True

    def test_generation_budget_matches_the_anchor_side(self, dp):
        
        
        # 256 tokens = the anchor-side max_new_tokens; the stop strings are
        # the Meno generation-terminating tokens.
        instance = dp.DriftProbeTask.construct_requests(
            None, {"prompt": "x", "check": {}}, "ctx"
        )
        kwargs = instance.arguments[1]
        assert kwargs["max_gen_toks"] == 256
        assert "<|im_end|>" in kwargs["until"]

    def test_construct_requests_absorbs_lm_eval_kwargs(self, dp):
        # build_all_requests lm-eval
        # chat_template
        
        # CMA-ES 2026-09-27
        
        # lm-eval's build_all_requests ALWAYS passes apply_chat_template /
        # chat_template (even False) + metadata; the override must absorb
        # them — otherwise Instance raises TypeError (the first-CMA-ES
        # population crash of 2026-09-27). The chat template is applied
        # upstream (fewshot_context), so the asserts only check that the
        # call survives and metadata flows through.
        for apply_chat_template in (True, False):
            instance = dp.DriftProbeTask.construct_requests(
                None,
                {"prompt": "x", "check": {}},
                "ctx",
                metadata=("drift_probe", 0, 1),
                apply_chat_template=apply_chat_template,
                chat_template=None,
            )
            assert instance.request_type == "generate_until"
            assert instance.arguments[0] == "ctx"
            assert instance.arguments[1]["max_gen_toks"] == 256
            assert "<|im_end|>" in instance.arguments[1]["until"]
            assert instance.metadata == ("drift_probe", 0, 1)


class TestWiring:
    'The consistency of the evolution configuration and the SLURM wrapper\n    (textual).'

    def test_merge_evolve_yaml_declares_the_task(self):
        text = MERGE_EVOLVE_YAML.read_text(encoding="utf-8")
        assert re.search(r"-\s*name:\s*drift_probe\b", text)
        
        
        # The entry block — up to the next task or a top-level key (the
        # bilingual comment is long).
        tail = text.split("name: drift_probe", 1)[1]
        block = tail.split("\n- name:", 1)[0].split("\nlimit:", 1)[0]
        assert re.search(r"weight:\s*0\.5\b", block)
        assert re.search(r'metric:\s*"probe_acc,none"', block)

    def test_slurm_tasks_and_smoke_include_it(self):
        text = MERGE_EVOLVE_SH.read_text(encoding="utf-8")
        assert re.search(r"^EVOLVE_TASKS=.*\bdrift_probe\b", text, re.MULTILINE)
        assert '"drift_probe": "probe_acc,none"' in text
