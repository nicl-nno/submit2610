'Tests for run_eval.py — CLI, aggregation, and the full pipeline (fakes, no GPU).'

from __future__ import annotations

import argparse
import json
import random

import pytest

import run_eval as eval_mod
from run_eval import (
    _aggregate_examples,
    _normalize_task_classes,
    _resolve_endpoint_settings,
    _stats,
    build_arg_parser,
    run_evaluation,
    set_seed,
)
from llm_training.inference.generate import GenerationConfig
from llm_training.reward.config import ScoredMetrics
from llm_training.schema import UnifiedExample


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeGenerator:
    'Returns a deterministic canned completion per example index.'

    def __init__(self, text="gen", n=1):
        self.text = text
        self.n = n
        self.calls = 0

    def generate(self, examples, cfg):
        self.calls += 1
        return [[self.text] * self.n for _ in examples]


class FakeService:
    'ScoreRequest → canned ScoredMetrics, distinguishing tool-use by class.'

    def __init__(self, with_ac: bool = True):
        # When with_ac is True we pretend a judge is configured and populate
        # the AC field; False mirrors --no-answer-correctness (AC is None).
        self.with_ac = with_ac
        self.requests = []

    def score_batch(self, reqs):
        out = []
        for r in reqs:
            if r.task_class in ("opencode_gen", "tool_calls_multiturn"):
                m = ScoredMetrics(
                    semantic_similarity=0.6,
                    combined=0.5,
                    answer_correctness=0.7 if self.with_ac else None,
                    format_score=0.9,
                )
            else:
                m = ScoredMetrics(
                    semantic_similarity=0.5,
                    combined=0.5,
                    answer_correctness=0.5 if self.with_ac else None,
                )
            out.append(m)
            self.requests.append(r)
        return out


def _ex(task_class="coder", user="q", reference="r"):
    return UnifiedExample(
        task_class=task_class,
        prompt=[{"role": "system", "content": "s"}, {"role": "user", "content": user}],
        reference=reference,
        reference_messages=[{"role": "assistant", "content": reference}],
    )


def _menon_qa_ex():
    query = (
        "==========\nDOCUMENT 1\n==========\n\nAU основан в 1959.\n\n"
        "==========\nCURRENT QUESTION\n==========\n\nКогда?"
    )
    return UnifiedExample(
        task_class="menon_qa",
        prompt=[{"role": "system", "content": "sys"}, {"role": "user", "content": query}],
        reference="1959",
        reference_messages=[{"role": "assistant", "content": "1959"}],
        extra={"current_datetime": "2026-03-01 00:00:00"},
    )


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------
class TestSetSeed:
    def test_python_random_reproducible(self):
        set_seed(123)
        a = [random.random() for _ in range(5)]
        set_seed(123)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_different_seeds_differ(self):
        set_seed(1)
        a = [random.random() for _ in range(5)]
        set_seed(2)
        b = [random.random() for _ in range(5)]
        assert a != b


# ---------------------------------------------------------------------------
# task-class normalization
# ---------------------------------------------------------------------------
class TestNormalizeTaskClasses:
    def test_none_returns_all_non_network(self):
        cls = _normalize_task_classes(None)
        assert "nerel_instruct" not in cls
        assert "coder" in cls and "menon_qa" in cls
        # 27 = 28 task classes − 1 network one (nerel_instruct); includes the
        # three Meno-stand classes since r5.
        
        assert "meno_rerank" in cls and "meno_final" in cls
        assert len(cls) == 27

    def test_explicit_list(self):
        assert _normalize_task_classes(["coder", "menon_qa"]) == ["coder", "menon_qa"]

    def test_unknown_raises(self):
        with pytest.raises(SystemExit):
            _normalize_task_classes(["not_a_class"])


# ---------------------------------------------------------------------------
# stats / aggregation
# ---------------------------------------------------------------------------
class TestStats:
    def test_empty(self):
        s = _stats([])
        assert s["n"] == 0 and s["mean"] == 0.0

    def test_single(self):
        s = _stats([0.5])
        assert s["mean"] == 0.5 and s["std"] == 0.0 and s["n"] == 1

    def test_multiple(self):
        s = _stats([0.0, 1.0])
        assert s["mean"] == 0.5 and s["n"] == 2


class TestAggregate:
    def test_per_class_and_overall(self):
        rows = [
            {"task_class": "coder", "metrics": {
                "combined": 0.8, "semantic_similarity": 0.8,
                "answer_correctness": 0.8, "format_score": None,
            }},
            {"task_class": "coder", "metrics": {
                "combined": 0.6, "semantic_similarity": 0.6,
                "answer_correctness": 0.6, "format_score": None,
            }},
            {"task_class": "menon_qa", "metrics": {
                "combined": 0.9, "semantic_similarity": 0.9,
                "answer_correctness": 0.9, "format_score": None,
            }},
        ]
        agg = _aggregate_examples(rows)
        assert set(agg["per_class"]) == {"coder", "menon_qa"}
        assert agg["per_class"]["coder"]["combined"]["mean"] == 0.7
        # For non-tool classes format_score must be None in the block.
        # None.
        assert agg["per_class"]["coder"]["format_score"] is None
        assert agg["per_class"]["menon_qa"]["semantic_similarity"]["mean"] == 0.9
        assert agg["overall"]["combined"]["mean"] == pytest.approx((0.8 + 0.6 + 0.9) / 3)

    def test_metric_failures_counted_per_class(self):
        # Two coder examples: one with a failed SemSim, one with a failed AC.
        
        rows = [
            {"task_class": "coder", "metrics": {
                "combined": 0.0, "semantic_similarity": 0.0,
                "answer_correctness": 0.5, "format_score": None,
                "failed_metrics": ("semantic_similarity",),
            }},
            {"task_class": "coder", "metrics": {
                "combined": 0.5, "semantic_similarity": 0.5,
                "answer_correctness": None, "format_score": None,
                "failed_metrics": ("answer_correctness",),
            }},
        ]
        agg = _aggregate_examples(rows)
        block = agg["per_class"]["coder"]
        assert block["metric_failures"] == {
            "semantic_similarity": 1,
            "answer_correctness": 1,
            "rubrics": 0,
            "docsem_check": 0,
            "fluency": 0,
        }

    def test_metric_failures_defaulted_when_field_absent(self):
        # Old-style rows without failed_metrics → counters are zero (graceful).
        
        rows = [
            {"task_class": "coder", "metrics": {
                "combined": 0.8, "semantic_similarity": 0.8,
                "answer_correctness": 0.8, "format_score": None,
            }},
        ]
        agg = _aggregate_examples(rows)
        assert agg["per_class"]["coder"]["metric_failures"] == {
            "semantic_similarity": 0, "answer_correctness": 0,
            "rubrics": 0, "docsem_check": 0, "fluency": 0,
        }

    def test_format_score_populated_for_tool_use(self):
        rows = [
            {"task_class": "opencode_gen", "metrics": {
                "combined": 0.6, "semantic_similarity": 0.6,
                "answer_correctness": 0.5, "format_score": 0.9,
            }},
        ]
        agg = _aggregate_examples(rows)
        assert agg["per_class"]["opencode_gen"]["format_score"]["mean"] == 0.9


# ---------------------------------------------------------------------------
# run_evaluation (the pipeline)
# ---------------------------------------------------------------------------
class TestRunEvaluation:
    def test_greedy_pipeline(self):
        gen = FakeGenerator(text="answer")
        svc = FakeService()
        examples = [_ex(), _ex("math_reasoner"), _ex("coder")]
        results = run_evaluation(
            gen, svc, examples, GenerationConfig(mode="greedy"),
            batch_size=2,
        )
        assert results["n_examples"] == 3
        assert set(results["per_class"]) == {"coder", "math_reasoner"}
        # coder appears twice → its mean is the SemSim value 0.5.
        
        assert results["per_class"]["coder"]["combined"]["mean"] == 0.5
        assert results["per_class"]["coder"]["n"] == 2
        # No detailed dump by default
        assert results["examples"] == []

    def test_rag_example_uses_semsim(self):
        # The RAG path is no longer special-cased — menon_qa now gets the same
        # treatment as every other class (SemSim + AC).
        
        gen = FakeGenerator(text="1959")
        svc = FakeService()
        results = run_evaluation(
            gen, svc, [_menon_qa_ex()], GenerationConfig(mode="greedy"),
        )
        block = results["per_class"]["menon_qa"]
        assert block["combined"]["mean"] == 0.5
        assert block["semantic_similarity"]["mean"] == 0.5
        assert block["answer_correctness"]["mean"] == 0.5

    def test_tool_use_pipeline(self):
        # A tool-use example flows through with format_score populated.
        # tool-use
        gen = FakeGenerator(
            text='<tool_call>{"name": "read", "arguments": {"path": "x"}}</tool_call>'
        )
        svc = FakeService()
        examples = [_ex(task_class="opencode_gen")]
        results = run_evaluation(
            gen, svc, examples, GenerationConfig(mode="greedy"),
        )
        block = results["per_class"]["opencode_gen"]
        assert block["format_score"]["mean"] == 0.9

    def test_no_answer_correctness_path(self):
        # When the judge is disabled (FakeService with_ac=False), AC is None.
        
        gen = FakeGenerator(text="answer")
        svc = FakeService(with_ac=False)
        results = run_evaluation(
            gen, svc, [_ex()], GenerationConfig(mode="greedy"),
        )
        block = results["per_class"]["coder"]
        assert block["answer_correctness"] is None
        # SemSim still populated.  SemSim
        assert block["semantic_similarity"]["mean"] == 0.5

    def test_sampling_averages_across_samples(self):
        gen = FakeGenerator(text="ans", n=3)
        svc = FakeService()
        results = run_evaluation(
            gen, svc, [_ex()], GenerationConfig(mode="sampling", num_samples=3),
        )
        # One example with 3 samples → per_class n counts the example (1), and
        # its combined is the mean across the 3 samples.
        
        
        assert results["per_class"]["coder"]["n"] == 1
        assert results["per_class"]["coder"]["combined"]["mean"] == 0.5

    def test_detailed_dumps_samples(self):
        gen = FakeGenerator(text="ans", n=2)
        svc = FakeService()
        results = run_evaluation(
            gen, svc, [_ex()], GenerationConfig(mode="sampling", num_samples=2),
            detailed=True,
        )
        row = results["examples"][0]
        assert row["n_samples"] == 2
        assert len(row["samples"]) == 2
        assert all("text" in s and "metrics" in s for s in row["samples"])

    def test_progress_callback(self):
        gen = FakeGenerator()
        svc = FakeService()
        seen = []
        run_evaluation(
            gen, svc, [_ex(), _ex(), _ex()], GenerationConfig(),
            batch_size=2, on_progress=lambda d, t: seen.append((d, t)),
        )
        assert seen  # at least one progress tick
        assert all(t == 3 for _, t in seen)

    def test_empty_examples(self):
        gen = FakeGenerator()
        svc = FakeService()
        results = run_evaluation(gen, svc, [], GenerationConfig())
        assert results["n_examples"] == 0
        assert results["per_class"] == {}


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------
class TestArgParser:
    def test_required_args(self):
        args = build_arg_parser().parse_args(
            ["--model", "M", "--judge-base-url", "http://x/v1", "--judge-model", "J"]
        )
        assert args.model == "M"
        assert args.generation_mode == "greedy"
        assert args.seed == 42
        assert args.num_samples == 1

    def test_judge_resilience_defaults(self):
        # judge-max-tokens / judge-max-retries default to sensible values that
        # avoid the RAGAS 1024-token truncation.
        # judge-max-tokens
        
        args = build_arg_parser().parse_args(
            ["--model", "M", "--judge-base-url", "u", "--judge-model", "J"]
        )
        assert args.judge_max_tokens == 4096
        assert args.judge_max_retries == 2

    def test_judge_resilience_overridable(self):
        args = build_arg_parser().parse_args(
            ["--model", "M", "--judge-base-url", "u", "--judge-model", "J",
             "--judge-max-tokens", "8192", "--judge-max-retries", "5"]
        )
        assert args.judge_max_tokens == 8192
        assert args.judge_max_retries == 5

    def test_judge_timeout_default_none(self):
        # --judge-timeout defaults to None → the RewardConfig default (120 s)
        # applies; an explicit value covers slow cloud judges (thinking
        # models, GLM on Z.AI).
        # --judge-timeout
        
        args = build_arg_parser().parse_args(
            ["--model", "M", "--judge-base-url", "u", "--judge-model", "J"]
        )
        assert args.judge_timeout is None
        args = build_arg_parser().parse_args(
            ["--model", "M", "--judge-base-url", "u", "--judge-model", "J",
             "--judge-timeout", "600"]
        )
        assert args.judge_timeout == 600.0

    def test_embedding_model_default(self):
        # --local-embedding-model defaults to bge-m3 (in-process, no HTTP).
        # --local-embedding-model
        args = build_arg_parser().parse_args(["--model", "M"])
        assert args.local_embedding_model == "BAAI/bge-m3"

    def test_task_append(self):
        args = build_arg_parser().parse_args(
            ["--model", "M", "--judge-base-url", "u", "--judge-model", "J",
             "--task", "coder", "--task", "menon_qa"]
        )
        assert args.task == ["coder", "menon_qa"]

    def test_missing_required_exits(self):
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args([])

    def test_no_answer_correctness_flag(self):
        # The --no-answer-correctness flag must be parsed and default to False.
        
        args = build_arg_parser().parse_args(
            ["--model", "M", "--no-answer-correctness"]
        )
        assert args.no_answer_correctness is True

    def test_rag_classes_flag_removed(self):
        # The legacy --rag-classes flag must be gone (no Faithfulness / NS).
        # NS).
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(
                ["--model", "M", "--rag-classes", "menon_qa"]
            )


# ---------------------------------------------------------------------------
# Endpoint setting resolution (CLI > env)

# ---------------------------------------------------------------------------
class TestResolveEndpointSettings:
    '_resolve_endpoint_settings: judge CLI flag > env var. Embeddings are\n    in-process and have no endpoint to resolve.\n\n    _resolve_endpoint_settings'

    @staticmethod
    def _ns(**kw):
        # Minimal namespace with every endpoint setting absent by default.
        
        base = dict(judge_base_url=None, judge_model=None)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_cli_flag_wins_over_env(self):
        ns = self._ns(judge_base_url="http://cli/v1", judge_model="cli-judge")
        env = {"JUDGE_BASE_URL": "http://env/v1", "JUDGE_MODEL": "env-judge"}
        jb, jm = _resolve_endpoint_settings(ns, env, require_judge=True)
        assert jb == "http://cli/v1"
        assert jm == "cli-judge"

    def test_env_used_when_cli_absent(self):
        ns = self._ns()
        env = {"JUDGE_BASE_URL": "http://env/v1", "JUDGE_MODEL": "env-judge"}
        jb, jm = _resolve_endpoint_settings(ns, env, require_judge=True)
        assert jb == "http://env/v1"
        assert jm == "env-judge"

    def test_missing_all_required_exits(self):
        ns = self._ns()
        with pytest.raises(SystemExit):
            _resolve_endpoint_settings(ns, {}, require_judge=True)

    def test_missing_judge_model_exits(self):
        ns = self._ns(judge_base_url="http://j/v1")
        with pytest.raises(SystemExit):
            _resolve_endpoint_settings(ns, {}, require_judge=True)

    def test_no_answer_correctness_makes_judge_optional(self):
        # When require_judge=False (i.e. --no-answer-correctness), no endpoint
        # settings are required at all (embeddings are always in-process).
        
        ns = self._ns()
        jb, jm = _resolve_endpoint_settings(ns, {}, require_judge=False)
        assert jb is None
        assert jm is None


# ---------------------------------------------------------------------------
# Full main() smoke test with everything faked
# ---------------------------------------------------------------------------
class TestMainSmoke:
    def _run_main(self, tmp_path, monkeypatch, tokenizer, extra_args=()):
        # Shared smoke setup: fake every heavy dependency, run main(), return
        # the parsed report.
        
        
        examples = [_ex(task_class="coder"), _menon_qa_ex()]

        class _BR:
            def __init__(self):
                self.train = examples
                self.stats = {"coder": {"loaded": 1, "kept": 1}}
                self.tool_names = frozenset()
                self.tool_schemas = {}

        monkeypatch.setattr(eval_mod, "load_tokenizer", lambda model: tokenizer)
        monkeypatch.setattr(eval_mod, "build_unified_dataset", lambda *a, **kw: _BR())
        monkeypatch.setattr(eval_mod, "PolicyGenerator", lambda **kw: FakeGenerator(text="ans"))
        monkeypatch.setattr(eval_mod, "make_ragas_scorer", lambda cfg: object())  # unused by FakeService

        class _FakeCMService:
            def __enter__(self):
                return FakeService()

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(eval_mod, "RewardService", lambda *a, **kw: _FakeCMService())

        out = tmp_path / "report.json"
        rc = eval_mod.main(
            [
                "--model", "anonymous/base-model",
                "--judge-base-url", "http://localhost:8001/v1",
                "--judge-model", "judge-m",
                "--local-embedding-model", "BAAI/bge-m3",
                "--task", "coder", "--task", "menon_qa",
                "--seed", "7",
                "--batch-size", "4",
                "--output", str(out),
                *extra_args,
            ]
        )
        assert rc == 0
        return json.loads(out.read_text(encoding="utf-8"))

    def test_writes_report(self, tmp_path, monkeypatch, tokenizer):
        report = self._run_main(tmp_path, monkeypatch, tokenizer)
        assert report["config"]["model"] == "anonymous/base-model"
        assert report["config"]["seed"] == 7
        # New judge-resilience settings must be recorded in the report config.
        
        assert report["config"]["judge_max_tokens"] == 4096
        assert report["config"]["judge_max_retries"] == 2
        # --judge-timeout absent → the RewardConfig default (120 s).
        # --judge-timeout
        assert report["config"]["judge_timeout_s"] == 120.0
        # New: tool-names count and compute_answer_correctness flag.
        
        assert "known_tool_names_count" in report["config"]
        assert report["config"]["compute_answer_correctness"] is True
        # API keys must be masked in the report (local endpoints → "<none>"/EMPTY mask).
        
        assert "api_key" in report["config"]["judge"]
        assert set(report["metrics"]["per_class"]) == {"coder", "menon_qa"}
        assert report["metrics"]["overall"]["n"] == 2
        # metric_failures block present on every class and overall.
        
        for cls in ("coder", "menon_qa"):
            assert "metric_failures" in report["metrics"]["per_class"][cls]
        assert "metric_failures" in report["metrics"]["overall"]
        # The new metric fields must be present (faithfulness/NS are gone).
        
        for cls in ("coder", "menon_qa"):
            block = report["metrics"]["per_class"][cls]
            assert "semantic_similarity" in block
            assert "answer_correctness" in block
            assert "format_score" in block
            assert "faithfulness" not in block
            assert "noise_sensitivity" not in block

    def test_judge_timeout_wired_into_reward_config(
        self, tmp_path, monkeypatch, tokenizer,
    ):
        # An explicit --judge-timeout must flow into RewardConfig and the
        # report (slow cloud judges: thinking models, GLM on Z.AI).
        
        
        report = self._run_main(
            tmp_path, monkeypatch, tokenizer, extra_args=("--judge-timeout", "300"),
        )
        assert report["config"]["judge_timeout_s"] == 300.0
