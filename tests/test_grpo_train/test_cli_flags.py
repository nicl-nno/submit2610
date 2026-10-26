'Tests for the grpo_train.py CLI parser (offline, module import only).\n\n\nCovers the r3 design-change flags: ``--reasoning-semsim`` (in-loop quality\nsignal for SGR reasoning turns) and ``--final-turn-share`` (final-turn boost\nof the trainable SGR classes), including defaults-off and the interaction\nwith the existing flags.'

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "grpo_train.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_train_cli", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tr = _load_module()


class TestParserDefaults:
    def test_reasoning_semsim_off_by_default(self):
        args = tr._build_arg_parser().parse_args([])
        assert args.reasoning_semsim is False

    def test_final_turn_share_off_by_default(self):
        args = tr._build_arg_parser().parse_args([])
        assert args.final_turn_share is None

    def test_max_shard_size_defaults_to_5gb(self):
        # The checkpoint discipline: shards pinned at 5 GB regardless of the
        # transformers version (v5.2+ changed the library default to 50 GB,
        # which would write one monolithic ~30 GB file).
        
        
        
        args = tr._build_arg_parser().parse_args([])
        assert args.max_shard_size == "5GB"

    def test_defaults_unchanged(self):
        # The refactor must not disturb the neighbouring knobs.
        
        args = tr._build_arg_parser().parse_args([])
        assert args.max_per_class == 200
        assert args.num_generations == 8
        assert args.temperature == pytest.approx(0.7)
        assert args.beta == 0.0
        assert args.val_task == list(tr.RECOMMENDED_VAL_TASKS)
        assert args.prepared_dir is None


class TestParserFlags:
    def test_reasoning_semsim_flag_parses(self):
        args = tr._build_arg_parser().parse_args(["--reasoning-semsim"])
        assert args.reasoning_semsim is True

    def test_final_turn_share_flag_parses(self):
        args = tr._build_arg_parser().parse_args(
            ["--final-turn-share", "0.3"]
        )
        assert args.final_turn_share == pytest.approx(0.3)

    def test_max_shard_size_flag_parses(self):
        args = tr._build_arg_parser().parse_args(
            ["--max-shard-size", "2GB"]
        )
        assert args.max_shard_size == "2GB"

    def test_flags_combine_with_existing(self):
        args = tr._build_arg_parser().parse_args(
            [
                "--num-generations", "16",
                "--grad-accum", "64",
                "--final-turn-share", "0.3",
                "--reasoning-semsim",
            ]
        )
        assert args.num_generations == 16
        assert args.grad_accum == 64
        assert args.final_turn_share == pytest.approx(0.3)
        assert args.reasoning_semsim is True


class TestValTaskFlags:
    'The merged-validation hold-out flags (repeatable --val-task).'

    def test_val_task_repeatable(self):
        args = tr._build_arg_parser().parse_args(
            ["--val-task", "summarizer", "math_reasoner"]
        )
        assert args.val_task == ["summarizer", "math_reasoner"]

    def test_test_task_flag_removed(self):
        # The separate --test-task split is gone: its data joins the val pool.
        
        with pytest.raises(SystemExit):
            tr._build_arg_parser().parse_args(["--test-task", "math_reasoner"])

    def test_prepared_dir_flag_parses(self):
        args = tr._build_arg_parser().parse_args(["--prepared-dir", "cache/grpo"])
        assert args.prepared_dir == "cache/grpo"


class TestTypoFlags:
    'The r4 keyboard-typo augmentation flags.'

    def test_typo_share_off_by_default(self):
        args = tr._build_arg_parser().parse_args([])
        assert args.typo_share == pytest.approx(0.0)
        assert args.typo_class is None
        assert args.no_typo_class is None

    def test_typo_share_parses(self):
        args = tr._build_arg_parser().parse_args(["--typo-share", "0.25"])
        assert args.typo_share == pytest.approx(0.25)

    def test_typo_classes_repeatable(self):
        args = tr._build_arg_parser().parse_args(
            [
                "--typo-class", "coder",
                "--typo-class", "menon_qa",
                "--no-typo-class", "mtrageval",
            ]
        )
        assert args.typo_class == ["coder", "menon_qa"]
        assert args.no_typo_class == ["mtrageval"]

    def test_typo_flags_combine_with_r3(self):
        args = tr._build_arg_parser().parse_args(
            [
                "--typo-share", "0.25",
                "--final-turn-share", "0.3",
                "--reasoning-semsim",
            ]
        )
        assert args.typo_share == pytest.approx(0.25)
        assert args.final_turn_share == pytest.approx(0.3)
        assert args.reasoning_semsim is True


class TestStreamFlags:
    'The balanced-stream flags (train-examples, up-sampling, network).'

    def test_epochs_flag_removed(self):
        # The old --epochs flag is gone: the training length is now given by
        # --train-examples.
        with pytest.raises(SystemExit):
            tr._build_arg_parser().parse_args(["--epochs", "3"])

    def test_train_examples_defaults_zero(self):
        args = tr._build_arg_parser().parse_args([])
        assert args.train_examples == 0
        assert args.max_upsample_ratio == pytest.approx(8.0)
        assert args.include_network is False

    def test_train_examples_and_ratio_parse(self):
        args = tr._build_arg_parser().parse_args(
            ["--train-examples", "50000", "--max-upsample-ratio", "4", "--include-network"]
        )
        assert args.train_examples == 50000
        assert args.max_upsample_ratio == pytest.approx(4.0)
        assert args.include_network is True


class TestSystemTypoFlags:
    'The instruction-noise (SYSTEM-prompt) augmentation flags.'

    def test_system_typo_share_off_by_default(self):
        args = tr._build_arg_parser().parse_args([])
        assert args.system_typo_share == pytest.approx(0.0)
        assert args.system_typo_class is None
        assert args.no_system_typo_class is None

    def test_system_typo_share_parses(self):
        args = tr._build_arg_parser().parse_args(["--system-typo-share", "0.15"])
        assert args.system_typo_share == pytest.approx(0.15)

    def test_system_typo_classes_repeatable(self):
        args = tr._build_arg_parser().parse_args(
            [
                "--system-typo-class", "nerel_instruct",
                "--system-typo-class", "menon_qa",
                "--no-system-typo-class", "coder",
            ]
        )
        assert args.system_typo_class == ["nerel_instruct", "menon_qa"]
        assert args.no_system_typo_class == ["coder"]


class TestJudgeFlags:
    'The anchor judge knobs (cloud judges incl. Z.AI GLM-5.3-Flash).'

    def test_defaults_preserve_previous_behaviour(self):
        # 8 in-flight requests / 4096 tokens / 600 s — exactly what the
        # hardcoded anchor config used before the flags existed.
        
        
        args = tr._build_arg_parser().parse_args([])
        assert args.judge_concurrency == 8
        assert args.judge_max_tokens == 4096
        assert args.judge_timeout == pytest.approx(600.0)

    def test_judge_flags_parse(self):
        # E.g. a Z.AI GLM-5.3-Flash judge: concurrency strictly below the
        # provider's 50-request limit, a raised token cap for the thinking
        # output, a longer timeout.
        
        
        args = tr._build_arg_parser().parse_args(
            [
                "--judge-concurrency", "32",
                "--judge-max-tokens", "8192",
                "--judge-timeout", "900",
            ]
        )
        assert args.judge_concurrency == 32
        assert args.judge_max_tokens == 8192
        assert args.judge_timeout == pytest.approx(900.0)

    def test_judge_preflight_on_by_default(self):
        # The pre-flight judge ping runs by default; --no-judge-preflight
        # is the explicit escape hatch.
        
        # --no-judge-preflight
        args = tr._build_arg_parser().parse_args([])
        assert args.no_judge_preflight is False
        args = tr._build_arg_parser().parse_args(["--no-judge-preflight"])
        assert args.no_judge_preflight is True

    def test_baseline_anchor_on_by_default(self):
        # The step-0 baseline anchor (the pre-training val measurement that
        # seeds best_val) runs by default whenever anchors are enabled;
        # --no-baseline-anchor is the explicit escape hatch.
        
        
        
        args = tr._build_arg_parser().parse_args([])
        assert args.no_baseline_anchor is False
        args = tr._build_arg_parser().parse_args(["--no-baseline-anchor"])
        assert args.no_baseline_anchor is True

    def test_anchor_generation_batching_flags(self):
        # The batched anchor generation: ON by default (LEFT-padded
        # packages under the KV budget, at most 8 rows each); 0 = the
        # legacy one-example-at-a-time escape.
        
        
        
        args = tr._build_arg_parser().parse_args([])
        assert args.anchor_gen_batch_tokens == 65536
        assert args.anchor_gen_max_batch == 8
        args = tr._build_arg_parser().parse_args(
            ["--anchor-gen-batch-tokens", "0", "--anchor-gen-max-batch", "2"]
        )
        assert args.anchor_gen_batch_tokens == 0
        assert args.anchor_gen_max_batch == 2

    def test_drift_probe_defaults(self):
        # The probe defaults: enabled (auto-loaded from src_data), WARN at
        # 0.05, the gate tolerance 0.10 (~2 sigma of the 76-item binomial
        # noise), the generation budget 256 tokens.
        
        
        args = tr._build_arg_parser().parse_args([])
        assert args.no_drift_probe is False
        assert args.drift_probe_warn == pytest.approx(0.05)
        assert args.drift_probe_gate == pytest.approx(0.10)
        assert args.drift_probe_max_new_tokens == 256
        args = tr._build_arg_parser().parse_args(
            ["--no-drift-probe", "--drift-probe-gate", "0", "--drift-probe-warn", "0.2"]
        )
        assert args.no_drift_probe is True
        assert args.drift_probe_gate == pytest.approx(0.0)
        assert args.drift_probe_warn == pytest.approx(0.2)

    def test_difficulty_filter_defaults(self):
        # The loss-based difficulty filter: the base Meno-Lite as the
        # separate scoring model, both tails ON in GRPO (easy 0.05 — the r2
        # zero-variance lesson — and hard 0.01 — the noise guard).
        
        
        args = tr._build_arg_parser().parse_args([])
        assert args.difficulty_model == "anonymous/base-model"
        assert args.difficulty_drop_easy == 0.05
        assert args.difficulty_drop_hard == 0.01
        assert args.difficulty_batch_size == 8
        assert args.no_difficulty_progress is False
        args = tr._build_arg_parser().parse_args(
            ["--difficulty-model", "", "--difficulty-drop-easy", "0",
             "--difficulty-drop-hard", "0.02", "--no-difficulty-progress"]
        )
        assert args.difficulty_model == ""
        assert args.difficulty_drop_easy == 0.0
        assert args.difficulty_drop_hard == pytest.approx(0.02)
        assert args.no_difficulty_progress is True
