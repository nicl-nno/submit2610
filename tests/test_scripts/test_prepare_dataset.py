'Tests for prepare_dataset.py — CLI parsing + offline dispatch helpers.\n\n\nThe heavy build/difficulty paths are NOT exercised here (they need the\nMeno-Lite tokenizer + a GPU); the offline pieces — the parser, the empty-model\ndifficulty no-op and the ``sft_warmup`` script loader — are covered.'

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "prepare_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_prepare_dataset", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pd = _load_module()


class TestParserDefaults:
    def test_pipeline_default_grpo(self):
        args = pd._build_arg_parser().parse_args(["--output-dir", "x"])
        assert args.pipeline == "grpo"

    def test_val_task_default(self):
        args = pd._build_arg_parser().parse_args(["--output-dir", "x"])
        assert args.val_task == list(pd.RECOMMENDED_VAL_TASKS)

    def test_difficulty_model_default(self):
        args = pd._build_arg_parser().parse_args(["--output-dir", "x"])
        assert args.difficulty_model == "anonymous/base-model"

    def test_val_task_repeatable(self):
        args = pd._build_arg_parser().parse_args(
            ["--output-dir", "x", "--val-task", "coder", "rewriter"]
        )
        assert args.val_task == ["coder", "rewriter"]

    def test_output_dir_required(self):
        with pytest.raises(SystemExit):
            pd._build_arg_parser().parse_args([])


class TestScoreDifficultyOffline:
    def test_empty_model_returns_empty(self):
        # No model → no scoring, no torch import (offline-safe).
        
        assert pd._score_difficulty([], "", attn="sdpa", batch_size=8,
                                    progress=False, desc="x") == {}


class TestImportSftWarmup:
    def test_loads_build_warmup_data(self):
        sw = pd._import_sft_warmup()
        assert callable(sw.build_warmup_data)
        assert callable(sw.filter_val_for_generation_budget)
