"Tests for grpo_train.py's adaptive ``_build_grpo_kwargs`` (TRL 0.20 ↔ 1.10).\n\n\nThe fake config classes mirror the two real stacks' field sets: TRL 0.20\n(``max_prompt_length``, ``use_liger_loss`` + ``use_liger_kernel``) and\nTRL 1.10 / transformers v5.2+ (``max_prompt_length`` removed, Liger via the\nsingle ``use_liger_kernel``, ``vllm_enable_sleep_mode`` present)."

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "grpo_train.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_train", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tr = _load_module()


def _make_config_cls(field_names: list[str]):
    """Build a fake dataclass-based GRPOConfig with the given field names."""
    namespace: dict[str, Any] = {"__annotations__": {}}
    for name in field_names:
        namespace["__annotations__"][name] = Any
        namespace[name] = dc_field(default=None)
    return dataclass(type("_FakeGRPOConfig", (), namespace))


def _base_kwargs():
    return dict(
        output_dir="out", num_generations=8, max_completion_length=8192,
        temperature=0.7, top_p=0.95, beta=0.0, lr=1e-6, epochs=1.0,
        per_device_batch=4, grad_accum=8, optim="adamw_bnb_8bit", seed=42,
        max_steps=0, vllm_mode="colocate", vllm_gpu_memory_utilization=0.4,
        vllm_server_host="localhost", vllm_server_port=8000, liger=True,
    )


COMMON_FIELDS = [
    "output_dir", "num_generations", "max_completion_length", "temperature",
    "top_p", "beta", "learning_rate", "num_train_epochs",
    "per_device_train_batch_size",
    "gradient_accumulation_steps", "optim", "seed", "bf16",
    "gradient_checkpointing", "logging_steps", "save_strategy", "report_to",
    "use_vllm", "vllm_mode", "vllm_gpu_memory_utilization",
]

TRL_020_STYLE = COMMON_FIELDS + [
    "max_prompt_length",          # removed in 1.10
    "use_liger_loss",             # the 0.20 fused-GRPO-loss flag
    "use_liger_kernel",
]

TRL_110_STYLE = COMMON_FIELDS + [
    "use_liger_kernel",           # 1.10: single flag drives the fused loss
    "vllm_enable_sleep_mode",     # 1.10-only
    "vllm_server_host", "vllm_server_port",
]


class TestCompletionBudget:
    def test_explicit_max_completion_length_always_set(self):
        for fields in (TRL_020_STYLE, TRL_110_STYLE):
            kw = tr._build_grpo_kwargs(_make_config_cls(fields), **_base_kwargs())
            assert kw["max_completion_length"] == 8192

    def test_epochs_wired_to_num_train_epochs(self):
        # --epochs used to be parsed but never forwarded, silently leaving
        # transformers' default num_train_epochs=3.0 in charge.
        
        for fields in (TRL_020_STYLE, TRL_110_STYLE):
            kw = tr._build_grpo_kwargs(
                _make_config_cls(fields),
                **{**_base_kwargs(), "epochs": 2.5},
            )
            assert kw["num_train_epochs"] == 2.5

    def test_missing_length_field_is_hard_error(self):
        # TRL's own default (256) silently truncates every completion — refuse
        # to run rather than zero all format scores.
        
        cls = _make_config_cls([f for f in TRL_110_STYLE if f != "max_completion_length"])
        with pytest.raises(RuntimeError, match="max_completion_length"):
            tr._build_grpo_kwargs(cls, **_base_kwargs())

    def test_max_prompt_length_none_where_present(self):
        # 0.20's default 512 truncates long tool-use prompts → None (no cap).
        
        kw = tr._build_grpo_kwargs(_make_config_cls(TRL_020_STYLE), **_base_kwargs())
        assert kw["max_prompt_length"] is None
        # 1.10 has no such field — nothing passed.
        kw2 = tr._build_grpo_kwargs(_make_config_cls(TRL_110_STYLE), **_base_kwargs())
        assert "max_prompt_length" not in kw2


class TestLiger:
    def test_trl020_style_sets_both_flags(self):
        kw = tr._build_grpo_kwargs(_make_config_cls(TRL_020_STYLE), **_base_kwargs())
        assert kw["use_liger_loss"] is True
        assert kw["use_liger_kernel"] is True

    def test_trl110_style_single_flag(self):
        kw = tr._build_grpo_kwargs(_make_config_cls(TRL_110_STYLE), **_base_kwargs())
        assert kw["use_liger_kernel"] is True
        assert "use_liger_loss" not in kw

    def test_no_liger_fields_warns_and_skips(self):
        # A future stack without either field: warning, liger silently off.
        
        fields = [f for f in TRL_110_STYLE
                  if f not in ("use_liger_kernel", "use_liger_loss")]
        kw = tr._build_grpo_kwargs(_make_config_cls(fields), **_base_kwargs())
        assert "use_liger_kernel" not in kw and "use_liger_loss" not in kw

    def test_liger_disabled_sets_nothing(self):
        kw = tr._build_grpo_kwargs(
            _make_config_cls(TRL_110_STYLE), **{**_base_kwargs(), "liger": False}
        )
        assert kw.get("use_liger_kernel") is None  # default of the fake field


class TestVllmModes:
    def test_colocate_sets_memory_fraction_and_sleep(self):
        kw = tr._build_grpo_kwargs(_make_config_cls(TRL_110_STYLE), **_base_kwargs())
        assert kw["vllm_mode"] == "colocate"
        assert kw["vllm_gpu_memory_utilization"] == 0.4
        assert kw["vllm_enable_sleep_mode"] is True

    def test_colocate_without_sleep_field_warns(self):
        # 0.20-style has no sleep field: warning path, still functional.
        kw = tr._build_grpo_kwargs(_make_config_cls(TRL_020_STYLE), **_base_kwargs())
        assert "vllm_enable_sleep_mode" not in kw

    def test_server_mode_sets_host_port(self):
        kw = tr._build_grpo_kwargs(
            _make_config_cls(TRL_110_STYLE),
            **{**_base_kwargs(), "vllm_mode": "server"},
        )
        assert kw["vllm_mode"] == "server"
        assert kw["vllm_server_host"] == "localhost"
        assert kw["vllm_server_port"] == 8000
        assert "vllm_enable_sleep_mode" not in kw

    def test_server_without_server_fields_is_error(self):
        fields = [f for f in TRL_110_STYLE
                  if not f.startswith("vllm_server") and f != "vllm_server_base_url"]
        with pytest.raises(RuntimeError, match="vllm_mode=server"):
            tr._build_grpo_kwargs(
                _make_config_cls(fields),
                **{**_base_kwargs(), "vllm_mode": "server"},
            )


class TestDivisibility:
    def test_divisible_ok(self):
        # 4 x 8 x 1 = 32 = 4 groups of 8.
        tr._build_grpo_kwargs(_make_config_cls(TRL_110_STYLE), **_base_kwargs())

    def test_indivisible_rejected(self):
        # 3 x 3 x 1 = 9 is not divisible by 8.
        with pytest.raises(ValueError, match="divisible"):
            tr._build_grpo_kwargs(
                _make_config_cls(TRL_110_STYLE),
                **{**_base_kwargs(), "per_device_batch": 3, "grad_accum": 3},
            )

    def test_world_size_accounted(self):
        # 3 x 3 x 2 = 18 is not divisible by 8.
        with pytest.raises(ValueError, match="divisible"):
            tr._build_grpo_kwargs(
                _make_config_cls(TRL_110_STYLE),
                **{**_base_kwargs(), "per_device_batch": 3, "grad_accum": 3},
                world_size=2,
            )


class TestUnknownFields:
    def test_unknown_kwargs_dropped(self):
        # Forward compatibility: fields the installed class does not know are
        # dropped with a warning, never a TypeError.
        
        cls = _make_config_cls(["output_dir", "max_completion_length"])
        kw = tr._build_grpo_kwargs(cls, **_base_kwargs())
        assert set(kw) <= {"output_dir", "max_completion_length"}
        assert kw["output_dir"] == "out"

    def test_max_steps_zero_omitted(self):
        kw = tr._build_grpo_kwargs(_make_config_cls(TRL_110_STYLE), **_base_kwargs())
        assert "max_steps" not in kw

    def test_max_steps_positive_passed(self):
        kw = tr._build_grpo_kwargs(
            _make_config_cls(TRL_110_STYLE + ["max_steps"]),
            **{**_base_kwargs(), "max_steps": 10},
        )
        assert kw["max_steps"] == 10


class TestRealInstalledConfig:
    def test_smoke_against_installed_grpoconfig(self):
        # The builder must produce a kwargs dict the INSTALLED GRPOConfig
        # accepts (dev stack: TRL 0.20) — the field filter guarantees it.
        
        # TRL 0.20).
        from dataclasses import fields as dc_fields
        from trl import GRPOConfig

        kw = tr._build_grpo_kwargs(GRPOConfig, **_base_kwargs())
        known = {f.name for f in dc_fields(GRPOConfig)}
        assert set(kw) <= known
        cfg = GRPOConfig(**kw)  # must not raise
        assert cfg.max_completion_length == 8192
