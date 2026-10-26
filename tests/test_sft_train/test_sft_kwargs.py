"Tests for sft_train.py's adaptive SFTConfig kwargs (offline).\n\n\nMirrors tests/test_grpo_train/test_grpo_kwargs.py: the real installed SFTConfig\n(TRL 0.20 in dev) must accept every kwarg untouched, and fake config classes\nsimulate the TRL 0.15.1 (vllm_env) / TRL 1.10 (cluster) landmines — the\ncompletion-only-loss routing and the length-cap ladder in particular (a silent\ndrop of either would change the ablation semantics)."

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCRIPT = Path(__file__).resolve().parents[2] / "sft_train.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_sft_train_kw", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


st = _load_module()


@dataclass
class _FakeSftConfig:
    'A minimal stand-in simulating older/newer SFTConfig field sets.'

    output_dir: str = "x"
    # The base fields every variant shares (TrainingArguments subset).
    learning_rate: float = 0.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    lr_scheduler_type: str = "linear"
    bf16: bool = False
    gradient_checkpointing: bool = False
    gradient_checkpointing_kwargs: dict = field(default_factory=dict)
    logging_steps: int = 5
    save_strategy: str = "no"
    report_to: list = field(default_factory=list)
    seed: int = 42
    remove_unused_columns: bool = False
    optim: str = "adamw_torch"
    weight_decay: float = 0.0
    num_train_epochs: float = 1.0
    # Variant-specific fields are appended by subclasses below.


@dataclass
class _ModernConfig(_FakeSftConfig):
    """TRL 0.14+/0.20/1.10-like: max_length + completion_only_loss."""

    max_length: int | None = None
    completion_only_loss: bool | None = None
    max_steps: int = -1
    warmup_ratio: float = 0.0
    use_liger_kernel: bool = False


@dataclass
class _LegacyDatasetKwargsConfig(_FakeSftConfig):
    """Older TRL: no completion_only_loss field, but dataset_kwargs exists."""

    max_length: int | None = None
    dataset_kwargs: dict = field(default_factory=dict)
    max_steps: int = -1
    warmup_ratio: float = 0.0


@dataclass
class _LegacySeqLengthConfig(_FakeSftConfig):
    """TRL 0.9-0.12-like: the max_seq_length spelling, dataset_kwargs."""

    max_seq_length: int | None = None
    dataset_kwargs: dict = field(default_factory=dict)
    max_steps: int = -1
    warmup_steps: int = 0


@dataclass
class _NoLengthConfig(_FakeSftConfig):
    """No length field at all — must be a hard error (loss fields present)."""

    completion_only_loss: bool | None = None
    dataset_kwargs: dict = field(default_factory=dict)
    max_steps: int = -1
    warmup_ratio: float = 0.0


@dataclass
class _NoCompletionLossConfig(_FakeSftConfig):
    """No completion_only_loss and no dataset_kwargs — a hard error."""

    max_length: int | None = None
    max_steps: int = -1


def _build(cfg_cls: Any, **overrides: Any) -> dict:
    return st._build_sft_kwargs(
        cfg_cls,
        output_dir="/tmp/x",
        lr=2e-4,
        per_device_batch=1,
        grad_accum=64,
        max_len=32768,
        optim="adamw_torch",
        seed=24,
        max_steps=782,
        **overrides,
    )


class TestRealInstalledConfig:
    def test_real_sft_config_accepts_all_kwargs(self):
        from trl import SFTConfig

        kwargs = _build(SFTConfig)
        cfg = SFTConfig(**kwargs)
        assert cfg.completion_only_loss is True
        assert cfg.max_length == 32768
        assert cfg.max_steps == 782
        assert cfg.num_train_epochs == 1
        assert cfg.label_names == ["labels"]

    def test_max_steps_zero_omitted(self):
        from trl import SFTConfig

        kwargs = st._build_sft_kwargs(
            SFTConfig, output_dir="/tmp/x", lr=1e-5, per_device_batch=1,
            grad_accum=16, max_len=8192, optim="adamw_torch", seed=42,
            max_steps=0,
        )
        assert "max_steps" not in kwargs


class TestCompletionOnlyLossRouting:
    def test_modern_field_used(self):
        kwargs = _build(_ModernConfig)
        assert kwargs["completion_only_loss"] is True
        assert "dataset_kwargs" not in kwargs

    def test_legacy_dataset_kwargs_fallback(self, caplog):
        with caplog.at_level(logging.WARNING):
            kwargs = _build(_LegacyDatasetKwargsConfig)
        assert kwargs["dataset_kwargs"] == {"completion_only_loss": True}
        assert "completion_only_loss" not in kwargs
        assert any("dataset_kwargs" in r.message for r in caplog.records)

    def test_missing_both_is_a_hard_error(self):
        # Training on the prompt tokens would silently change the ablation —
        # the kwargs builder must refuse to run.
        
        try:
            _build(_NoCompletionLossConfig)
        except RuntimeError as e:
            assert "completion_only_loss" in str(e)
        else:  # pragma: no cover
            raise AssertionError("RuntimeError not raised")


class TestLengthCapLadder:
    def test_max_length_preferred(self):
        kwargs = _build(_ModernConfig)
        assert kwargs["max_length"] == 32768

    def test_max_seq_length_fallback(self):
        kwargs = _build(_LegacySeqLengthConfig)
        assert kwargs["max_seq_length"] == 32768
        assert "max_length" not in kwargs

    def test_no_length_field_is_a_hard_error(self):
        try:
            _build(_NoLengthConfig)
        except RuntimeError as e:
            assert "max_length" in str(e)
        else:  # pragma: no cover
            raise AssertionError("RuntimeError not raised")


class TestLabelNames:
    def test_label_names_set_for_real_config(self):
        # transformers 4.53.x cannot auto-derive label_names from a
        # PeftModel (empty list + a warning per run) — the kwargs builder
        # always passes the explicit name.
        
        
        from trl import SFTConfig

        kwargs = _build(SFTConfig)
        assert kwargs["label_names"] == ["labels"]

    def test_label_names_dropped_when_field_absent(self):
        # The fakes simulate field sets WITHOUT label_names — the kwarg is
        # dropped by the forward-compatibility filter (a warning, not an
        # error; TrainingArguments declares it in every real version).
        
        
        kwargs = _build(_ModernConfig)
        assert "label_names" not in kwargs


class TestWarmupAndLiger:
    def test_warmup_ratio_used_when_present(self):
        kwargs = _build(_ModernConfig)
        assert kwargs["warmup_ratio"] == 0.03

    def test_warmup_steps_ratio_fallback(self):
        kwargs = _build(_LegacySeqLengthConfig)
        assert kwargs["warmup_steps"] == 0.03  # v5 float < 1 == ratio

    def test_liger_only_when_flag_and_field(self):
        kwargs = _build(_ModernConfig, liger=True)
        assert kwargs.get("use_liger_kernel") is True
        kwargs = _build(_ModernConfig, liger=False)
        assert "use_liger_kernel" not in kwargs

    def test_unknown_kwargs_dropped_with_warning(self, caplog):
        # A config genuinely lacking weight_decay / lr_scheduler_type → both
        # dropped with a warning, nothing else lost.
        
        

        @dataclass
        class _Sparse:
            output_dir: str = "x"
            learning_rate: float = 0.0
            per_device_train_batch_size: int = 1
            gradient_accumulation_steps: int = 1
            bf16: bool = False
            gradient_checkpointing: bool = False
            gradient_checkpointing_kwargs: dict = field(default_factory=dict)
            logging_steps: int = 5
            save_strategy: str = "no"
            report_to: list = field(default_factory=list)
            seed: int = 42
            remove_unused_columns: bool = False
            optim: str = "adamw_torch"
            num_train_epochs: float = 1.0
            max_length: int | None = None
            completion_only_loss: bool | None = None
            max_steps: int = -1
            warmup_ratio: float = 0.0

        with caplog.at_level(logging.WARNING):
            kwargs = _build(_Sparse)
        assert "weight_decay" not in kwargs
        assert "lr_scheduler_type" not in kwargs
        assert any("dropped unsupported kwargs" in r.message
                   for r in caplog.records)
        # Everything the fake DOES declare must survive.
        
        from dataclasses import fields as dc_fields

        known = {f.name for f in dc_fields(_Sparse)}
        assert set(kwargs) <= known
        assert {"completion_only_loss", "max_length", "warmup_ratio"} <= set(kwargs)
