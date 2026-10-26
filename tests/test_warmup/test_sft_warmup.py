'Tests for sft_warmup.py — data plumbing + CLI (offline, trainer mocked).\n\n\nThe script lives outside the installed package, so it is loaded by path via\n``importlib``. The heavy training path (transformers/trl) is never touched:\nthese tests exercise the data building, the prompt-completion conversion, the\nval budget filter, and the CLI parser.\nprompt-completion'

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "sft_warmup.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_sft_warmup", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sw = _load_module()


def _ex(cls: str, ref_len: int = 10) -> object:
    from llm_training.schema import UnifiedExample

    return UnifiedExample(
        task_class=cls,
        prompt=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        reference="слово " * ref_len,
        reference_messages=[{"role": "assistant", "content": "слово " * ref_len}],
        extra={"is_call_turn": True, "is_final_answer": True, "turn_index": 0},
    )


class TestToPromptCompletion:
    def test_rows_carry_prompt_and_completion(self, tokenizer):
        rows = sw.to_prompt_completion([_ex("coder")], tokenizer)
        assert len(rows) == 1
        # Standard (pre-rendered text) prompt-completion rows.
        
        assert isinstance(rows[0]["prompt"], str)
        assert isinstance(rows[0]["completion"], str)
        assert "<|im_start|>assistant" in rows[0]["prompt"]

    def test_completion_is_reference_suffix(self, tokenizer):
        # The completion must be exactly the rendered reference span: the
        # full-conversation render minus the prompt render (the same boundary
        # mask_example localises for the eval/GRPO path).
        # completion
        
        rows = sw.to_prompt_completion([_ex("coder")], tokenizer)
        from llm_training.tokenizer_utils import prepare_messages, render

        ex = _ex("coder")
        _, full = prepare_messages(ex, tokenizer)
        full_text = render(tokenizer, full, add_generation_prompt=False)
        assert rows[0]["prompt"] + rows[0]["completion"] == full_text

    def test_tool_call_rendered_in_completion(self, tokenizer):
        # An SGR final turn's completion carries the rendered <tool_call>.
        
        from llm_training.loaders import sgr_agent
        from llm_training.config import DataConfig
        import pathlib
        ex = next(iter(sgr_agent.load_class(
            "coder", tokenizer, DataConfig(data_dir=pathlib.Path("src_data"))
        )))
        rows = sw.to_prompt_completion([ex], tokenizer)
        assert "<tool_call>" in rows[0]["completion"]
        assert "finalanswertool" in rows[0]["completion"]

    def test_arrow_regression_mixed_tool_calls(self, tokenizer, data_dir):
        # Regression for the cluster crash: tool_calls_multiturn carries
        # heterogeneous value types in same-named arguments fields (int vs
        # str), which pyarrow cannot unify in a columnar schema
        # (ArrowInvalid: cannot mix struct and non-struct). The pre-rendered
        # TEXT rows must pass Dataset.from_list where the conversational
        # lists failed.
        
        
        
        from datasets import Dataset
        from llm_training.loaders import tool_calls_multiturn
        from llm_training.config import DataConfig

        examples = list(tool_calls_multiturn.load(
            tokenizer, DataConfig(data_dir=Path(data_dir))
        ))
        assert len(examples) > 50
        rows = sw.to_prompt_completion(examples[:100], tokenizer)
        ds = Dataset.from_list(rows)  # must not raise ArrowInvalid
        assert len(ds) == len(rows)


class TestValBudgetFilter:
    def test_long_references_dropped(self, tokenizer):
        short = _ex("coder", ref_len=5)     # ~a dozen tokens of reference
        long_ = _ex("coder", ref_len=5000)  # ~5000 tokens → above the budget
        kept, dropped = sw.filter_val_for_generation_budget([short, long_], tokenizer)
        # The dropped examples are RETURNED (a list, not a count) so the
        # caller can route them back to train.
        
        
        assert len(dropped) == 1
        assert dropped[0].reference == long_.reference
        assert len(kept) == 1
        assert kept[0].reference == short.reference


class TestParser:
    def test_max_shard_size_defaults_to_5gb(self):
        # The checkpoint discipline: shards pinned at 5 GB regardless of the
        # transformers version (v5.2+ changed the library default to 50 GB,
        # which would write one monolithic ~30 GB file).
        
        
        
        args = sw._build_arg_parser().parse_args([])
        assert args.max_shard_size == "5GB"

    def test_max_shard_size_flag_parses(self):
        args = sw._build_arg_parser().parse_args(["--max-shard-size", "2GB"])
        assert args.max_shard_size == "2GB"

    def test_parser_defaults_unchanged(self):
        # The extraction of _build_arg_parser() must not disturb the knobs.
        
        args = sw._build_arg_parser().parse_args([])
        assert args.model == "anonymous/base-model"
        assert args.max_len == 8192
        assert args.epochs == 5
        assert args.seed == 42

    def test_rehearsal_defaults(self):
        # The rehearsal mix is ON by default (--no-rehearsal reproduces the
        # pre-r5 catastrophic-forgetting configuration); the probe gate
        # defaults to 0.10 (≈2 sigma of the 76-item binomial noise).
        
        args = sw._build_arg_parser().parse_args([])
        assert args.no_rehearsal is False
        assert args.rehearsal_tasks is None
        assert args.warmup_probe_gate == 0.10
        assert args.no_drift_probe is False
        args = sw._build_arg_parser().parse_args(
            ["--no-rehearsal", "--warmup-probe-gate", "0",
             "--no-drift-probe", "--rehearsal-task", "nerel_instruct"]
        )
        assert args.no_rehearsal is True
        assert args.warmup_probe_gate == 0.0
        assert args.no_drift_probe is True
        assert args.rehearsal_tasks == ["nerel_instruct"]


class TestDefaultClasses:
    def test_includes_sgr_and_aux(self):
        assert set(sw.DEFAULT_CLASSES) == {
            *sw.SGR_CLASSES, "opencode_gen", "tool_calls_multiturn",
        }
        assert len(sw.DEFAULT_CLASSES) == 11

    def test_rehearsal_composition(self):
        # The anti-forgetting mix (the 2026-09 attribution): RU rehearsal +
        # EN rehearsal (multihop_rag, docsem_quant), drawn from the committed
        # src_data catalog; the Meno-stand classes are deliberately ABSENT
        # (their test split is the GRPO anchors' fixed-val pool).
        
        
        assert set(sw.REHEARSAL_CLASSES) == {
            "mtrageval", "menon_qa", "self_cognition", "asr_summarization",
            "rummlu_pro", "multihop_rag", "docsem_quant",
        }
        assert not set(sw.REHEARSAL_CLASSES) & {"meno_rerank", "meno_rewrite", "meno_final"}

    def test_budget_constants(self):
        # The val generation budget must comfortably exceed the reference
        # ceiling (the envelope wrapper adds overhead on top of the answer):
        # 4096 covers the longest SGR val reference span (3718 tokens, a
        # deep_research action turn) with ~296 tokens of envelope margin.
        assert sw.VAL_MAX_NEW_TOKENS > sw.VAL_MAX_REFERENCE_TOKENS
        assert sw.VAL_MAX_NEW_TOKENS == 4096
        assert sw.VAL_MAX_REFERENCE_TOKENS == 3800


class TestBuildWarmupData:
    def test_build_split_and_stats(self, tokenizer, data_dir):
        # A real (small) build over one SGR class + one aux class.
        split, tools, stats = sw.build_warmup_data(
            tokenizer,
            data_dir=Path(data_dir),
            task_classes=["coder", "opencode_gen"],
            max_per_class=50,
            max_len=16000,
            seed=42,
            val_ratio=0.2,
            exclude_tasks=[],
        )
        # coder: 18 examples, ~20% → val; opencode_gen: train only.
        assert stats["n_val"] > 0
        assert all(e.task_class != "opencode_gen" for e in split.val)
        assert any(e.task_class == "opencode_gen" for e in split.train)
        # finalanswertool is in the collected registry.
        assert "finalanswertool" in tools
        # Stats blocks are populated.
        assert stats["n_train"] == len(split.train)
        assert stats["n_val"] == len(split.val)


class TestBuildSftKwargs:
    "The adaptive SFTConfig kwargs builder (TRL 0.20 ↔ 1.10 compatibility).\n\n\n    The fake config classes below mirror the two real stacks' field sets:\n    TRL 0.20 + transformers 4.x (has ``warmup_ratio``; ``max_length`` via\n    TrainingArguments) and TRL 1.10 + transformers v5.2+ (``warmup_ratio``\n    removed; ``warmup_steps`` accepts a float ratio; ``max_length`` defaults\n    to a truncating 1024)."

    @staticmethod
    def _base_kwargs():
        return dict(
            output_dir="out", epochs=2, lr=1e-5, per_device_batch=2,
            grad_accum=8, max_len=8192, optim="adamw_torch", seed=42,
        )

    @staticmethod
    def _make_config_cls(field_names):
        """Build a fake dataclass-based SFTConfig with the given field names."""
        from dataclasses import dataclass, field as dc_field
        from typing import Any

        namespace: dict[str, Any] = {"__annotations__": {}}
        for name in field_names:
            namespace["__annotations__"][name] = Any
            namespace[name] = dc_field(default=None)
        return dataclass(type("_FakeSFTConfig", (), namespace))

    def test_trl020_style_uses_warmup_ratio(self):
        # The local dev stack: warmup_ratio exists → used verbatim, exactly as
        # before the fix (the local path stays byte-identical).
        cls = self._make_config_cls([
            "output_dir", "num_train_epochs", "learning_rate",
            "per_device_train_batch_size", "gradient_accumulation_steps",
            "lr_scheduler_type", "warmup_ratio", "warmup_steps", "bf16",
            "gradient_checkpointing", "completion_only_loss", "logging_steps",
            "save_strategy", "report_to", "seed", "remove_unused_columns",
            "optim", "max_length",
        ])
        kw = sw._build_sft_kwargs(cls, **self._base_kwargs())
        assert kw["warmup_ratio"] == 0.03
        assert "warmup_steps" not in kw
        assert kw["max_length"] == 8192

    def test_v52_style_falls_back_to_warmup_steps_ratio(self):
        # The cluster stack: warmup_ratio removed → warmup_steps=0.03 (in
        # transformers v5 a float < 1 is a ratio of steps — semantics kept).
        cls = self._make_config_cls([
            "output_dir", "num_train_epochs", "learning_rate",
            "per_device_train_batch_size", "gradient_accumulation_steps",
            "lr_scheduler_type", "warmup_steps", "bf16",
            "gradient_checkpointing", "completion_only_loss", "logging_steps",
            "save_strategy", "report_to", "seed", "remove_unused_columns",
            "optim", "max_length",
        ])
        kw = sw._build_sft_kwargs(cls, **self._base_kwargs())
        assert kw["warmup_steps"] == 0.03
        assert "warmup_ratio" not in kw
        assert kw["max_length"] == 8192

    def test_no_warmup_fields_warns_and_continues(self):
        # A hypothetical future without any warmup field: warning, not a crash.
        cls = self._make_config_cls([
            "output_dir", "num_train_epochs", "learning_rate",
            "per_device_train_batch_size", "gradient_accumulation_steps",
            "lr_scheduler_type", "bf16", "gradient_checkpointing",
            "completion_only_loss", "logging_steps", "save_strategy",
            "report_to", "seed", "remove_unused_columns", "optim",
            "max_length",
        ])
        kw = sw._build_sft_kwargs(cls, **self._base_kwargs())
        assert "warmup_ratio" not in kw and "warmup_steps" not in kw
        assert kw["max_length"] == 8192  # everything else survives

    def test_legacy_max_seq_length_name_supported(self):
        # A stack that only knows the legacy max_seq_length name.
        cls = self._make_config_cls([
            "output_dir", "num_train_epochs", "max_seq_length", "warmup_ratio",
        ])
        kw = sw._build_sft_kwargs(cls, **self._base_kwargs())
        assert kw["max_seq_length"] == 8192
        assert "max_length" not in kw

    def test_missing_length_cap_raises(self):
        # Without any length field TRL would truncate to its own default —
        # refuse to run rather than corrupt the warm-up silently.
        cls = self._make_config_cls([
            "output_dir", "num_train_epochs", "warmup_ratio",
        ])
        with pytest.raises(RuntimeError, match="max_length"):
            sw._build_sft_kwargs(cls, **self._base_kwargs())

    def test_unknown_kwargs_dropped_with_warning(self):
        # Forward compatibility: fields the installed class does not know are
        # dropped (with a warning), never a TypeError.
        cls = self._make_config_cls([
            "output_dir", "max_length", "warmup_steps",
        ])
        kw = sw._build_sft_kwargs(cls, **self._base_kwargs())
        assert set(kw) <= {"output_dir", "max_length", "warmup_steps"}
        assert kw["output_dir"] == "out"

    def test_gradient_checkpointing_kwargs_non_reentrant(self):
        # Non-reentrant checkpointing plays nicer with inputs requiring grads
        # and with Liger-patched modules; dropped on stacks without the field.
        cls = self._make_config_cls([
            "output_dir", "max_length", "gradient_checkpointing_kwargs",
        ])
        kw = sw._build_sft_kwargs(cls, **self._base_kwargs())
        assert kw["gradient_checkpointing_kwargs"] == {"use_reentrant": False}

    def test_gradient_checkpointing_kwargs_dropped_when_unknown(self):
        # Stacks that do not declare the field must not receive it (the
        # generic drop handles it — no TypeError).
        cls = self._make_config_cls(["output_dir", "max_length"])
        kw = sw._build_sft_kwargs(cls, **self._base_kwargs())
        assert "gradient_checkpointing_kwargs" not in kw

    def test_liger_flag_sets_use_liger_kernel_when_declared(self):
        # liger=True + the field declared → use_liger_kernel=True (the
        # Trainer-level path; the caller checks the field before passing).
        cls = self._make_config_cls([
            "output_dir", "max_length", "use_liger_kernel",
        ])
        kw = sw._build_sft_kwargs(cls, **self._base_kwargs(), liger=True)
        assert kw["use_liger_kernel"] is True

    def test_liger_off_by_default(self):
        # Without liger=True the kwarg is never passed — the default path
        # stays byte-identical to the pre-Liger behaviour.
        cls = self._make_config_cls([
            "output_dir", "max_length", "use_liger_kernel",
        ])
        kw = sw._build_sft_kwargs(cls, **self._base_kwargs())
        assert "use_liger_kernel" not in kw

    def test_liger_kwarg_dropped_when_field_absent(self):
        # Safety net: even if liger=True reaches the builder on a stack
        # without the field, the generic drop removes it (no TypeError).
        cls = self._make_config_cls(["output_dir", "max_length"])
        kw = sw._build_sft_kwargs(cls, **self._base_kwargs(), liger=True)
        assert "use_liger_kernel" not in kw

    def test_real_installed_sft_config(self):
        # Smoke against the actually installed TRL: the builder must produce a
        # kwargs dict the installed SFTConfig accepts without a TypeError.
        from dataclasses import fields as dc_fields
        from trl import SFTConfig

        kw = sw._build_sft_kwargs(SFTConfig, **self._base_kwargs())
        known = {f.name for f in dc_fields(SFTConfig)}
        assert set(kw) <= known  # everything passes the fields() filter
        cfg = SFTConfig(**kw)  # must not raise on the installed stack
        assert cfg.max_length == 8192
        if "gradient_checkpointing_kwargs" in known:
            assert cfg.gradient_checkpointing_kwargs == {"use_reentrant": False}

    def test_real_installed_sft_config_liger(self):
        # When the installed SFTConfig declares use_liger_kernel (TRL >=0.13),
        # liger=True must round-trip through the real config.
        from dataclasses import fields as dc_fields
        from trl import SFTConfig

        if "use_liger_kernel" not in {f.name for f in dc_fields(SFTConfig)}:
            pytest.skip("installed SFTConfig has no use_liger_kernel field")
        kw = sw._build_sft_kwargs(SFTConfig, **self._base_kwargs(), liger=True)
        cfg = SFTConfig(**kw)
        assert cfg.use_liger_kernel is True


class TestApplyLigerToModel:
    def test_missing_package_raises_bilingual_systemexit(self, monkeypatch):
        # An explicit --liger without liger-kernel installed must fail with a
        # clear bilingual message naming the package — never an raw ImportError.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("liger_kernel"):
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(SystemExit, match="liger-kernel"):
            sw._apply_liger_to_model(object())


# ---------------------------------------------------------------------------
# Envelope-preserving decode
# ---------------------------------------------------------------------------
class TestDecodeGeneration:
    'The regression suite for the special-token-stripping bug.\n\n\n    ``<tool_call>`` / ``</tool_call>`` are added special tokens of the\n    Meno-Lite tokenizer (ids 145123/145124); the legacy\n    ``decode(..., skip_special_tokens=True)`` deleted the envelope from the\n    generation, zeroing every final/action val turn via the strict harmonic\n    mean (cluster run: ``final=0.0000 action=0.0000``).'

    GEN = (
        '<tool_call>\n{"name": "finalanswertool", '
        '"arguments": {"answer": "ответ"}}\n</tool_call>'
    )

    def _ids(self, tokenizer):
        # The generation plus the stop token HF generate appends (twice, to
        # exercise the trailing-strip loop).
        
        
        ids = tokenizer.encode(self.GEN, add_special_tokens=False)
        return ids + [tokenizer.eos_token_id, tokenizer.eos_token_id]

    def test_envelope_preserved_and_stop_stripped(self, tokenizer):
        from llm_training.reward.scoring import format_check

        text = sw.decode_generation(tokenizer, self._ids(tokenizer))
        assert text.startswith("<tool_call>")
        assert text.endswith("</tool_call>")
        assert "<|im_end|>" not in text
        assert "<|endoftext|>" not in text
        # The envelope reaches format_check intact — a perfect call scores 1.0.
        
        assert format_check(text, "coder", frozenset({"finalanswertool"})) == 1.0

    def test_regression_legacy_decode_strips_envelope(self, tokenizer):
        # Documents the bug itself: the legacy decode loses the envelope,
        # the fixed one keeps it.
        
        ids = self._ids(tokenizer)
        legacy = tokenizer.decode(ids, skip_special_tokens=True)
        assert "<tool_call>" not in legacy
        fixed = sw.decode_generation(tokenizer, ids)
        assert "<tool_call>" in fixed
        assert "</tool_call>" in fixed

    def test_mid_text_control_tokens_removed(self, tokenizer):
        # Residual control tokens inside the continuation are removed too
        # (they cannot legitimately appear inside an assistant turn).
        
        ids = tokenizer.encode("текст", add_special_tokens=False)
        ids = (
            ids[:1]
            + [tokenizer.convert_tokens_to_ids("<|im_end|>")]
            + ids[1:]
            + [tokenizer.eos_token_id]
        )
        text = sw.decode_generation(tokenizer, ids)
        assert "<|im_end|>" not in text
        assert "<|endoftext|>" not in text
        assert "текст" in text


class _FakeValModel:
    'Records use_cache/eval/train; "generates" a fixed final-answer call.'

    def __init__(self, tokenizer, gen_text: str) -> None:
        self.config = type("Cfg", (), {"use_cache": False})()
        self.device = "cpu"
        self._tokenizer = tokenizer
        self._gen_ids = tokenizer.encode(gen_text, add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        self.use_cache_at_generate: list[bool] = []
        self.eval_calls = 0
        self.train_calls = 0

    def eval(self):  # noqa: D102
        self.eval_calls += 1

    def train(self):  # noqa: D102
        self.train_calls += 1

    def generate(self, **inputs):  # noqa: D102
        self.use_cache_at_generate.append(self.config.use_cache)
        prompt_ids = inputs["input_ids"][0].tolist()
        return [prompt_ids + self._gen_ids]


class TestRunGenerativeValidation:
    'End-to-end (fake model, real tokenizer/metrics) val-loop regression.'

    GEN = (
        '<tool_call>\n{"name": "finalanswertool", '
        '"arguments": {"answer": "Это финальный ответ."}}\n</tool_call>'
    )

    @staticmethod
    def _final_example():
        from llm_training.schema import UnifiedExample

        return UnifiedExample(
            task_class="coder",
            prompt=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            reference="Это финальный ответ.",
            reference_messages=[
                {"role": "assistant", "content": "Это финальный ответ."}
            ],
            extra={"is_call_turn": True, "is_final_answer": True, "turn_index": 0},
        )

    def test_final_turn_scores_nonzero(self, tokenizer, tmp_path):
        # The exact cluster regression: with the legacy skip_special_tokens
        # decode the final turn scored exactly 0.0; with the fix it must be
        # strictly positive (perfect envelope + identical content → ~1.0).
        
        
        model = _FakeValModel(tokenizer, self.GEN)
        dump = tmp_path / "val_generations.jsonl"
        breakdown = sw.run_generative_validation(
            model,
            tokenizer,
            [self._final_example()],
            frozenset({"finalanswertool"}),
            dump_path=dump,
            epoch=1.0,
            step=60,
        )
        assert breakdown["n"] == 1.0
        assert breakdown["final"] == pytest.approx(1.0)
        assert breakdown["aggregate"] == pytest.approx(1.0)

    def test_use_cache_enabled_during_generation_and_restored(self, tokenizer, tmp_path):
        model = _FakeValModel(tokenizer, self.GEN)
        assert model.config.use_cache is False  # the training-time state
        sw.run_generative_validation(
            model,
            tokenizer,
            [self._final_example()],
            frozenset({"finalanswertool"}),
            dump_path=tmp_path / "val_generations.jsonl",
        )
        assert model.use_cache_at_generate == [True]  # KV-cache was on
        assert model.config.use_cache is False  # restored for grad checkpointing
        assert model.eval_calls == 1
        assert model.train_calls == 1

    def test_dump_written_with_raw_generation(self, tokenizer, tmp_path):
        import json

        model = _FakeValModel(tokenizer, self.GEN)
        dump = tmp_path / "val_generations.jsonl"
        sw.run_generative_validation(
            model,
            tokenizer,
            [self._final_example()],
            frozenset({"finalanswertool"}),
            dump_path=dump,
            epoch=2.0,
            step=120,
        )
        lines = dump.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["epoch"] == 2.0
        assert rec["step"] == 120
        assert rec["task_class"] == "coder"
        assert rec["turn_kind"] == "final"
        assert "<tool_call>" in rec["generation"]
        assert "finalanswertool" in rec["generation"]
        assert rec["reference"] == "Это финальный ответ."
        assert rec["score"] == pytest.approx(1.0)

    def test_no_dump_when_path_none(self, tokenizer):
        # dump_path=None must be legal (no file opened, no crash).
        # dump_path=None
        model = _FakeValModel(tokenizer, self.GEN)
        breakdown = sw.run_generative_validation(
            model,
            tokenizer,
            [self._final_example()],
            frozenset({"finalanswertool"}),
        )
        assert breakdown["final"] == pytest.approx(1.0)


class TestCli:
    def test_parser_defaults(self):
        args = sw.main.__wrapped__ if hasattr(sw.main, "__wrapped__") else None
        parser_has = sw.main  # noqa: F841 - just ensure main is importable
        # Parse via constructing the same parser shape: use main with --help
        import contextlib, io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with pytest.raises(SystemExit) as ei:
                sw.main(["--help"])
        assert ei.value.code == 0
        assert "--exclude-task" in buf.getvalue()
        assert "--val-ratio" in buf.getvalue()

    def test_epochs_default_is_five(self, monkeypatch):
        # Capture the parsed namespace before the heavy imports run.
        
        import argparse

        captured: dict = {}
        real = argparse.ArgumentParser.parse_args

        def fake_parse(self, args=None, namespace=None):
            ns = real(self, args, namespace)
            captured.update(vars(ns))
            raise SystemExit(99)  # stop main() before model loading

        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fake_parse)
        with pytest.raises(SystemExit) as ei:
            sw.main([])
        assert ei.value.code == 99
        assert captured["epochs"] == 5

    def test_help_documents_new_memory_flags(self):
        # The memory-saving flags must be discoverable via --help.
        
        import contextlib, io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with pytest.raises(SystemExit) as ei:
                sw.main(["--help"])
        assert ei.value.code == 0
        help_text = buf.getvalue()
        assert "--attn" in help_text
        assert "flash_attention_2" in help_text
        assert "--liger" in help_text
        assert "liger-kernel" in help_text

    def test_attn_choices_validated(self):
        # An invalid attention implementation must be rejected at parse time.
        import contextlib, io

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            with pytest.raises(SystemExit) as ei:
                sw.main(["--attn", "flashinfer"])
        assert ei.value.code != 0

    def test_difficulty_filter_defaults(self):
        # The warm-up difficulty filter: the base Meno-Lite as the separate
        # scoring model, HARD tail only (0.01) — the easy tail is NOT cut
        # (the rehearsal mix is for retention, not learning).
        
        
        args = sw._build_arg_parser().parse_args([])
        assert args.difficulty_model == "anonymous/base-model"
        assert args.difficulty_drop_hard == 0.01
        assert args.difficulty_drop_easy == 0.0
        assert args.difficulty_batch_size == 8
        assert args.no_difficulty_progress is False
        args = sw._build_arg_parser().parse_args(
            ["--difficulty-model", "", "--difficulty-drop-hard", "0.02",
             "--no-difficulty-progress"]
        )
        assert args.difficulty_model == ""
        assert args.difficulty_drop_hard == pytest.approx(0.02)
        assert args.no_difficulty_progress is True
