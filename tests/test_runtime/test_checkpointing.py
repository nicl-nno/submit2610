'Tests for llm_training.runtime.checkpointing (offline, fakes only).\n\n\nCovers: the ``max_shard_size`` kwarg injection via the temporary\n``save_pretrained`` wrapper, ``kwargs.setdefault`` semantics (an explicit\nvalue wins), the attribute restore on exit (including on exceptions), the\n``unwrap_model`` resolution (the patch must land on the UNWRAPPED model —\nthat is what ``Trainer._save`` calls under DeepSpeed), the no-op paths that\nkeep the offline test fakes (``object()`` models) working, and the hard\nbfloat16 policy check.\n``save_pretrained``'

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from llm_training.runtime.checkpointing import (
    DEFAULT_MAX_SHARD_SIZE,
    assert_bfloat16_model,
    sharded_checkpoint_saves,
    unwrap_trainer_model,
)


class _FakeParam:
    def __init__(self, dtype):
        self.dtype = dtype


class _FakeModel:
    'A PreTrainedModel stand-in: class-level save_pretrained, params.'

    def __init__(self, dtype=torch.bfloat16):
        self.calls: list[tuple[str, tuple, dict]] = []
        self._params = {
            "model.embed_tokens.weight": _FakeParam(dtype),
            "lm_head.weight": _FakeParam(dtype),
        }

    def named_parameters(self):
        return iter(list(self._params.items()))

    def save_pretrained(self, save_directory, *args, **kwargs):
        self.calls.append((save_directory, args, kwargs))
        return "saved"


def _trainer_for(model, unwrap=None):
    if unwrap is None:
        unwrap = lambda m: m  # noqa: E731 - trivial identity for fakes
    return SimpleNamespace(
        model=model,
        accelerator=SimpleNamespace(unwrap_model=unwrap),
    )


class TestShardSizeInjection:
    def test_default_constant_is_5gb(self):
        # The cluster policy: 5 GB, same as transformers 4.x' own default
        # (v5.2+ changed the library default to 50 GB — hence the pin).
        
        
        assert DEFAULT_MAX_SHARD_SIZE == "5GB"

    def test_injects_max_shard_size(self):
        model = _FakeModel()
        trainer = _trainer_for(model)
        with sharded_checkpoint_saves(trainer, "5GB"):
            model.save_pretrained("out", state_dict={})
        assert model.calls[0][2]["max_shard_size"] == "5GB"

    def test_explicit_caller_value_wins(self):
        # setdefault: if a future transformers forwards its own
        # max_shard_size through Trainer._save, the library value wins.
        
        # max_shard_size
        model = _FakeModel()
        trainer = _trainer_for(model)
        with sharded_checkpoint_saves(trainer, "5GB"):
            model.save_pretrained("out", max_shard_size="1GB")
        assert model.calls[0][2]["max_shard_size"] == "1GB"

    def test_attribute_restored_on_exit(self):
        model = _FakeModel()
        trainer = _trainer_for(model)
        with sharded_checkpoint_saves(trainer, "5GB"):
            assert "save_pretrained" in vars(model)
        assert "save_pretrained" not in vars(model)
        model.save_pretrained("out")
        assert "max_shard_size" not in model.calls[0][2]

    def test_attribute_restored_on_exception(self):
        model = _FakeModel()
        trainer = _trainer_for(model)
        with pytest.raises(RuntimeError):
            with sharded_checkpoint_saves(trainer, "5GB"):
                raise RuntimeError("boom")
        assert "save_pretrained" not in vars(model)

    def test_falsy_size_disables_patch_but_keeps_dtype_check(self):
        model = _FakeModel()
        trainer = _trainer_for(model)
        with sharded_checkpoint_saves(trainer, ""):
            model.save_pretrained("out")
        assert "max_shard_size" not in model.calls[0][2]

    def test_patch_lands_on_unwrapped_model(self):
        # Under DeepSpeed trainer.model is the engine; Trainer._save calls
        # unwrap_model(...).save_pretrained — the patch must sit on the
        # unwrapped module, and the wrapper object must stay untouched.
        
        # unwrap_model(...).save_pretrained
        
        inner = _FakeModel()
        wrapper = SimpleNamespace(module=inner)
        trainer = _trainer_for(wrapper, unwrap=lambda m: m.module)
        with sharded_checkpoint_saves(trainer, "5GB"):
            trainer.accelerator.unwrap_model(trainer.model).save_pretrained(
                "out", state_dict={}
            )
        assert inner.calls[0][2]["max_shard_size"] == "5GB"
        assert "save_pretrained" not in vars(wrapper)

    def test_noop_for_object_model(self):
        # The offline test fakes carry object() models — the context manager
        # must not require a real save_pretrained.
        
        
        trainer = _trainer_for(object())
        with sharded_checkpoint_saves(trainer, "5GB"):
            pass

    def test_noop_without_accelerator(self):
        trainer = SimpleNamespace(model=_FakeModel())
        model = trainer.model
        with sharded_checkpoint_saves(trainer, "5GB"):
            model.save_pretrained("out")
        assert model.calls[0][2]["max_shard_size"] == "5GB"


class TestUnwrapTrainerModel:
    def test_prefers_unwrap_model(self):
        inner = object()
        trainer = _trainer_for(object(), unwrap=lambda m: inner)
        assert unwrap_trainer_model(trainer) is inner

    def test_falls_back_to_trainer_model(self):
        model = object()
        trainer = SimpleNamespace(model=model, accelerator=SimpleNamespace())
        assert unwrap_trainer_model(trainer) is model

    def test_none_model(self):
        trainer = SimpleNamespace(model=None, accelerator=SimpleNamespace())
        assert unwrap_trainer_model(trainer) is None

    def test_model_fed_as_trainer_returns_its_model_attr(self):
        # THE TRAP (the 2026-09-26 sft_train.py crash): a peft MODEL passed
        # where the TRAINER is expected resolves ``.model`` through peft's
        # ``__getattr__`` chain to the BARE base causal LM (and
        # ``.accelerator`` to None) — the helper then returns the bare base,
        # on which ``merge_and_unload`` does not exist. Simulated with a
        # plain namespace carrying ``.model`` (the call contract is
        # identical); the call sites must pass the TRAINER.
        
        
        # ``__getattr__`` peft
        
        # ``merge_and_unload``
        
        bare_base = object()
        peft_like = SimpleNamespace(model=bare_base)
        assert unwrap_trainer_model(peft_like) is bare_base


class TestBfloat16Policy:
    def test_bfloat16_model_passes(self):
        assert assert_bfloat16_model(_FakeModel(torch.bfloat16)) is None

    def test_fp16_model_raises_before_any_save(self):
        # The hard guarantee: a wrong-dtype checkpoint is never written —
        # the check fires at context ENTRY, before the first save.
        
        
        model = _FakeModel(torch.float16)
        trainer = _trainer_for(model)
        with pytest.raises(TypeError, match="bfloat16"):
            with sharded_checkpoint_saves(trainer, "5GB"):
                pass  # never reached
        assert model.calls == []

    def test_mixed_dtypes_raise_with_names(self):
        model = _FakeModel(torch.bfloat16)
        model._params["lm_head.weight"] = _FakeParam(torch.float32)
        with pytest.raises(TypeError, match="lm_head.weight"):
            assert_bfloat16_model(model)

    def test_object_model_skipped(self):
        # No named_parameters → nothing to check (test fakes).
        
        assert assert_bfloat16_model(object()) is None
