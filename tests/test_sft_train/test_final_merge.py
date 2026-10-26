'Tests for sft_train.py\'s final adapter merge (offline, tiny Qwen2 + LoRA).\n\n\nRegression scope\n``last-merged`` step called ``unwrap_trainer_model(trainer.MODEL)`` (the peft\nwrapper) instead of ``unwrap_trainer_model(trainer)``; the helper\'s\n``getattr(peft_model, "model")`` then resolved through peft\'s ``__getattr__``\nchain to the BARE ``Qwen2ForCausalLM`` (and ``accelerator`` to ``None``), so\n``.merge_and_unload()`` died with ``AttributeError`` AFTER a full 782-step\nrun. The tests below run the extracted ``_merge_final_adapter`` helper\nthrough the SAME fake-trainer shape (a namespace with ``.model`` and NO\n``accelerator`` — the fallback path that broke) and would fail with exactly\nthat error if the call shape regressed.\n``unwrap_trainer_model(trainer.MODEL)`` (peft\n``unwrap_trainer_model(trainer)``; ``getattr(peft_model, "model")``'

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Qwen2Config

_SCRIPT = Path(__file__).resolve().parents[2] / "sft_train.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_sft_train_merge", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


st = _load_module()


class _FakeTokenizer:
    'A tokenizer stand-in that records its save_pretrained directory.'

    def __init__(self):
        self.saved_to: list[str] = []

    def save_pretrained(self, out_dir, **kwargs):
        self.saved_to.append(str(out_dir))
        Path(out_dir).mkdir(parents=True, exist_ok=True)


def _tiny_peft_model(seed: int = 0):
    'A tiny bf16 Qwen2 wrapped in LoRA with a NON-zero B (merge ≠ no-op).'
    torch.manual_seed(seed)
    config = Qwen2Config(
        num_hidden_layers=2, hidden_size=64, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
    )
    base = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
    peft_model = get_peft_model(
        base,
        LoraConfig(
            r=8, lora_alpha=8, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        ),
    )
    # A fresh peft model has B = 0 (merge would be a numerical no-op) —
    # emulate a trained adapter with a non-zero B.
    
    
    for module in peft_model.modules():
        lora_b = getattr(module, "lora_B", None)
        if lora_b is not None:
            for adapter in lora_b:
                torch.nn.init.normal_(lora_b[adapter].weight, std=0.05)
    return peft_model


def _fake_trainer(peft_model) -> SimpleNamespace:
    # The PRODUCTION shape: .model present, no .accelerator attribute —
    # unwrap_trainer_model falls back to trainer.model.
    # unwrap_trainer_model
    
    return SimpleNamespace(model=peft_model)


class TestMergeFinalAdapter:
    def test_merges_and_saves_full_model(self, tmp_path):
        peft_model = _tiny_peft_model()
        q = (
            peft_model.base_model.model.model.layers[0]
            .self_attn.q_proj.base_layer.weight.detach().clone()
        )
        b = (
            peft_model.base_model.model.model.layers[0]
            .self_attn.q_proj.lora_B["default"].weight.detach()
        )
        a = (
            peft_model.base_model.model.model.layers[0]
            .self_attn.q_proj.lora_A["default"].weight.detach()
        )
        trainer = _fake_trainer(peft_model)
        tokenizer = _FakeTokenizer()
        out_dir = tmp_path / "last-merged"

        merged = st._merge_final_adapter(trainer, tokenizer, out_dir, "5GB")

        # The merged object is the BARE causal LM, adapter-free.
        
        assert type(merged).__name__ == "Qwen2ForCausalLM"
        assert not any("lora_A" in n for n, _ in merged.named_parameters())
        # Files written: model shards + config; the tokenizer rides along.
        
        assert (out_dir / "config.json").is_file()
        assert list(out_dir.glob("*.safetensors"))
        assert tokenizer.saved_to == [str(out_dir)]
        # The merge actually changed the frozen weight: W' ≈ W + B @ A
        # (scaling = alpha / r = 1), NOT the pristine base.
        
        # (scaling = alpha
        merged_q = merged.model.layers[0].self_attn.q_proj.weight.detach()
        manual = q.float() + (b.float() @ a.float())
        assert not torch.equal(q, merged_q)
        assert torch.allclose(merged_q.float(), manual, atol=0.05)

    def test_with_oplora_projection(self, tmp_path):
        # The OPLoRA belt-and-braces: the projection's no-op repair before
        # the merge, exactly as main() passes it.
        
        
        from llm_training.oplora import OploraProjection

        peft_model = _tiny_peft_model(seed=1)
        projection = OploraProjection.from_peft_model(peft_model, k=4)
        projection.project_weights()
        trainer = _fake_trainer(peft_model)

        merged = st._merge_final_adapter(
            trainer, _FakeTokenizer(), tmp_path / "last-merged",
            "5GB", projection,
        )
        assert type(merged).__name__ == "Qwen2ForCausalLM"
        assert (tmp_path / "last-merged" / "config.json").is_file()

    def test_production_crash_shape_is_the_bare_base(self):
        # THE TRAP, pinned: unwrap_trainer_model must take the TRAINER. Fed
        # the peft MODEL instead (the 2026-09-26 bug), it returns the bare
        # base causal LM — on which merge_and_unload does not exist.
        
        
        
        from llm_training.runtime.checkpointing import unwrap_trainer_model

        peft_model = _tiny_peft_model(seed=2)
        trainer = _fake_trainer(peft_model)

        got = unwrap_trainer_model(trainer.model)
        assert type(got).__name__ == "Qwen2ForCausalLM"
        with pytest.raises(AttributeError, match="merge_and_unload"):
            got.merge_and_unload()

        # The helper itself must resolve the peft wrapper through the SAME
        # trainer — this is what the fixed call shape does.
        
        
        assert unwrap_trainer_model(trainer) is peft_model
