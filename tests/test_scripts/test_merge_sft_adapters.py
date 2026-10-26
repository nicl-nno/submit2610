'Offline tests for scripts/merge_sft_adapters.py (no network, no GPU).\n\n\nThe script lives outside the package — loaded from its path via\n``importlib`` (the test_scripts convention). The end-to-end tests build a\nREAL tiny Qwen2 + LoRA pair in ``tmp_path`` (the base saved as a full model,\nthe adapter saved through peft with a NON-zero ``lora_B`` so the merge is\nnumerically visible; the tokenizer is constructed in-process — nothing is\ndownloaded), run ``main()`` in-process, and check the merged output: a plain\nfull model whose weights changed, plus the tokenizer copied from the adapter\ndirectory.\ntest_scripts'

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, models
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    Qwen2Config,
)

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "merge_sft_adapters.py"
)


@pytest.fixture(scope="module")
def msa():
    spec = importlib.util.spec_from_file_location("_merge_sft_adapters", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_merge_sft_adapters"] = mod
    spec.loader.exec_module(mod)
    return mod


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    'A minimal offline tokenizer (constructed, never downloaded).'
    inner = Tokenizer(
        models.WordLevel(vocab={"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=inner, pad_token="[PAD]", unk_token="[UNK]",
    )


@pytest.fixture()
def tiny_pair(tmp_path):
    'A tiny base model dir + a trained-looking adapter dir + the tokenizer.'
    torch.manual_seed(0)
    config = Qwen2Config(
        num_hidden_layers=2, hidden_size=64, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
    )
    base = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
    # Snapshot the pristine frozen weight BEFORE the wrap (the merge must
    # change it).
    
    
    q_before = base.model.layers[0].self_attn.q_proj.weight.detach().clone()
    tokenizer = _tiny_tokenizer()
    base_dir = tmp_path / "base"
    base.save_pretrained(str(base_dir))
    tokenizer.save_pretrained(str(base_dir))

    peft_model = get_peft_model(
        base,
        LoraConfig(
            r=8, lora_alpha=8, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        ),
    )
    # A fresh peft model has B = 0 — emulate a trained adapter.
    
    for module in peft_model.modules():
        lora_b = getattr(module, "lora_B", None)
        if lora_b is not None:
            for adapter in lora_b:
                torch.nn.init.normal_(lora_b[adapter].weight, std=0.05)
    adapter_dir = tmp_path / "best-60"
    peft_model.save_pretrained(str(adapter_dir))
    # sft_train.py saves the tokenizer INTO the adapter directory — mirror
    # that so the default tokenizer resolution picks it up.
    # sft_train.py
    
    tokenizer.save_pretrained(str(adapter_dir))

    return {
        "base_dir": base_dir,
        "adapter_dir": adapter_dir,
        "q_before": q_before,
    }


class TestMain:
    def test_merges_adapter_into_full_model(self, msa, tiny_pair, tmp_path):
        out_dir = tmp_path / "best-merged"
        rc = msa.main([
            "--model", str(tiny_pair["base_dir"]),
            "--adapter", str(tiny_pair["adapter_dir"]),
            "--output-dir", str(out_dir),
            "--device", "cpu",
        ])
        assert rc == 0
        merged = AutoModelForCausalLM.from_pretrained(
            str(out_dir), torch_dtype=torch.bfloat16,
        )
        q_after = merged.model.layers[0].self_attn.q_proj.weight.detach()
        assert not torch.equal(tiny_pair["q_before"], q_after)
        assert not any("lora_" in n for n, _ in merged.named_parameters())
        # The tokenizer came from the ADAPTER directory (the resolution
        # rule: adapter dir first, else the base).
        
        
        tok = AutoTokenizer.from_pretrained(str(out_dir))
        assert tok.pad_token == "[PAD]"

    def test_oplora_flag_runs(self, msa, tiny_pair, tmp_path):
        out_dir = tmp_path / "best-merged-oplora"
        rc = msa.main([
            "--model", str(tiny_pair["base_dir"]),
            "--adapter", str(tiny_pair["adapter_dir"]),
            "--output-dir", str(out_dir),
            "--oplora", "--oplora-k", "8",
            "--device", "cpu",
        ])
        assert rc == 0
        assert (out_dir / "config.json").is_file()
        assert list(out_dir.glob("*.safetensors"))

    def test_missing_adapter_dir_exits(self, msa, tiny_pair, tmp_path):
        with pytest.raises(SystemExit, match="не найден|not found"):
            msa.main([
                "--model", str(tiny_pair["base_dir"]),
                "--adapter", str(tmp_path / "no-such-adapter"),
                "--output-dir", str(tmp_path / "out"),
            ])

    def test_seed_makes_oplora_merge_reproducible(self, msa, tiny_pair, tmp_path):
        # torch.svd_lowrank draws a random start matrix: two runs with the
        # SAME --seed must produce bit-identical merged weights (the point
        # of the flag), regardless of the global RNG state before main().
        # torch.svd_lowrank
        
        
        torch.manual_seed(1234)
        out_a = tmp_path / "merged-a"
        assert msa.main([
            "--model", str(tiny_pair["base_dir"]),
            "--adapter", str(tiny_pair["adapter_dir"]),
            "--output-dir", str(out_a),
            "--oplora", "--oplora-k", "8", "--seed", "7",
            "--device", "cpu",
        ]) == 0
        torch.manual_seed(999)
        out_b = tmp_path / "merged-b"
        assert msa.main([
            "--model", str(tiny_pair["base_dir"]),
            "--adapter", str(tiny_pair["adapter_dir"]),
            "--output-dir", str(out_b),
            "--oplora", "--oplora-k", "8", "--seed", "7",
            "--device", "cpu",
        ]) == 0
        a = AutoModelForCausalLM.from_pretrained(
            str(out_a), torch_dtype=torch.bfloat16,
        )
        b = AutoModelForCausalLM.from_pretrained(
            str(out_b), torch_dtype=torch.bfloat16,
        )
        for (name_a, p_a), (name_b, p_b) in zip(
            a.named_parameters(), b.named_parameters()
        ):
            assert name_a == name_b
            assert torch.equal(p_a, p_b), f"weights diverged at {name_a}"


class TestBuildArgParser:
    def test_seed_default_is_42(self, msa):
        args = msa.build_arg_parser().parse_args(
            ["--model", "/m", "--adapter", "/a", "--output-dir", "/o"]
        )
        assert args.seed == 42


class TestResolveTokenizerPath:
    def test_explicit_flag_wins(self, msa, tmp_path):
        got = msa.resolve_tokenizer_path(
            "/explicit", tmp_path, "/base"
        )
        assert got == "/explicit"

    def test_adapter_dir_with_tokenizer_wins(self, msa, tmp_path):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        assert msa.resolve_tokenizer_path(None, adapter, "/base") == str(adapter)

    def test_falls_back_to_base(self, msa, tmp_path):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        assert msa.resolve_tokenizer_path(None, adapter, "/base") == "/base"
