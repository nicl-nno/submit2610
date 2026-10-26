'Tests for the sft_train.py CLI parser (offline, module import only).\n\n\nThe script is the grpo_train.py ablation twin: the DATA/anchor/typo/difficulty\nflags must stay mirrored, the GRPO/reward-only flags must be GONE, and the\nLoRA/OPLoRA flags carry the OPLoRA-paper defaults (r=64, alpha=64,\ndropout=0.05, k=128).\nGRPO/reward'

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "sft_train.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_sft_train_cli", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


st = _load_module()


class TestParserDefaults:
    def test_lora_on_by_default(self):
        args = st._build_arg_parser().parse_args([])
        assert args.no_lora is False
        assert args.lora_r == 64
        assert args.lora_alpha == 64
        assert args.lora_dropout == 0.05

    def test_lora_targets_default_paper_set(self):
        # The OPLoRA paper's module set for Qwen2.5-7B (no k_proj/gate_proj).
        
        args = st._build_arg_parser().parse_args([])
        assert set(args.lora_target_modules) == {
            "q_proj", "v_proj", "up_proj", "down_proj", "o_proj",
        }

    def test_oplora_off_by_default(self):
        args = st._build_arg_parser().parse_args([])
        assert args.oplora is False
        assert args.oplora_k == 128

    def test_length_and_batch_defaults(self):
        # 32768 keeps the whole dataset (the 32 000-token build rule binds);
        # batch 1 is the single-80GB-card budget.
        args = st._build_arg_parser().parse_args([])
        assert args.max_len == 32768
        assert args.per_device_batch == 1
        assert args.grad_accum == 64

    def test_lr_none_resolved_by_mode(self):
        # None → 2e-4 (LoRA) / 1e-5 (full FT), resolved in main().
        args = st._build_arg_parser().parse_args([])
        assert args.lr is None
        assert st.DEFAULT_LORA_LR == 2e-4
        assert st.DEFAULT_FULL_FT_LR == 1e-5

    def test_optim_and_attn_vllm_env_defaults(self):
        args = st._build_arg_parser().parse_args([])
        assert args.optim == "adamw_torch"  # LoRA-sized optimizer states
        assert args.attn == "sdpa"          # no flash-attn in vllm_env

    def test_data_flags_mirror_grpo_train(self):
        # The ablation twin knobs must default to the grpo_train.py values.
        
        args = st._build_arg_parser().parse_args([])
        assert args.val_task == ["summarizer", "math_reasoner"]
        assert args.max_per_class == 200
        assert args.max_upsample_ratio == 8.0
        assert args.difficulty_model == "anonymous/base-model"
        assert args.difficulty_drop_easy == 0.05
        assert args.difficulty_drop_hard == 0.01
        assert args.typo_share == 0.0
        assert args.system_typo_share == 0.0
        assert args.final_turn_share is None
        assert args.reasoning_keep_share is None
        assert args.seed == 42

    def test_anchor_flags_mirror_grpo_train(self):
        args = st._build_arg_parser().parse_args([])
        assert args.judge_anchor_every == 500
        assert args.anchor_patience == 3
        # The grpo_train r6 SLURM value: summarizer final turns carry the
        # whole answer inside a finalanswertool call; 2048 truncated them.
        assert args.anchor_max_new_tokens == 6144
        # The batched anchor generation: ON by default (LEFT-padded
        # packages under the KV budget); 0 = the legacy escape.
        
        assert args.anchor_gen_batch_tokens == 65536
        assert args.anchor_gen_max_batch == 8
        assert args.judge_concurrency == 8
        assert args.judge_max_tokens == 4096
        assert args.judge_timeout == 600.0
        assert args.drift_probe_gate == 0.10
        assert args.max_shard_size == "5GB"

    def test_prepared_dir_default_none(self):
        args = st._build_arg_parser().parse_args([])
        assert args.prepared_dir is None


class TestEmbeddingModelFlag:
    def test_default_none(self):
        # No flag → the anchor scorer uses the RewardConfig default
        # (BAAI/bge-m3); main() resolves flag > env > default.
        
        
        args = st._build_arg_parser().parse_args([])
        assert args.local_embedding_model is None

    def test_local_path_parses(self):
        args = st._build_arg_parser().parse_args(
            ["--local-embedding-model", "/path/to/storage/models/bge-m3"]
        )
        assert args.local_embedding_model == "/path/to/storage/models/bge-m3"

    def test_help_documents_env_and_cve_gate(self):
        import contextlib
        import io

        buf = io.StringIO()
        with (
            contextlib.redirect_stdout(buf),
            pytest.raises(SystemExit),
        ):
            st._build_arg_parser().parse_args(["--help"])
        text = buf.getvalue()
        # rsplit: the usage line also names the flag — the HELP entry is the
        # last occurrence.
        
        flag_block = text.rsplit("--local-embedding-model", 1)[1]
        flag_block = flag_block.split("\n\n", 1)[0]
        assert "LOCAL_EMBEDDING_MODEL" in flag_block
        assert "safetensors" in flag_block


class TestGrpoFlagsRemoved:
    @pytest.mark.parametrize("flag", [
        "--num-generations", "--max-completion-length", "--temperature",
        "--top-p", "--beta", "--vllm-mode", "--vllm-gpu-memory-utilization",
        "--format-epsilon", "--reasoning-semsim", "--no-schema-check",
        "--no-language-guard", "--no-fluency", "--no-rummlu-em",
        "--no-rerank-em",
    ])
    def test_grpo_and_reward_flags_are_gone(self, flag):
        # The GRPO/reward-only surface must not exist in the SFT twin.
        
        with pytest.raises(SystemExit):
            st._build_arg_parser().parse_args([flag, "1"])


class TestFlagCombinations:
    def test_oplora_with_lora_parses(self):
        args = st._build_arg_parser().parse_args(["--oplora", "--oplora-k", "16"])
        assert args.oplora is True
        assert args.oplora_k == 16

    def test_oplora_with_no_lora_rejected_in_main(self, capsys):
        # main() exits 2 before any heavy import — safe to run offline.
        assert st.main(["--oplora", "--no-lora"]) == 2
        assert "--oplora" in capsys.readouterr().err

    def test_no_lora_full_ft_escape(self):
        args = st._build_arg_parser().parse_args(["--no-lora"])
        assert args.no_lora is True

    def test_lora_knobs_overridable(self):
        args = st._build_arg_parser().parse_args(
            ["--lora-r", "32", "--lora-alpha", "32",
             "--lora-target-modules", "q_proj", "k_proj"]
        )
        assert args.lora_r == 32
        assert args.lora_alpha == 32
        assert args.lora_target_modules == ["q_proj", "k_proj"]
