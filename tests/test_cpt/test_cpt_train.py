'Tests for cpt_train.py / cpt_tokenize.py plumbing (offline, no trainer).'

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, script: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ct = _load_module("_cpt_train", "cpt_train.py")
cptok = _load_module("_cpt_tokenize", "cpt_tokenize.py")


class TestCptTrainParser:
    def test_repeatable_data_args(self):
        # The data paths are cpt_tokenize.py flags; cpt_train.py consumes
        # the cache they produce.
        
        
        args = cptok.build_arg_parser().parse_args([
            "--sft-data", "a.jsonl", "--sft-data", "b.jsonl",
            "--pt-data", "c.jsonl",
            "--val-sft-data", "v.jsonl",
        ])
        assert args.sft_data == ["a.jsonl", "b.jsonl"]
        assert args.pt_data == ["c.jsonl"]
        assert args.val_sft_data == ["v.jsonl"]

    def test_defaults(self):
        args = ct.build_arg_parser().parse_args([])
        assert args.attn == "sdpa"
        assert args.tokenizer is None
        assert not args.freeze_transformer
        assert not args.liger
        assert args.patience == 3
        assert args.eval_steps == 0
        assert args.max_steps == 0

    def test_max_shard_size_defaults_to_5gb(self):
        # The checkpoint discipline: shards pinned at 5 GB regardless of the
        # transformers version (v5.2+ changed the library default to 50 GB).
        
        
        args = ct.build_arg_parser().parse_args([])
        assert args.max_shard_size == "5GB"

    def test_max_shard_size_flag_parses(self):
        args = ct.build_arg_parser().parse_args(["--max-shard-size", "2GB"])
        assert args.max_shard_size == "2GB"


class TestBuildTrainerKwargs:
    def test_warmup_ratio_kept_when_declared(self):
        @dataclass
        class Cfg:
            output_dir: str = ""
            warmup_ratio: float = 0.0
            bf16: bool = False

        kwargs = ct._build_trainer_kwargs(
            Cfg, output_dir="o", epochs=1, max_steps=0, lr=1e-5,
            per_device_batch=2, grad_accum=8, optim="adamw_torch", seed=42,
            dataloader_num_workers=0,
        )
        assert kwargs["warmup_ratio"] == 0.03
        # Unknown-to-Cfg kwargs dropped, known ones kept.
        
        assert "output_dir" in kwargs
        assert "logging_steps" not in kwargs

    def test_warmup_steps_fallback(self):
        @dataclass
        class Cfg:
            output_dir: str = ""
            warmup_steps: int = 0

        kwargs = ct._build_trainer_kwargs(
            Cfg, output_dir="o", epochs=1, max_steps=0, lr=1e-5,
            per_device_batch=2, grad_accum=8, optim="adamw_torch", seed=42,
            dataloader_num_workers=0,
        )
        # v5 semantics: a float < 1 is a ratio of steps.
        
        assert kwargs["warmup_steps"] == 0.03
        assert "warmup_ratio" not in kwargs

    def test_max_steps_passthrough(self):
        @dataclass
        class Cfg:
            output_dir: str = ""
            max_steps: int = -1

        kwargs = ct._build_trainer_kwargs(
            Cfg, output_dir="o", epochs=1, max_steps=30, lr=1e-5,
            per_device_batch=2, grad_accum=8, optim="adamw_torch", seed=42,
            dataloader_num_workers=0,
        )
        assert kwargs["max_steps"] == 30


class TestFilterValRows:
    def _rows(self):
        return [
            {"n_reference_tokens": 10, "source": f"v#{i}"} for i in range(8)
        ] + [{"n_reference_tokens": 9999, "source": "v#long"}]

    def test_budget_filter(self):
        kept = ct.filter_val_rows(self._rows(), max_new_tokens=100,
                                  max_examples=0, seed=42)
        assert all(r["n_reference_tokens"] <= 100 for r in kept)
        assert len(kept) == 8

    def test_max_examples_seeded(self):
        a = ct.filter_val_rows(self._rows(), max_new_tokens=100,
                               max_examples=3, seed=42)
        b = ct.filter_val_rows(self._rows(), max_new_tokens=100,
                               max_examples=3, seed=42)
        c = ct.filter_val_rows(self._rows(), max_new_tokens=100,
                               max_examples=3, seed=43)
        assert [r["source"] for r in a] == [r["source"] for r in b]
        assert [r["source"] for r in a] != [r["source"] for r in c]
        assert len(a) == 3


class TestKindIndices:
    def test_indices_from_arrow_column(self):
        column = pa.chunked_array([pa.array(["sft", "pt", "sft", "pt", "pt"])])
        assert ct._kind_indices(column, "sft") == [0, 2]
        assert ct._kind_indices(column, "pt") == [1, 3, 4]


class TestExclusiveLock:
    def test_lock_excludes_second_fd(self, tmp_path):
        # While the lock is held, a second non-blocking flock on the same
        # file (another rank) must fail — the donor loads serialize.
        
        
        import fcntl

        lock_path = tmp_path / ".surgery.lock"
        with ct._exclusive_lock(lock_path):
            with open(lock_path, "w") as f, pytest.raises(BlockingIOError):
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Released after the context: a fresh acquisition succeeds.
        
        with ct._exclusive_lock(lock_path):
            pass


class TestTokenizeFingerprint:
    def test_sources_match(self, tmp_path):
        f = tmp_path / "a.jsonl"
        f.write_text("{}", encoding="utf-8")
        desc = [cptok._file_descriptor(f)]
        sources = {"sft": desc, "pt": [], "val": []}
        meta = {"version": 1, "tokenizer": "t", "max_len": 64,
                "sources": sources}
        assert cptok._sources_match(meta, "t", 64, sources)
        assert not cptok._sources_match(meta, "t", 32, sources)
        assert not cptok._sources_match(meta, "other", 64, sources)
        changed = {"sft": [dict(desc[0], size=1)], "pt": [], "val": []}
        assert not cptok._sources_match(meta, "t", 64, changed)

    def test_parser(self):
        args = cptok.build_arg_parser().parse_args([
            "--pt-data", "p.jsonl", "--max-len", "128", "--num-proc", "4",
        ])
        assert args.pt_data == ["p.jsonl"]
        assert args.max_len == 128
        assert args.num_proc == 4
