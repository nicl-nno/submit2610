'Tests for scripts/make_safetensors_model.py (offline, a fake snapshot).\n\n\nThe converter solves the vllm_env anchor failure: transformers >= 4.52\nrefuses ``torch.load`` of ``.bin`` checkpoints on torch < 2.6\n(CVE-2025-32434) and the ``BAAI/bge-m3`` HF repo has no\n``model.safetensors``. The tests run against a FAKE snapshot directory\n(a tiny ``pytorch_model.bin`` written by ``torch.save`` + the aux files);\n``huggingface_hub.snapshot_download`` is monkeypatched — no network.'

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_safetensors_model.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_make_safetensors_model", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sm = _load_module()


@pytest.fixture
def fake_snapshot(tmp_path):
    """A .bin-only model snapshot with the ST scaffolding + stray heads."""
    """Снапшот модели только с .bin, обвязкой ST и лишними головами."""
    snap = tmp_path / "snapshot"
    (snap / "1_Pooling").mkdir(parents=True)
    state = {"encoder.w": torch.eye(3), "encoder.b": torch.arange(3, dtype=torch.float32)}
    torch.save(state, snap / "pytorch_model.bin")
    (snap / "config.json").write_text(json.dumps({"model_type": "xlm-roberta"}))
    (snap / "tokenizer.json").write_text("{}")
    (snap / "modules.json").write_text("[]")
    (snap / "sentence_bert_config.json").write_text("{}")
    (snap / "1_Pooling" / "config.json").write_text("{}")
    # Heavy optional heads that must NOT be copied to --out.
    
    (snap / "colbert_linear.pt").write_bytes(b"pretend-weights")
    return snap, state


def _patch_hub(monkeypatch, snap):
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kwargs: str(snap)
    )


class TestRefusals:
    def test_refuses_on_old_torch(self, monkeypatch, tmp_path, capsys):
        # The converter itself performs the torch.load — on torch < 2.6 it
        # must refuse (running the vulnerable load would defeat the gate).
        
        monkeypatch.setattr(sm, "_torch_release", lambda: (2, 5, 1))
        rc = sm.main(["--out", str(tmp_path / "out")])
        assert rc == 2
        assert "CVE-2025-32434" in capsys.readouterr().err

    def test_refuses_when_no_bin(self, monkeypatch, tmp_path, capsys):
        # A source that already ships safetensors has nothing to convert.
        
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setattr(sm, "_torch_release", lambda: (2, 8, 0))
        _patch_hub(monkeypatch, empty)
        rc = sm.main(["--out", str(tmp_path / "out")])
        assert rc == 2
        assert "pytorch_model.bin" in capsys.readouterr().err


class TestConversion:
    def test_converts_copies_aux_and_excludes_bin(
        self, monkeypatch, tmp_path, fake_snapshot
    ):
        snap, state = fake_snapshot
        monkeypatch.setattr(sm, "_torch_release", lambda: (2, 8, 0))
        _patch_hub(monkeypatch, snap)
        out = tmp_path / "out"
        assert sm.main(["--source", "BAAI/bge-m3", "--out", str(out)]) == 0

        # The converted weights are bit-equal and carry the pt metadata
        # transformers requires of safetensors files.
        
        
        from safetensors import safe_open
        from safetensors.torch import load_file

        loaded = load_file(str(out / "model.safetensors"))
        assert set(loaded) == set(state)
        for key, value in state.items():
            assert torch.equal(loaded[key], value)
        with safe_open(str(out / "model.safetensors"), framework="pt") as f:
            assert f.metadata()["format"] == "pt"

        # The ST scaffolding (incl. the nested 1_Pooling) rides along.
        
        for rel in (
            "config.json",
            "tokenizer.json",
            "modules.json",
            "sentence_bert_config.json",
            "1_Pooling/config.json",
        ):
            assert (out / rel).is_file(), rel
        # The heavy originals never land in the output.
        
        assert not (out / "pytorch_model.bin").exists()
        assert not (out / "colbert_linear.pt").exists()

    def test_local_snapshot_path_as_source(
        self, monkeypatch, tmp_path, fake_snapshot
    ):
        # --source <local dir> bypasses the hub entirely (no monkeypatch
        # needed — the hub must not even be imported for the resolution).
        
        snap, _state = fake_snapshot
        monkeypatch.setattr(sm, "_torch_release", lambda: (2, 8, 0))
        out = tmp_path / "out"
        assert sm.main(["--source", str(snap), "--out", str(out)]) == 0
        assert (out / "model.safetensors").is_file()

    def test_usage_hint_printed(self, monkeypatch, tmp_path, fake_snapshot, capsys):
        snap, _state = fake_snapshot
        monkeypatch.setattr(sm, "_torch_release", lambda: (2, 8, 0))
        _patch_hub(monkeypatch, snap)
        out = tmp_path / "out"
        assert sm.main(["--out", str(out)]) == 0
        text = capsys.readouterr().out
        assert "EMBEDDING_MODEL" in text
        assert "--local-embedding-model" in text
