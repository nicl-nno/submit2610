'Offline tests for scripts/make_yarn_model.py (no network, no GPU).\n\n\nThe script lives outside the package — loaded from its path via ``importlib``\n(the test_scripts convention). All tests run against a fake model directory\nin ``tmp_path``: a minimal Qwen2 ``config.json`` plus dummy weight/tokenizer\nfiles.\ntest_scripts'

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_yarn_model.py"


@pytest.fixture(scope="module")
def mym():
    spec = importlib.util.spec_from_file_location("_make_yarn_model", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_make_yarn_model"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def model_dir(tmp_path):
    'A fake Qwen2 model dir: config.json + two dummy files + one subdir.'
    d = tmp_path / "model"
    d.mkdir()
    cfg = {
        "architectures": ["Qwen2ForCausalLM"],
        "max_position_embeddings": 32768,
        "rope_scaling": None,
        "rope_theta": 1000000.0,
        "vocab_size": 145152,
    }
    (d / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (d / "model-00001-of-00002.safetensors").write_bytes(b"weights-1")
    (d / "tokenizer.json").write_bytes(b"tokenizer")
    (d / "nested").mkdir()
    (d / "nested" / "ignored.bin").write_bytes(b"ignored")
    return d


class TestBuildOverlay:
    def test_builds_symlinks_and_patched_config(self, mym, model_dir, tmp_path):
        out = tmp_path / "model-yarn"
        assert mym.build_overlay(model_dir, out) == "built"
        # Non-config files become symlinks resolving into the source dir;
        # subdirectories are not linked (model dirs are flat).
        for name in ("model-00001-of-00002.safetensors", "tokenizer.json"):
            link = out / name
            assert link.is_symlink()
            assert link.resolve() == (model_dir / name).resolve()
        assert not (out / "nested").exists()
        # config.json is a real file with exactly the Qwen2.5-card block.
        cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
        assert cfg["rope_scaling"] == {
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "type": "yarn",
        }
        # max_position_embeddings untouched; the other keys survive.
        assert cfg["max_position_embeddings"] == 32768
        assert cfg["rope_theta"] == 1000000.0
        assert cfg["vocab_size"] == 145152

    def test_idempotent_reuse(self, mym, model_dir, tmp_path):
        out = tmp_path / "model-yarn"
        mym.build_overlay(model_dir, out)
        assert mym.build_overlay(model_dir, out) == "reused"

    def test_force_rebuilds_valid_overlay(self, mym, model_dir, tmp_path):
        out = tmp_path / "model-yarn"
        mym.build_overlay(model_dir, out)
        assert mym.build_overlay(model_dir, out, force=True) == "built"

    def test_stale_overlay_is_rebuilt(self, mym, model_dir, tmp_path):
        out = tmp_path / "model-yarn"
        mym.build_overlay(model_dir, out)
        cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
        cfg["rope_scaling"]["factor"] = 2.0
        (out / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        assert mym.build_overlay(model_dir, out) == "built"
        cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
        assert cfg["rope_scaling"]["factor"] == 4.0

    def test_stale_overlay_with_real_files_refuses_without_force(
        self, mym, model_dir, tmp_path,
    ):
        out = tmp_path / "model-yarn"
        out.mkdir()
        (out / "config.json").write_text("{}", encoding="utf-8")
        (out / "model-00001-of-00002.safetensors").write_bytes(b"real weights")
        with pytest.raises(SystemExit, match="--force"):
            mym.build_overlay(model_dir, out)
        assert (out / "model-00001-of-00002.safetensors").exists()
        assert mym.build_overlay(model_dir, out, force=True) == "built"

    def test_refuses_out_equal_to_model(self, mym, model_dir):
        with pytest.raises(SystemExit):
            mym.build_overlay(model_dir, model_dir)

    def test_refuses_out_containing_model(self, mym, model_dir, tmp_path):
        with pytest.raises(SystemExit):
            mym.build_overlay(model_dir, tmp_path)


class TestValidateSourceConfig:
    def test_rejects_missing_config(self, mym, tmp_path):
        with pytest.raises(SystemExit):
            mym.build_overlay(tmp_path, tmp_path / "out")

    def test_rejects_non_qwen2_arch(self, mym):
        with pytest.raises(SystemExit, match="Qwen2"):
            mym.validate_source_config({
                "architectures": ["LlamaForCausalLM"],
                "max_position_embeddings": 32768,
                "rope_scaling": None,
            })

    def test_rejects_existing_rope_scaling(self, mym):
        with pytest.raises(SystemExit, match="rope_scaling"):
            mym.validate_source_config({
                "architectures": ["Qwen2ForCausalLM"],
                "max_position_embeddings": 32768,
                "rope_scaling": {"type": "yarn", "factor": 4.0},
            })

    def test_rejects_factor_below_one(self, mym, model_dir, tmp_path):
        with pytest.raises(SystemExit, match="factor"):
            mym.build_overlay(model_dir, tmp_path / "out", factor=1.0)


class TestMain:
    def test_end_to_end(self, mym, model_dir, tmp_path, capsys):
        out = tmp_path / "model-yarn"
        rc = mym.main(["--model", str(model_dir), "--out", str(out)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "built" in captured.out
        assert "131072" in captured.out
        # The reminder about mergekit / static YaRN is printed.
        assert "mergekit" in captured.out
        # Second call reuses and still exits 0.
        assert mym.main(["--model", str(model_dir), "--out", str(out)]) == 0
        assert "reused" in capsys.readouterr().out

    def test_default_out_is_model_plus_yarn(self, mym, model_dir, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert mym.main(["--model", str(model_dir)]) == 0
        assert (model_dir.parent / f"{model_dir.name}-yarn" / "config.json").is_file()
