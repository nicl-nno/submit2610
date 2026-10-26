'Offline tests for lm_eval_tasks/scrolls_local.py (no network, no GPU).\n\n\nThe module lives outside the package — loaded from its path via\n``importlib`` (the same convention as ``tests/test_scripts``). The hub\ndownload is replaced by a fake returning a locally fabricated zip; the\nyaml↔class consistency of the shadow tasks is checked textually (without\nlm_eval) and, when lm_eval is importable, structurally.'

from __future__ import annotations

import importlib.util
import json
import re
import sys
import zipfile
from pathlib import Path

import pytest

LM_EVAL_TASKS_DIR = Path(__file__).resolve().parents[2] / "lm_eval_tasks"

EXPECTED_SHADOWS = {
    "scrolls_govreport": "scrolls_local.GovReportLocal",
    "scrolls_summscreenfd": "scrolls_local.SummScreenFDLocal",
    "scrolls_qmsum": "scrolls_local.QMSumLocal",
    "scrolls_qasper": "scrolls_local.QasperLocal",
    "scrolls_quality": "scrolls_local.QuALITYLocal",
    "scrolls_narrativeqa": "scrolls_local.NarrativeQALocal",
    "scrolls_contractnli": "scrolls_local.ContractNLILocal",
    "scrolls_narrativeqa_filtered": "scrolls_narrativeqa_filtered.NarrativeQAFiltered",
}


@pytest.fixture(scope="module")
def sl():
    'The scrolls_local module imported BY NAME (as the sibling task modules\n    and ``python scrolls_local.py`` see it).'
    if str(LM_EVAL_TASKS_DIR) not in sys.path:
        sys.path.insert(0, str(LM_EVAL_TASKS_DIR))
    import scrolls_local as mod

    return mod


def _make_source_zip(tmp_path: Path, name: str) -> Path:
    'Fabricate a tau/scrolls-style zip: ``<name>/{train,validation}.jsonl``.'
    zip_path = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for split in ("train", "validation"):
            rows = [
                {"id": f"{name}-{split}-{i}", "pid": f"{name}-{split}-{i}",
                 "input": f"input {i}\n\nQuestion: q{i}", "output": f"out {i}"}
                for i in range(3)
            ]
            zf.writestr(f"{name}/{split}.jsonl", "\n".join(json.dumps(r) for r in rows))
    return zip_path


class TestExtractedDir:
    def test_unpacks_once_and_is_idempotent(self, sl, tmp_path, monkeypatch):
        name = "gov_report"
        zip_path = _make_source_zip(tmp_path, name)
        calls = []

        def fake_download(**kwargs):
            calls.append(kwargs)
            return str(zip_path)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))

        data_dir = Path(sl._extracted_dir(name))

        assert data_dir.is_dir()
        assert (data_dir / "train.jsonl").is_file()
        assert (data_dir / "validation.jsonl").is_file()
        assert (data_dir.parent / ".done").is_file()
        assert len(calls) == 1
        assert calls[0]["repo_id"] == "tau/scrolls"
        assert calls[0]["filename"] == f"{name}.zip"

        again = Path(sl._extracted_dir(name))
        assert again == data_dir
        assert len(calls) == 1  # no re-download

    def test_short_circuits_on_done_marker(self, sl, tmp_path, monkeypatch):
        name = "qmsum"
        calls = []

        def fake_download(**kwargs):
            calls.append(kwargs)
            return str(_make_source_zip(tmp_path, name))

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))

        root = tmp_path / "hfhome" / "tau_scrolls_extracted" / name
        (root / name).mkdir(parents=True)
        (root / ".done").write_text("", encoding="utf-8")

        assert Path(sl._extracted_dir(name)) == root / name
        assert calls == []  # marker wins

    def test_concurrent_winner_result_is_kept(self, sl, tmp_path, monkeypatch):
        name = "quality"
        zip_path = _make_source_zip(tmp_path, name)

        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download", lambda **kwargs: str(zip_path)
        )
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))

        root = tmp_path / "hfhome" / "tau_scrolls_extracted" / name
        root.mkdir(parents=True)
        (root / "foreign_file").write_bytes(b"winner")  # rename fails

        out = Path(sl._extracted_dir(name))
        assert (root / ".done").is_file()
        assert (root / "foreign_file").read_bytes() == b"winner"
        assert not any(p.name.startswith(f"{name}.tmp.") for p in root.parent.iterdir())
        assert out == root / name


class TestShadows:
    def test_subsets_are_the_seven_scrolls_configs(self, sl):
        assert set(sl.SCROLLS_SUBSETS) == {
            "summ_screen_fd", "qasper", "qmsum", "narrative_qa",
            "gov_report", "contract_nli", "quality",
        }

    def test_yaml_task_and_class_fields(self):
        for yaml_path in sorted(LM_EVAL_TASKS_DIR.glob("scrolls_*.yaml")):
            text = yaml_path.read_text(encoding="utf-8")
            task = re.search(r"^task:\s*(\S+)$", text, re.MULTILINE).group(1)
            klass = re.search(r"^class:\s*!function\s+(\S+)$", text, re.MULTILINE).group(1)
            assert task == yaml_path.stem
            assert task in EXPECTED_SHADOWS
            assert klass == EXPECTED_SHADOWS[task]

    def test_shadow_classes_subclass_stock_and_mixin(self, sl):
        pytest.importorskip("lm_eval")
        if not hasattr(sl, "GovReportLocal"):
            pytest.skip("lm_eval без transformers / lm_eval without transformers")
        stock = {
            "GovReportLocal": "GovReport",
            "SummScreenFDLocal": "SummScreenFD",
            "QMSumLocal": "QMSum",
            "QasperLocal": "Qasper",
            "QuALITYLocal": "QuALITY",
            "NarrativeQALocal": "NarrativeQA",
            "ContractNLILocal": "ContractNLI",
        }
        import lm_eval.tasks.scrolls.task as stock_mod

        for local_name, stock_name in stock.items():
            local_cls = getattr(sl, local_name)
            stock_cls = getattr(stock_mod, stock_name)
            assert issubclass(local_cls, stock_cls)
            assert issubclass(local_cls, sl._LocalZipDownloadMixin)
            # download comes from the mixin
            assert local_cls.download is sl._LocalZipDownloadMixin.download

    def test_filtered_task_subclasses_local_narrativeqa(self, sl):
        pytest.importorskip("lm_eval")
        if not hasattr(sl, "NarrativeQALocal"):
            pytest.skip("lm_eval без transformers / lm_eval without transformers")
        spec = importlib.util.spec_from_file_location(
            "_scrolls_narrativeqa_filtered",
            LM_EVAL_TASKS_DIR / "scrolls_narrativeqa_filtered.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_scrolls_narrativeqa_filtered"] = mod
        spec.loader.exec_module(mod)
        assert issubclass(mod.NarrativeQAFiltered, sl.NarrativeQALocal)
        assert mod.NarrativeQAFiltered.PRUNE_MAX_TOKENS == 30000
        assert mod.NarrativeQAFiltered.PRUNE_NUM_PROC == 8
