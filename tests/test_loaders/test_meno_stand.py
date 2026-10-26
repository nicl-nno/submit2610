'Tests for the Meno-stand distillation loader (rerank/rewrite/final).\n\n\nOffline against the committed catalog: kind detection by the system-prompt\nmarkers, the fixed-val tagging of the test split, gold_label pre-parsing and\nthe malformed-record skip (one rewrite row has an empty response).'

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_training.config import DataConfig
from llm_training.loaders.meno_stand import _detect_kind, load_class

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cfg() -> DataConfig:
    return DataConfig(data_dir=REPO_ROOT / "src_data")


def _tok():
    return MagicMock()  # the loader never tokenizes


def test_catalog_present():  # pragma: no cover - committed with the repo
    for split in ("train", "test"):
        assert (REPO_ROOT / "src_data" / "meno_stand_distil" / f"{split}.jsonl").is_file()


class TestDetectKind:
    def test_rerank(self):
        assert _detect_kind({"system": "…оценивает, насколько документ пригоден…"}) == "rerank"

    def test_rewrite(self):
        assert _detect_kind({"system": "Вы — модуль переписывания вопросов (Query Rewriter)…"}) == "rewrite"

    def test_final(self):
        assert _detect_kind({"system": "Вы — Менон, виртуальный помощник AU…"}) == "final"


class TestLoadClass:
    def test_counts_and_tags(self, cfg):
        rerank = list(load_class("meno_rerank", _tok(), cfg))
        rewrite = list(load_class("meno_rewrite", _tok(), cfg))
        final = list(load_class("meno_final", _tok(), cfg))
        # 204 source rows; one rewrite record has an EMPTY response (train
        # #16) and is skipped as malformed → 186 + 8 + 9 = 203.
        
        assert len(rerank) == 186
        assert len(rewrite) == 8
        assert len(final) == 9
        # The test split carries fixed_val (63 + 4 + 4 = 71 anchor rows).
        
        assert sum(1 for e in rerank if e.extra.get("fixed_val")) == 63
        assert sum(1 for e in rewrite if e.extra.get("fixed_val")) == 4
        assert sum(1 for e in final if e.extra.get("fixed_val")) == 4
        # Train rows never carry fixed_val.
        assert all(not e.extra.get("fixed_val") for e in final if "train" in e.source)

    def test_gold_label_parsed(self, cfg):
        rerank = list(load_class("meno_rerank", _tok(), cfg))
        assert all(e.extra.get("gold_label") in (0, 1, 2) for e in rerank)
        # The heterogeneous references (9/186 with a space after the colon)
        # still parse.
        vals = {e.extra["gold_label"] for e in rerank}
        assert vals == {0, 1, 2}

    def test_meno_task_tag(self, cfg):
        for cls, want in (
            ("meno_rerank", "rerank"),
            ("meno_rewrite", "rewrite"),
            ("meno_final", "final"),
        ):
            exs = list(load_class(cls, _tok(), cfg))
            assert all(e.extra.get("meno_task") == want for e in exs)
            assert all(e.task_class == cls for e in exs)

    def test_source_is_split_addressed(self, cfg):
        final = list(load_class("meno_final", _tok(), cfg))
        assert all(
            e.source.startswith("meno_stand_distil/") and ".jsonl#" in e.source
            for e in final
        )

    def test_prompt_is_message_list(self, cfg):
        exs = list(load_class("meno_final", _tok(), cfg))
        assert exs[0].prompt and exs[0].prompt[0].get("role") == "system"
        assert exs[0].reference  # the response string

    def test_class_isolation(self, cfg):
        # load_class(meno_final) must never leak rerank/rewrite rows.
        # load_class(meno_final
        final = list(load_class("meno_final", _tok(), cfg))
        for e in final:
            assert "Query Rewriter" not in e.prompt[0].get("content", "")
            assert "пригоден" not in e.prompt[0].get("content", "")
