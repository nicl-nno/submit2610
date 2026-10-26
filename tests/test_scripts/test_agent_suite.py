'Offline tests of the agent-suite builder and the recall probe (stage 2).\n\n\nCovers the stage-2 gates of\n``specifications/2026-10-17-agent-suite-glm53-validation.md``: per-family\nselection with gold requirements and hard refusal, the no-leak prompt\ncontract (the source text never enters the prompt), the recall probe math,\nand the served-model arg validation of the pass driver.'

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from scripts.build_agent_suite import (
    build_extraction_row,
    build_qa_row,
    seed_fragment,
)
from scripts.build_agent_suite import (
    main as suite_main,
)

# ------------------------------------------------------------- fixtures


def _extraction_row(i: int, with_gold: bool = True, entities: int = 1) -> dict:
    text = (
        f"Текст документа номер {i}. Москва является столицей России. "
        "В городе проживают миллионы жителей и работают музеи."
    )
    payload: dict = {
        "stage": "one_stage",
        "language": "ru",
        "ontology": "nerel",
        "text": text,
        "entity_types": ["CITY", "PERSON"],
        "relation_types": [],
        "signatures": [],
    }
    if with_gold:
        names = ["Москва", "Санкт-Петербург", "Anonymous City"]
        payload["pseudo_gold"] = {
            "entities": [
                {
                    "entity_name": f"{name} {i}",
                    "entity_type": "CITY",
                    "description": "город",
                }
                for name in names[: max(1, entities)]
            ],
            "relations": [],
        }
    return {
        "id": f"ext-val-{i:06d}",
        "task_class": "ragu_entity_extraction",
        "prompt": [
            {"role": "system", "content": "системный промпт"},
            {"role": "user", "content": text},
        ],
        "extra": {
            "style": "scientific",
            "source": f"wikipedia/ru/Документ{i}",
            "target_language": "ru",
            "extraction": payload,
        },
    }


def _summary_row(i: int, with_ref: bool = True) -> dict:
    payload: dict = {
        "kind": "entity",
        "entity_name": f"Сущность номер {i}",
        "entity_type": "CITY",
        "descriptions": ["Описание первое про город.", "Описание второе."],
    }
    if with_ref:
        payload["reference"] = {
            "entity_name": f"Сущность номер {i}",
            "description": "Сводное описание города.",
        }
    return {
        "id": f"ext-sum-e-{i:06d}",
        "task_class": "ragu_entity_summary",
        "prompt": [
            {"role": "system", "content": "системный промпт"},
            {"role": "user", "content": "материал описаний"},
        ],
        "extra": {
            "style": "literary",
            "source": "gutenberg/pg1533",
            "doc_id": f"doc-{i}",
            "target_language": "ru",
            "summary": payload,
        },
    }


def _qa_row(i: int) -> dict:
    return {
        "id": f"qa-{i:06d}",
        "task_class": "fineweb_qa",
        "prompt": [
            {"role": "system", "content": "системный промпт"},
            {"role": "user", "content": f"Вопрос номер {i}?"},
        ],
        "reference": f"Короткий ответ номер {i}.",
        "extra": {
            "style": "qa",
            "source": "fineweb2",
            "target_language": "ru",
            "qa": {
                "question": f"Вопрос номер {i}?",
                "answer": f"Короткий ответ номер {i}.",
                "with_context": False,
            },
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


# ------------------------------------------------------------ seed fragment


def test_seed_fragment_deterministic_and_bounded() -> None:
    text = "Первое предложение здесь. Второе предложение длиннее и раскрывает тему далее. Третье."
    assert seed_fragment(text, 220) == seed_fragment(text, 220)
    long_text = ("Очень длинное предложение без границы " * 20).strip()
    cut = seed_fragment(long_text, 100)
    assert len(cut) <= 101 and cut.endswith("…")


# ------------------------------------------------------------ suite builder


def _write_inputs(tmp_path: Path) -> dict[str, str]:
    extraction = [_extraction_row(i) for i in range(3)]
    extraction.append(_extraction_row(3, with_gold=False))
    summary = [_summary_row(i) for i in range(2)]
    qa = [_qa_row(i) for i in range(2)]
    digest_file = tmp_path / "index" / "digest.txt"
    digest_file.parent.mkdir(parents=True, exist_ok=True)
    digest_file.write_text("sha256:" + "0" * 64 + "\n", encoding="utf-8")
    return {
        "extraction": _write_jsonl(tmp_path / "in" / "val_annotated.jsonl", extraction),
        "summary": _write_jsonl(tmp_path / "in" / "val_summaries.jsonl", summary),
        "qa": _write_jsonl(tmp_path / "in" / "val_qa.jsonl", qa),
        "digest": str(digest_file),
    }


def test_suite_selection_shape_and_no_leak(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    out_dir = tmp_path / "suite"
    suite_main(
        [
            "--extraction-jsonl", inputs["extraction"],
            "--summary-jsonl", inputs["summary"],
            "--qa-jsonl", inputs["qa"],
            "--index-digest-file", inputs["digest"],
            "--per-family", "2",
            "--fragment-chars", "40",
            "--out-dir", str(out_dir),
        ]
    )
    rows = [
        json.loads(line)
        for line in (out_dir / "suite.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["counts"] == {"extraction": 2, "summary": 2, "qa": 2}
    assert meta["index_digest"].startswith("sha256:")
    assert len({row["id"] for row in rows}) == 6
    for row in rows:
        rendered = json.dumps(row, ensure_ascii=False)
        assert row["tools"][0]["function"]["name"] == "searchdocs"
        assert row["id"].split("-")[1] == row["family"][:3] or row["id"].startswith(
            f"suite-{row['family'][:3]}"
        )
        assert "finalanswertool" in rendered
        assert row["gold"]
    extraction = [row for row in rows if row["family"] == "extraction"]
    for row in extraction:
        user = row["prompt"][-1]["content"]
        
        assert "миллионы жителей" not in user
        assert row["extra"]["seed_fragment"] in user
        assert row["gold"]["pseudo_gold"]["entities"]
    summary = [row for row in rows if row["family"] == "summary"]
    for row in summary:
        user = row["prompt"][-1]["content"]
        assert "Сущность номер" in user  
        assert "Описание первое" not in user  
        assert row["gold"]["material"]["descriptions"]
        assert row["gold"]["reference"]["description"]
    qa = [row for row in rows if row["family"] == "qa"]
    for row in qa:
        assert row["gold"]["question"] in row["prompt"][-1]["content"]
        assert row["gold"]["reference"]


def test_suite_excludes_ungolded_rows(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    out_dir = tmp_path / "suite"
    suite_main(
        [
            "--extraction-jsonl", inputs["extraction"],
            "--summary-jsonl", inputs["summary"],
            "--qa-jsonl", inputs["qa"],
            "--index-digest-file", inputs["digest"],
            "--per-family", "2",
            "--fragment-chars", "40",
            "--out-dir", str(out_dir),
        ]
    )
    rows = [
        json.loads(line)
        for line in (out_dir / "suite.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    extraction_sources = {
        row["source_row_id"] for row in rows if row["family"] == "extraction"
    }
    assert "ext-val-000003" not in extraction_sources  


def test_max_gold_entities_boundary(tmp_path: Path) -> None:
    'The optional boundary: OFF by default, ON excludes dense rows.'
    dense = _extraction_row(0, entities=3)  
    normal = [_extraction_row(1), _extraction_row(2), _extraction_row(3, with_gold=False)]
    extraction_file = _write_jsonl(tmp_path / "ext.jsonl", [dense, *normal])
    summary_file = _write_jsonl(tmp_path / "sum.jsonl", [_summary_row(i) for i in range(3)])
    qa_file = _write_jsonl(tmp_path / "qa.jsonl", [_qa_row(i) for i in range(3)])
    digest = _write_jsonl(tmp_path / "digest.txt", [])  # placeholder
    Path(digest).write_text("sha256:" + "1" * 64 + "\n", encoding="utf-8")

    def run(out_dir: str, extra: list[str]) -> list[dict]:
        suite_main(
            [
                "--extraction-jsonl", extraction_file,
                "--summary-jsonl", summary_file,
                "--qa-jsonl", qa_file,
                "--index-digest-file", digest,
                "--per-family", "3",
                "--out-dir", out_dir,
                *extra,
            ]
        )
        return [
            json.loads(line)
            for line in (Path(out_dir) / "suite.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    
    rows = run(str(tmp_path / "s1"), [])
    dense_ids = {
        row["source_row_id"]
        for row in rows
        if row["family"] == "extraction" and row["source_row_id"] == "ext-val-000000"
    }
    assert dense_ids

    
    rows = run(str(tmp_path / "s2"), ["--max-gold-entities", "2", "--per-family", "2"])
    extraction_sources = {
        row["source_row_id"] for row in rows if row["family"] == "extraction"
    }
    assert "ext-val-000000" not in extraction_sources


def test_suite_refuses_when_family_short(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    with pytest.raises(SystemExit):
        suite_main(
            [
                "--extraction-jsonl", inputs["extraction"],
                "--summary-jsonl", inputs["summary"],
                "--qa-jsonl", inputs["qa"],
                "--index-digest-file", inputs["digest"],
                "--per-family", "5",
                "--out-dir", str(tmp_path / "suite"),
            ]
        )


# ------------------------------------------------------------- recall probe


def _hash_encoder(dim: int = 32):
    def encode(texts: list[str]) -> np.ndarray:
        rows = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                number = int.from_bytes(digest, "big")
                rows[i, number % dim] += 1.0 + float((number >> 11) % 7)
        return rows

    return encode


def test_recall_probe_self_and_gold() -> None:
    from llm_training.extraction.retrieval import HybridIndex
    from scripts.measure_agent_suite_recall import evaluate_suite

    corpus = [_extraction_row(i) for i in range(3)]
    texts = [
        str(row["extra"]["extraction"]["text"]) for row in corpus
    ]
    ids = [row["id"] for row in corpus]
    index = HybridIndex.build(texts, ids, encode=_hash_encoder())
    texts_by_id = dict(zip(ids, texts))

    row = build_extraction_row(corpus[0], 220, "sha256:" + "0" * 64)
    qa = build_qa_row(_qa_row(0), "sha256:" + "0" * 64)
    
    alien_qa = build_qa_row(_qa_row(1), "sha256:" + "0" * 64)
    alien_qa["gold"]["reference"] = "Космический корабль приземлился на Марс."

    report = evaluate_suite(
        index, [row, qa, alien_qa], texts_by_id, top_ks=[1, 5], targets="all"
    )
    assert report["n_rows"] == 3
    assert report["variants"]["self"]["1"]["mean_recall"] == 1.0
    gold1 = report["variants"]["gold"]["1"]["mean_recall"]
    gold5 = report["variants"]["gold"]["5"]["mean_recall"]
    assert 0.0 <= gold1 <= 1.0 and gold5 >= gold1
    fam = report["per_family"]["qa"]["gold"]["5"]
    assert fam["n"] == 2


# --------------------------------------------------- pass driver: API mode


def test_api_generate_fn_survives_two_batches() -> None:
    'The 2026-10-17 pass-A crash: batch 2 died on a loop-bound semaphore.\n\n    the semaphore per batch; this test runs TWO consecutive generate calls.'
    import asyncio
    import types

    from scripts.generate_extraction_pass import make_api_generate_fn

    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            await asyncio.sleep(0)
            message = types.SimpleNamespace(content="ok")
            choice = types.SimpleNamespace(message=message, finish_reason="stop")
            return types.SimpleNamespace(choices=[choice])

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_FakeCompletions())
    )
    rows = [
        {"id": str(i), "prompt": [{"role": "user", "content": "задача"}]}
        for i in range(4)
    ]
    fn = make_api_generate_fn(
        "http://localhost:8024/v1", "m", "EMPTY", 2, 64, 42, client=client
    )
    first = fn(rows[:2])
    second = fn(rows[2:])
    assert first == ["ok", "ok"]
    assert second == ["ok", "ok"]


def test_pass_driver_requires_model_id_with_base_url(tmp_path: Path) -> None:
    from scripts.generate_extraction_pass import main as pass_main

    rows_file = _write_jsonl(
        tmp_path / "rows.jsonl",
        [
            {
                "id": "r1",
                "task_class": "ragu_entity_extraction",
                "prompt": [{"role": "user", "content": "задача"}],
            }
        ],
    )
    with pytest.raises(SystemExit):
        pass_main(
            [
                "--rows", rows_file,
                "--out", str(tmp_path / "gens.jsonl"),
                "--base-url", "http://localhost:8024/v1",
            ]
        )
    with pytest.raises(SystemExit):
        pass_main(
            [
                "--rows", rows_file,
                "--out", str(tmp_path / "gens.jsonl"),
            ]
        )
