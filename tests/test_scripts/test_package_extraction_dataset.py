'Offline tests of the Huggingface Datasets packaging (no network).'

from __future__ import annotations

import json

import pytest
import pyarrow.parquet as pq

from scripts.package_extraction_dataset import (
    compute_stats,
    main,
    read_rows,
)


def _rows() -> list[dict]:
    return [
        {
            "id": "ext-000000",
            "task_class": "ragu_entity_extraction",
            "prompt": [
                {"role": "system", "content": "Extract entities."},
                {"role": "user", "content": "Текст со вставками API."},
            ],
            "extra": {
                "style": "scientific",
                "source": "wikipedia/ru/x",
                "target_language": "ru",
                "code_switched": True,
            },
        },
        {
            "id": "qa-000000",
            "task_class": "fineweb_qa",
            "prompt": [{"role": "user", "content": "Question?"}],
            "reference": "Answer.",
            "extra": {"style": "qa", "target_language": "en"},
        },
    ]


def _write_rows(path, rows) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_read_rows_strict(tmp_path) -> None:
    good = tmp_path / "good.jsonl"
    _write_rows(good, _rows())
    assert len(read_rows([str(good)])) == 2
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"id": "x", "task_class": "y"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        read_rows([str(bad)])  # missing 'prompt'/'extra'


def test_package_roundtrip(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_rows(train, _rows())
    _write_rows(val, _rows()[:1])
    out = tmp_path / "ds"
    rc = main(
        [
            "--train-jsonl",
            str(train),
            "--val-jsonl",
            str(val),
            "--out-dir",
            str(out),
            "--dataset-name",
            "test-ds",
        ]
    )
    assert rc == 0
    # The hub layout: data/*.parquet + README.md + meta.json.
    assert (out / "data" / "train-00000-of-00001.parquet").is_file()
    assert (out / "data" / "val-00000-of-00001.parquet").is_file()
    assert (out / "README.md").is_file()
    assert (out / "meta.json").is_file()
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["train"]["rows"] == 2
    assert meta["train"]["code_switched"] == 1
    assert meta["val"]["rows"] == 1
    assert meta["inputs"]["train"][0]["sha256"]
    # LOSSLESS round-trip: the parquet restores the exact rows.
    table = pq.read_table(out / "data" / "train-00000-of-00001.parquet")
    assert table.column_names == [
        "id",
        "task_class",
        "prompt",
        "reference",
        "extra",
    ]
    restored = [
        {
            "id": table["id"][i].as_py(),
            "task_class": table["task_class"][i].as_py(),
            "prompt": json.loads(table["prompt"][i].as_py()),
            "reference": table["reference"][i].as_py(),
            "extra": json.loads(table["extra"][i].as_py()),
        }
        for i in range(table.num_rows)
    ]
    original = [
        json.loads(line)
        for line in train.read_text(encoding="utf-8").splitlines()
    ]
    # Contract: an absent reference comes back as an explicit None (the
    # column is fixed; semantically lossless for the consumers).
    for row in original:
        row.setdefault("reference", None)
    assert restored == original
    # The card carries the license table and the loading snippet.
    card = (out / "README.md").read_text(encoding="utf-8")
    assert "CC BY-SA 4.0" in card
    assert "load_dataset" in card
    assert "ODC-By" in card


def test_compute_stats() -> None:
    stats = compute_stats(_rows())
    assert stats["rows"] == 2
    assert stats["with_reference"] == 1
    assert stats["code_switched"] == 1
    assert stats["by_task_class"] == {
        "fineweb_qa": 1,
        "ragu_entity_extraction": 1,
    }
    assert stats["by_target_language"] == {"en": 1, "ru": 1}
    assert stats["by_style"] == {"qa": 1, "scientific": 1}
