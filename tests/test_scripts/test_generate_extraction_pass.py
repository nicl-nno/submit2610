'Offline tests of the generative-pass driver (no network, no GPU).\n\n\nThe vLLM engine of ``make_vllm_generate_fn`` is intentionally NOT exercised\nhere (heavy + GPU): ``run_pass`` takes the generator as an injectable seam.\n``make_vllm_generate_fn``'

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.build_extraction_corpus import _load_generations
from scripts.generate_extraction_pass import (
    compact,
    load_done,
    load_rows,
    run_pass,
    select_rows,
    write_records,
)


def _row(row_id: str, task_class: str = "ragu_entity_extraction") -> dict:
    return {
        "id": row_id,
        "task_class": task_class,
        "prompt": [{"role": "user", "content": "q"}],
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- load/select
def test_load_rows_and_rejects_promptless(tmp_path: Path) -> None:
    p = tmp_path / "rows.jsonl"
    _write_rows(p, [_row("a"), _row("b")])
    rows = load_rows(str(p))
    assert [r["id"] for r in rows] == ["a", "b"]

    bad = tmp_path / "bad.jsonl"
    _write_rows(bad, [{"id": "x", "task_class": "c"}])
    with pytest.raises(SystemExit):
        load_rows(str(bad))

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_rows(str(empty))


def test_select_rows_filters_and_defaults_to_all() -> None:
    rows = [
        _row("s1", "ragu_entity_extraction"),
        _row("s2", "ragu_relation_extraction"),
        _row("e1", "ragu_artifact_extraction"),
    ]
    assert len(select_rows(rows, None)) == 3
    assert len(select_rows(rows, [])) == 3
    only = select_rows(rows, ["ragu_entity_extraction"])
    assert [r["id"] for r in only] == ["s1"]


# ------------------------------------------------------------- resume/contract
def test_run_pass_writes_builder_contract(tmp_path: Path) -> None:
    "The output file is readable by the builder's ``_load_generations``."
    out = tmp_path / "gens.jsonl"
    rows = [_row("a"), _row("b")]

    def fake_generate(chunk: list[dict]) -> list[str]:
        return [f'{{"resp": "{r["id"]}"}}' for r in chunk]

    report = run_pass(rows, str(out), fake_generate)
    assert report["n_generated"] == 2
    assert report["n_empty"] == 0
    gens = _load_generations(str(out))
    assert gens == {"a": '{"resp": "a"}', "b": '{"resp": "b"}'}


def test_run_pass_resume_skips_done(tmp_path: Path) -> None:
    out = tmp_path / "gens.jsonl"
    rows = [_row("a"), _row("b"), _row("c")]
    # A previous interrupted run already produced "a".
    write_records(str(out), [{"row_id": "a", "response": "old-a"}])
    seen: list[str] = []

    def fake_generate(chunk: list[dict]) -> list[str]:
        seen.extend(r["id"] for r in chunk)
        return ["resp-" + r["id"] for r in chunk]

    report = run_pass(rows, str(out), fake_generate)
    assert seen == ["b", "c"]  # "a" skipped
    assert report["n_done"] == 1
    assert report["n_generated"] == 2
    gens = load_done(str(out))
    assert gens == {"a": "old-a", "b": "resp-b", "c": "resp-c"}


def test_run_pass_no_resume_rewrites(tmp_path: Path) -> None:
    out = tmp_path / "gens.jsonl"
    write_records(str(out), [{"row_id": "a", "response": "stale"}])
    rows = [_row("a")]

    def fake_generate(chunk: list[dict]) -> list[str]:
        return ["fresh" for _ in chunk]

    report = run_pass(rows, str(out), fake_generate, resume=False)
    assert report["n_done"] == 0
    assert load_done(str(out)) == {"a": "fresh"}


def test_run_pass_tolerates_torn_line_and_compacts(tmp_path: Path) -> None:
    'A torn write from an interrupted run is skipped, then compacted away.'
    out = tmp_path / "gens.jsonl"
    out.write_text(
        '{"row_id": "a", "response": "ok-a"}\n'
        '{"row_id": "b", "respo',  # torn mid-write
        encoding="utf-8",
    )
    rows = [_row("a"), _row("b")]
    seen: list[str] = []

    def fake_generate(chunk: list[dict]) -> list[str]:
        seen.extend(r["id"] for r in chunk)
        return ["fresh-" + r["id"] for r in chunk]

    report = run_pass(rows, str(out), fake_generate)
    assert seen == ["b"]  # torn line skipped -> "b" regenerated
    assert report["n_generated"] == 1
    # The final file is clean: every line parses, one record per row.
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln]
    assert all(json.loads(ln) for ln in lines)
    assert load_done(str(out)) == {"a": "ok-a", "b": "fresh-b"}


def test_run_pass_compact_drops_duplicates_last_wins(tmp_path: Path) -> None:
    out = tmp_path / "gens.jsonl"
    write_records(str(out), [
        {"row_id": "a", "response": "first"},
        {"row_id": "a", "response": "second"},
        {"row_id": "b", "response": "b1"},
    ])
    gens = compact(str(out))
    assert gens == {"a": "second", "b": "b1"}
    # File matches the returned map exactly (no leftover duplicate).
    file_gens = _load_generations(str(out))
    assert file_gens == gens


def test_compact_removes_fully_corrupt_file(tmp_path: Path) -> None:
    out = tmp_path / "gens.jsonl"
    out.write_text("garbage\nnot json at all\n", encoding="utf-8")
    assert compact(str(out)) == {}
    assert not out.exists()  # consumers fail loudly, never see 0 silently


def test_load_done_skips_malformed_records(tmp_path: Path) -> None:
    out = tmp_path / "gens.jsonl"
    out.write_text(
        '{"row_id": "a", "response": "ok"}\n'
        '{"row_id": ""}\n'
        '{"row_id": "c", "response": ""}\n'
        '["not", "a", "dict"]\n'
        '{"id": "d", "completion": "legacy-key"}\n',
        encoding="utf-8",
    )
    gens = load_done(str(out))
    assert gens == {"a": "ok", "d": "legacy-key"}


# ---------------------------------------------------------- batching/limits
def test_run_pass_batches_and_limit(tmp_path: Path) -> None:
    out = tmp_path / "gens.jsonl"
    rows = [_row(f"r{i}") for i in range(5)]
    batch_sizes: list[int] = []

    def fake_generate(chunk: list[dict]) -> list[str]:
        batch_sizes.append(len(chunk))
        return ["x" for _ in chunk]

    report = run_pass(rows, str(out), fake_generate, batch=2, limit=4)
    assert report["n_selected"] == 4  # limit applied first
    assert batch_sizes == [2, 2]
    assert report["n_generated"] == 4


def test_run_pass_counts_empty_responses(tmp_path: Path) -> None:
    out = tmp_path / "gens.jsonl"
    rows = [_row("a"), _row("b")]

    def fake_generate(chunk: list[dict]) -> list[str]:
        return ["ok" if r["id"] == "a" else "  " for r in chunk]

    report = run_pass(rows, str(out), fake_generate)
    assert report["n_generated"] == 1
    assert report["n_empty"] == 1
    # Empty responses are NOT written (the consumers skip them anyway) — the
    # next run retries them.
    assert load_done(str(out)) == {"a": "ok"}


def test_run_pass_rejects_misaligned_generator(tmp_path: Path) -> None:
    out = tmp_path / "gens.jsonl"

    def bad_generate(chunk: list[dict]) -> list[str]:
        return ["only-one"]

    with pytest.raises(ValueError, match="aligned"):
        run_pass([_row("a"), _row("b")], str(out), bad_generate)


def test_run_pass_rejects_bad_batch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch"):
        run_pass([_row("a")], str(tmp_path / "g.jsonl"),
                 lambda c: ["x"] * len(c), batch=0)


def test_run_pass_task_class_filter(tmp_path: Path) -> None:
    out = tmp_path / "gens.jsonl"
    rows = [
        _row("s1", "ragu_entity_extraction"),
        _row("o1", "ragu_artifact_extraction"),
    ]
    seen: list[str] = []

    def fake_generate(chunk: list[dict]) -> list[str]:
        seen.extend(r["id"] for r in chunk)
        return ["x" for _ in chunk]

    report = run_pass(
        rows, str(out), fake_generate,
        task_classes=["ragu_entity_extraction"],
    )
    assert seen == ["s1"]
    assert report["n_selected"] == 1
    assert report["n_total_in_file"] == 1


# ------------------------------------------------------------------- CLI (main)
def test_main_end_to_end_with_seamed_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    '``main()`` wires args -> run_pass with the vLLM generator swapped out.'
    import scripts.generate_extraction_pass as mod

    rows_path = tmp_path / "rows.jsonl"
    _write_rows(rows_path, [
        _row("s1", "ragu_entity_extraction"),
        _row("o1", "ragu_artifact_extraction"),
    ])
    out = tmp_path / "gens.jsonl"

    def fake_factory(*args: Any, **kwargs: Any):
        def generate(chunk: list[dict]) -> list[str]:
            return ["resp-" + r["id"] for r in chunk]

        return generate

    monkeypatch.setattr(mod, "make_vllm_generate_fn", fake_factory)
    rc = mod.main([
        "--rows", str(rows_path),
        "--out", str(out),
        "--model", "/fake/model",
        "--task-class", "ragu_entity_extraction",
        "--seed", "7",
    ])
    assert rc == 0
    assert load_done(str(out)) == {"s1": "resp-s1"}
