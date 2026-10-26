"Offline tests of the one-off annotation script (no network, no GPU).\n\n\nThe two judge passes are monkeypatched seams: the script's job under test is\nthe DISPATCH (both modes over the same pool, report printing, exit codes,\nargv) — the annotation algorithms themselves live in\n:mod:`llm_training.runtime.pseudo_gold` and are tested there."

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_training.runtime.pseudo_gold import PseudoGoldReport
from scripts import annotate_pseudo_gold as script


def _row(row_id: str, extra: dict) -> dict:
    return {
        "id": row_id,
        "task_class": "ragu_entity_extraction",
        "prompt": [{"role": "user", "content": "q"}],
        "extra": extra,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _patch(monkeypatch, ext_report: PseudoGoldReport,
           sum_report: PseudoGoldReport, *, annotate_ext=None,
           annotate_sum=None):
    """Patch both passes with annotators that cache a marker and report."""
    calls: dict[str, list[int]] = {"ext": [], "sum": []}

    async def fake_ext(rows, endpoint, **kwargs):
        calls["ext"].append(len(rows))
        if annotate_ext is not None:
            annotate_ext(rows)
        return rows, ext_report

    async def fake_sum(rows, endpoint, **kwargs):
        calls["sum"].append(len(rows))
        if annotate_sum is not None:
            annotate_sum(rows)
        return rows, sum_report

    monkeypatch.setattr(script, "annotate_rows", fake_ext)
    monkeypatch.setattr(script, "annotate_summary_references", fake_sum)
    return calls


def test_main_dispatches_both_modes_and_writes(tmp_path: Path,
                                               monkeypatch) -> None:
    """Both passes run over the SAME pool; annotations land in ONE file."""
    val = tmp_path / "val.jsonl"
    out = tmp_path / "val_annotated.jsonl"
    extraction = _row("e1", {"extraction": {"stage": "one_stage"}})
    summary = _row("s1", {"summary": {"kind": "entity"}})
    qa = _row("q1", {"qa": {"question": "q"}})
    _write_rows(val, [extraction, summary, qa])

    ext_report = PseudoGoldReport(
        n_rows=3, n_annotated=1, n_failed=0, model="fake-judge"
    )
    sum_report = PseudoGoldReport(
        n_rows=1, n_annotated=1, n_failed=0, model="fake-judge",
        label="summary-ref",
    )

    def mark_ext(rows):
        rows[0]["extra"]["extraction"]["pseudo_gold"] = {"entities": []}

    def mark_sum(rows):
        rows[1]["extra"]["summary"]["reference"] = {
            "response": "{}", "model": "fake-judge",
        }

    calls = _patch(monkeypatch, ext_report, sum_report,
                   annotate_ext=mark_ext, annotate_sum=mark_sum)

    rc = script.main([
        "--val", str(val), "--out", str(out),
        "--judge-base-url", "http://localhost:8001/v1",
        "--judge-model", "fake-judge",
    ])
    assert rc == 0
    # The summary pass received the rows RETURNED by the extraction pass
    # (one shared pool, one write at the end).
    assert calls["ext"] == [3] and calls["sum"] == [3]
    rows = _read_rows(out)
    assert rows[0]["extra"]["extraction"]["pseudo_gold"] == {"entities": []}
    assert rows[1]["extra"]["summary"]["reference"]["model"] == "fake-judge"


def test_main_prints_active_modes_and_stays_silent_on_empty_ones(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """An empty mode prints nothing; an active one prints its label line."""
    val = tmp_path / "val.jsonl"
    out = tmp_path / "out.jsonl"
    _write_rows(val, [_row("e1", {"extraction": {"stage": "one_stage"}})])

    # Extraction-only pool: the summary report is 0/0 → silent.
    ext_report = PseudoGoldReport(
        n_rows=1, n_annotated=1, n_failed=0, model="fake-judge"
    )
    sum_report = PseudoGoldReport(
        n_rows=0, n_annotated=0, n_failed=0, model="fake-judge",
        label="summary-ref",
    )
    _patch(monkeypatch, ext_report, sum_report)
    rc = script.main([
        "--val", str(val), "--out", str(out),
        "--judge-base-url", "http://localhost:8001/v1",
        "--judge-model", "fake-judge",
    ])
    assert rc == 0
    stdout = capsys.readouterr().out
    assert "pseudo-gold: 1/1 annotated" in stdout
    assert "summary-ref" not in stdout


def test_main_exit_codes_partial_versus_total_failure(tmp_path: Path,
                                                      monkeypatch) -> None:
    """A partial failure is usable (0); a total failure is an error (1)."""
    val = tmp_path / "val.jsonl"
    out = tmp_path / "out.jsonl"
    _write_rows(val, [_row("e1", {"extraction": {"stage": "one_stage"}})])

    partial = PseudoGoldReport(
        n_rows=2, n_annotated=1, n_failed=1, model="fake-judge"
    )
    _patch(monkeypatch, partial, partial)
    assert script.main([
        "--val", str(val), "--out", str(out),
        "--judge-base-url", "http://localhost:8001/v1",
        "--judge-model", "fake-judge",
    ]) == 0

    total = PseudoGoldReport(
        n_rows=1, n_annotated=0, n_failed=1, model="fake-judge"
    )
    _patch(monkeypatch, total, total)
    assert script.main([
        "--val", str(val), "--out", str(out),
        "--judge-base-url", "http://localhost:8001/v1",
        "--judge-model", "fake-judge",
    ]) == 1


def test_main_empty_pool_is_a_warning_not_an_error(tmp_path: Path,
                                                   monkeypatch, capsys) -> None:
    val = tmp_path / "val.jsonl"
    out = tmp_path / "out.jsonl"
    val.write_text("", encoding="utf-8")
    calls = _patch(
        monkeypatch,
        PseudoGoldReport(n_rows=0, n_annotated=0, n_failed=0, model="m"),
        PseudoGoldReport(n_rows=0, n_annotated=0, n_failed=0, model="m"),
    )
    rc = script.main([
        "--val", str(val), "--out", str(out),
        "--judge-base-url", "http://localhost:8001/v1",
        "--judge-model", "fake-judge",
    ])
    assert rc == 0
    assert "carries no rows" in capsys.readouterr().out
    # Neither judge pass ran on an empty pool.
    assert calls["ext"] == [] and calls["sum"] == []
    assert not out.exists()


def test_main_requires_judge_endpoint(tmp_path: Path, monkeypatch) -> None:
    """No flag and no env → argparse error (bilingual message), not a crash."""
    monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    val = tmp_path / "val.jsonl"
    _write_rows(val, [_row("e1", {"extraction": {"stage": "one_stage"}})])
    with pytest.raises(SystemExit) as exc:
        script.main(["--val", str(val), "--out", str(tmp_path / "o.jsonl")])
    assert exc.value.code == 2
