"Offline tests of run_extraction_eval.py's pure scoring half."

from __future__ import annotations

import json

import pytest

from run_extraction_eval import score_responses

_TEXT_RU = "Павел Дуров основал социальную сеть ВКонтакте в Санкт-Петербурге."

_ROWS = [
    {
        "id": "ext-val-000001",
        "task_class": "ragu_artifact_extraction",
        "prompt": [],
        "extra": {
            "style": "journalistic",
            "extraction": {
                "stage": "one_stage",
                "language": "ru",
                "text": _TEXT_RU,
                "entity_types": ["PERSON", "ORGANIZATION"],
                "relation_types": ["FOUNDED_BY"],
                "signatures": {
                    "FOUNDED_BY": {
                        "domain": ["ORGANIZATION"],
                        "range": ["PERSON"],
                    }
                },
                "pseudo_gold": {
                    "entities": [
                        {"entity_name": "Павел Дуров", "entity_type": "PERSON"},
                        {"entity_name": "ВКонтакте", "entity_type": "ORGANIZATION"},
                    ],
                    "relations": [],
                },
            },
        },
    },
]


def _good() -> str:
    return json.dumps(
        {
            "entities": [
                {"entity_name": "Павел Дуров", "entity_type": "PERSON",
                 "description": "d"},
                {"entity_name": "ВКонтакте", "entity_type": "ORGANIZATION",
                 "description": "d"},
            ],
            "relations": [
                {"source_entity": "ВКонтакте", "target_entity": "Павел Дуров",
                 "relation_type": "FOUNDED_BY", "description": "d",
                 "relationship_strength": 5},
            ],
        },
        ensure_ascii=False,
    )


def test_score_responses_aggregates() -> None:
    report = score_responses(_ROWS, [[_good(), "мусор"]])
    assert report["n_rows"] == 1 and report["n_samples"] == 2
    assert report["json_score"] == 0.5  # one perfect, one broken
    assert report["pseudo_gold"] == 0.5
    assert report["per_row"][0]["samples"][0]["val"] > 0.0
    assert report["per_row"][0]["samples"][1]["val"] == 0.0
    # The report keeps raw responses for diagnosis.
    assert "response" in report["per_row"][0]["samples"][0]
    # Extraction-only pool: the kind breakdown carries just that kind.
    assert report["val_by_kind"] == {"extraction": report["val"]}
    # The new summary/QA components are N/A here (no such rows).
    assert report["summary"] is None and report["qa"] is None


# ------------------------------------------------ summary / QA dispatch
_SUMMARY_PAYLOAD = {
    "kind": "entity",
    "entity_name": "метформин",
    "entity_type": "CHEM",
    "descriptions": [
        "метформин — препарат от диабета",
        "снижает глюкозу у пациентов",
    ],
}
_SUMMARY_RESPONSE = (
    '{"entity_name": "метформин", '
    '"description": "Метформин — препарат от диабета, снижающий глюкозу."}'
)
_SUMMARY_REFERENCE = (
    '{"entity_name": "метформин", '
    '"description": "Метформин применяется при диабете."}'
)


def _summary_row(with_reference: bool) -> dict:
    payload = dict(_SUMMARY_PAYLOAD)
    if with_reference:
        payload["reference"] = {
            "response": _SUMMARY_REFERENCE, "model": "fake-judge",
        }
    return {
        "id": "ext-sum-e-000001",
        "task_class": "ragu_entity_summary",
        "prompt": [{"role": "user", "content": "merge"}],
        "extra": {"style": "scientific", "summary": payload},
    }


def _qa_row() -> dict:
    return {
        "id": "qa-000001",
        "task_class": "fineweb_qa",
        "prompt": [{"role": "user", "content": "Столица Франции?"}],
        "reference": "Париж — столица Франции.",
        "extra": {"style": "qa", "qa": {"question": "Столица Франции?"}},
    }


class _FakeMetrics:
    def __init__(self, combined: float) -> None:
        self.combined = combined


class _FakeService:
    def __init__(self, combined: float = 0.6) -> None:
        self.combined = combined
        self.requests: list = []
        self.closed = False

    def score_batch(self, requests):
        self.requests = list(requests)
        return [_FakeMetrics(self.combined) for _ in requests]

    def close(self) -> None:
        self.closed = True


def test_score_summary_rows_with_reference_semsim() -> None:
    """Reference SemSim joins the structural combined (the anchor shape)."""
    from llm_training.reward.scoring import harmonic_mean
    from llm_training.reward.summarization import summarization_check

    structural = summarization_check(
        "ragu_entity_summary", {"summary": _SUMMARY_PAYLOAD},
        _SUMMARY_RESPONSE,
    )
    assert structural is not None
    report = score_responses(
        [_summary_row(with_reference=True)],
        [[_SUMMARY_RESPONSE]],
        summary_semsim=lambda a, b: 0.5,
    )
    assert report["val_by_kind"]["summary"] == pytest.approx(
        report["val"]
    )
    assert report["summary"] == pytest.approx(structural.combined)
    assert report["summary_semsim"] == pytest.approx(0.5)
    sample = report["per_row"][0]["samples"][0]
    assert sample["val"] == pytest.approx(
        harmonic_mean([structural.combined, 0.5])
    )


def test_score_summary_rows_without_reference_fall_back() -> None:
    """No cached reference → structural alone; the SemSim never fires."""
    from llm_training.reward.summarization import summarization_check

    called: list[bool] = []

    def semsim(a: str, b: str) -> float:
        called.append(True)
        return 0.5

    structural = summarization_check(
        "ragu_entity_summary", {"summary": _SUMMARY_PAYLOAD},
        _SUMMARY_RESPONSE,
    )
    assert structural is not None
    report = score_responses(
        [_summary_row(with_reference=False)],
        [[_SUMMARY_RESPONSE]],
        summary_semsim=semsim,
    )
    assert report["per_row"][0]["samples"][0]["val"] == pytest.approx(
        structural.combined
    )
    assert report["summary_semsim"] is None
    assert called == []


def test_score_qa_rows_batched_via_service() -> None:
    """QA rows score through ONE batch service call (the anchor pattern)."""
    service = _FakeService(combined=0.6)
    report = score_responses([_qa_row()], [["Париж"]], service=service)
    assert len(service.requests) == 1
    req = service.requests[0]
    assert req.task_class == "fineweb_qa"
    assert req.reference == "Париж — столица Франции."
    assert report["qa"] == pytest.approx(0.6)
    assert report["val_by_kind"]["qa"] == pytest.approx(0.6)
    assert report["per_row"][0]["val_mean"] == pytest.approx(0.6)


def test_score_qa_rows_without_service_measure_zero() -> None:
    """No service → the QA kind visibly measures 0.0 (never a crash)."""
    report = score_responses([_qa_row()], [["Париж"]], service=None)
    assert report["qa"] == pytest.approx(0.0)
    assert report["val_by_kind"]["qa"] == pytest.approx(0.0)


def test_score_mixed_pool_kind_restricted_components() -> None:
    """Extraction/summary/QA means never cross-contaminate in the report."""
    from llm_training.reward.summarization import summarization_check

    structural = summarization_check(
        "ragu_entity_summary", {"summary": _SUMMARY_PAYLOAD},
        _SUMMARY_RESPONSE,
    )
    assert structural is not None
    rows = [_ROWS[0], _summary_row(with_reference=False), _qa_row()]
    groups = [[_good()], [_SUMMARY_RESPONSE], [["Париж"]]]
    report = score_responses(rows, groups, service=_FakeService(0.6))
    assert set(report["val_by_kind"]) == {"extraction", "summary", "qa"}
    # The extraction json_score stays the extraction row's own value —
    # the summary envelope's json_score does NOT leak into it.
    assert report["json_score"] == pytest.approx(1.0)
    assert report["summary"] == pytest.approx(structural.combined)
    assert report["qa"] == pytest.approx(0.6)
    assert report["summary_semsim"] is None
    # Overall val = mean of the three per-row means.
    assert report["val"] == pytest.approx(
        sum(report["val_by_kind"].values()) / 3
    )
