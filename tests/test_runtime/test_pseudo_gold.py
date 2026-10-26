'Offline tests of the judge pseudo-gold annotation (no network, no GPU).'

from __future__ import annotations

import asyncio
import json

from llm_training.reward.summarization import SUMMARY_REFERENCE_KEY
from llm_training.runtime.endpoints import OpenAIEndpoint
from llm_training.runtime.pseudo_gold import (
    PSEUDO_GOLD_KEY,
    annotate_rows,
    annotate_summary_references,
)


class _Message:
    def __init__(self, content: str, reasoning: str | None = None) -> None:
        self.content = content
        if reasoning is not None:
            self.reasoning_content = reasoning


class _Choice:
    def __init__(self, message: _Message) -> None:
        self.message = message


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(_Message(content))]


class _Completions:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if not self._replies:
            raise RuntimeError("boom")
        return _Response(self._replies.pop(0))


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self.chat = _Chat(_Completions(replies))


_ANNOTATION = json.dumps(
    {
        "entities": [
            {
                "entity_name": "Павел Дуров",
                "entity_type": "PERSON",
                "description": "Основатель.",
            },
            {
                "entity_name": "ВКонтакте",
                "entity_type": "ORGANIZATION",
                "description": "Соцсеть.",
            },
        ],
        "relations": [
            {
                "source_entity": "ВКонтакте",
                "target_entity": "Павел Дуров",
                "relation_type": "FOUNDED_BY",
                "description": "Основал сеть.",
                "relationship_strength": 5,
            }
        ],
    },
    ensure_ascii=False,
)


def _val_row() -> dict:
    return {
        "id": "ext-val-000001",
        "task_class": "ragu_artifact_extraction",
        "prompt": [],
        "extra": {
            "style": "journalistic",
            "source": "wiki/ru/1",
            "target_language": "ru",
            "extraction": {
                "stage": "one_stage",
                "language": "ru",
                "text": "Павел Дуров основал ВКонтакте.",
                "entity_types": ["PERSON", "ORGANIZATION"],
                "relation_types": ["FOUNDED_BY"],
                "signatures": {},
            },
        },
    }


def _endpoint(monkeypatch, client: _FakeClient) -> OpenAIEndpoint:
    endpoint = OpenAIEndpoint(
        base_url="http://localhost:8001/v1", model="fake-judge"
    )
    monkeypatch.setattr(
        OpenAIEndpoint, "async_client", lambda self, **kwargs: client
    )
    return endpoint


def test_annotate_rows_cleans_and_caches(monkeypatch) -> None:
    # A fenced, think-block-prefixed reply must still parse.
    reply = (
        "<think>Рассуждаю...</think>"
        f"Вот разметка:\n```json\n{_ANNOTATION}\n```"
    )
    client = _FakeClient([reply])
    endpoint = _endpoint(monkeypatch, client)
    rows = [_val_row()]
    annotated, report = asyncio.run(
        annotate_rows(rows, endpoint, retries=0, retry_sleep_s=0.0)
    )
    assert report.n_annotated == 1 and report.n_failed == 0
    gold = annotated[0]["extra"]["extraction"][PSEUDO_GOLD_KEY]
    assert gold["model"] == "fake-judge"
    assert {e["entity_name"] for e in gold["entities"]} == {
        "Павел Дуров", "ВКонтакте",
    }
    # Idempotency: a second run makes no new judge calls.
    client2 = _FakeClient([])
    endpoint2 = _endpoint(monkeypatch, client2)
    _, report2 = asyncio.run(
        annotate_rows(annotated, endpoint2, retries=0, retry_sleep_s=0.0)
    )
    assert report2.n_annotated == 1 and report2.n_failed == 0
    assert client2.chat.completions.calls == 0


def test_annotate_rows_reasoning_content_fallback(monkeypatch) -> None:
    client = _FakeClient([])
    # The fake client returns content directly; emulate the thinking case by
    # a message whose content is empty — reuse the reply pipeline through a
    # tiny subclass is overkill; instead verify the sanitizer accepts the
    # raw JSON through the normal path and the fallback exists as a function.
    from llm_training.runtime.pseudo_gold import _message_text

    class _M:
        content = ""
        reasoning_content = _ANNOTATION

    assert _message_text(_M()) == _ANNOTATION
    endpoint = _endpoint(monkeypatch, client)
    rows = [_val_row()]
    _, report = asyncio.run(
        annotate_rows(rows, endpoint, retries=0, retry_sleep_s=0.0)
    )
    assert report.n_failed == 1  # the empty-replies client fails gracefully


def test_annotate_rows_retries_then_fails(monkeypatch) -> None:
    client = _FakeClient([])  # every create() raises
    endpoint = _endpoint(monkeypatch, client)
    rows = [_val_row()]
    annotated, report = asyncio.run(
        annotate_rows(rows, endpoint, retries=2, retry_sleep_s=0.0)
    )
    assert report.n_failed == 1 and report.n_annotated == 0
    assert PSEUDO_GOLD_KEY not in annotated[0]["extra"]["extraction"]
    assert client.chat.completions.calls == 3  # 1 + 2 retries


def test_pseudo_gold_score_metric() -> None:
    from llm_training.reward.extraction import pseudo_gold_score

    parsed = {
        "entities": [
            {"entity_name": "Павел Дуров", "entity_type": "PERSON"},
            {"entity_name": "ВКонтакте", "entity_type": "ORGANIZATION"},
        ],
        "relations": [
            {
                "source_entity": "ВКонтакте",
                "target_entity": "Павел Дуров",
                "relation_type": "FOUNDED_BY",
            }
        ],
    }
    gold = {
        "entities": [
            {"entity_name": "Павел Дуров", "entity_type": "PERSON"},
            {"entity_name": "ВКонтакте", "entity_type": "ORGANIZATION"},
            # An inflected variant that canonicalizes to the same stems.
            {"entity_name": "Санкт-Петербурге", "entity_type": "CITY"},
        ],
        "relations": [
            {
                "source_entity": "ВКонтакте",
                "target_entity": "Павел Дуров",
                "relation_type": "FOUNDED_BY",
            },
            {
                "source_entity": "ВКонтакте",
                "target_entity": "Санкт-Петербург",
                "relation_type": "TAKES_PLACE_IN",
            },
        ],
    }
    scored = pseudo_gold_score(parsed, gold, "ru")
    assert scored is not None
    # Entities: 2 common / 2 predicted / 3 gold → F1 = 0.8.
    assert abs(scored.entity_f1 - 0.8) < 1e-6
    # Relations: 1 common / 1 predicted / 2 gold → F1 = 2/3.
    assert abs(scored.relation_f1 - 2 / 3) < 1e-6
    assert scored.combined is not None and 0.0 < scored.combined < 1.0

    # Degenerate: no gold → N/A; empty prediction vs gold → 0.
    assert pseudo_gold_score(parsed, None, "ru") is None
    assert pseudo_gold_score(parsed, {"entities": []}, "ru") is None
    empty = pseudo_gold_score({"entities": [], "relations": []}, gold, "ru")
    assert empty is not None and empty.combined == 0.0


# ---------------------------------------------------------------------------
# Summary-reference mode
# ---------------------------------------------------------------------------
_SUMMARY_ROW = {
    "id": "ext-sum-e-000001",
    "task_class": "ragu_entity_summary",
    "prompt": [
        {"role": "system", "content": "Merge the descriptions into one."},
        {
            "role": "user",
            "content": "Entity: метформин\n1. Препарат от диабета.",
        },
    ],
    "extra": {
        "style": "scientific",
        "source": "wiki/ru/1",
        "summary": {
            "kind": "entity",
            "entity_name": "метформин",
            "entity_type": "CHEM",
            "descriptions": ["Препарат от диабета.", "Снижает глюкозу."],
        },
    },
}

_SUMMARY_REPLY = (
    "Рассуждаю...```json\n"
    '{"entity_name": "метформин", '
    '"description": "Метформин — препарат от диабета."}'
    "\n```"
)


def _summary_row() -> dict:
    return json.loads(json.dumps(_SUMMARY_ROW, ensure_ascii=False))


def test_annotate_summary_references_caches_and_idempotent(
    monkeypatch,
) -> None:
    """The judge reply is validated, cleaned and cached — once."""
    client = _FakeClient([_SUMMARY_REPLY])
    endpoint = _endpoint(monkeypatch, client)
    rows = [_summary_row()]
    annotated, report = asyncio.run(
        annotate_summary_references(
            rows, endpoint, retries=0, retry_sleep_s=0.0
        )
    )
    assert report.label == "summary-ref"
    assert report.n_rows == 1
    assert report.n_annotated == 1 and report.n_failed == 0
    ref = annotated[0]["extra"]["summary"][SUMMARY_REFERENCE_KEY]
    assert ref["model"] == "fake-judge"
    # The cached text is the CLEANED envelope (fences stripped).
    assert ref["response"].startswith("{")
    assert "препарат от диабета" in ref["response"]
    # Idempotency: a second run makes no new judge calls.
    client2 = _FakeClient([])
    endpoint2 = _endpoint(monkeypatch, client2)
    _, report2 = asyncio.run(
        annotate_summary_references(
            annotated, endpoint2, retries=0, retry_sleep_s=0.0
        )
    )
    assert report2.n_annotated == 1 and report2.n_failed == 0
    assert client2.chat.completions.calls == 0


def test_annotate_summary_references_invalid_reply_fails(
    monkeypatch,
) -> None:
    """An unparseable reply never becomes a reference (counted failed)."""
    client = _FakeClient(['{"foo": 1}'])
    endpoint = _endpoint(monkeypatch, client)
    rows = [_summary_row()]
    annotated, report = asyncio.run(
        annotate_summary_references(
            rows, endpoint, retries=0, retry_sleep_s=0.0
        )
    )
    assert report.n_failed == 1 and report.n_annotated == 0
    assert SUMMARY_REFERENCE_KEY not in annotated[0]["extra"]["summary"]


def test_annotate_summary_references_skips_foreign_and_malformed(
    monkeypatch,
) -> None:
    """Extraction rows and promptless summary rows are not judge failures."""
    client = _FakeClient([])
    endpoint = _endpoint(monkeypatch, client)
    malformed = _summary_row()
    malformed["prompt"] = []
    rows = [_val_row(), malformed]
    _, report = asyncio.run(
        annotate_summary_references(
            rows, endpoint, retries=0, retry_sleep_s=0.0
        )
    )
    assert client.chat.completions.calls == 0
    # Only the kind-valid summary row counts toward the mode denominator…
    assert report.n_rows == 1
    # …and a broken prompt is a data defect, not a judge failure.
    assert report.n_annotated == 0 and report.n_failed == 0


def test_annotate_summary_references_report_lines(monkeypatch) -> None:
    """The report line carries the mode label (operator log distinction)."""
    client = _FakeClient([_SUMMARY_REPLY])
    endpoint = _endpoint(monkeypatch, client)
    _, report = asyncio.run(
        annotate_summary_references(
            [_summary_row()], endpoint, retries=0, retry_sleep_s=0.0
        )
    )
    lines = report.format_lines()
    assert len(lines) == 1
    assert lines[0].startswith("summary-ref: 1/1 annotated")
    assert "failed: 0" in lines[0] and "fake-judge" in lines[0]
