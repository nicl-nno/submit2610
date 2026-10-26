'Offline tests of the agent-suite anchor runtime (no network, no GPU).\n\n\nCovers the stage-3 integration of\n``specifications/2026-10-17-agent-suite-glm53-validation.md``: the lockstep\npolicy rollout with LOCAL searchdocs execution, the deterministic scoring\n(envelope/schema/recall/grounding/answer, strict harmonic), and the\nselection-key fallback rules.'

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from llm_training.extraction.retrieval import HybridIndex
from llm_training.runtime.agent_suite import (
    AgentSuiteScorer,
    extract_tool_calls,
    selection_key,
    validate_searchdocs_args,
)

_DOCS = [
    "Москва является столицей России и крупнейшим городом страны.",  # doc-0
    "Река Волога протекает через Тверскую область на северо-западе.",  # doc-1
    "Отчёт компании за год показал рост выручки и новые рынки.",  # doc-2
    "Космический корабль приземлился на Марс после долгого полёта.",  # doc-3
]


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


def _suite_row(family: str, i: int, doc_id: str) -> dict:
    if family == "extraction":
        gold = {
            "pseudo_gold": {
                "entities": [
                    {
                        "entity_name": "Москва",
                        "entity_type": "CITY",
                        "description": "столица России",
                    }
                ],
                "relations": [],
            },
            "text": _DOCS[0],
        }
        user = (
            f'Excerpt:\n"""\n{seed_excerpt(_DOCS[0])}\n"""\n'
            "Extract per the schema."
        )
    elif family == "qa":
        gold = {"reference": "Москва — столица России.", "question": "Что такое Москва?"}
        user = "Question: Что такое Москва?"
    else:
        gold = {
            "reference": {"entity_name": "Москва", "description": "столица России"},
            "material": {
                "kind": "entity",
                "entity_name": "Москва",
                "entity_type": "CITY",
                "descriptions": ["столица России", "крупнейший город"],
            },
        }
        user = "Task: find fragments about Москва (CITY) and write the summary."
    return {
        "id": f"suite-{family}-{i:06d}",
        "family": family,
        "source_row_id": doc_id,
        "task_class": "ragu_entity_extraction",
        "prompt": [
            {"role": "system", "content": "системный промпт с протоколом SGR"},
            {"role": "user", "content": user},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "searchdocs",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer"},
                        },
                        "required": ["query", "top_k"],
                    },
                },
            }
        ],
        "gold": gold,
        "extra": {"family": family, "source_row_id": doc_id},
    }


def seed_excerpt(text: str, limit: int = 120) -> str:
    return " ".join(text.split())[:limit]


@pytest.fixture()
def suite_env(tmp_path: Path):
    'Frozen suite dir + corpus file; 3 tasks over 2 index documents.'
    encode = _hash_encoder()
    doc_ids = [f"doc-{i}" for i in range(2)]
    index = HybridIndex.build(_DOCS[:2], doc_ids, encode=encode)
    index_dir = tmp_path / "suite" / "index"
    saved = index.save(index_dir)
    (index_dir / "digest.txt").write_text(saved["digest"] + "\n", encoding="utf-8")

    corpus_rows = [
        {
            "id": doc_ids[i],
            "task_class": "ragu_entity_extraction",
            "extra": {"extraction": {"text": _DOCS[i]}},
        }
        for i in range(2)
    ]
    corpus_file = tmp_path / "corpus.jsonl"
    with corpus_file.open("w", encoding="utf-8") as fh:
        for row in corpus_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows = [
        _suite_row("extraction", 0, "doc-0"),
        _suite_row("qa", 1, "doc-0"),
        _suite_row("summary", 2, "doc-0"),
    ]
    with (tmp_path / "suite" / "suite.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    scorer = AgentSuiteScorer(
        suite_dir=tmp_path / "suite",
        corpus_paths=[str(corpus_file)],
        encode=encode,
    )
    return scorer, rows


def _scripted_generate(scripts: dict[str, list[str]]):
    'generate_batch keyed by a distinctive substring of the user message.'
    counters: dict[str, int] = {}

    def generate_batch(message_lists: list[list[dict[str, str]]]) -> list[str]:
        outputs: list[str] = []
        for messages in message_lists:
            user_texts = [
                m["content"] for m in messages if m["role"] == "user"
            ]
            matched = None
            for key, script in scripts.items():
                if any(key in text for text in user_texts):
                    matched = key
                    break
            assert matched is not None, "scripted generator: no script matched"
            turn = counters.get(matched, 0)
            counters[matched] = turn + 1
            outputs.append(script[min(turn, len(script) - 1)])
        return outputs

    return generate_batch


def _tool_call(name: str, arguments: dict) -> str:
    return (
        '<tool_call>{"name": "'
        + name
        + '", "arguments": '
        + json.dumps(arguments, ensure_ascii=False)
        + "}</tool_call>"
    )


# ------------------------------------------------------------------ parsing


def test_extract_tool_calls_tolerant() -> None:
    text = (
        "Начнём с поиска.\n```json\n"
        '<tool_call>{"name": "searchdocs", "arguments": '
        '{"query": "Москва столица", "top_k": 3}}</tool_call>\n```\n'
    )
    calls = extract_tool_calls(text)
    assert calls == [
        {"name": "searchdocs", "arguments": {"query": "Москва столица", "top_k": 3}}
    ]
    assert extract_tool_calls("никаких вызовов здесь нет") == []


def test_validate_searchdocs_args() -> None:
    assert validate_searchdocs_args({"query": "Москва столица", "top_k": 3}) == 1.0
    assert validate_searchdocs_args({"query": "коротко", "top_k": 3}) == 0.0
    assert validate_searchdocs_args({"query": "Москва столица", "top_k": 99}) == 0.0
    assert validate_searchdocs_args({"query": "Москва столица"}) == 0.0


# ------------------------------------------------------- rollout + scoring


def test_rollout_executes_searchdocs_and_scores(
    suite_env: tuple[AgentSuiteScorer, list[dict]],
) -> None:
    scorer, _rows = suite_env
    scripts = {
        
        "Extract per the schema.": [
            "Ищу документ: "
            + _tool_call("searchdocs", {"query": "Москва столица России", "top_k": 2}),
            _tool_call(
                "finalanswertool",
                {
                    "reasoning": "фрагмент найден и проверен",
                    "completed_steps": ["поиск", "извлечение"],
                    "answer": json.dumps(
                        {
                            "entities": [
                                {
                                    "entity_name": "Москва",
                                    "entity_type": "CITY",
                                    "description": "столица",
                                }
                            ],
                            "relations": [],
                        },
                        ensure_ascii=False,
                    ),
                    "status": "completed",
                },
            ),
        ],
        
        "Question: Что такое Москва?": [
            "Ищу ответ: "
            + _tool_call("searchdocs", {"query": "Москва столица России", "top_k": 2}),
            _tool_call(
                "finalanswertool",
                {
                    "reasoning": "ответ найден в чанке",
                    "completed_steps": ["поиск", "ответ"],
                    "answer": "Москва — столица России.",
                    "status": "completed",
                },
            ),
        ],
        
        "find fragments about Москва (CITY)": [
            "Ищу материал: "
            + _tool_call("searchdocs", {"query": "Москва крупный город", "top_k": 2}),
            _tool_call(
                "finalanswertool",
                {
                    "reasoning": "материал собран",
                    "completed_steps": ["поиск", "сводка"],
                    "answer": json.dumps(
                        {
                            "entity_name": "Москва",
                            "description": "столица России, крупнейший город",
                        },
                        ensure_ascii=False,
                    ),
                    "status": "completed",
                },
            ),
        ],
    }
    records = scorer.rollout(_scripted_generate(scripts))
    assert len(records) == 3
    assert all(record["accepted_envelope"] for record in records)
    assert all(record["search_calls"] >= 1 for record in records)
    breakdown = scorer.score(records)
    assert breakdown["n_scored"] == 3
    assert breakdown["envelope_rate"] == 1.0
    assert breakdown["schema_rate"] == 1.0
    assert breakdown["per_family"]["extraction"]["agent_val"] > 0.5
    
    assert breakdown["per_family"]["extraction"]["agent_val"] >= 0.5
    assert breakdown["agent_val"] > 0.5


def test_rollout_force_final_when_looping(
    suite_env: tuple[AgentSuiteScorer, list[dict]],
) -> None:
    scorer, _rows = suite_env
    looping = {
        "Extract per the schema.": [
            _tool_call("searchdocs", {"query": "Москва столица России", "top_k": 2})
        ]
        * 8,
        "Question: Что такое Москва?": [
            _tool_call("finalanswertool", {
                "reasoning": "r",
                "completed_steps": ["s"],
                "answer": "Москва — столица России.",
                "status": "completed",
            })
        ],
        "find fragments about Москва (CITY)": [
            _tool_call("finalanswertool", {
                "reasoning": "r",
                "completed_steps": ["s"],
                "answer": json.dumps(
                    {"entity_name": "Москва", "description": "город"},
                    ensure_ascii=False,
                ),
                "status": "completed",
            })
        ],
    }
    records = scorer.rollout(_scripted_generate(looping))
    looping_record = next(r for r in records if r["family"] == "extraction")
    assert looping_record["turns"] == 4  
    assert looping_record["accepted_envelope"] is False  
    breakdown = scorer.score(records)
    
    extraction_family = breakdown["per_family"]["extraction"]["agent_val"]
    assert extraction_family == 0.0


# ---------------------------------------------------------- selection key


def test_selection_key_harmonic_and_fallbacks() -> None:

    
    key, used = selection_key(0.6, 0.8, 141, 141)
    expected = (2 * 0.6 * 0.8) / (0.6 + 0.8)
    assert used and key == pytest.approx(expected)
    
    key, used = selection_key(0.6, None, 0, 141)
    assert key == 0.6 and not used
    
    key, used = selection_key(0.6, 0.9, 100, 141)
    assert key == 0.6 and not used
