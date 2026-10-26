'Loader: data B (distilled BenchmarkAgent loop) → ``distill_b_loop``.\n\n``distill_b_loop``\n\nSource: ``src_data/distill_loop/train_distill_b_loop.jsonl`` produced by\n``scripts/convert_distill_logs.py`` — the BenchmarkAgent loop faithfully\nreconstructed from the sgr-agent-core agent logs: system + task + Current\nDate, then reasoning calls (``reasoningtool``, canonical args validated\nclient-side by the OpenAI SDK), search/extract action calls and the final\n``finalanswertool`` (honest "not found" finals for the trap queries).\n\n``scripts/convert_distill_logs.py``\n\n**Turn-splitting** mirrors :mod:`llm_training.loaders.sgr_agent` (one\nexample per assistant turn, prior tool results folded into the prompt).\nThree turn kinds, dispatched by the same reward flags:\n\n\n* ``reasoningtool`` call → ``is_call_turn=True`` (loop-faithful: the model\n  must emit the reasoningtool call; scored SemSim + ``format_check``, the\n  same treatment the loop-format data A records get);\n* action call (``websearchtool`` / ``extractpagecontenttool``) →\n  ``is_call_turn=True`` (SemSim + ``format_check``);\n* ``finalanswertool`` → ``is_call_turn=True, is_final_answer=True``\n  (reference = the ``answer`` field).\n\nExtras carry ``lang`` / ``is_trap`` for run-level accounting. Train-only:\nthe anchor val pool is untouched (the frozen measuring instrument).'

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from llm_training.config import DataConfig
from llm_training.loaders.base import read_jsonl
from llm_training.loaders.sgr_agent import (
    _extract_answer_from_msg,
    _is_final_answer_turn,
    _normalize_message,
)
from llm_training.schema import UnifiedExample
from llm_training.tokenizer_utils import assistant_content_str

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer

CLASS = "distill_b_loop"
SUBDIR = "distill_loop"
FNAME = "train_distill_b_loop.jsonl"


def load_class(
    task_class: str, tokenizer: "PreTrainedTokenizer", cfg: DataConfig
) -> Iterator[UnifiedExample]:
    'Yield data-B examples, one per assistant turn.'
    path = Path(cfg.data_dir) / SUBDIR / FNAME
    if not path.exists():
        return
    for i, rec in enumerate(read_jsonl(path)):
        raw_messages = rec.get("messages") or []
        if not raw_messages:
            continue
        messages = [_normalize_message(m) for m in raw_messages]
        tools = rec.get("tools") or None
        rec_extra = rec.get("extra") or {}
        first_user_idx = next(
            (k for k, m in enumerate(messages) if m.get("role") == "user"),
            None,
        )
        if first_user_idx is None or first_user_idx < 1:
            continue  # need at least system+user
        running_prompt: list[dict[str, Any]] = [
            dict(m) for m in messages[: first_user_idx + 1]
        ]
        turn_index = 0
        for msg in messages[first_user_idx + 1 :]:
            if msg.get("role") != "assistant":
                # Tool results (and any interleaved turns) are context.
                
                running_prompt.append(dict(msg))
                continue
            msg = dict(msg)
            tool_calls = msg.get("tool_calls") or []
            is_final = _is_final_answer_turn(msg)
            if is_final:
                reference = _extract_answer_from_msg(msg)
                extra: dict[str, Any] = {
                    "is_call_turn": True,
                    "is_final_answer": True,
                    "turn_index": turn_index,
                    "lang": rec.get("lang"),
                    "is_trap": bool(rec_extra.get("trap", False)),
                }
            else:
                reference = assistant_content_str([msg])
                extra = {
                    "is_call_turn": bool(tool_calls),
                    "turn_index": turn_index,
                    "lang": rec.get("lang"),
                    "is_trap": bool(rec_extra.get("trap", False)),
                }
            if not reference:
                
                # A degenerate empty turn is context only, not a target.
                running_prompt.append(msg)
                continue
            yield UnifiedExample(
                task_class=task_class,
                prompt=[dict(m) for m in running_prompt],
                reference=reference,
                tools=tools,
                reference_messages=[msg],
                source=f"{SUBDIR}/{FNAME}#{i}#turn{turn_index}",
                extra=extra,
            )
            turn_index += 1
            running_prompt.append(msg)


def load(
    task_class: str, tokenizer: "PreTrainedTokenizer", cfg: DataConfig
) -> Iterator[UnifiedExample]:
    """Registry-compatible entry point (LOADERS[CLASS])."""
    yield from load_class(task_class, tokenizer, cfg)
