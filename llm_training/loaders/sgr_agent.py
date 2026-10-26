'Loader: SGR-agent catalog → 9 task classes (tool-call-faithful, turn-split).\n\n\nThe source (distilled from the sgr-agent-core SGR framework, formerly named\n"sharegpt") stores the Qwen2.5/Meno-Lite function-calling shape ``{messages,\ntools}`` where the agent **always finalizes via the ``finalanswertool``** call.\nThe export pipeline produced tool-call-faithful references: the trailing\n``assistant(tool_calls=finalanswertool)`` turn is kept verbatim (the framework\'s\n``--final-answer-as tool`` mode), so a student model trained on this catalog\nlearns the real SGR agent output protocol.\n\nformat function-calling Qwen2.5/Meno-Lite: ``{messages, tools}``\ntool-call-faithful\n\n**Turn-splitting.** The multi-step trajectories (the SGR classes that reason +\ncall action tools: ``data_analyst``, ``deep_research``, ``factual_qa``,\n``summarizer``; ``math_reasoner`` reasons then finalises without action tools)\nare split into **one example per assistant turn** — exactly as\n:mod:`llm_training.loaders.opencode_gen` already works — with each example\'s\n``prompt`` carrying the correct reference tool results as ``role:"tool"``\ncontext. No tool executor is needed in run_eval.py / the future grpo_train.py: the\npolicy always conditions on the distilled reference context and is scored on its\nsingle generated turn.\n\nThree turn kinds are emitted, scored differently by the reward dispatcher:\n\n* **reasoning** (assistant text, no ``tool_calls``): ``reference`` = the\n  reasoning text; ``extra={"is_call_turn": False, "is_reasoning_turn": True}``.\n  The reward scores fluency + an SGR-schema structural check (NOT SemSim — see\n  ``enable_reasoning_semsim`` / ``reasoning_check``); reasoning is open-ended,\n  so SemSim would reward paraphrase-imitation rather than quality.\n* **action** (assistant with an action tool call, e.g. ``pandas_analysis_tool``):\n  ``reference`` = the rendered turn; ``extra={"is_call_turn": True}`` → scored by\n  Semantic Similarity + ``format_check`` (validates the tool-call structure).\n* **final** (assistant with ``finalanswertool``): ``reference`` = the extracted\n  ``answer`` field (so SemSim/fluency/``code_check`` compare answer-vs-answer);\n  ``extra={"is_call_turn": True, "is_final_answer": True}`` → ``format_check``\n  validates the ``finalanswertool`` call and the reward extracts the model\'s\n  ``answer`` for content scoring.'

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Iterator

from llm_training.config import DataConfig
from llm_training.loaders.base import read_jsonl
from llm_training.schema import Message, UnifiedExample
from llm_training.tokenizer_utils import assistant_content_str

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

#: The 9 SGR-agent task classes = filename suffixes.

CLASSES = (
    "coder",
    "creative_writer",
    "data_analyst",
    "deep_research",
    "factual_qa",
    "instruction_following",
    "math_reasoner",
    "rewriter",
    "summarizer",
)
SUBDIR = "sgr_agent"

#: The SGR framework's finalization tool — every trajectory ends with a call to
#: it. The final turn's reference is its ``answer`` argument.

FINAL_ANSWER_TOOL = "finalanswertool"


def _normalize_message(msg: Message) -> Message:
    'Normalize one OpenAI message: ``arguments`` JSON-string → object.\n\n\n    OpenAI stores ``tool_calls[i].function.arguments`` as a JSON string; the\n    Meno-Lite chat template does ``arguments | tojson``, which would double-\n    encode a string. We decode it once at load time (provenance files are not\n    modified — a runtime normalization, the same convention as before). A\n    malformed ``arguments`` string is logged and replaced with an empty dict.'
    if not msg.get("tool_calls"):
        return msg
    out = dict(msg)
    tool_calls = out.get("tool_calls")
    if tool_calls:
        norm: list[dict[str, Any]] = []
        for tc in tool_calls:
            tc = dict(tc)
            fn = tc.get("function")
            if fn is not None:
                fn = dict(fn)
                args = fn.get("arguments")
                if isinstance(args, str):
                    if not args:
                        fn["arguments"] = {}
                    else:
                        try:
                            fn["arguments"] = json.loads(args)
                        except json.JSONDecodeError as e:
                            logger.warning(
                                "sgr_agent: malformed tool_call arguments "
                                "(treating as {}): %s", e,
                            )
                            fn["arguments"] = {}
                tc["function"] = fn
            norm.append(tc)
        out["tool_calls"] = norm
    return out


def _extract_answer_from_msg(msg: Message) -> str:
    'Pull the ``answer`` field out of a finalanswertool assistant message.'
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", tc)
        if fn.get("name") == FINAL_ANSWER_TOOL:
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            if isinstance(args, dict):
                return args.get("answer") or ""
    return ""


def _is_final_answer_turn(msg: Message) -> bool:
    return FINAL_ANSWER_TOOL in [
        (tc.get("function", tc) or {}).get("name")
        for tc in (msg.get("tool_calls") or [])
    ]


def _resolve_path(cfg: DataConfig, fname: str) -> tuple[Path, str]:
    'Pick the loop-format file when configured and present, else original.\n\n    ``scripts/convert_sgr_to_loop.py``'
    loop = getattr(cfg, "sgr_loop_subdir", None)
    if loop:
        loop_path = cfg.data_dir / loop / fname
        if loop_path.exists():
            return loop_path, f"{loop}/{fname}"
    return cfg.data_dir / SUBDIR / fname, f"{SUBDIR}/{fname}"


def load_class(
    task_class: str, tokenizer: "PreTrainedTokenizer", cfg: DataConfig
) -> Iterator[UnifiedExample]:
    'Yield SGR-agent examples for one task class, one per assistant turn.'
    for fname in (f"train_{task_class}.jsonl", f"val_{task_class}.jsonl"):
        path, source_prefix = _resolve_path(cfg, fname)
        if not path.exists():
            continue
        for i, rec in enumerate(read_jsonl(path)):
            raw_messages = rec.get("messages") or []
            if not raw_messages:
                continue
            messages = [_normalize_message(m) for m in raw_messages]
            tools = rec.get("tools") or None
            # Prefix = everything up to and including the first user turn (the
            # system + the task + the date preamble). The rest is the reference
            # trajectory we split on, assistant-turn by assistant-turn.
            
            
            first_user_idx = next(
                (k for k, m in enumerate(messages) if m.get("role") == "user"),
                None,
            )
            if first_user_idx is None or first_user_idx < 1:
                continue  # need at least system + user
            running_prompt: list[Message] = [dict(m) for m in messages[: first_user_idx + 1]]
            rest = messages[first_user_idx + 1 :]
            turn_index = 0
            for msg in rest:
                if msg.get("role") != "assistant":
                    # Non-assistant turns (tool results, interleaved user) are
                    # CONTEXT for the next assistant turn — fold them in.
                    
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
                    }
                else:
                    reference = assistant_content_str([msg])
                    extra = {
                        "is_call_turn": bool(tool_calls),
                        "turn_index": turn_index,
                    }
                    if not tool_calls:
                        # SGR reasoning phase (CoT text, no tool call).
                        
                        extra["is_reasoning_turn"] = True
                if not reference:
                    # A degenerate empty turn contributes nothing as a target;
                    # fold it into the prompt as context and move on.
                    
                    running_prompt.append(msg)
                    continue
                yield UnifiedExample(
                    task_class=task_class,
                    prompt=[dict(m) for m in running_prompt],
                    reference=reference,
                    tools=tools,
                    reference_messages=[msg],
                    source=f"{source_prefix}#{i}#turn{turn_index}",
                    extra=extra,
                )
                turn_index += 1
                running_prompt.append(msg)


def load(
    task_class: str, tokenizer: "PreTrainedTokenizer", cfg: DataConfig
) -> Iterator[UnifiedExample]:
    'Registry-compatible entry point delegating to :func:`load_class`.'
    yield from load_class(task_class, tokenizer, cfg)
