'Loader: OpenCode-Gen → ``opencode_gen`` (single-turn tool-use examples).\n\n\nThe source records store the Qwen2.5/Meno-Lite function-calling shape\n``{task_class, distilled_from, project, tools, messages}`` with 11 OpenCode tool\nschemas and full multi-turn trajectories (assistant ``tool_calls`` + tool results\nas ``role:"tool"``). The loader splits each trajectory into **one example per\nassistant turn** — mirroring how :mod:`llm_training.loaders.tool_calls_multiturn`\nalready works — so that:\n\ndistilled_from, project, tools, messages}``\n\n* each example is a single-turn problem: ``prompt`` = the conversation up to\n  (but not including) this assistant turn — **with the reference tool results\n  already in the prompt as ``role:"tool"``** — and ``reference_messages`` = just\n  this one assistant turn;\n* **no tool executor is needed** in either ``run_eval.py`` or the future\n  ``grpo_train.py``: the policy always sees the correct prior context (the distilled\n  reference tool results) and is scored on its single generated turn. This\n  dissolves the AGENTS.md "Open follow-ups (tool-use in GRPO)" roadmap item by\n  data design.'

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from llm_training.config import DataConfig
from llm_training.loaders.base import read_jsonl
from llm_training.schema import Message, UnifiedExample
from llm_training.tokenizer_utils import assistant_content_str

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer

CLASSES = ("opencode_gen",)
FILENAME = "opencode_gen.jsonl"
EXPECTED_NUM_TOOLS = 11  # OpenCode toolset


def load(tokenizer: "PreTrainedTokenizer", cfg: DataConfig) -> Iterator[UnifiedExample]:
    'Yield one ``opencode_gen`` example per assistant turn of each trajectory.'
    path = cfg.data_dir / FILENAME
    for i, rec in enumerate(read_jsonl(path)):
        messages = rec.get("messages") or []
        tools = rec.get("tools") or []
        if len(tools) != EXPECTED_NUM_TOOLS:
            # Defensive: the v2 corpus has exactly 11 tools; flag and skip oddities.
            
            continue
        # Locate the first user turn; the shared prefix is everything up to and
        # including it. The rest is the reference trajectory we split on.
        
        first_user_idx = next(
            (k for k, m in enumerate(messages) if m.get("role") == "user"), None
        )
        if first_user_idx is None or first_user_idx < 1:
            continue  # need a system + user at minimum
        prefix: list[Message] = [dict(m) for m in messages[: first_user_idx + 1]]
        rest: list[Message] = [dict(m) for m in messages[first_user_idx + 1 :]]
        if not rest:
            continue

        distilled_from = rec.get("distilled_from", "")
        project = rec.get("project", "")
        turn_index = 0  # counts assistant turns only
        # Walk the trajectory; accumulate prior turns into the running prompt so
        # each assistant turn is conditioned on the real reference tool results.
        
        running_prompt: list[Message] = list(prefix)
        for msg in rest:
            if msg.get("role") != "assistant":
                # Tool/user turns between assistant turns are CONTEXT for the
                # next assistant turn — fold them into the running prompt.
                # tool/user
                running_prompt.append(msg)
                continue
            this_turn = [msg]
            reference = assistant_content_str(this_turn)
            if not reference:
                running_prompt.append(msg)
                continue
            yield UnifiedExample(
                task_class="opencode_gen",
                prompt=[dict(m) for m in running_prompt],
                reference=reference,
                tools=list(tools),
                reference_messages=[dict(msg)],
                source=f"{FILENAME}#{i}#turn{turn_index}",
                extra={
                    "distilled_from": distilled_from,
                    "project": project,
                    "turn_index": turn_index,
                    "is_call_turn": bool(msg.get("tool_calls")),
                },
            )
            turn_index += 1
            running_prompt.append(msg)
