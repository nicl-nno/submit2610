'Loader: mt-RAGEval → ``mtrageval`` (8 sub-types merged into one class).\n\n\nSource ``mtrageval_ru.json`` is ``{<sub-type>: [{dialogue, context}, ...]}`` with\n8 sub-types (How-To, Factoid, Summarization, Explanation, Keyword, Non-Question,\nComposite, Opinion). All 8 are merged into the single class ``mtrageval``: the\ntiny sub-types (1–3 examples) would be meaningless as separate leave-task-out\nsplits. ``context`` is a list of ``{title, text}`` chunks (possibly empty).\nThe target is the **last assistant turn** of ``dialogue``; its placement\n(``system`` vs appended to the final ``user`` turn) is **varied** deterministically\nby ``seed + index``.'

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Iterator

from llm_training.config import DataConfig
from llm_training.loaders.base import read_json
from llm_training.schema import Message, UnifiedExample

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer

CLASSES = ("mtrageval",)
FILENAME = "mtrageval_ru.json"
SUBTYPES = (
    "How-To", "Factoid", "Summarization", "Explanation",
    "Keyword", "Non-Question", "Composite", "Opinion",
)


def _render_context(context: list[dict[str, Any]]) -> str:
    'Render the ``context`` chunk list into a single text block.'
    parts: list[str] = []
    for chunk in context or []:
        title = (chunk.get("title") or "").strip()
        text = (chunk.get("text") or "").strip()
        if title and text:
            parts.append(f"{title}\n{text}")
        elif text:
            parts.append(text)
    return "\n\n".join(parts)


def _place_context(
    prompt_messages: list[Message], context_text: str, placement: str
) -> list[Message]:
    'Return a copy of ``prompt_messages`` with ``context_text`` placed.\n\n\n    ``placement == "system"``: prepend/merge into a leading system message.\n    ``placement == "user"``: prepend to the last user message content\n    (context-before-question — the canonical order for retrieval-grounded QA:\n    the model reads the passages, then the question).'
    out = [dict(m) for m in prompt_messages]
    if not context_text:
        return out
    if placement == "system":
        if out and out[0].get("role") == "system":
            out[0] = {"role": "system", "content": out[0].get("content", "") + "\n\n" + context_text}
        else:
            out.insert(0, {"role": "system", "content": context_text})
    else:  # "user"
        # Prepend the context to the last user message (context-then-question).
        
        for j in range(len(out) - 1, -1, -1):
            if out[j].get("role") == "user":
                out[j] = {
                    "role": "user",
                    "content": context_text + "\n\n" + out[j].get("content", ""),
                }
                break
        else:
            # No user turn to attach to — fall back to a system message.
            
            out.insert(0, {"role": "system", "content": context_text})
    return out


def load(tokenizer: "PreTrainedTokenizer", cfg: DataConfig) -> Iterator[UnifiedExample]:
    """Yield ``mtrageval`` examples with varied context placement."""
    data = read_json(cfg.data_dir / FILENAME)
    global_idx = 0
    for subtype in SUBTYPES:
        items = data.get(subtype, []) or []
        for item in items:
            dialogue = item.get("dialogue") or []
            context = item.get("context") or []
            if not dialogue:
                continue
            # Find the last assistant turn = the reference.
            last_asst_idx = max(
                (k for k, m in enumerate(dialogue) if m.get("role") == "assistant"),
                default=-1,
            )
            if last_asst_idx < 0:
                continue
            reference = (dialogue[last_asst_idx].get("content") or "").strip()
            if not reference:
                continue
            prompt_messages: list[Message] = [dict(m) for m in dialogue[:last_asst_idx]]
            context_text = _render_context(context)
            rng = random.Random(cfg.seed + global_idx)
            placement = "system" if rng.random() < 0.5 else "user"
            prompt_messages = _place_context(prompt_messages, context_text, placement)
            yield UnifiedExample(
                task_class="mtrageval",
                prompt=prompt_messages,
                reference=reference,
                reference_messages=[{"role": "assistant", "content": reference}],
                source=f"{FILENAME}#{subtype}#{global_idx}",
                extra={
                    "subtype": subtype,
                    "context_placement": placement if context_text else "none",
                    "n_context_chunks": len(context),
                    # Store the rendered context verbatim so downstream code
                    # (e.g. extract_user_question in the reward layer) can strip
                    # it from the prompt without re-parsing the (variably
                    # placed) text. (Faithfulness / Noise Sensitivity were
                    # removed from the reward; see AGENTS.md.)
                    
                    
                    "context_text": context_text,
                },
            )
            global_idx += 1
