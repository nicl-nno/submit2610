'Dataset loaders for the unified GRPO dataset.\n\n\nEach loader reads one in-repo source under ``src_data/`` (NEREL_Instruct is the\nexception: it is fetched from HuggingFace at runtime) and yields\n:class:`~llm_training.schema.UnifiedExample` instances tagged with their task\nclass. Loaders never truncate: over-length examples are discarded downstream by\n:mod:`llm_training.build`.\n\n:class:`~llm_training.schema.UnifiedExample`'

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Protocol

logger = logging.getLogger(__name__)

from llm_training.config import DataConfig
from llm_training.schema import Message, UnifiedExample

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer


class LoaderFn(Protocol):
    'Loader protocol: yield examples from one source.\n\n\n    Loaders receive the tokenizer (needed only by tool-use loaders to bake tool\n    schemas into the system message) and the :class:`DataConfig`.'

    def __call__(
        self, tokenizer: "PreTrainedTokenizer", cfg: DataConfig
    ) -> Iterator[UnifiedExample]: ...


def parse_python_repr(text: str) -> Any:
    'Parse a Python-repr literal (single quotes, ``True``/``None``) to an object.\n\n\n    The ``tool_calls_multiturn`` source stores both the ``<tools>`` list and the\n    ``<tool_call>`` argument dicts as Python ``repr`` (not JSON).\n    ``<tool_call>``'
    return ast.literal_eval(text)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    'Iterate JSONL records from ``path`` (UTF-8), skipping unparseable lines.\n\n\n    A corrupt line is logged and skipped rather than aborting the whole load\n    (one bad line should not invalidate an entire class). Blank lines are\n    silently skipped'
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(
                    "Skipping unparseable JSONL line %s:%d: %s",
                    path, lineno, e,
                )


def read_json(path: Path) -> Any:
    'Read a JSON file (UTF-8).'
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def history_to_messages(
    system: str,
    query: str,
    response: str,
    history: list[Any] | None,
) -> tuple[list[Message], list[Message]]:
    'Turn a flat ``{system, query, response, history}`` record into messages.\n\n\n    Used by the ``multihop_rag`` / ``rummlu_pro`` / ``self_cognition`` / NEREL\n    loaders (they all share this shape). ``history`` is a list of prior\n    ``[user, assistant]`` pairs. Returns ``(prompt_messages, reference_messages)``\n    where ``prompt_messages`` is everything up to (but not including) the final\n    reference assistant turn.\n``rummlu_pro``\n``self_cognition``'
    prompt: list[Message] = []
    if system:
        prompt.append({"role": "system", "content": system})
    for pair in history or []:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        u, a = pair[0], pair[1]
        if u:
            prompt.append({"role": "user", "content": u})
        if a:
            prompt.append({"role": "assistant", "content": a})
    prompt.append({"role": "user", "content": query})
    reference_messages: list[Message] = [{"role": "assistant", "content": response}]
    return prompt, reference_messages
