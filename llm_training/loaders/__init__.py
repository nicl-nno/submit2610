"Loaders registry: maps every task class to its loader function.\n\n\n``LOADERS`` is a ``{task_class: LoaderFn}`` mapping built from every loader\nmodule. :mod:`llm_training.build` iterates over it to assemble the unified\ndataset. :func:`collect_tool_names` walks a set of unified examples once and\nreturns the union of every tool name declared in any tool-use example — used by\nthe reward layer's :func:`format_check` (validating ``<tool_call>`` names) and\ncollected automatically at build time so the registry always follows the data."

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Iterable

from llm_training.loaders.base import (
    LoaderFn,
    history_to_messages,
    parse_python_repr,
    read_json,
    read_jsonl,
)
from llm_training.schema import UnifiedExample

if TYPE_CHECKING:  # pragma: no cover
    from typing import Dict

# Import every loader module and register its task classes.

from llm_training.loaders import (  # noqa: E402
    asr_summarization,
    distill_loop,
    docsem_quant,
    meno_stand,
    menon_qa,
    mtrageval,
    multihop_rag,
    nerel,
    opencode_gen,
    ragu_icl,
    rummlu_pro,
    self_cognition,
    sgr_agent,
    tool_calls_multiturn,
)

_MODULES = (
    asr_summarization,
    docsem_quant,
    menon_qa,
    mtrageval,
    multihop_rag,
    nerel,
    opencode_gen,
    rummlu_pro,
    self_cognition,
    tool_calls_multiturn,
)

LOADERS: "Dict[str, LoaderFn]" = {}
for _mod in _MODULES:
    for _cls in getattr(_mod, "CLASSES", ()):
        LOADERS[_cls] = _mod.load

# SGR-agent: 9 classes share one loader parameterized by task_class (distilled
# from the sgr-agent-core SGR framework; tool-call-faithful, split per turn).
# SGR-agent: 9

for _cls in sgr_agent.CLASSES:
    LOADERS[_cls] = partial(sgr_agent.load_class, _cls)

# RAGU ICL: 6 classes share one loader parameterized by task_class.
# RAGU ICL: 6
for _cls in ragu_icl.CLASSES:
    LOADERS[_cls] = partial(ragu_icl.load_class, _cls)

# Meno-stand distillation: 3 classes share one loader parameterized by
# task_class (the task kind rides in the system prompt; test.jsonl rows are
# fixed-val — routed into the VAL pool by build.py).

# fixed-val).
for _cls in meno_stand.CLASSES:
    LOADERS[_cls] = partial(meno_stand.load_class, _cls)

# Data B: the distilled BenchmarkAgent loop (spec 0.2.1, decision №21) — a
# single class, train-only, per-turn split; registered under its own name.

# №21)
LOADERS[distill_loop.CLASS] = distill_loop.load


__all__ = [
    "LOADERS",
    "LoaderFn",
    "collect_tool_names",
    "collect_tool_schemas",
    "history_to_messages",
    "parse_python_repr",
    "read_json",
    "read_jsonl",
]


def _tool_name(tool: object) -> str | None:
    'Extract the tool name from either of the two common schema shapes.\n\n\n    Tool schemas appear in two forms across the data sources:\n    * the **OpenAI / Qwen2.5 function-call** shape —\n      ``{"type": "function", "function": {"name": str, ...}}``\n      (this is what ``opencode_gen.jsonl`` and ``tool_calls_multiturn.jsonl``\n      actually contain, and what the Qwen2.5 ``chat_template`` renders);\n    * the flat shape ``{"name": str, ...}``.\n\n    Both are accepted so the registry follows the data regardless of form.\n    "function":{"name":...}}``'
    if not isinstance(tool, dict):
        return None
    fn = tool.get("function")
    holder = fn if isinstance(fn, dict) else tool
    name = holder.get("name")
    return name if isinstance(name, str) and name else None


def collect_tool_names(examples: Iterable[UnifiedExample]) -> frozenset[str]:
    'Return the union of every tool name declared across the given examples.\n\n\n    Walks each example\'s ``tools`` list once and collects distinct names,\n    accepting both the OpenAI / Qwen2.5 function-call shape\n    (``{"type": "function", "function": {"name": ...}}`` — the form actually\n    present in ``opencode_gen.jsonl`` / ``tool_calls_multiturn.jsonl``) and the\n    flat shape (``{"name": ...}``). Examples without ``tools`` (the non-tool-use\n    classes) contribute nothing. Used by\n    :func:`llm_training.reward.scoring.format_check` to validate\n    ``<tool_call>`` names against the registry that actually appears in the\n    data — new tools propagate to the reward automatically, with no hard-coded\n    list to keep in sync.'
    names: set[str] = set()
    for ex in examples:
        for tool in ex.tools or []:
            name = _tool_name(tool)
            if name:
                names.add(name)
    return frozenset(names)


def _tool_schema(tool: object) -> "tuple[str, dict] | None":
    "Extract ``(name, parameter JSON schema)`` from either schema shape.\n\n\n    Mirrors :func:`_tool_name` (the OpenAI / Qwen2.5 function-call shape and\n    the flat shape); returns the tool's ``parameters`` dict only when it is a\n    non-empty mapping — a tool without a parameter schema has nothing for\n    :func:`llm_training.reward.schema_check.schema_check` to validate."
    if not isinstance(tool, dict):
        return None
    fn = tool.get("function")
    holder = fn if isinstance(fn, dict) else tool
    name = holder.get("name")
    if not (isinstance(name, str) and name):
        return None
    params = holder.get("parameters")
    if not isinstance(params, dict) or not params:
        return None
    return name, params


def collect_tool_schemas(
    examples: Iterable[UnifiedExample],
) -> dict[str, dict]:
    "Return the ``{tool name -> parameter JSON schema}`` registry.\n\n\n    The schema-level sibling of :func:`collect_tool_names`: walks each\n    example's ``tools`` list once and keeps the LAST declared parameter\n    schema per name (the same tool declared by several trajectories carries\n    the same schema in practice; the union of names is what\n    :func:`collect_tool_names` returns). Examples without ``tools``\n    contribute nothing. Used by\n    :func:`llm_training.reward.schema_check.schema_check` to validate the\n    ``arguments`` of generated ``<tool_call>`` blocks against the schemas\n    that actually ride in the prompts — new tools propagate automatically.\n    ``<tool_call>``"
    schemas: dict[str, dict] = {}
    for ex in examples:
        for tool in ex.tools or []:
            pair = _tool_schema(tool)
            if pair is not None:
                schemas[pair[0]] = pair[1]
    return schemas
