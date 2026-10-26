'Regression tests for collect_tool_names (the CRITICAL bug guard).\n\n\nBackground\n\n``collect_tool_names`` previously looked only at ``tool.get("name")`` and\nsilently returned an EMPTY set for the real data, because ``opencode_gen`` and\n``tool_calls_multiturn`` store tools in the OpenAI / Qwen2.5 function-call shape\n``{"type": "function", "function": {"name": ...}}``. That emptied\n``BuildResult.tool_names`` and silently disabled the 30 % "known tool name"\nshare of ``format_check``. These tests pin both shapes so it cannot regress.\n\n``collect_tool_names``\n``BuildResult.tool_names``'

from __future__ import annotations

from llm_training.loaders import collect_tool_names
from llm_training.schema import UnifiedExample


def _ex(tools):
    return UnifiedExample(
        task_class="opencode_gen",
        prompt=[{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
        reference="ref",
        reference_messages=[{"role": "assistant", "content": "ref"}],
        tools=tools,
    )


def test_openai_function_shape_collected():
    # The OpenAI / Qwen2.5 shape actually used in opencode_gen.jsonl and
    # tool_calls_multiturn.jsonl — the bug case.
    # formerly broken.
    tools = [
        {"type": "function", "function": {"name": "read", "parameters": {}}},
        {"type": "function", "function": {"name": "bash", "parameters": {}}},
    ]
    assert collect_tool_names([_ex(tools)]) == frozenset({"read", "bash"})


def test_flat_shape_collected():
    # The flat {"name": ...} shape is also accepted.
    
    tools = [{"name": "read"}, {"name": "write"}]
    assert collect_tool_names([_ex(tools)]) == frozenset({"read", "write"})


def test_mixed_shapes_collected():
    tools = [
        {"type": "function", "function": {"name": "read"}},
        {"name": "write"},
    ]
    assert collect_tool_names([_ex(tools)]) == frozenset({"read", "write"})


def test_no_tools_contributes_nothing():
    assert collect_tool_names([_ex(None), _ex([])]) == frozenset()


def test_collect_tool_names_against_real_data():
    # The actual production check that previously failed: collect from the REAL
    # opencode_gen + tool_calls_multiturn datasets and assert a non-empty set.
    
    import json
    from pathlib import Path

    from llm_training.loaders import _tool_name

    data_dir = Path(__file__).resolve().parents[2] / "src_data"
    names: set[str] = set()
    with (data_dir / "opencode_gen.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for tool in rec.get("tools", []):
                n = _tool_name(tool)
                if n:
                    names.add(n)
    # opencode_gen declares exactly 11 OpenCode tools.
    
    assert len(names) == 11
    assert "read" in names
