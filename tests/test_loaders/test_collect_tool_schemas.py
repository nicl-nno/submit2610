'Tests for collect_tool_schemas (the schema-level sibling of tool names).\n\n\nPins both data shapes (the OpenAI / Qwen2.5 function-call holder and the flat\none) and the degradation rules: tools without a parameter schema contribute\nnothing, later declarations of the same name win.'

from __future__ import annotations

from llm_training.loaders import collect_tool_names, collect_tool_schemas
from llm_training.schema import UnifiedExample


def _ex(tools):
    return UnifiedExample(
        task_class="opencode_gen",
        prompt=[{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
        reference="ref",
        reference_messages=[{"role": "assistant", "content": "ref"}],
        tools=tools,
    )


_OPENAI_SHAPE = {
    "type": "function",
    "function": {
        "name": "read",
        "parameters": {
            "type": "object",
            "properties": {"filePath": {"type": "string"}},
            "required": ["filePath"],
        },
    },
}

_FLAT_SHAPE = {
    "name": "grep",
    "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}},
}


def test_openai_function_shape_collected():
    schemas = collect_tool_schemas([_ex([_OPENAI_SHAPE])])
    assert set(schemas) == {"read"}
    assert schemas["read"]["required"] == ["filePath"]


def test_flat_shape_collected():
    schemas = collect_tool_schemas([_ex([_FLAT_SHAPE])])
    assert set(schemas) == {"grep"}


def test_no_tools_contributes_nothing():
    assert collect_tool_schemas([_ex(None)]) == {}
    assert collect_tool_schemas([_ex([])]) == {}


def test_tool_without_parameters_skipped():
    # A declared tool with no/empty parameters dict has nothing to validate.
    
    tools = [
        {"type": "function", "function": {"name": "bare"}},
        {"name": "empty", "parameters": {}},
        {"name": "bad", "parameters": "not-a-dict"},
        {"no": "name"},
    ]
    assert collect_tool_schemas([_ex(tools)]) == {}


def test_later_declaration_wins():
    a = {"name": "t", "parameters": {"properties": {"x": {"type": "string"}}}}
    b = {"name": "t", "parameters": {"properties": {"y": {"type": "integer"}}}}
    schemas = collect_tool_schemas([_ex([a]), _ex([b])])
    assert set(schemas["t"]["properties"]) == {"y"}


def test_names_still_collected_alongside():
    # The sibling registry keeps working on the same examples: a tool without
    # parameters still contributes its NAME to collect_tool_names.
    
    
    tools = [{"name": "bare"}]
    examples = [_ex(tools)]
    assert collect_tool_names(examples) == frozenset({"bare"})
    assert collect_tool_schemas(examples) == {}


def test_real_data_shapes(tmp_path):
    # Smoke over the real committed catalog: the OpenCode tool schemas
    # (11 tools) parse into the registry with non-empty parameter dicts.
    
    
    import json
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src_data" / "opencode_gen.jsonl"
    if not src.is_file():  # pragma: no cover - the catalog is committed
        return
    tools = json.loads(src.read_text(encoding="utf-8").splitlines()[0])["tools"]
    schemas = collect_tool_schemas([_ex(tools)])
    assert len(schemas) == len(tools)
    assert all(isinstance(v, dict) and v for v in schemas.values())
