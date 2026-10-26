'Tests for reward/schema_check.py — tool-call argument validation.\n\n\nPure offline logic over the stdlib-only JSON-Schema subset (required +\nproperties.type + enum, one object nesting level).'

from __future__ import annotations

from llm_training.reward.schema_check import schema_check

#: A registry in the two real data shapes (OpenAI function-call holder is
#: flattened by collect_tool_schemas — the reward sees name -> parameters).

#: parameters).
SCHEMAS = {
    "read": {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
        },
        "required": ["filePath"],
    },
    "finalanswertool": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "completed_steps": {"type": "array"},
            "answer": {"type": "string"},
            "status": {"type": "string", "enum": ["planned", "done"]},
        },
        "required": ["reasoning", "answer", "status"],
    },
    "nested": {
        "type": "object",
        "properties": {
            "options": {
                "type": "object",
                "properties": {"verbose": {"type": "boolean"}},
                "required": ["verbose"],
            },
        },
        "required": ["options"],
    },
}


def _call(name: str, arguments) -> str:
    import json

    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


class TestSchemaCheck:
    def test_none_for_non_tool_class(self):
        assert schema_check(_call("read", {"filePath": "/x"}), "menon_qa", SCHEMAS) is None

    def test_none_for_empty_registry(self):
        assert schema_check(_call("read", {"filePath": "/x"}), "opencode_gen", {}) is None
        assert schema_check(_call("read", {"filePath": "/x"}), "opencode_gen", None) is None

    def test_none_for_no_blocks(self):
        assert schema_check("просто текст", "opencode_gen", SCHEMAS) is None
        assert schema_check("", "opencode_gen", SCHEMAS) is None

    def test_none_for_unknown_tool_only(self):
        # Unknown names are format_check's concern; nothing to validate here.
        
        assert (
            schema_check(_call("nope", {}), "opencode_gen", SCHEMAS) is None
        )

    def test_perfect_call_scores_one(self):
        assert (
            schema_check(
                _call("read", {"filePath": "/x", "offset": 1}),
                "opencode_gen",
                SCHEMAS,
            )
            == 1.0
        )

    def test_missing_required_scores_half(self):
        # offset present with a wrong type would fail types too; here only
        # the required field is missing, types of the present ones hold.
        
        assert (
            schema_check(
                _call("read", {"offset": 1}), "opencode_gen", SCHEMAS
            )
            == 0.5
        )

    def test_wrong_type_scores_half(self):
        # filePath present (required ok) but not a string → types fail → 0.5.
        
        assert (
            schema_check(
                _call("read", {"filePath": 42}), "opencode_gen", SCHEMAS
            )
            == 0.5
        )

    def test_missing_required_and_wrong_type_scores_zero(self):
        # filePath absent (required fails) AND offset carries a wrong type
        # → both halves fail → 0.0. (A PRESENT required field with a wrong
        # type still earns the presence half — presence ≠ type validity.)
        
        
        
        assert (
            schema_check(
                _call("read", {"offset": "not-an-int"}),
                "opencode_gen",
                SCHEMAS,
            )
            == 0.0
        )

    def test_boolean_is_not_integer(self):
        # In JSON Schema boolean ≠ integer; Python's bool subclasses int —
        # the check must NOT be fooled.
        # bool ≠ integer
        assert (
            schema_check(
                _call("read", {"filePath": "/x", "offset": True}),
                "opencode_gen",
                SCHEMAS,
            )
            == 0.5
        )

    def test_enum_violation(self):
        # finalanswertool is an SGR tool — score through an SGR class ("coder"
        # is in TOOL_USE_CLASSES; there is no aggregate "sgr_agent" class).
        # finalanswertool
        good = _call(
            "finalanswertool",
            {"reasoning": "r", "answer": "a", "status": "done"},
        )
        bad = _call(
            "finalanswertool",
            {"reasoning": "r", "answer": "a", "status": "finished"},
        )
        assert schema_check(good, "coder", SCHEMAS) == 1.0
        assert schema_check(bad, "coder", SCHEMAS) == 0.5

    def test_nested_object_one_level(self):
        good = _call("nested", {"options": {"verbose": False}})
        bad_missing = _call("nested", {"options": {}})
        bad_type = _call("nested", {"options": {"verbose": "yes"}})
        assert schema_check(good, "opencode_gen", SCHEMAS) == 1.0
        assert schema_check(bad_missing, "opencode_gen", SCHEMAS) == 0.5
        assert schema_check(bad_type, "opencode_gen", SCHEMAS) == 0.5

    def test_non_dict_arguments_score_zero(self):
        assert (
            schema_check(_call("read", "filePath"), "opencode_gen", SCHEMAS)
            == 0.0
        )

    def test_multiple_calls_mean(self):
        # One perfect + one missing-required → mean(1.0, 0.5) = 0.75.
        resp = _call("read", {"filePath": "/x"}) + _call("read", {"offset": 2})
        assert schema_check(resp, "opencode_gen", SCHEMAS) == 0.75

    def test_unknown_call_skipped_in_mean(self):
        # The unknown-tool call contributes nothing; the known one scores.
        resp = _call("nope", {}) + _call("read", {"filePath": "/x"})
        assert schema_check(resp, "opencode_gen", SCHEMAS) == 1.0

    def test_python_repr_block_parsed(self):
        # The Qwen2.5 template sometimes emits Python repr inside
        # <tool_call> — mirror the format_check fallback.
        
        resp = "<tool_call>{'name': 'read', 'arguments': {'filePath': '/x'}}</tool_call>"
        assert schema_check(resp, "opencode_gen", SCHEMAS) == 1.0

    def test_tool_without_parameters_skipped(self):
        schemas = {"bare": {"type": "object"}}
        assert schema_check(_call("bare", {}), "opencode_gen", schemas) is None
