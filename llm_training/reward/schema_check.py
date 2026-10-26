"Deterministic tool-call ``schema_check`` — argument validation per schema.\n\n\n``format_check`` validates the tool-call ENVELOPE (valid JSON, known tool\nname, ``name``/``arguments`` keys) but never looks INSIDE ``arguments`` —\nyet the full JSON Schemas of every tool ride in the prompt (the ``<tools>``\nblock rendered into the system message; ``collect_tool_schemas`` lifts them\ninto a registry the same way ``collect_tool_names`` does for names). The\nbenchmark attribution (2026-09) showed the SFT/GRPO parents nearly double\nthe valid-JSON rate on hard schemas — a gain the evolutionary merge then\nlost; wiring the argument-level check into the in-loop reward trains exactly\nthat dimension and gives the merge/fitness something to preserve.\n``format_check``\n``arguments``\n\nThe validator is a **deliberate stdlib-only subset** of JSON Schema — no\n``jsonschema`` dependency (it is not a project dependency; the reward path\nmust stay deterministic across environments). The subset covers what real\ntool schemas in the data use: ``required`` presence, per-property ``type``\n(``string`` / ``number`` / ``integer`` / ``boolean`` / ``object`` / ``array``\n/ ``null``) and ``enum`` membership. Nested subschemas are checked when the\nproperty carries its own ``type``/``enum``/``required`` at the next level\n(one recursion for ``object`` properties).\n\nGrading per call\nGrading per call (graded, mirroring ``format_check``/``code_check``):\n\n* ``0.5`` — every schema-required argument is present in ``arguments``;\n* ``0.5`` — every present argument passes the ``type``/``enum`` check of its\n  property schema (a property absent from the schema is ignored — extra\n  arguments are ``format_check``'s envelope concern at most, not a type\n  violation).\n\nA response's calls score the MEAN of the per-call scores; calls of tools\nwithout a registered schema are skipped (``format_check`` already flags\nunknown names). ``None`` (not applicable, excluded from the harmonic\ncombine) when the class is not a tool-use class, no ``<tool_call>`` block\nparses, or no call references a schema-known tool."

from __future__ import annotations

from typing import Any

from llm_training.reward.scoring import (
    TOOL_USE_CLASSES,
    _TOOL_CALL_BLOCK_RE,
    safe_score,
)

#: JSON Schema ``type`` → the Python types satisfying it. ``integer`` accepts
#: ``int`` (but NOT ``bool`` — in Python ``bool`` is an ``int`` subclass, in
#: JSON Schema it is a distinct type). ``number`` accepts int and float.
# : ``type`` JSON Schema →
_TYPE_CHECKS: dict[str, Any] = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


def _parse_calls(response: str) -> list[dict[str, Any]]:
    'Parse ``<tool_call>`` blocks into dicts (json → Python literal fallback).'
    import ast
    import json

    calls: list[dict[str, Any]] = []
    for raw in _TOOL_CALL_BLOCK_RE.findall(response):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            try:
                data = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                continue
        if isinstance(data, dict):
            calls.append(data)
    return calls


def _property_ok(value: Any, prop_schema: dict[str, Any]) -> bool:
    'Check one value against its property schema (type/enum, one nesting).'
    enum = prop_schema.get("enum")
    if isinstance(enum, list):
        return value in enum
    type_name = prop_schema.get("type")
    if isinstance(type_name, str):
        check = _TYPE_CHECKS.get(type_name)
        if check is not None and not check(value):
            return False
        # One nesting level for object properties: required + subtypes.
        
        if isinstance(value, dict) and type_name == "object":
            sub_props = prop_schema.get("properties")
            if isinstance(sub_props, dict):
                for key, sub in sub_props.items():
                    if key in value and isinstance(sub, dict):
                        if not _property_ok(value[key], sub):
                            return False
            sub_required = prop_schema.get("required")
            if isinstance(sub_required, list):
                for key in sub_required:
                    if key not in value:
                        return False
    return True


def _call_schema_score(arguments: Any, schema: dict[str, Any]) -> float:
    "Graded [0, 0.5, 1.0] score of one call's arguments against the schema."
    props = schema.get("properties")
    props = props if isinstance(props, dict) else {}
    required = schema.get("required")
    required = required if isinstance(required, list) else []

    if not isinstance(arguments, dict):
        # Non-dict arguments (a string/JSON-encoded blob) fail BOTH halves.
        
        return 0.0

    required_ok = all(key in arguments for key in required)
    types_ok = all(
        _property_ok(arguments[key], sub)
        for key, sub in props.items()
        if key in arguments and isinstance(sub, dict)
    )
    score = 0.5 if required_ok else 0.0
    if types_ok:
        score += 0.5
    return score


def schema_check(
    response: str,
    task_class: str,
    known_tool_schemas: dict[str, dict[str, Any]] | None = None,
) -> float | None:
    "Graded [0, 1] argument-schema score for tool-call turns; ``None`` N/A.\n\n\n    Returns ``None`` (not applicable — excluded from the harmonic combine)\n    when the class is not in :data:`TOOL_USE_CLASSES`, no ``<tool_call>``\n    block parses into a dict, or every parsed call references a tool absent\n    from ``known_tool_schemas`` (unknown names are ``format_check``'s\n    concern; an empty registry means the check was never wired). Otherwise\n    returns the mean per-call score in ``[0, 1]`` — a call whose arguments\n    miss a required field or break a declared type scores below 1.0 (a\n    graded gradient toward schema-conform arguments, mirroring\n    ``format_check``'s envelope gradient)."
    if task_class not in TOOL_USE_CLASSES:
        return None
    if not known_tool_schemas:
        return None
    if not response or not response.strip():
        return None

    calls = _parse_calls(response)
    if not calls:
        return None

    scores: list[float] = []
    for call in calls:
        # The registry contract (collect_tool_schemas / RewardConfig
        # .known_tool_schemas): name -> the PARAMETER schema itself (not the
        # full OpenAI holder). Unknown names are skipped — format_check flags
        # them. A schema with neither ``properties`` nor ``required`` has
        # nothing checkable → skipped too (a tool declared without a
        # parameter schema never enters the registry in the first place).
        
        
        # properties/required
        params = known_tool_schemas.get(str(call.get("name") or ""))
        if not isinstance(params, dict) or not params:
            continue
        if "properties" not in params and "required" not in params:
            continue
        scores.append(_call_schema_score(call.get("arguments"), params))
    if not scores:
        return None
    return safe_score(sum(scores) / len(scores))
