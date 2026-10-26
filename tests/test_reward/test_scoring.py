'Tests for reward/scoring.py — pure score math + format-checks (no mocks).'

from __future__ import annotations

import math

import pytest

from llm_training.reward.scoring import (
    CODE_CHECK_CLASS,
    FINAL_ANSWER_TOOL,
    MIN_FLUENCY_WORDS,
    TOOL_USE_CLASSES,
    aggregate_fluency,
    code_check,
    content_response_text,
    count_words,
    extract_final_answer,
    format_check,
    harmonic_mean,
    reasoning_check,
    safe_score,
    split_prose_and_code,
)


class TestSafeScore:
    def test_passes_through_unit_interval(self):
        assert safe_score(0.0) == 0.0
        assert safe_score(1.0) == 1.0
        assert safe_score(0.5) == 0.5

    def test_none_is_zero(self):
        assert safe_score(None) == 0.0

    def test_clamps_above_one(self):
        assert safe_score(1.5) == 1.0
        assert safe_score(42.0) == 1.0

    def test_clamps_below_zero(self):
        assert safe_score(-0.3) == 0.0
        assert safe_score(-10.0) == 0.0

    def test_nan_and_inf_are_zero(self):
        assert safe_score(float("nan")) == 0.0
        assert safe_score(float("inf")) == 0.0
        assert safe_score(float("-inf")) == 0.0

    def test_garbage_is_zero(self):
        assert safe_score("not-a-number") == 0.0
        assert safe_score(object()) == 0.0


class TestHarmonicMean:
    def test_equal_values(self):
        assert harmonic_mean([0.5, 0.5, 0.5]) == pytest.approx(0.5)

    def test_all_ones(self):
        assert harmonic_mean([1.0, 1.0]) == pytest.approx(1.0)

    def test_known_three(self):
        # HM(1, 1/2, 1/4) = 3 / (1 + 2 + 4) = 3/7
        assert harmonic_mean([1.0, 0.5, 0.25]) == pytest.approx(3.0 / 7.0)

    def test_any_zero_is_zero(self):
        # Standard score-combination convention: a zero on any dimension zeroes
        # the harmonic mean (a fully failed metric kills the combined reward).
        
        assert harmonic_mean([0.0, 0.9, 0.9]) == 0.0
        assert harmonic_mean([0.9, 0.0]) == 0.0

    def test_empty_is_zero(self):
        # Invariant (defensive): an empty list must NOT raise; it returns 0.0
        # so a misuse cannot abort a GRPO batch. In production the caller always
        # supplies ≥1 active metric (Semantic Similarity).
        
        
        assert harmonic_mean([]) == 0.0

    def test_single_value(self):
        assert harmonic_mean([0.7]) == pytest.approx(0.7)

    def test_strictly_le_mean(self):
        # Harmonic mean <= arithmetic mean, equality only when all equal.
        
        hm = harmonic_mean([0.2, 0.8, 1.0])
        am = (0.2 + 0.8 + 1.0) / 3
        assert hm < am

    def test_returns_finite_float(self):
        for values in ([0.3, 0.7], [1.0, 1.0], [0.0, 0.5]):
            r = harmonic_mean(values)
            assert isinstance(r, float)
            assert not math.isnan(r)
            assert 0.0 <= r <= 1.0


class TestFormatCheckNonToolClasses:
    def test_none_for_non_tool_class(self):
        # Any class outside TOOL_USE_CLASSES must return None — the metric is
        # not applicable and must NOT drag the harmonic combine down.
        
        for cls in ("menon_qa", "mtrageval", "rummlu_pro", "multihop_rag"):
            assert format_check("anything", cls) is None

    def test_unknown_class_is_none(self):
        assert format_check("<tool_call>{}</tool_call>", "not_a_class") is None

    def test_tool_use_classes_registry(self):
        # Sanity: TOOL_USE_CLASSES contains the tool-use classes (OpenCode /
        # tool-calls-multiturn + the 9 SGR-agent classes).
        
        assert "opencode_gen" in TOOL_USE_CLASSES
        assert "tool_calls_multiturn" in TOOL_USE_CLASSES
        assert "coder" in TOOL_USE_CLASSES  # SGR-agent



class TestFormatCheckToolUse:
    def test_perfect_json_tool_call(self):
        # A single well-formed tool_call with a known name and both fields.
        
        response = '<tool_call>{"name": "read", "arguments": {"path": "x"}}</tool_call>'
        score = format_check(response, "opencode_gen", frozenset({"read", "write"}))
        assert score == pytest.approx(1.0)

    def test_missing_arguments_field(self):
        # JSON parses, name is known, but ``arguments`` is absent → fields fail.
        
        response = '<tool_call>{"name": "read"}</tool_call>'
        score = format_check(response, "opencode_gen", frozenset({"read"}))
        # JSON ok (0.4) + name ok (0.3) + fields fail (0.0) = 0.7
        assert score == pytest.approx(0.7)

    def test_unknown_tool_name(self):
        # JSON parses and fields are present, but name is not in the registry.
        
        response = '<tool_call>{"name": "hack", "arguments": {}}</tool_call>'
        score = format_check(response, "opencode_gen", frozenset({"read"}))
        # JSON ok (0.4) + name fail (0.0) + fields ok (0.3) = 0.7
        assert score == pytest.approx(0.7)

    def test_no_registry_skips_name_check_neutrally(self):
        # When known_tool_names is empty, the name check is skipped (awarded
        # neutrally). The operator did not collect a registry.
        
        response = '<tool_call>{"name": "whatever", "arguments": {}}</tool_call>'
        score = format_check(response, "opencode_gen", frozenset())
        assert score == pytest.approx(1.0)

    def test_invalid_json(self):
        response = '<tool_call>{not valid json}</tool_call>'
        score = format_check(response, "opencode_gen", frozenset({"read"}))
        # JSON fail (0.0) + name fail (0.0) + fields fail (0.0) = 0.0
        assert score == 0.0

    def test_python_repr_falls_back(self):
        # The Qwen template sometimes emits Python repr (single quotes, True);
        # the format_check falls back to ast.literal_eval.
        
        response = "<tool_call>{'name': 'read', 'arguments': {'path': 'x'}}</tool_call>"
        score = format_check(response, "opencode_gen", frozenset({"read"}))
        assert score == pytest.approx(1.0)

    def test_no_tool_call_block_is_zero(self):
        # A tool-use response with no <tool_call> block at all is structurally
        # broken (text-only answer when a tool call was expected).
        # tool-use
        response = "I will help you with that."
        assert format_check(response, "opencode_gen") == 0.0

    def test_empty_response_is_zero(self):
        assert format_check("", "opencode_gen") == 0.0
        assert format_check("   ", "opencode_gen") == 0.0

    def test_multiple_blocks_all_must_pass(self):
        # Two tool_call blocks; one malformed → JSON fails for the whole call.
        
        response = (
            '<tool_call>{"name": "read", "arguments": {}}</tool_call>'
            '<tool_call>not json</tool_call>'
        )
        score = format_check(response, "opencode_gen", frozenset({"read"}))
        # JSON fail (0.0) + name ok for first (0.3) + fields ok for first (0.3) = 0.6
        assert score == pytest.approx(0.6)

    def test_multiple_blocks_all_valid(self):
        response = (
            '<tool_call>{"name": "read", "arguments": {"path": "x"}}</tool_call>'
            '<tool_call>{"name": "write", "arguments": {"path": "y"}}</tool_call>'
        )
        score = format_check(response, "opencode_gen", frozenset({"read", "write"}))
        assert score == pytest.approx(1.0)

    def test_score_in_unit_interval(self):
        for response in [
            "<tool_call>{}</tool_call>",
            '<tool_call>{"name": "x", "arguments": {}}</tool_call>',
            "no tool call here",
            '<tool_call>{"name": "x"}</tool_call>',
        ]:
            s = format_check(response, "tool_calls_multiturn", frozenset({"x"}))
            assert s is None or (0.0 <= s <= 1.0)


# ---------------------------------------------------------------------------
# split_prose_and_code
# ---------------------------------------------------------------------------
class TestSplitProseAndCode:
    def test_extracts_python_and_strips_from_prose(self):
        text = "Вот решение:\n```python\nx = 1\n```\nГотово."
        prose, blocks = split_prose_and_code(text)
        assert ("python", "x = 1\n") in blocks
        assert "x = 1" not in prose
        assert "Вот решение" in prose
        assert "Готово" in prose

    def test_multiple_blocks_different_langs(self):
        text = (
            "```python\na = 1\n```\nпроза\n"
            "```rust\nlet b = 2;\n```\n"
            "```python\nc = 3\n```"
        )
        prose, blocks = split_prose_and_code(text)
        langs = [lang for lang, _ in blocks]
        assert langs == ["python", "rust", "python"]
        assert "a = 1" not in prose and "let b = 2" not in prose
        assert "проза" in prose

    def test_inline_code_stripped(self):
        prose, _ = split_prose_and_code("используйте `foo()` для вызова.")
        assert "foo()" not in prose
        assert "используйте" in prose

    def test_tool_blocks_stripped(self):
        text = (
            "ответ <tool_call>{'name':'x'}</tool_call> далее "
            "<tool_response>r</tool_response> конец"
        )
        prose, _ = split_prose_and_code(text)
        assert "<tool_call>" not in prose
        assert "<tool_response>" not in prose
        assert "ответ" in prose and "конец" in prose

    def test_unlabelled_fence_extracted(self):
        prose, blocks = split_prose_and_code("```\nplain\n```")
        assert ("", "plain\n") in blocks
        assert "plain" not in prose

    def test_empty_and_none(self):
        assert split_prose_and_code("") == ("", [])
        assert split_prose_and_code(None) == ("", [])  # type: ignore[arg-type]

    def test_whitespace_collapsed(self):
        prose, _ = split_prose_and_code("а\n\n\nб   в\tг")
        assert prose == "а б в г"

    def test_no_fences_returns_text_cleaned(self):
        # No code; inline code still stripped, whitespace collapsed.
        prose, blocks = split_prose_and_code("Простой `код` текст.")
        assert blocks == []
        assert "код" not in prose
        assert prose == "Простой текст."


# ---------------------------------------------------------------------------
# code_check
# ---------------------------------------------------------------------------
class TestCodeCheck:
    def test_none_for_non_coder_class(self):
        # The metric is not applicable to non-coder classes (excluded from the
        # harmonic combine, must NOT drag the score down).
        
        for cls in ("menon_qa", "math_reasoner", "opencode_gen", "tool_calls_multiturn"):
            assert code_check("```python\nx=1\n```", cls) is None

    def test_registry_constant(self):
        assert CODE_CHECK_CLASS == "coder"

    def test_none_for_coder_without_python(self):
        # A coder answer about Rust/SQL is structurally fine — N/A, not 0.
        
        rust = "```rust\nfn main(){}\n```"
        assert code_check(rust, "coder") is None

    def test_none_for_empty_response(self):
        assert code_check("", "coder") is None
        assert code_check("   ", "coder") is None

    def test_broken_python_is_zero(self):
        # Any SyntaxError → 0.0 (strong gradient toward compilable code).
        
        broken = "```python\ndef f(\n```"
        assert code_check(broken, "coder") == 0.0

    def test_valid_clean_function_is_one(self):
        good = "```python\ndef f(x):\n    return x + 1\n```"
        assert code_check(good, "coder") == pytest.approx(1.0)

    def test_comment_only_is_parse_only_plus_clean(self):
        # Parses + clean, but non-trivial fails (no def/statements).
        # parse(0.5) + nontrivial(0) + clean(0.2) = 0.7
        comment = "```python\n# просто комментарий\n```"
        assert code_check(comment, "coder") == pytest.approx(0.7)

    def test_pass_only_statement(self):
        # `pass` is a statement but not in _MEANINGFUL_STMT_TYPES → non-trivial
        # fails → 0.7. (parse + clean)
        
        pas = "```python\npass\n```"
        assert code_check(pas, "coder") == pytest.approx(0.7)

    def test_bare_except_penalty(self):
        # def + bare except → parse + nontrivial, anti-pattern fail.
        # parse(0.5) + nontrivial(0.3) + antipattern(0) = 0.8
        code = "```python\ndef f():\n    try:\n        x()\n    except:\n        pass\n```"
        assert code_check(code, "coder") == pytest.approx(0.8)

    def test_star_import_penalty(self):
        # `from os import *` alone: parse ok, no def/meaningful-stmt, antipattern.
        # parse(0.5) + nontrivial(0) + antipattern(0) = 0.5
        star = "```python\nfrom os import *\n```"
        assert code_check(star, "coder") == pytest.approx(0.5)

    def test_mutable_default_penalty(self):
        # def + mutable default → parse + nontrivial, anti-pattern fail.
        code = "```python\ndef f(x=[]):\n    return x\n```"
        assert code_check(code, "coder") == pytest.approx(0.8)

    def test_exec_penalty_in_function(self):
        # def + exec → parse + nontrivial, anti-pattern fail.
        code = "```python\ndef f():\n    exec('x')\n```"
        assert code_check(code, "coder") == pytest.approx(0.8)

    def test_multiple_blocks_all_valid(self):
        code = (
            "```python\ndef f():\n    return 1\n```\n"
            "```python\nclass C:\n    pass\n```"
        )
        assert code_check(code, "coder") == pytest.approx(1.0)

    def test_multiple_blocks_one_broken_is_zero(self):
        # One broken block among several → whole code_check 0.0.
        
        code = (
            "```python\ndef f():\n    return 1\n```\n"
            "```python\ndef (\n```"
        )
        assert code_check(code, "coder") == 0.0

    def test_python3_tag_recognised(self):
        good = "```python3\ndef f():\n    return 1\n```"
        assert code_check(good, "coder") == pytest.approx(1.0)

    def test_score_in_unit_interval_or_none(self):
        for response in [
            "```python\nx\n```",
            "```python\ndef f(): pass\n```",
            "```rust\nfn main(){}\n```",
            "no code at all",
        ]:
            s = code_check(response, "coder")
            assert s is None or (0.0 <= s <= 1.0)


# ---------------------------------------------------------------------------
# aggregate_fluency (mean of calibrated per-sentence scores) /

# ---------------------------------------------------------------------------
class TestAggregateFluency:
    def test_single_value_returns_it(self):
        # aggregate_fluency now means CALIBRATED per-sentence scores; the
        # word-count gate is upstream, so one score is a valid aggregate.
        # aggregate_fluency
        
        assert aggregate_fluency([0.8]) == pytest.approx(0.8)

    def test_two_values_returns_mean(self):
        assert aggregate_fluency([0.8, 0.6]) == pytest.approx(0.7)

    def test_clamps_with_safe_score(self):
        # Out-of-range values are clamped to [0,1] before averaging.
        
        assert aggregate_fluency([1.5, -0.2]) == pytest.approx(0.5)

    def test_none_values_zeroed(self):
        assert aggregate_fluency([None, 0.8]) == pytest.approx(0.4)

    def test_empty_is_none(self):
        assert aggregate_fluency([]) is None

    def test_default_word_floor_matches_constant(self):
        assert MIN_FLUENCY_WORDS == 3


# ---------------------------------------------------------------------------
# count_words (razdel word-count gate)
# ---------------------------------------------------------------------------
class TestCountWords:
    def test_counts_alpha_tokens(self):
        
        assert count_words("Мама мыла раму") == 3

    def test_hyphenated_is_one_word(self):
        # razdel keeps "
        
        assert count_words("Город Ростов-на-Дону красив") == 3

    def test_punctuation_ignored(self):
        assert count_words("Да!!! Или нет?") == 3

    def test_numbers_not_counted_as_words(self):
        # Pure-number tokens carry no alpha → excluded (they carry no fluency
        # signal). A token like "60
        
        assert count_words("Ему исполнилось 42 года") == 3
        assert count_words("60-летний мужчина пришёл") == 3

    def test_empty_is_zero(self):
        assert count_words("") == 0
        assert count_words("!!! ???") == 0

    def test_short_sentence_below_floor(self):
        # A 1- or 2-word sentence is below the MIN_FLUENCY_WORDS=3 gate.
        
        assert count_words("Да.") < MIN_FLUENCY_WORDS
        assert count_words("ОК спасибо") < MIN_FLUENCY_WORDS


# ---------------------------------------------------------------------------
# reasoning_check (SGR-schema structural score)
# ---------------------------------------------------------------------------
class TestReasoningCheck:
    def test_all_four_markers_is_one(self):
        text = (
            "Reasoning steps:\n- First\nCurrent situation: ok\n"
            "Plan status: going\nRemaining steps:\n- Next"
        )
        assert reasoning_check(text) == 1.0

    def test_partial_markers(self):
        # 2 of 4 markers → 0.5.
        text = "Reasoning steps:\n- a\nCurrent situation: x"
        assert reasoning_check(text) == 0.5

    def test_case_insensitive(self):
        assert reasoning_check("REASONING STEPS\ncurrent situation\nplan status\nremaining steps") == 1.0

    def test_no_markers_is_zero(self):
        assert reasoning_check("Просто какой-то текст без SGR-маркеров.") == 0.0

    def test_empty_is_zero(self):
        assert reasoning_check("") == 0.0
        assert reasoning_check("   ") == 0.0

    def test_clamped_to_unit(self):
        # Score is always in [0, 1].
        
        assert 0.0 <= reasoning_check("anything") <= 1.0


# ---------------------------------------------------------------------------
# extract_final_answer (finalanswertool answer extraction) /

# ---------------------------------------------------------------------------
class TestExtractFinalAnswer:
    def test_extracts_answer_from_tool_call(self):
        # Build the response with json.dumps so newlines in the answer are
        # properly escaped (as the chat template's `arguments | tojson` would).
        
        import json as _json
        answer = "## Решение\n```python\nx=1\n```"
        inner = _json.dumps(
            {"name": "finalanswertool",
             "arguments": {"answer": answer, "status": "completed"}},
            ensure_ascii=False,
        )
        resp = f"<tool_call>\n{inner}\n</tool_call>"
        assert extract_final_answer(resp) == answer

    def test_none_when_no_finalanswertool(self):
        resp = '<tool_call>\n{"name": "pandas_analysis_tool", "arguments": {}}</tool_call>'
        assert extract_final_answer(resp) is None

    def test_none_when_no_tool_call(self):
        assert extract_final_answer("plain text, no tool call") is None
        assert extract_final_answer("") is None

    def test_empty_answer_string(self):
        import json as _json
        inner = _json.dumps(
            {"name": "finalanswertool",
             "arguments": {"answer": "", "status": "completed"}},
        )
        resp = f"<tool_call>\n{inner}\n</tool_call>"
        # An empty answer field → empty string (not None; the call was valid).
        
        assert extract_final_answer(resp) == ""

    def test_picks_finalanswertool_when_multiple_blocks(self):
        import json as _json
        resp = (
            '<tool_call>\n'
            + _json.dumps({"name": "pandas_analysis_tool", "arguments": {}})
            + '\n</tool_call>\n<tool_call>\n'
            + _json.dumps({"name": "finalanswertool", "arguments": {"answer": "done"}})
            + "\n</tool_call>"
        )
        assert extract_final_answer(resp) == "done"

    def test_final_answer_tool_constant(self):
        assert FINAL_ANSWER_TOOL == "finalanswertool"


# ---------------------------------------------------------------------------
# content_response_text (the reward-equivalent offline-analysis population) /

# ---------------------------------------------------------------------------
class TestContentResponseText:
    @staticmethod
    def _envelope(answer: str) -> str:
        import json as _json
        return (
            "<tool_call>\n"
            + _json.dumps(
                {"name": "finalanswertool",
                 "arguments": {"answer": answer, "status": "completed"}},
                ensure_ascii=False,
            )
            + "\n</tool_call>"
        )

    def test_final_turn_yields_extracted_answer(self):
        # The offline analysis must score the answer PROSE, not the envelope —
        # the bug that once left final turns silently out of the Phase-0 pool.
        
        gen = self._envelope("Решение задачи с объяснением и выводом.")
        out = content_response_text(
            gen, {"is_final_answer": True, "is_call_turn": True}
        )
        assert out == "Решение задачи с объяснением и выводом."

    def test_final_turn_without_valid_call_is_none(self):
        # No valid finalanswertool → nothing to score offline (the reward maps
        # this to "" for penalization; the offline scripts skip the sample).
        
        assert content_response_text(
            "мусор без tool-call", {"is_final_answer": True, "is_call_turn": True}
        ) is None
        assert content_response_text(
            "<tool_call>{broken json</tool_call>",
            {"is_final_answer": True, "is_call_turn": True},
        ) is None

    def test_reasoning_and_plain_turns_pass_raw_text(self):
        assert content_response_text(
            "Reasoning steps:\n- a", {"is_reasoning_turn": True}
        ) == "Reasoning steps:\n- a"
        # Plain non-tool answers (no extra flags) pass through as-is.
        
        assert content_response_text("Обычный ответ без метаданных.", None) == (
            "Обычный ответ без метаданных."
        )

    def test_action_turn_passes_raw_text(self):
        # Action turns keep their raw text (the content line survives the
        # downstream tool-span stripping, mirroring the reward's fluency path).
        
        
        out = content_response_text(
            "Retrieve rows\n<tool_call>{\"name\": \"t\"}</tool_call>",
            {"is_call_turn": True},
        )
        assert out.startswith("Retrieve rows")

    def test_empty_is_none(self):
        assert content_response_text("", {"is_reasoning_turn": True}) is None
        assert content_response_text(None, None) is None
        assert content_response_text("   ", None) is None


# ---------------------------------------------------------------------------
# TOOL_USE_CLASSES includes the 9 SGR-agent classes /
# TOOL_USE_CLASSES
# ---------------------------------------------------------------------------
class TestToolUseClassesIncludesSGR:
    def test_sgr_classes_present(self):
        for cls in (
            "coder", "creative_writer", "data_analyst", "deep_research",
            "factual_qa", "instruction_following", "math_reasoner",
            "rewriter", "summarizer",
        ):
            assert cls in TOOL_USE_CLASSES

    def test_opencode_gen_still_present(self):
        assert "opencode_gen" in TOOL_USE_CLASSES
        assert "tool_calls_multiturn" in TOOL_USE_CLASSES
