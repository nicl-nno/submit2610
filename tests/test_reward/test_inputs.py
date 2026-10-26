'Tests for reward/inputs.py — per-class user-question extractors (no mocks).'

from __future__ import annotations

import pytest

from llm_training.reward.inputs import extract_user_question


def _prompt(*, system="s", user="u"):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class TestMenonQa:
    "The judge's ``user_input`` field must be the QUESTION, not the whole query."

    def test_extracts_current_question_and_instruction(self):
        # Full menon_qa query: abbreviation dict + 2 documents + question +
        # instruction. Only the question (with instruction) must reach the judge.
        
        query = (
            "==========\nABBREVIATION DICTIONARY\n==========\n\n"
            "AU — Anonymous University\n\n"
            "==========\nDOCUMENT 1\n==========\n\n"
            "AU основан в 1959 году.\n\n"
            "==========\nDOCUMENT 2\n==========\n\n"
            "Мехмат — лучший факультет.\n\n"
            "==========\nCURRENT QUESTION\n==========\n\nКогда основан AU?\n\n"
            "==========\nINSTRUCTION\n==========\nОтветьте на вопрос."
        )
        result = extract_user_question("menon_qa", _prompt(user=query), {})
        assert "Когда основан AU?" in result
        assert "Ответьте на вопрос" in result
        # Crucial: the documents and the abbreviation dictionary must NOT leak
        # into the judge's "question" — that was the original bug.
        
        assert "AU основан в 1959" not in result
        assert "Мехмат — лучший" not in result
        assert "ABBREVIATION DICTIONARY" not in result
        assert "DOCUMENT 1" not in result

    def test_question_without_instruction(self):
        # No INSTRUCTION section → just the question, no trailing newline.
        
        query = (
            "==========\nDOCUMENT 1\n==========\n\nКонтекст.\n\n"
            "==========\nCURRENT QUESTION\n==========\n\nЧто такое AU?"
        )
        result = extract_user_question("menon_qa", _prompt(user=query), {})
        assert result.strip() == "Что такое AU?"

    def test_chitchat_falls_back_to_full_text(self):
        # No CURRENT QUESTION section → no structured question to extract;
        # fall back to the full user text rather than returning empty.
        
        query = "Просто разговор без структуры."
        result = extract_user_question("menon_qa", _prompt(user=query), {})
        assert result == query

    def test_empty_query(self):
        assert extract_user_question("menon_qa", _prompt(user=""), {}) == ""

    def test_case_insensitive_header(self):
        query = "==========\ncurrent question\n==========\n\nТекст вопроса?"
        result = extract_user_question("menon_qa", _prompt(user=query), {})
        assert "Текст вопроса" in result


class TestMultihopRag:
    'The chunk list (retrieved context) must NOT leak into the question.'

    def test_extracts_before_marker(self):
        query = (
            "Carefully read the question and the chunks.\n"
            "The chunk list:\n"
            "Chunk A: Sam was CEO.\n"
            "Chunk B: FTX collapsed."
        )
        result = extract_user_question("multihop_rag", _prompt(user=query), {})
        assert "Carefully read the question" in result
        assert "Sam was CEO" not in result
        assert "FTX collapsed" not in result
        assert "The chunk list:" not in result

    def test_no_marker_falls_back_to_full(self):
        query = "Just a question with no chunk list marker."
        result = extract_user_question("multihop_rag", _prompt(user=query), {})
        assert result == query

    def test_empty_query(self):
        assert extract_user_question("multihop_rag", _prompt(user=""), {}) == ""


class TestMtrageval:
    'When the loader appended context_text to the user message, strip it.'

    def test_strips_appended_context(self):
        ctx = "Title 1\nBody of context.\n\nTitle 2\nSecond body."
        question = "What is the answer?"
        # Mirror exactly what the loader does: question + "\n\n" + context_text.
        
        user_text = question + "\n\n" + ctx
        result = extract_user_question(
            "mtrageval", _prompt(user=user_text), {"context_text": ctx}
        )
        assert result == question
        assert "Body of context" not in result

    def test_no_context_text_returns_full(self):
        # No context_text in extra → user message is the question verbatim.
        
        result = extract_user_question(
            "mtrageval", _prompt(user="What is 2+2?"), {}
        )
        assert result == "What is 2+2?"

    def test_context_not_at_end_returns_full_unchanged(self):
        # Defensive: if the user text does NOT end with the context block, leave
        # it alone (don't accidentally truncate a question that happens to contain
        # the context text by coincidence mid-message).
        
        ctx = "Some context."
        user_text = "Question mentioning Some context. in the middle"
        result = extract_user_question(
            "mtrageval", _prompt(user=user_text), {"context_text": ctx}
        )
        assert result == user_text


class TestDocsemQuant:
    "The document blocks must NOT leak into the judge's question."

    def test_extracts_after_last_marker(self):
        # A block body may itself contain the word "Question:" — the LAST
        # marker wins (the question line closes the message).
        
        
        query = (
            "Document blocks:\n"
            "b07: Team On-call staff. Question: how many are on call?\n"
            "b12: 98 total, 40 donated.\n\n"
            "Question: What is the requested result for the report's scenario?"
        )
        result = extract_user_question("docsem_quant", _prompt(user=query), {})
        assert result == (
            "What is the requested result for the report's scenario?"
        )
        assert "b07" not in result
        assert "98 total" not in result

    def test_no_marker_falls_back_to_full(self):
        query = "Just a question with no marker."
        result = extract_user_question("docsem_quant", _prompt(user=query), {})
        assert result == query

    def test_empty_query(self):
        assert extract_user_question("docsem_quant", _prompt(user=""), {}) == ""


class TestDefaultClasses:
    'For non-RAG classes the whole user message is the question already.'

    @pytest.mark.parametrize("cls", [
        "coder", "math_reasoner", "factual_qa", "rummlu_pro", "self_cognition",
        "asr_summarization", "rewriter", "summarizer", "creative_writer",
        "data_analyst", "deep_research", "instruction_following",
        "nerel_instruct", "opencode_gen", "tool_calls_multiturn",
        "ragu_artifact_extraction", "ragu_artifact_validation",
        "ragu_entity_extraction", "ragu_entity_validation",
        "ragu_relation_extraction", "ragu_relation_validation",
    ])
    def test_returns_full_user_text(self, cls):
        prompt = _prompt(user="This is the entire user message.")
        assert extract_user_question(cls, prompt, {}) == "This is the entire user message."

    def test_unknown_class_returns_full_user_text(self):
        # Unknown classes use the default extractor (full user text).
        
        result = extract_user_question("not_a_class", _prompt(user="hello"), {})
        assert result == "hello"


class TestEdgeCases:
    def test_no_user_message_returns_empty(self):
        # A prompt with only a system message must yield "".
        
        assert extract_user_question("menon_qa", [{"role": "system", "content": "s"}], {}) == ""
        assert extract_user_question("coder", [{"role": "system", "content": "s"}], {}) == ""

    def test_extra_none_is_handled(self):
        # extract_user_question must accept extra=None.
        
        result = extract_user_question(
            "mtrageval", _prompt(user="Question?"), None
        )
        assert result == "Question?"


class TestBuildRubricContext:
    'The judge-side context for the reasoning-turn rubric criteria.'

    def test_joins_system_user_and_rest(self):
        from llm_training.reward.inputs import build_rubric_context

        prompt = [
            {"role": "system", "content": "Задача."},
            {"role": "user", "content": "Дата"},
            {"role": "tool", "content": "результат 1"},
            {"role": "assistant", "content": "ход 1"},
        ]
        ctx = build_rubric_context(prompt)
        assert "Задача." in ctx
        assert "Дата" in ctx
        assert "результат 1" in ctx
        assert "ход 1" in ctx
        # The head keeps its order: system first, user second.
        
        assert ctx.index("Задача.") < ctx.index("Дата") < ctx.index("результат 1")

    def test_tail_truncation_keeps_head_and_recent_tail(self):
        from llm_training.reward.inputs import build_rubric_context

        prompt = [
            {"role": "system", "content": "З"},
            {"role": "user", "content": "П"},
            {"role": "tool", "content": "x" * 500},
            {"role": "tool", "content": "y" * 500},
        ]
        ctx = build_rubric_context(prompt, max_chars=300)
        assert len(ctx) <= 300
        assert "З" in ctx and "П" in ctx  # the head survives verbatim
        assert "y" * 100 in ctx  # the TAIL of the context survives
        assert not ctx.startswith("x")  # the head of a too-long tail is cut

    def test_empty_prompt_returns_empty(self):
        from llm_training.reward.inputs import build_rubric_context

        assert build_rubric_context([]) == ""
        assert build_rubric_context([{"role": "user", "content": "  "}]) == ""

    def test_last_user_message_wins(self):
        from llm_training.reward.inputs import build_rubric_context

        prompt = [
            {"role": "system", "content": "З"},
            {"role": "user", "content": "старое сообщение"},
            {"role": "tool", "content": "t"},
            {"role": "user", "content": "новое сообщение"},
        ]
        ctx = build_rubric_context(prompt)
        assert "новое сообщение" in ctx
        assert "старое сообщение" not in ctx  # only the last user message
