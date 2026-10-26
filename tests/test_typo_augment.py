'Tests of the offline keyboard-typo augmentation (llm_training.augment.typo).\n\n\nThe span-boundary tests run WITHOUT augmentex (the module imports it\nlazily); the augmentation-behaviour tests skip when augmentex is absent\n(core-only environments).'

from __future__ import annotations

import pytest

from llm_training.augment.typo import (
    DEFAULT_TYPO_CLASSES,
    SYSTEM_TYPO_CLASSES,
    TypoConfig,
    _levenshtein,
    _markers_intact,
    _menonqa_typed_ranges,
    _OPTION_MARKER_RE,
    apply_typos,
    apply_system_typos,
    augment_pool,
    augment_pool_system,
    resolve_classes,
    resolve_system_classes,
    system_spans,
    typed_spans,
)
from llm_training.schema import UnifiedExample

SGR_TASK = (
    "Подготовьте хронологическое саммаризацию ключевых событий из статьи "
    "https://ru.wikipedia.org/wiki/История_искусственного_интеллекта "
    "и выделите три наиболее значимых прорыва с обоснованием."
)
SGR_DATE = (
    "Current Date: 2026-06-29 22:13:59 (Year-Month-Day ISO format: "
    "YYYY-MM-DD HH:MM:SS)"
)


def _sgr_example() -> UnifiedExample:
    return UnifiedExample(
        task_class="summarizer",
        prompt=[
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": SGR_TASK},
            {"role": "user", "content": SGR_DATE},
            {"role": "assistant", "content": "Reasoning steps: ..."},
            {"role": "tool", "content": "extracted article body"},
        ],
        reference="Reasoning steps: ...",
        extra={"is_reasoning_turn": True},
        source="sgr_agent/val_summarizer.jsonl#0#turn0",
    )


def _span_texts(example: UnifiedExample) -> list[str]:
    return [
        (example.prompt[mi].get("content") or "")[s:e]
        for mi, s, e in typed_spans(example)
    ]


# ---------------------------------------------------------------- boundaries



class TestTypedSpans:
    def test_sgr_task_line_only(self):
        """The SGR task line is typed; the date preamble and tool results are not."""
        spans = typed_spans(_sgr_example())
        assert spans == [(1, 0, len(SGR_TASK))]

    def test_system_and_tool_roles_never_typed(self):
        ex = _sgr_example()
        touched = {mi for mi, _, _ in typed_spans(ex)}
        roles = {i: m["role"] for i, m in enumerate(ex.prompt)}
        for i in touched:
            assert roles[i] == "user"

    def test_menon_qa_documents_untouchable(self):
        docs = "D" * 500
        question = "Расскажи мне о поступлении на мехмат."
        instruction = "Ответьте, опираясь только на факты из документов."
        query = (
            "==========\nABBREVIATION DICTIONARY\n==========\n\nАБВ — ассоциация\n"
            f"\n==========\nDOCUMENT 1\n==========\n\n{docs}\n"
            f"\n==========\nCURRENT QUESTION\n==========\n\n{question}\n"
            f"\n==========\nINSTRUCTION\n==========\n\n{instruction}\n"
        )
        ex = UnifiedExample(
            task_class="menon_qa",
            prompt=[
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": query},
            ],
            reference="ответ",
            source="menon_qa.jsonl#0",
        )
        texts = _span_texts(ex)
        assert question in texts
        assert instruction in texts
        assert docs not in "".join(texts)
        assert "АБВ" not in "".join(texts)

    def test_menon_qa_without_sections_whole_message(self):
        ex = UnifiedExample(
            task_class="menon_qa",
            prompt=[
                {"role": "system", "content": "s"},
                {"role": "user", "content": "Привет! Как дела?"},
            ],
            reference="Привет!",
            source="menon_qa.jsonl#1",
        )
        assert _span_texts(ex) == ["Привет! Как дела?"]

    def test_mtrageval_context_prepended(self):
        ctx = "Контекстная статья о мехмате. " * 10
        question = "Расскажи о поступлении."
        ex = UnifiedExample(
            task_class="mtrageval",
            prompt=[
                {"role": "user", "content": ctx + "\n\n" + question},
            ],
            reference="ответ",
            extra={"context_text": ctx},
            source="mtrageval_ru.json#Factoid#0",
        )
        assert _span_texts(ex) == [question]

    def test_mtrageval_context_appended(self):
        ctx = "Статья для вопроса. " * 10
        question = "Что такое RAGEval?"
        ex = UnifiedExample(
            task_class="mtrageval",
            prompt=[{"role": "user", "content": question + "\n\n" + ctx}],
            reference="ответ",
            extra={"context_text": ctx},
            source="mtrageval_ru.json#Factoid#1",
        )
        assert _span_texts(ex) == [question]

    def test_mtrageval_context_in_system(self):
        ex = UnifiedExample(
            task_class="mtrageval",
            prompt=[
                {"role": "system", "content": "контекст в system"},
                {"role": "user", "content": "вопрос целиком"},
            ],
            reference="ответ",
            extra={"context_text": "контекст в system"},
            source="mtrageval_ru.json#Factoid#2",
        )
        assert _span_texts(ex) == ["вопрос целиком"]

    def test_multihop_rag_before_marker_only(self):
        question = "Who founded the company?"
        chunks = "chunk one ... chunk two ..."
        ex = UnifiedExample(
            task_class="multihop_rag",
            prompt=[
                {"role": "user", "content": question + "\nThe chunk list:\n" + chunks},
            ],
            reference="answer",
            source="multihop_rag.jsonl#0",
        )
        assert _span_texts(ex) == [question + "\n"]

    def test_multihop_rag_without_marker_skipped(self):
        ex = UnifiedExample(
            task_class="multihop_rag",
            prompt=[{"role": "user", "content": "no marker here at all"}],
            reference="answer",
            source="multihop_rag.jsonl#1",
        )
        assert typed_spans(ex) == []

    def test_docsem_quant_question_line_only(self):
        blocks = "b07: Team On-call staff ... Question: inline occurrence"
        question = "What is the requested result for the report's scenario?"
        ex = UnifiedExample(
            task_class="docsem_quant",
            prompt=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": f"Document blocks:\n{blocks}\n\nQuestion: {question}"},
            ],
            reference="{}",
            source="docsem_quant.jsonl#0",
        )
        # The LAST marker wins (a block body may itself contain the word).
        
        assert _span_texts(ex) == [question]

    def test_docsem_quant_without_marker_skipped(self):
        ex = UnifiedExample(
            task_class="docsem_quant",
            prompt=[{"role": "user", "content": "no marker here at all"}],
            reference="{}",
            source="docsem_quant.jsonl#1",
        )
        assert typed_spans(ex) == []

    def test_generic_classes_whole_user_messages(self):
        for cls in ("self_cognition", "rummlu_pro", "opencode_gen",
                    "tool_calls_multiturn", "coder"):
            ex = UnifiedExample(
                task_class=cls,
                prompt=[
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "первый вопрос"},
                    {"role": "assistant", "content": "ответ"},
                    {"role": "tool", "content": "tool output"},
                    {"role": "user", "content": "второй вопрос"},
                ],
                reference="ответ",
                source=f"{cls}#0",
            )
            assert _span_texts(ex) == ["первый вопрос", "второй вопрос"], cls

    def test_classes_without_boundary_yield_no_spans(self):
        for cls in ("asr_summarization", "ragu_entity_extraction",
                    "nerel_instruct"):
            ex = UnifiedExample(
                task_class=cls,
                prompt=[{"role": "user", "content": "любой текст"}],
                reference="ответ",
                source=f"{cls}#0",
            )
            assert typed_spans(ex) == []

    def test_no_user_messages(self):
        ex = UnifiedExample(
            task_class="coder",
            prompt=[{"role": "system", "content": "only system"}],
            reference="ответ",
            source="coder#1",
        )
        assert typed_spans(ex) == []


class TestMenonqaRanges:
    def test_exact_contents(self):
        question, instruction = "вопрос целиком", "инструкция целиком"
        query = (
            "==========\nDOCUMENT 1\n==========\n\nDOCBODY\n"
            f"\n==========\nCURRENT QUESTION\n==========\n\n{question}\n"
            f"\n==========\nINSTRUCTION\n==========\n\n{instruction}\n"
        )
        assert [query[s:e] for s, e in _menonqa_typed_ranges(query)] == [
            question,
            instruction,
        ]


# --------------------------------------------------------------------- config



class TestTypoConfig:
    def test_defaults(self):
        cfg = TypoConfig()
        assert cfg.share == 0.0
        assert cfg.classes == DEFAULT_TYPO_CLASSES
        assert "asr_summarization" not in cfg.classes

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"share": -0.1},
            {"share": 1.5},
            {"min_aug": 0},
            {"max_aug": 1, "min_aug": 2},
            {"platforms": ()},
            {"platforms": ("tablet",)},
            {"max_attempts": 0},
            {"copies": -1},
            {"classes": frozenset({"asr_summarization"})},
        ],
    )
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            TypoConfig(**kwargs)

    def test_resolve_classes_default_and_exclude(self):
        assert resolve_classes() == DEFAULT_TYPO_CLASSES
        resolved = resolve_classes(exclude=["menon_qa", "summarizer"])
        assert resolved == DEFAULT_TYPO_CLASSES - {"menon_qa", "summarizer"}

    def test_resolve_classes_explicit_include(self):
        resolved = resolve_classes(include=["coder", "menon_qa"])
        assert resolved == frozenset({"coder", "menon_qa"})

    def test_resolve_classes_unknown_include_rejected(self):
        with pytest.raises(ValueError):
            resolve_classes(include=["asr_summarization"])
        with pytest.raises(ValueError):
            resolve_classes(include=["no_such_class"])


class TestHelpers:
    def test_levenshtein(self):
        assert _levenshtein("кот", "кот") == 0
        assert _levenshtein("кот", "код") == 1
        assert _levenshtein("", "аб") == 2

    def test_option_marker_regex(self):
        text = "Выберите вариант:\nа. первый\nб. второй\nB) third"
        markers = _OPTION_MARKER_RE.findall(text)
        assert "а. " in markers and "B) " in markers

    def test_markers_intact(self):
        assert _markers_intact("а. кот\nб. пёс", "а. кот\nб. пёс", "rummlu_pro")
        altered = "а. кот\nб. пёс".replace("б.", "6.")
        assert not _markers_intact("а. кот\nб. пёс", altered, "rummlu_pro")
        assert _markers_intact("x", "y", "coder")


# ---------------------------------------------------------- augmentex-backed



def _aug_cfg(**overrides) -> TypoConfig:
    defaults = {"share": 1.0, "seed": 42}
    defaults.update(overrides)
    return TypoConfig(**defaults)


def _long_ru_example(source: str = "summarizer#0") -> UnifiedExample:
    return UnifiedExample(
        task_class="summarizer",
        prompt=[
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": SGR_TASK},
            {"role": "user", "content": SGR_DATE},
        ],
        reference="эталонный ответ",
        source=source,
    )


@pytest.fixture()
def _need_augmentex():
    pytest.importorskip("augmentex")


class TestApplyTypos:
    def test_light_changes_and_metadata(self, _need_augmentex):
        ex = _long_ru_example()
        new, n = apply_typos(ex, _aug_cfg())
        assert 1 <= n <= 2
        task = new.prompt[1]["content"]
        assert task != SGR_TASK
        assert _levenshtein(SGR_TASK, task) == n
        assert new.extra["typo_augmented"] is True
        assert new.extra["typo_platform"] in ("pc", "mobile")
        # The untouched parts stay byte-identical; the example itself is a copy.
        assert new.prompt[0] == ex.prompt[0]
        assert new.prompt[2]["content"] == SGR_DATE
        assert new.reference == ex.reference
        assert ex.prompt[1]["content"] == SGR_TASK

    def test_multi_span_splice_accumulates(self, monkeypatch):
        """Two spans in ONE message (menon_qa question + instruction) must
        both survive: the second splice builds on the first, not on the
        original content (a clobbering bug would keep only the last span)."""
        import llm_training.augment.typo as typo_mod

        def fake_typo(text: str, lang: str, platform: str, seed_int: int,
                      cfg: TypoConfig) -> str:
            return text.replace("о", "а", 1) if "о" in text else text

        monkeypatch.setattr(typo_mod, "_augmentex_typo", fake_typo)
        docs = "ДОКУМЕНТ ТЕЛА СТАТЬИ. " * 20
        question = "Расскажи мне о поступлении на мехмат."
        instruction = "Сформулируйте ответ кратко."
        query = (
            f"==========\nDOCUMENT 1\n==========\n\n{docs}\n"
            f"\n==========\nCURRENT QUESTION\n==========\n\n{question}\n"
            f"\n==========\nINSTRUCTION\n==========\n\n{instruction}\n"
        )
        ex = UnifiedExample(
            task_class="menon_qa",
            prompt=[
                {"role": "system", "content": "s"},
                {"role": "user", "content": query},
            ],
            reference="ответ",
            source="menon_qa.jsonl#7",
        )
        new, n = apply_typos(ex, TypoConfig(share=1.0, seed=42))
        out = new.prompt[1]["content"]
        assert n == 2
        assert question.replace("о", "а", 1) in out
        assert instruction.replace("о", "а", 1) in out
        assert question not in out and instruction not in out
        assert docs in out
        assert "CURRENT QUESTION" in out and "INSTRUCTION" in out

    def test_share_zero_off(self, _need_augmentex):
        ex = _long_ru_example()
        new, n = apply_typos(ex, _aug_cfg(share=0.0))
        assert new is ex and n == 0

    def test_no_letters_no_augmentation(self, _need_augmentex):
        ex = UnifiedExample(
            task_class="self_cognition",
            prompt=[{"role": "user", "content": "42 42 42 !!! ..."}],
            reference="ответ",
            source="self_cognition.jsonl#9",
        )
        new, n = apply_typos(ex, _aug_cfg())
        assert new is ex and n == 0

    def test_deterministic(self, _need_augmentex):
        ex = _long_ru_example()
        first, _ = apply_typos(ex, _aug_cfg())
        second, _ = apply_typos(ex, _aug_cfg())
        assert first.prompt == second.prompt

    def test_occurrence_changes_seed(self, _need_augmentex):
        ex = _long_ru_example()
        first, _ = apply_typos(ex, _aug_cfg(), occurrence=0)
        second, _ = apply_typos(ex, _aug_cfg(), occurrence=1)
        # Identical only in the (improbable) case of the same draw; the
        # seeds differ by construction, so assert the mechanism, not luck:
        # both must still be valid light corruptions of the same original.
        assert _levenshtein(SGR_TASK, first.prompt[1]["content"]) <= 2
        assert _levenshtein(SGR_TASK, second.prompt[1]["content"]) <= 2

    def test_excluded_class_untouched(self, _need_augmentex):
        ex = UnifiedExample(
            task_class="asr_summarization",
            prompt=[{"role": "user", "content": "Транскрипт распознанной речи."}],
            reference="саммари",
            source="asr_summarization/NN01",
        )
        new, n = apply_typos(ex, _aug_cfg())
        assert new is ex and n == 0

    def test_menon_qa_documents_survive(self, _need_augmentex):
        docs = "Документ один. " * 40
        question = "Расскажи мне о поступлении на мехмат южного федерального университета."
        query = (
            f"==========\nDOCUMENT 1\n==========\n\n{docs}\n"
            f"\n==========\nCURRENT QUESTION\n==========\n\n{question}\n"
        )
        ex = UnifiedExample(
            task_class="menon_qa",
            prompt=[
                {"role": "system", "content": "s"},
                {"role": "user", "content": query},
            ],
            reference="ответ",
            source="menon_qa.jsonl#5",
        )
        new, n = apply_typos(ex, _aug_cfg())
        assert n >= 1
        assert docs in new.prompt[1]["content"]
        assert "==========\nDOCUMENT 1\n==========\n\n" in new.prompt[1]["content"]
        assert "CURRENT QUESTION" in new.prompt[1]["content"]


class TestAugmentPool:
    def _pool(self, n: int = 6) -> list[UnifiedExample]:
        return [
            UnifiedExample(
                task_class="summarizer",
                prompt=[{"role": "user", "content": SGR_TASK + f" Вариант {i}."}],
                reference="эталон",
                source=f"summarizer#{i}",
            )
            for i in range(n)
        ]

    def test_share_zero_noop(self):
        pool = self._pool()
        out, stats = augment_pool(pool, TypoConfig(share=0.0, seed=7))
        assert out == pool and stats["enabled"] is False

    def test_exact_share_and_pool_size(self, _need_augmentex):
        pool = self._pool(n=6)
        out, stats = augment_pool(pool, _aug_cfg(share=0.5, seed=7))
        assert stats["selected"] == 3  # ceil(0.5 * 6)
        assert stats["augmented"] + stats["reverted_clean"] == 3
        assert len(out) == len(pool)
        augmented = [ex for ex in out if (ex.extra or {}).get("typo_augmented")]
        assert len(augmented) == stats["augmented"]

    def test_deterministic_pool(self, _need_augmentex):
        pool = self._pool()
        first, _ = augment_pool(pool, _aug_cfg(seed=11))
        second, _ = augment_pool(pool, _aug_cfg(seed=11))
        assert [ex.prompt for ex in first] == [ex.prompt for ex in second]

    def test_class_without_boundary_stays_clean(self, _need_augmentex):
        pool = [
            UnifiedExample(
                task_class="asr_summarization",
                prompt=[{"role": "user", "content": "Транскрипт речи."}],
                reference="саммари",
                source="asr#0",
            )
        ]
        out, stats = augment_pool(pool, _aug_cfg(share=1.0, seed=3))
        assert stats["eligible"] == 0 and stats["selected"] == 0
        assert out == pool

    def test_duplicates_augmented_independently(self, _need_augmentex):
        base = self._pool(n=1)
        pool = [*base, *base]
        out, stats = augment_pool(pool, _aug_cfg(share=1.0, seed=5))
        assert stats["selected"] == 2
        assert len(out) == 2
        assert stats["per_class"]["summarizer"]["selected"] == 2
        # The two copies of the same source must not be forced into the same
        # corruption: the occurrence number is part of the seed. Both stay
        # light corruptions of the original either way.
        originals = [ex.prompt[0]["content"] for ex in pool]
        for out_ex, orig in zip(out, originals):
            assert _levenshtein(orig, out_ex.prompt[0]["content"]) <= 2


class TestAugmentPoolMultiVariant:
    'Multi-variant scheme: clean original + K augmented copies.'

    def _pool(self, n: int = 6) -> list[UnifiedExample]:
        return [
            UnifiedExample(
                task_class="summarizer",
                prompt=[{"role": "user", "content": SGR_TASK + f" Вариант {i}."}],
                reference="эталон",
                source=f"summarizer#{i}",
            )
            for i in range(n)
        ]

    def test_pool_size_and_original_clean(self, _need_augmentex):
        pool = self._pool(n=6)
        out, stats = augment_pool(pool, _aug_cfg(share=1.0, seed=7, copies=2))
        assert stats["copies"] == 2
        assert stats["n_in"] == 6 and stats["n_out"] == 18
        assert len(out) == 3 * 6
        # The first N are the clean original — no typo flag, byte-identical.
        for orig, kept in zip(pool, out[:6]):
            assert kept.prompt == orig.prompt
            assert (kept.extra or {}).get("typo_augmented") is None

    def test_copies_carry_copy_index_and_differ(self, _need_augmentex):
        pool = self._pool(n=2)
        out, stats = augment_pool(pool, _aug_cfg(share=1.0, seed=9, copies=2))
        assert len(out) == 6
        # Copies 0 and 1 of source summarizer#0 land at indices 2 and 4.
        c0, c1 = out[2], out[4]
        assert (c0.extra or {}).get("typo_copy") == 0
        assert (c1.extra or {}).get("typo_copy") == 1
        # Both are light corruptions of the same original; the seeds differ
        # by construction, so assert the mechanism (valid, light) not luck.
        orig = pool[0].prompt[0]["content"]
        assert _levenshtein(orig, c0.prompt[0]["content"]) <= 2
        assert _levenshtein(orig, c1.prompt[0]["content"]) <= 2

    def test_deterministic(self, _need_augmentex):
        pool = self._pool()
        first, _ = augment_pool(pool, _aug_cfg(seed=11, copies=2))
        second, _ = augment_pool(pool, _aug_cfg(seed=11, copies=2))
        assert [ex.prompt for ex in first] == [ex.prompt for ex in second]

    def test_share_fraction_per_copy(self, _need_augmentex):
        pool = self._pool(n=6)
        out, stats = augment_pool(pool, _aug_cfg(share=0.5, seed=7, copies=2))
        assert stats["selected"] == 6  # ceil(0.5 * 6) = 3 per copy, two copies
        assert len(stats["per_copy"]) == 2
        assert all(c <= 3 for c in stats["per_copy"])
        assert sum(stats["per_copy"]) == stats["augmented"]
        assert len(out) == 18

    def test_noneligible_duplicated_identical(self, _need_augmentex):
        pool = [
            UnifiedExample(
                task_class="asr_summarization",
                prompt=[{"role": "user", "content": "Транскрипт речи."}],
                reference="саммари",
                source="asr#0",
            )
        ]
        out, stats = augment_pool(pool, _aug_cfg(share=1.0, seed=3, copies=2))
        assert stats["eligible"] == 0
        assert len(out) == 3  # original + 2 byte-identical copies
        assert out[0].prompt == out[1].prompt == out[2].prompt
        assert all((ex.extra or {}).get("typo_augmented") is None for ex in out)

    def test_copies_zero_legacy_size_unchanged(self, _need_augmentex):
        pool = self._pool(n=6)
        out, stats = augment_pool(pool, _aug_cfg(share=0.5, seed=7, copies=0))
        assert stats["copies"] == 0
        assert len(out) == len(pool)
        assert "per_copy" not in stats


# ----------------------------------------------------------------- system spans



def _sys_example(cls: str, system: str = "Вы — грамотный ассистент.") -> UnifiedExample:
    return UnifiedExample(
        task_class=cls,
        prompt=[
            {"role": "system", "content": system},
            {"role": "user", "content": "вопрос пользователя"},
        ],
        reference="ответ",
        source=f"{cls}#0",
    )


class TestSystemSpans:
    def test_plain_instruction_full_span(self):
        text = "Выделите все именованные сущности."
        ex = _sys_example("nerel_instruct", text)
        assert system_spans(ex) == [(0, 0, len(text))]

    def test_allowlisted_classes_have_spans(self):
        for cls in ("nerel_instruct", "menon_qa", "rummlu_pro",
                    "self_cognition", "coder"):
            assert system_spans(_sys_example(cls)) != [], cls

    def test_excluded_classes_no_spans(self):
        for cls in ("tool_calls_multiturn", "docsem_quant", "meno_rerank",
                    "mtrageval", "asr_summarization", "ragu_entity_extraction"):
            assert system_spans(_sys_example(cls)) == [], cls

    def test_tools_block_excluded(self):
        system = "Инструкция до блока. <tools>[...]</tools> После блока."
        ex = _sys_example("nerel_instruct", system)
        spans = system_spans(ex)
        texts = "".join(
            (ex.prompt[mi].get("content") or "")[s:e] for mi, s, e in spans
        )
        assert "<tools>" not in texts
        assert "Инструкция до блока." in texts
        assert "После блока." in texts

    def test_empty_system_no_spans(self):
        assert system_spans(_sys_example("nerel_instruct", "")) == []

    def test_no_system_message(self):
        ex = UnifiedExample(
            task_class="nerel_instruct",
            prompt=[{"role": "user", "content": "вопрос"}],
            reference="ответ",
            source="nerel#1",
        )
        assert system_spans(ex) == []


class TestSystemConfig:
    def test_system_share_validation(self):
        for bad in (-0.1, 1.5):
            with pytest.raises(ValueError, match="system_share"):
                TypoConfig(system_share=bad)
        with pytest.raises(ValueError, match="system_classes"):
            TypoConfig(system_classes=frozenset({"tool_calls_multiturn"}))
        assert TypoConfig(system_share=0.15).system_share == 0.15

    def test_resolve_system_classes(self):
        assert resolve_system_classes() == SYSTEM_TYPO_CLASSES
        assert resolve_system_classes(exclude=["nerel_instruct"]) == (
            SYSTEM_TYPO_CLASSES - {"nerel_instruct"}
        )
        assert resolve_system_classes(include=["nerel_instruct", "menon_qa"]) == (
            frozenset({"nerel_instruct", "menon_qa"})
        )
        with pytest.raises(ValueError):
            resolve_system_classes(include=["tool_calls_multiturn"])


class TestApplySystemTypos:
    def _nerel(self, source="nerel#0") -> UnifiedExample:
        return UnifiedExample(
            task_class="nerel_instruct",
            prompt=[
                {"role": "system", "content": "Выделите все именованные сущности из текста."},
                {"role": "user", "content": "Текст для разметки."},
            ],
            reference="ответ",
            source=source,
        )

    def test_system_changed_user_untouched(self, _need_augmentex):
        ex = self._nerel()
        new, n = apply_system_typos(ex, TypoConfig(system_share=1.0, seed=42))
        assert n >= 1
        assert new.extra["system_typo_augmented"] is True
        assert new.prompt[1]["content"] == "Текст для разметки."
        assert new.prompt[0]["content"] != ex.prompt[0]["content"]
        assert ex.prompt[0]["content"] == "Выделите все именованные сущности из текста."

    def test_system_share_zero_off(self):
        ex = self._nerel()
        new, n = apply_system_typos(ex, TypoConfig(system_share=0.0, seed=42))
        assert new is ex and n == 0

    def test_excluded_class_off(self, _need_augmentex):
        ex = _sys_example("tool_calls_multiturn", "Системный промпт с <tools>")
        new, n = apply_system_typos(ex, TypoConfig(system_share=1.0, seed=42))
        assert new is ex and n == 0


class TestAugmentPoolSystem:
    def _pool(self, n=6) -> list[UnifiedExample]:
        return [
            UnifiedExample(
                task_class="nerel_instruct",
                prompt=[
                    {"role": "system", "content": "Выделите все сущности из текста."},
                    {"role": "user", "content": f"Текст {i}."},
                ],
                reference="ответ",
                source=f"nerel#{i}",
            )
            for i in range(n)
        ]

    def test_share_zero_noop(self):
        pool = self._pool()
        out, stats = augment_pool_system(pool, TypoConfig(system_share=0.0, seed=7))
        assert out == pool and stats["enabled"] is False

    def test_exact_share_and_size(self, _need_augmentex):
        pool = self._pool(n=6)
        out, stats = augment_pool_system(pool, TypoConfig(system_share=0.5, seed=7))
        assert stats["selected"] == 3
        assert len(out) == len(pool)
        assert stats["augmented"] + stats["reverted_clean"] == 3

    def test_deterministic(self, _need_augmentex):
        pool = self._pool()
        first, _ = augment_pool_system(pool, TypoConfig(system_share=0.5, seed=11))
        second, _ = augment_pool_system(pool, TypoConfig(system_share=0.5, seed=11))
        assert [ex.prompt for ex in first] == [ex.prompt for ex in second]
