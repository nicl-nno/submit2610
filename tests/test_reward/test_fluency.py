'Tests for reward/fluency.py — FluencyScorer gating, cache, aggregation.\n\n\nExercises the **real** :class:`FluencyScorer` logic offline (no torch, no\nrazdel, no GPU) by injecting fake ``classifier`` / ``sentenize_fn`` /\n``word_count_fn`` / ``calibrator`` via the DI seams. The transformers / razdel\nlazy-import paths are not covered here (they are exercised by the @gpu e2e\ntest on a real machine'

from __future__ import annotations

import asyncio

import pytest

from llm_training.reward.calibration import IdentityCalibrator, IsotonicCalibrator
from llm_training.reward.config import RewardConfig
from llm_training.reward.fluency import (
    FluencyScorer,
    _resolve_fluency_label,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeClassifier:
    'Records calls; returns a fixed P(grammatical) per sentence.'

    def __init__(self, per_text_score: float = 0.9) -> None:
        self.per_text_score = per_text_score
        self.calls: list[list[str]] = []

    def classify(self, texts: list[str]) -> list[float]:
        self.calls.append(list(texts))
        return [self.per_text_score for _ in texts]


class _RaisingClassifier:
    def classify(self, texts: list[str]) -> list[float]:
        raise RuntimeError("model boom")


def _dot_sentenize(text: str) -> list[str]:
    "Trivial splitter on '.' — good enough for unit tests (not real Russian)."
    return [s.strip() for s in text.split(".") if s.strip()]


def _word_count(text: str) -> int:
    'Space-split alpha-token counter (fake of razdel-based count_words).'
    return sum(1 for w in text.split() if any(c.isalpha() for c in w))


def _cfg(**overrides) -> RewardConfig:
    base = dict(
        fluency_classes=frozenset({"menon_qa", "coder", "nerel_instruct"}),
        fluency_concurrency=2,
    )
    base.update(overrides)
    return RewardConfig(**base)


# A 3-word sentence passes the MIN_FLUENCY_WORDS=3 gate; a 1-2-word one does not.

_LONG = "Первое длинное предложение"


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------
class TestFluencyGating:
    pytestmark = pytest.mark.asyncio
    async def test_basic_multisentence_returns_calibrated_mean(self):
        clf = _FakeClassifier(per_text_score=0.8)
        scorer = FluencyScorer(
            _cfg(), classifier=clf, sentenize_fn=_dot_sentenize,
            word_count_fn=_word_count, calibrator=IdentityCalibrator(),
        )
        val = await scorer.score(f"{_LONG}. {_LONG}.", "menon_qa")
        assert val == pytest.approx(0.8)
        assert len(clf.calls) == 1
        assert len(clf.calls[0]) == 2  # two sentences classified in one batch

    async def test_class_outside_allowlist_is_none(self):
        clf = _FakeClassifier()
        scorer = FluencyScorer(
            _cfg(), classifier=clf, sentenize_fn=_dot_sentenize,
            word_count_fn=_word_count, calibrator=IdentityCalibrator(),
        )
        val = await scorer.score(f"{_LONG}. {_LONG}.", "math_reasoner")
        assert val is None
        assert clf.calls == []  # no classification ran

    async def test_too_few_words_per_sentence_is_none(self):
        clf = _FakeClassifier()
        scorer = FluencyScorer(
            _cfg(), classifier=clf, sentenize_fn=_dot_sentenize,
            word_count_fn=_word_count, calibrator=IdentityCalibrator(),
        )
        # Two sentences but each only 1 word → all dropped by the word gate.
        
        val = await scorer.score("Один. Два.", "menon_qa")
        assert val is None
        assert clf.calls == []

    async def test_single_long_sentence_is_scored(self):
        # A single sentence with ≥3 words IS scored now (the CoLa classifier is
        # sentence-level; one well-formed sentence is in-distribution).
        
        clf = _FakeClassifier(per_text_score=0.6)
        scorer = FluencyScorer(
            _cfg(), classifier=clf, sentenize_fn=_dot_sentenize,
            word_count_fn=_word_count, calibrator=IdentityCalibrator(),
        )
        val = await scorer.score(_LONG, "menon_qa")
        assert val == pytest.approx(0.6)
        assert clf.calls == [[_LONG]]

    async def test_empty_prose_is_none(self):
        clf = _FakeClassifier()
        scorer = FluencyScorer(
            _cfg(), classifier=clf, sentenize_fn=_dot_sentenize,
            word_count_fn=_word_count, calibrator=IdentityCalibrator(),
        )
        # Only code, no prose → N/A.
        
        val = await scorer.score("```python\nx = 1\n```", "coder")
        assert val is None
        assert clf.calls == []

    async def test_enable_fluency_false_is_none(self):
        clf = _FakeClassifier()
        scorer = FluencyScorer(
            _cfg(enable_fluency=False), classifier=clf,
            sentenize_fn=_dot_sentenize, word_count_fn=_word_count,
            calibrator=IdentityCalibrator(),
        )
        val = await scorer.score(f"{_LONG}. {_LONG}.", "menon_qa")
        assert val is None
        assert clf.calls == []

    async def test_code_stripped_before_segmentation(self):
        # A coder answer with prose + Python: the code is stripped, only the
        # prose sentences are classified.
        
        clf = _FakeClassifier(per_text_score=0.5)
        scorer = FluencyScorer(
            _cfg(), classifier=clf, sentenize_fn=_dot_sentenize,
            word_count_fn=_word_count, calibrator=IdentityCalibrator(),
        )
        response = f"{_LONG}. {_LONG}.\n```python\nx = 1\n```"
        val = await scorer.score(response, "coder")
        assert val == pytest.approx(0.5)
        # The code body must NOT have been classified as a sentence.
        
        classified = clf.calls[0]
        assert all("x = 1" not in t for t in classified)
        assert len(classified) == 2  # two prose sentences


# ---------------------------------------------------------------------------
# Sub-config gating (nerel_instruct)
# ---------------------------------------------------------------------------
class TestFluencySubconfigGating:
    pytestmark = pytest.mark.asyncio

    async def test_definition_subconfig_scored(self):
        clf = _FakeClassifier(per_text_score=0.7)
        scorer = FluencyScorer(
            _cfg(), classifier=clf, sentenize_fn=_dot_sentenize,
            word_count_fn=_word_count, calibrator=IdentityCalibrator(),
        )
        val = await scorer.score(
            _LONG, "nerel_instruct", extra={"config": "entity_definition"}
        )
        assert val == pytest.approx(0.7)

    async def test_extraction_subconfig_is_none(self):
        # The extraction sub-configs are entity lists outputs — fluency does NOT
        # apply (a deterministic F1 scores them instead).
        
        clf = _FakeClassifier()
        scorer = FluencyScorer(
            _cfg(), classifier=clf, sentenize_fn=_dot_sentenize,
            word_count_fn=_word_count, calibrator=IdentityCalibrator(),
        )
        val = await scorer.score(
            _LONG, "nerel_instruct", extra={"config": "entity_recognition"}
        )
        assert val is None
        assert clf.calls == []

    async def test_missing_config_metadata_is_none(self):
        # No extra / no config key → the subtask allowlist is unsatisfied.
        
        clf = _FakeClassifier()
        scorer = FluencyScorer(
            _cfg(), classifier=clf, sentenize_fn=_dot_sentenize,
            word_count_fn=_word_count, calibrator=IdentityCalibrator(),
        )
        val = await scorer.score(_LONG, "nerel_instruct")
        assert val is None
        assert clf.calls == []

    async def test_class_without_subtask_map_ignores_extra(self):
        # menon_qa has no subtask entry → extra is irrelevant, fluency runs.
        
        clf = _FakeClassifier(per_text_score=0.7)
        scorer = FluencyScorer(
            _cfg(), classifier=clf, sentenize_fn=_dot_sentenize,
            word_count_fn=_word_count, calibrator=IdentityCalibrator(),
        )
        val = await scorer.score(_LONG, "menon_qa", extra={"config": "whatever"})
        assert val == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Calibration wiring
# ---------------------------------------------------------------------------
class TestFluencyCalibration:
    pytestmark = pytest.mark.asyncio

    async def test_isotonic_calibrator_applied_per_sentence(self):
        # Raw 0.9 per sentence → calibrator maps it to 0.4; aggregation keeps it
        # 0.4 for a single sentence (calibration is now per-sentence, not on a
        # text-level mean).
        
        
        cal = IsotonicCalibrator(x=[0.8, 0.95], y=[0.0, 0.6])
        clf = _FakeClassifier(per_text_score=0.9)
        scorer = FluencyScorer(
            _cfg(), classifier=clf, sentenize_fn=_dot_sentenize,
            word_count_fn=_word_count, calibrator=cal,
        )
        val = await scorer.score(_LONG, "menon_qa")
        # np.interp(0.9; [0.8,0.95],[0.0,0.6]) = 0.6*(0.1/0.15)=0.4
        assert val == pytest.approx(0.4)

    async def test_default_calibrator_is_identity(self):
        # No calibrator_path configured → IdentityCalibrator (raw per-sentence).
        
        clf = _FakeClassifier(per_text_score=0.7)
        scorer = FluencyScorer(
            _cfg(), classifier=clf, sentenize_fn=_dot_sentenize,
            word_count_fn=_word_count,
        )
        assert isinstance(scorer._calibrator, IdentityCalibrator)
        val = await scorer.score(_LONG, "menon_qa")
        assert val == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class TestFluencyCache:
    pytestmark = pytest.mark.asyncio

    async def test_repeated_sentence_reuses_cache(self):
        # Two answers share a sentence: the shared sentence is classified once
        # (cache hit on the second answer).
        
        clf = _FakeClassifier(per_text_score=0.5)
        scorer = FluencyScorer(
            _cfg(), classifier=clf, sentenize_fn=_dot_sentenize,
            word_count_fn=_word_count, calibrator=IdentityCalibrator(),
        )
        await scorer.score(f"Общее длинное предложение. {_LONG} А.", "menon_qa")
        await scorer.score(f"Общее длинное предложение. {_LONG} Б.", "menon_qa")
        # The shared sentence is NOT re-classified in the second call.
        
        assert len(clf.calls) == 2
        assert clf.calls[0] == ["Общее длинное предложение", f"{_LONG} А"]
        assert clf.calls[1] == [f"{_LONG} Б"]

    async def test_cache_size_bounded(self):
        # Many distinct sentences do not raise and produce results.
        
        clf = _FakeClassifier(per_text_score=0.5)
        scorer = FluencyScorer(
            _cfg(fluency_concurrency=1), classifier=clf,
            sentenize_fn=_dot_sentenize, word_count_fn=_word_count,
            calibrator=IdentityCalibrator(),
        )
        text = ". ".join(f"Предложение номер {i} здесь" for i in range(50)) + "."
        val = await scorer.score(text, "menon_qa")
        assert val == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Failure propagation
# ---------------------------------------------------------------------------
class TestFluencyFailure:
    pytestmark = pytest.mark.asyncio

    async def test_model_error_propagates(self):
        # A model failure must raise (so RAGASScorer can record "fluency" in
        # failed_metrics) — it must NOT be swallowed into None.
        
        scorer = FluencyScorer(
            _cfg(), classifier=_RaisingClassifier(),
            sentenize_fn=_dot_sentenize, word_count_fn=_word_count,
            calibrator=IdentityCalibrator(),
        )
        with pytest.raises(RuntimeError, match="model boom"):
            await scorer.score(_LONG, "menon_qa")


# ---------------------------------------------------------------------------
# _resolve_fluency_label
# ---------------------------------------------------------------------------
class TestResolveFluencyLabel:
    def test_preferred_int_present(self):
        class _Cfg:
            id2label = {0: "unacceptable", 1: "acceptable"}
            label2id = {"unacceptable": 0, "acceptable": 1}

        assert _resolve_fluency_label(_Cfg(), 1) == 1

    def test_grammatical_label_present(self):
        # The grammar classifier: {0: "ungrammatical", 1: "grammatical"}.
        
        class _Cfg:
            id2label = {0: "ungrammatical", 1: "grammatical"}
            label2id = {"ungrammatical": 0, "grammatical": 1}

        assert _resolve_fluency_label(_Cfg(), 1) == 1

    def test_grammatical_resolved_by_name(self):
        # Positive label reachable by name even when the preferred index is
        # absent from id2label.
        
        class _Cfg:
            id2label = {0: "bad", 1: "good"}
            label2id = {"bad": 0, "good": 1, "grammatical": 1}

        assert _resolve_fluency_label(_Cfg(), 99) == 1

    def test_falls_back_to_name(self):
        class _Cfg:
            id2label = {0: "LABEL_0", 1: "LABEL_1"}
            label2id = {"LABEL_0": 0, "LABEL_1": 1}

        # preferred absent → name fallback LABEL_1.
        
        assert _resolve_fluency_label(_Cfg(), 99) == 1

    def test_acceptable_name_resolved(self):
        class _Cfg:
            id2label = {0: "no", 1: "yes"}
            label2id = {"no": 0, "yes": 1, "acceptable": 1}

        assert _resolve_fluency_label(_Cfg(), 99) == 1

    def test_empty_config_returns_preferred(self):
        class _Cfg:
            id2label = {}
            label2id = {}

        assert _resolve_fluency_label(_Cfg(), 1) == 1


# ---------------------------------------------------------------------------
# raw_fluency_per_sentence (offline analysis helper)
# ---------------------------------------------------------------------------
class TestRawFluencyPerSentence:
    def test_returns_unique_sentence_score_pairs(self):
        from llm_training.reward.fluency import raw_fluency_per_sentence

        clf = _FakeClassifier(per_text_score=0.6)
        cfg = _cfg()
        # Two texts sharing one sentence ("
        
        texts = [f"{_LONG} альфа. {_LONG} бета.", f"{_LONG} альфа. {_LONG} гамма."]
        out = raw_fluency_per_sentence(
            texts, cfg, classifier=clf, sentenize_fn=_dot_sentenize,
            word_count_fn=_word_count,
        )
        sentences = [s for s, _ in out]
        assert len(sentences) == len(set(sentences)) == 3  # all unique, 3 of them
        assert all(sc == pytest.approx(0.6) for _, sc in out)
        # The classifier saw each unique sentence exactly once (one batched call).
        
        assert len(clf.calls) == 1
        assert len(clf.calls[0]) == 3

    def test_short_sentences_dropped(self):
        from llm_training.reward.fluency import raw_fluency_per_sentence

        clf = _FakeClassifier(per_text_score=0.6)
        out = raw_fluency_per_sentence(
            ["Один. Два."], _cfg(),
            classifier=clf, sentenize_fn=_dot_sentenize,
            word_count_fn=_word_count,
        )
        assert out == []  # both sentences < 3 words → nothing survives
        assert clf.calls == []

    def test_code_only_text_is_none(self):
        from llm_training.reward.fluency import raw_fluency_per_sentence

        clf = _FakeClassifier()
        out = raw_fluency_per_sentence(
            ["```python\nx = 1\n```"], _cfg(),
            classifier=clf, sentenize_fn=_dot_sentenize,
            word_count_fn=_word_count,
        )
        assert out == []
        assert clf.calls == []  # nothing classified

    def test_empty_text_is_skipped(self):
        from llm_training.reward.fluency import raw_fluency_per_sentence

        out = raw_fluency_per_sentence(
            [""], _cfg(),
            classifier=_FakeClassifier(), sentenize_fn=_dot_sentenize,
            word_count_fn=_word_count,
        )
        assert out == []

    def test_uncalibrated_values(self):
        # The helper returns RAW scores (no calibrator applied) — distinguishing
        # it from FluencyScorer.score which calibrates per sentence.
        
        from llm_training.reward.fluency import raw_fluency_per_sentence

        clf = _FakeClassifier(per_text_score=0.97)
        out = raw_fluency_per_sentence(
            [_LONG], _cfg(), classifier=clf, sentenize_fn=_dot_sentenize,
            word_count_fn=_word_count,
        )
        assert len(out) == 1
        assert out[0][1] == pytest.approx(0.97)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
class TestFluencyConcurrency:
    pytestmark = pytest.mark.asyncio

    async def test_semaphore_limits_concurrent_classify(self):
        # With fluency_concurrency=1, two concurrent answers whose sentences are
        # all cache misses must not overlap on classify().
        
        concurrent = 0
        max_seen = 0

        class _Tracking:
            def classify(self, texts):
                nonlocal concurrent, max_seen
                concurrent += 1
                max_seen = max(max_seen, concurrent)
                # Simulate a forward pass duration long enough to overlap.
                import time

                time.sleep(0.05)
                concurrent -= 1
                return [0.5 for _ in texts]

        scorer = FluencyScorer(
            _cfg(fluency_concurrency=1), classifier=_Tracking(),
            sentenize_fn=_dot_sentenize, word_count_fn=_word_count,
            calibrator=IdentityCalibrator(),
        )
        await asyncio.gather(
            scorer.score(f"{_LONG} А. {_LONG} Б.", "menon_qa"),
            scorer.score(f"{_LONG} В. {_LONG} Г.", "menon_qa"),
        )
        assert max_seen == 1
