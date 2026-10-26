'Tests for runtime/difficulty.py — the loss-based data filter.\n\n\nThe pure logic is offline-testable in full: ``filter_by_difficulty`` (the\nper-class quantiles, the floor drop counts, the score-missing / small-class\nexemptions, the stats/report) and the helpers (``_plan_batches``, the\ndevice/attn resolvers, the scoring-summary logger). ``score_example_difficulty``\nis thin torch glue over ``mask_example``; its contract (reference-span CE,\nlength normalization, token-budget batching, padding/label shift) is pinned by\na tiny CPU Qwen2 model, and ``load_difficulty_model`` is pinned by the\ntiny-model CPU-fallback test (the r5 warm-up crash: the model silently scored\non CPU with ``flash_attention_2``).\n\n``filter_by_difficulty``\n``score_example_difficulty``'

from __future__ import annotations

import logging
import math

import pytest

from llm_training.runtime.difficulty import (
    DEFAULT_MAX_BATCH_TOKENS,
    _log_scoring_summary,
    _plan_batches,
    _resolve_scoring_attn,
    _resolve_scoring_device,
    filter_by_difficulty,
    load_difficulty_model,
    score_example_difficulty,
)
from llm_training.schema import UnifiedExample


def _ex(source, task_class="menon_qa"):
    return UnifiedExample(
        task_class=task_class,
        prompt=[{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
        reference="ref",
        reference_messages=[{"role": "assistant", "content": "ref"}],
        source=source,
    )


def test_noop_when_shares_zero():
    examples = [_ex(f"s{i}") for i in range(10)]
    kept, stats = filter_by_difficulty(
        examples, {f"s{i}": float(i) for i in range(10)}
    )
    assert kept == examples
    assert stats.enabled is False
    assert stats.n_input == stats.n_kept == 10
    assert stats.n_dropped == 0
    assert stats.dropped_per_class == {}
    assert "disabled" in stats.format_report()


def test_drops_both_tails_per_class():
    # class A: 10 examples loss 0..9; class B: 5 examples loss 0..4.
    # drop_easy 0.2 → floor(2) easiest; drop_hard 0.2 → floor(2) hardest.
    a = [_ex(f"a{i}", "menon_qa") for i in range(10)]
    b = [_ex(f"b{i}", "mtrageval") for i in range(5)]
    scores = {f"a{i}": float(i) for i in range(10)}
    scores.update({f"b{i}": float(i) for i in range(5)})
    kept, stats = filter_by_difficulty(
        [*a, *b], scores, drop_easy=0.2, drop_hard=0.2,
    )
    kept_a = {ex.source for ex in kept if ex.task_class == "menon_qa"}
    kept_b = {ex.source for ex in kept if ex.task_class == "mtrageval"}
    # A: drop a0,a1 (easiest) + a8,a9 (hardest) → keep a2..a7.
    assert kept_a == {f"a{i}" for i in range(2, 8)}
    # B: drop b0 (easiest) + b4 (hardest) → keep b1..b3.
    assert kept_b == {f"b{i}" for i in range(1, 4)}
    assert stats.dropped_per_class == {"menon_qa": 4, "mtrageval": 2}
    assert stats.n_dropped == 6
    assert stats.n_kept == 9
    a_stats = stats.classes["menon_qa"]
    assert a_stats.n_before == 10 and a_stats.n_after == 6
    assert a_stats.n_dropped_easy == 2 and a_stats.n_dropped_hard == 2
    assert a_stats.dropped_share == pytest.approx(0.4)
    assert stats.classes["mtrageval"].n_dropped_easy == 1
    assert stats.classes["mtrageval"].n_dropped_hard == 1


def test_per_class_not_global():
    # A small class must not be wiped out by a global-style quantile: its own
    # tail shares apply, and it must survive when it has few examples.
    a = [_ex(f"a{i}", "menon_qa") for i in range(30)]
    b = [_ex(f"b{i}", "mtrageval") for i in range(3)]  # below min_class_size
    scores = {f"a{i}": float(i) for i in range(30)}
    scores.update({f"b{i}": float(i) for i in range(3)})
    kept, stats = filter_by_difficulty(
        [*a, *b], scores, drop_easy=0.1, drop_hard=0.1,
    )
    # A: floor(3) each tail → 6 dropped; B: untouched (min_class_size=4).
    assert {ex.task_class for ex in kept} == {"menon_qa", "mtrageval"}
    assert sum(1 for ex in kept if ex.task_class == "mtrageval") == 3
    assert stats.dropped_per_class == {"menon_qa": 6}
    assert stats.classes["mtrageval"].skip_reason == "below_min_class_size"
    assert stats.n_classes_skipped_small == 1


def test_missing_score_kept():
    # An example the scorer never scored (a failure) is KEPT — the filter
    # must never drop data it could not evaluate.
    
    examples = [_ex(f"s{i}") for i in range(10)]
    scores = {f"s{i}": float(i) for i in range(10) if i != 5}
    kept, stats = filter_by_difficulty(
        examples, scores, drop_easy=0.5, drop_hard=0.5,
    )
    assert "s5" in {ex.source for ex in kept}
    assert stats.n_unscored == 1
    assert stats.classes["menon_qa"].n_unscored == 1
    # floor(0.5 * 9) = 4 per tail; s5 (unscored) and s4 survive.
    assert stats.n_dropped == 8
    assert stats.n_kept == 2
    assert {ex.source for ex in kept} == {"s4", "s5"}


def test_deterministic_and_stable():
    # Same input → same output (the index is the final tie-breaker under
    # duplicate losses).
    examples = [_ex(f"s{i}") for i in range(10)]
    scores = {f"s{i}": 0.5 for i in range(10)}  # all-equal losses
    kept1, stats1 = filter_by_difficulty(
        examples, scores, drop_easy=0.2, drop_hard=0.2
    )
    kept2, stats2 = filter_by_difficulty(
        examples, scores, drop_easy=0.2, drop_hard=0.2
    )
    assert [e.source for e in kept1] == [e.source for e in kept2]
    assert stats1.format_report() == stats2.format_report()
    # 4 dropped (2 each tail); the surviving order preserves the input order.
    assert len(kept1) == 6
    assert [e.source for e in kept1] == [f"s{i}" for i in range(2, 8)]


def test_drop_counts_are_floors():
    # n=9: floor(0.2*9)=1 per tail → 2 dropped, 7 kept.
    examples = [_ex(f"s{i}") for i in range(9)]
    scores = {f"s{i}": float(i) for i in range(9)}
    kept, stats = filter_by_difficulty(
        examples, scores, drop_easy=0.2, drop_hard=0.2,
    )
    assert len(kept) == 7
    assert stats.n_dropped == 2


def test_drop_shares_are_bounded():
    # A share >= 1 drops at most every scored example (no negative slicing).
    examples = [_ex(f"s{i}") for i in range(5)]
    scores = {f"s{i}": float(i) for i in range(5)}
    kept, stats = filter_by_difficulty(
        examples, scores, drop_easy=2.0, drop_hard=0.0,
    )
    assert kept == []  # floor(2.0 * 5) = 10 >= n → everything dropped
    assert stats.n_dropped == 5
    assert stats.n_kept == 0
    assert stats.dropped_share == pytest.approx(1.0)
    assert stats.classes["menon_qa"].n_dropped_easy == 5
    assert stats.classes["menon_qa"].n_dropped_hard == 0


@pytest.mark.parametrize("bad", [None, {}, {"x": 1}])
def test_empty_scores_is_noop(bad):
    examples = [_ex("s0"), _ex("s1")]
    kept, stats = filter_by_difficulty(
        examples, bad, drop_easy=0.5, drop_hard=0.5,
    )
    assert kept == examples
    assert stats.enabled is True
    assert stats.n_dropped == 0
    assert stats.n_unscored == 2
    assert stats.classes["menon_qa"].skip_reason == "unscored"
    assert stats.n_classes_unscored == 1


def test_stats_per_class_counts_and_shares():
    a = [_ex(f"a{i}", "menon_qa") for i in range(10)]
    b = [_ex(f"b{i}", "mtrageval") for i in range(6)]
    scores = {f"a{i}": float(i) for i in range(10)}
    scores.update({f"b{i}": float(i) for i in range(6)})
    kept, stats = filter_by_difficulty(
        [*a, *b], scores, drop_easy=0.2, drop_hard=0.0,
    )
    a_stats = stats.classes["menon_qa"]
    assert a_stats.n_before == 10
    assert a_stats.n_scored == 10
    assert a_stats.n_unscored == 0
    assert a_stats.n_dropped_easy == 2
    assert a_stats.n_dropped_hard == 0
    assert a_stats.n_dropped == 2
    assert a_stats.n_after == 8
    assert a_stats.dropped_share == pytest.approx(0.2)
    # floor(0.2 * 6) = 1 for class B.
    assert stats.classes["mtrageval"].n_dropped_easy == 1
    assert stats.n_input == 16
    assert stats.n_kept == 13
    assert stats.n_dropped == 3
    assert stats.dropped_share == pytest.approx(3 / 16)
    assert stats.n_classes == 2


def test_stats_duplicates_counted():
    # The final-turn boost duplicates rows with the same source; the
    # duplicate shares the score but is an independently droppable row.
    examples = [_ex("s0"), _ex("s1"), _ex("s2"), _ex("s3"), _ex("s4"), _ex("s0")]
    scores = {f"s{i}": float(i) for i in range(5)}
    kept, stats = filter_by_difficulty(
        examples, scores, drop_easy=0.2, drop_hard=0.2,
    )
    assert stats.n_input == 6
    assert stats.n_duplicates == 1
    assert stats.classes["menon_qa"].n_duplicates == 1
    assert stats.n_dropped == 2  # floor(0.2 * 6) = 1 per tail
    # idx0 (s0, easiest) and idx4 (s4, hardest) drop; the s0 copy survives.
    assert [ex.source for ex in kept] == ["s1", "s2", "s3", "s0"]
    # The loss summary weights UNIQUE sources only (duplicate not doubled).
    assert stats.loss_mean == pytest.approx(2.0)


def test_stats_skipped_classes_and_unscored():
    a = [_ex(f"a{i}", "menon_qa") for i in range(10)]
    b = [_ex(f"b{i}", "mtrageval") for i in range(3)]  # below min_class_size
    c = [_ex(f"c{i}", "rummlu_pro") for i in range(5)]  # no scores at all
    scores = {f"a{i}": float(i) for i in range(10)}
    scores.update({f"b{i}": float(i) for i in range(3)})
    kept, stats = filter_by_difficulty(
        [*a, *b, *c], scores, drop_easy=0.1, drop_hard=0.1,
    )
    assert stats.n_classes == 3
    assert stats.n_classes_skipped_small == 1
    assert stats.n_classes_unscored == 1
    assert stats.classes["mtrageval"].skip_reason == "below_min_class_size"
    assert stats.classes["rummlu_pro"].skip_reason == "unscored"
    assert stats.classes["rummlu_pro"].loss_min is None
    assert stats.n_unscored == 5
    report = stats.format_report()
    assert "skipped below min_class_size=4" in report
    assert "unscored classes: rummlu_pro" in report


def test_stats_loss_summary():
    examples = [_ex(f"s{i}") for i in range(5)] + [_ex("s0")]
    scores = {"s0": 0.5, "s1": 1.0, "s2": 1.5, "s3": 2.0, "s4": 2.5}
    _, stats = filter_by_difficulty(
        examples, scores, drop_easy=0.2, drop_hard=0.2,
    )
    assert stats.loss_min == pytest.approx(0.5)
    assert stats.loss_mean == pytest.approx(1.5)
    assert stats.loss_max == pytest.approx(2.5)
    assert stats.loss_summary == "0.500/1.500/2.500"
    cls = stats.classes["menon_qa"]
    assert (cls.loss_min, cls.loss_mean, cls.loss_max) == pytest.approx(
        (0.5, 1.5, 2.5)
    )
    _, no_scores = filter_by_difficulty(examples, {}, drop_easy=0.2)
    assert no_scores.loss_min is None
    assert no_scores.loss_summary == "-"


def test_format_report_content():
    examples = [_ex(f"s{i}", "coder") for i in range(6)]
    scores = {f"s{i}": float(i) for i in range(6)}
    _, stats = filter_by_difficulty(
        examples, scores, drop_easy=0.5, drop_hard=0.0,
    )
    report = stats.format_report()
    assert "per class" in report
    assert "drop_easy=0.500" in report
    assert "class" in report
    assert "coder" in report
    assert "TOTAL" in report
    assert "share" in report
    assert "loss min/mean/max" in report


def test_tail_overlap_reported():
    examples = [_ex(f"s{i}") for i in range(5)]
    scores = {f"s{i}": float(i) for i in range(5)}
    kept, stats = filter_by_difficulty(
        examples, scores, drop_easy=1.0, drop_hard=1.0,
    )
    assert kept == []
    assert stats.n_dropped == 5
    assert stats.n_tail_overlap == 5
    assert stats.classes["menon_qa"].n_dropped_easy == 5
    assert stats.classes["menon_qa"].n_dropped_hard == 5
    assert "tails overlap" in stats.format_report()


class TestPlanBatches:
    def test_respects_batch_size(self):
        assert _plan_batches(
            [10] * 5, batch_size=2, max_batch_tokens=10**9
        ) == [[0, 1], [2, 3], [4]]

    def test_respects_token_budget(self):
        # 4 x 5000 padded tokens: 3 fit 16384 (15000), the 4th opens a batch.
        assert _plan_batches(
            [5000] * 4, batch_size=8, max_batch_tokens=16384
        ) == [[0, 1, 2], [3]]

    def test_padding_is_counted(self):
        # The batch is padded to its longest sequence.
        assert _plan_batches(
            [1000, 8000], batch_size=8, max_batch_tokens=16384
        ) == [[0, 1]]
        assert _plan_batches(
            [1000, 9000], batch_size=8, max_batch_tokens=16384
        ) == [[0], [1]]

    def test_over_budget_example_gets_own_batch(self):
        assert _plan_batches(
            [30000, 100, 100], batch_size=8, max_batch_tokens=16384
        ) == [[0], [1, 2]]

    def test_empty(self):
        assert _plan_batches([], batch_size=8, max_batch_tokens=16384) == []

    def test_default_budget_constant(self):
        assert DEFAULT_MAX_BATCH_TOKENS == 16384


class TestResolvers:
    def test_device(self):
        assert _resolve_scoring_device(None, True) == "cuda"
        assert _resolve_scoring_device(None, False) == "cpu"
        assert _resolve_scoring_device("cpu", True) == "cpu"
        assert _resolve_scoring_device("cuda:1", False) == "cuda:1"

    def test_attn(self):
        assert _resolve_scoring_attn("flash_attention_2", "cuda") == (
            "flash_attention_2"
        )
        assert _resolve_scoring_attn("flash_attention_2", "cpu") == "sdpa"
        assert _resolve_scoring_attn("sdpa", "cpu") == "sdpa"
        assert _resolve_scoring_attn("eager", "cuda:1") == "eager"


def test_log_scoring_summary(caplog):
    with caplog.at_level(logging.INFO, logger="llm_training.runtime.difficulty"):
        _log_scoring_summary(
            n_input=10, n_unique=9, n_duplicates=1, n_empty_ref=2,
            n_scored=7, elapsed_s=1.25,
        )
    text = caplog.text
    assert "scored 7 of 9 unique" in text
    assert "1 duplicate source(s)" in text
    assert "2 empty-reference" in text
    assert "1.2s" in text


def test_log_scoring_summary_empty_is_warning(caplog):
    with caplog.at_level(
        logging.WARNING, logger="llm_training.runtime.difficulty"
    ):
        _log_scoring_summary(
            n_input=3, n_unique=3, n_duplicates=0, n_empty_ref=3,
            n_scored=0, elapsed_s=0.1,
        )
    assert caplog.records[-1].levelno == logging.WARNING
    assert "no example produced a score" in caplog.text


def _tiny_model_dir(tmp_path) -> str:
    'Tiny Qwen2 + minimal tokenizer on disk'
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    vocab = {
        "<pad>": 0, "<eos>": 1, "<unk>": 2, "system:": 3, "user:": 4,
        "assistant:": 5, "s": 6, "q": 7, "ref": 8,
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    )
    config = Qwen2Config(
        vocab_size=len(vocab),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    return str(tmp_path)


class TestLoadDifficultyModel:
    def test_cpu_fallback_uses_sdpa_and_stays_on_cpu(self, tmp_path):
        # The r5 warm-up crash: flash_attention_2 + a model left on CPU.
        # On CPU the attn must fall back to sdpa and the device stay CPU.
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        model_dir = _tiny_model_dir(tmp_path)
        model, tokenizer = load_difficulty_model(
            model_dir, attn="flash_attention_2", device="cpu"
        )
        assert model.device.type == "cpu"
        assert model.config._attn_implementation == "sdpa"
        assert tokenizer.pad_token_id is not None

    @pytest.mark.gpu
    def test_cuda_placement_when_available(self, tmp_path):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        model_dir = _tiny_model_dir(tmp_path)
        model, _ = load_difficulty_model(model_dir, attn="sdpa")
        assert model.device.type == "cuda"


def test_score_example_difficulty_tiny_model(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model_dir = _tiny_model_dir(tmp_path)
    model, tokenizer = load_difficulty_model(model_dir, attn="sdpa", device="cpu")
    # The duplicate source ("a") is scored once; the tiny max_batch_tokens
    # forces per-example batches (padding + label-shift path).
    examples = [_ex("a"), _ex("b"), _ex("a")]
    scores = score_example_difficulty(
        model, tokenizer, examples,
        batch_size=2, max_batch_tokens=8, progress=False,
    )
    assert set(scores) == {"a", "b"}
    assert all(math.isfinite(v) for v in scores.values())
