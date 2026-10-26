'Tests for build.py: length-discard, cap, split, determinism, baking.'

from __future__ import annotations

import pytest

from llm_training.build import (
    bake_example,
    build_full_dataset,
    build_train_stream,
    build_unified_dataset,
    to_dataset_dict,
)
from llm_training.config import DataConfig
from llm_training.schema import UnifiedExample

FAST_CLASSES = ["menon_qa", "self_cognition", "mtrageval"]


def _cfg(data_dir, **over) -> DataConfig:
    kw = dict(data_dir=data_dir, seed=42, max_per_class=200)
    kw.update(over)
    return DataConfig(**kw)


def test_build_offline_excludes_nerel_by_default(tokenizer, data_dir):
    # No network needed by default: nerel_instruct is excluded.
    
    result = build_unified_dataset(
        tokenizer, _cfg(data_dir), task_classes=FAST_CLASSES
    )
    assert "nerel_instruct" not in result.stats
    assert len(result.train) > 0


def test_length_discard_drops_overlong_reference(tokenizer, data_dir):
    # A synthetic example with a >8000-token reference must be discarded.
    
    huge_reference = "наука " * 40000  # well over 8000 tokens
    ex = UnifiedExample(
        task_class="menon_qa",
        prompt=[{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
        reference=huge_reference,
        reference_messages=[{"role": "assistant", "content": huge_reference}],
    )
    from llm_training.build import _passes_length_filter

    ok, reason = _passes_length_filter(ex, tokenizer, _cfg(data_dir))
    assert ok is False
    assert reason in {"total_too_long", "reference_too_long"}


def test_bpe_boundary_reference_not_discarded(tokenizer, data_dir):
    # A reference starting with "\n" triggers a benign BPE merge at the
    # ``<|im_start|>assistant\n`` + ``\n...`` boundary: ``prefix_ok`` is False
    # but the example is NOT malformed and must be kept.
    
    
    ex = UnifiedExample(
        task_class="coder",
        prompt=[{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
        reference="\n```bash\necho hello\n```",
        reference_messages=[{"role": "assistant", "content": "\n```bash\necho hello\n```"}],
    )
    from llm_training.masking import mask_example
    res = mask_example(ex, tokenizer)
    assert res.prefix_ok is False  # BPE boundary
    assert len(res.reference_ids) > 0
    from llm_training.build import _passes_length_filter

    ok, reason = _passes_length_filter(ex, tokenizer, _cfg(data_dir))
    assert ok is True, f"should be kept (got reason={reason})"


def test_cap_enforced(tokenizer, data_dir):
    # max_per_class=5 on self_cognition (136 records) -> 5 kept.
    result = build_unified_dataset(
        tokenizer, _cfg(data_dir, max_per_class=5),
        task_classes=["self_cognition"],
    )
    assert result.stats["self_cognition"]["kept"] == 5
    # small classes are NOT capped below their size.
    
    result2 = build_unified_dataset(
        tokenizer, _cfg(data_dir, max_per_class=200),
        task_classes=["mtrageval"],
    )
    assert result2.stats["mtrageval"]["kept"] == 45  # 45 total


def test_split_routes_whole_classes(tokenizer, data_dir):
    result = build_unified_dataset(
        tokenizer, _cfg(data_dir),
        val_task="menon_qa", test_task="self_cognition",
        task_classes=FAST_CLASSES,
    )
    assert {ex.task_class for ex in result.val} == {"menon_qa"}
    assert {ex.task_class for ex in result.test} == {"self_cognition"}
    # train has the remaining class(es), not val/test.
    assert "menon_qa" not in {ex.task_class for ex in result.train}
    assert "self_cognition" not in {ex.task_class for ex in result.train}
    assert "mtrageval" in {ex.task_class for ex in result.train}


def test_val_and_test_tasks_must_differ(tokenizer, data_dir):
    with pytest.raises(ValueError):
        build_unified_dataset(
            tokenizer, _cfg(data_dir),
            val_task="menon_qa", test_task="menon_qa",
            task_classes=FAST_CLASSES,
        )


def test_val_task_unknown_class_rejected(tokenizer, data_dir):
    # A typo in val_task must fail loudly, not silently produce an empty val
    # split with everything leaking into train.
    
    with pytest.raises(ValueError, match="val-task"):
        build_unified_dataset(
            tokenizer, _cfg(data_dir),
            val_task="menon_qa_typo", task_classes=FAST_CLASSES,
        )


def test_test_task_unknown_class_rejected(tokenizer, data_dir):
    with pytest.raises(ValueError, match="test-task"):
        build_unified_dataset(
            tokenizer, _cfg(data_dir),
            test_task="not_a_real_class", task_classes=FAST_CLASSES,
        )


def test_include_network_filters_nerel_even_with_explicit_task_classes(
    tokenizer, data_dir
):
    # The --include-network gate applies even when task_classes is explicitly
    # provided: nerel_instruct must be dropped without include_network=True.
    
    result = build_unified_dataset(
        tokenizer, _cfg(data_dir),
        task_classes=["menon_qa", "nerel_instruct"],
        include_network=False,
    )
    assert "nerel_instruct" not in result.stats
    assert {ex.task_class for ex in result.train} == {"menon_qa"}


def test_reference_length_uses_rendered_span(tokenizer, data_dir):
    # The reference-length discard counts the RENDERED reference span
    # (res.reference_ids), not the raw reference string. A reference whose
    # rendered form exceeds 8000 tokens is dropped with reason
    # "reference_too_long" even when its raw string is short enough.
    
    from llm_training.build import _passes_length_filter

    # Reference just over the 8000-token ceiling (rendered), but total stays
    # well under 32000 (so the reference_too_long branch is the one that fires,
    # not total_too_long).
    
    huge_reference = "наука " * 10000  # ~10k rendered tokens
    ex = UnifiedExample(
        task_class="menon_qa",
        prompt=[{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
        reference=huge_reference,
        reference_messages=[{"role": "assistant", "content": huge_reference}],
    )
    ok, reason = _passes_length_filter(ex, tokenizer, _cfg(data_dir))
    assert ok is False
    assert reason == "reference_too_long"


def test_discard_applied_before_cap(tokenizer, data_dir):
    # Discards happen BEFORE the per-class cap: a class with over-length
    # examples shows non-zero discard counts and the kept count is at most
    # max_per_class. Synthetic over-length reference built inline.
    
    from llm_training.build import _passes_length_filter

    long_ref = "длина " * 30000
    ex = UnifiedExample(
        task_class="menon_qa",
        prompt=[{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
        reference=long_ref,
        reference_messages=[{"role": "assistant", "content": long_ref}],
    )
    ok, reason = _passes_length_filter(ex, tokenizer, _cfg(data_dir))
    assert ok is False  # discarded


def test_determinism_same_seed_same_result(tokenizer, data_dir):
    r1 = build_unified_dataset(tokenizer, _cfg(data_dir, seed=7, max_per_class=10),
                               task_classes=FAST_CLASSES)
    r2 = build_unified_dataset(tokenizer, _cfg(data_dir, seed=7, max_per_class=10),
                               task_classes=FAST_CLASSES)
    assert [ex.source for ex in r1.train] == [ex.source for ex in r2.train]
    assert [ex.source for ex in r1.val] == [ex.source for ex in r2.val]


def test_different_seed_different_cap(tokenizer, data_dir):
    # With max_per_class=5 and a large class, different seeds yield different
    # sampled sources (with overwhelming probability).
    
    r1 = build_unified_dataset(tokenizer, _cfg(data_dir, seed=1, max_per_class=5),
                               task_classes=["self_cognition"])
    r2 = build_unified_dataset(tokenizer, _cfg(data_dir, seed=999, max_per_class=5),
                               task_classes=["self_cognition"])
    assert {ex.source for ex in r1.train} != {ex.source for ex in r2.train}


def test_bake_example_idempotent_for_non_tool(tokenizer, data_dir):
    result = build_unified_dataset(tokenizer, _cfg(data_dir), task_classes=["menon_qa"])
    ex = result.train[0]
    baked = bake_example(ex, tokenizer)
    assert baked.prompt == ex.prompt  # non-tool: no change


def test_bake_example_bakes_tools_for_tool_use(tokenizer, data_dir):
    result = build_unified_dataset(
        tokenizer, _cfg(data_dir), task_classes=["opencode_gen"]
    )
    ex = result.train[0]
    assert ex.tools  # has tools
    assert "<tools>" not in ex.prompt[0]["content"]  # not yet baked
    baked = bake_example(ex, tokenizer)
    assert "<tools>" in baked.prompt[0]["content"]
    # Idempotent: baking twice yields the same content.
    
    twice = bake_example(baked, tokenizer)
    assert twice.prompt[0]["content"] == baked.prompt[0]["content"]


def test_to_dataset_dict_produces_trl_columns(tokenizer, data_dir):
    result = build_unified_dataset(
        tokenizer, _cfg(data_dir),
        val_task="menon_qa", test_task="mtrageval",
        task_classes=FAST_CLASSES,
    )
    dd = to_dataset_dict(result, tokenizer)
    assert set(dd.keys()) == {"train", "validation", "test"}
    cols = dd["train"].column_names
    for required in ["prompt", "task_class", "reference"]:
        assert required in cols
    assert len(dd["validation"]) == len(result.val)
    assert len(dd["test"]) == len(result.test)
    # The prompt column is self-contained (tools baked) for tool-use examples.
    


# ---------------------------------------------------------------------------
# Final-turn boost (--final-turn-share)
# ---------------------------------------------------------------------------


def _n_final(examples):
    return sum(
        1 for ex in examples if (ex.extra or {}).get("is_final_answer")
    )


def test_final_turn_share_config_validation(data_dir):
    # The share must be in (0, 1) — 1.0 is unsatisfiable for a pool with
    # non-final turns; anything outside fails at DataConfig build.
    
    import pytest

    for bad in (0.0, -0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="final_turn_share"):
            DataConfig(data_dir=data_dir, final_turn_share=bad)
    assert DataConfig(data_dir=data_dir, final_turn_share=0.3).final_turn_share == 0.3
    assert DataConfig(data_dir=data_dir, final_turn_share=0.99).final_turn_share == 0.99


def test_final_turn_boost_off_by_default(tokenizer, data_dir):
    # No final_turn_share → the trainable SGR class keeps its capped size and
    # the boost stat is absent.
    
    result = build_unified_dataset(
        tokenizer, _cfg(data_dir, max_per_class=30),
        task_classes=["data_analyst"],
    )
    assert "final_turn_boost_added" not in result.stats["data_analyst"]
    assert len(result.train) == result.stats["data_analyst"]["kept"]


def test_final_turn_boost_raises_trainable_sgr_share(tokenizer, data_dir):
    # data_analyst carries reasoning/action/final turns; after the boost the
    # finals must reach the closed-form target share of the RESULTING pool,
    # originals preserved. The pre-boost composition comes from an unboosted
    # build with the same seed (the cap sample is per-class seeded).
    # data_analyst
    
    
    import math

    share = 0.3
    result = build_unified_dataset(
        tokenizer, _cfg(data_dir, max_per_class=30, final_turn_share=share),
        task_classes=["data_analyst"],
    )
    kept = result.stats["data_analyst"]["kept"]
    plain = build_unified_dataset(
        tokenizer, _cfg(data_dir, max_per_class=30),
        task_classes=["data_analyst"],
    )
    n_final_plain = _n_final(plain.train)
    target = math.ceil(share * (kept - n_final_plain) / (1.0 - share))
    expected_added = target - n_final_plain
    assert result.stats["data_analyst"]["final_turn_boost_added"] == expected_added
    assert len(result.train) == kept + expected_added
    assert _n_final(result.train) == target
    assert _n_final(result.train) >= share * len(result.train)
    # Every original source survives (duplicates only ADD rows).
    
    assert {ex.source for ex in plain.train} <= {ex.source for ex in result.train}


def test_final_turn_boost_never_touches_holdouts(tokenizer, data_dir):
    # summarizer is the val hold-out: its pool must stay EXACTLY the capped
    # sample — the anchor measuring instrument does not change composition.
    # summarizer
    plain = build_unified_dataset(
        tokenizer, _cfg(data_dir, max_per_class=30),
        task_classes=["summarizer", "data_analyst"],
        val_task="summarizer",
    )
    boosted = build_unified_dataset(
        tokenizer, _cfg(data_dir, max_per_class=30, final_turn_share=0.3),
        task_classes=["summarizer", "data_analyst"],
        val_task="summarizer",
    )
    assert [ex.source for ex in plain.val] == [ex.source for ex in boosted.val]
    assert "final_turn_boost_added" not in boosted.stats["summarizer"]
    assert boosted.stats["data_analyst"].get("final_turn_boost_added", 0) >= 0
    assert len(boosted.train) >= len(plain.train)


def test_final_turn_boost_noop_for_all_final_class(tokenizer, data_dir):
    # coder trajectories are a single final turn each: the class is already
    # at share 1.0, the boost adds nothing (stat recorded as 0).
    
    
    result = build_unified_dataset(
        tokenizer, _cfg(data_dir, max_per_class=20, final_turn_share=0.3),
        task_classes=["coder"],
    )
    assert result.stats["coder"]["final_turn_boost_added"] == 0
    assert len(result.train) == result.stats["coder"]["kept"]
    assert _n_final(result.train) == len(result.train)


# ---------------------------------------------------------------------------
# reasoning_keep_share — the deterministic reasoning-turn downsample /

# ---------------------------------------------------------------------------
def _n_reasoning(examples):
    return sum(
        1 for ex in examples if (ex.extra or {}).get("is_reasoning_turn")
    )


def test_reasoning_keep_share_config_validation(data_dir):
    # None / 1.0 = off; (0, 1) valid; 0 / >1 rejected.
    # None
    for good in (None, 1.0, 0.3, 0.99):
        DataConfig(data_dir=data_dir, reasoning_keep_share=good)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="reasoning_keep_share"):
            DataConfig(data_dir=data_dir, reasoning_keep_share=bad)


def test_reasoning_downsample_off_by_default(tokenizer, data_dir):
    # No reasoning_keep_share → the trainable SGR class keeps its capped
    # size, the downsample stat is absent.
    
    result = build_unified_dataset(
        tokenizer, _cfg(data_dir, max_per_class=30),
        task_classes=["data_analyst"],
    )
    assert "reasoning_downsampled" not in result.stats["data_analyst"]
    assert len(result.train) == result.stats["data_analyst"]["kept"]


def test_reasoning_downsample_thins_prefix_deterministically(tokenizer, data_dir):
    # share=0.5 keeps exactly floor(0.5 * N_reasoning) reasoning turns, every
    # non-reasoning example survives, and a rebuild with the same seed
    # reproduces the pool exactly.
    
    
    import math

    share = 0.5
    result = build_unified_dataset(
        tokenizer, _cfg(data_dir, max_per_class=30, reasoning_keep_share=share),
        task_classes=["data_analyst"],
    )
    plain = build_unified_dataset(
        tokenizer, _cfg(data_dir, max_per_class=30),
        task_classes=["data_analyst"],
    )
    n_reasoning_plain = _n_reasoning(plain.train)
    expected_kept = math.floor(share * n_reasoning_plain)
    assert result.stats["data_analyst"]["reasoning_downsampled"] == (
        n_reasoning_plain - expected_kept
    )
    assert _n_reasoning(result.train) == expected_kept
    assert len(result.train) == len(plain.train) - (n_reasoning_plain - expected_kept)
    # Every non-reasoning original survives (they carry no is_reasoning_turn).
    
    assert {ex.source for ex in plain.train if not (ex.extra or {}).get("is_reasoning_turn")} <= {
        ex.source for ex in result.train
    }
    # Deterministic rebuild → the same pool.
    
    again = build_unified_dataset(
        tokenizer, _cfg(data_dir, max_per_class=30, reasoning_keep_share=share),
        task_classes=["data_analyst"],
    )
    assert [ex.source for ex in again.train] == [ex.source for ex in result.train]


def test_reasoning_downsample_never_touches_holdouts(tokenizer, data_dir):
    # The val hold-out pool stays EXACTLY the capped sample — the anchor
    # measuring instrument does not change composition.
    
    plain = build_unified_dataset(
        tokenizer, _cfg(data_dir, max_per_class=30),
        task_classes=["summarizer", "data_analyst"],
        val_task="summarizer",
    )
    thinned = build_unified_dataset(
        tokenizer, _cfg(data_dir, max_per_class=30, reasoning_keep_share=0.3),
        task_classes=["summarizer", "data_analyst"],
        val_task="summarizer",
    )
    assert [ex.source for ex in plain.val] == [ex.source for ex in thinned.val]
    assert "reasoning_downsampled" not in thinned.stats["summarizer"]
    assert thinned.stats["data_analyst"].get("reasoning_downsampled", 0) >= 0
    assert len(thinned.train) <= len(plain.train)


def test_reasoning_downsample_composes_with_final_turn_boost(tokenizer, data_dir):
    # The downsample runs BEFORE the boost, so the boost's closed-form target
    # operates on the already-thinned pool — both stats land, the final pool
    # honours both constraints.
    
    
    result = build_unified_dataset(
        tokenizer,
        _cfg(data_dir, max_per_class=30, reasoning_keep_share=0.5, final_turn_share=0.3),
        task_classes=["data_analyst"],
    )
    stats = result.stats["data_analyst"]
    assert stats.get("reasoning_downsampled", 0) > 0
    assert "final_turn_boost_added" in stats
    assert _n_final(result.train) >= 0.3 * len(result.train)


# ---------------------------------------------------------------------------
# Fixed-val routing — the Meno-stand test split joins the val pool /

# ---------------------------------------------------------------------------
MENO_CLASSES = ["meno_rerank", "meno_rewrite", "meno_final"]


def test_fixed_val_rows_join_val_pool_before_cap(tokenizer, data_dir):
    # train.jsonl rows train; test.jsonl rows (extra["fixed_val"]) validate —
    # the classes participate in BOTH pools. The cap samples only the train
    # part, so the val rows survive even a tiny cap.
    
    
    result = build_unified_dataset(
        tokenizer,
        _cfg(data_dir, max_per_class=3),
        task_classes=MENO_CLASSES,
    )
    meno_val = [e for e in result.val if e.task_class in MENO_CLASSES]
    meno_train = [e for e in result.train if e.task_class in MENO_CLASSES]
    assert len(meno_val) == 71  # 63 rerank + 4 rewrite + 4 final (test split)
    assert all(e.extra.get("fixed_val") for e in meno_val)
    assert not any(e.extra.get("fixed_val") for e in meno_train)
    # The cap (3) applies to the train part only: rerank keeps 3 train rows
    # but ALL 63 val rows.
    rerank_train = [e for e in meno_train if e.task_class == "meno_rerank"]
    assert len(rerank_train) == 3
    assert result.stats["meno_rerank"]["fixed_val"] == 63
    assert result.stats["meno_rerank"]["kept"] == 3 + 63


def test_fixed_val_composes_with_holdout_val(tokenizer, data_dir):
    # With a leave-task-out hold-out: the val pool = the hold-out class PLUS
    # the fixed-val rows (the anchor pool gains the in-domain interpolation
    # component).
    result = build_unified_dataset(
        tokenizer,
        _cfg(data_dir, max_per_class=30),
        task_classes=[*MENO_CLASSES, "summarizer"],
        val_task="summarizer",
    )
    assert any(e.task_class == "summarizer" for e in result.val)
    assert sum(1 for e in result.val if e.task_class in MENO_CLASSES) == 71
    assert not any(e.task_class in MENO_CLASSES for e in result.train
                   if e.extra.get("fixed_val"))


def test_fixed_val_classes_rejected_as_holdouts(tokenizer, data_dir):
    # The Meno-stand classes already contribute fixed val rows — holding the
    # whole class out would collide with that routing.
    
    for flag in ("val_task", "test_task"):
        with pytest.raises(ValueError, match="fixed val rows"):
            build_unified_dataset(
                tokenizer,
                _cfg(data_dir, max_per_class=30),
                task_classes=MENO_CLASSES,
                **{flag: "meno_rerank"},
            )


def test_final_turn_boost_ignores_non_sgr_classes(tokenizer, data_dir):
    # A non-SGR class has no is_final_answer turns and is out of scope —
    # no boost stat even with the option on.
    
    result = build_unified_dataset(
        tokenizer, _cfg(data_dir, final_turn_share=0.5),
        task_classes=["menon_qa"],
    )
    assert "final_turn_boost_added" not in result.stats["menon_qa"]


def test_final_turn_boost_deterministic(tokenizer, data_dir):
    # Same seed + same share → identical train list (round-robin duplication
    # is RNG-free).
    
    kw = dict(
        max_per_class=30,
        final_turn_share=0.3,
    )
    r1 = build_unified_dataset(
        tokenizer, _cfg(data_dir, **kw), task_classes=["data_analyst"]
    )
    r2 = build_unified_dataset(
        tokenizer, _cfg(data_dir, **kw), task_classes=["data_analyst"]
    )
    assert [ex.source for ex in r1.train] == [ex.source for ex in r2.train]


# ---------------------------------------------------------------------------
# build_full_dataset — the FULL (uncapped) train pools + frozen val/test /
# build_full_dataset
# ---------------------------------------------------------------------------


def test_build_full_dataset_keeps_full_train_pools(tokenizer, data_dir):
    # self_cognition has 136 records: the full pool keeps all of them while
    # the legacy build_unified_dataset would cap it to max_per_class.
    
    full = build_full_dataset(
        tokenizer, _cfg(data_dir, max_per_class=5),
        task_classes=["self_cognition"],
    )
    assert len(full.train_pools["self_cognition"]) == 136
    legacy = build_unified_dataset(
        tokenizer, _cfg(data_dir, max_per_class=5),
        task_classes=["self_cognition"],
    )
    assert legacy.stats["self_cognition"]["kept"] == 5


def test_build_full_dataset_holdouts_stay_capped(tokenizer, data_dir):
    # The hold-out val/test pools are capped identically to the legacy build.
    
    full = build_full_dataset(
        tokenizer, _cfg(data_dir, max_per_class=10),
        val_task="menon_qa", test_task="self_cognition",
        task_classes=FAST_CLASSES,
    )
    legacy = build_unified_dataset(
        tokenizer, _cfg(data_dir, max_per_class=10),
        val_task="menon_qa", test_task="self_cognition",
        task_classes=FAST_CLASSES,
    )
    assert [ex.source for ex in full.val] == [ex.source for ex in legacy.val]
    assert [ex.source for ex in full.test] == [ex.source for ex in legacy.test]
    # mtrageval is a train class: full pool keeps all 45; the legacy build
    # caps it to 10.
    assert "mtrageval" in full.train_pools
    assert len(full.train_pools["mtrageval"]) == 45
    assert len([ex for ex in legacy.train if ex.task_class == "mtrageval"]) == 10


def test_build_full_dataset_multiple_val_holdouts(tokenizer, data_dir):
    # Multiple val hold-out classes route into the SAME (capped) val pool —
    # the merged-validation design (the former separate test split is folded
    # into val so the anchor validation becomes statistically more robust).
    
    
    full = build_full_dataset(
        tokenizer, _cfg(data_dir, max_per_class=10),
        val_task=["menon_qa", "self_cognition"],
        task_classes=FAST_CLASSES,
    )
    val_classes = {ex.task_class for ex in full.val}
    assert "menon_qa" in val_classes
    assert "self_cognition" in val_classes
    assert "menon_qa" not in full.train_pools
    assert "self_cognition" not in full.train_pools
    assert full.test == []
    assert "mtrageval" in full.train_pools


def test_build_rejects_val_test_collision(tokenizer, data_dir):
    # A --test-task that repeats a --val-task hold-out must be rejected (it
    # would double-book the anchor val pool).
    # --test-task
    with pytest.raises(ValueError):
        build_full_dataset(
            tokenizer, _cfg(data_dir),
            val_task=["menon_qa"], test_task="menon_qa",
            task_classes=FAST_CLASSES,
        )


# ---------------------------------------------------------------------------
# the balanced stream
# ---------------------------------------------------------------------------


def _pool(cls: str, n: int, extra: dict | None = None) -> list[UnifiedExample]:
    return [
        UnifiedExample(
            task_class=cls,
            prompt=[{"role": "user", "content": f"{cls} вопрос {i}"}],
            reference="эталон",
            source=f"{cls}#{i}",
            extra=dict(extra or {}),
        )
        for i in range(n)
    ]


def _pool_sizes(stream: list[UnifiedExample]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for ex in stream:
        sizes[ex.task_class] = sizes.get(ex.task_class, 0) + 1
    return sizes


def test_train_stream_exact_total():
    pools = {"a": _pool("menon_qa", 100), "b": _pool("self_cognition", 100)}
    stream, stats = build_train_stream(
        pools, train_examples=100, max_upsample_ratio=8.0, seed=42
    )
    assert len(stream) == 100
    assert stats["actual_total"] == 100
    sizes = _pool_sizes(stream)
    assert sizes == {"menon_qa": 50, "self_cognition": 50}


def test_train_stream_scarce_class_up_sampled_to_ratio():
    # pool=5, ratio=8 → allocated = min(desired, 40); with a big companion the
    # scarce class is up-sampled to 8× and the deficit goes to the abundant one.
    pools = {
        "menon_qa": _pool("menon_qa", 5),
        "self_cognition": _pool("self_cognition", 1000),
    }
    stream, stats = build_train_stream(
        pools, train_examples=200, max_upsample_ratio=8.0, seed=42
    )
    sizes = _pool_sizes(stream)
    assert sizes["menon_qa"] == 40  # 8 × 5, capped at the ratio
    assert sizes["self_cognition"] == 160  # the deficit redistributed
    assert len(stream) == 200
    assert stats["per_class"]["menon_qa"]["up_sampled"] == 35


def test_train_stream_deficit_redistributes_to_abundant():
    pools = {
        "menon_qa": _pool("menon_qa", 10),
        "self_cognition": _pool("self_cognition", 1000),
    }
    stream, stats = build_train_stream(
        pools, train_examples=300, max_upsample_ratio=8.0, seed=42
    )
    sizes = _pool_sizes(stream)
    # menon_qa capped at 8×10=80; self_cognition absorbs the rest fresh.
    assert sizes["menon_qa"] == 80
    assert sizes["self_cognition"] == 220
    assert len(stream) == 300


def test_train_stream_deterministic():
    pools = {"a": _pool("menon_qa", 50), "b": _pool("self_cognition", 7)}
    s1, _ = build_train_stream(
        pools, train_examples=60, max_upsample_ratio=8.0, seed=42
    )
    s2, _ = build_train_stream(
        pools, train_examples=60, max_upsample_ratio=8.0, seed=42
    )
    assert [ex.source for ex in s1] == [ex.source for ex in s2]


def test_train_stream_ratio_must_be_at_least_one():
    pools = {"menon_qa": _pool("menon_qa", 10)}
    with pytest.raises(ValueError, match="max_upsample_ratio"):
        build_train_stream(pools, train_examples=10, max_upsample_ratio=0.5, seed=42)


def test_train_stream_nerel_config_split_even():
    # NEREL's quota is split evenly across its 6 sub-configs.
    pools = {
        "nerel_instruct": [
            UnifiedExample(
                task_class="nerel_instruct",
                prompt=[{"role": "user", "content": f"q{i}"}],
                reference="r",
                source=f"nerel#{i}",
                extra={"config": cfg},
            )
            for cfg in ("entity_definition", "entity_normalization",
                        "entity_recognition", "relation_definition",
                        "relation_extraction", "relation_keywords")
            for i in range(60)
        ],
        "menon_qa": _pool("menon_qa", 1000),
    }
    stream, stats = build_train_stream(
        pools, train_examples=600, max_upsample_ratio=8.0, seed=42
    )
    nerel = [ex for ex in stream if ex.task_class == "nerel_instruct"]
    assert len(nerel) == 300  # half of 600, split among 6 configs
    per_config: dict[str, int] = {}
    for ex in nerel:
        per_config[ex.extra["config"]] = per_config.get(ex.extra["config"], 0) + 1
    assert set(per_config) == {
        "entity_definition", "entity_normalization", "entity_recognition",
        "relation_definition", "relation_extraction", "relation_keywords",
    }
    assert max(per_config.values()) - min(per_config.values()) <= 1


def test_train_stream_shortfall_warns_but_proceeds():
    # Total pool below train_examples → the stream is shorter than requested.
    pools = {"menon_qa": _pool("menon_qa", 6), "self_cognition": _pool("self_cognition", 4)}
    stream, stats = build_train_stream(
        pools, train_examples=1000, max_upsample_ratio=8.0, seed=42
    )
    # ratio cap: 8×6=48 + 8×4=32 = 80 max.
    assert stats["actual_total"] < 1000
    assert len(stream) == stats["actual_total"]
