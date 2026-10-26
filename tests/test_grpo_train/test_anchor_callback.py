'Tests for the judge-anchor ctrl-point callback (checkpoint selection).\n\n\nOffline with fakes: a fake trainer/accelerator, a fake scorer with scripted\nAC values, and monkeypatched ``run_anchor_scoring``. Covers: step gating,\nsave-iff-improve, the early-stop patience, and the main-process guard.'

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

_SCRIPT = Path(__file__).resolve().parents[2] / "grpo_train.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_train_anchor", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tr = _load_module()


class _FakeModel:
    'Stands in for the policy model: records save_pretrained kwargs.'

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def save_pretrained(self, path, **kwargs):
        self.calls.append((path, kwargs))


class _FakeTrainer:
    def __init__(self):
        self.accelerator = SimpleNamespace(
            is_main_process=True, unwrap_model=lambda m: m
        )
        self.model = _FakeModel()
        self.saved: list[str] = []

    def save_model(self, path):
        # Mirrors Trainer.save_model: the model save goes through
        # self.model.save_pretrained (what the shard-size wrapper patches).
        
        # self.model.save_pretrained
        self.saved.append(path)
        self.model.save_pretrained(path)


class _FakeTokenizer:
    def save_pretrained(self, path):
        pass


def _make_callback(
    trainer,
    ac_script,
    tmp_path,
    *,
    every=100,
    patience=3,
    probe_script=None,
    probe_baseline=None,
    probe_gate_tol=0.10,
    probe_boom=False,
    initial_best_val=None,
):
    'Build the callback with a scripted run_anchor_scoring fake.'
    calls = {"n": 0, "probe": 0}

    def _fake_run_anchor_scoring(model, tokenizer, examples, judge_scorer,
                                 reward_service, *, max_new_tokens, **kw):
        ac = ac_script[min(calls["n"], len(ac_script) - 1)]
        calls["n"] += 1
        # The full breakdown shape run_anchor_scoring returns: the selection
        # key is `val`; ac/rubric/semsim/fluency are diagnostics.
        # val.
        return {
            "val": ac, "ac": ac, "rubric": 0.7, "semsim": 0.5,
            "fluency": 0.9, "n": float(len(examples)),
        }

    def _fake_run_probe(model, tokenizer, items, *, max_new_tokens, **kw):
        calls["probe"] += 1
        if probe_boom:
            raise RuntimeError("probe boom")
        score = probe_script[min(calls["probe"] - 1, len(probe_script) - 1)]
        return {
            "score": score, "n": len(items),
            "per_check": {"min_words": {"pass_rate": score, "n": len(items)}},
            "failed_ids": [],
        }

    callback, state = tr.make_judge_anchor_callback(
        trainer=trainer,
        tokenizer=_FakeTokenizer(),
        anchor_examples=[SimpleNamespace()] * 4,
        get_judge_runtime=lambda: (object(), object()),
        output_dir=tmp_path,
        anchor_every=every,
        patience=patience,
        max_new_tokens=128,
        run_anchor_scoring_fn=_fake_run_anchor_scoring,
        probe_items=[SimpleNamespace()] * 8 if probe_script is not None else (),
        probe_max_new_tokens=64,
        probe_warn_drop=0.05,
        probe_gate_tol=probe_gate_tol,
        probe_baseline=probe_baseline,
        initial_best_val=initial_best_val,
        run_probe_fn=_fake_run_probe if probe_script is not None else None,
    )
    return callback, state, calls


def _step(cb, step):
    control = SimpleNamespace(should_training_stop=False)
    cb.on_step_end(None, SimpleNamespace(global_step=step), control)
    return control


class TestAnchorCallback:
    def test_non_anchor_steps_ignored(self, tmp_path):
        trainer = _FakeTrainer()
        cb, state, calls = _make_callback(trainer, [0.9], tmp_path, every=100)
        for step in (1, 50, 99, 101, 150):
            c = _step(cb, step)
            assert not c.should_training_stop
        assert calls["n"] == 0
        assert trainer.saved == []

    def test_anchor_steps_trigger_scoring(self, tmp_path):
        trainer = _FakeTrainer()
        cb, state, calls = _make_callback(trainer, [0.8], tmp_path, every=100)
        c = _step(cb, 100)
        assert calls["n"] == 1
        assert not c.should_training_stop
        # First anchor always improves (-1.0 → 0.8) → checkpoint saved.
        
        assert state["best_val"] == 0.8
        assert len(trainer.saved) == 1
        assert "best-100" in trainer.saved[0]

    def test_save_iff_improves(self, tmp_path):
        trainer = _FakeTrainer()
        # AC: 0.8 (save), 0.7 (no), 0.75 (no), 0.85 (save), 0.9 (save).
        cb, state, calls = _make_callback(
            trainer, [0.8, 0.7, 0.75, 0.85, 0.9], tmp_path, every=100, patience=10,
        )
        for step in (100, 200, 300, 400, 500):
            _step(cb, step)
        assert state["best_val"] == 0.9
        assert len(trainer.saved) == 3  # steps 100, 400, 500

    def test_early_stop_after_patience(self, tmp_path):
        trainer = _FakeTrainer()
        # AC stalls at 0.8 forever; patience=3 → stop at the 4th anchor.
        cb, state, calls = _make_callback(
            trainer, [0.8, 0.8, 0.8, 0.8], tmp_path, every=100, patience=3,
        )
        c = None
        for step in (100, 200, 300):
            c = _step(cb, step)
            assert not c.should_training_stop  # 1, 2 non-improvements — tolerable
        c = _step(cb, 400)
        assert c.should_training_stop  # 3rd non-improvement → stop
        assert state["no_improve"] == 3
        assert len(trainer.saved) == 1  # only the first anchor's checkpoint

    def test_recovery_resets_patience(self, tmp_path):
        trainer = _FakeTrainer()
        # 0.8 (save), 0.7, 0.75 (2 non-improvements), 0.9 (save, reset), then
        # 3 more stalls → stop only at the 6th anchor.
        cb, state, calls = _make_callback(
            trainer, [0.8, 0.7, 0.75, 0.9, 0.85, 0.87, 0.86], tmp_path,
            every=100, patience=3,
        )
        stopped_at = None
        for step in (100, 200, 300, 400, 500, 600, 700):
            c = _step(cb, step)
            if c.should_training_stop:
                stopped_at = step
                break
        assert stopped_at == 700

    def test_replica_processes_skipped(self, tmp_path):
        # Non-main processes must not generate/score/save (multi-GPU safety).
        
        trainer = _FakeTrainer()
        trainer.accelerator.is_main_process = False
        cb, state, calls = _make_callback(trainer, [0.9], tmp_path, every=100)
        c = _step(cb, 100)
        assert calls["n"] == 0
        assert trainer.saved == []
        assert state["best_val"] == -1.0

    def test_judge_unavailable_skips_gracefully(self, tmp_path):
        # No JUDGE_* env → the anchor logs a warning and training continues.
        
        trainer = _FakeTrainer()

        def _no_judge():
            raise RuntimeError("JUDGE_BASE_URL and JUDGE_MODEL required")

        import unittest.mock as mock

        cb, state = tr.make_judge_anchor_callback(
            trainer=trainer,
            tokenizer=_FakeTokenizer(),
            anchor_examples=[SimpleNamespace()],
            get_judge_runtime=_no_judge,
            output_dir=tmp_path,
            anchor_every=100,
            patience=3,
            max_new_tokens=128,
        )
        c = _step(cb, 100)
        assert not c.should_training_stop
        assert trainer.saved == []

    def test_scoring_failure_skips_anchor_not_training(self, tmp_path):
        # A crash inside run_anchor_scoring (e.g. a RewardService timeout or a
        # generation OOM) must skip THAT anchor — not abort a multi-hour run.
        
        trainer = _FakeTrainer()

        def _boom(model, tokenizer, examples, judge_scorer, reward_service, *,
                  max_new_tokens, **kw):
            raise TimeoutError("score_batch timed out")

        cb, state = tr.make_judge_anchor_callback(
            trainer=trainer,
            tokenizer=_FakeTokenizer(),
            anchor_examples=[SimpleNamespace()] * 4,
            get_judge_runtime=lambda: (object(), object()),
            output_dir=tmp_path,
            anchor_every=100,
            patience=3,
            max_new_tokens=128,
            run_anchor_scoring_fn=_boom,
        )
        c = _step(cb, 100)
        assert not c.should_training_stop
        assert trainer.saved == []
        assert state["best_val"] == -1.0  # failure is not counted as AC=0

    def test_empty_anchor_pool_skips(self, tmp_path):
        trainer = _FakeTrainer()
        cb, state = tr.make_judge_anchor_callback(
            trainer=trainer,
            tokenizer=_FakeTokenizer(),
            anchor_examples=[],
            get_judge_runtime=lambda: (object(), object()),
            output_dir=tmp_path,
            anchor_every=100,
            patience=3,
            max_new_tokens=128,
        )
        c = _step(cb, 100)
        assert not c.should_training_stop
        assert trainer.saved == []

    def test_best_save_pins_shard_size(self, tmp_path):
        # The best-<step>/ save goes through sharded_checkpoint_saves: the
        # model's save_pretrained receives max_shard_size explicitly, and
        # the temporary wrapper is gone after the save (no lingering
        # instance attribute).
        
        # save_pretrained
        
        trainer = _FakeTrainer()

        def _fake_run_anchor_scoring(model, tokenizer, examples, judge_scorer,
                                     reward_service, *, max_new_tokens, **kw):
            return {
                "val": 0.8, "ac": 0.8, "rubric": 0.7, "semsim": 0.5,
                "fluency": 0.9, "n": float(len(examples)),
            }

        cb, state = tr.make_judge_anchor_callback(
            trainer=trainer,
            tokenizer=_FakeTokenizer(),
            anchor_examples=[SimpleNamespace()],
            get_judge_runtime=lambda: (object(), object()),
            output_dir=tmp_path,
            anchor_every=100,
            patience=3,
            max_new_tokens=128,
            max_shard_size="5GB",
            run_anchor_scoring_fn=_fake_run_anchor_scoring,
        )
        _step(cb, 100)
        assert len(trainer.model.calls) == 1
        path, kwargs = trainer.model.calls[0]
        assert kwargs["max_shard_size"] == "5GB"
        assert "save_pretrained" not in vars(trainer.model)


class TestDriftProbeGate:
    'The probe gates the best-save; the frozen val number is untouched.'

    def test_val_improves_probe_ok_saves(self, tmp_path):
        # Baseline 0.8 (step 0, passed in); anchors: val 0.9 probe 0.85
        # (within base-0.10) → save.
        
        trainer = _FakeTrainer()
        cb, state, calls = _make_callback(
            trainer, [0.9], tmp_path,
            probe_script=[0.85], probe_baseline=0.8,
        )
        _step(cb, 100)
        assert state["best_val"] == 0.9
        assert len(trainer.saved) == 1
        assert state["probe_baseline"] == 0.8  # step-0 baseline wins
        assert calls["probe"] == 1

    def test_gate_blocks_save_and_counts_no_improve(self, tmp_path):
        # val improves (0.9 > -1) but the probe 0.60 < 0.8 - 0.10 → NO save,
        # best_val stays -1.0 (the next save must beat the SAME bar), and
        # no_improve grows — the r2-r4 lesson: a degraded checkpoint must
        # not ship through a val-only selector.
        
        
        trainer = _FakeTrainer()
        cb, state, calls = _make_callback(
            trainer, [0.9], tmp_path,
            probe_script=[0.60], probe_baseline=0.8,
        )
        c = _step(cb, 100)
        assert not c.should_training_stop
        assert trainer.saved == []
        assert state["best_val"] == -1.0
        assert state["no_improve"] == 1

    def test_gate_recovery_saves_again(self, tmp_path):
        # Blocked anchor (probe 0.60), then the probe recovers to 0.75 —
        # the same improving val saves on the next anchor.
        
        
        trainer = _FakeTrainer()
        cb, state, calls = _make_callback(
            trainer, [0.9, 0.92], tmp_path,
            probe_script=[0.60, 0.75], probe_baseline=0.8, patience=5,
        )
        _step(cb, 100)
        _step(cb, 200)
        assert state["best_val"] == 0.92
        assert len(trainer.saved) == 1
        assert "best-200" in trainer.saved[0]

    def test_persistent_gate_blocks_early_stop(self, tmp_path):
        # val keeps improving but the probe stays below the gate → patience
        # exhausts → early stop (stop instead of shipping degradation).
        
        trainer = _FakeTrainer()
        cb, state, calls = _make_callback(
            trainer, [0.9, 0.95, 0.97, 0.99], tmp_path,
            probe_script=[0.5, 0.5, 0.5, 0.5], probe_baseline=0.8,
            patience=3,
        )
        c = None
        for step in (100, 200):
            c = _step(cb, step)
            assert not c.should_training_stop
        c = _step(cb, 300)
        assert c.should_training_stop
        assert state["no_improve"] == 3
        assert trainer.saved == []

    def test_failed_probe_run_never_vetoes(self, tmp_path):
        # A measurement failure is not degradation: the probe raises, the
        # gate passes, the improving val still saves.
        
        
        trainer = _FakeTrainer()
        cb, state, calls = _make_callback(
            trainer, [0.9], tmp_path,
            probe_script=[1.0], probe_baseline=0.8, probe_boom=True,
        )
        _step(cb, 100)
        assert state["best_val"] == 0.9
        assert len(trainer.saved) == 1
        assert calls["probe"] == 1

    def test_baseline_falls_back_to_first_anchor(self, tmp_path):
        # No step-0 baseline → the first successful anchor becomes the
        # reference; its own save is compared against itself (passes).
        
        trainer = _FakeTrainer()
        cb, state, calls = _make_callback(
            trainer, [0.9], tmp_path, probe_script=[0.7], probe_baseline=None,
        )
        _step(cb, 100)
        assert state["probe_baseline"] == 0.7
        assert len(trainer.saved) == 1  # 0.7 >= 0.7 - 0.10

    def test_gate_disabled_by_zero_tolerance(self, tmp_path):
        # probe_gate_tol=0 → the gate is off; a deep probe drop must not
        # block the improving-val save (log+WARN only).
        # probe_gate_tol=0 →
        
        trainer = _FakeTrainer()
        cb, state, calls = _make_callback(
            trainer, [0.9], tmp_path,
            probe_script=[0.1], probe_baseline=0.8, probe_gate_tol=0.0,
        )
        _step(cb, 100)
        assert state["best_val"] == 0.9
        assert len(trainer.saved) == 1

    def test_probe_without_items_is_noop(self, tmp_path):
        # probe_items empty (the file was absent / --no-drift-probe): the
        # callback behaves exactly like the pre-probe version.
        
        trainer = _FakeTrainer()
        cb, state, calls = _make_callback(trainer, [0.9], tmp_path)
        _step(cb, 100)
        assert state["best_val"] == 0.9
        assert len(trainer.saved) == 1
        assert calls["probe"] == 0


class TestBaselineSeeding:
    'The step-0 baseline anchor seeds the best_val selection bar.'

    def test_seeded_baseline_blocks_worse_first_anchor(self, tmp_path):
        # Baseline 0.8 (step 0, seeded); the first in-training anchor is
        # WORSE (0.5) → no save, the bar stays 0.8, no_improve grows, and
        # best_step stays None. Before r5 the -1.0 start saved this
        # checkpoint unconditionally.
        
        
        trainer = _FakeTrainer()
        cb, state, calls = _make_callback(
            trainer, [0.5], tmp_path, initial_best_val=0.8,
        )
        c = _step(cb, 100)
        assert not c.should_training_stop
        assert trainer.saved == []
        assert state["best_val"] == 0.8
        assert state["best_step"] is None
        assert state["no_improve"] == 1

    def test_seeded_baseline_allows_better_first_anchor(self, tmp_path):
        # Baseline 0.8; the first anchor beats it (0.9) → save best-100,
        # the bar moves, patience resets, best_step recorded.
        
        
        trainer = _FakeTrainer()
        cb, state, calls = _make_callback(
            trainer, [0.9], tmp_path, initial_best_val=0.8,
        )
        c = _step(cb, 100)
        assert not c.should_training_stop
        assert len(trainer.saved) == 1
        assert "best-100" in trainer.saved[0]
        assert state["best_val"] == 0.9
        assert state["best_step"] == 100
        assert state["no_improve"] == 0

    def test_seeded_baseline_persistent_worse_early_stops(self, tmp_path):
        # The model never beats the pre-training checkpoint → patience
        # exhausts → early stop with ZERO in-run saves (do not ship a
        # checkpoint worse than the starting one — the step-0 lesson).
        
        
        trainer = _FakeTrainer()
        cb, state, calls = _make_callback(
            trainer, [0.5, 0.6, 0.7], tmp_path,
            initial_best_val=0.8, patience=3,
        )
        c = None
        for step in (100, 200):
            c = _step(cb, step)
            assert not c.should_training_stop
        c = _step(cb, 300)
        assert c.should_training_stop
        assert state["no_improve"] == 3
        assert trainer.saved == []
        assert state["best_val"] == 0.8
        assert state["best_step"] is None

    def test_anchor_every_zero_is_noop(self, tmp_path):
        # --judge-anchor-every 0 disables the anchors: no crash (the
        # modulo guard), no scoring, no saves.
        # --judge-anchor-every 0
        
        trainer = _FakeTrainer()
        cb, state, calls = _make_callback(
            trainer, [0.9], tmp_path, every=0,
        )
        c = _step(cb, 100)
        assert not c.should_training_stop
        assert calls["n"] == 0
        assert trainer.saved == []
        assert state["best_val"] == -1.0
