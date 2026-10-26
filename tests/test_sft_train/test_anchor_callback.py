"Tests for sft_train.py's judge-anchor callback save_best DI seam (offline).\n\n\nThe selection logic of ``make_judge_anchor_callback`` is the grpo_train.py\nverbatim copy; the ONLY delta is the ``save_best`` seam — the LoRA path saves\nthe peft adapter through a callable instead of the shard-size-wrapped full\nmodel save. These tests pin the seam behaviour; the selection itself\n(save-iff-improve, patience, the probe gate) is covered by\n``tests/test_grpo_train/test_anchor_callback.py`` against the same code.\n``save_best``\ngrpo_train."

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

_SCRIPT = Path(__file__).resolve().parents[2] / "sft_train.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_sft_train_anchor", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


st = _load_module()


class _FakeTrainer:
    def __init__(self):
        self.accelerator = SimpleNamespace(
            is_main_process=True, unwrap_model=lambda m: m
        )
        self.model = SimpleNamespace()
        self.saved: list[str] = []

    def save_model(self, path):
        self.saved.append(path)


class _FakeTokenizer:
    def __init__(self):
        self.saved_to: list[str] = []

    def save_pretrained(self, path):
        self.saved_to.append(path)


def _make(trainer, tokenizer, tmp_path, *, save_best=..., val=0.9):
    def _fake_run_anchor_scoring(model, tok, examples, scorer, service,
                                 *, max_new_tokens, **kw):
        return {
            "val": val, "ac": val, "rubric": 0.7, "semsim": 0.5,
            "fluency": 0.9, "n": float(len(examples)),
        }

    kwargs = {}
    if save_best is not ...:
        kwargs["save_best"] = save_best
    callback, state = st.make_judge_anchor_callback(
        trainer=trainer,
        tokenizer=tokenizer,
        anchor_examples=[SimpleNamespace()] * 4,
        get_judge_runtime=lambda: (object(), object()),
        output_dir=tmp_path,
        anchor_every=100,
        patience=3,
        max_new_tokens=128,
        run_anchor_scoring_fn=_fake_run_anchor_scoring,
        probe_items=(),
        run_probe_fn=lambda *a, **k: {
            "score": 1.0, "n": 1,
            "per_check": {}, "failed_ids": [],
        },
        **kwargs,
    )
    return callback, state


def _step(cb, step):
    control = SimpleNamespace(should_training_stop=False)
    cb.on_step_end(None, SimpleNamespace(global_step=step), control)
    return control


class TestSaveBestSeam:
    def test_default_path_saves_via_trainer_and_tokenizer(self, tmp_path):
        # save_best=None → the grpo_train.py verbatim block: trainer.save_model
        # + tokenizer.save_pretrained (the shard-size CM is a no-op on the
        # fake model without save_pretrained).
        # save_best=None →
        trainer = _FakeTrainer()
        tokenizer = _FakeTokenizer()
        cb, state = _make(trainer, tokenizer, tmp_path)
        _step(cb, 100)
        assert state["best_val"] == 0.9
        assert len(trainer.saved) == 1
        assert "best-100" in trainer.saved[0]
        assert len(tokenizer.saved_to) == 1

    def test_custom_save_best_replaces_the_save_mechanics(self, tmp_path):
        # The LoRA path: the callable owns the save (the adapter), the
        # callback itself must not call trainer.save_model.
        
        
        trainer = _FakeTrainer()
        tokenizer = _FakeTokenizer()
        seen: list[tuple[object, str]] = []

        def _save_peft(t, best_dir):
            seen.append((t, str(best_dir)))

        cb, state = _make(trainer, tokenizer, tmp_path, save_best=_save_peft)
        _step(cb, 100)
        assert state["best_val"] == 0.9  # selection intact
        assert len(seen) == 1
        assert seen[0][0] is trainer
        assert "best-100" in seen[0][1]
        assert trainer.saved == []       # the default path did not run
        assert tokenizer.saved_to == []  # the adapter saver owns the tokenizer

    def test_selection_logic_unchanged_with_custom_saver(self, tmp_path):
        # No improvement → no save call, patience advances (the seam must not
        # alter the selection semantics).
        
        
        trainer = _FakeTrainer()
        calls: list[str] = []

        def _fake_run_anchor_scoring(model, tok, examples, scorer, service,
                                     *, max_new_tokens, **kw):
            return {
                "val": 0.5, "ac": 0.5, "rubric": 0.7, "semsim": 0.5,
                "fluency": 0.9, "n": float(len(examples)),
            }

        callback, state = st.make_judge_anchor_callback(
            trainer=trainer,
            tokenizer=_FakeTokenizer(),
            anchor_examples=[SimpleNamespace()] * 4,
            get_judge_runtime=lambda: (object(), object()),
            output_dir=tmp_path,
            anchor_every=100,
            patience=2,
            max_new_tokens=128,
            run_anchor_scoring_fn=_fake_run_anchor_scoring,
            probe_items=(),
            initial_best_val=0.8,
            save_best=lambda t, d: calls.append(str(d)),
        )
        c1 = _step(callback, 100)
        assert not c1.should_training_stop
        assert calls == []
        assert state["no_improve"] == 1
        c2 = _step(callback, 200)
        assert c2.should_training_stop  # patience 2 reached
        assert calls == []
