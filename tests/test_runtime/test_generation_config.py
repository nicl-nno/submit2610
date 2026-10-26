'Tests for the greedy generation-config hygiene (offline).\n\n\n``unset_greedy_sampling_flags`` unsets the sampling-only flags a Qwen-line\ncheckpoint carries in ``generation_config.json`` so transformers\'\n``GenerationConfig.validate()`` stops logging "The following generation\nflags are not valid and may be ignored" on EVERY greedy ``.generate()``\ncall (drift probe / judge anchors / per-epoch validations). The\neos/pad/bos ids and length caps must survive — generation stopping depends\non them.\n``unset_greedy_sampling_flags``'

from __future__ import annotations

import logging
from types import SimpleNamespace

from llm_training.runtime.generation_config import (
    GREEDY_UNUSED_SAMPLING_FLAGS,
    unset_greedy_sampling_flags,
)


def _model_with_flags(**flags):
    gen_cfg = SimpleNamespace(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        typical_p=None,
        epsilon_cutoff=None,
        eta_cutoff=None,
        eos_token_id=151645,
        pad_token_id=151643,
        max_new_tokens=512,
    )
    for name, value in flags.items():
        setattr(gen_cfg, name, value)
    return SimpleNamespace(generation_config=gen_cfg)


class TestUnsetGreedySamplingFlags:
    def test_qwen_style_defaults_unset(self):
        model = _model_with_flags()
        unset = unset_greedy_sampling_flags(model)
        assert set(unset) == {"temperature", "top_p", "top_k"}
        gc = model.generation_config
        assert gc.temperature is None
        assert gc.top_p is None
        assert gc.top_k is None

    def test_stop_ids_and_length_caps_survive(self):
        # The whole-config reset would drop eos/pad — generation stopping
        # depends on them; the helper must touch ONLY the sampling flags.
        
        
        model = _model_with_flags()
        unset_greedy_sampling_flags(model)
        gc = model.generation_config
        assert gc.eos_token_id == 151645
        assert gc.pad_token_id == 151643
        assert gc.max_new_tokens == 512

    def test_clean_config_is_a_noop(self, caplog):
        model = _model_with_flags(
            temperature=None, top_p=None, top_k=None,
            typical_p=None, epsilon_cutoff=None, eta_cutoff=None,
        )
        with caplog.at_level(logging.INFO):
            unset = unset_greedy_sampling_flags(model)
        assert unset == []
        assert not any("hygiene" in r.message for r in caplog.records)

    def test_extra_sampling_flags_covered(self):
        # Future checkpoints may carry more greedy-unused sampling flags
        # (typical_p / cutoffs) — all of them are in the constant and unset.
        
        
        model = _model_with_flags(typical_p=0.9, epsilon_cutoff=0.1)
        unset = unset_greedy_sampling_flags(model)
        assert {"temperature", "top_p", "top_k", "typical_p",
                "epsilon_cutoff"} <= set(unset)
        assert model.generation_config.typical_p is None
        assert model.generation_config.eta_cutoff is None

    def test_peft_style_forwarding(self):
        # PeftModel forwards unknown attributes to the base — the helper
        # resolves model.generation_config the same way and mutates the ONE
        # shared config object.
        
        
        base = _model_with_flags()

        class _PeftLike:
            def __getattr__(self, name):
                return getattr(base, name)

        unset = unset_greedy_sampling_flags(_PeftLike())
        assert set(unset) == {"temperature", "top_p", "top_k"}
        assert base.generation_config.temperature is None

    def test_no_generation_config_is_a_noop(self):
        unset = unset_greedy_sampling_flags(SimpleNamespace())
        assert unset == []

    def test_flags_constant_contents(self):
        # Documented invariant: ONLY sampling flags, never stop ids / caps.
        
        assert set(GREEDY_UNUSED_SAMPLING_FLAGS) == {
            "temperature", "top_p", "top_k",
            "typical_p", "epsilon_cutoff", "eta_cutoff",
        }
