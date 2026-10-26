'Tests for llm_training.oplora (offline, CPU).\n\n\nCovers the projected-gradient OPLoRA implementation: the low-rank projection\nmath (invariants + equivalence to the dense double-sided projection + the\ntop-k singular-triple preservation of Proposition 2), and the peft integration\n(layer discovery, idempotent projection, _version drift detection + repair,\nthe duck-typed trainer callback, the forward guard, merge_and_unload).'

from __future__ import annotations

import pytest

from llm_training.oplora import (
    OploraProjection,
    project_left,
    project_right,
)

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------- math
class TestProjectionMath:
    def test_left_projection_removes_topk_component(self):
        torch.manual_seed(0)
        w = torch.randn(64, 48)
        u, _s, _vh = torch.linalg.svd(w, full_matrices=False)
        u_k = u[:, :16]
        b = torch.randn(64, 8)
        projected = project_left(b, u_k)
        assert torch.linalg.norm(u_k.T @ projected).item() == pytest.approx(
            0.0, abs=1e-4
        )

    def test_right_projection_removes_topk_component(self):
        torch.manual_seed(0)
        w = torch.randn(64, 48)
        _u, _s, vh = torch.linalg.svd(w, full_matrices=False)
        v_k = vh[:16, :].T
        a = torch.randn(8, 48)
        projected = project_right(a, v_k)
        assert torch.linalg.norm(projected @ v_k).item() == pytest.approx(
            0.0, abs=1e-4
        )

    def test_equivalence_to_dense_double_sided_projection(self):
        # (P_L B)(A P_R) == P_L (B A) P_R — associativity is what lets the
        # plain peft forward compute the OPLoRA update.
        # (P_L B)(A P_R) == P_L (B A) P_R
        
        torch.manual_seed(0)
        w = torch.randn(64, 48)
        u, _s, vh = torch.linalg.svd(w, full_matrices=False)
        u_k, v_k = u[:, :16], vh[:16, :].T
        b = torch.randn(64, 8)
        a = torch.randn(8, 48)
        lhs = project_left(b, u_k) @ project_right(a, v_k)
        pl = torch.eye(64) - u_k @ u_k.T
        pr = torch.eye(48) - v_k @ v_k.T
        rhs = pl @ (b @ a) @ pr
        assert torch.allclose(lhs, rhs, atol=1e-4)

    def test_proposition2_topk_triples_preserved(self):
        # U_kᵀ (W + ΔW) V_k == diag(S[:k]) for ΔW = P_L B A P_R.
        
        torch.manual_seed(0)
        w = torch.randn(64, 48)
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        u_k, v_k = u[:, :16], vh[:16, :].T
        b = torch.randn(64, 8)
        a = torch.randn(8, 48)
        delta = project_left(b, u_k) @ project_right(a, v_k)
        wp = w + delta
        assert torch.allclose(
            u_k.T @ wp @ v_k, torch.diag(s[:16]), atol=1e-3
        )


# ------------------------------------------------------------ peft fixtures
def _tiny_qwen():
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "qwen2",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=512,
    )
    return AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)


@pytest.fixture(scope="module")
def peft_model():
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    model = _tiny_qwen()
    lc = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj", "up_proj", "down_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lc)


@pytest.fixture(scope="module")
def projection(peft_model):
    torch.manual_seed(0)
    proj = OploraProjection.from_peft_model(
        peft_model, k=16, niter=4, progress=False
    )
    proj.project_weights()
    return proj


class TestFromPeftModel:
    def test_all_target_layers_found(self, projection):
        # 2 layers × 5 target modules.
        
        assert len(projection.layers) == 10
        assert projection.k == 16

    def test_bases_are_orthonormal(self, projection):
        for layer in projection.layers.values():
            eye_u = layer.u_k.T @ layer.u_k
            eye_v = layer.v_k.T @ layer.v_k
            assert torch.allclose(eye_u, torch.eye(layer.k), atol=1e-4)
            assert torch.allclose(eye_v, torch.eye(layer.k), atol=1e-4)

    def test_k_clamped_to_weight_dimensions(self, peft_model, caplog):
        torch.manual_seed(0)
        with caplog.at_level("WARNING"):
            proj = OploraProjection.from_peft_model(
                peft_model, k=100, niter=2, progress=False
            )
        # Every k is clamped to min(out, in) of its weight (k_proj/v_proj are
        # GQA-narrower than the hidden size); a clamping warning was logged.
        
        
        assert all(layer.k <= 64 for layer in proj.layers.values())
        assert any(layer.k < 64 for layer in proj.layers.values())
        assert any("clamped" in r.message for r in caplog.records)

    def test_no_lora_layers_is_an_error(self):
        with pytest.raises(RuntimeError, match="no LoRA layers"):
            OploraProjection.from_peft_model(
                _tiny_qwen(), k=8, progress=False
            )

    def test_dora_rejected(self, peft_model):
        for name, module in peft_model.named_modules():
            if hasattr(module, "lora_A"):
                module.use_dora = {"default": True}
                break
        try:
            with pytest.raises(NotImplementedError):
                OploraProjection.from_peft_model(
                    peft_model, k=8, progress=False
                )
        finally:
            for _n, module in peft_model.named_modules():
                if hasattr(module, "lora_A"):
                    module.use_dora = {"default": False}


class TestProjectionInvariants:
    def test_initial_projection_satisfies_invariants(self, projection):
        # B = 0 at peft init → the projection is trivially exact; perturb B
        # first so the invariants are non-vacuous.
        
        
        with torch.no_grad():
            for layer in projection.layers.values():
                layer.lora_b.normal_(0.0, 0.1)
        projection.project_weights()
        for layer in projection.layers.values():
            assert torch.linalg.norm(
                layer.u_k.T @ layer.lora_b.float()
            ).item() == pytest.approx(0.0, abs=1e-4)
            assert torch.linalg.norm(
                layer.lora_a.float() @ layer.v_k
            ).item() == pytest.approx(0.0, abs=1e-4)

    def test_interference_rho_near_zero(self, projection):
        report = projection.interference()
        assert report["max_rho"] == pytest.approx(0.0, abs=1e-3)

    def test_rho_none_when_delta_is_zero(self, projection):
        # A freshly initialised peft adapter (B = 0) has ΔW = 0 → rho None.
        
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(0)
        model = get_peft_model(
            _tiny_qwen(),
            LoraConfig(r=4, target_modules=["q_proj"], task_type="CAUSAL_LM"),
        )
        proj = OploraProjection.from_peft_model(
            model, k=8, progress=False
        )
        proj.project_weights()
        assert proj.interference()["mean_rho"] is None


class TestDriftAndRepair:
    def test_version_drift_detected_and_repaired(self, projection):
        layer = next(iter(projection.layers.values()))
        assert layer.drifted() is False
        # An optimizer-style in-place update kicks the layer out of the
        # subspace; repair_if_drifted must restore the invariants.
        # In-place
        
        with torch.no_grad():
            layer.lora_b.add_(torch.randn_like(layer.lora_b) * 0.1)
        assert layer.drifted() is True
        assert projection.repair_if_drifted() == 1
        assert layer.drifted() is False
        assert torch.linalg.norm(
            layer.u_k.T @ layer.lora_b.float()
        ).item() == pytest.approx(0.0, abs=1e-4)

    def test_adam_style_update_stays_confined_after_repair(self, projection):
        # Adam's elementwise rescaling breaks the subspace — the repair (not
        # a gradient projection) is what restores it; emulate a few steps.
        
        
        layer = next(iter(projection.layers.values()))
        for _ in range(3):
            with torch.no_grad():
                layer.lora_a.mul_(torch.rand_like(layer.lora_a) + 0.5)
                layer.lora_b.add_(torch.randn_like(layer.lora_b) * 0.05)
            projection.repair_if_drifted()
        assert torch.linalg.norm(
            layer.lora_a.float() @ layer.v_k
        ).item() == pytest.approx(0.0, abs=1e-3)


class TestHooksAndCallback:
    def test_trainer_callback_duck_typing(self, projection):
        cb = projection.make_trainer_callback()
        cb.on_step_end(None, None, None)
        cb.on_evaluate(None, None, None)
        cb.on_prediction_step(None, None, None)

    def test_trainer_callback_dispatches_every_event(self, projection):
        # Regression: the first cluster --oplora run crashed in
        # trainer.train() — transformers' CallbackHandler.call_event does
        # getattr(callback, event) for EVERY event and the former plain
        # duck-typed class lacked on_train_begin (AttributeError at step 0).
        
        # — call_event
        
        cb = projection.make_trainer_callback()
        events = [
            "on_init_end",
            "on_train_begin",
            "on_epoch_begin",
            "on_substep_end",
            "on_step_end",
            "on_epoch_end",
            "on_save",
            "on_evaluate",
            "on_prediction_step",
            "on_log",
            "on_train_end",
        ]
        for event in events:
            handler = getattr(cb, event, None)
            assert callable(handler), f"missing trainer event: {event}"
            handler(None, None, None)
        try:
            from transformers.trainer_callback import TrainerCallback
        except ImportError:
            return  # offline fallback: duck-typed dispatch is what we assert
        assert isinstance(cb, TrainerCallback)

    def test_forward_guard_repairs_and_removes(self, projection, peft_model):
        layer = next(iter(projection.layers.values()))
        with torch.no_grad():
            layer.lora_b.add_(torch.randn_like(layer.lora_b) * 0.1)
        handle = projection.register_forward_guard(peft_model)
        try:
            with torch.no_grad():
                peft_model(input_ids=torch.randint(0, 256, (1, 8)))
            # The forward itself triggered the repair.
            
            assert layer.drifted() is False
        finally:
            handle.remove()

    def test_generate_works_under_guard(self, projection, peft_model):
        handle = projection.register_forward_guard(peft_model)
        try:
            with torch.no_grad():
                out = peft_model.generate(
                    input_ids=torch.randint(0, 256, (1, 8)),
                    max_new_tokens=4,
                    do_sample=False,
                )
            assert out.shape[1] == 12
        finally:
            handle.remove()


class TestDeviceAlignment:
    @pytest.mark.gpu
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="needs a CUDA device"
    )
    def test_repair_after_model_move_to_gpu(self):
        # Regression (the 2026-09-25 run, step 1): the SVD bases are
        # computed while the model is still on its LOAD-time device; the
        # Trainer then moves the model to the GPU — u_k/v_k are plain
        # dataclass fields and do NOT ride model.to(). The first repair
        # after an optimizer step mixed CPU bases with CUDA parameters →
        # RuntimeError. The lazy alignment must move the bases and keep
        # the subspace invariant. A FRESH model: the module-scoped fixture
        # must not be poisoned with mixed devices.
        
        
        
        
        
        
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(0)
        model = get_peft_model(
            _tiny_qwen(),
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        proj = OploraProjection.from_peft_model(
            model, k=8, niter=2, progress=False
        )
        proj.project_weights()
        layer = next(iter(proj.layers.values()))
        assert layer.u_k.device.type == "cpu"
        # Emulate Trainer.__init__ moving the adapter params to the GPU —
        # the bases stay behind on the CPU exactly as on the cluster (they
        # are not module buffers).
        
        
        with torch.no_grad():
            layer.lora_a.data = layer.lora_a.data.to("cuda")
            layer.lora_b.data = layer.lora_b.data.to("cuda")
            layer.lora_b.add_(torch.randn_like(layer.lora_b) * 0.05)
        proj.repair_if_drifted()  # used to die with a device RuntimeError
        assert layer.u_k.device.type == "cuda"
        assert layer.v_k.device.type == "cuda"
        assert torch.linalg.norm(
            layer.u_k.T @ layer.lora_b.float()
        ).item() == pytest.approx(0.0, abs=1e-4)
        report = layer.interference()  # the aligned diagnostics path
        assert report["rho"] is not None

    def test_alignment_is_a_noop_when_devices_match(self, projection):
        # Matching devices must not reallocate the bases — the cached-copy
        # invariant (the copy is paid once per device move, never again).
        
        
        layer = next(iter(projection.layers.values()))
        u_before, v_before = layer.u_k, layer.v_k
        layer.project()
        assert layer.u_k is u_before
        assert layer.v_k is v_before


class TestMerge:
    def test_merge_and_unload_produces_base_model(self):
        # A FRESH model (merge_and_unload folds the adapter into the shared
        # base in place — it must not corrupt the module-scoped fixture).
        
        
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen2ForCausalLM

        torch.manual_seed(0)
        model = get_peft_model(
            _tiny_qwen(),
            LoraConfig(
                r=4,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        merged = model.merge_and_unload()
        assert isinstance(merged, Qwen2ForCausalLM)
