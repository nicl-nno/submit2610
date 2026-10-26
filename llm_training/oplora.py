"OPLoRA: double-sided orthogonal projection of LoRA updates (sft_train.py opt-in).\n\n\nImplements the method of `OPLoRA: Orthogonal Projection LoRA Prevents\nCatastrophic Forgetting during Parameter-Efficient Fine-Tuning\n<https://arxiv.org/abs/2510.13003>`_ (Xiong & Xie, 2025) on top of a standard\npeft LoRA model. The effective update is constrained to\n\n    ΔW = P_L · B · A · P_R,   P_L = I − U_k U_kᵀ,   P_R = I − V_k V_kᵀ,\n\nwhere ``U_k`` / ``V_k`` collect the top-k left / right singular vectors of the\nFROZEN weight ``W``. Proposition 2 of the paper: the top-k singular triples\n``(u_i, σ_i, v_i)`` of ``W`` are then preserved EXACTLY (``ΔW·V_k = 0`` and\n``U_kᵀ·ΔW = 0``), which keeps the dominant pre-trained knowledge intact while\nthe adapter learns in the residual subspace — the paper reports the best\nforgetting resistance among LoRA variants on Qwen2.5-7B (the Meno-Lite\nancestor), at a ~19 % training-time overhead.\n\n\n**Projected-gradient form\nand ``A`` (lora_A) as the trainable parameters and its plain forward computes\n``ΔW = B·A``. Whenever the parameters satisfy the invariants ``U_kᵀ·B = 0``\nand ``A·V_k = 0``, the plain forward EQUALS the OPLoRA update\n(``B = P_L·B``, ``A = A·P_R`` ⟹ ``B·A = P_L·B·A·P_R``). The invariants are\nenforced by RE-PROJECTING the parameters after every optimizer step with\nlow-rank corrections (never materializing the dense projectors — the same\nsequential trick as the paper's forward):\n\n    B ← B − U_k (U_kᵀ B),      A ← A − (A V_k) V_kᵀ.\n\nDrift detection uses the tensor ``_version`` counters, so the repair fires\nexactly when needed and for ANY optimizer — important because Adam's\nelementwise rescaling of the gradient does NOT preserve the subspace\n(projecting the *gradients* alone would be an approximation). The repair runs\nvia a forward pre-hook on the model (any forward — a training micro-batch,\nan anchor generation, a probe — always sees projected parameters) plus a\nTrainerCallback on step end; ``merge_and_unload`` then produces exactly\n``W + scaling·B·A`` with the projected factors.\n\n**Cost\n(:func:`compute_layer_spaces`, ``torch.svd_lowrank``; the paper measures\n~5.5 min for a 7B model), fp32 buffers ``U_k``/``V_k`` (~1 GB at k=128 for\nthe q/v/up/down/o set of a Qwen2.5-7B-class model) and a per-step repair of\nfour small matmuls per layer (negligible against the forward).\n\nThe truncated SVD is randomized; ``sft_train.py`` seeds the global torch RNG\nbefore the projection is built, so the spaces are reproducible within a run."

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Default number of top singular directions to preserve (the paper's
#: OPLoRA-128 shows the best forgetting resistance; k=16 the slightly better
#: task-specific variant — both were "not tuned" in the paper).


DEFAULT_OPLORA_K = 128


def project_left(tensor: Any, u_k: Any) -> Any:
    'Return ``P_L @ tensor = tensor − U_k (U_kᵀ tensor)`` (low-rank, fp32).'
    import torch

    t = tensor.detach().to(torch.float32)
    return t - u_k @ (u_k.T @ t)


def project_right(tensor: Any, v_k: Any) -> Any:
    'Return ``tensor @ P_R = tensor − (tensor V_k) V_kᵀ`` (low-rank, fp32).'
    import torch

    t = tensor.detach().to(torch.float32)
    return t - (t @ v_k) @ v_k.T


@dataclass
class LayerSpaces:
    'The OPLoRA bookkeeping for ONE adapted Linear layer.\n\n\n    ``lora_a`` / ``lora_b`` are the peft adapter parameters ([r, in] and\n    [out, r]); ``u_k`` / ``v_k`` are orthonormal fp32 bases of the top-k left /\n    right singular subspaces of the frozen weight; ``versions`` caches the\n    tensor ``_version`` counters after the last projection (drift detection).'

    module_name: str
    lora_a: Any
    lora_b: Any
    u_k: Any
    v_k: Any
    k: int
    versions: tuple[int, int] = (0, 0)

    def _align_subspace_devices(self) -> None:
        "Move the SVD bases onto the adapter parameters' device (cached).\n\n\n        The bases are computed by ``from_peft_model`` while the model is\n        still on its LOAD-time device (the Trainer moves it to the GPU only\n        later, in ``Trainer.__init__``), and they are plain dataclass fields\n        — NOT module buffers — so ``model.to(device)`` never moves them. The\n        first repair after that move would mix CPU bases with CUDA\n        parameters and die with ``RuntimeError: Expected all tensors to be\n        on the same device`` (the exact crash of the 2026-09-25 run, step\n        1). Align lazily at use time and cache the moved tensors back: the\n        copy is paid exactly once per layer per device move.\n``model.to(device)``"
        device = self.lora_b.device
        if self.u_k.device != device:
            self.u_k = self.u_k.to(device)
        if self.v_k.device != device:
            self.v_k = self.v_k.to(device)

    def project(self) -> None:
        'Re-project the adapter parameters into the orthogonal complement.'
        import torch

        self._align_subspace_devices()
        with torch.no_grad():
            self.lora_b.data.copy_(
                project_left(self.lora_b.data, self.u_k).to(self.lora_b.dtype)
            )
            self.lora_a.data.copy_(
                project_right(self.lora_a.data, self.v_k).to(self.lora_a.dtype)
            )
        self.versions = (self.lora_a._version, self.lora_b._version)

    def drifted(self) -> bool:
        'True when either parameter was mutated since the last projection.'
        return (
            self.lora_a._version,
            self.lora_b._version,
        ) != self.versions

    def delta_weight(self) -> Any:
        'The effective low-rank update ``B @ A`` in fp32 (no scaling).'
        import torch

        return self.lora_b.detach().to(torch.float32) @ self.lora_a.detach().to(
            torch.float32
        )

    def interference(self) -> dict[str, Any]:
        "The paper's subspace-alignment metric ρ_k for this layer.\n\n\n        ``rho`` = ‖U_kᵀ ΔW‖²_F / ‖ΔW‖²_F (0 = the update lives entirely in the\n        orthogonal complement — the OPLoRA goal; the paper reports ρ₁₂₈ ≈\n        0.003 thanks to the truncated-SVD approximation). ``None`` when\n        ``ΔW = 0`` (a freshly initialised adapter, B = 0). ``triple_error`` is\n        the top-k singular-triple preservation residual ‖U_kᵀ ΔW V_k‖_F."
        import torch

        self._align_subspace_devices()
        delta = self.delta_weight()
        denom = torch.linalg.norm(delta).item() ** 2
        left = torch.linalg.norm(self.u_k.T @ delta).item() ** 2
        triple = torch.linalg.norm(self.u_k.T @ delta @ self.v_k).item()
        return {
            "rho": (left / denom) if denom > 0.0 else None,
            "triple_error": triple,
            "delta_norm": math.sqrt(denom),
        }


@dataclass
class OploraProjection:
    'The built projection set over every adapted layer of a peft model.'

    layers: dict[str, LayerSpaces] = field(default_factory=dict)
    k: int = DEFAULT_OPLORA_K

    # ---------------------------------------------------------------- build
    @classmethod
    def from_peft_model(
        cls,
        model: Any,
        *,
        k: int = DEFAULT_OPLORA_K,
        niter: int = 4,
        oversample: int = 16,
        progress: bool = True,
    ) -> OploraProjection:
        'Compute the top-k singular spaces of every LoRA target weight.\n\n\n        Walks the peft model for modules carrying ``lora_A``/``lora_B`` (duck\n        typing — no peft internals are imported), reads the FROZEN base weight\n        of each, and runs a truncated SVD (:func:`torch.svd_lowrank`, fp32,\n        followed by a QR re-orthonormalisation so the projectors are exact).\n        ``k`` is clamped to ``min(out, in)`` of each weight with a warning.\n        DoRA layers are rejected (unsupported). ``progress`` adds a tqdm bar\n        (the one-time SVD of a 7B model takes minutes — the paper: 5.5 min).'
        import torch

        layers: dict[str, LayerSpaces] = {}
        items = list(model.named_modules())
        if progress:
            try:
                from tqdm import tqdm

                items = tqdm(
                    items, desc="OPLoRA truncated SVD", unit="module", leave=False
                )
            except ImportError:  # pragma: no cover - tqdm is a core dep
                pass
        for name, module in items:
            lora_a_dict = getattr(module, "lora_A", None)
            lora_b_dict = getattr(module, "lora_B", None)
            # peft stores the adapters in an nn.ModuleDict (duck-typed: both
            # dict and ModuleDict provide .keys()/__getitem__).
            
            
            if not (
                hasattr(lora_a_dict, "keys") and hasattr(lora_b_dict, "keys")
            ):
                continue
            if name in layers:
                continue
            use_dora = getattr(module, "use_dora", None)
            if isinstance(use_dora, dict) and any(use_dora.values()):
                raise NotImplementedError(
                    f'OPLoRA does not support DoRA adapters ({name}'
                )
            get_base = getattr(module, "get_base_layer", None)
            base_layer = get_base() if callable(get_base) else None
            base_weight = getattr(base_layer, "weight", None)
            if base_weight is None:
                logger.warning(
                    "OPLoRA: no base weight on %s; the layer is skipped.", name
                )
                continue
            for adapter in sorted(lora_a_dict):
                a = lora_a_dict[adapter].weight
                b = lora_b_dict[adapter].weight
                w = base_weight.detach().to(torch.float32)
                out_features, in_features = w.shape
                k_eff = min(k, out_features, in_features)
                if k_eff < k:
                    logger.warning(
                        "OPLoRA: k=%d clamped to %d for %s (%d×%d).",
                        k, k_eff, name, out_features, in_features,
                    )
                q = min(k_eff + oversample, out_features, in_features)
                u, _s, v = torch.svd_lowrank(w, q=q, niter=niter)
                u_k, _ = torch.linalg.qr(u[:, :k_eff])
                v_k, _ = torch.linalg.qr(v[:, :k_eff])
                layers[name] = LayerSpaces(
                    module_name=name,
                    lora_a=a,
                    lora_b=b,
                    u_k=u_k.contiguous(),
                    v_k=v_k.contiguous(),
                    k=k_eff,
                )
        if not layers:
            raise RuntimeError(
                'OPLoRA found no LoRA layers — wrap the model with peft first (get_peft_model) and without --no-lora.'
            )
        return cls(layers=layers, k=k)

    # ------------------------------------------------------------- project
    def project_weights(self) -> int:
        'Project every layer (idempotent); returns the number of layers.'
        n = 0
        for layer in self.layers.values():
            layer.project()
            n += 1
        return n

    def repair_if_drifted(self) -> int:
        'Re-project only the layers mutated since their last projection.'
        n = 0
        for layer in self.layers.values():
            if layer.drifted():
                layer.project()
                n += 1
        return n

    # -------------------------------------------------------- diagnostics
    def interference(self) -> dict[str, Any]:
        "Per-layer ρ_k + the preservation residual (the paper's metric)."
        per_layer: dict[str, Any] = {}
        rhos: list[float] = []
        for name, layer in sorted(self.layers.items()):
            report = layer.interference()
            per_layer[name] = report
            if report["rho"] is not None:
                rhos.append(report["rho"])
        return {
            "per_layer": per_layer,
            "mean_rho": (sum(rhos) / len(rhos)) if rhos else None,
            "max_rho": max(rhos) if rhos else None,
            "n_layers": len(self.layers),
            "k": self.k,
        }

    # -------------------------------------------------------------- hooks
    def make_trainer_callback(self) -> Any:
        "A TrainerCallback repairing the invariants after optimizer steps.\n\n\n        ``on_step_end`` fires after every optimizer update (before the other\n        step-end callbacks — register this one FIRST so anchor saves observe\n        projected parameters); ``on_evaluate`` covers evaluation entry points.\n        The repair is a no-op unless a ``_version`` counter moved.\n\n        The callback SUBCLASSES ``transformers.TrainerCallback`` when the\n        library is importable: transformers' ``CallbackHandler.call_event``\n        dispatches EVERY event via ``getattr(callback, event)`` and dies with\n        ``AttributeError`` on the first missing one (the first cluster\n        ``--oplora`` run crashed on ``on_train_begin`` — a plain duck-typed\n        class defining only the repair hooks is NOT enough; the dispatch is\n        identical across transformers 4.x/v5). Without\n        transformers the fallback base synthesises no-op ``on_*`` handlers so\n        the module stays importable in transformers-less offline tests.\n        ``on_train_begin``"
        try:
            from transformers.trainer_callback import (
                TrainerCallback as _CallbackBase,
            )
        except Exception:  # noqa: BLE001 - offline/transformers-less fallback

            class _CallbackBase:  # type: ignore[no-redef]
                def __getattr__(self, name: str) -> Any:
                    if name.startswith("on_"):
                        return lambda *args, **kwargs: None
                    raise AttributeError(name)

        class _OploraCallback(_CallbackBase):  # real TrainerCallback if present
            def __init__(self, projection: OploraProjection) -> None:
                super().__init__()
                self._projection = projection

            def on_step_end(self, args, state, control, **kwargs) -> None:
                self._projection.repair_if_drifted()

            def on_evaluate(self, args, state, control, **kwargs) -> None:
                self._projection.repair_if_drifted()

            def on_prediction_step(self, args, state, control, **kwargs) -> None:
                self._projection.repair_if_drifted()

        return _OploraCallback(self)

    def register_forward_guard(self, model: Any) -> Any:
        'Pre-forward hook: repair before ANY forward (training, anchors).'

        def _hook(_module: Any, _inputs: Any) -> None:
            self.repair_if_drifted()

        return model.register_forward_pre_hook(_hook)
