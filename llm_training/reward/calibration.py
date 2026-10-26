'Fluency-score calibration — load fitted thresholds, apply via numpy interp.\n\n\n**Currently unused by default.** The legacy ruRoberta-RuCoLa ``P(acceptable)``\nwas overconfident and squashed (most answers scored ≈ 0.98), which saturated the\nfluency reward and killed its gradient — hence the isotonic calibrator. The\nreplacement grammar classifier produces well-spread, reality-matching raw\nscores, so ``fluency_calibrator_path`` defaults to ``None`` and the reward path\napplies an :class:`IdentityCalibrator` (raw ``P(grammatical)`` used directly).\nThe calibrator machinery is kept for optional future use / ablations.\n\n:class:`IdentityCalibrator`\n\nThe module loads a fitted threshold table and applies it with\n:func:`numpy.interp` — **scikit-learn is never imported in the reward path**\n(only in the offline fit script ``scripts/calibrate_fluency.py``).\n\n**scikit-learn'

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from llm_training.reward.scoring import safe_score

# : JSON marker for the isotonic calibrator artifact.
_CALIBRATOR_KIND = "isotonic"

#: Minimum number of (x, y) thresholds for a meaningful calibrator.
# :
_MIN_THRESHOLDS = 2


@runtime_checkable
class Calibrator(Protocol):
    'Maps a raw fluency score to a calibrated ``[0, 1]`` value.'

    def apply(self, raw: float | None) -> float:
        """Return the calibrated score (``None`` / NaN / inf → ``0.0``)."""
        ...

    def describe(self) -> dict[str, Any]:
        'Log-safe description'
        ...


class IdentityCalibrator:
    'No-op calibrator: clamps the raw score to ``[0, 1]`` unchanged.\n\n\n    Used when no calibrator path is configured (the default with the grammar\n    classifier, whose raw scores are already well-spread). The fluency reward\n    then runs on the raw ``P(grammatical)`` directly.'

    def apply(self, raw: float | None) -> float:
        return safe_score(raw)

    def describe(self) -> dict[str, Any]:
        return {"kind": "identity"}


class IsotonicCalibrator:
    'Isotonic-regression fluency calibrator (thresholds loaded from JSON).\n\n\n    Built from paired ``(x, y)`` threshold arrays (strictly increasing ``x``,\n    monotonic ``y``). :meth:`apply` linearly interpolates between the points via\n    :func:`numpy.interp` and clamps to ``[0, 1]``; out-of-range inputs take the\n    endpoint values (constant extrapolation), matching the isotonic step\n    function. This mirrors the MERA ruDetox ``fluency_cal`` (interp1d over\n    fitted thresholds), but with **our own** thresholds fit on our distribution.'

    def __init__(self, x: Any, y: Any) -> None:
        xp = np.asarray(x, dtype=np.float64)
        yp = np.asarray(y, dtype=np.float64)
        if xp.ndim != 1 or yp.ndim != 1 or xp.shape[0] != yp.shape[0]:
            raise ValueError(
                'Calibrator x and y must be 1-D arrays of equal length.'
            )
        if xp.shape[0] < _MIN_THRESHOLDS:
            raise ValueError(
                f'Calibrator needs >= {_MIN_THRESHOLDS} thresholds, got {xp.shape[0]}'
            )
        # numpy.interp requires a strictly increasing xp. The offline fit script
        # emits sklearn's X_thresholds_ (strictly increasing by construction);
        # validate defensively so a malformed artifact fails loudly here, not
        # silently at reward time.
        # numpy.interp
        
        if not np.all(np.diff(xp) > 0):
            raise ValueError(
                'Calibrator x thresholds must be strictly increasing.'
            )
        if np.any(yp < 0.0) or np.any(yp > 1.0):
            raise ValueError(
                'Calibrator y thresholds must lie in [0, 1].'
            )
        self._xp = xp
        self._yp = yp

    @classmethod
    def from_file(cls, path: str | Path) -> "IsotonicCalibrator":
        'Load a calibrator from a JSON artifact written by the fit script.\n\n\n        Expected schema: ``{"kind": "isotonic", "x": [...], "y": [...]}``.'
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("kind") != _CALIBRATOR_KIND:
            raise ValueError(
                f"Calibrator artifact {p} is not an '{_CALIBRATOR_KIND}"
            )
        x = data.get("x")
        y = data.get("y")
        if x is None or y is None:
            raise ValueError(
                f'Calibrator artifact {p}'
            )
        return cls(x, y)

    def apply(self, raw: float | None) -> float:
        if raw is None:
            return 0.0
        try:
            r = float(raw)
        except (TypeError, ValueError):
            return 0.0
        if r != r or r in (float("inf"), float("-inf")):  # NaN / inf
            return 0.0
        # np.interp clamps out-of-range inputs to the endpoint y-values
        # (constant extrapolation) — exactly the isotonic step-function behaviour.
        calibrated = float(np.interp(r, self._xp, self._yp))
        return safe_score(calibrated)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": _CALIBRATOR_KIND,
            "n_thresholds": int(self._xp.shape[0]),
            "x_min": float(self._xp[0]),
            "x_max": float(self._xp[-1]),
            "y_min": float(self._yp.min()),
            "y_max": float(self._yp.max()),
        }


def load_calibrator(path: str | Path | None) -> Calibrator:
    'Resolve a calibrator from ``path`` (``None`` → :class:`IdentityCalibrator`).\n\n\n    This is the single entry point consumers use; ``RewardConfig`` wires its\n    ``fluency_calibrator_path`` through here. A ``None``/empty path returns an\n    :class:`IdentityCalibrator` (no calibration — for ablations or before\n    Phase 0 has produced an artifact).'
    if not path:
        return IdentityCalibrator()
    return IsotonicCalibrator.from_file(path)
