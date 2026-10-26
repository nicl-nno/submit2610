'Tests for reward/calibration.py — calibrator load + apply (no sklearn).'

from __future__ import annotations

import json
import math

import pytest

from llm_training.reward.calibration import (
    Calibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    load_calibrator,
)


class TestIdentityCalibrator:
    def test_passes_through_clamped(self):
        c = IdentityCalibrator()
        assert c.apply(0.5) == pytest.approx(0.5)
        assert c.apply(0.0) == pytest.approx(0.0)
        assert c.apply(1.0) == pytest.approx(1.0)

    def test_clamps_out_of_range(self):
        c = IdentityCalibrator()
        assert c.apply(1.5) == pytest.approx(1.0)
        assert c.apply(-0.3) == pytest.approx(0.0)

    def test_none_and_nan_are_zero(self):
        c = IdentityCalibrator()
        assert c.apply(None) == 0.0
        assert c.apply(float("nan")) == 0.0
        assert c.apply(float("inf")) == 0.0

    def test_describe(self):
        assert IdentityCalibrator().describe() == {"kind": "identity"}

    def test_satisfies_protocol(self):
        assert isinstance(IdentityCalibrator(), Calibrator)


class TestIsotonicCalibrator:
    def test_apply_interpolates_known_points(self):
        # Monotonic mapping: raw 0.5 (midpoint of x) → midpoint of y.
        
        c = IsotonicCalibrator(x=[0.0, 1.0], y=[0.0, 1.0])
        assert c.apply(0.5) == pytest.approx(0.5)
        assert c.apply(0.25) == pytest.approx(0.25)

    def test_apply_clamps_below_min_to_endpoint(self):
        # Out-of-range low → y[0] (constant extrapolation, isotonic step).
        
        c = IsotonicCalibrator(x=[0.2, 0.8], y=[0.1, 0.9])
        assert c.apply(0.0) == pytest.approx(0.1)
        assert c.apply(-1.0) == pytest.approx(0.1)

    def test_apply_clamps_above_max_to_endpoint(self):
        c = IsotonicCalibrator(x=[0.2, 0.8], y=[0.1, 0.9])
        assert c.apply(1.0) == pytest.approx(0.9)
        assert c.apply(5.0) == pytest.approx(0.9)

    def test_apply_handles_none_and_nan(self):
        c = IsotonicCalibrator(x=[0.0, 1.0], y=[0.0, 1.0])
        assert c.apply(None) == 0.0
        assert c.apply(float("nan")) == 0.0
        assert c.apply(float("inf")) == 0.0

    def test_non_monotonic_input_is_remapped(self):
        # A typical saturation-fix calibrator: raw scores clustered near 1.0
        # get spread out across [0, 1].
        
        c = IsotonicCalibrator(x=[0.90, 0.95, 0.99, 1.0], y=[0.0, 0.4, 0.8, 1.0])
        assert c.apply(0.975) > 0.4
        assert c.apply(0.975) < 0.8
        assert c.apply(0.90) == pytest.approx(0.0)
        assert c.apply(1.0) == pytest.approx(1.0)

    def test_describe(self):
        c = IsotonicCalibrator(x=[0.0, 0.5, 1.0], y=[0.1, 0.5, 0.9])
        d = c.describe()
        assert d["kind"] == "isotonic"
        assert d["n_thresholds"] == 3
        assert d["x_min"] == pytest.approx(0.0)
        assert d["x_max"] == pytest.approx(1.0)
        assert d["y_min"] == pytest.approx(0.1)
        assert d["y_max"] == pytest.approx(0.9)

    def test_satisfies_protocol(self):
        c = IsotonicCalibrator(x=[0.0, 1.0], y=[0.0, 1.0])
        assert isinstance(c, Calibrator)


class TestIsotonicCalibratorValidation:
    def test_rejects_non_strictly_increasing_x(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            IsotonicCalibrator(x=[0.0, 0.0, 1.0], y=[0.0, 0.5, 1.0])

    def test_rejects_decreasing_x(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            IsotonicCalibrator(x=[1.0, 0.0], y=[1.0, 0.0])

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="equal length"):
            IsotonicCalibrator(x=[0.0, 0.5, 1.0], y=[0.0, 1.0])

    def test_rejects_too_few_thresholds(self):
        with pytest.raises(ValueError, match="thresholds"):
            IsotonicCalibrator(x=[0.5], y=[0.5])

    def test_rejects_y_out_of_range(self):
        with pytest.raises(ValueError, match=r"y thresholds must lie in \[0, 1\]"):
            IsotonicCalibrator(x=[0.0, 1.0], y=[-0.1, 1.0])
        with pytest.raises(ValueError, match=r"y thresholds must lie in \[0, 1\]"):
            IsotonicCalibrator(x=[0.0, 1.0], y=[0.0, 1.5])


class TestIsotonicCalibratorFromFile:
    def test_loads_and_applies(self, tmp_path):
        art = tmp_path / "cal.json"
        art.write_text(
            json.dumps(
                {"kind": "isotonic", "x": [0.0, 0.5, 1.0], "y": [0.0, 0.4, 1.0]}
            ),
            encoding="utf-8",
        )
        c = IsotonicCalibrator.from_file(art)
        assert c.apply(0.5) == pytest.approx(0.4)
        assert c.apply(1.0) == pytest.approx(1.0)

    def test_rejects_wrong_kind(self, tmp_path):
        art = tmp_path / "cal.json"
        art.write_text(json.dumps({"kind": "something_else", "x": [], "y": []}),
                       encoding="utf-8")
        with pytest.raises(ValueError, match="isotonic"):
            IsotonicCalibrator.from_file(art)

    def test_rejects_missing_thresholds(self, tmp_path):
        art = tmp_path / "cal.json"
        art.write_text(json.dumps({"kind": "isotonic"}), encoding="utf-8")
        with pytest.raises(ValueError, match="x.*y"):
            IsotonicCalibrator.from_file(art)


class TestLoadCalibrator:
    def test_none_path_returns_identity(self):
        c = load_calibrator(None)
        assert isinstance(c, IdentityCalibrator)

    def test_empty_path_returns_identity(self):
        assert isinstance(load_calibrator(""), IdentityCalibrator)

    def test_path_returns_isotonic(self, tmp_path):
        art = tmp_path / "cal.json"
        art.write_text(
            json.dumps({"kind": "isotonic", "x": [0.0, 1.0], "y": [0.0, 1.0]}),
            encoding="utf-8",
        )
        c = load_calibrator(art)
        assert isinstance(c, IsotonicCalibrator)
        assert c.apply(0.5) == pytest.approx(0.5)
