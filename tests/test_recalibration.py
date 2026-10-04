"""Tests for model calibration diagnostics, surveillance, and post-hoc recalibration."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_detection.recalibration import (
    BaseRecalibrator,
    CalibrationBin,
    CalibrationDiagnostics,
    CalibrationDriftReport,
    CalibrationStrategy,
    IsotonicRecalibrator,
    PlattRecalibrator,
    RecalibrationMethod,
    TemperatureRecalibrator,
    _compute_bin_edges,
    _validate_arrays,
    _validate_probabilities,
    compute_calibration_diagnostics,
    compute_calibration_drift,
    create_recalibrator,
    load_recalibrator,
)


def test_validate_arrays_valid() -> None:
    probs = np.array([0.1, 0.5, 0.9])
    targets = np.array([0, 0, 1])
    p_out, t_out = _validate_arrays(probs, targets)
    assert np.array_equal(p_out, probs)
    assert np.array_equal(t_out, targets)


def test_validate_arrays_errors() -> None:
    # Empty probabilities
    with pytest.raises(ValueError, match="cannot be empty"):
        _validate_probabilities(np.array([]))
    with pytest.raises(ValueError, match="cannot be empty"):
        _validate_arrays(np.array([]), np.array([]))

    # Non-finite values
    with pytest.raises(ValueError, match="non-finite"):
        _validate_probabilities(np.array([0.1, np.nan, 0.5]))
    with pytest.raises(ValueError, match="non-finite"):
        _validate_probabilities(np.array([0.1, np.inf]))

    # Out of [0, 1] range
    with pytest.raises(ValueError, match=r"strictly in \[0\.0, 1\.0\]"):
        _validate_probabilities(np.array([-0.01, 0.5]))
    with pytest.raises(ValueError, match=r"strictly in \[0\.0, 1\.0\]"):
        _validate_probabilities(np.array([0.5, 1.05]))

    # Target shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        _validate_arrays(np.array([0.1, 0.5]), np.array([0]))

    # Target non-finite
    with pytest.raises(ValueError, match="Target array contains non-finite"):
        _validate_arrays(np.array([0.1, 0.5]), np.array([0, np.nan]))

    # Target non-binary
    with pytest.raises(ValueError, match="Binary target must contain only 0 and 1"):
        _validate_arrays(np.array([0.1, 0.5]), np.array([0, 2]))


def test_platt_recalibrator_fit_and_transform() -> None:
    np.random.seed(42)
    # Generate uncalibrated probabilities with severe overconfidence
    probs = np.linspace(0.01, 0.99, 100)
    # Binary targets with true sigmoid relationship
    y = (probs > 0.45).astype(int)

    recal = PlattRecalibrator()
    assert recal.method == RecalibrationMethod.SIGMOID
    assert getattr(recal, "is_fitted") is False
    # Unfitted transform returns original probabilities
    unfitted_out = recal.transform(probs)
    assert np.allclose(unfitted_out, probs)

    recal.fit(probs, y)
    assert getattr(recal, "is_fitted") is True
    assert recal.a_ >= 0.0

    calibrated = recal.transform(probs)
    assert len(calibrated) == len(probs)
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)
    # Must be monotonically non-decreasing
    assert np.all(np.diff(calibrated) >= -1e-6)

    # Test serialization round-trip
    payload = recal.to_dict()
    assert payload["method"] == "sigmoid"
    assert payload["is_fitted"] is True
    restored = PlattRecalibrator.from_dict(payload)
    assert restored.is_fitted is True
    assert restored.a_ == pytest.approx(recal.a_)
    assert restored.b_ == pytest.approx(recal.b_)
    assert np.allclose(restored.transform(probs), calibrated)


def test_platt_recalibrator_single_class() -> None:
    probs = np.array([0.1, 0.4, 0.7, 0.9])
    # All zeros
    y_zeros = np.zeros(4)
    recal = PlattRecalibrator()
    recal.fit(probs, y_zeros)
    assert recal.is_fitted
    calibrated = recal.transform(probs)
    assert np.all(calibrated < 0.5)

    # All ones
    y_ones = np.ones(4)
    recal2 = PlattRecalibrator()
    recal2.fit(probs, y_ones)
    assert recal2.is_fitted
    calibrated2 = recal2.transform(probs)
    assert np.all(calibrated2 > 0.5)


def test_isotonic_recalibrator_fit_and_transform() -> None:
    np.random.seed(42)
    probs = np.linspace(0.05, 0.95, 50)
    y = (probs > 0.5).astype(int)

    recal = IsotonicRecalibrator()
    assert recal.method == RecalibrationMethod.ISOTONIC
    assert getattr(recal, "is_fitted") is False
    # Unfitted
    assert np.allclose(recal.transform(probs), probs)

    recal.fit(probs, y)
    assert getattr(recal, "is_fitted") is True
    assert len(recal.x_thresholds_) > 0
    assert len(recal.y_thresholds_) > 0

    calibrated = recal.transform(probs)
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)
    # Monotonicity guarantee
    assert np.all(np.diff(calibrated) >= -1e-6)

    # Test serialization round-trip
    payload = recal.to_dict()
    assert payload["method"] == "isotonic"
    assert payload["is_fitted"] is True
    restored = IsotonicRecalibrator.from_dict(payload)
    assert restored.is_fitted is True
    assert np.allclose(restored.transform(probs), calibrated)


def test_temperature_recalibrator_fit_and_transform() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        TemperatureRecalibrator(temperature=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        TemperatureRecalibrator(temperature=-1.5)

    probs = np.linspace(0.1, 0.9, 40)
    y = (probs > 0.5).astype(int)

    recal = TemperatureRecalibrator()
    assert recal.method == RecalibrationMethod.TEMPERATURE
    # Default T=1.0 returns unchanged
    assert np.allclose(recal.transform(probs), probs)

    recal.fit(probs, y)
    assert recal.is_fitted
    assert recal.temperature_ > 0.0

    calibrated = recal.transform(probs)
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)
    # Temperature scaling strictly preserves rank order
    assert np.array_equal(np.argsort(probs), np.argsort(calibrated))

    # Serialization
    payload = recal.to_dict()
    assert payload["method"] == "temperature"
    restored = TemperatureRecalibrator.from_dict(payload)
    assert restored.temperature_ == pytest.approx(recal.temperature_)
    assert np.allclose(restored.transform(probs), calibrated)


def test_create_and_load_recalibrator() -> None:
    # Valid creations
    p = create_recalibrator("sigmoid")
    assert isinstance(p, PlattRecalibrator)
    iso = create_recalibrator("isotonic")
    assert isinstance(iso, IsotonicRecalibrator)
    t = create_recalibrator("temperature", temperature=1.5)
    assert isinstance(t, TemperatureRecalibrator)
    assert t.temperature_ == 1.5

    # Invalid creations
    with pytest.raises(ValueError, match="Unknown recalibration method"):
        create_recalibrator("invalid_method")

    # Round-trip load_recalibrator
    p_dict = p.to_dict()
    loaded_p = load_recalibrator(p_dict)
    assert isinstance(loaded_p, PlattRecalibrator)

    iso_dict = iso.to_dict()
    loaded_iso = load_recalibrator(iso_dict)
    assert isinstance(loaded_iso, IsotonicRecalibrator)

    t_dict = t.to_dict()
    loaded_t = load_recalibrator(t_dict)
    assert isinstance(loaded_t, TemperatureRecalibrator)

    # BaseRecalibrator.from_dict validation
    with pytest.raises(ValueError, match="missing required 'method'"):
        BaseRecalibrator.from_dict({})
    with pytest.raises(ValueError, match="Unknown recalibration method"):
        BaseRecalibrator.from_dict({"method": "unknown"})


def test_compute_bin_edges() -> None:
    probs = np.array([0.05, 0.1, 0.2, 0.5, 0.8, 0.95])

    # Uniform
    edges_u = _compute_bin_edges(probs, 5, CalibrationStrategy.UNIFORM)
    assert len(edges_u) == 6
    assert edges_u[0] == 0.0 and edges_u[-1] == 1.0

    # Quantile
    edges_q = _compute_bin_edges(probs, 4, CalibrationStrategy.QUANTILE)
    assert len(edges_q) >= 2
    assert edges_q[0] == 0.0 and edges_q[-1] == 1.0

    # Quantile with identical probabilities
    flat_probs = np.array([0.1, 0.1, 0.1, 0.1])
    edges_flat = _compute_bin_edges(flat_probs, 4, CalibrationStrategy.QUANTILE)
    assert len(edges_flat) == 5

    # Quantile with explicit 0.0 and 1.0 endpoints
    edges_full = _compute_bin_edges(np.array([0.0, 0.5, 1.0]), 2, CalibrationStrategy.QUANTILE)
    assert edges_full[0] == 0.0 and edges_full[-1] == 1.0

    # Tiered
    edges_t = _compute_bin_edges(
        probs, 6, CalibrationStrategy.TIERED, tiered_thresholds=(0.25, 0.75)
    )
    assert len(edges_t) >= 4
    assert 0.25 in edges_t
    assert 0.75 in edges_t

    # Tiered with default thresholds
    edges_t_default = _compute_bin_edges(probs, 6, CalibrationStrategy.TIERED)
    assert 0.20 in edges_t_default
    assert 0.80 in edges_t_default


def test_compute_calibration_diagnostics_uniform() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    diag = compute_calibration_diagnostics(y, probs, bins=4, strategy=CalibrationStrategy.UNIFORM)
    assert diag.rows == 8
    assert diag.bins == 4
    assert diag.strategy == "uniform"
    assert 0.0 <= diag.brier_score <= 1.0
    assert 0.0 <= diag.expected_calibration_error <= 1.0
    assert 0.0 <= diag.max_calibration_error <= 1.0
    assert 0.0 <= diag.reliability <= 1.0
    assert 0.0 <= diag.resolution <= 1.0
    assert 0.0 <= diag.uncertainty <= 1.0
    assert diag.monotonicity_violations == 0
    assert len(diag.detail) == 4

    payload = diag.to_dict()
    assert payload["rows"] == 8
    restored = CalibrationDiagnostics.from_dict(payload)
    assert restored.rows == diag.rows
    assert restored.brier_score == pytest.approx(diag.brier_score)
    assert restored.expected_calibration_error == pytest.approx(diag.expected_calibration_error)
    assert len(restored.detail) == len(diag.detail)


def test_compute_calibration_diagnostics_empty_bins_and_monotonicity() -> None:
    # Dataset with empty middle bins and a monotonicity violation
    y = np.array([1, 0, 0, 0, 0])
    probs = np.array([0.05, 0.06, 0.92, 0.93, 0.94])

    diag = compute_calibration_diagnostics(y, probs, bins=10, strategy=CalibrationStrategy.UNIFORM)
    assert diag.rows == 5
    assert any(b.count == 0 for b in diag.detail)
    # The low probability bin had 1 positive (50%), higher bin had 0 positives (0%) -> violation
    assert diag.monotonicity_violations >= 1


def test_compute_calibration_diagnostics_errors() -> None:
    y = np.array([0, 1])
    probs = np.array([0.2, 0.8])

    with pytest.raises(ValueError, match="between 2 and 50"):
        compute_calibration_diagnostics(y, probs, bins=1)
    with pytest.raises(ValueError, match="between 2 and 50"):
        compute_calibration_diagnostics(y, probs, bins=51)
    with pytest.raises(ValueError, match="Unknown calibration strategy"):
        compute_calibration_diagnostics(y, probs, bins=5, strategy="nonexistent")
    with pytest.raises(TypeError, match="Expected CalibrationStrategy or str"):
        compute_calibration_diagnostics(y, probs, bins=5, strategy=123)  # type: ignore[arg-type]


def test_compute_calibration_drift() -> None:
    bin1 = CalibrationBin(0, 0.0, 0.5, 50, 0.2, 0.2, 0.0)
    bin2 = CalibrationBin(1, 0.5, 1.0, 50, 0.8, 0.8, 0.0)

    ref = CalibrationDiagnostics(
        rows=100,
        bins=2,
        strategy="uniform",
        brier_score=0.05,
        expected_calibration_error=0.01,
        max_calibration_error=0.02,
        root_mean_squared_error=0.01,
        reliability=0.001,
        resolution=0.20,
        uncertainty=0.25,
        monotonicity_violations=0,
        detail=(bin1, bin2),
    )

    # Stable scenario
    drift_stable = compute_calibration_drift(ref, ref)
    assert drift_stable.status == "STABLE"
    assert drift_stable.ece_delta == pytest.approx(0.0)
    assert len(drift_stable.warnings) == 0

    # Warning scenario: moderate ECE increase
    cur_warn = CalibrationDiagnostics(
        rows=100,
        bins=2,
        strategy="uniform",
        brier_score=0.08,
        expected_calibration_error=0.045,  # delta = +0.035 >= 0.03
        max_calibration_error=0.06,
        root_mean_squared_error=0.045,
        reliability=0.002,
        resolution=0.18,
        uncertainty=0.25,
        monotonicity_violations=0,
        detail=(bin1, bin2),
    )
    drift_warn = compute_calibration_drift(ref, cur_warn)
    assert drift_warn.status == "WARNING"
    assert any("warning threshold" in w for w in drift_warn.warnings)

    # Degraded scenario: large ECE increase
    cur_degraded = CalibrationDiagnostics(
        rows=100,
        bins=2,
        strategy="uniform",
        brier_score=0.16,
        expected_calibration_error=0.09,  # delta = +0.08 >= 0.07
        max_calibration_error=0.12,
        root_mean_squared_error=0.09,
        reliability=0.008,
        resolution=0.15,
        uncertainty=0.25,
        monotonicity_violations=1,
        detail=(bin1, bin2),
    )
    drift_degraded = compute_calibration_drift(ref, cur_degraded)
    assert drift_degraded.status == "DEGRADED"
    assert any("alarm threshold" in w for w in drift_degraded.warnings)

    # Threshold validation errors
    with pytest.raises(ValueError, match="Invalid thresholds"):
        compute_calibration_drift(ref, cur_warn, warn_ece_delta=-0.01)
    with pytest.raises(ValueError, match="Invalid thresholds"):
        compute_calibration_drift(ref, cur_warn, warn_ece_delta=0.05, alarm_ece_delta=0.02)

    # Serialization round-trip
    payload = drift_degraded.to_dict()
    restored = CalibrationDriftReport.from_dict(payload)
    assert restored.status == "DEGRADED"
    assert restored.ece_delta == pytest.approx(drift_degraded.ece_delta)


def test_isotonic_rebuild_regressor() -> None:
    recal = IsotonicRecalibrator(x_thresholds=[0.0, 1.0], y_thresholds=[0.0, 1.0], is_fitted=True)
    recal._regressor = None  # noqa: SLF001
    # Calling transform should rebuild the regressor
    out = recal.transform(np.array([0.2, 0.8]))
    assert np.allclose(out, [0.2, 0.8])

    # Empty thresholds when is_fitted is True returns original probs
    recal_empty = IsotonicRecalibrator(is_fitted=True)
    assert np.allclose(recal_empty.transform(np.array([0.2, 0.8])), [0.2, 0.8])


def test_create_recalibrator_with_enum() -> None:
    recal_sig = create_recalibrator(RecalibrationMethod.SIGMOID)
    assert isinstance(recal_sig, PlattRecalibrator)
    recal_iso = create_recalibrator(RecalibrationMethod.ISOTONIC)
    assert isinstance(recal_iso, IsotonicRecalibrator)
    recal_temp = create_recalibrator(RecalibrationMethod.TEMPERATURE)
    assert isinstance(recal_temp, TemperatureRecalibrator)

    with pytest.raises(TypeError, match="Expected RecalibrationMethod or str"):
        create_recalibrator(12345)  # type: ignore[arg-type]


def test_platt_and_temperature_optimizer_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    probs = np.array([0.2, 0.8])
    y = np.array([0, 1])

    # Platt fallback
    class FakeResult:
        success = False
        fun = np.nan
        x = np.array([1.0, 0.0])

    monkeypatch.setattr(
        "fraud_detection.recalibration.minimize", lambda *_args, **_kwargs: FakeResult()
    )
    platt = PlattRecalibrator()
    platt.fit(probs, y)
    assert platt.a_ == 1.0 and platt.b_ == 0.0

    # Temperature fallback
    class FakeScalarResult:
        success = False
        fun = np.nan
        x = -1.0

    monkeypatch.setattr(
        "fraud_detection.recalibration.minimize_scalar",
        lambda *_args, **_kwargs: FakeScalarResult(),
    )
    temp = TemperatureRecalibrator()
    temp.fit(probs, y)
    assert temp.temperature_ == 1.0


def test_compute_bin_edges_edge_cases() -> None:
    # Quantile where probabilities are strictly within (0.2, 0.8)
    probs = np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
    edges = _compute_bin_edges(probs, 3, CalibrationStrategy.QUANTILE)
    assert edges[0] == 0.0
    assert edges[-1] == 1.0

    # Invalid strategy object
    with pytest.raises(ValueError, match="Unknown calibration strategy"):
        _compute_bin_edges(probs, 3, "unsupported_strategy")  # type: ignore[arg-type]


def test_diagnostics_with_enums_and_drift_branches() -> None:
    y = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.3, 0.7, 0.9])

    # Passing strategy as enum
    diag_q = compute_calibration_diagnostics(
        y, probs, bins=2, strategy=CalibrationStrategy.QUANTILE
    )
    assert diag_q.strategy == "quantile"

    diag_t = compute_calibration_diagnostics(y, probs, bins=3, strategy=CalibrationStrategy.TIERED)
    assert diag_t.strategy == "tiered"

    # Drift branch: empty bins divergence
    empty_detail = (CalibrationBin(0, 0.0, 1.0, 0, 0.5, 0.0, 0.0),)
    ref_empty = CalibrationDiagnostics(
        rows=0,
        bins=1,
        strategy="uniform",
        brier_score=0.1,
        expected_calibration_error=0.01,
        max_calibration_error=0.01,
        root_mean_squared_error=0.01,
        reliability=0.01,
        resolution=0.01,
        uncertainty=0.01,
        monotonicity_violations=0,
        detail=empty_detail,
    )
    drift_empty = compute_calibration_drift(ref_empty, ref_empty)
    assert drift_empty.max_divergence == 0.0

    # Drift branch: monotonicity violation increase causing WARNING
    ref_base = CalibrationDiagnostics(
        rows=10,
        bins=1,
        strategy="uniform",
        brier_score=0.1,
        expected_calibration_error=0.01,
        max_calibration_error=0.01,
        root_mean_squared_error=0.01,
        reliability=0.01,
        resolution=0.01,
        uncertainty=0.01,
        monotonicity_violations=0,
        detail=empty_detail,
    )
    cur_mono = CalibrationDiagnostics(
        rows=10,
        bins=1,
        strategy="uniform",
        brier_score=0.11,
        expected_calibration_error=0.015,  # ECE delta < 0.03
        max_calibration_error=0.02,
        root_mean_squared_error=0.015,
        reliability=0.01,
        resolution=0.01,
        uncertainty=0.01,
        monotonicity_violations=2,  # increased
        detail=empty_detail,
    )
    drift_mono = compute_calibration_drift(ref_base, cur_mono)
    assert drift_mono.status == "WARNING"

    # Drift branch: brier score degraded > 0.05 causing WARNING
    cur_brier = CalibrationDiagnostics(
        rows=10,
        bins=1,
        strategy="uniform",
        brier_score=0.18,  # delta = +0.08 > 0.05
        expected_calibration_error=0.015,
        max_calibration_error=0.02,
        root_mean_squared_error=0.015,
        reliability=0.01,
        resolution=0.01,
        uncertainty=0.01,
        monotonicity_violations=0,
        detail=empty_detail,
    )
    drift_brier = compute_calibration_drift(ref_base, cur_brier)
    assert drift_brier.status == "WARNING"
    assert drift_brier.drift_detected is True
    assert drift_brier.ece_shift == drift_brier.ece_delta
    assert drift_brier.brier_shift == drift_brier.brier_delta
    assert drift_brier.max_gap_shift == drift_brier.max_divergence
