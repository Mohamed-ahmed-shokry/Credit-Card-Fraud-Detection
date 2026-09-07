from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detection.drift import (
    DRIFT_THRESHOLD,
    STABLE_THRESHOLD,
    DriftError,
    assess_drift,
    build_reference_profile,
    default_thresholds,
    resolve_thresholds,
    surveillance_tripped,
)


def test_identical_distribution_is_stable() -> None:
    frame = pd.DataFrame(
        {
            "amount": np.linspace(1, 100, 1_000),
            "velocity": np.tile(np.arange(10), 100),
        }
    )
    profile = build_reference_profile(frame)

    report = assess_drift(profile, frame)

    assert report.overall_status == "stable"
    assert report.max_psi == pytest.approx(0.0)
    assert report.to_dict()["rows"] == 1_000
    assert [item.feature for item in report.features] == ["amount", "velocity"]


def test_shifted_distribution_is_flagged_and_ranked() -> None:
    rng = np.random.default_rng(5)
    reference = pd.DataFrame(
        {
            "stable": rng.normal(0, 1, 2_000),
            "shifted": rng.normal(0, 1, 2_000),
        }
    )
    current = reference.copy()
    current["shifted"] += 10

    report = assess_drift(build_reference_profile(reference), current)

    assert report.overall_status == "drifted"
    assert report.features[0].feature == "shifted"
    assert report.features[0].status == "drifted"
    assert report.features[1].status == "stable"


@pytest.mark.parametrize("bins", [1, 51])
def test_reference_profile_rejects_invalid_bin_count(bins: int) -> None:
    with pytest.raises(DriftError, match="bins"):
        build_reference_profile(pd.DataFrame({"x": [1.0, 2.0]}), bins=bins)


def test_drift_rejects_schema_mismatch() -> None:
    profile = build_reference_profile(pd.DataFrame({"x": [1.0, 2.0]}))

    with pytest.raises(DriftError, match="schema"):
        assess_drift(profile, pd.DataFrame({"wrong": [1.0]}))


def test_drift_rejects_non_finite_values() -> None:
    with pytest.raises(DriftError, match="non-finite"):
        build_reference_profile(pd.DataFrame({"x": [1.0, np.nan]}))

    profile = build_reference_profile(pd.DataFrame({"x": [1.0, 2.0]}))
    with pytest.raises(DriftError, match="non-finite"):
        assess_drift(profile, pd.DataFrame({"x": [np.inf]}))


def test_drift_rejects_empty_frames() -> None:
    with pytest.raises(DriftError, match="Reference features"):
        build_reference_profile(pd.DataFrame())
    with pytest.raises(DriftError, match="Current features"):
        assess_drift({}, pd.DataFrame())


def test_drift_rejects_empty_or_invalid_reference_profile_names() -> None:
    current = pd.DataFrame({"x": [1.0, 2.0]})

    with pytest.raises(DriftError, match="non-empty string feature names"):
        assess_drift({}, current)
    with pytest.raises(DriftError, match="non-empty string feature names"):
        assess_drift({"": {"edges": [None, 0, None], "proportions": [0.5, 0.5]}}, current)


@pytest.mark.parametrize(
    "profile",
    [
        {"x": {"edges": [0, None], "proportions": [1.0]}},
        {"x": {"edges": [None, 0, None]}},
        {"x": {"edges": [None, 0, None], "proportions": [1.0]}},
        {"x": {"edges": [None, 1, 0, None], "proportions": [0.3, 0.3, 0.4]}},
        {"x": {"edges": [None, float("inf"), None], "proportions": [0.5, 0.5]}},
        {"x": {"edges": [None, 0, None], "proportions": [-0.1, 1.1]}},
        {"x": {"edges": [None, 0, None], "proportions": [0.2, 0.2]}},
        {"x": {"edges": [None, 0, None], "proportions": [float("nan"), 0.0]}},
    ],
)
def test_drift_rejects_malformed_reference_profile(
    profile: dict[str, dict[str, object]],
) -> None:
    with pytest.raises(DriftError, match="Reference profile"):
        assess_drift(profile, pd.DataFrame({"x": [0.0, 1.0]}))


def test_drift_can_report_warning_level() -> None:
    profile = {"x": {"edges": [None, 0.5, None], "proportions": [0.5, 0.5]}}
    current = pd.DataFrame({"x": [0.0] * 68 + [1.0] * 32})

    report = assess_drift(profile, current)

    assert report.overall_status == "warning"


def test_default_thresholds_match_reference_cutoffs() -> None:
    assert default_thresholds() == {
        "warning_at": STABLE_THRESHOLD,
        "drift_at": DRIFT_THRESHOLD,
    }


def test_assess_drift_honors_persisted_thresholds() -> None:
    profile = {"x": {"edges": [None, 0.5, None], "proportions": [0.5, 0.5]}}
    current = pd.DataFrame({"x": [0.0] * 68 + [1.0] * 32})

    report = assess_drift(profile, current, thresholds={"warning_at": 0.5, "drift_at": 1.0})

    assert report.overall_status == "stable"
    assert report.to_dict()["thresholds"] == {"warning_at": 0.5, "drift_at": 1.0}


def test_assess_drift_reports_default_thresholds() -> None:
    profile = {"x": {"edges": [None, 0.5, None], "proportions": [0.5, 0.5]}}

    report = assess_drift(profile, pd.DataFrame({"x": [0.0, 1.0]}))

    assert report.to_dict()["thresholds"] == default_thresholds()


@pytest.mark.parametrize(
    "thresholds",
    [
        "not-a-mapping",
        {"warning_at": 0.1},
        {"warning_at": "high", "drift_at": 0.25},
        {"warning_at": float("nan"), "drift_at": 0.25},
        {"warning_at": -0.1, "drift_at": 0.25},
        {"warning_at": 0.25, "drift_at": 0.25},
        {"warning_at": 0.5, "drift_at": 0.25},
    ],
)
def test_resolve_thresholds_rejects_invalid_cutoffs(thresholds: object) -> None:
    with pytest.raises(DriftError, match="hresholds"):
        resolve_thresholds(thresholds)


def test_resolve_thresholds_defaults_to_reference_cutoffs() -> None:
    assert resolve_thresholds(None) == (STABLE_THRESHOLD, DRIFT_THRESHOLD)


@pytest.mark.parametrize(
    ("overall_status", "fail_on", "tripped"),
    [
        ("stable", None, False),
        ("stable", "warning", False),
        ("stable", "drifted", False),
        ("warning", "warning", True),
        ("warning", "drifted", False),
        ("drifted", "warning", True),
        ("drifted", "drifted", True),
    ],
)
def test_surveillance_tripped_compares_severity(
    overall_status: str, fail_on: str | None, tripped: bool
) -> None:
    assert surveillance_tripped(overall_status, fail_on) is tripped


@pytest.mark.parametrize("fail_on", ["bogus", ""])
def test_surveillance_tripped_rejects_unknown_levels(fail_on: str) -> None:
    with pytest.raises(DriftError, match="surveillance level"):
        surveillance_tripped("drifted", fail_on)


def test_drift_normalizes_non_string_feature_names() -> None:
    frame = pd.DataFrame({1: [0.0, 1.0, 2.0, 3.0]})

    report = assess_drift(build_reference_profile(frame), frame)

    assert report.features[0].feature == "1"
    assert report.max_psi == pytest.approx(0.0)


@pytest.mark.parametrize(
    "columns",
    [
        ["x", "x"],
        [1, "1"],
        ["", "x"],
    ],
)
def test_reference_profile_rejects_ambiguous_feature_names(
    columns: list[object],
) -> None:
    frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=columns)

    with pytest.raises(DriftError, match="feature names"):
        build_reference_profile(frame)


def test_drift_rejects_non_numeric_features_with_domain_error() -> None:
    with pytest.raises(DriftError, match="must be numeric"):
        build_reference_profile(pd.DataFrame({"x": ["not", "numeric"]}))


def test_reference_profile_rejects_numeric_overflow() -> None:
    frame = pd.DataFrame({"x": [-1e308, 1e308] * 10})

    with pytest.raises(DriftError, match="numeric overflow"):
        build_reference_profile(frame)
