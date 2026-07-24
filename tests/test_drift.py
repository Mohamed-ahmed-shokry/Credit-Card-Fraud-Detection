from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detection.drift import DriftError, assess_drift, build_reference_profile


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


@pytest.mark.parametrize(
    "profile",
    [
        {"x": {"edges": [0, None], "proportions": [1.0]}},
        {"x": {"edges": [None, 0, None]}},
        {"x": {"edges": [None, 0, None], "proportions": [1.0]}},
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
