from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detection.drift import (
    DRIFT_THRESHOLD,
    STABLE_THRESHOLD,
    DriftError,
    MultiWindowDriftReport,
    MultiWindowFeatureDrift,
    StreamingProfile,
    assess_drift,
    assess_multi_window_drift,
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


def test_multi_window_drift_stable() -> None:
    frame = pd.DataFrame(
        {
            "amount": np.tile(np.linspace(1, 100, 10), 100),
            "velocity": np.tile(np.arange(10), 100),
        }
    )
    profile = build_reference_profile(frame)

    report = assess_multi_window_drift(profile, frame, short_window_rows=100)

    assert isinstance(report, MultiWindowDriftReport)
    assert isinstance(report.features[0], MultiWindowFeatureDrift)
    assert report.overall_status == "stable"
    assert report.short_window_rows == 100
    assert report.long_window_rows == 1_000
    assert report.max_short_psi == pytest.approx(0.0)
    assert report.max_long_psi == pytest.approx(0.0)
    assert report.max_velocity == pytest.approx(0.0)

    d = report.to_dict()
    assert d["short_window_rows"] == 100
    assert d["long_window_rows"] == 1_000
    assert d["overall_status"] == "stable"
    assert len(d["features"]) == 2
    assert "velocity" in d["features"][0]


def test_multi_window_drift_detects_sudden_acceleration() -> None:
    rng = np.random.default_rng(42)
    reference = pd.DataFrame(
        {
            "stable_feat": rng.normal(0, 1, 1_500),
            "spiking_feat": rng.normal(0, 1, 1_500),
        }
    )
    profile = build_reference_profile(reference)

    # Current has mostly baseline, but the last 150 rows have an extreme shift
    current = pd.DataFrame(
        {
            "stable_feat": rng.normal(0, 1, 1_000),
            "spiking_feat": np.concatenate([rng.normal(0, 1, 850), rng.normal(15, 1, 150)]),
        }
    )

    report = assess_multi_window_drift(profile, current, short_window_rows=100)

    assert report.overall_status == "drifted"
    spiking = next(item for item in report.features if item.feature == "spiking_feat")
    assert spiking.short_psi > spiking.long_psi
    assert spiking.velocity > 0.0
    assert spiking.status == "drifted"
    assert report.max_velocity > 0.0


def test_multi_window_drift_rejects_invalid_short_window() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    profile = build_reference_profile(frame)

    with pytest.raises(DriftError, match=r"short_window_rows must be at least 2"):
        assess_multi_window_drift(profile, frame, short_window_rows=1)

    with pytest.raises(DriftError, match=r"less than short_window_rows"):
        assess_multi_window_drift(profile, frame, short_window_rows=10)


def test_multi_window_drift_honors_thresholds() -> None:
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 1_000)})
    profile = build_reference_profile(reference)
    current = pd.DataFrame({"x": rng.normal(1, 1, 500)})

    # Strict thresholds: turns small change into drifted
    report = assess_multi_window_drift(
        profile, current, short_window_rows=100, thresholds={"warning_at": 0.01, "drift_at": 0.02}
    )
    assert report.overall_status == "drifted"
    assert report.warning_at == 0.01
    assert report.drift_at == 0.02


def test_streaming_profile_updates_and_matches_batch_profile() -> None:
    rng = np.random.default_rng(123)
    frame = pd.DataFrame(
        {
            "amount": rng.uniform(10, 500, 1_000),
            "velocity": rng.normal(5, 2, 1_000),
        }
    )
    batch_ref = build_reference_profile(frame, bins=10)

    sp = StreamingProfile.from_reference_profile(batch_ref)

    # Feed in 5 chunks of 200 rows
    for i in range(5):
        chunk = frame.iloc[i * 200 : (i + 1) * 200]
        sp.update(chunk)

    stream_ref = sp.to_reference_profile()

    for feat in ["amount", "velocity"]:
        assert stream_ref[feat]["mean"] == pytest.approx(batch_ref[feat]["mean"], rel=1e-5)
        assert stream_ref[feat]["standard_deviation"] == pytest.approx(
            batch_ref[feat]["standard_deviation"], rel=1e-5
        )
        assert stream_ref[feat]["proportions"] == pytest.approx(
            batch_ref[feat]["proportions"], rel=1e-5
        )


def test_streaming_profile_serialization_roundtrip() -> None:
    frame = pd.DataFrame({"x": np.linspace(0, 100, 200)})
    sp = StreamingProfile({"x": [25.0, 50.0, 75.0]})
    sp.update(frame.iloc[:100])

    state = sp.to_dict()
    sp_restored = StreamingProfile.from_dict(state)

    # Update both with the second half
    sp.update(frame.iloc[100:])
    sp_restored.update(frame.iloc[100:])

    assert sp.to_reference_profile() == sp_restored.to_reference_profile()


def test_streaming_profile_validation_and_errors() -> None:
    with pytest.raises(DriftError, match="feature_edges must not be empty"):
        StreamingProfile({})

    with pytest.raises(DriftError, match="Feature names must be non-empty"):
        StreamingProfile({"": [1.0, 2.0]})

    with pytest.raises(DriftError, match="strictly increasing"):
        StreamingProfile({"x": [5.0, 2.0]})

    with pytest.raises(DriftError, match="Reference profile must not be empty"):
        StreamingProfile.from_reference_profile({})

    with pytest.raises(DriftError, match="Invalid edges in reference profile"):
        StreamingProfile.from_reference_profile({"x": {"edges": [None]}})

    sp = StreamingProfile({"x": [10.0, 20.0]})

    # Empty batch gracefully ignored
    sp.update(pd.DataFrame())
    sp.update([])

    with pytest.raises(DriftError, match="Batch must be a pandas DataFrame or list"):
        sp.update("not-a-batch")  # type: ignore[arg-type]

    with pytest.raises(DriftError, match="missing expected features"):
        sp.update(pd.DataFrame({"wrong": [1.0]}))

    with pytest.raises(DriftError, match="non-finite values"):
        sp.update(pd.DataFrame({"x": [np.nan]}))

    with pytest.raises(DriftError, match="missing 'features'"):
        StreamingProfile.from_dict({})

    with pytest.raises(DriftError, match="must be a non-empty mapping"):
        StreamingProfile.from_dict({"features": {}})


def test_streaming_profile_empty_to_reference_profile() -> None:
    sp = StreamingProfile({"feature_a": [10.0, 20.0]})
    ref = sp.to_reference_profile()
    assert "feature_a" in ref
    assert ref["feature_a"]["edges"] == [None, 10.0, 20.0, None]
    assert len(ref["feature_a"]["proportions"]) == 3
    assert pytest.approx(sum(ref["feature_a"]["proportions"])) == 1.0
    assert ref["feature_a"]["standard_deviation"] == 0.0
