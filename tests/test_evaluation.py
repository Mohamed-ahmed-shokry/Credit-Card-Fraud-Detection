from __future__ import annotations

import numpy as np
import pytest

from fraud_detection.evaluation import (
    calibration_report,
    evaluate_predictions,
    expected_classification_cost,
    select_cost_threshold,
    select_f1_threshold,
    summarize_thresholds,
)


def test_select_f1_threshold_finds_best_operating_point() -> None:
    y_true = np.array([0, 0, 0, 1, 1])
    probabilities = np.array([0.05, 0.2, 0.4, 0.45, 0.9])

    threshold = select_f1_threshold(y_true, probabilities)

    assert threshold == pytest.approx(0.45)


def test_select_f1_threshold_falls_back_when_curve_has_no_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def empty_curve(
        _y_true: np.ndarray,
        _probabilities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.array([]), np.array([]), np.array([])

    monkeypatch.setattr("fraud_detection.evaluation.precision_recall_curve", empty_curve)

    threshold = select_f1_threshold(np.array([0, 1]), np.array([0.1, 0.9]))

    assert threshold == 0.5


def test_evaluate_predictions_returns_complete_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.6, 0.9])

    metrics = evaluate_predictions(y_true, probabilities, threshold=0.5)

    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == 1.0
    assert metrics.f1 == pytest.approx(0.8)
    assert metrics.brier_score == pytest.approx(0.205)
    assert metrics.balanced_accuracy == pytest.approx(0.75)
    assert (
        metrics.true_negatives,
        metrics.false_positives,
        metrics.false_negatives,
        metrics.true_positives,
    ) == (1, 1, 0, 2)
    assert metrics.to_dict()["threshold"] == 0.5


def test_cost_threshold_minimizes_weighted_validation_mistakes() -> None:
    y_true = np.array([0, 0, 0, 1])
    probabilities = np.array([0.1, 0.4, 0.8, 0.7])

    threshold = select_cost_threshold(
        y_true,
        probabilities,
        false_positive_cost=1,
        false_negative_cost=10,
    )

    assert threshold == pytest.approx(0.7)
    assert expected_classification_cost(
        y_true,
        probabilities,
        threshold=threshold,
        false_positive_cost=1,
        false_negative_cost=10,
    ) == pytest.approx(0.25)


def test_cost_threshold_changes_with_business_costs() -> None:
    threshold = select_cost_threshold(
        np.array([0, 0, 0, 1]),
        np.array([0.1, 0.4, 0.8, 0.7]),
        false_positive_cost=10,
        false_negative_cost=1,
    )

    assert threshold == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("false_positive_cost", "false_negative_cost"),
    [(0, 1), (1, 0), (-1, 1), (1, np.inf)],
)
def test_cost_functions_reject_invalid_costs(
    false_positive_cost: float,
    false_negative_cost: float,
) -> None:
    with pytest.raises(ValueError, match="costs"):
        select_cost_threshold(
            np.array([0, 1]),
            np.array([0.1, 0.9]),
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_evaluate_predictions_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        evaluate_predictions(
            np.array([0, 1]),
            np.array([0.1, 0.9]),
            threshold=threshold,
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_expected_classification_cost_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        expected_classification_cost(
            np.array([0, 1]),
            np.array([0.1, 0.9]),
            threshold=threshold,
            false_positive_cost=1,
            false_negative_cost=1,
        )


def test_calibration_report_measures_reliability() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    report = calibration_report(y_true, probabilities, bins=2)

    assert report.rows == 8
    assert report.bins == 2
    assert len(report.detail) == 2
    assert [item.count for item in report.detail] == [4, 4]
    assert report.detail[0].fraction_positive == pytest.approx(0.0)
    assert report.detail[1].fraction_positive == pytest.approx(1.0)
    assert report.detail[0].mean_predicted == pytest.approx(0.25)
    assert report.detail[1].mean_predicted == pytest.approx(0.75)
    assert report.expected_calibration_error == pytest.approx(0.25)
    assert report.max_calibration_error == pytest.approx(0.25)
    assert report.to_dict()["rows"] == 8
    serialized_detail = report.to_dict()["detail"]
    assert isinstance(serialized_detail, list) and len(serialized_detail) == 2


def test_calibration_report_handles_empty_bins() -> None:
    report = calibration_report(np.array([0, 1]), np.array([0.05, 0.95]), bins=4)

    assert sum(item.count for item in report.detail) == 2
    assert report.expected_calibration_error >= 0.0
    assert report.brier_score == pytest.approx(
        report.reliability - report.resolution + report.uncertainty
    )


def test_calibration_report_matches_brier_decomposition() -> None:
    rng = np.random.default_rng(3)
    probabilities = rng.uniform(0.0, 1.0, size=200)
    y_true = (rng.uniform(0.0, 1.0, size=200) < probabilities).astype(int)
    if set(np.unique(y_true).tolist()) != {0, 1}:
        y_true[0], y_true[1] = 0, 1

    report = calibration_report(y_true, probabilities, bins=10)

    # Murphy's identity holds up to within-bin forecast variance, which
    # shrinks as bins narrow; 10 bins over 200 rows keeps it small.
    assert report.brier_score == pytest.approx(
        report.reliability - report.resolution + report.uncertainty, abs=0.01
    )
    assert 0.0 <= report.expected_calibration_error <= 1.0
    assert 0.0 <= report.max_calibration_error <= 1.0
    assert report.uncertainty == pytest.approx(0.25, abs=0.06)


def test_summarize_thresholds_reports_tradeoffs() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    tradeoff = summarize_thresholds(
        y_true,
        probabilities,
        [0.9, 0.5, 0.05],
        false_positive_cost=1,
        false_negative_cost=10,
    )

    assert tradeoff.rows == 8
    assert [row.threshold for row in tradeoff.detail] == [0.05, 0.5, 0.9]
    loose, middle, strict = tradeoff.detail
    assert loose.flagged == 8
    assert loose.recall == pytest.approx(1.0)
    assert strict.flagged == 1
    assert strict.true_positives == 1
    assert strict.false_positives == 0
    assert strict.precision == pytest.approx(1.0)
    assert middle.expected_cost_per_transaction == pytest.approx(0.0)
    assert middle.to_dict()["threshold"] == 0.5
    assert tradeoff.to_dict()["rows"] == 8


@pytest.mark.parametrize(
    ("thresholds", "message"),
    [
        ([0.5], "between 2 and 20"),
        ([0.1] * 21, "between 2 and 20"),
        ([-0.1, 0.5], "between 0 and 1"),
        ([0.5, 1.5], "between 0 and 1"),
        ([0.5, "high"], "only numbers"),
        ([0.5, True], "only numbers"),
    ],
)
def test_summarize_thresholds_rejects_invalid_candidates(
    thresholds: list[object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_thresholds(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            thresholds,  # type: ignore[arg-type]
            false_positive_cost=1,
            false_negative_cost=1,
        )


@pytest.mark.parametrize("bins", [1, 21])
def test_calibration_report_rejects_invalid_bin_count(bins: int) -> None:
    with pytest.raises(ValueError, match="bins"):
        calibration_report(np.array([0, 1]), np.array([0.2, 0.8]), bins=bins)


@pytest.mark.parametrize(
    ("y_true", "probabilities", "message"),
    [
        (np.array([[0, 1]]), np.array([0.1, 0.9]), "one-dimensional"),
        (np.array([0, 1]), np.array([[0.1, 0.9]]), "one-dimensional"),
        (np.array([0, 1]), np.array([0.1]), "same non-zero length"),
        (np.array([], dtype=int), np.array([]), "same non-zero length"),
        (np.array([0, 0]), np.array([0.1, 0.2]), "both binary labels"),
        (np.array([0, 1]), np.array([0.1, np.nan]), "finite"),
        (np.array([0, 1]), np.array([0.1, 1.2]), "between 0 and 1"),
    ],
)
def test_metric_functions_reject_invalid_vectors(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        select_f1_threshold(y_true, probabilities)
