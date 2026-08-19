from __future__ import annotations

import numpy as np
import pytest

from fraud_detection.evaluation import (
    evaluate_predictions,
    expected_classification_cost,
    select_cost_threshold,
    select_f1_threshold,
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
