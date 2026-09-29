from __future__ import annotations

import numpy as np
import pytest

from fraud_detection.evaluation import (
    DecisionAction,
    TieredMetrics,
    TieredThresholds,
    TieredThresholdTuningResult,
    TieredTuningObjective,
    assign_tiered_decisions,
    calibration_report,
    evaluate_predictions,
    evaluate_tiered_policy,
    expected_classification_cost,
    select_cost_threshold,
    select_f1_threshold,
    summarize_thresholds,
    tune_tiered_thresholds,
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


def test_tiered_thresholds_validation() -> None:
    t = TieredThresholds(review_threshold=0.3, deny_threshold=0.8)
    assert t.review_threshold == 0.3
    assert t.deny_threshold == 0.8
    assert t.to_dict() == {"review_threshold": 0.3, "deny_threshold": 0.8}

    # Equality is valid (binary policy boundary)
    t_eq = TieredThresholds(review_threshold=0.5, deny_threshold=0.5)
    assert t_eq.review_threshold == 0.5
    assert t_eq.deny_threshold == 0.5

    # Inverted thresholds
    with pytest.raises(ValueError, match=r"review_threshold .* cannot exceed deny_threshold"):
        TieredThresholds(review_threshold=0.8, deny_threshold=0.3)

    # Out of range
    with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
        TieredThresholds(review_threshold=-0.1, deny_threshold=0.5)
    with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
        TieredThresholds(review_threshold=0.2, deny_threshold=1.5)

    # Non-numeric / boolean
    with pytest.raises(TypeError, match="numeric"):
        TieredThresholds(review_threshold=True, deny_threshold=0.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="numeric"):
        TieredThresholds(review_threshold=0.2, deny_threshold=False)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real numbers"):
        TieredThresholds(review_threshold="bad", deny_threshold=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        TieredThresholds(review_threshold=float("nan"), deny_threshold=0.5)


def test_assign_tiered_decisions() -> None:
    probs = np.array([0.1, 0.3, 0.5, 0.8, 0.95])
    actions = assign_tiered_decisions(probs, review_threshold=0.3, deny_threshold=0.8)
    expected = np.array(
        [
            DecisionAction.ALLOW.value,
            DecisionAction.CHALLENGE.value,
            DecisionAction.CHALLENGE.value,
            DecisionAction.DENY.value,
            DecisionAction.DENY.value,
        ]
    )
    np.testing.assert_array_equal(actions, expected)

    # Empty or invalid vectors
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        assign_tiered_decisions(np.array([]), review_threshold=0.3, deny_threshold=0.8)
    with pytest.raises(ValueError, match="between 0 and 1"):
        assign_tiered_decisions(np.array([0.1, 1.5]), review_threshold=0.3, deny_threshold=0.8)


def test_evaluate_tiered_policy_metrics_and_costs() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1])
    probs = np.array([0.05, 0.2, 0.4, 0.85, 0.5, 0.9])
    # Tiers with review=0.3, deny=0.8:
    # ALLOW: idx 0 (y=0), idx 1 (y=0) -> allow_count=2, caught_fraud_allow=0, missed_fraud=0
    # CHALLENGE: idx 2 (y=0), idx 4 (y=1) -> review_count=2, caught_fraud_review=1, false_review=1
    # DENY: idx 3 (y=0), idx 5 (y=1) -> deny_count=2, caught_fraud_deny=1, false_deny=1

    metrics = evaluate_tiered_policy(
        y_true,
        probs,
        review_threshold=0.3,
        deny_threshold=0.8,
        manual_review_cost=10.0,
        false_deny_cost=60.0,
        missed_fraud_cost=300.0,
    )

    assert isinstance(metrics, TieredMetrics)
    assert metrics.rows == 6
    assert metrics.fraud_count == 2
    assert metrics.review_threshold == 0.3
    assert metrics.deny_threshold == 0.8
    assert metrics.allow_count == 2
    assert metrics.review_count == 2
    assert metrics.deny_count == 2
    assert metrics.allow_rate == pytest.approx(2 / 6)
    assert metrics.review_rate == pytest.approx(2 / 6)
    assert metrics.deny_rate == pytest.approx(2 / 6)
    assert metrics.caught_fraud_review == 1
    assert metrics.caught_fraud_deny == 1
    assert metrics.missed_fraud == 0
    assert metrics.review_precision == pytest.approx(0.5)
    assert metrics.deny_precision == pytest.approx(0.5)
    assert metrics.total_catch_rate == pytest.approx(1.0)

    # Cost: review_count (2) * 10 + false_deny (1) * 60 + missed_fraud (0) * 300 = 20 + 60 = 80
    # Expected cost per transaction: 80 / 6 = 13.333...
    assert metrics.expected_cost_per_transaction == pytest.approx(80.0 / 6.0)

    d = metrics.to_dict()
    assert d["rows"] == 6
    assert d["deny_count"] == 2


def test_evaluate_tiered_policy_validation() -> None:
    y_true = np.array([0, 1])
    probs = np.array([0.2, 0.8])

    with pytest.raises(ValueError, match="manual_review_cost must be a non-negative"):
        evaluate_tiered_policy(
            y_true,
            probs,
            review_threshold=0.3,
            deny_threshold=0.7,
            manual_review_cost=-1.0,
        )
    with pytest.raises(ValueError, match="false_deny_cost must be a non-negative"):
        evaluate_tiered_policy(
            y_true,
            probs,
            review_threshold=0.3,
            deny_threshold=0.7,
            false_deny_cost=float("inf"),
        )
    with pytest.raises(ValueError, match="missed_fraud_cost must be a non-negative"):
        evaluate_tiered_policy(
            y_true,
            probs,
            review_threshold=0.3,
            deny_threshold=0.7,
            missed_fraud_cost=True,  # type: ignore[arg-type]
        )


def test_tune_tiered_thresholds_cost_minimization() -> None:
    rng = np.random.default_rng(42)
    y_true = np.concatenate([np.zeros(900, dtype=int), np.ones(100, dtype=int)])
    # Non-frauds low scores, frauds higher scores
    probs_legit = rng.beta(0.5, 20.0, size=900)
    probs_fraud = rng.beta(5.0, 2.0, size=100)
    probs = np.concatenate([probs_legit, probs_fraud])

    result = tune_tiered_thresholds(
        y_true,
        probs,
        objective=TieredTuningObjective.COST_MINIMIZATION,
        manual_review_cost=5.0,
        false_deny_cost=50.0,
        missed_fraud_cost=200.0,
        steps=30,
    )

    assert isinstance(result, TieredThresholdTuningResult)
    assert result.objective == "cost_minimization"
    assert result.constraints_satisfied is True
    assert result.candidate_evaluations > 0
    assert (
        0.0
        <= result.best_thresholds.review_threshold
        <= result.best_thresholds.deny_threshold
        <= 1.0
    )
    assert result.best_metrics.expected_cost_per_transaction >= 0.0
    assert result.best_metrics.total_catch_rate > 0.5


def test_tune_tiered_thresholds_capacity_constrained() -> None:
    rng = np.random.default_rng(99)
    y_true = np.concatenate([np.zeros(900, dtype=int), np.ones(100, dtype=int)])
    probs_legit = rng.beta(0.5, 20.0, size=900)
    probs_fraud = rng.beta(4.0, 2.0, size=100)
    probs = np.concatenate([probs_legit, probs_fraud])

    max_rev = 0.08
    result = tune_tiered_thresholds(
        y_true,
        probs,
        objective="capacity_constrained",
        max_review_rate=max_rev,
        steps=35,
    )

    assert result.objective == "capacity_constrained"
    assert result.constraints_satisfied is True
    assert result.best_metrics.review_rate <= max_rev + 1e-6
    assert result.best_thresholds.review_threshold <= result.best_thresholds.deny_threshold


def test_tune_tiered_thresholds_with_min_deny_precision() -> None:
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    probs = np.array([0.02, 0.05, 0.10, 0.20, 0.85, 0.30, 0.60, 0.80, 0.90, 0.95])

    result = tune_tiered_thresholds(
        y_true,
        probs,
        objective="cost_minimization",
        min_deny_precision=0.75,
        steps=20,
    )

    assert result.constraints_satisfied is True
    if result.best_metrics.deny_count > 0:
        assert result.best_metrics.deny_precision >= 0.75 - 1e-6


def test_tune_tiered_thresholds_unsatisfied_constraints() -> None:
    y_true = np.array([0, 1])
    probs = np.array([0.9, 0.1])

    # Impossible constraint: inverted scores where max achievable precision is 0.5
    result = tune_tiered_thresholds(
        y_true,
        probs,
        objective="capacity_constrained",
        max_review_rate=0.5,
        min_deny_precision=0.95,
        steps=10,
    )

    assert isinstance(result, TieredThresholdTuningResult)
    # Result gracefully falls back to best candidate without crashing
    assert result.constraints_satisfied is False
    assert result.best_thresholds.review_threshold <= result.best_thresholds.deny_threshold


def test_tune_tiered_thresholds_validation_errors() -> None:
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])

    with pytest.raises(ValueError, match="Invalid objective"):
        tune_tiered_thresholds(y_true, probs, objective="unsupported_objective")

    with pytest.raises(
        ValueError, match="max_review_rate is required when objective is 'capacity_constrained'"
    ):
        tune_tiered_thresholds(
            y_true, probs, objective="capacity_constrained", max_review_rate=None
        )

    with pytest.raises(ValueError, match=r"max_review_rate must be a float between 0\.0 and 1\.0"):
        tune_tiered_thresholds(y_true, probs, max_review_rate=-0.1)

    with pytest.raises(ValueError, match=r"max_review_rate must be a float between 0\.0 and 1\.0"):
        tune_tiered_thresholds(y_true, probs, max_review_rate=True)  # type: ignore[arg-type]

    with pytest.raises(
        ValueError, match=r"min_deny_precision must be a float between 0\.0 and 1\.0"
    ):
        tune_tiered_thresholds(y_true, probs, min_deny_precision=1.5)

    with pytest.raises(
        ValueError, match=r"min_deny_precision must be a float between 0\.0 and 1\.0"
    ):
        tune_tiered_thresholds(y_true, probs, min_deny_precision=False)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="steps must be an integer >= 5"):
        tune_tiered_thresholds(y_true, probs, steps=3)

    with pytest.raises(ValueError, match="steps must be an integer >= 5"):
        tune_tiered_thresholds(y_true, probs, steps=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="manual_review_cost must be a non-negative finite number"):
        tune_tiered_thresholds(y_true, probs, manual_review_cost=-1.0)


def test_tune_tiered_thresholds_to_dict() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.05, 0.1, 0.2, 0.7, 0.8, 0.9])

    result = tune_tiered_thresholds(y_true, probs, steps=15)
    d = result.to_dict()
    assert "best_thresholds" in d
    assert "best_metrics" in d
    assert d["objective"] == "cost_minimization"
    assert "candidate_evaluations" in d
    assert "constraints_satisfied" in d
    assert d["best_thresholds"]["review_threshold"] <= d["best_thresholds"]["deny_threshold"]
