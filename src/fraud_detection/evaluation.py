"""Threshold selection and metrics for imbalanced fraud classification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class CalibrationBin:
    """One equal-width reliability bin of a calibration report."""

    bin_index: int
    lower: float
    upper: float
    count: int
    mean_predicted: float
    fraction_positive: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-compatible bin mapping."""
        return asdict(self)


@dataclass(frozen=True)
class CalibrationReport:
    """Serializable reliability analysis for predicted probabilities."""

    rows: int
    bins: int
    brier_score: float
    expected_calibration_error: float
    max_calibration_error: float
    reliability: float
    resolution: float
    uncertainty: float
    detail: tuple[CalibrationBin, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report mapping."""
        return {
            "rows": self.rows,
            "bins": self.bins,
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "max_calibration_error": self.max_calibration_error,
            "reliability": self.reliability,
            "resolution": self.resolution,
            "uncertainty": self.uncertainty,
            "detail": [item.to_dict() for item in self.detail],
        }


@dataclass(frozen=True)
class ThresholdRow:
    """Decision metrics at one candidate threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    expected_cost_per_transaction: float
    flagged: int
    flagged_rate: float
    true_positives: int
    false_positives: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-compatible row mapping."""
        return asdict(self)


@dataclass(frozen=True)
class ThresholdTradeoff:
    """Per-threshold decision metrics over shared labeled data."""

    rows: int
    detail: tuple[ThresholdRow, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report mapping."""
        return {
            "rows": self.rows,
            "detail": [item.to_dict() for item in self.detail],
        }


@dataclass(frozen=True)
class ClassificationMetrics:
    """Serializable binary-classification evaluation results."""

    threshold: float
    roc_auc: float
    average_precision: float
    brier_score: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-compatible metrics mapping."""
        return asdict(self)


class DecisionAction(StrEnum):
    """Action outcome of a tiered decision policy."""

    ALLOW = "ALLOW"
    CHALLENGE = "CHALLENGE"
    DENY = "DENY"


@dataclass(frozen=True)
class TieredThresholds:
    """Decision boundaries for three-tier risk routing."""

    review_threshold: float
    deny_threshold: float

    def __post_init__(self) -> None:
        if isinstance(self.review_threshold, bool) or isinstance(self.deny_threshold, bool):
            raise TypeError("Thresholds must be numeric, not boolean.")
        try:
            r = float(self.review_threshold)
            d = float(self.deny_threshold)
        except (TypeError, ValueError) as exc:
            raise TypeError("Thresholds must be real numbers.") from exc
        if not (np.isfinite(r) and np.isfinite(d)):
            raise ValueError("Thresholds must be finite.")
        if not (0.0 <= r <= 1.0) or not (0.0 <= d <= 1.0):
            raise ValueError("Thresholds must be between 0.0 and 1.0.")
        if r > d:
            raise ValueError(f"review_threshold ({r}) cannot exceed deny_threshold ({d}).")

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-compatible thresholds mapping."""
        return {
            "review_threshold": float(self.review_threshold),
            "deny_threshold": float(self.deny_threshold),
        }


@dataclass(frozen=True)
class TieredMetrics:
    """Performance, workload, and operational costs for a three-tier decision policy."""

    rows: int
    fraud_count: int
    review_threshold: float
    deny_threshold: float
    allow_count: int
    review_count: int
    deny_count: int
    allow_rate: float
    review_rate: float
    deny_rate: float
    caught_fraud_review: int
    caught_fraud_deny: int
    missed_fraud: int
    review_precision: float
    deny_precision: float
    total_catch_rate: float
    expected_cost_per_transaction: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible metrics mapping."""
        return asdict(self)


class TieredTuningObjective(StrEnum):
    """Optimization objective for tiered threshold tuning."""

    COST_MINIMIZATION = "cost_minimization"
    CAPACITY_CONSTRAINED = "capacity_constrained"


@dataclass(frozen=True)
class TieredThresholdTuningResult:
    """Outcome of automated decision tier threshold optimization."""

    best_thresholds: TieredThresholds
    best_metrics: TieredMetrics
    objective: str
    candidate_evaluations: int
    constraints_satisfied: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible tuning result mapping."""
        return {
            "best_thresholds": self.best_thresholds.to_dict(),
            "best_metrics": self.best_metrics.to_dict(),
            "objective": self.objective,
            "candidate_evaluations": self.candidate_evaluations,
            "constraints_satisfied": self.constraints_satisfied,
        }


@dataclass(frozen=True)
class PolicyTransitionMatrix:
    """Breakdown of decision shifts between baseline and candidate policies."""

    allow_to_allow: int
    allow_to_challenge: int
    allow_to_deny: int
    challenge_to_allow: int
    challenge_to_challenge: int
    challenge_to_deny: int
    deny_to_allow: int
    deny_to_challenge: int
    deny_to_deny: int
    total_records: int
    turnover_count: int
    turnover_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible transition matrix mapping."""
        return asdict(self)


@dataclass(frozen=True)
class PolicyBacktestReport:
    """Comparative backtest analysis between baseline and candidate policies."""

    rows: int
    baseline_thresholds: TieredThresholds
    candidate_thresholds: TieredThresholds
    transition_matrix: PolicyTransitionMatrix
    baseline_action_counts: dict[str, int]
    candidate_action_counts: dict[str, int]
    review_count_delta: int
    review_rate_delta: float
    deny_count_delta: int
    deny_rate_delta: float
    baseline_metrics: TieredMetrics | None = None
    candidate_metrics: TieredMetrics | None = None
    cost_delta: float | None = None
    fraud_catch_delta: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible backtest report mapping."""
        return {
            "rows": self.rows,
            "baseline_thresholds": self.baseline_thresholds.to_dict(),
            "candidate_thresholds": self.candidate_thresholds.to_dict(),
            "transition_matrix": self.transition_matrix.to_dict(),
            "baseline_action_counts": dict(self.baseline_action_counts),
            "candidate_action_counts": dict(self.candidate_action_counts),
            "review_count_delta": self.review_count_delta,
            "review_rate_delta": self.review_rate_delta,
            "deny_count_delta": self.deny_count_delta,
            "deny_rate_delta": self.deny_rate_delta,
            "baseline_metrics": self.baseline_metrics.to_dict() if self.baseline_metrics else None,
            "candidate_metrics": (
                self.candidate_metrics.to_dict() if self.candidate_metrics else None
            ),
            "cost_delta": self.cost_delta,
            "fraud_catch_delta": self.fraud_catch_delta,
        }


@dataclass(frozen=True)
class SliceMetricRow:
    """Performance and disparity metrics for a single data sub-population."""

    slice_name: str
    count: int
    percentage: float
    fraud_count: int
    fraud_rate: float
    flagged_count: int
    flagged_rate: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    recall_disparity: float
    fpr_disparity: float
    fraud_rate_disparity: float
    is_underperforming: bool
    underperformance_reasons: tuple[str, ...]
    allow_count: int | None = None
    review_count: int | None = None
    deny_count: int | None = None
    allow_rate: float | None = None
    review_rate: float | None = None
    deny_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible row mapping."""
        return {
            "slice_name": self.slice_name,
            "count": self.count,
            "percentage": self.percentage,
            "fraud_count": self.fraud_count,
            "fraud_rate": self.fraud_rate,
            "flagged_count": self.flagged_count,
            "flagged_rate": self.flagged_rate,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
            "recall_disparity": self.recall_disparity,
            "fpr_disparity": self.fpr_disparity,
            "fraud_rate_disparity": self.fraud_rate_disparity,
            "is_underperforming": self.is_underperforming,
            "underperformance_reasons": list(self.underperformance_reasons),
            "allow_count": self.allow_count,
            "review_count": self.review_count,
            "deny_count": self.deny_count,
            "allow_rate": self.allow_rate,
            "review_rate": self.review_rate,
            "deny_rate": self.deny_rate,
        }


@dataclass(frozen=True)
class SliceDisparityReport:
    """Disparity analysis comparing sub-population performance to global baseline."""

    total_records: int
    global_fraud_rate: float
    global_precision: float
    global_recall: float
    global_false_positive_rate: float
    min_recall_disparity: float
    max_fpr_disparity: float
    underperforming_slices: tuple[str, ...]
    slices: tuple[SliceMetricRow, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report mapping."""
        return {
            "total_records": self.total_records,
            "global_fraud_rate": self.global_fraud_rate,
            "global_precision": self.global_precision,
            "global_recall": self.global_recall,
            "global_false_positive_rate": self.global_false_positive_rate,
            "min_recall_disparity": self.min_recall_disparity,
            "max_fpr_disparity": self.max_fpr_disparity,
            "underperforming_slices": list(self.underperforming_slices),
            "slices": [s.to_dict() for s in self.slices],
        }


def select_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose the probability threshold that maximizes F1.

    Call this on validation data only. Test data must remain untouched until the
    threshold and model are finalized.
    """
    _validate_vectors(y_true, probabilities)
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5

    denominator = precision[:-1] + recall[:-1]
    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(f1_scores))])


def select_cost_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    false_positive_cost: float,
    false_negative_cost: float,
) -> float:
    """Choose the validation threshold with the lowest expected mistake cost.

    Costs are relative business weights. Ties favor higher recall so equally
    costly policies miss fewer fraudulent transactions.
    """
    _validate_vectors(y_true, probabilities)
    _validate_costs(false_positive_cost, false_negative_cost)

    false_positive_rates, true_positive_rates, thresholds = roc_curve(
        y_true,
        probabilities,
        drop_intermediate=False,
    )
    negative_count = int(np.sum(y_true == 0))
    positive_count = int(np.sum(y_true == 1))
    false_positives = false_positive_rates * negative_count
    false_negatives = (1.0 - true_positive_rates) * positive_count
    costs = (
        false_positives * false_positive_cost + false_negatives * false_negative_cost
    ) / y_true.size

    finite = np.isfinite(thresholds) & (thresholds <= 1.0)
    candidate_indices = np.flatnonzero(finite)
    candidate_costs = costs[candidate_indices]
    minimum_cost = float(np.min(candidate_costs))
    all_negative_cost = positive_count * false_negative_cost / y_true.size
    if probabilities.max() < 1.0 and all_negative_cost < minimum_cost:
        return 1.0
    tied_indices = candidate_indices[np.isclose(candidate_costs, minimum_cost)]
    best_index = tied_indices[int(np.argmax(true_positive_rates[tied_indices]))]
    return float(thresholds[best_index])


def expected_classification_cost(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    false_positive_cost: float,
    false_negative_cost: float,
) -> float:
    """Return average weighted mistake cost per transaction."""
    _validate_vectors(y_true, probabilities)
    _validate_costs(false_positive_cost, false_negative_cost)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    predictions = probabilities >= threshold
    false_positives = int(np.sum((predictions == 1) & (y_true == 0)))
    false_negatives = int(np.sum((predictions == 0) & (y_true == 1)))
    return float(
        (false_positives * false_positive_cost + false_negatives * false_negative_cost)
        / y_true.size
    )


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> ClassificationMetrics:
    """Evaluate probabilities at a fixed decision threshold."""
    _validate_vectors(y_true, probabilities)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    predictions = (probabilities >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return ClassificationMetrics(
        threshold=float(threshold),
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        average_precision=float(average_precision_score(y_true, probabilities)),
        brier_score=float(brier_score_loss(y_true, probabilities)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, predictions)),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )


def calibration_report(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> CalibrationReport:
    """Summarize reliability with equal-width bins plus a Brier decomposition.

    Reports the reliability curve (per-bin mean predicted probability against
    observed fraud fraction), the expected and maximum calibration errors, and
    Murphy's Brier-score decomposition (reliability, resolution, uncertainty)
    computed on the same bins. Lower Brier score, calibration errors, and
    reliability are better; higher resolution is better.
    """
    _validate_vectors(y_true, probabilities)
    if not 2 <= bins <= 20:
        raise ValueError("bins must be between 2 and 20")

    outcomes = y_true.astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, bins - 1)
    base_rate = float(np.mean(outcomes))
    detail: list[CalibrationBin] = []
    gaps: list[float] = []
    weights: list[float] = []
    for index in range(bins):
        mask = assignments == index
        count = int(np.sum(mask))
        if count == 0:
            detail.append(
                CalibrationBin(
                    bin_index=index,
                    lower=float(edges[index]),
                    upper=float(edges[index + 1]),
                    count=0,
                    mean_predicted=float((edges[index] + edges[index + 1]) / 2),
                    fraction_positive=0.0,
                )
            )
            gaps.append(0.0)
            weights.append(0.0)
            continue
        mean_predicted = float(np.mean(probabilities[mask]))
        fraction_positive = float(np.mean(outcomes[mask]))
        detail.append(
            CalibrationBin(
                bin_index=index,
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                count=count,
                mean_predicted=mean_predicted,
                fraction_positive=fraction_positive,
            )
        )
        gaps.append(abs(fraction_positive - mean_predicted))
        weights.append(count / y_true.size)

    weights_array = np.asarray(weights, dtype=float)
    gaps_array = np.asarray(gaps, dtype=float)
    fractions = np.asarray([item.fraction_positive for item in detail], dtype=float)
    reliability = float(np.sum(weights_array * gaps_array**2))
    resolution = float(np.sum(weights_array * (fractions - base_rate) ** 2))
    uncertainty = float(base_rate * (1.0 - base_rate))
    return CalibrationReport(
        rows=int(y_true.size),
        bins=bins,
        brier_score=float(brier_score_loss(y_true, probabilities)),
        expected_calibration_error=float(np.sum(weights_array * gaps_array)),
        max_calibration_error=float(np.max(gaps_array)),
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        detail=tuple(detail),
    )


def summarize_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: list[float],
    *,
    false_positive_cost: float,
    false_negative_cost: float,
) -> ThresholdTradeoff:
    """Score candidate thresholds on shared labeled data.

    Reports precision, recall, F1, and weighted expected cost per threshold so
    the operating point is chosen with eyes open. Use validation (or other
    held-out labeled) data: the tuned model threshold stays fixed while the
    candidates are explored.
    """
    _validate_vectors(y_true, probabilities)
    _validate_costs(false_positive_cost, false_negative_cost)
    if not 2 <= len(thresholds) <= 20:
        raise ValueError("thresholds must contain between 2 and 20 candidates")
    for threshold in thresholds:
        if isinstance(threshold, bool) or not isinstance(threshold, int | float):
            raise ValueError("thresholds must contain only numbers")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("thresholds must all fall between 0 and 1")

    detail = []
    for threshold in sorted(thresholds):
        predictions = (probabilities >= threshold).astype("int8")
        _, fp, _, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        detail.append(
            ThresholdRow(
                threshold=float(threshold),
                precision=float(precision_score(y_true, predictions, zero_division=0)),
                recall=float(recall_score(y_true, predictions, zero_division=0)),
                f1=float(f1_score(y_true, predictions, zero_division=0)),
                expected_cost_per_transaction=expected_classification_cost(
                    y_true,
                    probabilities,
                    threshold=float(threshold),
                    false_positive_cost=false_positive_cost,
                    false_negative_cost=false_negative_cost,
                ),
                flagged=int(fp + tp),
                flagged_rate=float((fp + tp) / y_true.size),
                true_positives=int(tp),
                false_positives=int(fp),
            )
        )
    return ThresholdTradeoff(rows=int(y_true.size), detail=tuple(detail))


def assign_tiered_decisions(
    probabilities: np.ndarray,
    *,
    review_threshold: float,
    deny_threshold: float,
) -> np.ndarray:
    """Return an array of DecisionAction strings ('ALLOW', 'CHALLENGE', 'DENY').

    ALLOW: probability < review_threshold
    CHALLENGE: review_threshold <= probability < deny_threshold
    DENY: probability >= deny_threshold
    """
    tiered = TieredThresholds(review_threshold=review_threshold, deny_threshold=deny_threshold)
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array")
    if not np.isfinite(probs).all() or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must be finite numbers between 0 and 1")

    actions = np.full(probs.shape, DecisionAction.ALLOW.value, dtype=object)
    actions[probs >= tiered.review_threshold] = DecisionAction.CHALLENGE.value
    actions[probs >= tiered.deny_threshold] = DecisionAction.DENY.value
    return actions


def evaluate_tiered_policy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    review_threshold: float,
    deny_threshold: float,
    manual_review_cost: float = 5.0,
    false_deny_cost: float = 50.0,
    missed_fraud_cost: float = 200.0,
) -> TieredMetrics:
    """Evaluate operational workload, precisions, and costs for a three-tier policy.

    ALLOW: probability < review_threshold
    CHALLENGE: review_threshold <= probability < deny_threshold
    DENY: probability >= deny_threshold
    """
    _validate_vectors(y_true, probabilities)
    tiered = TieredThresholds(review_threshold=review_threshold, deny_threshold=deny_threshold)
    r_thresh = float(tiered.review_threshold)
    d_thresh = float(tiered.deny_threshold)

    for cost_name, cost_val in [
        ("manual_review_cost", manual_review_cost),
        ("false_deny_cost", false_deny_cost),
        ("missed_fraud_cost", missed_fraud_cost),
    ]:
        if isinstance(cost_val, bool) or not np.isfinite(float(cost_val)) or float(cost_val) < 0.0:
            raise ValueError(f"{cost_name} must be a non-negative finite number")

    rows = int(y_true.size)
    frauds = int(np.sum(y_true == 1))

    is_deny = probabilities >= d_thresh
    is_review = (probabilities >= r_thresh) & (~is_deny)
    is_allow = probabilities < r_thresh

    deny_count = int(np.sum(is_deny))
    review_count = int(np.sum(is_review))
    allow_count = int(np.sum(is_allow))

    caught_fraud_deny = int(np.sum(is_deny & (y_true == 1)))
    caught_fraud_review = int(np.sum(is_review & (y_true == 1)))
    missed_fraud = int(np.sum(is_allow & (y_true == 1)))

    false_deny = int(np.sum(is_deny & (y_true == 0)))

    allow_rate = float(allow_count / rows) if rows > 0 else 0.0
    review_rate = float(review_count / rows) if rows > 0 else 0.0
    deny_rate = float(deny_count / rows) if rows > 0 else 0.0

    deny_precision = float(caught_fraud_deny / deny_count) if deny_count > 0 else 0.0
    review_precision = float(caught_fraud_review / review_count) if review_count > 0 else 0.0
    total_caught = caught_fraud_deny + caught_fraud_review
    total_catch_rate = float(total_caught / frauds) if frauds > 0 else 1.0

    total_cost = (
        review_count * float(manual_review_cost)
        + false_deny * float(false_deny_cost)
        + missed_fraud * float(missed_fraud_cost)
    )
    expected_cost_per_tx = float(total_cost / rows) if rows > 0 else 0.0

    return TieredMetrics(
        rows=rows,
        fraud_count=frauds,
        review_threshold=r_thresh,
        deny_threshold=d_thresh,
        allow_count=allow_count,
        review_count=review_count,
        deny_count=deny_count,
        allow_rate=allow_rate,
        review_rate=review_rate,
        deny_rate=deny_rate,
        caught_fraud_review=caught_fraud_review,
        caught_fraud_deny=caught_fraud_deny,
        missed_fraud=missed_fraud,
        review_precision=review_precision,
        deny_precision=deny_precision,
        total_catch_rate=total_catch_rate,
        expected_cost_per_transaction=expected_cost_per_tx,
    )


def tune_tiered_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    objective: str | TieredTuningObjective = TieredTuningObjective.COST_MINIMIZATION,
    manual_review_cost: float = 5.0,
    false_deny_cost: float = 50.0,
    missed_fraud_cost: float = 200.0,
    max_review_rate: float | None = None,
    min_deny_precision: float | None = None,
    steps: int = 50,
) -> TieredThresholdTuningResult:
    """Optimize three-tier decision boundaries on validation data.

    Evaluates candidate threshold pairs (r, d) with r <= d under either
    cost-minimization or review-capacity-constrained objectives.
    """
    _validate_vectors(y_true, probabilities)

    for cost_name, cost_val in [
        ("manual_review_cost", manual_review_cost),
        ("false_deny_cost", false_deny_cost),
        ("missed_fraud_cost", missed_fraud_cost),
    ]:
        if isinstance(cost_val, bool) or not np.isfinite(float(cost_val)) or float(cost_val) < 0.0:
            raise ValueError(f"{cost_name} must be a non-negative finite number")

    if max_review_rate is not None and (
        isinstance(max_review_rate, bool)
        or not np.isfinite(float(max_review_rate))
        or not 0.0 <= float(max_review_rate) <= 1.0
    ):
        raise ValueError("max_review_rate must be a float between 0.0 and 1.0")

    if min_deny_precision is not None and (
        isinstance(min_deny_precision, bool)
        or not np.isfinite(float(min_deny_precision))
        or not 0.0 <= float(min_deny_precision) <= 1.0
    ):
        raise ValueError("min_deny_precision must be a float between 0.0 and 1.0")

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 5:
        raise ValueError("steps must be an integer >= 5")

    try:
        norm_obj = (
            objective
            if isinstance(objective, TieredTuningObjective)
            else TieredTuningObjective(str(objective).lower())
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid objective {objective!r}: "
            "must be 'cost_minimization' or 'capacity_constrained'"
        ) from exc

    if norm_obj is TieredTuningObjective.CAPACITY_CONSTRAINED and max_review_rate is None:
        raise ValueError("max_review_rate is required when objective is 'capacity_constrained'")

    # Generate candidate cut points
    grid = np.linspace(0.005, 0.995, num=steps)
    quantiles = np.quantile(probabilities, np.linspace(0.01, 0.99, num=steps))
    raw_cuts = np.unique(np.clip(np.concatenate([grid, quantiles]), 0.001, 0.999))
    cuts = np.sort(raw_cuts)

    order = np.argsort(probabilities, kind="stable")
    p_sorted = probabilities[order]
    y_sorted = y_true[order]
    n_total = len(y_sorted)
    total_frauds = int(np.sum(y_sorted == 1))
    cum_frauds = np.cumsum(y_sorted == 1)

    r_cost = float(manual_review_cost)
    fd_cost = float(false_deny_cost)
    mf_cost = float(missed_fraud_cost)
    max_rev = float(max_review_rate) if max_review_rate is not None else None
    min_prec = float(min_deny_precision) if min_deny_precision is not None else None

    best_key: tuple[int, float, float, float, float] | None = None
    best_r = float(cuts[0])
    best_d = float(cuts[-1])
    best_satisfied = False
    candidate_evals = 0

    m_cuts = len(cuts)
    for i in range(m_cuts):
        r_val = float(cuts[i])
        idx_r = int(np.searchsorted(p_sorted, r_val, side="left"))
        missed = int(cum_frauds[idx_r - 1]) if idx_r > 0 else 0

        for j in range(i, m_cuts):
            d_val = float(cuts[j])
            candidate_evals += 1
            idx_d = int(np.searchsorted(p_sorted, d_val, side="left"))

            caught_deny = int(total_frauds - (cum_frauds[idx_d - 1] if idx_d > 0 else 0))
            caught_rev = total_frauds - caught_deny - missed

            deny_count = n_total - idx_d
            rev_count = idx_d - idx_r
            false_deny = deny_count - caught_deny

            rev_rate = rev_count / n_total
            deny_prec = (caught_deny / deny_count) if deny_count > 0 else 0.0
            catch_rate = (caught_deny + caught_rev) / total_frauds if total_frauds > 0 else 1.0

            cost = (rev_count * r_cost + false_deny * fd_cost + missed * mf_cost) / n_total

            penalty = 0.0
            satisfied = True
            if max_rev is not None and rev_rate > max_rev + 1e-9:
                penalty += rev_rate - max_rev
                satisfied = False
            if min_prec is not None and (deny_count == 0 or deny_prec < min_prec - 1e-9):
                penalty += (min_prec - deny_prec) if deny_count > 0 else min_prec
                satisfied = False

            if norm_obj is TieredTuningObjective.COST_MINIMIZATION:
                key = (0 if satisfied else 1, penalty, cost, -catch_rate, r_val - d_val)
            else:
                key = (0 if satisfied else 1, penalty, -catch_rate, cost, r_val - d_val)

            if best_key is None or key < best_key:
                best_key = key
                best_r = r_val
                best_d = d_val
                best_satisfied = satisfied

    best_metrics = evaluate_tiered_policy(
        y_true,
        probabilities,
        review_threshold=best_r,
        deny_threshold=best_d,
        manual_review_cost=manual_review_cost,
        false_deny_cost=false_deny_cost,
        missed_fraud_cost=missed_fraud_cost,
    )

    return TieredThresholdTuningResult(
        best_thresholds=TieredThresholds(
            review_threshold=best_metrics.review_threshold,
            deny_threshold=best_metrics.deny_threshold,
        ),
        best_metrics=best_metrics,
        objective=norm_obj.value,
        candidate_evaluations=candidate_evals,
        constraints_satisfied=best_satisfied,
    )


def backtest_policy_transition(
    probabilities: np.ndarray,
    baseline_thresholds: TieredThresholds,
    candidate_thresholds: TieredThresholds,
    y_true: np.ndarray | None = None,
    *,
    manual_review_cost: float = 5.0,
    false_deny_cost: float = 50.0,
    missed_fraud_cost: float = 200.0,
) -> PolicyBacktestReport:
    """Simulate operational impact and decision migrations between two policies."""
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array")
    if not np.isfinite(probs).all() or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must be finite numbers between 0 and 1")

    if not isinstance(baseline_thresholds, TieredThresholds):
        raise TypeError("baseline_thresholds must be a TieredThresholds instance")
    if not isinstance(candidate_thresholds, TieredThresholds):
        raise TypeError("candidate_thresholds must be a TieredThresholds instance")

    for cost_name, cost_val in [
        ("manual_review_cost", manual_review_cost),
        ("false_deny_cost", false_deny_cost),
        ("missed_fraud_cost", missed_fraud_cost),
    ]:
        if isinstance(cost_val, bool) or not np.isfinite(float(cost_val)) or float(cost_val) < 0.0:
            raise ValueError(f"{cost_name} must be a non-negative finite number")

    if y_true is not None:
        _validate_vectors(y_true, probs)

    b_actions = assign_tiered_decisions(
        probs,
        review_threshold=baseline_thresholds.review_threshold,
        deny_threshold=baseline_thresholds.deny_threshold,
    )
    c_actions = assign_tiered_decisions(
        probs,
        review_threshold=candidate_thresholds.review_threshold,
        deny_threshold=candidate_thresholds.deny_threshold,
    )

    n_rows = len(probs)
    b_allow = b_actions == DecisionAction.ALLOW.value
    b_review = b_actions == DecisionAction.CHALLENGE.value
    b_deny = b_actions == DecisionAction.DENY.value

    c_allow = c_actions == DecisionAction.ALLOW.value
    c_review = c_actions == DecisionAction.CHALLENGE.value
    c_deny = c_actions == DecisionAction.DENY.value

    turnover = int(np.sum(b_actions != c_actions))
    turnover_rate = float(turnover / n_rows) if n_rows > 0 else 0.0

    matrix = PolicyTransitionMatrix(
        allow_to_allow=int(np.sum(b_allow & c_allow)),
        allow_to_challenge=int(np.sum(b_allow & c_review)),
        allow_to_deny=int(np.sum(b_allow & c_deny)),
        challenge_to_allow=int(np.sum(b_review & c_allow)),
        challenge_to_challenge=int(np.sum(b_review & c_review)),
        challenge_to_deny=int(np.sum(b_review & c_deny)),
        deny_to_allow=int(np.sum(b_deny & c_allow)),
        deny_to_challenge=int(np.sum(b_deny & c_review)),
        deny_to_deny=int(np.sum(b_deny & c_deny)),
        total_records=n_rows,
        turnover_count=turnover,
        turnover_rate=turnover_rate,
    )

    b_counts = {
        DecisionAction.ALLOW.value: int(np.sum(b_allow)),
        DecisionAction.CHALLENGE.value: int(np.sum(b_review)),
        DecisionAction.DENY.value: int(np.sum(b_deny)),
    }
    c_counts = {
        DecisionAction.ALLOW.value: int(np.sum(c_allow)),
        DecisionAction.CHALLENGE.value: int(np.sum(c_review)),
        DecisionAction.DENY.value: int(np.sum(c_deny)),
    }

    review_delta = (
        c_counts[DecisionAction.CHALLENGE.value] - b_counts[DecisionAction.CHALLENGE.value]
    )
    review_rate_delta = float(review_delta / n_rows) if n_rows > 0 else 0.0
    deny_delta = c_counts[DecisionAction.DENY.value] - b_counts[DecisionAction.DENY.value]
    deny_rate_delta = float(deny_delta / n_rows) if n_rows > 0 else 0.0

    b_metrics = None
    c_metrics = None
    cost_delta = None
    fraud_catch_delta = None

    if y_true is not None:
        b_metrics = evaluate_tiered_policy(
            y_true,
            probs,
            review_threshold=baseline_thresholds.review_threshold,
            deny_threshold=baseline_thresholds.deny_threshold,
            manual_review_cost=manual_review_cost,
            false_deny_cost=false_deny_cost,
            missed_fraud_cost=missed_fraud_cost,
        )
        c_metrics = evaluate_tiered_policy(
            y_true,
            probs,
            review_threshold=candidate_thresholds.review_threshold,
            deny_threshold=candidate_thresholds.deny_threshold,
            manual_review_cost=manual_review_cost,
            false_deny_cost=false_deny_cost,
            missed_fraud_cost=missed_fraud_cost,
        )
        cost_delta = float(
            c_metrics.expected_cost_per_transaction - b_metrics.expected_cost_per_transaction
        )
        b_caught = b_metrics.caught_fraud_review + b_metrics.caught_fraud_deny
        c_caught = c_metrics.caught_fraud_review + c_metrics.caught_fraud_deny
        fraud_catch_delta = int(c_caught - b_caught)

    return PolicyBacktestReport(
        rows=n_rows,
        baseline_thresholds=baseline_thresholds,
        candidate_thresholds=candidate_thresholds,
        transition_matrix=matrix,
        baseline_action_counts=b_counts,
        candidate_action_counts=c_counts,
        review_count_delta=review_delta,
        review_rate_delta=review_rate_delta,
        deny_count_delta=deny_delta,
        deny_rate_delta=deny_rate_delta,
        baseline_metrics=b_metrics,
        candidate_metrics=c_metrics,
        cost_delta=cost_delta,
        fraud_catch_delta=fraud_catch_delta,
    )


def evaluate_slices(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    slices: np.ndarray | Sequence[Any],
    *,
    threshold: float = 0.5,
    tiered_thresholds: TieredThresholds | None = None,
    min_slice_size: int = 1,
    min_recall_disparity: float = 0.8,
    max_fpr_disparity: float = 1.5,
) -> SliceDisparityReport:
    """Evaluate decision metrics across categorical sub-population slices.

    Computes per-slice fraud rates, catch rates (recall), false-positive rates,
    precision, and disparity ratios relative to global model performance.
    Flags slices violating minimum recall disparity or maximum FPR disparity.

    Parameters
    ----------
    y_true:
        1D array of binary ground-truth labels (0 or 1).
    probabilities:
        1D array of predicted fraud probabilities in [0, 1].
    slices:
        1D array or sequence of categorical slice identifiers.
    threshold:
        Binary classification threshold used when tiered_thresholds is None.
    tiered_thresholds:
        Optional three-tier decision boundaries.
    min_slice_size:
        Minimum number of records required to evaluate a slice. Slices with
        fewer records are excluded. Must be >= 1.
    min_recall_disparity:
        Minimum acceptable ratio of slice recall to global recall (default: 0.8).
    max_fpr_disparity:
        Maximum acceptable ratio of slice FPR to global FPR (default: 1.5).

    Returns
    -------
    SliceDisparityReport
        Comprehensive sub-population performance and disparity report.
    """
    _validate_vectors(y_true, probabilities)

    slice_arr = np.asarray(slices)
    if slice_arr.ndim != 1:
        raise ValueError("slices must be one-dimensional")
    if slice_arr.shape[0] != y_true.shape[0]:
        raise ValueError("slices and y_true must have the same length")

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(float(threshold))
        or not (0.0 <= float(threshold) <= 1.0)
    ):
        raise ValueError("threshold must be a finite float between 0.0 and 1.0")

    if tiered_thresholds is not None and not isinstance(tiered_thresholds, TieredThresholds):
        raise TypeError("tiered_thresholds must be a TieredThresholds instance")

    if (
        isinstance(min_slice_size, bool)
        or not isinstance(min_slice_size, int)
        or min_slice_size < 1
    ):
        raise ValueError("min_slice_size must be an integer >= 1")

    if (
        isinstance(min_recall_disparity, bool)
        or not isinstance(min_recall_disparity, (int, float))
        or not np.isfinite(float(min_recall_disparity))
        or float(min_recall_disparity) <= 0.0
    ):
        raise ValueError("min_recall_disparity must be a positive finite float")

    if (
        isinstance(max_fpr_disparity, bool)
        or not isinstance(max_fpr_disparity, (int, float))
        or not np.isfinite(float(max_fpr_disparity))
        or float(max_fpr_disparity) <= 0.0
    ):
        raise ValueError("max_fpr_disparity must be a positive finite float")

    min_rec_disp = float(min_recall_disparity)
    max_fp_disp = float(max_fpr_disparity)

    n_total = len(y_true)
    y_arr = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)

    tiered_actions: np.ndarray | None = None
    if tiered_thresholds is not None:
        tiered_actions = assign_tiered_decisions(
            probs,
            review_threshold=tiered_thresholds.review_threshold,
            deny_threshold=tiered_thresholds.deny_threshold,
        )
        flagged = tiered_actions != DecisionAction.ALLOW.value
    else:
        flagged = probs >= float(threshold)

    g_fraud = int(np.sum(y_arr == 1))
    g_negs = n_total - g_fraud
    g_tp = int(np.sum(flagged & (y_arr == 1)))
    g_fp = int(np.sum(flagged & (y_arr == 0)))

    g_fraud_rate = float(g_fraud / n_total) if n_total > 0 else 0.0
    g_precision = float(g_tp / (g_tp + g_fp)) if (g_tp + g_fp) > 0 else 0.0
    g_recall = float(g_tp / g_fraud) if g_fraud > 0 else 1.0
    g_fpr = float(g_fp / g_negs) if g_negs > 0 else 0.0

    # Extract unique slices preserving deterministic order
    unique_slices = sorted(np.unique(slice_arr), key=str)

    rows: list[SliceMetricRow] = []
    underperforming_slices: list[str] = []

    for s_val in unique_slices:
        mask = slice_arr == s_val
        s_count = int(np.sum(mask))
        if s_count < min_slice_size:
            continue

        s_name = str(s_val)
        s_pct = float(s_count / n_total)
        s_y = y_arr[mask]
        s_flagged = flagged[mask]

        s_fraud = int(np.sum(s_y == 1))
        s_negs = s_count - s_fraud
        s_flagged_count = int(np.sum(s_flagged))
        s_flagged_rate = float(s_flagged_count / s_count)
        s_fraud_rate = float(s_fraud / s_count)

        s_tp = int(np.sum(s_flagged & (s_y == 1)))
        s_fp = int(np.sum(s_flagged & (s_y == 0)))
        s_fn = s_fraud - s_tp
        s_tn = s_negs - s_fp

        s_prec = float(s_tp / (s_tp + s_fp)) if (s_tp + s_fp) > 0 else 0.0
        s_rec = float(s_tp / s_fraud) if s_fraud > 0 else 1.0
        s_fpr = float(s_fp / s_negs) if s_negs > 0 else 0.0
        s_f1 = float(2 * s_prec * s_rec / (s_prec + s_rec)) if (s_prec + s_rec) > 0 else 0.0

        rec_disp = float(s_rec / g_recall) if g_recall > 0 else 1.0
        fpr_disp = float(s_fpr / g_fpr) if g_fpr > 0 else 1.0
        fr_disp = float(s_fraud_rate / g_fraud_rate) if g_fraud_rate > 0 else 1.0

        reasons: list[str] = []
        if s_fraud > 0 and rec_disp < min_rec_disp:
            reasons.append(
                f"Recall disparity {rec_disp:.3f} below minimum threshold {min_rec_disp:.3f}"
            )
        if s_negs > 0 and fpr_disp > max_fp_disp:
            reasons.append(
                f"False positive disparity {fpr_disp:.3f} "
                f"exceeds maximum threshold {max_fp_disp:.3f}"
            )

        is_under = len(reasons) > 0
        if is_under:
            underperforming_slices.append(s_name)

        s_allow_cnt = None
        s_rev_cnt = None
        s_deny_cnt = None
        s_allow_rate = None
        s_rev_rate = None
        s_deny_rate = None

        if tiered_actions is not None:
            s_actions = tiered_actions[mask]
            s_allow_cnt = int(np.sum(s_actions == DecisionAction.ALLOW.value))
            s_rev_cnt = int(np.sum(s_actions == DecisionAction.CHALLENGE.value))
            s_deny_cnt = int(np.sum(s_actions == DecisionAction.DENY.value))
            s_allow_rate = float(s_allow_cnt / s_count)
            s_rev_rate = float(s_rev_cnt / s_count)
            s_deny_rate = float(s_deny_cnt / s_count)

        rows.append(
            SliceMetricRow(
                slice_name=s_name,
                count=s_count,
                percentage=round(s_pct, 6),
                fraud_count=s_fraud,
                fraud_rate=round(s_fraud_rate, 6),
                flagged_count=s_flagged_count,
                flagged_rate=round(s_flagged_rate, 6),
                true_positives=s_tp,
                false_positives=s_fp,
                true_negatives=s_tn,
                false_negatives=s_fn,
                precision=round(s_prec, 6),
                recall=round(s_rec, 6),
                f1=round(s_f1, 6),
                false_positive_rate=round(s_fpr, 6),
                recall_disparity=round(rec_disp, 6),
                fpr_disparity=round(fpr_disp, 6),
                fraud_rate_disparity=round(fr_disp, 6),
                is_underperforming=is_under,
                underperformance_reasons=tuple(reasons),
                allow_count=s_allow_cnt,
                review_count=s_rev_cnt,
                deny_count=s_deny_cnt,
                allow_rate=round(s_allow_rate, 6) if s_allow_rate is not None else None,
                review_rate=round(s_rev_rate, 6) if s_rev_rate is not None else None,
                deny_rate=round(s_deny_rate, 6) if s_deny_rate is not None else None,
            )
        )

    return SliceDisparityReport(
        total_records=n_total,
        global_fraud_rate=round(g_fraud_rate, 6),
        global_precision=round(g_precision, 6),
        global_recall=round(g_recall, 6),
        global_false_positive_rate=round(g_fpr, 6),
        min_recall_disparity=min_rec_disp,
        max_fpr_disparity=max_fp_disp,
        underperforming_slices=tuple(underperforming_slices),
        slices=tuple(rows),
    )


def _validate_vectors(y_true: np.ndarray, probabilities: np.ndarray) -> None:
    if y_true.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("y_true and probabilities must be one-dimensional")
    if y_true.shape[0] != probabilities.shape[0] or y_true.size == 0:
        raise ValueError("y_true and probabilities must have the same non-zero length")
    if set(np.unique(y_true).tolist()) != {0, 1}:
        raise ValueError("y_true must contain both binary labels 0 and 1")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be between 0 and 1")


def _validate_costs(false_positive_cost: float, false_negative_cost: float) -> None:
    if (
        not np.isfinite(false_positive_cost)
        or not np.isfinite(false_negative_cost)
        or false_positive_cost <= 0
        or false_negative_cost <= 0
    ):
        raise ValueError("false-positive and false-negative costs must be finite and positive")
