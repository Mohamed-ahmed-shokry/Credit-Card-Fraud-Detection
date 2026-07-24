"""Threshold selection and metrics for imbalanced fraud classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Serializable binary-classification evaluation results."""

    threshold: float
    roc_auc: float
    average_precision: float
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
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, predictions)),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
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
