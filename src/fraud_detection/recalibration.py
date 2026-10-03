"""Model calibration surveillance, reliability diagnostics, and post-hoc recalibration.

This module provides:
1. Deterministic calibration diagnostics (ECE, MCE, RMSCE, Brier score decomposition,
   reliability bins) under uniform, quantile (equal-frequency), and tiered partitioning.
2. Calibration drift detection comparing reference vs production calibration profiles.
3. Post-hoc recalibrators:
   - PlattRecalibrator (logistic sigmoid scaling on log-odds with Laplace smoothing)
   - IsotonicRecalibrator (non-parametric monotonic step interpolation)
   - TemperatureRecalibrator (strictly rank-preserving temperature scaling)
4. Monotonicity validation guaranteeing calibrated probabilities remain in [0, 1].
"""

from __future__ import annotations

import abc
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

__all__ = [
    "BaseRecalibrator",
    "CalibrationBin",
    "CalibrationDiagnostics",
    "CalibrationDriftReport",
    "CalibrationStrategy",
    "IsotonicRecalibrator",
    "PlattRecalibrator",
    "RecalibrationMethod",
    "TemperatureRecalibrator",
    "compute_calibration_diagnostics",
    "compute_calibration_drift",
    "create_recalibrator",
    "load_recalibrator",
]

_EPSILON: float = 1e-12


class RecalibrationMethod(StrEnum):
    """Supported post-hoc probability recalibration algorithms."""

    SIGMOID = "sigmoid"
    ISOTONIC = "isotonic"
    TEMPERATURE = "temperature"


class CalibrationStrategy(StrEnum):
    """Bin partitioning strategies for probability calibration curves."""

    UNIFORM = "uniform"
    QUANTILE = "quantile"
    TIERED = "tiered"


@dataclass(frozen=True)
class CalibrationBin:
    """Diagnostic detail for a single probability bin."""

    bin_index: int
    lower: float
    upper: float
    count: int
    mean_predicted: float
    fraction_positive: float
    gap: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "bin_index": self.bin_index,
            "lower": round(self.lower, 6),
            "upper": round(self.upper, 6),
            "count": self.count,
            "mean_predicted": round(self.mean_predicted, 6),
            "fraction_positive": round(self.fraction_positive, 6),
            "gap": round(self.gap, 6),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CalibrationBin:
        """Construct from a dictionary payload."""
        return cls(
            bin_index=int(payload["bin_index"]),
            lower=float(payload["lower"]),
            upper=float(payload["upper"]),
            count=int(payload["count"]),
            mean_predicted=float(payload["mean_predicted"]),
            fraction_positive=float(payload["fraction_positive"]),
            gap=float(payload["gap"]),
        )


@dataclass(frozen=True)
class CalibrationDiagnostics:
    """Comprehensive calibration metrics and reliability curve decomposition."""

    rows: int
    bins: int
    strategy: str
    brier_score: float
    expected_calibration_error: float
    max_calibration_error: float
    root_mean_squared_error: float
    reliability: float
    resolution: float
    uncertainty: float
    monotonicity_violations: int
    detail: tuple[CalibrationBin, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize diagnostics to a clean dictionary."""
        return {
            "rows": self.rows,
            "bins": self.bins,
            "strategy": self.strategy,
            "brier_score": round(self.brier_score, 6),
            "expected_calibration_error": round(self.expected_calibration_error, 6),
            "max_calibration_error": round(self.max_calibration_error, 6),
            "root_mean_squared_error": round(self.root_mean_squared_error, 6),
            "reliability": round(self.reliability, 6),
            "resolution": round(self.resolution, 6),
            "uncertainty": round(self.uncertainty, 6),
            "monotonicity_violations": self.monotonicity_violations,
            "detail": [item.to_dict() for item in self.detail],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CalibrationDiagnostics:
        """Construct from a dictionary payload."""
        detail_raw = payload.get("detail", [])
        detail = tuple(
            CalibrationBin.from_dict(item) if isinstance(item, Mapping) else item
            for item in detail_raw
        )
        return cls(
            rows=int(payload["rows"]),
            bins=int(payload["bins"]),
            strategy=str(payload.get("strategy", CalibrationStrategy.UNIFORM.value)),
            brier_score=float(payload["brier_score"]),
            expected_calibration_error=float(payload["expected_calibration_error"]),
            max_calibration_error=float(payload["max_calibration_error"]),
            root_mean_squared_error=float(
                payload.get("root_mean_squared_error", payload["expected_calibration_error"])
            ),
            reliability=float(payload["reliability"]),
            resolution=float(payload["resolution"]),
            uncertainty=float(payload["uncertainty"]),
            monotonicity_violations=int(payload.get("monotonicity_violations", 0)),
            detail=detail,
        )


@dataclass(frozen=True)
class CalibrationDriftReport:
    """Surveillance report comparing current calibration against a baseline reference."""

    reference_ece: float
    current_ece: float
    ece_delta: float
    reference_brier: float
    current_brier: float
    brier_delta: float
    max_divergence: float
    status: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize drift report to dictionary."""
        return {
            "reference_ece": round(self.reference_ece, 6),
            "current_ece": round(self.current_ece, 6),
            "ece_delta": round(self.ece_delta, 6),
            "reference_brier": round(self.reference_brier, 6),
            "current_brier": round(self.current_brier, 6),
            "brier_delta": round(self.brier_delta, 6),
            "max_divergence": round(self.max_divergence, 6),
            "status": self.status,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CalibrationDriftReport:
        """Construct from dictionary payload."""
        return cls(
            reference_ece=float(payload["reference_ece"]),
            current_ece=float(payload["current_ece"]),
            ece_delta=float(payload["ece_delta"]),
            reference_brier=float(payload["reference_brier"]),
            current_brier=float(payload["current_brier"]),
            brier_delta=float(payload["brier_delta"]),
            max_divergence=float(payload["max_divergence"]),
            status=str(payload["status"]),
            warnings=tuple(payload.get("warnings", ())),
        )


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Validate that probability vector contains finite values within [0.0, 1.0]."""
    probs = np.asarray(probabilities, dtype=float).ravel()
    if probs.size == 0:
        raise ValueError("Probabilities array cannot be empty.")
    if not np.isfinite(probs).all():
        raise ValueError("Probabilities array contains non-finite (NaN or Inf) values.")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("Probabilities must lie strictly in [0.0, 1.0].")
    return probs


def _validate_arrays(
    probabilities: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate aligned probability and binary target vector inputs."""
    probs = _validate_probabilities(probabilities)
    targets = np.asarray(y_true, dtype=float).ravel()
    if targets.shape != probs.shape:
        raise ValueError(
            f"Shape mismatch: probabilities shape {probs.shape} != target shape {targets.shape}"
        )
    if not np.isfinite(targets).all():
        raise ValueError("Target array contains non-finite values.")
    unique_targets = set(np.unique(targets))
    if not unique_targets.issubset({0.0, 1.0}):
        raise ValueError(
            f"Binary target must contain only 0 and 1, got unique values: {unique_targets}"
        )
    return probs, targets


class BaseRecalibrator(abc.ABC):
    """Abstract base class for probability recalibrators."""

    @property
    @abc.abstractmethod
    def method(self) -> RecalibrationMethod:
        """Recalibration algorithm name."""
        ...

    @abc.abstractmethod
    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> Self:
        """Fit the recalibrator on predicted probabilities and binary labels."""
        ...

    @abc.abstractmethod
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Transform uncalibrated probabilities into calibrated probabilities."""
        ...

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize parameters to a dictionary."""
        ...

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BaseRecalibrator:
        """Deserialize recalibrator from a dictionary."""
        method_str = payload.get("method")
        if not method_str:
            raise ValueError("Recalibrator payload missing required 'method' field.")
        try:
            method = RecalibrationMethod(method_str)
        except ValueError as exc:
            raise ValueError(f"Unknown recalibration method: {method_str}") from exc

        if method == RecalibrationMethod.SIGMOID:
            return PlattRecalibrator.from_dict(payload)
        if method == RecalibrationMethod.ISOTONIC:
            return IsotonicRecalibrator.from_dict(payload)
        return TemperatureRecalibrator.from_dict(payload)


class PlattRecalibrator(BaseRecalibrator):
    """Platt scaling recalibration via regularized logistic regression on log-odds.

    Maps uncalibrated probability p -> p_cal = sigma(A * logit(p) + B) with A >= 0
    guaranteeing a strictly monotonic non-decreasing probability transformation.
    Employs Platt's Bayesian Laplace prior smoothing targets:
        t_plus = (N_pos + 1) / (N_pos + 2)
        t_minus = 1 / (N_neg + 2)
    to prevent overconfidence and infinite loss on separable classes.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0, *, is_fitted: bool = False) -> None:
        self.a_ = float(a)
        self.b_ = float(b)
        self.is_fitted = is_fitted

    @property
    def method(self) -> RecalibrationMethod:
        return RecalibrationMethod.SIGMOID

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> Self:
        probs, targets = _validate_arrays(probabilities, y_true)

        n_pos = float(np.sum(targets == 1.0))
        n_neg = float(np.sum(targets == 0.0))

        if n_pos == 0 or n_neg == 0:
            # Single-class fallback: flat empirical prior
            self.a_ = 0.0
            self.b_ = float(logit(np.clip((n_pos + 1.0) / (probs.size + 2.0), 1e-4, 1.0 - 1e-4)))
            self.is_fitted = True
            return self

        # Platt target smoothing
        t_pos = (n_pos + 1.0) / (n_pos + 2.0)
        t_neg = 1.0 / (n_neg + 2.0)
        smoothed_targets = np.where(targets == 1.0, t_pos, t_neg)

        # Compute log-odds clipped for stability
        clipped_probs = np.clip(probs, _EPSILON, 1.0 - _EPSILON)
        z = logit(clipped_probs)

        # Loss function: binary cross-entropy with slight L2 shrinkage on (A - 1.0) and B
        def loss_fn(params: np.ndarray) -> float:
            a_val, b_val = params[0], params[1]
            logits = a_val * z + b_val
            # Log-loss: - t * log(sigma(logits)) - (1 - t) * log(1 - sigma(logits))
            # Numerically stable: log(1 + exp(logits)) - t * logits
            loss = np.sum(np.logaddexp(0.0, logits) - smoothed_targets * logits)
            reg = 1e-4 * ((a_val - 1.0) ** 2 + b_val**2)
            return float(loss + reg)

        # Optimize with constraint A >= 0 for monotonic non-decreasing mapping
        initial_params = np.array([1.0, 0.0], dtype=float)
        result = minimize(
            loss_fn,
            initial_params,
            bounds=[(0.0, 50.0), (-50.0, 50.0)],
            method="L-BFGS-B",
        )
        if result.success or np.isfinite(result.fun):
            self.a_ = float(result.x[0])
            self.b_ = float(result.x[1])
        else:
            self.a_ = 1.0
            self.b_ = 0.0

        self.is_fitted = True
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        probs = _validate_probabilities(probabilities)
        if not self.is_fitted:
            return probs
        clipped_probs = np.clip(probs, _EPSILON, 1.0 - _EPSILON)
        z = logit(clipped_probs)
        scaled_logits = self.a_ * z + self.b_
        calibrated = expit(scaled_logits)
        return np.asarray(np.clip(calibrated, 0.0, 1.0), dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "a": round(self.a_, 8),
            "b": round(self.b_, 8),
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlattRecalibrator:
        return cls(
            a=float(payload.get("a", 1.0)),
            b=float(payload.get("b", 0.0)),
            is_fitted=bool(payload.get("is_fitted", True)),
        )


class IsotonicRecalibrator(BaseRecalibrator):
    """Non-parametric isotonic regression recalibrator.

    Fits a monotonically non-decreasing piecewise-constant or linear step model
    optimizing squared loss under monotonic order constraints.
    """

    def __init__(
        self,
        *,
        x_thresholds: list[float] | None = None,
        y_thresholds: list[float] | None = None,
        is_fitted: bool = False,
    ) -> None:
        self.x_thresholds_ = list(x_thresholds) if x_thresholds is not None else []
        self.y_thresholds_ = list(y_thresholds) if y_thresholds is not None else []
        self.is_fitted = is_fitted
        self._regressor: IsotonicRegression | None = None
        if self.is_fitted and self.x_thresholds_ and self.y_thresholds_:
            self._rebuild_regressor()

    def _rebuild_regressor(self) -> None:
        if not self.x_thresholds_ or not self.y_thresholds_:
            return
        reg = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )
        # Fit on stored knot coordinates
        x_arr = np.asarray(self.x_thresholds_, dtype=float)
        y_arr = np.asarray(self.y_thresholds_, dtype=float)
        reg.fit(x_arr, y_arr)
        self._regressor = reg

    @property
    def method(self) -> RecalibrationMethod:
        return RecalibrationMethod.ISOTONIC

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> Self:
        probs, targets = _validate_arrays(probabilities, y_true)

        reg = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )
        reg.fit(probs, targets)
        self._regressor = reg
        self.x_thresholds_ = [float(x) for x in reg.X_thresholds_]
        self.y_thresholds_ = [float(y) for y in reg.y_thresholds_]
        self.is_fitted = True
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        probs = _validate_probabilities(probabilities)
        if not self.is_fitted:
            return probs
        if self._regressor is None:
            self._rebuild_regressor()
        regressor = self._regressor
        if regressor is None:
            return probs
        calibrated = np.asarray(regressor.predict(probs), dtype=float)
        return np.clip(calibrated, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "x_thresholds": [round(x, 8) for x in self.x_thresholds_],
            "y_thresholds": [round(y, 8) for y in self.y_thresholds_],
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IsotonicRecalibrator:
        return cls(
            x_thresholds=payload.get("x_thresholds"),
            y_thresholds=payload.get("y_thresholds"),
            is_fitted=bool(payload.get("is_fitted", True)),
        )


class TemperatureRecalibrator(BaseRecalibrator):
    """Temperature scaling recalibration.

    Applies a single positive temperature scalar T:
        p_cal = sigma(logit(p) / T)
    Preserves exact prediction ranking while mitigating confidence over/under-estimation.
    """

    def __init__(self, temperature: float = 1.0, *, is_fitted: bool = False) -> None:
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        self.temperature_ = float(temperature)
        self.is_fitted = is_fitted

    @property
    def method(self) -> RecalibrationMethod:
        return RecalibrationMethod.TEMPERATURE

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> Self:
        probs, targets = _validate_arrays(probabilities, y_true)

        clipped_probs = np.clip(probs, _EPSILON, 1.0 - _EPSILON)
        z = logit(clipped_probs)

        # Optimize temperature T > 0 minimizing log-loss
        def loss_fn(t_val: float) -> float:
            logits = z / t_val
            loss = np.sum(np.logaddexp(0.0, logits) - targets * logits)
            return float(loss)

        result = minimize_scalar(
            loss_fn,
            bounds=(0.01, 20.0),
            method="bounded",
        )
        if result.success and np.isfinite(result.fun) and result.x > 0:
            self.temperature_ = float(result.x)
        else:
            self.temperature_ = 1.0

        self.is_fitted = True
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        probs = _validate_probabilities(probabilities)
        if not self.is_fitted or math.isclose(self.temperature_, 1.0):
            return probs
        clipped_probs = np.clip(probs, _EPSILON, 1.0 - _EPSILON)
        z = logit(clipped_probs)
        scaled_logits = z / self.temperature_
        calibrated = expit(scaled_logits)
        return np.asarray(np.clip(calibrated, 0.0, 1.0), dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "temperature": round(self.temperature_, 8),
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TemperatureRecalibrator:
        return cls(
            temperature=float(payload.get("temperature", 1.0)),
            is_fitted=bool(payload.get("is_fitted", True)),
        )


def create_recalibrator(method: RecalibrationMethod | str, **kwargs: Any) -> BaseRecalibrator:
    """Instantiate a recalibrator by method name."""
    if isinstance(method, RecalibrationMethod):
        method_enum = method
    elif isinstance(method, str):
        try:
            method_enum = RecalibrationMethod(method.lower())
        except ValueError as exc:
            raise ValueError(f"Unknown recalibration method '{method}'") from exc
    else:
        raise TypeError(f"Expected RecalibrationMethod or str, got {type(method).__name__}")

    if method_enum == RecalibrationMethod.SIGMOID:
        return PlattRecalibrator(**kwargs)
    if method_enum == RecalibrationMethod.ISOTONIC:
        return IsotonicRecalibrator(**kwargs)
    return TemperatureRecalibrator(**kwargs)


def load_recalibrator(payload: Mapping[str, Any]) -> BaseRecalibrator:
    """Deserialize a recalibrator from a dictionary representation."""
    return BaseRecalibrator.from_dict(payload)


def _compute_bin_edges(
    probabilities: np.ndarray,
    bins: int,
    strategy: CalibrationStrategy,
    tiered_thresholds: tuple[float, float] | None = None,
) -> np.ndarray:
    """Compute partition bin boundary edges based on the selected strategy."""
    if strategy == CalibrationStrategy.UNIFORM:
        return np.linspace(0.0, 1.0, bins + 1)

    if strategy == CalibrationStrategy.QUANTILE:
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        raw_edges = np.quantile(probabilities, quantiles)
        # Ensure unique, strictly monotonic edges
        unique_edges = np.unique(raw_edges)
        if unique_edges.size < 2:
            return np.linspace(0.0, 1.0, bins + 1)
        # Ensure full [0, 1] coverage or at least min to max
        if unique_edges[0] > 0.0:
            unique_edges = np.insert(unique_edges, 0, 0.0)
        if unique_edges[-1] < 1.0:
            unique_edges = np.append(unique_edges, 1.0)
        return np.unique(unique_edges)

    if strategy == CalibrationStrategy.TIERED:
        if tiered_thresholds is None:
            # Default to 0.20 (review) and 0.80 (deny) if unspecified
            r_thresh, d_thresh = 0.20, 0.80
        else:
            r_thresh, d_thresh = tiered_thresholds
        r_thresh = float(np.clip(r_thresh, 0.01, 0.98))
        d_thresh = float(np.clip(d_thresh, r_thresh + 0.01, 0.99))

        # Partition bins across allow [0, r], review [r, d], deny [d, 1]
        allow_bins = max(1, bins // 3)
        deny_bins = max(1, bins // 3)
        review_bins = max(1, bins - allow_bins - deny_bins)

        edges = np.concatenate(
            [
                np.linspace(0.0, r_thresh, allow_bins, endpoint=False),
                np.linspace(r_thresh, d_thresh, review_bins, endpoint=False),
                np.linspace(d_thresh, 1.0, deny_bins + 1),
            ]
        )
        return np.unique(edges)

    raise ValueError(f"Unknown calibration strategy: {strategy}")


def compute_calibration_diagnostics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
    strategy: CalibrationStrategy | str = CalibrationStrategy.UNIFORM,
    tiered_thresholds: tuple[float, float] | None = None,
) -> CalibrationDiagnostics:
    """Compute comprehensive calibration diagnostics with Brier decomposition.

    Supports uniform, quantile (equal frequency), and tiered decision binning.
    Handles empty bins and class imbalance gracefully.
    """
    probs, targets = _validate_arrays(probabilities, y_true)

    if not 2 <= bins <= 50:
        raise ValueError(f"Number of bins must be between 2 and 50, got {bins}")

    if isinstance(strategy, CalibrationStrategy):
        strat_enum = strategy
    elif isinstance(strategy, str):
        try:
            strat_enum = CalibrationStrategy(strategy.lower())
        except ValueError as exc:
            raise ValueError(f"Unknown calibration strategy '{strategy}'") from exc
    else:
        raise TypeError(f"Expected CalibrationStrategy or str, got {type(strategy).__name__}")

    edges = _compute_bin_edges(probs, bins, strat_enum, tiered_thresholds)
    actual_bins = len(edges) - 1

    # Assign samples to bins: clip ensures right-boundary sample lands in last bin
    assignments = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, actual_bins - 1)
    base_rate = float(np.mean(targets))

    detail: list[CalibrationBin] = []
    gaps: list[float] = []
    weights: list[float] = []

    for index in range(actual_bins):
        mask = assignments == index
        count = int(np.sum(mask))
        lower_edge = float(edges[index])
        upper_edge = float(edges[index + 1])

        if count == 0:
            mid = (lower_edge + upper_edge) / 2.0
            detail.append(
                CalibrationBin(
                    bin_index=index,
                    lower=lower_edge,
                    upper=upper_edge,
                    count=0,
                    mean_predicted=mid,
                    fraction_positive=0.0,
                    gap=0.0,
                )
            )
            gaps.append(0.0)
            weights.append(0.0)
            continue

        mean_pred = float(np.mean(probs[mask]))
        frac_pos = float(np.mean(targets[mask]))
        gap = abs(frac_pos - mean_pred)

        detail.append(
            CalibrationBin(
                bin_index=index,
                lower=lower_edge,
                upper=upper_edge,
                count=count,
                mean_predicted=mean_pred,
                fraction_positive=frac_pos,
                gap=gap,
            )
        )
        gaps.append(gap)
        weights.append(count / probs.size)

    weights_arr = np.asarray(weights, dtype=float)
    gaps_arr = np.asarray(gaps, dtype=float)
    fractions = np.asarray([b.fraction_positive for b in detail], dtype=float)

    ece = float(np.sum(weights_arr * gaps_arr))
    mce = float(np.max(gaps_arr)) if len(gaps_arr) > 0 else 0.0
    rmse = float(np.sqrt(np.sum(weights_arr * (gaps_arr**2))))

    reliability = float(np.sum(weights_arr * (gaps_arr**2)))
    resolution = float(np.sum(weights_arr * ((fractions - base_rate) ** 2)))
    uncertainty = float(base_rate * (1.0 - base_rate))
    brier = float(brier_score_loss(targets, probs))

    # Check monotonicity violations among non-empty bins
    non_empty = [b for b in detail if b.count > 0]
    monotonicity_violations = 0
    for i in range(len(non_empty) - 1):
        if (
            non_empty[i].mean_predicted < non_empty[i + 1].mean_predicted
            and non_empty[i].fraction_positive > non_empty[i + 1].fraction_positive
        ):
            monotonicity_violations += 1

    return CalibrationDiagnostics(
        rows=int(probs.size),
        bins=actual_bins,
        strategy=strat_enum.value,
        brier_score=brier,
        expected_calibration_error=ece,
        max_calibration_error=mce,
        root_mean_squared_error=rmse,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        monotonicity_violations=monotonicity_violations,
        detail=tuple(detail),
    )


def compute_calibration_drift(
    reference: CalibrationDiagnostics,
    current: CalibrationDiagnostics,
    *,
    warn_ece_delta: float = 0.03,
    alarm_ece_delta: float = 0.07,
) -> CalibrationDriftReport:
    """Compare baseline reference calibration against current calibration."""
    if warn_ece_delta < 0.0 or alarm_ece_delta < warn_ece_delta:
        raise ValueError("Invalid thresholds: alarm_ece_delta must be >= warn_ece_delta >= 0.")

    ece_delta = current.expected_calibration_error - reference.expected_calibration_error
    brier_delta = current.brier_score - reference.brier_score

    # Max gap divergence across common bins
    min_bins = min(len(reference.detail), len(current.detail))
    divergences: list[float] = []
    for i in range(min_bins):
        ref_b = reference.detail[i]
        cur_b = current.detail[i]
        if ref_b.count > 0 and cur_b.count > 0:
            divergences.append(abs(cur_b.gap - ref_b.gap))
    max_div = max(divergences) if divergences else 0.0

    warnings_list: list[str] = []
    if ece_delta >= alarm_ece_delta:
        status = "DEGRADED"
        warnings_list.append(
            f"Expected Calibration Error increased by {ece_delta:.4f} "
            f"(>= alarm threshold {alarm_ece_delta:.4f})"
        )
    elif ece_delta >= warn_ece_delta:
        status = "WARNING"
        warnings_list.append(
            f"Expected Calibration Error increased by {ece_delta:.4f} "
            f"(>= warning threshold {warn_ece_delta:.4f})"
        )
    else:
        status = "STABLE"

    if current.monotonicity_violations > reference.monotonicity_violations:
        warnings_list.append(
            f"Monotonicity violations increased from {reference.monotonicity_violations} "
            f"to {current.monotonicity_violations}"
        )
        if status == "STABLE":
            status = "WARNING"

    if brier_delta > 0.05:
        warnings_list.append(f"Brier score degraded by {brier_delta:.4f}")
        if status == "STABLE":
            status = "WARNING"

    return CalibrationDriftReport(
        reference_ece=reference.expected_calibration_error,
        current_ece=current.expected_calibration_error,
        ece_delta=ece_delta,
        reference_brier=reference.brier_score,
        current_brier=current.brier_score,
        brier_delta=brier_delta,
        max_divergence=max_div,
        status=status,
        warnings=tuple(warnings_list),
    )
