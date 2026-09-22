"""Dependency-light quantized runtime for supported edge model artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from fraud_detection.model import FraudModel

EDGE_SCHEMA_VERSION = 1
SUPPORTED_QUANTIZATION_BITS = 8


class EdgeArtifactError(ValueError):
    """Raised when an edge artifact cannot be built, loaded, or scored."""


@dataclass(frozen=True)
class EdgeModel:
    """Portable int8 logistic runtime with no scikit-learn dependency."""

    model_version: str
    feature_names: tuple[str, ...]
    threshold: float
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    quantized_weights: tuple[int, ...]
    weight_scale: float
    intercept: float
    quantization_bits: int
    prune_epsilon: float
    pruned_features: tuple[str, ...]
    schema_version: int = EDGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EDGE_SCHEMA_VERSION:
            raise EdgeArtifactError(
                f"Unsupported edge schema version {self.schema_version}; "
                f"expected {EDGE_SCHEMA_VERSION}."
            )
        if self.quantization_bits != SUPPORTED_QUANTIZATION_BITS:
            raise EdgeArtifactError("Only 8-bit edge quantization is supported.")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise EdgeArtifactError("Edge artifact feature names must be unique and non-empty.")
        size = len(self.feature_names)
        if not (
            len(self.scaler_mean) == len(self.scaler_scale) == len(self.quantized_weights) == size
        ):
            raise EdgeArtifactError("Edge artifact arrays must match the feature count.")
        if not 0.0 <= self.threshold <= 1.0:
            raise EdgeArtifactError("Edge artifact threshold must be between 0 and 1.")
        if not math.isfinite(self.weight_scale) or self.weight_scale <= 0:
            raise EdgeArtifactError("Edge artifact weight_scale must be finite and positive.")
        if not math.isfinite(self.prune_epsilon) or self.prune_epsilon < 0:
            raise EdgeArtifactError("Edge artifact prune_epsilon must be finite and non-negative.")
        if any(not math.isfinite(value) for value in (*self.scaler_mean, *self.scaler_scale)):
            raise EdgeArtifactError("Edge artifact scaler values must be finite.")
        if any(value <= 0 for value in self.scaler_scale):
            raise EdgeArtifactError("Edge artifact scaler scales must be positive.")
        if any(value < -128 or value > 127 for value in self.quantized_weights):
            raise EdgeArtifactError("Edge artifact weights must fit signed int8.")

    def score_record(self, record: Mapping[str, Any]) -> float:
        """Return the fraud probability for one numeric feature mapping."""
        values = self._ordered_values(record)
        linear_score = self.intercept
        for value, mean, scale, weight in zip(
            values,
            self.scaler_mean,
            self.scaler_scale,
            self.quantized_weights,
            strict=True,
        ):
            linear_score += ((value - mean) / scale) * weight * self.weight_scale
        return _sigmoid(linear_score)

    def score_records(self, records: Sequence[Mapping[str, Any]]) -> list[float]:
        """Return fraud probabilities for an ordered sequence of records."""
        if not records:
            raise EdgeArtifactError("At least one edge transaction is required.")
        return [self.score_record(record) for record in records]

    def predict_record(self, record: Mapping[str, Any]) -> bool:
        """Apply the persisted threshold to one record."""
        return self.score_record(record) >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible edge artifact mapping."""
        return {
            "schema_version": self.schema_version,
            "runtime": "fraud_detection.edge.EdgeModel",
            "model_version": self.model_version,
            "feature_names": list(self.feature_names),
            "threshold": self.threshold,
            "scaler_mean": list(self.scaler_mean),
            "scaler_scale": list(self.scaler_scale),
            "quantized_weights": list(self.quantized_weights),
            "weight_scale": self.weight_scale,
            "intercept": self.intercept,
            "quantization_bits": self.quantization_bits,
            "prune_epsilon": self.prune_epsilon,
            "pruned_features": list(self.pruned_features),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EdgeModel:
        """Validate and load an edge artifact mapping."""
        try:
            feature_names = tuple(payload["feature_names"])
            return cls(
                schema_version=int(payload["schema_version"]),
                model_version=str(payload["model_version"]),
                feature_names=feature_names,
                threshold=float(payload["threshold"]),
                scaler_mean=tuple(float(value) for value in payload["scaler_mean"]),
                scaler_scale=tuple(float(value) for value in payload["scaler_scale"]),
                quantized_weights=tuple(int(value) for value in payload["quantized_weights"]),
                weight_scale=float(payload["weight_scale"]),
                intercept=float(payload["intercept"]),
                quantization_bits=int(payload["quantization_bits"]),
                prune_epsilon=float(payload["prune_epsilon"]),
                pruned_features=tuple(str(value) for value in payload["pruned_features"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EdgeArtifactError(f"Invalid edge artifact: {exc}") from exc

    def _ordered_values(self, record: Mapping[str, Any]) -> tuple[float, ...]:
        expected = set(self.feature_names)
        provided = set(record)
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            message = "Input schema does not match the edge model: " + ", ".join(details)
            raise EdgeArtifactError(message)
        values: list[float] = []
        for feature in self.feature_names:
            raw_value = record[feature]
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise EdgeArtifactError(f"Edge feature {feature!r} must be numeric.")
            value = float(raw_value)
            if not math.isfinite(value):
                raise EdgeArtifactError(f"Edge feature {feature!r} must be finite.")
            values.append(value)
        return tuple(values)


def build_edge_model(
    model: FraudModel,
    *,
    prune_epsilon: float = 0.0,
) -> EdgeModel:
    """Build an int8 edge runtime from an uncalibrated logistic artifact."""
    if not math.isfinite(prune_epsilon) or prune_epsilon < 0:
        raise EdgeArtifactError("prune_epsilon must be finite and non-negative.")
    config = model.metadata.get("training_config", {})
    if not isinstance(config, dict) or config.get("estimator") != "logistic_regression":
        raise EdgeArtifactError(
            "Edge export currently supports logistic_regression artifacts only."
        )
    if config.get("calibration_method") != "none":
        raise EdgeArtifactError(
            "Edge export requires calibration_method='none'; calibrated artifacts "
            "cannot be reproduced by this dependency-light runtime."
        )
    estimator = model.estimator
    pipeline = getattr(estimator, "named_steps", None)
    if not isinstance(pipeline, dict) or "scale" not in pipeline or "classifier" not in pipeline:
        raise EdgeArtifactError(
            "Logistic artifact does not contain the expected scale/classifier pipeline."
        )
    scaler = pipeline["scale"]
    classifier = pipeline["classifier"]
    coefficients = getattr(classifier, "coef_", None)
    intercept = getattr(classifier, "intercept_", None)
    classes = getattr(classifier, "classes_", None)
    if coefficients is None or intercept is None or classes is None:
        raise EdgeArtifactError("Logistic artifact is not fitted for edge export.")
    if list(classes) != [0, 1] or len(coefficients) != 1:
        raise EdgeArtifactError("Edge export requires a fitted binary logistic classifier.")

    raw_weights = [float(value) for value in coefficients[0]]
    pruned_weights = [0.0 if abs(value) < prune_epsilon else value for value in raw_weights]
    max_weight = max((abs(value) for value in pruned_weights), default=0.0)
    weight_scale = max(max_weight / 127.0, 1e-12)
    quantized_weights = tuple(
        max(-128, min(127, round(value / weight_scale))) for value in pruned_weights
    )
    pruned_features = tuple(
        feature
        for feature, value in zip(model.feature_names, pruned_weights, strict=True)
        if value == 0.0
    )
    fingerprint = str(model.metadata.get("dataset_fingerprint", ""))
    return EdgeModel(
        model_version=fingerprint[:12],
        feature_names=model.feature_names,
        threshold=float(model.threshold),
        scaler_mean=tuple(float(value) for value in scaler.mean_),
        scaler_scale=tuple(float(value) for value in scaler.scale_),
        quantized_weights=quantized_weights,
        weight_scale=weight_scale,
        intercept=float(intercept[0]),
        quantization_bits=SUPPORTED_QUANTIZATION_BITS,
        prune_epsilon=prune_epsilon,
        pruned_features=pruned_features,
    )


def load_edge_model(path: Path | str) -> EdgeModel:
    """Load and validate a dependency-light edge artifact."""
    artifact_path = Path(path)
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EdgeArtifactError(f"Could not read edge artifact: {exc}") from exc
    if not isinstance(payload, dict):
        raise EdgeArtifactError("Edge artifact root must be a JSON object.")
    return EdgeModel.from_dict(payload)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)
