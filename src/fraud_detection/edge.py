"""Dependency-light quantized runtime for supported edge model artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fraud_detection.model import FraudModel

EDGE_SCHEMA_VERSION = 1
SUPPORTED_QUANTIZATION_BITS = 8


class EdgeDecisionAction(StrEnum):
    """Tiered decision routing action in edge runtime."""

    ALLOW = "ALLOW"
    CHALLENGE = "CHALLENGE"
    DENY = "DENY"


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

    def explain_record(
        self,
        record: Mapping[str, Any],
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Compute quantized linear feature contributions and top risk factors.

        For each feature: c_i = ((x_i - mean_i) / scale_i) * weight_i * weight_scale.
        Returns a dictionary containing:
            - probability: float
            - intercept: float
            - contributions: dict[str, float]
            - top_contributions: list[dict[str, Any]] (ranked by absolute contribution)
            - summary: human-readable explanation string
        """
        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not (1 <= top_k <= len(self.feature_names))
        ):
            raise EdgeArtifactError(
                f"top_k must be an integer between 1 and {len(self.feature_names)}."
            )
        values = self._ordered_values(record)
        contributions: dict[str, float] = {}
        linear_score = self.intercept
        for feature, value, mean, scale, weight in zip(
            self.feature_names,
            values,
            self.scaler_mean,
            self.scaler_scale,
            self.quantized_weights,
            strict=True,
        ):
            contribution = ((value - mean) / scale) * weight * self.weight_scale
            contributions[feature] = contribution
            linear_score += contribution

        prob = _sigmoid(linear_score)

        sorted_features = sorted(
            contributions.items(),
            key=lambda item: (-abs(item[1]), item[0]),
        )
        top_items = sorted_features[:top_k]
        top_contributions = [
            {
                "feature": feat,
                "contribution": round(contrib, 6),
                "direction": "increases_risk" if contrib >= 0 else "decreases_risk",
            }
            for feat, contrib in top_items
        ]
        factors = ", ".join(
            f"{item['feature']} ({float(item['contribution']):+.4f}, "
            f"{str(item['direction']).replace('_', ' ')})"
            for item in top_contributions
        )
        summary = f"Edge score: {prob:.2%}. Top contributing factors: {factors}."
        return {
            "probability": prob,
            "intercept": self.intercept,
            "contributions": contributions,
            "top_contributions": top_contributions,
            "summary": summary,
        }

    def explain_records(
        self,
        records: Sequence[Mapping[str, Any]],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Compute explanations for an ordered sequence of records."""
        if not records:
            raise EdgeArtifactError("At least one edge transaction is required.")
        return [self.explain_record(record, top_k=top_k) for record in records]

    def predict_decision_record(
        self,
        record: Mapping[str, Any],
        *,
        review_threshold: float | None = None,
        deny_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Classify a single record into ALLOW, CHALLENGE, or DENY."""
        if (review_threshold is None) ^ (deny_threshold is None):
            raise EdgeArtifactError(
                "review_threshold and deny_threshold must both be provided or both omitted."
            )
        eff_review = self.threshold if review_threshold is None else review_threshold
        eff_deny = self.threshold if deny_threshold is None else deny_threshold

        if (
            isinstance(eff_review, bool)
            or not math.isfinite(eff_review)
            or not (0.0 <= eff_review <= 1.0)
        ):
            raise EdgeArtifactError("review_threshold must be a finite float between 0.0 and 1.0.")
        if (
            isinstance(eff_deny, bool)
            or not math.isfinite(eff_deny)
            or not (0.0 <= eff_deny <= 1.0)
        ):
            raise EdgeArtifactError("deny_threshold must be a finite float between 0.0 and 1.0.")
        if eff_review > eff_deny:
            raise EdgeArtifactError(
                f"review_threshold ({eff_review}) cannot exceed deny_threshold ({eff_deny})."
            )

        prob = self.score_record(record)
        if prob < eff_review:
            decision = EdgeDecisionAction.ALLOW.value
            is_fraud = False
        elif prob >= eff_deny:
            decision = EdgeDecisionAction.DENY.value
            is_fraud = True
        else:
            decision = EdgeDecisionAction.CHALLENGE.value
            is_fraud = False

        return {
            "probability": prob,
            "decision": decision,
            "is_fraud": is_fraud,
            "review_threshold": float(eff_review),
            "deny_threshold": float(eff_deny),
        }

    def predict_decision_records(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        review_threshold: float | None = None,
        deny_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Predict tiered decisions for an ordered sequence of records."""
        if not records:
            raise EdgeArtifactError("At least one edge transaction is required.")
        return [
            self.predict_decision_record(
                record,
                review_threshold=review_threshold,
                deny_threshold=deny_threshold,
            )
            for record in records
        ]

    def score_batch_file(
        self,
        input_path: Path | str,
        output_path: Path | str,
        *,
        explain: bool = False,
        top_k: int = 3,
        review_threshold: float | None = None,
        deny_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Stream a batch file (CSV or JSONL) and write scored decisions."""
        return score_batch_file(
            self,
            input_path,
            output_path,
            explain=explain,
            top_k=top_k,
            review_threshold=review_threshold,
            deny_threshold=deny_threshold,
        )

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


def _stream_records(in_path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    suffix = in_path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        with in_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise EdgeArtifactError(f"Line {line_idx} is not valid JSON: {exc}") from exc
                if not isinstance(obj, dict):
                    raise EdgeArtifactError(f"Line {line_idx} must be a JSON object.")
                yield line_idx, obj
    elif suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        with in_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row_idx, row in enumerate(reader, start=1):
                converted: dict[str, Any] = {}
                for k, v in row.items():
                    if k is not None and v is not None:
                        try:
                            converted[k] = float(v)
                        except (ValueError, TypeError):
                            converted[k] = v
                yield row_idx, converted
    elif suffix == ".json":
        try:
            data = json.loads(in_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EdgeArtifactError(f"Invalid JSON in {in_path}: {exc}") from exc
        if not isinstance(data, list):
            raise EdgeArtifactError("JSON content must be a list of records.")
        for row_idx, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                raise EdgeArtifactError(f"Item {row_idx} must be a JSON object.")
            yield row_idx, item
    else:
        raise EdgeArtifactError(
            f"Unsupported input file format '{suffix}'. Expected .csv, .tsv, .json, or .jsonl."
        )


def score_batch_file(
    target_a: EdgeModel | Path | str,
    target_b: Path | str,
    target_c: Path | str | None = None,
    *,
    model: EdgeModel | Path | str | None = None,
    explain: bool = False,
    top_k: int = 3,
    review_threshold: float | None = None,
    deny_threshold: float | None = None,
) -> dict[str, Any]:
    """Stream a batch of transactions and write scored predictions.

    Supports JSONL (.jsonl, .ndjson), JSON (.json array or JSONL), and CSV (.csv, .tsv).
    """
    if isinstance(target_a, EdgeModel):
        edge_model = target_a
        in_path = Path(target_b)
        if target_c is None:
            raise EdgeArtifactError("output_path is required.")
        out_path = Path(target_c)
    elif target_c is not None:
        edge_model = target_a if isinstance(target_a, EdgeModel) else load_edge_model(target_a)
        in_path = Path(target_b)
        out_path = Path(target_c)
    else:
        if model is None:
            raise EdgeArtifactError("model must be provided as an argument to score_batch_file.")
        edge_model = model if isinstance(model, EdgeModel) else load_edge_model(model)
        in_path = Path(target_a)
        out_path = Path(target_b)

    if not in_path.is_file():
        raise EdgeArtifactError(f"Input file not found: {in_path}")

    out_suffix = out_path.suffix.lower()
    if out_suffix not in (".csv", ".tsv", ".jsonl", ".ndjson", ".json"):
        raise EdgeArtifactError(
            f"Unsupported output file format '{out_suffix}'. Expected .csv, .tsv, .json, or .jsonl."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    total_prob = 0.0
    decision_counts = {
        EdgeDecisionAction.ALLOW.value: 0,
        EdgeDecisionAction.CHALLENGE.value: 0,
        EdgeDecisionAction.DENY.value: 0,
    }

    if out_suffix in (".csv", ".tsv"):
        delimiter = "\t" if out_suffix == ".tsv" else ","
        fieldnames = ["row_id", "probability", "decision", "is_fraud"]
        if explain:
            fieldnames.extend(["summary", "explanation", "top_factors"])

        with out_path.open("w", encoding="utf-8", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()

            for row_idx, record in _stream_records(in_path):
                count += 1
                decision_res = edge_model.predict_decision_record(
                    record,
                    review_threshold=review_threshold,
                    deny_threshold=deny_threshold,
                )
                prob = decision_res["probability"]
                decision = decision_res["decision"]
                total_prob += prob
                decision_counts[decision] += 1

                row_id = record.get("id", record.get("row_id", row_idx))
                out_row: dict[str, Any] = {
                    "row_id": row_id,
                    "probability": round(prob, 6),
                    "decision": decision,
                    "is_fraud": int(decision_res["is_fraud"]),
                }
                if explain:
                    expl = edge_model.explain_record(record, top_k=top_k)
                    out_row["summary"] = expl["summary"]
                    out_row["explanation"] = expl["summary"]
                    out_row["top_factors"] = "; ".join(
                        f"{c['feature']}:{c['contribution']:+.4f}"
                        for c in expl["top_contributions"]
                    )
                writer.writerow(out_row)

    elif out_suffix in (".jsonl", ".ndjson"):
        with out_path.open("w", encoding="utf-8") as out_f:
            for row_idx, record in _stream_records(in_path):
                count += 1
                decision_res = edge_model.predict_decision_record(
                    record,
                    review_threshold=review_threshold,
                    deny_threshold=deny_threshold,
                )
                prob = decision_res["probability"]
                decision = decision_res["decision"]
                total_prob += prob
                decision_counts[decision] += 1

                row_id = record.get("id", record.get("row_id", row_idx))
                out_obj: dict[str, Any] = {
                    "row_id": row_id,
                    "probability": round(prob, 6),
                    "decision": decision,
                    "is_fraud": decision_res["is_fraud"],
                    "review_threshold": decision_res["review_threshold"],
                    "deny_threshold": decision_res["deny_threshold"],
                }
                if explain:
                    expl = edge_model.explain_record(record, top_k=top_k)
                    out_obj["summary"] = expl["summary"]
                    out_obj["explanation"] = expl["summary"]
                    out_obj["top_contributions"] = expl["top_contributions"]
                out_f.write(json.dumps(out_obj) + "\n")

    else:  # .json array
        json_rows: list[dict[str, Any]] = []
        for row_idx, record in _stream_records(in_path):
            count += 1
            decision_res = edge_model.predict_decision_record(
                record,
                review_threshold=review_threshold,
                deny_threshold=deny_threshold,
            )
            prob = decision_res["probability"]
            decision = decision_res["decision"]
            total_prob += prob
            decision_counts[decision] += 1

            row_id = record.get("id", record.get("row_id", row_idx))
            out_obj = {
                "row_id": row_id,
                "probability": round(prob, 6),
                "decision": decision,
                "is_fraud": decision_res["is_fraud"],
                "review_threshold": decision_res["review_threshold"],
                "deny_threshold": decision_res["deny_threshold"],
            }
            if explain:
                expl = edge_model.explain_record(record, top_k=top_k)
                out_obj["summary"] = expl["summary"]
                out_obj["explanation"] = expl["summary"]
                out_obj["top_contributions"] = expl["top_contributions"]
            json_rows.append(out_obj)

        with out_path.open("w", encoding="utf-8") as out_f:
            json.dump(json_rows, out_f, indent=2)

    if count == 0:
        raise EdgeArtifactError("Input file contains no records to score.")

    return {
        "rows_processed": count,
        "input_path": str(in_path),
        "output_path": str(out_path),
        "mean_probability": round(total_prob / count, 6),
        "decision_counts": decision_counts,
        "explained": explain,
    }
