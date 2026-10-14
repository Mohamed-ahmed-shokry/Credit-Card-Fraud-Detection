"""Automated closed-loop model retraining, challenger safety verification, and safe promotion."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from time import time
from typing import Any

import pandas as pd

from fraud_detection.data import ValidatedDataset
from fraud_detection.feedback import (
    DelayedLabelRecord,
    LabelIngestionStore,
)


class RetrainingError(Exception):
    """Base exception for closed-loop retraining and promotion errors."""


class DatasetBuildError(RetrainingError):
    """Raised when materialized retraining datasets fail validation or capacity checks."""


class VerificationError(RetrainingError):
    """Raised when candidate model verification encounters unrecoverable errors."""


class PromotionError(RetrainingError):
    """Raised when model promotion or hot-reload fails."""


class RetrainingStatus(StrEnum):
    """Lifecycle status of an automated retraining job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


class VerificationStatus(StrEnum):
    """Outcome status of challenger safety verification against active champion."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class PromotionType(StrEnum):
    """Promotion deployment target for verified candidate models."""

    CANARY = "canary"
    CHAMPION = "champion"
    NONE = "none"


@dataclass(frozen=True)
class RetrainingRecord:
    """Individual transaction observation paired with delayed ground-truth outcome and features."""

    transaction_id: str
    event_timestamp: float
    label: int
    features: dict[str, float]
    amount: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize record to dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "event_timestamp": self.event_timestamp,
            "label": self.label,
            "amount": self.amount,
            "features": dict(self.features),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrainingRecord:
        """Construct record from dictionary."""
        return cls(
            transaction_id=str(data["transaction_id"]),
            event_timestamp=float(data["event_timestamp"]),
            label=int(data["label"]),
            features={str(k): float(v) for k, v in data.get("features", {}).items()},
            amount=float(data.get("amount", 0.0)),
        )


@dataclass(frozen=True)
class MetricComparison:
    """Detailed evaluation comparison of a single metric between champion and challenger."""

    metric_name: str
    champion_value: float
    challenger_value: float
    delta: float
    threshold: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert metric comparison to JSON-serializable dictionary."""
        return {
            "metric_name": self.metric_name,
            "champion_value": round(float(self.champion_value), 6),
            "challenger_value": round(float(self.challenger_value), 6),
            "delta": round(float(self.delta), 6),
            "threshold": round(float(self.threshold), 6),
            "passed": bool(self.passed),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricComparison:
        """Construct MetricComparison from dictionary."""
        return cls(
            metric_name=str(data["metric_name"]),
            champion_value=float(data["champion_value"]),
            challenger_value=float(data["challenger_value"]),
            delta=float(data["delta"]),
            threshold=float(data["threshold"]),
            passed=bool(data["passed"]),
        )


@dataclass(frozen=True)
class ChallengerVerificationReport:
    """Comprehensive safety verification report comparing candidate against champion."""

    status: VerificationStatus
    comparisons: list[MetricComparison] = field(default_factory=list)
    slice_disparities: dict[str, float] = field(default_factory=dict)
    candidate_version: str = ""
    champion_version: str = ""
    summary: str = ""
    timestamp: float = field(default_factory=time)

    @property
    def passed(self) -> bool:
        """Return True if candidate is acceptable for deployment (PASSED or WARNING)."""
        return self.status != VerificationStatus.FAILED

    def to_dict(self) -> dict[str, Any]:
        """Convert verification report to JSON-serializable dictionary."""
        return {
            "status": self.status.value,
            "passed": self.passed,
            "candidate_version": self.candidate_version,
            "champion_version": self.champion_version,
            "summary": self.summary,
            "timestamp": self.timestamp,
            "comparisons": [comp.to_dict() for comp in self.comparisons],
            "slice_disparities": {k: round(float(v), 6) for k, v in self.slice_disparities.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChallengerVerificationReport:
        """Construct ChallengerVerificationReport from dictionary."""
        return cls(
            status=VerificationStatus(data["status"]),
            comparisons=[MetricComparison.from_dict(c) for c in data.get("comparisons", [])],
            slice_disparities={k: float(v) for k, v in data.get("slice_disparities", {}).items()},
            candidate_version=str(data.get("candidate_version", "")),
            champion_version=str(data.get("champion_version", "")),
            summary=str(data.get("summary", "")),
            timestamp=float(data.get("timestamp", 0.0)),
        )


@dataclass(frozen=True)
class RetrainingConfig:
    """Configuration governing automated retraining dataset materialization and training."""

    retrain_start_timestamp: float | None = None
    retrain_end_timestamp: float | None = None
    holdout_gap_seconds: float = 0.0
    estimator: str = "logistic_regression"
    split_strategy: str = "temporal"
    time_column: str = "Time"
    target_column: str = "Class"
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    min_training_samples: int = 20
    min_fraud_samples: int = 2
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.min_training_samples < 2:
            raise RetrainingError("min_training_samples must be at least 2.")
        if self.min_fraud_samples < 1:
            raise RetrainingError("min_fraud_samples must be at least 1.")
        if self.holdout_gap_seconds < 0.0:
            raise RetrainingError("holdout_gap_seconds cannot be negative.")
        if (
            self.retrain_start_timestamp is not None
            and self.retrain_end_timestamp is not None
            and self.retrain_end_timestamp <= self.retrain_start_timestamp
        ):
            raise RetrainingError(
                "retrain_end_timestamp must be greater than retrain_start_timestamp."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert RetrainingConfig to JSON-serializable dictionary."""
        return {
            "retrain_start_timestamp": self.retrain_start_timestamp,
            "retrain_end_timestamp": self.retrain_end_timestamp,
            "holdout_gap_seconds": self.holdout_gap_seconds,
            "estimator": self.estimator,
            "split_strategy": self.split_strategy,
            "time_column": self.time_column,
            "target_column": self.target_column,
            "test_size": self.test_size,
            "validation_size": self.validation_size,
            "random_state": self.random_state,
            "min_training_samples": self.min_training_samples,
            "min_fraud_samples": self.min_fraud_samples,
            "hyperparameters": dict(self.hyperparameters),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrainingConfig:
        """Construct RetrainingConfig from dictionary."""
        return cls(
            retrain_start_timestamp=data.get("retrain_start_timestamp"),
            retrain_end_timestamp=data.get("retrain_end_timestamp"),
            holdout_gap_seconds=float(data.get("holdout_gap_seconds", 0.0)),
            estimator=str(data.get("estimator", "logistic_regression")),
            split_strategy=str(data.get("split_strategy", "temporal")),
            time_column=str(data.get("time_column", "Time")),
            target_column=str(data.get("target_column", "Class")),
            test_size=float(data.get("test_size", 0.2)),
            validation_size=float(data.get("validation_size", 0.2)),
            random_state=int(data.get("random_state", 42)),
            min_training_samples=int(data.get("min_training_samples", 20)),
            min_fraud_samples=int(data.get("min_fraud_samples", 2)),
            hyperparameters=dict(data.get("hyperparameters", {})),
        )


class RetrainingDatasetBuilder:
    """Builds leak-free, validated training datasets from feedback labels and audit logs."""

    def __init__(self, config: RetrainingConfig | None = None) -> None:
        self.config = config or RetrainingConfig()

    def build_from_records(
        self,
        records: Sequence[RetrainingRecord],
        *,
        feature_names: Sequence[str] | None = None,
    ) -> ValidatedDataset:
        """Materialize a ValidatedDataset from RetrainingRecord instances.

        Applies temporal window filtering and sample volume assertions.
        """
        if not records:
            raise DatasetBuildError("No records provided for retraining dataset materialization.")

        filtered_records: list[RetrainingRecord] = []
        for r in records:
            ts = r.event_timestamp
            if (
                self.config.retrain_start_timestamp is not None
                and ts < self.config.retrain_start_timestamp
            ):
                continue
            if (
                self.config.retrain_end_timestamp is not None
                and ts > self.config.retrain_end_timestamp
            ):
                continue
            filtered_records.append(r)

        if len(filtered_records) < self.config.min_training_samples:
            raise DatasetBuildError(
                f"Materialized dataset has {len(filtered_records)} samples, below minimum "
                f"required {self.config.min_training_samples}."
            )

        fraud_count = sum(1 for r in filtered_records if r.label == 1)
        if fraud_count < self.config.min_fraud_samples:
            raise DatasetBuildError(
                f"Materialized dataset has {fraud_count} fraud samples, below minimum "
                f"required {self.config.min_fraud_samples}."
            )

        non_fraud_count = len(filtered_records) - fraud_count
        if non_fraud_count == 0:
            raise DatasetBuildError(
                "Materialized dataset contains only positive fraud samples (single-class dataset)."
            )

        # Build feature DataFrame
        rows: list[dict[str, Any]] = []
        labels: list[int] = []
        for r in filtered_records:
            row_dict = dict(r.features)
            # Ensure time column is present
            if self.config.time_column not in row_dict:
                row_dict[self.config.time_column] = r.event_timestamp
            rows.append(row_dict)
            labels.append(int(r.label))

        df_features = pd.DataFrame(rows)

        # If explicit feature_names are required, enforce presence and ordering
        if feature_names is not None:
            missing_features = [f for f in feature_names if f not in df_features.columns]
            if missing_features:
                raise DatasetBuildError(
                    f"Features missing from feedback records: {missing_features}"
                )
            df_features = df_features[list(feature_names)]

        # Ensure all columns are numeric
        for col in df_features.columns:
            df_features[col] = pd.to_numeric(df_features[col], errors="coerce")
            if df_features[col].isna().any():
                raise DatasetBuildError(
                    f"Feature column '{col}' contains non-numeric or missing values."
                )

        series_target = pd.Series(labels, name=self.config.target_column, dtype=int)
        return ValidatedDataset(features=df_features, target=series_target)

    def build_from_audit_and_labels(
        self,
        audit_log_path: Path | str,
        labels: LabelIngestionStore | Path | str | Sequence[DelayedLabelRecord],
        *,
        feature_names: Sequence[str] | None = None,
        unmatched_as_legitimate: bool = False,
    ) -> ValidatedDataset:
        """Match delayed labels against historical audit logs and materialize a ValidatedDataset."""
        audit_p = Path(audit_log_path)
        if not audit_p.is_file():
            raise DatasetBuildError(f"Audit log file not found: {audit_p}")

        label_records: Sequence[DelayedLabelRecord]
        if isinstance(labels, LabelIngestionStore):
            label_records = labels.all_records()
        elif isinstance(labels, (Path, str)):
            p = Path(labels)
            if not p.is_file():
                raise DatasetBuildError(f"Labels file not found: {p}")
            store = LabelIngestionStore()
            store.ingest_from_jsonl(p)
            label_records = store.all_records()
        else:
            label_records = labels

        labels_map = {str(r.transaction_id): r for r in label_records}
        retraining_records: list[RetrainingRecord] = []

        with audit_p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise DatasetBuildError(
                        f"Malformed audit event on line {line_no}: {exc}"
                    ) from exc

                if event.get("event_type") != "scoring":
                    continue

                payload = event.get("payload", {})
                predictions = payload.get("predictions", [])
                features_list = payload.get("features", [])
                req_id = payload.get("request_id")
                event_ts = float(event.get("timestamp", 0.0))

                for idx, pred in enumerate(predictions):
                    feat_dict = (
                        dict(features_list[idx])
                        if idx < len(features_list) and isinstance(features_list[idx], dict)
                        else {}
                    )
                    tx_id = (
                        pred.get("transaction_id")
                        or feat_dict.get("transaction_id")
                        or (f"{req_id}_{idx}" if req_id else f"tx_{line_no}_{idx}")
                    )

                    lbl = labels_map.get(str(tx_id))
                    if lbl is not None:
                        is_fraud = 1 if lbl.is_fraud else 0
                        ts = lbl.event_timestamp if lbl.event_timestamp > 0 else event_ts
                        amt = float(
                            lbl.amount
                            if lbl.amount > 0
                            else feat_dict.get("Amount", pred.get("amount", 0.0))
                        )
                        retraining_records.append(
                            RetrainingRecord(
                                transaction_id=str(tx_id),
                                event_timestamp=ts,
                                label=is_fraud,
                                features=feat_dict,
                                amount=amt,
                            )
                        )
                    elif unmatched_as_legitimate:
                        amt = float(feat_dict.get("Amount", pred.get("amount", 0.0)))
                        retraining_records.append(
                            RetrainingRecord(
                                transaction_id=str(tx_id),
                                event_timestamp=event_ts,
                                label=0,
                                features=feat_dict,
                                amount=amt,
                            )
                        )

        return self.build_from_records(retraining_records, feature_names=feature_names)
