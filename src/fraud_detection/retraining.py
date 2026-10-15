"""Automated closed-loop model retraining, challenger safety verification, and safe promotion."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.data import ValidatedDataset
from fraud_detection.evaluation import evaluate_predictions, evaluate_slices
from fraud_detection.feedback import (
    DelayedLabelRecord,
    LabelIngestionStore,
)
from fraud_detection.model import (
    CalibrationMethod,
    EstimatorType,
    FraudModel,
    SplitStrategy,
    TrainingConfig,
    train_model,
)
from fraud_detection.recalibration import compute_calibration_diagnostics


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


@dataclass(frozen=True)
class ChallengerTrainingResult:
    """Result of an automated challenger model training run."""

    model: FraudModel
    config: RetrainingConfig
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    model_version: str
    dataset_fingerprint: str
    training_duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Convert training result summary to JSON-serializable dictionary."""
        return {
            "model_version": self.model_version,
            "dataset_fingerprint": self.dataset_fingerprint,
            "training_duration_seconds": round(float(self.training_duration_seconds), 4),
            "validation_metrics": {
                k: round(float(v), 6) for k, v in self.validation_metrics.items()
            },
            "test_metrics": {k: round(float(v), 6) for k, v in self.test_metrics.items()},
            "config": self.config.to_dict(),
        }


class ChallengerTrainer:
    """Automates model training, hyperparameter mapping, and threshold tuning for challengers."""

    def __init__(self, config: RetrainingConfig | None = None) -> None:
        self.config = config or RetrainingConfig()

    def _build_training_config(self) -> TrainingConfig:
        """Map RetrainingConfig to model.TrainingConfig."""
        try:
            estimator_type = EstimatorType(self.config.estimator)
        except ValueError as exc:
            raise RetrainingError(
                f"Unsupported estimator type: '{self.config.estimator}'."
            ) from exc

        try:
            split_strategy = SplitStrategy(self.config.split_strategy)
        except ValueError as exc:
            raise RetrainingError(
                f"Unsupported split strategy: '{self.config.split_strategy}'."
            ) from exc

        hp = dict(self.config.hyperparameters)
        cal_method_str = str(hp.get("calibration_method", "sigmoid"))
        try:
            cal_method = CalibrationMethod(cal_method_str)
        except ValueError as exc:
            raise RetrainingError(f"Unsupported calibration method: '{cal_method_str}'.") from exc

        return TrainingConfig(
            test_size=self.config.test_size,
            validation_size=self.config.validation_size,
            random_state=self.config.random_state,
            estimator=estimator_type,
            split_strategy=split_strategy,
            time_column=self.config.time_column,
            temporal_gap=self.config.holdout_gap_seconds,
            max_iterations=int(hp.get("max_iterations", 1_000)),
            regularization=float(hp.get("regularization", 1.0)),
            calibration_method=cal_method,
            calibration_folds=int(hp.get("calibration_folds", 3)),
            n_estimators=int(hp.get("n_estimators", 100)),
            max_depth=hp.get("max_depth"),
            learning_rate=float(hp.get("learning_rate", 0.1)),
            tune_tiered=bool(hp.get("tune_tiered", False)),
            review_threshold=hp.get("review_threshold"),
            deny_threshold=hp.get("deny_threshold"),
        )

    def train(
        self,
        dataset: ValidatedDataset,
        *,
        provenance: dict[str, str] | None = None,
    ) -> ChallengerTrainingResult:
        """Train and evaluate candidate model on the provided dataset."""
        training_config = self._build_training_config()
        start_time = time()
        try:
            model = train_model(dataset, config=training_config, provenance=provenance)
        except Exception as exc:
            raise RetrainingError(f"Challenger model training failed: {exc}") from exc
        duration = time() - start_time

        metadata = model.metadata
        val_metrics = {
            k: float(v)
            for k, v in metadata.get("validation_metrics", {}).items()
            if isinstance(v, (int, float))
        }
        test_metrics = {
            k: float(v)
            for k, v in metadata.get("test_metrics", {}).items()
            if isinstance(v, (int, float))
        }

        if "average_precision" in val_metrics and "pr_auc" not in val_metrics:
            val_metrics["pr_auc"] = val_metrics["average_precision"]
        if "average_precision" in test_metrics and "pr_auc" not in test_metrics:
            test_metrics["pr_auc"] = test_metrics["average_precision"]

        return ChallengerTrainingResult(
            model=model,
            config=self.config,
            validation_metrics=val_metrics,
            test_metrics=test_metrics,
            model_version=str(metadata.get("model_version", "challenger")),
            dataset_fingerprint=str(metadata.get("dataset_fingerprint", "")),
            training_duration_seconds=duration,
        )


@dataclass(frozen=True)
class SafetyGatingThresholds:
    """Configurable thresholds determining whether a challenger qualifies for deployment."""

    min_pr_auc_delta: float = -0.02
    min_recall_delta: float = -0.03
    max_ece_increase: float = 0.03
    max_slice_disparity: float = 0.25
    max_net_cost_increase_pct: float = 0.05
    false_positive_cost: float = 1.0
    false_negative_cost: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        """Convert thresholds to JSON-serializable dictionary."""
        return {
            "min_pr_auc_delta": self.min_pr_auc_delta,
            "min_recall_delta": self.min_recall_delta,
            "max_ece_increase": self.max_ece_increase,
            "max_slice_disparity": self.max_slice_disparity,
            "max_net_cost_increase_pct": self.max_net_cost_increase_pct,
            "false_positive_cost": self.false_positive_cost,
            "false_negative_cost": self.false_negative_cost,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SafetyGatingThresholds:
        """Construct thresholds from dictionary."""
        return cls(
            min_pr_auc_delta=float(data.get("min_pr_auc_delta", -0.02)),
            min_recall_delta=float(data.get("min_recall_delta", -0.03)),
            max_ece_increase=float(data.get("max_ece_increase", 0.03)),
            max_slice_disparity=float(data.get("max_slice_disparity", 0.25)),
            max_net_cost_increase_pct=float(data.get("max_net_cost_increase_pct", 0.05)),
            false_positive_cost=float(data.get("false_positive_cost", 1.0)),
            false_negative_cost=float(data.get("false_negative_cost", 10.0)),
        )


class ChallengerSafetyVerifier:
    """Evaluates candidate model against production champion across safety gates."""

    def __init__(self, thresholds: SafetyGatingThresholds | None = None) -> None:
        self.thresholds = thresholds or SafetyGatingThresholds()

    def verify_candidate(
        self,
        candidate_model: FraudModel,
        champion_model: FraudModel,
        holdout_dataset: ValidatedDataset,
        *,
        slice_column: str | None = None,
    ) -> ChallengerVerificationReport:
        """Evaluate candidate vs. champion on identical holdout data and assess safety gating."""
        if len(holdout_dataset.features) == 0:
            raise VerificationError("Holdout dataset cannot be empty for challenger verification.")

        cand_features = holdout_dataset.features
        if candidate_model.feature_names and all(
            f in cand_features.columns for f in candidate_model.feature_names
        ):
            cand_features = cand_features[list(candidate_model.feature_names)]

        champ_features = holdout_dataset.features
        if champion_model.feature_names and all(
            f in champ_features.columns for f in champion_model.feature_names
        ):
            champ_features = champ_features[list(champion_model.feature_names)]

        # Predict probabilities
        cand_probs = np.asarray(candidate_model.predict_probabilities(cand_features), dtype=float)
        champ_probs = np.asarray(champion_model.predict_probabilities(champ_features), dtype=float)
        y_true = np.asarray(holdout_dataset.target, dtype=int)

        # Evaluate basic metrics
        cand_metrics = evaluate_predictions(y_true, cand_probs, threshold=candidate_model.threshold)
        champ_metrics = evaluate_predictions(
            y_true, champ_probs, threshold=champion_model.threshold
        )

        # Calibration ECE
        cand_cal = compute_calibration_diagnostics(y_true, cand_probs)
        champ_cal = compute_calibration_diagnostics(y_true, champ_probs)
        cand_ece = float(cand_cal.expected_calibration_error)
        champ_ece = float(champ_cal.expected_calibration_error)

        # Financial costs
        fp_cost = self.thresholds.false_positive_cost
        fn_cost = self.thresholds.false_negative_cost
        cand_cost = cand_metrics.false_positives * fp_cost + cand_metrics.false_negatives * fn_cost
        champ_cost = (
            champ_metrics.false_positives * fp_cost + champ_metrics.false_negatives * fn_cost
        )
        cost_diff_pct = (cand_cost - champ_cost) / max(1e-6, champ_cost)

        comparisons: list[MetricComparison] = []

        # 1. PR-AUC comparison
        pr_delta = cand_metrics.average_precision - champ_metrics.average_precision
        pr_passed = pr_delta >= self.thresholds.min_pr_auc_delta
        comparisons.append(
            MetricComparison(
                metric_name="pr_auc",
                champion_value=champ_metrics.average_precision,
                challenger_value=cand_metrics.average_precision,
                delta=pr_delta,
                threshold=self.thresholds.min_pr_auc_delta,
                passed=pr_passed,
            )
        )

        # 2. Recall comparison
        rec_delta = cand_metrics.recall - champ_metrics.recall
        rec_passed = rec_delta >= self.thresholds.min_recall_delta
        comparisons.append(
            MetricComparison(
                metric_name="recall",
                champion_value=champ_metrics.recall,
                challenger_value=cand_metrics.recall,
                delta=rec_delta,
                threshold=self.thresholds.min_recall_delta,
                passed=rec_passed,
            )
        )

        # 3. ECE comparison
        ece_delta = cand_ece - champ_ece
        ece_passed = ece_delta <= self.thresholds.max_ece_increase
        comparisons.append(
            MetricComparison(
                metric_name="expected_calibration_error",
                champion_value=champ_ece,
                challenger_value=cand_ece,
                delta=ece_delta,
                threshold=self.thresholds.max_ece_increase,
                passed=ece_passed,
            )
        )

        # 4. Cost comparison
        cost_passed = cost_diff_pct <= self.thresholds.max_net_cost_increase_pct
        comparisons.append(
            MetricComparison(
                metric_name="financial_cost_change_pct",
                champion_value=champ_cost,
                challenger_value=cand_cost,
                delta=cost_diff_pct,
                threshold=self.thresholds.max_net_cost_increase_pct,
                passed=cost_passed,
            )
        )

        # 5. Sub-population slice disparity
        slice_disparities: dict[str, float] = {}
        slice_passed = True
        if slice_column and slice_column in holdout_dataset.features.columns:
            slice_col_vals = holdout_dataset.features[slice_column].to_numpy()
            slice_rep = evaluate_slices(
                y_true, cand_probs, slice_col_vals, threshold=candidate_model.threshold
            )
            recalls = [r.recall for r in slice_rep.slices if r.count >= 5]
            if len(recalls) >= 2:
                disp = max(recalls) - min(recalls)
                slice_disparities[f"{slice_column}_disparity"] = disp
                if disp > self.thresholds.max_slice_disparity:
                    slice_passed = False
        elif "Amount" in holdout_dataset.features.columns:
            amt = holdout_dataset.features["Amount"].to_numpy()
            if len(amt) >= 10:
                q33, q66 = np.percentile(amt, [33, 66])
                amt_slices = np.where(amt <= q33, "low", np.where(amt <= q66, "medium", "high"))
                slice_rep = evaluate_slices(
                    y_true, cand_probs, amt_slices, threshold=candidate_model.threshold
                )
                recalls = [r.recall for r in slice_rep.slices if r.count >= 5]
                if len(recalls) >= 2:
                    disp = max(recalls) - min(recalls)
                    slice_disparities["amount_bracket_disparity"] = disp
                    if disp > self.thresholds.max_slice_disparity:
                        slice_passed = False

        if slice_disparities:
            max_disp = max(slice_disparities.values())
            comparisons.append(
                MetricComparison(
                    metric_name="max_slice_disparity",
                    champion_value=0.0,
                    challenger_value=max_disp,
                    delta=max_disp,
                    threshold=self.thresholds.max_slice_disparity,
                    passed=slice_passed,
                )
            )

        all_passed = all(comp.passed for comp in comparisons)
        status = VerificationStatus.PASSED if all_passed else VerificationStatus.FAILED
        failed_count = sum(1 for c in comparisons if not c.passed)
        if all_passed:
            summary = f"Candidate passed all {len(comparisons)} safety checks against champion."
        else:
            summary = f"Candidate failed {failed_count}/{len(comparisons)} safety checks."

        cand_ver = str(candidate_model.metadata.get("model_version", "challenger"))
        champ_ver = str(champion_model.metadata.get("model_version", "champion"))

        return ChallengerVerificationReport(
            status=status,
            comparisons=comparisons,
            slice_disparities=slice_disparities,
            candidate_version=cand_ver,
            champion_version=champ_ver,
            summary=summary,
            timestamp=time(),
        )
