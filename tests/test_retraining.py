"""Unit tests for automated closed-loop retraining and dataset materialization."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from fraud_detection.data import ValidatedDataset, generate_synthetic_data
from fraud_detection.feedback import (
    DelayedLabelRecord,
    DisputeType,
    LabelIngestionStore,
)
from fraud_detection.model import FraudModel
from fraud_detection.retraining import (
    ChallengerTrainer,
    ChallengerTrainingResult,
    ChallengerVerificationReport,
    DatasetBuildError,
    MetricComparison,
    PromotionType,
    RetrainingConfig,
    RetrainingDatasetBuilder,
    RetrainingError,
    RetrainingRecord,
    RetrainingStatus,
    VerificationError,
    VerificationStatus,
)


def _make_dummy_records(
    count: int = 30,
    fraud_count: int = 5,
    base_time: float = 1_000.0,
) -> list[RetrainingRecord]:
    """Helper to generate dummy RetrainingRecord instances."""
    records = []
    for i in range(count):
        is_fraud = 1 if i < fraud_count else 0
        records.append(
            RetrainingRecord(
                transaction_id=f"tx_{i:04d}",
                event_timestamp=base_time + float(i * 10),
                label=is_fraud,
                features={"V1": float(i) * 0.1, "V2": float(i) * -0.1, "Amount": 50.0 + float(i)},
                amount=50.0 + float(i),
            )
        )
    return records


def test_retraining_enums_and_exceptions() -> None:
    """Test enums and exception hierarchies."""
    assert RetrainingStatus.COMPLETED == "completed"
    assert VerificationStatus.PASSED == "passed"
    assert PromotionType.CANARY == "canary"

    assert issubclass(DatasetBuildError, RetrainingError)
    assert issubclass(VerificationError, RetrainingError)


def test_retraining_record_serialization() -> None:
    """Test RetrainingRecord serialization and deserialization."""
    rec = RetrainingRecord(
        transaction_id="tx_123",
        event_timestamp=1000.0,
        label=1,
        features={"V1": 1.5, "Amount": 42.0},
        amount=42.0,
    )
    d = rec.to_dict()
    assert d["transaction_id"] == "tx_123"
    assert d["label"] == 1
    assert d["features"]["V1"] == 1.5

    restored = RetrainingRecord.from_dict(d)
    assert restored.transaction_id == "tx_123"
    assert restored.event_timestamp == 1000.0
    assert restored.label == 1
    assert restored.features["V1"] == 1.5
    assert restored.amount == 42.0


def test_metric_comparison_serialization() -> None:
    """Test MetricComparison round-trip serialization."""
    comp = MetricComparison(
        metric_name="pr_auc",
        champion_value=0.825,
        challenger_value=0.845,
        delta=0.02,
        threshold=-0.01,
        passed=True,
    )
    data = comp.to_dict()
    assert data["metric_name"] == "pr_auc"
    assert data["passed"] is True

    restored = MetricComparison.from_dict(data)
    assert restored.metric_name == comp.metric_name
    assert restored.champion_value == pytest.approx(comp.champion_value)
    assert restored.challenger_value == pytest.approx(comp.challenger_value)
    assert restored.passed == comp.passed


def test_challenger_verification_report_serialization() -> None:
    """Test ChallengerVerificationReport serialization and status properties."""
    comp = MetricComparison("recall", 0.8, 0.85, 0.05, -0.02, True)
    report = ChallengerVerificationReport(
        status=VerificationStatus.PASSED,
        comparisons=[comp],
        slice_disparities={"high_amount": 0.05},
        candidate_version="candidate_v2",
        champion_version="champion_v1",
        summary="Candidate passed all safety checks.",
        timestamp=1700000000.0,
    )
    assert report.passed is True
    data = report.to_dict()
    assert data["status"] == "passed"
    assert data["passed"] is True
    assert len(data["comparisons"]) == 1

    restored = ChallengerVerificationReport.from_dict(data)
    assert restored.status == VerificationStatus.PASSED
    assert restored.candidate_version == "candidate_v2"
    assert restored.slice_disparities["high_amount"] == pytest.approx(0.05)

    failed_report = ChallengerVerificationReport(status=VerificationStatus.FAILED)
    assert failed_report.passed is False


def test_retraining_config_validation_and_serialization() -> None:
    """Test RetrainingConfig validation and serialization."""
    cfg = RetrainingConfig(
        retrain_start_timestamp=100.0,
        retrain_end_timestamp=500.0,
        holdout_gap_seconds=3600.0,
        min_training_samples=30,
        min_fraud_samples=3,
        hyperparameters={"C": 0.5},
    )
    data = cfg.to_dict()
    restored = RetrainingConfig.from_dict(data)
    assert restored.retrain_start_timestamp == 100.0
    assert restored.retrain_end_timestamp == 500.0
    assert restored.holdout_gap_seconds == 3600.0
    assert restored.hyperparameters["C"] == 0.5

    # Validation errors
    with pytest.raises(RetrainingError, match="min_training_samples must be at least 2"):
        RetrainingConfig(min_training_samples=1)

    with pytest.raises(RetrainingError, match="min_fraud_samples must be at least 1"):
        RetrainingConfig(min_fraud_samples=0)

    with pytest.raises(RetrainingError, match="holdout_gap_seconds cannot be negative"):
        RetrainingConfig(holdout_gap_seconds=-5.0)

    with pytest.raises(RetrainingError, match="retrain_end_timestamp must be greater"):
        RetrainingConfig(retrain_start_timestamp=500.0, retrain_end_timestamp=100.0)


def test_dataset_builder_from_records() -> None:
    """Test RetrainingDatasetBuilder successfully builds a ValidatedDataset."""
    records = _make_dummy_records(count=25, fraud_count=4, base_time=100.0)
    cfg = RetrainingConfig(min_training_samples=10, min_fraud_samples=2)
    builder = RetrainingDatasetBuilder(cfg)

    dataset = builder.build_from_records(records)
    assert isinstance(dataset, ValidatedDataset)
    assert len(dataset.features) == 25
    assert len(dataset.target) == 25
    assert dataset.target.sum() == 4
    assert "Time" in dataset.features.columns
    assert "V1" in dataset.features.columns


def test_dataset_builder_temporal_filtering() -> None:
    """Test RetrainingDatasetBuilder applies temporal window bounds."""
    records = _make_dummy_records(count=30, fraud_count=6, base_time=100.0)
    # Timestamps are 100, 110, 120, ..., 390
    cfg = RetrainingConfig(
        retrain_start_timestamp=150.0,
        retrain_end_timestamp=250.0,
        min_training_samples=5,
        min_fraud_samples=1,
    )
    builder = RetrainingDatasetBuilder(cfg)
    dataset = builder.build_from_records(records)
    assert len(dataset.features) == 11
    assert dataset.features["Time"].min() >= 150.0
    assert dataset.features["Time"].max() <= 250.0


def test_dataset_builder_capacity_and_class_checks() -> None:
    """Test RetrainingDatasetBuilder enforces sample thresholds and multi-class requirements."""
    builder = RetrainingDatasetBuilder(
        RetrainingConfig(min_training_samples=20, min_fraud_samples=3)
    )

    # Empty records
    with pytest.raises(DatasetBuildError, match="No records provided"):
        builder.build_from_records([])

    # Too few records
    records_few = _make_dummy_records(count=10, fraud_count=3)
    with pytest.raises(DatasetBuildError, match="below minimum required 20"):
        builder.build_from_records(records_few)

    # Too few fraud samples
    records_low_fraud = _make_dummy_records(count=25, fraud_count=1)
    with pytest.raises(DatasetBuildError, match="below minimum required 3"):
        builder.build_from_records(records_low_fraud)

    # Single-class dataset (only fraud)
    records_all_fraud = _make_dummy_records(count=25, fraud_count=25)
    with pytest.raises(DatasetBuildError, match="single-class dataset"):
        builder.build_from_records(records_all_fraud)


def test_dataset_builder_feature_selection_and_validation() -> None:
    """Test RetrainingDatasetBuilder feature order validation and missing feature handling."""
    records = _make_dummy_records(count=20, fraud_count=3)
    builder = RetrainingDatasetBuilder(
        RetrainingConfig(min_training_samples=10, min_fraud_samples=2)
    )

    # Reordering features
    dataset = builder.build_from_records(records, feature_names=["V2", "V1"])
    assert dataset.feature_names == ("V2", "V1")

    # Missing feature
    with pytest.raises(DatasetBuildError, match="Features missing"):
        builder.build_from_records(records, feature_names=["NonExistentFeature"])


def test_dataset_builder_from_audit_and_labels() -> None:
    """Test RetrainingDatasetBuilder matches audit logs with delayed labels."""
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        audit_file = tmp_path / "audit.jsonl"
        labels_file = tmp_path / "labels.jsonl"

        # Write audit events
        audit_lines = []
        labels_list = []
        for i in range(25):
            tx_id = f"tx_audit_{i:04d}"
            is_fraud = 1 if i < 4 else 0
            ts = 1_000.0 + float(i * 10)
            audit_lines.append(
                json.dumps(
                    {
                        "event_type": "scoring",
                        "timestamp": ts,
                        "model_version": "v1",
                        "payload": {
                            "predictions": [
                                {
                                    "transaction_id": tx_id,
                                    "fraud_probability": 0.9 if is_fraud else 0.1,
                                    "decision": "DENY" if is_fraud else "ALLOW",
                                    "amount": 100.0,
                                }
                            ],
                            "features": [{"V1": float(i), "Amount": 100.0, "Time": ts}],
                        },
                    }
                )
            )
            labels_list.append(
                DelayedLabelRecord(
                    transaction_id=tx_id,
                    event_timestamp=ts,
                    label_timestamp=ts + 50.0,
                    dispute_type=DisputeType.CHARGEBACK if is_fraud else DisputeType.FRIENDLY_FRAUD,
                    is_fraud=bool(is_fraud),
                    amount=100.0,
                )
            )

        audit_file.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
        store = LabelIngestionStore()
        for lbl in labels_list:
            store.ingest_record(lbl)
        store.save_to_jsonl(labels_file)

        builder = RetrainingDatasetBuilder(
            RetrainingConfig(min_training_samples=10, min_fraud_samples=2)
        )

        # Using LabelIngestionStore instance
        ds_from_store = builder.build_from_audit_and_labels(audit_file, store)
        assert len(ds_from_store.features) == 25
        assert ds_from_store.target.sum() == 4

        # Using labels JSONL path
        ds_from_path = builder.build_from_audit_and_labels(audit_file, labels_file)
        assert len(ds_from_path.features) == 25

        # Missing files
        with pytest.raises(DatasetBuildError, match="Audit log file not found"):
            builder.build_from_audit_and_labels(tmp_path / "missing_audit.jsonl", store)

        with pytest.raises(DatasetBuildError, match="Labels file not found"):
            builder.build_from_audit_and_labels(audit_file, tmp_path / "missing_labels.jsonl")


def test_challenger_trainer_training() -> None:
    """Test ChallengerTrainer trains and evaluates a candidate FraudModel."""
    raw_df = generate_synthetic_data(rows=250, fraud_rate=0.1, random_state=42)
    features = raw_df.drop(columns=["Class"])
    target = raw_df["Class"]
    dataset = ValidatedDataset(features=features, target=target)

    cfg = RetrainingConfig(
        estimator="logistic_regression",
        split_strategy="temporal",
        time_column="Time",
        test_size=0.2,
        validation_size=0.2,
        random_state=42,
    )
    trainer = ChallengerTrainer(cfg)
    result = trainer.train(dataset, provenance={"trigger_id": "trig_001"})

    assert isinstance(result, ChallengerTrainingResult)
    assert isinstance(result.model, FraudModel)
    assert result.model_version
    assert result.dataset_fingerprint
    assert result.training_duration_seconds > 0.0
    assert "f1" in result.validation_metrics
    assert "pr_auc" in result.validation_metrics
    assert "f1" in result.test_metrics
    assert "pr_auc" in result.test_metrics

    # Test serialization
    data = result.to_dict()
    assert data["model_version"] == result.model_version
    assert "validation_metrics" in data
    assert "test_metrics" in data
    assert "config" in data


def test_challenger_trainer_hyperparameters() -> None:
    """Test ChallengerTrainer with custom hyperparameters."""
    raw_df = generate_synthetic_data(rows=250, fraud_rate=0.1, random_state=42)
    dataset = ValidatedDataset(features=raw_df.drop(columns=["Class"]), target=raw_df["Class"])

    cfg = RetrainingConfig(
        estimator="hist_gradient_boosting",
        split_strategy="stratified",
        test_size=0.2,
        validation_size=0.2,
        hyperparameters={
            "max_depth": 3,
            "learning_rate": 0.05,
            "calibration_method": "none",
        },
    )
    trainer = ChallengerTrainer(cfg)
    result = trainer.train(dataset)
    assert isinstance(result.model, FraudModel)
    assert "HistGradientBoosting" in str(result.model.metadata.get("estimator"))


def test_challenger_trainer_invalid_configs() -> None:
    """Test ChallengerTrainer error handling for unsupported configurations."""
    raw_df = generate_synthetic_data(rows=250, fraud_rate=0.1, random_state=42)
    dataset = ValidatedDataset(features=raw_df.drop(columns=["Class"]), target=raw_df["Class"])

    # Unsupported estimator
    bad_est = ChallengerTrainer(RetrainingConfig(estimator="unsupported_estimator"))
    with pytest.raises(RetrainingError, match="Unsupported estimator type"):
        bad_est.train(dataset)

    # Unsupported split strategy
    bad_split = ChallengerTrainer(RetrainingConfig(split_strategy="invalid_split"))
    with pytest.raises(RetrainingError, match="Unsupported split strategy"):
        bad_split.train(dataset)

    # Unsupported calibration method
    bad_cal = ChallengerTrainer(
        RetrainingConfig(hyperparameters={"calibration_method": "invalid_cal"})
    )
    with pytest.raises(RetrainingError, match="Unsupported calibration method"):
        bad_cal.train(dataset)


def test_challenger_trainer_training_failure() -> None:
    """Test ChallengerTrainer handles underlying training failures gracefully."""
    # Build dataset where one split class is too small for calibration folds
    raw_df = generate_synthetic_data(rows=200, fraud_rate=0.01, random_state=42)
    # Only ~2 fraud cases across entire dataset; 3-fold CV inside split will fail
    dataset = ValidatedDataset(features=raw_df.drop(columns=["Class"]), target=raw_df["Class"])

    trainer = ChallengerTrainer(
        RetrainingConfig(
            test_size=0.3,
            validation_size=0.3,
            hyperparameters={"calibration_folds": 5},
        )
    )
    with pytest.raises(RetrainingError, match="Challenger model training failed"):
        trainer.train(dataset)
