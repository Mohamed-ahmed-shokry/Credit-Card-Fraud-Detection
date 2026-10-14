"""Unit and integration tests for feedback ingestion and delayed label storage."""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path

import pytest

from fraud_detection.feedback import (
    DelayedLabelRecord,
    DisputeType,
    FeedbackCohort,
    FeedbackError,
    FeedbackSurveillanceAnalyzer,
    LabelIngestionStore,
    LabelMaturationCurve,
    LabelSource,
    MatchedFeedbackRecord,
    RealizedMetricsReport,
    RetrainingTriggerEvent,
    RetrainingTriggerPolicy,
    compute_realized_metrics,
    match_labels_with_audit_log,
    partition_into_cohorts,
)


def test_delayed_label_record_creation_and_properties() -> None:
    rec = DelayedLabelRecord(
        transaction_id="tx_1001",
        event_timestamp=1000.0,
        label_timestamp=87400.0,  # 86400s (1 day) later
        dispute_type=DisputeType.CHARGEBACK,
        is_fraud=True,
        amount=150.50,
        label_source=LabelSource.NETWORK_REPORT,
        entity_key="card_999",
        notes="Network TC40 report",
    )
    assert rec.transaction_id == "tx_1001"
    assert rec.delay_seconds == 86400.0
    assert rec.delay_days == 1.0
    assert rec.is_fraud is True

    d = rec.to_dict()
    assert d["transaction_id"] == "tx_1001"
    assert d["dispute_type"] == "chargeback"
    assert d["delay_days"] == 1.0

    # Reconstruct from dict
    rec2 = DelayedLabelRecord.from_dict(d)
    assert rec2 == rec


def test_delayed_label_record_validation() -> None:
    with pytest.raises(FeedbackError, match="non-empty string"):
        DelayedLabelRecord(
            transaction_id="",
            event_timestamp=100.0,
            label_timestamp=200.0,
            dispute_type=DisputeType.CHARGEBACK,
        )

    with pytest.raises(FeedbackError, match="cannot precede"):
        DelayedLabelRecord(
            transaction_id="tx_1",
            event_timestamp=500.0,
            label_timestamp=100.0,
            dispute_type=DisputeType.CHARGEBACK,
        )

    with pytest.raises(FeedbackError, match="cannot be negative"):
        DelayedLabelRecord(
            transaction_id="tx_1",
            event_timestamp=100.0,
            label_timestamp=200.0,
            dispute_type=DisputeType.CHARGEBACK,
            amount=-10.0,
        )

    with pytest.raises(FeedbackError, match="Expected dict"):
        DelayedLabelRecord.from_dict("not_a_dict")  # type: ignore[arg-type]

    with pytest.raises(FeedbackError, match="Failed to parse"):
        DelayedLabelRecord.from_dict({"transaction_id": "tx_bad"})


def test_label_ingestion_store_crud_and_deduplication() -> None:
    store = LabelIngestionStore()
    assert len(store) == 0

    r1 = DelayedLabelRecord(
        transaction_id="tx_1",
        event_timestamp=100.0,
        label_timestamp=200.0,
        dispute_type=DisputeType.MERCHANT_INQUIRY,
        is_fraud=False,
        amount=50.0,
    )
    added = store.ingest_record(r1)
    assert added is True
    assert len(store) == 1
    assert store.get_record("tx_1") == r1
    assert store.get_record("non_existent") is None

    # Duplicate with same fraud status but older timestamp is ignored
    r1_older = DelayedLabelRecord(
        transaction_id="tx_1",
        event_timestamp=100.0,
        label_timestamp=150.0,
        dispute_type=DisputeType.MERCHANT_INQUIRY,
        is_fraud=False,
        amount=50.0,
    )
    assert store.ingest_record(r1_older) is False
    assert len(store) == 1

    # Conflict resolution: fraud takes precedence over non-fraud
    r1_fraud = DelayedLabelRecord(
        transaction_id="tx_1",
        event_timestamp=100.0,
        label_timestamp=180.0,
        dispute_type=DisputeType.CHARGEBACK,
        is_fraud=True,
        amount=50.0,
    )
    assert store.ingest_record(r1_fraud) is True
    assert len(store) == 1
    assert store.get_record("tx_1") is not None
    assert store.get_record("tx_1").is_fraud is True  # type: ignore[union-attr]

    with pytest.raises(FeedbackError, match="Expected DelayedLabelRecord"):
        store.ingest_record("invalid")  # type: ignore[arg-type]


def test_label_ingestion_store_csv_and_jsonl(tmp_path: Path) -> None:
    store = LabelIngestionStore()

    # CSV ingestion
    csv_file = tmp_path / "labels.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "transaction_id",
                "event_timestamp",
                "label_timestamp",
                "dispute_type",
                "is_fraud",
                "amount",
                "label_source",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "transaction_id": "tx_csv_1",
                "event_timestamp": "1000.0",
                "label_timestamp": "2000.0",
                "dispute_type": "chargeback",
                "is_fraud": "True",
                "amount": "120.0",
                "label_source": "network_report",
            }
        )
        writer.writerow(
            {
                "transaction_id": "tx_csv_2",
                "event_timestamp": "1500.0",
                "label_timestamp": "3000.0",
                "dispute_type": "fraud_dispute",
                "is_fraud": "true",
                "amount": "75.0",
                "label_source": "merchant_feed",
            }
        )

    count = store.ingest_from_csv(csv_file)
    assert count == 2
    assert len(store) == 2

    with pytest.raises(FeedbackError, match="CSV file not found"):
        store.ingest_from_csv(tmp_path / "missing.csv")

    # JSONL save and reload
    jsonl_file = tmp_path / "labels.jsonl"
    store.save_to_jsonl(jsonl_file)
    assert jsonl_file.is_file()

    new_store = LabelIngestionStore()
    loaded_count = new_store.ingest_from_jsonl(jsonl_file)
    assert loaded_count == 2
    assert len(new_store) == 2
    assert [r.transaction_id for r in new_store.all_records()] == ["tx_csv_1", "tx_csv_2"]

    with pytest.raises(FeedbackError, match="JSONL file not found"):
        new_store.ingest_from_jsonl(tmp_path / "missing.jsonl")

    corrupt_jsonl = tmp_path / "corrupt.jsonl"
    corrupt_jsonl.write_text("{not valid json\n", encoding="utf-8")
    with pytest.raises(FeedbackError, match="Malformed JSON on line 1"):
        new_store.ingest_from_jsonl(corrupt_jsonl)


def test_label_ingestion_store_concurrency() -> None:
    store = LabelIngestionStore()
    threads = []
    errors: list[Exception] = []

    def worker(worker_id: int) -> None:
        try:
            for i in range(50):
                tx_id = f"tx_c_{worker_id}_{i}"
                rec = DelayedLabelRecord(
                    transaction_id=tx_id,
                    event_timestamp=float(i),
                    label_timestamp=float(i + 10),
                    dispute_type=DisputeType.CHARGEBACK,
                    is_fraud=True,
                )
                store.ingest_record(rec)
        except (FeedbackError, RuntimeError, ValueError) as exc:
            errors.append(exc)

    for w in range(5):
        t = threading.Thread(target=worker, args=(w,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert not errors
    assert len(store) == 250


def test_matched_feedback_record_serialization() -> None:
    matched = MatchedFeedbackRecord(
        transaction_id="tx_m1",
        event_timestamp=1000.0,
        actual_fraud=True,
        predicted_probability=0.88,
        predicted_decision="deny",
        amount=250.0,
        label_timestamp=87400.0,
        dispute_type=DisputeType.CHARGEBACK,
        model_version="v1.0.0",
        matched_rule="rule_high_risk",
        entity_key="card_123",
    )
    assert matched.delay_seconds == 86400.0
    assert matched.delay_days == 1.0

    d = matched.to_dict()
    assert d["transaction_id"] == "tx_m1"
    assert d["actual_fraud"] is True
    assert d["predicted_decision"] == "deny"
    assert d["delay_days"] == 1.0

    recon = MatchedFeedbackRecord.from_dict(d)
    assert recon.transaction_id == matched.transaction_id
    assert recon.actual_fraud == matched.actual_fraud
    assert recon.dispute_type == DisputeType.CHARGEBACK


def test_match_labels_with_audit_log(tmp_path: Path) -> None:
    audit_file = tmp_path / "scoring_audit.jsonl"
    events = [
        # Scoring event 1: contains 2 transactions with request_id
        {
            "event_type": "scoring",
            "timestamp": 1000.0,
            "model_version": "v1.0",
            "payload": {
                "request_id": "req_1",
                "predictions": [
                    {"transaction_id": "tx_alpha", "fraud_probability": 0.95, "decision": "deny"},
                    {"transaction_id": "tx_beta", "fraud_probability": 0.15, "decision": "allow"},
                ],
                "features": [
                    {"Amount": 200.0, "card_id": "c1"},
                    {"Amount": 30.0, "card_id": "c2"},
                ],
            },
        },
        # Non-scoring event (should be skipped)
        {
            "event_type": "promotion",
            "timestamp": 1500.0,
            "payload": {"status": "promoted"},
        },
        # Scoring event 2: without transaction_id in prediction (derives from req_id)
        {
            "event_type": "scoring",
            "timestamp": 2000.0,
            "model_version": "v1.0",
            "payload": {
                "request_id": "req_2",
                "predictions": [
                    {"fraud_probability": 0.45, "is_fraud": False},
                ],
                "features": [{"Amount": 80.0}],
            },
        },
    ]

    with audit_file.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    store = LabelIngestionStore()
    store.ingest_record(
        DelayedLabelRecord(
            transaction_id="tx_alpha",
            event_timestamp=1000.0,
            label_timestamp=87400.0,
            dispute_type=DisputeType.CHARGEBACK,
            is_fraud=True,
            amount=200.0,
        )
    )

    # Matching with unmatched treated as legitimate (default)
    matched = match_labels_with_audit_log(audit_file, store, unmatched_as_legitimate=True)
    assert len(matched) == 3

    # tx_alpha is actual fraud
    alpha = next(r for r in matched if r.transaction_id == "tx_alpha")
    assert alpha.actual_fraud is True
    assert alpha.predicted_probability == 0.95
    assert alpha.dispute_type == DisputeType.CHARGEBACK
    assert alpha.delay_days == 1.0

    # tx_beta is unmatched -> actual fraud False
    beta = next(r for r in matched if r.transaction_id == "tx_beta")
    assert beta.actual_fraud is False
    assert beta.dispute_type is None
    assert beta.delay_days is None

    # Matching with unmatched omitted
    only_matched = match_labels_with_audit_log(audit_file, store, unmatched_as_legitimate=False)
    assert len(only_matched) == 1
    assert only_matched[0].transaction_id == "tx_alpha"

    # Error handling
    with pytest.raises(FeedbackError, match="Audit log file not found"):
        match_labels_with_audit_log(tmp_path / "missing_audit.jsonl", store)

    corrupt_audit = tmp_path / "corrupt_audit.jsonl"
    corrupt_audit.write_text("{malformed json\n", encoding="utf-8")
    with pytest.raises(FeedbackError, match="Malformed audit event on line 1"):
        match_labels_with_audit_log(corrupt_audit, store)


def test_partition_into_cohorts() -> None:
    records = [
        MatchedFeedbackRecord(
            transaction_id="t1",
            event_timestamp=10.0,
            actual_fraud=True,
            predicted_probability=0.9,
            predicted_decision="deny",
            label_timestamp=100.0,
        ),
        MatchedFeedbackRecord(
            transaction_id="t2",
            event_timestamp=20.0,
            actual_fraud=False,
            predicted_probability=0.1,
            predicted_decision="allow",
        ),
        MatchedFeedbackRecord(
            transaction_id="t3",
            event_timestamp=150.0,
            actual_fraud=False,
            predicted_probability=0.2,
            predicted_decision="allow",
        ),
    ]

    cohorts = partition_into_cohorts(records, cohort_window_seconds=100.0)
    assert len(cohorts) == 2
    assert isinstance(cohorts[0], FeedbackCohort)
    assert cohorts[0].cohort_id == "cohort_0"
    assert cohorts[0].total_count == 2
    assert cohorts[0].fraud_count == 1
    assert cohorts[0].fraud_rate == 0.5
    assert cohorts[0].mean_delay_days > 0.0

    assert cohorts[1].cohort_id == "cohort_1"
    assert cohorts[1].total_count == 1
    assert cohorts[1].fraud_count == 0

    d = cohorts[0].to_dict()
    assert d["total_count"] == 2
    assert d["fraud_count"] == 1

    # Empty records
    assert partition_into_cohorts([]) == []

    # Invalid window seconds
    with pytest.raises(FeedbackError, match="must be positive"):
        partition_into_cohorts(records, cohort_window_seconds=-10.0)


def test_label_maturation_curve() -> None:
    # Empty records
    empty_curve = LabelMaturationCurve.fit_from_records([])
    assert empty_curve.total_labels_observed == 0
    assert empty_curve.maturation_fraction(10.0) == 0.0

    # Fit with realistic label delays (days)
    delays = [5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0]
    records = [
        DelayedLabelRecord(
            transaction_id=f"tx_{i}",
            event_timestamp=1000.0,
            label_timestamp=1000.0 + d * 86400.0,
            dispute_type=DisputeType.CHARGEBACK,
        )
        for i, d in enumerate(delays)
    ]

    curve = LabelMaturationCurve.fit_from_records(records)
    assert curve.total_labels_observed == 8
    assert curve.p50_delay_days > 0.0
    assert curve.max_delay_days == 90.0

    # Test maturation fraction
    assert curve.maturation_fraction(0.0) == 0.0
    assert curve.maturation_fraction(-5.0) == 0.0
    assert curve.maturation_fraction(90.0) == 1.0
    assert curve.maturation_fraction(120.0) == 1.0
    assert 0.0 < curve.maturation_fraction(30.0) < 1.0

    # Test cohort maturity filtering
    cohort_early = FeedbackCohort(
        cohort_id="c_early",
        start_timestamp=1000.0,
        end_timestamp=2000.0,
        records=(),
    )
    cohort_recent = FeedbackCohort(
        cohort_id="c_recent",
        start_timestamp=1000.0 + 85 * 86400.0,
        end_timestamp=1000.0 + 89 * 86400.0,
        records=(),
    )

    # Current time is 100 days after t=1000
    now_ts = 1000.0 + 100 * 86400.0

    # c_early ended at t=2000 (~0.01 days), elapsed is ~100 days -> >90 days -> 100% mature
    assert curve.is_cohort_mature(cohort_early, now_ts, min_maturation_fraction=0.90) is True

    # c_recent ended at day 89, elapsed is only 11 days -> maturation fraction is low (<90%)
    assert curve.is_cohort_mature(cohort_recent, now_ts, min_maturation_fraction=0.90) is False

    # Filter mature cohorts
    mature = curve.filter_mature_cohorts(
        [cohort_early, cohort_recent],
        now_ts,
        min_maturation_fraction=0.90,
    )
    assert len(mature) == 1
    assert mature[0].cohort_id == "c_early"

    # Filter with min_elapsed_days
    mature_elapsed = curve.filter_mature_cohorts(
        [cohort_early, cohort_recent],
        now_ts,
        min_elapsed_days=150.0,
    )
    assert len(mature_elapsed) == 0

    # Serialization
    d = curve.to_dict()
    assert d["total_labels_observed"] == 8
    assert d["max_delay_days"] == 90.0

    recon = LabelMaturationCurve.from_dict(d)
    assert recon.total_labels_observed == curve.total_labels_observed
    assert recon.max_delay_days == curve.max_delay_days
    assert len(recon.delay_cdf_points) == len(curve.delay_cdf_points)


def test_compute_realized_metrics() -> None:
    # Empty records
    empty_report = compute_realized_metrics([])
    assert empty_report.total_transactions == 0
    assert empty_report.realized_precision == 0.0
    assert empty_report.realized_auprc == 0.0

    # Populated records
    records = [
        MatchedFeedbackRecord(
            transaction_id="t1",
            event_timestamp=100.0,
            actual_fraud=True,
            predicted_probability=0.9,
            predicted_decision="deny",
            amount=100.0,
        ),
        MatchedFeedbackRecord(
            transaction_id="t2",
            event_timestamp=105.0,
            actual_fraud=False,
            predicted_probability=0.6,
            predicted_decision="review",
            amount=50.0,
        ),
        MatchedFeedbackRecord(
            transaction_id="t3",
            event_timestamp=110.0,
            actual_fraud=False,
            predicted_probability=0.05,
            predicted_decision="allow",
            amount=20.0,
        ),
        MatchedFeedbackRecord(
            transaction_id="t4",
            event_timestamp=115.0,
            actual_fraud=True,
            predicted_probability=0.2,
            predicted_decision="allow",
            amount=80.0,
        ),
    ]

    report = compute_realized_metrics(
        records,
        positive_decisions=("deny", "review"),
        manual_review_cost=15.0,
        cohort_id="c_test",
    )
    assert report.total_transactions == 4
    assert report.actual_fraud_count == 2
    assert report.actual_fraud_rate == 0.5
    assert report.predicted_positive_count == 2
    assert report.true_positives == 1
    assert report.false_positives == 1
    assert report.true_negatives == 1
    assert report.false_negatives == 1
    assert report.realized_precision == 0.5
    assert report.realized_recall == 0.5
    assert report.realized_f1 == 0.5
    assert report.realized_fpr == 0.5
    assert report.total_fraud_loss == 180.0
    assert report.prevented_fraud_amount == 100.0
    assert report.missed_fraud_amount == 80.0
    assert report.false_positive_cost == 15.0
    assert report.net_financial_savings == 85.0
    assert report.decision_counts == {"deny": 1, "review": 1, "allow": 2}
    assert report.realized_auprc > 0.0
    assert report.cohort_id == "c_test"

    # Serialization
    d = report.to_dict()
    assert d["total_transactions"] == 4
    assert d["net_financial_savings"] == 85.0

    recon = RealizedMetricsReport.from_dict(d)
    assert recon.total_transactions == report.total_transactions
    assert recon.realized_precision == report.realized_precision
    assert recon.net_financial_savings == report.net_financial_savings


def test_retraining_trigger_policy_validation_and_serialization() -> None:
    # Default policy
    policy = RetrainingTriggerPolicy()
    assert policy.min_precision == 0.70
    assert policy.min_auprc == 0.60
    assert policy.max_fpr == 0.05
    assert policy.max_missed_fraud_rate == 0.30
    assert policy.min_matured_transactions == 10
    assert policy.recommended_retrain_window_days == 90.0
    assert policy.recommended_temporal_gap_days == 14.0

    d = policy.to_dict()
    recon = RetrainingTriggerPolicy.from_dict(d)
    assert recon == policy

    # Validation errors
    with pytest.raises(FeedbackError, match="min_precision"):
        RetrainingTriggerPolicy(min_precision=-0.1)
    with pytest.raises(FeedbackError, match="min_precision"):
        RetrainingTriggerPolicy(min_precision=1.1)

    with pytest.raises(FeedbackError, match="min_auprc"):
        RetrainingTriggerPolicy(min_auprc=-0.1)
    with pytest.raises(FeedbackError, match="min_auprc"):
        RetrainingTriggerPolicy(min_auprc=1.1)

    with pytest.raises(FeedbackError, match="max_fpr"):
        RetrainingTriggerPolicy(max_fpr=-0.1)
    with pytest.raises(FeedbackError, match="max_fpr"):
        RetrainingTriggerPolicy(max_fpr=1.1)

    with pytest.raises(FeedbackError, match="max_missed_fraud_rate"):
        RetrainingTriggerPolicy(max_missed_fraud_rate=-0.1)
    with pytest.raises(FeedbackError, match="max_missed_fraud_rate"):
        RetrainingTriggerPolicy(max_missed_fraud_rate=1.1)

    with pytest.raises(FeedbackError, match="min_matured_transactions"):
        RetrainingTriggerPolicy(min_matured_transactions=0)

    with pytest.raises(FeedbackError, match="recommended_retrain_window_days"):
        RetrainingTriggerPolicy(recommended_retrain_window_days=0.0)

    with pytest.raises(FeedbackError, match="recommended_temporal_gap_days"):
        RetrainingTriggerPolicy(recommended_temporal_gap_days=-1.0)


def test_retraining_trigger_event_serialization() -> None:
    rep = RealizedMetricsReport(
        total_transactions=10,
        actual_fraud_count=3,
        actual_fraud_rate=0.3,
        predicted_positive_count=4,
        true_positives=2,
        false_positives=2,
        true_negatives=5,
        false_negatives=1,
        realized_precision=0.5,
        realized_recall=0.6667,
        realized_f1=0.5714,
        realized_fpr=0.2857,
        realized_auprc=0.45,
        total_fraud_loss=300.0,
        prevented_fraud_amount=200.0,
        missed_fraud_amount=100.0,
        false_positive_cost=30.0,
        net_financial_savings=170.0,
        decision_counts={"deny": 4, "allow": 6},
        cohort_id="c_2026_01",
    )
    event = RetrainingTriggerEvent(
        trigger_id="trig_1",
        triggered_at=1000000.0,
        breached_metrics={
            "precision": {"observed": 0.5, "threshold": 0.7},
            "auprc": {"observed": 0.45, "threshold": 0.6},
        },
        realized_metrics=rep,
        recommended_retrain_start=100000.0,
        recommended_retrain_end=900000.0,
        recommended_temporal_gap_seconds=100000.0,
        reason="precision breached; auprc breached",
        cohort_id="c_2026_01",
    )

    d = event.to_dict()
    assert d["trigger_id"] == "trig_1"
    assert d["cohort_id"] == "c_2026_01"

    recon = RetrainingTriggerEvent.from_dict(d)
    assert recon.trigger_id == event.trigger_id
    assert recon.breached_metrics == event.breached_metrics
    assert recon.realized_metrics.total_transactions == 10


def test_feedback_surveillance_analyzer_insufficient_samples() -> None:
    analyzer = FeedbackSurveillanceAnalyzer()
    records = [
        MatchedFeedbackRecord(
            transaction_id="t1",
            event_timestamp=100.0,
            actual_fraud=False,
            predicted_probability=0.01,
            predicted_decision="allow",
            amount=50.0,
        )
    ]
    report, trigger = analyzer.evaluate_cohort(records, current_timestamp=1000.0)
    assert report.total_transactions == 1
    assert trigger is None


def test_feedback_surveillance_analyzer_no_breach() -> None:
    analyzer = FeedbackSurveillanceAnalyzer()
    # High performance records exceeding default policy
    records: list[MatchedFeedbackRecord] = [
        MatchedFeedbackRecord(
            transaction_id=f"neg_{i}",
            event_timestamp=100.0,
            actual_fraud=False,
            predicted_probability=0.01,
            predicted_decision="allow",
            amount=50.0,
        )
        for i in range(8)
    ]
    records.extend(
        MatchedFeedbackRecord(
            transaction_id=f"pos_{i}",
            event_timestamp=100.0,
            actual_fraud=True,
            predicted_probability=0.95,
            predicted_decision="deny",
            amount=200.0,
        )
        for i in range(4)
    )

    report, trigger = analyzer.evaluate_cohort(records, current_timestamp=1000.0)
    assert report.total_transactions == 12
    assert report.realized_precision == 1.0
    assert report.realized_fpr == 0.0
    assert trigger is None


def test_feedback_surveillance_analyzer_breach_detection() -> None:
    analyzer = FeedbackSurveillanceAnalyzer()
    policy = RetrainingTriggerPolicy(
        min_precision=0.8,
        min_auprc=0.7,
        max_fpr=0.05,
        max_missed_fraud_rate=0.2,
        min_matured_transactions=5,
        recommended_retrain_window_days=30.0,
        recommended_temporal_gap_days=7.0,
    )

    # Build cohort with poor precision and missed frauds
    records: list[MatchedFeedbackRecord] = [
        MatchedFeedbackRecord(
            transaction_id=f"fp_{i}",
            event_timestamp=100.0,
            actual_fraud=False,
            predicted_probability=0.8,
            predicted_decision="deny",
            amount=50.0,
        )
        for i in range(5)
    ]
    records.extend(
        MatchedFeedbackRecord(
            transaction_id=f"fn_{i}",
            event_timestamp=100.0,
            actual_fraud=True,
            predicted_probability=0.1,
            predicted_decision="allow",
            amount=100.0,
        )
        for i in range(5)
    )

    now = 1000000.0
    report, trigger = analyzer.evaluate_cohort(
        records,
        policy=policy,
        current_timestamp=now,
        cohort_id="c_breach",
    )
    assert report.total_transactions == 10
    assert trigger is not None
    assert "precision" in trigger.breached_metrics
    assert "fpr" in trigger.breached_metrics
    assert "missed_fraud_rate" in trigger.breached_metrics
    assert trigger.cohort_id == "c_breach"
    assert trigger.triggered_at == now
    gap_sec = 7.0 * 86400.0
    assert trigger.recommended_temporal_gap_seconds == gap_sec
    assert trigger.recommended_retrain_end == now - gap_sec
    assert trigger.recommended_retrain_start == now - gap_sec - (30.0 * 86400.0)


def test_feedback_coverage_edge_cases(tmp_path: Path) -> None:
    # 1. DelayedLabelRecord same fraud status update if newer label_timestamp (lines 152-153)
    store = LabelIngestionStore()
    r1 = DelayedLabelRecord(
        transaction_id="tx_up",
        event_timestamp=100.0,
        label_timestamp=200.0,
        dispute_type=DisputeType.MERCHANT_INQUIRY,
        is_fraud=True,
    )
    r2 = DelayedLabelRecord(
        transaction_id="tx_up",
        event_timestamp=100.0,
        label_timestamp=300.0,
        dispute_type=DisputeType.CHARGEBACK,
        is_fraud=True,
    )
    assert store.ingest_record(r1) is True
    assert store.ingest_record(r2) is True
    assert store.get_record("tx_up").label_timestamp == 300.0  # type: ignore[union-attr]

    # 2. ingest_from_dicts (lines 167-168)
    dict_records = [
        {
            "transaction_id": "tx_d1",
            "event_timestamp": 100.0,
            "label_timestamp": 200.0,
            "dispute_type": "chargeback",
            "is_fraud": True,
        }
    ]
    assert store.ingest_from_dicts(dict_records) == 1

    # 3. ingest_from_jsonl with empty lines (line 193)
    jsonl_path = tmp_path / "labels.jsonl"
    jsonl_path.write_text(
        "\n"
        + json.dumps(
            {
                "transaction_id": "tx_j1",
                "event_timestamp": 100.0,
                "label_timestamp": 200.0,
                "dispute_type": "chargeback",
                "is_fraud": True,
            }
        )
        + "\n\n",
        encoding="utf-8",
    )
    assert store.ingest_from_jsonl(jsonl_path) == 1

    # 4. match_labels_with_audit_log with Sequence[DelayedLabelRecord] & empty line in audit log
    # (lines 357, 364)
    audit_file = tmp_path / "audit.jsonl"
    audit_file.write_text(
        "\n"
        + json.dumps(
            {
                "event_type": "scoring",
                "timestamp": 100.0,
                "payload": {
                    "predictions": [
                        {
                            "transaction_id": "tx_j1",
                            "fraud_probability": 0.85,
                            "decision": "deny",
                            "amount": 120.0,
                        }
                    ]
                },
            }
        )
        + "\n\n",
        encoding="utf-8",
    )
    seq_records = [
        DelayedLabelRecord(
            transaction_id="tx_j1",
            event_timestamp=100.0,
            label_timestamp=200.0,
            dispute_type=DisputeType.CHARGEBACK,
            is_fraud=True,
        )
    ]
    matched = match_labels_with_audit_log(audit_file, seq_records)
    assert len(matched) == 1
    assert matched[0].transaction_id == "tx_j1"

    # Ingest records when duplicate older record is ignored (returns False, lines 161->160)
    older_rec = DelayedLabelRecord(
        transaction_id="tx_up",
        event_timestamp=100.0,
        label_timestamp=250.0,
        dispute_type=DisputeType.CHARGEBACK,
        is_fraud=True,
    )
    assert store.ingest_records([older_rec]) == 0

    # 5. fit_from_records with non-empty delays but all actual frauds (auprc 1.0)
    all_fraud_records = [
        MatchedFeedbackRecord(
            transaction_id="f1",
            event_timestamp=100.0,
            actual_fraud=True,
            predicted_probability=0.9,
            predicted_decision="deny",
            amount=50.0,
            label_timestamp=100.0 + 1.5 * 86400.0,
        ),
        MatchedFeedbackRecord(
            transaction_id="f2",
            event_timestamp=100.0,
            actual_fraud=True,
            predicted_probability=0.8,
            predicted_decision="deny",
            amount=50.0,
            label_timestamp=100.0 + 2.5 * 86400.0,
        ),
    ]
    rep = compute_realized_metrics(all_fraud_records)
    assert rep.realized_auprc == 1.0

    # 6. fit_from_records with delay_days None or negative (lines 561->559)
    class DummyRecord:
        delay_days = -5.0

    curve = LabelMaturationCurve.fit_from_records([DummyRecord()])  # type: ignore[list-item]
    assert curve.total_labels_observed == 0

    # 7. Surveillance when actual_fraud_count is 0 (line 1005->1013)
    analyzer = FeedbackSurveillanceAnalyzer()
    policy_missed_only = RetrainingTriggerPolicy(
        min_precision=None,
        min_auprc=None,
        max_fpr=None,
        max_missed_fraud_rate=0.2,
        min_matured_transactions=10,
    )
    zero_fraud_records = [
        MatchedFeedbackRecord(
            transaction_id=f"zf_{i}",
            event_timestamp=100.0,
            actual_fraud=False,
            predicted_probability=0.01,
            predicted_decision="allow",
            amount=20.0,
        )
        for i in range(12)
    ]
    rep_zf, trig_zf = analyzer.evaluate_cohort(zero_fraud_records, policy=policy_missed_only)
    assert rep_zf.actual_fraud_count == 0
    assert trig_zf is None
