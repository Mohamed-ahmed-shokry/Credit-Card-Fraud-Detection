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
    LabelIngestionStore,
    LabelSource,
    MatchedFeedbackRecord,
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
