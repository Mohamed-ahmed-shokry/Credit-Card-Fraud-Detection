"""Unit and integration tests for feedback ingestion and delayed label storage."""

from __future__ import annotations

import csv
import threading
from pathlib import Path

import pytest

from fraud_detection.feedback import (
    DelayedLabelRecord,
    DisputeType,
    FeedbackError,
    LabelIngestionStore,
    LabelSource,
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
