"""Delayed feedback ingestion, label maturation modeling, and realized surveillance."""

from __future__ import annotations

import csv
import json
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class FeedbackError(Exception):
    """Raised when feedback ingestion or surveillance operations encounter errors."""


class DisputeType(StrEnum):
    """Category of asynchronous dispute or ground-truth label notification."""

    CHARGEBACK = "chargeback"
    FRAUD_DISPUTE = "fraud_dispute"
    MERCHANT_INQUIRY = "merchant_inquiry"
    FRIENDLY_FRAUD = "friendly_fraud"


class LabelSource(StrEnum):
    """Origin of the delayed ground-truth feedback label."""

    NETWORK_REPORT = "network_report"
    MANUAL_REVIEW = "manual_review"
    MERCHANT_FEED = "merchant_feed"


@dataclass(frozen=True)
class DelayedLabelRecord:
    """Asynchronous ground-truth fraud label for a historically evaluated transaction."""

    transaction_id: str
    event_timestamp: float
    label_timestamp: float
    dispute_type: DisputeType
    is_fraud: bool = True
    amount: float = 0.0
    label_source: LabelSource = LabelSource.NETWORK_REPORT
    entity_key: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.transaction_id or not str(self.transaction_id).strip():
            raise FeedbackError("transaction_id must be a non-empty string.")
        if self.label_timestamp < self.event_timestamp:
            raise FeedbackError(
                f"label_timestamp ({self.label_timestamp}) cannot precede event_timestamp "
                f"({self.event_timestamp})."
            )
        if self.amount < 0.0:
            raise FeedbackError(f"amount cannot be negative, got {self.amount}.")

    @property
    def delay_seconds(self) -> float:
        """Elapsed delay in seconds between the transaction event and label arrival."""
        return self.label_timestamp - self.event_timestamp

    @property
    def delay_days(self) -> float:
        """Elapsed delay in days between transaction event and label arrival."""
        return self.delay_seconds / 86400.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize delayed label record to dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "event_timestamp": self.event_timestamp,
            "label_timestamp": self.label_timestamp,
            "dispute_type": self.dispute_type.value,
            "is_fraud": self.is_fraud,
            "amount": self.amount,
            "label_source": self.label_source.value,
            "entity_key": self.entity_key,
            "notes": self.notes,
            "delay_days": round(self.delay_days, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DelayedLabelRecord:
        """Construct DelayedLabelRecord from a dictionary."""
        if not isinstance(data, dict):
            raise FeedbackError(f"Expected dict, got {type(data).__name__}")
        try:
            raw_dispute = data["dispute_type"]
            dispute_type = (
                raw_dispute
                if isinstance(raw_dispute, DisputeType)
                else DisputeType(str(raw_dispute))
            )
            raw_source = data.get("label_source", LabelSource.NETWORK_REPORT)
            label_source = (
                raw_source if isinstance(raw_source, LabelSource) else LabelSource(str(raw_source))
            )
            return cls(
                transaction_id=str(data["transaction_id"]),
                event_timestamp=float(data["event_timestamp"]),
                label_timestamp=float(data["label_timestamp"]),
                dispute_type=dispute_type,
                is_fraud=bool(data.get("is_fraud", True)),
                amount=float(data.get("amount", 0.0)),
                label_source=label_source,
                entity_key=str(data["entity_key"]) if data.get("entity_key") is not None else None,
                notes=str(data["notes"]) if data.get("notes") is not None else None,
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise FeedbackError(f"Failed to parse DelayedLabelRecord from dict: {exc}") from exc


class LabelIngestionStore:
    """Thread-safe storage and deduplication engine for asynchronous ground-truth labels."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, DelayedLabelRecord] = {}

    def ingest_record(self, record: DelayedLabelRecord) -> bool:
        """Ingest a single label record with deduplication and fraud precedence resolution.

        Returns True if a new record was added or existing record was updated.
        """
        if not isinstance(record, DelayedLabelRecord):
            raise FeedbackError(f"Expected DelayedLabelRecord, got {type(record).__name__}")
        with self._lock:
            existing = self._records.get(record.transaction_id)
            if existing is None:
                self._records[record.transaction_id] = record
                return True

            # If existing is non-fraud and new is fraud, fraud takes precedence
            if record.is_fraud and not existing.is_fraud:
                self._records[record.transaction_id] = record
                return True

            # If same fraud status, newer label_timestamp updates
            if (
                record.is_fraud == existing.is_fraud
                and record.label_timestamp >= existing.label_timestamp
            ):
                self._records[record.transaction_id] = record
                return True

            return False

    def ingest_records(self, records: Iterable[DelayedLabelRecord]) -> int:
        """Ingest multiple label records. Returns the count of added/updated records."""
        count = 0
        for rec in records:
            if self.ingest_record(rec):
                count += 1
        return count

    def ingest_from_dicts(self, dicts: Iterable[dict[str, Any]]) -> int:
        """Parse and ingest delayed label records from dictionaries."""
        records = [DelayedLabelRecord.from_dict(d) for d in dicts]
        return self.ingest_records(records)

    def ingest_from_csv(self, path: Path | str) -> int:
        """Ingest delayed label records from a CSV file."""
        csv_path = Path(path)
        if not csv_path.is_file():
            raise FeedbackError(f"CSV file not found: {csv_path}")

        records: list[DelayedLabelRecord] = []
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records.extend(DelayedLabelRecord.from_dict(row) for row in reader)
        return self.ingest_records(records)

    def ingest_from_jsonl(self, path: Path | str) -> int:
        """Ingest delayed label records from a JSONL file."""
        jsonl_path = Path(path)
        if not jsonl_path.is_file():
            raise FeedbackError(f"JSONL file not found: {jsonl_path}")

        records: list[DelayedLabelRecord] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                    records.append(DelayedLabelRecord.from_dict(payload))
                except json.JSONDecodeError as exc:
                    raise FeedbackError(f"Malformed JSON on line {line_no}: {exc}") from exc
        return self.ingest_records(records)

    def get_record(self, transaction_id: str) -> DelayedLabelRecord | None:
        """Look up a label record by transaction ID."""
        with self._lock:
            return self._records.get(transaction_id)

    def all_records(self) -> list[DelayedLabelRecord]:
        """Return all stored label records ordered by event timestamp."""
        with self._lock:
            return sorted(
                self._records.values(), key=lambda r: (r.event_timestamp, r.label_timestamp)
            )

    def save_to_jsonl(self, path: Path | str) -> None:
        """Serialize all stored label records to a JSONL file atomically."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(f"{out_path.suffix}.tmp")
        records = self.all_records()
        with tmp_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict()) + "\n")
        tmp_path.replace(out_path)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
