"""Delayed feedback ingestion, label maturation modeling, and realized surveillance."""

from __future__ import annotations

import csv
import json
import threading
from collections.abc import Iterable, Sequence
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


@dataclass(frozen=True)
class MatchedFeedbackRecord:
    """Historical transaction decision joined with delayed ground-truth feedback."""

    transaction_id: str
    event_timestamp: float
    actual_fraud: bool
    predicted_probability: float
    predicted_decision: str
    amount: float = 0.0
    label_timestamp: float | None = None
    dispute_type: DisputeType | None = None
    model_version: str | None = None
    matched_rule: str | None = None
    entity_key: str | None = None

    @property
    def delay_seconds(self) -> float | None:
        """Elapsed delay in seconds if a ground-truth label was matched."""
        if self.label_timestamp is None:
            return None
        return max(0.0, self.label_timestamp - self.event_timestamp)

    @property
    def delay_days(self) -> float | None:
        """Elapsed delay in days if a ground-truth label was matched."""
        sec = self.delay_seconds
        return sec / 86400.0 if sec is not None else None

    def to_dict(self) -> dict[str, Any]:
        """Serialize matched record to dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "event_timestamp": self.event_timestamp,
            "actual_fraud": self.actual_fraud,
            "predicted_probability": round(self.predicted_probability, 6),
            "predicted_decision": self.predicted_decision,
            "amount": self.amount,
            "label_timestamp": self.label_timestamp,
            "dispute_type": self.dispute_type.value if self.dispute_type is not None else None,
            "model_version": self.model_version,
            "matched_rule": self.matched_rule,
            "entity_key": self.entity_key,
            "delay_days": round(self.delay_days, 4) if self.delay_days is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatchedFeedbackRecord:
        """Reconstruct matched record from dictionary."""
        raw_dispute = data.get("dispute_type")
        dispute_type = (
            raw_dispute
            if isinstance(raw_dispute, DisputeType) or raw_dispute is None
            else DisputeType(str(raw_dispute))
        )
        return cls(
            transaction_id=str(data["transaction_id"]),
            event_timestamp=float(data["event_timestamp"]),
            actual_fraud=bool(data["actual_fraud"]),
            predicted_probability=float(data["predicted_probability"]),
            predicted_decision=str(data["predicted_decision"]),
            amount=float(data.get("amount", 0.0)),
            label_timestamp=(
                float(data["label_timestamp"]) if data.get("label_timestamp") is not None else None
            ),
            dispute_type=dispute_type,
            model_version=str(data["model_version"])
            if data.get("model_version") is not None
            else None,
            matched_rule=str(data["matched_rule"])
            if data.get("matched_rule") is not None
            else None,
            entity_key=str(data["entity_key"]) if data.get("entity_key") is not None else None,
        )


@dataclass(frozen=True)
class FeedbackCohort:
    """Cohort of historical transactions grouped by observation time window."""

    cohort_id: str
    start_timestamp: float
    end_timestamp: float
    records: tuple[MatchedFeedbackRecord, ...]

    @property
    def total_count(self) -> int:
        return len(self.records)

    @property
    def fraud_count(self) -> int:
        return sum(1 for r in self.records if r.actual_fraud)

    @property
    def fraud_rate(self) -> float:
        return self.fraud_count / self.total_count if self.total_count > 0 else 0.0

    @property
    def mean_delay_days(self) -> float:
        delays = [r.delay_days for r in self.records if r.delay_days is not None]
        return sum(delays) / len(delays) if delays else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort_id": self.cohort_id,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "total_count": self.total_count,
            "fraud_count": self.fraud_count,
            "fraud_rate": round(self.fraud_rate, 4),
            "mean_delay_days": round(self.mean_delay_days, 4),
        }


def match_labels_with_audit_log(
    audit_log_path: Path | str,
    label_store: LabelIngestionStore | Iterable[DelayedLabelRecord],
    *,
    unmatched_as_legitimate: bool = True,
) -> list[MatchedFeedbackRecord]:
    """Match historical scoring audit log events with ingested ground-truth labels."""
    audit_file = Path(audit_log_path)
    if not audit_file.is_file():
        raise FeedbackError(f"Audit log file not found: {audit_file}")

    if isinstance(label_store, LabelIngestionStore):
        labels_map = {r.transaction_id: r for r in label_store.all_records()}
    else:
        labels_map = {r.transaction_id: r for r in label_store}

    matched_records: list[MatchedFeedbackRecord] = []
    with audit_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                ev = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise FeedbackError(f"Malformed audit event on line {line_no}: {exc}") from exc

            if ev.get("event_type") != "scoring":
                continue

            payload = ev.get("payload", {})
            event_ts = float(ev.get("timestamp", 0.0))
            model_ver = ev.get("model_version")
            req_id = payload.get("request_id")
            predictions = payload.get("predictions", [])
            features = payload.get("features", [])

            for idx, pred in enumerate(predictions):
                # Derive transaction id from prediction payload, request_id, or feature row
                tx_id = (
                    pred.get("transaction_id")
                    or (f"{req_id}_{idx}" if req_id else None)
                    or (
                        str(features[idx].get("transaction_id"))
                        if idx < len(features)
                        and isinstance(features[idx], dict)
                        and features[idx].get("transaction_id")
                        else f"tx_line{line_no}_{idx}"
                    )
                )

                prob = float(pred.get("fraud_probability", 0.0))
                dec = str(pred.get("decision", "deny" if pred.get("is_fraud") else "allow")).lower()
                amount = float(
                    pred.get("amount")
                    or (
                        features[idx].get("Amount")
                        if idx < len(features)
                        and isinstance(features[idx], dict)
                        and features[idx].get("Amount") is not None
                        else 0.0
                    )
                )
                matched_rule = pred.get("matched_rule")
                entity_key = pred.get("entity_key") or (
                    features[idx].get("card_id")
                    if idx < len(features) and isinstance(features[idx], dict)
                    else None
                )

                lbl = labels_map.get(str(tx_id))
                if lbl is not None:
                    matched_records.append(
                        MatchedFeedbackRecord(
                            transaction_id=str(tx_id),
                            event_timestamp=lbl.event_timestamp
                            if lbl.event_timestamp > 0
                            else event_ts,
                            actual_fraud=lbl.is_fraud,
                            predicted_probability=prob,
                            predicted_decision=dec,
                            amount=lbl.amount if lbl.amount > 0 else amount,
                            label_timestamp=lbl.label_timestamp,
                            dispute_type=lbl.dispute_type,
                            model_version=model_ver,
                            matched_rule=matched_rule,
                            entity_key=lbl.entity_key or entity_key,
                        )
                    )
                elif unmatched_as_legitimate:
                    matched_records.append(
                        MatchedFeedbackRecord(
                            transaction_id=str(tx_id),
                            event_timestamp=event_ts,
                            actual_fraud=False,
                            predicted_probability=prob,
                            predicted_decision=dec,
                            amount=amount,
                            label_timestamp=None,
                            dispute_type=None,
                            model_version=model_ver,
                            matched_rule=matched_rule,
                            entity_key=entity_key,
                        )
                    )

    return sorted(matched_records, key=lambda r: r.event_timestamp)


def partition_into_cohorts(
    records: Sequence[MatchedFeedbackRecord],
    *,
    cohort_window_seconds: float = 86400.0,
) -> list[FeedbackCohort]:
    """Partition matched records into sequential temporal cohorts."""
    if cohort_window_seconds <= 0.0:
        raise FeedbackError(f"cohort_window_seconds must be positive, got {cohort_window_seconds}")
    if not records:
        return []

    sorted_records = sorted(records, key=lambda r: r.event_timestamp)
    min_ts = sorted_records[0].event_timestamp
    max_ts = sorted_records[-1].event_timestamp

    cohorts: list[FeedbackCohort] = []
    cohort_start = min_ts
    cohort_idx = 0

    while cohort_start <= max_ts:
        cohort_end = cohort_start + cohort_window_seconds
        c_records = [
            r
            for r in sorted_records
            if cohort_start <= r.event_timestamp < cohort_end
            or (r.event_timestamp == max_ts and cohort_end > max_ts)
        ]
        cohorts.append(
            FeedbackCohort(
                cohort_id=f"cohort_{cohort_idx}",
                start_timestamp=cohort_start,
                end_timestamp=cohort_end,
                records=tuple(c_records),
            )
        )
        cohort_start = cohort_end
        cohort_idx += 1

    return cohorts
