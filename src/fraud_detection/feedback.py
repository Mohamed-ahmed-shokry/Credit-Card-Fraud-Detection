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

import numpy as np
from sklearn.metrics import average_precision_score


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


@dataclass(frozen=True)
class LabelMaturationCurve:
    """Empirical maturation curve estimating cumulative label arrival over elapsed days."""

    total_labels_observed: int
    p50_delay_days: float
    p75_delay_days: float
    p90_delay_days: float
    p95_delay_days: float
    p99_delay_days: float
    max_delay_days: float
    delay_cdf_points: tuple[tuple[float, float], ...]

    def maturation_fraction(self, elapsed_days: float) -> float:
        """Estimate cumulative proportion of labels arrived by elapsed_days."""
        if elapsed_days <= 0.0 or self.total_labels_observed == 0:
            return 0.0
        if not self.delay_cdf_points or elapsed_days >= self.max_delay_days:
            return 1.0

        days = [pt[0] for pt in self.delay_cdf_points]
        fractions = [pt[1] for pt in self.delay_cdf_points]
        return float(np.interp(elapsed_days, days, fractions))

    def is_cohort_mature(
        self,
        cohort: FeedbackCohort,
        current_timestamp: float,
        *,
        min_maturation_fraction: float = 0.90,
        min_elapsed_days: float | None = None,
    ) -> bool:
        """Return True if the cohort has matured sufficiently for unbiased evaluation."""
        elapsed_sec = max(0.0, current_timestamp - cohort.end_timestamp)
        elapsed_days = elapsed_sec / 86400.0
        if min_elapsed_days is not None and elapsed_days < min_elapsed_days:
            return False
        return self.maturation_fraction(elapsed_days) >= min_maturation_fraction

    def filter_mature_cohorts(
        self,
        cohorts: Sequence[FeedbackCohort],
        current_timestamp: float,
        *,
        min_maturation_fraction: float = 0.90,
        min_elapsed_days: float | None = None,
    ) -> list[FeedbackCohort]:
        """Filter input cohorts to only include matured cohorts."""
        return [
            c
            for c in cohorts
            if self.is_cohort_mature(
                c,
                current_timestamp,
                min_maturation_fraction=min_maturation_fraction,
                min_elapsed_days=min_elapsed_days,
            )
        ]

    @classmethod
    def fit_from_records(
        cls,
        records: Iterable[DelayedLabelRecord | MatchedFeedbackRecord],
    ) -> LabelMaturationCurve:
        """Fit empirical maturation curve from observed delay durations."""
        delays: list[float] = []
        for r in records:
            d = getattr(r, "delay_days", None)
            if d is not None and d >= 0.0:
                delays.append(float(d))

        if not delays:
            return cls(
                total_labels_observed=0,
                p50_delay_days=0.0,
                p75_delay_days=0.0,
                p90_delay_days=0.0,
                p95_delay_days=0.0,
                p99_delay_days=0.0,
                max_delay_days=0.0,
                delay_cdf_points=(),
            )

        delays_arr = np.sort(np.array(delays, dtype=float))
        n = len(delays_arr)
        p50 = float(np.percentile(delays_arr, 50))
        p75 = float(np.percentile(delays_arr, 75))
        p90 = float(np.percentile(delays_arr, 90))
        p95 = float(np.percentile(delays_arr, 95))
        p99 = float(np.percentile(delays_arr, 99))
        max_d = float(delays_arr[-1])

        # Sample up to 100 quantiles for CDF curve
        grid = np.linspace(0.01, 1.0, min(n, 100))
        q_days = np.quantile(delays_arr, grid)
        cdf_points = tuple(
            (float(round(day, 4)), float(round(q, 4))) for day, q in zip(q_days, grid, strict=True)
        )

        return cls(
            total_labels_observed=n,
            p50_delay_days=round(p50, 4),
            p75_delay_days=round(p75, 4),
            p90_delay_days=round(p90, 4),
            p95_delay_days=round(p95, 4),
            p99_delay_days=round(p99, 4),
            max_delay_days=round(max_d, 4),
            delay_cdf_points=cdf_points,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize maturation curve to dictionary."""
        return {
            "total_labels_observed": self.total_labels_observed,
            "p50_delay_days": self.p50_delay_days,
            "p75_delay_days": self.p75_delay_days,
            "p90_delay_days": self.p90_delay_days,
            "p95_delay_days": self.p95_delay_days,
            "p99_delay_days": self.p99_delay_days,
            "max_delay_days": self.max_delay_days,
            "delay_cdf_points": [list(pt) for pt in self.delay_cdf_points],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabelMaturationCurve:
        """Construct maturation curve from dictionary."""
        raw_pts = data.get("delay_cdf_points", [])
        pts = tuple((float(pt[0]), float(pt[1])) for pt in raw_pts)
        return cls(
            total_labels_observed=int(data["total_labels_observed"]),
            p50_delay_days=float(data["p50_delay_days"]),
            p75_delay_days=float(data["p75_delay_days"]),
            p90_delay_days=float(data["p90_delay_days"]),
            p95_delay_days=float(data["p95_delay_days"]),
            p99_delay_days=float(data["p99_delay_days"]),
            max_delay_days=float(data["max_delay_days"]),
            delay_cdf_points=pts,
        )


@dataclass(frozen=True)
class RealizedMetricsReport:
    """Realized performance and business impact metrics computed on matured feedback."""

    total_transactions: int
    actual_fraud_count: int
    actual_fraud_rate: float
    predicted_positive_count: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    realized_precision: float
    realized_recall: float
    realized_f1: float
    realized_fpr: float
    realized_auprc: float
    total_fraud_loss: float
    prevented_fraud_amount: float
    missed_fraud_amount: float
    false_positive_cost: float
    net_financial_savings: float
    decision_counts: dict[str, int]
    cohort_id: str | None = None
    start_timestamp: float | None = None
    end_timestamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize realized metrics report to dictionary."""
        return {
            "total_transactions": self.total_transactions,
            "actual_fraud_count": self.actual_fraud_count,
            "actual_fraud_rate": round(self.actual_fraud_rate, 4),
            "predicted_positive_count": self.predicted_positive_count,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "realized_precision": round(self.realized_precision, 4),
            "realized_recall": round(self.realized_recall, 4),
            "realized_f1": round(self.realized_f1, 4),
            "realized_fpr": round(self.realized_fpr, 4),
            "realized_auprc": round(self.realized_auprc, 4),
            "total_fraud_loss": round(self.total_fraud_loss, 2),
            "prevented_fraud_amount": round(self.prevented_fraud_amount, 2),
            "missed_fraud_amount": round(self.missed_fraud_amount, 2),
            "false_positive_cost": round(self.false_positive_cost, 2),
            "net_financial_savings": round(self.net_financial_savings, 2),
            "decision_counts": dict(self.decision_counts),
            "cohort_id": self.cohort_id,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RealizedMetricsReport:
        """Construct report from dictionary."""
        return cls(
            total_transactions=int(data["total_transactions"]),
            actual_fraud_count=int(data["actual_fraud_count"]),
            actual_fraud_rate=float(data["actual_fraud_rate"]),
            predicted_positive_count=int(data["predicted_positive_count"]),
            true_positives=int(data["true_positives"]),
            false_positives=int(data["false_positives"]),
            true_negatives=int(data["true_negatives"]),
            false_negatives=int(data["false_negatives"]),
            realized_precision=float(data["realized_precision"]),
            realized_recall=float(data["realized_recall"]),
            realized_f1=float(data["realized_f1"]),
            realized_fpr=float(data["realized_fpr"]),
            realized_auprc=float(data["realized_auprc"]),
            total_fraud_loss=float(data["total_fraud_loss"]),
            prevented_fraud_amount=float(data["prevented_fraud_amount"]),
            missed_fraud_amount=float(data["missed_fraud_amount"]),
            false_positive_cost=float(data["false_positive_cost"]),
            net_financial_savings=float(data["net_financial_savings"]),
            decision_counts=dict(data.get("decision_counts", {})),
            cohort_id=data.get("cohort_id"),
            start_timestamp=data.get("start_timestamp"),
            end_timestamp=data.get("end_timestamp"),
        )


def compute_realized_metrics(
    records: Sequence[MatchedFeedbackRecord],
    *,
    positive_decisions: tuple[str, ...] = ("deny", "review", "challenge"),
    manual_review_cost: float = 15.0,
    cohort_id: str | None = None,
    start_timestamp: float | None = None,
    end_timestamp: float | None = None,
) -> RealizedMetricsReport:
    """Compute realized accuracy, precision/recall, AUPRC, and financial impact metrics."""
    if not records:
        return RealizedMetricsReport(
            total_transactions=0,
            actual_fraud_count=0,
            actual_fraud_rate=0.0,
            predicted_positive_count=0,
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
            realized_precision=0.0,
            realized_recall=0.0,
            realized_f1=0.0,
            realized_fpr=0.0,
            realized_auprc=0.0,
            total_fraud_loss=0.0,
            prevented_fraud_amount=0.0,
            missed_fraud_amount=0.0,
            false_positive_cost=0.0,
            net_financial_savings=0.0,
            decision_counts={},
            cohort_id=cohort_id,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )

    norm_pos = tuple(d.lower() for d in positive_decisions)
    total = len(records)
    tp = fp = tn = fn = 0
    fraud_count = 0
    pred_pos_count = 0
    total_loss = 0.0
    prevented_amt = 0.0
    missed_amt = 0.0
    dec_counts: dict[str, int] = {}

    y_true: list[int] = []
    y_score: list[float] = []

    for r in records:
        dec = r.predicted_decision.lower()
        dec_counts[dec] = dec_counts.get(dec, 0) + 1
        is_pred_pos = dec in norm_pos
        is_actual = r.actual_fraud

        y_true.append(1 if is_actual else 0)
        y_score.append(r.predicted_probability)

        if is_actual:
            fraud_count += 1
            total_loss += r.amount

        if is_pred_pos:
            pred_pos_count += 1

        if is_actual and is_pred_pos:
            tp += 1
            prevented_amt += r.amount
        elif not is_actual and is_pred_pos:
            fp += 1
        elif not is_actual and not is_pred_pos:
            tn += 1
        else:
            fn += 1
            missed_amt += r.amount

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    y_true_arr = np.array(y_true, dtype=int)
    y_score_arr = np.array(y_score, dtype=float)
    if len(np.unique(y_true_arr)) > 1:
        auprc = float(average_precision_score(y_true_arr, y_score_arr))
    else:
        auprc = 1.0 if np.all(y_true_arr == 1) else 0.0

    fp_cost = fp * manual_review_cost
    net_savings = prevented_amt - fp_cost

    return RealizedMetricsReport(
        total_transactions=total,
        actual_fraud_count=fraud_count,
        actual_fraud_rate=fraud_count / total,
        predicted_positive_count=pred_pos_count,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        realized_precision=precision,
        realized_recall=recall,
        realized_f1=f1,
        realized_fpr=fpr,
        realized_auprc=auprc,
        total_fraud_loss=total_loss,
        prevented_fraud_amount=prevented_amt,
        missed_fraud_amount=missed_amt,
        false_positive_cost=fp_cost,
        net_financial_savings=net_savings,
        decision_counts=dec_counts,
        cohort_id=cohort_id,
        start_timestamp=start_timestamp
        if start_timestamp is not None
        else records[0].event_timestamp,
        end_timestamp=end_timestamp if end_timestamp is not None else records[-1].event_timestamp,
    )
