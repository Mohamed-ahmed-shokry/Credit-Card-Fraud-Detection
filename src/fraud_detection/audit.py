"""Structured audit event export with redaction guarantees and JSONL sink."""

from __future__ import annotations

import json
import math
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import numpy as np
import pandas as pd

from fraud_detection import __version__
from fraud_detection.evaluation import (
    PolicyBacktestReport,
    TieredThresholds,
    backtest_policy_transition,
)

# Sensitive key keywords that must always be masked in audit payloads
DEFAULT_SENSITIVE_KEYS: tuple[str, ...] = (
    "card",
    "pan",
    "account",
    "token",
    "secret",
    "key",
    "password",
    "passwd",
    "ssn",
    "cvv",
    "cvc",
    "pin",
    "auth",
    "authorization",
    "cookie",
    "email",
    "phone",
    "user_id",
    "customer_id",
    "credit_card",
    "card_number",
)

_PAN_CANDIDATE_REGEX = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


def _is_luhn_valid(candidate: str) -> bool:
    """Verify if a numeric candidate passes the Luhn check for payment card numbers."""
    digits = [int(c) for c in candidate if c.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        digit = d
        if i % 2 == 1:
            digit = digit * 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _mask_pans_in_string(text: str, mask_replacement: str = "[REDACTED_PAN]") -> str:
    """Detect and mask Luhn-valid primary account numbers (PANs) within a string."""

    def _replace_match(match: re.Match[str]) -> str:
        raw_match = match.group(0)
        if _is_luhn_valid(raw_match):
            return mask_replacement
        return raw_match

    return _PAN_CANDIDATE_REGEX.sub(_replace_match, text)


def redact_data(
    data: Any,
    *,
    sensitive_keys: tuple[str, ...] = DEFAULT_SENSITIVE_KEYS,
    mask_value: str = "[REDACTED]",
    mask_pan_values: bool = True,
) -> Any:
    """Recursively redact sensitive key values and cardholder data patterns.

    Guarantees:
    - Any dictionary key whose normalized name contains any keyword in `sensitive_keys`
      has its value replaced with `mask_value`.
    - Any string value containing a Luhn-valid payment card number has the PAN masked.
    - Non-finite floating point numbers (NaN, Inf) are converted to None for JSON safety.
    """
    if isinstance(data, Mapping):
        redacted_dict: dict[str, Any] = {}
        for key, value in data.items():
            str_key = str(key)
            normalized_key = str_key.lower().replace("-", "_").strip()
            is_sensitive = any(sensitive in normalized_key for sensitive in sensitive_keys)
            if is_sensitive:
                redacted_dict[str_key] = mask_value
            else:
                redacted_dict[str_key] = redact_data(
                    value,
                    sensitive_keys=sensitive_keys,
                    mask_value=mask_value,
                    mask_pan_values=mask_pan_values,
                )
        return redacted_dict

    if isinstance(data, (list, tuple)):
        return [
            redact_data(
                item,
                sensitive_keys=sensitive_keys,
                mask_value=mask_value,
                mask_pan_values=mask_pan_values,
            )
            for item in data
        ]

    if isinstance(data, str):
        if mask_pan_values:
            return _mask_pans_in_string(data)
        return data

    if isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data

    return data


@dataclass(frozen=True)
class AuditEvent:
    """Structured audit event recording an operational decision or scoring action."""

    event_type: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    event_id: str = field(default_factory=lambda: uuid4().hex)
    model_version: str | None = None
    dataset_fingerprint: str | None = None
    code_version: str | None = field(default=__version__)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, redact: bool = True) -> dict[str, Any]:
        """Convert the audit event to a dictionary, applying redaction by default."""
        final_payload = redact_data(self.payload) if redact else self.payload
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "model_version": self.model_version,
            "dataset_fingerprint": self.dataset_fingerprint,
            "code_version": self.code_version,
            "payload": final_payload,
        }

    def to_json(self, *, redact: bool = True) -> str:
        """Serialize the audit event to a single-line valid JSON string."""
        return json.dumps(self.to_dict(redact=redact), sort_keys=True, allow_nan=False)


class AuditSink(Protocol):
    """Protocol for audit event sinks."""

    def emit(self, event: AuditEvent) -> None:
        """Record an audit event."""
        ...

    def close(self) -> None:
        """Close the sink and release any resources."""
        ...


class NullAuditSink:
    """No-op sink when audit event logging is not configured."""

    def emit(self, event: AuditEvent) -> None:
        """No-op."""

    def close(self) -> None:
        """No-op."""


class JsonlAuditSink:
    """Thread-safe append-only JSONL audit sink."""

    def __init__(self, path: Path | str, *, redact: bool = True) -> None:
        self.path = Path(path)
        self.redact = redact
        self._lock = threading.Lock()

    def emit(self, event: AuditEvent) -> None:
        """Write an audit event as a single line of JSON to the destination file."""
        serialized = event.to_json(redact=self.redact) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(serialized)
                f.flush()

    def close(self) -> None:
        """No-op for file-based append sink."""


def build_scoring_audit_event(
    *,
    model_version: str | None,
    dataset_fingerprint: str | None,
    threshold: float,
    predictions: Sequence[dict[str, Any]],
    features: Sequence[Mapping[str, Any]] | Sequence[Mapping[Any, Any]] | None = None,
    fallback_applied: bool = False,
    fallback_reason: str | None = None,
    request_id: str | None = None,
    client_metadata: dict[str, Any] | None = None,
    review_threshold: float | None = None,
    deny_threshold: float | None = None,
    decision_counts: Mapping[str, int] | None = None,
) -> AuditEvent:
    """Construct a structured scoring audit event."""
    fraud_count = sum(1 for p in predictions if p.get("is_fraud"))
    payload: dict[str, Any] = {
        "batch_size": len(predictions),
        "threshold": threshold,
        "fraud_count": fraud_count,
        "predictions": list(predictions),
    }
    if review_threshold is not None:
        payload["review_threshold"] = review_threshold
    if deny_threshold is not None:
        payload["deny_threshold"] = deny_threshold
    if decision_counts is not None:
        payload["decision_counts"] = dict(decision_counts)
    else:
        counts: dict[str, int] = {}
        for p in predictions:
            dec = p.get("decision")
            if dec:
                counts[str(dec)] = counts.get(str(dec), 0) + 1
        if counts:
            payload["decision_counts"] = counts

    if features is not None:
        payload["features"] = list(features)
    if fallback_applied:
        payload["fallback_applied"] = True
        if fallback_reason is not None:
            payload["fallback_reason"] = fallback_reason
    if request_id is not None:
        payload["request_id"] = request_id
    if client_metadata is not None:
        payload["client_metadata"] = client_metadata

    return AuditEvent(
        event_type="scoring",
        model_version=model_version,
        dataset_fingerprint=dataset_fingerprint,
        payload=payload,
    )


def build_promotion_audit_event(
    *,
    model_version: str | None,
    dataset_fingerprint: str | None,
    bundle_summary: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> AuditEvent:
    """Construct a structured promotion evaluation audit event."""
    payload: dict[str, Any] = {
        "bundle": bundle_summary,
    }
    if metadata is not None:
        payload["metadata"] = metadata

    return AuditEvent(
        event_type="promotion",
        model_version=model_version,
        dataset_fingerprint=dataset_fingerprint,
        payload=payload,
    )


def build_shadow_scoring_audit_event(
    *,
    shadow_model_version: str | None,
    shadow_dataset_fingerprint: str | None,
    evaluated_count: int,
    discrepancy_count: int,
    discrepancies: Sequence[dict[str, Any]],
    request_id: str | None = None,
) -> AuditEvent:
    """Construct a structured shadow scoring audit event."""
    payload: dict[str, Any] = {
        "evaluated_count": evaluated_count,
        "discrepancy_count": discrepancy_count,
        "discrepancies": list(discrepancies),
    }
    if request_id is not None:
        payload["request_id"] = request_id

    return AuditEvent(
        event_type="shadow_scoring",
        model_version=shadow_model_version,
        dataset_fingerprint=shadow_dataset_fingerprint,
        payload=payload,
    )


@dataclass(frozen=True)
class DiscrepancyDetail:
    """Record of a single replayed transaction discrepancy."""

    event_id: str
    transaction_index: int
    logged_probability: float
    replayed_probability: float
    score_difference: float
    logged_decision: bool
    replayed_decision: bool
    decision_flipped: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "event_id": self.event_id,
            "transaction_index": self.transaction_index,
            "logged_probability": self.logged_probability,
            "replayed_probability": self.replayed_probability,
            "score_difference": self.score_difference,
            "logged_decision": self.logged_decision,
            "replayed_decision": self.replayed_decision,
            "decision_flipped": self.decision_flipped,
        }


@dataclass(frozen=True)
class AuditReplayReport:
    """Result of replaying scoring audit events against a target model."""

    total_events: int
    scoring_events: int
    replayed_events: int
    skipped_events: int
    total_transactions: int
    score_discrepancies: int
    decision_flips: int
    max_absolute_difference: float
    mean_absolute_difference: float
    tolerance: float
    applied_threshold: float | None
    status: str  # "MATCH", "DIVERGENT", or "EMPTY"
    discrepancies: tuple[DiscrepancyDetail, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "total_events": self.total_events,
            "scoring_events": self.scoring_events,
            "replayed_events": self.replayed_events,
            "skipped_events": self.skipped_events,
            "total_transactions": self.total_transactions,
            "score_discrepancies": self.score_discrepancies,
            "decision_flips": self.decision_flips,
            "max_absolute_difference": self.max_absolute_difference,
            "mean_absolute_difference": self.mean_absolute_difference,
            "tolerance": self.tolerance,
            "applied_threshold": self.applied_threshold,
            "status": self.status,
            "discrepancies": [d.to_dict() for d in self.discrepancies],
        }


def replay_audit_log(
    audit_log_path: Path | str,
    model: Any,
    *,
    data_path: Path | str | None = None,
    threshold: float | None = None,
    tolerance: float = 1e-4,
    max_discrepancies_to_record: int = 100,
) -> AuditReplayReport:
    """Replay scoring events from a JSONL audit log against a model to detect divergence."""
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise ValueError("tolerance must be a finite, non-negative float.")

    if (
        isinstance(max_discrepancies_to_record, bool)
        or not isinstance(max_discrepancies_to_record, int)
        or max_discrepancies_to_record < 0
    ):
        raise ValueError("max_discrepancies_to_record must be a non-negative integer.")

    if threshold is not None and (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not (0.0 <= threshold <= 1.0)
    ):
        raise ValueError("threshold must be a finite float between 0.0 and 1.0.")

    if isinstance(model, (str, Path)):
        from fraud_detection.model import load_model

        model = load_model(model)

    audit_path = Path(audit_log_path)
    if not audit_path.is_file():
        raise FileNotFoundError(f"Audit log file not found: {audit_path}")

    external_df: pd.DataFrame | None = None
    external_offset = 0
    if data_path is not None:
        p = Path(data_path)
        if not p.is_file():
            raise FileNotFoundError(f"Transaction data file not found: {p}")
        external_df = pd.read_csv(p)

    total_events = 0
    scoring_events = 0
    replayed_events = 0
    skipped_events = 0
    total_transactions = 0
    score_discrepancies = 0
    decision_flips = 0
    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    discrepancies_list: list[DiscrepancyDetail] = []

    with audit_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            total_events += 1
            try:
                event_dict = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(event_dict, dict):
                continue
            if event_dict.get("event_type") != "scoring":
                continue

            scoring_events += 1
            payload = event_dict.get("payload")
            if not isinstance(payload, dict):
                skipped_events += 1
                continue

            logged_predictions = payload.get("predictions")
            if not isinstance(logged_predictions, list) or not logged_predictions:
                skipped_events += 1
                continue

            inline_features = payload.get("features")
            frame: pd.DataFrame
            if isinstance(inline_features, list) and inline_features:
                if len(inline_features) != len(logged_predictions):
                    skipped_events += 1
                    continue
                frame = pd.DataFrame(inline_features)
            elif external_df is not None:
                n_rows = len(logged_predictions)
                if external_offset + n_rows > len(external_df):
                    skipped_events += 1
                    continue
                frame = external_df.iloc[external_offset : external_offset + n_rows].copy()
                external_offset += n_rows
            else:
                skipped_events += 1
                continue

            if hasattr(model, "feature_names"):
                feat_names = list(model.feature_names)
                extra_cols = [c for c in frame.columns if c not in feat_names]
                if extra_cols:
                    frame = frame.drop(columns=extra_cols, errors="ignore")

            try:
                probs = model.predict_probabilities(frame)
            except Exception:  # noqa: BLE001
                skipped_events += 1
                continue

            if len(probs) != len(logged_predictions):
                skipped_events += 1
                continue

            if threshold is not None:
                applied_thresh = float(threshold)
            else:
                raw_thresh = payload.get("threshold")
                if raw_thresh is None:
                    raw_thresh = getattr(model, "threshold", 0.5)
                if raw_thresh is None:
                    raw_thresh = 0.5
                try:
                    applied_thresh = float(raw_thresh)
                    if not math.isfinite(applied_thresh) or not (0.0 <= applied_thresh <= 1.0):
                        applied_thresh = 0.5
                except (TypeError, ValueError):
                    applied_thresh = 0.5

            decisions = np.asarray(probs) >= applied_thresh
            event_id = str(event_dict.get("event_id", f"event_{line_num}"))

            for idx, (pred_dict, prob, dec) in enumerate(
                zip(logged_predictions, probs, decisions, strict=True)
            ):
                if not isinstance(pred_dict, dict):
                    continue
                raw_prob = pred_dict.get("fraud_probability")
                if raw_prob is None:
                    continue
                try:
                    logged_prob = float(raw_prob)
                    if not math.isfinite(logged_prob):
                        continue
                except (TypeError, ValueError):
                    continue

                total_transactions += 1
                logged_dec = bool(pred_dict.get("is_fraud", False))
                prob_float = float(prob)
                dec_bool = bool(dec)

                diff = abs(prob_float - logged_prob)
                flipped = dec_bool != logged_dec
                if diff > max_abs_diff:
                    max_abs_diff = diff
                sum_abs_diff += diff

                is_score_disc = diff > tolerance
                if is_score_disc:
                    score_discrepancies += 1
                if flipped:
                    decision_flips += 1

                can_record = (is_score_disc or flipped) and len(
                    discrepancies_list
                ) < max_discrepancies_to_record
                if can_record:
                    discrepancies_list.append(
                        DiscrepancyDetail(
                            event_id=event_id,
                            transaction_index=idx,
                            logged_probability=logged_prob,
                            replayed_probability=prob_float,
                            score_difference=round(diff, 6),
                            logged_decision=logged_dec,
                            replayed_decision=dec_bool,
                            decision_flipped=flipped,
                        )
                    )

            replayed_events += 1

    mean_abs_diff = (sum_abs_diff / total_transactions) if total_transactions > 0 else 0.0

    if replayed_events == 0 or total_transactions == 0:
        status = "EMPTY"
    elif score_discrepancies > 0 or decision_flips > 0:
        status = "DIVERGENT"
    else:
        status = "MATCH"

    return AuditReplayReport(
        total_events=total_events,
        scoring_events=scoring_events,
        replayed_events=replayed_events,
        skipped_events=skipped_events,
        total_transactions=total_transactions,
        score_discrepancies=score_discrepancies,
        decision_flips=decision_flips,
        max_absolute_difference=round(max_abs_diff, 6),
        mean_absolute_difference=round(mean_abs_diff, 6),
        tolerance=tolerance,
        applied_threshold=threshold,
        status=status,
        discrepancies=tuple(discrepancies_list),
    )


def backtest_audit_policy(
    audit_log_path: Path | str,
    candidate_thresholds: TieredThresholds,
    *,
    baseline_thresholds: TieredThresholds | None = None,
    y_true: np.ndarray | Sequence[int] | None = None,
    manual_review_cost: float = 5.0,
    false_deny_cost: float = 50.0,
    missed_fraud_cost: float = 200.0,
) -> PolicyBacktestReport:
    """Stream a JSONL audit log and backtest candidate tiered decision thresholds.

    Extracts recorded scoring probabilities from the audit log and performs
    counterfactual backtesting against either an explicitly supplied baseline
    policy or one inferred from the audit log metadata.

    Parameters
    ----------
    audit_log_path:
        Path to the JSONL audit event log file.
    candidate_thresholds:
        Target candidate TieredThresholds to evaluate.
    baseline_thresholds:
        Optional baseline TieredThresholds. If None, inferred from the first
        valid thresholds in scoring event payloads, or defaults to (0.5, 0.5).
    y_true:
        Optional ground truth binary labels matching the extracted transactions.
    manual_review_cost:
        Operational cost per challenged transaction.
    false_deny_cost:
        Friction/support cost per legitimate transaction denied.
    missed_fraud_cost:
        Loss cost per undetected fraud allowed.

    Returns
    -------
    PolicyBacktestReport
        Comprehensive migration report with transition matrix and workload deltas.
    """
    if not isinstance(candidate_thresholds, TieredThresholds):
        raise TypeError("candidate_thresholds must be a TieredThresholds instance")
    if baseline_thresholds is not None and not isinstance(baseline_thresholds, TieredThresholds):
        raise TypeError("baseline_thresholds must be a TieredThresholds instance")

    audit_path = Path(audit_log_path)
    if not audit_path.is_file():
        raise FileNotFoundError(f"Audit log file not found: {audit_path}")

    probs_list: list[float] = []
    inferred_baseline: TieredThresholds | None = None

    with audit_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event_dict = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event_dict, dict):
                continue
            if event_dict.get("event_type") != "scoring":
                continue
            payload = event_dict.get("payload")
            if not isinstance(payload, dict):
                continue

            if inferred_baseline is None and baseline_thresholds is None:
                r_th = payload.get("review_threshold")
                d_th = payload.get("deny_threshold")
                if r_th is not None and d_th is not None:
                    try:
                        r_f = float(r_th)
                        d_f = float(d_th)
                        if (
                            not isinstance(r_th, bool)
                            and not isinstance(d_th, bool)
                            and math.isfinite(r_f)
                            and math.isfinite(d_f)
                            and 0.0 <= r_f <= d_f <= 1.0
                        ):
                            inferred_baseline = TieredThresholds(r_f, d_f)
                    except (TypeError, ValueError):
                        pass
                if inferred_baseline is None and "threshold" in payload:
                    th = payload.get("threshold")
                    if th is not None and not isinstance(th, bool):
                        try:
                            th_f = float(th)
                            if math.isfinite(th_f) and 0.0 <= th_f <= 1.0:
                                inferred_baseline = TieredThresholds(th_f, th_f)
                        except (TypeError, ValueError):
                            pass

            predictions = payload.get("predictions")
            if not isinstance(predictions, list):
                continue
            for pred in predictions:
                if not isinstance(pred, dict):
                    continue
                raw_prob = pred.get("fraud_probability")
                if raw_prob is None or isinstance(raw_prob, bool):
                    continue
                try:
                    prob = float(raw_prob)
                    if math.isfinite(prob) and 0.0 <= prob <= 1.0:
                        probs_list.append(prob)
                except (TypeError, ValueError):
                    continue

    if not probs_list:
        raise ValueError(f"No valid scoring transactions found in audit log: {audit_path}")

    y_arr: np.ndarray | None = None
    if y_true is not None:
        y_arr = np.asarray(y_true, dtype=int)
        if len(y_arr) != len(probs_list):
            raise ValueError(
                f"Length of y_true ({len(y_arr)}) does not match number of "
                f"scored transactions ({len(probs_list)})"
            )

    effective_baseline = baseline_thresholds or inferred_baseline or TieredThresholds(0.5, 0.5)

    return backtest_policy_transition(
        probabilities=np.asarray(probs_list, dtype=float),
        baseline_thresholds=effective_baseline,
        candidate_thresholds=candidate_thresholds,
        y_true=y_arr,
        manual_review_cost=manual_review_cost,
        false_deny_cost=false_deny_cost,
        missed_fraud_cost=missed_fraud_cost,
    )
