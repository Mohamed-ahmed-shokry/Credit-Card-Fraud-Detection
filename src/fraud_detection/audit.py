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

from fraud_detection import __version__

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
            is_sensitive = any(
                sensitive in normalized_key for sensitive in sensitive_keys
            )
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
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
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
    request_id: str | None = None,
    client_metadata: dict[str, Any] | None = None,
) -> AuditEvent:
    """Construct a structured scoring audit event."""
    fraud_count = sum(1 for p in predictions if p.get("is_fraud"))
    payload: dict[str, Any] = {
        "batch_size": len(predictions),
        "threshold": threshold,
        "fraud_count": fraud_count,
        "predictions": list(predictions),
    }
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
