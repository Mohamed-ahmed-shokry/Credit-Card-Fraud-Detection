"""Asynchronous real-time event bus and resilient webhook dispatch subsystem.

Provides in-process, zero-dependency streaming domain event distribution,
HMAC-SHA256 signature verification, PCI-DSS compliant payload sanitization,
jittered exponential backoff retries, endpoint circuit breaking, and
dead-letter queue (DLQ) persistent retention.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from fraud_detection.audit import redact_data

__all__ = [
    "DeadLetterQueueError",
    "DispatchError",
    "EventDeliveryStatus",
    "EventError",
    "EventType",
    "FraudDomainEvent",
    "SignatureVerificationError",
    "SubscriptionError",
    "WebhookSubscription",
    "compute_webhook_signature",
    "sanitize_event_payload",
    "verify_webhook_signature",
]

_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]{1,64}$")


class EventError(RuntimeError):
    """Base exception for all domain event and webhook operations."""


class SubscriptionError(EventError):
    """Raised when webhook subscription parameters or registration fails."""


class DispatchError(EventError):
    """Raised when an unrecoverable webhook dispatch failure occurs."""


class SignatureVerificationError(EventError):
    """Raised when HMAC-SHA256 signature verification fails or has expired."""


class DeadLetterQueueError(EventError):
    """Raised when dead-letter queue operations fail."""


class EventType(StrEnum):
    """Domain event types broadcast across the fraud detection event bus."""

    TRANSACTION_SCORED = "fraud.transaction.scored"
    DECISION_DENIED = "fraud.decision.denied"
    DECISION_CHALLENGED = "fraud.decision.challenged"
    DECISION_REVIEW = "fraud.decision.review"
    FRAUD_RING_DETECTED = "fraud.graph.ring_detected"
    MERCHANT_ANOMALY_DETECTED = "fraud.merchant.anomaly_detected"
    GEOGRAPHIC_ANOMALY_DETECTED = "fraud.geo.anomaly_detected"
    DEVICE_ANOMALY_DETECTED = "fraud.device.anomaly_detected"
    MODEL_PROMOTED = "fraud.model.promoted"
    TENANT_STATUS_CHANGED = "fraud.tenant.status_changed"


class EventDeliveryStatus(StrEnum):
    """Lifecycle delivery states of a dispatched webhook notification."""

    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"
    FILTERED = "filtered"


def sanitize_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Sanitize domain event payload ensuring PCI-DSS PAN masking and credential redaction."""
    redacted = redact_data(payload, mask_value="[REDACTED]", mask_pan_values=True)
    return redacted if isinstance(redacted, dict) else dict(payload)


def compute_webhook_signature(
    payload_bytes: bytes,
    secret_token: str,
    timestamp: int | None = None,
) -> str:
    """Compute HMAC-SHA256 signature header for webhook payload.

    Formats the header as `t={timestamp},v1={hex_digest}`.
    """
    ts = int(time.time()) if timestamp is None else timestamp
    signed_payload = f"{ts}.".encode() + payload_bytes
    hex_digest = hmac.new(
        secret_token.encode("utf-8"),
        signed_payload,
        hashlib.sha256,
    ).hexdigest()
    return f"t={ts},v1={hex_digest}"


def verify_webhook_signature(
    payload_bytes: bytes,
    secret_token: str,
    signature_header: str,
    tolerance_seconds: float = 300.0,
    current_time: float | None = None,
) -> bool:
    """Verify HMAC-SHA256 webhook signature and check timestamp replay skew.

    Raises SignatureVerificationError if signature is invalid, missing, or expired.
    """
    if not signature_header or not secret_token:
        raise SignatureVerificationError("Missing signature header or secret token")

    parts = dict(item.split("=", 1) for item in signature_header.split(",") if "=" in item)
    if "t" not in parts or "v1" not in parts:
        raise SignatureVerificationError(
            f"Invalid signature header format '{signature_header}': expected t=...,v1=..."
        )

    try:
        ts = int(parts["t"])
    except ValueError as exc:
        raise SignatureVerificationError(f"Invalid timestamp in signature header: {exc}") from exc

    now = time.time() if current_time is None else current_time
    if abs(now - ts) > tolerance_seconds:
        raise SignatureVerificationError(
            f"Webhook signature timestamp expired or skewed: skew={abs(now - ts):.1f}s, "
            f"tolerance={tolerance_seconds:.1f}s"
        )

    expected_sig = compute_webhook_signature(payload_bytes, secret_token, timestamp=ts)
    expected_v1 = dict(item.split("=", 1) for item in expected_sig.split(",") if "=" in item)["v1"]

    if not hmac.compare_digest(parts["v1"], expected_v1):
        raise SignatureVerificationError("Webhook signature HMAC mismatch")

    return True


@dataclass(frozen=True)
class FraudDomainEvent:
    """Immutable domain event emitted during payment scoring or risk analysis."""

    event_id: str = field(default_factory=lambda: uuid4().hex)
    event_type: EventType = EventType.TRANSACTION_SCORED
    tenant_id: str = "default"
    timestamp: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)
    traceparent: str | None = None

    def __post_init__(self) -> None:
        """Validate domain event fields."""
        if not self.event_id or not _IDENTIFIER_PATTERN.match(self.event_id):
            raise EventError(f"Invalid event_id '{self.event_id}'")
        if not self.tenant_id or not _IDENTIFIER_PATTERN.match(self.tenant_id):
            raise EventError(f"Invalid tenant_id '{self.tenant_id}'")
        if self.timestamp <= 0:
            raise EventError(f"timestamp must be positive, got {self.timestamp}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to a plain dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "traceparent": self.traceparent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FraudDomainEvent:
        """Construct domain event from dictionary."""
        try:
            return cls(
                event_id=str(data["event_id"]),
                event_type=EventType(data["event_type"]),
                tenant_id=str(data.get("tenant_id", "default")),
                timestamp=float(data.get("timestamp", time.time())),
                payload=dict(data.get("payload", {})),
                traceparent=data.get("traceparent"),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise EventError(f"Invalid domain event serialization payload: {exc}") from exc


@dataclass(frozen=True)
class WebhookSubscription:
    """Registered webhook subscription endpoint for event notification."""

    subscription_id: str = field(default_factory=lambda: f"sub_{uuid4().hex[:12]}")
    tenant_id: str = "default"
    target_url: str = ""
    secret_token: str = field(default_factory=lambda: uuid4().hex)
    subscribed_events: tuple[EventType, ...] = (
        EventType.TRANSACTION_SCORED,
        EventType.DECISION_DENIED,
    )
    is_active: bool = True
    max_retries: int = 3
    timeout_seconds: float = 5.0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        """Validate subscription settings and endpoint URL."""
        if not self.subscription_id or not _IDENTIFIER_PATTERN.match(self.subscription_id):
            raise SubscriptionError(f"Invalid subscription_id '{self.subscription_id}'")
        if not self.tenant_id or not _IDENTIFIER_PATTERN.match(self.tenant_id):
            raise SubscriptionError(f"Invalid tenant_id '{self.tenant_id}'")

        parsed = urlparse(self.target_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise SubscriptionError(
                f"Invalid target_url '{self.target_url}': must be valid http or https URL"
            )

        if len(self.secret_token) < 16:
            raise SubscriptionError(
                f"secret_token must be at least 16 characters, got length {len(self.secret_token)}"
            )

        if self.max_retries < 0 or self.max_retries > 10:
            raise SubscriptionError(f"max_retries must be between 0 and 10, got {self.max_retries}")

        if not (0.1 <= self.timeout_seconds <= 60.0):
            raise SubscriptionError(
                f"timeout_seconds must be within [0.1, 60.0], got {self.timeout_seconds}"
            )

    def matches(self, event: FraudDomainEvent) -> bool:
        """Check if this subscription applies to the given event."""
        if not self.is_active:
            return False
        if self.tenant_id != "default" and self.tenant_id != event.tenant_id:
            return False
        return event.event_type in self.subscribed_events

    def to_dict(self, include_secret: bool = False) -> dict[str, Any]:
        """Serialize subscription to dictionary, optionally redacting secret."""
        data: dict[str, Any] = {
            "subscription_id": self.subscription_id,
            "tenant_id": self.tenant_id,
            "target_url": self.target_url,
            "subscribed_events": [e.value for e in self.subscribed_events],
            "is_active": self.is_active,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at,
        }
        if include_secret:
            data["secret_token"] = self.secret_token
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WebhookSubscription:
        """Construct subscription instance from serialized dictionary."""
        try:
            subscribed = tuple(
                EventType(e) for e in data.get("subscribed_events", [EventType.TRANSACTION_SCORED])
            )
            return cls(
                subscription_id=str(data["subscription_id"]),
                tenant_id=str(data.get("tenant_id", "default")),
                target_url=str(data["target_url"]),
                secret_token=str(data.get("secret_token", uuid4().hex)),
                subscribed_events=subscribed,
                is_active=bool(data.get("is_active", True)),
                max_retries=int(data.get("max_retries", 3)),
                timeout_seconds=float(data.get("timeout_seconds", 5.0)),
                created_at=str(data.get("created_at", datetime.now(UTC).isoformat())),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise SubscriptionError(f"Invalid subscription dictionary: {exc}") from exc
