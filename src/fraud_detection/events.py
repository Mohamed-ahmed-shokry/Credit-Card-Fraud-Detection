"""Asynchronous real-time event bus and resilient webhook dispatch subsystem.

Provides in-process, zero-dependency streaming domain event distribution,
HMAC-SHA256 signature verification, PCI-DSS compliant payload sanitization,
jittered exponential backoff retries, endpoint circuit breaking, and
dead-letter queue (DLQ) persistent retention.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from fraud_detection.audit import redact_data

__all__ = [
    "DeadLetterQueue",
    "DeadLetterQueueError",
    "DeadLetterRecord",
    "DispatchConfig",
    "DispatchError",
    "EventBus",
    "EventDeliveryStatus",
    "EventError",
    "EventType",
    "FraudDomainEvent",
    "SignatureVerificationError",
    "SubscriptionError",
    "SubscriptionRegistry",
    "WebhookDeliveryResult",
    "WebhookDispatcher",
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


@dataclass(frozen=True)
class DeadLetterRecord:
    """Archived record of an undeliverable webhook notification."""

    record_id: str = field(default_factory=lambda: uuid4().hex)
    event: FraudDomainEvent = field(default_factory=FraudDomainEvent)
    subscription_id: str = ""
    target_url: str = ""
    error_reason: str = ""
    attempt_count: int = 1
    failed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize dead letter record to dictionary."""
        return {
            "record_id": self.record_id,
            "event": self.event.to_dict(),
            "subscription_id": self.subscription_id,
            "target_url": self.target_url,
            "error_reason": self.error_reason,
            "attempt_count": self.attempt_count,
            "failed_at": self.failed_at,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeadLetterRecord:
        """Construct dead letter record from dictionary."""
        try:
            return cls(
                record_id=str(data["record_id"]),
                event=FraudDomainEvent.from_dict(data["event"]),
                subscription_id=str(data["subscription_id"]),
                target_url=str(data["target_url"]),
                error_reason=str(data["error_reason"]),
                attempt_count=int(data.get("attempt_count", 1)),
                failed_at=str(data.get("failed_at", datetime.now(UTC).isoformat())),
                payload=dict(data.get("payload", {})),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise DeadLetterQueueError(f"Invalid DeadLetterRecord dictionary: {exc}") from exc


class DeadLetterQueue:
    """Thread-safe bounded queue retaining undelivered webhook records."""

    def __init__(self, max_size: int = 10_000) -> None:
        if max_size <= 0:
            raise DeadLetterQueueError(f"max_size must be positive, got {max_size}")
        self._max_size = max_size
        self._lock = threading.RLock()
        self._records: OrderedDict[str, DeadLetterRecord] = OrderedDict()

    @property
    def max_size(self) -> int:
        """Return maximum capacity of the dead letter queue."""
        return self._max_size

    def push(self, record: DeadLetterRecord) -> None:
        """Push a failed delivery record, evicting the oldest if capacity is exceeded."""
        with self._lock:
            if len(self._records) >= self._max_size:
                self._records.popitem(last=False)
            self._records[record.record_id] = record

    def get(self, record_id: str) -> DeadLetterRecord | None:
        """Retrieve a specific dead letter record by identifier."""
        with self._lock:
            return self._records.get(record_id)

    def remove(self, record_id: str) -> bool:
        """Remove a record by identifier. Returns True if removed."""
        with self._lock:
            return self._records.pop(record_id, None) is not None

    def acknowledge(self, record_id: str) -> bool:
        """Acknowledge and remove a record from the DLQ. Returns True if removed."""
        return self.remove(record_id)

    @property
    def size(self) -> int:
        """Return current count of queued dead letter records."""
        return self.count()

    def list_records(
        self,
        limit: int = 100,
        tenant_id: str | None = None,
    ) -> list[DeadLetterRecord]:
        """List dead letter records filtered optionally by tenant."""
        with self._lock:
            records = list(self._records.values())
            if tenant_id is not None:
                records = [r for r in records if r.event.tenant_id == tenant_id]
            return records[: max(1, limit)]

    def clear(self, tenant_id: str | None = None) -> int:
        """Clear all or tenant-specific records from the queue. Returns count cleared."""
        with self._lock:
            if tenant_id is None:
                cleared = len(self._records)
                self._records.clear()
                return cleared
            keys_to_delete = [k for k, r in self._records.items() if r.event.tenant_id == tenant_id]
            for k in keys_to_delete:
                del self._records[k]
            return len(keys_to_delete)

    def count(self, tenant_id: str | None = None) -> int:
        """Return total count of queued dead letter records."""
        with self._lock:
            if tenant_id is None:
                return len(self._records)
            return sum(1 for r in self._records.values() if r.event.tenant_id == tenant_id)

    def __len__(self) -> int:
        """Return total count of queued dead letter records."""
        return self.count()


@dataclass(frozen=True)
class DispatchConfig:
    """Configurable retry, timeout, and circuit breaker policies for webhook dispatch."""

    max_retries: int = 3
    initial_backoff_seconds: float = 0.5
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 10.0
    jitter: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_seconds: float = 30.0
    http_timeout_seconds: float = 5.0
    dlq_max_size: int = 10_000

    def __post_init__(self) -> None:
        """Validate dispatcher configuration parameters."""
        if self.max_retries < 0:
            raise DispatchError(f"max_retries must be non-negative, got {self.max_retries}")
        if self.initial_backoff_seconds <= 0:
            raise DispatchError(
                f"initial_backoff_seconds must be positive, got {self.initial_backoff_seconds}"
            )
        if self.backoff_multiplier < 1.0:
            raise DispatchError(f"backoff_multiplier must be >= 1.0, got {self.backoff_multiplier}")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise DispatchError("max_backoff_seconds must be >= initial_backoff_seconds")
        if self.circuit_breaker_failure_threshold <= 0:
            raise DispatchError("circuit_breaker_failure_threshold must be positive")
        if self.circuit_breaker_recovery_seconds <= 0:
            raise DispatchError("circuit_breaker_recovery_seconds must be positive")
        if self.http_timeout_seconds <= 0:
            raise DispatchError("http_timeout_seconds must be positive")


@dataclass(frozen=True)
class WebhookDeliveryResult:
    """Result of attempting to dispatch a webhook notification."""

    delivery_id: str = field(default_factory=lambda: uuid4().hex)
    subscription_id: str = ""
    status: EventDeliveryStatus = EventDeliveryStatus.PENDING
    status_code: int | None = None
    error_message: str | None = None
    attempt_count: int = 1
    duration_seconds: float = 0.0


class WebhookDispatcher:
    """Resilient HTTP webhook dispatcher with exponential backoff and circuit breaking."""

    def __init__(
        self,
        config: DispatchConfig | None = None,
        dlq: DeadLetterQueue | None = None,
        transport: Callable[[str, bytes, dict[str, str], float], tuple[int, bytes]] | None = None,
    ) -> None:
        self._config = config or DispatchConfig()
        self._dlq = dlq or DeadLetterQueue(max_size=self._config.dlq_max_size)
        self._transport = transport or self._default_http_transport
        self._lock = threading.RLock()
        # Circuit breaker state: subscription_id -> (consecutive_failures, last_failure_timestamp)
        self._cb_state: dict[str, tuple[int, float]] = {}

    @property
    def config(self) -> DispatchConfig:
        """Return the dispatcher configuration."""
        return self._config

    @property
    def dlq(self) -> DeadLetterQueue:
        """Return the dead letter queue."""
        return self._dlq

    def is_circuit_open(self, subscription_id: str) -> bool:
        """Return True if the circuit breaker is currently open for the subscription."""
        with self._lock:
            state = self._cb_state.get(subscription_id)
            if state is None:
                return False
            failures, last_fail = state
            if failures >= self._config.circuit_breaker_failure_threshold:
                return (time.time() - last_fail) < self._config.circuit_breaker_recovery_seconds
            return False

    def reset_circuit_breaker(self, subscription_id: str) -> None:
        """Reset the circuit breaker for a given subscription."""
        with self._lock:
            self._cb_state.pop(subscription_id, None)

    def dispatch(
        self,
        subscription: WebhookSubscription,
        event: FraudDomainEvent,
    ) -> WebhookDeliveryResult:
        """Dispatch a signed domain event to a webhook subscription endpoint."""
        start_time = time.perf_counter()
        sub_id = subscription.subscription_id

        # Circuit breaker check
        if self.is_circuit_open(sub_id):
            duration = time.perf_counter() - start_time
            return WebhookDeliveryResult(
                subscription_id=sub_id,
                status=EventDeliveryStatus.FAILED,
                error_message="Circuit breaker open due to consecutive endpoint failures",
                attempt_count=0,
                duration_seconds=duration,
            )

        sanitized_payload = sanitize_event_payload(event.payload)
        envelope = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "tenant_id": event.tenant_id,
            "timestamp": event.timestamp,
            "payload": sanitized_payload,
            "traceparent": event.traceparent,
        }
        payload_bytes = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
        signature_header = compute_webhook_signature(
            payload_bytes,
            subscription.secret_token,
            timestamp=int(event.timestamp),
        )

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "CreditCardFraudDetection-Webhook/1.0",
            "X-Fraud-Signature": signature_header,
            "X-Event-Type": event.event_type.value,
            "X-Tenant-ID": event.tenant_id,
            "X-Event-ID": event.event_id,
        }
        if event.traceparent:
            headers["traceparent"] = event.traceparent

        max_attempts = max(1, subscription.max_retries + 1)
        last_error = ""
        last_status_code: int | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                status_code, _response_bytes = self._transport(
                    subscription.target_url,
                    payload_bytes,
                    headers,
                    subscription.timeout_seconds,
                )
                last_status_code = status_code
                if 200 <= status_code < 300:
                    # Success: reset circuit breaker
                    with self._lock:
                        self._cb_state.pop(sub_id, None)
                    duration = time.perf_counter() - start_time
                    return WebhookDeliveryResult(
                        subscription_id=sub_id,
                        status=EventDeliveryStatus.DELIVERED,
                        status_code=status_code,
                        attempt_count=attempt,
                        duration_seconds=duration,
                    )
                # 4xx client errors (except 429) are terminal, do not retry
                if 400 <= status_code < 500 and status_code != 429:
                    last_error = f"HTTP {status_code} client error"
                    break

                last_error = f"HTTP {status_code} server error"
            except Exception as exc:  # noqa: BLE001
                last_error = f"Network transport error: {exc}"

            # Retry backoff with jitter if attempts remain
            if attempt < max_attempts:
                backoff = min(
                    self._config.max_backoff_seconds,
                    self._config.initial_backoff_seconds
                    * (self._config.backoff_multiplier ** (attempt - 1)),
                )
                if self._config.jitter:
                    backoff = random.uniform(0.5 * backoff, backoff)  # noqa: S311
                time.sleep(backoff)

        # Failure path: record in circuit breaker and DLQ
        with self._lock:
            current_fails = self._cb_state.get(sub_id, (0, 0.0))[0]
            self._cb_state[sub_id] = (current_fails + 1, time.time())

        # Write to dead letter queue
        dlq_record = DeadLetterRecord(
            event=event,
            subscription_id=sub_id,
            target_url=subscription.target_url,
            error_reason=last_error,
            attempt_count=max_attempts,
            payload=envelope,
        )
        self._dlq.push(dlq_record)

        duration = time.perf_counter() - start_time
        return WebhookDeliveryResult(
            subscription_id=sub_id,
            status=EventDeliveryStatus.DEAD_LETTER,
            status_code=last_status_code,
            error_message=last_error,
            attempt_count=max_attempts,
            duration_seconds=duration,
        )

    @staticmethod
    def _default_http_transport(
        url: str,
        payload_bytes: bytes,
        headers: dict[str, str],
        timeout: float,
    ) -> tuple[int, bytes]:
        """Perform standard HTTP POST dispatch using urllib."""
        req = urllib.request.Request(url, data=payload_bytes, headers=headers, method="POST")  # noqa: S310
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return int(resp.status), resp.read()
        except urllib.error.HTTPError as exc:
            return int(exc.code), exc.read()


class SubscriptionRegistry:
    """Thread-safe registry for managing active webhook subscriptions."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscriptions: dict[str, WebhookSubscription] = {}

    def register(self, subscription: WebhookSubscription, overwrite: bool = False) -> None:
        """Register a new webhook subscription."""
        with self._lock:
            sub_id = subscription.subscription_id
            if sub_id in self._subscriptions and not overwrite:
                raise SubscriptionError(
                    f"Subscription '{sub_id}' already registered. Set overwrite=True to update."
                )
            self._subscriptions[sub_id] = subscription

    def get(self, subscription_id: str) -> WebhookSubscription | None:
        """Retrieve a subscription by identifier."""
        with self._lock:
            return self._subscriptions.get(subscription_id)

    def remove(self, subscription_id: str) -> bool:
        """Remove a subscription by identifier. Returns True if removed."""
        with self._lock:
            return self._subscriptions.pop(subscription_id, None) is not None

    def list_subscriptions(
        self,
        tenant_id: str | None = None,
    ) -> list[WebhookSubscription]:
        """List subscriptions, optionally filtered by tenant."""
        with self._lock:
            subs = list(self._subscriptions.values())
            if tenant_id is not None:
                subs = [s for s in subs if s.tenant_id == tenant_id]
            return subs

    def find_matching(self, event: FraudDomainEvent) -> list[WebhookSubscription]:
        """Find active subscriptions that match the event type and tenant."""
        with self._lock:
            return [s for s in self._subscriptions.values() if s.matches(event)]

    def count(self, tenant_id: str | None = None) -> int:
        """Return total count of registered subscriptions."""
        with self._lock:
            if tenant_id is None:
                return len(self._subscriptions)
            return sum(1 for s in self._subscriptions.values() if s.tenant_id == tenant_id)

    def export_manifest(self) -> dict[str, Any]:
        """Export all subscriptions to a serializable dictionary manifest."""
        with self._lock:
            return {
                "version": "1.0",
                "exported_at": datetime.now(UTC).isoformat(),
                "subscriptions": [
                    s.to_dict(include_secret=True) for s in self._subscriptions.values()
                ],
            }

    @classmethod
    def import_manifest(cls, data: dict[str, Any]) -> SubscriptionRegistry:
        """Construct a SubscriptionRegistry from an exported manifest."""
        registry = cls()
        subs_data = data.get("subscriptions", [])
        if not isinstance(subs_data, list):
            raise SubscriptionError("Invalid manifest format: 'subscriptions' must be a list")
        for item in subs_data:
            sub = WebhookSubscription.from_dict(item)
            registry.register(sub, overwrite=True)
        return registry


class EventBus:
    """In-process thread-safe streaming event bus for domain events and webhooks."""

    def __init__(
        self,
        registry: SubscriptionRegistry | None = None,
        dispatcher: WebhookDispatcher | None = None,
        max_workers: int = 4,
    ) -> None:
        self._registry = registry or SubscriptionRegistry()
        self._dispatcher = dispatcher or WebhookDispatcher()
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix="fraud-event-bus",
        )
        # In-memory listeners: token -> (event_type | None, tenant_id | None, callback)
        self._listeners: dict[
            str, tuple[EventType | None, str | None, Callable[[FraudDomainEvent], None]]
        ] = {}

    @property
    def registry(self) -> SubscriptionRegistry:
        """Return subscription registry."""
        return self._registry

    @property
    def listeners_count(self) -> int:
        """Return count of registered in-memory subscribers."""
        with self._lock:
            return len(self._listeners)

    @property
    def dispatcher(self) -> WebhookDispatcher:
        """Return webhook dispatcher."""
        return self._dispatcher

    def subscribe(
        self,
        callback: Callable[[FraudDomainEvent], None],
        event_type: EventType | None = None,
        tenant_id: str | None = None,
    ) -> str:
        """Register an in-memory listener callback, returning a cancellation token."""
        with self._lock:
            token = f"listener_{uuid4().hex[:12]}"
            self._listeners[token] = (event_type, tenant_id, callback)
            return token

    def unsubscribe(self, token: str) -> bool:
        """Unregister an in-memory listener by token."""
        with self._lock:
            return self._listeners.pop(token, None) is not None

    def publish(
        self,
        event: FraudDomainEvent,
        synchronous: bool = False,
    ) -> int:
        """Publish a domain event to all matching in-memory listeners and webhooks."""
        with self._lock:
            listeners_to_notify = [
                cb
                for et, tid, cb in self._listeners.values()
                if (et is None or et == event.event_type)
                and (tid is None or tid == "default" or tid == event.tenant_id)
            ]
            matching_webhooks = self._registry.find_matching(event)

        dispatched_count = len(listeners_to_notify) + len(matching_webhooks)

        # Notify in-memory listeners
        for listener in listeners_to_notify:
            with contextlib.suppress(Exception):
                listener(event)

        # Dispatch webhooks
        for sub in matching_webhooks:
            if synchronous:
                self._dispatcher.dispatch(sub, event)
            else:
                self._pool.submit(self._dispatcher.dispatch, sub, event)

        return dispatched_count

    def close(self, _timeout: float = 2.0) -> None:
        """Shutdown background worker pool cleanly."""
        self._pool.shutdown(wait=True, cancel_futures=True)
