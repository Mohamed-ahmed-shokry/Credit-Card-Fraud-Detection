"""Unit tests for domain events and webhook models."""

from __future__ import annotations

import pytest

from fraud_detection.events import (
    DeadLetterQueue,
    DeadLetterQueueError,
    DeadLetterRecord,
    DispatchConfig,
    DispatchError,
    EventBus,
    EventDeliveryStatus,
    EventError,
    EventType,
    FraudDomainEvent,
    SignatureVerificationError,
    SubscriptionError,
    SubscriptionRegistry,
    WebhookDispatcher,
    WebhookSubscription,
    compute_webhook_signature,
    sanitize_event_payload,
    verify_webhook_signature,
)


def test_fraud_domain_event_defaults_and_validation() -> None:
    """Validate default construction, serialization, and boundary constraints."""
    event = FraudDomainEvent(
        event_id="evt_12345",
        event_type=EventType.DECISION_DENIED,
        tenant_id="fintech_alpha",
        payload={"score": 0.94, "action": "DENY"},
    )
    assert event.event_id == "evt_12345"
    assert event.event_type == EventType.DECISION_DENIED
    assert event.tenant_id == "fintech_alpha"
    assert event.payload["score"] == 0.94

    # Serialization roundtrip
    data = event.to_dict()
    assert data["event_id"] == "evt_12345"
    assert data["event_type"] == "fraud.decision.denied"

    restored = FraudDomainEvent.from_dict(data)
    assert restored.event_id == event.event_id
    assert restored.event_type == event.event_type
    assert restored.tenant_id == event.tenant_id

    # Validation errors
    with pytest.raises(EventError, match="Invalid event_id"):
        FraudDomainEvent(event_id="bad id with spaces!")

    with pytest.raises(EventError, match="Invalid tenant_id"):
        FraudDomainEvent(tenant_id="")

    with pytest.raises(EventError, match="timestamp must be positive"):
        FraudDomainEvent(timestamp=-10.0)

    with pytest.raises(EventError, match="Invalid domain event serialization"):
        FraudDomainEvent.from_dict({"event_id": "ok", "event_type": "unknown_type"})


def test_webhook_subscription_validation_and_matching() -> None:
    """Validate subscription creation, URL validation, matching, and secret handling."""
    test_signing_key = "sample_test_signing_token_12345"
    sub = WebhookSubscription(
        subscription_id="sub_test1",
        tenant_id="fintech_alpha",
        target_url="https://api.example.com/fraud/webhooks",
        secret_token=test_signing_key,
        subscribed_events=(EventType.DECISION_DENIED, EventType.FRAUD_RING_DETECTED),
    )
    assert sub.is_active is True
    assert sub.timeout_seconds == 5.0
    assert sub.max_retries == 3

    # Event matching
    matched_event = FraudDomainEvent(
        event_id="evt_01",
        event_type=EventType.DECISION_DENIED,
        tenant_id="fintech_alpha",
    )
    assert sub.matches(matched_event) is True

    # Different event type
    unmatched_event = FraudDomainEvent(
        event_id="evt_02",
        event_type=EventType.TRANSACTION_SCORED,
        tenant_id="fintech_alpha",
    )
    assert sub.matches(unmatched_event) is False

    # Different tenant
    other_tenant_event = FraudDomainEvent(
        event_id="evt_03",
        event_type=EventType.DECISION_DENIED,
        tenant_id="other_tenant",
    )
    assert sub.matches(other_tenant_event) is False

    # Serialization without secret
    public_dict = sub.to_dict(include_secret=False)
    assert "secret_token" not in public_dict
    assert public_dict["target_url"] == "https://api.example.com/fraud/webhooks"

    # Serialization with secret and roundtrip
    private_dict = sub.to_dict(include_secret=True)
    assert "secret_token" in private_dict
    restored = WebhookSubscription.from_dict(private_dict)
    assert restored.subscription_id == sub.subscription_id
    assert restored.secret_token == sub.secret_token

    # Validation errors
    with pytest.raises(SubscriptionError, match="Invalid target_url"):
        WebhookSubscription(target_url="not-a-valid-url")

    with pytest.raises(SubscriptionError, match="secret_token must be at least 16"):
        WebhookSubscription(
            target_url="https://api.example.com/hook",
            secret_token="short_key",  # noqa: S106
        )

    with pytest.raises(SubscriptionError, match="max_retries must be between 0 and 10"):
        WebhookSubscription(
            target_url="https://api.example.com/hook",
            secret_token=test_signing_key,
            max_retries=15,
        )

    with pytest.raises(SubscriptionError, match="timeout_seconds must be within"):
        WebhookSubscription(
            target_url="https://api.example.com/hook",
            secret_token=test_signing_key,
            timeout_seconds=0.01,
        )


def test_event_delivery_status_enum() -> None:
    """Validate status enum values."""
    assert EventDeliveryStatus.PENDING == "pending"
    assert EventDeliveryStatus.DELIVERED == "delivered"
    assert EventDeliveryStatus.FAILED == "failed"
    assert EventDeliveryStatus.DEAD_LETTER == "dead_letter"


def test_webhook_signature_computation_and_verification() -> None:
    """Validate HMAC-SHA256 signature generation, verification, and timestamp tolerance."""
    payload = b'{"event_id": "evt_test", "action": "DENY"}'
    signing_secret = "test_secret_key_12345678"  # noqa: S105
    ts = 1700000000

    sig = compute_webhook_signature(payload, signing_secret, timestamp=ts)
    assert sig.startswith(f"t={ts},v1=")

    # Successful verification within tolerance
    assert (
        verify_webhook_signature(
            payload,
            signing_secret,
            sig,
            tolerance_seconds=60.0,
            current_time=float(ts + 10),
        )
        is True
    )

    # Verification failure: wrong secret
    with pytest.raises(SignatureVerificationError, match="HMAC mismatch"):
        verify_webhook_signature(
            payload,
            "wrong_secret_12345678",
            sig,
            tolerance_seconds=60.0,
            current_time=float(ts + 10),
        )

    # Verification failure: payload modified
    tampered_payload = b'{"event_id": "evt_test", "action": "ALLOW"}'
    with pytest.raises(SignatureVerificationError, match="HMAC mismatch"):
        verify_webhook_signature(
            tampered_payload,
            signing_secret,
            sig,
            tolerance_seconds=60.0,
            current_time=float(ts + 10),
        )

    # Verification failure: timestamp expired (replay attack)
    with pytest.raises(SignatureVerificationError, match="expired or skewed"):
        verify_webhook_signature(
            payload,
            signing_secret,
            sig,
            tolerance_seconds=60.0,
            current_time=float(ts + 300),
        )

    # Verification failure: malformed header
    with pytest.raises(SignatureVerificationError, match="Invalid signature header format"):
        verify_webhook_signature(
            payload, signing_secret, "malformed_header", tolerance_seconds=60.0
        )

    with pytest.raises(SignatureVerificationError, match="Missing signature header"):
        verify_webhook_signature(payload, "", sig)


def test_sanitize_event_payload_redaction() -> None:
    """Validate PCI-DSS Luhn PAN masking and credential redaction."""
    raw_payload = {
        "transaction_id": "tx_999",
        "card_number": "4532015112830366",  # Valid Luhn Visa
        "amount": 250.0,
        "metadata": {
            "api_key": "secret_abc123",
            "password_hash": "hidden_hash",
            "notes": "Payment using card 4532015112830366 in Paris",
        },
    }
    sanitized = sanitize_event_payload(raw_payload)
    assert sanitized["transaction_id"] == "tx_999"
    assert sanitized["card_number"] == "[REDACTED]"
    assert sanitized["amount"] == 250.0
    assert sanitized["metadata"]["api_key"] == "[REDACTED]"
    assert sanitized["metadata"]["password_hash"] == "[REDACTED]"  # noqa: S105
    assert "[REDACTED_PAN]" in sanitized["metadata"]["notes"]


def test_dead_letter_queue_operations() -> None:
    """Validate DeadLetterQueue capacity bounding, tenant filtering, and retrieval."""
    dlq = DeadLetterQueue(max_size=3)
    assert dlq.max_size == 3

    event1 = FraudDomainEvent(event_id="e1", tenant_id="tenant_a")
    event2 = FraudDomainEvent(event_id="e2", tenant_id="tenant_b")
    event3 = FraudDomainEvent(event_id="e3", tenant_id="tenant_a")
    event4 = FraudDomainEvent(event_id="e4", tenant_id="tenant_b")

    rec1 = DeadLetterRecord(record_id="r1", event=event1, error_reason="HTTP 500")
    rec2 = DeadLetterRecord(record_id="r2", event=event2, error_reason="Timeout")
    rec3 = DeadLetterRecord(record_id="r3", event=event3, error_reason="HTTP 502")
    rec4 = DeadLetterRecord(record_id="r4", event=event4, error_reason="HTTP 503")

    dlq.push(rec1)
    dlq.push(rec2)
    dlq.push(rec3)
    assert dlq.count() == 3
    assert dlq.count("tenant_a") == 2
    assert dlq.count("tenant_b") == 1

    # Over capacity evicts oldest (rec1)
    dlq.push(rec4)
    assert dlq.count() == 3
    assert dlq.get("r1") is None
    assert dlq.get("r4") is not None

    # Filtering by tenant
    tenant_b_recs = dlq.list_records(tenant_id="tenant_b")
    assert len(tenant_b_recs) == 2
    assert all(r.event.tenant_id == "tenant_b" for r in tenant_b_recs)

    # Remove single record
    assert dlq.remove("r2") is True
    assert dlq.get("r2") is None
    assert dlq.remove("non_existent") is False

    # Clear tenant-specific
    cleared = dlq.clear(tenant_id="tenant_b")
    assert cleared == 1
    assert dlq.count() == 1

    # Clear all
    assert dlq.clear() == 1
    assert dlq.count() == 0

    # Serialization roundtrip
    rec_dict = rec3.to_dict()
    restored = DeadLetterRecord.from_dict(rec_dict)
    assert restored.record_id == "r3"
    assert restored.error_reason == "HTTP 502"

    with pytest.raises(DeadLetterQueueError, match="max_size must be positive"):
        DeadLetterQueue(max_size=0)

    with pytest.raises(DeadLetterQueueError, match="Invalid DeadLetterRecord dictionary"):
        DeadLetterRecord.from_dict({"record_id": "r"})


def test_dispatch_config_validation() -> None:
    """Validate DispatchConfig constraints and parameters."""
    cfg = DispatchConfig(max_retries=2, initial_backoff_seconds=0.1)
    assert cfg.max_retries == 2

    with pytest.raises(DispatchError, match="max_retries must be non-negative"):
        DispatchConfig(max_retries=-1)

    with pytest.raises(DispatchError, match="initial_backoff_seconds must be positive"):
        DispatchConfig(initial_backoff_seconds=0)

    with pytest.raises(DispatchError, match=r"backoff_multiplier must be >= 1.0"):
        DispatchConfig(backoff_multiplier=0.5)

    with pytest.raises(DispatchError, match="max_backoff_seconds must be >="):
        DispatchConfig(initial_backoff_seconds=10.0, max_backoff_seconds=5.0)

    with pytest.raises(DispatchError, match="circuit_breaker_failure_threshold"):
        DispatchConfig(circuit_breaker_failure_threshold=0)

    with pytest.raises(DispatchError, match="circuit_breaker_recovery_seconds"):
        DispatchConfig(circuit_breaker_recovery_seconds=0)

    with pytest.raises(DispatchError, match="http_timeout_seconds"):
        DispatchConfig(http_timeout_seconds=0)


def test_webhook_dispatcher_lifecycle_and_retries() -> None:
    """Validate WebhookDispatcher delivery, retries, terminal 4xx, and circuit breaking."""
    call_counts: dict[str, int] = {"count": 0}

    def mock_transport_success(
        _url: str, _payload: bytes, headers: dict[str, str], _timeout: float
    ) -> tuple[int, bytes]:
        call_counts["count"] += 1
        assert "X-Fraud-Signature" in headers
        assert "X-Event-Type" in headers
        assert "X-Tenant-ID" in headers
        return 200, b'{"received": true}'

    cfg = DispatchConfig(
        max_retries=2,
        initial_backoff_seconds=0.01,
        circuit_breaker_failure_threshold=2,
        circuit_breaker_recovery_seconds=1.0,
    )
    dispatcher = WebhookDispatcher(config=cfg, transport=mock_transport_success)

    sub = WebhookSubscription(
        subscription_id="sub_delivery",
        tenant_id="fintech_alpha",
        target_url="https://api.example.com/events",
        secret_token="secret_key_1234567890",  # noqa: S106
        max_retries=2,
    )
    event = FraudDomainEvent(
        event_id="evt_success",
        event_type=EventType.TRANSACTION_SCORED,
        tenant_id="fintech_alpha",
    )

    # 1. Successful dispatch
    res = dispatcher.dispatch(sub, event)
    assert res.status == EventDeliveryStatus.DELIVERED
    assert res.status_code == 200
    assert res.attempt_count == 1
    assert dispatcher.dlq.count() == 0

    # 2. Retrying server error (500) leading to dead letter queue
    fail_counts: dict[str, int] = {"count": 0}

    def mock_transport_500(
        _url: str, _payload: bytes, _headers: dict[str, str], _timeout: float
    ) -> tuple[int, bytes]:
        fail_counts["count"] += 1
        return 500, b'{"error": "internal error"}'

    dispatcher_fail = WebhookDispatcher(config=cfg, transport=mock_transport_500)
    res_fail = dispatcher_fail.dispatch(sub, event)
    assert res_fail.status == EventDeliveryStatus.DEAD_LETTER
    assert res_fail.status_code == 500
    assert res_fail.attempt_count == 3  # 1 initial + 2 retries
    assert fail_counts["count"] == 3
    assert dispatcher_fail.dlq.count() == 1

    # 3. Terminal client error (400): fails immediately without retrying
    client_err_counts: dict[str, int] = {"count": 0}

    def mock_transport_400(
        _url: str, _payload: bytes, _headers: dict[str, str], _timeout: float
    ) -> tuple[int, bytes]:
        client_err_counts["count"] += 1
        return 400, b'{"error": "bad request"}'

    dispatcher_client_err = WebhookDispatcher(config=cfg, transport=mock_transport_400)
    res_400 = dispatcher_client_err.dispatch(sub, event)
    assert res_400.status == EventDeliveryStatus.DEAD_LETTER
    assert res_400.status_code == 400
    assert client_err_counts["count"] == 1  # No retry

    # 4. Circuit breaker activation after threshold reached
    # Dispatch again on dispatcher_fail to trip circuit breaker (threshold = 2)
    dispatcher_fail.dispatch(sub, event)
    assert dispatcher_fail.is_circuit_open("sub_delivery") is True

    # Immediate rejection when circuit breaker is open
    res_cb = dispatcher_fail.dispatch(sub, event)
    assert res_cb.status == EventDeliveryStatus.FAILED
    assert res_cb.attempt_count == 0
    assert "Circuit breaker open" in str(res_cb.error_message)

    # Reset circuit breaker
    dispatcher_fail.reset_circuit_breaker("sub_delivery")
    assert dispatcher_fail.is_circuit_open("sub_delivery") is False


def test_subscription_registry_crud_and_matching() -> None:
    """Validate SubscriptionRegistry CRUD, filtering, manifest import/export, and matching."""
    reg = SubscriptionRegistry()
    assert reg.count() == 0

    sub1 = WebhookSubscription(
        subscription_id="sub_alpha",
        tenant_id="fintech_a",
        target_url="https://api.a.com/hook",
        secret_token="secret_key_1234567890",  # noqa: S106
        subscribed_events=(EventType.DECISION_DENIED,),
    )
    sub2 = WebhookSubscription(
        subscription_id="sub_beta",
        tenant_id="fintech_b",
        target_url="https://api.b.com/hook",
        secret_token="secret_key_0987654321",  # noqa: S106
        subscribed_events=(EventType.TRANSACTION_SCORED, EventType.DECISION_DENIED),
    )

    reg.register(sub1)
    reg.register(sub2)
    assert reg.count() == 2
    assert reg.count("fintech_a") == 1
    assert reg.get("sub_alpha") == sub1

    # Duplicate rejection without overwrite
    with pytest.raises(SubscriptionError, match="already registered"):
        reg.register(sub1, overwrite=False)

    # Overwrite
    sub1_mod = WebhookSubscription(
        subscription_id="sub_alpha",
        tenant_id="fintech_a",
        target_url="https://api.a.com/v2/hook",
        secret_token="secret_key_1234567890",  # noqa: S106
    )
    reg.register(sub1_mod, overwrite=True)
    assert reg.get("sub_alpha") == sub1_mod

    # Matching logic
    evt_denied_a = FraudDomainEvent(
        event_id="e1",
        event_type=EventType.DECISION_DENIED,
        tenant_id="fintech_a",
    )
    matches = reg.find_matching(evt_denied_a)
    assert len(matches) == 1
    assert matches[0].subscription_id == "sub_alpha"

    # Manifest export and import
    manifest = reg.export_manifest()
    assert manifest["version"] == "1.0"
    assert len(manifest["subscriptions"]) == 2

    restored_reg = SubscriptionRegistry.import_manifest(manifest)
    assert restored_reg.count() == 2
    assert restored_reg.get("sub_beta") is not None

    with pytest.raises(SubscriptionError, match="'subscriptions' must be a list"):
        SubscriptionRegistry.import_manifest({"subscriptions": "invalid"})

    # Remove
    assert reg.remove("sub_alpha") is True
    assert reg.remove("sub_alpha") is False
    assert reg.count() == 1


def test_event_bus_in_memory_and_webhook_dispatch() -> None:
    """Validate EventBus in-memory listener callbacks, filtering, and webhook dispatches."""
    reg = SubscriptionRegistry()
    delivered_payloads: list[bytes] = []

    def mock_transport(
        _url: str, payload: bytes, _headers: dict[str, str], _timeout: float
    ) -> tuple[int, bytes]:
        delivered_payloads.append(payload)
        return 200, b'{"ok": true}'

    dispatcher = WebhookDispatcher(transport=mock_transport)
    bus = EventBus(registry=reg, dispatcher=dispatcher, max_workers=2)

    received_events: list[FraudDomainEvent] = []

    def listener(event: FraudDomainEvent) -> None:
        received_events.append(event)

    # Subscribe in-memory
    token = bus.subscribe(listener, event_type=EventType.DECISION_DENIED)

    # Register webhook
    sub = WebhookSubscription(
        subscription_id="sub_bus",
        tenant_id="fintech_alpha",
        target_url="https://api.example.com/fraud",
        secret_token="secret_key_1234567890",  # noqa: S106
        subscribed_events=(EventType.DECISION_DENIED,),
    )
    reg.register(sub)

    # Publish matched event
    matched = FraudDomainEvent(
        event_id="evt_bus_01",
        event_type=EventType.DECISION_DENIED,
        tenant_id="fintech_alpha",
        payload={"score": 0.95},
    )
    count = bus.publish(matched, synchronous=True)
    assert count == 2  # 1 in-memory + 1 webhook
    assert len(received_events) == 1
    assert len(delivered_payloads) == 1

    # Publish unmatched event (TRANSACTION_SCORED)
    unmatched = FraudDomainEvent(
        event_id="evt_bus_02",
        event_type=EventType.TRANSACTION_SCORED,
        tenant_id="fintech_alpha",
    )
    count_unmatched = bus.publish(unmatched, synchronous=True)
    assert count_unmatched == 0
    assert len(received_events) == 1

    # Unsubscribe in-memory
    assert bus.unsubscribe(token) is True
    assert bus.unsubscribe("invalid_token") is False

    bus.close()
