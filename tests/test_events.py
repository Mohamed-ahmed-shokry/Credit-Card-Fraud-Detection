"""Unit tests for domain events and webhook models."""

from __future__ import annotations

import pytest

from fraud_detection.events import (
    EventDeliveryStatus,
    EventError,
    EventType,
    FraudDomainEvent,
    SignatureVerificationError,
    SubscriptionError,
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
