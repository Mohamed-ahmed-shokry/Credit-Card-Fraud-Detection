"""Unit tests for domain events and webhook models."""

from __future__ import annotations

import pytest

from fraud_detection.events import (
    EventDeliveryStatus,
    EventError,
    EventType,
    FraudDomainEvent,
    SubscriptionError,
    WebhookSubscription,
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
