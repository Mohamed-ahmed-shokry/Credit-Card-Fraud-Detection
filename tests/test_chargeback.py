"""Unit and integration tests for Card Scheme Chargeback & Representment Gateway."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fraud_detection.chargeback import (
    ALLOWED_DISPUTE_TRANSITIONS,
    CardScheme,
    ChargebackCategory,
    ChargebackDispute,
    ChargebackError,
    ChargebackStage,
    DisputeNotFoundError,
    DisputeValidationError,
    InvalidDisputeStateTransitionError,
    ReasonCodeDefinition,
    ReasonCodeNotFoundError,
    ReasonCodeRegistry,
    RepresentmentAction,
    RepresentmentError,
)


def test_chargeback_exception_hierarchy() -> None:
    """Verify that all chargeback errors inherit from ChargebackError."""
    assert issubclass(DisputeNotFoundError, ChargebackError)
    assert issubclass(InvalidDisputeStateTransitionError, ChargebackError)
    assert issubclass(DisputeValidationError, ChargebackError)
    assert issubclass(RepresentmentError, ChargebackError)
    assert issubclass(ReasonCodeNotFoundError, ChargebackError)
    err = DisputeNotFoundError("Dispute not found")
    assert str(err) == "Dispute not found"


def test_scheme_and_category_enums() -> None:
    """Verify enum members and values for card schemes, categories, and stages."""
    assert CardScheme.VISA.value == "visa"
    assert CardScheme.MASTERCARD.value == "mastercard"
    assert CardScheme.AMEX.value == "amex"
    assert CardScheme.DISCOVER.value == "discover"
    assert CardScheme.JCB.value == "jcb"

    assert ChargebackCategory.FRAUD_CARD_NOT_PRESENT.value == "fraud_card_not_present"
    assert ChargebackCategory.FRIENDLY_FRAUD_FIRST_PARTY.value == "friendly_fraud_first_party"
    assert ChargebackCategory.CONSUMER_DISPUTE.value == "consumer_dispute"

    assert ChargebackStage.FIRST_CHARGEBACK.value == "first_chargeback"
    assert ChargebackStage.REPRESENTED.value == "represented"
    assert ChargebackStage.RESOLVED_WON.value == "resolved_won"
    assert ChargebackStage.RESOLVED_LOST.value == "resolved_lost"

    assert RepresentmentAction.SUBMIT_REPRESENTMENT.value == "submit_representment"
    assert RepresentmentAction.ACCEPT_LIABILITY.value == "accept_liability"


def test_allowed_dispute_transitions() -> None:
    """Verify state machine transition definitions."""
    assert (
        ChargebackStage.REPRESENTED in ALLOWED_DISPUTE_TRANSITIONS[ChargebackStage.FIRST_CHARGEBACK]
    )
    assert ChargebackStage.RESOLVED_WON in ALLOWED_DISPUTE_TRANSITIONS[ChargebackStage.REPRESENTED]
    assert ChargebackStage.RESOLVED_LOST in ALLOWED_DISPUTE_TRANSITIONS[ChargebackStage.REPRESENTED]
    # Terminal stages have empty transition sets
    assert len(ALLOWED_DISPUTE_TRANSITIONS[ChargebackStage.RESOLVED_WON]) == 0
    assert len(ALLOWED_DISPUTE_TRANSITIONS[ChargebackStage.RESOLVED_LOST]) == 0
    assert len(ALLOWED_DISPUTE_TRANSITIONS[ChargebackStage.ACCEPTED_LIABILITY]) == 0
    assert len(ALLOWED_DISPUTE_TRANSITIONS[ChargebackStage.EXPIRED]) == 0


def test_reason_code_registry() -> None:
    """Verify reason code registry lookups, additions, fallbacks, and serialization."""
    registry = ReasonCodeRegistry()

    # Visa 10.4 lookup
    v_104 = registry.lookup(CardScheme.VISA, "10.4")
    assert v_104.scheme == CardScheme.VISA
    assert v_104.category == ChargebackCategory.FRAUD_CARD_NOT_PRESENT
    assert v_104.compelling_evidence_eligible is True
    assert v_104.default_response_days == 30
    assert "code" in v_104.to_dict()

    # Mastercard 4837 lookup
    mc_4837 = registry.lookup(CardScheme.MASTERCARD, "4837")
    assert mc_4837.category == ChargebackCategory.FRAUD_CARD_NOT_PRESENT
    assert mc_4837.default_response_days == 45

    # Unregistered lookup raises ReasonCodeNotFoundError
    with pytest.raises(ReasonCodeNotFoundError, match="Reason code '9999' not registered"):
        registry.lookup(CardScheme.VISA, "9999")

    # find_or_fallback handles unregistered cleanly
    fallback = registry.find_or_fallback(CardScheme.VISA, "9999")
    assert fallback.code == "9999"
    assert fallback.category == ChargebackCategory.OTHER
    assert fallback.compelling_evidence_eligible is False

    # Custom registration
    custom_defn = ReasonCodeDefinition(
        scheme=CardScheme.VISA,
        code="CUSTOM_1",
        description="Custom Test Reason",
        category=ChargebackCategory.AUTHORIZATION_ERROR,
        default_response_days=21,
        compelling_evidence_eligible=False,
    )
    registry.register(custom_defn)
    assert registry.lookup(CardScheme.VISA, "CUSTOM_1").default_response_days == 21

    # list_codes
    visa_codes = registry.list_codes(scheme=CardScheme.VISA)
    assert len(visa_codes) >= 5
    all_codes = registry.list_codes()
    assert len(all_codes) >= len(visa_codes)


def test_chargeback_dispute_model_and_validation() -> None:
    """Verify ChargebackDispute validation, defaults, deadline computation, and serialization."""
    # Validation errors
    with pytest.raises(DisputeValidationError, match="transaction_id cannot be empty"):
        ChargebackDispute(transaction_id="   ")

    with pytest.raises(DisputeValidationError, match="disputed_amount cannot be negative"):
        ChargebackDispute(transaction_id="tx_123", disputed_amount=-15.0)

    # Valid dispute creation
    dispute = ChargebackDispute(
        transaction_id="tx_valid_001",
        tenant_id="fintech_eu",
        scheme=CardScheme.VISA,
        reason_code="10.4",
        disputed_amount=299.95,
        currency="EUR",
    )
    assert dispute.dispute_id.startswith("dsp_")
    assert dispute.stage == ChargebackStage.FIRST_CHARGEBACK
    assert dispute.response_deadline != ""
    assert dispute.is_terminal is False
    assert dispute.is_expired is False
    assert len(dispute.audit_trail) == 1
    assert dispute.audit_trail[0]["action"] == "dispute_ingested"

    # Audit entry addition
    dispute.add_audit_entry(
        action="analyst_assigned",
        actor="lead_analyst",
        details={"analyst": "bob"},
    )
    assert len(dispute.audit_trail) == 2
    assert dispute.audit_trail[1]["action"] == "analyst_assigned"

    # to_dict and from_dict roundtrip
    d = dispute.to_dict()
    assert d["transaction_id"] == "tx_valid_001"
    assert d["scheme"] == "visa"
    assert d["disputed_amount"] == 299.95

    hydrated = ChargebackDispute.from_dict(d)
    assert hydrated.dispute_id == dispute.dispute_id
    assert hydrated.transaction_id == dispute.transaction_id
    assert hydrated.disputed_amount == dispute.disputed_amount
    assert hydrated.stage == dispute.stage
    assert len(hydrated.audit_trail) == len(dispute.audit_trail)


def test_chargeback_dispute_expiration_and_terminal() -> None:
    """Verify is_expired and is_terminal properties across lifecycle stages."""
    # Non-expired dispute with future deadline
    d_future = ChargebackDispute(transaction_id="tx_future")
    d_future.response_deadline = (datetime.now(UTC) + timedelta(days=10)).isoformat()
    assert d_future.is_expired is False

    # Expired dispute with past deadline
    d_past = ChargebackDispute(transaction_id="tx_past")
    d_past.response_deadline = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    assert d_past.is_expired is True

    # Naive deadline (no tzinfo)
    d_naive = ChargebackDispute(transaction_id="tx_naive")
    d_naive.response_deadline = (
        datetime.now(UTC).replace(tzinfo=None) - timedelta(days=2)
    ).strftime("%Y-%m-%dT%H:%M:%S")
    assert d_past.is_expired is True

    # Invalid deadline format returns False
    d_inv = ChargebackDispute(transaction_id="tx_inv")
    d_inv.response_deadline = "invalid_date_format"
    assert d_inv.is_expired is False

    # Terminal disputes are not flagged as active-expired
    d_past.stage = ChargebackStage.RESOLVED_WON
    assert d_past.is_expired is False
    assert d_past.is_terminal is True
