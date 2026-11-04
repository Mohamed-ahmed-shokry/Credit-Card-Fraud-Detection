"""Unit and integration tests for Card Scheme Chargeback & Representment Gateway."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

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
    assert d_naive.is_expired is True

    # Invalid deadline format returns False
    d_inv = ChargebackDispute(transaction_id="tx_inv")
    d_inv.response_deadline = "invalid_date_format"
    assert d_inv.is_expired is False

    # Terminal disputes are not flagged as active-expired
    d_resolved = ChargebackDispute(
        transaction_id="tx_resolved",
        stage=ChargebackStage.RESOLVED_WON,
        response_deadline=(datetime.now(UTC) - timedelta(days=2)).isoformat(),
    )
    assert d_resolved.is_expired is False
    assert d_resolved.is_terminal is True


def test_representment_viability_engine_3ds_and_ce3() -> None:
    """Verify representment viability evaluation for 3DS liability shift and CE3.0."""
    from fraud_detection.chargeback import (
        RepresentmentViability,
        evaluate_representment_viability,
    )

    dispute = ChargebackDispute(
        transaction_id="tx_viability_1",
        scheme=CardScheme.VISA,
        reason_code="10.4",
        category=ChargebackCategory.FRAUD_CARD_NOT_PRESENT,
        disputed_amount=150.0,
    )

    # 1. Low score / Accept liability with no evidence
    v_empty = evaluate_representment_viability(dispute)
    assert isinstance(v_empty, RepresentmentViability)
    assert v_empty.winnability_score == 0.0
    assert v_empty.liability_shift_proven is False
    assert v_empty.compelling_evidence_qualified is False
    assert v_empty.recommended_action == RepresentmentAction.ACCEPT_LIABILITY
    assert len(v_empty.evidence_requirements) >= 2
    assert "winnability_score" in v_empty.to_dict()

    # 2. 3DS liability shift with ECI 05
    v_3ds = evaluate_representment_viability(
        dispute,
        transaction_context={"eci": "05", "is_3ds_authenticated": True},
    )
    assert v_3ds.liability_shift_proven is True
    assert v_3ds.winnability_score >= 0.55
    assert dispute.liability_shift_proven is True
    assert v_3ds.recommended_action == RepresentmentAction.REQUEST_MORE_INFO

    # 3. 3DS verified challenge session + CE3.0 qualifying transactions + fingerprinted evidence
    priors = [
        {
            "transaction_id": "tx_old_1",
            "device_id": "dev_abc",
            "ip_address": "1.2.3.4",
            "user_id": "u_99",
        },
        {
            "transaction_id": "tx_old_2",
            "device_id": "dev_abc",
            "ip_address": "1.2.3.4",
            "user_id": "u_99",
        },
        {"transaction_id": "tx_fraud", "device_id": "dev_abc", "is_fraud": True},  # Ignored
        "not_a_dict_prior",  # Skipped safely
    ]
    evidence_pkg = {
        "package_id": "pkg_123",
        "sha256_fingerprint": "a" * 64,
    }
    v_full = evaluate_representment_viability(
        dispute,
        transaction_context={"device_id": "dev_abc", "ip_address": "1.2.3.4", "user_id": "u_99"},
        challenge_session={"status": "verified"},
        prior_customer_transactions=priors,
        evidence_package=evidence_pkg,
    )
    assert v_full.liability_shift_proven is True
    assert v_full.compelling_evidence_qualified is True
    assert v_full.winnability_score >= 0.70
    assert v_full.recommended_action == RepresentmentAction.SUBMIT_REPRESENTMENT
    assert dispute.winnability_score == v_full.winnability_score


def test_representment_viability_auth_and_consumer_disputes() -> None:
    """Verify viability scoring for authorization and consumer delivery disputes."""
    from fraud_detection.chargeback import evaluate_representment_viability

    # Authorization error dispute
    d_auth = ChargebackDispute(
        transaction_id="tx_auth_1",
        scheme=CardScheme.MASTERCARD,
        reason_code="4808",
        category=ChargebackCategory.AUTHORIZATION_ERROR,
    )
    v_auth_no_code = evaluate_representment_viability(d_auth)
    assert v_auth_no_code.recommended_action == RepresentmentAction.ACCEPT_LIABILITY

    v_auth_with_code = evaluate_representment_viability(
        d_auth,
        transaction_context={"auth_code": "AUTH_99124"},
    )
    assert v_auth_with_code.winnability_score >= 0.40
    assert v_auth_with_code.recommended_action == RepresentmentAction.REQUEST_MORE_INFO

    # Consumer dispute (Goods Not Received)
    d_consumer = ChargebackDispute(
        transaction_id="tx_consumer_1",
        scheme=CardScheme.VISA,
        reason_code="13.1",
        category=ChargebackCategory.CONSUMER_DISPUTE,
    )
    v_cons_tracking = evaluate_representment_viability(
        d_consumer,
        transaction_context={"tracking_number": "1Z9999999999999999"},
        evidence_package={"package_id": "pkg_unhashed"},  # Unfingerprinted package branch
    )
    assert v_cons_tracking.winnability_score >= 0.45
    assert v_cons_tracking.recommended_action == RepresentmentAction.REQUEST_MORE_INFO


def test_chargeback_store_indexing_and_eviction() -> None:
    """Verify ChargebackStore indices, filtering, capacity eviction, and deletion."""
    from fraud_detection.chargeback import ChargebackStore

    store = ChargebackStore(max_capacity=3)

    d1 = ChargebackDispute(
        transaction_id="tx_s1",
        tenant_id="tenant_a",
        scheme=CardScheme.VISA,
        scheme_dispute_id="vrol_1001",
        disputed_amount=100.0,
    )
    d2 = ChargebackDispute(
        transaction_id="tx_s2",
        tenant_id="tenant_b",
        scheme=CardScheme.MASTERCARD,
        scheme_dispute_id="mcom_2002",
        disputed_amount=250.0,
    )
    store.put(d1)
    store.put(d2)

    assert store.count() == 2
    assert store.count("tenant_a") == 1
    assert store.get(d1.dispute_id) == d1
    assert store.get_by_transaction("tx_s1") == d1
    assert store.get_by_scheme_id(CardScheme.VISA, "vrol_1001") == d1

    # Filtered list
    res_a = store.list_disputes(tenant_id="tenant_a")
    assert len(res_a) == 1
    assert res_a[0].dispute_id == d1.dispute_id

    res_mc = store.list_disputes(scheme=CardScheme.MASTERCARD)
    assert len(res_mc) == 1

    # Capacity eviction
    d3 = ChargebackDispute(transaction_id="tx_s3", disputed_amount=50.0)
    d4 = ChargebackDispute(transaction_id="tx_s4", disputed_amount=75.0)
    store.put(d3)
    assert store.count() == 3
    store.put(d4)  # Evicts d1
    assert store.count() == 3
    assert store.get(d1.dispute_id) is None
    assert store.get_by_transaction("tx_s1") is None
    assert store.get_by_scheme_id(CardScheme.VISA, "vrol_1001") is None

    # Delete
    assert store.delete("nonexistent_id") is False
    assert store.delete(d2.dispute_id) is True
    assert store.get(d2.dispute_id) is None
    assert store.get_by_scheme_id(CardScheme.MASTERCARD, "mcom_2002") is None

    # Clear
    store.clear()
    assert store.count() == 0


def test_chargeback_manager_lifecycle_and_resolution() -> None:
    """Verify ChargebackManager ingestion, stage transitions, representment, and summary."""
    from fraud_detection.chargeback import (
        ChargebackManager,
        ChargebackStore,
    )

    store = ChargebackStore()
    mgr = ChargebackManager(store=store)

    # 1. Ingest with auto-evaluation
    disp = mgr.ingest_chargeback(
        transaction_id="tx_life_001",
        scheme=CardScheme.VISA,
        reason_code="10.4",
        disputed_amount=450.0,
        currency="USD",
        tenant_id="fintech_us",
        scheme_dispute_id="vrol_life_1",
        transaction_context={"eci": "05", "is_3ds_authenticated": True},
    )
    assert disp.dispute_id.startswith("dsp_")
    assert disp.liability_shift_proven is True
    assert disp.stage == ChargebackStage.FIRST_CHARGEBACK
    assert mgr.get_dispute(disp.dispute_id) == disp

    # Missing dispute raises DisputeNotFoundError
    with pytest.raises(DisputeNotFoundError, match="Chargeback dispute 'unknown' not found"):
        mgr.get_dispute("unknown")

    # 2. Stage transition: same stage is no-op
    curr = mgr.update_stage(disp.dispute_id, ChargebackStage.FIRST_CHARGEBACK)
    assert curr.stage == ChargebackStage.FIRST_CHARGEBACK

    # 3. Invalid transition (FIRST_CHARGEBACK directly to RESOLVED_WON)
    with pytest.raises(InvalidDisputeStateTransitionError, match="Illegal transition"):
        mgr.update_stage(disp.dispute_id, ChargebackStage.RESOLVED_WON)

    # 4. Represent dispute
    disp_rep = mgr.represent_dispute(
        disp.dispute_id,
        package_id="pkg_represent_001",
        evidence_fingerprint="f" * 64,
        notes="Representing with 3DS ECI 05 evidence.",
        actor="senior_analyst",
    )
    assert disp_rep.stage == ChargebackStage.REPRESENTED
    assert disp_rep.representment_package_id == "pkg_represent_001"
    assert disp_rep.represented_at is not None

    # 5. Resolve dispute with won outcome
    disp_won = mgr.resolve_dispute(
        disp.dispute_id,
        outcome=ChargebackStage.RESOLVED_WON,
        actor="visa_network",
        notes="Issuer conceded liability.",
    )
    assert disp_won.stage == ChargebackStage.RESOLVED_WON
    assert disp_won.resolved_at is not None
    assert disp_won.is_terminal is True

    # 6. Representing a terminal dispute raises InvalidDisputeStateTransitionError
    with pytest.raises(
        InvalidDisputeStateTransitionError, match="Cannot represent terminal dispute"
    ):
        mgr.represent_dispute(disp.dispute_id)

    # 7. Invalid resolution outcome
    d_unres = mgr.ingest_chargeback(
        transaction_id="tx_unres",
        scheme=CardScheme.MASTERCARD,
        reason_code="4837",
        disputed_amount=120.0,
    )
    with pytest.raises(DisputeValidationError, match="Invalid outcome 'first_chargeback'"):
        mgr.resolve_dispute(d_unres.dispute_id, outcome=ChargebackStage.FIRST_CHARGEBACK)

    # 8. Summary aggregation
    summary = mgr.get_summary()
    assert summary["total_disputes"] == 2
    assert summary["total_disputed_amount"] == 570.0
    assert summary["by_stage"]["resolved_won"] == 1
    assert summary["by_stage"]["first_chargeback"] == 1
    assert summary["by_scheme"]["visa"] == 1
    assert summary["by_scheme"]["mastercard"] == 1


def test_chargeback_edge_cases_and_branch_coverage() -> None:
    """Cover edge cases across dispute parsing, store deletion, filters, and summary metrics."""
    from fraud_detection.chargeback import (
        ChargebackManager,
        ChargebackStore,
        evaluate_representment_viability,
    )

    # 1. Fallback when filing_date is unparseable
    d_bad_date = ChargebackDispute(transaction_id="tx_bad_date", filing_date="not-a-valid-date")
    assert d_bad_date.response_deadline != ""

    # 2. is_expired with naive deadline (tzinfo is None)
    d_naive = ChargebackDispute(transaction_id="tx_naive_exp")
    d_naive.response_deadline = (
        datetime.now(UTC).replace(tzinfo=None) - timedelta(days=5)
    ).strftime("%Y-%m-%dT%H:%M:%S")
    assert d_naive.is_expired is True

    # 3. Viability with only 1 matching prior transaction (not meeting CE3.0 >=2 requirement)
    d_fraud = ChargebackDispute(
        transaction_id="tx_ce3_one",
        scheme=CardScheme.VISA,
        reason_code="10.4",
        category=ChargebackCategory.FRAUD_CARD_NOT_PRESENT,
    )
    v_one = evaluate_representment_viability(
        d_fraud,
        transaction_context={"device_id": "dev_one", "ip_address": "9.9.9.9", "user_id": "u_one"},
        prior_customer_transactions=[
            {"transaction_id": "tx_prev", "device_id": "dev_one"},  # Only 1 match
        ],
    )
    assert v_one.compelling_evidence_qualified is False
    assert any(
        "Requires at least 2 qualifying prior transactions" in r
        for r in v_one.evidence_requirements
    )

    # 4. Viability consumer dispute without tracking or delivery confirmation
    d_cons_notrack = ChargebackDispute(
        transaction_id="tx_notrack",
        scheme=CardScheme.VISA,
        reason_code="13.1",
        category=ChargebackCategory.CONSUMER_DISPUTE,
    )
    v_notrack = evaluate_representment_viability(d_cons_notrack)
    assert any("Proof of merchandise delivery" in r for r in v_notrack.evidence_requirements)

    # 5. Store delete index collision edge cases
    store = ChargebackStore()
    d_shared1 = ChargebackDispute(transaction_id="tx_shared_del", scheme_dispute_id="dup_id")
    d_shared2 = ChargebackDispute(transaction_id="tx_shared_del", scheme_dispute_id="dup_id")
    store.put(d_shared1)
    store.put(d_shared2)  # Overwrites tx and scheme indices to point to d_shared2
    assert store.delete(d_shared1.dispute_id) is True  # Delete d_shared1 without breaking d_shared2
    assert store.get_by_transaction("tx_shared_del") == d_shared2

    # 6. Store list_disputes with category, stage, and expired_only filters
    d_exp = ChargebackDispute(
        transaction_id="tx_exp_store",
        category=ChargebackCategory.PROCESSING_ERROR,
        stage=ChargebackStage.PRE_ARBITRATION,
    )
    d_exp.response_deadline = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    store.put(d_exp)

    res_cat = store.list_disputes(category=ChargebackCategory.PROCESSING_ERROR)
    assert len(res_cat) == 1
    res_stage = store.list_disputes(stage=ChargebackStage.PRE_ARBITRATION)
    assert len(res_stage) == 1
    res_exp = store.list_disputes(expired_only=True)
    assert len(res_exp) == 1

    # 7. Ingest with auto_evaluate=False
    mgr = ChargebackManager(store=store)
    d_no_eval = mgr.ingest_chargeback(
        transaction_id="tx_no_eval",
        scheme=CardScheme.AMEX,
        reason_code="FR2",
        disputed_amount=50.0,
        auto_evaluate=False,
    )
    assert d_no_eval.winnability_score == 0.0

    # 8. Summary counting expired disputes and high winnability disputes
    d_high = mgr.ingest_chargeback(
        transaction_id="tx_high_win",
        scheme=CardScheme.VISA,
        reason_code="10.4",
        disputed_amount=300.0,
        transaction_context={"eci": "05", "is_3ds_authenticated": True},
        evidence_package={"package_id": "pkg_win", "evidence_fingerprint": "a" * 64},
    )
    assert d_high.winnability_score >= 0.70
    summary = mgr.get_summary()
    assert summary["expired_count"] >= 1
    assert summary["high_winnability_count"] >= 1


def test_chargeback_event_bus_and_feedback_integration() -> None:
    """Verify EventBus domain event publishing and LabelIngestionStore integration."""
    from fraud_detection.chargeback import ChargebackManager, ChargebackStore
    from fraud_detection.events import EventBus, EventType, FraudDomainEvent
    from fraud_detection.feedback import DelayedLabelRecord, DisputeType, LabelIngestionStore

    published_events: list[FraudDomainEvent] = []
    bus = EventBus()
    bus.subscribe(lambda ev: published_events.append(ev))

    store = ChargebackStore()
    mgr = ChargebackManager(store=store, event_bus=bus)

    # 1. Ingest chargeback publishes CHARGEBACK_INGESTED
    dispute = mgr.ingest_chargeback(
        transaction_id="tx_event_1",
        scheme=CardScheme.VISA,
        reason_code="10.4",
        disputed_amount=120.0,
        tenant_id="tenant_alpha",
    )
    assert any(
        ev.event_type == EventType.CHARGEBACK_INGESTED
        and ev.tenant_id == "tenant_alpha"
        and ev.payload["dispute_id"] == dispute.dispute_id
        for ev in published_events
    )

    # 2. Represent dispute publishes CHARGEBACK_REPRESENTED
    mgr.represent_dispute(
        dispute.dispute_id,
        package_id="pkg_test",
        evidence_fingerprint="f" * 64,
        notes="Strong proof",
    )
    assert any(
        ev.event_type == EventType.CHARGEBACK_REPRESENTED
        and ev.payload["dispute_id"] == dispute.dispute_id
        for ev in published_events
    )

    # 3. Resolve dispute publishes CHARGEBACK_RESOLVED
    mgr.resolve_dispute(
        dispute.dispute_id,
        outcome=ChargebackStage.RESOLVED_LOST,
        notes="Issuer rejected representment",
    )
    assert any(
        ev.event_type == EventType.CHARGEBACK_RESOLVED
        and ev.payload["stage"] == ChargebackStage.RESOLVED_LOST.value
        for ev in published_events
    )

    # 4. Feedback label publication for fraud dispute resolved lost
    label_store = LabelIngestionStore()
    record1 = mgr.publish_feedback_label(dispute.dispute_id, label_store=label_store)
    assert isinstance(record1, DelayedLabelRecord)
    assert record1.is_fraud is True
    assert record1.dispute_type == DisputeType.FRAUD_DISPUTE
    assert record1.amount == 120.0
    assert record1.transaction_id == "tx_event_1"

    # 5. Non-fraud category (consumer dispute) resolves to is_fraud=False
    dispute_consumer = mgr.ingest_chargeback(
        transaction_id="tx_event_consumer",
        scheme=CardScheme.MASTERCARD,
        reason_code="4853",
        disputed_amount=75.0,
    )
    record2 = mgr.publish_feedback_label(dispute_consumer.dispute_id, label_store=label_store)
    assert record2.is_fraud is False
    assert record2.dispute_type == DisputeType.CHARGEBACK

    # 6. Fraud dispute resolved won yields is_fraud=False
    dispute_won = mgr.ingest_chargeback(
        transaction_id="tx_event_won",
        scheme=CardScheme.VISA,
        reason_code="10.4",
        disputed_amount=250.0,
    )
    mgr.represent_dispute(dispute_won.dispute_id)
    mgr.resolve_dispute(dispute_won.dispute_id, outcome=ChargebackStage.RESOLVED_WON)
    record3 = mgr.publish_feedback_label(dispute_won.dispute_id, label_store=label_store)
    assert record3.is_fraud is False

    # 7. Explicit is_fraud override and future event timestamp clamped
    future_ts = 9999999999.0
    record4 = mgr.publish_feedback_label(
        dispute_won.dispute_id,
        label_store=label_store,
        is_fraud=True,
        event_timestamp=future_ts,
    )
    assert record4.is_fraud is True
    assert record4.event_timestamp <= record4.label_timestamp

    # 8. Resilience when event_bus raises an exception
    class FailingBus:
        def publish(self, _event: Any) -> None:
            raise RuntimeError("Kafka broker down")

    failing_mgr = ChargebackManager(event_bus=FailingBus())
    d_fail = failing_mgr.ingest_chargeback(
        transaction_id="tx_bus_fail",
        scheme=CardScheme.AMEX,
        reason_code="FR2",
        disputed_amount=30.0,
    )
    assert d_fail.transaction_id == "tx_bus_fail"
