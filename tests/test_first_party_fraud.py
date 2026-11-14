"""Unit tests for the Phase 51 First-Party Fraud & Refund Abuse Detection Engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fraud_detection.cases import CaseManager
from fraud_detection.events import EventBus, EventType, FraudDomainEvent
from fraud_detection.first_party_fraud import (
    WARDROBING_CATEGORIES,
    ClaimOutcome,
    FirstPartyAccountProfile,
    FirstPartyAccountStore,
    FirstPartyAction,
    FirstPartyAnomalyType,
    FirstPartyAssessment,
    FirstPartyEventRecord,
    FirstPartyFraudEngine,
    FirstPartyRiskLevel,
    FriendlyFraudAnalyzer,
    OrderClaimType,
    PromoAbuseDetector,
    RefundAbuseAnalyzer,
    _coerce_utc_datetime,
)


def test_first_party_enums_and_model_round_trip() -> None:
    assert FirstPartyRiskLevel.CRITICAL.value == "critical"
    assert FirstPartyAction.RESTRICT_INSTANT_REFUND.value == "restrict_instant_refund"
    assert FirstPartyAnomalyType.WARDROBING_PATTERN.value == "wardrobing_pattern"
    assert OrderClaimType.INR_CLAIM.value == "inr_claim"
    assert ClaimOutcome.APPROVED.value == "approved"
    assert "apparel" in WARDROBING_CATEGORIES

    naive = datetime(2026, 11, 14, 12, 0, 0)  # noqa: DTZ001
    assert _coerce_utc_datetime(naive).tzinfo is UTC
    assert _coerce_utc_datetime("2026-11-14T12:00:00+00:00").tzinfo is UTC

    record = FirstPartyEventRecord(
        event_id="evt-1",
        account_id="acct-1",
        claim_type=OrderClaimType.PURCHASE,
        amount=250.0,
        timestamp=datetime(2026, 11, 14, 12, 0, 0, tzinfo=UTC),
        order_id="ord-1",
        merchant_category="apparel",
        avs_cvv_matched=True,
        three_ds_authenticated=True,
        delivery_confirmed=True,
    )
    restored_record = FirstPartyEventRecord.from_dict(record.to_dict())
    assert restored_record == record

    with pytest.raises(ValueError, match="event_id"):
        FirstPartyEventRecord(
            event_id="",
            account_id="acct-1",
            claim_type=OrderClaimType.PURCHASE,
            amount=10.0,
            timestamp=datetime.now(UTC),
        )
    with pytest.raises(ValueError, match="account_id"):
        FirstPartyEventRecord(
            event_id="evt-2",
            account_id="",
            claim_type=OrderClaimType.PURCHASE,
            amount=10.0,
            timestamp=datetime.now(UTC),
        )
    with pytest.raises(ValueError, match="amount"):
        FirstPartyEventRecord(
            event_id="evt-3",
            account_id="acct-1",
            claim_type=OrderClaimType.PURCHASE,
            amount=-5.0,
            timestamp=datetime.now(UTC),
        )

    profile = FirstPartyAccountProfile(
        account_id="acct-1",
        events=[record],
        linked_accounts={"acct-2"},
        risk_level=FirstPartyRiskLevel.MEDIUM,
        last_action=FirstPartyAction.REQUIRE_SIGNATURE_DELIVERY,
        last_assessed_at=datetime(2026, 11, 14, 13, 0, 0, tzinfo=UTC),
        last_assessment_id="fpa-1",
    )
    restored_profile = FirstPartyAccountProfile.from_dict(profile.to_dict())
    assert restored_profile.account_id == "acct-1"
    assert restored_profile.linked_accounts == {"acct-2"}
    assert len(restored_profile.events) == 1

    assessment = FirstPartyAssessment(
        assessment_id="fpa-1",
        account_id="acct-1",
        tenant_id="default",
        risk_score=0.68,
        risk_level=FirstPartyRiskLevel.HIGH,
        action=FirstPartyAction.RESTRICT_INSTANT_REFUND,
        anomalies=(FirstPartyAnomalyType.HIGH_RETURN_RATIO,),
        return_ratio=0.72,
        purchase_volume=1000.0,
        refund_volume=720.0,
        dispute_count=1,
        inr_claim_count=1,
        promo_redemption_count=2,
        linked_account_count=1,
        reasons=("Elevated return ratio",),
        evaluated_at=datetime(2026, 11, 14, 13, 0, 0, tzinfo=UTC),
    )
    restored_assessment = FirstPartyAssessment.from_dict(assessment.to_dict())
    assert restored_assessment == assessment


def test_refund_abuse_analyzer_detects_wardrobing_inr_and_skew() -> None:
    analyzer = RefundAbuseAnalyzer()
    empty = analyzer.analyze([])
    assert empty.return_ratio == 0.0
    assert empty.anomalies == ()

    now = datetime(2026, 11, 14, 12, 0, 0, tzinfo=UTC)
    events = [
        FirstPartyEventRecord(
            event_id="p-1",
            account_id="acct-1",
            claim_type=OrderClaimType.PURCHASE,
            amount=100.0,
            timestamp=now - timedelta(days=10),
            merchant_category="apparel",
        ),
        FirstPartyEventRecord(
            event_id="p-2",
            account_id="acct-1",
            claim_type=OrderClaimType.PURCHASE,
            amount=100.0,
            timestamp=now - timedelta(days=8),
            merchant_category="electronics",
        ),
        FirstPartyEventRecord(
            event_id="r-1",
            account_id="acct-1",
            claim_type=OrderClaimType.REFUND_RETURN,
            amount=200.0,
            timestamp=now - timedelta(days=5),
            merchant_category="apparel",
        ),
        FirstPartyEventRecord(
            event_id="r-2",
            account_id="acct-1",
            claim_type=OrderClaimType.REFUND_RETURN,
            amount=180.0,
            timestamp=now - timedelta(days=3),
            merchant_category="electronics",
        ),
        FirstPartyEventRecord(
            event_id="inr-1",
            account_id="acct-1",
            claim_type=OrderClaimType.INR_CLAIM,
            amount=150.0,
            timestamp=now - timedelta(days=2),
        ),
        FirstPartyEventRecord(
            event_id="inr-2",
            account_id="acct-1",
            claim_type=OrderClaimType.INR_CLAIM,
            amount=150.0,
            timestamp=now - timedelta(days=1),
        ),
    ]
    metrics = analyzer.analyze(events, now=now)
    assert metrics.purchase_volume == 200.0
    assert metrics.refund_volume == 680.0
    assert metrics.return_ratio == 2.0
    assert metrics.inr_claim_count == 2
    assert metrics.wardrobing_count == 2
    assert metrics.high_value_skew is True
    assert FirstPartyAnomalyType.HIGH_RETURN_RATIO in metrics.anomalies
    assert FirstPartyAnomalyType.WARDROBING_PATTERN in metrics.anomalies
    assert FirstPartyAnomalyType.INR_CLAIM_BURST in metrics.anomalies
    assert FirstPartyAnomalyType.HIGH_VALUE_REFUND_SKEW in metrics.anomalies

    refund_only = analyzer.analyze([events[2]], now=now)
    assert refund_only.return_ratio == 1.0


def test_friendly_fraud_and_promo_abuse_analyzers() -> None:
    ff = FriendlyFraudAnalyzer()
    empty_ff = ff.analyze([])
    assert empty_ff.dispute_count == 0
    assert empty_ff.anomalies == ()

    now = datetime(2026, 11, 14, 12, 0, 0, tzinfo=UTC)
    disputes = [
        FirstPartyEventRecord(
            event_id="d-1",
            account_id="acct-ff",
            claim_type=OrderClaimType.FRIENDLY_DISPUTE,
            amount=320.0,
            timestamp=now - timedelta(days=3),
            three_ds_authenticated=True,
            delivery_confirmed=True,
        ),
        FirstPartyEventRecord(
            event_id="d-2",
            account_id="acct-ff",
            claim_type=OrderClaimType.FRIENDLY_DISPUTE,
            amount=410.0,
            timestamp=now - timedelta(days=1),
            avs_cvv_matched=True,
            signature_captured=True,
        ),
    ]
    ff_metrics = ff.analyze(disputes, now=now)
    assert ff_metrics.dispute_count == 2
    assert ff_metrics.verified_dispute_count == 2
    assert ff_metrics.recent_dispute_count == 2
    assert FirstPartyAnomalyType.SERIAL_FRIENDLY_FRAUD in ff_metrics.anomalies
    assert FirstPartyAnomalyType.PRE_DISPUTE_VELOCITY in ff_metrics.anomalies

    promo = PromoAbuseDetector()
    promo_events = [
        FirstPartyEventRecord(
            event_id=f"pr-{i}",
            account_id="acct-promo",
            claim_type=OrderClaimType.PROMO_REDEMPTION,
            amount=25.0,
            timestamp=now - timedelta(hours=i),
            promo_code=f"WELCOME{i}",
        )
        for i in range(3)
    ]
    promo_metrics = promo.analyze(promo_events, linked_accounts={"acct-a", "acct-b"})
    assert promo_metrics.promo_redemption_count == 3
    assert promo_metrics.unique_promo_codes == 3
    assert promo_metrics.linked_account_count == 2
    assert FirstPartyAnomalyType.PROMO_BONUS_ABUSE in promo_metrics.anomalies
    assert FirstPartyAnomalyType.LINKED_ACCOUNT_CYCLING in promo_metrics.anomalies


def test_store_eviction_linking_and_engine_case_escalation() -> None:
    store = FirstPartyAccountStore(
        max_accounts=2,
        max_assessments=2,
        max_events_per_account=20,
    )
    now = datetime(2026, 11, 14, 12, 0, 0, tzinfo=UTC)
    store.record_event(
        FirstPartyEventRecord(
            event_id="e-old",
            account_id="acct-old",
            claim_type=OrderClaimType.PURCHASE,
            amount=50.0,
            timestamp=now,
        )
    )
    store.record_event(
        FirstPartyEventRecord(
            event_id="e-a",
            account_id="acct-a",
            claim_type=OrderClaimType.PURCHASE,
            amount=50.0,
            timestamp=now,
            shipping_address_hash="ship-1",
            payment_instrument_hash="pay-1",
        )
    )
    # Third account evicts acct-old and links with acct-a
    store.record_event(
        FirstPartyEventRecord(
            event_id="e-b",
            account_id="acct-b",
            claim_type=OrderClaimType.PURCHASE,
            amount=50.0,
            timestamp=now,
            shipping_address_hash="ship-1",
            payment_instrument_hash="pay-1",
        )
    )
    assert store.get_profile("acct-old") is None
    profile_b = store.get_profile("acct-b")
    assert profile_b is not None
    assert "acct-a" in profile_b.linked_accounts

    case_mgr = CaseManager()
    engine = FirstPartyFraudEngine(store=store, case_manager=case_mgr)

    with pytest.raises(KeyError):
        engine.evaluate_account("missing-acct")

    # Drive acct-b to critical/block score with serial disputes + INR + wardrobing
    engine.record_event(
        FirstPartyEventRecord(
            event_id="e-b2",
            account_id="acct-b",
            claim_type=OrderClaimType.REFUND_RETURN,
            amount=250.0,
            timestamp=now,
            merchant_category="apparel",
        )
    )
    engine.record_event(
        FirstPartyEventRecord(
            event_id="e-b3",
            account_id="acct-b",
            claim_type=OrderClaimType.REFUND_RETURN,
            amount=250.0,
            timestamp=now,
            merchant_category="electronics",
        )
    )
    engine.record_event(
        FirstPartyEventRecord(
            event_id="e-b4",
            account_id="acct-b",
            claim_type=OrderClaimType.INR_CLAIM,
            amount=100.0,
            timestamp=now,
        )
    )
    engine.record_event(
        FirstPartyEventRecord(
            event_id="e-b5",
            account_id="acct-b",
            claim_type=OrderClaimType.INR_CLAIM,
            amount=100.0,
            timestamp=now,
        )
    )
    engine.record_event(
        FirstPartyEventRecord(
            event_id="e-b6",
            account_id="acct-b",
            claim_type=OrderClaimType.FRIENDLY_DISPUTE,
            amount=300.0,
            timestamp=now,
            three_ds_authenticated=True,
        )
    )
    assessment = engine.record_and_evaluate(
        FirstPartyEventRecord(
            event_id="e-b7",
            account_id="acct-b",
            claim_type=OrderClaimType.FRIENDLY_DISPUTE,
            amount=300.0,
            timestamp=now,
            delivery_confirmed=True,
        ),
        now=now,
    )
    assert assessment.risk_level == FirstPartyRiskLevel.CRITICAL
    assert assessment.action == FirstPartyAction.REVOKE_PRIVILEGES_AND_BLOCK
    assert assessment.case_id is not None
    assert len(case_mgr.store.list_cases()) == 1
    assert engine.get_assessment(assessment.assessment_id) == assessment
    assert len(engine.list_assessments(account_id="acct-b")) == 1

    # Export/load state round-trip
    exported = store.export_state()
    restored_store = FirstPartyAccountStore()
    restored_store.load_state(exported)
    assert restored_store.summary()["total_accounts"] == 2
    assert restored_store.get_assessment(assessment.assessment_id) is not None


def test_event_bus_publishing_and_policy_tiers() -> None:
    bus = EventBus()
    received: list[EventType] = []

    def _capture(event: FraudDomainEvent) -> None:
        received.append(event.event_type)

    bus.subscribe(_capture)
    engine = FirstPartyFraudEngine(event_bus=bus)
    now = datetime(2026, 11, 14, 12, 0, 0, tzinfo=UTC)

    # Low risk -> ALLOW
    low = engine.record_and_evaluate(
        FirstPartyEventRecord(
            event_id="ev-low",
            account_id="acct-low",
            claim_type=OrderClaimType.PURCHASE,
            amount=100.0,
            timestamp=now,
        ),
        now=now,
    )
    assert low.action == FirstPartyAction.ALLOW
    assert EventType.FIRST_PARTY_EVALUATED in received

    # Medium/High -> RESTRICT_INSTANT_REFUND
    received.clear()
    engine.record_event(
        FirstPartyEventRecord(
            event_id="ev-r1",
            account_id="acct-mid",
            claim_type=OrderClaimType.PURCHASE,
            amount=100.0,
            timestamp=now,
        )
    )
    engine.record_event(
        FirstPartyEventRecord(
            event_id="ev-r2",
            account_id="acct-mid",
            claim_type=OrderClaimType.REFUND_RETURN,
            amount=100.0,
            timestamp=now,
            merchant_category="apparel",
        )
    )
    mid = engine.record_and_evaluate(
        FirstPartyEventRecord(
            event_id="ev-r3",
            account_id="acct-mid",
            claim_type=OrderClaimType.REFUND_RETURN,
            amount=100.0,
            timestamp=now,
            merchant_category="jewelry",
        ),
        now=now,
    )
    assert mid.action == FirstPartyAction.RESTRICT_INSTANT_REFUND
    assert EventType.FIRST_PARTY_ABUSE_DETECTED in received
    assert EventType.REFUND_RESTRICTED in received

    # Critical -> REVOKE_PRIVILEGES_AND_BLOCK
    received.clear()
    for idx in range(3):
        engine.record_event(
            FirstPartyEventRecord(
                event_id=f"ev-c-{idx}",
                account_id="acct-crit",
                claim_type=OrderClaimType.FRIENDLY_DISPUTE,
                amount=300.0,
                timestamp=now,
                three_ds_authenticated=True,
            )
        )
    for idx in range(3):
        engine.record_event(
            FirstPartyEventRecord(
                event_id=f"ev-ci-{idx}",
                account_id="acct-crit",
                claim_type=OrderClaimType.INR_CLAIM,
                amount=200.0,
                timestamp=now,
            )
        )
    crit = engine.record_and_evaluate(
        FirstPartyEventRecord(
            event_id="ev-cp",
            account_id="acct-crit",
            claim_type=OrderClaimType.PROMO_REDEMPTION,
            amount=50.0,
            timestamp=now,
            promo_code="FREE50",
        ),
        now=now,
    )
    assert crit.action == FirstPartyAction.REVOKE_PRIVILEGES_AND_BLOCK
    assert EventType.ACCOUNT_PRIVILEGES_REVOKED in received
    assert len(engine.list_assessments(risk_level=FirstPartyRiskLevel.CRITICAL)) == 1
    assert engine.get_profile("acct-crit") is not None
    assert engine.summary()["total_accounts"] == 3


def test_edge_cases_and_remaining_policy_tiers() -> None:
    tiny_store = FirstPartyAccountStore(max_events_per_account=2, max_assessments=1)
    now = datetime(2026, 11, 14, 12, 0, 0, tzinfo=UTC)
    for idx in range(3):
        tiny_store.record_event(
            FirstPartyEventRecord(
                event_id=f"te-{idx}",
                account_id="acct-tiny",
                claim_type=OrderClaimType.PURCHASE,
                amount=50.0,
                timestamp=now,
            )
        )
    profile = tiny_store.get_profile("acct-tiny")
    assert profile is not None
    assert len(profile.events) == 2

    high_store = FirstPartyAccountStore(max_assessments=1)
    engine = FirstPartyFraudEngine(store=high_store)
    assert engine.get_assessment("missing") is None

    # Medium tier -> REQUIRE_SIGNATURE_DELIVERY (ratio 1.0 -> 0.25)
    engine.record_event(
        FirstPartyEventRecord(
            event_id="m-1",
            account_id="acct-med",
            claim_type=OrderClaimType.PURCHASE,
            amount=100.0,
            timestamp=now,
        )
    )
    med = engine.record_and_evaluate(
        FirstPartyEventRecord(
            event_id="m-2",
            account_id="acct-med",
            claim_type=OrderClaimType.REFUND_RETURN,
            amount=100.0,
            timestamp=now,
            merchant_category="books",
        ),
        now=now,
    )
    assert med.risk_level == FirstPartyRiskLevel.MEDIUM
    assert med.action == FirstPartyAction.REQUIRE_SIGNATURE_DELIVERY

    # High tier -> DENY_REFUND_MANUAL_REVIEW (0.65..0.84):
    # 3 linked accounts (0.20) + promo abuse (0.15) + serial friendly fraud (0.40) => 0.75
    for idx, peer in enumerate(["p1", "p2", "p3"]):
        high_store.record_event(
            FirstPartyEventRecord(
                event_id=f"lk-{idx}",
                account_id=peer,
                claim_type=OrderClaimType.PURCHASE,
                amount=20.0,
                timestamp=now,
                shipping_address_hash="shared-ship",
                payment_instrument_hash="shared-pay",
            )
        )
    engine.record_event(
        FirstPartyEventRecord(
            event_id="h-1",
            account_id="acct-high",
            claim_type=OrderClaimType.PROMO_REDEMPTION,
            amount=10.0,
            timestamp=now,
            shipping_address_hash="shared-ship",
            payment_instrument_hash="shared-pay",
        )
    )
    engine.record_event(
        FirstPartyEventRecord(
            event_id="h-2",
            account_id="acct-high",
            claim_type=OrderClaimType.PROMO_REDEMPTION,
            amount=10.0,
            timestamp=now,
        )
    )
    engine.record_event(
        FirstPartyEventRecord(
            event_id="h-3",
            account_id="acct-high",
            claim_type=OrderClaimType.FRIENDLY_DISPUTE,
            amount=100.0,
            timestamp=now,
            three_ds_authenticated=True,
        )
    )
    high = engine.record_and_evaluate(
        FirstPartyEventRecord(
            event_id="h-4",
            account_id="acct-high",
            claim_type=OrderClaimType.FRIENDLY_DISPUTE,
            amount=100.0,
            timestamp=now - timedelta(days=30),
            delivery_confirmed=True,
        ),
        now=now,
    )
    assert high.risk_level == FirstPartyRiskLevel.HIGH
    assert high.action == FirstPartyAction.DENY_REFUND_MANUAL_REVIEW
    # max_assessments=1 evicted med
    assert engine.get_assessment(med.assessment_id) is None
