"""Unit tests for the Phase 51 First-Party Fraud & Refund Abuse Detection Engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fraud_detection.first_party_fraud import (
    WARDROBING_CATEGORIES,
    ClaimOutcome,
    FirstPartyAccountProfile,
    FirstPartyAction,
    FirstPartyAnomalyType,
    FirstPartyAssessment,
    FirstPartyEventRecord,
    FirstPartyRiskLevel,
    OrderClaimType,
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
