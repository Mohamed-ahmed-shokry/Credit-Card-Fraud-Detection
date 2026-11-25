"""Unit tests for Phase 57 APP Scam & Social Engineering Defense Engine."""

from __future__ import annotations

import pytest

from fraud_detection.app_scam import (
    AppPaymentContext,
    AppScamAssessment,
    CoolingOffHoldStatus,
    ScamCoolingOffHold,
    ScamInterventionAction,
    ScamRiskLevel,
    ScamSignal,
    ScamSignalType,
    ScamTypology,
    _clamp,
    _parse_iso,
)


def test_app_scam_helpers_and_enums() -> None:
    assert _clamp(-0.5) == 0.0
    assert _clamp(1.5) == 1.0
    assert _clamp(0.42) == 0.42
    naive = _parse_iso("2026-11-25T10:00:00")
    assert naive.tzinfo is not None
    aware = _parse_iso("2026-11-25T10:00:00+00:00")
    assert aware == naive
    assert ScamTypology.IMPERSONATION_BANK == "impersonation_bank"
    assert ScamRiskLevel.CRITICAL == "critical"
    assert ScamInterventionAction.COOLING_OFF_HOLD == "cooling_off_hold"
    assert ScamSignalType.ACTIVE_CALL_OR_SCREEN_SHARE == "active_call_or_screen_share"
    assert CoolingOffHoldStatus.ACTIVE == "active"


def test_app_payment_context_validation_and_roundtrip() -> None:
    ctx = AppPaymentContext(
        sender_account_id=" acct_sender_1 ",
        payee_account_id=" acct_payee_9 ",
        amount=4500.0,
        currency=" gbp ",
        payment_rail=" Faster_Payments ",
        tenant_id=" ",
        cop_name_match=" NO_MATCH ",
        active_phone_call=True,
        session_hesitation_seconds=-5.0,
        stated_purpose=" ",
        sender_age_years=74,
        sender_balance_before=5000.0,
        metadata={"channel": "mobile"},
    )
    assert ctx.sender_account_id == "acct_sender_1"
    assert ctx.payee_account_id == "acct_payee_9"
    assert ctx.currency == "GBP"
    assert ctx.payment_rail == "faster_payments"
    assert ctx.tenant_id == "default"
    assert ctx.cop_name_match == "no_match"
    assert ctx.session_hesitation_seconds == 0.0
    assert ctx.stated_purpose == "general"

    restored = AppPaymentContext.from_dict(ctx.to_dict())
    assert restored == ctx

    with pytest.raises(ValueError, match="must not be empty"):
        AppPaymentContext(sender_account_id=" ", payee_account_id="p1", amount=100.0)
    with pytest.raises(ValueError, match="greater than zero"):
        AppPaymentContext(sender_account_id="s1", payee_account_id="p1", amount=0.0)


def test_scam_signal_hold_and_assessment_roundtrip() -> None:
    sig = ScamSignal(
        signal_type="active_call_or_screen_share",
        severity=1.4,
        detail="Active voice call during transfer",
        evidence={"active_phone_call": True},
    )
    assert sig.signal_type is ScamSignalType.ACTIVE_CALL_OR_SCREEN_SHARE
    assert sig.severity == 1.0
    assert ScamSignal.from_dict(sig.to_dict()) == sig

    hold = ScamCoolingOffHold(
        payment_id="app_123",
        sender_account_id="s1",
        payee_account_id="p1",
        amount=3000.0,
        typology="impersonation_bank",
        status="active",
        hold_duration_minutes=0,
        reason="Coached transfer",
    )
    assert hold.typology is ScamTypology.IMPERSONATION_BANK
    assert hold.status is CoolingOffHoldStatus.ACTIVE
    assert hold.hold_duration_minutes == 1
    assert ScamCoolingOffHold.from_dict(hold.to_dict()) == hold

    assessment = AppScamAssessment(
        payment_id="app_123",
        sender_account_id="s1",
        payee_account_id="p1",
        amount=3000.0,
        risk_score=0.82,
        risk_level="high",
        typology="impersonation_bank",
        action="cooling_off_hold",
        signals=[sig],
        customer_warning="Pause and verify.",
        hold_id=hold.hold_id,
        case_id="case_1",
    )
    assert assessment.risk_level is ScamRiskLevel.HIGH
    assert assessment.typology is ScamTypology.IMPERSONATION_BANK
    assert assessment.action is ScamInterventionAction.COOLING_OFF_HOLD
    assert AppScamAssessment.from_dict(assessment.to_dict()) == assessment
