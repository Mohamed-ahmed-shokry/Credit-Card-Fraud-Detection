"""Unit tests for Phase 57 APP Scam & Social Engineering Defense Engine."""

from __future__ import annotations

import pytest

from fraud_detection.app_scam import (
    AppPaymentContext,
    AppScamAssessment,
    BehavioralCoachingAnalyzer,
    CoolingOffHoldStatus,
    PayeeBeneficiaryAnalyzer,
    ScamCoolingOffHold,
    ScamInterventionAction,
    ScamRiskLevel,
    ScamSignal,
    ScamSignalType,
    ScamTypology,
    TransferEscalationAnalyzer,
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


def test_behavioral_coaching_analyzer_detects_all_branches() -> None:
    analyzer = BehavioralCoachingAnalyzer()

    # Clean low-risk payment
    clean = AppPaymentContext(
        sender_account_id="s1",
        payee_account_id="p1",
        amount=120.0,
        is_new_payee=False,
        sender_age_years=40,
    )
    assert analyzer.analyze(clean) == []

    # Active call only + extreme hesitation without copy-paste
    call_only = AppPaymentContext(
        sender_account_id="s1",
        payee_account_id="p1",
        amount=200.0,
        active_phone_call=True,
        session_hesitation_seconds=135.0,
    )
    sigs = analyzer.analyze(call_only)
    types = {s.signal_type for s in sigs}
    assert ScamSignalType.ACTIVE_CALL_OR_SCREEN_SHARE in types
    assert ScamSignalType.BEHAVIORAL_HESITATION_OR_COPY_PASTE in types

    # RAT only
    rat_only = AppPaymentContext(
        sender_account_id="s1",
        payee_account_id="p1",
        amount=200.0,
        screen_share_or_rat_detected=True,
    )
    rat_sigs = analyzer.analyze(rat_only)
    assert rat_sigs[0].severity == 0.85

    # Combined call + RAT + urgency + safe_account + copy-paste + elder
    severe = AppPaymentContext(
        sender_account_id="s1",
        payee_account_id="p1",
        amount=5000.0,
        active_phone_call=True,
        screen_share_or_rat_detected=True,
        urgency_keywords_detected=True,
        stated_purpose="safe_account",
        payee_details_pasted=True,
        session_hesitation_seconds=60.0,
        sender_age_years=72,
    )
    severe_sigs = analyzer.analyze(severe)
    assert {s.signal_type for s in severe_sigs} == {
        ScamSignalType.ACTIVE_CALL_OR_SCREEN_SHARE,
        ScamSignalType.URGENCY_OR_SCRIPTED_COACHING,
        ScamSignalType.BEHAVIORAL_HESITATION_OR_COPY_PASTE,
        ScamSignalType.ELDER_OR_VULNERABLE_ANOMALY,
    }


def test_payee_beneficiary_analyzer_detects_all_branches() -> None:
    analyzer = PayeeBeneficiaryAnalyzer()

    clean = AppPaymentContext(
        sender_account_id="s1",
        payee_account_id="p1",
        amount=300.0,
        is_new_payee=False,
        cop_name_match="match",
        payment_rail="faster_payments",
    )
    assert analyzer.analyze(clean) == []

    risky = AppPaymentContext(
        sender_account_id="s1",
        payee_account_id="p2",
        amount=6500.0,
        is_new_payee=True,
        cop_name_match="no_match",
        payment_rail="crypto_exchange",
        stated_purpose="bitcoin_investment",
    )
    sigs = analyzer.analyze(risky)
    assert {s.signal_type for s in sigs} == {
        ScamSignalType.NEW_PAYEE_HIGH_VALUE,
        ScamSignalType.BENEFICIARY_NAME_MISMATCH,
        ScamSignalType.CRYPTO_OR_HIGH_RISK_RAIL,
    }
    by_type = {s.signal_type: s for s in sigs}
    assert by_type[ScamSignalType.NEW_PAYEE_HIGH_VALUE].severity == 0.75
    assert by_type[ScamSignalType.BENEFICIARY_NAME_MISMATCH].severity == 0.8
    assert by_type[ScamSignalType.CRYPTO_OR_HIGH_RISK_RAIL].severity == 0.75

    moderate = AppPaymentContext(
        sender_account_id="s1",
        payee_account_id="p3",
        amount=2000.0,
        is_new_payee=True,
        cop_name_match="close_match",
        payment_rail="faster_payments",
        stated_purpose="crypto",
    )
    mod_by_type = {s.signal_type: s for s in analyzer.analyze(moderate)}
    assert mod_by_type[ScamSignalType.NEW_PAYEE_HIGH_VALUE].severity == 0.5
    assert mod_by_type[ScamSignalType.BENEFICIARY_NAME_MISMATCH].severity == 0.45
    assert mod_by_type[ScamSignalType.CRYPTO_OR_HIGH_RISK_RAIL].severity == 0.55


def test_transfer_escalation_analyzer_detects_all_branches() -> None:
    analyzer = TransferEscalationAnalyzer(window_minutes=60, escalation_multiplier=3.0)
    t0 = "2026-11-25T10:00:00+00:00"
    t1 = "2026-11-25T10:15:00+00:00"
    old_t = "2026-11-25T07:00:00+00:00"

    prior_small = AppPaymentContext(
        payment_id="app_prior",
        sender_account_id="s1",
        payee_account_id="p1",
        amount=100.0,
        initiated_at=t0,
    )
    stale = AppPaymentContext(
        payment_id="app_stale",
        sender_account_id="s1",
        payee_account_id="p1",
        amount=50.0,
        initiated_at=old_t,
    )

    # No escalation, no drain
    normal = AppPaymentContext(
        payment_id="app_norm",
        sender_account_id="s1",
        payee_account_id="p1",
        amount=150.0,
        sender_balance_before=5000.0,
        initiated_at=t1,
    )
    assert analyzer.analyze(normal, [normal, prior_small, stale]) == []

    # Escalation only (500 >= 100 * 3)
    escalated_only = AppPaymentContext(
        payment_id="app_esc",
        sender_account_id="s1",
        payee_account_id="p1",
        amount=500.0,
        sender_balance_before=5000.0,
        initiated_at=t1,
    )
    esc_sigs = analyzer.analyze(escalated_only, [prior_small])
    assert len(esc_sigs) == 1
    assert esc_sigs[0].severity == 0.75

    # Balance drain only (new payee, 90% of balance)
    drain_only = AppPaymentContext(
        payment_id="app_drain",
        sender_account_id="s1",
        payee_account_id="p_new",
        amount=4500.0,
        is_new_payee=True,
        sender_balance_before=5000.0,
        initiated_at=t1,
    )
    drain_sigs = analyzer.analyze(drain_only, [])
    assert len(drain_sigs) == 1
    assert drain_sigs[0].severity == 0.7

    # Both escalation and balance drain
    both = AppPaymentContext(
        payment_id="app_both",
        sender_account_id="s1",
        payee_account_id="p1",
        amount=4800.0,
        sender_balance_before=5000.0,
        initiated_at=t1,
    )
    both_sigs = analyzer.analyze(both, [prior_small])
    assert len(both_sigs) == 1
    assert both_sigs[0].severity == 0.9
