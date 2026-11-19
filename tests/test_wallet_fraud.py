"""Unit tests for the Digital Wallet Provisioning & Network Token Fraud Defense Engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fraud_detection.wallet_fraud import (
    IdentityTelemetryAnalyzer,
    ProvisioningPath,
    ProvisioningVelocityAnalyzer,
    TokenLifecycleAnalyzer,
    TokenStatus,
    WalletAnomalyType,
    WalletProvider,
    WalletProvisioningRequest,
    WalletRiskAssessment,
    WalletRiskLevel,
    WalletTokenRecord,
)


def test_wallet_domain_models_round_trip_and_validation() -> None:
    req = WalletProvisioningRequest(
        card_hash=" card_abc ",
        device_id=" dev_1 ",
        wallet_account_id=" wal_1 ",
        wallet_provider="google_pay",
        tenant_id="",
        wallet_account_age_days=-5,
        device_country="gb",
        issuer_country="us",
        phone_match=False,
        sim_swap_detected=True,
        otp_attempts=-2,
    )
    assert req.card_hash == "card_abc"
    assert req.device_id == "dev_1"
    assert req.wallet_account_id == "wal_1"
    assert req.wallet_provider is WalletProvider.GOOGLE_PAY
    assert req.tenant_id == "default"
    assert req.wallet_account_age_days == 0
    assert req.otp_attempts == 0
    assert req.device_country == "GB"
    assert req.issuer_country == "US"

    restored_req = WalletProvisioningRequest.from_dict(req.to_dict())
    assert restored_req.wallet_provider is WalletProvider.GOOGLE_PAY
    assert restored_req.sim_swap_detected is True

    with pytest.raises(ValueError, match="must not be empty"):
        WalletProvisioningRequest(card_hash=" ", device_id="dev", wallet_account_id="wal")

    tok = WalletTokenRecord(
        card_hash="card_abc",
        device_id="dev_1",
        wallet_account_id="wal_1",
        wallet_provider="apple_pay",
        status="pending_idv",
        total_spend_amount=125.456,
        transaction_count=2,
    )
    assert tok.wallet_provider is WalletProvider.APPLE_PAY
    assert tok.status is TokenStatus.PENDING_IDV
    restored_tok = WalletTokenRecord.from_dict(tok.to_dict())
    assert restored_tok.total_spend_amount == 125.46
    assert restored_tok.status is TokenStatus.PENDING_IDV

    with pytest.raises(ValueError, match="must not be empty"):
        WalletTokenRecord(
            card_hash="card_abc",
            device_id=" ",
            wallet_account_id="wal_1",
            wallet_provider=WalletProvider.APPLE_PAY,
        )

    assessment = WalletRiskAssessment(
        assessment_id="wra_1",
        request_id=req.request_id,
        card_hash="card_abc",
        device_id="dev_1",
        wallet_account_id="wal_1",
        wallet_provider="samsung_pay",
        tenant_id="acme",
        risk_score=1.45,
        risk_level="critical",
        provisioning_path="red_path",
        anomalies=[WalletAnomalyType.PHONE_MISMATCH_SIM_SWAP],
        rationale=["SIM swap detected"],
    )
    assert assessment.risk_score == 1.0
    assert assessment.risk_level is WalletRiskLevel.CRITICAL
    assert assessment.provisioning_path is ProvisioningPath.RED_PATH
    restored_ass = WalletRiskAssessment.from_dict(assessment.to_dict())
    assert restored_ass.anomalies == [WalletAnomalyType.PHONE_MISMATCH_SIM_SWAP]


def test_wallet_analyzers_detect_all_eight_anomalies() -> None:
    vel = ProvisioningVelocityAnalyzer(max_cards_per_device=2, max_wallets_per_card=2)
    existing = [
        WalletTokenRecord(
            card_hash="c1",
            device_id="dev_bad",
            wallet_account_id="w1",
            wallet_provider=WalletProvider.APPLE_PAY,
            status=TokenStatus.SUSPENDED,
        ),
        WalletTokenRecord(
            card_hash="c2",
            device_id="dev_bad",
            wallet_account_id="w2",
            wallet_provider=WalletProvider.GOOGLE_PAY,
        ),
        WalletTokenRecord(
            card_hash="c3",
            device_id="dev_other",
            wallet_account_id="w3",
            wallet_provider=WalletProvider.SAMSUNG_PAY,
        ),
        WalletTokenRecord(
            card_hash="c3",
            device_id="dev_other_2",
            wallet_account_id="w4",
            wallet_provider=WalletProvider.APPLE_PAY,
        ),
    ]
    req = WalletProvisioningRequest(
        card_hash="c3",
        device_id="dev_bad",
        wallet_account_id="w5",
        wallet_account_age_days=2,
        device_country="RU",
        issuer_country="US",
        phone_match=False,
        sim_swap_detected=True,
        otp_attempts=4,
    )
    v_score, v_anoms, v_rat = vel.analyze(req, existing)
    assert v_score > 0.8
    assert set(v_anoms) == {
        WalletAnomalyType.DEVICE_PROVISIONING_VELOCITY,
        WalletAnomalyType.COMPROMISED_DEVICE_REUSE,
        WalletAnomalyType.CARD_WALLET_FAN_OUT,
    }
    assert len(v_rat) == 3

    ident = IdentityTelemetryAnalyzer(min_account_age_days=14, otp_burst_threshold=3)
    i_score, i_anoms, i_rat = ident.analyze(req)
    assert i_score == 1.0
    assert set(i_anoms) == {
        WalletAnomalyType.PHONE_MISMATCH_SIM_SWAP,
        WalletAnomalyType.OTP_INTERCEPTION_BURST,
        WalletAnomalyType.GEO_ISSUER_MISMATCH,
        WalletAnomalyType.LOW_WALLET_ACCOUNT_TENURE,
    }
    assert len(i_rat) == 4

    # Clean request triggers zero identity anomalies
    clean_req = WalletProvisioningRequest(
        card_hash="c_ok",
        device_id="dev_ok",
        wallet_account_id="w_ok",
    )
    c_score, c_anoms, _ = ident.analyze(clean_req)
    assert c_score == 0.0
    assert c_anoms == []

    # Phone mismatch without SIM swap
    phone_only = WalletProvisioningRequest(
        card_hash="c_p", device_id="dev_p", wallet_account_id="w_p", phone_match=False
    )
    p_score, p_anoms, _ = ident.analyze(phone_only)
    assert p_score > 0.0
    assert p_anoms == [WalletAnomalyType.PHONE_MISMATCH_SIM_SWAP]

    # TokenLifecycleAnalyzer
    now = datetime.now(UTC)
    lifecycle = TokenLifecycleAnalyzer(burst_window_seconds=3600, high_value_threshold=1000.0)
    fresh_token = WalletTokenRecord(
        card_hash="c_ok",
        device_id="dev_ok",
        wallet_account_id="w_ok",
        wallet_provider=WalletProvider.APPLE_PAY,
        provisioned_at=(now - timedelta(minutes=10)).isoformat(),
        activated_at="2026-11-19T20:00:00",  # naive timestamp
    )
    # Normal low spend -> 0.0
    low_s, low_a, _ = lifecycle.analyze_spend(
        fresh_token, 100.0, reference_time=datetime(2026, 11, 19, 20, 5, tzinfo=UTC)
    )
    assert low_s == 0.0 and low_a == []

    # High-value spend within 5 minutes -> POST_PROVISIONING_HIGH_VALUE_BURST
    high_s, high_a, high_r = lifecycle.analyze_spend(
        fresh_token, 1500.0, reference_time=datetime(2026, 11, 19, 20, 5, tzinfo=UTC)
    )
    assert high_s == 0.85
    assert high_a == [WalletAnomalyType.POST_PROVISIONING_HIGH_VALUE_BURST]
    assert len(high_r) == 1
