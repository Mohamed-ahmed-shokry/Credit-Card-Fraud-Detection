"""Unit tests for the Digital Wallet Provisioning & Network Token Fraud Defense Engine."""

from __future__ import annotations

import pytest

from fraud_detection.wallet_fraud import (
    ProvisioningPath,
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
