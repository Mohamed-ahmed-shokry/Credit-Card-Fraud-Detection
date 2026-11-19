"""Unit tests for the Digital Wallet Provisioning & Network Token Fraud Defense Engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fraud_detection.cases import CaseManager
from fraud_detection.events import EventBus, EventType, FraudDomainEvent
from fraud_detection.wallet_fraud import (
    IdentityTelemetryAnalyzer,
    ProvisioningPath,
    ProvisioningVelocityAnalyzer,
    TokenLifecycleAnalyzer,
    TokenStatus,
    WalletAnomalyType,
    WalletFraudEngine,
    WalletProvider,
    WalletProvisioningRequest,
    WalletRiskAssessment,
    WalletRiskLevel,
    WalletTokenRecord,
    WalletTokenStore,
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


def test_wallet_store_eviction_filtering_and_persistence(tmp_path: Path) -> None:
    store = WalletTokenStore(max_tokens=2, max_assessments=2)
    for idx in range(3):
        tok = WalletTokenRecord(
            token_id=f"tok_{idx}",
            card_hash=f"card_{idx}",
            device_id=f"dev_{idx}",
            wallet_account_id=f"wal_{idx}",
            wallet_provider=WalletProvider.APPLE_PAY,
            tenant_id="default" if idx < 2 else "acme",
            status=TokenStatus.ACTIVE if idx < 2 else TokenStatus.SUSPENDED,
        )
        store.save_token(tok)
    store.save_token(tok)
    assert store.get_token("tok_0") is None
    assert len(store.list_tokens(card_hash="card_2")) == 1
    assert len(store.list_tokens(device_id="dev_2")) == 1
    assert len(store.list_tokens(tenant_id="acme")) == 1
    assert len(store.list_tokens(status=TokenStatus.SUSPENDED)) == 1

    for idx in range(3):
        ass = WalletRiskAssessment(
            assessment_id=f"wra_{idx}",
            request_id=f"wpr_{idx}",
            card_hash=f"card_{idx}",
            device_id=f"dev_{idx}",
            wallet_account_id=f"wal_{idx}",
            wallet_provider=WalletProvider.GOOGLE_PAY,
            tenant_id="default" if idx < 2 else "acme",
            risk_score=0.2 if idx < 2 else 0.9,
            risk_level=WalletRiskLevel.LOW if idx < 2 else WalletRiskLevel.CRITICAL,
            provisioning_path=(
                ProvisioningPath.GREEN_PATH if idx < 2 else ProvisioningPath.RED_PATH
            ),
        )
        store.save_assessment(ass)
    store.save_assessment(ass)
    assert store.get_assessment("wra_0") is None
    assert len(store.list_assessments(card_hash="card_2")) == 1
    assert len(store.list_assessments(device_id="dev_2")) == 1
    assert len(store.list_assessments(tenant_id="acme")) == 1
    assert len(store.list_assessments(path=ProvisioningPath.RED_PATH)) == 1

    summary = store.summary()
    assert summary["token_count"] == 2
    assert summary["assessment_count"] == 2

    json_path = store.save_json(tmp_path / "wallet_store.json")
    loaded = WalletTokenStore.load_json(json_path)
    assert loaded.summary()["token_count"] == 2
    assert WalletTokenStore.load_json(tmp_path / "missing.json").summary()["token_count"] == 0


def test_wallet_engine_paths_lifecycle_spend_burst_and_events() -> None:
    bus = EventBus()
    case_mgr = CaseManager()
    events: list[EventType] = []

    def _capture(ev: FraudDomainEvent) -> None:
        events.append(ev.event_type)

    bus.subscribe(_capture)
    engine = WalletFraudEngine(case_manager=case_mgr, event_bus=bus)

    # 1. GREEN_PATH provisioning -> ACTIVE token + WALLET_PROVISIONING_EVALUATED & ACTIVATED
    green = engine.evaluate_provisioning(
        WalletProvisioningRequest(
            card_hash="card_green",
            device_id="dev_green",
            wallet_account_id="wal_green",
        )
    )
    assert green.provisioning_path is ProvisioningPath.GREEN_PATH
    assert green.risk_level is WalletRiskLevel.LOW
    assert green.token_id is not None
    assert events == [EventType.WALLET_PROVISIONING_EVALUATED, EventType.WALLET_TOKEN_ACTIVATED]

    # Normal spend -> LOW, token stays ACTIVE
    normal_spend = engine.record_token_spend(green.token_id, 50.0)
    assert normal_spend.risk_level is WalletRiskLevel.LOW
    tok_green = engine.store.get_token(green.token_id)
    assert tok_green is not None and tok_green.status is TokenStatus.ACTIVE

    # Post-provisioning high-value burst -> auto-suspends token & escalates CaseManager case
    events.clear()
    burst_spend = engine.record_token_spend(green.token_id, 2000.0)
    assert burst_spend.risk_level is WalletRiskLevel.CRITICAL
    assert WalletAnomalyType.POST_PROVISIONING_HIGH_VALUE_BURST in burst_spend.anomalies
    assert burst_spend.correlated_case_id is not None
    assert EventType.WALLET_TOKEN_SUSPENDED in events
    tok_after_burst = engine.store.get_token(green.token_id)
    assert tok_after_burst is not None and tok_after_burst.status is TokenStatus.SUSPENDED

    # 2. YELLOW_PATH provisioning (phone mismatch + new account = 0.46) -> PENDING_IDV
    events.clear()
    yellow = engine.evaluate_provisioning(
        WalletProvisioningRequest(
            card_hash="card_yellow",
            device_id="dev_yellow",
            wallet_account_id="wal_yellow",
            phone_match=False,
            wallet_account_age_days=3,
        )
    )
    assert yellow.provisioning_path is ProvisioningPath.YELLOW_PATH
    assert yellow.risk_level is WalletRiskLevel.MEDIUM
    assert yellow.token_id is not None
    assert events == [EventType.WALLET_PROVISIONING_EVALUATED]

    # Complete ID&V verified -> ACTIVE
    activated = engine.complete_idv(yellow.token_id, verified=True)
    assert activated.status is TokenStatus.ACTIVE
    assert EventType.WALLET_TOKEN_ACTIVATED in events

    # Complete ID&V failed on another yellow token -> DEACTIVATED
    yellow2 = engine.evaluate_provisioning(
        WalletProvisioningRequest(
            card_hash="card_yellow2",
            device_id="dev_yellow2",
            wallet_account_id="wal_yellow2",
            phone_match=False,
            wallet_account_age_days=3,
        )
    )
    assert yellow2.token_id is not None
    failed_idv = engine.complete_idv(yellow2.token_id, verified=False)
    assert failed_idv.status is TokenStatus.DEACTIVATED

    # Manual deactivate_token
    deact = engine.deactivate_token(yellow.token_id, reason="card_replaced")
    assert deact.status is TokenStatus.DEACTIVATED

    # 3. RED_PATH provisioning (HIGH 0.70..0.84 and CRITICAL >= 0.85)
    events.clear()
    red_high = engine.evaluate_provisioning(
        WalletProvisioningRequest(
            card_hash="card_red_high",
            device_id="dev_red_high",
            wallet_account_id="wal_red_high",
            sim_swap_detected=True,
            otp_attempts=3,
        )
    )
    assert red_high.provisioning_path is ProvisioningPath.RED_PATH
    assert red_high.risk_level is WalletRiskLevel.HIGH
    assert red_high.token_id is None
    assert red_high.correlated_case_id is not None
    assert EventType.WALLET_PROVISIONING_DECLINED in events

    # Missing token errors
    with pytest.raises(KeyError, match="not found"):
        engine.complete_idv("tok_missing")
    with pytest.raises(KeyError, match="not found"):
        engine.suspend_token("tok_missing")
    with pytest.raises(KeyError, match="not found"):
        engine.deactivate_token("tok_missing")
    with pytest.raises(KeyError, match="not found"):
        engine.record_token_spend("tok_missing", 10.0)

    # Engine without CaseManager or EventBus
    bare = WalletFraudEngine()
    bare_red = bare.evaluate_provisioning(
        WalletProvisioningRequest(
            card_hash="card_bare",
            device_id="dev_bare",
            wallet_account_id="wal_bare",
            sim_swap_detected=True,
            otp_attempts=5,
            device_country="RU",
            issuer_country="US",
        )
    )
    assert bare_red.correlated_case_id is None
    assert bare_red.risk_level is WalletRiskLevel.CRITICAL
