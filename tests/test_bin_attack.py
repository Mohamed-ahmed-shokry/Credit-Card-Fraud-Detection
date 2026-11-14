"""Unit tests for the Phase 52 Card-Testing & BIN Enumeration Attack Defense Engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fraud_detection.bin_attack import (
    PROBING_RESPONSE_CODES,
    AuthResponseCode,
    BinAttackAction,
    BinAttackAnomalyType,
    BinAttackAssessment,
    BinAttackDefenseEngine,
    BinAttackRiskLevel,
    BinAttackStore,
    BinAuthAttempt,
    BinPrefixProfile,
    EnumerationAnalyzer,
    VelocityDeclineAnalyzer,
    _coerce_utc_datetime,
)
from fraud_detection.cases import CaseManager
from fraud_detection.events import EventBus, EventType


def test_bin_attack_enums_and_model_round_trip() -> None:
    assert BinAttackRiskLevel.CRITICAL.value == "critical"
    assert BinAttackAction.TEMP_BLOCK_BIN_PREFIX.value == "temp_block_bin_prefix"
    assert BinAttackAnomalyType.SEQUENTIAL_PAN_ENUMERATION.value == "sequential_pan_enumeration"
    assert AuthResponseCode.CVV_MISMATCH in PROBING_RESPONSE_CODES

    naive = datetime(2026, 11, 15, 0, 30, 0)  # noqa: DTZ001
    assert _coerce_utc_datetime(naive).tzinfo is UTC
    assert _coerce_utc_datetime("2026-11-15T00:30:00+00:00").tzinfo is UTC

    attempt = BinAuthAttempt(
        attempt_id="att-1",
        bin_prefix="411111",
        last4="0001",
        amount=1.0,
        response_code=AuthResponseCode.CVV_MISMATCH,
        timestamp=datetime(2026, 11, 15, 0, 30, 0, tzinfo=UTC),
        merchant_id="m-1",
        ip_address="203.0.113.10",
    )
    assert BinAuthAttempt.from_dict(attempt.to_dict()) == attempt

    with pytest.raises(ValueError, match="attempt_id"):
        BinAuthAttempt(
            attempt_id="",
            bin_prefix="411111",
            last4="0001",
            amount=1.0,
            response_code=AuthResponseCode.APPROVED,
            timestamp=datetime.now(UTC),
        )
    with pytest.raises(ValueError, match="bin_prefix"):
        BinAuthAttempt(
            attempt_id="att-2",
            bin_prefix="411",
            last4="0001",
            amount=1.0,
            response_code=AuthResponseCode.APPROVED,
            timestamp=datetime.now(UTC),
        )
    with pytest.raises(ValueError, match="last4"):
        BinAuthAttempt(
            attempt_id="att-3",
            bin_prefix="411111",
            last4="12",
            amount=1.0,
            response_code=AuthResponseCode.APPROVED,
            timestamp=datetime.now(UTC),
        )
    with pytest.raises(ValueError, match="amount"):
        BinAuthAttempt(
            attempt_id="att-4",
            bin_prefix="411111",
            last4="0001",
            amount=-1.0,
            response_code=AuthResponseCode.APPROVED,
            timestamp=datetime.now(UTC),
        )

    profile = BinPrefixProfile(
        bin_prefix="411111",
        attempts=[attempt],
        risk_level=BinAttackRiskLevel.HIGH,
        last_action=BinAttackAction.RATE_LIMIT_MERCHANT_IP,
        last_assessed_at=datetime(2026, 11, 15, 0, 31, 0, tzinfo=UTC),
        last_assessment_id="bin_1",
    )
    restored_profile = BinPrefixProfile.from_dict(profile.to_dict())
    assert restored_profile.bin_prefix == "411111"
    assert len(restored_profile.attempts) == 1

    assessment = BinAttackAssessment(
        assessment_id="bin_1",
        bin_prefix="411111",
        tenant_id="default",
        risk_score=0.72,
        risk_level=BinAttackRiskLevel.HIGH,
        action=BinAttackAction.TEMP_BLOCK_BIN_PREFIX,
        anomalies=(BinAttackAnomalyType.SEQUENTIAL_PAN_ENUMERATION,),
        total_attempts=5,
        decline_ratio=0.8,
        micro_auth_ratio=1.0,
        unique_ips=1,
        unique_cards=5,
        sequential_run_length=5,
        probing_count=4,
        reasons=("Sequential PAN enumeration detected",),
        evaluated_at=datetime(2026, 11, 15, 0, 31, 0, tzinfo=UTC),
    )
    assert BinAttackAssessment.from_dict(assessment.to_dict()) == assessment


def test_enumeration_analyzer_detects_sequential_pan_and_probing() -> None:
    analyzer = EnumerationAnalyzer()
    empty = analyzer.analyze([])
    assert empty.sequential_run_length == 0
    assert empty.anomalies == ()

    now = datetime(2026, 11, 15, 0, 30, 0, tzinfo=UTC)
    attempts = [
        BinAuthAttempt(
            attempt_id=f"seq-{i}",
            bin_prefix="411111",
            last4=f"{1000 + i:04d}",
            amount=1.0,
            response_code=AuthResponseCode.CVV_MISMATCH,
            timestamp=now - timedelta(seconds=10 * i),
        )
        for i in range(4)
    ]
    # Add a non-sequential outlier to cover run-reset branch
    attempts.append(
        BinAuthAttempt(
            attempt_id="seq-outlier",
            bin_prefix="411111",
            last4="9999",
            amount=1.0,
            response_code=AuthResponseCode.APPROVED,
            timestamp=now,
        )
    )
    metrics = analyzer.analyze(attempts, now=now)
    assert metrics.sequential_run_length == 4
    assert metrics.probing_count == 4
    assert metrics.unique_last4_count == 5
    assert BinAttackAnomalyType.SEQUENTIAL_PAN_ENUMERATION in metrics.anomalies
    assert BinAttackAnomalyType.CVV_EXPIRY_PROBING in metrics.anomalies


def test_velocity_decline_analyzer_detects_micro_auth_and_distributed_sweeps() -> None:
    analyzer = VelocityDeclineAnalyzer()
    assert analyzer.analyze([]).total_attempts == 0

    now = datetime(2026, 11, 15, 0, 30, 0, tzinfo=UTC)
    old_only = [
        BinAuthAttempt(
            attempt_id="old-1",
            bin_prefix="411111",
            last4="0001",
            amount=1.0,
            response_code=AuthResponseCode.APPROVED,
            timestamp=now - timedelta(hours=2),
        )
    ]
    assert analyzer.analyze(old_only, now=now).total_attempts == 0

    # Single IP probing 4 cards with $1 micro-auths and high declines
    single_ip_attempts = [
        BinAuthAttempt(
            attempt_id=f"m-{i}",
            bin_prefix="411111",
            last4=f"000{i}",
            amount=1.0,
            response_code=AuthResponseCode.DO_NOT_HONOR,
            timestamp=now - timedelta(seconds=5 * i),
            ip_address="198.51.100.1",
        )
        for i in range(4)
    ]
    m1 = analyzer.analyze(single_ip_attempts, now=now)
    assert m1.decline_ratio == 1.0
    assert m1.micro_auth_ratio == 1.0
    assert m1.max_cards_per_ip == 4
    assert BinAttackAnomalyType.MICRO_AUTH_VELOCITY in m1.anomalies
    assert BinAttackAnomalyType.HIGH_DECLINE_RATIO in m1.anomalies
    assert BinAttackAnomalyType.MULTI_CARD_IP_BURST in m1.anomalies

    # Distributed sweep across 4 distinct IPs and 4 distinct cards
    dist_attempts = [
        BinAuthAttempt(
            attempt_id=f"d-{i}",
            bin_prefix="520000",
            last4=f"100{i}",
            amount=2.0,
            response_code=AuthResponseCode.CVV_MISMATCH,
            timestamp=now - timedelta(seconds=5 * i),
            ip_address=f"198.51.100.{i + 1}",
        )
        for i in range(4)
    ]
    m2 = analyzer.analyze(dist_attempts, now=now)
    assert m2.unique_ips == 4
    assert BinAttackAnomalyType.DISTRIBUTED_BIN_SWEEP in m2.anomalies


def test_store_eviction_and_engine_case_escalation() -> None:
    store = BinAttackStore(max_bins=2, max_assessments=2, max_attempts_per_bin=10)
    now = datetime(2026, 11, 15, 0, 30, 0, tzinfo=UTC)

    store.record_attempt(
        BinAuthAttempt(
            attempt_id="old-b",
            bin_prefix="400000",
            last4="0001",
            amount=50.0,
            response_code=AuthResponseCode.APPROVED,
            timestamp=now,
        )
    )
    store.record_attempt(
        BinAuthAttempt(
            attempt_id="b1",
            bin_prefix="411111",
            last4="0001",
            amount=50.0,
            response_code=AuthResponseCode.APPROVED,
            timestamp=now,
        )
    )
    # Third BIN evicts 400000
    store.record_attempt(
        BinAuthAttempt(
            attempt_id="b2",
            bin_prefix="422222",
            last4="1000",
            amount=1.0,
            response_code=AuthResponseCode.CVV_MISMATCH,
            timestamp=now,
            ip_address="203.0.113.5",
        )
    )
    assert store.get_profile("400000") is None

    case_mgr = CaseManager()
    engine = BinAttackDefenseEngine(store=store, case_manager=case_mgr)
    with pytest.raises(KeyError):
        engine.evaluate_bin("999999")

    # Drive 422222 to CRITICAL (BLOCK_GATEWAY_SESSION) with sequential + probing + micro-auth
    for idx in range(1, 6):
        engine.record_attempt(
            BinAuthAttempt(
                attempt_id=f"b2-{idx}",
                bin_prefix="422222",
                last4=f"{1000 + idx:04d}",
                amount=1.0,
                response_code=AuthResponseCode.CVV_MISMATCH,
                timestamp=now + timedelta(seconds=idx),
                ip_address="203.0.113.5",
            )
        )
    assessment = engine.evaluate_bin("422222", now=now + timedelta(seconds=10))
    assert assessment.risk_level == BinAttackRiskLevel.CRITICAL
    assert assessment.action == BinAttackAction.BLOCK_GATEWAY_SESSION
    assert assessment.case_id is not None
    assert len(case_mgr.store.list_cases()) == 1
    assert engine.get_assessment(assessment.assessment_id) == assessment
    assert len(engine.list_assessments(bin_prefix="422222")) == 1

    exported = store.export_state()
    restored = BinAttackStore()
    restored.load_state(exported)
    assert restored.summary()["total_bins"] == 2
    assert restored.get_assessment(assessment.assessment_id) is not None


def test_event_bus_publishing_and_policy_tiers() -> None:
    bus = EventBus()
    received: list[EventType] = []
    bus.subscribe(lambda event: received.append(event.event_type))

    tiny_store = BinAttackStore(max_attempts_per_bin=3, max_assessments=1)
    now = datetime(2026, 11, 15, 0, 30, 0, tzinfo=UTC)
    for idx in range(4):
        tiny_store.record_attempt(
            BinAuthAttempt(
                attempt_id=f"tr-{idx}",
                bin_prefix="499999",
                last4=f"000{idx}",
                amount=10.0,
                response_code=AuthResponseCode.APPROVED,
                timestamp=now,
            )
        )
    prof_tr = tiny_store.get_profile("499999")
    assert prof_tr is not None
    assert len(prof_tr.attempts) == 3

    store = BinAttackStore(max_assessments=20)
    engine = BinAttackDefenseEngine(store=store, event_bus=bus)
    assert engine.get_assessment("missing") is None

    # LOW -> ALLOW
    low = engine.record_and_evaluate(
        BinAuthAttempt(
            attempt_id="low-1",
            bin_prefix="400001",
            last4="0001",
            amount=50.0,
            response_code=AuthResponseCode.APPROVED,
            timestamp=now,
        ),
        now=now,
    )
    assert low.action == BinAttackAction.ALLOW
    assert EventType.BIN_ATTACK_EVALUATED in received

    # MEDIUM -> CHALLENGE_3DS (sequential run of 4 -> 0.30)
    received.clear()
    for idx in range(3):
        engine.record_attempt(
            BinAuthAttempt(
                attempt_id=f"med-{idx}",
                bin_prefix="400002",
                last4=f"200{idx}",
                amount=20.0,
                response_code=AuthResponseCode.APPROVED,
                timestamp=now,
            )
        )
    med = engine.record_and_evaluate(
        BinAuthAttempt(
            attempt_id="med-3",
            bin_prefix="400002",
            last4="2003",
            amount=20.0,
            response_code=AuthResponseCode.APPROVED,
            timestamp=now,
        ),
        now=now,
    )
    assert med.risk_level == BinAttackRiskLevel.MEDIUM
    assert med.action == BinAttackAction.CHALLENGE_3DS
    assert EventType.BIN_ENUMERATION_DETECTED in received

    # HIGH -> RATE_LIMIT_MERCHANT_IP (0.45..0.64): sequential (0.30) + cvv probing (0.20) = 0.50
    for idx in range(4):
        engine.record_attempt(
            BinAuthAttempt(
                attempt_id=f"rl-{idx}",
                bin_prefix="400003",
                last4=f"300{idx}",
                amount=20.0,
                response_code=AuthResponseCode.CVV_MISMATCH,
                timestamp=now,
                ip_address="10.0.0.1" if idx < 3 else "10.0.0.2",
            )
        )
    # Add 2 approved to keep decline ratio < 0.70 (4/6 = 0.667)
    for idx in range(2):
        engine.record_attempt(
            BinAuthAttempt(
                attempt_id=f"rl-ok-{idx}",
                bin_prefix="400003",
                last4=f"900{idx}",
                amount=20.0,
                response_code=AuthResponseCode.APPROVED,
                timestamp=now,
            )
        )
    rl = engine.evaluate_bin("400003", now=now)
    assert rl.action == BinAttackAction.RATE_LIMIT_MERCHANT_IP

    # HIGH -> TEMP_BLOCK_BIN_PREFIX (0.65..0.84):
    # sequential (0.30) + probing (0.20) + high decline (0.15) + 0.05 = 0.70
    received.clear()
    for idx in range(4):
        engine.record_attempt(
            BinAuthAttempt(
                attempt_id=f"tb-{idx}",
                bin_prefix="400004",
                last4=f"400{idx}",
                amount=20.0,
                response_code=AuthResponseCode.CVV_MISMATCH,
                timestamp=now,
                ip_address="10.0.1.1" if idx < 3 else "10.0.1.2",
            )
        )
    tb = engine.evaluate_bin("400004", now=now)
    assert tb.action == BinAttackAction.TEMP_BLOCK_BIN_PREFIX
    assert EventType.CARD_TESTING_BLOCKED in received
    assert len(engine.list_assessments(risk_level=BinAttackRiskLevel.HIGH)) >= 2
    assert engine.get_profile("400004") is not None
    assert engine.summary()["total_bins"] == 4
