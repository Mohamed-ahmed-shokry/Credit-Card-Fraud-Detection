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
    BinAttackRiskLevel,
    BinAuthAttempt,
    BinPrefixProfile,
    EnumerationAnalyzer,
    _coerce_utc_datetime,
)


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
