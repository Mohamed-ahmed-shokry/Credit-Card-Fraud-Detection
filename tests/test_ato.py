"""Unit tests for Account Takeover (ATO) domain models and enums."""

from datetime import UTC, datetime

import pytest

from fraud_detection.ato import (
    AccountMutationRecord,
    AccountMutationType,
    AtoAction,
    AtoRiskAssessment,
    AtoRiskFactor,
    LoginEvent,
    LoginOutcome,
)


def test_ato_enums() -> None:
    """Verify all enum values match expected identifiers."""
    assert AtoRiskFactor.NEW_DEVICE.value == "new_device"
    assert AtoRiskFactor.IMPOSSIBLE_TRAVEL.value == "impossible_travel"
    assert AtoRiskFactor.RECENT_CREDENTIAL_CHANGE.value == "recent_credential_change"
    assert AtoRiskFactor.RECENT_MFA_RESET.value == "recent_mfa_reset"
    assert AtoRiskFactor.CREDENTIAL_STUFFING_BURST.value == "credential_stuffing_burst"
    assert AtoRiskFactor.PASSWORD_SPRAYING.value == "password_spraying"
    assert AtoRiskFactor.DORMANT_ACCOUNT_REACTIVE.value == "dormant_account_reactive"
    assert AtoRiskFactor.VELOCITY_SPIKE.value == "velocity_spike"
    assert AtoRiskFactor.ACCOUNT_QUARANTINED.value == "account_quarantined"

    assert AtoAction.ALLOW.value == "allow"
    assert AtoAction.CHALLENGE.value == "challenge"
    assert AtoAction.QUARANTINE_ACCOUNT.value == "quarantine_account"
    assert AtoAction.BLOCK.value == "block"

    assert LoginOutcome.SUCCESS.value == "success"
    assert LoginOutcome.FAILED_CREDENTIALS.value == "failed_credentials"
    assert LoginOutcome.FAILED_MFA.value == "failed_mfa"
    assert LoginOutcome.BLOCKED.value == "blocked"

    assert AccountMutationType.PASSWORD_CHANGE.value == "password_change"
    assert AccountMutationType.EMAIL_CHANGE.value == "email_change"
    assert AccountMutationType.PHONE_CHANGE.value == "phone_change"
    assert AccountMutationType.MFA_RESET.value == "mfa_reset"
    assert AccountMutationType.SHIPPING_ADDRESS_CHANGE.value == "shipping_address_change"


def test_account_mutation_record_serialization() -> None:
    """Test AccountMutationRecord to_dict and from_dict roundtrip."""
    now = datetime(2026, 11, 8, 12, 0, 0, tzinfo=UTC)
    record = AccountMutationRecord(
        mutation_type=AccountMutationType.PASSWORD_CHANGE,
        timestamp=now,
        channel="mobile_app",
        metadata={"ip": "1.2.3.4", "device": "ios_v12"},
    )

    data = record.to_dict()
    assert data["mutation_id"].startswith("mut_")
    assert data["mutation_type"] == "password_change"
    assert data["timestamp"] == now.isoformat()
    assert data["channel"] == "mobile_app"
    assert data["metadata"]["ip"] == "1.2.3.4"

    rebuilt = AccountMutationRecord.from_dict(data)
    assert rebuilt.mutation_id == record.mutation_id
    assert rebuilt.mutation_type == AccountMutationType.PASSWORD_CHANGE
    assert rebuilt.timestamp == now
    assert rebuilt.channel == "mobile_app"
    assert rebuilt.metadata == record.metadata

    # Test defaults
    naive_data = {"mutation_type": "email_change", "timestamp": "2026-11-08T10:00:00"}
    naive_rebuilt = AccountMutationRecord.from_dict(naive_data)
    assert naive_rebuilt.timestamp.tzinfo == UTC
    assert naive_rebuilt.channel == "web"

    # Test fallback timestamp when missing
    missing_ts = AccountMutationRecord.from_dict({"mutation_type": "mfa_reset"})
    assert isinstance(missing_ts.timestamp, datetime)
    assert missing_ts.timestamp.tzinfo == UTC


def test_login_event_serialization() -> None:
    """Test LoginEvent creation and roundtrip serialization."""
    now = datetime(2026, 11, 8, 12, 30, 0, tzinfo=UTC)
    event = LoginEvent(
        user_id="usr_12345",
        tenant_id="tenant_a",
        timestamp=now,
        ip_address="198.51.100.42",
        outcome=LoginOutcome.FAILED_CREDENTIALS,
        device_fingerprint="fp_abc123",
        latitude=40.7128,
        longitude=-74.0060,
        user_agent="Mozilla/5.0",
    )

    data = event.to_dict()
    assert data["event_id"].startswith("lgn_")
    assert data["user_id"] == "usr_12345"
    assert data["outcome"] == "failed_credentials"
    assert data["latitude"] == 40.7128

    rebuilt = LoginEvent.from_dict(data)
    assert rebuilt.event_id == event.event_id
    assert rebuilt.user_id == "usr_12345"
    assert rebuilt.outcome == LoginOutcome.FAILED_CREDENTIALS
    assert rebuilt.latitude == 40.7128
    assert rebuilt.longitude == -74.0060

    # Test defaults
    minimal_data = {"user_id": "usr_999"}
    minimal_rebuilt = LoginEvent.from_dict(minimal_data)
    assert minimal_rebuilt.tenant_id == "default"
    assert minimal_rebuilt.outcome == LoginOutcome.SUCCESS
    assert minimal_rebuilt.ip_address == "127.0.0.1"


def test_ato_risk_assessment_validation_and_serialization() -> None:
    """Test AtoRiskAssessment invariant validation and roundtrip."""
    with pytest.raises(ValueError, match="risk_score must be in range"):
        AtoRiskAssessment(
            user_id="usr_test",
            risk_score=1.5,
            recommended_action=AtoAction.ALLOW,
        )

    with pytest.raises(ValueError, match="risk_score must be in range"):
        AtoRiskAssessment(
            user_id="usr_test",
            risk_score=-0.1,
            recommended_action=AtoAction.ALLOW,
        )

    now = datetime.now(UTC)
    assessment = AtoRiskAssessment(
        user_id="usr_victim",
        tenant_id="corp_1",
        risk_score=0.85,
        recommended_action=AtoAction.QUARANTINE_ACCOUNT,
        factors_triggered=(
            AtoRiskFactor.IMPOSSIBLE_TRAVEL,
            AtoRiskFactor.RECENT_CREDENTIAL_CHANGE,
        ),
        score_breakdown={"travel": 0.45, "mutation": 0.40},
        impossible_travel_kmh=1250.5,
        is_new_device=True,
        hours_since_last_mutation=0.75,
        quarantined=True,
        timestamp=now,
        details={"case_id": "case_101"},
    )

    data = assessment.to_dict()
    assert data["assessment_id"].startswith("ato_")
    assert data["risk_score"] == 0.85
    assert data["recommended_action"] == "quarantine_account"
    assert "impossible_travel" in data["factors_triggered"]
    assert data["impossible_travel_kmh"] == 1250.5
    assert data["is_new_device"] is True
    assert data["quarantined"] is True

    rebuilt = AtoRiskAssessment.from_dict(data)
    assert rebuilt.assessment_id == assessment.assessment_id
    assert rebuilt.user_id == "usr_victim"
    assert rebuilt.risk_score == 0.85
    assert rebuilt.recommended_action == AtoAction.QUARANTINE_ACCOUNT
    assert AtoRiskFactor.IMPOSSIBLE_TRAVEL in rebuilt.factors_triggered
    assert rebuilt.impossible_travel_kmh == 1250.5
    assert rebuilt.quarantined is True


def test_credential_stuffing_ip_spraying() -> None:
    """Test detection of an IP spraying failed passwords across multiple accounts."""
    from datetime import timedelta

    from fraud_detection.ato import CredentialStuffingDetector

    detector = CredentialStuffingDetector(window_seconds=300, ip_spray_user_threshold=3)
    now = datetime(2026, 11, 8, 14, 0, 0, tzinfo=UTC)

    # 1st failure: user_1
    detector.record_attempt(
        LoginEvent(
            user_id="user_1",
            ip_address="203.0.113.5",
            outcome=LoginOutcome.FAILED_CREDENTIALS,
            timestamp=now,
        )
    )
    is_spray, score, count = detector.check_ip_spraying("203.0.113.5", now)
    assert not is_spray
    assert count == 1
    assert score == 0.0

    # 2nd failure: user_2
    detector.record_attempt(
        LoginEvent(
            user_id="user_2",
            ip_address="203.0.113.5",
            outcome=LoginOutcome.FAILED_CREDENTIALS,
            timestamp=now + timedelta(seconds=10),
        )
    )
    is_spray, score, count = detector.check_ip_spraying("203.0.113.5", now + timedelta(seconds=10))
    assert not is_spray
    assert count == 2

    # 3rd failure: user_3 -> triggers threshold
    detector.record_attempt(
        LoginEvent(
            user_id="user_3",
            ip_address="203.0.113.5",
            outcome=LoginOutcome.FAILED_CREDENTIALS,
            timestamp=now + timedelta(seconds=20),
        )
    )
    is_spray, score, count = detector.check_ip_spraying("203.0.113.5", now + timedelta(seconds=20))
    assert is_spray
    assert count == 3
    assert score >= 0.5

    # Evaluation returns PASSWORD_SPRAYING factor
    eval_event = LoginEvent(
        user_id="user_4",
        ip_address="203.0.113.5",
        outcome=LoginOutcome.FAILED_CREDENTIALS,
        timestamp=now + timedelta(seconds=25),
    )
    eval_score, factors = detector.evaluate_stuffing_risk(eval_event)
    assert AtoRiskFactor.PASSWORD_SPRAYING in factors
    assert eval_score >= 0.5


def test_credential_stuffing_distributed_brute_force() -> None:
    """Test detection of a single user attacked from multiple distinct IPs."""
    from datetime import timedelta

    from fraud_detection.ato import CredentialStuffingDetector

    detector = CredentialStuffingDetector(window_seconds=300, distributed_user_ip_threshold=3)
    now = datetime(2026, 11, 8, 14, 0, 0, tzinfo=UTC)

    # User targeted from 3 different IPs
    for i in range(3):
        detector.record_attempt(
            LoginEvent(
                user_id="target_vip",
                ip_address=f"192.0.2.{10 + i}",
                outcome=LoginOutcome.FAILED_CREDENTIALS,
                timestamp=now + timedelta(seconds=i * 5),
            )
        )

    is_bf, score, count = detector.check_distributed_brute_force(
        "target_vip", now + timedelta(seconds=20)
    )
    assert is_bf
    assert count == 3
    assert score >= 0.5

    eval_event = LoginEvent(
        user_id="target_vip",
        ip_address="192.0.2.99",
        timestamp=now + timedelta(seconds=20),
    )
    eval_score, factors = detector.evaluate_stuffing_risk(eval_event)
    assert AtoRiskFactor.CREDENTIAL_STUFFING_BURST in factors
    assert eval_score >= 0.5


def test_credential_stuffing_subnet_burst_and_window_expiry() -> None:
    """Test subnet-level burst detection and window expiry."""
    from datetime import timedelta

    from fraud_detection.ato import CredentialStuffingDetector

    detector = CredentialStuffingDetector(window_seconds=60, subnet_spray_threshold=3)
    t0 = datetime(2026, 11, 8, 15, 0, 0, tzinfo=UTC)

    # 3 failures across different IPs in 198.51.100.0/24 targeting 3 distinct users
    for i in range(3):
        detector.record_attempt(
            LoginEvent(
                user_id=f"user_{i}",
                ip_address=f"198.51.100.{10 + i}",
                outcome=LoginOutcome.FAILED_CREDENTIALS,
                timestamp=t0 + timedelta(seconds=i * 2),
            )
        )

    is_sub, score, count = detector.check_subnet_burst("198.51.100.50", t0 + timedelta(seconds=10))
    assert is_sub
    assert count == 3
    assert score >= 0.4

    # Test window expiry: at t0 + 120 seconds, previous attempts are pruned
    t_later = t0 + timedelta(seconds=120)
    is_sub_expired, score_expired, count_expired = detector.check_subnet_burst(
        "198.51.100.50", t_later
    )
    assert not is_sub_expired
    assert count_expired == 0
    assert score_expired == 0.0

    # Test IPv6 subnet extraction
    detector.record_attempt(
        LoginEvent(
            user_id="ipv6_victim",
            ip_address="2001:db8:85a3::8a2e:370:7334",
            outcome=LoginOutcome.FAILED_CREDENTIALS,
            timestamp=t_later,
        )
    )
    is_sub_v6, _, _ = detector.check_subnet_burst("2001:db8:85a3::1", t_later)
    assert not is_sub_v6  # only 1 attempt, below threshold of 3

    # Test invalid IP fallback
    invalid_sub = detector._extract_subnet("not-an-ip")  # noqa: SLF001
    assert invalid_sub == "not-an-ip"

    # Querying unrecorded IP / user
    assert detector.check_ip_spraying("10.0.0.1")[0] is False
    assert detector.check_distributed_brute_force("unknown_user")[0] is False
    assert detector.check_subnet_burst("10.0.0.1")[0] is False
