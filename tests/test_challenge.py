"""Unit and integration tests for Risk-Based Step-Up Challenge Orchestration (Phase 44)."""

import pytest

from fraud_detection.challenge import (
    ChallengeError,
    ChallengeExpiredError,
    ChallengeLockedOutError,
    ChallengeMaxAttemptsExceededError,
    ChallengeMethod,
    ChallengeSession,
    ChallengeSessionNotFoundError,
    ChallengeStatus,
    InvalidChallengeTokenError,
    RateLimitExceededError,
)


def test_challenge_models_and_validation() -> None:
    """Verify ChallengeSession field validation, serialization, and status enums."""
    token_h = "hash_" + "a" * 59
    salt_s = "salt_" + "b" * 27
    session = ChallengeSession(
        challenge_id="ch-1001",
        transaction_id="tx-999",
        user_id="usr-42",
        tenant_id="tenant-alpha",
        method=ChallengeMethod.SMS_OTP,
        target_masked="+1 ***-***-1234",
        token_hash=token_h,
        token_salt=salt_s,
        attempts_count=0,
        max_attempts=3,
        status=ChallengeStatus.PENDING,
        created_at="2026-11-01T00:00:00+00:00",
        expires_at="2026-11-01T00:15:00+00:00",
        metadata={"channel": "mobile_app"},
    )

    assert session.challenge_id == "ch-1001"
    assert session.method == ChallengeMethod.SMS_OTP
    assert session.status == ChallengeStatus.PENDING
    assert session.is_expired is True  # In past

    d = session.to_dict()
    assert d["challenge_id"] == "ch-1001"
    assert d["method"] == "sms_otp"
    assert d["status"] == "pending"

    rehydrated = ChallengeSession.from_dict(d)
    assert rehydrated.challenge_id == session.challenge_id
    assert rehydrated.method == session.method
    assert rehydrated.status == session.status


def test_challenge_exceptions_hierarchy() -> None:
    """Verify challenge exception inheritance hierarchy."""
    assert issubclass(ChallengeSessionNotFoundError, ChallengeError)
    assert issubclass(ChallengeExpiredError, ChallengeError)
    assert issubclass(ChallengeLockedOutError, ChallengeError)
    assert issubclass(ChallengeMaxAttemptsExceededError, ChallengeError)
    assert issubclass(InvalidChallengeTokenError, ChallengeError)
    assert issubclass(RateLimitExceededError, ChallengeError)


def test_challenge_models_invariants() -> None:
    """Verify ChallengeSession raises ValueError on invalid inputs."""
    token_h = "hash_" + "a" * 59
    salt_s = "salt_" + "b" * 27

    with pytest.raises(ValueError, match="challenge_id must be a non-empty string"):
        ChallengeSession(
            challenge_id="",
            transaction_id="tx",
            user_id="u",
            tenant_id="t",
            method=ChallengeMethod.SMS_OTP,
            target_masked="x",
            token_hash=token_h,
            token_salt=salt_s,
        )

    with pytest.raises(ValueError, match="attempts_count cannot be negative"):
        ChallengeSession(
            challenge_id="ch",
            transaction_id="tx",
            user_id="u",
            tenant_id="t",
            method=ChallengeMethod.SMS_OTP,
            target_masked="x",
            token_hash=token_h,
            token_salt=salt_s,
            attempts_count=-1,
        )

    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        ChallengeSession(
            challenge_id="ch",
            transaction_id="tx",
            user_id="u",
            tenant_id="t",
            method=ChallengeMethod.SMS_OTP,
            target_masked="x",
            token_hash=token_h,
            token_salt=salt_s,
            max_attempts=0,
        )

    with pytest.raises(ValueError, match="Invalid challenge method"):
        ChallengeSession(
            challenge_id="ch",
            transaction_id="tx",
            user_id="u",
            tenant_id="t",
            method="unknown_method",  # type: ignore[arg-type]
            target_masked="x",
            token_hash=token_h,
            token_salt=salt_s,
        )
