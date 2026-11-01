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
    generate_salt,
    generate_secure_otp,
    hash_challenge_token,
    mask_target_destination,
    verify_challenge_token,
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


def test_generate_secure_otp() -> None:
    """Verify cryptographically secure OTP generation length and character sets."""
    otp6 = generate_secure_otp(length=6, numeric_only=True)
    assert len(otp6) == 6
    assert otp6.isdigit()

    otp8 = generate_secure_otp(length=8, numeric_only=False)
    assert len(otp8) == 8
    assert all((c.isalnum() and c.isupper()) or c.isdigit() for c in otp8)

    with pytest.raises(ValueError, match="OTP length must be between 4 and 32"):
        generate_secure_otp(length=3)
    with pytest.raises(ValueError, match="OTP length must be between 4 and 32"):
        generate_secure_otp(length=33)


def test_hash_and_verify_challenge_token() -> None:
    """Verify constant-time HMAC-SHA256 token hashing and verification."""
    salt = generate_salt(num_bytes=16)
    assert len(salt) == 32
    candidate_val = "".join(str(i) for i in [4, 9, 2, 0, 1, 8])
    wrong_val = "".join(str(i) for i in [4, 9, 2, 0, 1, 9])
    h = hash_challenge_token(candidate_val, salt)

    assert verify_challenge_token(candidate_val, h, salt) is True
    assert verify_challenge_token(wrong_val, h, salt) is False
    assert verify_challenge_token("", h, salt) is False
    assert verify_challenge_token(candidate_val, "", salt) is False
    assert verify_challenge_token(candidate_val, h, "") is False

    with pytest.raises(ValueError, match="Salt length must be at least 8 bytes"):
        generate_salt(num_bytes=4)
    with pytest.raises(ValueError, match="token must be a non-empty string"):
        hash_challenge_token("", salt)
    with pytest.raises(ValueError, match="salt must be a non-empty string"):
        hash_challenge_token(candidate_val, "")


def test_mask_target_destination() -> None:
    """Verify PII masking across email, SMS phone numbers, and generic identifiers."""
    masked_email = mask_target_destination("johndoe@example.com", ChallengeMethod.EMAIL_OTP)
    assert masked_email == "j***e@example.com"

    masked_short_email = mask_target_destination("ab@test.org", ChallengeMethod.EMAIL_OTP)
    assert masked_short_email == "a***@test.org"

    masked_phone = mask_target_destination("+14155552671", ChallengeMethod.SMS_OTP)
    assert masked_phone == "+1 ***-***-2671"

    masked_intl = mask_target_destination("+442071234567", ChallengeMethod.SMS_OTP)
    assert masked_intl == "+44 ***-***-4567"

    masked_short_phone = mask_target_destination("1234567", ChallengeMethod.SMS_OTP)
    assert masked_short_phone == "***4567"

    masked_generic = mask_target_destination("user_security_token_99", ChallengeMethod.OUT_OF_BAND)
    assert masked_generic == "us***99"

    assert mask_target_destination("", ChallengeMethod.SMS_OTP) == "***"
