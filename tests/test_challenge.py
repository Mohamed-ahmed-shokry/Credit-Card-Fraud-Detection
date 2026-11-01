"""Unit and integration tests for Risk-Based Step-Up Challenge Orchestration (Phase 44)."""

import pytest

from fraud_detection.challenge import (
    ChallengeError,
    ChallengeExpiredError,
    ChallengeLockedOutError,
    ChallengeManager,
    ChallengeMaxAttemptsExceededError,
    ChallengeMethod,
    ChallengePolicy,
    ChallengeSession,
    ChallengeSessionNotFoundError,
    ChallengeStatus,
    ChallengeStore,
    InMemoryNotificationAdapter,
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


def test_challenge_policy_validation() -> None:
    """Verify ChallengePolicy constraint checks."""
    policy = ChallengePolicy(default_ttl_seconds=120, max_attempts=5)
    assert policy.default_ttl_seconds == 120
    assert policy.max_attempts == 5

    with pytest.raises(ValueError, match="default_ttl_seconds must be at least 10"):
        ChallengePolicy(default_ttl_seconds=5)
    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        ChallengePolicy(max_attempts=0)
    with pytest.raises(ValueError, match="lockout_duration_seconds cannot be negative"):
        ChallengePolicy(lockout_duration_seconds=-1)
    with pytest.raises(ValueError, match="resend_cooldown_seconds cannot be negative"):
        ChallengePolicy(resend_cooldown_seconds=-1)
    with pytest.raises(ValueError, match="otp_length must be between 4 and 32"):
        ChallengePolicy(otp_length=2)


def test_challenge_store_operations() -> None:
    """Verify ChallengeStore storage, retrieval, lockout, and pruning."""
    store = ChallengeStore()
    token_h = "hash_" + "x" * 59
    salt_s = "salt_" + "y" * 27
    session = ChallengeSession(
        challenge_id="ch_test_1",
        transaction_id="tx_1",
        user_id="user_1",
        tenant_id="tenant_1",
        method=ChallengeMethod.EMAIL_OTP,
        target_masked="u***@example.com",
        token_hash=token_h,
        token_salt=salt_s,
    )

    store.put(session)
    assert len(store) == 1
    assert store.get("ch_test_1") is session
    assert store.get_by_transaction("tx_1") is session
    assert store.get_by_transaction("tx_nonexistent") is None

    sessions = store.list_sessions(tenant_id="tenant_1")
    assert len(sessions) == 1
    assert store.list_sessions(tenant_id="tenant_other") == []

    # Test lockout
    from datetime import UTC, datetime, timedelta

    assert store.is_user_locked_out("tenant_1", "user_1") is False
    lock_until = datetime.now(UTC) + timedelta(minutes=10)
    store.set_user_lockout("tenant_1", "user_1", lock_until)
    assert store.is_user_locked_out("tenant_1", "user_1") is True

    # Test prune
    past = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    session.expires_at = past
    pruned = store.prune_expired()
    assert pruned == 1
    assert session.status == ChallengeStatus.EXPIRED

    assert store.delete("ch_test_1") is True
    assert store.delete("ch_test_1") is False
    assert len(store) == 0

    store.clear()
    assert store.count() == 0


def test_challenge_manager_lifecycle_success() -> None:
    """Verify successful end-to-end challenge issuance, notification, and verification."""
    adapter = InMemoryNotificationAdapter()
    manager = ChallengeManager(adapter=adapter)

    session, raw_token = manager.create_challenge(
        transaction_id="tx_flow_1",
        user_id="alice",
        tenant_id="tenant_main",
        destination="alice@company.com",
        method=ChallengeMethod.EMAIL_OTP,
        metadata={"risk_score": 0.85},
    )

    assert session.status == ChallengeStatus.PENDING
    assert session.target_masked == "a***e@company.com"
    assert len(adapter.dispatches) == 1
    assert adapter.dispatches[0]["raw_token"] == raw_token

    # Verify challenge with correct token
    verified_session = manager.verify_challenge(session.challenge_id, raw_token)
    assert verified_session.status == ChallengeStatus.VERIFIED
    assert verified_session.verified_at is not None

    # Cannot verify twice
    with pytest.raises(ChallengeError, match="already verified"):
        manager.verify_challenge(session.challenge_id, raw_token)


def test_challenge_manager_lockout_and_attempts() -> None:
    """Verify invalid attempts count, lockout triggering, and blocked operations."""
    policy = ChallengePolicy(max_attempts=2, lockout_duration_seconds=300)
    manager = ChallengeManager(policy=policy)

    session, _ = manager.create_challenge(
        transaction_id="tx_flow_2",
        user_id="bob",
        tenant_id="tenant_main",
        destination="+15551234567",
        method=ChallengeMethod.SMS_OTP,
    )

    wrong_tok_1 = "000000"
    wrong_tok_2 = "999999"

    with pytest.raises(InvalidChallengeTokenError, match="1 attempt\\(s\\) remaining"):
        manager.verify_challenge(session.challenge_id, wrong_tok_1)

    assert session.attempts_count == 1

    # Second wrong attempt -> max attempts reached -> locked out!
    with pytest.raises(ChallengeMaxAttemptsExceededError, match="Max attempts exceeded"):
        manager.verify_challenge(session.challenge_id, wrong_tok_2)

    assert session.status == ChallengeStatus.LOCKED_OUT
    assert manager.store.is_user_locked_out("tenant_main", "bob") is True

    # Subsequent verification rejected as locked out
    with pytest.raises(ChallengeLockedOutError):
        manager.verify_challenge(session.challenge_id, wrong_tok_1)

    # Creating a new challenge while user is locked out is rejected
    with pytest.raises(ChallengeLockedOutError):
        manager.create_challenge(
            transaction_id="tx_flow_3",
            user_id="bob",
            tenant_id="tenant_main",
            destination="+15551234567",
        )


def test_challenge_manager_resend_and_cooldown() -> None:
    """Verify token resending, cooldown enforcement, and expiration handling."""
    policy = ChallengePolicy(resend_cooldown_seconds=60)
    adapter = InMemoryNotificationAdapter()
    manager = ChallengeManager(policy=policy, adapter=adapter)

    session, token1 = manager.create_challenge(
        transaction_id="tx_flow_4",
        user_id="carol",
        tenant_id="tenant_main",
        destination="carol@domain.com",
        method=ChallengeMethod.EMAIL_OTP,
    )

    # Immediate resend triggers rate limit
    with pytest.raises(RateLimitExceededError, match="Resend cooldown active"):
        manager.resend_challenge(session.challenge_id)

    # Fast forward created_at and last_resend_at
    from datetime import UTC, datetime, timedelta

    past_dt = (datetime.now(UTC) - timedelta(seconds=70)).isoformat()
    session.created_at = past_dt
    session.last_resend_at = past_dt

    # Resend succeeds
    updated_session, token2 = manager.resend_challenge(session.challenge_id)
    assert len(adapter.dispatches) == 2
    assert updated_session.challenge_id == session.challenge_id
    assert updated_session.token_hash != session.token_hash or token1 != token2

    # Verify with new token succeeds
    res = manager.verify_challenge(session.challenge_id, token2)
    assert res.status == ChallengeStatus.VERIFIED


def test_challenge_manager_expiration_and_cancellation() -> None:
    """Verify expired token rejection and cancellation."""
    from datetime import UTC, datetime, timedelta

    manager = ChallengeManager()
    session, raw_token = manager.create_challenge(
        transaction_id="tx_flow_5",
        user_id="david",
        tenant_id="tenant_main",
        destination="david@domain.com",
        method=ChallengeMethod.EMAIL_OTP,
    )

    # Make session expired
    session.expires_at = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
    with pytest.raises(ChallengeExpiredError, match="has expired"):
        manager.verify_challenge(session.challenge_id, raw_token)

    # Create another session and cancel it
    session2, _ = manager.create_challenge(
        transaction_id="tx_flow_6",
        user_id="eve",
        tenant_id="tenant_main",
        destination="eve@domain.com",
        method=ChallengeMethod.EMAIL_OTP,
    )
    cancelled = manager.cancel_challenge(session2.challenge_id, reason="Customer abandoned")
    assert cancelled.status == ChallengeStatus.CANCELLED
    assert cancelled.metadata["cancellation_reason"] == "Customer abandoned"

    with pytest.raises(ChallengeError, match="is inactive"):
        manager.verify_challenge(session2.challenge_id, "123456")

    # Nonexistent session
    with pytest.raises(ChallengeSessionNotFoundError):
        manager.verify_challenge("ch_nonexistent", "123456")
    with pytest.raises(ChallengeSessionNotFoundError):
        manager.resend_challenge("ch_nonexistent")
    with pytest.raises(ChallengeSessionNotFoundError):
        manager.cancel_challenge("ch_nonexistent")
