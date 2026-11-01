"""Risk-Based Step-Up Authentication & Challenge Orchestration (Phase 44).

Implements Strong Customer Authentication (SCA) step-up challenges and 3-D Secure
(3DS 2.3) orchestration for friction decisions. Provides cryptographically secure
OTP generation, constant-time verification, multi-channel challenge methods,
rate-limiting, brute-force lockout, and thread-safe session lifecycle management.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import string
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def generate_secure_otp(length: int = 6, numeric_only: bool = True) -> str:
    """Generate a cryptographically secure one-time password (OTP).

    Args:
        length: Number of characters in the OTP (must be between 4 and 32).
        numeric_only: If True, uses digits '0'-'9'; otherwise alphanumeric.

    Returns:
        The generated OTP string.
    """
    if length < 4 or length > 32:
        raise ValueError("OTP length must be between 4 and 32 characters")

    alphabet = string.digits if numeric_only else string.digits + string.ascii_uppercase
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_salt(num_bytes: int = 16) -> str:
    """Generate a cryptographically secure random hexadecimal salt."""
    if num_bytes < 8:
        raise ValueError("Salt length must be at least 8 bytes")
    return secrets.token_hex(num_bytes)


def hash_challenge_token(token: str, salt: str) -> str:
    """Compute an HMAC-SHA256 hash of a verification token with salt.

    Args:
        token: The plaintext token or OTP string.
        salt: The hexadecimal salt string.

    Returns:
        Hex-encoded HMAC-SHA256 digest string.
    """
    if not token or not isinstance(token, str):
        raise ValueError("token must be a non-empty string")
    if not salt or not isinstance(salt, str):
        raise ValueError("salt must be a non-empty string")

    key = salt.encode("utf-8")
    msg = token.strip().encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def verify_challenge_token(token: str, token_hash: str, salt: str) -> bool:
    """Verify a challenge token against its stored hash using constant-time comparison.

    Args:
        token: The candidate token provided by the user.
        token_hash: The stored expected HMAC-SHA256 hex digest.
        salt: The salt used when creating the hash.

    Returns:
        True if the token matches, False otherwise.
    """
    if not token or not isinstance(token, str):
        return False
    if not token_hash or not isinstance(token_hash, str):
        return False
    if not salt or not isinstance(salt, str):
        return False

    computed = hash_challenge_token(token, salt)
    return hmac.compare_digest(computed, token_hash)


def mask_target_destination(target: str, method: ChallengeMethod) -> str:
    """Mask a destination address (phone number, email) to protect PII.

    Args:
        target: The raw destination string (e.g. phone number or email).
        method: The challenge method.

    Returns:
        A masked string preserving minimal identifiable parts.
    """
    if not target or not isinstance(target, str):
        return "***"

    clean_target = target.strip()
    if method == ChallengeMethod.EMAIL_OTP or "@" in clean_target:
        parts = clean_target.split("@", 1)
        name, domain = parts[0], parts[1] if len(parts) > 1 else ""
        if len(name) <= 2:
            masked_name = name[0] + "***" if name else "***"
        else:
            masked_name = name[0] + "***" + name[-1]
        return f"{masked_name}@{domain}" if domain else masked_name

    if method == ChallengeMethod.SMS_OTP:
        digits_only = "".join(c for c in clean_target if c.isdigit())
        if len(digits_only) >= 10:
            last4 = digits_only[-4:]
            country = "+" + digits_only[:-10] if len(digits_only) > 10 else "+1"
            return f"{country} ***-***-{last4}"
        if len(clean_target) > 4:
            return "*" * (len(clean_target) - 4) + clean_target[-4:]
        return "***"

    if len(clean_target) > 4:
        return clean_target[:2] + "***" + clean_target[-2:]
    return "***"


class ChallengeError(Exception):
    """Base exception for all challenge orchestration errors."""


class ChallengeSessionNotFoundError(ChallengeError):
    """Raised when a challenge session is not found."""


class ChallengeExpiredError(ChallengeError):
    """Raised when an operation is attempted on an expired challenge session."""


class ChallengeLockedOutError(ChallengeError):
    """Raised when a user or session is temporarily locked out due to excessive failed attempts."""


class ChallengeMaxAttemptsExceededError(ChallengeError):
    """Raised when the maximum number of verification attempts has been exceeded."""


class InvalidChallengeTokenError(ChallengeError):
    """Raised when an invalid verification token or code is provided."""


class RateLimitExceededError(ChallengeError):
    """Raised when challenge token resend or verification exceeds allowed rate limits."""


class ChallengeMethod(StrEnum):
    """Supported strong customer authentication (SCA) challenge methods."""

    SMS_OTP = "sms_otp"
    EMAIL_OTP = "email_otp"
    TOTP = "totp"
    BIOMETRIC_PUSH = "biometric_push"
    KBA_SECURITY_QUESTIONS = "kba_security_questions"
    OUT_OF_BAND = "out_of_band"


class ChallengeStatus(StrEnum):
    """Lifecycle statuses for a step-up challenge session."""

    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"
    LOCKED_OUT = "locked_out"
    CANCELLED = "cancelled"


@dataclass(frozen=False)
class ChallengeSession:
    """Represents an active or resolved step-up authentication challenge session."""

    challenge_id: str
    transaction_id: str
    user_id: str
    tenant_id: str
    method: ChallengeMethod
    target_masked: str
    token_hash: str
    token_salt: str
    attempts_count: int = 0
    max_attempts: int = 3
    status: ChallengeStatus = ChallengeStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    expires_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_attempt_at: str | None = None
    last_resend_at: str | None = None
    verified_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate invariant constraints upon instantiation."""
        if not self.challenge_id or not isinstance(self.challenge_id, str):
            raise ValueError("challenge_id must be a non-empty string")
        if not self.transaction_id or not isinstance(self.transaction_id, str):
            raise ValueError("transaction_id must be a non-empty string")
        if not self.user_id or not isinstance(self.user_id, str):
            raise ValueError("user_id must be a non-empty string")
        if not self.tenant_id or not isinstance(self.tenant_id, str):
            raise ValueError("tenant_id must be a non-empty string")
        raw_method: Any = self.method
        if not isinstance(raw_method, ChallengeMethod):
            try:
                self.method = ChallengeMethod(raw_method)
            except ValueError as err:
                raise ValueError(f"Invalid challenge method: {raw_method}") from err
        raw_status: Any = self.status
        if not isinstance(raw_status, ChallengeStatus):
            try:
                self.status = ChallengeStatus(raw_status)
            except ValueError as err:
                raise ValueError(f"Invalid challenge status: {raw_status}") from err
        if self.attempts_count < 0:
            raise ValueError("attempts_count cannot be negative")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    @property
    def is_expired(self) -> bool:
        """Return True if the challenge session has passed its expiration time."""
        try:
            exp = datetime.fromisoformat(self.expires_at)
            now = datetime.now(UTC)
            return now >= exp
        except (ValueError, TypeError):
            return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize the challenge session to a JSON-compatible dictionary."""
        return {
            "challenge_id": self.challenge_id,
            "transaction_id": self.transaction_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "method": self.method.value,
            "target_masked": self.target_masked,
            "token_hash": self.token_hash,
            "token_salt": self.token_salt,
            "attempts_count": self.attempts_count,
            "max_attempts": self.max_attempts,
            "status": self.status.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_attempt_at": self.last_attempt_at,
            "last_resend_at": self.last_resend_at,
            "verified_at": self.verified_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChallengeSession:
        """Deserialize a challenge session from a dictionary."""
        return cls(
            challenge_id=str(data["challenge_id"]),
            transaction_id=str(data["transaction_id"]),
            user_id=str(data["user_id"]),
            tenant_id=str(data["tenant_id"]),
            method=ChallengeMethod(data["method"]),
            target_masked=str(data["target_masked"]),
            token_hash=str(data["token_hash"]),
            token_salt=str(data["token_salt"]),
            attempts_count=int(data.get("attempts_count", 0)),
            max_attempts=int(data.get("max_attempts", 3)),
            status=ChallengeStatus(data.get("status", ChallengeStatus.PENDING.value)),
            created_at=str(data["created_at"]),
            expires_at=str(data["expires_at"]),
            last_attempt_at=data.get("last_attempt_at"),
            last_resend_at=data.get("last_resend_at"),
            verified_at=data.get("verified_at"),
            metadata=dict(data.get("metadata", {})),
        )
