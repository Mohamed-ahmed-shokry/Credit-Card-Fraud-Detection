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
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol


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


@dataclass(frozen=True)
class ChallengePolicy:
    """Configurable step-up challenge and strong customer authentication policy."""

    allowed_methods: tuple[ChallengeMethod, ...] = (
        ChallengeMethod.SMS_OTP,
        ChallengeMethod.EMAIL_OTP,
        ChallengeMethod.TOTP,
        ChallengeMethod.BIOMETRIC_PUSH,
    )
    default_ttl_seconds: int = 300
    max_attempts: int = 3
    lockout_duration_seconds: int = 900
    resend_cooldown_seconds: int = 30
    otp_length: int = 6
    numeric_only: bool = True

    def __post_init__(self) -> None:
        """Validate policy bounds."""
        if self.default_ttl_seconds < 10:
            raise ValueError("default_ttl_seconds must be at least 10 seconds")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.lockout_duration_seconds < 0:
            raise ValueError("lockout_duration_seconds cannot be negative")
        if self.resend_cooldown_seconds < 0:
            raise ValueError("resend_cooldown_seconds cannot be negative")
        if self.otp_length < 4 or self.otp_length > 32:
            raise ValueError("otp_length must be between 4 and 32")


class ChallengeStore:
    """Thread-safe in-memory store for challenge sessions and user lockouts."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, ChallengeSession] = {}
        self._by_tx: dict[str, str] = {}
        self._user_lockouts: dict[tuple[str, str], datetime] = {}

    def put(self, session: ChallengeSession) -> None:
        """Store or update a challenge session."""
        with self._lock:
            self._sessions[session.challenge_id] = session
            self._by_tx[session.transaction_id] = session.challenge_id

    def get(self, challenge_id: str) -> ChallengeSession | None:
        """Retrieve a challenge session by its challenge ID."""
        with self._lock:
            return self._sessions.get(challenge_id)

    def get_by_transaction(self, transaction_id: str) -> ChallengeSession | None:
        """Retrieve the latest challenge session for a transaction ID."""
        with self._lock:
            ch_id = self._by_tx.get(transaction_id)
            if ch_id:
                return self._sessions.get(ch_id)
            return None

    def delete(self, challenge_id: str) -> bool:
        """Remove a challenge session by ID."""
        with self._lock:
            session = self._sessions.pop(challenge_id, None)
            if session:
                if self._by_tx.get(session.transaction_id) == challenge_id:
                    self._by_tx.pop(session.transaction_id, None)
                return True
            return False

    def set_user_lockout(self, tenant_id: str, user_id: str, until: datetime) -> None:
        """Record a temporary lockout for a specific user within a tenant."""
        with self._lock:
            self._user_lockouts[(tenant_id, user_id)] = until

    def is_user_locked_out(self, tenant_id: str, user_id: str) -> bool:
        """Check whether a user is currently locked out."""
        with self._lock:
            until = self._user_lockouts.get((tenant_id, user_id))
            if until is None:
                return False
            now = datetime.now(UTC)
            if now >= until:
                self._user_lockouts.pop((tenant_id, user_id), None)
                return False
            return True

    def prune_expired(self) -> int:
        """Prune expired sessions and clean up expired lockouts."""
        with self._lock:
            now = datetime.now(UTC)
            expired_ids = [s.challenge_id for s in self._sessions.values() if s.is_expired]
            for ch_id in expired_ids:
                s = self._sessions[ch_id]
                s.status = ChallengeStatus.EXPIRED

            expired_lockouts = [k for k, until in self._user_lockouts.items() if now >= until]
            for k in expired_lockouts:
                self._user_lockouts.pop(k, None)

            return len(expired_ids)

    def list_sessions(
        self, tenant_id: str | None = None, limit: int = 100
    ) -> list[ChallengeSession]:
        """List active and historical challenge sessions."""
        with self._lock:
            res: list[ChallengeSession] = []
            for s in reversed(list(self._sessions.values())):
                if tenant_id is None or s.tenant_id == tenant_id:
                    res.append(s)
                if len(res) >= limit:
                    break
            return res

    def count(self) -> int:
        """Return the total number of stored challenge sessions."""
        with self._lock:
            return len(self._sessions)

    def __len__(self) -> int:
        return self.count()

    def clear(self) -> None:
        """Clear all stored sessions and lockouts."""
        with self._lock:
            self._sessions.clear()
            self._by_tx.clear()
            self._user_lockouts.clear()


class ChallengeNotificationAdapter(Protocol):
    """Interface for dispatching challenge tokens to external communication channels."""

    def dispatch(
        self,
        session: ChallengeSession,
        raw_token: str,
        destination: str,
    ) -> bool:
        """Send a challenge token to the user destination."""
        ...


class InMemoryNotificationAdapter:
    """In-memory mock adapter recording dispatched challenge notifications."""

    def __init__(self) -> None:
        self._dispatches: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def dispatch(
        self,
        session: ChallengeSession,
        raw_token: str,
        destination: str,
    ) -> bool:
        """Record dispatched notification event."""
        with self._lock:
            self._dispatches.append(
                {
                    "challenge_id": session.challenge_id,
                    "method": session.method.value,
                    "destination": destination,
                    "target_masked": session.target_masked,
                    "raw_token": raw_token,
                    "dispatched_at": datetime.now(UTC).isoformat(),
                }
            )
            return True

    @property
    def dispatches(self) -> list[dict[str, Any]]:
        """Return a copy of dispatched messages."""
        with self._lock:
            return list(self._dispatches)

    def clear(self) -> None:
        """Clear stored dispatches."""
        with self._lock:
            self._dispatches.clear()


class ChallengeManager:
    """Orchestrates creation, verification, rate limiting, and lifecycle for step-up challenges."""

    def __init__(
        self,
        policy: ChallengePolicy | None = None,
        store: ChallengeStore | None = None,
        adapter: ChallengeNotificationAdapter | None = None,
    ) -> None:
        self.policy = policy or ChallengePolicy()
        self.store = store or ChallengeStore()
        self.adapter = adapter or InMemoryNotificationAdapter()
        self._lock = threading.RLock()

    def create_challenge(
        self,
        transaction_id: str,
        user_id: str,
        tenant_id: str,
        destination: str,
        method: ChallengeMethod = ChallengeMethod.SMS_OTP,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[ChallengeSession, str]:
        """Issue a new challenge session for a high-risk or friction transaction."""
        with self._lock:
            if self.store.is_user_locked_out(tenant_id, user_id):
                raise ChallengeLockedOutError(
                    f"User '{user_id}' in tenant '{tenant_id}' is temporarily locked out"
                )

            if method not in self.policy.allowed_methods:
                raise ValueError(f"Challenge method '{method}' is not allowed by policy")

            challenge_id = f"ch_{uuid.uuid4().hex}"
            now = datetime.now(UTC)
            expires_at = (now + timedelta(seconds=self.policy.default_ttl_seconds)).isoformat()
            raw_otp = generate_secure_otp(
                length=self.policy.otp_length, numeric_only=self.policy.numeric_only
            )
            salt = generate_salt(num_bytes=16)
            token_h = hash_challenge_token(raw_otp, salt)
            masked = mask_target_destination(destination, method)

            session = ChallengeSession(
                challenge_id=challenge_id,
                transaction_id=transaction_id,
                user_id=user_id,
                tenant_id=tenant_id,
                method=method,
                target_masked=masked,
                token_hash=token_h,
                token_salt=salt,
                attempts_count=0,
                max_attempts=self.policy.max_attempts,
                status=ChallengeStatus.PENDING,
                created_at=now.isoformat(),
                expires_at=expires_at,
                metadata=metadata or {},
            )

            self.store.put(session)
            self.adapter.dispatch(session, raw_otp, destination)
            return session, raw_otp

    def verify_challenge(self, challenge_id: str, token: str) -> ChallengeSession:
        """Verify an entered challenge token against the active session."""
        with self._lock:
            session = self.store.get(challenge_id)
            if session is None:
                raise ChallengeSessionNotFoundError(f"Challenge session '{challenge_id}' not found")

            if self.store.is_user_locked_out(session.tenant_id, session.user_id):
                raise ChallengeLockedOutError(
                    f"User '{session.user_id}' in tenant '{session.tenant_id}' is locked out"
                )

            if session.status == ChallengeStatus.VERIFIED:
                raise ChallengeError(f"Challenge session '{challenge_id}' is already verified")
            if session.status == ChallengeStatus.LOCKED_OUT:
                raise ChallengeLockedOutError(f"Challenge session '{challenge_id}' is locked out")
            if session.status in (ChallengeStatus.FAILED, ChallengeStatus.CANCELLED):
                raise ChallengeError(
                    f"Challenge session '{challenge_id}' is inactive ({session.status.value})"
                )

            if session.is_expired:
                session.status = ChallengeStatus.EXPIRED
                raise ChallengeExpiredError(f"Challenge session '{challenge_id}' has expired")

            session.attempts_count += 1
            session.last_attempt_at = datetime.now(UTC).isoformat()

            matched = verify_challenge_token(token, session.token_hash, session.token_salt)
            if matched:
                session.status = ChallengeStatus.VERIFIED
                session.verified_at = datetime.now(UTC).isoformat()
                return session

            if session.attempts_count >= session.max_attempts:
                session.status = ChallengeStatus.LOCKED_OUT
                lockout_until = datetime.now(UTC) + timedelta(
                    seconds=self.policy.lockout_duration_seconds
                )
                self.store.set_user_lockout(session.tenant_id, session.user_id, lockout_until)
                raise ChallengeMaxAttemptsExceededError(
                    f"Max attempts exceeded for challenge '{challenge_id}'. User locked out."
                )

            raise InvalidChallengeTokenError(
                f"Invalid token for challenge '{challenge_id}'. "
                f"{session.max_attempts - session.attempts_count} attempt(s) remaining."
            )

    def resend_challenge(
        self, challenge_id: str, destination: str | None = None
    ) -> tuple[ChallengeSession, str]:
        """Regenerate and resend an OTP token subject to cooldown rate-limiting."""
        with self._lock:
            session = self.store.get(challenge_id)
            if session is None:
                raise ChallengeSessionNotFoundError(f"Challenge session '{challenge_id}' not found")

            if session.status not in (ChallengeStatus.PENDING, ChallengeStatus.EXPIRED):
                raise ChallengeError(
                    f"Cannot resend token for challenge in status '{session.status.value}'"
                )

            now = datetime.now(UTC)
            last_activity = session.last_resend_at or session.created_at
            if last_activity:
                try:
                    last_dt = datetime.fromisoformat(last_activity)
                    diff = (now - last_dt).total_seconds()
                    if diff < self.policy.resend_cooldown_seconds:
                        remaining = max(1, int(self.policy.resend_cooldown_seconds - diff))
                        raise RateLimitExceededError(
                            f"Resend cooldown active. Please wait {remaining} seconds."
                        )
                except (ValueError, TypeError):
                    pass

            raw_otp = generate_secure_otp(
                length=self.policy.otp_length, numeric_only=self.policy.numeric_only
            )
            salt = generate_salt(num_bytes=16)
            token_h = hash_challenge_token(raw_otp, salt)

            session.token_hash = token_h
            session.token_salt = salt
            session.attempts_count = 0
            session.status = ChallengeStatus.PENDING
            session.last_resend_at = now.isoformat()
            session.expires_at = (
                now + timedelta(seconds=self.policy.default_ttl_seconds)
            ).isoformat()

            dest = destination or session.target_masked
            self.adapter.dispatch(session, raw_otp, dest)
            return session, raw_otp

    def cancel_challenge(self, challenge_id: str, reason: str = "") -> ChallengeSession:
        """Cancel a pending challenge session."""
        with self._lock:
            session = self.store.get(challenge_id)
            if session is None:
                raise ChallengeSessionNotFoundError(f"Challenge session '{challenge_id}' not found")
            session.status = ChallengeStatus.CANCELLED
            if reason:
                session.metadata["cancellation_reason"] = reason
            return session

    def get_session(self, challenge_id: str) -> ChallengeSession | None:
        """Look up a challenge session by ID."""
        return self.store.get(challenge_id)
