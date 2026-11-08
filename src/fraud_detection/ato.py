"""Account Takeover (ATO) Detection, Credential Stuffing Defense & Identity Session Anomaly Engine.

Provides domain models, sliding-window credential stuffing detection, user account
baselines with mutation decay penalties, multi-factor ATO risk evaluation, and
thread-safe store with automated defense orchestration (challenge/quarantine).
"""

from __future__ import annotations

import ipaddress
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import uuid4


class AtoRiskFactor(StrEnum):
    """Specific risk factors triggering account takeover suspicion."""

    NEW_DEVICE = "new_device"
    IMPOSSIBLE_TRAVEL = "impossible_travel"
    RECENT_CREDENTIAL_CHANGE = "recent_credential_change"
    RECENT_MFA_RESET = "recent_mfa_reset"
    CREDENTIAL_STUFFING_BURST = "credential_stuffing_burst"
    PASSWORD_SPRAYING = "password_spraying"  # noqa: S105
    DORMANT_ACCOUNT_REACTIVE = "dormant_account_reactive"
    VELOCITY_SPIKE = "velocity_spike"
    ACCOUNT_QUARANTINED = "account_quarantined"


class AtoAction(StrEnum):
    """Enforcement policy actions triggered by ATO assessment."""

    ALLOW = "allow"
    CHALLENGE = "challenge"
    QUARANTINE_ACCOUNT = "quarantine_account"
    BLOCK = "block"


class LoginOutcome(StrEnum):
    """Authentication attempt resolution status."""

    SUCCESS = "success"
    FAILED_CREDENTIALS = "failed_credentials"
    FAILED_MFA = "failed_mfa"
    BLOCKED = "blocked"


class AccountMutationType(StrEnum):
    """Sensitive account profile changes evaluated for takeover risk."""

    PASSWORD_CHANGE = "password_change"  # noqa: S105
    EMAIL_CHANGE = "email_change"
    PHONE_CHANGE = "phone_change"
    MFA_RESET = "mfa_reset"
    SHIPPING_ADDRESS_CHANGE = "shipping_address_change"


@dataclass(frozen=True)
class AccountMutationRecord:
    """Immutable log of a sensitive account profile mutation."""

    mutation_type: AccountMutationType
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    channel: str = "web"
    mutation_id: str = field(default_factory=lambda: f"mut_{uuid4().hex[:12]}")
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize mutation record to dictionary."""
        return {
            "mutation_id": self.mutation_id,
            "mutation_type": self.mutation_type.value,
            "timestamp": self.timestamp.isoformat(),
            "channel": self.channel,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AccountMutationRecord:
        """Instantiate mutation record from dictionary."""
        raw_ts = data.get("timestamp")
        if isinstance(raw_ts, str):
            ts = datetime.fromisoformat(raw_ts)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
        elif isinstance(raw_ts, datetime):
            ts = raw_ts if raw_ts.tzinfo is not None else raw_ts.replace(tzinfo=UTC)
        else:
            ts = datetime.now(UTC)

        m_type_raw = data.get("mutation_type", AccountMutationType.PASSWORD_CHANGE.value)
        m_type = (
            m_type_raw
            if isinstance(m_type_raw, AccountMutationType)
            else AccountMutationType(m_type_raw)
        )

        return cls(
            mutation_id=str(data.get("mutation_id", f"mut_{uuid4().hex[:12]}")),
            mutation_type=m_type,
            timestamp=ts,
            channel=str(data.get("channel", "web")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class LoginEvent:
    """Telemetry record of a single user authentication attempt."""

    user_id: str
    tenant_id: str = "default"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    ip_address: str = "127.0.0.1"
    outcome: LoginOutcome = LoginOutcome.SUCCESS
    device_fingerprint: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    user_agent: str | None = None
    event_id: str = field(default_factory=lambda: f"lgn_{uuid4().hex[:12]}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize login event to dictionary."""
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp.isoformat(),
            "ip_address": self.ip_address,
            "outcome": self.outcome.value,
            "device_fingerprint": self.device_fingerprint,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "user_agent": self.user_agent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoginEvent:
        """Instantiate login event from dictionary."""
        raw_ts = data.get("timestamp")
        if isinstance(raw_ts, str):
            ts = datetime.fromisoformat(raw_ts)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
        elif isinstance(raw_ts, datetime):
            ts = raw_ts if raw_ts.tzinfo is not None else raw_ts.replace(tzinfo=UTC)
        else:
            ts = datetime.now(UTC)

        outcome_raw = data.get("outcome", LoginOutcome.SUCCESS.value)
        outcome = (
            outcome_raw if isinstance(outcome_raw, LoginOutcome) else LoginOutcome(outcome_raw)
        )

        lat = data.get("latitude")
        lon = data.get("longitude")

        return cls(
            event_id=str(data.get("event_id", f"lgn_{uuid4().hex[:12]}")),
            user_id=str(data["user_id"]),
            tenant_id=str(data.get("tenant_id", "default")),
            timestamp=ts,
            ip_address=str(data.get("ip_address", "127.0.0.1")),
            outcome=outcome,
            device_fingerprint=data.get("device_fingerprint"),
            latitude=float(lat) if lat is not None else None,
            longitude=float(lon) if lon is not None else None,
            user_agent=data.get("user_agent"),
        )


@dataclass(frozen=True)
class AtoRiskAssessment:
    """Comprehensive evaluation record generated by AtoEngine."""

    user_id: str
    risk_score: float
    recommended_action: AtoAction
    tenant_id: str = "default"
    factors_triggered: tuple[AtoRiskFactor, ...] = field(default_factory=tuple)
    score_breakdown: dict[str, float] = field(default_factory=dict)
    impossible_travel_kmh: float | None = None
    is_new_device: bool = False
    hours_since_last_mutation: float | None = None
    quarantined: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = field(default_factory=dict)
    assessment_id: str = field(default_factory=lambda: f"ato_{uuid4().hex[:12]}")

    def __post_init__(self) -> None:
        """Validate bounded invariants."""
        if not (0.0 <= self.risk_score <= 1.0):
            msg = f"risk_score must be in range [0.0, 1.0], got {self.risk_score}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize risk assessment to dictionary."""
        return {
            "assessment_id": self.assessment_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "risk_score": round(self.risk_score, 4),
            "recommended_action": self.recommended_action.value,
            "factors_triggered": [f.value for f in self.factors_triggered],
            "score_breakdown": {k: round(v, 4) for k, v in self.score_breakdown.items()},
            "impossible_travel_kmh": (
                round(self.impossible_travel_kmh, 2)
                if self.impossible_travel_kmh is not None
                else None
            ),
            "is_new_device": self.is_new_device,
            "hours_since_last_mutation": (
                round(self.hours_since_last_mutation, 2)
                if self.hours_since_last_mutation is not None
                else None
            ),
            "quarantined": self.quarantined,
            "timestamp": self.timestamp.isoformat(),
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AtoRiskAssessment:
        """Instantiate assessment from dictionary."""
        raw_ts = data.get("timestamp")
        if isinstance(raw_ts, str):
            ts = datetime.fromisoformat(raw_ts)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
        elif isinstance(raw_ts, datetime):
            ts = raw_ts if raw_ts.tzinfo is not None else raw_ts.replace(tzinfo=UTC)
        else:
            ts = datetime.now(UTC)

        action_raw = data.get("recommended_action", AtoAction.ALLOW.value)
        action = action_raw if isinstance(action_raw, AtoAction) else AtoAction(action_raw)

        factors_raw = data.get("factors_triggered", [])
        factors = tuple(
            f if isinstance(f, AtoRiskFactor) else AtoRiskFactor(f) for f in factors_raw
        )

        speed = data.get("impossible_travel_kmh")
        mutation_hrs = data.get("hours_since_last_mutation")

        return cls(
            assessment_id=str(data.get("assessment_id", f"ato_{uuid4().hex[:12]}")),
            user_id=str(data["user_id"]),
            tenant_id=str(data.get("tenant_id", "default")),
            risk_score=float(data["risk_score"]),
            recommended_action=action,
            factors_triggered=factors,
            score_breakdown=dict(data.get("score_breakdown", {})),
            impossible_travel_kmh=float(speed) if speed is not None else None,
            is_new_device=bool(data.get("is_new_device", False)),
            hours_since_last_mutation=float(mutation_hrs) if mutation_hrs is not None else None,
            quarantined=bool(data.get("quarantined", False)),
            timestamp=ts,
            details=dict(data.get("details", {})),
        )


class CredentialStuffingDetector:
    """Sliding-window credential stuffing, password spraying and brute force detector."""

    def __init__(
        self,
        window_seconds: int = 300,
        ip_spray_user_threshold: int = 3,
        distributed_user_ip_threshold: int = 3,
        subnet_spray_threshold: int = 5,
    ) -> None:
        self.window_seconds = window_seconds
        self.ip_spray_user_threshold = ip_spray_user_threshold
        self.distributed_user_ip_threshold = distributed_user_ip_threshold
        self.subnet_spray_threshold = subnet_spray_threshold
        self._lock = threading.RLock()
        self._ip_events: dict[str, deque[LoginEvent]] = defaultdict(deque)
        self._user_events: dict[str, deque[LoginEvent]] = defaultdict(deque)
        self._subnet_events: dict[str, deque[LoginEvent]] = defaultdict(deque)

    def _extract_subnet(self, ip_str: str) -> str:
        """Extract /24 for IPv4 or /48 for IPv6 subnet prefix."""
        try:
            ip = ipaddress.ip_address(ip_str)
            if ip.version == 4:
                return str(ipaddress.ip_network(f"{ip}/24", strict=False))
            return str(ipaddress.ip_network(f"{ip}/48", strict=False))
        except ValueError:
            return ip_str

    def _prune_deque(self, q: deque[LoginEvent], cutoff: datetime) -> None:
        """Prune events older than cutoff from front of deque."""
        while q and q[0].timestamp < cutoff:
            q.popleft()

    def record_attempt(self, event: LoginEvent) -> None:
        """Record an authentication event and maintain sliding window."""
        with self._lock:
            cutoff = event.timestamp - timedelta(seconds=self.window_seconds)
            subnet = self._extract_subnet(event.ip_address)

            ip_q = self._ip_events[event.ip_address]
            ip_q.append(event)
            self._prune_deque(ip_q, cutoff)

            usr_q = self._user_events[event.user_id]
            usr_q.append(event)
            self._prune_deque(usr_q, cutoff)

            sub_q = self._subnet_events[subnet]
            sub_q.append(event)
            self._prune_deque(sub_q, cutoff)

    def check_ip_spraying(
        self, ip_address: str, at_time: datetime | None = None
    ) -> tuple[bool, float, int]:
        """Check if an IP is spraying credentials across multiple user accounts."""
        ref_time = at_time or datetime.now(UTC)
        cutoff = ref_time - timedelta(seconds=self.window_seconds)

        with self._lock:
            q = self._ip_events.get(ip_address)
            if not q:
                return False, 0.0, 0
            self._prune_deque(q, cutoff)

            # Count distinct users failed from this IP
            failed_users = {
                e.user_id
                for e in q
                if e.outcome in (LoginOutcome.FAILED_CREDENTIALS, LoginOutcome.FAILED_MFA)
            }
            count = len(failed_users)
            if count >= self.ip_spray_user_threshold:
                score = min(1.0, 0.5 + 0.1 * (count - self.ip_spray_user_threshold))
                return True, score, count
            return False, 0.0, count

    def check_distributed_brute_force(
        self, user_id: str, at_time: datetime | None = None
    ) -> tuple[bool, float, int]:
        """Check if a single user account is being brute-forced from multiple IPs."""
        ref_time = at_time or datetime.now(UTC)
        cutoff = ref_time - timedelta(seconds=self.window_seconds)

        with self._lock:
            q = self._user_events.get(user_id)
            if not q:
                return False, 0.0, 0
            self._prune_deque(q, cutoff)

            failed_ips = {
                e.ip_address
                for e in q
                if e.outcome in (LoginOutcome.FAILED_CREDENTIALS, LoginOutcome.FAILED_MFA)
            }
            count = len(failed_ips)
            if count >= self.distributed_user_ip_threshold:
                score = min(1.0, 0.5 + 0.1 * (count - self.distributed_user_ip_threshold))
                return True, score, count
            return False, 0.0, count

    def check_subnet_burst(
        self, ip_address: str, at_time: datetime | None = None
    ) -> tuple[bool, float, int]:
        """Check if a subnet (/24 or /48) is performing a distributed burst attack."""
        ref_time = at_time or datetime.now(UTC)
        cutoff = ref_time - timedelta(seconds=self.window_seconds)
        subnet = self._extract_subnet(ip_address)

        with self._lock:
            q = self._subnet_events.get(subnet)
            if not q:
                return False, 0.0, 0
            self._prune_deque(q, cutoff)

            failed_users = {
                e.user_id
                for e in q
                if e.outcome in (LoginOutcome.FAILED_CREDENTIALS, LoginOutcome.FAILED_MFA)
            }
            count = len(failed_users)
            if count >= self.subnet_spray_threshold:
                score = min(1.0, 0.4 + 0.1 * (count - self.subnet_spray_threshold))
                return True, score, count
            return False, 0.0, count

    def evaluate_stuffing_risk(self, event: LoginEvent) -> tuple[float, list[AtoRiskFactor]]:
        """Evaluate credential stuffing risk for an event and return triggered factors."""
        factors: list[AtoRiskFactor] = []
        is_spray, spray_score, _ = self.check_ip_spraying(event.ip_address, event.timestamp)
        if is_spray:
            factors.append(AtoRiskFactor.PASSWORD_SPRAYING)

        is_bf, bf_score, _ = self.check_distributed_brute_force(event.user_id, event.timestamp)
        if is_bf:
            factors.append(AtoRiskFactor.CREDENTIAL_STUFFING_BURST)

        is_sub, sub_score, _ = self.check_subnet_burst(event.ip_address, event.timestamp)
        if is_sub and AtoRiskFactor.CREDENTIAL_STUFFING_BURST not in factors:
            factors.append(AtoRiskFactor.CREDENTIAL_STUFFING_BURST)

        max_score = max(spray_score, bf_score, sub_score)
        return max_score, factors
