"""Account Takeover (ATO) Detection, Credential Stuffing Defense & Identity Session Anomaly Engine.

Provides domain models, sliding-window credential stuffing detection, user account
baselines with mutation decay penalties, multi-factor ATO risk evaluation, and
thread-safe store with automated defense orchestration (challenge/quarantine).
"""

from __future__ import annotations

import ipaddress
import math
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


@dataclass
class AccountProfile:
    """Historical behavior baseline, known devices/locations, and quarantine state."""

    user_id: str
    tenant_id: str = "default"
    known_devices: set[str] = field(default_factory=set)
    known_locations: list[tuple[float, float]] = field(default_factory=list)
    last_known_ip: str | None = None
    last_known_location: tuple[float, float] | None = None
    last_active_at: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    mutations: list[AccountMutationRecord] = field(default_factory=list)
    is_quarantined: bool = False
    quarantine_reason: str | None = None
    quarantined_at: datetime | None = None
    release_reason: str | None = None
    released_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize account profile to dictionary."""
        return {
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "known_devices": sorted(self.known_devices),
            "known_locations": [list(loc) for loc in self.known_locations],
            "last_known_ip": self.last_known_ip,
            "last_known_location": (
                list(self.last_known_location) if self.last_known_location else None
            ),
            "last_active_at": self.last_active_at.isoformat() if self.last_active_at else None,
            "created_at": self.created_at.isoformat(),
            "mutations": [m.to_dict() for m in self.mutations],
            "is_quarantined": self.is_quarantined,
            "quarantine_reason": self.quarantine_reason,
            "quarantined_at": self.quarantined_at.isoformat() if self.quarantined_at else None,
            "release_reason": self.release_reason,
            "released_at": self.released_at.isoformat() if self.released_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AccountProfile:
        """Instantiate account profile from dictionary."""

        def parse_ts(val: Any) -> datetime | None:
            if not val:
                return None
            if isinstance(val, datetime):
                return val if val.tzinfo is not None else val.replace(tzinfo=UTC)
            dt = datetime.fromisoformat(str(val))
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)

        devices = set(data.get("known_devices", []))
        locs = [(float(c[0]), float(c[1])) for c in data.get("known_locations", [])]

        last_loc_raw = data.get("last_known_location")
        last_loc = (
            (float(last_loc_raw[0]), float(last_loc_raw[1]))
            if last_loc_raw and len(last_loc_raw) >= 2
            else None
        )

        mutations = [
            AccountMutationRecord.from_dict(m)
            for m in data.get("mutations", [])
            if isinstance(m, dict)
        ]

        created_at = parse_ts(data.get("created_at")) or datetime.now(UTC)

        return cls(
            user_id=str(data["user_id"]),
            tenant_id=str(data.get("tenant_id", "default")),
            known_devices=devices,
            known_locations=locs,
            last_known_ip=data.get("last_known_ip"),
            last_known_location=last_loc,
            last_active_at=parse_ts(data.get("last_active_at")),
            created_at=created_at,
            mutations=mutations,
            is_quarantined=bool(data.get("is_quarantined", False)),
            quarantine_reason=data.get("quarantine_reason"),
            quarantined_at=parse_ts(data.get("quarantined_at")),
            release_reason=data.get("release_reason"),
            released_at=parse_ts(data.get("released_at")),
        )


class AccountProfileTracker:
    """Maintains user behavioral baselines, known devices/locations, and mutation decay."""

    def __init__(
        self,
        mutation_risk_window_hours: float = 72.0,
        dormancy_threshold_days: int = 90,
    ) -> None:
        self.mutation_risk_window_hours = mutation_risk_window_hours
        self.dormancy_threshold_days = dormancy_threshold_days
        self._lock = threading.RLock()
        self._profiles: dict[str, AccountProfile] = {}

    def get_or_create_profile(self, user_id: str, tenant_id: str = "default") -> AccountProfile:
        """Get or initialize user account profile."""
        with self._lock:
            if user_id not in self._profiles:
                self._profiles[user_id] = AccountProfile(user_id=user_id, tenant_id=tenant_id)
            return self._profiles[user_id]

    def record_mutation(
        self,
        user_id: str,
        mutation: AccountMutationRecord,
        tenant_id: str = "default",
    ) -> None:
        """Log a sensitive profile mutation for the user."""
        with self._lock:
            profile = self.get_or_create_profile(user_id, tenant_id)
            profile.mutations.append(mutation)

    def record_login_success(self, event: LoginEvent) -> None:
        """Update user baseline on successful authentication."""
        with self._lock:
            profile = self.get_or_create_profile(event.user_id, event.tenant_id)
            if event.device_fingerprint:
                profile.known_devices.add(event.device_fingerprint)
            profile.last_known_ip = event.ip_address
            if event.latitude is not None and event.longitude is not None:
                loc = (event.latitude, event.longitude)
                profile.last_known_location = loc
                if loc not in profile.known_locations:
                    profile.known_locations.append(loc)
            profile.last_active_at = event.timestamp

    def is_device_recognized(self, user_id: str, device_fingerprint: str | None) -> bool:
        """Check if device fingerprint matches user's known devices."""
        if not device_fingerprint:
            return False
        with self._lock:
            profile = self._profiles.get(user_id)
            if not profile:
                return False
            return device_fingerprint in profile.known_devices

    def calculate_mutation_penalty(
        self, user_id: str, at_time: datetime | None = None
    ) -> tuple[float, float | None, list[AtoRiskFactor]]:
        """Calculate time-decayed penalty and triggered factors for recent sensitive mutations."""
        ref_time = at_time or datetime.now(UTC)
        with self._lock:
            profile = self._profiles.get(user_id)
            if not profile or not profile.mutations:
                return 0.0, None, []

            # Find the most recent mutation
            most_recent: AccountMutationRecord | None = None
            min_hours: float | None = None

            for m in profile.mutations:
                diff_sec = (ref_time - m.timestamp).total_seconds()
                diff_hrs = max(0.0, diff_sec / 3600.0)
                if diff_hrs <= self.mutation_risk_window_hours and (
                    min_hours is None or diff_hrs < min_hours
                ):
                    min_hours = diff_hrs
                    most_recent = m

            if most_recent is None or min_hours is None:
                return 0.0, None, []

            factors: list[AtoRiskFactor] = []
            if most_recent.mutation_type in (
                AccountMutationType.PASSWORD_CHANGE,
                AccountMutationType.EMAIL_CHANGE,
                AccountMutationType.PHONE_CHANGE,
            ):
                factors.append(AtoRiskFactor.RECENT_CREDENTIAL_CHANGE)
            elif most_recent.mutation_type == AccountMutationType.MFA_RESET:
                factors.append(AtoRiskFactor.RECENT_MFA_RESET)

            # Time decay step function
            if min_hours <= 1.0:
                penalty = 0.40
            elif min_hours <= 24.0:
                penalty = 0.20
            else:
                penalty = 0.10

            return penalty, min_hours, factors

    def is_dormant_reactivation(
        self, user_id: str, current_time: datetime | None = None
    ) -> tuple[bool, float]:
        """Detect sudden activity on accounts inactive for longer than threshold."""
        ref_time = current_time or datetime.now(UTC)
        with self._lock:
            profile = self._profiles.get(user_id)
            if not profile or profile.last_active_at is None:
                return False, 0.0

            days_inactive = (ref_time - profile.last_active_at).total_seconds() / 86400.0
            if days_inactive >= self.dormancy_threshold_days:
                return True, 0.15
            return False, 0.0

    def quarantine_account(self, user_id: str, reason: str, tenant_id: str = "default") -> None:
        """Place an account under security quarantine."""
        with self._lock:
            profile = self.get_or_create_profile(user_id, tenant_id)
            profile.is_quarantined = True
            profile.quarantine_reason = reason
            profile.quarantined_at = datetime.now(UTC)
            profile.released_at = None

    def release_account(self, user_id: str, reason: str, tenant_id: str = "default") -> None:
        """Release an account from quarantine."""
        with self._lock:
            profile = self.get_or_create_profile(user_id, tenant_id)
            profile.is_quarantined = False
            profile.quarantine_reason = None
            profile.release_reason = reason
            profile.released_at = datetime.now(UTC)

    def is_quarantined(self, user_id: str) -> bool:
        """Check if an account is currently quarantined."""
        with self._lock:
            profile = self._profiles.get(user_id)
            return profile.is_quarantined if profile else False


def calculate_haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in kilometers."""
    radius_earth_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius_earth_km * c


class AtoStore:
    """Thread-safe storage for ATO assessments, telemetry events, and audit logs."""

    def __init__(self, max_capacity: int = 10_000) -> None:
        self.max_capacity = max_capacity
        self._lock = threading.RLock()
        self._assessments: dict[str, AtoRiskAssessment] = {}
        self._user_assessments: dict[str, list[str]] = defaultdict(list)
        self._events: deque[LoginEvent] = deque(maxlen=max_capacity)

    def record_assessment(self, assessment: AtoRiskAssessment) -> None:
        """Store an assessment and index it by user."""
        with self._lock:
            if len(self._assessments) >= self.max_capacity:
                # Evict oldest entry
                oldest_id = next(iter(self._assessments))
                del self._assessments[oldest_id]

            self._assessments[assessment.assessment_id] = assessment
            self._user_assessments[assessment.user_id].append(assessment.assessment_id)

    def get_assessment(self, assessment_id: str) -> AtoRiskAssessment | None:
        """Retrieve an assessment by unique identifier."""
        with self._lock:
            return self._assessments.get(assessment_id)

    def get_user_assessments(self, user_id: str, limit: int = 50) -> list[AtoRiskAssessment]:
        """Retrieve recent assessments for a user."""
        with self._lock:
            ids = self._user_assessments.get(user_id, [])
            selected_ids = ids[-limit:] if limit > 0 else ids
            return [
                self._assessments[aid] for aid in reversed(selected_ids) if aid in self._assessments
            ]

    def list_assessments(
        self, tenant_id: str | None = None, limit: int = 100
    ) -> list[AtoRiskAssessment]:
        """List assessments with optional tenant filter."""
        with self._lock:
            items = list(self._assessments.values())
            if tenant_id:
                items = [a for a in items if a.tenant_id == tenant_id]
            return items[-limit:] if limit > 0 else items

    def record_event(self, event: LoginEvent) -> None:
        """Store a telemetry event."""
        with self._lock:
            self._events.append(event)

    def list_events(self, user_id: str | None = None, limit: int = 100) -> list[LoginEvent]:
        """List recent events with optional user filter."""
        with self._lock:
            evs = list(self._events)
            if user_id:
                evs = [e for e in evs if e.user_id == user_id]
            return evs[-limit:] if limit > 0 else evs


class AtoEngine:
    """Core multi-factor Account Takeover evaluation and defense orchestration engine."""

    def __init__(
        self,
        stuffing_detector: CredentialStuffingDetector | None = None,
        profile_tracker: AccountProfileTracker | None = None,
        store: AtoStore | None = None,
        auto_quarantine: bool = True,
        challenge_threshold: float = 0.40,
        quarantine_threshold: float = 0.75,
        impossible_travel_speed_kmh: float = 800.0,
    ) -> None:
        self.stuffing_detector = stuffing_detector or CredentialStuffingDetector()
        self.profile_tracker = profile_tracker or AccountProfileTracker()
        self.store = store or AtoStore()
        self.auto_quarantine = auto_quarantine
        self.challenge_threshold = challenge_threshold
        self.quarantine_threshold = quarantine_threshold
        self.impossible_travel_speed_kmh = impossible_travel_speed_kmh
        self._lock = threading.RLock()

    def evaluate_attempt(
        self,
        event: LoginEvent,
        action_type: str = "login",
        transaction_amount: float | None = None,
    ) -> AtoRiskAssessment:
        """Evaluate an authentication attempt or transaction for account takeover indicators."""
        with self._lock:
            # Check quarantine status first
            if self.profile_tracker.is_quarantined(event.user_id):
                assessment = AtoRiskAssessment(
                    user_id=event.user_id,
                    tenant_id=event.tenant_id,
                    risk_score=1.0,
                    recommended_action=AtoAction.BLOCK,
                    factors_triggered=(AtoRiskFactor.ACCOUNT_QUARANTINED,),
                    score_breakdown={"quarantine": 1.0},
                    quarantined=True,
                    timestamp=event.timestamp,
                    details={"reason": "Account is actively quarantined for security breach"},
                )
                self.store.record_assessment(assessment)
                self.store.record_event(event)
                return assessment

            factors: list[AtoRiskFactor] = []
            score_breakdown: dict[str, float] = {}

            # 1. Device familiarity
            is_new_device = False
            device_score = 0.0
            if event.device_fingerprint:
                is_recognized = self.profile_tracker.is_device_recognized(
                    event.user_id, event.device_fingerprint
                )
                if not is_recognized:
                    is_new_device = True
                    device_score = 0.25
                    factors.append(AtoRiskFactor.NEW_DEVICE)
                    score_breakdown["device"] = device_score
            else:
                is_new_device = True
                device_score = 0.15
                score_breakdown["device"] = device_score

            # 2. Impossible travel velocity
            travel_score = 0.0
            calculated_speed_kmh: float | None = None
            profile = self.profile_tracker.get_or_create_profile(event.user_id, event.tenant_id)

            if (
                profile.last_known_location
                and profile.last_active_at
                and event.latitude is not None
                and event.longitude is not None
            ):
                dist_km = calculate_haversine_distance_km(
                    profile.last_known_location[0],
                    profile.last_known_location[1],
                    event.latitude,
                    event.longitude,
                )
                time_diff_hours = (
                    event.timestamp - profile.last_active_at
                ).total_seconds() / 3600.0

                if time_diff_hours > 0.0:
                    speed_kmh = dist_km / time_diff_hours
                    calculated_speed_kmh = speed_kmh
                    if speed_kmh >= self.impossible_travel_speed_kmh:
                        factors.append(AtoRiskFactor.IMPOSSIBLE_TRAVEL)
                        travel_score = min(
                            0.40, 0.25 + 0.0001 * (speed_kmh - self.impossible_travel_speed_kmh)
                        )
                        score_breakdown["travel"] = travel_score

            # 3. Sensitive profile mutation decay penalty
            mutation_penalty, min_mutation_hrs, mut_factors = (
                self.profile_tracker.calculate_mutation_penalty(event.user_id, event.timestamp)
            )
            if mutation_penalty > 0.0:
                score_breakdown["mutation"] = mutation_penalty
                factors.extend(mut_factors)

            # 4. Dormancy sudden reactivation
            is_dormant, dormant_penalty = self.profile_tracker.is_dormant_reactivation(
                event.user_id, event.timestamp
            )
            if is_dormant:
                score_breakdown["dormancy"] = dormant_penalty
                factors.append(AtoRiskFactor.DORMANT_ACCOUNT_REACTIVE)

            # 5. Credential stuffing / password spraying
            stuffing_score, stuffing_factors = self.stuffing_detector.evaluate_stuffing_risk(event)
            if stuffing_score > 0.0:
                score_breakdown["credential_stuffing"] = stuffing_score
                for sf in stuffing_factors:
                    if sf not in factors:
                        factors.append(sf)

            # Aggregate composite risk score [0.0, 1.0]
            total_score = sum(score_breakdown.values())
            calibrated_score = min(1.0, max(0.0, total_score))

            # Determine policy action
            if calibrated_score >= self.quarantine_threshold:
                action = AtoAction.QUARANTINE_ACCOUNT
            elif calibrated_score >= self.challenge_threshold:
                action = AtoAction.CHALLENGE
            else:
                action = AtoAction.ALLOW

            quarantined = False
            if action == AtoAction.QUARANTINE_ACCOUNT and self.auto_quarantine:
                self.profile_tracker.quarantine_account(
                    event.user_id,
                    reason=f"Automated quarantine triggered (score {calibrated_score:.4f})",
                    tenant_id=event.tenant_id,
                )
                quarantined = True

            # Record event in stuffing detector
            self.stuffing_detector.record_attempt(event)

            # If outcome is success and action is allow, update profile baseline
            if event.outcome == LoginOutcome.SUCCESS and action == AtoAction.ALLOW:
                self.profile_tracker.record_login_success(event)

            assessment = AtoRiskAssessment(
                user_id=event.user_id,
                tenant_id=event.tenant_id,
                risk_score=calibrated_score,
                recommended_action=action,
                factors_triggered=tuple(factors),
                score_breakdown=score_breakdown,
                impossible_travel_kmh=calculated_speed_kmh,
                is_new_device=is_new_device,
                hours_since_last_mutation=min_mutation_hrs,
                quarantined=quarantined,
                timestamp=event.timestamp,
                details={
                    "action_type": action_type,
                    "transaction_amount": transaction_amount,
                },
            )

            self.store.record_assessment(assessment)
            self.store.record_event(event)
            return assessment
