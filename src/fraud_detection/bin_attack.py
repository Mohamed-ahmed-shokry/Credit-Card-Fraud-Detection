"""Card-testing and BIN enumeration attack defense engine.

Detects automated payment-gateway probing, sequential PAN/BIN enumeration,
CVV/expiry brute-force sweeps, micro-authorization velocity spikes, and
distributed botnet BIN attacks across merchants and IP/device clusters.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

__all__ = [
    "PROBING_RESPONSE_CODES",
    "AuthResponseCode",
    "BinAttackAction",
    "BinAttackAnomalyType",
    "BinAttackAssessment",
    "BinAttackRiskLevel",
    "BinAttackStore",
    "BinAuthAttempt",
    "BinPrefixProfile",
    "EnumerationAnalyzer",
    "EnumerationMetrics",
    "VelocityDeclineAnalyzer",
    "VelocityDeclineMetrics",
]


class BinAttackRiskLevel(StrEnum):
    """Risk severity tier for card-testing and BIN enumeration attacks."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BinAttackAction(StrEnum):
    """Recommended gateway defense action for BIN attack assessments."""

    ALLOW = "allow"
    CHALLENGE_3DS = "challenge_3ds"
    RATE_LIMIT_MERCHANT_IP = "rate_limit_merchant_ip"
    TEMP_BLOCK_BIN_PREFIX = "temp_block_bin_prefix"
    BLOCK_GATEWAY_SESSION = "block_gateway_session"


class BinAttackAnomalyType(StrEnum):
    """Specific behavioral indicators of card-testing and BIN attacks."""

    SEQUENTIAL_PAN_ENUMERATION = "sequential_pan_enumeration"
    CVV_EXPIRY_PROBING = "cvv_expiry_probing"
    MICRO_AUTH_VELOCITY = "micro_auth_velocity"
    HIGH_DECLINE_RATIO = "high_decline_ratio"
    DISTRIBUTED_BIN_SWEEP = "distributed_bin_sweep"
    MULTI_CARD_IP_BURST = "multi_card_ip_burst"


class AuthResponseCode(StrEnum):
    """Normalized issuer/gateway authorization response codes."""

    APPROVED = "approved"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    DO_NOT_HONOR = "do_not_honor"
    CVV_MISMATCH = "cvv_mismatch"
    INVALID_EXPIRY = "invalid_expiry"
    EXPIRED_CARD = "expired_card"
    INVALID_PAN = "invalid_pan"
    PICKUP_CARD = "pickup_card"


PROBING_RESPONSE_CODES: frozenset[AuthResponseCode] = frozenset(
    {
        AuthResponseCode.CVV_MISMATCH,
        AuthResponseCode.INVALID_EXPIRY,
        AuthResponseCode.EXPIRED_CARD,
        AuthResponseCode.INVALID_PAN,
    }
)


def _coerce_utc_datetime(value: datetime | str) -> datetime:
    """Parse an ISO-8601 string or normalize a datetime to UTC."""
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class BinAuthAttempt:
    """Single card authorization attempt tracked for BIN attack / card-testing defense."""

    attempt_id: str
    bin_prefix: str
    last4: str
    amount: float
    response_code: AuthResponseCode
    timestamp: datetime
    merchant_id: str = "default_merchant"
    ip_address: str = ""
    device_id: str = ""
    card_fingerprint: str = ""
    currency: str = "USD"
    tenant_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.attempt_id:
            raise ValueError("attempt_id must be non-empty.")
        if not (6 <= len(self.bin_prefix) <= 8 and self.bin_prefix.isdigit()):
            raise ValueError("bin_prefix must be 6 to 8 digits.")
        if not (len(self.last4) == 4 and self.last4.isdigit()):
            raise ValueError("last4 must be 4 digits.")
        if self.amount < 0:
            raise ValueError("amount must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "bin_prefix": self.bin_prefix,
            "last4": self.last4,
            "amount": float(self.amount),
            "response_code": self.response_code.value,
            "timestamp": self.timestamp.astimezone(UTC).isoformat(),
            "merchant_id": self.merchant_id,
            "ip_address": self.ip_address,
            "device_id": self.device_id,
            "card_fingerprint": self.card_fingerprint,
            "currency": self.currency,
            "tenant_id": self.tenant_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BinAuthAttempt:
        return cls(
            attempt_id=str(payload["attempt_id"]),
            bin_prefix=str(payload["bin_prefix"]),
            last4=str(payload["last4"]),
            amount=float(payload["amount"]),
            response_code=AuthResponseCode(payload["response_code"]),
            timestamp=_coerce_utc_datetime(payload["timestamp"]),
            merchant_id=str(payload.get("merchant_id", "default_merchant")),
            ip_address=str(payload.get("ip_address", "")),
            device_id=str(payload.get("device_id", "")),
            card_fingerprint=str(payload.get("card_fingerprint", "")),
            currency=str(payload.get("currency", "USD")),
            tenant_id=str(payload.get("tenant_id", "default")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class BinPrefixProfile:
    """Mutable surveillance profile for a BIN prefix."""

    bin_prefix: str
    tenant_id: str = "default"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    attempts: list[BinAuthAttempt] = field(default_factory=list)
    risk_level: BinAttackRiskLevel = BinAttackRiskLevel.LOW
    last_action: BinAttackAction = BinAttackAction.ALLOW
    last_assessed_at: datetime | None = None
    last_assessment_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bin_prefix": self.bin_prefix,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.astimezone(UTC).isoformat(),
            "attempts": [a.to_dict() for a in self.attempts],
            "risk_level": self.risk_level.value,
            "last_action": self.last_action.value,
            "last_assessed_at": (
                self.last_assessed_at.astimezone(UTC).isoformat()
                if self.last_assessed_at is not None
                else None
            ),
            "last_assessment_id": self.last_assessment_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BinPrefixProfile:
        last_raw = payload.get("last_assessed_at")
        return cls(
            bin_prefix=str(payload["bin_prefix"]),
            tenant_id=str(payload.get("tenant_id", "default")),
            created_at=_coerce_utc_datetime(payload["created_at"]),
            attempts=[BinAuthAttempt.from_dict(a) for a in payload.get("attempts", [])],
            risk_level=BinAttackRiskLevel(payload.get("risk_level", BinAttackRiskLevel.LOW.value)),
            last_action=BinAttackAction(payload.get("last_action", BinAttackAction.ALLOW.value)),
            last_assessed_at=_coerce_utc_datetime(last_raw) if last_raw is not None else None,
            last_assessment_id=payload.get("last_assessment_id"),
        )


@dataclass(frozen=True, slots=True)
class BinAttackAssessment:
    """Risk assessment output produced by BinAttackDefenseEngine."""

    assessment_id: str
    bin_prefix: str
    tenant_id: str
    risk_score: float
    risk_level: BinAttackRiskLevel
    action: BinAttackAction
    anomalies: tuple[BinAttackAnomalyType, ...]
    total_attempts: int
    decline_ratio: float
    micro_auth_ratio: float
    unique_ips: int
    unique_cards: int
    sequential_run_length: int
    probing_count: int
    case_id: str | None = None
    reasons: tuple[str, ...] = ()
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "bin_prefix": self.bin_prefix,
            "tenant_id": self.tenant_id,
            "risk_score": round(float(self.risk_score), 6),
            "risk_level": self.risk_level.value,
            "action": self.action.value,
            "anomalies": [a.value for a in self.anomalies],
            "total_attempts": self.total_attempts,
            "decline_ratio": round(float(self.decline_ratio), 6),
            "micro_auth_ratio": round(float(self.micro_auth_ratio), 6),
            "unique_ips": self.unique_ips,
            "unique_cards": self.unique_cards,
            "sequential_run_length": self.sequential_run_length,
            "probing_count": self.probing_count,
            "case_id": self.case_id,
            "reasons": list(self.reasons),
            "evaluated_at": self.evaluated_at.astimezone(UTC).isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BinAttackAssessment:
        return cls(
            assessment_id=str(payload["assessment_id"]),
            bin_prefix=str(payload["bin_prefix"]),
            tenant_id=str(payload.get("tenant_id", "default")),
            risk_score=float(payload["risk_score"]),
            risk_level=BinAttackRiskLevel(payload["risk_level"]),
            action=BinAttackAction(payload["action"]),
            anomalies=tuple(BinAttackAnomalyType(a) for a in payload.get("anomalies", [])),
            total_attempts=int(payload["total_attempts"]),
            decline_ratio=float(payload["decline_ratio"]),
            micro_auth_ratio=float(payload["micro_auth_ratio"]),
            unique_ips=int(payload["unique_ips"]),
            unique_cards=int(payload["unique_cards"]),
            sequential_run_length=int(payload["sequential_run_length"]),
            probing_count=int(payload["probing_count"]),
            case_id=payload.get("case_id"),
            reasons=tuple(str(r) for r in payload.get("reasons", [])),
            evaluated_at=_coerce_utc_datetime(payload["evaluated_at"]),
        )


@dataclass(frozen=True, slots=True)
class EnumerationMetrics:
    """Computed sequential PAN enumeration and CVV/expiry probing metrics."""

    sequential_run_length: int
    probing_count: int
    unique_last4_count: int
    anomalies: tuple[BinAttackAnomalyType, ...]
    reasons: tuple[str, ...]


class EnumerationAnalyzer:
    """Detects sequential PAN/last4 enumeration and CVV/expiry brute-force probing."""

    def __init__(
        self,
        *,
        window_seconds: int = 600,
        sequential_run_threshold: int = 4,
        max_stride: int = 2,
        probing_threshold: int = 3,
    ) -> None:
        self.window_seconds = window_seconds
        self.sequential_run_threshold = sequential_run_threshold
        self.max_stride = max_stride
        self.probing_threshold = probing_threshold

    def analyze(
        self,
        attempts: list[BinAuthAttempt],
        *,
        now: datetime | None = None,
    ) -> EnumerationMetrics:
        if not attempts:
            return EnumerationMetrics(
                sequential_run_length=0,
                probing_count=0,
                unique_last4_count=0,
                anomalies=(),
                reasons=(),
            )
        ref_time = now or max(a.timestamp for a in attempts)
        cutoff = ref_time - timedelta(seconds=self.window_seconds)
        window_attempts = [a for a in attempts if a.timestamp >= cutoff]

        unique_last4 = sorted({int(a.last4) for a in window_attempts})
        best_run = 1 if unique_last4 else 0
        current_run = 1
        for idx in range(1, len(unique_last4)):
            if 1 <= (unique_last4[idx] - unique_last4[idx - 1]) <= self.max_stride:
                current_run += 1
                best_run = max(best_run, current_run)
            else:
                current_run = 1

        probing_count = sum(1 for a in window_attempts if a.response_code in PROBING_RESPONSE_CODES)

        anomalies: list[BinAttackAnomalyType] = []
        reasons: list[str] = []

        if best_run >= self.sequential_run_threshold:
            anomalies.append(BinAttackAnomalyType.SEQUENTIAL_PAN_ENUMERATION)
            reasons.append(f"Sequential last4 run of {best_run} within {self.window_seconds}s")
        if probing_count >= self.probing_threshold:
            anomalies.append(BinAttackAnomalyType.CVV_EXPIRY_PROBING)
            reasons.append(
                f"{probing_count} CVV/expiry/PAN probing declines within {self.window_seconds}s"
            )

        return EnumerationMetrics(
            sequential_run_length=best_run,
            probing_count=probing_count,
            unique_last4_count=len(unique_last4),
            anomalies=tuple(anomalies),
            reasons=tuple(reasons),
        )


@dataclass(frozen=True, slots=True)
class VelocityDeclineMetrics:
    """Computed micro-auth velocity, decline ratio, and IP/card sweep metrics."""

    total_attempts: int
    decline_ratio: float
    micro_auth_ratio: float
    unique_ips: int
    unique_cards: int
    max_cards_per_ip: int
    anomalies: tuple[BinAttackAnomalyType, ...]
    reasons: tuple[str, ...]


class VelocityDeclineAnalyzer:
    """Detects micro-auth velocity spikes, high decline ratios, and distributed BIN sweeps."""

    def __init__(
        self,
        *,
        window_seconds: int = 300,
        micro_auth_amount: float = 5.0,
        micro_auth_threshold: int = 4,
        min_attempts_for_decline: int = 4,
        decline_ratio_threshold: float = 0.70,
        distributed_ip_threshold: int = 4,
        multi_card_ip_threshold: int = 4,
    ) -> None:
        self.window_seconds = window_seconds
        self.micro_auth_amount = micro_auth_amount
        self.micro_auth_threshold = micro_auth_threshold
        self.min_attempts_for_decline = min_attempts_for_decline
        self.decline_ratio_threshold = decline_ratio_threshold
        self.distributed_ip_threshold = distributed_ip_threshold
        self.multi_card_ip_threshold = multi_card_ip_threshold

    def analyze(
        self,
        attempts: list[BinAuthAttempt],
        *,
        now: datetime | None = None,
    ) -> VelocityDeclineMetrics:
        if not attempts:
            return VelocityDeclineMetrics(
                total_attempts=0,
                decline_ratio=0.0,
                micro_auth_ratio=0.0,
                unique_ips=0,
                unique_cards=0,
                max_cards_per_ip=0,
                anomalies=(),
                reasons=(),
            )
        ref_time = now or max(a.timestamp for a in attempts)
        cutoff = ref_time - timedelta(seconds=self.window_seconds)
        window_attempts = [a for a in attempts if a.timestamp >= cutoff]
        if not window_attempts:
            return VelocityDeclineMetrics(
                total_attempts=0,
                decline_ratio=0.0,
                micro_auth_ratio=0.0,
                unique_ips=0,
                unique_cards=0,
                max_cards_per_ip=0,
                anomalies=(),
                reasons=(),
            )

        total = len(window_attempts)
        declines = sum(1 for a in window_attempts if a.response_code != AuthResponseCode.APPROVED)
        micro_auths = sum(1 for a in window_attempts if a.amount <= self.micro_auth_amount)
        decline_ratio = declines / total
        micro_auth_ratio = micro_auths / total

        cards_by_ip: dict[str, set[str]] = defaultdict(set)
        all_cards: set[str] = set()
        for att in window_attempts:
            card_key = att.card_fingerprint or f"{att.bin_prefix}:{att.last4}"
            all_cards.add(card_key)
            if att.ip_address:
                cards_by_ip[att.ip_address].add(card_key)

        unique_ips = len(cards_by_ip)
        unique_cards = len(all_cards)
        max_cards_per_ip = max((len(c) for c in cards_by_ip.values()), default=0)

        anomalies: list[BinAttackAnomalyType] = []
        reasons: list[str] = []

        if micro_auths >= self.micro_auth_threshold:
            anomalies.append(BinAttackAnomalyType.MICRO_AUTH_VELOCITY)
            reasons.append(
                f"{micro_auths} micro-authorizations <= ${self.micro_auth_amount:.2f} "
                f"within {self.window_seconds}s"
            )
        if total >= self.min_attempts_for_decline and decline_ratio >= self.decline_ratio_threshold:
            anomalies.append(BinAttackAnomalyType.HIGH_DECLINE_RATIO)
            reasons.append(f"Decline ratio {decline_ratio:.2f} across {total} attempts")
        if (
            unique_ips >= self.distributed_ip_threshold
            and unique_cards >= self.distributed_ip_threshold
        ):
            anomalies.append(BinAttackAnomalyType.DISTRIBUTED_BIN_SWEEP)
            reasons.append(
                f"Distributed BIN sweep across {unique_ips} IPs and {unique_cards} cards"
            )
        if max_cards_per_ip >= self.multi_card_ip_threshold:
            anomalies.append(BinAttackAnomalyType.MULTI_CARD_IP_BURST)
            reasons.append(f"Single IP probed {max_cards_per_ip} distinct cards")

        return VelocityDeclineMetrics(
            total_attempts=total,
            decline_ratio=decline_ratio,
            micro_auth_ratio=micro_auth_ratio,
            unique_ips=unique_ips,
            unique_cards=unique_cards,
            max_cards_per_ip=max_cards_per_ip,
            anomalies=tuple(anomalies),
            reasons=tuple(reasons),
        )


class BinAttackStore:
    """Thread-safe in-memory store for BIN prefix profiles and assessments."""

    def __init__(
        self,
        *,
        max_bins: int = 10_000,
        max_assessments: int = 20_000,
        max_attempts_per_bin: int = 500,
    ) -> None:
        self._lock = threading.RLock()
        self._max_bins = max_bins
        self._max_assessments = max_assessments
        self._max_attempts_per_bin = max_attempts_per_bin
        self._profiles: OrderedDict[str, BinPrefixProfile] = OrderedDict()
        self._assessments: OrderedDict[str, BinAttackAssessment] = OrderedDict()
        self._tenant_index: dict[str, set[str]] = {}

    def record_attempt(self, attempt: BinAuthAttempt) -> BinPrefixProfile:
        with self._lock:
            profile = self._profiles.get(attempt.bin_prefix)
            if profile is None:
                while len(self._profiles) >= self._max_bins:
                    evicted_bin, evicted = self._profiles.popitem(last=False)
                    self._tenant_index.get(evicted.tenant_id, set()).discard(evicted_bin)
                profile = BinPrefixProfile(
                    bin_prefix=attempt.bin_prefix,
                    tenant_id=attempt.tenant_id,
                    created_at=attempt.timestamp,
                )
            else:
                self._profiles.move_to_end(attempt.bin_prefix)

            profile.attempts.append(attempt)
            if len(profile.attempts) > self._max_attempts_per_bin:
                profile.attempts = profile.attempts[-self._max_attempts_per_bin :]

            self._profiles[attempt.bin_prefix] = profile
            self._tenant_index.setdefault(attempt.tenant_id, set()).add(attempt.bin_prefix)
            return profile

    def get_profile(self, bin_prefix: str) -> BinPrefixProfile | None:
        with self._lock:
            return self._profiles.get(bin_prefix)

    def save_assessment(self, assessment: BinAttackAssessment) -> None:
        with self._lock:
            while len(self._assessments) >= self._max_assessments:
                self._assessments.popitem(last=False)
            self._assessments[assessment.assessment_id] = assessment

    def get_assessment(self, assessment_id: str) -> BinAttackAssessment | None:
        with self._lock:
            return self._assessments.get(assessment_id)

    def list_assessments(
        self,
        *,
        bin_prefix: str | None = None,
        risk_level: BinAttackRiskLevel | None = None,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[BinAttackAssessment]:
        with self._lock:
            items = list(self._assessments.values())
            items.reverse()
            filtered = [
                item
                for item in items
                if (bin_prefix is None or item.bin_prefix == bin_prefix)
                and (risk_level is None or item.risk_level == risk_level)
                and (tenant_id is None or item.tenant_id == tenant_id)
            ]
            return filtered[:limit]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            risk_counts: dict[str, int] = {level.value: 0 for level in BinAttackRiskLevel}
            for profile in self._profiles.values():
                risk_counts[profile.risk_level.value] += 1
            return {
                "total_bins": len(self._profiles),
                "total_assessments": len(self._assessments),
                "risk_distribution": risk_counts,
            }

    def export_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "profiles": [p.to_dict() for p in self._profiles.values()],
                "assessments": [a.to_dict() for a in self._assessments.values()],
            }

    def load_state(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._profiles.clear()
            self._assessments.clear()
            self._tenant_index.clear()
            for p_raw in payload.get("profiles", []):
                profile = BinPrefixProfile.from_dict(p_raw)
                self._profiles[profile.bin_prefix] = profile
                self._tenant_index.setdefault(profile.tenant_id, set()).add(profile.bin_prefix)
            for a_raw in payload.get("assessments", []):
                assessment = BinAttackAssessment.from_dict(a_raw)
                self._assessments[assessment.assessment_id] = assessment
