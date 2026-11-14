"""Card-testing and BIN enumeration attack defense engine.

Detects automated payment-gateway probing, sequential PAN/BIN enumeration,
CVV/expiry brute-force sweeps, micro-authorization velocity spikes, and
distributed botnet BIN attacks across merchants and IP/device clusters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

__all__ = [
    "PROBING_RESPONSE_CODES",
    "AuthResponseCode",
    "BinAttackAction",
    "BinAttackAnomalyType",
    "BinAttackAssessment",
    "BinAttackRiskLevel",
    "BinAuthAttempt",
    "BinPrefixProfile",
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
