"""First-party fraud, refund abuse, and friendly fraud detection engine.

Detects legitimate cardholders and account owners exploiting merchant refund,
chargeback, and promotional policies via serial friendly-fraud disputes,
high-ratio wardrobing/return abuse, Item-Not-Received (INR) claim bursts,
and multi-account promo/referral cycling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

__all__ = [
    "WARDROBING_CATEGORIES",
    "ClaimOutcome",
    "FirstPartyAccountProfile",
    "FirstPartyAction",
    "FirstPartyAnomalyType",
    "FirstPartyAssessment",
    "FirstPartyEventRecord",
    "FirstPartyRiskLevel",
    "OrderClaimType",
]


class FirstPartyRiskLevel(StrEnum):
    """Risk severity tier for first-party fraud and refund abuse."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FirstPartyAction(StrEnum):
    """Recommended policy action for first-party abuse assessments."""

    ALLOW = "allow"
    REQUIRE_SIGNATURE_DELIVERY = "require_signature_delivery"
    RESTRICT_INSTANT_REFUND = "restrict_instant_refund"
    DENY_REFUND_MANUAL_REVIEW = "deny_refund_manual_review"
    REVOKE_PRIVILEGES_AND_BLOCK = "revoke_privileges_and_block"


class FirstPartyAnomalyType(StrEnum):
    """Specific behavioral indicators of first-party abuse."""

    SERIAL_FRIENDLY_FRAUD = "serial_friendly_fraud"
    HIGH_RETURN_RATIO = "high_return_ratio"
    WARDROBING_PATTERN = "wardrobing_pattern"
    INR_CLAIM_BURST = "inr_claim_burst"
    HIGH_VALUE_REFUND_SKEW = "high_value_refund_skew"
    PROMO_BONUS_ABUSE = "promo_bonus_abuse"
    LINKED_ACCOUNT_CYCLING = "linked_account_cycling"
    PRE_DISPUTE_VELOCITY = "pre_dispute_velocity"


class OrderClaimType(StrEnum):
    """Lifecycle event types tracked for first-party abuse analysis."""

    PURCHASE = "purchase"
    REFUND_RETURN = "refund_return"
    INR_CLAIM = "inr_claim"
    FRIENDLY_DISPUTE = "friendly_dispute"
    PROMO_REDEMPTION = "promo_redemption"


class ClaimOutcome(StrEnum):
    """Resolution status of a refund, INR claim, or dispute."""

    APPROVED = "approved"
    PENDING = "pending"
    DENIED = "denied"
    REVERSED = "reversed"


WARDROBING_CATEGORIES: frozenset[str] = frozenset(
    {
        "apparel",
        "luxury_fashion",
        "electronics",
        "cameras",
        "jewelry",
        "footwear",
    }
)


def _coerce_utc_datetime(value: datetime | str) -> datetime:
    """Parse an ISO-8601 string or normalize a datetime to UTC."""
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class FirstPartyEventRecord:
    """Single order, refund, INR claim, dispute, or promo event for an account."""

    event_id: str
    account_id: str
    claim_type: OrderClaimType
    amount: float
    timestamp: datetime
    order_id: str = ""
    currency: str = "USD"
    merchant_category: str = "general"
    outcome: ClaimOutcome = ClaimOutcome.APPROVED
    promo_code: str = ""
    shipping_address_hash: str = ""
    payment_instrument_hash: str = ""
    device_id: str = ""
    avs_cvv_matched: bool = False
    three_ds_authenticated: bool = False
    delivery_confirmed: bool = False
    signature_captured: bool = False
    tenant_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must be non-empty.")
        if not self.account_id:
            raise ValueError("account_id must be non-empty.")
        if self.amount < 0:
            raise ValueError("amount must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "account_id": self.account_id,
            "claim_type": self.claim_type.value,
            "amount": float(self.amount),
            "timestamp": self.timestamp.astimezone(UTC).isoformat(),
            "order_id": self.order_id,
            "currency": self.currency,
            "merchant_category": self.merchant_category,
            "outcome": self.outcome.value,
            "promo_code": self.promo_code,
            "shipping_address_hash": self.shipping_address_hash,
            "payment_instrument_hash": self.payment_instrument_hash,
            "device_id": self.device_id,
            "avs_cvv_matched": self.avs_cvv_matched,
            "three_ds_authenticated": self.three_ds_authenticated,
            "delivery_confirmed": self.delivery_confirmed,
            "signature_captured": self.signature_captured,
            "tenant_id": self.tenant_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FirstPartyEventRecord:
        return cls(
            event_id=str(payload["event_id"]),
            account_id=str(payload["account_id"]),
            claim_type=OrderClaimType(payload["claim_type"]),
            amount=float(payload["amount"]),
            timestamp=_coerce_utc_datetime(payload["timestamp"]),
            order_id=str(payload.get("order_id", "")),
            currency=str(payload.get("currency", "USD")),
            merchant_category=str(payload.get("merchant_category", "general")),
            outcome=ClaimOutcome(payload.get("outcome", ClaimOutcome.APPROVED.value)),
            promo_code=str(payload.get("promo_code", "")),
            shipping_address_hash=str(payload.get("shipping_address_hash", "")),
            payment_instrument_hash=str(payload.get("payment_instrument_hash", "")),
            device_id=str(payload.get("device_id", "")),
            avs_cvv_matched=bool(payload.get("avs_cvv_matched", False)),
            three_ds_authenticated=bool(payload.get("three_ds_authenticated", False)),
            delivery_confirmed=bool(payload.get("delivery_confirmed", False)),
            signature_captured=bool(payload.get("signature_captured", False)),
            tenant_id=str(payload.get("tenant_id", "default")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class FirstPartyAccountProfile:
    """Mutable behavioral state for an account under first-party fraud surveillance."""

    account_id: str
    tenant_id: str = "default"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    events: list[FirstPartyEventRecord] = field(default_factory=list)
    linked_accounts: set[str] = field(default_factory=set)
    risk_level: FirstPartyRiskLevel = FirstPartyRiskLevel.LOW
    last_action: FirstPartyAction = FirstPartyAction.ALLOW
    last_assessed_at: datetime | None = None
    last_assessment_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.astimezone(UTC).isoformat(),
            "events": [e.to_dict() for e in self.events],
            "linked_accounts": sorted(self.linked_accounts),
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
    def from_dict(cls, payload: dict[str, Any]) -> FirstPartyAccountProfile:
        last_assessed_raw = payload.get("last_assessed_at")
        return cls(
            account_id=str(payload["account_id"]),
            tenant_id=str(payload.get("tenant_id", "default")),
            created_at=_coerce_utc_datetime(payload["created_at"]),
            events=[FirstPartyEventRecord.from_dict(item) for item in payload.get("events", [])],
            linked_accounts={str(a) for a in payload.get("linked_accounts", [])},
            risk_level=FirstPartyRiskLevel(
                payload.get("risk_level", FirstPartyRiskLevel.LOW.value)
            ),
            last_action=FirstPartyAction(payload.get("last_action", FirstPartyAction.ALLOW.value)),
            last_assessed_at=(
                _coerce_utc_datetime(last_assessed_raw) if last_assessed_raw is not None else None
            ),
            last_assessment_id=payload.get("last_assessment_id"),
        )


@dataclass(frozen=True, slots=True)
class FirstPartyAssessment:
    """Risk assessment output produced by FirstPartyFraudEngine."""

    assessment_id: str
    account_id: str
    tenant_id: str
    risk_score: float
    risk_level: FirstPartyRiskLevel
    action: FirstPartyAction
    anomalies: tuple[FirstPartyAnomalyType, ...]
    return_ratio: float
    purchase_volume: float
    refund_volume: float
    dispute_count: int
    inr_claim_count: int
    promo_redemption_count: int
    linked_account_count: int
    case_id: str | None = None
    reasons: tuple[str, ...] = ()
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "account_id": self.account_id,
            "tenant_id": self.tenant_id,
            "risk_score": round(float(self.risk_score), 6),
            "risk_level": self.risk_level.value,
            "action": self.action.value,
            "anomalies": [a.value for a in self.anomalies],
            "return_ratio": round(float(self.return_ratio), 6),
            "purchase_volume": round(float(self.purchase_volume), 4),
            "refund_volume": round(float(self.refund_volume), 4),
            "dispute_count": self.dispute_count,
            "inr_claim_count": self.inr_claim_count,
            "promo_redemption_count": self.promo_redemption_count,
            "linked_account_count": self.linked_account_count,
            "case_id": self.case_id,
            "reasons": list(self.reasons),
            "evaluated_at": self.evaluated_at.astimezone(UTC).isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FirstPartyAssessment:
        return cls(
            assessment_id=str(payload["assessment_id"]),
            account_id=str(payload["account_id"]),
            tenant_id=str(payload.get("tenant_id", "default")),
            risk_score=float(payload["risk_score"]),
            risk_level=FirstPartyRiskLevel(payload["risk_level"]),
            action=FirstPartyAction(payload["action"]),
            anomalies=tuple(FirstPartyAnomalyType(a) for a in payload.get("anomalies", [])),
            return_ratio=float(payload["return_ratio"]),
            purchase_volume=float(payload["purchase_volume"]),
            refund_volume=float(payload["refund_volume"]),
            dispute_count=int(payload["dispute_count"]),
            inr_claim_count=int(payload["inr_claim_count"]),
            promo_redemption_count=int(payload["promo_redemption_count"]),
            linked_account_count=int(payload["linked_account_count"]),
            case_id=payload.get("case_id"),
            reasons=tuple(str(r) for r in payload.get("reasons", [])),
            evaluated_at=_coerce_utc_datetime(payload["evaluated_at"]),
        )
