"""First-party fraud, refund abuse, and friendly fraud detection engine.

Detects legitimate cardholders and account owners exploiting merchant refund,
chargeback, and promotional policies via serial friendly-fraud disputes,
high-ratio wardrobing/return abuse, Item-Not-Received (INR) claim bursts,
and multi-account promo/referral cycling.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

__all__ = [
    "WARDROBING_CATEGORIES",
    "ClaimOutcome",
    "FirstPartyAction",
    "FirstPartyAnomalyType",
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
