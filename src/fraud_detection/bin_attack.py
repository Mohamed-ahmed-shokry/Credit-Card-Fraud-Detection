"""Card-testing and BIN enumeration attack defense engine.

Detects automated payment-gateway probing, sequential PAN/BIN enumeration,
CVV/expiry brute-force sweeps, micro-authorization velocity spikes, and
distributed botnet BIN attacks across merchants and IP/device clusters.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

__all__ = [
    "PROBING_RESPONSE_CODES",
    "AuthResponseCode",
    "BinAttackAction",
    "BinAttackAnomalyType",
    "BinAttackRiskLevel",
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
