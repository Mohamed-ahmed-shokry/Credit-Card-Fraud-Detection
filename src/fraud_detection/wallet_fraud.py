"""Digital Wallet Provisioning & Network Token Fraud Defense Engine.

Scores digital wallet tokenization requests (Apple Pay, Google Pay, Samsung Pay,
Click-to-Pay, Merchant Tokens) modeled on Visa VTS and Mastercard MDES issuer
decisioning (GREEN_PATH / YELLOW_PATH / RED_PATH), manages network token
lifecycles, and detects post-provisioning high-value spend bursts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_iso(ts: str) -> datetime:
    parsed = datetime.fromisoformat(ts)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class WalletProvider(StrEnum):
    """Supported digital wallet and network tokenization providers."""

    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"
    SAMSUNG_PAY = "samsung_pay"
    CLICK_TO_PAY = "click_to_pay"
    MERCHANT_TOKEN = "merchant_token"  # noqa: S105


class ProvisioningPath(StrEnum):
    """Issuer VTS/MDES provisioning decisioning paths."""

    GREEN_PATH = "green_path"
    YELLOW_PATH = "yellow_path"
    RED_PATH = "red_path"


class WalletRiskLevel(StrEnum):
    """Calibrated risk tier for digital wallet provisioning and token activity."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TokenStatus(StrEnum):
    """Lifecycle states of a provisioned network payment token."""

    PENDING_IDV = "pending_idv"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"


class WalletAnomalyType(StrEnum):
    """Fraud anomaly signals detected during wallet provisioning and token spend."""

    DEVICE_PROVISIONING_VELOCITY = "device_provisioning_velocity"
    CARD_WALLET_FAN_OUT = "card_wallet_fan_out"
    PHONE_MISMATCH_SIM_SWAP = "phone_mismatch_sim_swap"
    OTP_INTERCEPTION_BURST = "otp_interception_burst"
    GEO_ISSUER_MISMATCH = "geo_issuer_mismatch"
    LOW_WALLET_ACCOUNT_TENURE = "low_wallet_account_tenure"
    POST_PROVISIONING_HIGH_VALUE_BURST = "post_provisioning_high_value_burst"
    COMPROMISED_DEVICE_REUSE = "compromised_device_reuse"
