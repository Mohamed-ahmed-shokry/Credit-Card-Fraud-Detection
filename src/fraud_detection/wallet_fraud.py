"""Digital Wallet Provisioning & Network Token Fraud Defense Engine.

Scores digital wallet tokenization requests (Apple Pay, Google Pay, Samsung Pay,
Click-to-Pay, Merchant Tokens) modeled on Visa VTS and Mastercard MDES issuer
decisioning (GREEN_PATH / YELLOW_PATH / RED_PATH), manages network token
lifecycles, and detects post-provisioning high-value spend bursts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


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


@dataclass(slots=True)
class WalletProvisioningRequest:
    """Inbound digital wallet tokenization request for issuer decisioning."""

    card_hash: str
    device_id: str
    wallet_account_id: str
    wallet_provider: WalletProvider | str = WalletProvider.APPLE_PAY
    tenant_id: str = "default"
    request_id: str = field(default_factory=lambda: f"wpr_{uuid4().hex[:12]}")
    wallet_account_age_days: int = 365
    device_country: str = "US"
    issuer_country: str = "US"
    phone_match: bool = True
    sim_swap_detected: bool = False
    otp_attempts: int = 0
    requested_at: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.card_hash = self.card_hash.strip()
        self.device_id = self.device_id.strip()
        self.wallet_account_id = self.wallet_account_id.strip()
        if not self.card_hash or not self.device_id or not self.wallet_account_id:
            raise ValueError("card_hash, device_id, and wallet_account_id must not be empty")
        if not isinstance(self.wallet_provider, WalletProvider):
            self.wallet_provider = WalletProvider(str(self.wallet_provider).lower())
        self.tenant_id = self.tenant_id.strip() or "default"
        self.wallet_account_age_days = max(0, int(self.wallet_account_age_days))
        self.otp_attempts = max(0, int(self.otp_attempts))
        self.device_country = self.device_country.strip().upper() or "US"
        self.issuer_country = self.issuer_country.strip().upper() or "US"

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "card_hash": self.card_hash,
            "device_id": self.device_id,
            "wallet_account_id": self.wallet_account_id,
            "wallet_provider": WalletProvider(self.wallet_provider).value,
            "tenant_id": self.tenant_id,
            "wallet_account_age_days": self.wallet_account_age_days,
            "device_country": self.device_country,
            "issuer_country": self.issuer_country,
            "phone_match": self.phone_match,
            "sim_swap_detected": self.sim_swap_detected,
            "otp_attempts": self.otp_attempts,
            "requested_at": self.requested_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WalletProvisioningRequest:
        return cls(
            request_id=str(payload.get("request_id") or f"wpr_{uuid4().hex[:12]}"),
            card_hash=str(payload["card_hash"]),
            device_id=str(payload["device_id"]),
            wallet_account_id=str(payload["wallet_account_id"]),
            wallet_provider=WalletProvider(
                str(payload.get("wallet_provider", "apple_pay")).lower()
            ),
            tenant_id=str(payload.get("tenant_id") or "default"),
            wallet_account_age_days=int(payload.get("wallet_account_age_days", 365)),
            device_country=str(payload.get("device_country", "US")),
            issuer_country=str(payload.get("issuer_country", "US")),
            phone_match=bool(payload.get("phone_match", True)),
            sim_swap_detected=bool(payload.get("sim_swap_detected", False)),
            otp_attempts=int(payload.get("otp_attempts", 0)),
            requested_at=str(payload.get("requested_at") or _utc_now_iso()),
            metadata=dict(payload.get("metadata", {})),
        )
