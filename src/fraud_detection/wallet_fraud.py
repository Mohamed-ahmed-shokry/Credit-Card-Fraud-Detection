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


@dataclass(slots=True)
class WalletTokenRecord:
    """Provisioned network payment token lifecycle record."""

    card_hash: str
    device_id: str
    wallet_account_id: str
    wallet_provider: WalletProvider | str
    tenant_id: str = "default"
    token_id: str = field(default_factory=lambda: f"tok_{uuid4().hex[:12]}")
    status: TokenStatus | str = TokenStatus.ACTIVE
    provisioned_at: str = field(default_factory=_utc_now_iso)
    activated_at: str | None = None
    updated_at: str = field(default_factory=_utc_now_iso)
    total_spend_amount: float = 0.0
    transaction_count: int = 0
    suspension_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.card_hash = self.card_hash.strip()
        self.device_id = self.device_id.strip()
        self.wallet_account_id = self.wallet_account_id.strip()
        if not self.card_hash or not self.device_id or not self.wallet_account_id:
            raise ValueError("card_hash, device_id, and wallet_account_id must not be empty")
        if not isinstance(self.wallet_provider, WalletProvider):
            self.wallet_provider = WalletProvider(str(self.wallet_provider).lower())
        if not isinstance(self.status, TokenStatus):
            self.status = TokenStatus(str(self.status).lower())
        self.tenant_id = self.tenant_id.strip() or "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "card_hash": self.card_hash,
            "device_id": self.device_id,
            "wallet_account_id": self.wallet_account_id,
            "wallet_provider": WalletProvider(self.wallet_provider).value,
            "tenant_id": self.tenant_id,
            "status": TokenStatus(self.status).value,
            "provisioned_at": self.provisioned_at,
            "activated_at": self.activated_at,
            "updated_at": self.updated_at,
            "total_spend_amount": round(self.total_spend_amount, 2),
            "transaction_count": self.transaction_count,
            "suspension_reason": self.suspension_reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WalletTokenRecord:
        return cls(
            token_id=str(payload.get("token_id") or f"tok_{uuid4().hex[:12]}"),
            card_hash=str(payload["card_hash"]),
            device_id=str(payload["device_id"]),
            wallet_account_id=str(payload["wallet_account_id"]),
            wallet_provider=WalletProvider(
                str(payload.get("wallet_provider", "apple_pay")).lower()
            ),
            tenant_id=str(payload.get("tenant_id") or "default"),
            status=TokenStatus(str(payload.get("status", "active")).lower()),
            provisioned_at=str(payload.get("provisioned_at") or _utc_now_iso()),
            activated_at=payload.get("activated_at"),
            updated_at=str(payload.get("updated_at") or _utc_now_iso()),
            total_spend_amount=float(payload.get("total_spend_amount", 0.0)),
            transaction_count=int(payload.get("transaction_count", 0)),
            suspension_reason=payload.get("suspension_reason"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class WalletRiskAssessment:
    """Output of a digital wallet provisioning or post-provisioning risk evaluation."""

    assessment_id: str
    request_id: str
    card_hash: str
    device_id: str
    wallet_account_id: str
    wallet_provider: WalletProvider | str
    tenant_id: str
    risk_score: float
    risk_level: WalletRiskLevel | str
    provisioning_path: ProvisioningPath | str
    anomalies: list[WalletAnomalyType] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    token_id: str | None = None
    correlated_case_id: str | None = None
    evaluated_at: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.risk_score = round(_clamp(self.risk_score), 6)
        if not isinstance(self.wallet_provider, WalletProvider):
            self.wallet_provider = WalletProvider(str(self.wallet_provider).lower())
        if not isinstance(self.risk_level, WalletRiskLevel):
            self.risk_level = WalletRiskLevel(str(self.risk_level).lower())
        if not isinstance(self.provisioning_path, ProvisioningPath):
            self.provisioning_path = ProvisioningPath(str(self.provisioning_path).lower())
        self.anomalies = [WalletAnomalyType(str(a).lower()) for a in self.anomalies]

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "request_id": self.request_id,
            "card_hash": self.card_hash,
            "device_id": self.device_id,
            "wallet_account_id": self.wallet_account_id,
            "wallet_provider": WalletProvider(self.wallet_provider).value,
            "tenant_id": self.tenant_id,
            "risk_score": self.risk_score,
            "risk_level": WalletRiskLevel(self.risk_level).value,
            "provisioning_path": ProvisioningPath(self.provisioning_path).value,
            "anomalies": [a.value for a in self.anomalies],
            "rationale": list(self.rationale),
            "token_id": self.token_id,
            "correlated_case_id": self.correlated_case_id,
            "evaluated_at": self.evaluated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WalletRiskAssessment:
        return cls(
            assessment_id=str(payload["assessment_id"]),
            request_id=str(payload["request_id"]),
            card_hash=str(payload["card_hash"]),
            device_id=str(payload["device_id"]),
            wallet_account_id=str(payload["wallet_account_id"]),
            wallet_provider=WalletProvider(
                str(payload.get("wallet_provider", "apple_pay")).lower()
            ),
            tenant_id=str(payload.get("tenant_id") or "default"),
            risk_score=float(payload.get("risk_score", 0.0)),
            risk_level=WalletRiskLevel(str(payload.get("risk_level", "low")).lower()),
            provisioning_path=ProvisioningPath(
                str(payload.get("provisioning_path", "green_path")).lower()
            ),
            anomalies=[WalletAnomalyType(str(a).lower()) for a in payload.get("anomalies", [])],
            rationale=[str(r) for r in payload.get("rationale", [])],
            token_id=payload.get("token_id"),
            correlated_case_id=payload.get("correlated_case_id"),
            evaluated_at=str(payload.get("evaluated_at") or _utc_now_iso()),
            metadata=dict(payload.get("metadata", {})),
        )


class ProvisioningVelocityAnalyzer:
    """Detects multi-card device velocity, card fan-out, and compromised device reuse."""

    def __init__(
        self,
        *,
        max_cards_per_device: int = 3,
        max_wallets_per_card: int = 3,
    ) -> None:
        self.max_cards_per_device = max(1, max_cards_per_device)
        self.max_wallets_per_card = max(1, max_wallets_per_card)

    def analyze(
        self,
        request: WalletProvisioningRequest,
        existing_tokens: list[WalletTokenRecord],
    ) -> tuple[float, list[WalletAnomalyType], list[str]]:
        score = 0.0
        anomalies: list[WalletAnomalyType] = []
        rationale: list[str] = []

        device_tokens = [
            t
            for t in existing_tokens
            if t.device_id == request.device_id and t.tenant_id == request.tenant_id
        ]
        distinct_cards_on_device = {t.card_hash for t in device_tokens} | {request.card_hash}
        if len(distinct_cards_on_device) > self.max_cards_per_device:
            score += 0.35
            anomalies.append(WalletAnomalyType.DEVICE_PROVISIONING_VELOCITY)
            rationale.append(
                f"Device '{request.device_id}' provisioned {len(distinct_cards_on_device)} "
                f"distinct cards (limit {self.max_cards_per_device})"
            )

        if any(t.status in {TokenStatus.SUSPENDED, TokenStatus.DEACTIVATED} for t in device_tokens):
            score += 0.30
            anomalies.append(WalletAnomalyType.COMPROMISED_DEVICE_REUSE)
            rationale.append(
                f"Device '{request.device_id}' has prior suspended or deactivated tokens"
            )

        card_tokens = [
            t
            for t in existing_tokens
            if t.card_hash == request.card_hash and t.tenant_id == request.tenant_id
        ]
        distinct_wallets_for_card = {t.wallet_account_id for t in card_tokens} | {
            request.wallet_account_id
        }
        if len(distinct_wallets_for_card) > self.max_wallets_per_card:
            score += 0.30
            anomalies.append(WalletAnomalyType.CARD_WALLET_FAN_OUT)
            rationale.append(
                f"Card provisioned across {len(distinct_wallets_for_card)} distinct wallets "
                f"(limit {self.max_wallets_per_card})"
            )

        return _clamp(score), anomalies, rationale
