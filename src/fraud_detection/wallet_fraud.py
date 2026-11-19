"""Digital Wallet Provisioning & Network Token Fraud Defense Engine.

Scores digital wallet tokenization requests (Apple Pay, Google Pay, Samsung Pay,
Click-to-Pay, Merchant Tokens) modeled on Visa VTS and Mastercard MDES issuer
decisioning (GREEN_PATH / YELLOW_PATH / RED_PATH), manages network token
lifecycles, and detects post-provisioning high-value spend bursts.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from fraud_detection.cases import CaseManager
from fraud_detection.events import EventBus, EventType, FraudDomainEvent


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


class IdentityTelemetryAnalyzer:
    """Evaluates SIM-swap, phone mismatch, OTP retry bursts, geo mismatch, and account tenure."""

    def __init__(
        self,
        *,
        min_account_age_days: int = 14,
        otp_burst_threshold: int = 3,
    ) -> None:
        self.min_account_age_days = max(1, min_account_age_days)
        self.otp_burst_threshold = max(1, otp_burst_threshold)

    def analyze(
        self,
        request: WalletProvisioningRequest,
    ) -> tuple[float, list[WalletAnomalyType], list[str]]:
        score = 0.0
        anomalies: list[WalletAnomalyType] = []
        rationale: list[str] = []

        if request.sim_swap_detected or not request.phone_match:
            delta = 0.45 if request.sim_swap_detected else 0.28
            score += delta
            anomalies.append(WalletAnomalyType.PHONE_MISMATCH_SIM_SWAP)
            detail = (
                "Carrier SIM swap detected"
                if request.sim_swap_detected
                else "Device phone does not match issuer cardholder phone on file"
            )
            rationale.append(detail)

        if request.otp_attempts >= self.otp_burst_threshold:
            score += 0.30
            anomalies.append(WalletAnomalyType.OTP_INTERCEPTION_BURST)
            rationale.append(
                f"High OTP retry velocity ({request.otp_attempts} attempts >= "
                f"{self.otp_burst_threshold})"
            )

        if request.device_country != request.issuer_country:
            score += 0.22
            anomalies.append(WalletAnomalyType.GEO_ISSUER_MISMATCH)
            rationale.append(
                f"Device country ({request.device_country}) mismatches issuer country "
                f"({request.issuer_country})"
            )

        if request.wallet_account_age_days < self.min_account_age_days:
            score += 0.18
            anomalies.append(WalletAnomalyType.LOW_WALLET_ACCOUNT_TENURE)
            rationale.append(
                f"Newly created wallet account ({request.wallet_account_age_days} days < "
                f"{self.min_account_age_days} days)"
            )

        return _clamp(score), anomalies, rationale


class TokenLifecycleAnalyzer:
    """Detects post-provisioning high-value spend bursts immediately after activation."""

    def __init__(
        self,
        *,
        burst_window_seconds: int = 3600,
        high_value_threshold: float = 1500.0,
    ) -> None:
        self.burst_window_seconds = max(60, burst_window_seconds)
        self.high_value_threshold = max(1.0, float(high_value_threshold))

    def analyze_spend(
        self,
        token: WalletTokenRecord,
        amount: float,
        *,
        reference_time: datetime | None = None,
    ) -> tuple[float, list[WalletAnomalyType], list[str]]:
        now = reference_time or datetime.now(UTC)
        anchor_ts = token.activated_at or token.provisioned_at
        elapsed_seconds = max(0.0, (now - _parse_iso(anchor_ts)).total_seconds())
        cumulative = token.total_spend_amount + max(0.0, float(amount))

        if elapsed_seconds <= self.burst_window_seconds and cumulative >= self.high_value_threshold:
            return (
                0.85,
                [WalletAnomalyType.POST_PROVISIONING_HIGH_VALUE_BURST],
                [
                    f"Rapid post-provisioning spend of ${cumulative:,.2f} within "
                    f"{int(elapsed_seconds)}s of token activation"
                ],
            )
        return 0.0, [], []


class WalletTokenStore:
    """Thread-safe in-memory ledger for network tokens and provisioning assessments."""

    def __init__(
        self,
        *,
        max_tokens: int = 20000,
        max_assessments: int = 20000,
    ) -> None:
        self._lock = threading.RLock()
        self.max_tokens = max(1, max_tokens)
        self.max_assessments = max(1, max_assessments)
        self._tokens: dict[str, WalletTokenRecord] = {}
        self._token_order: list[str] = []
        self._assessments: dict[str, WalletRiskAssessment] = {}
        self._assessment_order: list[str] = []

    def _evict_tokens_if_needed(self) -> None:
        while len(self._token_order) > self.max_tokens:
            oldest_id = self._token_order.pop(0)
            self._tokens.pop(oldest_id, None)

    def _evict_assessments_if_needed(self) -> None:
        while len(self._assessment_order) > self.max_assessments:
            oldest_id = self._assessment_order.pop(0)
            self._assessments.pop(oldest_id, None)

    def save_token(self, token: WalletTokenRecord) -> None:
        with self._lock:
            if token.token_id not in self._tokens:
                self._token_order.append(token.token_id)
            self._tokens[token.token_id] = token
            self._evict_tokens_if_needed()

    def get_token(self, token_id: str) -> WalletTokenRecord | None:
        with self._lock:
            return self._tokens.get(token_id)

    def list_tokens(
        self,
        *,
        card_hash: str | None = None,
        device_id: str | None = None,
        tenant_id: str | None = None,
        status: TokenStatus | None = None,
        limit: int = 100,
    ) -> list[WalletTokenRecord]:
        with self._lock:
            items = [self._tokens[tid] for tid in reversed(self._token_order)]
            if card_hash is not None:
                items = [t for t in items if t.card_hash == card_hash]
            if device_id is not None:
                items = [t for t in items if t.device_id == device_id]
            if tenant_id is not None:
                items = [t for t in items if t.tenant_id == tenant_id]
            if status is not None:
                items = [t for t in items if t.status is status]
            return items[: max(1, limit)]

    def save_assessment(self, assessment: WalletRiskAssessment) -> None:
        with self._lock:
            if assessment.assessment_id not in self._assessments:
                self._assessment_order.append(assessment.assessment_id)
            self._assessments[assessment.assessment_id] = assessment
            self._evict_assessments_if_needed()

    def get_assessment(self, assessment_id: str) -> WalletRiskAssessment | None:
        with self._lock:
            return self._assessments.get(assessment_id)

    def list_assessments(
        self,
        *,
        card_hash: str | None = None,
        device_id: str | None = None,
        tenant_id: str | None = None,
        path: ProvisioningPath | None = None,
        limit: int = 100,
    ) -> list[WalletRiskAssessment]:
        with self._lock:
            items = [self._assessments[aid] for aid in reversed(self._assessment_order)]
            if card_hash is not None:
                items = [a for a in items if a.card_hash == card_hash]
            if device_id is not None:
                items = [a for a in items if a.device_id == device_id]
            if tenant_id is not None:
                items = [a for a in items if a.tenant_id == tenant_id]
            if path is not None:
                items = [a for a in items if a.provisioning_path is path]
            return items[: max(1, limit)]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            by_status: dict[str, int] = {st.value: 0 for st in TokenStatus}
            for tok in self._tokens.values():
                key = TokenStatus(tok.status).value
                by_status[key] = by_status.get(key, 0) + 1

            by_path: dict[str, int] = {p.value: 0 for p in ProvisioningPath}
            for ass in self._assessments.values():
                pkey = ProvisioningPath(ass.provisioning_path).value
                by_path[pkey] = by_path.get(pkey, 0) + 1

            return {
                "token_count": len(self._tokens),
                "assessment_count": len(self._assessments),
                "by_token_status": by_status,
                "by_provisioning_path": by_path,
            }

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_tokens": self.max_tokens,
                "max_assessments": self.max_assessments,
                "tokens": [self._tokens[tid].to_dict() for tid in self._token_order],
                "assessments": [self._assessments[aid].to_dict() for aid in self._assessment_order],
            }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WalletTokenStore:
        store = cls(
            max_tokens=int(payload.get("max_tokens", 20000)),
            max_assessments=int(payload.get("max_assessments", 20000)),
        )
        for t_raw in payload.get("tokens", []):
            tok = WalletTokenRecord.from_dict(t_raw)
            store._tokens[tok.token_id] = tok
            store._token_order.append(tok.token_id)
        for a_raw in payload.get("assessments", []):
            ass = WalletRiskAssessment.from_dict(a_raw)
            store._assessments[ass.assessment_id] = ass
            store._assessment_order.append(ass.assessment_id)
        return store

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> WalletTokenStore:
        source = Path(path)
        if not source.exists():
            return cls()
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))


class WalletFraudEngine:
    """Orchestrates VTS/MDES digital wallet provisioning, token lifecycle, and spend defense."""

    def __init__(
        self,
        *,
        store: WalletTokenStore | None = None,
        velocity_analyzer: ProvisioningVelocityAnalyzer | None = None,
        identity_analyzer: IdentityTelemetryAnalyzer | None = None,
        lifecycle_analyzer: TokenLifecycleAnalyzer | None = None,
        case_manager: CaseManager | None = None,
        event_bus: EventBus | None = None,
        yellow_path_threshold: float = 0.30,
        red_path_threshold: float = 0.70,
    ) -> None:
        self.store = store or WalletTokenStore()
        self.velocity_analyzer = velocity_analyzer or ProvisioningVelocityAnalyzer()
        self.identity_analyzer = identity_analyzer or IdentityTelemetryAnalyzer()
        self.lifecycle_analyzer = lifecycle_analyzer or TokenLifecycleAnalyzer()
        self.case_manager = case_manager
        self.event_bus = event_bus
        self.yellow_path_threshold = float(yellow_path_threshold)
        self.red_path_threshold = float(red_path_threshold)
        self._lock = threading.RLock()

    def _emit_event(
        self,
        event_type: EventType,
        *,
        tenant_id: str,
        payload: dict[str, Any],
    ) -> None:
        if self.event_bus is None:
            return
        self.event_bus.publish(
            FraudDomainEvent(
                event_type=event_type,
                tenant_id=tenant_id,
                payload=payload,
            )
        )

    def _route_score(self, score: float) -> tuple[WalletRiskLevel, ProvisioningPath]:
        if score >= 0.85:
            return WalletRiskLevel.CRITICAL, ProvisioningPath.RED_PATH
        if score >= self.red_path_threshold:
            return WalletRiskLevel.HIGH, ProvisioningPath.RED_PATH
        if score >= self.yellow_path_threshold:
            return WalletRiskLevel.MEDIUM, ProvisioningPath.YELLOW_PATH
        return WalletRiskLevel.LOW, ProvisioningPath.GREEN_PATH

    def _escalate_case(
        self,
        *,
        card_hash: str,
        device_id: str,
        score: float,
        tenant_id: str,
        reason: str,
        anomalies: list[WalletAnomalyType],
    ) -> str | None:
        if self.case_manager is None:
            return None
        created = self.case_manager.create_case(
            transaction_id=f"wallet:{card_hash}:{device_id}",
            fraud_score=score,
            tenant_id=tenant_id,
            metadata={
                "source": "wallet_fraud_engine",
                "card_hash": card_hash,
                "device_id": device_id,
                "reason": reason,
                "anomalies": [a.value for a in anomalies],
            },
        )
        return created.case_id

    def evaluate_provisioning(
        self,
        request: WalletProvisioningRequest,
        *,
        reference_time: datetime | None = None,
    ) -> WalletRiskAssessment:
        now = reference_time or datetime.now(UTC)
        with self._lock:
            existing = self.store.list_tokens(tenant_id=request.tenant_id, limit=50000)
            vel_score, vel_anoms, vel_rat = self.velocity_analyzer.analyze(request, existing)
            id_score, id_anoms, id_rat = self.identity_analyzer.analyze(request)

            composite = round(_clamp(vel_score + id_score), 6)
            anomalies = list(dict.fromkeys([*vel_anoms, *id_anoms]))
            rationale = [*vel_rat, *id_rat]
            risk_level, path = self._route_score(composite)

            token_id: str | None = None
            case_id: str | None = None

            if path is ProvisioningPath.GREEN_PATH:
                tok = WalletTokenRecord(
                    card_hash=request.card_hash,
                    device_id=request.device_id,
                    wallet_account_id=request.wallet_account_id,
                    wallet_provider=request.wallet_provider,
                    tenant_id=request.tenant_id,
                    status=TokenStatus.ACTIVE,
                    provisioned_at=now.isoformat(),
                    activated_at=now.isoformat(),
                    updated_at=now.isoformat(),
                )
                self.store.save_token(tok)
                token_id = tok.token_id
            elif path is ProvisioningPath.YELLOW_PATH:
                tok = WalletTokenRecord(
                    card_hash=request.card_hash,
                    device_id=request.device_id,
                    wallet_account_id=request.wallet_account_id,
                    wallet_provider=request.wallet_provider,
                    tenant_id=request.tenant_id,
                    status=TokenStatus.PENDING_IDV,
                    provisioned_at=now.isoformat(),
                    activated_at=None,
                    updated_at=now.isoformat(),
                )
                self.store.save_token(tok)
                token_id = tok.token_id
            else:
                case_id = self._escalate_case(
                    card_hash=request.card_hash,
                    device_id=request.device_id,
                    score=composite,
                    tenant_id=request.tenant_id,
                    reason="red_path_provisioning_declined",
                    anomalies=anomalies,
                )

            assessment = WalletRiskAssessment(
                assessment_id=f"wra_{uuid4().hex[:12]}",
                request_id=request.request_id,
                card_hash=request.card_hash,
                device_id=request.device_id,
                wallet_account_id=request.wallet_account_id,
                wallet_provider=request.wallet_provider,
                tenant_id=request.tenant_id,
                risk_score=composite,
                risk_level=risk_level,
                provisioning_path=path,
                anomalies=anomalies,
                rationale=rationale,
                token_id=token_id,
                correlated_case_id=case_id,
                evaluated_at=now.isoformat(),
                metadata=dict(request.metadata),
            )
            self.store.save_assessment(assessment)

            base_payload = {
                "assessment_id": assessment.assessment_id,
                "card_hash": assessment.card_hash,
                "device_id": assessment.device_id,
                "wallet_provider": WalletProvider(assessment.wallet_provider).value,
                "risk_score": assessment.risk_score,
                "provisioning_path": path.value,
                "token_id": token_id,
            }
            self._emit_event(
                EventType.WALLET_PROVISIONING_EVALUATED,
                tenant_id=request.tenant_id,
                payload=base_payload,
            )
            if path is ProvisioningPath.GREEN_PATH:
                self._emit_event(
                    EventType.WALLET_TOKEN_ACTIVATED,
                    tenant_id=request.tenant_id,
                    payload=base_payload,
                )
            elif path is ProvisioningPath.RED_PATH:
                self._emit_event(
                    EventType.WALLET_PROVISIONING_DECLINED,
                    tenant_id=request.tenant_id,
                    payload={**base_payload, "case_id": case_id},
                )
            return assessment

    def complete_idv(
        self,
        token_id: str,
        *,
        verified: bool = True,
        reference_time: datetime | None = None,
    ) -> WalletTokenRecord:
        now = reference_time or datetime.now(UTC)
        with self._lock:
            tok = self.store.get_token(token_id)
            if tok is None:
                raise KeyError(f"Wallet token '{token_id}' not found")
            if verified:
                tok.status = TokenStatus.ACTIVE
                tok.activated_at = now.isoformat()
                tok.updated_at = now.isoformat()
                self.store.save_token(tok)
                self._emit_event(
                    EventType.WALLET_TOKEN_ACTIVATED,
                    tenant_id=tok.tenant_id,
                    payload={
                        "token_id": tok.token_id,
                        "card_hash": tok.card_hash,
                        "device_id": tok.device_id,
                        "source": "idv_verified",
                    },
                )
            else:
                tok.status = TokenStatus.DEACTIVATED
                tok.suspension_reason = "idv_failed"
                tok.updated_at = now.isoformat()
                self.store.save_token(tok)
                self._emit_event(
                    EventType.WALLET_PROVISIONING_DECLINED,
                    tenant_id=tok.tenant_id,
                    payload={
                        "token_id": tok.token_id,
                        "card_hash": tok.card_hash,
                        "reason": "idv_failed",
                    },
                )
            return tok

    def suspend_token(
        self,
        token_id: str,
        *,
        reason: str = "fraud_suspected",
        reference_time: datetime | None = None,
    ) -> WalletTokenRecord:
        now = reference_time or datetime.now(UTC)
        with self._lock:
            tok = self.store.get_token(token_id)
            if tok is None:
                raise KeyError(f"Wallet token '{token_id}' not found")
            tok.status = TokenStatus.SUSPENDED
            tok.suspension_reason = reason
            tok.updated_at = now.isoformat()
            self.store.save_token(tok)
            self._emit_event(
                EventType.WALLET_TOKEN_SUSPENDED,
                tenant_id=tok.tenant_id,
                payload={
                    "token_id": tok.token_id,
                    "card_hash": tok.card_hash,
                    "device_id": tok.device_id,
                    "reason": reason,
                },
            )
            return tok

    def deactivate_token(
        self,
        token_id: str,
        *,
        reason: str = "operator_deactivated",
        reference_time: datetime | None = None,
    ) -> WalletTokenRecord:
        now = reference_time or datetime.now(UTC)
        with self._lock:
            tok = self.store.get_token(token_id)
            if tok is None:
                raise KeyError(f"Wallet token '{token_id}' not found")
            tok.status = TokenStatus.DEACTIVATED
            tok.suspension_reason = reason
            tok.updated_at = now.isoformat()
            self.store.save_token(tok)
            self._emit_event(
                EventType.WALLET_TOKEN_SUSPENDED,
                tenant_id=tok.tenant_id,
                payload={
                    "token_id": tok.token_id,
                    "card_hash": tok.card_hash,
                    "device_id": tok.device_id,
                    "reason": reason,
                    "permanent": True,
                },
            )
            return tok

    def record_token_spend(
        self,
        token_id: str,
        amount: float,
        *,
        reference_time: datetime | None = None,
    ) -> WalletRiskAssessment:
        now = reference_time or datetime.now(UTC)
        with self._lock:
            tok = self.store.get_token(token_id)
            if tok is None:
                raise KeyError(f"Wallet token '{token_id}' not found")
            score, anomalies, rationale = self.lifecycle_analyzer.analyze_spend(
                tok, amount, reference_time=now
            )
            tok.total_spend_amount = round(tok.total_spend_amount + max(0.0, float(amount)), 2)
            tok.transaction_count += 1
            tok.updated_at = now.isoformat()
            self.store.save_token(tok)

            risk_level, path = self._route_score(score)
            case_id: str | None = None
            if anomalies:
                self.suspend_token(
                    tok.token_id,
                    reason="post_provisioning_high_value_burst",
                    reference_time=now,
                )
                case_id = self._escalate_case(
                    card_hash=tok.card_hash,
                    device_id=tok.device_id,
                    score=score,
                    tenant_id=tok.tenant_id,
                    reason="post_provisioning_high_value_burst",
                    anomalies=anomalies,
                )

            assessment = WalletRiskAssessment(
                assessment_id=f"wra_{uuid4().hex[:12]}",
                request_id=f"spend_{uuid4().hex[:10]}",
                card_hash=tok.card_hash,
                device_id=tok.device_id,
                wallet_account_id=tok.wallet_account_id,
                wallet_provider=tok.wallet_provider,
                tenant_id=tok.tenant_id,
                risk_score=score,
                risk_level=risk_level,
                provisioning_path=path,
                anomalies=anomalies,
                rationale=rationale,
                token_id=tok.token_id,
                correlated_case_id=case_id,
                evaluated_at=now.isoformat(),
                metadata={"spend_amount": float(amount)},
            )
            self.store.save_assessment(assessment)
            return assessment
