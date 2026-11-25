"""Authorized Push Payment (APP) Scam & Social Engineering Defense Engine.

Detects genuine-customer authorized push payments initiated under social
engineering coercion or deception (bank/government impersonation, investment
and crypto scams, romance/pig-butchering, invoice redirection / BEC, and
remote-access-tool coaching), applies graduated friction and cooling-off holds,
and escalates high-risk cases to CaseManager and EventBus.
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


class ScamTypology(StrEnum):
    """Classified Authorized Push Payment (APP) social engineering scam typology."""

    IMPERSONATION_BANK = "impersonation_bank"
    IMPERSONATION_GOVERNMENT = "impersonation_government"
    INVESTMENT_CRYPTO = "investment_crypto"
    ROMANCE_PIG_BUTCHERING = "romance_pig_butchering"
    INVOICE_REDIRECTION_BEC = "invoice_redirection_bec"
    PURCHASE_MARKETPLACE = "purchase_marketplace"
    REMOTE_ACCESS_COACHING = "remote_access_coaching"
    UNKNOWN = "unknown"


class ScamRiskLevel(StrEnum):
    """Calibrated risk tier for an APP payment evaluation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ScamInterventionAction(StrEnum):
    """Graduated customer-friction and containment intervention actions."""

    ALLOW = "allow"
    DYNAMIC_WARNING = "dynamic_warning"
    COOLING_OFF_HOLD = "cooling_off_hold"
    CALLBACK_VERIFICATION = "callback_verification"
    BLOCK_AND_ESCALATE = "block_and_escalate"


class ScamSignalType(StrEnum):
    """Distinct behavioral, payee, and transfer-pattern APP scam signals."""

    ACTIVE_CALL_OR_SCREEN_SHARE = "active_call_or_screen_share"
    NEW_PAYEE_HIGH_VALUE = "new_payee_high_value"
    URGENCY_OR_SCRIPTED_COACHING = "urgency_or_scripted_coaching"
    BEHAVIORAL_HESITATION_OR_COPY_PASTE = "behavioral_hesitation_or_copy_paste"
    BENEFICIARY_NAME_MISMATCH = "beneficiary_name_mismatch"
    RAPID_ESCALATING_TRANSFERS = "rapid_escalating_transfers"
    CRYPTO_OR_HIGH_RISK_RAIL = "crypto_or_high_risk_rail"
    ELDER_OR_VULNERABLE_ANOMALY = "elder_or_vulnerable_anomaly"


class CoolingOffHoldStatus(StrEnum):
    """Lifecycle states of a mandatory APP scam cooling-off payment hold."""

    ACTIVE = "active"
    RELEASED = "released"
    CONFIRMED_SCAM_CANCELLED = "confirmed_scam_cancelled"


@dataclass(slots=True)
class AppPaymentContext:
    """Inbound authorized push payment context for social engineering evaluation."""

    sender_account_id: str
    payee_account_id: str
    amount: float
    currency: str = "USD"
    payment_rail: str = "faster_payments"
    tenant_id: str = "default"
    payment_id: str = field(default_factory=lambda: f"app_{uuid4().hex[:12]}")
    is_new_payee: bool = True
    cop_name_match: str = "match"
    active_phone_call: bool = False
    screen_share_or_rat_detected: bool = False
    payee_details_pasted: bool = False
    session_hesitation_seconds: float = 0.0
    urgency_keywords_detected: bool = False
    stated_purpose: str = "general"
    sender_age_years: int | None = None
    sender_balance_before: float | None = None
    initiated_at: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sender_account_id = self.sender_account_id.strip()
        self.payee_account_id = self.payee_account_id.strip()
        if not self.sender_account_id or not self.payee_account_id:
            raise ValueError("sender_account_id and payee_account_id must not be empty")
        self.amount = float(self.amount)
        if self.amount <= 0:
            raise ValueError("amount must be greater than zero")
        self.currency = self.currency.strip().upper() or "USD"
        self.payment_rail = self.payment_rail.strip().lower() or "faster_payments"
        self.tenant_id = self.tenant_id.strip() or "default"
        self.cop_name_match = self.cop_name_match.strip().lower() or "match"
        self.session_hesitation_seconds = max(0.0, float(self.session_hesitation_seconds))
        self.stated_purpose = self.stated_purpose.strip().lower() or "general"
        if self.sender_age_years is not None:
            self.sender_age_years = max(0, int(self.sender_age_years))
        if self.sender_balance_before is not None:
            self.sender_balance_before = max(0.0, float(self.sender_balance_before))

    def to_dict(self) -> dict[str, Any]:
        return {
            "payment_id": self.payment_id,
            "sender_account_id": self.sender_account_id,
            "payee_account_id": self.payee_account_id,
            "amount": self.amount,
            "currency": self.currency,
            "payment_rail": self.payment_rail,
            "tenant_id": self.tenant_id,
            "is_new_payee": self.is_new_payee,
            "cop_name_match": self.cop_name_match,
            "active_phone_call": self.active_phone_call,
            "screen_share_or_rat_detected": self.screen_share_or_rat_detected,
            "payee_details_pasted": self.payee_details_pasted,
            "session_hesitation_seconds": self.session_hesitation_seconds,
            "urgency_keywords_detected": self.urgency_keywords_detected,
            "stated_purpose": self.stated_purpose,
            "sender_age_years": self.sender_age_years,
            "sender_balance_before": self.sender_balance_before,
            "initiated_at": self.initiated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AppPaymentContext:
        return cls(
            payment_id=str(payload.get("payment_id", f"app_{uuid4().hex[:12]}")),
            sender_account_id=str(payload["sender_account_id"]),
            payee_account_id=str(payload["payee_account_id"]),
            amount=float(payload["amount"]),
            currency=str(payload.get("currency", "USD")),
            payment_rail=str(payload.get("payment_rail", "faster_payments")),
            tenant_id=str(payload.get("tenant_id", "default")),
            is_new_payee=bool(payload.get("is_new_payee", True)),
            cop_name_match=str(payload.get("cop_name_match", "match")),
            active_phone_call=bool(payload.get("active_phone_call", False)),
            screen_share_or_rat_detected=bool(payload.get("screen_share_or_rat_detected", False)),
            payee_details_pasted=bool(payload.get("payee_details_pasted", False)),
            session_hesitation_seconds=float(payload.get("session_hesitation_seconds", 0.0)),
            urgency_keywords_detected=bool(payload.get("urgency_keywords_detected", False)),
            stated_purpose=str(payload.get("stated_purpose", "general")),
            sender_age_years=(
                int(payload["sender_age_years"])
                if payload.get("sender_age_years") is not None
                else None
            ),
            sender_balance_before=(
                float(payload["sender_balance_before"])
                if payload.get("sender_balance_before") is not None
                else None
            ),
            initiated_at=str(payload.get("initiated_at", _utc_now_iso())),
            metadata=dict(payload.get("metadata", {})),
        )
