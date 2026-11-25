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


@dataclass(slots=True)
class ScamSignal:
    """Individual social engineering or beneficiary risk signal."""

    signal_type: ScamSignalType | str
    severity: float
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.signal_type, ScamSignalType):
            self.signal_type = ScamSignalType(str(self.signal_type).lower())
        self.severity = round(_clamp(self.severity), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": ScamSignalType(self.signal_type).value,
            "severity": self.severity,
            "detail": self.detail,
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScamSignal:
        return cls(
            signal_type=ScamSignalType(str(payload["signal_type"]).lower()),
            severity=float(payload.get("severity", 0.0)),
            detail=str(payload.get("detail", "")),
            evidence=dict(payload.get("evidence", {})),
        )


@dataclass(slots=True)
class ScamCoolingOffHold:
    """Reversible cooling-off hold placed on a suspected APP scam payment."""

    payment_id: str
    sender_account_id: str
    payee_account_id: str
    amount: float
    typology: ScamTypology | str
    tenant_id: str = "default"
    hold_id: str = field(default_factory=lambda: f"hold_{uuid4().hex[:12]}")
    status: CoolingOffHoldStatus | str = CoolingOffHoldStatus.ACTIVE
    hold_duration_minutes: int = 120
    reason: str = ""
    created_at: str = field(default_factory=_utc_now_iso)
    resolved_at: str | None = None
    resolution_note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.typology, ScamTypology):
            self.typology = ScamTypology(str(self.typology).lower())
        if not isinstance(self.status, CoolingOffHoldStatus):
            self.status = CoolingOffHoldStatus(str(self.status).lower())
        self.hold_duration_minutes = max(1, int(self.hold_duration_minutes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "hold_id": self.hold_id,
            "payment_id": self.payment_id,
            "sender_account_id": self.sender_account_id,
            "payee_account_id": self.payee_account_id,
            "amount": self.amount,
            "typology": ScamTypology(self.typology).value,
            "tenant_id": self.tenant_id,
            "status": CoolingOffHoldStatus(self.status).value,
            "hold_duration_minutes": self.hold_duration_minutes,
            "reason": self.reason,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "resolution_note": self.resolution_note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScamCoolingOffHold:
        return cls(
            hold_id=str(payload.get("hold_id", f"hold_{uuid4().hex[:12]}")),
            payment_id=str(payload["payment_id"]),
            sender_account_id=str(payload["sender_account_id"]),
            payee_account_id=str(payload["payee_account_id"]),
            amount=float(payload["amount"]),
            typology=ScamTypology(str(payload.get("typology", "unknown")).lower()),
            tenant_id=str(payload.get("tenant_id", "default")),
            status=CoolingOffHoldStatus(str(payload.get("status", "active")).lower()),
            hold_duration_minutes=int(payload.get("hold_duration_minutes", 120)),
            reason=str(payload.get("reason", "")),
            created_at=str(payload.get("created_at", _utc_now_iso())),
            resolved_at=(
                str(payload["resolved_at"]) if payload.get("resolved_at") is not None else None
            ),
            resolution_note=str(payload.get("resolution_note", "")),
        )


@dataclass(slots=True)
class AppScamAssessment:
    """Calibrated APP scam risk assessment and intervention recommendation."""

    payment_id: str
    sender_account_id: str
    payee_account_id: str
    amount: float
    risk_score: float
    risk_level: ScamRiskLevel | str
    typology: ScamTypology | str
    action: ScamInterventionAction | str
    tenant_id: str = "default"
    assessment_id: str = field(default_factory=lambda: f"asa_{uuid4().hex[:12]}")
    signals: list[ScamSignal] = field(default_factory=list)
    customer_warning: str = ""
    hold_id: str | None = None
    case_id: str | None = None
    assessed_at: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.risk_score = round(_clamp(self.risk_score), 4)
        if not isinstance(self.risk_level, ScamRiskLevel):
            self.risk_level = ScamRiskLevel(str(self.risk_level).lower())
        if not isinstance(self.typology, ScamTypology):
            self.typology = ScamTypology(str(self.typology).lower())
        if not isinstance(self.action, ScamInterventionAction):
            self.action = ScamInterventionAction(str(self.action).lower())

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "payment_id": self.payment_id,
            "sender_account_id": self.sender_account_id,
            "payee_account_id": self.payee_account_id,
            "amount": self.amount,
            "risk_score": self.risk_score,
            "risk_level": ScamRiskLevel(self.risk_level).value,
            "typology": ScamTypology(self.typology).value,
            "action": ScamInterventionAction(self.action).value,
            "tenant_id": self.tenant_id,
            "signals": [sig.to_dict() for sig in self.signals],
            "customer_warning": self.customer_warning,
            "hold_id": self.hold_id,
            "case_id": self.case_id,
            "assessed_at": self.assessed_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AppScamAssessment:
        return cls(
            assessment_id=str(payload.get("assessment_id", f"asa_{uuid4().hex[:12]}")),
            payment_id=str(payload["payment_id"]),
            sender_account_id=str(payload["sender_account_id"]),
            payee_account_id=str(payload["payee_account_id"]),
            amount=float(payload["amount"]),
            risk_score=float(payload["risk_score"]),
            risk_level=ScamRiskLevel(str(payload["risk_level"]).lower()),
            typology=ScamTypology(str(payload.get("typology", "unknown")).lower()),
            action=ScamInterventionAction(str(payload["action"]).lower()),
            tenant_id=str(payload.get("tenant_id", "default")),
            signals=[ScamSignal.from_dict(item) for item in payload.get("signals", [])],
            customer_warning=str(payload.get("customer_warning", "")),
            hold_id=str(payload["hold_id"]) if payload.get("hold_id") is not None else None,
            case_id=str(payload["case_id"]) if payload.get("case_id") is not None else None,
            assessed_at=str(payload.get("assessed_at", _utc_now_iso())),
            metadata=dict(payload.get("metadata", {})),
        )


_COERCION_PURPOSE_KEYWORDS = frozenset(
    {"safe_account", "fraud_department", "hmrc", "irs", "police", "tax_fine", "arrest", "court"}
)


class BehavioralCoachingAnalyzer:
    """Detects live social engineering coaching, screen-sharing, and vulnerability cues."""

    def __init__(
        self,
        *,
        hesitation_threshold_seconds: float = 45.0,
        extreme_hesitation_seconds: float = 120.0,
        vulnerable_age_years: int = 65,
        vulnerable_amount_threshold: float = 1000.0,
    ) -> None:
        self.hesitation_threshold_seconds = max(1.0, float(hesitation_threshold_seconds))
        self.extreme_hesitation_seconds = max(
            self.hesitation_threshold_seconds, float(extreme_hesitation_seconds)
        )
        self.vulnerable_age_years = max(18, int(vulnerable_age_years))
        self.vulnerable_amount_threshold = max(1.0, float(vulnerable_amount_threshold))

    def analyze(self, context: AppPaymentContext) -> list[ScamSignal]:
        signals: list[ScamSignal] = []

        if context.screen_share_or_rat_detected or context.active_phone_call:
            if context.screen_share_or_rat_detected and context.active_phone_call:
                severity = 0.95
                detail = "Active voice call combined with remote-access/screen-share session."
            elif context.screen_share_or_rat_detected:
                severity = 0.85
                detail = "Remote-access tool or screen-sharing detected during payment setup."
            else:
                severity = 0.65
                detail = "Customer initiated payment while on an active phone call."
            signals.append(
                ScamSignal(
                    signal_type=ScamSignalType.ACTIVE_CALL_OR_SCREEN_SHARE,
                    severity=severity,
                    detail=detail,
                    evidence={
                        "active_phone_call": context.active_phone_call,
                        "screen_share_or_rat_detected": context.screen_share_or_rat_detected,
                    },
                )
            )

        coercion_purpose = any(
            token in context.stated_purpose for token in _COERCION_PURPOSE_KEYWORDS
        )
        if context.urgency_keywords_detected or coercion_purpose:
            severity = 0.85 if (context.urgency_keywords_detected and coercion_purpose) else 0.6
            signals.append(
                ScamSignal(
                    signal_type=ScamSignalType.URGENCY_OR_SCRIPTED_COACHING,
                    severity=severity,
                    detail="Urgency pressure or scripted safe-account/authority coercion cues.",
                    evidence={
                        "urgency_keywords_detected": context.urgency_keywords_detected,
                        "stated_purpose": context.stated_purpose,
                    },
                )
            )

        if (
            context.payee_details_pasted
            and context.session_hesitation_seconds >= self.hesitation_threshold_seconds
        ) or context.session_hesitation_seconds >= self.extreme_hesitation_seconds:
            severity = (
                0.7
                if (
                    context.payee_details_pasted
                    and context.session_hesitation_seconds >= self.hesitation_threshold_seconds
                )
                else 0.45
            )
            signals.append(
                ScamSignal(
                    signal_type=ScamSignalType.BEHAVIORAL_HESITATION_OR_COPY_PASTE,
                    severity=severity,
                    detail="Copy-pasted beneficiary details or prolonged session hesitation.",
                    evidence={
                        "payee_details_pasted": context.payee_details_pasted,
                        "session_hesitation_seconds": context.session_hesitation_seconds,
                    },
                )
            )

        if (
            context.sender_age_years is not None
            and context.sender_age_years >= self.vulnerable_age_years
            and context.amount >= self.vulnerable_amount_threshold
            and (
                context.is_new_payee
                or context.active_phone_call
                or context.screen_share_or_rat_detected
            )
        ):
            signals.append(
                ScamSignal(
                    signal_type=ScamSignalType.ELDER_OR_VULNERABLE_ANOMALY,
                    severity=0.6,
                    detail="High-value transfer from vulnerable age cohort with scam cues.",
                    evidence={
                        "sender_age_years": context.sender_age_years,
                        "amount": context.amount,
                    },
                )
            )

        return signals
