"""Authorized Push Payment (APP) Scam & Social Engineering Defense Engine.

Detects genuine-customer authorized push payments initiated under social
engineering coercion or deception (bank/government impersonation, investment
and crypto scams, romance/pig-butchering, invoice redirection / BEC, and
remote-access-tool coaching), applies graduated friction and cooling-off holds,
and escalates high-risk cases to CaseManager and EventBus.
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
