"""Card Scheme Chargeback Ingestion, Reason Code Taxonomy & Automated Representment Gateway.

Provides standardized card network dispute ingestion, scheme reason code resolution
(Visa, Mastercard, Amex, Discover), representment viability scoring, CE3.0/First-Party
Trust compelling evidence qualification, and thread-safe dispute lifecycle management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ChargebackError(Exception):
    """Base exception for all chargeback and representment gateway errors."""


class DisputeNotFoundError(ChargebackError):
    """Raised when a referenced dispute ID does not exist."""


class InvalidDisputeStateTransitionError(ChargebackError):
    """Raised when an illegal dispute lifecycle transition is attempted."""


class DisputeValidationError(ChargebackError):
    """Raised when dispute parameters fail integrity or business validation."""


class RepresentmentError(ChargebackError):
    """Raised when evidence packaging or representment submission fails."""


class ReasonCodeNotFoundError(ChargebackError):
    """Raised when an unrecognized scheme reason code is queried."""


# ---------------------------------------------------------------------------
# Enums & Taxonomy
# ---------------------------------------------------------------------------


class CardScheme(StrEnum):
    """Major card schemes recognized in dispute representment."""

    VISA = "visa"
    MASTERCARD = "mastercard"
    AMEX = "amex"
    DISCOVER = "discover"
    JCB = "jcb"
    OTHER = "other"


class ChargebackCategory(StrEnum):
    """Normalized dispute reason categories spanning all card networks."""

    FRAUD_CARD_NOT_PRESENT = "fraud_card_not_present"
    FRAUD_CARD_PRESENT = "fraud_card_present"
    FRIENDLY_FRAUD_FIRST_PARTY = "friendly_fraud_first_party"
    UNRECOGNIZED_TRANSACTION = "unrecognized_transaction"
    AUTHORIZATION_ERROR = "authorization_error"
    PROCESSING_ERROR = "processing_error"
    CONSUMER_DISPUTE = "consumer_dispute"
    OTHER = "other"


class ChargebackStage(StrEnum):
    """Formal lifecycle progression stages of a card scheme dispute."""

    RETRIEVAL_REQUEST = "retrieval_request"
    FIRST_CHARGEBACK = "first_chargeback"
    PRE_ARBITRATION = "pre_arbitration"
    ARBITRATION = "arbitration"
    REPRESENTED = "represented"
    RESOLVED_WON = "resolved_won"
    RESOLVED_LOST = "resolved_lost"
    ACCEPTED_LIABILITY = "accepted_liability"
    EXPIRED = "expired"


class RepresentmentAction(StrEnum):
    """Automated operational recommendation for handling an incoming dispute."""

    SUBMIT_REPRESENTMENT = "submit_representment"
    ACCEPT_LIABILITY = "accept_liability"
    REQUEST_MORE_INFO = "request_more_info"


# Finite State Machine for Scheme Disputes
ALLOWED_DISPUTE_TRANSITIONS: dict[ChargebackStage, set[ChargebackStage]] = {
    ChargebackStage.RETRIEVAL_REQUEST: {
        ChargebackStage.FIRST_CHARGEBACK,
        ChargebackStage.RESOLVED_WON,
        ChargebackStage.ACCEPTED_LIABILITY,
        ChargebackStage.EXPIRED,
    },
    ChargebackStage.FIRST_CHARGEBACK: {
        ChargebackStage.REPRESENTED,
        ChargebackStage.ACCEPTED_LIABILITY,
        ChargebackStage.EXPIRED,
    },
    ChargebackStage.REPRESENTED: {
        ChargebackStage.PRE_ARBITRATION,
        ChargebackStage.RESOLVED_WON,
        ChargebackStage.RESOLVED_LOST,
    },
    ChargebackStage.PRE_ARBITRATION: {
        ChargebackStage.ARBITRATION,
        ChargebackStage.RESOLVED_WON,
        ChargebackStage.RESOLVED_LOST,
        ChargebackStage.ACCEPTED_LIABILITY,
    },
    ChargebackStage.ARBITRATION: {
        ChargebackStage.RESOLVED_WON,
        ChargebackStage.RESOLVED_LOST,
    },
    ChargebackStage.RESOLVED_WON: set(),
    ChargebackStage.RESOLVED_LOST: set(),
    ChargebackStage.ACCEPTED_LIABILITY: set(),
    ChargebackStage.EXPIRED: set(),
}


# ---------------------------------------------------------------------------
# Reason Code Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReasonCodeDefinition:
    """Standardized scheme reason code metadata and representment requirements."""

    scheme: CardScheme
    code: str
    description: str
    category: ChargebackCategory
    default_response_days: int
    compelling_evidence_eligible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme.value,
            "code": self.code,
            "description": self.description,
            "category": self.category.value,
            "default_response_days": self.default_response_days,
            "compelling_evidence_eligible": self.compelling_evidence_eligible,
        }


class ReasonCodeRegistry:
    """Central catalog of card scheme dispute reason codes and rules."""

    def __init__(self) -> None:
        self._registry: dict[tuple[CardScheme, str], ReasonCodeDefinition] = {}
        self._populate_defaults()

    def _populate_defaults(self) -> None:
        defaults = [
            # Visa Reason Codes
            ReasonCodeDefinition(
                scheme=CardScheme.VISA,
                code="10.4",
                description="Other Fraud - Card-Absent Environment",
                category=ChargebackCategory.FRAUD_CARD_NOT_PRESENT,
                default_response_days=30,
                compelling_evidence_eligible=True,
            ),
            ReasonCodeDefinition(
                scheme=CardScheme.VISA,
                code="10.5",
                description="Visa Fraud Monitoring Program",
                category=ChargebackCategory.FRAUD_CARD_NOT_PRESENT,
                default_response_days=30,
                compelling_evidence_eligible=False,
            ),
            ReasonCodeDefinition(
                scheme=CardScheme.VISA,
                code="10.3",
                description="Other Fraud - Card-Present Environment",
                category=ChargebackCategory.FRAUD_CARD_PRESENT,
                default_response_days=30,
                compelling_evidence_eligible=False,
            ),
            ReasonCodeDefinition(
                scheme=CardScheme.VISA,
                code="13.1",
                description="Merchandise/Services Not Received",
                category=ChargebackCategory.CONSUMER_DISPUTE,
                default_response_days=30,
                compelling_evidence_eligible=True,
            ),
            ReasonCodeDefinition(
                scheme=CardScheme.VISA,
                code="11.3",
                description="No Authorization",
                category=ChargebackCategory.AUTHORIZATION_ERROR,
                default_response_days=30,
                compelling_evidence_eligible=False,
            ),
            # Mastercard Reason Codes
            ReasonCodeDefinition(
                scheme=CardScheme.MASTERCARD,
                code="4837",
                description="No Cardholder Authorization",
                category=ChargebackCategory.FRAUD_CARD_NOT_PRESENT,
                default_response_days=45,
                compelling_evidence_eligible=True,
            ),
            ReasonCodeDefinition(
                scheme=CardScheme.MASTERCARD,
                code="4863",
                description="Cardholder Does Not Recognize - Potential Fraud",
                category=ChargebackCategory.UNRECOGNIZED_TRANSACTION,
                default_response_days=45,
                compelling_evidence_eligible=True,
            ),
            ReasonCodeDefinition(
                scheme=CardScheme.MASTERCARD,
                code="4853",
                description="Cardholder Dispute - Defective/Not as Described",
                category=ChargebackCategory.CONSUMER_DISPUTE,
                default_response_days=45,
                compelling_evidence_eligible=True,
            ),
            ReasonCodeDefinition(
                scheme=CardScheme.MASTERCARD,
                code="4808",
                description="Authorization-Related Chargeback",
                category=ChargebackCategory.AUTHORIZATION_ERROR,
                default_response_days=45,
                compelling_evidence_eligible=False,
            ),
            # American Express Reason Codes
            ReasonCodeDefinition(
                scheme=CardScheme.AMEX,
                code="FR2",
                description="Fraud Full Recourse",
                category=ChargebackCategory.FRAUD_CARD_NOT_PRESENT,
                default_response_days=20,
                compelling_evidence_eligible=True,
            ),
            ReasonCodeDefinition(
                scheme=CardScheme.AMEX,
                code="F29",
                description="Card Not Present Fraud",
                category=ChargebackCategory.FRAUD_CARD_NOT_PRESENT,
                default_response_days=20,
                compelling_evidence_eligible=True,
            ),
            ReasonCodeDefinition(
                scheme=CardScheme.AMEX,
                code="C08",
                description="Goods/Services Not Received",
                category=ChargebackCategory.CONSUMER_DISPUTE,
                default_response_days=20,
                compelling_evidence_eligible=True,
            ),
            # Discover Reason Codes
            ReasonCodeDefinition(
                scheme=CardScheme.DISCOVER,
                code="7030",
                description="Fraud - Card Not Present Transaction",
                category=ChargebackCategory.FRAUD_CARD_NOT_PRESENT,
                default_response_days=30,
                compelling_evidence_eligible=True,
            ),
            ReasonCodeDefinition(
                scheme=CardScheme.DISCOVER,
                code="4755",
                description="Unauthorized Purchase",
                category=ChargebackCategory.FRAUD_CARD_NOT_PRESENT,
                default_response_days=30,
                compelling_evidence_eligible=True,
            ),
        ]
        for item in defaults:
            self._registry[(item.scheme, item.code.upper())] = item

    def register(self, definition: ReasonCodeDefinition) -> None:
        """Add or override a reason code definition in the registry."""
        self._registry[(definition.scheme, definition.code.upper())] = definition

    def lookup(self, scheme: CardScheme, code: str) -> ReasonCodeDefinition:
        """Resolve a reason code definition or raise ReasonCodeNotFoundError."""
        key = (scheme, code.strip().upper())
        item = self._registry.get(key)
        if item is None:
            raise ReasonCodeNotFoundError(
                f"Reason code '{code}' not registered for scheme '{scheme.value}'"
            )
        return item

    def find_or_fallback(
        self,
        scheme: CardScheme,
        code: str,
        fallback_category: ChargebackCategory = ChargebackCategory.OTHER,
    ) -> ReasonCodeDefinition:
        """Resolve definition if registered, else build a dynamic fallback definition."""
        try:
            return self.lookup(scheme, code)
        except ReasonCodeNotFoundError:
            return ReasonCodeDefinition(
                scheme=scheme,
                code=code.strip().upper(),
                description=f"Unregistered reason code {code}",
                category=fallback_category,
                default_response_days=30,
                compelling_evidence_eligible=False,
            )

    def list_codes(self, scheme: CardScheme | None = None) -> list[ReasonCodeDefinition]:
        """Return all registered reason code definitions, optionally filtered by scheme."""
        if scheme is not None:
            return [defn for (s, _), defn in self._registry.items() if s == scheme]
        return list(self._registry.values())


# Global default registry instance
DEFAULT_REASON_REGISTRY = ReasonCodeRegistry()


# ---------------------------------------------------------------------------
# Chargeback Dispute Model
# ---------------------------------------------------------------------------


@dataclass
class ChargebackDispute:
    """Core domain model representing a card network chargeback dispute."""

    transaction_id: str
    tenant_id: str = "default"
    scheme: CardScheme = CardScheme.VISA
    reason_code: str = "10.4"
    category: ChargebackCategory = ChargebackCategory.FRAUD_CARD_NOT_PRESENT
    disputed_amount: float = 0.0
    currency: str = "USD"
    scheme_dispute_id: str | None = None
    dispute_id: str = field(default_factory=lambda: f"dsp_{uuid4().hex[:12]}")
    stage: ChargebackStage = ChargebackStage.FIRST_CHARGEBACK
    filing_date: str = ""
    response_deadline: str = ""
    liability_shift_proven: bool = False
    winnability_score: float = 0.0
    recommended_action: RepresentmentAction = RepresentmentAction.REQUEST_MORE_INFO
    representment_package_id: str | None = None
    evidence_fingerprint: str | None = None
    representment_notes: str = ""
    represented_at: str | None = None
    resolved_at: str | None = None
    audit_trail: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.transaction_id or not self.transaction_id.strip():
            raise DisputeValidationError("transaction_id cannot be empty")
        if self.disputed_amount < 0.0:
            raise DisputeValidationError("disputed_amount cannot be negative")

        now_str = datetime.now(UTC).isoformat()
        if not self.created_at:
            self.created_at = now_str
        if not self.updated_at:
            self.updated_at = now_str
        if not self.filing_date:
            self.filing_date = now_str

        # Compute default response deadline if absent
        if not self.response_deadline:
            try:
                defn = DEFAULT_REASON_REGISTRY.find_or_fallback(self.scheme, self.reason_code)
                days = defn.default_response_days
            except (ChargebackError, KeyError):
                days = 30
            try:
                base_dt = datetime.fromisoformat(self.filing_date)
            except (ValueError, TypeError):
                base_dt = datetime.now(UTC)
            self.response_deadline = (base_dt + timedelta(days=days)).isoformat()

        if not self.audit_trail:
            self.add_audit_entry(
                action="dispute_ingested",
                actor="system",
                details={
                    "scheme": self.scheme.value,
                    "reason_code": self.reason_code,
                    "amount": self.disputed_amount,
                    "stage": self.stage.value,
                },
            )

    @property
    def is_expired(self) -> bool:
        """Check if the dispute deadline has passed without representment."""
        if self.stage in (
            ChargebackStage.RESOLVED_WON,
            ChargebackStage.RESOLVED_LOST,
            ChargebackStage.ACCEPTED_LIABILITY,
            ChargebackStage.EXPIRED,
            ChargebackStage.REPRESENTED,
        ):
            return False
        try:
            deadline = datetime.fromisoformat(self.response_deadline)
            now = datetime.now(UTC)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=UTC)
            return now > deadline
        except (ValueError, TypeError):
            return False

    @property
    def is_terminal(self) -> bool:
        """Check if the dispute has reached an immutable outcome."""
        return self.stage in (
            ChargebackStage.RESOLVED_WON,
            ChargebackStage.RESOLVED_LOST,
            ChargebackStage.ACCEPTED_LIABILITY,
            ChargebackStage.EXPIRED,
        )

    def add_audit_entry(
        self, action: str, actor: str, details: dict[str, Any] | None = None
    ) -> None:
        """Record an immutable audit trail entry."""
        entry_time = datetime.now(UTC).isoformat()
        self.audit_trail.append(
            {
                "entry_id": f"aud_{uuid4().hex[:12]}",
                "action": action,
                "actor": actor,
                "timestamp": entry_time,
                "details": details or {},
            }
        )
        self.updated_at = entry_time

    def to_dict(self) -> dict[str, Any]:
        """Serialize dispute record to JSON-compatible dictionary."""
        return {
            "dispute_id": self.dispute_id,
            "transaction_id": self.transaction_id,
            "tenant_id": self.tenant_id,
            "scheme": self.scheme.value,
            "scheme_dispute_id": self.scheme_dispute_id,
            "reason_code": self.reason_code,
            "category": self.category.value,
            "disputed_amount": self.disputed_amount,
            "currency": self.currency,
            "stage": self.stage.value,
            "filing_date": self.filing_date,
            "response_deadline": self.response_deadline,
            "is_expired": self.is_expired,
            "is_terminal": self.is_terminal,
            "liability_shift_proven": self.liability_shift_proven,
            "winnability_score": round(self.winnability_score, 4),
            "recommended_action": self.recommended_action.value,
            "representment_package_id": self.representment_package_id,
            "evidence_fingerprint": self.evidence_fingerprint,
            "representment_notes": self.representment_notes,
            "represented_at": self.represented_at,
            "resolved_at": self.resolved_at,
            "audit_trail": self.audit_trail,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChargebackDispute:
        """Hydrate dispute record from dictionary."""
        d = dict(data)
        scheme = CardScheme(d.get("scheme", CardScheme.VISA.value))
        cat = ChargebackCategory(d.get("category", ChargebackCategory.OTHER.value))
        stage = ChargebackStage(d.get("stage", ChargebackStage.FIRST_CHARGEBACK.value))
        rec_act = RepresentmentAction(
            d.get("recommended_action", RepresentmentAction.REQUEST_MORE_INFO.value)
        )

        return cls(
            transaction_id=str(d.get("transaction_id", "")),
            tenant_id=str(d.get("tenant_id", "default")),
            scheme=scheme,
            reason_code=str(d.get("reason_code", "10.4")),
            category=cat,
            disputed_amount=float(d.get("disputed_amount", 0.0)),
            currency=str(d.get("currency", "USD")),
            scheme_dispute_id=d.get("scheme_dispute_id"),
            dispute_id=str(d.get("dispute_id", f"dsp_{uuid4().hex[:12]}")),
            stage=stage,
            filing_date=str(d.get("filing_date", "")),
            response_deadline=str(d.get("response_deadline", "")),
            liability_shift_proven=bool(d.get("liability_shift_proven", False)),
            winnability_score=float(d.get("winnability_score", 0.0)),
            recommended_action=rec_act,
            representment_package_id=d.get("representment_package_id"),
            evidence_fingerprint=d.get("evidence_fingerprint"),
            representment_notes=str(d.get("representment_notes", "")),
            represented_at=d.get("represented_at"),
            resolved_at=d.get("resolved_at"),
            audit_trail=list(d.get("audit_trail", [])),
            metadata=dict(d.get("metadata", {})),
            created_at=str(d.get("created_at", "")),
            updated_at=str(d.get("updated_at", "")),
        )
