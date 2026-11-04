"""Card Scheme Chargeback Ingestion, Reason Code Taxonomy & Automated Representment Gateway.

Provides standardized card network dispute ingestion, scheme reason code resolution
(Visa, Mastercard, Amex, Discover), representment viability scoring, CE3.0/First-Party
Trust compelling evidence qualification, and thread-safe dispute lifecycle management.
"""

from __future__ import annotations

import logging
import threading
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


# ---------------------------------------------------------------------------
# Representment Viability Engine (Liability Shift & CE3.0 / First-Party Trust)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepresentmentViability:
    """Outcome of evaluating dispute winnability, liability shift, and evidence."""

    winnability_score: float
    liability_shift_proven: bool
    compelling_evidence_qualified: bool
    recommended_action: RepresentmentAction
    reasons: list[str] = field(default_factory=list)
    evidence_requirements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "winnability_score": round(self.winnability_score, 4),
            "liability_shift_proven": self.liability_shift_proven,
            "compelling_evidence_qualified": self.compelling_evidence_qualified,
            "recommended_action": self.recommended_action.value,
            "reasons": self.reasons,
            "evidence_requirements": self.evidence_requirements,
        }


def evaluate_representment_viability(
    dispute: ChargebackDispute,
    transaction_context: dict[str, Any] | None = None,
    challenge_session: dict[str, Any] | None = None,
    prior_customer_transactions: list[Any] | None = None,
    evidence_package: dict[str, Any] | None = None,
    registry: ReasonCodeRegistry | None = None,
) -> RepresentmentViability:
    """Evaluate dispute winnability, liability shift, and representment recommendation.

    Analyzes 3DS/SCA authentication cryptograms (ECI indicators), Visa Compelling
    Evidence 3.0 (CE3.0) and Mastercard First-Party Trust qualifying historical transactions,
    and compiled dispute evidence dossiers. Updates the dispute record in place.
    """
    reg = registry or DEFAULT_REASON_REGISTRY
    defn = reg.find_or_fallback(dispute.scheme, dispute.reason_code, dispute.category)

    score: float = 0.0
    reasons: list[str] = []
    requirements: list[str] = []
    liability_shift = False
    ce3_qualified = False

    tx_ctx = transaction_context or {}
    ch_sess = challenge_session or {}

    # 1. Evaluate 3-D Secure Liability Shift
    # Valid ECIs: 05/5 (Visa/Amex full 3DS), 02/2 (Mastercard full 3DS), 06/6, 01/1 (attempted)
    eci = str(tx_ctx.get("eci", "")).strip().zfill(2)
    is_3ds_verified = (
        ch_sess.get("status") == "verified" or tx_ctx.get("is_3ds_authenticated") is True
    )
    has_valid_eci = eci in ("05", "02", "06", "01")

    if is_3ds_verified or has_valid_eci:
        liability_shift = True
        score += 0.55
        scheme_name = dispute.scheme.value.upper()
        eci_desc = eci or "verified"
        reasons.append(f"Liability shift proven under {scheme_name} 3DS rules (ECI: {eci_desc})")
    else:
        requirements.append("3DS authentication proof or ECI cryptogram")

    # 2. Evaluate Compelling Evidence (Visa CE3.0 / Mastercard First-Party Trust)
    # Applicable for fraud reason codes (e.g. Visa 10.4, Mastercard 4837)
    if defn.compelling_evidence_eligible and prior_customer_transactions:
        qualifying_txs = 0
        current_dev = tx_ctx.get("device_id") or tx_ctx.get("fingerprint_hash")
        current_ip = tx_ctx.get("ip_address") or tx_ctx.get("client_ip")
        current_user = tx_ctx.get("user_id") or tx_ctx.get("customer_id")

        for prior in prior_customer_transactions:
            if not isinstance(prior, dict):
                continue
            # Must be prior non-fraud transaction
            if prior.get("is_fraud") or prior.get("chargeback") or prior.get("disputed"):
                continue

            matches = 0
            if current_dev and prior.get("device_id") == current_dev:
                matches += 1
            if current_ip and prior.get("ip_address") == current_ip:
                matches += 1
            if current_user and prior.get("user_id") == current_user:
                matches += 1

            if matches >= 1:
                qualifying_txs += 1

        # CE3.0 / First-Party Trust requires at least 2 qualifying prior transactions
        if qualifying_txs >= 2:
            ce3_qualified = True
            score += 0.35
            reasons.append(
                f"Qualified for CE3.0/First-Party Trust ({qualifying_txs} matching prior records)"
            )
        else:
            requirements.append(
                f"Requires at least 2 qualifying prior transactions (found {qualifying_txs})"
            )
    elif defn.compelling_evidence_eligible:
        requirements.append("Prior customer transaction history for CE3.0/First-Party Trust")

    # 3. Evaluate Dispute Evidence Package Dossier (cases.py)
    if evidence_package is not None:
        has_fp = bool(
            evidence_package.get("evidence_fingerprint")
            or evidence_package.get("package_fingerprint")
            or evidence_package.get("sha256_fingerprint")
        )
        if has_fp:
            score += 0.20
            reasons.append("Tamper-evident canonical SHA-256 dispute evidence dossier attached")
        else:
            score += 0.10
            reasons.append("Dispute evidence package provided without cryptographic fingerprint")
    else:
        requirements.append("DisputeEvidencePackage compiled dossier")

    # 4. Reason Category Baseline Adjustments
    if dispute.category == ChargebackCategory.AUTHORIZATION_ERROR:
        if tx_ctx.get("auth_code"):
            score += 0.40
            reasons.append(f"Valid authorization code provided ({tx_ctx.get('auth_code')})")
        else:
            requirements.append("Issuer authorization code proof")
    elif dispute.category == ChargebackCategory.CONSUMER_DISPUTE:
        if tx_ctx.get("tracking_number") or tx_ctx.get("delivery_confirmation"):
            score += 0.35
            reasons.append("Proof of delivery / tracking number available")
        else:
            requirements.append("Proof of merchandise delivery or service fulfillment")

    final_score = round(min(1.0, max(0.0, score)), 4)

    if final_score >= 0.70:
        recommended_action = RepresentmentAction.SUBMIT_REPRESENTMENT
    elif final_score >= 0.35:
        recommended_action = RepresentmentAction.REQUEST_MORE_INFO
    else:
        recommended_action = RepresentmentAction.ACCEPT_LIABILITY

    # Update dispute model attributes
    dispute.winnability_score = final_score
    dispute.liability_shift_proven = liability_shift
    dispute.recommended_action = recommended_action
    if reasons:
        dispute.add_audit_entry(
            action="viability_evaluated",
            actor="system",
            details={
                "score": round(final_score, 4),
                "liability_shift": liability_shift,
                "ce3_qualified": ce3_qualified,
                "recommended_action": recommended_action.value,
            },
        )

    return RepresentmentViability(
        winnability_score=final_score,
        liability_shift_proven=liability_shift,
        compelling_evidence_qualified=ce3_qualified,
        recommended_action=recommended_action,
        reasons=reasons,
        evidence_requirements=requirements,
    )


# ---------------------------------------------------------------------------
# Thread-Safe Chargeback Store & Manager
# ---------------------------------------------------------------------------


class ChargebackStore:
    """Thread-safe, multi-index repository for card scheme disputes."""

    def __init__(self, max_capacity: int = 10000) -> None:
        self._max_capacity = max_capacity
        self._lock = threading.RLock()
        self._disputes: dict[str, ChargebackDispute] = {}
        self._by_tx: dict[str, str] = {}
        self._by_scheme_id: dict[tuple[CardScheme, str], str] = {}

    def put(self, dispute: ChargebackDispute) -> None:
        """Store or update a dispute record with multi-index synchronization."""
        with self._lock:
            if (
                len(self._disputes) >= self._max_capacity
                and dispute.dispute_id not in self._disputes
            ):
                oldest_id = next(iter(self._disputes))
                self.delete(oldest_id)

            self._disputes[dispute.dispute_id] = dispute
            self._by_tx[dispute.transaction_id] = dispute.dispute_id
            if dispute.scheme_dispute_id:
                self._by_scheme_id[(dispute.scheme, dispute.scheme_dispute_id)] = dispute.dispute_id

    def get(self, dispute_id: str) -> ChargebackDispute | None:
        """Retrieve a dispute by dispute ID."""
        with self._lock:
            return self._disputes.get(dispute_id)

    def get_by_transaction(self, transaction_id: str) -> ChargebackDispute | None:
        """Retrieve the latest dispute linked to a transaction ID."""
        with self._lock:
            d_id = self._by_tx.get(transaction_id)
            if d_id:
                return self._disputes.get(d_id)
            return None

    def get_by_scheme_id(
        self, scheme: CardScheme, scheme_dispute_id: str
    ) -> ChargebackDispute | None:
        """Retrieve a dispute by card scheme and scheme reference identifier."""
        with self._lock:
            d_id = self._by_scheme_id.get((scheme, scheme_dispute_id))
            if d_id:
                return self._disputes.get(d_id)
            return None

    def delete(self, dispute_id: str) -> bool:
        """Remove a dispute from all indices."""
        with self._lock:
            dispute = self._disputes.pop(dispute_id, None)
            if dispute is not None:
                if self._by_tx.get(dispute.transaction_id) == dispute_id:
                    self._by_tx.pop(dispute.transaction_id, None)
                if dispute.scheme_dispute_id:
                    key = (dispute.scheme, dispute.scheme_dispute_id)
                    if self._by_scheme_id.get(key) == dispute_id:
                        self._by_scheme_id.pop(key, None)
                return True
            return False

    def list_disputes(
        self,
        tenant_id: str | None = None,
        scheme: CardScheme | None = None,
        category: ChargebackCategory | None = None,
        stage: ChargebackStage | None = None,
        expired_only: bool = False,
        limit: int = 100,
    ) -> list[ChargebackDispute]:
        """Query stored disputes with multi-criteria filtering."""
        with self._lock:
            results: list[ChargebackDispute] = []
            for d in self._disputes.values():
                if tenant_id and d.tenant_id != tenant_id:
                    continue
                if scheme and d.scheme != scheme:
                    continue
                if category and d.category != category:
                    continue
                if stage and d.stage != stage:
                    continue
                if expired_only and not d.is_expired:
                    continue
                results.append(d)
                if len(results) >= limit:
                    break
            results.sort(key=lambda item: item.created_at, reverse=True)
            return results

    def count(self, tenant_id: str | None = None) -> int:
        """Return total stored dispute count, optionally filtered by tenant."""
        with self._lock:
            if tenant_id is None:
                return len(self._disputes)
            return sum(1 for d in self._disputes.values() if d.tenant_id == tenant_id)

    def clear(self) -> None:
        """Flush all stored dispute records and secondary indices."""
        with self._lock:
            self._disputes.clear()
            self._by_tx.clear()
            self._by_scheme_id.clear()


class ChargebackManager:
    """Orchestrates dispute ingestion, representment viability, and lifecycle state."""

    def __init__(
        self,
        store: ChargebackStore | None = None,
        registry: ReasonCodeRegistry | None = None,
    ) -> None:
        self.store = store or ChargebackStore()
        self.registry = registry or DEFAULT_REASON_REGISTRY
        self._lock = threading.RLock()

    def ingest_chargeback(
        self,
        transaction_id: str,
        scheme: CardScheme,
        reason_code: str,
        disputed_amount: float,
        currency: str = "USD",
        tenant_id: str = "default",
        scheme_dispute_id: str | None = None,
        filing_date: str | None = None,
        response_deadline: str | None = None,
        metadata: dict[str, Any] | None = None,
        auto_evaluate: bool = True,
        transaction_context: dict[str, Any] | None = None,
        challenge_session: dict[str, Any] | None = None,
        prior_customer_transactions: list[Any] | None = None,
        evidence_package: dict[str, Any] | None = None,
    ) -> ChargebackDispute:
        """Ingest, validate, and optionally evaluate an incoming card scheme chargeback."""
        with self._lock:
            defn = self.registry.find_or_fallback(scheme, reason_code)
            dispute = ChargebackDispute(
                transaction_id=transaction_id,
                tenant_id=tenant_id,
                scheme=scheme,
                reason_code=reason_code,
                category=defn.category,
                disputed_amount=disputed_amount,
                currency=currency,
                scheme_dispute_id=scheme_dispute_id,
                filing_date=filing_date or "",
                response_deadline=response_deadline or "",
                metadata=metadata or {},
            )

            if auto_evaluate:
                evaluate_representment_viability(
                    dispute=dispute,
                    transaction_context=transaction_context,
                    challenge_session=challenge_session,
                    prior_customer_transactions=prior_customer_transactions,
                    evidence_package=evidence_package,
                    registry=self.registry,
                )

            self.store.put(dispute)
            logger.info(
                "Ingested chargeback dispute %s for tx %s (scheme: %s, code: %s, amount: %s %s)",
                dispute.dispute_id,
                dispute.transaction_id,
                dispute.scheme.value,
                dispute.reason_code,
                dispute.disputed_amount,
                dispute.currency,
            )
            return dispute

    def get_dispute(self, dispute_id: str) -> ChargebackDispute:
        """Retrieve a dispute or raise DisputeNotFoundError."""
        with self._lock:
            dispute = self.store.get(dispute_id)
            if dispute is None:
                raise DisputeNotFoundError(f"Chargeback dispute '{dispute_id}' not found")
            return dispute

    def update_stage(
        self,
        dispute_id: str,
        new_stage: ChargebackStage,
        actor: str = "system",
        notes: str = "",
    ) -> ChargebackDispute:
        """Transition dispute to a new lifecycle stage enforcing state machine rules."""
        with self._lock:
            dispute = self.get_dispute(dispute_id)
            if new_stage == dispute.stage:
                return dispute

            allowed = ALLOWED_DISPUTE_TRANSITIONS.get(dispute.stage, set())
            if new_stage not in allowed:
                raise InvalidDisputeStateTransitionError(
                    f"Illegal transition from '{dispute.stage.value}' to '{new_stage.value}'. "
                    f"Allowed transitions: {[s.value for s in allowed]}"
                )

            old_stage = dispute.stage
            dispute.stage = new_stage
            if dispute.is_terminal and not dispute.resolved_at:
                dispute.resolved_at = datetime.now(UTC).isoformat()

            dispute.add_audit_entry(
                action="stage_transitioned",
                actor=actor,
                details={
                    "old_stage": old_stage.value,
                    "new_stage": new_stage.value,
                    "notes": notes,
                },
            )
            self.store.put(dispute)
            return dispute

    def represent_dispute(
        self,
        dispute_id: str,
        package_id: str | None = None,
        evidence_fingerprint: str | None = None,
        notes: str = "",
        actor: str = "analyst",
    ) -> ChargebackDispute:
        """Submit dispute representment attaching evidence package and cryptographic digest."""
        with self._lock:
            dispute = self.get_dispute(dispute_id)
            if dispute.is_terminal:
                stage_val = dispute.stage.value
                raise InvalidDisputeStateTransitionError(
                    f"Cannot represent terminal dispute '{dispute_id}' in stage '{stage_val}'"
                )

            dispute = self.update_stage(
                dispute_id,
                ChargebackStage.REPRESENTED,
                actor=actor,
                notes=notes,
            )
            dispute.representment_package_id = package_id
            dispute.evidence_fingerprint = evidence_fingerprint
            dispute.representment_notes = notes
            dispute.represented_at = datetime.now(UTC).isoformat()

            dispute.add_audit_entry(
                action="dispute_represented",
                actor=actor,
                details={
                    "package_id": package_id,
                    "evidence_fingerprint": evidence_fingerprint,
                    "notes": notes,
                },
            )
            self.store.put(dispute)
            return dispute

    def resolve_dispute(
        self,
        dispute_id: str,
        outcome: ChargebackStage,
        actor: str = "system",
        notes: str = "",
    ) -> ChargebackDispute:
        """Finalize dispute with a terminal resolution outcome."""
        with self._lock:
            valid_terminal = [
                ChargebackStage.RESOLVED_WON.value,
                ChargebackStage.RESOLVED_LOST.value,
                ChargebackStage.ACCEPTED_LIABILITY.value,
            ]
            if outcome not in (
                ChargebackStage.RESOLVED_WON,
                ChargebackStage.RESOLVED_LOST,
                ChargebackStage.ACCEPTED_LIABILITY,
            ):
                raise DisputeValidationError(
                    f"Invalid outcome '{outcome.value}'. Allowed: {valid_terminal}"
                )

            dispute = self.update_stage(dispute_id, outcome, actor=actor, notes=notes)
            dispute.resolved_at = datetime.now(UTC).isoformat()
            self.store.put(dispute)
            return dispute

    def get_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
        """Aggregate financial exposure, win rates, and stage counts across disputes."""
        with self._lock:
            disputes = self.store.list_disputes(tenant_id=tenant_id, limit=100000)
            total_count = len(disputes)
            total_disputed_amount = sum(d.disputed_amount for d in disputes)

            by_stage = {s.value: 0 for s in ChargebackStage}
            by_scheme = {sc.value: 0 for sc in CardScheme}
            by_category = {c.value: 0 for c in ChargebackCategory}

            expired_count = 0
            liability_shift_count = 0
            ce3_qualified_count = 0

            for d in disputes:
                by_stage[d.stage.value] = by_stage.get(d.stage.value, 0) + 1
                by_scheme[d.scheme.value] = by_scheme.get(d.scheme.value, 0) + 1
                by_category[d.category.value] = by_category.get(d.category.value, 0) + 1
                if d.is_expired:
                    expired_count += 1
                if d.liability_shift_proven:
                    liability_shift_count += 1
                if d.winnability_score >= 0.70:
                    ce3_qualified_count += 1

            return {
                "total_disputes": total_count,
                "total_disputed_amount": round(total_disputed_amount, 2),
                "by_stage": by_stage,
                "by_scheme": by_scheme,
                "by_category": by_category,
                "expired_count": expired_count,
                "liability_shift_proven_count": liability_shift_count,
                "high_winnability_count": ce3_qualified_count,
            }
