"""Fraud Case Management, investigation SLA tracking, and dispute evidence packaging."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import uuid4


class CaseError(Exception):
    """Base exception for all fraud case management operations."""


class CaseNotFoundError(CaseError):
    """Raised when a requested case ID cannot be found."""


class InvalidCaseStateTransitionError(CaseError):
    """Raised when an illegal case lifecycle state transition is attempted."""


class CaseAssignmentError(CaseError):
    """Raised when an analyst assignment fails validation."""


class EvidencePackagingError(CaseError):
    """Raised when dispute evidence packaging or canonicalization fails."""


class CaseValidationError(CaseError):
    """Raised when case fields violate invariant validation rules."""


class CasePriority(StrEnum):
    """Severity and urgency classification for investigation cases."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def compute_priority(
    fraud_score: float,
    amount: float,
    is_vip: bool = False,
) -> CasePriority:
    """Derive case priority based on model fraud score, transaction amount, and VIP flags.

    Args:
        fraud_score: Predicted probability of fraud (0.0 to 1.0).
        amount: Transaction monetary amount.
        is_vip: Whether customer holds elevated VIP tier.

    Returns:
        Computed CasePriority enum.
    """
    if is_vip and fraud_score >= 0.5:
        return CasePriority.CRITICAL
    if fraud_score >= 0.9 or amount >= 5000.0:
        return CasePriority.CRITICAL
    if fraud_score >= 0.75 or amount >= 1000.0:
        return CasePriority.HIGH
    if fraud_score >= 0.45 or amount >= 250.0:
        return CasePriority.MEDIUM
    return CasePriority.LOW


class CaseStatus(StrEnum):
    """Operational lifecycle states of a fraud investigation case."""

    OPEN = "open"
    INVESTIGATING = "investigating"
    PENDING_CUSTOMER = "pending_customer"
    REPRESENTED = "represented"
    RESOLVED = "resolved"
    CLOSED = "closed"


ALLOWED_STATUS_TRANSITIONS: dict[CaseStatus, set[CaseStatus]] = {
    CaseStatus.OPEN: {CaseStatus.INVESTIGATING, CaseStatus.CLOSED},
    CaseStatus.INVESTIGATING: {
        CaseStatus.PENDING_CUSTOMER,
        CaseStatus.REPRESENTED,
        CaseStatus.RESOLVED,
        CaseStatus.CLOSED,
    },
    CaseStatus.PENDING_CUSTOMER: {
        CaseStatus.INVESTIGATING,
        CaseStatus.RESOLVED,
        CaseStatus.CLOSED,
    },
    CaseStatus.REPRESENTED: {
        CaseStatus.RESOLVED,
        CaseStatus.CLOSED,
    },
    CaseStatus.RESOLVED: {
        CaseStatus.CLOSED,
        CaseStatus.INVESTIGATING,  # Re-opening
    },
    CaseStatus.CLOSED: {
        CaseStatus.INVESTIGATING,  # Re-opening
    },
}


class CaseDisposition(StrEnum):
    """Final resolution classification of an investigated fraud case."""

    UNRESOLVED = "unresolved"
    CONFIRMED_FRAUD = "confirmed_fraud"
    FALSE_POSITIVE = "false_positive"
    MERCHANT_ERROR = "merchant_error"
    ACCOUNT_TAKEOVER = "account_takeover"
    FRIENDLY_FRAUD = "friendly_fraud"


@dataclass
class CaseNote:
    """Immutable analyst commentary or investigation finding."""

    author: str
    text: str
    note_id: str = field(default_factory=lambda: f"note_{uuid4().hex[:12]}")
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if not self.author.strip():
            raise CaseValidationError("Note author cannot be empty")
        if not self.text.strip():
            raise CaseValidationError("Note text cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """Serialize note to JSON-compatible dictionary."""
        return {
            "note_id": self.note_id,
            "author": self.author,
            "text": self.text,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseNote:
        """Hydrate note from dictionary representation."""
        return cls(
            note_id=str(data.get("note_id") or f"note_{uuid4().hex[:12]}"),
            author=str(data.get("author", "")),
            text=str(data.get("text", "")),
            timestamp=str(data.get("timestamp") or datetime.now(UTC).isoformat()),
        )


@dataclass
class FraudCase:
    """Full operational record representing a suspicious transaction investigation."""

    transaction_id: str
    case_id: str = field(default_factory=lambda: f"case_{uuid4().hex[:12]}")
    tenant_id: str = "default"
    assigned_analyst: str | None = None
    status: CaseStatus = CaseStatus.OPEN
    priority: CasePriority = CasePriority.MEDIUM
    fraud_score: float = 0.0
    amount: float = 0.0
    currency: str = "USD"
    disposition: CaseDisposition = CaseDisposition.UNRESOLVED
    notes: list[CaseNote] = field(default_factory=list)
    audit_trail: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    sla_deadline_hours: float = 24.0
    sla_expires_at: str = field(default="")
    resolved_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.transaction_id or not self.transaction_id.strip():
            raise CaseValidationError("transaction_id must be a non-empty string")
        if not self.case_id or not self.case_id.strip():
            raise CaseValidationError("case_id must be a non-empty string")
        if not self.tenant_id or not self.tenant_id.strip():
            raise CaseValidationError("tenant_id must be a non-empty string")
        if self.fraud_score < 0.0 or self.fraud_score > 1.0:
            raise CaseValidationError("fraud_score must be between 0.0 and 1.0")
        if self.amount < 0.0:
            raise CaseValidationError("amount cannot be negative")
        if self.sla_deadline_hours <= 0.0:
            raise CaseValidationError("sla_deadline_hours must be strictly positive")

        # Compute default sla_expires_at if not populated
        if not self.sla_expires_at:
            try:
                base_dt = datetime.fromisoformat(self.created_at)
            except (ValueError, TypeError):
                base_dt = datetime.now(UTC)
            self.sla_expires_at = (base_dt + timedelta(hours=self.sla_deadline_hours)).isoformat()

        # Seed initial audit trail if empty
        if not self.audit_trail:
            self.add_audit_entry(
                action="case_created",
                actor="system",
                details={
                    "priority": self.priority.value,
                    "status": self.status.value,
                    "fraud_score": self.fraud_score,
                },
            )

    @property
    def is_sla_breached(self) -> bool:
        """Evaluate if the case is currently past its SLA deadline without resolution."""
        if self.status in (CaseStatus.RESOLVED, CaseStatus.CLOSED):
            return False
        try:
            deadline = datetime.fromisoformat(self.sla_expires_at)
            now = datetime.now(UTC)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=UTC)
            return now > deadline
        except (ValueError, TypeError):
            return False

    @property
    def is_resolved(self) -> bool:
        """Check whether the case has reached a terminal resolved or closed status."""
        return self.status in (CaseStatus.RESOLVED, CaseStatus.CLOSED)

    def add_audit_entry(
        self,
        action: str,
        actor: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append an immutable audit trail entry tracking state or attribute changes."""
        now_str = datetime.now(UTC).isoformat()
        entry: dict[str, Any] = {
            "entry_id": f"aud_{uuid4().hex[:12]}",
            "action": action,
            "actor": actor,
            "timestamp": now_str,
            "details": details or {},
        }
        self.audit_trail.append(entry)
        self.updated_at = now_str

    def to_dict(self) -> dict[str, Any]:
        """Serialize case model to full JSON-compatible dictionary."""
        return {
            "case_id": self.case_id,
            "transaction_id": self.transaction_id,
            "tenant_id": self.tenant_id,
            "assigned_analyst": self.assigned_analyst,
            "status": self.status.value,
            "priority": self.priority.value,
            "fraud_score": self.fraud_score,
            "amount": self.amount,
            "currency": self.currency,
            "disposition": self.disposition.value,
            "notes": [n.to_dict() for n in self.notes],
            "audit_trail": list(self.audit_trail),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sla_deadline_hours": self.sla_deadline_hours,
            "sla_expires_at": self.sla_expires_at,
            "is_sla_breached": self.is_sla_breached,
            "resolved_at": self.resolved_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FraudCase:
        """Hydrate case instance from dictionary representation."""
        status_val = data.get("status", CaseStatus.OPEN.value)
        status = CaseStatus(status_val) if isinstance(status_val, str) else CaseStatus.OPEN

        prio_val = data.get("priority", CasePriority.MEDIUM.value)
        priority = CasePriority(prio_val) if isinstance(prio_val, str) else CasePriority.MEDIUM

        disp_val = data.get("disposition", CaseDisposition.UNRESOLVED.value)
        disposition = (
            CaseDisposition(disp_val) if isinstance(disp_val, str) else CaseDisposition.UNRESOLVED
        )

        notes_raw = data.get("notes") or []
        notes = [CaseNote.from_dict(n) if isinstance(n, dict) else n for n in notes_raw]

        return cls(
            case_id=str(data.get("case_id", "")),
            transaction_id=str(data.get("transaction_id", "")),
            tenant_id=str(data.get("tenant_id", "default")),
            assigned_analyst=data.get("assigned_analyst"),
            status=status,
            priority=priority,
            fraud_score=float(data.get("fraud_score", 0.0)),
            amount=float(data.get("amount", 0.0)),
            currency=str(data.get("currency", "USD")),
            disposition=disposition,
            notes=notes,
            audit_trail=list(data.get("audit_trail") or []),
            created_at=str(data.get("created_at") or datetime.now(UTC).isoformat()),
            updated_at=str(data.get("updated_at") or datetime.now(UTC).isoformat()),
            sla_deadline_hours=float(data.get("sla_deadline_hours", 24.0)),
            sla_expires_at=str(data.get("sla_expires_at", "")),
            resolved_at=data.get("resolved_at"),
            metadata=dict(data.get("metadata") or {}),
        )
