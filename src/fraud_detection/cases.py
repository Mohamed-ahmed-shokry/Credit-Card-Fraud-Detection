"""Fraud Case Management, investigation SLA tracking, and dispute evidence packaging."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
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


@dataclass
class DisputeEvidencePackage:
    """Cryptographic, tamper-evident dispute evidence dossier for card network representment."""

    case_id: str
    transaction_id: str
    tenant_id: str
    evidence_items: dict[str, Any]
    package_id: str = field(default_factory=lambda: f"evpkg_{uuid4().hex[:12]}")
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    device_fingerprint: dict[str, Any] | None = None
    geolocation_data: dict[str, Any] | None = None
    challenge_verification: dict[str, Any] | None = None
    merchant_profile: dict[str, Any] | None = None
    sha256_fingerprint: str = field(default="")

    def __post_init__(self) -> None:
        if not self.case_id or not self.case_id.strip():
            raise EvidencePackagingError("case_id must be a non-empty string")
        if not self.transaction_id or not self.transaction_id.strip():
            raise EvidencePackagingError("transaction_id must be a non-empty string")
        if not self.tenant_id or not self.tenant_id.strip():
            raise EvidencePackagingError("tenant_id must be a non-empty string")

        if not self.sha256_fingerprint:
            self.sha256_fingerprint = self.compute_fingerprint()

    def _canonical_payload(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "case_id": self.case_id,
            "transaction_id": self.transaction_id,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at,
            "evidence_items": self.evidence_items,
            "device_fingerprint": self.device_fingerprint,
            "geolocation_data": self.geolocation_data,
            "challenge_verification": self.challenge_verification,
            "merchant_profile": self.merchant_profile,
        }

    def to_canonical_json(self) -> str:
        """Produce deterministic sorted canonical JSON representation for hashing and export."""
        return json.dumps(
            self._canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )

    def compute_fingerprint(self) -> str:
        """Compute SHA-256 cryptographic digest across all canonical evidence attributes."""
        canonical_bytes = self.to_canonical_json().encode("utf-8")
        return hashlib.sha256(canonical_bytes).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify that the evidence package payload matches the cryptographic SHA-256 digest."""
        calculated = self.compute_fingerprint()
        return hmac.compare_digest(calculated, self.sha256_fingerprint)

    def to_dict(self) -> dict[str, Any]:
        """Serialize evidence package to JSON dictionary."""
        payload = self._canonical_payload()
        payload["sha256_fingerprint"] = self.sha256_fingerprint
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DisputeEvidencePackage:
        """Hydrate evidence package from dictionary representation."""
        return cls(
            package_id=str(data.get("package_id") or f"evpkg_{uuid4().hex[:12]}"),
            case_id=str(data.get("case_id", "")),
            transaction_id=str(data.get("transaction_id", "")),
            tenant_id=str(data.get("tenant_id", "default")),
            created_at=str(data.get("created_at") or datetime.now(UTC).isoformat()),
            evidence_items=dict(data.get("evidence_items") or {}),
            device_fingerprint=data.get("device_fingerprint"),
            geolocation_data=data.get("geolocation_data"),
            challenge_verification=data.get("challenge_verification"),
            merchant_profile=data.get("merchant_profile"),
            sha256_fingerprint=str(data.get("sha256_fingerprint", "")),
        )


def compile_evidence_package(
    case: FraudCase,
    transaction_data: dict[str, Any],
    device_data: dict[str, Any] | None = None,
    geo_data: dict[str, Any] | None = None,
    challenge_data: dict[str, Any] | None = None,
    merchant_data: dict[str, Any] | None = None,
    additional_evidence: dict[str, Any] | None = None,
) -> DisputeEvidencePackage:
    """Compile multi-modal fraud evaluation evidence into a tamper-evident dispute dossier.

    Args:
        case: FraudCase instance being represented or investigated.
        transaction_data: Base transaction attributes (amount, currency, timestamps).
        device_data: Optional device fingerprint metadata (Phase 40).
        geo_data: Optional geolocation velocity metrics (Phase 39).
        challenge_data: Optional step-up authentication verification proof (Phase 44).
        merchant_data: Optional merchant risk profiling metrics (Phase 41).
        additional_evidence: Optional custom evidence key-values (receipts, delivery proof).

    Returns:
        Fingerprinted DisputeEvidencePackage.
    """
    evidence_items = {
        "case_priority": case.priority.value,
        "case_fraud_score": case.fraud_score,
        "case_disposition": case.disposition.value,
        "transaction": dict(transaction_data),
        "notes_count": len(case.notes),
        "audit_events_count": len(case.audit_trail),
    }
    if additional_evidence:
        evidence_items.update(additional_evidence)

    return DisputeEvidencePackage(
        case_id=case.case_id,
        transaction_id=case.transaction_id,
        tenant_id=case.tenant_id,
        evidence_items=evidence_items,
        device_fingerprint=dict(device_data) if device_data is not None else None,
        geolocation_data=dict(geo_data) if geo_data is not None else None,
        challenge_verification=dict(challenge_data) if challenge_data is not None else None,
        merchant_profile=dict(merchant_data) if merchant_data is not None else None,
    )


class CaseStore:
    """Thread-safe multi-index in-memory store for fraud cases and dispute packages."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cases: dict[str, FraudCase] = {}
        self._by_tx: dict[str, str] = {}
        self._evidence_packages: dict[str, DisputeEvidencePackage] = {}
        self._pkg_by_case: dict[str, list[str]] = {}

    def put(self, case: FraudCase) -> None:
        """Store or update a FraudCase instance."""
        with self._lock:
            self._cases[case.case_id] = case
            self._by_tx[case.transaction_id] = case.case_id

    def get(self, case_id: str) -> FraudCase | None:
        """Retrieve a case by its unique case_id."""
        with self._lock:
            return self._cases.get(case_id)

    def get_by_transaction(self, transaction_id: str) -> FraudCase | None:
        """Retrieve the latest case associated with a transaction_id."""
        with self._lock:
            case_id = self._by_tx.get(transaction_id)
            if case_id:
                return self._cases.get(case_id)
            return None

    def delete(self, case_id: str) -> bool:
        """Remove a case by ID from the store."""
        with self._lock:
            case = self._cases.pop(case_id, None)
            if case is not None:
                if self._by_tx.get(case.transaction_id) == case_id:
                    self._by_tx.pop(case.transaction_id, None)
                self._pkg_by_case.pop(case_id, None)
                return True
            return False

    def list_cases(
        self,
        tenant_id: str | None = None,
        status: CaseStatus | None = None,
        priority: CasePriority | None = None,
        assigned_analyst: str | None = None,
        sla_breached_only: bool = False,
        limit: int = 100,
    ) -> list[FraudCase]:
        """Query and filter cases by tenant, status, priority, analyst, and SLA."""
        with self._lock:
            results: list[FraudCase] = []
            for case in self._cases.values():
                if tenant_id and case.tenant_id != tenant_id:
                    continue
                if status and case.status != status:
                    continue
                if priority and case.priority != priority:
                    continue
                if assigned_analyst and case.assigned_analyst != assigned_analyst:
                    continue
                if sla_breached_only and not case.is_sla_breached:
                    continue
                results.append(case)
                if len(results) >= limit:
                    break
            # Sort by created_at descending
            results.sort(key=lambda c: c.created_at, reverse=True)
            return results

    def put_evidence_package(self, pkg: DisputeEvidencePackage) -> None:
        """Index and store a DisputeEvidencePackage."""
        with self._lock:
            self._evidence_packages[pkg.package_id] = pkg
            if pkg.case_id not in self._pkg_by_case:
                self._pkg_by_case[pkg.case_id] = []
            if pkg.package_id not in self._pkg_by_case[pkg.case_id]:
                self._pkg_by_case[pkg.case_id].append(pkg.package_id)

    def get_evidence_package(self, package_id: str) -> DisputeEvidencePackage | None:
        """Retrieve an evidence package by ID."""
        with self._lock:
            return self._evidence_packages.get(package_id)

    def list_evidence_packages(self, case_id: str) -> list[DisputeEvidencePackage]:
        """List all evidence packages compiled for a given case."""
        with self._lock:
            pkg_ids = self._pkg_by_case.get(case_id, [])
            return [
                self._evidence_packages[pid] for pid in pkg_ids if pid in self._evidence_packages
            ]

    def count(self) -> int:
        """Total number of active cases in store."""
        with self._lock:
            return len(self._cases)

    def clear(self) -> None:
        """Reset all case and package storage."""
        with self._lock:
            self._cases.clear()
            self._by_tx.clear()
            self._evidence_packages.clear()
            self._pkg_by_case.clear()


class CaseManager:
    """Orchestrates case lifecycle, analyst assignment, SLA queries, and evidence generation."""

    def __init__(self, store: CaseStore | None = None) -> None:
        self.store = store or CaseStore()
        self._lock = threading.RLock()

    def create_case(
        self,
        transaction_id: str,
        fraud_score: float = 0.0,
        amount: float = 0.0,
        currency: str = "USD",
        tenant_id: str = "default",
        priority: CasePriority | None = None,
        sla_deadline_hours: float = 24.0,
        metadata: dict[str, Any] | None = None,
        auto_assign: str | None = None,
    ) -> FraudCase:
        """Create and persist a new fraud case."""
        with self._lock:
            resolved_priority = priority or compute_priority(fraud_score, amount)
            case = FraudCase(
                transaction_id=transaction_id,
                tenant_id=tenant_id,
                fraud_score=fraud_score,
                amount=amount,
                currency=currency,
                priority=resolved_priority,
                sla_deadline_hours=sla_deadline_hours,
                metadata=metadata or {},
            )
            if auto_assign:
                case.assigned_analyst = auto_assign
                case.status = CaseStatus.INVESTIGATING
                case.add_audit_entry(
                    action="case_assigned",
                    actor="system",
                    details={"analyst": auto_assign},
                )
            self.store.put(case)
            return case

    def get_case(self, case_id: str) -> FraudCase:
        """Retrieve case by ID or raise CaseNotFoundError."""
        with self._lock:
            case = self.store.get(case_id)
            if case is None:
                raise CaseNotFoundError(f"Case '{case_id}' not found")
            return case

    def assign_case(
        self,
        case_id: str,
        analyst: str,
        actor: str = "system",
    ) -> FraudCase:
        """Assign an analyst to an active case, transitioning from OPEN to INVESTIGATING."""
        with self._lock:
            case = self.get_case(case_id)
            if not analyst or not analyst.strip():
                raise CaseAssignmentError("Analyst identifier cannot be empty")

            old_analyst = case.assigned_analyst
            case.assigned_analyst = analyst.strip()
            if case.status == CaseStatus.OPEN:
                case.status = CaseStatus.INVESTIGATING

            case.add_audit_entry(
                action="case_assigned",
                actor=actor,
                details={"old_analyst": old_analyst, "new_analyst": case.assigned_analyst},
            )
            self.store.put(case)
            return case

    def update_status(
        self,
        case_id: str,
        new_status: CaseStatus,
        actor: str,
        reason: str = "",
    ) -> FraudCase:
        """Transition case to a new lifecycle status enforcing state machine rules."""
        with self._lock:
            case = self.get_case(case_id)
            if new_status == case.status:
                return case

            allowed = ALLOWED_STATUS_TRANSITIONS.get(case.status, set())
            if new_status not in allowed:
                raise InvalidCaseStateTransitionError(
                    f"Illegal transition from '{case.status.value}' to '{new_status.value}'. "
                    f"Allowed: {[s.value for s in allowed]}"
                )

            old_status = case.status
            case.status = new_status
            if new_status == CaseStatus.RESOLVED and not case.resolved_at:
                case.resolved_at = datetime.now(UTC).isoformat()

            case.add_audit_entry(
                action="status_changed",
                actor=actor,
                details={
                    "old_status": old_status.value,
                    "new_status": new_status.value,
                    "reason": reason,
                },
            )
            self.store.put(case)
            return case

    def add_note(self, case_id: str, author: str, text: str) -> CaseNote:
        """Append an investigation note to the case audit log."""
        with self._lock:
            case = self.get_case(case_id)
            note = CaseNote(author=author, text=text)
            case.notes.append(note)
            case.add_audit_entry(
                action="note_added",
                actor=author,
                details={"note_id": note.note_id, "snippet": text[:40]},
            )
            self.store.put(case)
            return note

    def resolve_case(
        self,
        case_id: str,
        disposition: CaseDisposition,
        actor: str,
        notes: str = "",
    ) -> FraudCase:
        """Finalize a case with a resolution disposition."""
        with self._lock:
            case = self.get_case(case_id)
            if disposition == CaseDisposition.UNRESOLVED:
                raise CaseValidationError("Case cannot be resolved with UNRESOLVED disposition")

            case.disposition = disposition
            case.resolved_at = datetime.now(UTC).isoformat()
            if case.status != CaseStatus.RESOLVED:
                # Transition status to RESOLVED
                case.status = CaseStatus.RESOLVED

            if notes.strip():
                case.notes.append(CaseNote(author=actor, text=f"Resolution: {notes}"))

            case.add_audit_entry(
                action="case_resolved",
                actor=actor,
                details={
                    "disposition": disposition.value,
                    "notes": notes,
                },
            )
            self.store.put(case)
            return case

    def generate_evidence_package(
        self,
        case_id: str,
        transaction_data: dict[str, Any],
        device_data: dict[str, Any] | None = None,
        geo_data: dict[str, Any] | None = None,
        challenge_data: dict[str, Any] | None = None,
        merchant_data: dict[str, Any] | None = None,
        additional_evidence: dict[str, Any] | None = None,
        actor: str = "system",
    ) -> DisputeEvidencePackage:
        """Generate, fingerprint, and persist a DisputeEvidencePackage for a case."""
        with self._lock:
            case = self.get_case(case_id)
            pkg = compile_evidence_package(
                case=case,
                transaction_data=transaction_data,
                device_data=device_data,
                geo_data=geo_data,
                challenge_data=challenge_data,
                merchant_data=merchant_data,
                additional_evidence=additional_evidence,
            )
            self.store.put_evidence_package(pkg)
            case.add_audit_entry(
                action="evidence_packaged",
                actor=actor,
                details={
                    "package_id": pkg.package_id,
                    "sha256_fingerprint": pkg.sha256_fingerprint,
                },
            )
            self.store.put(case)
            return pkg

    def get_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
        """Aggregate high-level metrics across all cases."""
        with self._lock:
            cases = self.store.list_cases(tenant_id=tenant_id, limit=10000)
            status_counts = {s.value: 0 for s in CaseStatus}
            priority_counts = {p.value: 0 for p in CasePriority}
            sla_breached_count = 0

            for c in cases:
                status_counts[c.status.value] = status_counts.get(c.status.value, 0) + 1
                priority_counts[c.priority.value] = priority_counts.get(c.priority.value, 0) + 1
                if c.is_sla_breached:
                    sla_breached_count += 1

            return {
                "total_cases": len(cases),
                "by_status": status_counts,
                "by_priority": priority_counts,
                "sla_breached_count": sla_breached_count,
            }
