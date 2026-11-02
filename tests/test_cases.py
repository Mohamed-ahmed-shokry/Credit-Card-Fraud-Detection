"""Unit and integration tests for Fraud Case Management and Dispute Evidence Engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fraud_detection.cases import (
    CaseAssignmentError,
    CaseDisposition,
    CaseError,
    CaseNote,
    CaseNotFoundError,
    CasePriority,
    CaseStatus,
    CaseValidationError,
    EvidencePackagingError,
    FraudCase,
    InvalidCaseStateTransitionError,
    compute_priority,
)


def test_case_exception_hierarchy() -> None:
    """Validate that custom case exceptions inherit from CaseError."""
    assert issubclass(CaseNotFoundError, CaseError)
    assert issubclass(InvalidCaseStateTransitionError, CaseError)
    assert issubclass(CaseAssignmentError, CaseError)
    assert issubclass(EvidencePackagingError, CaseError)
    assert issubclass(CaseValidationError, CaseError)
    err = CaseNotFoundError("Case not found")
    assert str(err) == "Case not found"


def test_compute_priority_rules() -> None:
    """Verify priority computation thresholds and VIP escalation."""
    # Critical by VIP + elevated score
    assert compute_priority(0.55, 100.0, is_vip=True) == CasePriority.CRITICAL

    # Critical by high score >= 0.9 or large amount >= 5000
    assert compute_priority(0.92, 10.0, is_vip=False) == CasePriority.CRITICAL
    assert compute_priority(0.2, 5500.0, is_vip=False) == CasePriority.CRITICAL

    # High by score >= 0.75 or amount >= 1000
    assert compute_priority(0.80, 50.0, is_vip=False) == CasePriority.HIGH
    assert compute_priority(0.3, 1200.0, is_vip=False) == CasePriority.HIGH

    # Medium by score >= 0.45 or amount >= 250
    assert compute_priority(0.48, 50.0, is_vip=False) == CasePriority.MEDIUM
    assert compute_priority(0.1, 300.0, is_vip=False) == CasePriority.MEDIUM

    # Low otherwise
    assert compute_priority(0.2, 50.0, is_vip=False) == CasePriority.LOW


def test_case_note_creation_and_validation() -> None:
    """Validate CaseNote fields, validation guards, and dictionary serialization."""
    note = CaseNote(author="analyst_alice", text="Transaction verified with customer via phone.")
    assert note.note_id.startswith("note_")
    assert note.author == "analyst_alice"
    assert note.text == "Transaction verified with customer via phone."
    assert note.timestamp is not None

    d = note.to_dict()
    assert d["author"] == "analyst_alice"
    assert d["text"] == "Transaction verified with customer via phone."

    hydrated = CaseNote.from_dict(d)
    assert hydrated.note_id == note.note_id
    assert hydrated.author == note.author
    assert hydrated.text == note.text

    with pytest.raises(CaseValidationError, match="author cannot be empty"):
        CaseNote(author="   ", text="valid text")

    with pytest.raises(CaseValidationError, match="text cannot be empty"):
        CaseNote(author="alice", text="")


def test_fraud_case_models_and_invariants() -> None:
    """Verify FraudCase invariants, SLA computation, serialization, and audit logging."""
    case = FraudCase(
        transaction_id="tx_test_101",
        tenant_id="fintech_corp",
        fraud_score=0.88,
        amount=1500.0,
        currency="USD",
        priority=CasePriority.HIGH,
        sla_deadline_hours=12.0,
    )
    assert case.case_id.startswith("case_")
    assert case.status == CaseStatus.OPEN
    assert case.disposition == CaseDisposition.UNRESOLVED
    assert len(case.audit_trail) == 1
    assert case.audit_trail[0]["action"] == "case_created"
    assert case.is_sla_breached is False
    assert case.is_resolved is False

    # Check SLA expiration string is populated
    assert case.sla_expires_at != ""

    # Test audit entry append
    case.add_audit_entry(
        action="analyst_assigned",
        actor="lead_analyst",
        details={"analyst": "bob"},
    )
    assert len(case.audit_trail) == 2
    assert case.audit_trail[1]["action"] == "analyst_assigned"

    # Test serialization & hydration
    case_dict = case.to_dict()
    assert case_dict["case_id"] == case.case_id
    assert case_dict["transaction_id"] == "tx_test_101"
    assert case_dict["tenant_id"] == "fintech_corp"
    assert case_dict["fraud_score"] == 0.88

    restored = FraudCase.from_dict(case_dict)
    assert restored.case_id == case.case_id
    assert restored.transaction_id == case.transaction_id
    assert restored.priority == CasePriority.HIGH
    assert restored.amount == 1500.0


def test_fraud_case_sla_breach_detection() -> None:
    """Verify SLA breach detection when deadline has elapsed."""
    past_time = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    case = FraudCase(
        transaction_id="tx_sla_past",
        sla_deadline_hours=1.0,
        created_at=(datetime.now(UTC) - timedelta(hours=5)).isoformat(),
        sla_expires_at=past_time,
    )
    assert case.is_sla_breached is True

    # When resolved or closed, SLA is not breached
    case.status = CaseStatus.RESOLVED
    assert case.is_sla_breached is False
    assert case.is_resolved is True


def test_fraud_case_validation_errors() -> None:
    """Verify bounds and validation errors on invalid inputs."""
    with pytest.raises(CaseValidationError, match="transaction_id"):
        FraudCase(transaction_id="")

    with pytest.raises(CaseValidationError, match="case_id"):
        FraudCase(transaction_id="tx_1", case_id="")

    with pytest.raises(CaseValidationError, match="tenant_id"):
        FraudCase(transaction_id="tx_1", tenant_id="")

    with pytest.raises(CaseValidationError, match="fraud_score"):
        FraudCase(transaction_id="tx_1", fraud_score=1.5)

    with pytest.raises(CaseValidationError, match="amount"):
        FraudCase(transaction_id="tx_1", amount=-10.0)

    with pytest.raises(CaseValidationError, match="sla_deadline_hours"):
        FraudCase(transaction_id="tx_1", sla_deadline_hours=0.0)
