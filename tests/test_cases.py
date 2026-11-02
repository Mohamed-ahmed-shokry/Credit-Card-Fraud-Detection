"""Unit and integration tests for Fraud Case Management and Dispute Evidence Engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fraud_detection.cases import (
    CaseAssignmentError,
    CaseDisposition,
    CaseError,
    CaseManager,
    CaseNote,
    CaseNotFoundError,
    CasePriority,
    CaseStatus,
    CaseStore,
    CaseValidationError,
    DisputeEvidencePackage,
    EvidencePackagingError,
    FraudCase,
    InvalidCaseStateTransitionError,
    compile_evidence_package,
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


def test_dispute_evidence_package_fingerprint_and_integrity() -> None:
    """Validate deterministic canonical hashing and tamper-evident integrity checking."""
    pkg = DisputeEvidencePackage(
        case_id="case_123",
        transaction_id="tx_abc",
        tenant_id="tenant_fintech",
        evidence_items={"amount": 499.99, "currency": "USD", "reason": "fraud_dispute"},
        device_fingerprint={"device_id": "dev_001", "risk_tier": "elevated"},
    )
    assert pkg.package_id.startswith("evpkg_")
    assert len(pkg.sha256_fingerprint) == 64
    assert pkg.verify_integrity() is True

    # Check canonical JSON deterministic structure
    canonical = pkg.to_canonical_json()
    assert '"case_id":"case_123"' in canonical
    assert '"device_id":"dev_001"' in canonical

    # Tampering test: alter internal evidence item
    pkg.evidence_items["amount"] = 9999.99
    assert pkg.verify_integrity() is False

    # Restoring original restores integrity
    pkg.evidence_items["amount"] = 499.99
    assert pkg.verify_integrity() is True


def test_compile_evidence_package_multi_modal() -> None:
    """Verify multi-modal compiler aggregates device, geo, 3DS challenge, and merchant context."""
    case = FraudCase(
        transaction_id="tx_multi_001",
        tenant_id="tenant_alpha",
        fraud_score=0.91,
        amount=1200.0,
        priority=CasePriority.CRITICAL,
    )
    case.notes.append(CaseNote(author="analyst_1", text="Suspicious IP hop"))

    tx_data = {"card_network": "Visa", "auth_code": "V12345", "avs_result": "MATCH"}
    device_data = {"device_cookie_age_days": 0.5, "is_proxy": True}
    geo_data = {"country": "FR", "speed_kmh": 1200.0, "impossible_travel": True}
    challenge_data = {"status": "verified", "method": "sms_otp", "attempts": 1}
    merchant_data = {"mcc": "5732", "merchant_risk_tier": "high"}

    pkg = compile_evidence_package(
        case=case,
        transaction_data=tx_data,
        device_data=device_data,
        geo_data=geo_data,
        challenge_data=challenge_data,
        merchant_data=merchant_data,
        additional_evidence={"chargeback_reason_code": "10.4"},
    )

    assert pkg.case_id == case.case_id
    assert pkg.transaction_id == "tx_multi_001"
    assert pkg.tenant_id == "tenant_alpha"
    assert pkg.device_fingerprint == device_data
    assert pkg.geolocation_data == geo_data
    assert pkg.challenge_verification == challenge_data
    assert pkg.merchant_profile == merchant_data
    assert pkg.evidence_items["chargeback_reason_code"] == "10.4"
    assert pkg.evidence_items["case_fraud_score"] == 0.91
    assert pkg.verify_integrity() is True

    # Serialization and hydration
    d = pkg.to_dict()
    assert d["sha256_fingerprint"] == pkg.sha256_fingerprint
    hydrated = DisputeEvidencePackage.from_dict(d)
    assert hydrated.package_id == pkg.package_id
    assert hydrated.verify_integrity() is True


def test_evidence_package_validation_errors() -> None:
    """Verify bounds and validation errors on evidence packages."""
    with pytest.raises(EvidencePackagingError, match="case_id"):
        DisputeEvidencePackage(
            case_id="",
            transaction_id="tx_1",
            tenant_id="t1",
            evidence_items={},
        )

    with pytest.raises(EvidencePackagingError, match="transaction_id"):
        DisputeEvidencePackage(
            case_id="c1",
            transaction_id="",
            tenant_id="t1",
            evidence_items={},
        )

    with pytest.raises(EvidencePackagingError, match="tenant_id"):
        DisputeEvidencePackage(
            case_id="c1",
            transaction_id="tx_1",
            tenant_id="",
            evidence_items={},
        )


def test_case_store_operations() -> None:
    """Validate thread-safe CaseStore indexing, queries, filtering, and package storage."""
    store = CaseStore()
    c1 = FraudCase(transaction_id="tx_s1", tenant_id="t1", priority=CasePriority.HIGH)
    c2 = FraudCase(transaction_id="tx_s2", tenant_id="t2", priority=CasePriority.LOW)
    c3 = FraudCase(
        transaction_id="tx_s3",
        tenant_id="t1",
        priority=CasePriority.CRITICAL,
        sla_expires_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(),
    )

    store.put(c1)
    store.put(c2)
    store.put(c3)

    assert store.count() == 3
    assert store.get(c1.case_id) == c1
    assert store.get_by_transaction("tx_s2") == c2
    assert store.get_by_transaction("nonexistent") is None

    # Filter by tenant
    t1_cases = store.list_cases(tenant_id="t1")
    assert len(t1_cases) == 2

    # Filter by priority
    low_cases = store.list_cases(priority=CasePriority.LOW)
    assert len(low_cases) == 1
    assert low_cases[0].case_id == c2.case_id

    # Filter by SLA breached
    breached = store.list_cases(sla_breached_only=True)
    assert len(breached) == 1
    assert breached[0].case_id == c3.case_id

    # Evidence package indexing
    pkg = DisputeEvidencePackage(
        case_id=c1.case_id,
        transaction_id="tx_s1",
        tenant_id="t1",
        evidence_items={"test": True},
    )
    store.put_evidence_package(pkg)
    assert store.get_evidence_package(pkg.package_id) == pkg
    assert len(store.list_evidence_packages(c1.case_id)) == 1

    # Delete
    assert store.delete(c1.case_id) is True
    assert store.get(c1.case_id) is None
    assert store.delete("nonexistent") is False
    assert store.count() == 2

    store.clear()
    assert store.count() == 0


def test_case_manager_lifecycle_and_transitions() -> None:
    """Verify CaseManager workflows: creation, assignment, transition rules, and resolution."""
    mgr = CaseManager()

    # 1. Create case with auto-calculated priority
    case = mgr.create_case(
        transaction_id="tx_flow_01",
        fraud_score=0.85,
        amount=2500.0,
        tenant_id="fintech_beta",
    )
    assert case.priority == CasePriority.HIGH
    assert case.status == CaseStatus.OPEN
    assert case.assigned_analyst is None

    # 2. Assign analyst -> auto transitions to INVESTIGATING
    assigned = mgr.assign_case(case.case_id, analyst="analyst_charlie", actor="supervisor")
    assert assigned.assigned_analyst == "analyst_charlie"
    assert assigned.status == CaseStatus.INVESTIGATING

    # 3. Add note
    note = mgr.add_note(
        case.case_id,
        author="analyst_charlie",
        text="Called bank, suspicious card.",
    )
    assert note.text == "Called bank, suspicious card."
    assert len(assigned.notes) == 1

    # 4. Status transitions
    mgr.update_status(case.case_id, CaseStatus.PENDING_CUSTOMER, actor="analyst_charlie")
    assert assigned.status == CaseStatus.PENDING_CUSTOMER

    # Illegal transition: PENDING_CUSTOMER cannot go directly to REPRESENTED
    with pytest.raises(InvalidCaseStateTransitionError, match="Illegal transition"):
        mgr.update_status(case.case_id, CaseStatus.REPRESENTED, actor="analyst_charlie")

    # Transition to INVESTIGATING
    mgr.update_status(case.case_id, CaseStatus.INVESTIGATING, actor="analyst_charlie")

    # 5. Resolve case
    resolved = mgr.resolve_case(
        case.case_id,
        disposition=CaseDisposition.CONFIRMED_FRAUD,
        actor="analyst_charlie",
        notes="Cardholder confirmed stolen credentials.",
    )
    assert resolved.status == CaseStatus.RESOLVED
    assert resolved.disposition == CaseDisposition.CONFIRMED_FRAUD
    assert resolved.resolved_at is not None

    # 6. Cannot resolve with UNRESOLVED disposition
    with pytest.raises(CaseValidationError, match="cannot be resolved with UNRESOLVED"):
        mgr.resolve_case(
            case.case_id,
            disposition=CaseDisposition.UNRESOLVED,
            actor="analyst_charlie",
        )

    # 7. Generate evidence package through manager
    pkg = mgr.generate_evidence_package(
        case.case_id,
        transaction_data={"amount": 2500.0},
        device_data={"ip": "1.2.3.4"},
        actor="analyst_charlie",
    )
    assert pkg.case_id == case.case_id
    assert pkg.verify_integrity() is True
    assert len(mgr.store.list_evidence_packages(case.case_id)) == 1

    # 8. Summary statistics
    summary = mgr.get_summary(tenant_id="fintech_beta")
    assert summary["total_cases"] == 1
    assert summary["by_status"]["resolved"] == 1
    assert summary["by_priority"]["high"] == 1


def test_case_manager_error_paths() -> None:
    """Verify error checking for nonexistent cases and invalid assignment."""
    mgr = CaseManager()

    with pytest.raises(CaseNotFoundError, match="not found"):
        mgr.get_case("case_missing")

    with pytest.raises(CaseNotFoundError, match="not found"):
        mgr.assign_case("case_missing", analyst="bob")

    # Create case then attempt empty analyst assignment
    c = mgr.create_case(transaction_id="tx_err")
    with pytest.raises(CaseAssignmentError, match="Analyst identifier cannot be empty"):
        mgr.assign_case(c.case_id, analyst="   ")
