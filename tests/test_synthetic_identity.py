"""Unit and integration tests for Synthetic Identity Fraud (SIF) engine."""

from __future__ import annotations

from fraud_detection.synthetic_identity import (
    AuthorizedUserRecord,
    IdentityProfile,
    SifAction,
    SifAnomalyType,
    SifAssessment,
    SifRiskLevel,
    SsnValidationResult,
    SsnValidationStatus,
)


def test_sif_enums_values_and_roundtrip() -> None:
    assert SifRiskLevel.LOW == "low"
    assert SifRiskLevel.CRITICAL == "critical"
    assert SifAction.APPROVE == "approve"
    assert SifAction.DECLINE_BLOCK == "decline_block"
    assert SifAnomalyType.SSN_DOB_MISMATCH == "ssn_dob_mismatch"
    assert SifAnomalyType.DECEASED_SSN_MATCH == "deceased_ssn_match"
    assert SsnValidationStatus.VALID == "valid"
    assert SsnValidationStatus.DOB_INCONSISTENT == "dob_inconsistent"


def test_identity_profile_and_serialization() -> None:
    user = AuthorizedUserRecord(user_id="u_99", name="Jane Doe", relationship="spouse")
    profile = IdentityProfile(
        applicant_id="app_100",
        ssn="123-45-6789",
        first_name="John",
        last_name="Doe",
        dob="1990-05-15",
        address="123 Main St, Springfield, IL",
        phone="+12175551234",
        email="john.doe@example.com",
        credit_line_requested=10000.0,
        authorized_users=[user],
        tenant_id="tenant_a",
    )
    assert profile.ssn_clean() == "123456789"
    assert profile.ssn_masked() == "XXX-XX-6789"
    assert len(profile.ssn_hash()) == 64

    data = profile.to_dict()
    assert data["applicant_id"] == "app_100"
    assert data["ssn"] == "XXX-XX-6789"
    assert len(data["authorized_users"]) == 1

    restored = IdentityProfile.from_dict(data)
    assert restored.applicant_id == "app_100"
    assert restored.first_name == "John"
    assert len(restored.authorized_users) == 1
    assert restored.authorized_users[0].name == "Jane Doe"


def test_ssn_validation_result_roundtrip() -> None:
    res = SsnValidationResult(
        is_valid=True,
        status=SsnValidationStatus.VALID,
        area_number=123,
        group_number=45,
        serial_number=6789,
        state_issued="Massachusetts",
        estimated_issuance_year_range=(1985, 1987),
        is_deceased=False,
        details="Matches SSA allocation",
    )
    data = res.to_dict()
    assert data["is_valid"] is True
    assert data["state_issued"] == "Massachusetts"
    assert data["estimated_issuance_year_range"] == [1985, 1987]

    restored = SsnValidationResult.from_dict(data)
    assert restored.is_valid is True
    assert restored.status == SsnValidationStatus.VALID
    assert restored.estimated_issuance_year_range == (1985, 1987)


def test_sif_assessment_roundtrip() -> None:
    ssn_res = SsnValidationResult(
        is_valid=False,
        status=SsnValidationStatus.DOB_INCONSISTENT,
        area_number=999,
        group_number=99,
        serial_number=9999,
        state_issued="Unknown",
        details="SSN issued before birth",
    )
    assessment = SifAssessment(
        assessment_id="sif_test123",
        applicant_id="app_456",
        tenant_id="tenant_b",
        risk_score=0.85,
        risk_level=SifRiskLevel.HIGH,
        action=SifAction.MANUAL_REVIEW,
        triggered_anomalies=[
            SifAnomalyType.SSN_DOB_MISMATCH,
            SifAnomalyType.BURNER_PHONE_DETECTED,
        ],
        anomaly_breakdown={"ssn_dob_mismatch": 0.6, "burner_phone": 0.25},
        identity_fragmentation_score=0.4,
        ssn_validation=ssn_res,
        metadata={"source": "online_application"},
    )
    data = assessment.to_dict()
    assert data["risk_score"] == 0.85
    assert data["risk_level"] == "high"
    assert data["action"] == "manual_review"
    assert len(data["triggered_anomalies"]) == 2
    assert data["ssn_validation"]["status"] == "dob_inconsistent"

    restored = SifAssessment.from_dict(data)
    assert restored.assessment_id == "sif_test123"
    assert restored.risk_level == SifRiskLevel.HIGH
    assert restored.action == SifAction.MANUAL_REVIEW
    assert SifAnomalyType.SSN_DOB_MISMATCH in restored.triggered_anomalies
    assert restored.ssn_validation is not None
    assert restored.ssn_validation.status == SsnValidationStatus.DOB_INCONSISTENT
