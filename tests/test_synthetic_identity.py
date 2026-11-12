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
    SsnValidator,
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


def test_ssn_validator_format_and_prohibited_areas() -> None:
    validator = SsnValidator()

    # Invalid length
    res = validator.validate("123-45")
    assert res.is_valid is False
    assert res.status == SsnValidationStatus.INVALID_FORMAT

    # Group 00 or serial 0000
    res_grp0 = validator.validate("050-00-1234")
    assert res_grp0.is_valid is False
    assert res_grp0.status == SsnValidationStatus.RESERVED_OR_SPECIAL

    res_ser0 = validator.validate("050-12-0000")
    assert res_ser0.is_valid is False
    assert res_ser0.status == SsnValidationStatus.RESERVED_OR_SPECIAL

    # Area 000, 666, 900+
    res_000 = validator.validate("000-12-3456")
    assert res_000.is_valid is False
    assert res_000.status == SsnValidationStatus.RESERVED_OR_SPECIAL

    res_666 = validator.validate("666-12-3456")
    assert res_666.is_valid is False
    assert res_666.status == SsnValidationStatus.RESERVED_OR_SPECIAL

    res_950 = validator.validate("950-12-3456")
    assert res_950.is_valid is False
    assert res_950.status == SsnValidationStatus.RESERVED_OR_SPECIAL


def test_ssn_validator_deceased_master_file() -> None:
    validator = SsnValidator()
    # Check default deceased
    res = validator.validate("042-12-9999")
    assert res.is_valid is False
    assert res.status == SsnValidationStatus.DECEASED
    assert res.is_deceased is True
    assert "Arthur Pendelton" in res.details

    # Add dynamic deceased
    validator.add_deceased_record("050-11-2222", "Test Deceased", "2024-01-01", "NY")
    res_dyn = validator.validate("050-11-2222")
    assert res_dyn.is_valid is False
    assert res_dyn.status == SsnValidationStatus.DECEASED


def test_ssn_validator_historical_areas_and_dob_consistency() -> None:
    validator = SsnValidator()

    # Pre-2011 valid New York SSN (050-134)
    res_ny = validator.validate("060-12-3456", dob="1980-01-01")
    assert res_ny.is_valid is True
    assert res_ny.status == SsnValidationStatus.VALID
    assert res_ny.state_issued == "New York"
    assert res_ny.estimated_issuance_year_range == (1936, 2011)

    # Railroad retirement (700-728) stopped in 1963; applicant born 1995 is impossible!
    res_rr = validator.validate("710-12-3456", dob="1995-05-20")
    assert res_rr.is_valid is False
    assert res_rr.status == SsnValidationStatus.DOB_INCONSISTENT
    assert "Railroad Retirement" in res_rr.details

    # Pre-2011 unissued area (e.g. 740) claimed with pre-2011 DOB
    res_unissued = validator.validate("740-12-3456", dob="1985-02-14")
    assert res_unissued.is_valid is False
    assert res_unissued.status == SsnValidationStatus.UNISSUED_AREA

    # Same unissued area claimed with post-2011 DOB -> post-randomization
    res_rand = validator.validate("740-12-3456", dob="2015-08-10")
    assert res_rand.is_valid is True
    assert res_rand.status == SsnValidationStatus.ISSUED_POST_RANDOMIZATION


def test_group_order_helper() -> None:
    from fraud_detection.synthetic_identity import get_group_order

    # Odd 01-09 -> 1..5
    assert get_group_order(1) == 1
    assert get_group_order(9) == 5
    # Even 10-98 -> 6..50
    assert get_group_order(10) == 6
    assert get_group_order(98) == 50
    # Even 02-08 -> 51..54
    assert get_group_order(2) == 51
    assert get_group_order(8) == 54
    # Odd 11-99 -> 55..99
    assert get_group_order(11) == 55
    assert get_group_order(99) == 99
    # Invalid
    assert get_group_order(0) == 0
    assert get_group_order(100) == 0


def test_pii_inconsistency_engine_fragmentation() -> None:
    from fraud_detection.synthetic_identity import (
        PiiInconsistencyEngine,
    )

    engine = PiiInconsistencyEngine()

    target = IdentityProfile(
        applicant_id="app_new",
        ssn="050-12-3456",
        first_name="Alice",
        last_name="Johnson",
        dob="1992-06-15",
        address="742 Evergreen Terrace, Springfield, IL",
        phone="+12175550001",
        email="alice.j@example.com",
    )

    # 1. Stolen SSN with different name
    hist1 = IdentityProfile(
        applicant_id="app_old_1",
        ssn="050-12-3456",
        first_name="Robert",
        last_name="Smith",
        dob="1985-01-20",
        address="100 Oak St, Chicago, IL",
        phone="+13125559999",
        email="rsmith@example.com",
    )

    score, matches, anomalies = engine.evaluate_inconsistency(target, [hist1])
    assert score > 0.8
    assert SifAnomalyType.PII_ELEMENT_FRAGMENTATION in anomalies
    assert SifAnomalyType.SSN_DOB_MISMATCH in anomalies
    assert len(matches) == 2
    match_dict = matches[0].to_dict()
    assert match_dict["overlap_field"] == "ssn"
    assert "Robert Smith" in match_dict["description"]


def test_pii_inconsistency_shared_contact_and_address() -> None:
    from fraud_detection.synthetic_identity import PiiInconsistencyEngine

    engine = PiiInconsistencyEngine()
    target = IdentityProfile(
        applicant_id="target_app",
        ssn="123-45-6789",
        first_name="Charlie",
        last_name="Brown",
        dob="1995-10-10",
        address="456 Elm Ave Apt 2, Boston, MA",
        phone="+16175551212",
        email="charlie.b@testmail.com",
    )
    # Different person with same phone, email, and address
    hist = IdentityProfile(
        applicant_id="hist_app",
        ssn="987-65-4321",
        first_name="Danielle",
        last_name="Vance",
        dob="1991-03-05",
        address="456 Elm Ave Apt 2, Boston, MA",
        phone="+16175551212",
        email="charlie.b@testmail.com",
    )

    score, matches, _ = engine.evaluate_inconsistency(target, [hist])
    assert score > 0.5
    fields = {m.overlap_field for m in matches}
    assert "phone" in fields
    assert "email" in fields
    assert "address" in fields
