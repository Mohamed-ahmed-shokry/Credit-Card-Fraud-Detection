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


def test_synthetic_identity_tracker_drop_and_burner() -> None:
    from fraud_detection.synthetic_identity import SyntheticIdentityTracker

    tracker = SyntheticIdentityTracker()

    # Drop address PO Box
    assert tracker.is_drop_address("P.O. Box 742, Springfield, IL") is True
    assert tracker.is_drop_address("Suite PMB 450, 100 Main St") is True
    assert tracker.is_drop_address("123 Maple Street, Chicago, IL") is False

    # Burner phone
    assert tracker.is_burner_phone("18005551234") is True
    assert tracker.is_burner_phone("+12125550199") is True
    assert tracker.is_burner_phone("1111111111") is True
    assert tracker.is_burner_phone("+12175558901") is False


def test_synthetic_identity_tracker_piggybacking_and_velocity() -> None:
    from datetime import timedelta

    from fraud_detection.synthetic_identity import (
        AuthorizedUserRecord,
        IdentityProfile,
        SyntheticIdentityTracker,
    )

    tracker = SyntheticIdentityTracker(velocity_window_hours=1.0, max_applications_per_ssn=2)

    # Authorized user piggybacking
    u1 = AuthorizedUserRecord(user_id="u1", name="Tradeline Buyer 1", relationship="other")
    u2 = AuthorizedUserRecord(user_id="u2", name="Tradeline Buyer 2", relationship="broker")
    profile = IdentityProfile(
        applicant_id="app_abuse",
        ssn="050-12-3456",
        first_name="David",
        last_name="Copperfield",
        dob="1990-01-01",
        address="100 Grand Ave, New York, NY",
        phone="+12125558888",
        email="david@example.com",
        authorized_users=[u1, u2],
    )

    score, anomalies, details = tracker.check_and_record(profile)
    assert SifAnomalyType.AUTHORIZED_USER_ABUSE in anomalies
    assert details["abuse_reason"] != ""
    assert score >= 0.5

    # Velocity: second application (below limit 2)
    p2 = IdentityProfile(
        applicant_id="app_abuse_2",
        ssn="050-12-3456",
        first_name="David",
        last_name="Copperfield",
        dob="1990-01-01",
        address="100 Grand Ave, New York, NY",
        phone="+12125558888",
        email="david@example.com",
    )
    _score2, anomalies2, _ = tracker.check_and_record(p2)
    assert SifAnomalyType.SYNTHETIC_VELOCITY_BURST not in anomalies2

    # Third application: exceeds max_applications_per_ssn (2)
    p3 = IdentityProfile(
        applicant_id="app_abuse_3",
        ssn="050-12-3456",
        first_name="David",
        last_name="Copperfield",
        dob="1990-01-01",
        address="100 Grand Ave, New York, NY",
        phone="+12125558888",
        email="david@example.com",
    )
    _score3, anomalies3, _ = tracker.check_and_record(p3)
    assert SifAnomalyType.SYNTHETIC_VELOCITY_BURST in anomalies3

    # Fast forward time beyond window
    future_time = profile.created_at + timedelta(hours=2.5)
    p4 = IdentityProfile(
        applicant_id="app_abuse_4",
        ssn="050-12-3456",
        first_name="David",
        last_name="Copperfield",
        dob="1990-01-01",
        address="100 Grand Ave, New York, NY",
        phone="+12125558888",
        email="david@example.com",
        created_at=future_time,
    )
    _, anomalies4, _ = tracker.check_and_record(p4, now=future_time)
    # The old applications should have slid out of the window!
    assert SifAnomalyType.SYNTHETIC_VELOCITY_BURST not in anomalies4


def test_synthetic_identity_store_crud_and_indexing() -> None:
    from fraud_detection.synthetic_identity import (
        IdentityProfile,
        SifAssessment,
        SyntheticIdentityStore,
    )

    store = SyntheticIdentityStore(max_records=3)
    p1 = IdentityProfile(
        applicant_id="app1",
        ssn="111-22-3333",
        first_name="First",
        last_name="Person",
        dob="1990-01-01",
        address="10 Main St",
        phone="+12125551111",
        email="p1@example.com",
        tenant_id="t1",
    )
    store.add_profile(p1)
    assert store.get_profile("app1") is not None
    assert len(store.find_profiles_by_ssn(p1.ssn_hash())) == 1
    assert len(store.find_profiles_by_phone("12125551111")) == 1
    assert len(store.find_profiles_by_email("p1@example.com")) == 1
    assert len(store.list_profiles("t1")) == 1

    # Capacity eviction
    p2 = IdentityProfile(
        applicant_id="app2",
        ssn="222-33-4444",
        first_name="Second",
        last_name="Person",
        dob="1991-01-01",
        address="20 Main St",
        phone="+12125552222",
        email="p2@example.com",
    )
    p3 = IdentityProfile(
        applicant_id="app3",
        ssn="333-44-5555",
        first_name="Third",
        last_name="Person",
        dob="1992-01-01",
        address="30 Main St",
        phone="+12125553333",
        email="p3@example.com",
    )
    p4 = IdentityProfile(
        applicant_id="app4",
        ssn="444-55-6666",
        first_name="Fourth",
        last_name="Person",
        dob="1993-01-01",
        address="40 Main St",
        phone="+12125554444",
        email="p4@example.com",
    )
    store.add_profile(p2)
    store.add_profile(p3)
    store.add_profile(p4)
    # app1 should have been evicted due to max_records=3
    assert store.get_profile("app1") is None
    assert store.get_profile("app4") is not None

    # Assessment indexing
    ass = SifAssessment(assessment_id="ass_1", applicant_id="app4", tenant_id="t1")
    store.add_assessment(ass)
    assert store.get_assessment("ass_1") is not None
    assert len(store.list_assessments("t1")) == 1

    store.clear()
    assert store.get_profile("app4") is None
    assert store.get_assessment("ass_1") is None


def test_synthetic_identity_engine_decisions_and_escalations() -> None:
    from unittest.mock import MagicMock

    from fraud_detection.synthetic_identity import (
        IdentityProfile,
        SifAction,
        SifAnomalyType,
        SifRiskLevel,
        SyntheticIdentityEngine,
    )

    mock_case_mgr = MagicMock()
    mock_challenge_mgr = MagicMock()

    engine = SyntheticIdentityEngine(
        case_manager=mock_case_mgr,
        challenge_manager=mock_challenge_mgr,
    )

    # 1. Clean, legitimate applicant -> APPROVE
    legit = IdentityProfile(
        applicant_id="legit_1",
        ssn="050-12-3456",  # Valid NY
        first_name="Sarah",
        last_name="Connor",
        dob="1984-05-12",
        address="100 Cyberdyne Way, New York, NY",
        phone="+12125557777",
        email="sarah@resistance.org",
    )
    ass_legit = engine.evaluate(legit)
    assert ass_legit.action == SifAction.APPROVE
    assert ass_legit.risk_level == SifRiskLevel.LOW
    assert ass_legit.risk_score < 0.35
    assert not ass_legit.triggered_anomalies

    # 2. Deceased SSN -> DECLINE_BLOCK
    deceased = IdentityProfile(
        applicant_id="dec_1",
        ssn="042-12-9999",  # Arthur Pendelton
        first_name="Arthur",
        last_name="Pendelton",
        dob="1950-01-01",
        address="10 Ghost St, Hartford, CT",
        phone="+12035559999",
        email="ghost@example.com",
    )
    ass_dec = engine.evaluate(deceased)
    assert ass_dec.action == SifAction.DECLINE_BLOCK
    assert ass_dec.risk_level == SifRiskLevel.CRITICAL
    assert ass_dec.risk_score == 1.0
    assert SifAnomalyType.DECEASED_SSN_MATCH in ass_dec.triggered_anomalies

    # 3. High risk PII fragmentation -> MANUAL_REVIEW + Case Escalation
    # Stolen SSN from legit_1 with different name
    stolen = IdentityProfile(
        applicant_id="stolen_1",
        ssn="050-12-3456",  # Reusing Sarah's SSN!
        first_name="Miles",
        last_name="Dyson",
        dob="1970-08-25",
        address="200 Tech Blvd, New York, NY",
        phone="+12125556666",
        email="dyson@example.com",
    )
    ass_stolen = engine.evaluate(stolen)
    assert ass_stolen.action in (SifAction.MANUAL_REVIEW, SifAction.DECLINE_BLOCK)
    assert mock_case_mgr.create_case.called

    # 4. Burner phone step-up challenge -> STEP_UP_IDENTITY_PROOFING + ChallengeManager
    burner_app = IdentityProfile(
        applicant_id="burner_1",
        ssn="060-12-3456",
        first_name="John",
        last_name="Doe",
        dob="1988-03-15",
        address="500 Pine St, New York, NY",
        phone="18005551234",  # Burner toll-free
        email="johndoe1800@example.com",
    )
    ass_burner = engine.evaluate(burner_app)
    assert ass_burner.action == SifAction.STEP_UP_IDENTITY_PROOFING
    assert ass_burner.risk_level == SifRiskLevel.MEDIUM
    assert mock_challenge_mgr.create_challenge.called


def test_synthetic_identity_engine_with_event_bus() -> None:
    from fraud_detection.events import EventBus, EventType, FraudDomainEvent
    from fraud_detection.synthetic_identity import (
        IdentityProfile,
        SyntheticIdentityEngine,
    )

    bus = EventBus()
    received: list[FraudDomainEvent] = []
    bus.subscribe(lambda ev: received.append(ev))

    engine = SyntheticIdentityEngine(event_bus=bus)

    # Legitimate applicant -> SIF_APPLICATION_EVALUATED and SIF_IDENTITY_CONFIRMED
    legit = IdentityProfile(
        applicant_id="app_event_1",
        ssn="050-12-3456",
        first_name="Diana",
        last_name="Prince",
        dob="1984-01-01",
        address="700 Themyscira Blvd, New York, NY",
        phone="+12125554321",
        email="diana@example.com",
    )
    engine.evaluate(legit)

    event_types = [e.event_type for e in received]
    assert EventType.SIF_APPLICATION_EVALUATED in event_types
    assert EventType.SIF_IDENTITY_CONFIRMED in event_types

    # Deceased applicant -> SIF_APPLICATION_EVALUATED, SIF_ANOMALY_DETECTED,
    # and SIF_APPLICATION_DECLINED
    dec = IdentityProfile(
        applicant_id="app_event_2",
        ssn="042-12-9999",
        first_name="Arthur",
        last_name="Pendelton",
        dob="1950-01-01",
        address="10 Ghost St, Hartford, CT",
        phone="+12035559999",
        email="ghost@example.com",
    )
    engine.evaluate(dec)

    event_types_2 = [e.event_type for e in received]
    assert EventType.SIF_ANOMALY_DETECTED in event_types_2
    assert EventType.SIF_APPLICATION_DECLINED in event_types_2
    bus.close()
