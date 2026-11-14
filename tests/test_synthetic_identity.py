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


def test_synthetic_identity_uncovered_branches_and_edge_cases() -> None:
    from datetime import UTC, datetime

    from fraud_detection.synthetic_identity import (
        AuthorizedUserRecord,
        IdentityProfile,
        PiiInconsistencyEngine,
        SifAction,
        SifAnomalyType,
        SifAssessment,
        SifRiskLevel,
        SsnValidationStatus,
        SsnValidator,
        SyntheticIdentityEngine,
        SyntheticIdentityStore,
        SyntheticIdentityTracker,
    )

    # 1. IdentityProfile short SSN masking
    short_prof = IdentityProfile(
        applicant_id="app_short",
        ssn="123",
        first_name="Short",
        last_name="SSN",
        dob="1990-01-01",
        address="123 St",
        phone="+12125550000",
        email="short@example.com",
    )
    assert short_prof.ssn_masked() == "XXX-XX-XXXX"

    # 2. SsnValidator add_deceased_record invalid length & lookup_area unmapped
    val = SsnValidator(unissued_areas=set())
    val.add_deceased_record(ssn="123", name="Invalid", dod="2020-01-01")
    assert not val.is_deceased("123")[0]

    # Validate with no dob, malformed dob, and unmapped area
    res_no_dob = val.validate("050-12-3456", dob=None)
    assert res_no_dob.is_valid
    res_bad_dob = val.validate("050-12-3456", dob="not-a-date")
    assert res_bad_dob.is_valid
    res_unmapped = val.validate("750-12-3456")
    assert res_unmapped.status == SsnValidationStatus.VALID
    assert "Unmapped" in res_unmapped.state_issued

    # Post-2011 randomized area with birth year >= 2011
    val_default = SsnValidator()
    res_post_2011 = val_default.validate("750-12-3456", dob="2015-05-01")
    assert res_post_2011.status == SsnValidationStatus.ISSUED_POST_RANDOMIZATION

    # 3. PiiInconsistencyEngine self-skip & same name/DOB with different SSN
    pii = PiiInconsistencyEngine()
    target_p = IdentityProfile(
        applicant_id="target_1",
        ssn="050-12-3456",
        first_name="Alice",
        last_name="Smith",
        dob="1990-01-01",
        phone="+12125551111",
        email="alice@example.com",
        address="100 Maple St, Boston, MA",
    )
    same_id_p = IdentityProfile(
        applicant_id="target_1",
        ssn="050-12-3456",
        first_name="Alice",
        last_name="Smith",
        dob="1990-01-01",
        phone="+12125551111",
        email="alice@example.com",
        address="100 Maple St, Boston, MA",
    )
    clone_diff_ssn = IdentityProfile(
        applicant_id="target_clone",
        ssn="060-12-3456",
        first_name="Alice",
        last_name="Smith",
        dob="1990-01-01",
        phone="+12125559999",
        email="alice.clone@example.com",
        address="200 Oak St, Boston, MA",
    )
    score, _matches, anomalies = pii.evaluate_inconsistency(target_p, [same_id_p, clone_diff_ssn])
    assert score > 0.0
    assert SifAnomalyType.PII_ELEMENT_FRAGMENTATION in anomalies

    # 4. SyntheticIdentityTracker phone velocity & address velocity spikes
    tracker = SyntheticIdentityTracker(
        max_applications_per_phone=1,
        max_applications_per_address=1,
    )
    now = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
    p_phone1 = IdentityProfile(
        applicant_id="p1",
        ssn="050-11-1111",
        first_name="One",
        last_name="Test",
        dob="1990-01-01",
        phone="+12125558888",
        address="300 Pine St, Boston, MA",
        email="one@example.com",
    )
    p_phone2 = IdentityProfile(
        applicant_id="p2",
        ssn="050-22-2222",
        first_name="Two",
        last_name="Test",
        dob="1990-01-01",
        phone="+12125558888",
        address="300 Pine St, Boston, MA",
        email="two@example.com",
    )
    tracker.check_and_record(p_phone1, now=now)
    # Second evaluation triggers velocity burst on phone and address
    _, tracker_anomalies, _ = tracker.check_and_record(p_phone2, now=now)
    assert SifAnomalyType.SYNTHETIC_VELOCITY_BURST in tracker_anomalies
    assert SifAnomalyType.HIGH_RISK_ADDRESS_DROP in tracker_anomalies

    # 5. SyntheticIdentityStore eviction, removal, counts, and listing
    store = SyntheticIdentityStore(max_records=2)
    p_a = IdentityProfile(
        applicant_id="p_a",
        ssn="050-11-0001",
        first_name="A",
        last_name="X",
        dob="1990-01-01",
        address="Addr A",
        phone="+12125550001",
        email="a@example.com",
    )
    p_b = IdentityProfile(
        applicant_id="p_b",
        ssn="050-11-0002",
        first_name="B",
        last_name="Y",
        dob="1990-01-01",
        address="Addr B",
        phone="",
        email="",
    )
    p_c = IdentityProfile(
        applicant_id="p_c",
        ssn="050-11-0003",
        first_name="C",
        last_name="Z",
        dob="1990-01-01",
        address="Addr C",
        phone="+12125550003",
        email="c@example.com",
    )

    store.add_profile(p_a)
    store.add_profile(p_a)  # Duplicate add tests branch when app_id already in index
    store.add_profile(p_b)
    assert store.profile_count == 2
    store.add_profile(p_c)  # Triggers eviction of p_a
    assert store.profile_count == 2
    assert store.get_profile("p_a") is None
    assert store.get_profile("p_c") is not None

    store.remove_profile("nonexistent_id")  # Tests remove_profile when prof is None
    store.remove_profile("p_b")  # Tests remove_profile when phone and email are empty
    assert store.get_profile("p_b") is None
    all_profiles = store.list_profiles()
    assert len(all_profiles) == 1

    # Assessments eviction and listing
    ass_1 = SifAssessment(
        assessment_id="ass_1",
        applicant_id="p_a",
        risk_score=0.1,
        risk_level=SifRiskLevel.LOW,
        action=SifAction.APPROVE,
        triggered_anomalies=[],
        ssn_validation=res_no_dob,
    )
    ass_2 = SifAssessment(
        assessment_id="ass_2",
        applicant_id="p_b",
        risk_score=0.2,
        risk_level=SifRiskLevel.LOW,
        action=SifAction.APPROVE,
        triggered_anomalies=[],
        ssn_validation=res_no_dob,
    )
    ass_3 = SifAssessment(
        assessment_id="ass_3",
        applicant_id="p_c",
        risk_score=0.3,
        risk_level=SifRiskLevel.LOW,
        action=SifAction.APPROVE,
        triggered_anomalies=[],
        ssn_validation=res_no_dob,
    )
    store.add_assessment(ass_1)
    store.add_assessment(ass_2)
    assert store.assessment_count == 2
    store.add_assessment(ass_3)  # Triggers eviction of ass_1
    assert store.assessment_count == 2
    assert store.get_assessment("ass_1") is None
    assert len(store.list_assessments()) == 2
    assert len(store.list_assessments(tenant_id="default")) == 2

    # 6. SyntheticIdentityEngine status mappings and floor checks
    engine = SyntheticIdentityEngine(store=store)

    # Invalid SSN structure status mapping
    invalid_ssn_prof = IdentityProfile(
        applicant_id="p_inv_ssn",
        ssn="123",
        first_name="Invalid",
        last_name="SSN",
        dob="1990-01-01",
        address="10 Bad St",
        phone="+12125550010",
        email="inv@example.com",
    )
    eval_inv = engine.evaluate(invalid_ssn_prof)
    assert SifAnomalyType.INVALID_SSN_STRUCTURE in eval_inv.triggered_anomalies

    # DOB inconsistent status mapping (RRB SSN born after 1963)
    dob_inconsistent_prof = IdentityProfile(
        applicant_id="p_rrb",
        ssn="700-12-3456",
        first_name="RRB",
        last_name="Applicant",
        dob="1990-01-01",
        address="20 RRB Way",
        phone="+12125550020",
        email="rrb@example.com",
    )
    eval_rrb = engine.evaluate(dob_inconsistent_prof)
    assert SifAnomalyType.SSN_DOB_MISMATCH in eval_rrb.triggered_anomalies

    # Unissued area pre-2011 status mapping
    unissued_prof = IdentityProfile(
        applicant_id="p_unissued",
        ssn="750-12-3456",
        first_name="Unissued",
        last_name="Applicant",
        dob="1990-01-01",
        address="30 Unissued St",
        phone="+12125550030",
        email="unissued@example.com",
    )
    eval_unissued = engine.evaluate(unissued_prof)
    assert SifAnomalyType.UNISSUED_SSN_AREA in eval_unissued.triggered_anomalies

    # Authorized user abuse standalone floor check
    user_rel_1 = AuthorizedUserRecord(user_id="u1", name="Rel 1", relationship="other")
    user_rel_2 = AuthorizedUserRecord(user_id="u2", name="Rel 2", relationship="broker")
    piggyback_prof = IdentityProfile(
        applicant_id="p_piggy",
        ssn="050-12-3456",
        first_name="Piggy",
        last_name="Backer",
        dob="1990-01-01",
        address="40 Normal St",
        phone="+12125550040",
        email="piggy@example.com",
        authorized_users=[user_rel_1, user_rel_2],
    )
    eval_piggy = engine.evaluate(piggyback_prof)
    assert SifAnomalyType.AUTHORIZED_USER_ABUSE in eval_piggy.triggered_anomalies
    assert eval_piggy.risk_score >= 0.55

    # Drop address trigger in tracker
    drop_prof = IdentityProfile(
        applicant_id="p_drop",
        ssn="050-12-3456",
        first_name="Drop",
        last_name="User",
        dob="1990-01-01",
        address="100 Commercial Mail Drop Suite PMB 99",
        phone="+12125550050",
        email="drop@example.com",
    )
    _, drop_anomalies, _ = tracker.check_and_record(drop_prof, now=now)
    assert SifAnomalyType.HIGH_RISK_ADDRESS_DROP in drop_anomalies
