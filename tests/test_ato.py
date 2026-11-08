"""Unit tests for Account Takeover (ATO) domain models and enums."""

from datetime import UTC, datetime

import pytest

from fraud_detection.ato import (
    AccountMutationRecord,
    AccountMutationType,
    AtoAction,
    AtoRiskAssessment,
    AtoRiskFactor,
    LoginEvent,
    LoginOutcome,
)


def test_ato_enums() -> None:
    """Verify all enum values match expected identifiers."""
    assert AtoRiskFactor.NEW_DEVICE.value == "new_device"
    assert AtoRiskFactor.IMPOSSIBLE_TRAVEL.value == "impossible_travel"
    assert AtoRiskFactor.RECENT_CREDENTIAL_CHANGE.value == "recent_credential_change"
    assert AtoRiskFactor.RECENT_MFA_RESET.value == "recent_mfa_reset"
    assert AtoRiskFactor.CREDENTIAL_STUFFING_BURST.value == "credential_stuffing_burst"
    assert AtoRiskFactor.PASSWORD_SPRAYING.value == "password_spraying"
    assert AtoRiskFactor.DORMANT_ACCOUNT_REACTIVE.value == "dormant_account_reactive"
    assert AtoRiskFactor.VELOCITY_SPIKE.value == "velocity_spike"
    assert AtoRiskFactor.ACCOUNT_QUARANTINED.value == "account_quarantined"

    assert AtoAction.ALLOW.value == "allow"
    assert AtoAction.CHALLENGE.value == "challenge"
    assert AtoAction.QUARANTINE_ACCOUNT.value == "quarantine_account"
    assert AtoAction.BLOCK.value == "block"

    assert LoginOutcome.SUCCESS.value == "success"
    assert LoginOutcome.FAILED_CREDENTIALS.value == "failed_credentials"
    assert LoginOutcome.FAILED_MFA.value == "failed_mfa"
    assert LoginOutcome.BLOCKED.value == "blocked"

    assert AccountMutationType.PASSWORD_CHANGE.value == "password_change"
    assert AccountMutationType.EMAIL_CHANGE.value == "email_change"
    assert AccountMutationType.PHONE_CHANGE.value == "phone_change"
    assert AccountMutationType.MFA_RESET.value == "mfa_reset"
    assert AccountMutationType.SHIPPING_ADDRESS_CHANGE.value == "shipping_address_change"


def test_account_mutation_record_serialization() -> None:
    """Test AccountMutationRecord to_dict and from_dict roundtrip."""
    now = datetime(2026, 11, 8, 12, 0, 0, tzinfo=UTC)
    record = AccountMutationRecord(
        mutation_type=AccountMutationType.PASSWORD_CHANGE,
        timestamp=now,
        channel="mobile_app",
        metadata={"ip": "1.2.3.4", "device": "ios_v12"},
    )

    data = record.to_dict()
    assert data["mutation_id"].startswith("mut_")
    assert data["mutation_type"] == "password_change"
    assert data["timestamp"] == now.isoformat()
    assert data["channel"] == "mobile_app"
    assert data["metadata"]["ip"] == "1.2.3.4"

    rebuilt = AccountMutationRecord.from_dict(data)
    assert rebuilt.mutation_id == record.mutation_id
    assert rebuilt.mutation_type == AccountMutationType.PASSWORD_CHANGE
    assert rebuilt.timestamp == now
    assert rebuilt.channel == "mobile_app"
    assert rebuilt.metadata == record.metadata

    # Test defaults
    naive_data = {"mutation_type": "email_change", "timestamp": "2026-11-08T10:00:00"}
    naive_rebuilt = AccountMutationRecord.from_dict(naive_data)
    assert naive_rebuilt.timestamp.tzinfo == UTC
    assert naive_rebuilt.channel == "web"

    # Test fallback timestamp when missing
    missing_ts = AccountMutationRecord.from_dict({"mutation_type": "mfa_reset"})
    assert isinstance(missing_ts.timestamp, datetime)
    assert missing_ts.timestamp.tzinfo == UTC


def test_login_event_serialization() -> None:
    """Test LoginEvent creation and roundtrip serialization."""
    now = datetime(2026, 11, 8, 12, 30, 0, tzinfo=UTC)
    event = LoginEvent(
        user_id="usr_12345",
        tenant_id="tenant_a",
        timestamp=now,
        ip_address="198.51.100.42",
        outcome=LoginOutcome.FAILED_CREDENTIALS,
        device_fingerprint="fp_abc123",
        latitude=40.7128,
        longitude=-74.0060,
        user_agent="Mozilla/5.0",
    )

    data = event.to_dict()
    assert data["event_id"].startswith("lgn_")
    assert data["user_id"] == "usr_12345"
    assert data["outcome"] == "failed_credentials"
    assert data["latitude"] == 40.7128

    rebuilt = LoginEvent.from_dict(data)
    assert rebuilt.event_id == event.event_id
    assert rebuilt.user_id == "usr_12345"
    assert rebuilt.outcome == LoginOutcome.FAILED_CREDENTIALS
    assert rebuilt.latitude == 40.7128
    assert rebuilt.longitude == -74.0060

    # Test defaults
    minimal_data = {"user_id": "usr_999"}
    minimal_rebuilt = LoginEvent.from_dict(minimal_data)
    assert minimal_rebuilt.tenant_id == "default"
    assert minimal_rebuilt.outcome == LoginOutcome.SUCCESS
    assert minimal_rebuilt.ip_address == "127.0.0.1"


def test_ato_risk_assessment_validation_and_serialization() -> None:
    """Test AtoRiskAssessment invariant validation and roundtrip."""
    with pytest.raises(ValueError, match="risk_score must be in range"):
        AtoRiskAssessment(
            user_id="usr_test",
            risk_score=1.5,
            recommended_action=AtoAction.ALLOW,
        )

    with pytest.raises(ValueError, match="risk_score must be in range"):
        AtoRiskAssessment(
            user_id="usr_test",
            risk_score=-0.1,
            recommended_action=AtoAction.ALLOW,
        )

    now = datetime.now(UTC)
    assessment = AtoRiskAssessment(
        user_id="usr_victim",
        tenant_id="corp_1",
        risk_score=0.85,
        recommended_action=AtoAction.QUARANTINE_ACCOUNT,
        factors_triggered=(
            AtoRiskFactor.IMPOSSIBLE_TRAVEL,
            AtoRiskFactor.RECENT_CREDENTIAL_CHANGE,
        ),
        score_breakdown={"travel": 0.45, "mutation": 0.40},
        impossible_travel_kmh=1250.5,
        is_new_device=True,
        hours_since_last_mutation=0.75,
        quarantined=True,
        timestamp=now,
        details={"case_id": "case_101"},
    )

    data = assessment.to_dict()
    assert data["assessment_id"].startswith("ato_")
    assert data["risk_score"] == 0.85
    assert data["recommended_action"] == "quarantine_account"
    assert "impossible_travel" in data["factors_triggered"]
    assert data["impossible_travel_kmh"] == 1250.5
    assert data["is_new_device"] is True
    assert data["quarantined"] is True

    rebuilt = AtoRiskAssessment.from_dict(data)
    assert rebuilt.assessment_id == assessment.assessment_id
    assert rebuilt.user_id == "usr_victim"
    assert rebuilt.risk_score == 0.85
    assert rebuilt.recommended_action == AtoAction.QUARANTINE_ACCOUNT
    assert AtoRiskFactor.IMPOSSIBLE_TRAVEL in rebuilt.factors_triggered
    assert rebuilt.impossible_travel_kmh == 1250.5
    assert rebuilt.quarantined is True


def test_credential_stuffing_ip_spraying() -> None:
    """Test detection of an IP spraying failed passwords across multiple accounts."""
    from datetime import timedelta

    from fraud_detection.ato import CredentialStuffingDetector

    detector = CredentialStuffingDetector(window_seconds=300, ip_spray_user_threshold=3)
    now = datetime(2026, 11, 8, 14, 0, 0, tzinfo=UTC)

    # 1st failure: user_1
    detector.record_attempt(
        LoginEvent(
            user_id="user_1",
            ip_address="203.0.113.5",
            outcome=LoginOutcome.FAILED_CREDENTIALS,
            timestamp=now,
        )
    )
    is_spray, score, count = detector.check_ip_spraying("203.0.113.5", now)
    assert not is_spray
    assert count == 1
    assert score == 0.0

    # 2nd failure: user_2
    detector.record_attempt(
        LoginEvent(
            user_id="user_2",
            ip_address="203.0.113.5",
            outcome=LoginOutcome.FAILED_CREDENTIALS,
            timestamp=now + timedelta(seconds=10),
        )
    )
    is_spray, score, count = detector.check_ip_spraying("203.0.113.5", now + timedelta(seconds=10))
    assert not is_spray
    assert count == 2

    # 3rd failure: user_3 -> triggers threshold
    detector.record_attempt(
        LoginEvent(
            user_id="user_3",
            ip_address="203.0.113.5",
            outcome=LoginOutcome.FAILED_CREDENTIALS,
            timestamp=now + timedelta(seconds=20),
        )
    )
    is_spray, score, count = detector.check_ip_spraying("203.0.113.5", now + timedelta(seconds=20))
    assert is_spray
    assert count == 3
    assert score >= 0.5

    # Evaluation returns PASSWORD_SPRAYING factor
    eval_event = LoginEvent(
        user_id="user_4",
        ip_address="203.0.113.5",
        outcome=LoginOutcome.FAILED_CREDENTIALS,
        timestamp=now + timedelta(seconds=25),
    )
    eval_score, factors = detector.evaluate_stuffing_risk(eval_event)
    assert AtoRiskFactor.PASSWORD_SPRAYING in factors
    assert eval_score >= 0.5


def test_credential_stuffing_distributed_brute_force() -> None:
    """Test detection of a single user attacked from multiple distinct IPs."""
    from datetime import timedelta

    from fraud_detection.ato import CredentialStuffingDetector

    detector = CredentialStuffingDetector(window_seconds=300, distributed_user_ip_threshold=3)
    now = datetime(2026, 11, 8, 14, 0, 0, tzinfo=UTC)

    # User targeted from 3 different IPs
    for i in range(3):
        detector.record_attempt(
            LoginEvent(
                user_id="target_vip",
                ip_address=f"192.0.2.{10 + i}",
                outcome=LoginOutcome.FAILED_CREDENTIALS,
                timestamp=now + timedelta(seconds=i * 5),
            )
        )

    is_bf, score, count = detector.check_distributed_brute_force(
        "target_vip", now + timedelta(seconds=20)
    )
    assert is_bf
    assert count == 3
    assert score >= 0.5

    eval_event = LoginEvent(
        user_id="target_vip",
        ip_address="192.0.2.99",
        timestamp=now + timedelta(seconds=20),
    )
    eval_score, factors = detector.evaluate_stuffing_risk(eval_event)
    assert AtoRiskFactor.CREDENTIAL_STUFFING_BURST in factors
    assert eval_score >= 0.5


def test_credential_stuffing_subnet_burst_and_window_expiry() -> None:
    """Test subnet-level burst detection and window expiry."""
    from datetime import timedelta

    from fraud_detection.ato import CredentialStuffingDetector

    detector = CredentialStuffingDetector(window_seconds=60, subnet_spray_threshold=3)
    t0 = datetime(2026, 11, 8, 15, 0, 0, tzinfo=UTC)

    # 3 failures across different IPs in 198.51.100.0/24 targeting 3 distinct users
    for i in range(3):
        detector.record_attempt(
            LoginEvent(
                user_id=f"user_{i}",
                ip_address=f"198.51.100.{10 + i}",
                outcome=LoginOutcome.FAILED_CREDENTIALS,
                timestamp=t0 + timedelta(seconds=i * 2),
            )
        )

    is_sub, score, count = detector.check_subnet_burst("198.51.100.50", t0 + timedelta(seconds=10))
    assert is_sub
    assert count == 3
    assert score >= 0.4

    # Test window expiry: at t0 + 120 seconds, previous attempts are pruned
    t_later = t0 + timedelta(seconds=120)
    is_sub_expired, score_expired, count_expired = detector.check_subnet_burst(
        "198.51.100.50", t_later
    )
    assert not is_sub_expired
    assert count_expired == 0
    assert score_expired == 0.0

    # Test IPv6 subnet extraction
    detector.record_attempt(
        LoginEvent(
            user_id="ipv6_victim",
            ip_address="2001:db8:85a3::8a2e:370:7334",
            outcome=LoginOutcome.FAILED_CREDENTIALS,
            timestamp=t_later,
        )
    )
    is_sub_v6, _, _ = detector.check_subnet_burst("2001:db8:85a3::1", t_later)
    assert not is_sub_v6  # only 1 attempt, below threshold of 3

    # Test invalid IP fallback
    invalid_sub = detector._extract_subnet("not-an-ip")  # noqa: SLF001
    assert invalid_sub == "not-an-ip"

    # Querying unrecorded IP / user
    assert detector.check_ip_spraying("10.0.0.1")[0] is False
    assert detector.check_distributed_brute_force("unknown_user")[0] is False
    assert detector.check_subnet_burst("10.0.0.1")[0] is False


def test_account_profile_tracker_devices_and_locations() -> None:
    """Test AccountProfileTracker tracking known devices and locations."""
    from fraud_detection.ato import AccountProfileTracker

    tracker = AccountProfileTracker()
    now = datetime(2026, 11, 8, 16, 0, 0, tzinfo=UTC)

    # Initial state: unknown device
    assert tracker.is_device_recognized("user_alice", "dev_laptop") is False

    # Record login
    login_ev = LoginEvent(
        user_id="user_alice",
        ip_address="198.51.100.12",
        outcome=LoginOutcome.SUCCESS,
        device_fingerprint="dev_laptop",
        latitude=37.7749,
        longitude=-122.4194,
        timestamp=now,
    )
    tracker.record_login_success(login_ev)

    # Now recognized
    assert tracker.is_device_recognized("user_alice", "dev_laptop") is True
    assert tracker.is_device_recognized("user_alice", "dev_phone") is False
    assert tracker.is_device_recognized("user_alice", None) is False

    profile = tracker.get_or_create_profile("user_alice")
    assert profile.last_known_ip == "198.51.100.12"
    assert profile.last_known_location == (37.7749, -122.4194)
    assert (37.7749, -122.4194) in profile.known_locations
    assert profile.last_active_at == now

    # Roundtrip serialization of AccountProfile
    dict_rep = profile.to_dict()
    assert dict_rep["user_id"] == "user_alice"
    assert "dev_laptop" in dict_rep["known_devices"]
    rebuilt_profile = type(profile).from_dict(dict_rep)
    assert rebuilt_profile.user_id == "user_alice"
    assert "dev_laptop" in rebuilt_profile.known_devices
    assert rebuilt_profile.last_known_location == (37.7749, -122.4194)


def test_account_profile_tracker_mutation_penalties() -> None:
    """Test time-decayed mutation risk penalties."""
    from datetime import timedelta

    from fraud_detection.ato import AccountProfileTracker

    tracker = AccountProfileTracker(mutation_risk_window_hours=72.0)
    now = datetime(2026, 11, 8, 16, 0, 0, tzinfo=UTC)

    # No mutations -> 0 penalty
    penalty, hrs, factors = tracker.calculate_mutation_penalty("user_bob", now)
    assert penalty == 0.0
    assert hrs is None
    assert factors == []

    # Mutation 30 minutes ago (<= 1.0 hour) -> penalty 0.40
    mut1 = AccountMutationRecord(
        mutation_type=AccountMutationType.PASSWORD_CHANGE,
        timestamp=now - timedelta(minutes=30),
    )
    tracker.record_mutation("user_bob", mut1)
    penalty, hrs, factors = tracker.calculate_mutation_penalty("user_bob", now)
    assert penalty == 0.40
    assert hrs is not None and hrs < 1.0
    assert AtoRiskFactor.RECENT_CREDENTIAL_CHANGE in factors

    # Mutation 12 hours ago (<= 24.0 hours) -> penalty 0.20
    mut2 = AccountMutationRecord(
        mutation_type=AccountMutationType.MFA_RESET,
        timestamp=now - timedelta(hours=12),
    )
    tracker = AccountProfileTracker(mutation_risk_window_hours=72.0)
    tracker.record_mutation("user_charlie", mut2)
    penalty, hrs, factors = tracker.calculate_mutation_penalty("user_charlie", now)
    assert penalty == 0.20
    assert hrs is not None and 11.0 < hrs < 13.0
    assert AtoRiskFactor.RECENT_MFA_RESET in factors

    # Mutation 48 hours ago (<= 72.0 hours) -> penalty 0.10
    mut3 = AccountMutationRecord(
        mutation_type=AccountMutationType.EMAIL_CHANGE,
        timestamp=now - timedelta(hours=48),
    )
    tracker = AccountProfileTracker(mutation_risk_window_hours=72.0)
    tracker.record_mutation("user_david", mut3)
    penalty, hrs, factors = tracker.calculate_mutation_penalty("user_david", now)
    assert penalty == 0.10
    assert hrs is not None and 47.0 < hrs < 49.0
    assert AtoRiskFactor.RECENT_CREDENTIAL_CHANGE in factors

    # Mutation 100 hours ago (> 72 hours) -> 0.0
    mut4 = AccountMutationRecord(
        mutation_type=AccountMutationType.PASSWORD_CHANGE,
        timestamp=now - timedelta(hours=100),
    )
    tracker = AccountProfileTracker(mutation_risk_window_hours=72.0)
    tracker.record_mutation("user_eve", mut4)
    penalty, hrs, factors = tracker.calculate_mutation_penalty("user_eve", now)
    assert penalty == 0.0
    assert hrs is None
    assert factors == []


def test_account_profile_tracker_dormancy_and_quarantine() -> None:
    """Test dormancy detection and quarantine lifecycle."""
    from datetime import timedelta

    from fraud_detection.ato import AccountProfileTracker

    tracker = AccountProfileTracker(dormancy_threshold_days=90)
    now = datetime(2026, 11, 8, 16, 0, 0, tzinfo=UTC)

    # Fresh account without last_active_at
    is_dormant, penalty = tracker.is_dormant_reactivation("user_fresh", now)
    assert not is_dormant
    assert penalty == 0.0

    # Account active 100 days ago
    login_old = LoginEvent(
        user_id="user_dormant",
        timestamp=now - timedelta(days=100),
    )
    tracker.record_login_success(login_old)

    is_dormant, penalty = tracker.is_dormant_reactivation("user_dormant", now)
    assert is_dormant
    assert penalty == 0.15

    # Account active 10 days ago
    login_recent = LoginEvent(
        user_id="user_active",
        timestamp=now - timedelta(days=10),
    )
    tracker.record_login_success(login_recent)
    is_dormant, penalty = tracker.is_dormant_reactivation("user_active", now)
    assert not is_dormant
    assert penalty == 0.0

    # Quarantine and release
    assert tracker.is_quarantined("user_compromised") is False
    tracker.quarantine_account("user_compromised", "Credential stuffing detection confirmed")
    assert tracker.is_quarantined("user_compromised") is True

    profile = tracker.get_or_create_profile("user_compromised")
    assert profile.is_quarantined is True
    assert profile.quarantine_reason == "Credential stuffing detection confirmed"
    assert profile.quarantined_at is not None

    tracker.release_account("user_compromised", "Identity verified via analyst")
    assert tracker.is_quarantined("user_compromised") is False
    assert profile.is_quarantined is False
    assert profile.quarantine_reason is None
    assert profile.released_at is not None


def test_haversine_distance_calculation() -> None:
    """Test haversine distance calculation between known coordinates."""
    from fraud_detection.ato import calculate_haversine_distance_km

    # London (51.5074, -0.1278) to Paris (48.8566, 2.3522) ~ 343 km
    dist = calculate_haversine_distance_km(51.5074, -0.1278, 48.8566, 2.3522)
    assert 340.0 < dist < 350.0

    # Same location -> 0 km
    assert calculate_haversine_distance_km(40.0, -74.0, 40.0, -74.0) == 0.0


def test_ato_store_crud_and_indexing() -> None:
    """Test AtoStore persistence, capacity eviction, and user indexing."""
    from fraud_detection.ato import AtoStore

    store = AtoStore(max_capacity=5)
    now = datetime.now(UTC)

    # Record 6 assessments to trigger capacity eviction of oldest
    for i in range(6):
        assessment = AtoRiskAssessment(
            user_id="user_store_test",
            risk_score=0.1 * i,
            recommended_action=AtoAction.ALLOW,
            timestamp=now,
        )
        store.record_assessment(assessment)

    # Store size capped at 5
    assessments = store.list_assessments()
    assert len(assessments) == 5

    # Check user assessments query
    user_assessments = store.get_user_assessments("user_store_test", limit=3)
    assert len(user_assessments) == 3

    # Check event logging
    ev = LoginEvent(user_id="user_store_test", timestamp=now)
    store.record_event(ev)
    assert len(store.list_events(user_id="user_store_test")) == 1
    assert len(store.list_events(user_id="other_user")) == 0


def test_ato_engine_full_evaluation_flow() -> None:
    """Test AtoEngine multi-factor risk scoring, impossible travel, and auto-quarantine."""
    from datetime import timedelta

    from fraud_detection.ato import AccountProfileTracker, AtoEngine, CredentialStuffingDetector

    profile_tracker = AccountProfileTracker(dormancy_threshold_days=60)
    stuffing_detector = CredentialStuffingDetector()
    engine = AtoEngine(
        stuffing_detector=stuffing_detector,
        profile_tracker=profile_tracker,
        auto_quarantine=True,
        challenge_threshold=0.35,
        quarantine_threshold=0.70,
    )
    t0 = datetime(2026, 11, 8, 12, 0, 0, tzinfo=UTC)

    # Step 1: Normal initial login from London (success, recognized device created)
    ev1 = LoginEvent(
        user_id="user_traveler",
        device_fingerprint="device_laptop_win",
        latitude=51.5074,
        longitude=-0.1278,
        timestamp=t0,
        outcome=LoginOutcome.SUCCESS,
    )
    res1 = engine.evaluate_attempt(ev1)
    assert res1.recommended_action == AtoAction.ALLOW
    assert not res1.quarantined

    # Step 2: 30 minutes later, login from Tokyo (35.6762, 139.6503) from a new device!
    # Distance is ~9560 km, elapsed time 0.5 hours -> speed ~ 19,120 km/h -> IMPOSSIBLE TRAVEL!
    t1 = t0 + timedelta(minutes=30)
    ev2 = LoginEvent(
        user_id="user_traveler",
        device_fingerprint="device_unknown_hacker",
        latitude=35.6762,
        longitude=139.6503,
        timestamp=t1,
        outcome=LoginOutcome.SUCCESS,
    )
    res2 = engine.evaluate_attempt(ev2)
    assert AtoRiskFactor.NEW_DEVICE in res2.factors_triggered
    assert AtoRiskFactor.IMPOSSIBLE_TRAVEL in res2.factors_triggered
    assert res2.impossible_travel_kmh is not None and res2.impossible_travel_kmh > 10_000.0
    assert res2.risk_score >= 0.50
    assert res2.recommended_action in (AtoAction.CHALLENGE, AtoAction.QUARANTINE_ACCOUNT)

    # Step 3: Add recent password change mutation -> pushes score over quarantine threshold!
    profile_tracker.record_mutation(
        "user_traveler",
        AccountMutationRecord(
            mutation_type=AccountMutationType.PASSWORD_CHANGE,
            timestamp=t1 - timedelta(minutes=20),
        ),
    )
    t2 = t1 + timedelta(minutes=5)
    ev3 = LoginEvent(
        user_id="user_traveler",
        device_fingerprint="device_unknown_hacker",
        latitude=35.6762,
        longitude=139.6503,
        timestamp=t2,
        outcome=LoginOutcome.SUCCESS,
    )
    res3 = engine.evaluate_attempt(ev3)
    assert res3.risk_score >= 0.70
    assert res3.recommended_action == AtoAction.QUARANTINE_ACCOUNT
    assert res3.quarantined is True
    assert profile_tracker.is_quarantined("user_traveler") is True

    # Step 4: Subsequent attempt while quarantined -> immediate BLOCK
    t3 = t2 + timedelta(minutes=2)
    ev4 = LoginEvent(
        user_id="user_traveler",
        timestamp=t3,
    )
    res4 = engine.evaluate_attempt(ev4)
    assert res4.recommended_action == AtoAction.BLOCK
    assert res4.risk_score == 1.0
    assert AtoRiskFactor.ACCOUNT_QUARANTINED in res4.factors_triggered


def test_ato_event_bus_integration() -> None:
    """Test publishing of ATO lifecycle domain events across EventBus."""
    from fraud_detection.ato import AtoEngine
    from fraud_detection.events import EventBus, EventType, FraudDomainEvent

    bus = EventBus()
    received_events: list[FraudDomainEvent] = []

    def handle_event(event: FraudDomainEvent) -> None:
        received_events.append(event)

    bus.subscribe(handle_event, event_type=EventType.ATO_LOGIN_EVALUATED)
    bus.subscribe(handle_event, event_type=EventType.ATO_ACCOUNT_QUARANTINED)
    bus.subscribe(handle_event, event_type=EventType.ATO_ACCOUNT_RELEASED)
    bus.subscribe(handle_event, event_type=EventType.ATO_CREDENTIAL_STUFFING_BURST)

    engine = AtoEngine(event_bus=bus, auto_quarantine=True, quarantine_threshold=0.50)
    now = datetime.now(UTC)

    # 1. Normal login -> ATO_LOGIN_EVALUATED
    ev1 = LoginEvent(
        user_id="user_event_test",
        device_fingerprint="device_1",
        timestamp=now,
    )
    engine.evaluate_attempt(ev1)
    assert len(received_events) >= 1
    types = [e.event_type for e in received_events]
    assert EventType.ATO_LOGIN_EVALUATED in types

    # 2. Manual quarantine -> ATO_ACCOUNT_QUARANTINED
    engine.quarantine_account("user_event_test", "Analyst confirmed suspicion")
    types = [e.event_type for e in received_events]
    assert EventType.ATO_ACCOUNT_QUARANTINED in types

    # 3. Release account -> ATO_ACCOUNT_RELEASED
    engine.release_account("user_event_test", "Identity verified via telephone")
    types = [e.event_type for e in received_events]
    assert EventType.ATO_ACCOUNT_RELEASED in types
