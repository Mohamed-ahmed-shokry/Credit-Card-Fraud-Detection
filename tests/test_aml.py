"""Tests for AML transaction monitoring, typology detection, and SAR filing."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fraud_detection.aml import (
    AmlAction,
    AmlAlert,
    AmlRiskLevel,
    AmlTransactionEvent,
    AmlTypology,
    CircularFlowAlert,
    CircularFlowDetector,
    HighRiskJurisdictionDetector,
    JurisdictionRiskAssessment,
    LayeringAlert,
    LayeringDetector,
    SarDossier,
    SarStatus,
    ScreeningMatch,
    StructuringAlert,
    StructuringDetector,
    TransactionDirection,
    WatchlistEntry,
    WatchlistScreeningEngine,
    WatchlistType,
    jaro_similarity,
    jaro_winkler_similarity,
    levenshtein_distance,
    levenshtein_similarity,
    soundex,
)


def test_aml_enums() -> None:
    """Verify string enum mappings and values."""
    assert AmlTypology.STRUCTURING_SMURFING == "structuring_smurfing"
    assert AmlTypology.RAPID_MOVEMENT_LAYERING == "rapid_movement_layering"
    assert AmlTypology.ROUND_TRIP_CIRCULAR == "round_trip_circular"
    assert AmlTypology.SANCTION_PEP_MATCH == "sanction_pep_match"
    assert AmlTypology.HIGH_RISK_JURISDICTION == "high_risk_jurisdiction"
    assert AmlTypology.HIGH_VELOCITY_BURST == "high_velocity_burst"

    assert AmlRiskLevel.LOW == "low"
    assert AmlRiskLevel.CRITICAL == "critical"

    assert AmlAction.PASS == "pass"  # noqa: S105
    assert AmlAction.FILE_SAR_BLOCK == "file_sar_block"

    assert SarStatus.DRAFT == "draft"
    assert SarStatus.FILED == "filed"

    assert WatchlistType.OFAC_SDN == "ofac_sdn"
    assert TransactionDirection.INFLOW == "inflow"
    assert TransactionDirection.OUTFLOW == "outflow"


def test_watchlist_entry_serialization() -> None:
    """Verify WatchlistEntry serialization and deserialization."""
    entry = WatchlistEntry(
        entity_id="w-101",
        name="Viktor Bout",
        watchlist_type=WatchlistType.OFAC_SDN,
        country="RU",
        aliases=("Merchant of Death", "Boris"),
        listed_date="2008-03-06",
        notes="High risk arms dealer",
    )
    d = entry.to_dict()
    assert d["entity_id"] == "w-101"
    assert d["name"] == "Viktor Bout"
    assert d["watchlist_type"] == "ofac_sdn"
    assert "Boris" in d["aliases"]

    restored = WatchlistEntry.from_dict(d)
    assert restored.entity_id == entry.entity_id
    assert restored.name == entry.name
    assert restored.watchlist_type == WatchlistType.OFAC_SDN
    assert restored.aliases == ("Merchant of Death", "Boris")


def test_screening_match_serialization() -> None:
    """Verify ScreeningMatch serialization and deserialization."""
    entry = WatchlistEntry(
        entity_id="w-102",
        name="John Doe",
        watchlist_type=WatchlistType.PEP,
        country="US",
    )
    now = datetime.now(UTC)
    match = ScreeningMatch(
        match_id="m-501",
        entity_name="Jon Doe",
        matched_entry=entry,
        similarity_score=0.92,
        match_algorithm="jaro_winkler",
        risk_level=AmlRiskLevel.HIGH,
        screened_at=now,
    )
    d = match.to_dict()
    assert d["match_id"] == "m-501"
    assert d["similarity_score"] == 0.92
    assert d["matched_entry"]["name"] == "John Doe"

    restored = ScreeningMatch.from_dict(d)
    assert restored.match_id == "m-501"
    assert restored.entity_name == "Jon Doe"
    assert restored.matched_entry.name == "John Doe"
    assert restored.similarity_score == 0.92
    assert restored.risk_level == AmlRiskLevel.HIGH


def test_aml_transaction_event_serialization() -> None:
    """Verify AmlTransactionEvent serialization and deserialization."""
    now = datetime.now(UTC)
    tx = AmlTransactionEvent(
        transaction_id="tx-999",
        user_id="usr-123",
        amount=9500.0,
        currency="USD",
        direction=TransactionDirection.INFLOW,
        counterparty_id="usr-456",
        country="US",
        payment_method="ach",
        timestamp=now,
        tenant_id="tenant-alpha",
        metadata={"channel": "web"},
    )
    d = tx.to_dict()
    assert d["transaction_id"] == "tx-999"
    assert d["amount"] == 9500.0
    assert d["direction"] == "inflow"
    assert d["metadata"]["channel"] == "web"

    restored = AmlTransactionEvent.from_dict(d)
    assert restored.transaction_id == "tx-999"
    assert restored.amount == 9500.0
    assert restored.direction == TransactionDirection.INFLOW
    assert restored.metadata["channel"] == "web"


def test_structuring_alert_serialization() -> None:
    """Verify StructuringAlert serialization and deserialization."""
    alert = StructuringAlert(
        alert_id="str-1",
        user_id="usr-abc",
        transaction_ids=("tx-1", "tx-2", "tx-3"),
        total_amount=28500.0,
        transaction_count=3,
        window_seconds=86400.0,
        average_amount=9500.0,
        threshold_proximity=0.95,
    )
    d = alert.to_dict()
    assert d["total_amount"] == 28500.0
    assert d["transaction_count"] == 3
    assert len(d["transaction_ids"]) == 3

    restored = StructuringAlert.from_dict(d)
    assert restored.alert_id == "str-1"
    assert restored.transaction_count == 3
    assert restored.total_amount == 28500.0


def test_aml_alert_serialization() -> None:
    """Verify AmlAlert serialization and deserialization."""
    alert = AmlAlert(
        alert_id="alt-1",
        user_id="usr-xyz",
        typology=AmlTypology.STRUCTURING_SMURFING,
        risk_level=AmlRiskLevel.HIGH,
        recommended_action=AmlAction.ENHANCED_DUE_DILIGENCE,
        risk_score=0.82,
        trigger_details={"smurfing_streak": 3},
        transaction_ids=("tx-1", "tx-2"),
        tenant_id="tenant-beta",
    )
    d = alert.to_dict()
    assert d["alert_id"] == "alt-1"
    assert d["typology"] == "structuring_smurfing"
    assert d["risk_score"] == 0.82

    restored = AmlAlert.from_dict(d)
    assert restored.alert_id == "alt-1"
    assert restored.typology == AmlTypology.STRUCTURING_SMURFING
    assert restored.recommended_action == AmlAction.ENHANCED_DUE_DILIGENCE


def test_sar_dossier_serialization() -> None:
    """Verify SarDossier serialization and deserialization."""
    sar = SarDossier(
        sar_id="sar-001",
        user_id="usr-xyz",
        narrative="Customer conducted sequential ATM cash loads under $10,000 threshold.",
        alert_ids=("alt-1",),
        status=SarStatus.PENDING_REVIEW,
        suspect_details={"full_name": "Suspicious Individual", "ssn_hash": "abc..."},
        summary_financials={"total_suspicious_amount": 28500.0, "currency": "USD"},
        jurisdictions=("US",),
        filing_date="",
        tenant_id="tenant-beta",
        audit_trail=({"action": "created", "by": "system"},),
    )
    d = sar.to_dict()
    assert d["sar_id"] == "sar-001"
    assert d["status"] == "pending_review"
    assert len(d["audit_trail"]) == 1

    restored = SarDossier.from_dict(d)
    assert restored.sar_id == "sar-001"
    assert restored.status == SarStatus.PENDING_REVIEW
    assert restored.narrative.startswith("Customer conducted")
    assert restored.summary_financials["total_suspicious_amount"] == 28500.0


def test_soundex() -> None:
    """Verify American Soundex phonetic encoding."""
    assert soundex("Robert") == "R163"
    assert soundex("Rupert") == "R163"
    assert soundex("Rubin") == "R150"
    assert soundex("Ashcraft") == "A261"
    assert soundex("Ashcroft") == "A261"
    assert soundex("") == ""
    assert soundex("12345") == ""


def test_levenshtein() -> None:
    """Verify Levenshtein distance and similarity computation."""
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("", "test") == 4
    assert levenshtein_distance("same", "same") == 0

    assert levenshtein_similarity("same", "same") == 1.0
    assert levenshtein_similarity("", "") == 1.0
    assert levenshtein_similarity("kitten", "sitting") == 1.0 - (3 / 7)
    assert levenshtein_similarity("abc", "xyz") == 0.0


def test_jaro_winkler() -> None:
    """Verify Jaro and Jaro-Winkler string similarity calculations."""
    assert jaro_similarity("same", "same") == 1.0
    assert jaro_similarity("", "test") == 0.0
    assert jaro_similarity("test", "") == 0.0
    assert jaro_similarity("abc", "xyz") == 0.0

    # Standard benchmark: MARTHA vs MARHTA
    martha_jaro = jaro_similarity("MARTHA", "MARHTA")
    assert 0.94 < martha_jaro < 0.95

    # Jaro-Winkler gives higher score when prefix matches
    jw = jaro_winkler_similarity("MARTHA", "MARHTA")
    assert jw >= martha_jaro
    assert 0.96 < jw < 0.97

    # Exact match edge case
    assert jaro_winkler_similarity("ALICE", "ALICE") == 1.0
    assert jaro_winkler_similarity("", "") == 1.0
    assert jaro_winkler_similarity("A", "Z") == 0.0


def test_watchlist_screening_engine_crud() -> None:
    """Verify CRUD and listing operations on WatchlistScreeningEngine."""
    engine = WatchlistScreeningEngine()
    assert engine.count() == 0

    entry1 = WatchlistEntry(
        entity_id="sdn-1",
        name="Viktor Bout",
        watchlist_type=WatchlistType.OFAC_SDN,
        country="RU",
        aliases=("Boris",),
    )
    entry2 = WatchlistEntry(
        entity_id="pep-1",
        name="Alexander Lukashenko",
        watchlist_type=WatchlistType.PEP,
        country="BY",
    )

    engine.add_entry(entry1)
    engine.add_entry(entry2)
    assert engine.count() == 2

    assert engine.get_entry("sdn-1") == entry1
    assert engine.get_entry("non-existent") is None

    sdn_entries = engine.list_entries(watchlist_type=WatchlistType.OFAC_SDN)
    assert len(sdn_entries) == 1
    assert sdn_entries[0].entity_id == "sdn-1"

    all_entries = engine.list_entries()
    assert len(all_entries) == 2

    assert engine.remove_entry("sdn-1") is True
    assert engine.remove_entry("sdn-1") is False
    assert engine.count() == 1


def test_watchlist_screening_matching_algorithms() -> None:
    """Verify exact, token-sorted, typo-fuzzy, and phonetic watchlist matches."""
    engine = WatchlistScreeningEngine(default_threshold=0.80)
    entry = WatchlistEntry(
        entity_id="target-1",
        name="Osama Bin Laden",
        watchlist_type=WatchlistType.OFAC_SDN,
        country="SA",
        aliases=("Usama Bin Ladin", "The Emir"),
    )
    engine.add_entry(entry)

    # 1. Exact match
    hits = engine.screen_entity("Osama Bin Laden")
    assert len(hits) == 1
    assert hits[0].similarity_score == 1.0
    assert hits[0].match_algorithm == "exact"
    assert hits[0].risk_level == AmlRiskLevel.CRITICAL

    # 2. Alias match
    hits_alias = engine.screen_entity("Usama Bin Ladin")
    assert len(hits_alias) == 1
    assert hits_alias[0].similarity_score == 1.0

    # 3. Token-sorted match (e.g. inverted name)
    hits_sorted = engine.screen_entity("Bin Laden Osama")
    assert len(hits_sorted) == 1
    assert hits_sorted[0].similarity_score >= 0.98

    # 4. Typo / fuzzy match
    hits_typo = engine.screen_entity("Osama Bin Ladon", threshold=0.85)
    assert len(hits_typo) == 1
    assert hits_typo[0].similarity_score >= 0.90
    assert hits_typo[0].risk_level in (AmlRiskLevel.HIGH, AmlRiskLevel.CRITICAL)

    # 5. Threshold filter cutoff
    hits_unrelated = engine.screen_entity("John Smith", threshold=0.85)
    assert len(hits_unrelated) == 0

    # 6. Country filter
    hits_wrong_country = engine.screen_entity("Osama Bin Laden", country="US")
    assert len(hits_wrong_country) == 0

    hits_right_country = engine.screen_entity("Osama Bin Laden", country="SA")
    assert len(hits_right_country) == 1

    # 7. Blank string handling
    assert engine.screen_entity("") == []
    assert engine.screen_entity("   !@#$%   ") == []


def test_structuring_detector() -> None:
    """Verify structuring and smurfing detection logic."""
    detector = StructuringDetector(reporting_threshold=10000.0, sub_threshold_ratio=0.80)
    now = datetime.now(UTC)

    # 1. Single transaction under threshold -> no alert
    tx1 = AmlTransactionEvent(
        transaction_id="tx-1",
        user_id="usr-smurf",
        amount=9500.0,
        timestamp=now,
    )
    assert detector.record_and_evaluate(tx1) is None

    # 2. Second sub-threshold transaction -> triggers StructuringAlert
    tx2 = AmlTransactionEvent(
        transaction_id="tx-2",
        user_id="usr-smurf",
        amount=9200.0,
        timestamp=now + timedelta(minutes=15),
    )
    alert = detector.record_and_evaluate(tx2)
    assert alert is not None
    assert alert.user_id == "usr-smurf"
    assert alert.transaction_count == 2
    assert alert.total_amount == 18700.0
    assert alert.threshold_proximity > 0.90
    assert alert.to_dict()["transaction_count"] == 2

    # 3. Multiple small transactions accumulating above threshold
    detector.clear()
    t_a = AmlTransactionEvent("tx-a", "usr-acc", 3500.0, timestamp=now)
    t_b = AmlTransactionEvent("tx-b", "usr-acc", 4000.0, timestamp=now + timedelta(minutes=5))
    t_c = AmlTransactionEvent("tx-c", "usr-acc", 3000.0, timestamp=now + timedelta(minutes=10))

    assert detector.record_and_evaluate(t_a) is None
    assert detector.record_and_evaluate(t_b) is None
    alert_acc = detector.record_and_evaluate(t_c)
    assert alert_acc is not None
    assert alert_acc.transaction_count == 3
    assert alert_acc.total_amount == 10500.0


def test_layering_detector() -> None:
    """Verify rapid flow-through layering and dwell time detection."""
    detector = LayeringDetector(
        max_dwell_hours=2.0,
        min_flow_amount=2000.0,
        min_turnover_ratio=0.80,
    )
    now = datetime.now(UTC)

    # Inflow of $10,000
    inflow = AmlTransactionEvent(
        transaction_id="in-1",
        user_id="usr-layer",
        amount=10000.0,
        direction=TransactionDirection.INFLOW,
        timestamp=now,
    )
    assert detector.record_and_evaluate(inflow) is None

    # Outflow of $9,500 after 30 minutes (95% turnover, 30 min dwell)
    outflow = AmlTransactionEvent(
        transaction_id="out-1",
        user_id="usr-layer",
        amount=9500.0,
        direction=TransactionDirection.OUTFLOW,
        timestamp=now + timedelta(minutes=30),
    )
    alert = detector.record_and_evaluate(outflow)
    assert alert is not None
    assert alert.user_id == "usr-layer"
    assert alert.inflow_amount == 10000.0
    assert alert.outflow_amount == 9500.0
    assert alert.turnover_ratio == 0.95
    assert 29.0 < alert.dwell_minutes < 31.0
    assert LayeringAlert.from_dict(alert.to_dict()).alert_id == alert.alert_id

    # Low amount outflow should not trigger
    detector.clear()
    detector.record_and_evaluate(inflow)
    small_outflow = AmlTransactionEvent(
        transaction_id="out-small",
        user_id="usr-layer",
        amount=500.0,
        direction=TransactionDirection.OUTFLOW,
        timestamp=now + timedelta(minutes=10),
    )
    assert detector.record_and_evaluate(small_outflow) is None


def test_circular_flow_detector() -> None:
    """Verify circular and round-trip transaction loop detection."""
    detector = CircularFlowDetector(window_seconds=86400.0, min_amount=500.0, amount_tolerance=0.10)
    now = datetime.now(UTC)

    # Alice sends Bob $5,000 (Alice is user, Bob is counterparty)
    tx_fwd = AmlTransactionEvent(
        transaction_id="tx-fwd",
        user_id="Alice",
        counterparty_id="Bob",
        amount=5000.0,
        direction=TransactionDirection.OUTFLOW,
        timestamp=now,
    )
    assert detector.record_and_evaluate(tx_fwd) is None

    # Bob sends Alice $4,900 after 2 hours
    tx_ret = AmlTransactionEvent(
        transaction_id="tx-ret",
        user_id="Bob",
        counterparty_id="Alice",
        amount=4900.0,
        direction=TransactionDirection.OUTFLOW,
        timestamp=now + timedelta(hours=2),
    )
    alert = detector.record_and_evaluate(tx_ret)
    assert alert is not None
    assert alert.origin_user_id == "Bob"
    assert alert.counterparty_id == "Alice"
    assert alert.forward_amount == 5000.0
    assert alert.return_amount == 4900.0
    assert 1.9 < alert.cycle_hours < 2.1
    assert CircularFlowAlert.from_dict(alert.to_dict()).alert_id == alert.alert_id

    # Clear and verify empty
    detector.clear()
    assert detector.record_and_evaluate(tx_ret) is None


def test_high_risk_jurisdiction_detector() -> None:
    """Verify FATF blacklist, graylist, and custom jurisdiction evaluation."""
    detector = HighRiskJurisdictionDetector()

    # Blacklist evaluation
    ir_eval = detector.evaluate("IR")
    assert ir_eval.is_high_risk is True
    assert ir_eval.fatf_tier == "black_list"
    assert ir_eval.risk_score == 1.0

    # Graylist evaluation
    sy_eval = detector.evaluate("SY")
    assert sy_eval.is_high_risk is True
    assert sy_eval.fatf_tier == "gray_list"
    assert sy_eval.risk_score == 0.60

    # Standard jurisdiction evaluation
    us_eval = detector.evaluate("US")
    assert us_eval.is_high_risk is False
    assert us_eval.fatf_tier == "standard"
    assert us_eval.risk_score == 0.0

    # Custom addition and removal
    detector.add_jurisdiction("XX", tier="black_list")
    assert detector.evaluate("XX").fatf_tier == "black_list"
    assert detector.remove_jurisdiction("XX") is True
    assert detector.evaluate("XX").is_high_risk is False

    # Serialization
    d = ir_eval.to_dict()
    restored = JurisdictionRiskAssessment.from_dict(d)
    assert restored.country == "IR"
    assert restored.risk_score == 1.0
