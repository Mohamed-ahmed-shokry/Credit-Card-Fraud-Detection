"""Unit tests for real-time merchant & MCC risk profiling and chargeback surveillance."""

from __future__ import annotations

import pytest

from fraud_detection.merchant import (
    MERCHANT_FEATURE_NAMES,
    MCCCategory,
    MCCRiskProfile,
    MerchantAnomalyReport,
    MerchantFeatureExtractor,
    MerchantProfile,
    MerchantRiskError,
    MerchantTracker,
    MerchantTrackerConfig,
    MerchantTransactionEvent,
    calculate_empirical_bayes_fraud_rate,
    calculate_wilson_upper_bound,
    resolve_mcc_profile,
)


def test_mcc_risk_profile_validation_and_serialization() -> None:
    profile = MCCRiskProfile(
        mcc=" 7995 ",
        category=MCCCategory.GAMBLING_CRYPTO,
        baseline_fraud_rate=0.045,
        risk_score=0.90,
        description="Betting and Casino Gambling",
    )
    assert profile.mcc == "7995"
    data = profile.to_dict()
    assert data["mcc"] == "7995"
    assert data["category"] == "gambling_crypto"
    restored = MCCRiskProfile.from_dict(data)
    assert restored == profile

    with pytest.raises(MerchantRiskError, match="non-empty"):
        MCCRiskProfile("", MCCCategory.UNKNOWN, 0.01, 0.30)
    with pytest.raises(MerchantRiskError, match="4-digit"):
        MCCRiskProfile("123", MCCCategory.UNKNOWN, 0.01, 0.30)
    with pytest.raises(MerchantRiskError, match="Invalid MCCCategory"):
        MCCRiskProfile("5411", "not_a_category", 0.01, 0.30)  # type: ignore[arg-type]
    with pytest.raises(MerchantRiskError, match="baseline_fraud_rate"):
        MCCRiskProfile("5411", MCCCategory.UNKNOWN, -0.1, 0.30)
    with pytest.raises(MerchantRiskError, match="baseline_fraud_rate"):
        MCCRiskProfile("5411", MCCCategory.UNKNOWN, True, 0.30)  # type: ignore[arg-type]
    with pytest.raises(MerchantRiskError, match="risk_score"):
        MCCRiskProfile("5411", MCCCategory.UNKNOWN, 0.01, 1.5)
    with pytest.raises(MerchantRiskError, match="risk_score"):
        MCCRiskProfile("5411", MCCCategory.UNKNOWN, 0.01, False)  # type: ignore[arg-type]
    with pytest.raises(MerchantRiskError, match="description"):
        MCCRiskProfile("5411", MCCCategory.UNKNOWN, 0.01, 0.30, 123)  # type: ignore[arg-type]
    with pytest.raises(MerchantRiskError, match="mapping"):
        MCCRiskProfile.from_dict("invalid")  # type: ignore[arg-type]
    with pytest.raises(MerchantRiskError, match="Invalid MCCRiskProfile data"):
        MCCRiskProfile.from_dict({"mcc": "5411"})


def test_resolve_mcc_profile_known_and_ranges() -> None:
    assert resolve_mcc_profile("7995").category == MCCCategory.GAMBLING_CRYPTO
    assert resolve_mcc_profile(5411).category == MCCCategory.GROCERY_UTILITIES
    assert resolve_mcc_profile("5411.0").mcc == "5411"
    assert resolve_mcc_profile("742").mcc == "0742"
    assert resolve_mcc_profile("3501").category == MCCCategory.TRAVEL_HOSPITALITY
    assert resolve_mcc_profile("4814").category == MCCCategory.GROCERY_UTILITIES
    assert resolve_mcc_profile("5651").category == MCCCategory.RETAIL_COMMERCE
    assert resolve_mcc_profile("6211").category == MCCCategory.MONEY_TRANSFER
    assert resolve_mcc_profile("7996").category == MCCCategory.GAMBLING_CRYPTO
    assert resolve_mcc_profile("8011").category == MCCCategory.UNKNOWN
    assert resolve_mcc_profile(None).mcc == "0000"
    assert resolve_mcc_profile(True).mcc == "0000"
    assert resolve_mcc_profile("").mcc == "0000"
    assert resolve_mcc_profile("ABCD").mcc == "0000"


def test_empirical_bayes_and_wilson_upper_bound() -> None:
    rate0 = calculate_empirical_bayes_fraud_rate(0, 0, prior_rate=0.02, prior_strength=20.0)
    assert rate0 == pytest.approx(0.02)
    rate_obs = calculate_empirical_bayes_fraud_rate(2, 20, prior_rate=0.02, prior_strength=20.0)
    assert rate_obs == pytest.approx((2.0 + 0.4) / 40.0)

    for bad_args in [
        (-1, 10, 0.01, 20.0),
        (True, 10, 0.01, 20.0),
        (0, -1, 0.01, 20.0),
        (0, False, 0.01, 20.0),
        (5, 3, 0.01, 20.0),
        (1, 10, -0.1, 20.0),
        (1, 10, True, 20.0),
        (1, 10, 0.01, 0.0),
        (1, 10, 0.01, False),
    ]:
        with pytest.raises(MerchantRiskError):
            calculate_empirical_bayes_fraud_rate(*bad_args)  # type: ignore[arg-type]

    assert calculate_wilson_upper_bound(0, 0) == 0.0
    wb = calculate_wilson_upper_bound(5, 100)
    assert 0.05 < wb < 0.15
    for bad_w in [(-1, 10, 1.96), (0, -1, 1.96), (5, 2, 1.96), (1, 10, 0.0), (1, 10, True)]:
        with pytest.raises(MerchantRiskError):
            calculate_wilson_upper_bound(*bad_w)  # type: ignore[arg-type]


def test_merchant_transaction_event_validation_and_serialization() -> None:
    ev = MerchantTransactionEvent(
        merchant_id=" m_100 ",
        timestamp=1000.0,
        amount=49.99,
        mcc="5411",
        card_id=" card_1 ",
        is_fraud=True,
        is_chargeback=False,
    )
    assert ev.merchant_id == "m_100"
    assert ev.card_id == "card_1"
    d = ev.to_dict()
    assert MerchantTransactionEvent.from_dict(d) == ev

    with pytest.raises(MerchantRiskError, match="merchant_id"):
        MerchantTransactionEvent("", 100.0, 10.0)
    with pytest.raises(MerchantRiskError, match="timestamp"):
        MerchantTransactionEvent("m1", True, 10.0)  # type: ignore[arg-type]
    with pytest.raises(MerchantRiskError, match="amount"):
        MerchantTransactionEvent("m1", 100.0, -1.0)
    with pytest.raises(MerchantRiskError, match="card_id"):
        MerchantTransactionEvent("m1", 100.0, 10.0, card_id="   ")
    with pytest.raises(MerchantRiskError, match="is_fraud"):
        MerchantTransactionEvent("m1", 100.0, 10.0, is_fraud="yes")  # type: ignore[arg-type]
    with pytest.raises(MerchantRiskError, match="is_chargeback"):
        MerchantTransactionEvent("m1", 100.0, 10.0, is_chargeback=1)  # type: ignore[arg-type]
    with pytest.raises(MerchantRiskError, match="mapping"):
        MerchantTransactionEvent.from_dict(123)  # type: ignore[arg-type]
    with pytest.raises(MerchantRiskError, match="Invalid MerchantTransactionEvent"):
        MerchantTransactionEvent.from_dict({})


def test_merchant_tracker_config_validation() -> None:
    cfg = MerchantTrackerConfig()
    assert cfg.window_seconds == 86400.0
    assert cfg.max_merchants == 50000

    invalid_kwargs = [
        {"window_seconds": 0.0},
        {"window_seconds": True},
        {"max_merchants": 0},
        {"max_merchants": False},
        {"prior_strength": -1.0},
        {"prior_strength": True},
        {"chargeback_rate_threshold": 0.0},
        {"chargeback_rate_threshold": 1.5},
        {"min_chargeback_count": 0},
        {"bust_out_zscore_threshold": 0.0},
        {"min_history_for_zscore": 1},
        {"micro_tx_amount_cutoff": -0.1},
        {"micro_tx_ratio_threshold": 0.0},
        {"min_distinct_cards_for_testing": 0},
    ]
    for kwargs in invalid_kwargs:
        with pytest.raises(MerchantRiskError):
            MerchantTrackerConfig(**kwargs)  # type: ignore[arg-type]


def test_merchant_tracker_lifecycle_welford_and_anomalies() -> None:
    cfg = MerchantTrackerConfig(
        window_seconds=3600.0,
        max_merchants=2,
        prior_strength=5.0,
        chargeback_rate_threshold=0.10,
        min_chargeback_count=2,
        bust_out_zscore_threshold=2.5,
        min_history_for_zscore=3,
        micro_tx_amount_cutoff=5.0,
        micro_tx_ratio_threshold=0.60,
        min_distinct_cards_for_testing=3,
    )
    tracker = MerchantTracker(cfg)
    assert tracker.get_profile("m_missing") is None
    assert tracker.get_tracked_merchant_count() == 0

    # Record 3 baseline identical-amount transactions
    for i in range(3):
        prof, rep = tracker.record_event(
            MerchantTransactionEvent("m1", 1000.0 + i * 10, 20.0, mcc="5411", card_id=f"c{i}")
        )
        assert isinstance(prof, MerchantProfile)
        assert isinstance(rep, MerchantAnomalyReport)
        assert not rep.has_anomaly
        assert rep.severity == "LOW"

    # Test MEDIUM ticket outlier with zero-std fallback
    tracker2 = MerchantTracker(cfg)
    for i in range(3):
        tracker2.record_event(
            MerchantTransactionEvent("m_med", 100.0 + i, 20.0, mcc="5411", card_id="c1")
        )
    _, rep_med = tracker2.record_event(
        MerchantTransactionEvent("m_med", 105.0, 50.0, mcc="5411", card_id="c1")
    )
    assert rep_med.is_ticket_outlier
    assert not rep_med.is_bust_out_surge
    assert rep_med.severity == "MEDIUM"
    assert any("ticket_size_outlier" in r for r in rep_med.reasons)

    # Now test variably sized transactions on m1 + chargebacks + bust-out surge -> CRITICAL
    tracker.record_event(
        MerchantTransactionEvent(
            "m1", 1050.0, 25.0, mcc="7995", card_id="c3", is_fraud=True, is_chargeback=True
        )
    )
    prof_crit, rep_crit = tracker.record_event(
        MerchantTransactionEvent(
            "m1", 1060.0, 500.0, mcc="7995", card_id="c4", is_fraud=True, is_chargeback=True
        )
    )
    assert rep_crit.is_high_chargeback
    assert rep_crit.is_bust_out_surge
    assert rep_crit.severity == "CRITICAL"
    assert prof_crit.to_dict()["merchant_id"] == "m1"
    assert rep_crit.to_dict()["has_anomaly"] is True

    # Test card testing micro-tx ratio (HIGH severity) and window eviction
    for idx in range(3):
        _, rep_ct = tracker.record_event(
            MerchantTransactionEvent(
                "m2", 5000.0 + idx * 5, 1.00, mcc="5817", card_id=f"card_{idx}"
            )
        )
    assert rep_ct.is_card_testing
    assert rep_ct.severity == "HIGH"

    # Advance timestamp past window_seconds (3600s) for m2
    prof_evicted, _ = tracker.record_event(
        MerchantTransactionEvent("m2", 9500.0, 15.0, mcc="5817", card_id="card_new")
    )
    assert prof_evicted.distinct_cards_count == 1

    # Test LRU eviction when adding m3 (max_merchants=2 -> m1 evicted)
    tracker.record_event(MerchantTransactionEvent("m3", 9600.0, 10.0, mcc="5411"))
    assert tracker.get_tracked_merchant_count() == 2
    assert tracker.get_profile("m1") is None
    assert tracker.get_profile("m2") is not None
    assert tracker.get_profile("m3") is not None

    stats = tracker.get_stats()
    assert stats["tracked_merchants"] == 2
    assert stats["total_evaluations"] > 0
    assert stats["total_anomalies"] > 0

    cleared = tracker.reset()
    assert cleared == 2
    assert tracker.get_tracked_merchant_count() == 0


def test_merchant_feature_extractor_single_and_batch() -> None:
    extractor = MerchantFeatureExtractor()
    with pytest.raises(MerchantRiskError, match="record must be a Mapping"):
        extractor.extract_features("not-a-dict")  # type: ignore[arg-type]

    # Test fallback defaults with malformed fields and mutate_tracker=False
    f_readonly = extractor.extract_features(
        {
            "mcc": "bad_mcc",
            "amount": "not_num",
            "timestamp": "not_ts",
            "is_fraud": "0",
            "is_chargeback": "false",
        },
        mutate_tracker=False,
    )
    assert set(f_readonly.keys()) == set(MERCHANT_FEATURE_NAMES)
    assert f_readonly["merchant_tx_count_24h"] == 1.0
    assert extractor.tracker.get_tracked_merchant_count() == 0

    records = [
        {
            "merchant": "m_casino",
            "merchant_category_code": "7995",
            "amt": 25.0,
            "unix_time": 1000.0,
            "cc_num": "card_a",
            "Class": 1,
            "is_chargeback": True,
        },
        {
            "merch_id": "m_casino",
            "category_code": "7995",
            "Amount": 35.0,
            "Time": 1010.0,
            "account_id": "card_b",
            "is_fraud": True,
            "is_chargeback": True,
        },
        {
            "merchant_id": "m_casino",
            "mcc": "7995",
            "amount": 500.0,
            "event_time": 1020.0,
            "user_id": "card_c",
            "is_fraud": True,
            "is_chargeback": True,
        },
    ]
    batch_features = extractor.extract_batch(records, mutate_tracker=True)
    assert len(batch_features) == 3
    assert batch_features[0]["merchant_tx_count_24h"] == 1.0
    assert batch_features[1]["merchant_tx_count_24h"] == 2.0
    assert batch_features[2]["merchant_tx_count_24h"] == 3.0
    assert batch_features[2]["merchant_distinct_cards_24h"] == 3.0
    assert batch_features[2]["merchant_mcc_risk_score"] == 0.9
    assert batch_features[2]["merchant_is_high_chargeback"] == 1.0
