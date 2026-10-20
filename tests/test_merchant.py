"""Unit tests for real-time merchant & MCC risk profiling and chargeback surveillance."""

from __future__ import annotations

import pytest

from fraud_detection.merchant import (
    MCCCategory,
    MCCRiskProfile,
    MerchantRiskError,
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
