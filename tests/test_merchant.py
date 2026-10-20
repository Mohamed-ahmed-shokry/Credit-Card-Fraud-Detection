"""Unit tests for real-time merchant & MCC risk profiling and chargeback surveillance."""

from __future__ import annotations

import pytest

from fraud_detection.merchant import (
    MCCCategory,
    MCCRiskProfile,
    MerchantRiskError,
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
