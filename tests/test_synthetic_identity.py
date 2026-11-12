"""Unit and integration tests for Synthetic Identity Fraud (SIF) engine."""

from __future__ import annotations

from fraud_detection.synthetic_identity import (
    SifAction,
    SifAnomalyType,
    SifRiskLevel,
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
