"""Unit tests for the Unified Cross-Engine Risk Orchestrator (Phase 53)."""

from __future__ import annotations

import pytest

from fraud_detection.risk_orchestrator import (
    CrossSignalPattern,
    DomainRiskSignal,
    EntityRiskProfile,
    OrchestratorAction,
    OrchestratorPolicy,
    RiskDomain,
    UnifiedRiskAssessment,
    UnifiedRiskLevel,
)


def test_domain_models_round_trip_and_validation() -> None:
    sig = DomainRiskSignal(
        entity_id=" ent-101 ",
        domain=RiskDomain.ATO,
        score=1.42,
        tenant_id="",
        source_assessment_id="ato_123",
        transaction_id="tx_99",
        reason_codes=["NEW_DEVICE", "IMPOSSIBLE_TRAVEL"],
        metadata={"ip": "203.0.113.10"},
    )
    assert sig.entity_id == "ent-101"
    assert sig.score == 1.0
    assert sig.tenant_id == "default"

    restored_sig = DomainRiskSignal.from_dict(sig.to_dict())
    assert restored_sig.entity_id == "ent-101"
    assert restored_sig.domain is RiskDomain.ATO
    assert restored_sig.reason_codes == ["NEW_DEVICE", "IMPOSSIBLE_TRAVEL"]

    with pytest.raises(ValueError, match="entity_id"):
        DomainRiskSignal(entity_id="   ", domain=RiskDomain.AML, score=0.5)

    policy = OrchestratorPolicy(elevated_threshold=0.6, pattern_uplift=0.1)
    restored_policy = OrchestratorPolicy.from_dict(policy.to_dict())
    assert restored_policy.elevated_threshold == 0.6
    assert restored_policy.weight_for(RiskDomain.ATO) > 1.0

    profile = EntityRiskProfile(
        entity_id="ent-101",
        tenant_id="acme",
        domain_scores={"ato": 0.9},
        signals=[sig],
        high_risk_streak=2,
    )
    restored_profile = EntityRiskProfile.from_dict(profile.to_dict())
    assert restored_profile.high_risk_streak == 2
    assert len(restored_profile.signals) == 1

    assessment = UnifiedRiskAssessment(
        assessment_id="ora_1",
        entity_id="ent-101",
        tenant_id="acme",
        composite_score=0.88,
        risk_level=UnifiedRiskLevel.CRITICAL,
        action=OrchestratorAction.BLOCK_AND_ESCALATE,
        domain_scores={"ato": 0.9, "mule": 0.8},
        patterns_detected=[CrossSignalPattern.ATO_TO_MULE_EXFILTRATION],
        contributing_signal_ids=[sig.signal_id],
        correlated_case_id="case_1",
        case_created=True,
        rationale=["ATO + Mule compound exfiltration detected"],
    )
    restored_assessment = UnifiedRiskAssessment.from_dict(assessment.to_dict())
    assert restored_assessment.risk_level is UnifiedRiskLevel.CRITICAL
    assert restored_assessment.action is OrchestratorAction.BLOCK_AND_ESCALATE
    assert restored_assessment.patterns_detected == [CrossSignalPattern.ATO_TO_MULE_EXFILTRATION]
