"""Unit tests for the Unified Cross-Engine Risk Orchestrator (Phase 53)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fraud_detection.cases import CaseManager, CaseStatus
from fraud_detection.events import EventBus, EventType, FraudDomainEvent
from fraud_detection.risk_orchestrator import (
    CrossSignalAnalyzer,
    CrossSignalPattern,
    DomainRiskSignal,
    EntityCaseCorrelator,
    EntityRiskProfile,
    OrchestratorAction,
    OrchestratorPolicy,
    RiskDomain,
    UnifiedRiskAssessment,
    UnifiedRiskLevel,
    UnifiedRiskOrchestrator,
    UnifiedRiskStore,
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


def test_cross_signal_analyzer_detects_all_compound_patterns() -> None:
    analyzer = CrossSignalAnalyzer()
    policy = OrchestratorPolicy(elevated_threshold=0.55)

    # Low scores -> no patterns
    low_patterns, low_rationale = analyzer.analyze(
        {"ato": 0.2, "aml": 0.3}, policy=policy, high_risk_streak=0
    )
    assert low_patterns == []
    assert low_rationale == []

    # ATO + Mule + Synthetic + AML + First-Party + streak >= 2 -> all 5 patterns
    patterns, rationale = analyzer.analyze(
        {
            "ato": 0.78,
            "mule": 0.82,
            "synthetic_identity": 0.75,
            "aml": 0.68,
            "first_party": 0.61,
        },
        policy=policy,
        high_risk_streak=2,
    )
    assert set(patterns) == {
        CrossSignalPattern.ATO_TO_MULE_EXFILTRATION,
        CrossSignalPattern.SYNTHETIC_TO_AML_LAYERING,
        CrossSignalPattern.SYNTHETIC_BUST_OUT,
        CrossSignalPattern.MULTI_VECTOR_SYNDICATE,
        CrossSignalPattern.ESCALATED_REPEAT_OFFENDER,
    }
    assert len(rationale) == 5


def test_store_eviction_filtering_and_persistence(tmp_path: Path) -> None:
    store = UnifiedRiskStore(max_entities=2, max_assessments=2, max_signals_per_entity=2)
    for idx in range(3):
        store.record_signal(
            DomainRiskSignal(entity_id=f"e{idx}", domain=RiskDomain.ATO, score=0.2 + idx * 0.1)
        )
        store.record_signal(
            DomainRiskSignal(entity_id=f"e{idx}", domain=RiskDomain.AML, score=0.3 + idx * 0.1)
        )
        store.record_signal(
            DomainRiskSignal(entity_id=f"e{idx}", domain=RiskDomain.MULE, score=0.4 + idx * 0.1)
        )
    assert store.get_profile("e0") is None
    e2 = store.get_profile("e2")
    assert e2 is not None
    assert len(e2.signals) == 2

    for idx in range(3):
        store.save_assessment(
            UnifiedRiskAssessment(
                assessment_id=f"ora_{idx}",
                entity_id=f"e{idx}",
                tenant_id="default",
                composite_score=0.5,
                risk_level=UnifiedRiskLevel.MEDIUM,
                action=OrchestratorAction.STEP_UP_CHALLENGE,
            )
        )
    assert store.get_assessment("ora_0") is None
    assert len(store.list_assessments(entity_id="e2", tenant_id="default")) == 1
    assert store.list_assessments(risk_level=UnifiedRiskLevel.HIGH) == []

    json_path = store.save_json(tmp_path / "orchestrator.json")
    loaded = UnifiedRiskStore.load_json(json_path)
    assert loaded.summary()["assessment_count"] == 2
    assert UnifiedRiskStore.load_json(tmp_path / "missing.json").summary()["entity_count"] == 0


def test_orchestrator_scoring_window_and_case_correlation() -> None:
    case_mgr = CaseManager()
    orchestrator = UnifiedRiskOrchestrator(case_manager=case_mgr)

    # Empty entity -> LOW / ALLOW
    empty = orchestrator.evaluate_entity("ent-empty")
    assert empty.risk_level is UnifiedRiskLevel.LOW
    assert empty.action is OrchestratorAction.ALLOW

    # Medium single-domain score -> STEP_UP_CHALLENGE
    med = orchestrator.evaluate_entity(
        "ent-med",
        signals=[DomainRiskSignal(entity_id="ent-med", domain=RiskDomain.TRANSACTION, score=0.42)],
    )
    assert med.risk_level is UnifiedRiskLevel.MEDIUM
    assert med.action is OrchestratorAction.STEP_UP_CHALLENGE
    assert med.correlated_case_id is None

    # Elevated single-domain score -> HOLD_AND_REVIEW
    elev = orchestrator.evaluate_entity(
        "ent-elev",
        signals=[DomainRiskSignal(entity_id="ent-elev", domain=RiskDomain.TRANSACTION, score=0.58)],
    )
    assert elev.risk_level is UnifiedRiskLevel.MEDIUM
    assert elev.action is OrchestratorAction.HOLD_AND_REVIEW

    # High risk -> creates case; second evaluation reuses active case
    first = orchestrator.evaluate_entity(
        "ent-high",
        signals=[
            DomainRiskSignal(entity_id="ent-high", domain=RiskDomain.ATO, score=0.72),
        ],
    )
    assert first.risk_level is UnifiedRiskLevel.HIGH
    assert first.action is OrchestratorAction.RESTRICT_ENTITY
    assert first.correlated_case_id is not None
    assert first.case_created is True
    first_case_id = first.correlated_case_id

    # Add Mule signal -> ATO_TO_MULE_EXFILTRATION -> CRITICAL, reuses same case!
    second = orchestrator.evaluate_entity(
        "ent-high",
        signals=[
            DomainRiskSignal(entity_id="ent-high", domain=RiskDomain.MULE, score=0.80),
        ],
    )
    assert second.risk_level is UnifiedRiskLevel.CRITICAL
    assert second.action is OrchestratorAction.BLOCK_AND_ESCALATE
    assert CrossSignalPattern.ATO_TO_MULE_EXFILTRATION in second.patterns_detected
    assert second.correlated_case_id == first_case_id
    assert second.case_created is False

    # Close existing case -> next evaluation creates a new case
    case = case_mgr.get_case(first_case_id)
    case.status = CaseStatus.CLOSED
    case_mgr.store.put(case)

    third = orchestrator.evaluate_entity(
        "ent-high",
        signals=[DomainRiskSignal(entity_id="ent-high", domain=RiskDomain.ATO, score=0.88)],
    )
    assert third.case_created is True
    assert third.correlated_case_id != first_case_id


def test_correlator_handles_missing_case_id_and_stale_window() -> None:
    case_mgr = CaseManager()
    correlator = EntityCaseCorrelator()
    profile = EntityRiskProfile(entity_id="ent-stale", active_case_id="missing_case_id")
    assessment = UnifiedRiskAssessment(
        assessment_id="ora_x",
        entity_id="ent-stale",
        tenant_id="default",
        composite_score=0.9,
        risk_level=UnifiedRiskLevel.CRITICAL,
        action=OrchestratorAction.BLOCK_AND_ESCALATE,
    )
    case_id, created = correlator.correlate(
        case_manager=case_mgr, profile=profile, assessment=assessment
    )
    assert created is True
    assert profile.active_case_id == case_id

    # Stale signal outside correlation window falls back to profile.domain_scores or ignores stale
    now = datetime.now(UTC)
    old_ts = (now - timedelta(days=5)).isoformat()
    orchestrator = UnifiedRiskOrchestrator(
        policy=OrchestratorPolicy(correlation_window_seconds=3600)
    )
    orchestrator.ingest_signal(
        DomainRiskSignal(
            entity_id="ent-window",
            domain=RiskDomain.ATO,
            score=0.85,
            timestamp=old_ts,
        )
    )
    res = orchestrator.evaluate_entity("ent-window", reference_time=now)
    assert res.contributing_signal_ids == []
    assert res.domain_scores["ato"] == 0.85


def test_orchestrator_emits_lifecycle_events() -> None:
    bus = EventBus()
    case_mgr = CaseManager()
    received: list[EventType] = []

    def _capture(ev: FraudDomainEvent) -> None:
        received.append(ev.event_type)

    bus.subscribe(_capture)
    orchestrator = UnifiedRiskOrchestrator(case_manager=case_mgr, event_bus=bus)
    orchestrator.evaluate_entity(
        "ent-events",
        tenant_id="tenant-a",
        signals=[
            DomainRiskSignal(
                entity_id="ent-events", domain=RiskDomain.ATO, score=0.78, tenant_id="tenant-a"
            ),
            DomainRiskSignal(
                entity_id="ent-events", domain=RiskDomain.MULE, score=0.81, tenant_id="tenant-a"
            ),
        ],
    )
    assert received == [
        EventType.ENTITY_RISK_ORCHESTRATED,
        EventType.CROSS_SIGNAL_PATTERN_DETECTED,
        EventType.UNIFIED_CASE_CORRELATED,
        EventType.ENTITY_ESCALATION_TRIGGERED,
    ]


def test_orchestrator_edge_branches_and_string_domain() -> None:
    bus = EventBus()
    received: list[EventType] = []
    bus.subscribe(lambda ev: received.append(ev.event_type))

    orch = UnifiedRiskOrchestrator(event_bus=bus)
    # String domain coercion + naive ISO timestamp
    sig = DomainRiskSignal(
        entity_id="ent-edge",
        domain="ato",
        score=0.72,
        tenant_id="t1",
        timestamp="2026-11-16T01:00:00",
    )
    orch.ingest_signal(sig)
    high = orch.evaluate_entity(
        "ent-edge",
        tenant_id="t1",
        reference_time=datetime(2026, 11, 16, 1, 5, tzinfo=UTC),
    )
    assert high.risk_level is UnifiedRiskLevel.HIGH
    assert high.correlated_case_id is None
    assert EventType.ENTITY_RISK_ORCHESTRATED in received
    assert EventType.ENTITY_ESCALATION_TRIGGERED in received

    # Low-risk evaluation resets high_risk_streak and emits only ENTITY_RISK_ORCHESTRATED
    received.clear()
    prof = orch.store.get_profile("ent-edge", tenant_id="t1")
    assert prof is not None
    prof.signals.clear()
    prof.domain_scores.clear()
    low = orch.evaluate_entity("ent-edge", tenant_id="t1")
    assert low.risk_level is UnifiedRiskLevel.LOW
    assert prof.high_risk_streak == 0
    assert received == [EventType.ENTITY_RISK_ORCHESTRATED]

    # Filtering by tenant_id and risk_level
    assert len(orch.store.list_assessments(tenant_id="t1", risk_level=UnifiedRiskLevel.HIGH)) == 1
