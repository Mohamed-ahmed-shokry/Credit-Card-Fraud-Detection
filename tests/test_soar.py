"""Unit tests for the Automated SOAR Response Playbooks Engine (Phase 54)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fraud_detection.cases import CaseManager, CaseStatus
from fraud_detection.events import EventBus, EventType, FraudDomainEvent
from fraud_detection.risk_orchestrator import (
    CrossSignalPattern,
    OrchestratorAction,
    UnifiedRiskAssessment,
    UnifiedRiskLevel,
)
from fraud_detection.soar import (
    ActiveContainmentRecord,
    ContainmentControlType,
    ContainmentState,
    PlaybookCatalog,
    PlaybookExecutionRecord,
    PlaybookRunStatus,
    PlaybookStepResult,
    PlaybookStepSpec,
    ResponsePlaybook,
    SoarEngine,
    SoarStore,
    build_default_playbooks,
)


def test_soar_domain_models_round_trip_and_validation() -> None:
    step = PlaybookStepSpec(
        control_type="lock_account",
        description="Lock hijacked account",
        ttl_seconds=3600,
        parameters={"notify_user": True},
    )
    assert step.control_type is ContainmentControlType.LOCK_ACCOUNT
    restored_step = PlaybookStepSpec.from_dict(step.to_dict())
    assert restored_step.control_type is ContainmentControlType.LOCK_ACCOUNT
    assert restored_step.ttl_seconds == 3600
    assert restored_step.parameters == {"notify_user": True}

    with pytest.raises(ValueError, match="ttl_seconds"):
        PlaybookStepSpec(control_type=ContainmentControlType.FREEZE_CARD, ttl_seconds=0)

    playbook = ResponsePlaybook(
        playbook_id=" pb_ato_mule ",
        name="ATO to Mule Containment",
        description="Locks account and holds payouts",
        trigger_patterns=[CrossSignalPattern.ATO_TO_MULE_EXFILTRATION],
        min_risk_level="high",
        min_composite_score=0.70,
        steps=[step],
        priority=10,
    )
    assert playbook.playbook_id == "pb_ato_mule"
    assert playbook.min_risk_level is UnifiedRiskLevel.HIGH
    restored_pb = ResponsePlaybook.from_dict(playbook.to_dict())
    assert restored_pb.playbook_id == "pb_ato_mule"
    assert restored_pb.trigger_patterns == [CrossSignalPattern.ATO_TO_MULE_EXFILTRATION]
    assert len(restored_pb.steps) == 1

    with pytest.raises(ValueError, match="playbook_id"):
        ResponsePlaybook(playbook_id="   ", name="Invalid", description="Empty ID")

    now = datetime.now(UTC)
    future = (now + timedelta(hours=1)).isoformat()
    past = (now - timedelta(hours=1)).isoformat()

    containment = ActiveContainmentRecord(
        entity_id=" ent_500 ",
        control_type="freeze_card",
        tenant_id="",
        state="active",
        reason="Compound exfiltration",
        expires_at=future,
    )
    assert containment.entity_id == "ent_500"
    assert containment.tenant_id == "default"
    assert containment.control_type is ContainmentControlType.FREEZE_CARD
    assert containment.state is ContainmentState.ACTIVE
    assert containment.is_expired(now) is False

    containment.expires_at = past
    assert containment.is_expired(now) is True
    containment.expires_at = None
    assert containment.is_expired(now) is False

    # Naive timestamp handling in is_expired
    containment.expires_at = "2020-01-01T00:00:00"
    assert containment.is_expired(now) is True

    restored_cnt = ActiveContainmentRecord.from_dict(containment.to_dict())
    assert restored_cnt.entity_id == "ent_500"

    with pytest.raises(ValueError, match="entity_id"):
        ActiveContainmentRecord(entity_id="  ", control_type=ContainmentControlType.LOCK_ACCOUNT)

    step_res = PlaybookStepResult(
        control_type="revoke_sessions",
        applied=True,
        containment_id=containment.containment_id,
        message="Revoked active tokens",
    )
    restored_res = PlaybookStepResult.from_dict(step_res.to_dict())
    assert restored_res.control_type is ContainmentControlType.REVOKE_SESSIONS
    assert restored_res.applied is True

    exec_rec = PlaybookExecutionRecord(
        execution_id="soar_1",
        playbook_id="pb_ato_mule",
        entity_id="ent_500",
        tenant_id="acme",
        status="completed",
        composite_score=0.89,
        risk_level="critical",
        source_assessment_id="ora_1",
        triggered_patterns=[CrossSignalPattern.ATO_TO_MULE_EXFILTRATION],
        step_results=[step_res],
        correlated_case_id="case_1",
    )
    restored_exec = PlaybookExecutionRecord.from_dict(exec_rec.to_dict())
    assert restored_exec.status is PlaybookRunStatus.COMPLETED
    assert restored_exec.risk_level is UnifiedRiskLevel.CRITICAL
    assert len(restored_exec.step_results) == 1


def test_playbook_catalog_matching_and_registration() -> None:
    defaults = build_default_playbooks()
    assert len(defaults) == 5
    catalog = PlaybookCatalog()
    assert len(catalog.list_playbooks()) == 5
    assert len(catalog.list_playbooks(enabled_only=True)) == 5

    # Low risk assessment -> no playbook matched
    low_assessment = UnifiedRiskAssessment(
        assessment_id="ora_low",
        entity_id="ent_low",
        tenant_id="default",
        composite_score=0.20,
        risk_level=UnifiedRiskLevel.LOW,
        action=OrchestratorAction.ALLOW,
    )
    assert catalog.select_playbook(low_assessment) is None

    # ATO to Mule pattern -> matches pb_ato_mule_exfiltration
    ato_mule = UnifiedRiskAssessment(
        assessment_id="ora_am",
        entity_id="ent_am",
        tenant_id="default",
        composite_score=0.82,
        risk_level=UnifiedRiskLevel.HIGH,
        action=OrchestratorAction.RESTRICT_ENTITY,
        patterns_detected=[CrossSignalPattern.ATO_TO_MULE_EXFILTRATION],
    )
    matched = catalog.select_playbook(ato_mule)
    assert matched is not None
    assert matched.playbook_id == "pb_ato_mule_exfiltration"

    # High risk without pattern -> matches fallback pb_high_risk_step_up
    fallback_assess = UnifiedRiskAssessment(
        assessment_id="ora_fb",
        entity_id="ent_fb",
        tenant_id="default",
        composite_score=0.72,
        risk_level=UnifiedRiskLevel.HIGH,
        action=OrchestratorAction.RESTRICT_ENTITY,
    )
    fb = catalog.select_playbook(fallback_assess)
    assert fb is not None
    assert fb.playbook_id == "pb_high_risk_step_up"

    # Explicit playbook selection and disabled playbook handling
    explicit = catalog.select_playbook(low_assessment, playbook_id="pb_synthetic_bust_out")
    assert explicit is not None
    assert explicit.playbook_id == "pb_synthetic_bust_out"

    with pytest.raises(KeyError, match="not found"):
        catalog.select_playbook(low_assessment, playbook_id="pb_nonexistent")

    disabled_pb = ResponsePlaybook(
        playbook_id="pb_disabled",
        name="Disabled",
        description="Disabled playbook",
        enabled=False,
    )
    catalog.register(disabled_pb)
    assert catalog.get("pb_disabled") is disabled_pb
    assert catalog.select_playbook(fallback_assess, playbook_id="pb_disabled") is None


def test_soar_store_eviction_filtering_expiry_and_persistence(tmp_path: Path) -> None:
    store = SoarStore(max_executions=2, max_containments=2)
    now = datetime.now(UTC)

    for idx in range(3):
        exe = PlaybookExecutionRecord(
            execution_id=f"soar_{idx}",
            playbook_id="pb_high_risk_step_up",
            entity_id=f"ent_{idx}",
            tenant_id="default" if idx < 2 else "acme",
            status=PlaybookRunStatus.COMPLETED if idx < 2 else PlaybookRunStatus.SKIPPED,
            composite_score=0.75,
            risk_level=UnifiedRiskLevel.HIGH,
        )
        store.save_execution(exe)
    # Re-saving existing execution_id does not duplicate order
    store.save_execution(exe)
    assert store.get_execution("soar_0") is None
    assert len(store.list_executions(entity_id="ent_2")) == 1
    assert len(store.list_executions(tenant_id="acme")) == 1
    assert len(store.list_executions(status=PlaybookRunStatus.SKIPPED)) == 1

    for idx in range(3):
        cnt = ActiveContainmentRecord(
            containment_id=f"cnt_{idx}",
            entity_id=f"ent_{idx}",
            tenant_id="default" if idx < 2 else "acme",
            control_type=ContainmentControlType.LOCK_ACCOUNT,
            expires_at=(now + timedelta(seconds=10 if idx == 2 else -10)).isoformat(),
        )
        store.save_containment(cnt)
    store.save_containment(cnt)
    assert store.get_containment("cnt_0") is None

    # cnt_1 is expired, cnt_2 is active
    all_cnt = store.list_containments(reference_time=now)
    assert len(all_cnt) == 2
    cnt_1 = store.get_containment("cnt_1")
    assert cnt_1 is not None and cnt_1.state is ContainmentState.EXPIRED

    active_acme = store.list_containments(
        entity_id="ent_2", tenant_id="acme", active_only=True, reference_time=now
    )
    assert len(active_acme) == 1
    assert store.list_containments(entity_id="other", reference_time=now) == []
    assert store.list_containments(tenant_id="other", reference_time=now) == []

    # Future reference time expires cnt_2 during summary()
    future = now + timedelta(minutes=5)
    summary = store.summary(reference_time=future)
    assert summary["execution_count"] == 2
    assert summary["active_containments"] == 0

    json_path = store.save_json(tmp_path / "soar_store.json")
    loaded = SoarStore.load_json(json_path)
    assert loaded.summary()["execution_count"] == 2
    assert SoarStore.load_json(tmp_path / "missing.json").summary()["execution_count"] == 0


def test_soar_engine_execution_idempotency_rollback_and_events() -> None:
    bus = EventBus()
    case_mgr = CaseManager()
    events: list[EventType] = []

    def _on_event(ev: FraudDomainEvent) -> None:
        events.append(ev.event_type)

    bus.subscribe(_on_event)
    engine = SoarEngine(case_manager=case_mgr, event_bus=bus)

    # 1. Low risk -> SKIPPED, no events emitted
    low_assess = UnifiedRiskAssessment(
        assessment_id="ora_low",
        entity_id="ent_low",
        tenant_id="default",
        composite_score=0.15,
        risk_level=UnifiedRiskLevel.LOW,
        action=OrchestratorAction.ALLOW,
    )
    skipped = engine.execute_for_assessment(low_assess)
    assert skipped.status is PlaybookRunStatus.SKIPPED
    assert events == []

    # 2. High risk ATO -> Mule exfiltration -> executes pb_ato_mule_exfiltration & creates case
    high_assess = UnifiedRiskAssessment(
        assessment_id="ora_high_1",
        entity_id="ent_victim",
        tenant_id="t1",
        composite_score=0.88,
        risk_level=UnifiedRiskLevel.CRITICAL,
        action=OrchestratorAction.BLOCK_AND_ESCALATE,
        patterns_detected=[CrossSignalPattern.ATO_TO_MULE_EXFILTRATION],
    )
    run1 = engine.execute_for_assessment(high_assess)
    assert run1.status is PlaybookRunStatus.COMPLETED
    assert run1.playbook_id == "pb_ato_mule_exfiltration"
    assert run1.correlated_case_id is not None
    assert EventType.SOAR_PLAYBOOK_TRIGGERED in events
    assert EventType.SOAR_CONTAINMENT_APPLIED in events
    assert EventType.SOAR_PLAYBOOK_COMPLETED in events

    # Active containments count for ent_victim should be 3 (revoke, lock, hold)
    active = engine.store.list_containments(
        entity_id="ent_victim", tenant_id="t1", active_only=True
    )
    assert len(active) == 3
    summary = engine.store.summary()
    assert summary["active_containments"] == 3

    # 3. Re-running on same entity with existing correlated case -> containments already active,
    #    case escalation still updates existing case -> PARTIAL status
    high_assess_2 = UnifiedRiskAssessment(
        assessment_id="ora_high_2",
        entity_id="ent_victim",
        tenant_id="t1",
        composite_score=0.91,
        risk_level=UnifiedRiskLevel.CRITICAL,
        action=OrchestratorAction.BLOCK_AND_ESCALATE,
        patterns_detected=[CrossSignalPattern.ATO_TO_MULE_EXFILTRATION],
        correlated_case_id=run1.correlated_case_id,
    )
    run2 = engine.execute_for_assessment(high_assess_2)
    assert run2.status is PlaybookRunStatus.PARTIAL
    assert run2.correlated_case_id == run1.correlated_case_id

    # 4. Rollback run1 -> releases all 3 containments and emits SOAR_CONTAINMENT_ROLLED_BACK
    events.clear()
    rolled = engine.rollback_execution(run1.execution_id, reason="false_alarm_verified")
    assert rolled.status is PlaybookRunStatus.ROLLED_BACK
    assert len(rolled.metadata["released_containment_ids"]) == 3
    assert EventType.SOAR_CONTAINMENT_ROLLED_BACK in events
    assert (
        engine.store.list_containments(entity_id="ent_victim", tenant_id="t1", active_only=True)
        == []
    )

    # Re-releasing an already released containment is a no-op
    first_cnt_id = rolled.metadata["released_containment_ids"][0]
    released_again = engine.release_containment(first_cnt_id)
    assert released_again.state is ContainmentState.RELEASED

    with pytest.raises(KeyError, match="not found"):
        engine.release_containment("cnt_missing")
    with pytest.raises(KeyError, match="not found"):
        engine.rollback_execution("soar_missing")


def test_soar_engine_edge_branches_without_case_manager_or_event_bus() -> None:
    # Engine without CaseManager or EventBus
    engine = SoarEngine()
    assess = UnifiedRiskAssessment(
        assessment_id="ora_no_cm",
        entity_id="ent_nocm",
        tenant_id="default",
        composite_score=0.85,
        risk_level=UnifiedRiskLevel.CRITICAL,
        action=OrchestratorAction.BLOCK_AND_ESCALATE,
        patterns_detected=[CrossSignalPattern.ATO_TO_MULE_EXFILTRATION],
    )
    # First run: 3 containments applied, ESCALATE_CASE skipped (no CaseManager) -> PARTIAL
    run1 = engine.execute_for_assessment(assess)
    assert run1.status is PlaybookRunStatus.PARTIAL
    # Second run: all 3 containments already active + ESCALATE_CASE False -> SKIPPED
    run2 = engine.execute_for_assessment(assess)
    assert run2.status is PlaybookRunStatus.SKIPPED

    # Engine with CaseManager where correlated_case_id is missing or closed
    case_mgr = CaseManager()
    engine_cm = SoarEngine(case_manager=case_mgr)
    assess_missing_case = UnifiedRiskAssessment(
        assessment_id="ora_mc",
        entity_id="ent_mc",
        tenant_id="default",
        composite_score=0.85,
        risk_level=UnifiedRiskLevel.CRITICAL,
        action=OrchestratorAction.BLOCK_AND_ESCALATE,
        patterns_detected=[CrossSignalPattern.SYNTHETIC_TO_AML_LAYERING],
        correlated_case_id="nonexistent_case",
    )
    run_mc = engine_cm.execute_for_assessment(assess_missing_case)
    assert run_mc.correlated_case_id is not None
    assert run_mc.correlated_case_id != "nonexistent_case"

    # Close that case, then run again -> creates a new case
    closed_case = case_mgr.get_case(run_mc.correlated_case_id)
    closed_case.status = CaseStatus.CLOSED
    case_mgr.store.put(closed_case)
    assess_closed = UnifiedRiskAssessment(
        assessment_id="ora_closed",
        entity_id="ent_mc",
        tenant_id="default",
        composite_score=0.89,
        risk_level=UnifiedRiskLevel.CRITICAL,
        action=OrchestratorAction.BLOCK_AND_ESCALATE,
        patterns_detected=[CrossSignalPattern.SYNTHETIC_TO_AML_LAYERING],
        correlated_case_id=closed_case.case_id,
    )
    run_closed = engine_cm.execute_for_assessment(assess_closed)
    assert run_closed.correlated_case_id != closed_case.case_id
