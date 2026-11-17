"""Unit tests for the Automated SOAR Response Playbooks Engine (Phase 54)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fraud_detection.risk_orchestrator import CrossSignalPattern, UnifiedRiskLevel
from fraud_detection.soar import (
    ActiveContainmentRecord,
    ContainmentControlType,
    ContainmentState,
    PlaybookExecutionRecord,
    PlaybookRunStatus,
    PlaybookStepResult,
    PlaybookStepSpec,
    ResponsePlaybook,
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
