"""Unit tests for the Event-Driven Cross-Engine Automation Bridge (Phase 56)."""

from __future__ import annotations

import pytest

from fraud_detection.automation_bridge import (
    BridgeDispatchRecord,
    BridgeDispatchStatus,
    BridgePolicy,
    BridgeRuleAction,
)
from fraud_detection.risk_orchestrator import UnifiedRiskLevel


def test_bridge_domain_models_round_trip_and_validation() -> None:
    policy = BridgePolicy(
        enabled=True,
        auto_trigger_soar=True,
        auto_suspend_wallet_tokens=False,
        auto_forward_wallet_signals=True,
        min_soar_risk_level="critical",
        cooldown_seconds=-10,
    )
    assert policy.min_soar_risk_level is UnifiedRiskLevel.CRITICAL
    assert policy.cooldown_seconds == 0
    restored_policy = BridgePolicy.from_dict(policy.to_dict())
    assert restored_policy.min_soar_risk_level is UnifiedRiskLevel.CRITICAL
    assert restored_policy.auto_suspend_wallet_tokens is False

    rec = BridgeDispatchRecord(
        entity_id=" ent_1 ",
        action="full_containment_cascade",
        status="executed",
        tenant_id="",
        source_event_id="ev_1",
        source_event_type="fraud.orchestrator.entity_escalation_triggered",
        soar_execution_id="soar_1",
        suspended_token_ids=["tok_1", "tok_2"],
        detail="Cascaded to SOAR and suspended 2 tokens",
    )
    assert rec.entity_id == "ent_1"
    assert rec.tenant_id == "default"
    assert rec.action is BridgeRuleAction.FULL_CONTAINMENT_CASCADE
    assert rec.status is BridgeDispatchStatus.EXECUTED
    restored_rec = BridgeDispatchRecord.from_dict(rec.to_dict())
    assert restored_rec.suspended_token_ids == ["tok_1", "tok_2"]
    assert restored_rec.action is BridgeRuleAction.FULL_CONTAINMENT_CASCADE

    with pytest.raises(ValueError, match="entity_id"):
        BridgeDispatchRecord(
            entity_id="  ",
            action=BridgeRuleAction.TRIGGER_SOAR_PLAYBOOK,
            status=BridgeDispatchStatus.EXECUTED,
        )
