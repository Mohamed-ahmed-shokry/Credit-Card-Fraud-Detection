"""Unit tests for the Event-Driven Cross-Engine Automation Bridge (Phase 56)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fraud_detection.automation_bridge import (
    AutomationBridgeStore,
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


def test_bridge_store_eviction_filtering_cooldowns_and_persistence(tmp_path: Path) -> None:
    store = AutomationBridgeStore(max_dispatches=2)
    for idx in range(3):
        rec = BridgeDispatchRecord(
            dispatch_id=f"ab_{idx}",
            entity_id=f"ent_{idx}",
            tenant_id="default" if idx < 2 else "acme",
            action=(
                BridgeRuleAction.FULL_CONTAINMENT_CASCADE
                if idx < 2
                else BridgeRuleAction.INGEST_ORCHESTRATOR_SIGNAL
            ),
            status=(
                BridgeDispatchStatus.EXECUTED if idx < 2 else BridgeDispatchStatus.SKIPPED_COOLDOWN
            ),
            soar_execution_id=f"soar_{idx}" if idx < 2 else None,
            suspended_token_ids=["tok_a"] if idx == 1 else [],
        )
        store.save_dispatch(rec)
    store.save_dispatch(rec)
    assert store.get_dispatch("ab_0") is None
    assert len(store.list_dispatches(entity_id="ent_2")) == 1
    assert len(store.list_dispatches(tenant_id="acme")) == 1
    assert len(store.list_dispatches(action=BridgeRuleAction.INGEST_ORCHESTRATOR_SIGNAL)) == 1
    assert len(store.list_dispatches(status=BridgeDispatchStatus.SKIPPED_COOLDOWN)) == 1

    now = datetime.now(UTC)
    assert (
        store.is_in_cooldown(
            tenant_id="default",
            entity_id="ent_1",
            action=BridgeRuleAction.FULL_CONTAINMENT_CASCADE,
            cooldown_seconds=0,
            reference_time=now,
        )
        is False
    )
    assert (
        store.is_in_cooldown(
            tenant_id="default",
            entity_id="ent_1",
            action=BridgeRuleAction.FULL_CONTAINMENT_CASCADE,
            cooldown_seconds=60,
            reference_time=now,
        )
        is False
    )
    store.record_cooldown(
        tenant_id="default",
        entity_id="ent_1",
        action=BridgeRuleAction.FULL_CONTAINMENT_CASCADE,
        reference_time=now,
    )
    assert (
        store.is_in_cooldown(
            tenant_id="default",
            entity_id="ent_1",
            action=BridgeRuleAction.FULL_CONTAINMENT_CASCADE,
            cooldown_seconds=60,
            reference_time=now + timedelta(seconds=30),
        )
        is True
    )
    assert (
        store.is_in_cooldown(
            tenant_id="default",
            entity_id="ent_1",
            action=BridgeRuleAction.FULL_CONTAINMENT_CASCADE,
            cooldown_seconds=60,
            reference_time=now + timedelta(seconds=61),
        )
        is False
    )
    # Naive timestamp in cooldown map
    store.record_cooldown(
        tenant_id="default",
        entity_id="ent_naive",
        action=BridgeRuleAction.FULL_CONTAINMENT_CASCADE,
        reference_time=datetime(2026, 11, 22, 12, 0, 0),  # noqa: DTZ001
    )
    assert (
        store.is_in_cooldown(
            tenant_id="default",
            entity_id="ent_naive",
            action=BridgeRuleAction.FULL_CONTAINMENT_CASCADE,
            cooldown_seconds=60,
            reference_time=datetime(2026, 11, 22, 12, 0, 20, tzinfo=UTC),
        )
        is True
    )

    summary = store.summary()
    assert summary["dispatch_count"] == 2
    assert summary["soar_triggered_count"] == 1
    assert summary["tokens_suspended_count"] == 1

    json_path = store.save_json(tmp_path / "bridge_store.json")
    loaded = AutomationBridgeStore.load_json(json_path)
    assert loaded.summary()["dispatch_count"] == 2
    assert (
        AutomationBridgeStore.load_json(tmp_path / "missing.json").summary()["dispatch_count"] == 0
    )
