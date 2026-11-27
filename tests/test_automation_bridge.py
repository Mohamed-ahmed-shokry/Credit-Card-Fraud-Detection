"""Unit tests for the Event-Driven Cross-Engine Automation Bridge (Phase 56)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fraud_detection.app_scam import (
    AppPaymentContext,
    AppScamDefenseEngine,
    CoolingOffHoldStatus,
)
from fraud_detection.automation_bridge import (
    AutomationBridgeStore,
    BridgeDispatchRecord,
    BridgeDispatchStatus,
    BridgePolicy,
    BridgeRuleAction,
    CrossEngineAutomationBridge,
)
from fraud_detection.cases import CaseManager
from fraud_detection.events import EventBus, EventType, FraudDomainEvent
from fraud_detection.mule_detection import MuleDetectionEngine
from fraud_detection.risk_orchestrator import (
    CrossSignalPattern,
    DomainRiskSignal,
    RiskDomain,
    UnifiedRiskLevel,
    UnifiedRiskOrchestrator,
)
from fraud_detection.soar import SoarEngine
from fraud_detection.wallet_fraud import (
    TokenStatus,
    WalletFraudEngine,
    WalletProvisioningRequest,
)


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


def test_cross_engine_bridge_live_event_bus_cascade_and_wallet_feedback() -> None:
    bus = EventBus()
    case_mgr = CaseManager()
    orch = UnifiedRiskOrchestrator(case_manager=case_mgr, event_bus=bus)
    soar = SoarEngine(case_manager=case_mgr, event_bus=bus)
    wallet = WalletFraudEngine(case_manager=case_mgr, event_bus=bus)

    bridge_events: list[EventType] = []

    def _capture(ev: FraudDomainEvent) -> None:
        if ev.event_type in {
            EventType.AUTOMATION_BRIDGE_EXECUTED,
            EventType.AUTOMATION_BRIDGE_SKIPPED,
        }:
            bridge_events.append(ev.event_type)

    bus.subscribe(_capture)

    bridge = CrossEngineAutomationBridge(
        event_bus=bus,
        orchestrator=orch,
        soar_engine=soar,
        wallet_engine=wallet,
        policy=BridgePolicy(cooldown_seconds=300),
    )
    assert bridge.is_attached is False
    assert bridge.attach() is True
    assert bridge.is_attached is True
    # Idempotent attach
    assert bridge.attach() is False

    # 1. Provision an active wallet token for card_99
    green = wallet.evaluate_provisioning(
        WalletProvisioningRequest(
            card_hash="card_99",
            device_id="dev_99",
            wallet_account_id="wal_99",
        )
    )
    assert green.token_id is not None

    # 2. High-risk ATO + Mule evaluation on entity card_99 -> emits CROSS_SIGNAL_PATTERN_DETECTED
    #    & ENTITY_ESCALATION_TRIGGERED -> bridge triggers SOAR & suspends card_99's wallet token!
    orch.evaluate_entity(
        "card_99",
        signals=[
            DomainRiskSignal(entity_id="card_99", domain=RiskDomain.ATO, score=0.82),
            DomainRiskSignal(entity_id="card_99", domain=RiskDomain.MULE, score=0.85),
        ],
    )
    tok = wallet.store.get_token(green.token_id)
    assert tok is not None and tok.status is TokenStatus.SUSPENDED
    assert soar.store.summary()["execution_count"] == 1
    assert EventType.AUTOMATION_BRIDGE_EXECUTED in bridge_events
    # Second event (ENTITY_ESCALATION_TRIGGERED) was cooled down -> SKIPPED_COOLDOWN
    assert EventType.AUTOMATION_BRIDGE_SKIPPED in bridge_events

    # 3. Reverse cascade: RED_PATH wallet provisioning decline feeds signal into Orchestrator
    wallet.evaluate_provisioning(
        WalletProvisioningRequest(
            card_hash="card_red_88",
            device_id="dev_red_88",
            wallet_account_id="wal_red_88",
            sim_swap_detected=True,
            otp_attempts=4,
        )
    )
    prof = orch.store.get_profile("card_red_88")
    assert prof is not None and len(prof.signals) == 1

    # 4. Pause bridge -> skips both cascade and wallet signal forwarding
    bridge.pause()
    paused_rec = bridge.trigger_cascade(
        entity_id="card_paused",
        composite_score=0.90,
        risk_level=UnifiedRiskLevel.CRITICAL,
        patterns=[CrossSignalPattern.ATO_TO_MULE_EXFILTRATION],
    )
    assert paused_rec.status is BridgeDispatchStatus.SKIPPED_DISABLED
    wallet.evaluate_provisioning(
        WalletProvisioningRequest(
            card_hash="card_paused_wal",
            device_id="dev_pw",
            wallet_account_id="wal_pw",
            sim_swap_detected=True,
            otp_attempts=4,
        )
    )
    assert orch.store.get_profile("card_paused_wal") is None

    # 5. Resume & update_policy & detach
    bridge.resume()
    bridge.update_policy(
        BridgePolicy(
            enabled=True,
            auto_trigger_soar=False,
            auto_suspend_wallet_tokens=False,
            auto_forward_wallet_signals=False,
            cooldown_seconds=0,
        )
    )
    noop_rec = bridge.trigger_cascade(entity_id="ent_noop", composite_score=0.90)
    assert noop_rec.soar_execution_id is None
    assert noop_rec.suspended_token_ids == []

    assert bridge.detach() == 6
    assert bridge.is_attached is False
    assert bridge.detach() == 0


def test_cross_engine_bridge_edge_branches_and_bare_init() -> None:
    bare = CrossEngineAutomationBridge()
    assert bare.attach() is False
    assert bare.detach() == 0

    # Manual cascade with no soar_engine, no wallet_engine, no event_bus
    rec = bare.trigger_cascade(entity_id="ent_bare", composite_score=0.88)
    assert rec.status is BridgeDispatchStatus.EXECUTED
    assert rec.soar_execution_id is None
    assert rec.suspended_token_ids == []

    # Unrelated event type & missing payload fields -> ignored (returns None)
    assert bare.handle_event(FraudDomainEvent(event_type=EventType.TRANSACTION_SCORED)) is None
    assert (
        bare.handle_event(
            FraudDomainEvent(event_type=EventType.ENTITY_ESCALATION_TRIGGERED, payload={})
        )
        is None
    )
    assert (
        bare.handle_event(
            FraudDomainEvent(event_type=EventType.WALLET_PROVISIONING_DECLINED, payload={})
        )
        is None
    )

    # Orchestrator present, wallet signal with empty payload -> ignored
    orch = UnifiedRiskOrchestrator()
    bridge_o = CrossEngineAutomationBridge(orchestrator=orch)
    assert (
        bridge_o.handle_event(
            FraudDomainEvent(event_type=EventType.WALLET_PROVISIONING_DECLINED, payload={})
        )
        is None
    )

    # Detach when listener was already removed externally
    bus = EventBus()
    b_ext = CrossEngineAutomationBridge(event_bus=bus)
    b_ext.attach()
    for tok in list(b_ext._subscription_tokens):  # noqa: SLF001
        bus.unsubscribe(tok)
    assert b_ext.detach() == 0


def test_app_scam_to_mule_and_soar_live_event_bus_correlation() -> None:
    bus = EventBus()
    orch = UnifiedRiskOrchestrator(event_bus=bus)
    soar = SoarEngine(event_bus=bus)
    mule = MuleDetectionEngine()
    app_scam = AppScamDefenseEngine(event_bus=bus)
    bridge = CrossEngineAutomationBridge(
        event_bus=bus,
        orchestrator=orch,
        soar_engine=soar,
        mule_engine=mule,
        policy=BridgePolicy(cooldown_seconds=300),
    )
    assert bridge.attach() is True

    # 1. Critical APP scam block fires APP_SCAM_BLOCKED -> correlates Mule + Orchestrator + SOAR
    blocked = app_scam.evaluate_payment(
        AppPaymentContext(
            sender_account_id="victim_1",
            payee_account_id="mule_beneficiary_1",
            amount=8500.0,
            active_phone_call=True,
            screen_share_or_rat_detected=True,
            stated_purpose="safe_account",
            tenant_id="t_scam",
        )
    )
    assert blocked.action.value == "block_and_escalate"

    dispatches = bridge.store.list_dispatches(
        entity_id="mule_beneficiary_1",
        action=BridgeRuleAction.CORRELATE_APP_SCAM_MULE,
    )
    assert len(dispatches) == 1
    rec = dispatches[0]
    assert rec.status is BridgeDispatchStatus.EXECUTED
    assert rec.orchestrator_signal_id is not None
    assert rec.soar_execution_id is not None
    assert "app_scam_to_mule_beneficiary" in rec.metadata["patterns_detected"]
    assert mule.store.get_profile("mule_beneficiary_1") is not None
    assert bridge.store.summary()["app_scam_correlations_count"] == 1

    # 2. Second correlation within cooldown -> SKIPPED_COOLDOWN
    cooled = bridge.correlate_app_scam_incident(
        sender_account_id="victim_2",
        payee_account_id="mule_beneficiary_1",
        amount=4000.0,
        tenant_id="t_scam",
    )
    assert cooled.status is BridgeDispatchStatus.SKIPPED_COOLDOWN

    # 3. Cooling-off hold resolved as RELEASED -> ignored; CONFIRMED_SCAM_CANCELLED -> correlated
    held_ok = app_scam.evaluate_payment(
        AppPaymentContext(
            sender_account_id="victim_3",
            payee_account_id="payee_released",
            amount=600.0,
            is_new_payee=False,
            active_phone_call=True,
        )
    )
    assert held_ok.hold_id is not None
    app_scam.resolve_hold(held_ok.hold_id, outcome=CoolingOffHoldStatus.RELEASED)
    assert bridge.store.list_dispatches(entity_id="payee_released") == []

    held_scam = app_scam.evaluate_payment(
        AppPaymentContext(
            sender_account_id="victim_4",
            payee_account_id="mule_beneficiary_2",
            amount=700.0,
            is_new_payee=False,
            active_phone_call=True,
        )
    )
    assert held_scam.hold_id is not None
    app_scam.resolve_hold(held_scam.hold_id, outcome=CoolingOffHoldStatus.CONFIRMED_SCAM_CANCELLED)
    dispatches_2 = bridge.store.list_dispatches(
        entity_id="mule_beneficiary_2",
        action=BridgeRuleAction.CORRELATE_APP_SCAM_MULE,
    )
    assert len(dispatches_2) == 1
    assert dispatches_2[0].status is BridgeDispatchStatus.EXECUTED

    # 4. Paused bridge -> SKIPPED_DISABLED
    bridge.pause()
    paused_rec = bridge.correlate_app_scam_incident(
        sender_account_id="victim_5",
        payee_account_id="mule_paused",
        amount=1000.0,
    )
    assert paused_rec.status is BridgeDispatchStatus.SKIPPED_DISABLED
    bridge.resume()

    # 5. Missing payee_account_id in event payload -> ignored (None)
    assert (
        bridge.handle_event(FraudDomainEvent(event_type=EventType.APP_SCAM_BLOCKED, payload={}))
        is None
    )

    # 6. Mule-only bridge (no orchestrator, no soar) + policy roundtrip
    policy = BridgePolicy.from_dict(
        {
            "auto_forward_app_scam_signals": False,
            "auto_correlate_mule_beneficiaries": True,
            "cooldown_seconds": 0,
        }
    )
    assert policy.to_dict()["auto_forward_app_scam_signals"] is False
    mule_only_bridge = CrossEngineAutomationBridge(mule_engine=MuleDetectionEngine(), policy=policy)
    bare_rec = mule_only_bridge.correlate_app_scam_incident(
        sender_account_id="v_bare",
        payee_account_id="p_bare",
        amount=500.0,
        risk_score=0.75,
    )
    assert bare_rec.status is BridgeDispatchStatus.EXECUTED
    assert bare_rec.soar_execution_id is None
    assert bare_rec.orchestrator_signal_id is None
