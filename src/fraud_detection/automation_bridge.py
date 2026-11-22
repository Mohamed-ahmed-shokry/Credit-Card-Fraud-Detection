"""Event-Driven Cross-Engine Automation Bridge (Orchestrator <-> SOAR <-> Wallet).

Subscribes to ``EventBus`` domain events to automatically cascade high-severity
cross-engine risk escalations into reversible SOAR containment playbooks and
active digital wallet token suspensions, while feeding wallet provisioning
declines back into ``UnifiedRiskOrchestrator`` with per-entity cooldown guards
and zero-recursion guarantees.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_iso(ts: str) -> datetime:
    parsed = datetime.fromisoformat(ts)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


class BridgeRuleAction(StrEnum):
    """Automated cross-engine actions executed by the automation bridge."""

    TRIGGER_SOAR_PLAYBOOK = "trigger_soar_playbook"
    SUSPEND_ENTITY_WALLET_TOKENS = "suspend_entity_wallet_tokens"
    INGEST_ORCHESTRATOR_SIGNAL = "ingest_orchestrator_signal"
    FULL_CONTAINMENT_CASCADE = "full_containment_cascade"


class BridgeDispatchStatus(StrEnum):
    """Execution outcome of an automated cross-engine bridge dispatch."""

    EXECUTED = "executed"
    SKIPPED_COOLDOWN = "skipped_cooldown"
    SKIPPED_DISABLED = "skipped_disabled"
    FAILED = "failed"
