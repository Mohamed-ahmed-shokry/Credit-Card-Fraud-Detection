"""Event-Driven Cross-Engine Automation Bridge (Orchestrator <-> SOAR <-> Wallet).

Subscribes to ``EventBus`` domain events to automatically cascade high-severity
cross-engine risk escalations into reversible SOAR containment playbooks and
active digital wallet token suspensions, while feeding wallet provisioning
declines back into ``UnifiedRiskOrchestrator`` with per-entity cooldown guards
and zero-recursion guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from fraud_detection.risk_orchestrator import UnifiedRiskLevel


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


@dataclass(slots=True)
class BridgePolicy:
    """Configurable toggles, risk floor, and cooldown guardrails for the bridge."""

    enabled: bool = True
    auto_trigger_soar: bool = True
    auto_suspend_wallet_tokens: bool = True
    auto_forward_wallet_signals: bool = True
    min_soar_risk_level: UnifiedRiskLevel | str = UnifiedRiskLevel.HIGH
    cooldown_seconds: int = 300

    def __post_init__(self) -> None:
        if not isinstance(self.min_soar_risk_level, UnifiedRiskLevel):
            self.min_soar_risk_level = UnifiedRiskLevel(str(self.min_soar_risk_level).lower())
        self.cooldown_seconds = max(0, int(self.cooldown_seconds))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "auto_trigger_soar": self.auto_trigger_soar,
            "auto_suspend_wallet_tokens": self.auto_suspend_wallet_tokens,
            "auto_forward_wallet_signals": self.auto_forward_wallet_signals,
            "min_soar_risk_level": UnifiedRiskLevel(self.min_soar_risk_level).value,
            "cooldown_seconds": self.cooldown_seconds,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BridgePolicy:
        return cls(
            enabled=bool(payload.get("enabled", True)),
            auto_trigger_soar=bool(payload.get("auto_trigger_soar", True)),
            auto_suspend_wallet_tokens=bool(payload.get("auto_suspend_wallet_tokens", True)),
            auto_forward_wallet_signals=bool(payload.get("auto_forward_wallet_signals", True)),
            min_soar_risk_level=UnifiedRiskLevel(
                str(payload.get("min_soar_risk_level", "high")).lower()
            ),
            cooldown_seconds=int(payload.get("cooldown_seconds", 300)),
        )


@dataclass(slots=True)
class BridgeDispatchRecord:
    """Audit record of an automated cross-engine cascade or signal-forwarding dispatch."""

    entity_id: str
    action: BridgeRuleAction | str
    status: BridgeDispatchStatus | str
    tenant_id: str = "default"
    dispatch_id: str = field(default_factory=lambda: f"ab_{uuid4().hex[:12]}")
    source_event_id: str | None = None
    source_event_type: str = "manual"
    soar_execution_id: str | None = None
    suspended_token_ids: list[str] = field(default_factory=list)
    orchestrator_signal_id: str | None = None
    detail: str = ""
    dispatched_at: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.entity_id = self.entity_id.strip()
        if not self.entity_id:
            raise ValueError("entity_id must not be empty")
        if not isinstance(self.action, BridgeRuleAction):
            self.action = BridgeRuleAction(str(self.action).lower())
        if not isinstance(self.status, BridgeDispatchStatus):
            self.status = BridgeDispatchStatus(str(self.status).lower())
        self.tenant_id = self.tenant_id.strip() or "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dispatch_id": self.dispatch_id,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "action": BridgeRuleAction(self.action).value,
            "status": BridgeDispatchStatus(self.status).value,
            "source_event_id": self.source_event_id,
            "source_event_type": self.source_event_type,
            "soar_execution_id": self.soar_execution_id,
            "suspended_token_ids": list(self.suspended_token_ids),
            "orchestrator_signal_id": self.orchestrator_signal_id,
            "detail": self.detail,
            "dispatched_at": self.dispatched_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BridgeDispatchRecord:
        return cls(
            dispatch_id=str(payload.get("dispatch_id") or f"ab_{uuid4().hex[:12]}"),
            entity_id=str(payload["entity_id"]),
            tenant_id=str(payload.get("tenant_id") or "default"),
            action=BridgeRuleAction(str(payload["action"]).lower()),
            status=BridgeDispatchStatus(str(payload["status"]).lower()),
            source_event_id=payload.get("source_event_id"),
            source_event_type=str(payload.get("source_event_type", "manual")),
            soar_execution_id=payload.get("soar_execution_id"),
            suspended_token_ids=[str(t) for t in payload.get("suspended_token_ids", [])],
            orchestrator_signal_id=payload.get("orchestrator_signal_id"),
            detail=str(payload.get("detail", "")),
            dispatched_at=str(payload.get("dispatched_at") or _utc_now_iso()),
            metadata=dict(payload.get("metadata", {})),
        )
