"""Event-Driven Cross-Engine Automation Bridge (Orchestrator <-> SOAR <-> Wallet).

Subscribes to ``EventBus`` domain events to automatically cascade high-severity
cross-engine risk escalations into reversible SOAR containment playbooks and
active digital wallet token suspensions, while feeding wallet provisioning
declines back into ``UnifiedRiskOrchestrator`` with per-entity cooldown guards
and zero-recursion guarantees.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
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


class AutomationBridgeStore:
    """Thread-safe bounded audit ledger and cooldown tracker for bridge dispatches."""

    def __init__(self, *, max_dispatches: int = 10000) -> None:
        self._lock = threading.RLock()
        self.max_dispatches = max(1, max_dispatches)
        self._dispatches: dict[str, BridgeDispatchRecord] = {}
        self._dispatch_order: list[str] = []
        self._cooldowns: dict[tuple[str, str, str], str] = {}

    def _evict_if_needed(self) -> None:
        while len(self._dispatch_order) > self.max_dispatches:
            oldest_id = self._dispatch_order.pop(0)
            self._dispatches.pop(oldest_id, None)

    def save_dispatch(self, record: BridgeDispatchRecord) -> None:
        with self._lock:
            if record.dispatch_id not in self._dispatches:
                self._dispatch_order.append(record.dispatch_id)
            self._dispatches[record.dispatch_id] = record
            self._evict_if_needed()

    def get_dispatch(self, dispatch_id: str) -> BridgeDispatchRecord | None:
        with self._lock:
            return self._dispatches.get(dispatch_id)

    def list_dispatches(
        self,
        *,
        entity_id: str | None = None,
        tenant_id: str | None = None,
        action: BridgeRuleAction | None = None,
        status: BridgeDispatchStatus | None = None,
        limit: int = 100,
    ) -> list[BridgeDispatchRecord]:
        with self._lock:
            items = [self._dispatches[did] for did in reversed(self._dispatch_order)]
            if entity_id is not None:
                items = [d for d in items if d.entity_id == entity_id]
            if tenant_id is not None:
                items = [d for d in items if d.tenant_id == tenant_id]
            if action is not None:
                items = [d for d in items if d.action is action]
            if status is not None:
                items = [d for d in items if d.status is status]
            return items[: max(1, limit)]

    def is_in_cooldown(
        self,
        *,
        tenant_id: str,
        entity_id: str,
        action: BridgeRuleAction,
        cooldown_seconds: int,
        reference_time: datetime | None = None,
    ) -> bool:
        if cooldown_seconds <= 0:
            return False
        now = reference_time or datetime.now(UTC)
        key = (tenant_id, entity_id, action.value)
        with self._lock:
            last_ts = self._cooldowns.get(key)
            if last_ts is None:
                return False
            elapsed = (now - _parse_iso(last_ts)).total_seconds()
            return elapsed < cooldown_seconds

    def record_cooldown(
        self,
        *,
        tenant_id: str,
        entity_id: str,
        action: BridgeRuleAction,
        reference_time: datetime | None = None,
    ) -> None:
        now = reference_time or datetime.now(UTC)
        key = (tenant_id, entity_id, action.value)
        with self._lock:
            self._cooldowns[key] = now.isoformat()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            by_status: dict[str, int] = {st.value: 0 for st in BridgeDispatchStatus}
            by_action: dict[str, int] = {act.value: 0 for act in BridgeRuleAction}
            tokens_suspended = 0
            soar_triggered = 0
            for rec in self._dispatches.values():
                s_key = BridgeDispatchStatus(rec.status).value
                a_key = BridgeRuleAction(rec.action).value
                by_status[s_key] = by_status.get(s_key, 0) + 1
                by_action[a_key] = by_action.get(a_key, 0) + 1
                tokens_suspended += len(rec.suspended_token_ids)
                if rec.soar_execution_id:
                    soar_triggered += 1
            return {
                "dispatch_count": len(self._dispatches),
                "soar_triggered_count": soar_triggered,
                "tokens_suspended_count": tokens_suspended,
                "by_status": by_status,
                "by_action": by_action,
            }

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_dispatches": self.max_dispatches,
                "dispatches": [self._dispatches[did].to_dict() for did in self._dispatch_order],
            }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AutomationBridgeStore:
        store = cls(max_dispatches=int(payload.get("max_dispatches", 10000)))
        for raw in payload.get("dispatches", []):
            rec = BridgeDispatchRecord.from_dict(raw)
            store._dispatches[rec.dispatch_id] = rec
            store._dispatch_order.append(rec.dispatch_id)
        return store

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> AutomationBridgeStore:
        source = Path(path)
        if not source.exists():
            return cls()
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))
