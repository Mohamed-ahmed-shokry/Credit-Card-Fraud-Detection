"""Event-Driven Cross-Engine Automation Bridge (Orchestrator <-> SOAR <-> Wallet).

Subscribes to ``EventBus`` domain events to automatically cascade high-severity
cross-engine risk escalations into reversible SOAR containment playbooks and
active digital wallet token suspensions, while feeding wallet provisioning
declines back into ``UnifiedRiskOrchestrator`` with per-entity cooldown guards
and zero-recursion guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

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
