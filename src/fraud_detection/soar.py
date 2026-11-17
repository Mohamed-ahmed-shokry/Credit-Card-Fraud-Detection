"""Automated SOAR Incident Response Playbooks & Compensating Control Containment Engine.

Executes deterministic, auditable, reversible containment playbooks in response
to high-severity cross-engine entity risk assessments (``UnifiedRiskAssessment``)
and compound attack patterns, managing active compensating controls with TTL
expiration and full operator rollback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from fraud_detection.risk_orchestrator import CrossSignalPattern, UnifiedRiskLevel


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_iso(ts: str) -> datetime:
    parsed = datetime.fromisoformat(ts)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


class ContainmentControlType(StrEnum):
    """Compensating containment controls executable by a SOAR playbook."""

    LOCK_ACCOUNT = "lock_account"
    FREEZE_CARD = "freeze_card"
    HOLD_PAYOUTS = "hold_payouts"
    REVOKE_SESSIONS = "revoke_sessions"
    REQUIRE_STEP_UP = "require_step_up"
    QUARANTINE_DEVICE = "quarantine_device"
    ESCALATE_CASE = "escalate_case"


class PlaybookRunStatus(StrEnum):
    """Execution lifecycle status of a SOAR response playbook run."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    ROLLED_BACK = "rolled_back"
    SKIPPED = "skipped"


class ContainmentState(StrEnum):
    """State of an individual compensating control applied to an entity."""

    ACTIVE = "active"
    RELEASED = "released"
    EXPIRED = "expired"


@dataclass(slots=True)
class PlaybookStepSpec:
    """Declarative specification for a single compensating control step in a playbook."""

    control_type: ContainmentControlType | str
    description: str = ""
    ttl_seconds: int | None = 86400
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.control_type, ContainmentControlType):
            self.control_type = ContainmentControlType(str(self.control_type).lower())
        if self.ttl_seconds is not None and self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_type": ContainmentControlType(self.control_type).value,
            "description": self.description,
            "ttl_seconds": self.ttl_seconds,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlaybookStepSpec:
        ttl_raw = payload.get("ttl_seconds", 86400)
        return cls(
            control_type=ContainmentControlType(str(payload["control_type"]).lower()),
            description=str(payload.get("description", "")),
            ttl_seconds=int(ttl_raw) if ttl_raw is not None else None,
            parameters=dict(payload.get("parameters", {})),
        )


@dataclass(slots=True)
class ResponsePlaybook:
    """Declarative SOAR response playbook triggered by compound risk patterns or scores."""

    playbook_id: str
    name: str
    description: str
    trigger_patterns: list[CrossSignalPattern] = field(default_factory=list)
    min_risk_level: UnifiedRiskLevel | str = UnifiedRiskLevel.HIGH
    min_composite_score: float = 0.65
    steps: list[PlaybookStepSpec] = field(default_factory=list)
    priority: int = 100
    enabled: bool = True

    def __post_init__(self) -> None:
        self.playbook_id = self.playbook_id.strip()
        if not self.playbook_id:
            raise ValueError("playbook_id must not be empty")
        self.trigger_patterns = [CrossSignalPattern(str(p).lower()) for p in self.trigger_patterns]
        if not isinstance(self.min_risk_level, UnifiedRiskLevel):
            self.min_risk_level = UnifiedRiskLevel(str(self.min_risk_level).lower())

    def to_dict(self) -> dict[str, Any]:
        return {
            "playbook_id": self.playbook_id,
            "name": self.name,
            "description": self.description,
            "trigger_patterns": [p.value for p in self.trigger_patterns],
            "min_risk_level": UnifiedRiskLevel(self.min_risk_level).value,
            "min_composite_score": self.min_composite_score,
            "steps": [step.to_dict() for step in self.steps],
            "priority": self.priority,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResponsePlaybook:
        return cls(
            playbook_id=str(payload["playbook_id"]),
            name=str(payload["name"]),
            description=str(payload.get("description", "")),
            trigger_patterns=[
                CrossSignalPattern(str(p).lower()) for p in payload.get("trigger_patterns", [])
            ],
            min_risk_level=UnifiedRiskLevel(str(payload.get("min_risk_level", "high")).lower()),
            min_composite_score=float(payload.get("min_composite_score", 0.65)),
            steps=[PlaybookStepSpec.from_dict(s) for s in payload.get("steps", [])],
            priority=int(payload.get("priority", 100)),
            enabled=bool(payload.get("enabled", True)),
        )
