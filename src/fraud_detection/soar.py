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
from uuid import uuid4

from fraud_detection.risk_orchestrator import (
    CrossSignalPattern,
    UnifiedRiskAssessment,
    UnifiedRiskLevel,
)


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


@dataclass(slots=True)
class ActiveContainmentRecord:
    """Tracks an active or released compensating control applied to an entity."""

    entity_id: str
    control_type: ContainmentControlType | str
    tenant_id: str = "default"
    containment_id: str = field(default_factory=lambda: f"cnt_{uuid4().hex[:12]}")
    execution_id: str | None = None
    playbook_id: str | None = None
    state: ContainmentState | str = ContainmentState.ACTIVE
    reason: str = ""
    applied_at: str = field(default_factory=_utc_now_iso)
    expires_at: str | None = None
    released_at: str | None = None
    release_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.entity_id = self.entity_id.strip()
        if not self.entity_id:
            raise ValueError("entity_id must not be empty")
        if not isinstance(self.control_type, ContainmentControlType):
            self.control_type = ContainmentControlType(str(self.control_type).lower())
        if not isinstance(self.state, ContainmentState):
            self.state = ContainmentState(str(self.state).lower())
        self.tenant_id = self.tenant_id.strip() or "default"

    def is_expired(self, reference_time: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        now = reference_time or datetime.now(UTC)
        return _parse_iso(self.expires_at) <= now

    def to_dict(self) -> dict[str, Any]:
        return {
            "containment_id": self.containment_id,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "control_type": ContainmentControlType(self.control_type).value,
            "execution_id": self.execution_id,
            "playbook_id": self.playbook_id,
            "state": ContainmentState(self.state).value,
            "reason": self.reason,
            "applied_at": self.applied_at,
            "expires_at": self.expires_at,
            "released_at": self.released_at,
            "release_reason": self.release_reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ActiveContainmentRecord:
        return cls(
            containment_id=str(payload.get("containment_id") or f"cnt_{uuid4().hex[:12]}"),
            entity_id=str(payload["entity_id"]),
            tenant_id=str(payload.get("tenant_id") or "default"),
            control_type=ContainmentControlType(str(payload["control_type"]).lower()),
            execution_id=payload.get("execution_id"),
            playbook_id=payload.get("playbook_id"),
            state=ContainmentState(str(payload.get("state", "active")).lower()),
            reason=str(payload.get("reason", "")),
            applied_at=str(payload.get("applied_at") or _utc_now_iso()),
            expires_at=payload.get("expires_at"),
            released_at=payload.get("released_at"),
            release_reason=payload.get("release_reason"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class PlaybookStepResult:
    """Outcome of executing a single step within a SOAR playbook."""

    control_type: ContainmentControlType | str
    applied: bool
    containment_id: str | None = None
    case_id: str | None = None
    message: str = ""
    executed_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        if not isinstance(self.control_type, ContainmentControlType):
            self.control_type = ContainmentControlType(str(self.control_type).lower())

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_type": ContainmentControlType(self.control_type).value,
            "applied": self.applied,
            "containment_id": self.containment_id,
            "case_id": self.case_id,
            "message": self.message,
            "executed_at": self.executed_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlaybookStepResult:
        return cls(
            control_type=ContainmentControlType(str(payload["control_type"]).lower()),
            applied=bool(payload.get("applied", True)),
            containment_id=payload.get("containment_id"),
            case_id=payload.get("case_id"),
            message=str(payload.get("message", "")),
            executed_at=str(payload.get("executed_at") or _utc_now_iso()),
        )


@dataclass(slots=True)
class PlaybookExecutionRecord:
    """Audit record of a full SOAR playbook execution and optional rollback."""

    execution_id: str
    playbook_id: str
    entity_id: str
    tenant_id: str
    status: PlaybookRunStatus | str
    composite_score: float
    risk_level: UnifiedRiskLevel | str
    source_assessment_id: str | None = None
    triggered_patterns: list[CrossSignalPattern] = field(default_factory=list)
    step_results: list[PlaybookStepResult] = field(default_factory=list)
    correlated_case_id: str | None = None
    executed_at: str = field(default_factory=_utc_now_iso)
    rolled_back_at: str | None = None
    rollback_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, PlaybookRunStatus):
            self.status = PlaybookRunStatus(str(self.status).lower())
        if not isinstance(self.risk_level, UnifiedRiskLevel):
            self.risk_level = UnifiedRiskLevel(str(self.risk_level).lower())
        self.triggered_patterns = [
            CrossSignalPattern(str(p).lower()) for p in self.triggered_patterns
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "playbook_id": self.playbook_id,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "status": PlaybookRunStatus(self.status).value,
            "composite_score": round(self.composite_score, 6),
            "risk_level": UnifiedRiskLevel(self.risk_level).value,
            "source_assessment_id": self.source_assessment_id,
            "triggered_patterns": [p.value for p in self.triggered_patterns],
            "step_results": [r.to_dict() for r in self.step_results],
            "correlated_case_id": self.correlated_case_id,
            "executed_at": self.executed_at,
            "rolled_back_at": self.rolled_back_at,
            "rollback_reason": self.rollback_reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlaybookExecutionRecord:
        return cls(
            execution_id=str(payload["execution_id"]),
            playbook_id=str(payload["playbook_id"]),
            entity_id=str(payload["entity_id"]),
            tenant_id=str(payload.get("tenant_id") or "default"),
            status=PlaybookRunStatus(str(payload["status"]).lower()),
            composite_score=float(payload.get("composite_score", 0.0)),
            risk_level=UnifiedRiskLevel(str(payload.get("risk_level", "high")).lower()),
            source_assessment_id=payload.get("source_assessment_id"),
            triggered_patterns=[
                CrossSignalPattern(str(p).lower()) for p in payload.get("triggered_patterns", [])
            ],
            step_results=[PlaybookStepResult.from_dict(r) for r in payload.get("step_results", [])],
            correlated_case_id=payload.get("correlated_case_id"),
            executed_at=str(payload.get("executed_at") or _utc_now_iso()),
            rolled_back_at=payload.get("rolled_back_at"),
            rollback_reason=payload.get("rollback_reason"),
            metadata=dict(payload.get("metadata", {})),
        )


_RISK_LEVEL_ORDER: dict[UnifiedRiskLevel, int] = {
    UnifiedRiskLevel.LOW: 0,
    UnifiedRiskLevel.MEDIUM: 1,
    UnifiedRiskLevel.HIGH: 2,
    UnifiedRiskLevel.CRITICAL: 3,
}


def build_default_playbooks() -> list[ResponsePlaybook]:
    """Build standard SOAR containment playbooks for compound fraud patterns."""
    return [
        ResponsePlaybook(
            playbook_id="pb_ato_mule_exfiltration",
            name="ATO to Mule Exfiltration Containment",
            description=(
                "Locks compromised account, revokes sessions, holds payouts, and escalates case"
            ),
            trigger_patterns=[CrossSignalPattern.ATO_TO_MULE_EXFILTRATION],
            min_risk_level=UnifiedRiskLevel.HIGH,
            min_composite_score=0.65,
            priority=10,
            steps=[
                PlaybookStepSpec(
                    control_type=ContainmentControlType.REVOKE_SESSIONS,
                    description="Terminate active authentication sessions immediately",
                    ttl_seconds=86400,
                ),
                PlaybookStepSpec(
                    control_type=ContainmentControlType.LOCK_ACCOUNT,
                    description="Lock compromised account pending security verification",
                    ttl_seconds=86400,
                ),
                PlaybookStepSpec(
                    control_type=ContainmentControlType.HOLD_PAYOUTS,
                    description="Hold outbound ACH, wire, and off-ramp payouts",
                    ttl_seconds=172800,
                ),
                PlaybookStepSpec(
                    control_type=ContainmentControlType.ESCALATE_CASE,
                    description="Escalate correlated investigation case to priority review",
                    ttl_seconds=None,
                ),
            ],
        ),
        ResponsePlaybook(
            playbook_id="pb_synthetic_aml_layering",
            name="Synthetic Identity AML Layering Containment",
            description="Freezes cards, holds payouts, locks account, and escalates AML case",
            trigger_patterns=[CrossSignalPattern.SYNTHETIC_TO_AML_LAYERING],
            min_risk_level=UnifiedRiskLevel.HIGH,
            min_composite_score=0.65,
            priority=20,
            steps=[
                PlaybookStepSpec(
                    control_type=ContainmentControlType.FREEZE_CARD,
                    description="Freeze associated credit and debit instruments",
                    ttl_seconds=172800,
                ),
                PlaybookStepSpec(
                    control_type=ContainmentControlType.HOLD_PAYOUTS,
                    description="Hold outbound layering transfers",
                    ttl_seconds=172800,
                ),
                PlaybookStepSpec(
                    control_type=ContainmentControlType.LOCK_ACCOUNT,
                    description="Lock synthetic identity account profile",
                    ttl_seconds=172800,
                ),
                PlaybookStepSpec(
                    control_type=ContainmentControlType.ESCALATE_CASE,
                    description="Escalate case for regulatory AML/SAR review",
                    ttl_seconds=None,
                ),
            ],
        ),
        ResponsePlaybook(
            playbook_id="pb_synthetic_bust_out",
            name="Synthetic Bust-Out & Refund Abuse Containment",
            description="Freezes card credit line, holds refunds/payouts, and requires step-up",
            trigger_patterns=[CrossSignalPattern.SYNTHETIC_BUST_OUT],
            min_risk_level=UnifiedRiskLevel.HIGH,
            min_composite_score=0.65,
            priority=30,
            steps=[
                PlaybookStepSpec(
                    control_type=ContainmentControlType.FREEZE_CARD,
                    description="Freeze card credit line against bust-out drawdowns",
                    ttl_seconds=86400,
                ),
                PlaybookStepSpec(
                    control_type=ContainmentControlType.HOLD_PAYOUTS,
                    description="Hold instant refund and promotional disbursements",
                    ttl_seconds=86400,
                ),
                PlaybookStepSpec(
                    control_type=ContainmentControlType.REQUIRE_STEP_UP,
                    description="Require step-up identity proofing on account actions",
                    ttl_seconds=86400,
                ),
            ],
        ),
        ResponsePlaybook(
            playbook_id="pb_multi_vector_syndicate",
            name="Multi-Vector Syndicate Full Containment",
            description=(
                "Applies full device, session, card, and account quarantine across 3+ domains"
            ),
            trigger_patterns=[
                CrossSignalPattern.MULTI_VECTOR_SYNDICATE,
                CrossSignalPattern.ESCALATED_REPEAT_OFFENDER,
            ],
            min_risk_level=UnifiedRiskLevel.HIGH,
            min_composite_score=0.70,
            priority=5,
            steps=[
                PlaybookStepSpec(
                    control_type=ContainmentControlType.QUARANTINE_DEVICE,
                    description="Quarantine associated device and browser fingerprints",
                    ttl_seconds=259200,
                ),
                PlaybookStepSpec(
                    control_type=ContainmentControlType.REVOKE_SESSIONS,
                    description="Revoke all active session tokens",
                    ttl_seconds=259200,
                ),
                PlaybookStepSpec(
                    control_type=ContainmentControlType.FREEZE_CARD,
                    description="Freeze payment instruments",
                    ttl_seconds=259200,
                ),
                PlaybookStepSpec(
                    control_type=ContainmentControlType.LOCK_ACCOUNT,
                    description="Lock entity account",
                    ttl_seconds=259200,
                ),
                PlaybookStepSpec(
                    control_type=ContainmentControlType.ESCALATE_CASE,
                    description="Escalate multi-vector syndicate case",
                    ttl_seconds=None,
                ),
            ],
        ),
        ResponsePlaybook(
            playbook_id="pb_high_risk_step_up",
            name="High-Risk Entity Step-Up & Payout Hold",
            description=(
                "Fallback playbook for high/critical composite risk without specific pattern match"
            ),
            trigger_patterns=[],
            min_risk_level=UnifiedRiskLevel.HIGH,
            min_composite_score=0.65,
            priority=100,
            steps=[
                PlaybookStepSpec(
                    control_type=ContainmentControlType.REQUIRE_STEP_UP,
                    description="Enforce step-up MFA challenge on next transaction",
                    ttl_seconds=43200,
                ),
                PlaybookStepSpec(
                    control_type=ContainmentControlType.HOLD_PAYOUTS,
                    description="Place temporary hold on high-risk disbursements",
                    ttl_seconds=43200,
                ),
            ],
        ),
    ]


class PlaybookCatalog:
    """Registry and matcher for SOAR response playbooks."""

    def __init__(self, playbooks: list[ResponsePlaybook] | None = None) -> None:
        items = playbooks if playbooks is not None else build_default_playbooks()
        self._playbooks: dict[str, ResponsePlaybook] = {p.playbook_id: p for p in items}

    def register(self, playbook: ResponsePlaybook) -> None:
        self._playbooks[playbook.playbook_id] = playbook

    def get(self, playbook_id: str) -> ResponsePlaybook | None:
        return self._playbooks.get(playbook_id)

    def list_playbooks(self, *, enabled_only: bool = False) -> list[ResponsePlaybook]:
        items = list(self._playbooks.values())
        if enabled_only:
            items = [p for p in items if p.enabled]
        return sorted(items, key=lambda p: (p.priority, p.playbook_id))

    def select_playbook(
        self,
        assessment: UnifiedRiskAssessment,
        *,
        playbook_id: str | None = None,
    ) -> ResponsePlaybook | None:
        if playbook_id:
            explicit = self._playbooks.get(playbook_id)
            if explicit is None:
                raise KeyError(f"Playbook '{playbook_id}' not found in catalog")
            return explicit if explicit.enabled else None

        level_rank = _RISK_LEVEL_ORDER.get(assessment.risk_level, 0)
        detected_set = set(assessment.patterns_detected)

        for pb in self.list_playbooks(enabled_only=True):
            min_rank = _RISK_LEVEL_ORDER.get(UnifiedRiskLevel(pb.min_risk_level), 0)
            if level_rank < min_rank or assessment.composite_score < pb.min_composite_score:
                continue
            if pb.trigger_patterns and not (set(pb.trigger_patterns) & detected_set):
                continue
            return pb
        return None
