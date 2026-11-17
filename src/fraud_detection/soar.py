"""Automated SOAR Incident Response Playbooks & Compensating Control Containment Engine.

Executes deterministic, auditable, reversible containment playbooks in response
to high-severity cross-engine entity risk assessments (``UnifiedRiskAssessment``)
and compound attack patterns, managing active compensating controls with TTL
expiration and full operator rollback.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from fraud_detection.cases import CaseManager, CaseNotFoundError, CaseStatus
from fraud_detection.events import EventBus, EventType, FraudDomainEvent
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


class SoarStore:
    """Thread-safe in-memory store for SOAR playbook executions and entity containments."""

    def __init__(
        self,
        *,
        max_executions: int = 10000,
        max_containments: int = 20000,
    ) -> None:
        self._lock = threading.RLock()
        self.max_executions = max(1, max_executions)
        self.max_containments = max(1, max_containments)
        self._executions: dict[str, PlaybookExecutionRecord] = {}
        self._execution_order: list[str] = []
        self._containments: dict[str, ActiveContainmentRecord] = {}
        self._containment_order: list[str] = []

    def _evict_executions_if_needed(self) -> None:
        while len(self._execution_order) > self.max_executions:
            oldest_id = self._execution_order.pop(0)
            self._executions.pop(oldest_id, None)

    def _evict_containments_if_needed(self) -> None:
        while len(self._containment_order) > self.max_containments:
            oldest_id = self._containment_order.pop(0)
            self._containments.pop(oldest_id, None)

    def save_execution(self, record: PlaybookExecutionRecord) -> None:
        with self._lock:
            if record.execution_id not in self._executions:
                self._execution_order.append(record.execution_id)
            self._executions[record.execution_id] = record
            self._evict_executions_if_needed()

    def get_execution(self, execution_id: str) -> PlaybookExecutionRecord | None:
        with self._lock:
            return self._executions.get(execution_id)

    def list_executions(
        self,
        *,
        entity_id: str | None = None,
        tenant_id: str | None = None,
        status: PlaybookRunStatus | None = None,
        limit: int = 100,
    ) -> list[PlaybookExecutionRecord]:
        with self._lock:
            items = [
                self._executions[eid]
                for eid in reversed(self._execution_order)
                if eid in self._executions
            ]
            if entity_id is not None:
                items = [e for e in items if e.entity_id == entity_id]
            if tenant_id is not None:
                items = [e for e in items if e.tenant_id == tenant_id]
            if status is not None:
                items = [e for e in items if e.status is status]
            return items[: max(1, limit)]

    def save_containment(self, record: ActiveContainmentRecord) -> None:
        with self._lock:
            if record.containment_id not in self._containments:
                self._containment_order.append(record.containment_id)
            self._containments[record.containment_id] = record
            self._evict_containments_if_needed()

    def get_containment(self, containment_id: str) -> ActiveContainmentRecord | None:
        with self._lock:
            return self._containments.get(containment_id)

    def list_containments(
        self,
        *,
        entity_id: str | None = None,
        tenant_id: str | None = None,
        active_only: bool = False,
        reference_time: datetime | None = None,
    ) -> list[ActiveContainmentRecord]:
        now = reference_time or datetime.now(UTC)
        with self._lock:
            results: list[ActiveContainmentRecord] = []
            for cid in reversed(self._containment_order):
                rec = self._containments.get(cid)
                if rec is None:
                    continue
                if rec.state is ContainmentState.ACTIVE and rec.is_expired(now):
                    rec.state = ContainmentState.EXPIRED
                if entity_id is not None and rec.entity_id != entity_id:
                    continue
                if tenant_id is not None and rec.tenant_id != tenant_id:
                    continue
                if active_only and rec.state is not ContainmentState.ACTIVE:
                    continue
                results.append(rec)
            return results

    def find_active_control(
        self,
        entity_id: str,
        control_type: ContainmentControlType,
        *,
        tenant_id: str = "default",
        reference_time: datetime | None = None,
    ) -> ActiveContainmentRecord | None:
        for rec in self.list_containments(
            entity_id=entity_id,
            tenant_id=tenant_id,
            active_only=True,
            reference_time=reference_time,
        ):
            if rec.control_type is control_type:
                return rec
        return None

    def summary(self, *, reference_time: datetime | None = None) -> dict[str, Any]:
        now = reference_time or datetime.now(UTC)
        with self._lock:
            by_status: dict[str, int] = {st.value: 0 for st in PlaybookRunStatus}
            for exe in self._executions.values():
                status_key = PlaybookRunStatus(exe.status).value
                by_status[status_key] = by_status.get(status_key, 0) + 1

            by_control: dict[str, int] = {ct.value: 0 for ct in ContainmentControlType}
            active_count = 0
            for rec in self._containments.values():
                if rec.state is ContainmentState.ACTIVE and rec.is_expired(now):
                    rec.state = ContainmentState.EXPIRED
                if rec.state is ContainmentState.ACTIVE:
                    active_count += 1
                    ctrl_key = ContainmentControlType(rec.control_type).value
                    by_control[ctrl_key] = by_control.get(ctrl_key, 0) + 1

            return {
                "execution_count": len(self._executions),
                "containment_count": len(self._containments),
                "active_containments": active_count,
                "by_status": by_status,
                "active_by_control": by_control,
            }

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_executions": self.max_executions,
                "max_containments": self.max_containments,
                "executions": [
                    self._executions[eid].to_dict()
                    for eid in self._execution_order
                    if eid in self._executions
                ],
                "containments": [
                    self._containments[cid].to_dict()
                    for cid in self._containment_order
                    if cid in self._containments
                ],
            }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SoarStore:
        store = cls(
            max_executions=int(payload.get("max_executions", 10000)),
            max_containments=int(payload.get("max_containments", 20000)),
        )
        for e_raw in payload.get("executions", []):
            exe = PlaybookExecutionRecord.from_dict(e_raw)
            store._executions[exe.execution_id] = exe
            store._execution_order.append(exe.execution_id)
        for c_raw in payload.get("containments", []):
            cnt = ActiveContainmentRecord.from_dict(c_raw)
            store._containments[cnt.containment_id] = cnt
            store._containment_order.append(cnt.containment_id)
        return store

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> SoarStore:
        source = Path(path)
        if not source.exists():
            return cls()
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))


class SoarEngine:
    """Executes deterministic SOAR containment playbooks with TTL tracking and rollback."""

    def __init__(
        self,
        *,
        catalog: PlaybookCatalog | None = None,
        store: SoarStore | None = None,
        case_manager: CaseManager | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.catalog = catalog or PlaybookCatalog()
        self.store = store or SoarStore()
        self.case_manager = case_manager
        self.event_bus = event_bus
        self._lock = threading.RLock()

    def _emit_event(
        self,
        event_type: EventType,
        *,
        tenant_id: str,
        payload: dict[str, Any],
    ) -> None:
        if self.event_bus is None:
            return
        self.event_bus.publish(
            FraudDomainEvent(
                event_type=event_type,
                tenant_id=tenant_id,
                payload=payload,
            )
        )

    def _escalate_case(
        self,
        *,
        assessment: UnifiedRiskAssessment,
        playbook: ResponsePlaybook,
        execution_id: str,
    ) -> str | None:
        if self.case_manager is None:
            return assessment.correlated_case_id

        if assessment.correlated_case_id:
            try:
                existing = self.case_manager.get_case(assessment.correlated_case_id)
                if existing.status not in {CaseStatus.RESOLVED, CaseStatus.CLOSED}:
                    existing.metadata["soar_execution_id"] = execution_id
                    existing.metadata["soar_playbook_id"] = playbook.playbook_id
                    existing.add_audit_entry(
                        action="soar_playbook_escalated",
                        actor="soar_engine",
                        details={
                            "execution_id": execution_id,
                            "playbook_id": playbook.playbook_id,
                        },
                    )
                    self.case_manager.store.put(existing)
                    return existing.case_id
            except CaseNotFoundError:
                pass

        created = self.case_manager.create_case(
            transaction_id=f"soar:{assessment.entity_id}:{execution_id}",
            fraud_score=assessment.composite_score,
            tenant_id=assessment.tenant_id,
            metadata={
                "source": "soar_engine",
                "entity_id": assessment.entity_id,
                "execution_id": execution_id,
                "playbook_id": playbook.playbook_id,
                "patterns": [p.value for p in assessment.patterns_detected],
            },
        )
        return created.case_id

    def execute_for_assessment(
        self,
        assessment: UnifiedRiskAssessment,
        *,
        playbook_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        reference_time: datetime | None = None,
    ) -> PlaybookExecutionRecord:
        now = reference_time or datetime.now(UTC)
        with self._lock:
            playbook = self.catalog.select_playbook(assessment, playbook_id=playbook_id)
            execution_id = f"soar_{uuid4().hex[:12]}"

            if playbook is None:
                skipped = PlaybookExecutionRecord(
                    execution_id=execution_id,
                    playbook_id=playbook_id or "none",
                    entity_id=assessment.entity_id,
                    tenant_id=assessment.tenant_id,
                    status=PlaybookRunStatus.SKIPPED,
                    composite_score=assessment.composite_score,
                    risk_level=assessment.risk_level,
                    source_assessment_id=assessment.assessment_id,
                    triggered_patterns=list(assessment.patterns_detected),
                    correlated_case_id=assessment.correlated_case_id,
                    executed_at=now.isoformat(),
                    metadata=dict(metadata or {}),
                )
                self.store.save_execution(skipped)
                return skipped

            self._emit_event(
                EventType.SOAR_PLAYBOOK_TRIGGERED,
                tenant_id=assessment.tenant_id,
                payload={
                    "execution_id": execution_id,
                    "playbook_id": playbook.playbook_id,
                    "entity_id": assessment.entity_id,
                    "composite_score": assessment.composite_score,
                    "risk_level": assessment.risk_level.value,
                },
            )

            step_results: list[PlaybookStepResult] = []
            correlated_case_id = assessment.correlated_case_id

            for step in playbook.steps:
                ctrl = ContainmentControlType(step.control_type)
                if ctrl is ContainmentControlType.ESCALATE_CASE:
                    escalated_case_id = self._escalate_case(
                        assessment=assessment,
                        playbook=playbook,
                        execution_id=execution_id,
                    )
                    if escalated_case_id:
                        correlated_case_id = escalated_case_id
                    step_results.append(
                        PlaybookStepResult(
                            control_type=ctrl,
                            applied=escalated_case_id is not None,
                            case_id=escalated_case_id,
                            message=(
                                f"Escalated investigation case {escalated_case_id}"
                                if escalated_case_id
                                else "Case escalation skipped (no CaseManager configured)"
                            ),
                            executed_at=now.isoformat(),
                        )
                    )
                    continue

                existing_cnt = self.store.find_active_control(
                    assessment.entity_id,
                    ctrl,
                    tenant_id=assessment.tenant_id,
                    reference_time=now,
                )
                if existing_cnt is not None:
                    step_results.append(
                        PlaybookStepResult(
                            control_type=ctrl,
                            applied=False,
                            containment_id=existing_cnt.containment_id,
                            message=(
                                f"Control '{ctrl.value}' already active "
                                f"({existing_cnt.containment_id})"
                            ),
                            executed_at=now.isoformat(),
                        )
                    )
                    continue

                expires_at = (
                    (now + timedelta(seconds=step.ttl_seconds)).isoformat()
                    if step.ttl_seconds is not None
                    else None
                )
                containment = ActiveContainmentRecord(
                    entity_id=assessment.entity_id,
                    tenant_id=assessment.tenant_id,
                    control_type=ctrl,
                    execution_id=execution_id,
                    playbook_id=playbook.playbook_id,
                    state=ContainmentState.ACTIVE,
                    reason=step.description or f"Applied by {playbook.playbook_id}",
                    applied_at=now.isoformat(),
                    expires_at=expires_at,
                    metadata=dict(step.parameters),
                )
                self.store.save_containment(containment)
                self._emit_event(
                    EventType.SOAR_CONTAINMENT_APPLIED,
                    tenant_id=assessment.tenant_id,
                    payload={
                        "execution_id": execution_id,
                        "playbook_id": playbook.playbook_id,
                        "containment_id": containment.containment_id,
                        "entity_id": assessment.entity_id,
                        "control_type": ctrl.value,
                        "expires_at": expires_at,
                    },
                )
                step_results.append(
                    PlaybookStepResult(
                        control_type=ctrl,
                        applied=True,
                        containment_id=containment.containment_id,
                        message=step.description or f"Applied {ctrl.value}",
                        executed_at=now.isoformat(),
                    )
                )

            applied_any = any(r.applied for r in step_results)
            all_applied = bool(step_results) and all(r.applied for r in step_results)
            if all_applied:
                status = PlaybookRunStatus.COMPLETED
            elif applied_any:
                status = PlaybookRunStatus.PARTIAL
            else:
                status = PlaybookRunStatus.SKIPPED

            record = PlaybookExecutionRecord(
                execution_id=execution_id,
                playbook_id=playbook.playbook_id,
                entity_id=assessment.entity_id,
                tenant_id=assessment.tenant_id,
                status=status,
                composite_score=assessment.composite_score,
                risk_level=assessment.risk_level,
                source_assessment_id=assessment.assessment_id,
                triggered_patterns=list(assessment.patterns_detected),
                step_results=step_results,
                correlated_case_id=correlated_case_id,
                executed_at=now.isoformat(),
                metadata=dict(metadata or {}),
            )
            self.store.save_execution(record)
            self._emit_event(
                EventType.SOAR_PLAYBOOK_COMPLETED,
                tenant_id=assessment.tenant_id,
                payload={
                    "execution_id": record.execution_id,
                    "playbook_id": record.playbook_id,
                    "entity_id": record.entity_id,
                    "status": PlaybookRunStatus(record.status).value,
                    "correlated_case_id": record.correlated_case_id,
                },
            )
            return record

    def release_containment(
        self,
        containment_id: str,
        *,
        reason: str = "operator_release",
        reference_time: datetime | None = None,
    ) -> ActiveContainmentRecord:
        now = reference_time or datetime.now(UTC)
        with self._lock:
            rec = self.store.get_containment(containment_id)
            if rec is None:
                raise KeyError(f"Containment '{containment_id}' not found")
            if rec.state is ContainmentState.ACTIVE:
                rec.state = ContainmentState.RELEASED
                rec.released_at = now.isoformat()
                rec.release_reason = reason
                self.store.save_containment(rec)
                self._emit_event(
                    EventType.SOAR_CONTAINMENT_ROLLED_BACK,
                    tenant_id=rec.tenant_id,
                    payload={
                        "containment_id": rec.containment_id,
                        "execution_id": rec.execution_id,
                        "entity_id": rec.entity_id,
                        "control_type": ContainmentControlType(rec.control_type).value,
                        "reason": reason,
                    },
                )
            return rec

    def rollback_execution(
        self,
        execution_id: str,
        *,
        reason: str = "operator_rollback",
        reference_time: datetime | None = None,
    ) -> PlaybookExecutionRecord:
        now = reference_time or datetime.now(UTC)
        with self._lock:
            record = self.store.get_execution(execution_id)
            if record is None:
                raise KeyError(f"Playbook execution '{execution_id}' not found")
            released_ids: list[str] = []
            for step in record.step_results:
                if step.containment_id and step.applied:
                    cnt = self.store.get_containment(step.containment_id)
                    if cnt is not None and cnt.state is ContainmentState.ACTIVE:
                        self.release_containment(
                            cnt.containment_id,
                            reason=reason,
                            reference_time=now,
                        )
                        released_ids.append(cnt.containment_id)
            record.status = PlaybookRunStatus.ROLLED_BACK
            record.rolled_back_at = now.isoformat()
            record.rollback_reason = reason
            record.metadata["released_containment_ids"] = released_ids
            self.store.save_execution(record)
            return record
