"""Unified Cross-Engine Risk Orchestrator & Entity Case Correlation.

Aggregates risk signals across specialized fraud domains (ATO, AML, Synthetic
Identity, Mule Detection, First-Party Fraud, Transaction Scoring), detects
compound multi-vector attack patterns, computes calibrated composite entity
risk scores, and correlates investigation cases in ``CaseManager``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _parse_iso(ts: str) -> datetime:
    parsed = datetime.fromisoformat(ts)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


class RiskDomain(StrEnum):
    """Supported domain engines contributing risk signals to an entity."""

    ATO = "ato"
    AML = "aml"
    SYNTHETIC_IDENTITY = "synthetic_identity"
    MULE = "mule"
    FIRST_PARTY = "first_party"
    TRANSACTION = "transaction"
    CUSTOM = "custom"


class UnifiedRiskLevel(StrEnum):
    """Calibrated composite risk tier for an orchestrated entity."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OrchestratorAction(StrEnum):
    """Automated policy action selected by the unified risk orchestrator."""

    ALLOW = "allow"
    STEP_UP_CHALLENGE = "step_up_challenge"
    HOLD_AND_REVIEW = "hold_and_review"
    RESTRICT_ENTITY = "restrict_entity"
    BLOCK_AND_ESCALATE = "block_and_escalate"


class CrossSignalPattern(StrEnum):
    """Compound multi-engine attack patterns detected across domains."""

    ATO_TO_MULE_EXFILTRATION = "ato_to_mule_exfiltration"
    SYNTHETIC_TO_AML_LAYERING = "synthetic_to_aml_layering"
    SYNTHETIC_BUST_OUT = "synthetic_bust_out"
    MULTI_VECTOR_SYNDICATE = "multi_vector_syndicate"
    ESCALATED_REPEAT_OFFENDER = "escalated_repeat_offender"


DEFAULT_DOMAIN_WEIGHTS: dict[str, float] = {
    RiskDomain.ATO.value: 1.15,
    RiskDomain.AML.value: 1.15,
    RiskDomain.SYNTHETIC_IDENTITY.value: 1.10,
    RiskDomain.MULE.value: 1.15,
    RiskDomain.FIRST_PARTY.value: 0.95,
    RiskDomain.TRANSACTION.value: 1.00,
    RiskDomain.CUSTOM.value: 0.90,
}


@dataclass(slots=True)
class DomainRiskSignal:
    """Normalized risk signal emitted by an individual domain engine."""

    entity_id: str
    domain: RiskDomain | str
    score: float
    tenant_id: str = "default"
    signal_id: str = field(default_factory=lambda: f"sig_{uuid4().hex[:12]}")
    timestamp: str = field(default_factory=_utc_now_iso)
    source_assessment_id: str | None = None
    transaction_id: str | None = None
    reason_codes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.entity_id = self.entity_id.strip()
        if not self.entity_id:
            raise ValueError("entity_id must not be empty")
        if not isinstance(self.domain, RiskDomain):
            self.domain = RiskDomain(str(self.domain).lower())
        self.score = round(_clamp(self.score), 6)
        self.tenant_id = self.tenant_id.strip() or "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "entity_id": self.entity_id,
            "domain": RiskDomain(self.domain).value,
            "score": self.score,
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp,
            "source_assessment_id": self.source_assessment_id,
            "transaction_id": self.transaction_id,
            "reason_codes": list(self.reason_codes),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DomainRiskSignal:
        return cls(
            signal_id=str(payload.get("signal_id") or f"sig_{uuid4().hex[:12]}"),
            entity_id=str(payload["entity_id"]),
            domain=RiskDomain(str(payload["domain"]).lower()),
            score=float(payload["score"]),
            tenant_id=str(payload.get("tenant_id") or "default"),
            timestamp=str(payload.get("timestamp") or _utc_now_iso()),
            source_assessment_id=payload.get("source_assessment_id"),
            transaction_id=payload.get("transaction_id"),
            reason_codes=[str(x) for x in payload.get("reason_codes", [])],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class OrchestratorPolicy:
    """Configurable weights, thresholds, and correlation settings."""

    domain_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DOMAIN_WEIGHTS))
    elevated_threshold: float = 0.55
    medium_threshold: float = 0.35
    high_threshold: float = 0.65
    critical_threshold: float = 0.85
    pattern_uplift: float = 0.08
    correlation_window_seconds: int = 86400
    auto_correlate_cases: bool = True

    def weight_for(self, domain: RiskDomain) -> float:
        return max(0.1, float(self.domain_weights.get(domain.value, 1.0)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_weights": dict(self.domain_weights),
            "elevated_threshold": self.elevated_threshold,
            "medium_threshold": self.medium_threshold,
            "high_threshold": self.high_threshold,
            "critical_threshold": self.critical_threshold,
            "pattern_uplift": self.pattern_uplift,
            "correlation_window_seconds": self.correlation_window_seconds,
            "auto_correlate_cases": self.auto_correlate_cases,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OrchestratorPolicy:
        return cls(
            domain_weights={
                str(k): float(v)
                for k, v in payload.get("domain_weights", DEFAULT_DOMAIN_WEIGHTS).items()
            },
            elevated_threshold=float(payload.get("elevated_threshold", 0.55)),
            medium_threshold=float(payload.get("medium_threshold", 0.35)),
            high_threshold=float(payload.get("high_threshold", 0.65)),
            critical_threshold=float(payload.get("critical_threshold", 0.85)),
            pattern_uplift=float(payload.get("pattern_uplift", 0.08)),
            correlation_window_seconds=int(payload.get("correlation_window_seconds", 86400)),
            auto_correlate_cases=bool(payload.get("auto_correlate_cases", True)),
        )


@dataclass(slots=True)
class EntityRiskProfile:
    """Aggregated cross-domain risk state for a single entity."""

    entity_id: str
    tenant_id: str = "default"
    domain_scores: dict[str, float] = field(default_factory=dict)
    signals: list[DomainRiskSignal] = field(default_factory=list)
    last_updated: str = field(default_factory=_utc_now_iso)
    last_assessment_id: str | None = None
    active_case_id: str | None = None
    high_risk_streak: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "signal_count": len(self.signals),
            "domain_scores": dict(self.domain_scores),
            "signals": [sig.to_dict() for sig in self.signals],
            "last_updated": self.last_updated,
            "last_assessment_id": self.last_assessment_id,
            "active_case_id": self.active_case_id,
            "high_risk_streak": self.high_risk_streak,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EntityRiskProfile:
        return cls(
            entity_id=str(payload["entity_id"]),
            tenant_id=str(payload.get("tenant_id") or "default"),
            domain_scores={str(k): float(v) for k, v in payload.get("domain_scores", {}).items()},
            signals=[DomainRiskSignal.from_dict(item) for item in payload.get("signals", [])],
            last_updated=str(payload.get("last_updated") or _utc_now_iso()),
            last_assessment_id=payload.get("last_assessment_id"),
            active_case_id=payload.get("active_case_id"),
            high_risk_streak=int(payload.get("high_risk_streak", 0)),
        )


@dataclass(slots=True)
class UnifiedRiskAssessment:
    """Output of a cross-engine entity risk orchestration evaluation."""

    assessment_id: str
    entity_id: str
    tenant_id: str
    composite_score: float
    risk_level: UnifiedRiskLevel
    action: OrchestratorAction
    domain_scores: dict[str, float] = field(default_factory=dict)
    patterns_detected: list[CrossSignalPattern] = field(default_factory=list)
    contributing_signal_ids: list[str] = field(default_factory=list)
    correlated_case_id: str | None = None
    case_created: bool = False
    rationale: list[str] = field(default_factory=list)
    evaluated_at: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "composite_score": round(self.composite_score, 6),
            "risk_level": self.risk_level.value,
            "action": self.action.value,
            "domain_scores": dict(self.domain_scores),
            "patterns_detected": [p.value for p in self.patterns_detected],
            "contributing_signal_ids": list(self.contributing_signal_ids),
            "correlated_case_id": self.correlated_case_id,
            "case_created": self.case_created,
            "rationale": list(self.rationale),
            "evaluated_at": self.evaluated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> UnifiedRiskAssessment:
        return cls(
            assessment_id=str(payload["assessment_id"]),
            entity_id=str(payload["entity_id"]),
            tenant_id=str(payload.get("tenant_id") or "default"),
            composite_score=float(payload["composite_score"]),
            risk_level=UnifiedRiskLevel(str(payload["risk_level"]).lower()),
            action=OrchestratorAction(str(payload["action"]).lower()),
            domain_scores={str(k): float(v) for k, v in payload.get("domain_scores", {}).items()},
            patterns_detected=[
                CrossSignalPattern(str(p).lower()) for p in payload.get("patterns_detected", [])
            ],
            contributing_signal_ids=[str(s) for s in payload.get("contributing_signal_ids", [])],
            correlated_case_id=payload.get("correlated_case_id"),
            case_created=bool(payload.get("case_created", False)),
            rationale=[str(r) for r in payload.get("rationale", [])],
            evaluated_at=str(payload.get("evaluated_at") or _utc_now_iso()),
            metadata=dict(payload.get("metadata", {})),
        )


class CrossSignalAnalyzer:
    """Detects compound multi-engine attack patterns from per-domain risk scores."""

    def analyze(
        self,
        domain_scores: dict[str, float],
        *,
        policy: OrchestratorPolicy,
        high_risk_streak: int = 0,
    ) -> tuple[list[CrossSignalPattern], list[str]]:
        patterns: list[CrossSignalPattern] = []
        rationale: list[str] = []
        thr = policy.elevated_threshold

        ato = domain_scores.get(RiskDomain.ATO.value, 0.0)
        aml = domain_scores.get(RiskDomain.AML.value, 0.0)
        synthetic = domain_scores.get(RiskDomain.SYNTHETIC_IDENTITY.value, 0.0)
        mule = domain_scores.get(RiskDomain.MULE.value, 0.0)
        first_party = domain_scores.get(RiskDomain.FIRST_PARTY.value, 0.0)

        if ato >= thr and mule >= thr:
            patterns.append(CrossSignalPattern.ATO_TO_MULE_EXFILTRATION)
            rationale.append(
                f"ATO ({ato:.2f}) + Mule ({mule:.2f}) indicates account takeover exfiltration"
            )
        if synthetic >= thr and aml >= thr:
            patterns.append(CrossSignalPattern.SYNTHETIC_TO_AML_LAYERING)
            rationale.append(
                f"Synthetic identity ({synthetic:.2f}) + AML ({aml:.2f}) indicates layering"
            )
        if synthetic >= thr and first_party >= thr:
            patterns.append(CrossSignalPattern.SYNTHETIC_BUST_OUT)
            rationale.append(
                f"Synthetic identity ({synthetic:.2f}) + First-party ({first_party:.2f}) "
                "indicates bust-out credit/refund abuse"
            )

        elevated_domains = [d for d, s in domain_scores.items() if s >= thr]
        if len(elevated_domains) >= 3:
            patterns.append(CrossSignalPattern.MULTI_VECTOR_SYNDICATE)
            rationale.append(
                f"{len(elevated_domains)} domains elevated ({', '.join(sorted(elevated_domains))})"
            )

        if high_risk_streak >= 2 and domain_scores and max(domain_scores.values()) >= thr:
            patterns.append(CrossSignalPattern.ESCALATED_REPEAT_OFFENDER)
            rationale.append(f"Repeat elevated risk across {high_risk_streak} consecutive sweeps")

        return patterns, rationale
