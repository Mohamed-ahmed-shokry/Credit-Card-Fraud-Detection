"""Unified Cross-Engine Risk Orchestrator & Entity Case Correlation.

Aggregates risk signals across specialized fraud domains (ATO, AML, Synthetic
Identity, Mule Detection, First-Party Fraud, Transaction Scoring), detects
compound multi-vector attack patterns, computes calibrated composite entity
risk scores, and correlates investigation cases in ``CaseManager``.
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
    APP_SCAM = "app_scam"
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
    RiskDomain.APP_SCAM.value: 1.15,
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


class EntityCaseCorrelator:
    """Correlates cross-engine entity assessments into unified CaseManager cases."""

    def correlate(
        self,
        *,
        case_manager: CaseManager,
        profile: EntityRiskProfile,
        assessment: UnifiedRiskAssessment,
    ) -> tuple[str, bool]:
        if profile.active_case_id:
            try:
                existing = case_manager.get_case(profile.active_case_id)
                if existing.status not in {CaseStatus.RESOLVED, CaseStatus.CLOSED}:
                    existing.fraud_score = max(existing.fraud_score, assessment.composite_score)
                    existing.metadata["cross_engine_domains"] = dict(assessment.domain_scores)
                    existing.metadata["patterns"] = [p.value for p in assessment.patterns_detected]
                    existing.metadata["last_orchestrator_assessment_id"] = assessment.assessment_id
                    existing.add_audit_entry(
                        action="orchestrator_correlated",
                        actor="risk_orchestrator",
                        details={
                            "assessment_id": assessment.assessment_id,
                            "composite_score": assessment.composite_score,
                            "patterns": [p.value for p in assessment.patterns_detected],
                        },
                    )
                    case_manager.store.put(existing)
                    return existing.case_id, False
            except CaseNotFoundError:
                pass

        created = case_manager.create_case(
            transaction_id=f"entity:{profile.entity_id}",
            fraud_score=assessment.composite_score,
            tenant_id=profile.tenant_id,
            metadata={
                "source": "risk_orchestrator",
                "entity_id": profile.entity_id,
                "assessment_id": assessment.assessment_id,
                "cross_engine_domains": dict(assessment.domain_scores),
                "patterns": [p.value for p in assessment.patterns_detected],
            },
        )
        profile.active_case_id = created.case_id
        return created.case_id, True


class UnifiedRiskStore:
    """Thread-safe in-memory store for entity risk profiles and assessments."""

    def __init__(
        self,
        *,
        max_entities: int = 5000,
        max_assessments: int = 10000,
        max_signals_per_entity: int = 200,
    ) -> None:
        self._lock = threading.RLock()
        self.max_entities = max(1, max_entities)
        self.max_assessments = max(1, max_assessments)
        self.max_signals_per_entity = max(1, max_signals_per_entity)
        self._profiles: dict[tuple[str, str], EntityRiskProfile] = {}
        self._assessments: dict[str, UnifiedRiskAssessment] = {}
        self._assessment_order: list[str] = []

    def _evict_entities_if_needed(self) -> None:
        while len(self._profiles) > self.max_entities:
            oldest_key = min(
                self._profiles,
                key=lambda k: self._profiles[k].last_updated,
            )
            del self._profiles[oldest_key]

    def _evict_assessments_if_needed(self) -> None:
        while len(self._assessment_order) > self.max_assessments:
            oldest_id = self._assessment_order.pop(0)
            self._assessments.pop(oldest_id, None)

    def get_or_create_profile(
        self, entity_id: str, tenant_id: str = "default"
    ) -> EntityRiskProfile:
        key = (tenant_id.strip() or "default", entity_id.strip())
        with self._lock:
            profile = self._profiles.get(key)
            if profile is None:
                profile = EntityRiskProfile(entity_id=key[1], tenant_id=key[0])
                self._profiles[key] = profile
                self._evict_entities_if_needed()
            return profile

    def get_profile(self, entity_id: str, tenant_id: str = "default") -> EntityRiskProfile | None:
        key = (tenant_id.strip() or "default", entity_id.strip())
        with self._lock:
            return self._profiles.get(key)

    def record_signal(self, signal: DomainRiskSignal) -> EntityRiskProfile:
        with self._lock:
            profile = self.get_or_create_profile(signal.entity_id, signal.tenant_id)
            profile.signals.append(signal)
            if len(profile.signals) > self.max_signals_per_entity:
                profile.signals = profile.signals[-self.max_signals_per_entity :]
            domain_key = RiskDomain(signal.domain).value
            profile.domain_scores[domain_key] = max(
                profile.domain_scores.get(domain_key, 0.0), signal.score
            )
            profile.last_updated = _utc_now_iso()
            return profile

    def save_assessment(self, assessment: UnifiedRiskAssessment) -> None:
        with self._lock:
            if assessment.assessment_id not in self._assessments:
                self._assessment_order.append(assessment.assessment_id)
            self._assessments[assessment.assessment_id] = assessment
            self._evict_assessments_if_needed()

    def get_assessment(self, assessment_id: str) -> UnifiedRiskAssessment | None:
        with self._lock:
            return self._assessments.get(assessment_id)

    def list_assessments(
        self,
        *,
        entity_id: str | None = None,
        tenant_id: str | None = None,
        risk_level: UnifiedRiskLevel | None = None,
        limit: int = 100,
    ) -> list[UnifiedRiskAssessment]:
        with self._lock:
            items = [
                self._assessments[aid]
                for aid in reversed(self._assessment_order)
                if aid in self._assessments
            ]
            if entity_id is not None:
                items = [a for a in items if a.entity_id == entity_id]
            if tenant_id is not None:
                items = [a for a in items if a.tenant_id == tenant_id]
            if risk_level is not None:
                items = [a for a in items if a.risk_level is risk_level]
            return items[: max(1, limit)]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            by_level: dict[str, int] = {lvl.value: 0 for lvl in UnifiedRiskLevel}
            by_pattern: dict[str, int] = {pat.value: 0 for pat in CrossSignalPattern}
            correlated_cases = 0
            for assessment in self._assessments.values():
                by_level[assessment.risk_level.value] = (
                    by_level.get(assessment.risk_level.value, 0) + 1
                )
                for pat in assessment.patterns_detected:
                    by_pattern[pat.value] = by_pattern.get(pat.value, 0) + 1
                if assessment.correlated_case_id:
                    correlated_cases += 1
            return {
                "entity_count": len(self._profiles),
                "assessment_count": len(self._assessments),
                "correlated_assessments": correlated_cases,
                "by_risk_level": by_level,
                "by_pattern": by_pattern,
            }

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_entities": self.max_entities,
                "max_assessments": self.max_assessments,
                "max_signals_per_entity": self.max_signals_per_entity,
                "profiles": [p.to_dict() for p in self._profiles.values()],
                "assessments": [
                    self._assessments[aid].to_dict()
                    for aid in self._assessment_order
                    if aid in self._assessments
                ],
            }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> UnifiedRiskStore:
        store = cls(
            max_entities=int(payload.get("max_entities", 5000)),
            max_assessments=int(payload.get("max_assessments", 10000)),
            max_signals_per_entity=int(payload.get("max_signals_per_entity", 200)),
        )
        for p_raw in payload.get("profiles", []):
            profile = EntityRiskProfile.from_dict(p_raw)
            store._profiles[(profile.tenant_id, profile.entity_id)] = profile
        for a_raw in payload.get("assessments", []):
            assessment = UnifiedRiskAssessment.from_dict(a_raw)
            store._assessments[assessment.assessment_id] = assessment
            store._assessment_order.append(assessment.assessment_id)
        return store

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> UnifiedRiskStore:
        source = Path(path)
        if not source.exists():
            return cls()
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))


class UnifiedRiskOrchestrator:
    """Orchestrates cross-engine signals, compound pattern detection, and case correlation."""

    def __init__(
        self,
        *,
        policy: OrchestratorPolicy | None = None,
        store: UnifiedRiskStore | None = None,
        analyzer: CrossSignalAnalyzer | None = None,
        correlator: EntityCaseCorrelator | None = None,
        case_manager: CaseManager | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.policy = policy or OrchestratorPolicy()
        self.store = store or UnifiedRiskStore()
        self.analyzer = analyzer or CrossSignalAnalyzer()
        self.correlator = correlator or EntityCaseCorrelator()
        self.case_manager = case_manager
        self.event_bus = event_bus
        self._lock = threading.RLock()

    def ingest_signal(self, signal: DomainRiskSignal) -> EntityRiskProfile:
        with self._lock:
            return self.store.record_signal(signal)

    def _active_domain_scores(
        self, profile: EntityRiskProfile, reference_time: datetime
    ) -> tuple[dict[str, float], list[str]]:
        cutoff = reference_time - timedelta(seconds=self.policy.correlation_window_seconds)
        scores: dict[str, float] = {}
        signal_ids: list[str] = []
        for sig in profile.signals:
            if _parse_iso(sig.timestamp) >= cutoff:
                domain_key = RiskDomain(sig.domain).value
                scores[domain_key] = max(scores.get(domain_key, 0.0), sig.score)
                signal_ids.append(sig.signal_id)
        if not scores and profile.domain_scores:
            scores = dict(profile.domain_scores)
        return scores, signal_ids

    def _score_and_route(
        self,
        domain_scores: dict[str, float],
        patterns: list[CrossSignalPattern],
    ) -> tuple[float, UnifiedRiskLevel, OrchestratorAction]:
        if not domain_scores:
            return 0.0, UnifiedRiskLevel.LOW, OrchestratorAction.ALLOW

        total_weight = sum(self.policy.weight_for(RiskDomain(d)) for d in domain_scores)
        weighted_avg = (
            sum(s * self.policy.weight_for(RiskDomain(d)) for d, s in domain_scores.items())
            / total_weight
        )
        peak = max(domain_scores.values())
        base = 0.65 * peak + 0.35 * weighted_avg
        uplift = min(0.25, len(patterns) * self.policy.pattern_uplift)
        composite = round(_clamp(base + uplift), 6)

        if composite >= self.policy.critical_threshold:
            return composite, UnifiedRiskLevel.CRITICAL, OrchestratorAction.BLOCK_AND_ESCALATE
        if composite >= self.policy.high_threshold:
            return composite, UnifiedRiskLevel.HIGH, OrchestratorAction.RESTRICT_ENTITY
        if composite >= self.policy.elevated_threshold:
            return composite, UnifiedRiskLevel.MEDIUM, OrchestratorAction.HOLD_AND_REVIEW
        if composite >= self.policy.medium_threshold:
            return composite, UnifiedRiskLevel.MEDIUM, OrchestratorAction.STEP_UP_CHALLENGE
        return composite, UnifiedRiskLevel.LOW, OrchestratorAction.ALLOW

    def evaluate_entity(
        self,
        entity_id: str,
        *,
        tenant_id: str = "default",
        signals: list[DomainRiskSignal] | None = None,
        metadata: dict[str, Any] | None = None,
        reference_time: datetime | None = None,
    ) -> UnifiedRiskAssessment:
        with self._lock:
            for sig in signals or []:
                self.store.record_signal(sig)

            profile = self.store.get_or_create_profile(entity_id, tenant_id=tenant_id)
            now = reference_time or datetime.now(UTC)
            domain_scores, signal_ids = self._active_domain_scores(profile, now)
            profile.domain_scores = dict(domain_scores)

            patterns, rationale = self.analyzer.analyze(
                domain_scores,
                policy=self.policy,
                high_risk_streak=profile.high_risk_streak,
            )
            composite_score, risk_level, action = self._score_and_route(domain_scores, patterns)
            if not rationale and domain_scores:
                top_domain = max(domain_scores, key=lambda k: domain_scores[k])
                rationale.append(
                    f"Dominant risk domain '{top_domain}' scored {domain_scores[top_domain]:.2f}"
                )

            if risk_level in {UnifiedRiskLevel.HIGH, UnifiedRiskLevel.CRITICAL}:
                profile.high_risk_streak += 1
            else:
                profile.high_risk_streak = 0

            assessment = UnifiedRiskAssessment(
                assessment_id=f"ora_{uuid4().hex[:12]}",
                entity_id=profile.entity_id,
                tenant_id=profile.tenant_id,
                composite_score=composite_score,
                risk_level=risk_level,
                action=action,
                domain_scores=domain_scores,
                patterns_detected=patterns,
                contributing_signal_ids=signal_ids,
                rationale=rationale,
                evaluated_at=now.isoformat(),
                metadata=dict(metadata or {}),
            )

            if (
                self.case_manager is not None
                and self.policy.auto_correlate_cases
                and risk_level in {UnifiedRiskLevel.HIGH, UnifiedRiskLevel.CRITICAL}
            ):
                case_id, created = self.correlator.correlate(
                    case_manager=self.case_manager,
                    profile=profile,
                    assessment=assessment,
                )
                assessment.correlated_case_id = case_id
                assessment.case_created = created

            profile.last_assessment_id = assessment.assessment_id
            profile.last_updated = now.isoformat()
            self.store.save_assessment(assessment)
            self._emit_events(assessment)
            return assessment

    def _emit_events(self, assessment: UnifiedRiskAssessment) -> None:
        if self.event_bus is None:
            return
        base_payload = {
            "assessment_id": assessment.assessment_id,
            "entity_id": assessment.entity_id,
            "composite_score": assessment.composite_score,
            "risk_level": assessment.risk_level.value,
            "action": assessment.action.value,
        }
        self.event_bus.publish(
            FraudDomainEvent(
                event_type=EventType.ENTITY_RISK_ORCHESTRATED,
                tenant_id=assessment.tenant_id,
                payload=base_payload,
            )
        )
        if assessment.patterns_detected:
            self.event_bus.publish(
                FraudDomainEvent(
                    event_type=EventType.CROSS_SIGNAL_PATTERN_DETECTED,
                    tenant_id=assessment.tenant_id,
                    payload={
                        **base_payload,
                        "patterns": [p.value for p in assessment.patterns_detected],
                    },
                )
            )
        if assessment.correlated_case_id:
            self.event_bus.publish(
                FraudDomainEvent(
                    event_type=EventType.UNIFIED_CASE_CORRELATED,
                    tenant_id=assessment.tenant_id,
                    payload={
                        **base_payload,
                        "case_id": assessment.correlated_case_id,
                        "case_created": assessment.case_created,
                    },
                )
            )
        if assessment.risk_level in {UnifiedRiskLevel.HIGH, UnifiedRiskLevel.CRITICAL}:
            self.event_bus.publish(
                FraudDomainEvent(
                    event_type=EventType.ENTITY_ESCALATION_TRIGGERED,
                    tenant_id=assessment.tenant_id,
                    payload=base_payload,
                )
            )
