from __future__ import annotations

import hashlib
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from fraud_detection.model import FraudModel


class TrafficSplitStrategy(StrEnum):
    """Supported traffic routing strategies between Champion and Challenger models."""

    CHAMPION_ONLY = "champion_only"
    CHALLENGER_ONLY = "challenger_only"
    HASH = "hash"
    PERCENTAGE = "percentage"
    SHADOW = "shadow"
    CANARY = "canary"


class ModelRole(StrEnum):
    """Serving role assigned to a model candidate."""

    CHAMPION = "champion"
    CHALLENGER = "challenger"


class CanaryStatus(StrEnum):
    """Operational health state of active canary deployment."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ROLLED_BACK = "rolled_back"


class RoutingError(ValueError):
    """Base exception for routing configuration and evaluation errors."""


@dataclass(frozen=True)
class CanaryConfig:
    """Governance thresholds and automated rollback safeguards for canary routing."""

    max_discrepancy_rate: float = 0.15
    max_probability_divergence: float = 0.25
    min_evaluations: int = 10
    auto_rollback: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.max_discrepancy_rate <= 1.0):
            raise RoutingError(
                f"max_discrepancy_rate must be between 0.0 and 1.0, got {self.max_discrepancy_rate}"
            )
        if not (0.0 <= self.max_probability_divergence <= 1.0):
            raise RoutingError(
                "max_probability_divergence must be between 0.0 and 1.0, "
                f"got {self.max_probability_divergence}"
            )
        if self.min_evaluations < 1:
            raise RoutingError(
                f"min_evaluations must be at least 1, got {self.min_evaluations}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration parameters to dictionary."""
        return {
            "max_discrepancy_rate": round(self.max_discrepancy_rate, 4),
            "max_probability_divergence": round(self.max_probability_divergence, 4),
            "min_evaluations": self.min_evaluations,
            "auto_rollback": self.auto_rollback,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CanaryConfig:
        """Construct from dictionary payload."""
        return cls(
            max_discrepancy_rate=float(payload.get("max_discrepancy_rate", 0.15)),
            max_probability_divergence=float(payload.get("max_probability_divergence", 0.25)),
            min_evaluations=int(payload.get("min_evaluations", 10)),
            auto_rollback=bool(payload.get("auto_rollback", True)),
        )


@dataclass(frozen=True)
class RoutingPolicy:
    """Configurable routing specification governing model traffic partitioning."""

    strategy: TrafficSplitStrategy = TrafficSplitStrategy.CHAMPION_ONLY
    challenger_weight: float = 0.0
    entity_key: str = "card_id"
    shadow_challenger: bool = False
    canary_config: CanaryConfig = field(default_factory=CanaryConfig)
    salt: str = "fraud_routing_v1"

    def __post_init__(self) -> None:
        raw_strategy: Any = self.strategy
        if not isinstance(raw_strategy, TrafficSplitStrategy):
            try:
                object.__setattr__(self, "strategy", TrafficSplitStrategy(str(raw_strategy)))
            except ValueError as exc:
                valid = [s.value for s in TrafficSplitStrategy]
                raise RoutingError(
                    f"Unsupported routing strategy {raw_strategy!r}. Valid strategies: {valid}"
                ) from exc

        if not (0.0 <= self.challenger_weight <= 1.0):
            raise RoutingError(
                f"challenger_weight must be in [0.0, 1.0], got {self.challenger_weight}"
            )

        if not isinstance(self.entity_key, str) or not self.entity_key.strip():
            raise RoutingError("entity_key must be a non-empty string.")

        if not isinstance(self.salt, str) or not self.salt:
            raise RoutingError("salt must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize policy to dictionary."""
        return {
            "strategy": self.strategy.value,
            "challenger_weight": round(self.challenger_weight, 4),
            "entity_key": self.entity_key,
            "shadow_challenger": self.shadow_challenger,
            "canary_config": self.canary_config.to_dict(),
            "salt": self.salt,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RoutingPolicy:
        """Construct from dictionary payload."""
        canary_raw = payload.get("canary_config")
        canary_cfg = (
            CanaryConfig.from_dict(canary_raw)
            if isinstance(canary_raw, Mapping)
            else CanaryConfig()
        )
        return cls(
            strategy=TrafficSplitStrategy(
                str(payload.get("strategy", TrafficSplitStrategy.CHAMPION_ONLY.value))
            ),
            challenger_weight=float(payload.get("challenger_weight", 0.0)),
            entity_key=str(payload.get("entity_key", "card_id")),
            shadow_challenger=bool(payload.get("shadow_challenger", False)),
            canary_config=canary_cfg,
            salt=str(payload.get("salt", "fraud_routing_v1")),
        )


@dataclass(frozen=True)
class RouteDecision:
    """Model routing outcome for an evaluated transaction."""

    role: ModelRole
    model_version: str
    reason: str
    is_shadow_candidate: bool = False
    entity_key_value: str | None = None
    hash_bucket: float | None = None


@dataclass(frozen=True)
class DivergenceReport:
    """Comparative diagnostic report quantifying divergence between models."""

    total_samples: int
    discrepancies: int
    flip_rate: float
    mean_probability_divergence: float
    max_probability_divergence: float
    champion_flagged_rate: float
    challenger_flagged_rate: float
    confusion_matrix: dict[str, int]
    canary_status: CanaryStatus
    safeguard_tripped: bool
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize divergence report to dictionary."""
        return {
            "total_samples": self.total_samples,
            "discrepancies": self.discrepancies,
            "flip_rate": round(self.flip_rate, 4),
            "mean_probability_divergence": round(self.mean_probability_divergence, 4),
            "max_probability_divergence": round(self.max_probability_divergence, 4),
            "champion_flagged_rate": round(self.champion_flagged_rate, 4),
            "challenger_flagged_rate": round(self.challenger_flagged_rate, 4),
            "confusion_matrix": dict(self.confusion_matrix),
            "canary_status": self.canary_status.value,
            "safeguard_tripped": self.safeguard_tripped,
            "recommendation": self.recommendation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DivergenceReport:
        """Construct from dictionary payload."""
        return cls(
            total_samples=int(payload["total_samples"]),
            discrepancies=int(payload["discrepancies"]),
            flip_rate=float(payload["flip_rate"]),
            mean_probability_divergence=float(payload["mean_probability_divergence"]),
            max_probability_divergence=float(payload["max_probability_divergence"]),
            champion_flagged_rate=float(payload["champion_flagged_rate"]),
            challenger_flagged_rate=float(payload["challenger_flagged_rate"]),
            confusion_matrix=dict(payload.get("confusion_matrix", {})),
            canary_status=CanaryStatus(
                str(payload.get("canary_status", CanaryStatus.HEALTHY.value))
            ),
            safeguard_tripped=bool(payload.get("safeguard_tripped", False)),
            recommendation=str(payload.get("recommendation", "UNKNOWN")),
        )


class RoutingMetricsTracker:
    """Thread-safe stateful telemetry accumulator tracking routing and divergence statistics."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.champion_requests: int = 0
        self.challenger_requests: int = 0
        self.shadow_evaluations: int = 0
        self.discrepancy_count: int = 0
        self.total_probability_diff: float = 0.0
        self.max_probability_diff: float = 0.0
        self.canary_status: CanaryStatus = CanaryStatus.HEALTHY
        self.rollback_reason: str | None = None

    def record_route(self, role: ModelRole, count: int = 1) -> None:
        """Record model routing executions."""
        with self._lock:
            if role == ModelRole.CHAMPION:
                self.champion_requests += count
            else:
                self.challenger_requests += count

    def record_comparison(
        self,
        champion_prob: float,
        champion_dec: bool,
        challenger_prob: float,
        challenger_dec: bool,
        canary_config: CanaryConfig | None = None,
    ) -> bool:
        """Record a single Champion vs Challenger inference comparison.

        Returns True if a decision flip occurred (primary != challenger).
        """
        prob_diff = abs(float(champion_prob) - float(challenger_prob))
        has_discrepancy = bool(champion_dec) != bool(challenger_dec)

        with self._lock:
            self.shadow_evaluations += 1
            if has_discrepancy:
                self.discrepancy_count += 1
            self.total_probability_diff += prob_diff
            if prob_diff > self.max_probability_diff:
                self.max_probability_diff = prob_diff

            if canary_config is not None:
                self._evaluate_safeguard_locked(canary_config)

        return has_discrepancy

    def record_batch_comparison(
        self,
        champion_probs: Sequence[float],
        champion_decs: Sequence[bool],
        challenger_probs: Sequence[float],
        challenger_decs: Sequence[bool],
        canary_config: CanaryConfig | None = None,
    ) -> int:
        """Record multiple aligned Champion vs Challenger inference comparisons.

        Returns the number of decision discrepancies found in the batch.
        """
        if len(champion_probs) != len(challenger_probs):
            raise RoutingError("Probabilities array lengths must match for comparison.")

        batch_discrepancies = 0
        batch_prob_diff = 0.0
        batch_max_diff = 0.0

        for c_prob, c_dec, ch_prob, ch_dec in zip(
            champion_probs, champion_decs, challenger_probs, challenger_decs, strict=True
        ):
            diff = abs(float(c_prob) - float(ch_prob))
            batch_prob_diff += diff
            if diff > batch_max_diff:
                batch_max_diff = diff
            if bool(c_dec) != bool(ch_dec):
                batch_discrepancies += 1

        with self._lock:
            self.shadow_evaluations += len(champion_probs)
            self.discrepancy_count += batch_discrepancies
            self.total_probability_diff += batch_prob_diff
            if batch_max_diff > self.max_probability_diff:
                self.max_probability_diff = batch_max_diff

            if canary_config is not None:
                self._evaluate_safeguard_locked(canary_config)

        return batch_discrepancies

    def _evaluate_safeguard_locked(self, canary_config: CanaryConfig) -> None:
        """Evaluate canary safeguard conditions under the acquired lock."""
        if self.canary_status == CanaryStatus.ROLLED_BACK:
            return

        if self.shadow_evaluations < canary_config.min_evaluations:
            return

        flip_rate = self.discrepancy_count / max(1, self.shadow_evaluations)
        mean_divergence = self.total_probability_diff / max(1, self.shadow_evaluations)

        discrepancy_breached = flip_rate > canary_config.max_discrepancy_rate
        divergence_breached = mean_divergence > canary_config.max_probability_divergence

        if discrepancy_breached or divergence_breached:
            reasons: list[str] = []
            if discrepancy_breached:
                reasons.append(
                    f"flip_rate {flip_rate:.4f} > max {canary_config.max_discrepancy_rate:.4f}"
                )
            if divergence_breached:
                reasons.append(
                    f"mean_divergence {mean_divergence:.4f} > "
                    f"max {canary_config.max_probability_divergence:.4f}"
                )
            reason_str = "; ".join(reasons)

            if canary_config.auto_rollback:
                self.canary_status = CanaryStatus.ROLLED_BACK
                self.rollback_reason = f"Canary safeguard tripped: {reason_str}"
            else:
                self.canary_status = CanaryStatus.DEGRADED
                self.rollback_reason = f"Canary divergence warning: {reason_str}"

    def trigger_rollback(self, reason: str = "Manual operational rollback") -> None:
        """Force manual rollback of canary routing to Champion."""
        with self._lock:
            self.canary_status = CanaryStatus.ROLLED_BACK
            self.rollback_reason = reason

    def reset_canary(self) -> None:
        """Reset canary metrics and status back to HEALTHY."""
        with self._lock:
            self.canary_status = CanaryStatus.HEALTHY
            self.rollback_reason = None
            self.discrepancy_count = 0
            self.shadow_evaluations = 0
            self.total_probability_diff = 0.0
            self.max_probability_diff = 0.0

    def get_summary(self) -> dict[str, Any]:
        """Return a snapshot of accumulated routing metrics and canary status."""
        with self._lock:
            total_routed = self.champion_requests + self.challenger_requests
            evals = self.shadow_evaluations
            flip_rate = self.discrepancy_count / evals if evals > 0 else 0.0
            mean_div = self.total_probability_diff / evals if evals > 0 else 0.0

            return {
                "total_routed": total_routed,
                "champion_requests": self.champion_requests,
                "challenger_requests": self.challenger_requests,
                "shadow_evaluations": self.shadow_evaluations,
                "discrepancy_count": self.discrepancy_count,
                "flip_rate": round(flip_rate, 4),
                "mean_probability_divergence": round(mean_div, 4),
                "max_probability_divergence": round(self.max_probability_diff, 4),
                "canary_status": self.canary_status.value,
                "rollback_reason": self.rollback_reason,
            }


def compute_entity_hash_bucket(entity_value: str, salt: str = "fraud_routing_v1") -> float:
    """Compute a deterministic hash bucket in [0.0, 1.0) for a given entity key.

    Guarantees stable stickiness without external session state stores.
    """
    key_bytes = f"{salt}:{entity_value}".encode()
    digest = hashlib.sha256(key_bytes).hexdigest()
    # 32-bit unsigned integer normalization
    bucket_int = int(digest[:8], 16)
    return float(bucket_int / 0xFFFFFFFF)


def resolve_entity_key_value(
    record: Mapping[str, Any], entity_key: str = "card_id"
) -> tuple[str, bool]:
    """Extract named entity key value or compute a deterministic payload signature.

    Returns (key_value, is_explicit_key).
    """
    if entity_key in record:
        val = record[entity_key]
        if val is not None:
            str_val = str(val).strip()
            if str_val:
                return str_val, True

    # Fallback: compute deterministic digest of sorted record keys & values
    items_to_hash: list[str] = []
    for k in sorted(record.keys()):
        v = record[k]
        if v is not None and not isinstance(v, (dict, list)):
            items_to_hash.append(f"{k}={v}")

    fallback_str = "|".join(items_to_hash) if items_to_hash else "empty_record"
    return fallback_str, False


class ChampionChallengerRouter:
    """Multi-model traffic routing coordinator managing Champion and Challenger topologies."""

    def __init__(
        self,
        champion_model: FraudModel,
        challenger_model: FraudModel | None = None,
        policy: RoutingPolicy | None = None,
        metrics_tracker: RoutingMetricsTracker | None = None,
    ) -> None:
        self.champion_model = champion_model
        self.challenger_model = challenger_model
        self.policy = policy if policy is not None else RoutingPolicy()
        self.metrics = metrics_tracker if metrics_tracker is not None else RoutingMetricsTracker()

    @property
    def champion_version(self) -> str:
        """Version identifier of active Champion model."""
        return str(
            self.champion_model.metadata.get(
                "version",
                self.champion_model.metadata.get("dataset_fingerprint", "champion")[:12],
            )
        )

    @property
    def challenger_version(self) -> str | None:
        """Version identifier of candidate Challenger model, if configured."""
        if self.challenger_model is None:
            return None
        return str(
            self.challenger_model.metadata.get(
                "version",
                self.challenger_model.metadata.get("dataset_fingerprint", "challenger")[:12],
            )
        )

    def route_record(self, record: Mapping[str, Any]) -> RouteDecision:
        """Determine routing destination for a single transaction record."""
        # 1. Without challenger model, all traffic routes exclusively to Champion
        if self.challenger_model is None:
            return RouteDecision(
                role=ModelRole.CHAMPION,
                model_version=self.champion_version,
                reason="no_challenger_configured",
                is_shadow_candidate=False,
            )

        strategy = self.policy.strategy

        # 2. Strategy: CHAMPION_ONLY
        if strategy == TrafficSplitStrategy.CHAMPION_ONLY:
            return RouteDecision(
                role=ModelRole.CHAMPION,
                model_version=self.champion_version,
                reason="champion_only_policy",
                is_shadow_candidate=self.policy.shadow_challenger,
            )

        # 3. Strategy: CHALLENGER_ONLY
        if strategy == TrafficSplitStrategy.CHALLENGER_ONLY:
            return RouteDecision(
                role=ModelRole.CHALLENGER,
                model_version=self.challenger_version or "challenger",
                reason="challenger_only_policy",
                is_shadow_candidate=False,
            )

        # 4. Strategy: SHADOW
        if strategy == TrafficSplitStrategy.SHADOW:
            return RouteDecision(
                role=ModelRole.CHAMPION,
                model_version=self.champion_version,
                reason="shadow_mirror_policy",
                is_shadow_candidate=True,
            )

        # 5. Check Canary Safeguard Status
        if (
            strategy == TrafficSplitStrategy.CANARY
            and self.metrics.canary_status == CanaryStatus.ROLLED_BACK
        ):
            return RouteDecision(
                role=ModelRole.CHAMPION,
                model_version=self.champion_version,
                reason="canary_rolled_back",
                is_shadow_candidate=False,
            )

        # 6. Strategy: PERCENTAGE (Random weighted split)
        if strategy == TrafficSplitStrategy.PERCENTAGE:
            rand_val = secrets.SystemRandom().random()
            if rand_val < self.policy.challenger_weight:
                return RouteDecision(
                    role=ModelRole.CHALLENGER,
                    model_version=self.challenger_version or "challenger",
                    reason="percentage_split",
                    is_shadow_candidate=False,
                    hash_bucket=rand_val,
                )
            return RouteDecision(
                role=ModelRole.CHAMPION,
                model_version=self.champion_version,
                reason="percentage_split",
                is_shadow_candidate=self.policy.shadow_challenger,
                hash_bucket=rand_val,
            )

        # 7. Strategy: HASH (or default CANARY traffic split)
        entity_val, _ = resolve_entity_key_value(record, self.policy.entity_key)
        bucket = compute_entity_hash_bucket(entity_val, self.policy.salt)

        if bucket < self.policy.challenger_weight:
            return RouteDecision(
                role=ModelRole.CHALLENGER,
                model_version=self.challenger_version or "challenger",
                reason=f"{strategy.value}_hash_split",
                is_shadow_candidate=False,
                entity_key_value=entity_val,
                hash_bucket=bucket,
            )

        return RouteDecision(
            role=ModelRole.CHAMPION,
            model_version=self.champion_version,
            reason=f"{strategy.value}_hash_split",
            is_shadow_candidate=self.policy.shadow_challenger,
            entity_key_value=entity_val,
            hash_bucket=bucket,
        )

    def route_batch(self, records: Sequence[Mapping[str, Any]]) -> list[RouteDecision]:
        """Determine routing destinations for a sequence of records."""
        return [self.route_record(r) for r in records]

    def evaluate_divergence(
        self,
        champion_probabilities: np.ndarray,
        challenger_probabilities: np.ndarray,
        champion_threshold: float,
        challenger_threshold: float,
        canary_config: CanaryConfig | None = None,
    ) -> DivergenceReport:
        """Evaluate divergence between Champion and Challenger probability distributions."""
        c_probs = np.asarray(champion_probabilities, dtype=float).ravel()
        ch_probs = np.asarray(challenger_probabilities, dtype=float).ravel()

        if c_probs.size == 0 or ch_probs.size == 0:
            raise RoutingError("Probability arrays cannot be empty for divergence evaluation.")
        if c_probs.shape != ch_probs.shape:
            raise RoutingError(
                f"Shape mismatch: champion {c_probs.shape} != challenger {ch_probs.shape}"
            )

        total_samples = int(c_probs.size)
        c_decisions = c_probs >= champion_threshold
        ch_decisions = ch_probs >= challenger_threshold

        discrepancies = int(np.sum(c_decisions != ch_decisions))
        flip_rate = float(discrepancies / total_samples)

        prob_diffs = np.abs(c_probs - ch_probs)
        mean_div = float(np.mean(prob_diffs))
        max_div = float(np.max(prob_diffs))

        c_flagged_rate = float(np.mean(c_decisions))
        ch_flagged_rate = float(np.mean(ch_decisions))

        # Confusion matrix between models
        both_deny = int(np.sum(c_decisions & ch_decisions))
        c_deny_ch_allow = int(np.sum(c_decisions & ~ch_decisions))
        c_allow_ch_deny = int(np.sum(~c_decisions & ch_decisions))
        both_allow = int(np.sum(~c_decisions & ~ch_decisions))

        confusion = {
            "champion_deny_challenger_deny": both_deny,
            "champion_deny_challenger_allow": c_deny_ch_allow,
            "champion_allow_challenger_deny": c_allow_ch_deny,
            "champion_allow_challenger_allow": both_allow,
        }

        cfg = canary_config if canary_config is not None else self.policy.canary_config
        safeguard_tripped = False
        status = CanaryStatus.HEALTHY

        if total_samples >= cfg.min_evaluations and (
            flip_rate > cfg.max_discrepancy_rate or mean_div > cfg.max_probability_divergence
        ):
            safeguard_tripped = True
            status = CanaryStatus.ROLLED_BACK if cfg.auto_rollback else CanaryStatus.DEGRADED

        if safeguard_tripped:
            recommendation = (
                "ROLLBACK_RECOMMENDED"
                if status == CanaryStatus.ROLLED_BACK
                else "CANARY_WARN_HIGH_DIVERGENCE"
            )
        elif flip_rate <= 0.05 and mean_div <= 0.10:
            recommendation = "PROCEED_ROLLOUT"
        else:
            recommendation = "CANARY_MONITOR"

        return DivergenceReport(
            total_samples=total_samples,
            discrepancies=discrepancies,
            flip_rate=flip_rate,
            mean_probability_divergence=mean_div,
            max_probability_divergence=max_div,
            champion_flagged_rate=c_flagged_rate,
            challenger_flagged_rate=ch_flagged_rate,
            confusion_matrix=confusion,
            canary_status=status,
            safeguard_tripped=safeguard_tripped,
            recommendation=recommendation,
        )


def evaluate_model_divergence(
    champion_model: FraudModel,
    challenger_model: FraudModel,
    data: pd.DataFrame,
    canary_config: CanaryConfig | None = None,
) -> DivergenceReport:
    """Evaluate batch divergence between two fitted FraudModel instances on a DataFrame."""
    if data.empty:
        raise RoutingError("Cannot evaluate model divergence on empty dataset.")

    # Align feature frames
    champion_probs = champion_model.predict_probabilities(data)
    challenger_probs = challenger_model.predict_probabilities(data)

    router = ChampionChallengerRouter(
        champion_model=champion_model,
        challenger_model=challenger_model,
        policy=RoutingPolicy(canary_config=canary_config or CanaryConfig()),
    )

    return router.evaluate_divergence(
        champion_probabilities=champion_probs,
        challenger_probabilities=challenger_probs,
        champion_threshold=champion_model.threshold,
        challenger_threshold=challenger_model.threshold,
        canary_config=canary_config,
    )
