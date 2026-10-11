"""FastAPI application for online fraud scoring."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import secrets
import time
from collections.abc import AsyncIterator, Callable, Collection
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Annotated, Any, Self, cast
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from fraud_detection import __version__
from fraud_detection.audit import (
    AuditSink,
    JsonlAuditSink,
    NullAuditSink,
    build_scoring_audit_event,
    build_shadow_scoring_audit_event,
)
from fraud_detection.evaluation import DecisionAction, TieredThresholds
from fraud_detection.explanations import ExplanationProvider
from fraud_detection.features import FeatureStoreProtocol, FileFeatureStore
from fraud_detection.model import FraudModel, ModelArtifactError, load_model
from fraud_detection.pipeline import (
    DecisionGraph,
    DecisionGraphExecutor,
    create_default_fraud_pipeline,
)
from fraud_detection.recalibration import BaseRecalibrator
from fraud_detection.routing import (
    CanaryConfig,
    CanaryStatus,
    ChampionChallengerRouter,
    ModelRole,
    RoutingMetricsTracker,
    RoutingPolicy,
    TrafficSplitStrategy,
)
from fraud_detection.rules import RulePrecedence, RuleSet
from fraud_detection.telemetry import (
    DEFAULT_OTLP_TIMEOUT_SECONDS,
    DEFAULT_SERVICE_NAME,
    OTLP_ENDPOINT_ENVIRONMENT_VARIABLE,
    OTLP_SERVICE_NAME_ENVIRONMENT_VARIABLE,
    NullTraceExporter,
    OTLPHttpTraceExporter,
    TraceContext,
    TraceExporter,
    TraceSpan,
    parse_traceparent,
)
from fraud_detection.velocity import (
    VelocityConfig,
    VelocityWindowBuffer,
    enrich_record_velocity,
)

MODEL_PATH_ENVIRONMENT_VARIABLE = "FRAUD_MODEL_PATH"
AUDIT_LOG_ENVIRONMENT_VARIABLE = "FRAUD_AUDIT_LOG_PATH"
FALLBACK_MODE_ENVIRONMENT_VARIABLE = "FRAUD_FALLBACK_MODE"
FALLBACK_SCORE_ENVIRONMENT_VARIABLE = "FRAUD_FALLBACK_SCORE"
FALLBACK_AMOUNT_THRESHOLD_ENVIRONMENT_VARIABLE = "FRAUD_FALLBACK_AMOUNT_THRESHOLD"
DEGRADED_MODE_ENVIRONMENT_VARIABLE = "FRAUD_DEGRADED_MODE"
SHADOW_MODEL_PATH_ENVIRONMENT_VARIABLE = "FRAUD_SHADOW_MODEL_PATH"
ATTESTATION_PATH_ENVIRONMENT_VARIABLE = "FRAUD_ATTESTATION_PATH"
TRUST_BUNDLE_PATH_ENVIRONMENT_VARIABLE = "FRAUD_TRUST_BUNDLE_PATH"
CIRCUIT_BREAKER_ENABLED_ENVIRONMENT_VARIABLE = "FRAUD_CIRCUIT_BREAKER_ENABLED"
CIRCUIT_BREAKER_FAILURE_THRESHOLD_ENVIRONMENT_VARIABLE = "FRAUD_CIRCUIT_BREAKER_FAILURE_THRESHOLD"
CIRCUIT_BREAKER_RECOVERY_TIMEOUT_ENVIRONMENT_VARIABLE = "FRAUD_CIRCUIT_BREAKER_RECOVERY_TIMEOUT"
CIRCUIT_BREAKER_LATENCY_BUDGET_ENVIRONMENT_VARIABLE = "FRAUD_CIRCUIT_BREAKER_LATENCY_BUDGET_MS"
OTLP_TIMEOUT_ENVIRONMENT_VARIABLE = "FRAUD_OTLP_TIMEOUT_SECONDS"
API_KEYS_ENVIRONMENT_VARIABLE = "FRAUD_API_KEYS"
RATE_LIMIT_REQUESTS_ENVIRONMENT_VARIABLE = "FRAUD_RATE_LIMIT_REQUESTS"
RATE_LIMIT_WINDOW_SECONDS_ENVIRONMENT_VARIABLE = "FRAUD_RATE_LIMIT_WINDOW_SECONDS"
MAX_CONCURRENT_SCORING_ENVIRONMENT_VARIABLE = "FRAUD_MAX_CONCURRENT_SCORING"
ENABLE_CHAOS_HEADER_ENVIRONMENT_VARIABLE = "FRAUD_ENABLE_CHAOS_HEADER"
RULES_PATH_ENVIRONMENT_VARIABLE = "FRAUD_RULES_PATH"
RULE_PRECEDENCE_ENVIRONMENT_VARIABLE = "FRAUD_RULE_PRECEDENCE"
VELOCITY_CONFIG_PATH_ENVIRONMENT_VARIABLE = "FRAUD_VELOCITY_CONFIG_PATH"
FEATURE_STORE_PATH_ENVIRONMENT_VARIABLE = "FRAUD_FEATURE_STORE_PATH"
CHALLENGER_MODEL_PATH_ENVIRONMENT_VARIABLE = "FRAUD_CHALLENGER_MODEL_PATH"
ROUTING_STRATEGY_ENVIRONMENT_VARIABLE = "FRAUD_ROUTING_STRATEGY"
CHALLENGER_WEIGHT_ENVIRONMENT_VARIABLE = "FRAUD_CHALLENGER_WEIGHT"
ROUTING_ENTITY_KEY_ENVIRONMENT_VARIABLE = "FRAUD_ROUTING_ENTITY_KEY"
CANARY_MAX_DISCREPANCY_ENVIRONMENT_VARIABLE = "FRAUD_CANARY_MAX_DISCREPANCY"
CANARY_MAX_DIVERGENCE_ENVIRONMENT_VARIABLE = "FRAUD_CANARY_MAX_DIVERGENCE"
CANARY_AUTO_ROLLBACK_ENVIRONMENT_VARIABLE = "FRAUD_CANARY_AUTO_ROLLBACK"
PIPELINE_ENABLED_ENVIRONMENT_VARIABLE = "FRAUD_PIPELINE_ENABLED"
PIPELINE_CACHE_ENABLED_ENVIRONMENT_VARIABLE = "FRAUD_PIPELINE_CACHE_ENABLED"
PIPELINE_SPECULATIVE_ENVIRONMENT_VARIABLE = "FRAUD_PIPELINE_SPECULATIVE_ENABLED"
PIPELINE_CACHE_TTL_ENVIRONMENT_VARIABLE = "FRAUD_PIPELINE_CACHE_TTL_SECONDS"
PIPELINE_CACHE_SIZE_ENVIRONMENT_VARIABLE = "FRAUD_PIPELINE_CACHE_MAX_SIZE"
REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time-Ms"
MAX_REQUEST_BODY_BYTES = 2 * 1024 * 1024
RATE_LIMIT_TRACKED_KEY_LIMIT = 10_000
SHADOW_SHUTDOWN_TIMEOUT_SECONDS = 5.0
OPERATIONAL_PATHS = frozenset(
    {
        "/health",
        "/metrics",
        "/live",
        "/ready",
        "/v1/routing/status",
        "/v1/routing/rollback",
        "/v1/routing/reset",
        "/v1/features/stats",
        "/v1/pipeline/topology",
        "/v1/pipeline/cache/stats",
        "/v1/pipeline/cache/clear",
    }
)
UNMATCHED_ROUTE_LABEL = "unmatched"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
logger = logging.getLogger(__name__)

TransactionValue = Annotated[float, Field(strict=True, allow_inf_nan=False)]


@dataclass(frozen=True)
class _RateLimitDecision:
    """Outcome of one fixed-window rate-limit check."""

    allowed: bool
    remaining: int
    retry_after_seconds: int


@dataclass(frozen=True)
class _OperationalStatus:
    """Fallback, degraded, circuit-breaker, shadow, and rules detail for operational endpoints."""

    fallback_mode: str
    degraded_mode: bool
    circuit_breaker: dict[str, Any] | None
    shadow_model_version: str | None
    rules_count: int | None = None
    rule_precedence: str | None = None
    velocity_enabled: bool | None = None
    velocity_entities_count: int | None = None
    recalibration_enabled: bool | None = None
    recalibration_method: str | None = None
    routing_strategy: str | None = None
    challenger_model_version: str | None = None
    canary_status: str | None = None
    features_enabled: bool | None = None
    features_views_count: int | None = None
    features_entities_count: int | None = None
    pipeline_enabled: bool | None = None
    pipeline_nodes_count: int | None = None

    def response_fields(self) -> dict[str, Any]:
        """Return the response fields shared by `/health` and `/ready`."""
        fields: dict[str, Any] = {
            "degraded_mode": True if self.degraded_mode else None,
            "fallback_mode": self.fallback_mode if self.fallback_mode != "raise" else None,
            "circuit_breaker": self.circuit_breaker,
            "shadow_model_version": self.shadow_model_version,
        }
        if self.rules_count is not None:
            fields["rules_count"] = self.rules_count
        if self.rule_precedence is not None:
            fields["rule_precedence"] = self.rule_precedence
        if self.velocity_enabled is not None:
            fields["velocity_enabled"] = self.velocity_enabled
        if self.velocity_entities_count is not None:
            fields["velocity_entities_count"] = self.velocity_entities_count
        if self.recalibration_enabled is not None:
            fields["recalibration_enabled"] = self.recalibration_enabled
        if self.recalibration_method is not None:
            fields["recalibration_method"] = self.recalibration_method
        if self.routing_strategy is not None:
            fields["routing_strategy"] = self.routing_strategy
        if self.challenger_model_version is not None:
            fields["challenger_model_version"] = self.challenger_model_version
        if self.canary_status is not None:
            fields["canary_status"] = self.canary_status
        if self.features_enabled is not None:
            fields["features_enabled"] = self.features_enabled
        if self.features_views_count is not None:
            fields["features_views_count"] = self.features_views_count
        if self.features_entities_count is not None:
            fields["features_entities_count"] = self.features_entities_count
        if self.pipeline_enabled is not None:
            fields["pipeline_enabled"] = self.pipeline_enabled
        if self.pipeline_nodes_count is not None:
            fields["pipeline_nodes_count"] = self.pipeline_nodes_count
        return fields


class _ScoringOverloadedError(Exception):
    """Raised internally when the scoring concurrency cap sheds a request."""


class _RateLimiter:
    """Simple in-memory rate limiter using fixed window."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}
        self._last_eviction = 0.0

    @property
    def tracked_clients(self) -> int:
        """Number of clients currently tracked in the window."""
        return len(self._requests)

    def _evict_idle_keys(self, window_start: float) -> None:
        """Drop clients with no request inside the current window.

        Without this the in-memory window grows with every client address the process
        ever sees, which is unbounded for a long-lived server.
        """
        stale = [
            key
            for key, timestamps in self._requests.items()
            if not timestamps or timestamps[-1] <= window_start
        ]
        for key in stale:
            del self._requests[key]

    def check(self, key: str) -> _RateLimitDecision:
        """Record one request and report whether it fits the window."""
        now = time.time()
        window_start = now - self.window_seconds
        # Swept at most once per window so the scan cannot dominate request latency.
        if (
            len(self._requests) >= RATE_LIMIT_TRACKED_KEY_LIMIT
            and now - self._last_eviction >= self.window_seconds
        ):
            self._evict_idle_keys(window_start)
            self._last_eviction = now
        timestamps = [ts for ts in self._requests.get(key, []) if ts > window_start]
        if len(timestamps) >= self.max_requests:
            self._requests[key] = timestamps
            retry_after = max(0, math.ceil(timestamps[0] + self.window_seconds - now))
            return _RateLimitDecision(allowed=False, remaining=0, retry_after_seconds=retry_after)
        timestamps.append(now)
        self._requests[key] = timestamps
        return _RateLimitDecision(
            allowed=True,
            remaining=self.max_requests - len(timestamps),
            retry_after_seconds=0,
        )


def _resolve_api_keys(api_keys: list[str] | None) -> list[str] | None:
    """Resolve API keys from an explicit argument or the environment."""
    if api_keys is not None:
        return list(api_keys) or None
    raw = os.getenv(API_KEYS_ENVIRONMENT_VARIABLE, "")
    keys = [part.strip() for part in raw.split(",") if part.strip()]
    return keys or None


def _api_key_is_valid(candidate: str | None, accepted: Collection[str]) -> bool:
    """Check a candidate key against every accepted key in constant time.

    Every configured key is compared even after a match so the loop duration
    does not depend on which key (if any) matched.
    """
    if not candidate:
        return False
    candidate_bytes = candidate.encode()
    matched = False
    for accepted_key in accepted:
        if secrets.compare_digest(candidate_bytes, accepted_key.encode()):
            matched = True
    return matched


def _resolve_rate_limit_requests(rate_limit_requests: int) -> int:
    """Resolve the per-window request cap from an explicit argument or the environment."""
    if rate_limit_requests > 0:
        return rate_limit_requests
    raw = os.getenv(RATE_LIMIT_REQUESTS_ENVIRONMENT_VARIABLE, "").strip()
    if not raw:
        return 0
    try:
        resolved = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{RATE_LIMIT_REQUESTS_ENVIRONMENT_VARIABLE} must be an integer, got {raw!r}."
        ) from exc
    if resolved < 0:
        raise ValueError(
            f"{RATE_LIMIT_REQUESTS_ENVIRONMENT_VARIABLE} must be >= 0, got {resolved}."
        )
    return resolved


def _resolve_rate_limit_window_seconds(rate_limit_window_seconds: float) -> float:
    """Resolve the rate-limit window from an explicit non-default argument or the environment."""
    if rate_limit_window_seconds != 60.0:
        return rate_limit_window_seconds
    raw = os.getenv(RATE_LIMIT_WINDOW_SECONDS_ENVIRONMENT_VARIABLE, "").strip()
    if not raw:
        return rate_limit_window_seconds
    try:
        resolved = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{RATE_LIMIT_WINDOW_SECONDS_ENVIRONMENT_VARIABLE} must be a number, got {raw!r}."
        ) from exc
    if resolved <= 0:
        raise ValueError(
            f"{RATE_LIMIT_WINDOW_SECONDS_ENVIRONMENT_VARIABLE} must be > 0, got {resolved}."
        )
    return resolved


def _resolve_max_concurrent_scoring(max_concurrent_scoring: int) -> int:
    """Resolve the scoring concurrency cap from an explicit argument or the environment."""
    if max_concurrent_scoring < 0:
        raise ValueError(f"max_concurrent_scoring must be >= 0, got {max_concurrent_scoring}.")
    if max_concurrent_scoring > 0:
        return max_concurrent_scoring
    raw = os.getenv(MAX_CONCURRENT_SCORING_ENVIRONMENT_VARIABLE, "").strip()
    if not raw:
        return 0
    try:
        resolved = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{MAX_CONCURRENT_SCORING_ENVIRONMENT_VARIABLE} must be an integer, got {raw!r}."
        ) from exc
    if resolved < 0:
        raise ValueError(
            f"{MAX_CONCURRENT_SCORING_ENVIRONMENT_VARIABLE} must be >= 0, got {resolved}."
        )
    return resolved


def _resolve_positive_int(name: str, raw: str) -> int:
    """Parse a strictly positive integer environment value with a named error."""
    try:
        resolved = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc
    if resolved < 1:
        raise ValueError(f"{name} must be >= 1, got {resolved}.")
    return resolved


def _resolve_positive_float(name: str, raw: str) -> float:
    """Parse a strictly positive float environment value with a named error."""
    try:
        resolved = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}.") from exc
    if resolved <= 0:
        raise ValueError(f"{name} must be > 0, got {resolved}.")
    return resolved


def _resolve_optional_positive_float(name: str, raw: str | None) -> float | None:
    """Parse an optional strictly positive float environment value with a named error."""
    if raw is None or not raw.strip():
        return None
    return _resolve_positive_float(name, raw.strip())


class CircuitBreaker:
    """Automated operational circuit breaker protecting scoring endpoints."""

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        latency_budget_ms: float | None = None,
        half_open_success_threshold: int = 1,
        on_trip: Callable[[], None] | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1.")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive.")
        if latency_budget_ms is not None and latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive.")
        if half_open_success_threshold < 1:
            raise ValueError("half_open_success_threshold must be at least 1.")

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.latency_budget_ms = latency_budget_ms
        self.half_open_success_threshold = half_open_success_threshold
        self.on_trip = on_trip
        # Scoring runs concurrently in the threadpool, so every state transition is
        # guarded: without it a lost update can erase a failure and keep the breaker
        # closed while the model is failing.
        self._lock = RLock()

        self.state = "closed"
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_failure_time: float | None = None
        self.last_state_change = time.time()

    def allow_request(self) -> bool:
        """Determine whether an incoming request may proceed to primary model scoring."""
        with self._lock:
            return self._allow_request_locked()

    def _allow_request_locked(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            now = time.time()
            if (
                self.last_failure_time is not None
                and (now - self.last_failure_time) >= self.recovery_timeout
            ):
                self.state = "half_open"
                self.consecutive_successes = 0
                self.last_state_change = now
                logger.info("CircuitBreaker transitioned from open to half_open (probing).")
                return True
            return False
        return True

    def record_success(self) -> None:
        """Record a successful primary evaluation within latency budget."""
        with self._lock:
            if self.state == "half_open":
                self.consecutive_successes += 1
                if self.consecutive_successes >= self.half_open_success_threshold:
                    self.state = "closed"
                    self.consecutive_failures = 0
                    self.consecutive_successes = 0
                    self.last_state_change = time.time()
                    logger.info("CircuitBreaker recovered: transitioned from half_open to closed.")
            elif self.state == "closed":
                self.consecutive_failures = 0

    def record_failure(self, reason: str = "exception") -> None:
        """Record a failure (exception or latency budget breach)."""
        with self._lock:
            now = time.time()
            self.last_failure_time = now
            self.consecutive_failures += 1
            if self.state == "half_open":
                self.state = "open"
                self.last_state_change = now
                if self.on_trip is not None:
                    self.on_trip()
                logger.warning(
                    "CircuitBreaker probe failed (%s); transitioned back to open.", reason
                )
            elif self.state == "closed":
                if self.consecutive_failures >= self.failure_threshold:
                    self.state = "open"
                    self.last_state_change = now
                    if self.on_trip is not None:
                        self.on_trip()
                    logger.warning(
                        "CircuitBreaker tripped to open after %d consecutive failures (%s).",
                        self.consecutive_failures,
                        reason,
                    )

    def trip(self) -> None:
        """Force the circuit breaker to OPEN state."""
        with self._lock:
            self.state = "open"
            self.last_failure_time = time.time()
            self.last_state_change = time.time()
            if self.on_trip is not None:
                self.on_trip()

    def reset(self) -> None:
        """Reset the circuit breaker to CLOSED state."""
        with self._lock:
            self.state = "closed"
            self.consecutive_failures = 0
            self.consecutive_successes = 0
            self.last_state_change = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Return circuit breaker status dictionary."""
        with self._lock:
            return {
                "state": self.state,
                "consecutive_failures": self.consecutive_failures,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "latency_budget_ms": self.latency_budget_ms,
            }


class PredictionRequest(BaseModel):
    """Bounded batch of numeric transaction feature mappings."""

    model_config = ConfigDict(extra="forbid")

    transactions: Annotated[
        list[dict[str, TransactionValue]],
        Field(min_length=1, max_length=1_000),
    ]
    explain: bool = False
    explain_llm: bool = False
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    review_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    deny_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    include_shadow: bool = False

    @model_validator(mode="after")
    def validate_tiered_thresholds(self) -> Self:
        if (self.review_threshold is None) ^ (self.deny_threshold is None):
            raise ValueError(
                "review_threshold and deny_threshold must both be provided or both omitted."
            )
        if (
            self.review_threshold is not None
            and self.deny_threshold is not None
            and self.review_threshold > self.deny_threshold
        ):
            raise ValueError("review_threshold cannot exceed deny_threshold.")
        return self


class PredictionResult(BaseModel):
    """Fraud score and thresholded decision for one transaction."""

    fraud_probability: float
    is_fraud: bool
    decision: str = "ALLOW"
    matched_rule: str | None = None
    rule_action: str | None = None
    contributions: dict[str, float] | None = None
    explanation: str | None = None
    raw_probability: float | None = None


class PredictionResponse(BaseModel):
    """Ordered batch prediction response."""

    model_version: str
    threshold: float
    model_threshold: float
    review_threshold: float | None = None
    deny_threshold: float | None = None
    predictions: list[PredictionResult]
    fallback_applied: bool = False
    fallback_reason: str | None = None
    model_role: str | None = None
    routed_model_version: str | None = None
    shadow_predictions: list[dict[str, Any]] | None = None


class ScoreRequest(BaseModel):
    """One numeric transaction feature mapping."""

    model_config = ConfigDict(extra="forbid")

    transaction: dict[str, TransactionValue]
    explain: bool = False
    explain_llm: bool = False
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    review_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    deny_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    include_shadow: bool = False

    @model_validator(mode="after")
    def validate_tiered_thresholds(self) -> Self:
        if (self.review_threshold is None) ^ (self.deny_threshold is None):
            raise ValueError(
                "review_threshold and deny_threshold must both be provided or both omitted."
            )
        if (
            self.review_threshold is not None
            and self.deny_threshold is not None
            and self.review_threshold > self.deny_threshold
        ):
            raise ValueError("review_threshold cannot exceed deny_threshold.")
        return self


class ScoreResponse(BaseModel):
    """Single-transaction prediction response."""

    model_version: str
    threshold: float
    model_threshold: float
    review_threshold: float | None = None
    deny_threshold: float | None = None
    prediction: PredictionResult
    fallback_applied: bool = False
    fallback_reason: str | None = None
    model_role: str | None = None
    routed_model_version: str | None = None
    shadow_prediction: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """Readiness and loaded model information."""

    status: str
    service_version: str
    model_created_at: str
    feature_count: int
    threshold: float
    degraded_mode: bool | None = None
    fallback_mode: str | None = None
    circuit_breaker: dict[str, Any] | None = None
    shadow_model_version: str | None = None
    rules_count: int | None = None
    rule_precedence: str | None = None
    velocity_enabled: bool | None = None
    velocity_entities_count: int | None = None
    recalibration_enabled: bool | None = None
    recalibration_method: str | None = None
    routing_strategy: str | None = None
    challenger_model_version: str | None = None
    canary_status: str | None = None
    features_enabled: bool | None = None
    features_views_count: int | None = None
    features_entities_count: int | None = None
    pipeline_enabled: bool | None = None
    pipeline_nodes_count: int | None = None


class LivenessResponse(BaseModel):
    """Process liveness information."""

    status: str
    service_version: str


class ReadinessResponse(BaseModel):
    """Scoring-path readiness information."""

    status: str
    service_version: str
    detail: str | None = None
    model_created_at: str | None = None
    feature_count: int | None = None
    threshold: float | None = None
    degraded_mode: bool | None = None
    fallback_mode: str | None = None
    circuit_breaker: dict[str, Any] | None = None
    shadow_model_version: str | None = None
    rules_count: int | None = None
    rule_precedence: str | None = None
    velocity_enabled: bool | None = None
    velocity_entities_count: int | None = None
    recalibration_enabled: bool | None = None
    recalibration_method: str | None = None
    routing_strategy: str | None = None
    challenger_model_version: str | None = None
    canary_status: str | None = None
    pipeline_enabled: bool | None = None
    pipeline_nodes_count: int | None = None


class RoutingStatusResponse(BaseModel):
    """Multi-model traffic routing topology and canary telemetry status."""

    champion_model_version: str
    challenger_model_version: str | None = None
    routing_policy: dict[str, Any]
    metrics: dict[str, Any]
    canary_status: str
    rollback_reason: str | None = None


class PipelineScoreRequest(BaseModel):
    """Transaction input mapping for decision graph execution."""

    model_config = ConfigDict(extra="forbid")

    transaction: dict[str, TransactionValue]
    explain: bool = False


class PipelineScoreResponse(BaseModel):
    """Decision, trace, and telemetry returned by the decision graph pipeline."""

    execution_id: str
    decision: str
    is_fraud: bool
    fraud_probability: float
    raw_probability: float | None = None
    matched_rule: str | None = None
    rule_action: str | None = None
    contributions: dict[str, float] | None = None
    total_latency_ms: float
    degraded: bool
    degraded_nodes: list[str]
    execution_path: list[str]
    stage_results: dict[str, Any]


class PipelineTopologyResponse(BaseModel):
    """Topological graph description for the registered decision pipeline."""

    name: str
    node_count: int
    edge_count: int
    nodes: dict[str, Any]
    edges: list[dict[str, str]]
    execution_plan: dict[str, Any]


class PipelineCacheStatsResponse(BaseModel):
    """Runtime cache statistics for decision graph pipeline stages."""

    total_hits: int
    total_misses: int
    total_lookups: int
    hit_ratio: float
    total_evictions: int
    total_expirations: int
    total_size: int
    nodes: dict[str, Any]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def hits(self) -> int:
        return self.total_hits

    @computed_field  # type: ignore[prop-decorator]
    @property
    def misses(self) -> int:
        return self.total_misses

    @computed_field  # type: ignore[prop-decorator]
    @property
    def size(self) -> int:
        return self.total_size


class PipelineCacheClearResponse(BaseModel):
    """Result of clearing the pipeline stage execution cache."""

    status: str
    message: str | None = None


def create_app(
    *,
    model: FraudModel | None = None,
    model_path: Path | str | None = None,
    api_keys: list[str] | None = None,
    rate_limit_requests: int = 0,
    rate_limit_window_seconds: float = 60.0,
    max_concurrent_scoring: int = 0,
    audit_sink: AuditSink | None = None,
    explanation_provider: ExplanationProvider | None = None,
    fallback_mode: str | None = None,
    fallback_score: float | None = None,
    fallback_amount_threshold: float | None = None,
    degraded_mode: bool | None = None,
    enable_chaos_header: bool | None = None,
    shadow_model: FraudModel | None = None,
    shadow_model_path: Path | str | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    latency_budget_ms: float | None = None,
    circuit_breaker_failure_threshold: int = 5,
    circuit_breaker_recovery_timeout: float = 30.0,
    trace_exporter: TraceExporter | None = None,
    otlp_endpoint: str | None = None,
    otlp_service_name: str | None = None,
    otlp_timeout_seconds: float = DEFAULT_OTLP_TIMEOUT_SECONDS,
    attestation_path: Path | str | None = None,
    trust_bundle_path: Path | str | None = None,
    rules: RuleSet | None = None,
    rules_path: Path | str | None = None,
    rule_precedence: RulePrecedence | None = None,
    velocity_config: VelocityConfig | None = None,
    velocity_config_path: Path | str | None = None,
    challenger_model: FraudModel | None = None,
    challenger_model_path: Path | str | None = None,
    router: ChampionChallengerRouter | None = None,
    routing_policy: RoutingPolicy | None = None,
    routing_strategy: TrafficSplitStrategy | str | None = None,
    challenger_weight: float | None = None,
    routing_entity_key: str | None = None,
    canary_config: CanaryConfig | None = None,
    canary_max_discrepancy: float | None = None,
    canary_max_divergence: float | None = None,
    canary_auto_rollback: bool | None = None,
    feature_store: FeatureStoreProtocol | None = None,
    feature_store_path: Path | str | None = None,
    pipeline: DecisionGraph | None = None,
    enable_pipeline: bool = False,
    pipeline_executor: DecisionGraphExecutor | None = None,
    enable_pipeline_cache: bool = False,
    enable_speculative_pipeline: bool = False,
    pipeline_cache_ttl_seconds: float = 60.0,
    pipeline_cache_max_size: int = 1000,
) -> FastAPI:
    """Create an application using an injected model or a trusted artifact path.

    Args:
        model: Pre-loaded model instance.
        model_path: Path to model artifact directory.
        api_keys: Optional list of valid API keys. If provided, enables API key
            authentication via the `X-API-Key` header. When omitted, the
            comma-separated `FRAUD_API_KEYS` environment variable is used.
        rate_limit_requests: Maximum requests per window. If > 0, enables rate
            limiting per client IP. When not set, `FRAUD_RATE_LIMIT_REQUESTS`
            is used.
        rate_limit_window_seconds: Time window for rate limiting in seconds.
            When left at the default, `FRAUD_RATE_LIMIT_WINDOW_SECONDS` is used.
        max_concurrent_scoring: Maximum scoring requests allowed to run at the
            same time; 0 disables the cap. When 0, `FRAUD_MAX_CONCURRENT_SCORING`
            is used. Excess scoring requests receive HTTP 503 with `Retry-After`.
        audit_sink: Optional structured audit sink. If omitted, checks
            the `FRAUD_AUDIT_LOG_PATH` environment variable or defaults to NullAuditSink.
        explanation_provider: Optional explanation provider for natural language risk
            summaries. Defaults to offline deterministic TemplateExplanationProvider.
        fallback_mode: Fallback policy when degraded or on model failure:
            'raise' (default), 'rule' (heuristic on Amount), or 'constant'.
        fallback_score: Fixed score for 'constant' fallback mode (default: 0.5).
        fallback_amount_threshold: Amount cutoff for 'rule' fallback mode (default: 1000.0).
        degraded_mode: When True, bypasses primary model and uses fallback policy. Requires a
            fallback policy other than 'raise'.
        enable_chaos_header: When True, honor the `X-Simulate-Degraded` request header so a
            degraded-state fallback can be exercised on demand. Defaults to False, and when
            unset reads `FRAUD_ENABLE_CHAOS_HEADER`; the header is ignored unless enabled
            because any client able to reach the service could otherwise force fallback
            scoring for its own requests.
        shadow_model: Pre-loaded shadow challenger model instance.
        shadow_model_path: Path to shadow challenger model artifact directory.
        circuit_breaker: Optional pre-configured CircuitBreaker instance.
        latency_budget_ms: Max scoring latency (ms) before tripping circuit breaker. Applied
            to an injected `circuit_breaker` as well, overriding the instance's own budget
            and making the effective budget visible in the operational endpoints.
        circuit_breaker_failure_threshold: Consecutive failures before opening circuit breaker.
        circuit_breaker_recovery_timeout: Seconds before probing recovery in half-open state.
        trace_exporter: Optional injectable trace exporter; defaults to NullTraceExporter.
        otlp_endpoint: Optional OTLP/HTTP collector endpoint. Environment fallback is
            ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``.
        otlp_service_name: Service name included in exported spans.
        otlp_timeout_seconds: Collector request timeout for the optional OTLP exporter.
        attestation_path: Optional signed deployment attestation required before model load.
        trust_bundle_path: Optional rotation-aware trust bundle paired with attestation_path.
        rules: Optional pre-loaded declarative RuleSet.
        rules_path: Optional path to JSON or YAML declarative rules file.
        rule_precedence: Optional RulePrecedence strategy governing rule vs model priority.
        velocity_config: Optional pre-configured VelocityConfig specification.
        velocity_config_path: Optional path to JSON file defining VelocityConfig.
    """
    logging.basicConfig(level=logging.INFO)

    raw_mode = (
        (
            fallback_mode
            if fallback_mode is not None
            else os.getenv(FALLBACK_MODE_ENVIRONMENT_VARIABLE, "raise")
        )
        .lower()
        .strip()
    )
    if raw_mode not in {"raise", "rule", "constant"}:
        raise ValueError(
            f"Invalid fallback_mode '{raw_mode}'; must be 'raise', 'rule', or 'constant'."
        )
    resolved_fallback_mode = raw_mode

    raw_score = (
        fallback_score
        if fallback_score is not None
        else float(os.getenv(FALLBACK_SCORE_ENVIRONMENT_VARIABLE, "0.5"))
    )
    if not (0.0 <= raw_score <= 1.0):
        raise ValueError("fallback_score must be between 0.0 and 1.0.")
    resolved_fallback_score = raw_score

    resolved_fallback_amount = (
        fallback_amount_threshold
        if fallback_amount_threshold is not None
        else float(os.getenv(FALLBACK_AMOUNT_THRESHOLD_ENVIRONMENT_VARIABLE, "1000.0"))
    )

    resolved_degraded_mode = (
        degraded_mode
        if degraded_mode is not None
        else os.getenv(DEGRADED_MODE_ENVIRONMENT_VARIABLE, "false").lower() in {"1", "true", "yes"}
    )
    if resolved_degraded_mode and resolved_fallback_mode == "raise":
        # Degraded mode only means something with a fallback policy to activate; with
        # 'raise' the primary model would keep serving while /health reported degraded.
        raise ValueError(
            f"{DEGRADED_MODE_ENVIRONMENT_VARIABLE}/degraded_mode requires a fallback policy; "
            f"set fallback_mode to 'rule' or 'constant' instead of "
            f"'{resolved_fallback_mode}'."
        )

    resolved_enable_chaos_header = (
        enable_chaos_header
        if enable_chaos_header is not None
        else os.getenv(ENABLE_CHAOS_HEADER_ENVIRONMENT_VARIABLE, "false").lower()
        in {"1", "true", "yes"}
    )

    resolved_shadow_path = shadow_model_path or os.getenv(SHADOW_MODEL_PATH_ENVIRONMENT_VARIABLE)
    resolved_attestation_path = attestation_path or os.getenv(ATTESTATION_PATH_ENVIRONMENT_VARIABLE)
    resolved_trust_bundle_path = trust_bundle_path or os.getenv(
        TRUST_BUNDLE_PATH_ENVIRONMENT_VARIABLE
    )
    if (resolved_attestation_path is None) != (resolved_trust_bundle_path is None):
        raise ValueError(
            f"{ATTESTATION_PATH_ENVIRONMENT_VARIABLE} and "
            f"{TRUST_BUNDLE_PATH_ENVIRONMENT_VARIABLE} must be configured together."
        )

    cb_enabled = (
        circuit_breaker is not None
        or latency_budget_ms is not None
        or os.getenv(CIRCUIT_BREAKER_ENABLED_ENVIRONMENT_VARIABLE, "false").lower()
        in {"1", "true", "yes"}
    )
    resolved_latency_budget_ms = _resolve_optional_positive_float(
        CIRCUIT_BREAKER_LATENCY_BUDGET_ENVIRONMENT_VARIABLE,
        os.getenv(CIRCUIT_BREAKER_LATENCY_BUDGET_ENVIRONMENT_VARIABLE),
    )
    if latency_budget_ms is not None:
        resolved_latency_budget_ms = latency_budget_ms

    resolved_cb: CircuitBreaker | None = None
    if circuit_breaker is not None:
        resolved_cb = circuit_breaker
        if resolved_latency_budget_ms is not None:
            resolved_cb.latency_budget_ms = resolved_latency_budget_ms
    elif cb_enabled:
        fail_thresh = _resolve_positive_int(
            CIRCUIT_BREAKER_FAILURE_THRESHOLD_ENVIRONMENT_VARIABLE,
            os.getenv(
                CIRCUIT_BREAKER_FAILURE_THRESHOLD_ENVIRONMENT_VARIABLE,
                str(circuit_breaker_failure_threshold),
            ).strip(),
        )
        recov_timeout = _resolve_positive_float(
            CIRCUIT_BREAKER_RECOVERY_TIMEOUT_ENVIRONMENT_VARIABLE,
            os.getenv(
                CIRCUIT_BREAKER_RECOVERY_TIMEOUT_ENVIRONMENT_VARIABLE,
                str(circuit_breaker_recovery_timeout),
            ).strip(),
        )
        resolved_cb = CircuitBreaker(
            failure_threshold=fail_thresh,
            recovery_timeout=recov_timeout,
            latency_budget_ms=resolved_latency_budget_ms,
        )

    resolved_rules_path = rules_path or os.getenv(RULES_PATH_ENVIRONMENT_VARIABLE)
    resolved_rules: RuleSet | None = rules
    if resolved_rules is None and resolved_rules_path is not None:
        resolved_rules = RuleSet.load_file(resolved_rules_path)

    raw_precedence = os.getenv(RULE_PRECEDENCE_ENVIRONMENT_VARIABLE)
    if rule_precedence is not None:
        resolved_precedence = rule_precedence
    elif raw_precedence:
        resolved_precedence = RulePrecedence(raw_precedence.strip().lower())
    else:
        resolved_precedence = RulePrecedence.RULES_OVERRIDE_MODEL

    resolved_velocity_path = velocity_config_path or os.getenv(
        VELOCITY_CONFIG_PATH_ENVIRONMENT_VARIABLE
    )
    resolved_velocity_config: VelocityConfig | None = velocity_config
    if resolved_velocity_config is None and resolved_velocity_path is not None:
        resolved_velocity_config = VelocityConfig.load_file(resolved_velocity_path)

    resolved_velocity_buffer: VelocityWindowBuffer | None = (
        VelocityWindowBuffer(config=resolved_velocity_config)
        if resolved_velocity_config is not None
        else None
    )

    resolved_feature_store_path = feature_store_path or os.getenv(
        FEATURE_STORE_PATH_ENVIRONMENT_VARIABLE
    )
    resolved_feature_store: FeatureStoreProtocol | None = feature_store
    if resolved_feature_store is None and resolved_feature_store_path is not None:
        resolved_feature_store = FileFeatureStore.from_file(resolved_feature_store_path)

    raw_pipeline_env = os.getenv(PIPELINE_ENABLED_ENVIRONMENT_VARIABLE)
    resolved_pipeline_enabled = enable_pipeline or (
        raw_pipeline_env.lower() in ("true", "1", "yes")
        if raw_pipeline_env is not None
        else False
    ) or (pipeline is not None)

    raw_cache_env = os.getenv(PIPELINE_CACHE_ENABLED_ENVIRONMENT_VARIABLE)
    resolved_pipeline_cache_enabled = enable_pipeline_cache or (
        raw_cache_env.lower() in ("true", "1", "yes")
        if raw_cache_env is not None
        else False
    )

    raw_spec_env = os.getenv(PIPELINE_SPECULATIVE_ENVIRONMENT_VARIABLE)
    resolved_pipeline_speculative_enabled = enable_speculative_pipeline or (
        raw_spec_env.lower() in ("true", "1", "yes")
        if raw_spec_env is not None
        else False
    )

    raw_ttl_env = os.getenv(PIPELINE_CACHE_TTL_ENVIRONMENT_VARIABLE)
    resolved_pipeline_cache_ttl = (
        float(raw_ttl_env)
        if raw_ttl_env is not None
        else pipeline_cache_ttl_seconds
    )

    raw_size_env = os.getenv(PIPELINE_CACHE_SIZE_ENVIRONMENT_VARIABLE)
    resolved_pipeline_cache_max_size = (
        int(raw_size_env)
        if raw_size_env is not None
        else pipeline_cache_max_size
    )

    metrics_registry = CollectorRegistry()
    request_counter = Counter(
        "http_requests_total",
        "Total HTTP requests handled.",
        ["method", "path", "status_code"],
        registry=metrics_registry,
    )
    request_duration = Histogram(
        "http_request_duration_seconds",
        "HTTP request duration in seconds.",
        ["method", "path"],
        registry=metrics_registry,
    )
    prediction_counter = Counter(
        "fraud_predictions_total",
        "Total scored transactions by decision.",
        ["is_fraud"],
        registry=metrics_registry,
    )
    decision_counter = Counter(
        "fraud_decisions_total",
        "Total scored transactions by decision policy tier.",
        ["action"],
        registry=metrics_registry,
    )
    score_histogram = Histogram(
        "fraud_output_score",
        "Distribution of model prediction fraud scores.",
        buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
        registry=metrics_registry,
    )
    fallback_counter = Counter(
        "fraud_fallback_predictions_total",
        "Total fallback predictions executed due to degraded mode or runtime error.",
        ["mode", "reason"],
        registry=metrics_registry,
    )
    circuit_breaker_gauge = Gauge(
        "fraud_circuit_breaker_state",
        "State of the scoring circuit breaker (0=closed, 1=half_open, 2=open).",
        registry=metrics_registry,
    )
    circuit_breaker_tripped_counter = Counter(
        "fraud_circuit_breaker_tripped_total",
        "Total times the scoring circuit breaker has tripped to open.",
        registry=metrics_registry,
    )
    shadow_evaluations_counter = Counter(
        "fraud_shadow_evaluations_total",
        "Total transactions evaluated by shadow challenger model.",
        ["has_discrepancy"],
        registry=metrics_registry,
    )
    shadow_discrepancies_counter = Counter(
        "fraud_shadow_discrepancies_total",
        "Total shadow challenger classification discrepancies with primary model.",
        ["shadow_decision", "primary_decision"],
        registry=metrics_registry,
    )
    scoring_rejected_counter = Counter(
        "fraud_scoring_rejected_total",
        "Total requests rejected before scoring by an optional guardrail.",
        ["reason"],
        registry=metrics_registry,
    )
    rules_counter = Counter(
        "fraud_rules_triggered_total",
        "Total transactions triggering declarative decision rules.",
        ["rule_id", "action"],
        registry=metrics_registry,
    )
    velocity_enrichments_counter = Counter(
        "fraud_velocity_enrichments_total",
        "Total transactions enriched with sliding window velocity features.",
        registry=metrics_registry,
    )
    velocity_entities_gauge = Gauge(
        "fraud_velocity_entities_active",
        "Number of active unique entities tracked in the velocity window buffer.",
        registry=metrics_registry,
    )
    recalibrated_predictions_counter = Counter(
        "fraud_recalibrated_predictions_total",
        "Total predictions processed with post-hoc probability recalibration.",
        ["method"],
        registry=metrics_registry,
    )
    routing_predictions_counter = Counter(
        "fraud_routing_predictions_total",
        "Total prediction decisions routed by model role and version.",
        ["route", "model_version"],
        registry=metrics_registry,
    )
    shadow_discrepancy_total_counter = Counter(
        "fraud_shadow_discrepancy_total",
        "Total shadow challenger classification discrepancies with primary model.",
        ["shadow_decision", "primary_decision"],
        registry=metrics_registry,
    )
    canary_status_gauge = Gauge(
        "fraud_canary_status",
        "Current operational status of canary routing (1=active, 0=inactive).",
        ["status"],
        registry=metrics_registry,
    )
    feature_lookups_counter = Counter(
        "fraud_feature_lookups_total",
        "Total online feature lookups performed by the feature store.",
        ["entity_key", "status"],
        registry=metrics_registry,
    )
    pipeline_stage_duration_histogram = Histogram(
        "fraud_pipeline_stage_duration_seconds",
        "Duration of decision graph pipeline stages in seconds.",
        ["stage", "status"],
        registry=metrics_registry,
    )
    pipeline_executions_counter = Counter(
        "fraud_pipeline_executions_total",
        "Total decision graph pipeline executions.",
        ["status", "degraded"],
        registry=metrics_registry,
    )
    pipeline_cache_hits_counter = Counter(
        "fraud_pipeline_cache_hits_total",
        "Total pipeline stage cache hits.",
        ["stage"],
        registry=metrics_registry,
    )
    pipeline_cache_misses_counter = Counter(
        "fraud_pipeline_cache_misses_total",
        "Total pipeline stage cache misses.",
        ["stage"],
        registry=metrics_registry,
    )
    pipeline_speculative_executions_counter = Counter(
        "fraud_pipeline_speculative_executions_total",
        "Total speculative pipeline evaluations launched.",
        registry=metrics_registry,
    )
    pipeline_speculative_hits_counter = Counter(
        "fraud_pipeline_speculative_hits_total",
        "Total speculative pipeline evaluation hits adopted.",
        registry=metrics_registry,
    )

    if resolved_cb is not None:
        # Keep a caller-supplied callback: the metrics counter is chained onto it
        # instead of replacing it, so an injected breaker's own instrumentation runs.
        caller_on_trip = resolved_cb.on_trip
        if caller_on_trip is None:

            def on_trip() -> None:
                circuit_breaker_tripped_counter.inc()

            resolved_cb.on_trip = on_trip
        else:

            def on_trip_with_caller() -> None:
                caller_on_trip()
                circuit_breaker_tripped_counter.inc()

            resolved_cb.on_trip = on_trip_with_caller

    resolved_trace_exporter = trace_exporter
    if resolved_trace_exporter is None:
        configured_otlp_endpoint = otlp_endpoint or os.getenv(OTLP_ENDPOINT_ENVIRONMENT_VARIABLE)
        if configured_otlp_endpoint:
            configured_service_name = (
                otlp_service_name
                or os.getenv(OTLP_SERVICE_NAME_ENVIRONMENT_VARIABLE)
                or DEFAULT_SERVICE_NAME
            )
            configured_timeout = float(
                os.getenv(OTLP_TIMEOUT_ENVIRONMENT_VARIABLE, str(otlp_timeout_seconds))
            )
            resolved_trace_exporter = OTLPHttpTraceExporter(
                configured_otlp_endpoint,
                service_name=configured_service_name,
                timeout_seconds=configured_timeout,
            )
        else:
            resolved_trace_exporter = NullTraceExporter()

    resolved_rate_limit_requests = _resolve_rate_limit_requests(rate_limit_requests)
    resolved_rate_limit_window_seconds = _resolve_rate_limit_window_seconds(
        rate_limit_window_seconds
    )
    rate_limiter = (
        _RateLimiter(resolved_rate_limit_requests, resolved_rate_limit_window_seconds)
        if resolved_rate_limit_requests > 0
        else None
    )
    resolved_api_keys = _resolve_api_keys(api_keys)
    api_key_set = set(resolved_api_keys) if resolved_api_keys else None
    resolved_max_concurrent_scoring = _resolve_max_concurrent_scoring(max_concurrent_scoring)
    scoring_semaphore = (
        asyncio.Semaphore(resolved_max_concurrent_scoring)
        if resolved_max_concurrent_scoring > 0
        else None
    )

    resolved_audit_sink: AuditSink
    if audit_sink is not None:
        resolved_audit_sink = audit_sink
    else:
        configured_audit_path = os.getenv(AUDIT_LOG_ENVIRONMENT_VARIABLE)
        if configured_audit_path:
            resolved_audit_sink = JsonlAuditSink(Path(configured_audit_path))
        else:
            resolved_audit_sink = NullAuditSink()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if resolved_attestation_path is not None and resolved_trust_bundle_path is not None:
            from fraud_detection.trust import verify_admission_files

            admission_valid, admission_message = verify_admission_files(
                resolved_attestation_path,
                resolved_trust_bundle_path,
            )
            if not admission_valid:
                raise RuntimeError(f"Deployment admission failed: {admission_message}")

        loaded_model = model
        if loaded_model is None:
            configured_path = model_path or os.getenv(MODEL_PATH_ENVIRONMENT_VARIABLE)
            if configured_path is None:
                raise RuntimeError(
                    f"No model configured. Set {MODEL_PATH_ENVIRONMENT_VARIABLE} "
                    "or pass model_path to create_app()."
                )
            loaded_model = load_model(configured_path)

        resolved_challenger_path = (
            challenger_model_path
            or os.getenv(CHALLENGER_MODEL_PATH_ENVIRONMENT_VARIABLE)
            or resolved_shadow_path
        )
        loaded_challenger = challenger_model or shadow_model
        if loaded_challenger is None and resolved_challenger_path is not None:
            loaded_challenger = load_model(resolved_challenger_path)

        active_router = router
        if active_router is None and loaded_model is not None:
            if routing_policy is not None:
                active_router = ChampionChallengerRouter(
                    champion_model=loaded_model,
                    challenger_model=loaded_challenger,
                    policy=routing_policy,
                )
            else:
                strat_str = routing_strategy or os.getenv(ROUTING_STRATEGY_ENVIRONMENT_VARIABLE)
                if strat_str is not None:
                    parsed_strat = TrafficSplitStrategy(str(strat_str))
                elif (
                    loaded_challenger is not None
                    and challenger_model is None
                    and challenger_model_path is None
                    and (shadow_model is not None or resolved_shadow_path is not None)
                ):
                    parsed_strat = TrafficSplitStrategy.SHADOW
                else:
                    parsed_strat = TrafficSplitStrategy.CHAMPION_ONLY

                c_weight = challenger_weight
                if c_weight is None:
                    raw_w = os.getenv(CHALLENGER_WEIGHT_ENVIRONMENT_VARIABLE)
                    c_weight = float(raw_w) if raw_w is not None else 0.0

                e_key = str(
                    routing_entity_key
                    or os.getenv(ROUTING_ENTITY_KEY_ENVIRONMENT_VARIABLE)
                    or "card_id"
                )

                c_disc = canary_max_discrepancy
                if c_disc is None:
                    raw_d = os.getenv(CANARY_MAX_DISCREPANCY_ENVIRONMENT_VARIABLE)
                    c_disc = float(raw_d) if raw_d is not None else 0.15

                c_div = canary_max_divergence
                if c_div is None:
                    raw_div = os.getenv(CANARY_MAX_DIVERGENCE_ENVIRONMENT_VARIABLE)
                    c_div = float(raw_div) if raw_div is not None else 0.25

                c_auto = canary_auto_rollback
                if c_auto is None:
                    raw_auto = os.getenv(CANARY_AUTO_ROLLBACK_ENVIRONMENT_VARIABLE)
                    c_auto = (
                        raw_auto.lower() in ("true", "1", "yes")
                        if raw_auto is not None
                        else True
                    )

                built_canary_cfg = canary_config or CanaryConfig(
                    max_discrepancy_rate=c_disc,
                    max_probability_divergence=c_div,
                    auto_rollback=c_auto,
                )
                built_policy = RoutingPolicy(
                    strategy=parsed_strat,
                    challenger_weight=c_weight,
                    entity_key=e_key,
                    canary_config=built_canary_cfg,
                    shadow_challenger=(
                        parsed_strat in (TrafficSplitStrategy.SHADOW, TrafficSplitStrategy.CANARY)
                    ),
                )
                active_router = ChampionChallengerRouter(
                    champion_model=loaded_model,
                    challenger_model=loaded_challenger,
                    policy=built_policy,
                )

        resolved_pipeline_graph: DecisionGraph | None = pipeline
        if (
            resolved_pipeline_graph is None
            and resolved_pipeline_enabled
            and loaded_model is not None
        ):
            resolved_pipeline_graph = create_default_fraud_pipeline(
                model=loaded_model,
                rules=resolved_rules,
                rule_precedence=resolved_precedence,
                feature_store=resolved_feature_store,
                velocity_buffer=resolved_velocity_buffer,
                velocity_config=resolved_velocity_config,
                recalibrator=getattr(loaded_model, "recalibrator", None),
                enable_cache=resolved_pipeline_cache_enabled,
                cache_ttl_seconds=resolved_pipeline_cache_ttl,
                cache_max_size=resolved_pipeline_cache_max_size,
                speculative_inference=resolved_pipeline_speculative_enabled,
            )

        resolved_pipeline_exec = pipeline_executor
        if resolved_pipeline_exec is None and resolved_pipeline_graph is not None:
            resolved_pipeline_exec = DecisionGraphExecutor(
                resolved_pipeline_graph,
                enable_cache=resolved_pipeline_cache_enabled,
                enable_speculative=resolved_pipeline_speculative_enabled,
            )
        elif resolved_pipeline_exec is not None and resolved_pipeline_graph is None:
            resolved_pipeline_graph = resolved_pipeline_exec.graph

        application.state.model = loaded_model
        application.state.shadow_model = loaded_challenger
        application.state.challenger_model = loaded_challenger
        application.state.router = active_router
        application.state.circuit_breaker = resolved_cb
        application.state.audit_sink = resolved_audit_sink
        application.state.explanation_provider = explanation_provider
        application.state.fallback_mode = resolved_fallback_mode
        application.state.fallback_score = resolved_fallback_score
        application.state.fallback_amount_threshold = resolved_fallback_amount
        application.state.degraded_mode = resolved_degraded_mode
        application.state.enable_chaos_header = resolved_enable_chaos_header
        application.state.fallback_counter = fallback_counter
        application.state.shadow_evaluations_counter = shadow_evaluations_counter
        application.state.shadow_discrepancies_counter = shadow_discrepancies_counter
        application.state.circuit_breaker_tripped_counter = circuit_breaker_tripped_counter
        application.state.circuit_breaker_gauge = circuit_breaker_gauge
        application.state.trace_exporter = resolved_trace_exporter
        application.state.rules = resolved_rules
        application.state.rule_precedence = resolved_precedence
        application.state.rules_counter = rules_counter
        application.state.velocity_config = resolved_velocity_config
        application.state.velocity_buffer = resolved_velocity_buffer
        application.state.velocity_enrichments_counter = velocity_enrichments_counter
        application.state.velocity_entities_gauge = velocity_entities_gauge
        application.state.routing_predictions_counter = routing_predictions_counter
        application.state.shadow_discrepancy_total_counter = shadow_discrepancy_total_counter
        application.state.canary_status_gauge = canary_status_gauge
        application.state.feature_store = resolved_feature_store
        application.state.feature_lookups_counter = feature_lookups_counter
        application.state.pipeline_enabled = resolved_pipeline_enabled
        application.state.pipeline_graph = resolved_pipeline_graph
        application.state.pipeline_executor = resolved_pipeline_exec
        application.state.pipeline_stage_duration_histogram = pipeline_stage_duration_histogram
        application.state.pipeline_executions_counter = pipeline_executions_counter
        application.state.pipeline_cache_hits_counter = pipeline_cache_hits_counter
        application.state.pipeline_cache_misses_counter = pipeline_cache_misses_counter
        application.state.pipeline_speculative_executions_counter = (
            pipeline_speculative_executions_counter
        )
        application.state.pipeline_speculative_hits_counter = (
            pipeline_speculative_hits_counter
        )
        shadow_tasks: set[asyncio.Task[None]] = set()
        application.state.shadow_tasks = shadow_tasks
        yield
        await _drain_shadow_tasks(shadow_tasks, SHADOW_SHUTDOWN_TIMEOUT_SECONDS)
        resolved_audit_sink.close()

    application = FastAPI(
        title="Credit Card Fraud Detection API",
        summary="Low-latency scoring with a validation-tuned fraud threshold.",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @application.middleware("http")
    async def api_key_middleware(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if (
            api_key_set is not None
            and request.url.path not in OPERATIONAL_PATHS
            and not _api_key_is_valid(request.headers.get("X-API-Key"), api_key_set)
        ):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )
        return await call_next(request)

    @application.middleware("http")
    async def rate_limit_middleware(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        # Registered after the API-key middleware, so it runs outside it: unauthenticated
        # traffic has to consume the client budget, otherwise key guessing is unmetered.
        if rate_limiter is None or request.url.path in OPERATIONAL_PATHS:
            return await call_next(request)
        client_ip = request.client.host if request.client else "unknown"
        decision = rate_limiter.check(client_ip)
        if not decision.allowed:
            scoring_rejected_counter.labels(reason="rate_limited").inc()
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={
                    "Retry-After": str(decision.retry_after_seconds),
                    "X-RateLimit-Limit": str(rate_limiter.max_requests),
                    "X-RateLimit-Remaining": "0",
                },
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
        return response

    @application.middleware("http")
    async def request_context(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        incoming_trace_context = parse_traceparent(request.headers.get("traceparent"))
        server_trace_context = (
            incoming_trace_context.child()
            if incoming_trace_context is not None
            else TraceContext.new()
        )
        request.state.trace_context = server_trace_context
        supplied_request_id = request.headers.get(REQUEST_ID_HEADER)
        request_id = (
            supplied_request_id
            if supplied_request_id is not None
            and _REQUEST_ID_PATTERN.fullmatch(supplied_request_id)
            else uuid4().hex
        )
        request.state.request_id = request_id
        started_at = perf_counter()
        started_at_unix_nano = time.time_ns()

        def export_trace(status_code: int, *, error_type: str | None = None) -> None:
            attributes: dict[str, str | int | float | bool] = {
                "http.request.method": request.method,
                "http.route": request.url.path,
                "http.response.status_code": status_code,
                "fraud.request_id": request_id,
            }
            loaded_model = getattr(request.app.state, "model", None)
            if loaded_model is not None:
                attributes["fraud.model_version"] = str(
                    loaded_model.metadata.get("dataset_fingerprint", "")
                )[:12]
            if error_type is not None:
                attributes["error.type"] = error_type
            span = TraceSpan(
                name=f"HTTP {request.url.path}",
                context=server_trace_context,
                parent_span_id=(
                    incoming_trace_context.span_id if incoming_trace_context is not None else None
                ),
                start_time_unix_nano=started_at_unix_nano,
                end_time_unix_nano=time.time_ns(),
                attributes=attributes,
                status_code=status_code,
            )
            try:
                resolved_trace_exporter.export(span)
            except Exception:
                logger.exception("Failed to export request trace.")

        def stamp_response(response: Response, duration_ms: float) -> Response:
            response.headers[REQUEST_ID_HEADER] = request_id
            response.headers[PROCESS_TIME_HEADER] = f"{duration_ms:.3f}"
            response.headers["traceparent"] = server_trace_context.traceparent
            return response

        try:
            response: Response | None = await _request_body_error(request)
            if response is None:
                response = await call_next(request)
        except Exception as exc:
            duration_ms = (perf_counter() - started_at) * 1_000
            request_counter.labels(
                method=request.method,
                path=_metric_path(request),
                status_code="500",
            ).inc()
            request_duration.labels(method=request.method, path=_metric_path(request)).observe(
                duration_ms / 1_000
            )
            logger.exception(
                "request_failed method=%s path=%s duration_ms=%.3f request_id=%s",
                request.method,
                request.url.path,
                duration_ms,
                request_id,
            )
            export_trace(500, error_type=type(exc).__name__)
            # Answer here rather than re-raising: the sanitized body must still carry the
            # correlation headers, which an outer error handler would not be able to add.
            return stamp_response(
                JSONResponse(status_code=500, content={"detail": "Internal server error."}),
                duration_ms,
            )

        duration_ms = (perf_counter() - started_at) * 1_000
        request_counter.labels(
            method=request.method,
            path=_metric_path(request),
            status_code=str(response.status_code),
        ).inc()
        request_duration.labels(method=request.method, path=_metric_path(request)).observe(
            duration_ms / 1_000
        )
        stamp_response(response, duration_ms)
        export_trace(response.status_code)
        logger.info(
            "request_completed method=%s path=%s status_code=%d duration_ms=%.3f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )
        return response

    @application.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        _request: Request,
        exception: RequestValidationError,
    ) -> JSONResponse:
        sanitized_errors = [
            {
                "type": error["type"],
                "loc": error["loc"],
                "msg": error["msg"],
            }
            for error in exception.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": sanitized_errors})

    @application.exception_handler(ModelArtifactError)
    async def model_artifact_error_handler(
        _request: Request,
        exception: ModelArtifactError,
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exception)})

    @application.get(
        "/health",
        response_model=HealthResponse,
        response_model_exclude_none=True,
        responses={503: {"description": "No model is loaded."}},
        tags=["operations"],
    )
    async def health(request: Request, response: Response) -> HealthResponse | JSONResponse:
        loaded = getattr(request.app.state, "model", None)
        if loaded is None:
            # Same answer as /ready: an unloaded model is a service problem to report,
            # not an internal error for the probe to trip over.
            response.status_code = 503
            return JSONResponse(status_code=503, content={"detail": "Model not loaded."})
        status = _operational_status(request)
        return HealthResponse(
            status="degraded" if status.degraded_mode else "ready",
            service_version=__version__,
            model_created_at=str(loaded.metadata["created_at"]),
            feature_count=len(loaded.feature_names),
            threshold=loaded.threshold,
            **status.response_fields(),
        )

    @application.get(
        "/live",
        response_model=LivenessResponse,
        response_model_exclude_none=True,
        tags=["operations"],
    )
    async def live() -> LivenessResponse:
        return LivenessResponse(status="alive", service_version=__version__)

    @application.get(
        "/ready",
        response_model=ReadinessResponse,
        response_model_exclude_none=True,
        responses={
            200: {"description": "Service can accept scoring traffic."},
            503: {"description": "Service cannot accept scoring traffic."},
        },
        tags=["operations"],
    )
    async def ready(request: Request, response: Response) -> ReadinessResponse:
        loaded = getattr(request.app.state, "model", None)
        if loaded is None:
            response.status_code = 503
            return ReadinessResponse(
                status="not_ready",
                service_version=__version__,
                detail="Model not loaded.",
            )
        status = _operational_status(request)
        cannot_serve = (
            status.circuit_breaker is not None
            and status.circuit_breaker["state"] == "open"
            and status.fallback_mode == "raise"
        )
        if cannot_serve:
            response.status_code = 503
        return ReadinessResponse(
            status="not_ready" if cannot_serve else "ready",
            service_version=__version__,
            detail="Circuit breaker is open and fallback mode is raise." if cannot_serve else None,
            model_created_at=str(loaded.metadata["created_at"]),
            feature_count=len(loaded.feature_names),
            threshold=loaded.threshold,
            **status.response_fields(),
        )

    def _primary_results(
        loaded: FraudModel,
        frame: pd.DataFrame,
        probabilities: np.ndarray,
        *,
        raw_probabilities: np.ndarray | None = None,
        applied_threshold: float,
        eff_review: float,
        eff_deny: float,
        explain: bool,
        explain_llm: bool,
        rules: RuleSet | None = None,
        rule_precedence: RulePrecedence = RulePrecedence.RULES_OVERRIDE_MODEL,
    ) -> list[PredictionResult]:
        """Classify primary probabilities using tiered thresholds and build
        per-transaction results.
        """
        for prob in probabilities:
            score_histogram.observe(float(prob))

        final_decisions, matched_rule_ids, _ = loaded.predict_decisions_with_details(
            frame,
            probabilities=probabilities,
            review_threshold=eff_review,
            deny_threshold=eff_deny,
            rules=rules,
            rule_precedence=rule_precedence,
        )

        decision_actions = [str(act) for act in final_decisions]
        is_fraud_flags: list[bool] = [act == DecisionAction.DENY.value for act in decision_actions]

        local_explanations = loaded.explain_local(frame) if explain or explain_llm else None
        natural_language = (
            loaded.explain_local_natural_language(
                frame,
                probabilities,
                threshold=applied_threshold,
                contributions=local_explanations,
                decisions=decision_actions,
                provider=explanation_provider,
            )
            if explain_llm
            else None
        )
        results: list[PredictionResult] = []
        for index, (prob, is_fraud, action, matched_id) in enumerate(
            zip(probabilities, is_fraud_flags, decision_actions, matched_rule_ids, strict=True)
        ):
            results.append(
                PredictionResult(
                    fraud_probability=float(prob),
                    is_fraud=is_fraud,
                    decision=action,
                    matched_rule=matched_id,
                    rule_action=action if matched_id is not None else None,
                    contributions=(
                        local_explanations[index]
                        if explain and local_explanations is not None
                        else None
                    ),
                    explanation=(natural_language[index] if natural_language else None),
                    raw_probability=(
                        float(raw_probabilities[index]) if raw_probabilities is not None else None
                    ),
                )
            )
        fraud_sum = sum(1 for f in is_fraud_flags if f)
        prediction_counter.labels(is_fraud="true").inc(fraud_sum)
        prediction_counter.labels(is_fraud="false").inc(len(is_fraud_flags) - fraud_sum)
        for action in (
            DecisionAction.ALLOW.value,
            DecisionAction.CHALLENGE.value,
            DecisionAction.DENY.value,
        ):
            act_count = sum(1 for a in decision_actions if a == action)
            if act_count > 0:
                decision_counter.labels(action=action).inc(act_count)
        for matched_id, act in zip(matched_rule_ids, decision_actions, strict=True):
            if matched_id is not None:
                rules_counter.labels(rule_id=matched_id, action=act).inc()
        recal = getattr(loaded, "recalibrator", None)
        if isinstance(recal, BaseRecalibrator):
            recalibrated_predictions_counter.labels(method=recal.method.value).inc(len(results))
        return results

    def _require_probabilities(probabilities: Any, expected_rows: int) -> np.ndarray:
        """Return the model's probabilities, rejecting a response that cannot answer the batch.

        Raises:
            RuntimeError: The model returned no probabilities, or one row per requested
                transaction was not returned. Both are model failures, so they must reach
                the circuit breaker and the fallback policy rather than fail late.
        """
        if probabilities is None:
            raise RuntimeError("Primary model returned no probabilities.")
        if len(probabilities) != expected_rows:
            raise RuntimeError(
                f"Primary model returned probabilities for {len(probabilities)} "
                f"of {expected_rows} transactions."
            )
        return cast("np.ndarray", probabilities)

    def score_frame(
        loaded: FraudModel,
        frame: pd.DataFrame,
        *,
        explain: bool,
        explain_llm: bool,
        threshold: float | None,
        review_threshold: float | None = None,
        deny_threshold: float | None = None,
        fallback_mode: str = "raise",
        fallback_score: float = 0.5,
        fallback_amount_threshold: float = 1000.0,
        force_degraded: bool = False,
        circuit_breaker: CircuitBreaker | None = None,
        rules: RuleSet | None = None,
        rule_precedence: RulePrecedence = RulePrecedence.RULES_OVERRIDE_MODEL,
    ) -> tuple[float, list[PredictionResult], bool, str | None, float, float]:
        """Score transactions with runtime guardrails and resilient degraded-state fallback."""
        # Reject a request whose feature schema does not match the artifact before any
        # guardrail state changes: a malformed request is a client error, never a model
        # failure, so it must not record a circuit-breaker failure or be answered with a
        # fabricated fallback score.
        loaded.validate_features(frame)
        raw_th = getattr(loaded, "threshold", 0.5) if threshold is None else threshold
        applied_threshold = float(raw_th) if isinstance(raw_th, (int, float)) else 0.5

        tiered = getattr(loaded, "tiered_thresholds", None)
        if review_threshold is not None and deny_threshold is not None:
            eff_review = float(review_threshold)
            eff_deny = float(deny_threshold)
        elif isinstance(tiered, TieredThresholds):
            eff_review = float(tiered.review_threshold)
            eff_deny = float(tiered.deny_threshold)
        else:
            eff_review = applied_threshold
            eff_deny = applied_threshold

        fallback_applied = False
        fallback_reason: str | None = None
        results: list[PredictionResult] = []

        if force_degraded and fallback_mode != "raise":
            fallback_applied = True
            fallback_reason = "degraded_mode_active"
        elif circuit_breaker is not None and not circuit_breaker.allow_request():
            if fallback_mode == "raise":
                raise RuntimeError("Circuit breaker is OPEN: primary model scoring is suspended.")
            fallback_applied = True
            fallback_reason = "circuit_breaker_open"
        else:
            t0 = perf_counter()
            try:
                recal = getattr(loaded, "recalibrator", None)
                if isinstance(recal, BaseRecalibrator):
                    raw_probabilities = _require_probabilities(
                        loaded.predict_probabilities(frame, raw=True), len(frame)
                    )
                    probabilities = _require_probabilities(
                        recal.transform(raw_probabilities), len(frame)
                    )
                else:
                    raw_probabilities = None
                    probabilities = _require_probabilities(
                        loaded.predict_probabilities(frame), len(frame)
                    )
                elapsed_ms = (perf_counter() - t0) * 1000.0
                if circuit_breaker is not None:
                    if (
                        circuit_breaker.latency_budget_ms is not None
                        and elapsed_ms > circuit_breaker.latency_budget_ms
                    ):
                        circuit_breaker.record_failure(reason="latency_budget_exceeded")
                    else:
                        circuit_breaker.record_success()
                results = _primary_results(
                    loaded,
                    frame,
                    probabilities,
                    raw_probabilities=raw_probabilities,
                    applied_threshold=applied_threshold,
                    eff_review=eff_review,
                    eff_deny=eff_deny,
                    explain=explain,
                    explain_llm=explain_llm,
                    rules=rules,
                    rule_precedence=rule_precedence,
                )
            except Exception as exc:
                if circuit_breaker is not None:
                    circuit_breaker.record_failure(reason=f"model_exception: {type(exc).__name__}")
                if fallback_mode == "raise":
                    raise
                logger.warning(
                    "Primary model scoring failed; activating fallback guardrail (%s): %s",
                    fallback_mode,
                    exc,
                )
                fallback_applied = True
                fallback_reason = f"model_exception: {type(exc).__name__}"

        if not fallback_applied:
            return applied_threshold, results, False, None, eff_review, eff_deny

        if fallback_mode == "constant":
            prob_const = float(fallback_score)
            if prob_const < eff_review:
                act_const = DecisionAction.ALLOW.value
                flag_const = False
            elif prob_const >= eff_deny:
                act_const = DecisionAction.DENY.value
                flag_const = True
            else:
                act_const = DecisionAction.CHALLENGE.value
                flag_const = False
            results = [
                PredictionResult(
                    fraud_probability=prob_const,
                    is_fraud=flag_const,
                    decision=act_const,
                    contributions=None,
                    explanation="Fallback policy (constant) applied.",
                )
                for _ in range(len(frame))
            ]
        else:
            amounts = frame["Amount"] if "Amount" in frame.columns else [0.0] * len(frame)
            for amount in amounts:
                amt_val = float(amount)
                is_high = amt_val >= fallback_amount_threshold
                prob = 1.0 if is_high else 0.0
                if prob < eff_review:
                    act_rule = DecisionAction.ALLOW.value
                    flag_rule = False
                elif prob >= eff_deny:
                    act_rule = DecisionAction.DENY.value
                    flag_rule = True
                else:
                    act_rule = DecisionAction.CHALLENGE.value
                    flag_rule = False
                rule_exp = (
                    f"Fallback rule applied: Amount ({amt_val:.2f}) "
                    f"{'>=' if is_high else '<'} {fallback_amount_threshold:.2f}."
                )
                results.append(
                    PredictionResult(
                        fraud_probability=prob,
                        is_fraud=flag_rule,
                        decision=act_rule,
                        contributions=None,
                        explanation=rule_exp,
                    )
                )
        fraud_count = sum(1 for r in results if r.is_fraud)
        prediction_counter.labels(is_fraud="true").inc(fraud_count)
        prediction_counter.labels(is_fraud="false").inc(len(results) - fraud_count)
        for action in (
            DecisionAction.ALLOW.value,
            DecisionAction.CHALLENGE.value,
            DecisionAction.DENY.value,
        ):
            act_count = sum(1 for r in results if r.decision == action)
            if act_count > 0:
                decision_counter.labels(action=action).inc(act_count)
        fallback_counter.labels(mode=fallback_mode, reason=fallback_reason or "unknown").inc(
            len(results)
        )
        return applied_threshold, results, True, fallback_reason, eff_review, eff_deny

    async def _score_with_guardrails(
        request: Request,
        loaded: FraudModel,
        frame: pd.DataFrame,
        *,
        features: list[dict[str, float]],
        explain: bool,
        explain_llm: bool,
        threshold: float | None,
        review_threshold: float | None = None,
        deny_threshold: float | None = None,
        include_shadow: bool = False,
        enriched_features: list[str] | None = None,
        feature_views_applied: list[str] | None = None,
    ) -> tuple[
        float,
        list[PredictionResult],
        bool,
        str | None,
        float,
        float,
        str,
        str,
        list[dict[str, Any]] | None,
    ]:
        """Run the shared scoring pipeline: cap, inference, audit emit, and shadow scheduling.

        Raises:
            _ScoringOverloadedError: The scoring concurrency cap is already saturated.
        """
        force_degraded = bool(getattr(request.app.state, "degraded_mode", False)) or (
            bool(getattr(request.app.state, "enable_chaos_header", False))
            and request.headers.get("X-Simulate-Degraded", "").lower() == "true"
        )
        fb_mode = str(getattr(request.app.state, "fallback_mode", "raise"))
        fb_score = float(getattr(request.app.state, "fallback_score", 0.5))
        fb_amount = float(getattr(request.app.state, "fallback_amount_threshold", 1000.0))
        cb: CircuitBreaker | None = getattr(request.app.state, "circuit_breaker", None)
        active_rules: RuleSet | None = getattr(request.app.state, "rules", None)
        active_rule_precedence: RulePrecedence = getattr(
            request.app.state, "rule_precedence", RulePrecedence.RULES_OVERRIDE_MODEL
        )

        if scoring_semaphore is not None:
            if scoring_semaphore.locked():
                scoring_rejected_counter.labels(reason="concurrency_cap").inc()
                raise _ScoringOverloadedError
            await scoring_semaphore.acquire()

        router: ChampionChallengerRouter | None = getattr(request.app.state, "router", None)
        model_to_use = loaded
        model_role = "champion"
        routed_version = _model_version(loaded)
        decisions = []
        is_mixed = False
        champ_indices: list[int] = []
        chall_indices: list[int] = []

        if router is not None and router.challenger_model is not None:
            decisions = router.route_batch(features)
            r_counter: Counter | None = getattr(
                request.app.state, "routing_predictions_counter", None
            )
            for d in decisions:
                router.metrics.record_route(d.role)
                if r_counter is not None:
                    r_counter.labels(route=d.role.value, model_version=d.model_version).inc()

            chall_count = sum(1 for d in decisions if d.role == ModelRole.CHALLENGER)
            if chall_count == len(decisions):
                model_to_use = router.challenger_model
                model_role = "challenger"
                routed_version = router.challenger_version or "challenger"
            elif chall_count == 0:
                model_to_use = router.champion_model
                model_role = "champion"
                routed_version = router.champion_version
            else:
                is_mixed = True
                model_role = "mixed"
                routed_version = f"{router.champion_version}+{router.challenger_version}"
                champ_indices = [
                    i for i, d in enumerate(decisions) if d.role == ModelRole.CHAMPION
                ]
                chall_indices = [
                    i for i, d in enumerate(decisions) if d.role == ModelRole.CHALLENGER
                ]

        try:
            if not is_mixed:
                (
                    applied_threshold,
                    results,
                    fallback_applied,
                    fallback_reason,
                    eff_review,
                    eff_deny,
                ) = await run_in_threadpool(
                    score_frame,
                    model_to_use,
                    frame,
                    explain=explain,
                    explain_llm=explain_llm,
                    threshold=threshold,
                    review_threshold=review_threshold,
                    deny_threshold=deny_threshold,
                    fallback_mode=fb_mode,
                    fallback_score=fb_score,
                    fallback_amount_threshold=fb_amount,
                    force_degraded=force_degraded,
                    circuit_breaker=cb,
                    rules=active_rules,
                    rule_precedence=active_rule_precedence,
                )
            else:
                champ_res: list[PredictionResult] = []
                chall_res: list[PredictionResult] = []
                applied_threshold = loaded.threshold
                eff_review = 0.0
                eff_deny = 1.0
                fallback_applied = False
                fallback_reason = None
                if champ_indices and router is not None:
                    (
                        applied_threshold,
                        champ_res,
                        fallback_applied,
                        fallback_reason,
                        eff_review,
                        eff_deny,
                    ) = await run_in_threadpool(
                        score_frame,
                        router.champion_model,
                        frame.iloc[champ_indices],
                        explain=explain,
                        explain_llm=explain_llm,
                        threshold=threshold,
                        review_threshold=review_threshold,
                        deny_threshold=deny_threshold,
                        fallback_mode=fb_mode,
                        fallback_score=fb_score,
                        fallback_amount_threshold=fb_amount,
                        force_degraded=force_degraded,
                        circuit_breaker=cb,
                        rules=active_rules,
                        rule_precedence=active_rule_precedence,
                    )
                if chall_indices and router is not None and router.challenger_model is not None:
                    (
                        _,
                        chall_res,
                        _,
                        _,
                        _,
                        _,
                    ) = await run_in_threadpool(
                        score_frame,
                        router.challenger_model,
                        frame.iloc[chall_indices],
                        explain=explain,
                        explain_llm=explain_llm,
                        threshold=threshold,
                        review_threshold=review_threshold,
                        deny_threshold=deny_threshold,
                        fallback_mode=fb_mode,
                        fallback_score=fb_score,
                        fallback_amount_threshold=fb_amount,
                        force_degraded=force_degraded,
                        circuit_breaker=cb,
                        rules=active_rules,
                        rule_precedence=active_rule_precedence,
                    )
                merged_results: list[PredictionResult] = [None] * len(decisions)  # type: ignore[list-item]
                for idx, res in zip(champ_indices, champ_res, strict=True):
                    merged_results[idx] = res
                for idx, res in zip(chall_indices, chall_res, strict=True):
                    merged_results[idx] = res
                results = merged_results
        finally:
            if scoring_semaphore is not None:
                scoring_semaphore.release()

        sh_model: FraudModel | None = getattr(request.app.state, "shadow_model", None)
        shadow_results_sync: list[dict[str, Any]] | None = None
        if include_shadow and sh_model is not None:
            try:
                sh_probs = await run_in_threadpool(sh_model.predict_probabilities, frame)
                sh_decs = sh_probs >= sh_model.threshold
                shadow_results_sync = [
                    {
                        "shadow_probability": float(sp),
                        "shadow_decision": bool(sd),
                        "shadow_model_version": _model_version(sh_model),
                    }
                    for sp, sd in zip(sh_probs, sh_decs, strict=True)
                ]
            except Exception:
                logger.exception("Synchronous shadow scoring evaluation failed.")
                shadow_results_sync = None

        sink: AuditSink = getattr(request.app.state, "audit_sink", resolved_audit_sink)
        try:
            audit_event = build_scoring_audit_event(
                model_version=str(loaded.metadata.get("dataset_fingerprint", ""))[:12],
                dataset_fingerprint=str(loaded.metadata.get("dataset_fingerprint", "")),
                threshold=applied_threshold,
                review_threshold=eff_review,
                deny_threshold=eff_deny,
                features=features,
                fallback_applied=fallback_applied,
                fallback_reason=fallback_reason,
                predictions=[
                    {
                        "fraud_probability": r.fraud_probability,
                        "raw_probability": r.raw_probability,
                        "is_fraud": r.is_fraud,
                        "decision": r.decision,
                        "matched_rule": r.matched_rule,
                        "rule_action": r.rule_action,
                        "contributions": r.contributions,
                        "explanation": r.explanation,
                    }
                    for r in results
                ],
                request_id=getattr(request.state, "request_id", None),
                model_role=model_role,
                routed_model_version=routed_version,
                shadow_predictions=shadow_results_sync,
                enriched_features=enriched_features,
                feature_views_applied=feature_views_applied,
            )
            await run_in_threadpool(sink.emit, audit_event)
        except Exception:
            logger.exception("Failed to emit scoring audit event.")

        # Asynchronous non-blocking shadow evaluation
        should_shadow_async = sh_model is not None and model_role == "champion" and not is_mixed
        if should_shadow_async and sh_model is not None:
            shadow_tasks: set[asyncio.Task[None]] = request.app.state.shadow_tasks
            shadow_task = asyncio.ensure_future(
                run_in_threadpool(
                    _evaluate_shadow_traffic,
                    shadow_model=sh_model,
                    frame=frame,
                    primary_results=results,
                    audit_sink=sink,
                    request_id=getattr(request.state, "request_id", None),
                    shadow_evaluations_counter=getattr(
                        request.app.state, "shadow_evaluations_counter", None
                    ),
                    shadow_discrepancies_counter=getattr(
                        request.app.state, "shadow_discrepancies_counter", None
                    ),
                    primary_model_version=routed_version,
                    router=router,
                    shadow_discrepancy_total_counter=getattr(
                        request.app.state, "shadow_discrepancy_total_counter", None
                    ),
                    canary_status_gauge=getattr(
                        request.app.state, "canary_status_gauge", None
                    ),
                )
            )
            shadow_tasks.add(shadow_task)
            shadow_task.add_done_callback(shadow_tasks.discard)

        return (
            applied_threshold,
            results,
            fallback_applied,
            fallback_reason,
            eff_review,
            eff_deny,
            model_role,
            routed_version,
            shadow_results_sync,
        )

    def _enrich_records_with_feature_store(
        request: Request,
        records: list[dict[str, float]],
        expected_cols: set[str],
    ) -> tuple[list[dict[str, float]], list[str], list[str]]:
        store: FeatureStoreProtocol | None = getattr(request.app.state, "feature_store", None)
        if store is None:
            return records, [], []

        counter: Counter | None = getattr(request.app.state, "feature_lookups_counter", None)
        applied_views: list[str] = []
        enriched_features: list[str] = []
        header_entity = request.headers.get("X-Entity-ID")

        enriched_records: list[dict[str, float]] = []
        for rec in records:
            tx = dict(rec)
            for vn in store.list_views():
                view = store.get_view(vn)
                if view is None:
                    continue
                eid = tx.get(view.entity_key)
                if eid is None and header_entity:
                    eid = header_entity  # type: ignore[assignment]
                if eid is not None:
                    obs_t = tx.get("Time")
                    as_of = float(obs_t) if obs_t is not None else None
                    str_eid = (
                        str(int(eid))
                        if isinstance(eid, (int, float)) and float(eid).is_integer()
                        else str(eid)
                    )
                    res = store.lookup_online(
                        entity_key=view.entity_key,
                        entity_id=str_eid,
                        view_name=view.name,
                        as_of_time=as_of,
                    )
                    if not res.found and str_eid != str(eid):
                        alt_res = store.lookup_online(
                            entity_key=view.entity_key,
                            entity_id=str(eid),
                            view_name=view.name,
                            as_of_time=as_of,
                        )
                        if alt_res.found:
                            res = alt_res
                    if counter is not None:
                        counter.labels(
                            entity_key=view.entity_key,
                            status="hit" if res.found else "miss",
                        ).inc()
                    if res.found:
                        if view.name not in applied_views:
                            applied_views.append(view.name)
                        for fname, fval in res.values.items():
                            if fname in expected_cols or fname in tx:
                                tx[fname] = float(fval)
                                if fname not in enriched_features:
                                    enriched_features.append(fname)
            enriched_records.append(tx)

        pruned_records: list[dict[str, float]] = [
            {k: float(v) for k, v in rec_item.items() if k in expected_cols}
            for rec_item in enriched_records
        ]

        return pruned_records, enriched_features, applied_views

    @application.post(
        "/v1/predict",
        response_model=PredictionResponse,
        responses={503: {"description": "Scoring concurrency cap reached."}},
        tags=["predictions"],
    )
    async def predict(
        payload: PredictionRequest,
        request: Request,
    ) -> PredictionResponse | JSONResponse:
        loaded = _model_from_request(request)
        vel_buf: VelocityWindowBuffer | None = getattr(request.app.state, "velocity_buffer", None)
        vel_cfg: VelocityConfig | None = getattr(request.app.state, "velocity_config", None)
        vel_counter: Counter | None = getattr(
            request.app.state, "velocity_enrichments_counter", None
        )
        vel_gauge: Gauge | None = getattr(request.app.state, "velocity_entities_gauge", None)

        if vel_buf is not None and vel_cfg is not None:
            enriched_records: list[dict[str, float]] = []
            expected_cols = set(loaded.feature_names)
            for tx in payload.transactions:
                header_entity = request.headers.get("X-Entity-ID")
                tx_copy = dict(tx)
                if header_entity and vel_cfg.entity_key not in tx_copy:
                    tx_copy[vel_cfg.entity_key] = header_entity  # type: ignore[assignment]
                enriched = enrich_record_velocity(tx_copy, vel_buf, vel_cfg, update=True)
                frame_row: dict[str, float] = {}
                for k, v in enriched.items():
                    if k in expected_cols or (k in tx and k != vel_cfg.entity_key):
                        frame_row[k] = float(v)
                enriched_records.append(frame_row)
            frame_records = enriched_records
            if vel_counter is not None:
                vel_counter.inc(len(payload.transactions))
            if vel_gauge is not None:
                vel_gauge.set(vel_buf.entity_count)
        else:
            frame_records = payload.transactions

        (
            frame_records,
            enriched_features,
            feature_views_applied,
        ) = _enrich_records_with_feature_store(
            request, frame_records, set(loaded.feature_names)
        )

        try:
            (
                applied_threshold,
                results,
                fallback_applied,
                fallback_reason,
                eff_review,
                eff_deny,
                model_role,
                routed_version,
                shadow_results,
            ) = await _score_with_guardrails(
                request,
                loaded,
                pd.DataFrame(frame_records),
                features=frame_records,
                explain=payload.explain,
                explain_llm=payload.explain_llm,
                threshold=payload.threshold,
                review_threshold=payload.review_threshold,
                deny_threshold=payload.deny_threshold,
                include_shadow=payload.include_shadow,
                enriched_features=enriched_features or None,
                feature_views_applied=feature_views_applied or None,
            )
        except _ScoringOverloadedError:
            return _scoring_overloaded_response()
        return PredictionResponse(
            model_version=_model_version(loaded),
            threshold=applied_threshold,
            model_threshold=loaded.threshold,
            review_threshold=eff_review,
            deny_threshold=eff_deny,
            predictions=results,
            fallback_applied=fallback_applied,
            fallback_reason=fallback_reason,
            model_role=model_role,
            routed_model_version=routed_version,
            shadow_predictions=shadow_results,
        )

    @application.post(
        "/v1/score",
        response_model=ScoreResponse,
        responses={503: {"description": "Scoring concurrency cap reached."}},
        tags=["predictions"],
    )
    async def score(
        payload: ScoreRequest,
        request: Request,
    ) -> ScoreResponse | JSONResponse:
        loaded = _model_from_request(request)
        vel_buf: VelocityWindowBuffer | None = getattr(request.app.state, "velocity_buffer", None)
        vel_cfg: VelocityConfig | None = getattr(request.app.state, "velocity_config", None)
        vel_counter: Counter | None = getattr(
            request.app.state, "velocity_enrichments_counter", None
        )
        vel_gauge: Gauge | None = getattr(request.app.state, "velocity_entities_gauge", None)

        if vel_buf is not None and vel_cfg is not None:
            expected_cols = set(loaded.feature_names)
            header_entity = request.headers.get("X-Entity-ID")
            tx_copy = dict(payload.transaction)
            if header_entity and vel_cfg.entity_key not in tx_copy:
                tx_copy[vel_cfg.entity_key] = header_entity  # type: ignore[assignment]
            enriched = enrich_record_velocity(tx_copy, vel_buf, vel_cfg, update=True)
            frame_row: dict[str, float] = {}
            for k, v in enriched.items():
                if k in expected_cols or (k in payload.transaction and k != vel_cfg.entity_key):
                    frame_row[k] = float(v)
            frame_records = [frame_row]
            if vel_counter is not None:
                vel_counter.inc()
            if vel_gauge is not None:
                vel_gauge.set(vel_buf.entity_count)
        else:
            frame_records = [payload.transaction]

        (
            frame_records,
            enriched_features,
            feature_views_applied,
        ) = _enrich_records_with_feature_store(
            request, frame_records, set(loaded.feature_names)
        )

        try:
            (
                applied_threshold,
                results,
                fallback_applied,
                fallback_reason,
                eff_review,
                eff_deny,
                model_role,
                routed_version,
                shadow_results,
            ) = await _score_with_guardrails(
                request,
                loaded,
                pd.DataFrame(frame_records),
                features=frame_records,
                explain=payload.explain,
                explain_llm=payload.explain_llm,
                threshold=payload.threshold,
                review_threshold=payload.review_threshold,
                deny_threshold=payload.deny_threshold,
                include_shadow=payload.include_shadow,
                enriched_features=enriched_features or None,
                feature_views_applied=feature_views_applied or None,
            )
        except _ScoringOverloadedError:
            return _scoring_overloaded_response()
        return ScoreResponse(
            model_version=_model_version(loaded),
            threshold=applied_threshold,
            model_threshold=loaded.threshold,
            review_threshold=eff_review,
            deny_threshold=eff_deny,
            prediction=results[0],
            fallback_applied=fallback_applied,
            fallback_reason=fallback_reason,
            model_role=model_role,
            routed_model_version=routed_version,
            shadow_prediction=shadow_results[0] if shadow_results else None,
        )

    @application.get("/v1/velocity/stats", tags=["velocity"])
    async def velocity_stats(request: Request) -> dict[str, Any]:
        """Return operational metrics of the in-memory velocity window buffer."""
        buf: VelocityWindowBuffer | None = getattr(request.app.state, "velocity_buffer", None)
        if buf is None:
            return {
                "enabled": False,
                "entities_tracked": 0,
                "total_events": 0,
                "max_entities": 0,
                "max_events_per_entity": 0,
                "configured_windows": [],
            }
        out_stats = buf.stats()
        out_stats["enabled"] = True
        return out_stats

    @application.get("/v1/velocity/profile/{entity_id}", tags=["velocity"])
    async def velocity_profile(entity_id: str, request: Request) -> Any:
        """Inspect real-time sliding window velocity profile for an entity."""
        buf: VelocityWindowBuffer | None = getattr(request.app.state, "velocity_buffer", None)
        if buf is None:
            return JSONResponse(
                status_code=404,
                content={"detail": "Velocity tracking is not enabled on this service."},
            )
        return buf.get_entity_profile(entity_id)

    @application.get("/v1/features/stats", tags=["features"])
    async def feature_store_stats(request: Request) -> dict[str, Any]:
        """Return operational metrics of the integrated feature store."""
        fs: FeatureStoreProtocol | None = getattr(request.app.state, "feature_store", None)
        if fs is None:
            return {
                "enabled": False,
                "views_count": 0,
                "entities_count": 0,
                "views": [],
            }
        base_stats = fs.get_stats()
        views_info = []
        for v_name in fs.list_views():
            view = fs.get_view(v_name)
            if view is not None:
                views_info.append(
                    {
                        "name": view.name,
                        "entity_key": view.entity_key,
                        "features": [f.name for f in view.features],
                        "ttl_seconds": view.ttl_seconds,
                    }
                )
        return {
            "enabled": True,
            **base_stats,
            "views_detail": views_info,
        }

    @application.get("/v1/features/lookup/{entity_key}/{entity_id}", tags=["features"])
    async def feature_store_lookup(
        entity_key: str, entity_id: str, request: Request
    ) -> Any:
        """Inspect online feature values for a specific entity."""
        fs: FeatureStoreProtocol | None = getattr(request.app.state, "feature_store", None)
        if fs is None:
            return JSONResponse(
                status_code=404,
                content={"detail": "Feature store is not enabled on this service."},
            )
        result = fs.lookup_online(entity_key=entity_key, entity_id=entity_id)
        if not result.found:
            return JSONResponse(
                status_code=404,
                content={
                    "detail": (
                        f"Entity '{entity_id}' with key '{entity_key}' not found in feature store."
                    )
                },
            )
        return result.to_dict()

    @application.get(
        "/v1/routing/status",
        response_model=RoutingStatusResponse,
        responses={503: {"description": "Model not loaded."}},
        tags=["routing"],
    )
    async def routing_status(request: Request) -> RoutingStatusResponse | JSONResponse:
        """Inspect multi-model routing topology, weights, and canary telemetry status."""
        loaded = getattr(request.app.state, "model", None)
        if loaded is None:
            return JSONResponse(status_code=503, content={"detail": "Model not loaded."})
        router: ChampionChallengerRouter | None = getattr(request.app.state, "router", None)
        champ_ver = _model_version(loaded)
        if router is None:
            return RoutingStatusResponse(
                champion_model_version=champ_ver,
                challenger_model_version=None,
                routing_policy=RoutingPolicy().to_dict(),
                metrics=RoutingMetricsTracker().get_summary(),
                canary_status="healthy",
                rollback_reason=None,
            )
        return RoutingStatusResponse(
            champion_model_version=router.champion_version,
            challenger_model_version=router.challenger_version,
            routing_policy=router.policy.to_dict(),
            metrics=router.metrics.get_summary(),
            canary_status=router.metrics.canary_status.value,
            rollback_reason=router.metrics.rollback_reason,
        )

    @application.post(
        "/v1/routing/rollback",
        response_model=RoutingStatusResponse,
        responses={503: {"description": "Model not loaded."}},
        tags=["routing"],
    )
    async def routing_rollback(
        request: Request,
        reason: str = "Operational rollback triggered via API",
    ) -> RoutingStatusResponse | JSONResponse:
        """Trigger manual operational rollback of canary routing back to Champion."""
        loaded = getattr(request.app.state, "model", None)
        if loaded is None:
            return JSONResponse(status_code=503, content={"detail": "Model not loaded."})
        router: ChampionChallengerRouter | None = getattr(request.app.state, "router", None)
        if router is not None:
            router.metrics.trigger_rollback(reason)
        return await routing_status(request)

    @application.post(
        "/v1/routing/reset",
        response_model=RoutingStatusResponse,
        responses={503: {"description": "Model not loaded."}},
        tags=["routing"],
    )
    async def routing_reset(request: Request) -> RoutingStatusResponse | JSONResponse:
        """Reset canary divergence metrics and operational status back to HEALTHY."""
        loaded = getattr(request.app.state, "model", None)
        if loaded is None:
            return JSONResponse(status_code=503, content={"detail": "Model not loaded."})
        router: ChampionChallengerRouter | None = getattr(request.app.state, "router", None)
        if router is not None:
            router.metrics.reset_canary()
        return await routing_status(request)

    @application.get(
        "/v1/pipeline/topology",
        response_model=PipelineTopologyResponse,
        responses={404: {"description": "Decision graph pipeline is not enabled."}},
        tags=["pipeline"],
    )
    async def pipeline_topology(request: Request) -> PipelineTopologyResponse | JSONResponse:
        """Inspect the registered decision graph pipeline topology and execution plan."""
        graph: DecisionGraph | None = getattr(request.app.state, "pipeline_graph", None)
        if graph is None:
            return JSONResponse(
                status_code=404,
                content={"detail": "Decision graph pipeline is not enabled on this service."},
            )
        data = graph.to_dict()
        return PipelineTopologyResponse(
            name=data["name"],
            node_count=data["node_count"],
            edge_count=data["edge_count"],
            nodes=data["nodes"],
            edges=data["edges"],
            execution_plan=data["execution_plan"],
        )

    @application.post(
        "/v1/pipeline/score",
        response_model=PipelineScoreResponse,
        responses={404: {"description": "Decision graph pipeline is not enabled."}},
        tags=["pipeline"],
    )
    async def pipeline_score(
        payload: PipelineScoreRequest,
        request: Request,
    ) -> PipelineScoreResponse | JSONResponse:
        """Execute the real-time decision graph pipeline for a single transaction."""
        executor: DecisionGraphExecutor | None = getattr(
            request.app.state, "pipeline_executor", None
        )
        if executor is None:
            return JSONResponse(
                status_code=404,
                content={"detail": "Decision graph pipeline is not enabled on this service."},
            )

        metadata = {"explain": payload.explain}
        result = await executor.execute(payload.transaction, metadata=metadata)

        stage_dur_hist: Histogram | None = getattr(
            request.app.state, "pipeline_stage_duration_histogram", None
        )
        if stage_dur_hist is not None:
            for node_name, stage_res in result.stage_results.items():
                stage_dur_hist.labels(
                    stage=node_name,
                    status=stage_res.status.value,
                ).observe(stage_res.latency_ms / 1000.0)

        exec_counter: Counter | None = getattr(
            request.app.state, "pipeline_executions_counter", None
        )
        if exec_counter is not None:
            exec_counter.labels(
                status="success" if result.success else "failed",
                degraded=str(result.degraded).lower(),
            ).inc()

        cache_hits_counter: Counter | None = getattr(
            request.app.state, "pipeline_cache_hits_counter", None
        )
        if cache_hits_counter is not None:
            for stage_name, stage_res in result.stage_results.items():
                if stage_res.cached:
                    cache_hits_counter.labels(stage=stage_name).inc()

        cache_misses_counter: Counter | None = getattr(
            request.app.state, "pipeline_cache_misses_counter", None
        )
        if cache_misses_counter is not None:
            for stage_name, stage_res in result.stage_results.items():
                node_obj = executor.graph.nodes.get(stage_name)
                if not stage_res.cached and (
                    executor.cache.get_policy(stage_name) is not None
                    or getattr(node_obj, "cache_policy", None) is not None
                ):
                    cache_misses_counter.labels(stage=stage_name).inc()

        spec_exec_counter: Counter | None = getattr(
            request.app.state, "pipeline_speculative_executions_counter", None
        )
        if spec_exec_counter is not None and result.speculative_executed:
            spec_exec_counter.inc()

        spec_hit_counter: Counter | None = getattr(
            request.app.state, "pipeline_speculative_hits_counter", None
        )
        if spec_hit_counter is not None and result.speculative_hit:
            spec_hit_counter.inc()

        sink: AuditSink = getattr(request.app.state, "audit_sink", resolved_audit_sink)
        loaded_model_for_audit: FraudModel | None = getattr(request.app.state, "model", None)
        try:
            audit_event = build_scoring_audit_event(
                model_version=(
                    _model_version(loaded_model_for_audit)
                    if loaded_model_for_audit is not None
                    else "pipeline"
                ),
                dataset_fingerprint=(
                    str(loaded_model_for_audit.metadata.get("dataset_fingerprint", ""))
                    if loaded_model_for_audit is not None
                    else "pipeline"
                ),
                threshold=(
                    loaded_model_for_audit.threshold
                    if loaded_model_for_audit is not None
                    else 0.5
                ),
                features=[payload.transaction],
                fallback_applied=result.degraded,
                fallback_reason="pipeline_degraded" if result.degraded else None,
                predictions=[
                    {
                        "fraud_probability": result.fraud_probability,
                        "raw_probability": result.raw_probability,
                        "is_fraud": result.is_fraud,
                        "decision": result.decision,
                        "matched_rule": result.matched_rule,
                        "rule_action": result.rule_action,
                        "contributions": result.contributions,
                        "explanation": None,
                    }
                ],
                request_id=getattr(request.state, "request_id", None),
                execution_trace=result.to_dict(),
                pipeline_stages=[s.to_dict() for s in result.stage_results.values()],
                degraded_nodes=result.degraded_nodes,
                cache_hits=result.cache_hits,
                cache_misses=result.cache_misses,
                speculative_executed=result.speculative_executed,
                speculative_hit=result.speculative_hit,
            )
            await run_in_threadpool(sink.emit, audit_event)
        except Exception:
            logger.exception("Failed to emit pipeline scoring audit event.")

        return PipelineScoreResponse(
            execution_id=result.execution_id,
            decision=result.decision,
            is_fraud=result.is_fraud,
            fraud_probability=result.fraud_probability,
            raw_probability=result.raw_probability,
            matched_rule=result.matched_rule,
            rule_action=result.rule_action,
            contributions=result.contributions,
            total_latency_ms=result.total_latency_ms,
            degraded=result.degraded,
            degraded_nodes=result.degraded_nodes,
            execution_path=result.execution_path,
            stage_results={k: v.to_dict() for k, v in result.stage_results.items()},
        )

    @application.get(
        "/v1/pipeline/cache/stats",
        response_model=PipelineCacheStatsResponse,
        responses={
            404: {"description": "Decision graph pipeline or cache is not enabled."},
        },
        tags=["pipeline"],
    )
    async def pipeline_cache_stats(
        request: Request,
    ) -> PipelineCacheStatsResponse | JSONResponse:
        """Retrieve operational statistics for pipeline node caching."""
        executor: DecisionGraphExecutor | None = getattr(
            request.app.state, "pipeline_executor", None
        )
        if executor is None:
            return JSONResponse(
                status_code=404,
                content={"detail": "Decision graph pipeline is not enabled on this service."},
            )
        if not executor.enable_cache:
            return JSONResponse(
                status_code=404,
                content={"detail": "Pipeline caching is not enabled."},
            )
        return PipelineCacheStatsResponse(**executor.cache.stats_dict())

    @application.post(
        "/v1/pipeline/cache/clear",
        response_model=PipelineCacheClearResponse,
        responses={
            404: {"description": "Decision graph pipeline or cache is not enabled."},
        },
        tags=["pipeline"],
    )
    async def pipeline_cache_clear(
        request: Request,
    ) -> PipelineCacheClearResponse | JSONResponse:
        """Clear all cached entries across all pipeline stages."""
        executor: DecisionGraphExecutor | None = getattr(
            request.app.state, "pipeline_executor", None
        )
        if executor is None:
            return JSONResponse(
                status_code=404,
                content={"detail": "Decision graph pipeline is not enabled on this service."},
            )
        if not executor.enable_cache:
            return JSONResponse(
                status_code=404,
                content={"detail": "Pipeline caching is not enabled."},
            )
        executor.cache.clear()
        return PipelineCacheClearResponse(
            status="cleared",
            message="Pipeline cache cleared successfully.",
        )

    @application.get("/metrics", tags=["operations"])
    async def metrics() -> Response:
        cb: CircuitBreaker | None = getattr(application.state, "circuit_breaker", None)
        if cb is not None:
            gauge: Gauge | None = getattr(application.state, "circuit_breaker_gauge", None)
            if gauge is not None:
                state_map = {"closed": 0.0, "half_open": 1.0, "open": 2.0}
                gauge.set(state_map.get(cb.state, 0.0))
        return Response(
            content=generate_latest(metrics_registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    return application


def _evaluate_shadow_traffic(
    *,
    shadow_model: FraudModel,
    frame: pd.DataFrame,
    primary_results: list[PredictionResult],
    audit_sink: AuditSink,
    request_id: str | None,
    shadow_evaluations_counter: Counter | None = None,
    shadow_discrepancies_counter: Counter | None = None,
    primary_model_version: str | None = None,
    router: ChampionChallengerRouter | None = None,
    shadow_discrepancy_total_counter: Counter | None = None,
    canary_status_gauge: Gauge | None = None,
) -> None:
    """Evaluate candidate transactions against shadow model asynchronously."""
    try:
        shadow_probabilities = shadow_model.predict_probabilities(frame)
        shadow_decisions = shadow_probabilities >= shadow_model.threshold

        discrepancies = 0
        discrepancy_details: list[dict[str, Any]] = []
        for idx, (p_res, s_prob, s_dec) in enumerate(
            zip(primary_results, shadow_probabilities, shadow_decisions, strict=True)
        ):
            s_dec_bool = bool(s_dec)
            if p_res.is_fraud != s_dec_bool:
                discrepancies += 1
                discrepancy_details.append(
                    {
                        "index": idx,
                        "primary_probability": p_res.fraud_probability,
                        "primary_decision": p_res.is_fraud,
                        "shadow_probability": float(s_prob),
                        "shadow_decision": s_dec_bool,
                    }
                )
                if shadow_discrepancies_counter is not None:
                    shadow_discrepancies_counter.labels(
                        shadow_decision=str(s_dec_bool).lower(),
                        primary_decision=str(p_res.is_fraud).lower(),
                    ).inc()
                if shadow_discrepancy_total_counter is not None:
                    shadow_discrepancy_total_counter.labels(
                        shadow_decision=str(s_dec_bool).lower(),
                        primary_decision=str(p_res.is_fraud).lower(),
                    ).inc()

        has_discrepancy = discrepancies > 0
        if shadow_evaluations_counter is not None:
            shadow_evaluations_counter.labels(has_discrepancy=str(has_discrepancy).lower()).inc(
                len(primary_results)
            )

        primary_probs = [p.fraud_probability for p in primary_results]
        prob_diffs = [
            abs(p - float(s))
            for p, s in zip(primary_probs, shadow_probabilities, strict=True)
        ]
        mean_div = float(np.mean(prob_diffs)) if prob_diffs else 0.0
        max_div = float(np.max(prob_diffs)) if prob_diffs else 0.0

        if router is not None:
            router.metrics.record_batch_comparison(
                champion_probs=primary_probs,
                champion_decs=[p.is_fraud for p in primary_results],
                challenger_probs=[float(s) for s in shadow_probabilities],
                challenger_decs=[bool(s) for s in shadow_decisions],
                canary_config=router.policy.canary_config,
            )
            if canary_status_gauge is not None:
                for c_stat in CanaryStatus:
                    canary_status_gauge.labels(status=c_stat.value).set(
                        1.0 if router.metrics.canary_status == c_stat else 0.0
                    )

        shadow_event = build_shadow_scoring_audit_event(
            shadow_model_version=_model_version(shadow_model),
            shadow_dataset_fingerprint=str(shadow_model.metadata.get("dataset_fingerprint", "")),
            evaluated_count=len(primary_results),
            discrepancy_count=discrepancies,
            discrepancies=discrepancy_details,
            request_id=request_id,
            mean_probability_divergence=mean_div,
            max_probability_divergence=max_div,
            primary_model_version=primary_model_version,
        )
        audit_sink.emit(shadow_event)
    except Exception:
        logger.exception("Shadow scoring evaluation failed.")


async def _request_body_error(request: Request) -> JSONResponse | None:
    if request.method not in _BODY_METHODS:
        return None

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length."})
        if declared_length < 0:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length."})
        if declared_length > MAX_REQUEST_BODY_BYTES:
            return _request_too_large_response()

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_REQUEST_BODY_BYTES:
            return _request_too_large_response()
        body.extend(chunk)
    # Starlette has no public API to seed the body cache; without this, the
    # stream we just consumed above would appear empty to route handlers.
    request._body = bytes(body)  # noqa: SLF001
    return None


def _request_too_large_response() -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={"detail": f"Request body exceeds the {MAX_REQUEST_BODY_BYTES}-byte limit."},
    )


def _metric_path(request: Request) -> str:
    """Return a bounded metric label for a request.

    The matched route template is used instead of the raw path so an undeclared path
    cannot create a new time series per request, which would let any client grow the
    registry without limit.
    """
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    return route_path if isinstance(route_path, str) else UNMATCHED_ROUTE_LABEL


def _scoring_overloaded_response() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "Too many concurrent scoring requests."},
        headers={"Retry-After": "1"},
    )


async def _drain_shadow_tasks(
    tasks: Collection[asyncio.Task[None]],
    timeout_seconds: float,
) -> None:
    """Wait for in-flight shadow evaluations so a shutdown does not silently drop them."""
    pending = [task for task in tasks if not task.done()]
    if not pending:
        return
    _, unfinished = await asyncio.wait(pending, timeout=timeout_seconds)
    for task in unfinished:
        logger.warning("Cancelling a shadow evaluation that outlived the shutdown grace period.")
        task.cancel()
    if unfinished:
        await asyncio.gather(*unfinished, return_exceptions=True)


def _model_from_request(request: Request) -> FraudModel:
    return cast(FraudModel, request.app.state.model)


def _model_version(model: FraudModel) -> str:
    return str(model.metadata.get("dataset_fingerprint", ""))[:12]


def _shadow_model_version(shadow_model: FraudModel | None) -> str | None:
    if shadow_model is None:
        return None
    return _model_version(shadow_model)


def _operational_status(request: Request) -> _OperationalStatus:
    """Collect the circuit-breaker, fallback, degraded, shadow, and rules detail of a service."""
    circuit_breaker: CircuitBreaker | None = getattr(request.app.state, "circuit_breaker", None)
    degraded_mode = bool(getattr(request.app.state, "degraded_mode", False))
    if circuit_breaker is not None and circuit_breaker.state == "open":
        degraded_mode = True
    active_rules: RuleSet | None = getattr(request.app.state, "rules", None)
    prec = getattr(request.app.state, "rule_precedence", None)
    prec_str = (
        prec.value
        if isinstance(prec, RulePrecedence)
        else (str(prec) if prec is not None else None)
    )
    vel_buf: VelocityWindowBuffer | None = getattr(request.app.state, "velocity_buffer", None)
    loaded_model: FraudModel | None = getattr(request.app.state, "model", None)
    recalibration_enabled: bool | None = None
    recalibration_method: str | None = None
    recal = getattr(loaded_model, "recalibrator", None)
    if isinstance(recal, BaseRecalibrator):
        recalibration_enabled = True
        recalibration_method = recal.method.value

    router: ChampionChallengerRouter | None = getattr(request.app.state, "router", None)
    routing_strategy: str | None = None
    challenger_model_version: str | None = None
    canary_status: str | None = None
    if router is not None and (
        router.challenger_model is not None
        or router.policy.strategy != TrafficSplitStrategy.CHAMPION_ONLY
    ):
        routing_strategy = router.policy.strategy.value
        challenger_model_version = router.challenger_version
        canary_status = router.metrics.canary_status.value

    fs: FeatureStoreProtocol | None = getattr(request.app.state, "feature_store", None)
    features_enabled: bool | None = None
    features_views_count: int | None = None
    features_entities_count: int | None = None
    if fs is not None:
        features_enabled = True
        fs_stats = fs.get_stats()
        features_views_count = int(fs_stats.get("views_count", len(fs.list_views())))
        features_entities_count = int(fs_stats.get("total_entities", 0))

    pipe_graph: DecisionGraph | None = getattr(request.app.state, "pipeline_graph", None)
    raw_pipe_enabled: bool = bool(getattr(request.app.state, "pipeline_enabled", False))
    pipe_enabled: bool | None = True if (raw_pipe_enabled and pipe_graph is not None) else None
    pipe_nodes_count: int | None = (
        len(pipe_graph.nodes) if pipe_enabled and pipe_graph is not None else None
    )

    return _OperationalStatus(
        fallback_mode=str(getattr(request.app.state, "fallback_mode", "raise")),
        degraded_mode=degraded_mode,
        circuit_breaker=circuit_breaker.to_dict() if circuit_breaker is not None else None,
        shadow_model_version=_shadow_model_version(
            cast(FraudModel | None, getattr(request.app.state, "shadow_model", None))
        ),
        rules_count=len(active_rules) if active_rules is not None else None,
        rule_precedence=prec_str if active_rules is not None else None,
        velocity_enabled=True if vel_buf is not None else None,
        velocity_entities_count=vel_buf.entity_count if vel_buf is not None else None,
        recalibration_enabled=recalibration_enabled,
        recalibration_method=recalibration_method,
        routing_strategy=routing_strategy,
        challenger_model_version=challenger_model_version,
        canary_status=canary_status,
        features_enabled=features_enabled,
        features_views_count=features_views_count,
        features_entities_count=features_entities_count,
        pipeline_enabled=pipe_enabled,
        pipeline_nodes_count=pipe_nodes_count,
    )


def app_from_environment() -> FastAPI:
    """Uvicorn factory that resolves the model path during application startup."""
    return create_app()
