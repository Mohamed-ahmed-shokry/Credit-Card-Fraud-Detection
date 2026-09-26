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
from typing import Annotated, Any, cast
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
from pydantic import BaseModel, ConfigDict, Field
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
from fraud_detection.explanations import ExplanationProvider
from fraud_detection.model import FraudModel, ModelArtifactError, load_model
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
REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time-Ms"
MAX_REQUEST_BODY_BYTES = 2 * 1024 * 1024
SHADOW_SHUTDOWN_TIMEOUT_SECONDS = 5.0
OPERATIONAL_PATHS = frozenset({"/health", "/metrics", "/live", "/ready"})
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
    """Fallback, degraded, circuit-breaker, and shadow detail for operational endpoints."""

    fallback_mode: str
    degraded_mode: bool
    circuit_breaker: dict[str, Any] | None
    shadow_model_version: str | None

    def response_fields(self) -> dict[str, Any]:
        """Return the response fields shared by `/health` and `/ready`."""
        return {
            "degraded_mode": True if self.degraded_mode else None,
            "fallback_mode": self.fallback_mode if self.fallback_mode != "raise" else None,
            "circuit_breaker": self.circuit_breaker,
            "shadow_model_version": self.shadow_model_version,
        }


class _ScoringOverloadedError(Exception):
    """Raised internally when the scoring concurrency cap sheds a request."""


class _RateLimiter:
    """Simple in-memory rate limiter using fixed window."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    def check(self, key: str) -> _RateLimitDecision:
        """Record one request and report whether it fits the window."""
        now = time.time()
        window_start = now - self.window_seconds
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


class PredictionResult(BaseModel):
    """Fraud score and thresholded decision for one transaction."""

    fraud_probability: float
    is_fraud: bool
    contributions: dict[str, float] | None = None
    explanation: str | None = None


class PredictionResponse(BaseModel):
    """Ordered batch prediction response."""

    model_version: str
    threshold: float
    model_threshold: float
    predictions: list[PredictionResult]
    fallback_applied: bool = False
    fallback_reason: str | None = None


class ScoreRequest(BaseModel):
    """One numeric transaction feature mapping."""

    model_config = ConfigDict(extra="forbid")

    transaction: dict[str, TransactionValue]
    explain: bool = False
    explain_llm: bool = False
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class ScoreResponse(BaseModel):
    """Single-transaction prediction response."""

    model_version: str
    threshold: float
    model_threshold: float
    prediction: PredictionResult
    fallback_applied: bool = False
    fallback_reason: str | None = None


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

        loaded_shadow = shadow_model
        if loaded_shadow is None and resolved_shadow_path is not None:
            loaded_shadow = load_model(resolved_shadow_path)

        application.state.model = loaded_model
        application.state.shadow_model = loaded_shadow
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
    async def rate_limit_middleware(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if rate_limiter is None or request.url.path in OPERATIONAL_PATHS:
            return await call_next(request)
        client_ip = request.client.host if request.client else "unknown"
        decision = rate_limiter.check(client_ip)
        if not decision.allowed:
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
                path=request.url.path,
                status_code="500",
            ).inc()
            request_duration.labels(method=request.method, path=request.url.path).observe(
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
            path=request.url.path,
            status_code=str(response.status_code),
        ).inc()
        request_duration.labels(method=request.method, path=request.url.path).observe(
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
        tags=["operations"],
    )
    async def health(request: Request) -> HealthResponse:
        loaded = _model_from_request(request)
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
        applied_threshold: float,
        explain: bool,
        explain_llm: bool,
    ) -> list[PredictionResult]:
        """Threshold primary probabilities and build the per-transaction results."""
        decisions = probabilities >= applied_threshold
        local_explanations = loaded.explain_local(frame) if explain or explain_llm else None
        natural_language = (
            loaded.explain_local_natural_language(
                frame,
                probabilities,
                threshold=applied_threshold,
                contributions=local_explanations,
                provider=explanation_provider,
            )
            if explain_llm
            else None
        )
        results: list[PredictionResult] = []
        for index, (probability, decision) in enumerate(zip(probabilities, decisions, strict=True)):
            results.append(
                PredictionResult(
                    fraud_probability=float(probability),
                    is_fraud=bool(decision),
                    contributions=(
                        local_explanations[index]
                        if explain and local_explanations is not None
                        else None
                    ),
                    explanation=(natural_language[index] if natural_language else None),
                )
            )
        prediction_counter.labels(is_fraud="true").inc(int(decisions.sum()))
        prediction_counter.labels(is_fraud="false").inc(int((~decisions).sum()))
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
        fallback_mode: str = "raise",
        fallback_score: float = 0.5,
        fallback_amount_threshold: float = 1000.0,
        force_degraded: bool = False,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> tuple[float, list[PredictionResult], bool, str | None]:
        """Score transactions with runtime guardrails and resilient degraded-state fallback."""
        # Reject a request whose feature schema does not match the artifact before any
        # guardrail state changes: a malformed request is a client error, never a model
        # failure, so it must not record a circuit-breaker failure or be answered with a
        # fabricated fallback score.
        loaded.validate_features(frame)
        applied_threshold = loaded.threshold if threshold is None else threshold
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
                    applied_threshold=applied_threshold,
                    explain=explain,
                    explain_llm=explain_llm,
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
            return applied_threshold, results, False, None

        if fallback_mode == "constant":
            dec_const = fallback_score >= applied_threshold
            results = [
                PredictionResult(
                    fraud_probability=float(fallback_score),
                    is_fraud=bool(dec_const),
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
                is_fraud = bool(prob >= applied_threshold)
                rule_exp = (
                    f"Fallback rule applied: Amount ({amt_val:.2f}) "
                    f"{'>=' if is_high else '<'} {fallback_amount_threshold:.2f}."
                )
                results.append(
                    PredictionResult(
                        fraud_probability=prob,
                        is_fraud=is_fraud,
                        contributions=None,
                        explanation=rule_exp,
                    )
                )
        fraud_count = sum(1 for r in results if r.is_fraud)
        prediction_counter.labels(is_fraud="true").inc(fraud_count)
        prediction_counter.labels(is_fraud="false").inc(len(results) - fraud_count)
        fallback_counter.labels(mode=fallback_mode, reason=fallback_reason or "unknown").inc(
            len(results)
        )
        return applied_threshold, results, True, fallback_reason

    async def _score_with_guardrails(
        request: Request,
        loaded: FraudModel,
        frame: pd.DataFrame,
        *,
        features: list[dict[str, float]],
        explain: bool,
        explain_llm: bool,
        threshold: float | None,
    ) -> tuple[float, list[PredictionResult], bool, str | None]:
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

        if scoring_semaphore is not None:
            if scoring_semaphore.locked():
                raise _ScoringOverloadedError
            await scoring_semaphore.acquire()
        try:
            applied_threshold, results, fallback_applied, fallback_reason = await run_in_threadpool(
                score_frame,
                loaded,
                frame,
                explain=explain,
                explain_llm=explain_llm,
                threshold=threshold,
                fallback_mode=fb_mode,
                fallback_score=fb_score,
                fallback_amount_threshold=fb_amount,
                force_degraded=force_degraded,
                circuit_breaker=cb,
            )
        finally:
            if scoring_semaphore is not None:
                scoring_semaphore.release()
        sink: AuditSink = getattr(request.app.state, "audit_sink", resolved_audit_sink)
        try:
            audit_event = build_scoring_audit_event(
                model_version=str(loaded.metadata.get("dataset_fingerprint", ""))[:12],
                dataset_fingerprint=str(loaded.metadata.get("dataset_fingerprint", "")),
                threshold=applied_threshold,
                features=features,
                fallback_applied=fallback_applied,
                fallback_reason=fallback_reason,
                predictions=[
                    {
                        "fraud_probability": r.fraud_probability,
                        "is_fraud": r.is_fraud,
                        "contributions": r.contributions,
                        "explanation": r.explanation,
                    }
                    for r in results
                ],
                request_id=getattr(request.state, "request_id", None),
            )
            await run_in_threadpool(sink.emit, audit_event)
        except Exception:
            logger.exception("Failed to emit scoring audit event.")

        sh_model: FraudModel | None = getattr(request.app.state, "shadow_model", None)
        if sh_model is not None:
            # Scheduled on the event loop instead of as a response background task: a
            # background task still runs before the request is finished, which held the
            # connection open for the whole challenger evaluation.
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
                )
            )
            shadow_tasks.add(shadow_task)
            shadow_task.add_done_callback(shadow_tasks.discard)

        return applied_threshold, results, fallback_applied, fallback_reason

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
        try:
            (
                applied_threshold,
                results,
                fallback_applied,
                fallback_reason,
            ) = await _score_with_guardrails(
                request,
                loaded,
                pd.DataFrame(payload.transactions),
                features=payload.transactions,
                explain=payload.explain,
                explain_llm=payload.explain_llm,
                threshold=payload.threshold,
            )
        except _ScoringOverloadedError:
            return _scoring_overloaded_response()
        return PredictionResponse(
            model_version=str(loaded.metadata["dataset_fingerprint"])[:12],
            threshold=applied_threshold,
            model_threshold=loaded.threshold,
            predictions=results,
            fallback_applied=fallback_applied,
            fallback_reason=fallback_reason,
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
        try:
            (
                applied_threshold,
                results,
                fallback_applied,
                fallback_reason,
            ) = await _score_with_guardrails(
                request,
                loaded,
                pd.DataFrame([payload.transaction]),
                features=[payload.transaction],
                explain=payload.explain,
                explain_llm=payload.explain_llm,
                threshold=payload.threshold,
            )
        except _ScoringOverloadedError:
            return _scoring_overloaded_response()
        return ScoreResponse(
            model_version=str(loaded.metadata["dataset_fingerprint"])[:12],
            threshold=applied_threshold,
            model_threshold=loaded.threshold,
            prediction=results[0],
            fallback_applied=fallback_applied,
            fallback_reason=fallback_reason,
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

        has_discrepancy = discrepancies > 0
        if shadow_evaluations_counter is not None:
            shadow_evaluations_counter.labels(has_discrepancy=str(has_discrepancy).lower()).inc(
                len(primary_results)
            )

        shadow_event = build_shadow_scoring_audit_event(
            shadow_model_version=str(shadow_model.metadata.get("dataset_fingerprint", ""))[:12],
            shadow_dataset_fingerprint=str(shadow_model.metadata.get("dataset_fingerprint", "")),
            evaluated_count=len(primary_results),
            discrepancy_count=discrepancies,
            discrepancies=discrepancy_details,
            request_id=request_id,
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


def _shadow_model_version(shadow_model: FraudModel | None) -> str | None:
    if shadow_model is None:
        return None
    return str(shadow_model.metadata["dataset_fingerprint"])[:12]


def _operational_status(request: Request) -> _OperationalStatus:
    """Collect the circuit-breaker, fallback, degraded, and shadow detail of a service."""
    circuit_breaker: CircuitBreaker | None = getattr(request.app.state, "circuit_breaker", None)
    degraded_mode = bool(getattr(request.app.state, "degraded_mode", False))
    if circuit_breaker is not None and circuit_breaker.state == "open":
        degraded_mode = True
    return _OperationalStatus(
        fallback_mode=str(getattr(request.app.state, "fallback_mode", "raise")),
        degraded_mode=degraded_mode,
        circuit_breaker=circuit_breaker.to_dict() if circuit_breaker is not None else None,
        shadow_model_version=_shadow_model_version(
            cast(FraudModel | None, getattr(request.app.state, "shadow_model", None))
        ),
    )


def app_from_environment() -> FastAPI:
    """Uvicorn factory that resolves the model path during application startup."""
    return create_app()
