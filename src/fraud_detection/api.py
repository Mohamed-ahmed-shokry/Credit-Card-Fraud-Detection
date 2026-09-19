"""FastAPI application for online fraud scoring."""

from __future__ import annotations

import logging
import math
import os
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Annotated, cast
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from fraud_detection import __version__
from fraud_detection.audit import (
    AuditSink,
    JsonlAuditSink,
    NullAuditSink,
    build_scoring_audit_event,
)
from fraud_detection.explanations import ExplanationProvider
from fraud_detection.model import FraudModel, ModelArtifactError, load_model

MODEL_PATH_ENVIRONMENT_VARIABLE = "FRAUD_MODEL_PATH"
AUDIT_LOG_ENVIRONMENT_VARIABLE = "FRAUD_AUDIT_LOG_PATH"
FALLBACK_MODE_ENVIRONMENT_VARIABLE = "FRAUD_FALLBACK_MODE"
FALLBACK_SCORE_ENVIRONMENT_VARIABLE = "FRAUD_FALLBACK_SCORE"
FALLBACK_AMOUNT_THRESHOLD_ENVIRONMENT_VARIABLE = "FRAUD_FALLBACK_AMOUNT_THRESHOLD"
DEGRADED_MODE_ENVIRONMENT_VARIABLE = "FRAUD_DEGRADED_MODE"
REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time-Ms"
MAX_REQUEST_BODY_BYTES = 2 * 1024 * 1024
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



def create_app(
    *,
    model: FraudModel | None = None,
    model_path: Path | str | None = None,
    api_keys: list[str] | None = None,
    rate_limit_requests: int = 0,
    rate_limit_window_seconds: float = 60.0,
    audit_sink: AuditSink | None = None,
    explanation_provider: ExplanationProvider | None = None,
    fallback_mode: str | None = None,
    fallback_score: float | None = None,
    fallback_amount_threshold: float | None = None,
    degraded_mode: bool | None = None,
) -> FastAPI:
    """Create an application using an injected model or a trusted artifact path.

    Args:
        model: Pre-loaded model instance.
        model_path: Path to model artifact directory.
        api_keys: Optional list of valid API keys. If provided, enables API key
            authentication via the `X-API-Key` header.
        rate_limit_requests: Maximum requests per window. If > 0, enables rate
            limiting per client IP.
        rate_limit_window_seconds: Time window for rate limiting in seconds.
        audit_sink: Optional structured audit sink. If omitted, checks
            the `FRAUD_AUDIT_LOG_PATH` environment variable or defaults to NullAuditSink.
        explanation_provider: Optional explanation provider for natural language risk
            summaries. Defaults to offline deterministic TemplateExplanationProvider.
        fallback_mode: Fallback policy when degraded or on model failure:
            'raise' (default), 'rule' (heuristic on Amount), or 'constant'.
        fallback_score: Fixed score for 'constant' fallback mode (default: 0.5).
        fallback_amount_threshold: Amount cutoff for 'rule' fallback mode (default: 1000.0).
        degraded_mode: When True, bypasses primary model and uses fallback policy.
    """
    logging.basicConfig(level=logging.INFO)

    raw_mode = (
        fallback_mode
        if fallback_mode is not None
        else os.getenv(FALLBACK_MODE_ENVIRONMENT_VARIABLE, "raise")
    ).lower().strip()
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

    rate_limiter = (
        _RateLimiter(rate_limit_requests, rate_limit_window_seconds)
        if rate_limit_requests > 0
        else None
    )
    api_key_set = set(api_keys) if api_keys else None

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
        loaded_model = model
        if loaded_model is None:
            configured_path = model_path or os.getenv(MODEL_PATH_ENVIRONMENT_VARIABLE)
            if configured_path is None:
                raise RuntimeError(
                    f"No model configured. Set {MODEL_PATH_ENVIRONMENT_VARIABLE} "
                    "or pass model_path to create_app()."
                )
            loaded_model = load_model(configured_path)
        application.state.model = loaded_model
        application.state.audit_sink = resolved_audit_sink
        application.state.explanation_provider = explanation_provider
        application.state.fallback_mode = resolved_fallback_mode
        application.state.fallback_score = resolved_fallback_score
        application.state.fallback_amount_threshold = resolved_fallback_amount
        application.state.degraded_mode = resolved_degraded_mode
        application.state.fallback_counter = fallback_counter
        yield
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
        if rate_limiter is None:
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
        if api_key_set is not None:
            api_key = request.headers.get("X-API-Key")
            if api_key not in api_key_set:
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
        supplied_request_id = request.headers.get(REQUEST_ID_HEADER)
        request_id = (
            supplied_request_id
            if supplied_request_id is not None
            and _REQUEST_ID_PATTERN.fullmatch(supplied_request_id)
            else uuid4().hex
        )
        request.state.request_id = request_id
        started_at = perf_counter()
        try:
            response: Response | None = await _request_body_error(request)
            if response is None:
                response = await call_next(request)
        except Exception:
            duration_seconds = perf_counter() - started_at
            request_counter.labels(
                method=request.method,
                path=request.url.path,
                status_code="500",
            ).inc()
            request_duration.labels(method=request.method, path=request.url.path).observe(
                duration_seconds
            )
            logger.exception(
                "request_failed method=%s path=%s duration_ms=%.3f request_id=%s",
                request.method,
                request.url.path,
                duration_seconds * 1_000,
                request_id,
            )
            raise

        duration_ms = (perf_counter() - started_at) * 1_000
        request_counter.labels(
            method=request.method,
            path=request.url.path,
            status_code=str(response.status_code),
        ).inc()
        request_duration.labels(method=request.method, path=request.url.path).observe(
            duration_ms / 1_000
        )
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[PROCESS_TIME_HEADER] = f"{duration_ms:.3f}"
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
        is_degraded = bool(getattr(request.app.state, "degraded_mode", False))
        fb_mode = str(getattr(request.app.state, "fallback_mode", "raise"))
        return HealthResponse(
            status="degraded" if is_degraded else "ready",
            service_version=__version__,
            model_created_at=str(loaded.metadata["created_at"]),
            feature_count=len(loaded.feature_names),
            threshold=loaded.threshold,
            degraded_mode=True if is_degraded else None,
            fallback_mode=fb_mode if fb_mode != "raise" else None,
        )

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
    ) -> tuple[float, list[PredictionResult], bool, str | None]:
        """Score transactions with runtime guardrails and resilient degraded-state fallback."""
        applied_threshold = loaded.threshold if threshold is None else threshold
        fallback_applied = False
        fallback_reason: str | None = None
        probabilities = None

        if force_degraded and fallback_mode != "raise":
            fallback_applied = True
            fallback_reason = "degraded_mode_active"
        else:
            try:
                probabilities = loaded.predict_probabilities(frame)
            except Exception as exc:
                if fallback_mode == "raise":
                    raise
                logger.warning(
                    "Primary model scoring failed; activating fallback guardrail (%s): %s",
                    fallback_mode,
                    exc,
                )
                fallback_applied = True
                fallback_reason = f"model_exception: {type(exc).__name__}"

        if fallback_applied:
            results: list[PredictionResult] = []
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
            fallback_counter.labels(
                mode=fallback_mode, reason=fallback_reason or "unknown"
            ).inc(len(results))
            return applied_threshold, results, True, fallback_reason

        if probabilities is None:
            raise RuntimeError("Primary model returned no probabilities.")
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
        results = []
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
        return applied_threshold, results, False, None

    @application.post(
        "/v1/predict",
        response_model=PredictionResponse,
        tags=["predictions"],
    )
    async def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        loaded = _model_from_request(request)
        frame = pd.DataFrame(payload.transactions)
        force_degraded = bool(getattr(request.app.state, "degraded_mode", False)) or (
            request.headers.get("X-Simulate-Degraded", "").lower() == "true"
        )
        fb_mode = str(getattr(request.app.state, "fallback_mode", "raise"))
        fb_score = float(getattr(request.app.state, "fallback_score", 0.5))
        fb_amount = float(getattr(request.app.state, "fallback_amount_threshold", 1000.0))

        applied_threshold, results, fallback_applied, fallback_reason = score_frame(
            loaded,
            frame,
            explain=payload.explain,
            explain_llm=payload.explain_llm,
            threshold=payload.threshold,
            fallback_mode=fb_mode,
            fallback_score=fb_score,
            fallback_amount_threshold=fb_amount,
            force_degraded=force_degraded,
        )
        sink: AuditSink = getattr(request.app.state, "audit_sink", resolved_audit_sink)
        try:
            audit_event = build_scoring_audit_event(
                model_version=str(loaded.metadata.get("dataset_fingerprint", ""))[:12],
                dataset_fingerprint=str(loaded.metadata.get("dataset_fingerprint", "")),
                threshold=applied_threshold,
                features=payload.transactions,
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
            sink.emit(audit_event)
        except Exception:
            logger.exception("Failed to emit scoring audit event.")
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
        tags=["predictions"],
    )
    async def score(payload: ScoreRequest, request: Request) -> ScoreResponse:
        loaded = _model_from_request(request)
        frame = pd.DataFrame([payload.transaction])
        force_degraded = bool(getattr(request.app.state, "degraded_mode", False)) or (
            request.headers.get("X-Simulate-Degraded", "").lower() == "true"
        )
        fb_mode = str(getattr(request.app.state, "fallback_mode", "raise"))
        fb_score = float(getattr(request.app.state, "fallback_score", 0.5))
        fb_amount = float(getattr(request.app.state, "fallback_amount_threshold", 1000.0))

        applied_threshold, results, fallback_applied, fallback_reason = score_frame(
            loaded,
            frame,
            explain=payload.explain,
            explain_llm=payload.explain_llm,
            threshold=payload.threshold,
            fallback_mode=fb_mode,
            fallback_score=fb_score,
            fallback_amount_threshold=fb_amount,
            force_degraded=force_degraded,
        )
        sink: AuditSink = getattr(request.app.state, "audit_sink", resolved_audit_sink)
        try:
            audit_event = build_scoring_audit_event(
                model_version=str(loaded.metadata.get("dataset_fingerprint", ""))[:12],
                dataset_fingerprint=str(loaded.metadata.get("dataset_fingerprint", "")),
                threshold=applied_threshold,
                features=[payload.transaction],
                fallback_applied=fallback_applied,
                fallback_reason=fallback_reason,
                predictions=[
                    {
                        "fraud_probability": results[0].fraud_probability,
                        "is_fraud": results[0].is_fraud,
                        "contributions": results[0].contributions,
                        "explanation": results[0].explanation,
                    }
                ],
                request_id=getattr(request.state, "request_id", None),
            )
            sink.emit(audit_event)
        except Exception:
            logger.exception("Failed to emit scoring audit event.")
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
        return Response(
            content=generate_latest(metrics_registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    return application


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


def _model_from_request(request: Request) -> FraudModel:
    return cast(FraudModel, request.app.state.model)


def app_from_environment() -> FastAPI:
    """Uvicorn factory that resolves the model path during application startup."""
    return create_app()
