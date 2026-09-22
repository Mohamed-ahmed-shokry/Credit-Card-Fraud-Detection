"""Explanation provider interface with timeouts, redaction, cost controls, and fallback."""

from __future__ import annotations

import concurrent.futures
import logging
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter, time
from typing import Any, Protocol

from fraud_detection.audit import redact_data

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExplanationRequest:
    """Request payload for an individual transaction explanation."""

    probability: float
    threshold: float
    decision: str
    contributions: dict[str, float]
    features: dict[str, Any] = field(default_factory=dict)
    top_k: int = 3


@dataclass(frozen=True)
class ExplanationResult:
    """Outcome of generating a natural-language explanation."""

    explanation: str
    provider: str
    latency_ms: float = 0.0
    tokens_used: int | None = None
    fallback_triggered: bool = False
    fallback_reason: str | None = None


class ExplanationProvider(Protocol):
    """Protocol defining the interface for explanation providers."""

    @property
    def name(self) -> str:
        """Provider name."""
        ...

    def explain(self, request: ExplanationRequest) -> ExplanationResult:
        """Generate an explanation for a single transaction request."""
        ...

    def explain_batch(self, requests: Sequence[ExplanationRequest]) -> list[ExplanationResult]:
        """Generate explanations for a sequence of transaction requests."""
        ...


class TemplateExplanationProvider:
    """Deterministic, offline, template-based explanation provider (default)."""

    name: str = "template"

    def explain(self, request: ExplanationRequest) -> ExplanationResult:
        """Generate a deterministic explanation using the standard risk template."""
        start_time = perf_counter()
        ranked = sorted(
            request.contributions.items(),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[: request.top_k]
        factors = ", ".join(
            f"{feature} ({float(value):+.3f}, "
            f"{'increases' if float(value) >= 0 else 'decreases'} fraud risk)"
            for feature, value in ranked
        )
        prob = float(request.probability)
        risk_level = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.3 else "LOW"
        decision = "FRAUD" if prob >= request.threshold else "LEGITIMATE"

        text = (
            f"Transaction classified as {decision} "
            f"(fraud probability: {prob:.2%}, "
            f"applied threshold: {request.threshold:.2f}, risk level: {risk_level}). "
            f"Top contributing factors: {factors}."
        )
        duration_ms = (perf_counter() - start_time) * 1_000.0
        return ExplanationResult(
            explanation=text,
            provider=self.name,
            latency_ms=duration_ms,
        )

    def explain_batch(self, requests: Sequence[ExplanationRequest]) -> list[ExplanationResult]:
        """Generate explanations for all requests."""
        return [self.explain(req) for req in requests]


class CostController:
    """In-memory rate and token budget limiter for external provider calls."""

    def __init__(
        self,
        *,
        max_requests_per_minute: int | None = None,
        max_tokens_budget: int | None = None,
    ) -> None:
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_budget = max_tokens_budget
        self._timestamps: deque[float] = deque()
        self._tokens_consumed: int = 0
        self._lock = Lock()

    def check_and_record(self, tokens_used: int = 0) -> tuple[bool, str | None]:
        """Check whether budget allows the request and record usage."""
        with self._lock:
            now = time()
            if self.max_requests_per_minute is not None:
                cutoff = now - 60.0
                while self._timestamps and self._timestamps[0] < cutoff:
                    self._timestamps.popleft()
                if len(self._timestamps) >= self.max_requests_per_minute:
                    return False, "Rate limit exceeded (requests per minute)"
                self._timestamps.append(now)

            if self.max_tokens_budget is not None:
                if self._tokens_consumed + tokens_used > self.max_tokens_budget:
                    return False, "Token budget exhausted"
                self._tokens_consumed += tokens_used

            return True, None


class ExternalExplanationProvider:
    """Isolated provider boundary for external LLM calls with timeouts, redaction, and fallback."""

    def __init__(
        self,
        endpoint_fn: Callable[[dict[str, Any]], str],
        *,
        name: str = "external_llm",
        timeout_seconds: float = 2.0,
        redact_inputs: bool = True,
        cost_controller: CostController | None = None,
        fallback_provider: ExplanationProvider | None = None,
    ) -> None:
        self._name = name
        self.endpoint_fn = endpoint_fn
        self.timeout_seconds = timeout_seconds
        self.redact_inputs = redact_inputs
        self.cost_controller = cost_controller
        self.fallback_provider = fallback_provider or TemplateExplanationProvider()

    @property
    def name(self) -> str:
        """Provider name."""
        return self._name

    def explain(self, request: ExplanationRequest) -> ExplanationResult:
        """Generate an explanation via external endpoint, falling back on error or timeout."""
        start_time = perf_counter()

        # 1. Cost & budget check
        if self.cost_controller is not None:
            allowed, reason = self.cost_controller.check_and_record()
            if not allowed:
                logger.warning("Cost controller blocked external explanation: %s", reason)
                fb_res = self.fallback_provider.explain(request)
                return ExplanationResult(
                    explanation=fb_res.explanation,
                    provider=fb_res.provider,
                    latency_ms=(perf_counter() - start_time) * 1_000.0,
                    fallback_triggered=True,
                    fallback_reason=reason,
                )

        # 2. Redact payload before sending to external boundary
        payload: dict[str, Any] = {
            "probability": request.probability,
            "threshold": request.threshold,
            "decision": request.decision,
            "contributions": request.contributions,
            "features": request.features,
            "top_k": request.top_k,
        }
        if self.redact_inputs:
            payload = redact_data(payload)

        # 3. Call with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.endpoint_fn, payload)
            try:
                explanation_text = future.result(timeout=self.timeout_seconds)
                duration_ms = (perf_counter() - start_time) * 1_000.0
                return ExplanationResult(
                    explanation=explanation_text,
                    provider=self.name,
                    latency_ms=duration_ms,
                    fallback_triggered=False,
                )
            except concurrent.futures.TimeoutError:
                logger.warning(
                    "External explanation timed out after %.2fs. Using fallback.",
                    self.timeout_seconds,
                )
                fb_res = self.fallback_provider.explain(request)
                return ExplanationResult(
                    explanation=fb_res.explanation,
                    provider=fb_res.provider,
                    latency_ms=(perf_counter() - start_time) * 1_000.0,
                    fallback_triggered=True,
                    fallback_reason=f"Timeout after {self.timeout_seconds}s",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "External explanation failed with error: %s. Using fallback.",
                    exc,
                )
                fb_res = self.fallback_provider.explain(request)
                return ExplanationResult(
                    explanation=fb_res.explanation,
                    provider=fb_res.provider,
                    latency_ms=(perf_counter() - start_time) * 1_000.0,
                    fallback_triggered=True,
                    fallback_reason=str(exc),
                )

    def explain_batch(self, requests: Sequence[ExplanationRequest]) -> list[ExplanationResult]:
        """Generate explanations for all requests sequentially or via pool."""
        return [self.explain(req) for req in requests]
