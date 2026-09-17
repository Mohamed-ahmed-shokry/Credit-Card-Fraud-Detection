"""Tests for explanation provider interface, template, external wrapper, and fallbacks."""

from __future__ import annotations

import time
from typing import Any

from fastapi.testclient import TestClient

from fraud_detection.api import create_app
from fraud_detection.data import generate_synthetic_data, validate_frame
from fraud_detection.explanations import (
    CostController,
    ExplanationRequest,
    ExplanationResult,
    ExternalExplanationProvider,
    TemplateExplanationProvider,
)
from fraud_detection.model import train_model


def test_template_explanation_provider() -> None:
    provider = TemplateExplanationProvider()
    req = ExplanationRequest(
        probability=0.85,
        threshold=0.5,
        decision="FRAUD",
        contributions={"V1": 0.45, "V2": -0.3, "V3": 0.1},
        features={"V1": 1.2, "V2": -0.5, "V3": 0.3},
        top_k=2,
    )
    res = provider.explain(req)

    assert res.provider == "template"
    assert res.fallback_triggered is False
    assert "FRAUD" in res.explanation
    assert "85.00%" in res.explanation
    assert "HIGH" in res.explanation
    assert "V1 (+0.450, increases fraud risk)" in res.explanation
    assert "V2 (-0.300, decreases fraud risk)" in res.explanation


def test_external_provider_successful_call() -> None:
    def fake_llm(payload: dict[str, Any]) -> str:
        return f"Risk explanation for prob {payload['probability']:.2f}"

    provider = ExternalExplanationProvider(
        endpoint_fn=fake_llm,
        timeout_seconds=1.0,
    )
    req = ExplanationRequest(
        probability=0.75,
        threshold=0.5,
        decision="FRAUD",
        contributions={"V1": 0.2},
    )
    res = provider.explain(req)

    assert res.provider == "external_llm"
    assert res.fallback_triggered is False
    assert res.explanation == "Risk explanation for prob 0.75"


def test_external_provider_timeout_fallback() -> None:
    def slow_llm(_payload: dict[str, Any]) -> str:
        time.sleep(0.5)
        return "Should not arrive"

    provider = ExternalExplanationProvider(
        endpoint_fn=slow_llm,
        timeout_seconds=0.05,
    )
    req = ExplanationRequest(
        probability=0.9,
        threshold=0.5,
        decision="FRAUD",
        contributions={"V1": 0.5},
    )
    res = provider.explain(req)

    assert res.fallback_triggered is True
    assert "Timeout after 0.05s" in str(res.fallback_reason)
    assert res.provider == "template"
    assert "FRAUD" in res.explanation


def test_external_provider_exception_fallback() -> None:
    def broken_llm(_payload: dict[str, Any]) -> str:
        raise ConnectionResetError("Connection dropped by remote server")

    provider = ExternalExplanationProvider(
        endpoint_fn=broken_llm,
        timeout_seconds=1.0,
    )
    req = ExplanationRequest(
        probability=0.2,
        threshold=0.5,
        decision="LEGITIMATE",
        contributions={"V1": -0.1},
    )
    res = provider.explain(req)

    assert res.fallback_triggered is True
    assert "Connection dropped" in str(res.fallback_reason)
    assert res.provider == "template"
    assert "LEGITIMATE" in res.explanation


def test_external_provider_redacts_payload() -> None:
    captured_payload: dict[str, Any] = {}

    def inspect_payload(payload: dict[str, Any]) -> str:
        nonlocal captured_payload
        captured_payload = payload
        return "Explanation generated"

    provider = ExternalExplanationProvider(
        endpoint_fn=inspect_payload,
        redact_inputs=True,
    )
    req = ExplanationRequest(
        probability=0.8,
        threshold=0.5,
        decision="FRAUD",
        contributions={"V1": 0.3},
        features={
            "token": "secret_api_token",
            "card_number": "4532015112830366",
            "V1": 0.5,
        },
    )
    res = provider.explain(req)
    assert res.fallback_triggered is False
    assert captured_payload["features"]["token"] == "[REDACTED]"  # noqa: S105
    assert captured_payload["features"]["card_number"] == "[REDACTED]"
    assert captured_payload["features"]["V1"] == 0.5


def test_cost_controller_rate_and_token_limits() -> None:
    controller = CostController(max_requests_per_minute=2, max_tokens_budget=100)

    # First request
    ok1, reason1 = controller.check_and_record(tokens_used=60)
    assert ok1 is True
    assert reason1 is None

    # Second request exceeds budget
    ok2, reason2 = controller.check_and_record(tokens_used=50)
    assert ok2 is False
    assert reason2 == "Token budget exhausted"

    # Rate limiting on separate controller
    rate_controller = CostController(max_requests_per_minute=2)
    assert rate_controller.check_and_record()[0] is True
    assert rate_controller.check_and_record()[0] is True
    ok_blocked, reason_blocked = rate_controller.check_and_record()
    assert ok_blocked is False
    assert reason_blocked == "Rate limit exceeded (requests per minute)"

    # Test sliding window eviction of old timestamps
    rate_controller._timestamps.append(time.time() - 120.0)  # noqa: SLF001
    assert rate_controller.check_and_record()[0] is False  # cleans up old timestamp


def test_external_provider_cost_controller_blocking() -> None:
    blocked_controller = CostController(max_requests_per_minute=1)
    blocked_controller.check_and_record()  # consume the 1 allowed

    provider = ExternalExplanationProvider(
        endpoint_fn=lambda _p: "Not called",
        cost_controller=blocked_controller,
    )
    req = ExplanationRequest(
        probability=0.8,
        threshold=0.5,
        decision="FRAUD",
        contributions={"V1": 0.2},
    )
    res = provider.explain(req)
    assert res.fallback_triggered is True
    assert res.provider == "template"
    assert "Rate limit exceeded" in str(res.fallback_reason)


def test_external_provider_unredacted_and_batch() -> None:
    captured: list[dict[str, Any]] = []

    def record_payload(p: dict[str, Any]) -> str:
        captured.append(p)
        return "Custom batch result"

    provider = ExternalExplanationProvider(
        endpoint_fn=record_payload,
        redact_inputs=False,
    )
    reqs = [
        ExplanationRequest(
            probability=0.7,
            threshold=0.5,
            decision="FRAUD",
            contributions={"V1": 0.1},
            features={"token": "unmasked_secret"},
        )
    ]
    results = provider.explain_batch(reqs)
    assert len(results) == 1
    assert results[0].explanation == "Custom batch result"
    assert captured[0]["features"]["token"] == "unmasked_secret"  # noqa: S105


def test_model_explain_with_custom_provider() -> None:
    dataset = validate_frame(generate_synthetic_data(rows=500, fraud_rate=0.08, random_state=42))
    model = train_model(dataset)
    features = dataset.features.iloc[:2]
    probs = model.predict_probabilities(features)

    class CustomProvider:
        name = "custom_test"

        def explain(self, request: ExplanationRequest) -> ExplanationResult:
            return ExplanationResult(
                explanation=f"Custom reason for {request.decision}",
                provider=self.name,
            )

        def explain_batch(
            self, requests: list[ExplanationRequest]
        ) -> list[ExplanationResult]:
            return [self.explain(r) for r in requests]

    exps = model.explain_local_natural_language(
        features,
        probs,
        provider=CustomProvider(),  # type: ignore[arg-type]
    )
    assert len(exps) == 2
    assert all("Custom reason for" in exp for exp in exps)


def test_api_predict_with_custom_explanation_provider() -> None:
    dataset = validate_frame(generate_synthetic_data(rows=500, fraud_rate=0.08, random_state=42))
    model = train_model(dataset)

    class FastProvider:
        name = "fast_custom"

        def explain(self, request: ExplanationRequest) -> ExplanationResult:
            return ExplanationResult(
                explanation=f"API Custom: {request.decision}",
                provider=self.name,
            )

        def explain_batch(
            self, requests: list[ExplanationRequest]
        ) -> list[ExplanationResult]:
            return [self.explain(r) for r in requests]

    app = create_app(
        model=model,
        explanation_provider=FastProvider(),  # type: ignore[arg-type]
    )
    records = dataset.features.iloc[:2].to_dict(orient="records")

    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/predict",
            json={"transactions": records, "explain_llm": True},
        )
        assert response.status_code == 200
        preds = response.json()["predictions"]
        assert len(preds) == 2
        for p in preds:
            assert p["explanation"].startswith("API Custom:")
