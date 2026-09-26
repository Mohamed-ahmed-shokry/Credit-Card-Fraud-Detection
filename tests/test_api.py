from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_detection import __version__
from fraud_detection.api import (
    API_KEYS_ENVIRONMENT_VARIABLE,
    AUDIT_LOG_ENVIRONMENT_VARIABLE,
    CIRCUIT_BREAKER_ENABLED_ENVIRONMENT_VARIABLE,
    CIRCUIT_BREAKER_FAILURE_THRESHOLD_ENVIRONMENT_VARIABLE,
    CIRCUIT_BREAKER_LATENCY_BUDGET_ENVIRONMENT_VARIABLE,
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT_ENVIRONMENT_VARIABLE,
    DEGRADED_MODE_ENVIRONMENT_VARIABLE,
    ENABLE_CHAOS_HEADER_ENVIRONMENT_VARIABLE,
    MAX_CONCURRENT_SCORING_ENVIRONMENT_VARIABLE,
    MAX_REQUEST_BODY_BYTES,
    MODEL_PATH_ENVIRONMENT_VARIABLE,
    PROCESS_TIME_HEADER,
    RATE_LIMIT_REQUESTS_ENVIRONMENT_VARIABLE,
    RATE_LIMIT_WINDOW_SECONDS_ENVIRONMENT_VARIABLE,
    REQUEST_ID_HEADER,
    SHADOW_MODEL_PATH_ENVIRONMENT_VARIABLE,
    CircuitBreaker,
    _api_key_is_valid,
    app_from_environment,
    create_app,
)
from fraud_detection.audit import JsonlAuditSink
from fraud_detection.data import ValidatedDataset, generate_synthetic_data, validate_frame
from fraud_detection.model import FraudModel, save_model, train_model, validate_artifact
from fraud_detection.signing import load_private_key, load_public_key, write_keypair
from fraud_detection.telemetry import TraceContext, TraceSpan
from fraud_detection.trust import TrustBundle, write_trust_bundle


class _RecordingTraceExporter:
    def __init__(self) -> None:
        self.spans: list[TraceSpan] = []

    def export(self, span: TraceSpan) -> None:
        self.spans.append(span)


class _FailingTraceExporter:
    def export(self, _span: TraceSpan) -> None:
        raise RuntimeError("collector unavailable")


@pytest.fixture(scope="module")
def api_context() -> tuple[TestClient, FraudModel, ValidatedDataset]:
    dataset = validate_frame(generate_synthetic_data(rows=800, fraud_rate=0.1))
    model = train_model(dataset)
    return TestClient(create_app(model=model)), model, dataset


@pytest.fixture(scope="module")
def client(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> Iterator[TestClient]:
    test_client, _, _ = api_context
    with test_client:
        yield test_client


def test_health_reports_loaded_model(
    client: TestClient,
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, _ = api_context

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service_version": __version__,
        "model_created_at": model.metadata["created_at"],
        "feature_count": 30,
        "threshold": model.threshold,
    }


def test_live_reports_process_liveness(client: TestClient) -> None:
    response = client.get("/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive", "service_version": __version__}


def test_ready_reports_loaded_model(
    client: TestClient,
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, _ = api_context

    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["service_version"] == __version__
    assert body["model_created_at"] == model.metadata["created_at"]
    assert body["feature_count"] == 30
    assert body["threshold"] == model.threshold


def test_ready_returns_503_when_circuit_open_with_raise_fallback(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, _ = api_context
    cb = CircuitBreaker(failure_threshold=1)
    cb.trip()
    app = create_app(model=model, circuit_breaker=cb, fallback_mode="raise")

    with TestClient(app) as test_client:
        response = test_client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert "Circuit breaker is open" in body["detail"]
    assert body["circuit_breaker"]["state"] == "open"


def test_ready_stays_available_with_open_circuit_and_constant_fallback(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, _ = api_context
    cb = CircuitBreaker(failure_threshold=1)
    cb.trip()
    app = create_app(model=model, circuit_breaker=cb, fallback_mode="constant")

    with TestClient(app) as test_client:
        response = test_client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["degraded_mode"] is True
    assert body["circuit_breaker"]["state"] == "open"


def test_traceparent_is_continued_and_span_is_exported(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, _ = api_context
    exporter = _RecordingTraceExporter()
    incoming = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"

    with TestClient(create_app(model=model, trace_exporter=exporter)) as trace_client:
        response = trace_client.get("/health", headers={"traceparent": incoming})

    assert response.status_code == 200
    returned_context = TraceContext.from_traceparent(response.headers["traceparent"])
    assert returned_context.trace_id == "0123456789abcdef0123456789abcdef"
    assert returned_context.span_id != "0123456789abcdef"
    assert len(exporter.spans) == 1
    span = exporter.spans[0]
    assert span.context == returned_context
    assert span.parent_span_id == "0123456789abcdef"
    assert span.attributes["http.response.status_code"] == 200
    assert span.attributes["fraud.model_version"] == model.metadata["dataset_fingerprint"][:12]


def test_invalid_traceparent_starts_new_trace_and_export_failure_is_isolated(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, _ = api_context

    with TestClient(create_app(model=model, trace_exporter=_FailingTraceExporter())) as client:
        response = client.get("/health", headers={"traceparent": "invalid"})

    assert response.status_code == 200
    context = TraceContext.from_traceparent(response.headers["traceparent"])
    assert context.trace_id != "invalid"


def test_otlp_environment_configures_optional_exporter(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, _ = api_context
    configured: dict[str, object] = {}

    class _ConfiguredExporter(_RecordingTraceExporter):
        def __init__(
            self,
            endpoint: str,
            *,
            service_name: str,
            timeout_seconds: float,
        ) -> None:
            super().__init__()
            configured.update(
                endpoint=endpoint,
                service_name=service_name,
                timeout_seconds=timeout_seconds,
            )

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://collector:4318")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "fraud-api")
    monkeypatch.setenv("FRAUD_OTLP_TIMEOUT_SECONDS", "0.25")
    monkeypatch.setattr("fraud_detection.api.OTLPHttpTraceExporter", _ConfiguredExporter)

    with TestClient(create_app(model=model)) as configured_client:
        response = configured_client.get("/health")

    assert response.status_code == 200
    assert configured == {
        "endpoint": "http://collector:4318",
        "service_name": "fraud-api",
        "timeout_seconds": 0.25,
    }


def test_admission_gate_runs_before_model_load(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    tmp_path: Path,
) -> None:
    _, model, _ = api_context
    signed_model = deepcopy(model)
    signed_model.metadata["lineage"].update(
        {"git_commit": "b" * 40, "git_repository": "https://example.test/admission.git"}
    )
    artifact_path = tmp_path / "artifact"
    save_model(signed_model, artifact_path)
    private_key_path = tmp_path / "private.pem"
    public_key_path = tmp_path / "public.pem"
    write_keypair(private_key_path, public_key_path)
    report = validate_artifact(artifact_path, strict=True)
    attestation_path = tmp_path / "attestation.json"
    attestation_path.write_text(
        json.dumps(
            report.to_attestation(signing_key=load_private_key(private_key_path)),
            indent=2,
        ),
        encoding="utf-8",
    )
    bundle_path = tmp_path / "trust-bundle.json"
    write_trust_bundle(
        bundle_path,
        TrustBundle.from_public_keys((load_public_key(public_key_path),)),
    )

    with TestClient(
        create_app(
            model_path=artifact_path,
            attestation_path=attestation_path,
            trust_bundle_path=bundle_path,
        )
    ) as admitted_client:
        assert admitted_client.get("/health").status_code == 200

    tampered = json.loads(attestation_path.read_text(encoding="utf-8"))
    tampered["status"] = "FAILED"
    attestation_path.write_text(json.dumps(tampered), encoding="utf-8")
    blocked_app = create_app(
        model_path=tmp_path / "missing-model",
        attestation_path=attestation_path,
        trust_bundle_path=bundle_path,
    )
    with pytest.raises(RuntimeError, match="Deployment admission failed"), TestClient(blocked_app):
        pass


def test_admission_paths_must_be_configured_together(api_context) -> None:
    _, model, _ = api_context
    with pytest.raises(ValueError, match="must be configured together"):
        create_app(model=model, attestation_path="attestation.json")


def test_predict_scores_ordered_batch(
    client: TestClient,
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    records = dataset.features.iloc[:3].to_dict(orient="records")

    response = client.post("/v1/predict", json={"transactions": records})

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == model.metadata["dataset_fingerprint"][:12]
    assert body["threshold"] == model.threshold
    assert len(body["predictions"]) == 3
    assert all(0 <= item["fraud_probability"] <= 1 for item in body["predictions"])
    assert all(isinstance(item["is_fraud"], bool) for item in body["predictions"])


def test_predict_defaults_to_tuned_threshold(
    client: TestClient,
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    records = dataset.features.iloc[:3].to_dict(orient="records")

    response = client.post("/v1/predict", json={"transactions": records})

    assert response.status_code == 200
    body = response.json()
    assert body["threshold"] == model.threshold
    assert body["model_threshold"] == model.threshold


def test_predict_supports_threshold_override(
    client: TestClient,
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    records = dataset.features.iloc[:3].to_dict(orient="records")

    response = client.post("/v1/predict", json={"transactions": records, "threshold": 0.0})

    assert response.status_code == 200
    body = response.json()
    assert body["threshold"] == 0.0
    assert body["model_threshold"] == model.threshold
    assert all(item["is_fraud"] for item in body["predictions"])


@pytest.mark.parametrize("threshold", [-0.1, 1.5, "high"])
def test_predict_rejects_invalid_threshold_override(
    client: TestClient,
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    threshold: object,
) -> None:
    _, _, dataset = api_context
    records = dataset.features.iloc[:1].to_dict(orient="records")

    response = client.post("/v1/predict", json={"transactions": records, "threshold": threshold})

    assert response.status_code == 422


def test_score_returns_single_prediction(
    client: TestClient,
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    record = dataset.features.iloc[0].to_dict()

    response = client.post("/v1/score", json={"transaction": record})

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == model.metadata["dataset_fingerprint"][:12]
    assert body["threshold"] == model.threshold
    assert body["model_threshold"] == model.threshold
    assert 0 <= body["prediction"]["fraud_probability"] <= 1
    assert isinstance(body["prediction"]["is_fraud"], bool)
    assert body["prediction"]["contributions"] is None


def test_score_supports_explain_and_threshold_override(
    client: TestClient,
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    record = dataset.features.iloc[0].to_dict()

    response = client.post(
        "/v1/score",
        json={
            "transaction": record,
            "explain": True,
            "explain_llm": True,
            "threshold": 0.0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["threshold"] == 0.0
    assert body["model_threshold"] == model.threshold
    assert body["prediction"]["is_fraud"] is True
    assert set(body["prediction"]["contributions"]) == set(model.feature_names)
    assert "Top contributing factors:" in body["prediction"]["explanation"]


def test_predict_supports_natural_language_explanation(
    client: TestClient,
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, _, dataset = api_context
    records = dataset.features.iloc[:2].to_dict(orient="records")

    response = client.post(
        "/v1/predict",
        json={"transactions": records, "explain_llm": True, "threshold": 0.0},
    )

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert all(item["is_fraud"] for item in predictions)
    assert all("Top contributing factors:" in item["explanation"] for item in predictions)
    assert all(item["contributions"] is None for item in predictions)


@pytest.mark.parametrize(
    "payload",
    [
        {"transaction": {"wrong": 1.0}},
        {"transaction": {}, "threshold": 1.5},
        {"transactions": []},
    ],
)
def test_score_rejects_invalid_requests(client: TestClient, payload: dict[str, object]) -> None:
    response = client.post("/v1/score", json=payload)

    assert response.status_code == 422


def test_score_counts_toward_prediction_metrics(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    single = dataset.features.iloc[:1].to_dict(orient="records")[0]
    isolated_app = create_app(model=model)

    with TestClient(isolated_app) as isolated_client:
        isolated_client.post("/v1/score", json={"transaction": single})
        response = isolated_client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert 'http_requests_total{method="POST",path="/v1/score",status_code="200"} 1.0' in body
    fraud_count = float(
        next(
            line.rsplit(" ", 1)[1]
            for line in body.splitlines()
            if line.startswith('fraud_predictions_total{is_fraud="true"}')
        )
    )
    legitimate_count = float(
        next(
            line.rsplit(" ", 1)[1]
            for line in body.splitlines()
            if line.startswith('fraud_predictions_total{is_fraud="false"}')
        )
    )
    assert fraud_count + legitimate_count == 1


def test_predict_returns_actionable_schema_error(client: TestClient) -> None:
    response = client.post("/v1/predict", json={"transactions": [{"wrong": 1.0}]})

    assert response.status_code == 422
    assert "Input schema does not match" in response.json()["detail"]


@pytest.mark.parametrize("invalid_value", ["1.25", True, float("inf")])
def test_predict_rejects_coerced_or_non_finite_feature_values(
    client: TestClient,
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    invalid_value: object,
) -> None:
    _, _, dataset = api_context
    transaction = dataset.features.iloc[0].to_dict()
    transaction["V1"] = invalid_value

    payload = {"transactions": [transaction]}
    if invalid_value == float("inf"):
        response = client.post(
            "/v1/predict",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
    else:
        response = client.post("/v1/predict", json=payload)

    assert response.status_code == 422
    assert all("input" not in error for error in response.json()["detail"])


@pytest.mark.parametrize(
    "payload",
    [
        {"transactions": []},
        {"transactions": [{"V1": "not-a-number"}]},
        {"transactions": [{}]},
        {"transactions": [{"V1": 1.0}], "unexpected": True},
    ],
)
def test_predict_rejects_invalid_request_shape(client: TestClient, payload: object) -> None:
    response = client.post("/v1/predict", json=payload)

    assert response.status_code == 422


def test_api_rejects_declared_oversized_body_with_request_context(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/predict",
        content=b"{}",
        headers={
            "Content-Length": str(MAX_REQUEST_BODY_BYTES + 1),
            "Content-Type": "application/json",
            REQUEST_ID_HEADER: "oversized-request",
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"].endswith("-byte limit.")
    assert response.headers[REQUEST_ID_HEADER] == "oversized-request"
    assert float(response.headers[PROCESS_TIME_HEADER]) >= 0


@pytest.mark.parametrize("content_length", ["not-a-number", "-1"])
def test_api_rejects_malformed_content_length_header(
    client: TestClient,
    content_length: str,
) -> None:
    response = client.post(
        "/v1/predict",
        content=b"{}",
        headers={"Content-Length": content_length, "Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid Content-Length."


def test_api_logs_and_answers_unhandled_errors_with_correlation_headers(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, model, dataset = api_context

    class _BrokenModel:
        feature_names = model.feature_names
        threshold = model.threshold
        metadata = model.metadata

        def validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
            return model.validate_features(features)

        def predict_probabilities(self, _features: object) -> None:
            raise RuntimeError("boom")

    broken_app = create_app(model=cast(FraudModel, _BrokenModel()))
    with (
        caplog.at_level(logging.ERROR, logger="fraud_detection.api"),
        TestClient(broken_app) as broken_client,
    ):
        response = broken_client.post(
            "/v1/predict",
            json={"transactions": dataset.features.iloc[:1].to_dict(orient="records")},
            headers={REQUEST_ID_HEADER: "failed-request"},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert response.headers[REQUEST_ID_HEADER] == "failed-request"
    assert float(response.headers[PROCESS_TIME_HEADER]) >= 0
    assert response.headers["traceparent"].startswith("00-")
    assert "request_failed" in caplog.text
    assert "failed-request" in caplog.text
    assert "boom" not in response.text


def test_metrics_endpoint_reports_request_and_prediction_counters(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    isolated_app = create_app(model=model)

    with TestClient(isolated_app) as isolated_client:
        isolated_client.get("/health")
        isolated_client.post(
            "/v1/predict",
            json={"transactions": dataset.features.iloc[:5].to_dict(orient="records")},
        )
        response = isolated_client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert 'http_requests_total{method="GET",path="/health",status_code="200"} 1.0' in body
    assert 'http_requests_total{method="POST",path="/v1/predict",status_code="200"} 1.0' in body
    fraud_count = float(
        next(
            line.rsplit(" ", 1)[1]
            for line in body.splitlines()
            if line.startswith('fraud_predictions_total{is_fraud="true"}')
        )
    )
    legitimate_count = float(
        next(
            line.rsplit(" ", 1)[1]
            for line in body.splitlines()
            if line.startswith('fraud_predictions_total{is_fraud="false"}')
        )
    )
    assert fraud_count + legitimate_count == 5


def test_metrics_endpoint_records_unhandled_errors_as_500(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context

    class _BrokenModel:
        feature_names = model.feature_names
        threshold = model.threshold
        metadata = model.metadata

        def validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
            return model.validate_features(features)

        def predict_probabilities(self, _features: object) -> None:
            raise RuntimeError("boom")

    broken_app = create_app(model=cast(FraudModel, _BrokenModel()))
    with TestClient(broken_app, raise_server_exceptions=False) as broken_client:
        broken_client.post(
            "/v1/predict",
            json={"transactions": dataset.features.iloc[:1].to_dict(orient="records")},
        )
        response = broken_client.get("/metrics")

    assert 'http_requests_total{method="POST",path="/v1/predict",status_code="500"} 1.0' in (
        response.text
    )


def test_api_rejects_chunked_body_that_crosses_limit(client: TestClient) -> None:
    def oversized_body() -> Iterator[bytes]:
        yield b'{"transactions":[{"x":"'
        yield b"x" * MAX_REQUEST_BODY_BYTES
        yield b'"}]}'

    response = client.post(
        "/v1/predict",
        content=oversized_body(),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413


def test_openapi_describes_versioned_prediction_endpoint(client: TestClient) -> None:
    document = client.get("/openapi.json").json()

    assert document["info"]["version"] == __version__
    assert "/v1/predict" in document["paths"]
    assert "/v1/score" in document["paths"]
    assert "/live" in document["paths"]
    assert "/ready" in document["paths"]


def test_request_context_propagates_safe_correlation_id(client: TestClient) -> None:
    response = client.get("/health", headers={REQUEST_ID_HEADER: "audit-request_123"})

    assert response.headers[REQUEST_ID_HEADER] == "audit-request_123"
    assert float(response.headers[PROCESS_TIME_HEADER]) >= 0


def test_request_context_replaces_unsafe_correlation_id(client: TestClient) -> None:
    response = client.get("/health", headers={REQUEST_ID_HEADER: "unsafe id value"})

    generated_request_id = response.headers[REQUEST_ID_HEADER]
    assert generated_request_id != "unsafe id value"
    assert len(generated_request_id) == 32
    assert generated_request_id.isalnum()


def test_environment_factory_loads_persisted_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, _ = api_context
    artifact_directory = tmp_path / "artifact"
    save_model(model, artifact_directory)
    monkeypatch.setenv(MODEL_PATH_ENVIRONMENT_VARIABLE, str(artifact_directory))

    with TestClient(app_from_environment()) as environment_client:
        response = environment_client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_created_at"] == model.metadata["created_at"]


def test_environment_factory_requires_model_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(MODEL_PATH_ENVIRONMENT_VARIABLE, raising=False)

    with (
        pytest.raises(RuntimeError, match="No model configured"),
        TestClient(app_from_environment()),
    ):
        pass


def test_predict_endpoint_supports_explain_parameter(
    client: TestClient,
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    records = dataset.features.iloc[:3].to_dict(orient="records")

    response = client.post("/v1/predict", json={"transactions": records, "explain": True})

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 3
    for prediction in body["predictions"]:
        assert "contributions" in prediction
        assert prediction["contributions"] is not None
        assert isinstance(prediction["contributions"], dict)
        assert set(prediction["contributions"].keys()) == set(model.feature_names)


def test_predict_endpoint_works_without_explain_parameter(
    client: TestClient,
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, _model, dataset = api_context
    records = dataset.features.iloc[:3].to_dict(orient="records")

    response = client.post("/v1/predict", json={"transactions": records})

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 3
    for prediction in body["predictions"]:
        assert "contributions" in prediction
        assert prediction["contributions"] is None


def test_api_key_middleware_allows_valid_key(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    records = dataset.features.iloc[:1].to_dict(orient="records")
    app = create_app(model=model, api_keys=["valid-key"])

    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/predict",
            json={"transactions": records},
            headers={"X-API-Key": "valid-key"},
        )

    assert response.status_code == 200


def test_api_key_middleware_rejects_invalid_key(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    records = dataset.features.iloc[:1].to_dict(orient="records")
    app = create_app(model=model, api_keys=["valid-key"])

    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/predict",
            json={"transactions": records},
            headers={"X-API-Key": "invalid-key"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_api_key_middleware_rejects_missing_key(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    records = dataset.features.iloc[:1].to_dict(orient="records")
    app = create_app(model=model, api_keys=["valid-key"])

    with TestClient(app) as test_client:
        response = test_client.post("/v1/predict", json={"transactions": records})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_api_key_middleware_exempts_operational_endpoints(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, _ = api_context
    app = create_app(model=model, api_keys=["valid-key"])

    with TestClient(app) as test_client:
        health = test_client.get("/health")
        metrics = test_client.get("/metrics")
        live = test_client.get("/live")
        ready = test_client.get("/ready")

    assert health.status_code == 200
    assert metrics.status_code == 200
    assert live.status_code == 200
    assert ready.status_code == 200


def test_api_key_validation_semantics() -> None:
    accepted = {"first-key", "second-key"}

    assert _api_key_is_valid("first-key", accepted) is True
    assert _api_key_is_valid("second-key", accepted) is True
    assert _api_key_is_valid("wrong-key", accepted) is False
    assert _api_key_is_valid(None, accepted) is False
    assert _api_key_is_valid("", accepted) is False
    assert _api_key_is_valid("first-key", set()) is False


def test_api_key_comparison_checks_every_configured_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import secrets as real_secrets

    original_compare_digest = real_secrets.compare_digest
    comparisons: list[tuple[bytes, bytes]] = []

    def counting_compare_digest(a: bytes, b: bytes) -> bool:
        comparisons.append((a, b))
        return original_compare_digest(a, b)

    monkeypatch.setattr("fraud_detection.api.secrets.compare_digest", counting_compare_digest)

    assert _api_key_is_valid("second-key", {"first-key", "second-key", "third-key"}) is True
    assert len(comparisons) == 3


def _slow_wrapper(
    model: FraudModel,
    *,
    started: threading.Event | None = None,
    release: threading.Event | None = None,
    delay_seconds: float = 0.0,
) -> MagicMock:
    wrapped = MagicMock(wraps=model)
    wrapped.metadata = model.metadata
    wrapped.threshold = model.threshold
    wrapped.feature_names = model.feature_names

    def slow_predict(frame: Any) -> Any:
        if started is not None:
            started.set()
        if release is not None:
            assert release.wait(timeout=10), "scoring release event was never set"
        elif delay_seconds > 0:
            time.sleep(delay_seconds)
        return model.predict_probabilities(frame)

    wrapped.predict_probabilities.side_effect = slow_predict
    return wrapped


def test_operational_probes_stay_responsive_during_scoring(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    record = dataset.features.iloc[0].to_dict()
    started = threading.Event()
    release = threading.Event()
    slow_model = _slow_wrapper(model, started=started, release=release)
    app = create_app(model=slow_model)

    with TestClient(app) as test_client, ThreadPoolExecutor(max_workers=1) as pool:
        scoring = pool.submit(
            test_client.post,
            "/v1/predict",
            json={"transactions": [record]},
        )
        assert started.wait(timeout=10), "scoring never started"
        live = test_client.get("/live")
        ready = test_client.get("/ready")
        health = test_client.get("/health")
        release.set()
        response = scoring.result(timeout=15)

    assert live.status_code == 200
    assert ready.status_code == 200
    assert health.status_code == 200
    assert response.status_code == 200


def test_concurrent_scoring_requests_overlap_in_threadpool(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    record = dataset.features.iloc[0].to_dict()
    lock = threading.Lock()
    active = 0
    max_active = 0

    wrapped = MagicMock(wraps=model)
    wrapped.metadata = model.metadata
    wrapped.threshold = model.threshold
    wrapped.feature_names = model.feature_names

    def tracked_predict(frame: Any) -> Any:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.2)
        with lock:
            active -= 1
        return model.predict_probabilities(frame)

    wrapped.predict_probabilities.side_effect = tracked_predict
    app = create_app(model=wrapped)

    with TestClient(app) as test_client, ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            test_client.post,
            "/v1/score",
            json={"transaction": record},
        )
        second = pool.submit(
            test_client.post,
            "/v1/score",
            json={"transaction": record},
        )
        responses = [first.result(timeout=15), second.result(timeout=15)]

    assert [r.status_code for r in responses] == [200, 200]
    assert max_active >= 2, "scoring requests did not overlap on the threadpool"


def test_slow_audit_emit_does_not_block_probes(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    from fraud_detection.audit import AuditEvent, AuditSink

    _, model, dataset = api_context
    record = dataset.features.iloc[0].to_dict()
    emit_started = threading.Event()
    emit_release = threading.Event()

    class SlowSink:
        def emit(self, _event: AuditEvent) -> None:
            emit_started.set()
            assert emit_release.wait(timeout=10), "audit emit release was never set"

        def close(self) -> None:
            return None

    sink = SlowSink()
    app = create_app(model=model, audit_sink=cast(AuditSink, sink))

    with TestClient(app) as test_client, ThreadPoolExecutor(max_workers=1) as pool:
        scoring = pool.submit(
            test_client.post,
            "/v1/predict",
            json={"transactions": [record]},
        )
        assert emit_started.wait(timeout=10), "audit emit never started"
        live = test_client.get("/live")
        emit_release.set()
        response = scoring.result(timeout=15)

    assert live.status_code == 200
    assert response.status_code == 200


def test_concurrency_cap_sheds_load_with_503(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    record = dataset.features.iloc[0].to_dict()
    started = threading.Event()
    release = threading.Event()
    slow_model = _slow_wrapper(model, started=started, release=release)
    app = create_app(model=slow_model, max_concurrent_scoring=1)

    with TestClient(app) as test_client, ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(
            test_client.post,
            "/v1/predict",
            json={"transactions": [record]},
        )
        assert started.wait(timeout=10), "first scoring never started"
        rejected = test_client.post("/v1/predict", json={"transactions": [record]})
        live = test_client.get("/live")
        release.set()
        accepted = first.result(timeout=15)

    assert accepted.status_code == 200
    assert rejected.status_code == 503
    assert rejected.json()["detail"] == "Too many concurrent scoring requests."
    assert rejected.headers["Retry-After"] == "1"
    assert live.status_code == 200


def test_concurrency_cap_reads_environment(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, dataset = api_context
    record = dataset.features.iloc[0].to_dict()
    started = threading.Event()
    release = threading.Event()
    slow_model = _slow_wrapper(model, started=started, release=release)
    monkeypatch.setenv(MAX_CONCURRENT_SCORING_ENVIRONMENT_VARIABLE, "1")
    app = create_app(model=slow_model)

    with TestClient(app) as test_client, ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(
            test_client.post,
            "/v1/predict",
            json={"transactions": [record]},
        )
        assert started.wait(timeout=10), "first scoring never started"
        rejected = test_client.post("/v1/score", json={"transaction": record})
        release.set()
        accepted = first.result(timeout=15)

    assert accepted.status_code == 200
    assert rejected.status_code == 503


def test_invalid_concurrency_configuration_is_rejected(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, _ = api_context

    with pytest.raises(ValueError, match="max_concurrent_scoring must be >= 0"):
        create_app(model=model, max_concurrent_scoring=-1)

    monkeypatch.setenv(MAX_CONCURRENT_SCORING_ENVIRONMENT_VARIABLE, "not-a-number")
    with pytest.raises(ValueError, match=MAX_CONCURRENT_SCORING_ENVIRONMENT_VARIABLE):
        create_app(model=model)

    monkeypatch.setenv(MAX_CONCURRENT_SCORING_ENVIRONMENT_VARIABLE, "-2")
    with pytest.raises(ValueError, match=">= 0"):
        create_app(model=model)


def test_rate_limit_middleware_allows_within_limit(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    records = dataset.features.iloc[:1].to_dict(orient="records")
    app = create_app(model=model, rate_limit_requests=5, rate_limit_window_seconds=60)

    with TestClient(app) as test_client:
        for expected_remaining in (4, 3, 2, 1, 0):
            response = test_client.post("/v1/predict", json={"transactions": records})
            assert response.status_code == 200
            assert response.headers["X-RateLimit-Limit"] == "5"
            assert response.headers["X-RateLimit-Remaining"] == str(expected_remaining)


def test_rate_limit_middleware_rejects_over_limit(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    records = dataset.features.iloc[:1].to_dict(orient="records")
    app = create_app(model=model, rate_limit_requests=2, rate_limit_window_seconds=60)

    with TestClient(app) as test_client:
        response1 = test_client.post("/v1/predict", json={"transactions": records})
        assert response1.status_code == 200
        response2 = test_client.post("/v1/predict", json={"transactions": records})
        assert response2.status_code == 200
        response3 = test_client.post("/v1/predict", json={"transactions": records})
        assert response3.status_code == 429
        assert response3.json()["detail"] == "Rate limit exceeded"
        assert response3.headers["X-RateLimit-Limit"] == "2"
        assert response3.headers["X-RateLimit-Remaining"] == "0"
        assert int(response3.headers["Retry-After"]) >= 0


def test_rate_limit_middleware_exempts_operational_endpoints(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, _ = api_context
    app = create_app(model=model, rate_limit_requests=1, rate_limit_window_seconds=60)

    with TestClient(app) as test_client:
        first = test_client.get("/health")
        second = test_client.get("/health")
        live = test_client.get("/live")
        ready = test_client.get("/ready")
        metrics = test_client.get("/metrics")

    assert first.status_code == 200
    assert second.status_code == 200
    assert live.status_code == 200
    assert ready.status_code == 200
    assert metrics.status_code == 200
    assert "X-RateLimit-Limit" not in first.headers


def test_environment_configures_api_keys(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, dataset = api_context
    records = dataset.features.iloc[:1].to_dict(orient="records")
    monkeypatch.setenv(API_KEYS_ENVIRONMENT_VARIABLE, "env-key-1, env-key-2")

    app = create_app(model=model)
    with TestClient(app) as test_client:
        rejected = test_client.post("/v1/predict", json={"transactions": records})
        accepted = test_client.post(
            "/v1/predict",
            json={"transactions": records},
            headers={"X-API-Key": "env-key-2"},
        )

    assert rejected.status_code == 401
    assert accepted.status_code == 200


def test_explicit_api_keys_override_environment(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, dataset = api_context
    records = dataset.features.iloc[:1].to_dict(orient="records")
    monkeypatch.setenv(API_KEYS_ENVIRONMENT_VARIABLE, "env-key")

    app = create_app(model=model, api_keys=["explicit-key"])
    with TestClient(app) as test_client:
        rejected = test_client.post(
            "/v1/predict",
            json={"transactions": records},
            headers={"X-API-Key": "env-key"},
        )
        accepted = test_client.post(
            "/v1/predict",
            json={"transactions": records},
            headers={"X-API-Key": "explicit-key"},
        )

    assert rejected.status_code == 401
    assert accepted.status_code == 200


def test_environment_configures_rate_limiting(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, dataset = api_context
    records = dataset.features.iloc[:1].to_dict(orient="records")
    monkeypatch.setenv(RATE_LIMIT_REQUESTS_ENVIRONMENT_VARIABLE, "1")
    monkeypatch.setenv(RATE_LIMIT_WINDOW_SECONDS_ENVIRONMENT_VARIABLE, "60")

    app = create_app(model=model)
    with TestClient(app) as test_client:
        first = test_client.post("/v1/predict", json={"transactions": records})
        second = test_client.post("/v1/predict", json={"transactions": records})

    assert first.status_code == 200
    assert first.headers["X-RateLimit-Limit"] == "1"
    assert second.status_code == 429


def test_invalid_rate_limit_environment_is_rejected(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, _ = api_context
    monkeypatch.setenv(RATE_LIMIT_REQUESTS_ENVIRONMENT_VARIABLE, "not-a-number")

    with pytest.raises(ValueError, match=RATE_LIMIT_REQUESTS_ENVIRONMENT_VARIABLE):
        create_app(model=model)


def test_negative_rate_limit_requests_environment_is_rejected(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, _ = api_context
    monkeypatch.setenv(RATE_LIMIT_REQUESTS_ENVIRONMENT_VARIABLE, "-1")

    with pytest.raises(ValueError, match=">= 0"):
        create_app(model=model)


def test_invalid_rate_limit_window_environment_is_rejected(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, _ = api_context
    monkeypatch.setenv(RATE_LIMIT_WINDOW_SECONDS_ENVIRONMENT_VARIABLE, "not-a-number")

    with pytest.raises(ValueError, match=RATE_LIMIT_WINDOW_SECONDS_ENVIRONMENT_VARIABLE):
        create_app(model=model)


def test_non_positive_rate_limit_window_environment_is_rejected(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, _ = api_context
    monkeypatch.setenv(RATE_LIMIT_WINDOW_SECONDS_ENVIRONMENT_VARIABLE, "0")

    with pytest.raises(ValueError, match="> 0"):
        create_app(model=model)


def test_explicit_rate_limit_window_overrides_environment(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, _ = api_context
    monkeypatch.setenv(RATE_LIMIT_WINDOW_SECONDS_ENVIRONMENT_VARIABLE, "not-a-number")

    app = create_app(model=model, rate_limit_requests=1, rate_limit_window_seconds=15.0)

    assert app is not None


def test_ready_returns_503_when_model_missing(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, _ = api_context

    with TestClient(create_app(model=model)) as test_client:
        del test_client.app.state.model
        response = test_client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["detail"] == "Model not loaded."


def test_predict_emits_audit_event_to_configured_sink(
    tmp_path: Path,
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    records = dataset.features.iloc[:2].to_dict(orient="records")
    audit_file = tmp_path / "audit" / "scoring.jsonl"
    sink = JsonlAuditSink(audit_file)
    app = create_app(model=model, audit_sink=sink)

    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/predict",
            json={"transactions": records, "explain": True},
            headers={"X-Request-ID": "test-req-123"},
        )
        assert response.status_code == 200

    assert audit_file.exists()
    lines = audit_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event_type"] == "scoring"
    assert event["model_version"] == str(model.metadata["dataset_fingerprint"])[:12]
    assert event["payload"]["batch_size"] == 2
    assert event["payload"]["request_id"] == "test-req-123"
    assert len(event["payload"]["predictions"]) == 2


def test_predict_emits_audit_event_via_environment_variable(
    tmp_path: Path,
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, dataset = api_context
    audit_file = tmp_path / "env_audit.jsonl"
    monkeypatch.setenv(AUDIT_LOG_ENVIRONMENT_VARIABLE, str(audit_file))
    records = dataset.features.iloc[:1].to_dict(orient="records")
    app = create_app(model=model)

    with TestClient(app) as test_client:
        response = test_client.post("/v1/predict", json={"transactions": records})
        assert response.status_code == 200

    assert audit_file.exists()
    lines = audit_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event_type"] == "scoring"
    assert event["payload"]["batch_size"] == 1


def test_score_emits_audit_event_to_configured_sink(
    tmp_path: Path,
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    record = dataset.features.iloc[0].to_dict()
    audit_file = tmp_path / "score_audit.jsonl"
    sink = JsonlAuditSink(audit_file)
    app = create_app(model=model, audit_sink=sink)

    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/score",
            json={"transaction": record},
            headers={"X-Request-ID": "single-score-999"},
        )
        assert response.status_code == 200

    assert audit_file.exists()
    lines = audit_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event_type"] == "scoring"
    assert event["payload"]["batch_size"] == 1
    assert event["payload"]["request_id"] == "single-score-999"


def test_predict_and_score_gracefully_handle_audit_sink_failure(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context

    class FailingAuditSink:
        def emit(self, _event: object) -> None:
            raise RuntimeError("Audit storage unavailable")

        def close(self) -> None:
            pass

    app = create_app(model=model, audit_sink=FailingAuditSink())
    record = dataset.features.iloc[0].to_dict()

    with TestClient(app) as test_client:
        pred_resp = test_client.post("/v1/predict", json={"transactions": [record]})
        assert pred_resp.status_code == 200
        score_resp = test_client.post("/v1/score", json={"transaction": record})
        assert score_resp.status_code == 200


def test_fallback_mode_constant_on_model_error(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    record = dataset.features.iloc[0].to_dict()

    class FaultyModel:
        threshold = 0.5
        feature_names = model.feature_names
        metadata = model.metadata

        def validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
            return model.validate_features(features)

        def predict_probabilities(self, _frame: Any) -> list[float]:
            raise RuntimeError("Underlying estimator corrupted")

    app = create_app(
        model=FaultyModel(),  # type: ignore[arg-type]
        fallback_mode="constant",
        fallback_score=0.75,
    )
    with TestClient(app) as test_client:
        response = test_client.post("/v1/predict", json={"transactions": [record]})
        assert response.status_code == 200
        data = response.json()
        assert data["fallback_applied"] is True
        assert "model_exception: RuntimeError" in data["fallback_reason"]
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["fraud_probability"] == 0.75
        assert data["predictions"][0]["is_fraud"] is True

        # Check /metrics has fallback counter
        metrics_resp = test_client.get("/metrics")
        assert metrics_resp.status_code == 200
        assert "fraud_fallback_predictions_total" in metrics_resp.text


def test_fallback_mode_rule_based_on_amount(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    rec_low = dataset.features.iloc[0].to_dict()
    rec_low["Amount"] = 50.0
    rec_high = dataset.features.iloc[1].to_dict()
    rec_high["Amount"] = 1500.0

    app = create_app(
        model=model,
        fallback_mode="rule",
        fallback_amount_threshold=1000.0,
        degraded_mode=True,
    )
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/predict",
            json={"transactions": [rec_low, rec_high]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["fallback_applied"] is True
        assert data["fallback_reason"] == "degraded_mode_active"
        assert len(data["predictions"]) == 2
        # Low amount -> non-fraud
        assert data["predictions"][0]["fraud_probability"] == 0.0
        assert data["predictions"][0]["is_fraud"] is False
        assert "Amount (50.00) < 1000.00" in data["predictions"][0]["explanation"]
        # High amount -> fraud
        assert data["predictions"][1]["fraud_probability"] == 1.0
        assert data["predictions"][1]["is_fraud"] is True
        assert "Amount (1500.00) >= 1000.00" in data["predictions"][1]["explanation"]

        # Also test /v1/score single endpoint
        score_resp = test_client.post("/v1/score", json={"transaction": rec_high})
        assert score_resp.status_code == 200
        score_data = score_resp.json()
        assert score_data["fallback_applied"] is True
        assert score_data["prediction"]["is_fraud"] is True


def test_simulate_degraded_header_and_health_status(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    record = dataset.features.iloc[0].to_dict()

    app = create_app(
        model=model,
        fallback_mode="constant",
        fallback_score=0.42,
        degraded_mode=False,
        enable_chaos_header=True,
    )
    with TestClient(app) as test_client:
        # Normal health
        h_ready = test_client.get("/health")
        assert h_ready.status_code == 200
        assert h_ready.json()["status"] == "ready"
        assert h_ready.json().get("fallback_mode") == "constant"

        # Normal predict (not degraded)
        p_normal = test_client.post("/v1/predict", json={"transactions": [record]})
        assert p_normal.json()["fallback_applied"] is False

        # Chaos testing via header
        p_chaos = test_client.post(
            "/v1/predict",
            json={"transactions": [record]},
            headers={"X-Simulate-Degraded": "true"},
        )
        assert p_chaos.status_code == 200
        assert p_chaos.json()["fallback_applied"] is True
        assert p_chaos.json()["predictions"][0]["fraud_probability"] == 0.42

        # Single-transaction endpoint honors it too.
        s_chaos = test_client.post(
            "/v1/score",
            json={"transaction": record},
            headers={"X-Simulate-Degraded": "true"},
        )
        assert s_chaos.json()["fallback_applied"] is True

    # Now with degraded_mode=True on app
    app_degraded = create_app(
        model=model,
        fallback_mode="constant",
        degraded_mode=True,
    )
    with TestClient(app_degraded) as test_client:
        h_deg = test_client.get("/health")
        assert h_deg.status_code == 200
        assert h_deg.json()["status"] == "degraded"
        assert h_deg.json()["degraded_mode"] is True


def test_simulate_degraded_header_is_ignored_unless_enabled(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, dataset = api_context
    record = dataset.features.iloc[0].to_dict()

    default_app = create_app(model=model, fallback_mode="constant", fallback_score=0.42)
    with TestClient(default_app) as test_client:
        response = test_client.post(
            "/v1/predict",
            json={"transactions": [record]},
            headers={"X-Simulate-Degraded": "true"},
        )
    assert response.status_code == 200
    assert response.json()["fallback_applied"] is False

    monkeypatch.setenv(ENABLE_CHAOS_HEADER_ENVIRONMENT_VARIABLE, "true")
    env_app = create_app(model=model, fallback_mode="constant", fallback_score=0.42)
    with TestClient(env_app) as test_client:
        response = test_client.post(
            "/v1/score",
            json={"transaction": record},
            headers={"X-Simulate-Degraded": "true"},
        )
    assert response.json()["fallback_applied"] is True
    assert response.json()["prediction"]["fraud_probability"] == 0.42


def test_fallback_configuration_validation(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    import pytest

    _, model, _ = api_context
    with pytest.raises(ValueError, match="Invalid fallback_mode"):
        create_app(model=model, fallback_mode="unsupported_mode")

    with pytest.raises(ValueError, match=r"fallback_score must be between 0\.0 and 1\.0"):
        create_app(model=model, fallback_score=1.5)

    with pytest.raises(ValueError, match="requires a fallback policy"):
        create_app(model=model, degraded_mode=True)

    with pytest.raises(ValueError, match="requires a fallback policy"):
        create_app(model=model, degraded_mode=True, fallback_mode="raise")


def test_degraded_mode_environment_requires_a_fallback_policy(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, _ = api_context
    monkeypatch.setenv(DEGRADED_MODE_ENVIRONMENT_VARIABLE, "true")
    with pytest.raises(ValueError, match=DEGRADED_MODE_ENVIRONMENT_VARIABLE):
        create_app(model=model)


def test_primary_model_returns_none_probabilities(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    rec = dataset.features.iloc[0].to_dict()

    mock_model = MagicMock(wraps=model)
    mock_model.predict_probabilities.return_value = None

    app = create_app(model=mock_model)
    with TestClient(app) as test_client:
        response = test_client.post("/v1/predict", json={"transactions": [rec]})

    # The default fallback mode is 'raise', so the model failure surfaces as a 500.
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}


def test_unknown_feature_is_rejected_without_touching_the_circuit_breaker(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    circuit_breaker = CircuitBreaker(failure_threshold=1)
    app = create_app(
        model=model,
        fallback_mode="constant",
        fallback_score=0.99,
        circuit_breaker=circuit_breaker,
    )

    with TestClient(app) as test_client:
        for _ in range(3):
            response = test_client.post("/v1/score", json={"transaction": {"NotAFeature": 1.0}})
            assert response.status_code == 422
            assert "NotAFeature" in response.json()["detail"]

        health = test_client.get("/health")
        assert health.json()["status"] == "ready"
        assert health.json()["circuit_breaker"]["state"] == "closed"

        scored = test_client.post(
            "/v1/score", json={"transaction": dataset.features.iloc[0].to_dict()}
        )
        assert scored.status_code == 200
        assert scored.json()["fallback_applied"] is False


def test_missing_feature_is_rejected_without_touching_the_circuit_breaker(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    partial = dict(dataset.features.iloc[0].drop("V1").to_dict())
    circuit_breaker = CircuitBreaker(failure_threshold=1)
    app = create_app(
        model=model,
        fallback_mode="constant",
        fallback_score=0.99,
        circuit_breaker=circuit_breaker,
    )

    with TestClient(app) as test_client:
        response = test_client.post("/v1/predict", json={"transactions": [partial]})

    assert response.status_code == 422
    assert "V1" in response.json()["detail"]
    assert circuit_breaker.consecutive_failures == 0
    assert circuit_breaker.state == "closed"


def test_schema_violation_is_rejected_even_when_degraded_mode_is_active(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, _ = api_context
    app = create_app(model=model, fallback_mode="constant", degraded_mode=True)

    with TestClient(app) as test_client:
        response = test_client.post("/v1/score", json={"transaction": {"NotAFeature": 1.0}})

    assert response.status_code == 422


def test_missing_probabilities_activate_the_fallback_and_record_a_failure(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    rec = dataset.features.iloc[0].to_dict()
    broken_model = MagicMock(wraps=model)
    broken_model.metadata = model.metadata
    broken_model.threshold = model.threshold
    broken_model.feature_names = model.feature_names
    broken_model.predict_probabilities.return_value = None
    circuit_breaker = CircuitBreaker(failure_threshold=5)
    app = create_app(
        model=broken_model,
        fallback_mode="constant",
        fallback_score=0.31,
        circuit_breaker=circuit_breaker,
    )

    with TestClient(app) as test_client:
        response = test_client.post("/v1/predict", json={"transactions": [rec]})

    assert response.status_code == 200
    assert response.json()["fallback_applied"] is True
    assert response.json()["fallback_reason"] == "model_exception: RuntimeError"
    assert response.json()["predictions"][0]["fraud_probability"] == 0.31
    assert circuit_breaker.consecutive_failures == 1
    assert circuit_breaker.state == "closed"


def test_short_probability_array_activates_the_fallback(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    records = dataset.features.iloc[:3].to_dict(orient="records")
    short_model = MagicMock(wraps=model)
    short_model.metadata = model.metadata
    short_model.threshold = model.threshold
    short_model.feature_names = model.feature_names
    short_model.predict_probabilities.side_effect = lambda _frame: np.array([0.2, 0.4])
    app = create_app(model=short_model, fallback_mode="constant", fallback_score=0.6)

    with TestClient(app) as test_client:
        response = test_client.post("/v1/predict", json={"transactions": records})

    assert response.status_code == 200
    assert response.json()["fallback_applied"] is True
    assert response.json()["fallback_reason"] == "model_exception: RuntimeError"
    assert len(response.json()["predictions"]) == 3


def test_malformed_circuit_breaker_environment_is_rejected(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, _ = api_context
    monkeypatch.setenv(CIRCUIT_BREAKER_ENABLED_ENVIRONMENT_VARIABLE, "true")

    monkeypatch.setenv(CIRCUIT_BREAKER_FAILURE_THRESHOLD_ENVIRONMENT_VARIABLE, "many")
    with pytest.raises(
        ValueError, match=f"{CIRCUIT_BREAKER_FAILURE_THRESHOLD_ENVIRONMENT_VARIABLE}"
    ):
        create_app(model=model)

    monkeypatch.setenv(CIRCUIT_BREAKER_FAILURE_THRESHOLD_ENVIRONMENT_VARIABLE, "0")
    with pytest.raises(ValueError, match=">= 1"):
        create_app(model=model)

    monkeypatch.delenv(CIRCUIT_BREAKER_FAILURE_THRESHOLD_ENVIRONMENT_VARIABLE)
    monkeypatch.setenv(CIRCUIT_BREAKER_RECOVERY_TIMEOUT_ENVIRONMENT_VARIABLE, "soon")
    with pytest.raises(
        ValueError, match=f"{CIRCUIT_BREAKER_RECOVERY_TIMEOUT_ENVIRONMENT_VARIABLE}"
    ):
        create_app(model=model)

    monkeypatch.delenv(CIRCUIT_BREAKER_RECOVERY_TIMEOUT_ENVIRONMENT_VARIABLE)
    monkeypatch.setenv(CIRCUIT_BREAKER_LATENCY_BUDGET_ENVIRONMENT_VARIABLE, "-3")
    with pytest.raises(ValueError, match=f"{CIRCUIT_BREAKER_LATENCY_BUDGET_ENVIRONMENT_VARIABLE}"):
        create_app(model=model)


def test_injected_circuit_breaker_adopts_the_latency_budget(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, _ = api_context
    circuit_breaker = CircuitBreaker(failure_threshold=99)
    create_app(model=model, circuit_breaker=circuit_breaker, latency_budget_ms=12.5)
    assert circuit_breaker.latency_budget_ms == 12.5

    env_breaker = CircuitBreaker(failure_threshold=99)
    monkeypatch.setenv(CIRCUIT_BREAKER_LATENCY_BUDGET_ENVIRONMENT_VARIABLE, "7.5")
    create_app(model=model, circuit_breaker=env_breaker)
    assert env_breaker.latency_budget_ms == 7.5

    # The effective budget is reported so an operator can see what is enforced.
    app = create_app(model=model, circuit_breaker=env_breaker)
    with TestClient(app) as test_client:
        assert test_client.get("/health").json()["circuit_breaker"]["latency_budget_ms"] == 7.5


def test_injected_circuit_breaker_keeps_its_trip_callback(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, _ = api_context
    tripped: list[str] = []
    circuit_breaker = CircuitBreaker(failure_threshold=1, on_trip=lambda: tripped.append("trip"))

    app = create_app(model=model, circuit_breaker=circuit_breaker)
    with TestClient(app) as test_client:
        circuit_breaker.trip()
        metrics = test_client.get("/metrics").text

    assert tripped == ["trip"]
    assert "fraud_circuit_breaker_tripped_total 1.0" in metrics


def test_circuit_breaker_serializes_concurrent_state_transitions() -> None:
    trip_entered = threading.Event()
    release_trip = threading.Event()

    def on_trip() -> None:
        trip_entered.set()
        assert release_trip.wait(timeout=10), "circuit breaker trip was never released"

    circuit_breaker = CircuitBreaker(failure_threshold=1, on_trip=on_trip)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(circuit_breaker.record_failure)
        assert trip_entered.wait(timeout=10), "circuit breaker never tripped"
        second = pool.submit(circuit_breaker.record_failure)
        time.sleep(0.1)
        failures_while_tripping = circuit_breaker.consecutive_failures
        release_trip.set()
        first.result(timeout=10)
        second.result(timeout=10)

    # The second failure must not be applied while the first transition is in flight.
    assert failures_while_tripping == 1
    assert circuit_breaker.consecutive_failures == 2


def test_circuit_breaker_unit() -> None:
    tripped_count = 0

    def on_trip() -> None:
        nonlocal tripped_count
        tripped_count += 1

    cb = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=0.05,
        latency_budget_ms=10.0,
        half_open_success_threshold=2,
        on_trip=on_trip,
    )

    assert cb.state == "closed"
    assert cb.allow_request() is True

    cb.record_failure()
    assert cb.state == "closed"
    assert cb.consecutive_failures == 1

    cb.record_success()
    assert cb.consecutive_failures == 0

    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "open"
    assert tripped_count == 1
    assert cb.allow_request() is False

    time.sleep(0.06)
    assert cb.allow_request() is True
    assert cb.state == "half_open"

    cb.record_failure()
    assert cb.state == "open"
    assert tripped_count == 2
    assert cb.allow_request() is False

    time.sleep(0.06)
    assert cb.allow_request() is True
    assert cb.state == "half_open"
    cb.record_success()
    assert cb.state == "half_open"
    cb.record_success()
    assert cb.state == "closed"
    assert cb.consecutive_failures == 0

    cb.trip()
    assert cb.state == "open"
    assert tripped_count == 3
    cb.reset()
    assert cb.state == "closed"

    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError, match="recovery_timeout"):
        CircuitBreaker(recovery_timeout=0)
    with pytest.raises(ValueError, match="latency_budget_ms"):
        CircuitBreaker(latency_budget_ms=0)
    with pytest.raises(ValueError, match="half_open_success_threshold"):
        CircuitBreaker(half_open_success_threshold=0)


def test_circuit_breaker_api_routing_and_health(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    rec = dataset.features.iloc[0].to_dict()

    mock_model = MagicMock(wraps=model)
    mock_model.metadata = model.metadata
    mock_model.threshold = model.threshold
    mock_model.feature_names = model.feature_names
    mock_model.predict_probabilities.side_effect = RuntimeError("simulated model crash")

    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.08)
    app = create_app(
        model=mock_model,
        circuit_breaker=cb,
        fallback_mode="constant",
        fallback_score=0.88,
    )

    with TestClient(app) as client:
        r1 = client.post("/v1/score", json={"transaction": rec})
        assert r1.status_code == 200
        assert r1.json()["fallback_applied"] is True
        assert "model_exception" in r1.json()["fallback_reason"]
        assert cb.state == "closed"

        r2 = client.post("/v1/score", json={"transaction": rec})
        assert r2.status_code == 200
        assert cb.state == "open"

        mock_model.predict_probabilities.reset_mock()
        r3 = client.post("/v1/score", json={"transaction": rec})
        assert r3.status_code == 200
        assert r3.json()["fallback_applied"] is True
        assert r3.json()["fallback_reason"] == "circuit_breaker_open"
        assert r3.json()["prediction"]["fraud_probability"] == 0.88
        mock_model.predict_probabilities.assert_not_called()

        h = client.get("/health")
        assert h.status_code == 200
        h_data = h.json()
        assert h_data["status"] == "degraded"
        assert h_data["circuit_breaker"]["state"] == "open"

        m = client.get("/metrics")
        assert "fraud_circuit_breaker_state 2.0" in m.text
        assert "fraud_circuit_breaker_tripped_total 1.0" in m.text

        time.sleep(0.1)
        mock_model.predict_probabilities.side_effect = None
        mock_model.predict_probabilities.return_value = np.array([0.05])

        r4 = client.post("/v1/score", json={"transaction": rec})
        assert r4.status_code == 200
        assert r4.json()["fallback_applied"] is False
        assert cb.state == "closed"

        h_ready = client.get("/health")
        assert h_ready.json()["status"] == "ready"
        assert h_ready.json()["circuit_breaker"]["state"] == "closed"


def test_circuit_breaker_raise_when_open(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    rec = dataset.features.iloc[0].to_dict()

    cb = CircuitBreaker(failure_threshold=1)
    cb.trip()

    app = create_app(model=model, circuit_breaker=cb, fallback_mode="raise")
    with TestClient(app) as client:
        response = client.post("/v1/predict", json={"transactions": [rec]})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}


def test_circuit_breaker_latency_sla_breach(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
) -> None:
    _, model, dataset = api_context
    rec = dataset.features.iloc[0].to_dict()

    mock_model = MagicMock(wraps=model)
    mock_model.metadata = model.metadata
    mock_model.threshold = model.threshold
    mock_model.feature_names = model.feature_names

    def slow_predict(_frame: Any) -> np.ndarray:
        time.sleep(0.02)
        return np.array([0.1])

    mock_model.predict_probabilities.side_effect = slow_predict

    cb = CircuitBreaker(failure_threshold=1, latency_budget_ms=5.0)
    app = create_app(model=mock_model, circuit_breaker=cb)

    with TestClient(app) as client:
        client.post("/v1/score", json={"transaction": rec})
        assert cb.state == "open"


def test_traffic_shadowing_and_metrics(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    tmp_path: Path,
) -> None:
    _, primary_model, dataset = api_context
    recs = [dataset.features.iloc[i].to_dict() for i in range(5)]

    shadow_model = MagicMock(wraps=primary_model)
    shadow_model.metadata = {
        "created_at": "2026-09-20",
        "dataset_fingerprint": "shadow9876543210",
    }
    shadow_model.threshold = 0.5
    shadow_model.predict_probabilities.side_effect = lambda df: np.array([0.99] * len(df))

    sink = JsonlAuditSink(tmp_path / "audit_shadow.jsonl")
    app = create_app(
        model=primary_model,
        shadow_model=shadow_model,
        audit_sink=sink,
    )

    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200
        assert h.json()["shadow_model_version"] == "shadow987654"

        res = client.post("/v1/predict", json={"transactions": recs})
        assert res.status_code == 200

        res_single = client.post("/v1/score", json={"transaction": recs[0]})
        assert res_single.status_code == 200

        m = client.get("/metrics")
        assert "fraud_shadow_evaluations_total" in m.text

    sink.close()
    lines = [
        json.loads(line)
        for line in (tmp_path / "audit_shadow.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    shadow_events = [event for event in lines if event.get("event_type") == "shadow_scoring"]
    assert len(shadow_events) >= 2
    assert shadow_events[0]["payload"]["evaluated_count"] == 5
    assert shadow_events[0]["model_version"] == "shadow987654"


def test_create_app_with_circuit_breaker_and_shadow_env_vars(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, primary_model, _ = api_context
    shadow_path = tmp_path / "shadow_model.joblib"
    save_model(primary_model, shadow_path)

    monkeypatch.setenv(CIRCUIT_BREAKER_ENABLED_ENVIRONMENT_VARIABLE, "true")
    monkeypatch.setenv(CIRCUIT_BREAKER_FAILURE_THRESHOLD_ENVIRONMENT_VARIABLE, "3")
    monkeypatch.setenv(CIRCUIT_BREAKER_RECOVERY_TIMEOUT_ENVIRONMENT_VARIABLE, "2.5")
    monkeypatch.setenv(CIRCUIT_BREAKER_LATENCY_BUDGET_ENVIRONMENT_VARIABLE, "75.0")
    monkeypatch.setenv(SHADOW_MODEL_PATH_ENVIRONMENT_VARIABLE, str(shadow_path))

    app = create_app(model=primary_model)
    with TestClient(app) as test_client:
        h = test_client.get("/health")
        assert h.status_code == 200
        health_data = h.json()
        assert health_data["circuit_breaker"]["state"] == "closed"
        assert health_data["shadow_model_version"] is not None


def test_traffic_shadowing_exception_handling(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    tmp_path: Path,
) -> None:
    _, primary_model, dataset = api_context
    recs = [dataset.features.iloc[i].to_dict() for i in range(2)]

    shadow_model = MagicMock(wraps=primary_model)
    shadow_model.metadata = {
        "created_at": "2026-09-20",
        "dataset_fingerprint": "err_fingerprint_123",
    }
    shadow_model.threshold = 0.5
    shadow_model.predict_probabilities.side_effect = RuntimeError("Shadow execution exploded")

    sink = JsonlAuditSink(tmp_path / "audit_shadow_err.jsonl")
    app = create_app(
        model=primary_model,
        shadow_model=shadow_model,
        audit_sink=sink,
    )

    with TestClient(app) as client:
        res = client.post("/v1/predict", json={"transactions": recs})
        assert res.status_code == 200
        assert len(res.json()["predictions"]) == 2

    sink.close()
