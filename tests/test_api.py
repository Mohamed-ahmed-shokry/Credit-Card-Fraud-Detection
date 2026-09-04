from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient

from fraud_detection.api import (
    MAX_REQUEST_BODY_BYTES,
    MODEL_PATH_ENVIRONMENT_VARIABLE,
    PROCESS_TIME_HEADER,
    REQUEST_ID_HEADER,
    app_from_environment,
    create_app,
)
from fraud_detection.data import ValidatedDataset, generate_synthetic_data, validate_frame
from fraud_detection.model import FraudModel, save_model, train_model


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
        "service_version": "0.1.0",
        "model_created_at": model.metadata["created_at"],
        "feature_count": 30,
        "threshold": model.threshold,
    }


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


def test_api_logs_and_reraises_unhandled_errors(
    api_context: tuple[TestClient, FraudModel, ValidatedDataset],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, model, dataset = api_context

    class _BrokenModel:
        feature_names = model.feature_names
        threshold = model.threshold
        metadata = model.metadata

        def predict_probabilities(self, _features: object) -> None:
            raise RuntimeError("boom")

    broken_app = create_app(model=cast(FraudModel, _BrokenModel()))
    with (
        caplog.at_level(logging.ERROR, logger="fraud_detection.api"),
        TestClient(broken_app, raise_server_exceptions=False) as broken_client,
    ):
        response = broken_client.post(
            "/v1/predict",
            json={"transactions": dataset.features.iloc[:1].to_dict(orient="records")},
        )

    assert response.status_code == 500
    assert "request_failed" in caplog.text


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

    assert document["info"]["version"] == "0.1.0"
    assert "/v1/predict" in document["paths"]


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
