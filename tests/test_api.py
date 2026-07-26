from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fraud_detection.api import (
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
