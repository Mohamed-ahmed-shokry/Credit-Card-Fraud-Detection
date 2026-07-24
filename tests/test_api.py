from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from fraud_detection.api import create_app
from fraud_detection.data import ValidatedDataset, generate_synthetic_data, validate_frame
from fraud_detection.model import FraudModel, train_model


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
