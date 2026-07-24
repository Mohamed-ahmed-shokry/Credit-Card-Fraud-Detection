from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import ValidatedDataset, generate_synthetic_data, validate_frame
from fraud_detection.model import (
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    MODEL_FILENAME,
    FraudModel,
    ModelArtifactError,
    TrainingConfig,
    load_model,
    save_model,
    train_model,
)


@pytest.fixture(scope="module")
def trained_model() -> tuple[FraudModel, ValidatedDataset]:
    dataset = validate_frame(generate_synthetic_data(rows=1_500, fraud_rate=0.08))
    return train_model(dataset), dataset


def test_train_model_produces_reproducible_model_card(
    trained_model: tuple[FraudModel, ValidatedDataset],
) -> None:
    model, dataset = trained_model

    assert 0.0 <= model.threshold <= 1.0
    assert model.feature_names == dataset.feature_names
    assert model.metadata["row_count"] == 1_500
    assert model.metadata["fraud_count"] == int(dataset.target.sum())
    assert sum(model.metadata["splits"].values()) == 1_500
    assert model.metadata["test_metrics"]["roc_auc"] > 0.7
    assert len(model.metadata["dataset_fingerprint"]) == 64

    repeated = train_model(dataset)
    assert repeated.threshold == pytest.approx(model.threshold)
    assert repeated.metadata["dataset_fingerprint"] == model.metadata["dataset_fingerprint"]
    assert repeated.metadata["test_metrics"] == model.metadata["test_metrics"]


def test_predictions_apply_threshold_and_accept_reordered_columns(
    trained_model: tuple[FraudModel, ValidatedDataset],
) -> None:
    model, dataset = trained_model
    features = dataset.features.iloc[:10, ::-1]

    probabilities = model.predict_probabilities(features)
    predictions = model.predict(features)

    assert probabilities.shape == (10,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    np.testing.assert_array_equal(predictions, probabilities >= model.threshold)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop(columns="V1"), "missing"),
        (lambda frame: frame.assign(extra=1), "unexpected"),
        (lambda frame: frame.assign(V1="invalid"), "must be numeric"),
        (lambda frame: frame.assign(V1=np.nan), "missing or infinite"),
        (lambda frame: frame.iloc[0:0], "At least one"),
    ],
)
def test_prediction_rejects_schema_violations(
    trained_model: tuple[FraudModel, ValidatedDataset],
    mutate: object,
    message: str,
) -> None:
    model, dataset = trained_model
    invalid = mutate(dataset.features.iloc[:2].copy())  # type: ignore[operator]

    with pytest.raises(ModelArtifactError, match=message):
        model.predict_probabilities(invalid)


def test_save_and_load_model_round_trip(
    tmp_path: Path,
    trained_model: tuple[FraudModel, ValidatedDataset],
) -> None:
    model, dataset = trained_model

    model_path = save_model(model, tmp_path / "artifact")
    restored = load_model(model_path.parent)

    assert model_path.is_file()
    manifest = json.loads((model_path.parent / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {MODEL_FILENAME, METADATA_FILENAME}
    assert all(len(digest) == 64 for digest in manifest["files"].values())
    metadata = json.loads((model_path.parent / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["dataset_fingerprint"] == model.metadata["dataset_fingerprint"]
    np.testing.assert_allclose(
        restored.predict_probabilities(dataset.features.iloc[:5]),
        model.predict_probabilities(dataset.features.iloc[:5]),
    )


def test_load_model_rejects_invalid_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactError, match="does not exist"):
        load_model(tmp_path / "missing")

    invalid_path = tmp_path / "invalid.joblib"
    joblib.dump({"not": "a model"}, invalid_path)
    with pytest.raises(ModelArtifactError, match="FraudModel"):
        load_model(invalid_path)


def test_load_model_detects_artifact_tampering(
    tmp_path: Path,
    trained_model: tuple[FraudModel, ValidatedDataset],
) -> None:
    model, _ = trained_model
    artifact_directory = tmp_path / "artifact"
    save_model(model, artifact_directory)
    metadata_path = artifact_directory / METADATA_FILENAME
    metadata_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="integrity check failed"):
        load_model(artifact_directory)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"test_size": 0.01},
        {"validation_size": 0.9},
        {"test_size": 0.4, "validation_size": 0.3},
        {"max_iterations": 10},
        {"regularization": 0},
    ],
)
def test_training_config_rejects_invalid_values(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)  # type: ignore[arg-type]


def test_train_model_rejects_too_few_fraud_rows() -> None:
    features = pd.DataFrame({"x": range(20)})
    target = pd.Series([0] * 15 + [1] * 5, name="Class", dtype="int8")

    with pytest.raises(ValueError, match="at least 6"):
        train_model(ValidatedDataset(features, target))
