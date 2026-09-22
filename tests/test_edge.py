from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fraud_detection.data import generate_synthetic_data, validate_frame
from fraud_detection.edge import (
    EdgeArtifactError,
    EdgeModel,
    build_edge_model,
    load_edge_model,
)
from fraud_detection.model import (
    CalibrationMethod,
    EstimatorType,
    FraudModel,
    TrainingConfig,
    train_model,
)


@pytest.fixture(scope="module")
def edge_inputs():
    dataset = validate_frame(generate_synthetic_data(rows=600, fraud_rate=0.1))
    model = train_model(
        dataset,
        config=TrainingConfig(calibration_method=CalibrationMethod.NONE),
    )
    return model, dataset


def _minimal_edge_model() -> EdgeModel:
    return EdgeModel(
        model_version="model",
        feature_names=("first", "second"),
        threshold=0.5,
        scaler_mean=(0.0, 0.0),
        scaler_scale=(1.0, 1.0),
        quantized_weights=(10, -10),
        weight_scale=0.1,
        intercept=0.0,
        quantization_bits=8,
        prune_epsilon=0.0,
        pruned_features=(),
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": 2}, "schema version"),
        ({"quantization_bits": 4}, "8-bit"),
        ({"feature_names": ()}, "feature names"),
        ({"feature_names": ("first", "first")}, "feature names"),
        ({"scaler_mean": (0.0,)}, "arrays"),
        ({"threshold": 2.0}, "threshold"),
        ({"weight_scale": 0.0}, "weight_scale"),
        ({"prune_epsilon": -1.0}, "prune_epsilon"),
        ({"scaler_mean": (float("nan"), 0.0)}, "scaler values"),
        ({"scaler_scale": (0.0, 1.0)}, "scaler scales"),
        ({"quantized_weights": (128, 0)}, "int8"),
    ],
)
def test_edge_artifact_validation(changes: dict[str, object], message: str) -> None:
    with pytest.raises(EdgeArtifactError, match=message):
        replace(_minimal_edge_model(), **changes)


def test_edge_runtime_rejects_empty_records_and_invalid_values() -> None:
    edge_model = _minimal_edge_model()

    with pytest.raises(EdgeArtifactError, match="At least one"):
        edge_model.score_records([])
    with pytest.raises(EdgeArtifactError, match="unexpected"):
        edge_model.score_record({"first": 0.0, "second": 0.0, "extra": 1.0})
    with pytest.raises(EdgeArtifactError, match="numeric"):
        edge_model.score_record({"first": True, "second": 0.0})


def test_edge_loader_rejects_unreadable_json_and_non_object(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    with pytest.raises(EdgeArtifactError, match="Could not read"):
        load_edge_model(missing_path)

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("not json", encoding="utf-8")
    with pytest.raises(EdgeArtifactError, match="Could not read"):
        load_edge_model(invalid_path)

    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")
    with pytest.raises(EdgeArtifactError, match="root"):
        load_edge_model(list_path)


def test_edge_export_rejects_invalid_pruning_and_pipeline_contracts(edge_inputs) -> None:
    model, _dataset = edge_inputs
    with pytest.raises(EdgeArtifactError, match="prune_epsilon"):
        build_edge_model(model, prune_epsilon=float("nan"))

    metadata = {
        "training_config": {"estimator": "logistic_regression", "calibration_method": "none"}
    }
    base = FraudModel(SimpleNamespace(), 0.5, ("first",), metadata)
    with pytest.raises(EdgeArtifactError, match="scale/classifier"):
        build_edge_model(base)

    base.estimator = SimpleNamespace(
        named_steps={
            "scale": SimpleNamespace(mean_=[0.0], scale_=[1.0]),
            "classifier": SimpleNamespace(),
        }
    )
    with pytest.raises(EdgeArtifactError, match="not fitted"):
        build_edge_model(base)

    base.estimator.named_steps["classifier"] = SimpleNamespace(
        coef_=[[1.0]], intercept_=[0.0], classes_=[0, 2]
    )
    with pytest.raises(EdgeArtifactError, match="binary"):
        build_edge_model(base)


def test_edge_runtime_matches_source_model_and_round_trips(edge_inputs, tmp_path: Path) -> None:
    model, dataset = edge_inputs
    edge_model = build_edge_model(model)
    records = dataset.features.iloc[:25].to_dict(orient="records")

    edge_probabilities = edge_model.score_records(records)
    source_probabilities = model.predict_probabilities(dataset.features.iloc[:25])

    assert (
        max(
            abs(edge - source)
            for edge, source in zip(edge_probabilities, source_probabilities, strict=True)
        )
        < 0.005
    )
    assert [edge_model.predict_record(record) for record in records] == [
        bool(value >= model.threshold) for value in source_probabilities
    ]

    artifact_path = tmp_path / "edge-model.json"
    artifact_path.write_text(json.dumps(edge_model.to_dict()), encoding="utf-8")
    restored = load_edge_model(artifact_path)
    np.testing.assert_allclose(restored.score_records(records), edge_probabilities)


def test_edge_export_records_pruned_features(edge_inputs) -> None:
    model, _dataset = edge_inputs
    edge_model = build_edge_model(model, prune_epsilon=1_000.0)

    assert edge_model.pruned_features == model.feature_names
    assert edge_model.quantized_weights == (0,) * len(model.feature_names)


def test_edge_runtime_rejects_schema_and_non_finite_values(edge_inputs) -> None:
    model, dataset = edge_inputs
    edge_model = build_edge_model(model)
    record = dataset.features.iloc[0].to_dict()

    with pytest.raises(EdgeArtifactError, match="missing"):
        edge_model.score_record({key: value for key, value in record.items() if key != "V1"})
    record["V1"] = float("nan")
    with pytest.raises(EdgeArtifactError, match="finite"):
        edge_model.score_record(record)


def test_edge_export_rejects_calibrated_and_tree_models() -> None:
    dataset = validate_frame(generate_synthetic_data(rows=400, fraud_rate=0.1))

    calibrated = train_model(dataset)
    with pytest.raises(EdgeArtifactError, match="calibration_method='none'"):
        build_edge_model(calibrated)

    tree_model = train_model(
        dataset,
        config=TrainingConfig(
            estimator=EstimatorType.RANDOM_FOREST,
            calibration_method=CalibrationMethod.NONE,
        ),
    )
    with pytest.raises(EdgeArtifactError, match="logistic_regression"):
        build_edge_model(tree_model)


def test_edge_loader_rejects_malformed_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "malformed.json"
    artifact_path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")

    with pytest.raises(EdgeArtifactError, match="Invalid edge artifact"):
        load_edge_model(artifact_path)
