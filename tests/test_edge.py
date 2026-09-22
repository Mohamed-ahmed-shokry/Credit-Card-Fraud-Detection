from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fraud_detection.data import generate_synthetic_data, validate_frame
from fraud_detection.edge import EdgeArtifactError, build_edge_model, load_edge_model
from fraud_detection.model import CalibrationMethod, EstimatorType, TrainingConfig, train_model


@pytest.fixture(scope="module")
def edge_inputs():
    dataset = validate_frame(generate_synthetic_data(rows=600, fraud_rate=0.1))
    model = train_model(
        dataset,
        config=TrainingConfig(calibration_method=CalibrationMethod.NONE),
    )
    return model, dataset


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
