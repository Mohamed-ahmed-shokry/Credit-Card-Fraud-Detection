from __future__ import annotations

import json
import warnings
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn

from fraud_detection.data import ValidatedDataset, generate_synthetic_data, validate_frame
from fraud_detection.model import (
    ARTIFACT_VERSION,
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    MODEL_FILENAME,
    CalibrationMethod,
    EstimatorType,
    FraudModel,
    ModelArtifactError,
    SplitStrategy,
    ThresholdStrategy,
    TrainingConfig,
    _extract_feature_effects,
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
    assert "expected_cost_per_transaction" in model.metadata["test_metrics"]
    assert "brier_score" in model.metadata["test_metrics"]
    assert model.metadata["calibration"] == {
        "method": "sigmoid",
        "folds": 3,
        "jobs": 1,
    }
    assert model.metadata["runtime_versions"]["scikit_learn"] == sklearn.__version__
    assert set(model.metadata["runtime_versions"]) == {
        "python",
        "joblib",
        "numpy",
        "pandas",
        "scikit_learn",
        "scipy",
    }
    assert all(model.metadata["runtime_versions"].values())
    assert model.estimator.n_jobs == 1
    assert model.artifact_version == ARTIFACT_VERSION
    assert len(model.metadata["dataset_fingerprint"]) == 64
    assert len(model.metadata["reference_profile"]) == 30
    assert model.metadata["drift_thresholds"] == {"warning_at": 0.1, "drift_at": 0.25}
    effects = model.metadata["feature_effects"]
    assert len(effects) == 30
    assert [effect["rank"] for effect in effects] == list(range(1, 31))
    assert all(
        effects[index]["absolute_effect"] >= effects[index + 1]["absolute_effect"]
        for index in range(len(effects) - 1)
    )
    assert {effect["direction"] for effect in effects} <= {
        "higher_fraud_risk",
        "lower_fraud_risk",
    }

    repeated = train_model(dataset)
    assert repeated.threshold == pytest.approx(model.threshold)
    assert repeated.metadata["dataset_fingerprint"] == model.metadata["dataset_fingerprint"]
    assert repeated.metadata["test_metrics"] == model.metadata["test_metrics"]


def test_prediction_rejects_duplicate_feature_names(
    trained_model: tuple[FraudModel, ValidatedDataset],
) -> None:
    model, dataset = trained_model
    duplicated = dataset.features.iloc[:2].copy()
    duplicated.columns = [duplicated.columns[0]] * len(duplicated.columns)

    with pytest.raises(ModelArtifactError, match="must be unique"):
        model.predict_probabilities(duplicated)


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


class _InvalidProbabilityEstimator:
    def __init__(self, output: object) -> None:
        self.output = output

    def predict_proba(self, _features: pd.DataFrame) -> object:
        return self.output


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ([["invalid", "values"]], "non-numeric"),
        (np.array([0.1, 0.9]), "binary-class shape"),
        (np.array([[0.1, np.nan]]), "finite range"),
        (np.array([[-0.1, 1.1]]), "finite range"),
    ],
)
def test_predictions_reject_invalid_estimator_outputs(
    trained_model: tuple[FraudModel, ValidatedDataset],
    output: object,
    message: str,
) -> None:
    model, dataset = trained_model
    invalid_model = deepcopy(model)
    invalid_model.estimator = _InvalidProbabilityEstimator(output)

    with pytest.raises(ModelArtifactError, match=message):
        invalid_model.predict_probabilities(dataset.features.iloc[:1])


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


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda model: setattr(model, "threshold", float("nan")), "decision threshold"),
        (
            lambda model: model.metadata["test_metrics"].update({"roc_auc": float("nan")}),
            "finite JSON-compatible",
        ),
    ],
)
def test_save_model_rejects_invalid_artifact_before_creating_directory(
    tmp_path: Path,
    trained_model: tuple[FraudModel, ValidatedDataset],
    mutate: object,
    message: str,
) -> None:
    model, _ = trained_model
    invalid_model = deepcopy(model)
    mutate(invalid_model)  # type: ignore[operator]
    artifact_directory = tmp_path / "artifact"

    with pytest.raises(ModelArtifactError, match=message):
        save_model(invalid_model, artifact_directory)

    assert not artifact_directory.exists()


def test_load_model_rejects_invalid_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactError, match="does not exist"):
        load_model(tmp_path / "missing")

    invalid_path = tmp_path / "invalid.joblib"
    joblib.dump({"not": "a model"}, invalid_path)
    with pytest.raises(ModelArtifactError, match="FraudModel"):
        load_model(invalid_path)

    corrupted_path = tmp_path / "corrupted.joblib"
    corrupted_path.write_bytes(b"not a pickle stream")
    with pytest.raises(ModelArtifactError, match="Could not load model artifact"):
        load_model(corrupted_path)


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


def test_load_model_requires_valid_integrity_manifest(
    tmp_path: Path,
    trained_model: tuple[FraudModel, ValidatedDataset],
) -> None:
    model, _ = trained_model
    artifact_directory = tmp_path / "artifact"
    save_model(model, artifact_directory)
    manifest_path = artifact_directory / MANIFEST_FILENAME
    manifest_path.unlink()
    with pytest.raises(ModelArtifactError, match="manifest does not exist"):
        load_model(artifact_directory)

    manifest_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="manifest is invalid"):
        load_model(artifact_directory)

    manifest_path.write_text('{"artifact_version": NaN}\n', encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="non-standard JSON"):
        load_model(artifact_directory)

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "hash_algorithm": "md5",
        "files": {MODEL_FILENAME: "0" * 64, METADATA_FILENAME: "0" * 64},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="manifest is invalid"):
        load_model(artifact_directory)

    model_path = artifact_directory / MODEL_FILENAME
    manifest["hash_algorithm"] = "sha256"
    manifest["files"][MODEL_FILENAME] = sha256(model_path.read_bytes()).hexdigest()
    del manifest["files"][METADATA_FILENAME]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="integrity entry is missing"):
        load_model(artifact_directory)


def test_load_model_rejects_metadata_that_is_not_a_json_object(
    tmp_path: Path,
    trained_model: tuple[FraudModel, ValidatedDataset],
) -> None:
    model, _ = trained_model
    artifact_directory = tmp_path / "artifact"
    save_model(model, artifact_directory)
    metadata_path = artifact_directory / METADATA_FILENAME
    metadata_path.write_text("[]\n", encoding="utf-8")
    manifest_path = artifact_directory / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][METADATA_FILENAME] = sha256(metadata_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="must be a JSON object"):
        load_model(artifact_directory)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda model: setattr(model, "threshold", float("nan")), "decision threshold"),
        (lambda model: setattr(model, "feature_names", ("V1", "V1")), "feature schema"),
        (lambda model: setattr(model, "estimator", object()), "probability prediction"),
        (
            lambda model: setattr(model.estimator, "classes_", np.array([1, 0])),
            "binary class order",
        ),
        (lambda model: setattr(model, "artifact_version", 999), "artifact version"),
        (lambda model: setattr(model, "metadata", "not a mapping"), "must be a mapping"),
        (
            lambda model: model.metadata.update({"feature_count": 0}),
            "feature_count",
        ),
        (
            lambda model: model.metadata.pop("created_at"),
            "created_at",
        ),
        (
            lambda model: model.metadata.pop("scikit_learn_version"),
            "scikit_learn_version",
        ),
        (
            lambda model: model.metadata.pop("dataset_fingerprint"),
            "dataset_fingerprint",
        ),
        (
            lambda model: model.metadata.update({"scikit_learn_version": "0.0"}),
            "version mismatch",
        ),
        (
            lambda model: model.metadata["test_metrics"].update({"roc_auc": float("nan")}),
            "finite JSON-compatible",
        ),
    ],
)
def test_load_model_rejects_structurally_invalid_models(
    tmp_path: Path,
    trained_model: tuple[FraudModel, ValidatedDataset],
    mutate: object,
    message: str,
) -> None:
    model, _ = trained_model
    invalid_model = deepcopy(model)
    mutate(invalid_model)  # type: ignore[operator]
    artifact_path = tmp_path / "invalid.joblib"
    joblib.dump(invalid_model, artifact_path)

    with pytest.raises(ModelArtifactError, match=message):
        load_model(artifact_path)


def test_load_model_rejects_metadata_that_disagrees_with_model(
    tmp_path: Path,
    trained_model: tuple[FraudModel, ValidatedDataset],
) -> None:
    model, _ = trained_model
    artifact_directory = tmp_path / "artifact"
    save_model(model, artifact_directory)
    metadata_path = artifact_directory / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["created_at"] = "changed-after-model-save"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    manifest_path = artifact_directory / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][METADATA_FILENAME] = sha256(metadata_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="does not match"):
        load_model(artifact_directory)


def test_load_model_rejects_directory_version_mismatch_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trained_model: tuple[FraudModel, ValidatedDataset],
) -> None:
    model, _ = trained_model
    artifact_directory = tmp_path / "artifact"
    save_model(model, artifact_directory)
    metadata_path = artifact_directory / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["scikit_learn_version"] = "0.0"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    manifest_path = artifact_directory / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][METADATA_FILENAME] = sha256(metadata_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fail_if_deserialized(_path: Path) -> object:
        raise AssertionError("joblib.load must not run after a preflight mismatch")

    monkeypatch.setattr(joblib, "load", fail_if_deserialized)

    with pytest.raises(ModelArtifactError, match="version mismatch"):
        load_model(artifact_directory)


def test_load_model_converts_sklearn_version_warning_to_artifact_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = tmp_path / "model.joblib"
    artifact_path.write_bytes(b"placeholder")

    def warn_about_version(_path: Path) -> object:
        warnings.warn(
            sklearn.exceptions.InconsistentVersionWarning(
                estimator_name="LogisticRegression",
                current_sklearn_version=sklearn.__version__,
                original_sklearn_version="0.0",
            ),
            stacklevel=2,
        )
        raise AssertionError("warning should have been converted to an error")

    monkeypatch.setattr(joblib, "load", warn_about_version)

    with pytest.raises(ModelArtifactError, match="incompatible scikit-learn"):
        load_model(artifact_path)


def test_load_model_rejects_nonstandard_metadata_json(
    tmp_path: Path,
    trained_model: tuple[FraudModel, ValidatedDataset],
) -> None:
    model, _ = trained_model
    artifact_directory = tmp_path / "artifact"
    save_model(model, artifact_directory)
    metadata_path = artifact_directory / METADATA_FILENAME
    metadata_path.write_text('{"invalid_number": NaN}\n', encoding="utf-8")
    manifest_path = artifact_directory / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][METADATA_FILENAME] = sha256(metadata_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="non-standard JSON"):
        load_model(artifact_directory)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"test_size": 0.01},
        {"validation_size": 0.9},
        {"test_size": 0.4, "validation_size": 0.3},
        {"estimator": "unknown"},
        {"max_iterations": 10},
        {"regularization": 0},
        {"threshold_strategy": "unknown"},
        {"false_positive_cost": 0},
        {"false_negative_cost": float("inf")},
        {"calibration_method": "unknown"},
        {"calibration_folds": 1},
        {"calibration_folds": 11},
        {"calibration_jobs": 0},
        {"calibration_jobs": -2},
        {"split_strategy": "unknown"},
        {"time_column": " "},
        {"cost_policy": ""},
        {"cost_policy": "  "},
        {"n_estimators": 5},
        {"max_depth": 0},
        {"learning_rate": 0},
        {"l2_regularization": -1},
        {"max_bins": 1},
    ],
)
def test_training_config_rejects_invalid_values(
    kwargs: dict[str, float | int | str],
) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)  # type: ignore[arg-type]


def test_train_model_rejects_too_few_fraud_rows() -> None:
    features = pd.DataFrame({"x": range(20)})
    target = pd.Series([0] * 15 + [1] * 5, name="Class", dtype="int8")

    with pytest.raises(ValueError, match="at least 6"):
        train_model(ValidatedDataset(features, target))


def test_train_model_rejects_minority_class_too_small_for_splits() -> None:
    rng = np.random.default_rng(1)
    features = pd.DataFrame({"x": rng.normal(size=120)})
    target = pd.Series([0] * 114 + [1] * 6, name="Class", dtype="int8")
    config = TrainingConfig(test_size=0.05, validation_size=0.05)

    with pytest.raises(ValueError, match="too small for the configured"):
        train_model(ValidatedDataset(features, target), config=config)


def test_train_model_rejects_calibration_folds_exceeding_minority_class_size() -> None:
    rng = np.random.default_rng(0)
    features = pd.DataFrame({"x": rng.normal(size=100)})
    target = pd.Series([0] * 90 + [1] * 10, name="Class", dtype="int8")
    config = TrainingConfig(calibration_folds=10)

    with pytest.raises(ValueError, match="at least as many rows as calibration_folds"):
        train_model(ValidatedDataset(features, target), config=config)


def test_train_model_supports_cost_sensitive_thresholds() -> None:
    dataset = validate_frame(generate_synthetic_data(rows=800, fraud_rate=0.1))
    config = TrainingConfig(
        threshold_strategy=ThresholdStrategy.COST,
        false_positive_cost=1,
        false_negative_cost=25,
        calibration_jobs=-1,
    )

    model = train_model(dataset, config=config)

    assert model.metadata["training_config"]["threshold_strategy"] == "cost"
    assert model.metadata["training_config"]["false_negative_cost"] == 25
    assert model.metadata["calibration"]["jobs"] == -1
    assert model.estimator.n_jobs == -1


def test_train_model_can_disable_probability_calibration() -> None:
    dataset = validate_frame(generate_synthetic_data(rows=500, fraud_rate=0.1))

    model = train_model(
        dataset,
        config=TrainingConfig(calibration_method=CalibrationMethod.NONE),
    )

    assert model.metadata["estimator"] == "LogisticRegression"
    assert model.metadata["calibration"] == {
        "method": "none",
        "folds": None,
        "jobs": None,
    }


def test_train_model_supports_random_forest_estimator() -> None:
    dataset = validate_frame(generate_synthetic_data(rows=800, fraud_rate=0.1))

    model = train_model(
        dataset,
        config=TrainingConfig(
            estimator=EstimatorType.RANDOM_FOREST,
            calibration_method=CalibrationMethod.NONE,
        ),
    )

    assert model.metadata["estimator"] == "RandomForestClassifier"
    assert model.metadata["training_config"]["estimator"] == "random_forest"
    assert 0.0 <= model.threshold <= 1.0
    probabilities = model.predict_probabilities(dataset.features.iloc[:10])
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))

    effects = model.metadata["feature_effects"]
    assert len(effects) == 30
    assert all(effect["method"] == "feature_importance" for effect in effects)
    assert all(effect["direction"] is None for effect in effects)
    assert all(effect["coefficient"] >= 0.0 for effect in effects)
    assert [effect["rank"] for effect in effects] == list(range(1, 31))
    assert all(
        effects[index]["absolute_effect"] >= effects[index + 1]["absolute_effect"]
        for index in range(len(effects) - 1)
    )


def test_train_model_supports_calibrated_random_forest(tmp_path: Path) -> None:
    dataset = validate_frame(generate_synthetic_data(rows=800, fraud_rate=0.1))

    model = train_model(dataset, config=TrainingConfig(estimator=EstimatorType.RANDOM_FOREST))

    assert model.metadata["estimator"] == "CalibratedClassifierCV(RandomForestClassifier)"
    model_path = save_model(model, tmp_path / "artifact")
    restored = load_model(model_path.parent)
    np.testing.assert_allclose(
        restored.predict_probabilities(dataset.features.iloc[:5]),
        model.predict_probabilities(dataset.features.iloc[:5]),
    )


def test_train_model_supports_hist_gradient_boosting(tmp_path: Path) -> None:
    dataset = validate_frame(generate_synthetic_data(rows=800, fraud_rate=0.1))

    model = train_model(
        dataset, config=TrainingConfig(estimator=EstimatorType.HIST_GRADIENT_BOOSTING)
    )

    assert model.metadata["estimator"] == "CalibratedClassifierCV(HistGradientBoostingClassifier)"
    model_path = save_model(model, tmp_path / "artifact")
    restored = load_model(model_path.parent)
    np.testing.assert_allclose(
        restored.predict_probabilities(dataset.features.iloc[:5]),
        model.predict_probabilities(dataset.features.iloc[:5]),
    )
    effects = model.metadata["feature_effects"]
    assert len(effects) == 30
    assert all(effect["method"] == "permutation_importance" for effect in effects)
    assert all(effect["direction"] is None for effect in effects)
    assert sum(effect["coefficient"] for effect in effects) > 0


def test_train_model_applies_random_forest_hyperparameters() -> None:
    dataset = validate_frame(generate_synthetic_data(rows=500, fraud_rate=0.1))

    model = train_model(
        dataset,
        config=TrainingConfig(
            estimator=EstimatorType.RANDOM_FOREST,
            n_estimators=25,
            max_depth=4,
            calibration_method=CalibrationMethod.NONE,
        ),
    )

    classifier = model.estimator.named_steps["classifier"]
    assert classifier.n_estimators == 25
    assert classifier.max_depth == 4


def test_train_model_supports_chronological_evaluation() -> None:
    dataset = validate_frame(generate_synthetic_data(rows=1_000, fraud_rate=0.1))

    model = train_model(
        dataset,
        config=TrainingConfig(
            split_strategy=SplitStrategy.TEMPORAL,
            calibration_method=CalibrationMethod.NONE,
        ),
    )

    ranges = model.metadata["split_time_ranges"]
    assert ranges["train"]["maximum"] <= ranges["validation"]["minimum"]
    assert ranges["validation"]["maximum"] <= ranges["test"]["minimum"]
    assert model.metadata["training_config"]["split_strategy"] == "temporal"


def test_temporal_split_requires_time_feature() -> None:
    dataset = validate_frame(generate_synthetic_data(rows=500, fraud_rate=0.1).drop(columns="Time"))

    with pytest.raises(ValueError, match="requires feature column"):
        train_model(
            dataset,
            config=TrainingConfig(split_strategy=SplitStrategy.TEMPORAL),
        )


def test_temporal_split_rejects_single_class_windows() -> None:
    features = pd.DataFrame({"Time": range(100), "amount": range(100)})
    target = pd.Series([0] * 90 + [1] * 10, name="Class", dtype="int8")

    with pytest.raises(ValueError, match="must contain both target classes"):
        train_model(
            ValidatedDataset(features, target),
            config=TrainingConfig(
                split_strategy=SplitStrategy.TEMPORAL,
                calibration_method=CalibrationMethod.NONE,
            ),
        )


def test_explain_local_returns_per_transaction_contributions(
    trained_model: tuple[FraudModel, ValidatedDataset],
) -> None:
    model, dataset = trained_model
    features = dataset.features.iloc[:5]

    explanations = model.explain_local(features)

    assert len(explanations) == 5
    assert all(isinstance(exp, dict) for exp in explanations)
    assert all(set(exp.keys()) == set(model.feature_names) for exp in explanations)
    assert all(all(isinstance(v, float) for v in exp.values()) for exp in explanations)


def test_explain_local_requires_scaler_stats(
    trained_model: tuple[FraudModel, ValidatedDataset],
) -> None:
    model, dataset = trained_model
    model.metadata.pop("scaler_mean", None)
    model.metadata.pop("scaler_scale", None)

    with pytest.raises(ModelArtifactError, match="Scaler statistics not available"):
        model.explain_local(dataset.features.iloc[:1])


def test_explain_local_requires_feature_effects(
    trained_model: tuple[FraudModel, ValidatedDataset],
) -> None:
    model, dataset = trained_model
    model.metadata.pop("feature_effects", None)

    with pytest.raises(ModelArtifactError, match="Feature effects not available"):
        model.explain_local(dataset.features.iloc[:1])


def test_explain_local_random_forest() -> None:
    dataset = validate_frame(generate_synthetic_data(rows=500, fraud_rate=0.1, random_state=1))
    model = train_model(
        dataset,
        config=TrainingConfig(
            estimator=EstimatorType.RANDOM_FOREST,
            calibration_method=CalibrationMethod.NONE,
        ),
    )
    features = dataset.features.iloc[:3]

    explanations = model.explain_local(features)

    assert len(explanations) == 3
    assert all(isinstance(exp, dict) for exp in explanations)
    assert all(set(exp.keys()) == set(model.feature_names) for exp in explanations)
    assert all(all(isinstance(v, float) for v in exp.values()) for exp in explanations)

    model.metadata.pop("feature_effects", None)
    with pytest.raises(ModelArtifactError, match="Feature effects not available"):
        model.explain_local(features.iloc[:1])


def test_load_model_supports_bare_joblib_file(
    tmp_path: Path, trained_model: tuple[FraudModel, ValidatedDataset]
) -> None:
    model, dataset = trained_model
    artifact_path = tmp_path / "model.joblib"
    joblib.dump(model, artifact_path)

    restored = load_model(artifact_path)

    np.testing.assert_allclose(
        restored.predict_probabilities(dataset.features.iloc[:5]),
        model.predict_probabilities(dataset.features.iloc[:5]),
    )


def test_extract_feature_effects_requires_permutation_data() -> None:
    stub = SimpleNamespace(named_steps={"classifier": SimpleNamespace()})

    with pytest.raises(ValueError, match="Permutation training data"):
        _extract_feature_effects(stub, ("x",), calibration_method=CalibrationMethod.NONE)
