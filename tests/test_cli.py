from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import joblib
import pandas as pd
import pytest
from typer.testing import CliRunner

from fraud_detection import __version__
from fraud_detection.cli import _git_info, _resolve_report_costs, app
from fraud_detection.data import (
    DEFAULT_TARGET,
    generate_synthetic_data,
    validate_frame,
)
from fraud_detection.features import (
    FeatureDefinition,
    FeatureSnapshot,
    FeatureType,
    FeatureView,
    FileFeatureStore,
)
from fraud_detection.model import (
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    MODEL_FILENAME,
    CalibrationMethod,
    FraudModel,
    ModelArtifactError,
    TrainingConfig,
    load_model,
    save_model,
    train_model,
    verify_attestation,
)
from fraud_detection.recalibration import CalibrationDiagnostics
from fraud_detection.rules import (
    DecisionRule,
    RuleAction,
    RuleCondition,
    RuleOperator,
    RulePrecedence,
    RuleSet,
)
from fraud_detection.signing import write_keypair
from fraud_detection.velocity import VelocityConfig, VelocityWindow

runner = CliRunner()


@pytest.fixture(scope="module")
def trained_model() -> FraudModel:
    dataset = validate_frame(generate_synthetic_data(rows=300, random_state=3))
    return train_model(dataset)


@pytest.fixture(scope="module")
def trained_artifact(tmp_path_factory: pytest.TempPathFactory, trained_model: FraudModel) -> Path:
    artifact_directory = tmp_path_factory.mktemp("artifact")
    save_model(trained_model, artifact_directory)
    return artifact_directory


def test_export_edge_command_validates_and_protects_output(tmp_path: Path) -> None:
    dataset = validate_frame(generate_synthetic_data(rows=500, random_state=11))
    model = train_model(
        dataset,
        config=TrainingConfig(calibration_method=CalibrationMethod.NONE),
    )
    artifact_path = tmp_path / "artifact"
    save_model(model, artifact_path)
    validation_path = tmp_path / "validation.csv"
    validation = dataset.features.copy()
    validation[DEFAULT_TARGET] = dataset.target
    validation.to_csv(validation_path, index=False)
    edge_path = tmp_path / "edge-model.json"

    exported = runner.invoke(
        app,
        [
            "export-edge",
            str(artifact_path),
            "--output",
            str(edge_path),
            "--validation-data",
            str(validation_path),
        ],
    )

    assert exported.exit_code == 0, exported.output
    summary = json.loads(exported.stdout)
    assert summary["quantization_bits"] == 8
    assert summary["validation"]["rows"] == 500
    payload = json.loads(edge_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["validation"]["max_absolute_probability_error"] <= 0.01

    protected = runner.invoke(
        app,
        ["export-edge", str(artifact_path), "--output", str(edge_path)],
    )
    assert protected.exit_code == 2
    assert "Pass --overwrite" in protected.stderr


def test_export_edge_command_rejects_calibrated_artifact(
    trained_artifact: Path,
    tmp_path: Path,
) -> None:
    result = runner.invoke(
        app,
        ["export-edge", str(trained_artifact), "--output", str(tmp_path / "edge.json")],
    )

    assert result.exit_code == 2
    assert "calibration_method='none'" in result.stderr


def test_version_flag_prints_installed_version_and_exits() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_cli_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_path = tmp_path / "transactions.csv"
    artifact_path = tmp_path / "artifact"
    predictions_path = tmp_path / "predictions.csv"
    drift_path = tmp_path / "reports" / "drift.json"

    generated = runner.invoke(
        app,
        [
            "generate-data",
            "--output",
            str(data_path),
            "--rows",
            "800",
            "--fraud-rate",
            "0.08",
            "--seed",
            "9",
        ],
    )
    assert generated.exit_code == 0, generated.output
    generated_summary = json.loads(generated.stdout)
    assert generated_summary["rows"] == 800
    assert data_path.is_file()

    artifact_path.mkdir()
    trained = runner.invoke(
        app,
        [
            "train",
            str(data_path),
            "--output",
            str(artifact_path),
            "--seed",
            "9",
            "--threshold-strategy",
            "cost",
            "--false-negative-cost",
            "20",
            "--calibration-method",
            "sigmoid",
            "--calibration-folds",
            "4",
            "--split-strategy",
            "temporal",
        ],
    )
    assert trained.exit_code == 0, trained.output
    training_summary = json.loads(trained.stdout)
    assert training_summary["test_metrics"]["roc_auc"] > 0.7
    assert (artifact_path / MODEL_FILENAME).is_file()
    assert (artifact_path / METADATA_FILENAME).is_file()
    metadata = json.loads((artifact_path / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["training_config"]["threshold_strategy"] == "cost"
    assert metadata["training_config"]["false_negative_cost"] == 20
    assert metadata["calibration"] == {"method": "sigmoid", "folds": 4, "jobs": 1}
    assert metadata["training_config"]["split_strategy"] == "temporal"
    assert len(metadata["lineage"]["git_commit"]) == 40
    assert metadata["lineage"]["git_repository"]
    assert (
        metadata["split_time_ranges"]["train"]["maximum"]
        <= metadata["split_time_ranges"]["validation"]["minimum"]
    )

    inspected = runner.invoke(app, ["inspect", str(artifact_path)])
    assert inspected.exit_code == 0, inspected.output
    assert json.loads(inspected.stdout)["row_count"] == 800

    explained = runner.invoke(app, ["explain", str(artifact_path), "--top", "5"])
    assert explained.exit_code == 0, explained.output
    explanation = json.loads(explained.stdout)
    assert len(explanation["effects"]) == 5
    assert [effect["rank"] for effect in explanation["effects"]] == [1, 2, 3, 4, 5]

    drifted = runner.invoke(
        app,
        [
            "drift",
            str(artifact_path),
            str(data_path),
            "--output",
            str(drift_path),
        ],
    )
    assert drifted.exit_code == 0, drifted.output
    drift_summary = json.loads(drifted.stdout)
    assert drift_summary["rows"] == 800
    assert len(drift_summary["features"]) == 30
    assert json.loads(drift_path.read_text(encoding="utf-8")) == drift_summary

    drifted_without_output = runner.invoke(app, ["drift", str(artifact_path), str(data_path)])
    assert drifted_without_output.exit_code == 0, drifted_without_output.output
    assert json.loads(drifted_without_output.stdout) == drift_summary

    predicted = runner.invoke(
        app,
        [
            "predict",
            str(artifact_path),
            str(data_path),
            "--output",
            str(predictions_path),
        ],
    )
    assert predicted.exit_code == 0, predicted.output
    prediction_summary = json.loads(predicted.stdout)
    assert prediction_summary["rows"] == 800
    scored = pd.read_csv(predictions_path)
    assert {"fraud_probability", "is_fraud"}.issubset(scored.columns)

    server_call: dict[str, Any] = {}

    def fake_run(application: object, *, host: str, port: int) -> None:
        server_call.update(application=application, host=host, port=port)

    monkeypatch.setattr("fraud_detection.cli.uvicorn.run", fake_run)
    served = runner.invoke(
        app,
        ["serve", str(artifact_path), "--host", "0.0.0.0", "--port", "9000"],
    )
    assert served.exit_code == 0, served.output
    assert server_call["host"] == "0.0.0.0"
    assert server_call["port"] == 9000

    protected = runner.invoke(
        app,
        ["train", str(data_path), "--output", str(artifact_path)],
    )
    assert protected.exit_code == 2
    assert "Pass --overwrite" in protected.stderr


def test_serve_passes_middleware_options_to_create_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = tmp_path / "artifact"
    artifact_path.mkdir()
    created: dict[str, Any] = {}

    def fake_load_model(_path: Path | str) -> object:
        return object()

    def fake_create_app(**kwargs: Any) -> object:
        created.update(kwargs)
        return object()

    def fake_run(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("fraud_detection.cli.load_model", fake_load_model)
    monkeypatch.setattr("fraud_detection.api.create_app", fake_create_app)
    monkeypatch.setattr("fraud_detection.cli.uvicorn.run", fake_run)

    served = runner.invoke(
        app,
        [
            "serve",
            str(artifact_path),
            "--api-key",
            "key-one",
            "--api-key",
            "key-two",
            "--rate-limit-requests",
            "25",
            "--rate-limit-window-seconds",
            "15",
            "--max-concurrent-scoring",
            "4",
        ],
    )

    assert served.exit_code == 0, served.output
    assert created["api_keys"] == ["key-one", "key-two"]
    assert created["rate_limit_requests"] == 25
    assert created["rate_limit_window_seconds"] == 15.0
    assert created["max_concurrent_scoring"] == 4


def test_train_and_explain_support_random_forest_estimator(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    artifact_path = tmp_path / "artifact"
    generate_synthetic_data(rows=800, fraud_rate=0.08, random_state=5).to_csv(
        data_path, index=False
    )

    trained = runner.invoke(
        app,
        ["train", str(data_path), "--output", str(artifact_path), "--estimator", "random_forest"],
    )
    assert trained.exit_code == 0, trained.output
    training_summary = json.loads(trained.stdout)
    assert training_summary["estimator"] == "CalibratedClassifierCV(RandomForestClassifier)"

    explained = runner.invoke(app, ["explain", str(artifact_path), "--top", "3"])
    assert explained.exit_code == 0, explained.output
    effects = json.loads(explained.stdout)["effects"]
    assert all(effect["method"] == "feature_importance" for effect in effects)
    assert all(effect["direction"] is None for effect in effects)


def test_train_and_explain_support_hist_gradient_boosting_estimator(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    artifact_path = tmp_path / "artifact"
    generate_synthetic_data(rows=800, fraud_rate=0.08, random_state=5).to_csv(
        data_path, index=False
    )

    trained = runner.invoke(
        app,
        [
            "train",
            str(data_path),
            "--output",
            str(artifact_path),
            "--estimator",
            "hist_gradient_boosting",
        ],
    )
    assert trained.exit_code == 0, trained.output
    training_summary = json.loads(trained.stdout)
    assert training_summary["estimator"] == "CalibratedClassifierCV(HistGradientBoostingClassifier)"

    explained = runner.invoke(app, ["explain", str(artifact_path), "--top", "3"])
    assert explained.exit_code == 0, explained.output
    effects = json.loads(explained.stdout)["effects"]
    assert all(effect["method"] == "permutation_importance" for effect in effects)
    assert all(effect["direction"] is None for effect in effects)


def test_train_honors_tree_hyperparameters(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    artifact_path = tmp_path / "artifact"
    generate_synthetic_data(rows=600, fraud_rate=0.1, random_state=18).to_csv(
        data_path, index=False
    )

    trained = runner.invoke(
        app,
        [
            "train",
            str(data_path),
            "--output",
            str(artifact_path),
            "--estimator",
            "random_forest",
            "--n-estimators",
            "10",
            "--max-depth",
            "3",
            "--calibration-method",
            "none",
        ],
    )

    assert trained.exit_code == 0, trained.output
    metadata = json.loads((artifact_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["training_config"]["n_estimators"] == 10
    assert metadata["training_config"]["max_depth"] == 3


def test_compare_defaults_to_every_estimator_on_the_same_split(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=1200, fraud_rate=0.08, random_state=6).to_csv(
        data_path, index=False
    )

    compared = runner.invoke(app, ["compare", str(data_path)])

    assert compared.exit_code == 0, compared.output
    results = json.loads(compared.stdout)["results"]
    assert {result["estimator"] for result in results} == {
        "CalibratedClassifierCV(LogisticRegression)",
        "CalibratedClassifierCV(RandomForestClassifier)",
        "CalibratedClassifierCV(HistGradientBoostingClassifier)",
    }
    actual_positives = {
        result["test_metrics"]["true_positives"] + result["test_metrics"]["false_negatives"]
        for result in results
    }
    actual_negatives = {
        result["test_metrics"]["true_negatives"] + result["test_metrics"]["false_positives"]
        for result in results
    }
    assert len(actual_positives) == 1, "every estimator must see the same test split"
    assert len(actual_negatives) == 1, "every estimator must see the same test split"


def test_compare_reports_dataset_errors(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=300, fraud_rate=0.1, random_state=8).to_csv(data_path, index=False)

    result = runner.invoke(app, ["compare", str(data_path), "--target", "missing_column"])

    assert result.exit_code == 2
    assert "missing_column" in result.stderr


def test_compare_supports_selecting_specific_estimators(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=500, fraud_rate=0.08, random_state=7).to_csv(
        data_path, index=False
    )

    compared = runner.invoke(
        app,
        ["compare", str(data_path), "--estimator", "logistic_regression"],
    )

    assert compared.exit_code == 0, compared.output
    results = json.loads(compared.stdout)["results"]
    assert len(results) == 1
    assert results[0]["estimator"] == "CalibratedClassifierCV(LogisticRegression)"


def test_compare_supports_hyperparameter_sweep_logistic_regression(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=800, fraud_rate=0.08, random_state=9).to_csv(
        data_path, index=False
    )

    compared = runner.invoke(
        app,
        [
            "compare",
            str(data_path),
            "--estimator",
            "logistic_regression",
            "--param-name",
            "regularization",
            "--param-values",
            "0.1,1.0,10.0",
        ],
    )

    assert compared.exit_code == 0, compared.output
    results = json.loads(compared.stdout)["results"]
    assert len(results) == 3
    for result in results:
        assert result["estimator"] == "CalibratedClassifierCV(LogisticRegression)"
        assert "hyperparameter" in result
        assert "regularization" in result["hyperparameter"]


def test_compare_supports_hyperparameter_sweep_hist_gradient_boosting(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=800, fraud_rate=0.08, random_state=10).to_csv(
        data_path, index=False
    )

    compared = runner.invoke(
        app,
        [
            "compare",
            str(data_path),
            "--estimator",
            "hist_gradient_boosting",
            "--param-name",
            "learning_rate",
            "--param-values",
            "0.01,0.1,0.2",
        ],
    )

    assert compared.exit_code == 0, compared.output
    results = json.loads(compared.stdout)["results"]
    assert len(results) == 3
    for result in results:
        assert result["estimator"] == "CalibratedClassifierCV(HistGradientBoostingClassifier)"
        assert "hyperparameter" in result
        assert "learning_rate" in result["hyperparameter"]


def test_compare_rejects_hyperparameter_sweep_with_multiple_estimators(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=500, fraud_rate=0.08, random_state=11).to_csv(
        data_path, index=False
    )

    compared = runner.invoke(
        app,
        [
            "compare",
            str(data_path),
            "--estimator",
            "logistic_regression",
            "--estimator",
            "random_forest",
            "--param-name",
            "regularization",
            "--param-values",
            "0.1,1.0",
        ],
    )

    assert compared.exit_code == 2
    assert "requires exactly one estimator" in compared.stderr


def test_compare_rejects_hyperparameter_sweep_with_missing_param(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=500, fraud_rate=0.08, random_state=12).to_csv(
        data_path, index=False
    )

    compared = runner.invoke(
        app,
        [
            "compare",
            str(data_path),
            "--estimator",
            "logistic_regression",
            "--param-name",
            "regularization",
        ],
    )

    assert compared.exit_code == 2
    assert "Both --param-name and --param-values must be provided" in compared.stderr


def test_compare_rejects_empty_hyperparameter_values(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=500, fraud_rate=0.08, random_state=17).to_csv(
        data_path, index=False
    )

    compared = runner.invoke(
        app,
        [
            "compare",
            str(data_path),
            "--estimator",
            "logistic_regression",
            "--param-name",
            "regularization",
            "--param-values",
            " , ",
        ],
    )

    assert compared.exit_code == 2
    assert "at least one value" in compared.stderr


def test_compare_rejects_unknown_hyperparameter(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=500, fraud_rate=0.08, random_state=13).to_csv(
        data_path, index=False
    )

    compared = runner.invoke(
        app,
        [
            "compare",
            str(data_path),
            "--estimator",
            "logistic_regression",
            "--param-name",
            "not_a_param",
            "--param-values",
            "1,2",
        ],
    )

    assert compared.exit_code == 2
    assert "Unknown hyperparameter" in compared.stderr


def test_compare_rejects_incompatible_hyperparameter(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=500, fraud_rate=0.08, random_state=14).to_csv(
        data_path, index=False
    )

    compared = runner.invoke(
        app,
        [
            "compare",
            str(data_path),
            "--estimator",
            "random_forest",
            "--param-name",
            "regularization",
            "--param-values",
            "0.1,1.0",
        ],
    )

    assert compared.exit_code == 2
    assert "does not apply to" in compared.stderr


def test_compare_rejects_invalid_hyperparameter_value(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=500, fraud_rate=0.08, random_state=15).to_csv(
        data_path, index=False
    )

    compared = runner.invoke(
        app,
        [
            "compare",
            str(data_path),
            "--estimator",
            "logistic_regression",
            "--param-name",
            "regularization",
            "--param-values",
            "0.1,not_a_number",
        ],
    )

    assert compared.exit_code == 2
    assert "Invalid value" in compared.stderr


def test_compare_supports_hyperparameter_sweep_random_forest(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=800, fraud_rate=0.08, random_state=16).to_csv(
        data_path, index=False
    )

    compared = runner.invoke(
        app,
        [
            "compare",
            str(data_path),
            "--estimator",
            "random_forest",
            "--param-name",
            "n_estimators",
            "--param-values",
            "10,25",
        ],
    )

    assert compared.exit_code == 0, compared.output
    results = json.loads(compared.stdout)["results"]
    assert len(results) == 2
    for result in results:
        assert result["estimator"] == "CalibratedClassifierCV(RandomForestClassifier)"
        assert "hyperparameter" in result
        assert "n_estimators" in result["hyperparameter"]


def _save_model_missing_metadata_key(
    model: FraudModel,
    destination: Path,
    remove_key: str,
) -> Path:
    mutated = deepcopy(model)
    del mutated.metadata[remove_key]
    save_model(mutated, destination)
    return destination


def test_predict_protects_existing_output(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    output = tmp_path / "predictions.csv"
    data_path.write_text("x\n1\n", encoding="utf-8")
    output.write_text("keep me", encoding="utf-8")

    result = runner.invoke(
        app,
        ["predict", str(trained_artifact), str(data_path), "--output", str(output)],
    )

    assert result.exit_code == 2
    assert "Pass --overwrite" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep me"


def test_explain_reports_missing_feature_effects(tmp_path: Path, trained_model: FraudModel) -> None:
    artifact = _save_model_missing_metadata_key(
        trained_model,
        tmp_path / "artifact",
        remove_key="feature_effects",
    )

    result = runner.invoke(app, ["explain", str(artifact)])

    assert result.exit_code == 2
    assert "does not contain feature effects" in result.stderr


def test_drift_reports_missing_reference_profile(tmp_path: Path, trained_model: FraudModel) -> None:
    artifact = _save_model_missing_metadata_key(
        trained_model,
        tmp_path / "artifact",
        remove_key="reference_profile",
    )
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=200, random_state=4).to_csv(data_path, index=False)

    result = runner.invoke(app, ["drift", str(artifact), str(data_path)])

    assert result.exit_code == 2
    assert "does not contain a reference profile" in result.stderr


def test_promote_reports_missing_reference_profile(
    tmp_path: Path, trained_model: FraudModel
) -> None:
    artifact = _save_model_missing_metadata_key(
        trained_model,
        tmp_path / "artifact",
        remove_key="reference_profile",
    )
    heldout_path = tmp_path / "heldout.csv"
    recent_path = tmp_path / "recent.csv"
    generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=86).to_csv(
        heldout_path, index=False
    )
    generate_synthetic_data(rows=200, random_state=87).to_csv(recent_path, index=False)

    result = runner.invoke(app, ["promote", str(artifact), str(heldout_path), str(recent_path)])

    assert result.exit_code == 2
    assert "does not contain a reference profile" in result.stderr


@pytest.mark.parametrize("command", ["inspect", "explain", "serve"])
def test_commands_report_invalid_artifact_errors(tmp_path: Path, command: str) -> None:
    empty_artifact = tmp_path / "empty_artifact"
    empty_artifact.mkdir()

    result = runner.invoke(app, [command, str(empty_artifact)])

    assert result.exit_code == 2
    assert "Error:" in result.stderr


def test_generate_data_protects_existing_file(tmp_path: Path) -> None:
    output = tmp_path / "existing.csv"
    output.write_text("keep me", encoding="utf-8")

    result = runner.invoke(app, ["generate-data", "--output", str(output)])

    assert result.exit_code == 2
    assert "Pass --overwrite" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep me"


def test_generate_data_preserves_existing_file_when_atomic_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "existing.csv"
    output.write_text("keep me", encoding="utf-8")

    def fail_after_partial_write(
        _frame: pd.DataFrame,
        path: Path,
        *,
        index: bool,
    ) -> None:
        assert index is False
        path.write_text("partial output", encoding="utf-8")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_after_partial_write)

    result = runner.invoke(
        app,
        ["generate-data", "--output", str(output), "--overwrite"],
    )

    assert result.exit_code == 2
    assert "simulated disk failure" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep me"
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("command", ["predict", "drift"])
def test_predict_and_drift_report_empty_transactions_file(
    tmp_path: Path,
    trained_artifact: Path,
    command: str,
) -> None:
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")

    result = runner.invoke(app, [command, str(trained_artifact), str(empty_csv)])

    assert result.exit_code == 2
    assert "No columns to parse from file" in result.stderr


def _shifted_transactions(rows: int = 200, random_state: int = 91) -> pd.DataFrame:
    frame = generate_synthetic_data(rows=rows, fraud_rate=0.1, random_state=random_state)
    shifted = frame.copy()
    shifted[frame.columns.drop(DEFAULT_TARGET)] += 10.0
    return shifted


def test_drift_fail_on_trips_on_shifted_data(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "recent.csv"
    _shifted_transactions().to_csv(data_path, index=False)

    tripped = runner.invoke(
        app, ["drift", str(trained_artifact), str(data_path), "--fail-on", "drifted"]
    )

    assert tripped.exit_code == 1
    assert json.loads(tripped.stdout)["overall_status"] == "drifted"
    assert "tripped" in tripped.stderr

    silent = runner.invoke(app, ["drift", str(trained_artifact), str(data_path)])

    assert silent.exit_code == 0
    assert json.loads(silent.stdout)["overall_status"] == "drifted"


def test_drift_fail_on_passes_on_stable_data(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "recent.csv"
    generate_synthetic_data(rows=300, random_state=3).to_csv(data_path, index=False)

    result = runner.invoke(
        app, ["drift", str(trained_artifact), str(data_path), "--fail-on", "drifted"]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["overall_status"] == "stable"


def test_drift_fail_on_rejects_unknown_level(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "recent.csv"
    generate_synthetic_data(rows=200, random_state=92).to_csv(data_path, index=False)

    result = runner.invoke(
        app, ["drift", str(trained_artifact), str(data_path), "--fail-on", "bogus"]
    )

    assert result.exit_code == 2
    assert "surveillance level" in result.stderr


def test_drift_protects_existing_report(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    data_path = tmp_path / "transactions.csv"
    output = tmp_path / "drift.json"
    model_path.write_bytes(b"placeholder")
    data_path.write_text("x\n1\n", encoding="utf-8")
    output.write_text("keep me", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "drift",
            str(model_path),
            str(data_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "Pass --overwrite" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep me"


def test_calibration_reports_reliability(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    report_path = tmp_path / "reports" / "calibration.json"
    generate_synthetic_data(rows=300, fraud_rate=0.1, random_state=11).to_csv(
        data_path, index=False
    )

    result = runner.invoke(
        app,
        [
            "calibration",
            str(trained_artifact),
            str(data_path),
            "--output",
            str(report_path),
            "--bins",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["rows"] == 300
    assert body["bins"] == 5
    assert len(body["detail"]) == 5
    assert 0.0 <= body["expected_calibration_error"] <= 1.0
    assert "model_version" in body
    assert json.loads(report_path.read_text(encoding="utf-8")) == body


def test_calibration_protects_existing_report(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    data_path = tmp_path / "transactions.csv"
    output = tmp_path / "calibration.json"
    model_path.write_bytes(b"placeholder")
    data_path.write_text("x\n1\n", encoding="utf-8")
    output.write_text("keep me", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "calibration",
            str(model_path),
            str(data_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "Pass --overwrite" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep me"


def test_calibration_reports_missing_label_column(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=200, random_state=12).to_csv(data_path, index=False)

    result = runner.invoke(
        app,
        ["calibration", str(trained_artifact), str(data_path), "--target", "missing_column"],
    )

    assert result.exit_code == 2
    assert "missing_column" in result.stderr


def test_stability_reports_metric_spread(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=600, fraud_rate=0.1, random_state=31).to_csv(
        data_path, index=False
    )

    result = runner.invoke(
        app,
        [
            "stability",
            str(data_path),
            "--estimator",
            "logistic_regression",
            "--repeats",
            "2",
            "--seed",
            "7",
            "--calibration-method",
            "none",
        ],
    )

    assert result.exit_code == 0, result.output
    results = json.loads(result.stdout)["results"]
    assert len(results) == 1
    summary = results[0]
    assert summary["estimator"] == "LogisticRegression"
    assert summary["repeats"] == 2
    assert [run["seed"] for run in summary["runs"]] == [7, 8]
    expected_metrics = {
        "roc_auc",
        "average_precision",
        "brier_score",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
    }
    assert set(summary["test_metrics_mean"]) == expected_metrics
    assert set(summary["test_metrics_std"]) == expected_metrics
    assert all(value >= 0 for value in summary["test_metrics_std"].values())
    for run in summary["runs"]:
        assert set(run["test_metrics"]) >= expected_metrics


def test_stability_reports_dataset_errors(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=300, fraud_rate=0.1, random_state=32).to_csv(
        data_path, index=False
    )

    result = runner.invoke(app, ["stability", str(data_path), "--target", "missing_column"])

    assert result.exit_code == 2
    assert "missing_column" in result.stderr


def test_benchmark_reports_throughput(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    report_path = tmp_path / "reports" / "benchmark.json"
    generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=21).to_csv(
        data_path, index=False
    )

    result = runner.invoke(
        app,
        [
            "benchmark",
            str(trained_artifact),
            str(data_path),
            "--batch-sizes",
            "2,4",
            "--repeat",
            "2",
            "--output",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["rows_available"] == 200
    assert body["repeat"] == 2
    assert [item["batch_size"] for item in body["results"]] == [2, 4]
    for item in body["results"]:
        assert item["median_ms"] > 0
        assert item["ms_per_transaction"] > 0
        assert item["transactions_per_second"] > 0
    assert json.loads(report_path.read_text(encoding="utf-8")) == body


@pytest.mark.parametrize("batch_sizes", ["0,4", "2,abc", " , ", "200000"])
def test_benchmark_rejects_invalid_batch_sizes(
    tmp_path: Path, trained_artifact: Path, batch_sizes: str
) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=200, random_state=22).to_csv(data_path, index=False)

    result = runner.invoke(
        app,
        ["benchmark", str(trained_artifact), str(data_path), "--batch-sizes", batch_sizes],
    )

    assert result.exit_code == 2
    assert "batch" in result.stderr.lower()


def test_thresholds_reports_tradeoffs(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    report_path = tmp_path / "reports" / "thresholds.json"
    generate_synthetic_data(rows=300, fraud_rate=0.1, random_state=41).to_csv(
        data_path, index=False
    )

    result = runner.invoke(
        app,
        [
            "thresholds",
            str(trained_artifact),
            str(data_path),
            "--output",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["rows"] == 300
    assert [row["threshold"] for row in body["detail"]] == [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]
    assert body["model_threshold_metrics"]["threshold"] == body["model_threshold"]
    assert body["false_positive_cost"] == 1.0
    assert body["false_negative_cost"] == 10.0
    assert body["cost_policy"] == {
        "name": "default",
        "false_positive_cost": 1.0,
        "false_negative_cost": 10.0,
    }
    assert json.loads(report_path.read_text(encoding="utf-8")) == body


def test_thresholds_supports_custom_candidates_and_costs(
    tmp_path: Path, trained_artifact: Path
) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=42).to_csv(
        data_path, index=False
    )

    result = runner.invoke(
        app,
        [
            "thresholds",
            str(trained_artifact),
            str(data_path),
            "--thresholds",
            "0.8,0.2",
            "--false-positive-cost",
            "2",
            "--false-negative-cost",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert [row["threshold"] for row in body["detail"]] == [0.2, 0.8]
    assert body["false_positive_cost"] == 2.0
    assert body["false_negative_cost"] == 5.0
    assert body["cost_policy"]["name"] == "custom"


def test_train_records_named_cost_policy(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    artifact_path = tmp_path / "artifact"
    generate_synthetic_data(rows=600, fraud_rate=0.1, random_state=61).to_csv(
        data_path, index=False
    )

    trained = runner.invoke(
        app,
        [
            "train",
            str(data_path),
            "--output",
            str(artifact_path),
            "--cost-policy",
            "strict-recall",
            "--false-negative-cost",
            "25",
            "--calibration-method",
            "none",
        ],
    )

    assert trained.exit_code == 0, trained.output
    metadata = json.loads((artifact_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["training_config"]["cost_policy"] == "strict-recall"
    assert metadata["cost_policy"] == {
        "name": "strict-recall",
        "false_positive_cost": 1.0,
        "false_negative_cost": 25.0,
    }


@pytest.mark.parametrize(
    ("thresholds", "message"),
    [
        ("0.5", "between 2 and 20"),
        ("0.2,high", "every entry must be a number"),
        ("0.2,1.5", "between 0 and 1"),
    ],
)
def test_thresholds_rejects_invalid_candidates(
    tmp_path: Path, trained_artifact: Path, thresholds: str, message: str
) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=200, random_state=43).to_csv(data_path, index=False)

    result = runner.invoke(
        app,
        ["thresholds", str(trained_artifact), str(data_path), "--thresholds", thresholds],
    )

    assert result.exit_code == 2
    assert message in result.stderr


def test_thresholds_reports_missing_label_column(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=200, random_state=44).to_csv(data_path, index=False)

    result = runner.invoke(
        app,
        ["thresholds", str(trained_artifact), str(data_path), "--target", "missing_column"],
    )

    assert result.exit_code == 2
    assert "missing_column" in result.stderr


def test_thresholds_protects_existing_report(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    data_path = tmp_path / "transactions.csv"
    output = tmp_path / "thresholds.json"
    model_path.write_bytes(b"placeholder")
    data_path.write_text("x\n1\n", encoding="utf-8")
    output.write_text("keep me", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "thresholds",
            str(model_path),
            str(data_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "Pass --overwrite" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep me"


class _StubModel:
    def __init__(self, metadata: object) -> None:
        self.metadata = metadata


def test_resolve_report_costs_prefers_overrides() -> None:
    model = _StubModel({"training_config": {"false_positive_cost": 3.0}})

    assert _resolve_report_costs(model, 2.0, None) == ("custom", 2.0, 10.0)


def test_resolve_report_costs_reads_named_policy() -> None:
    model = _StubModel(
        {
            "cost_policy": {
                "name": "strict-recall",
                "false_positive_cost": 1,
                "false_negative_cost": 25,
            }
        }
    )

    assert _resolve_report_costs(model, None, None) == ("strict-recall", 1.0, 25.0)


def test_resolve_report_costs_tolerates_null_metadata_blocks() -> None:
    model = _StubModel({"cost_policy": None, "training_config": None})

    assert _resolve_report_costs(model, None, None) == ("default", 1.0, 10.0)


def test_resolve_report_costs_falls_back_to_training_config() -> None:
    model = _StubModel({"training_config": {"false_positive_cost": 2, "false_negative_cost": 7}})

    assert _resolve_report_costs(model, None, None) == ("default", 2.0, 7.0)


@pytest.mark.parametrize(
    "metadata",
    [
        {"cost_policy": "oops"},
        {"cost_policy": {"name": " ", "false_positive_cost": 1}},
        "oops",
    ],
)
def test_resolve_report_costs_rejects_malformed_policy(metadata: object) -> None:
    with pytest.raises(ValueError, match="artifact"):
        _resolve_report_costs(_StubModel(metadata), None, None)


def test_resolve_report_costs_rejects_malformed_training_config() -> None:
    with pytest.raises(ValueError, match="training_config is invalid"):
        _resolve_report_costs(_StubModel({"training_config": "oops"}), None, None)

    with pytest.raises(ValueError, match="Invalid classification costs"):
        _resolve_report_costs(
            _StubModel({"training_config": {"false_positive_cost": "high"}}), None, None
        )


def test_rolling_reports_metric_spread_over_prefixes(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=800, fraud_rate=0.1, random_state=71).to_csv(
        data_path, index=False
    )

    result = runner.invoke(
        app,
        [
            "rolling",
            str(data_path),
            "--estimator",
            "logistic_regression",
            "--origins",
            "2",
            "--calibration-method",
            "none",
        ],
    )

    assert result.exit_code == 0, result.output
    results = json.loads(result.stdout)["results"]
    assert len(results) == 1
    summary = results[0]
    assert summary["estimator"] == "LogisticRegression"
    assert summary["origins"] == 2
    assert [run["origin"] for run in summary["runs"]] == [0, 1]
    assert [run["rows"] for run in summary["runs"]] == [600, 800]
    assert (
        summary["runs"][0]["test_time_range"]["maximum"]
        <= summary["runs"][1]["test_time_range"]["maximum"]
    )
    expected_metrics = {
        "roc_auc",
        "average_precision",
        "brier_score",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
    }
    assert set(summary["test_metrics_mean"]) == expected_metrics
    assert set(summary["test_metrics_std"]) == expected_metrics
    assert all(value >= 0 for value in summary["test_metrics_std"].values())


def test_rolling_requires_time_feature(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=500, fraud_rate=0.1, random_state=72).drop(columns="Time").to_csv(
        data_path, index=False
    )

    result = runner.invoke(app, ["rolling", str(data_path)])

    assert result.exit_code == 2
    assert "requires feature column" in result.stderr


def test_rolling_reports_dataset_errors(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=300, fraud_rate=0.1, random_state=73).to_csv(
        data_path, index=False
    )

    result = runner.invoke(app, ["rolling", str(data_path), "--target", "missing_column"])

    assert result.exit_code == 2
    assert "missing_column" in result.stderr


def test_promote_assembles_evidence_bundle(tmp_path: Path, trained_artifact: Path) -> None:
    heldout_path = tmp_path / "heldout.csv"
    recent_path = tmp_path / "recent.csv"
    bundle_path = tmp_path / "reports" / "promotion.json"
    generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=81).to_csv(
        heldout_path, index=False
    )
    generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=82).to_csv(
        recent_path, index=False
    )

    result = runner.invoke(
        app,
        [
            "promote",
            str(trained_artifact),
            str(heldout_path),
            str(recent_path),
            "--thresholds",
            "0.2,0.8",
            "--bins",
            "4",
            "--batch-sizes",
            "2",
            "--output",
            str(bundle_path),
        ],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert set(body) == {
        "model_version",
        "model",
        "calibration",
        "thresholds",
        "drift",
        "benchmark",
    }
    assert body["model"]["cost_policy"]["name"] == "default"
    assert len(body["calibration"]["detail"]) == 4
    assert [row["threshold"] for row in body["thresholds"]["detail"]] == [0.2, 0.8]
    assert body["drift"]["rows"] == 200
    assert [item["batch_size"] for item in body["benchmark"]["results"]] == [2]
    assert json.loads(bundle_path.read_text(encoding="utf-8")) == body


def test_promote_reports_missing_heldout_label(tmp_path: Path, trained_artifact: Path) -> None:
    heldout_path = tmp_path / "heldout.csv"
    recent_path = tmp_path / "recent.csv"
    generate_synthetic_data(rows=200, random_state=83).to_csv(heldout_path, index=False)
    generate_synthetic_data(rows=200, random_state=84).to_csv(recent_path, index=False)

    result = runner.invoke(
        app,
        [
            "promote",
            str(trained_artifact),
            str(heldout_path),
            str(recent_path),
            "--target",
            "missing_column",
        ],
    )

    assert result.exit_code == 2
    assert "missing_column" in result.stderr


def test_promote_reports_recent_schema_mismatch(tmp_path: Path, trained_artifact: Path) -> None:
    heldout_path = tmp_path / "heldout.csv"
    recent_path = tmp_path / "recent.csv"
    generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=85).to_csv(
        heldout_path, index=False
    )
    pd.DataFrame({"wrong": [1.0, 2.0]}).to_csv(recent_path, index=False)

    result = runner.invoke(
        app, ["promote", str(trained_artifact), str(heldout_path), str(recent_path)]
    )

    assert result.exit_code == 2
    assert "Input schema does not match" in result.stderr


def test_promote_protects_existing_bundle(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    heldout_path = tmp_path / "heldout.csv"
    recent_path = tmp_path / "recent.csv"
    output = tmp_path / "promotion.json"
    model_path.write_bytes(b"placeholder")
    heldout_path.write_text("x\n1\n", encoding="utf-8")
    recent_path.write_text("x\n1\n", encoding="utf-8")
    output.write_text("keep me", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "promote",
            str(model_path),
            str(heldout_path),
            str(recent_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "Pass --overwrite" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep me"


def test_benchmark_reports_without_output_file(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=200, random_state=25).to_csv(data_path, index=False)

    result = runner.invoke(
        app,
        ["benchmark", str(trained_artifact), str(data_path), "--batch-sizes", "3"],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert [item["batch_size"] for item in body["results"]] == [3]


def test_benchmark_protects_existing_report(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    data_path = tmp_path / "transactions.csv"
    output = tmp_path / "benchmark.json"
    model_path.write_bytes(b"placeholder")
    data_path.write_text("x\n1\n", encoding="utf-8")
    output.write_text("keep me", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "benchmark",
            str(model_path),
            str(data_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "Pass --overwrite" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep me"


def test_benchmark_reports_write_failures(
    tmp_path: Path, trained_artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=200, random_state=23).to_csv(data_path, index=False)

    def fail_write(_content: str, _destination: Path) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr("fraud_detection.cli._atomic_write_text", fail_write)

    result = runner.invoke(
        app,
        [
            "benchmark",
            str(trained_artifact),
            str(data_path),
            "--batch-sizes",
            "2",
            "--output",
            str(tmp_path / "benchmark.json"),
        ],
    )

    assert result.exit_code == 2
    assert "simulated disk failure" in result.stderr


def test_calibration_reports_without_output_file(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=24).to_csv(
        data_path, index=False
    )

    result = runner.invoke(
        app, ["calibration", str(trained_artifact), str(data_path), "--bins", "4"]
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["rows"] == 200
    assert len(body["detail"]) == 4


def test_train_reports_invalid_output_directory_without_replacing_file(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "transactions.csv"
    output = tmp_path / "model"
    output.write_text("keep me", encoding="utf-8")
    generated = runner.invoke(
        app,
        [
            "generate-data",
            "--output",
            str(data_path),
            "--rows",
            "300",
            "--fraud-rate",
            "0.1",
        ],
    )
    assert generated.exit_code == 0, generated.output

    result = runner.invoke(
        app,
        [
            "train",
            str(data_path),
            "--output",
            str(output),
            "--overwrite",
            "--calibration-method",
            "none",
        ],
    )

    assert result.exit_code == 2
    assert "Error:" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep me"


def test_predict_reports_schema_error(tmp_path: Path) -> None:
    data_path = tmp_path / "training.csv"
    artifact_path = tmp_path / "artifact"
    invalid_path = tmp_path / "invalid.csv"
    runner.invoke(
        app,
        [
            "generate-data",
            "--output",
            str(data_path),
            "--rows",
            "600",
            "--fraud-rate",
            "0.1",
        ],
    )
    trained = runner.invoke(app, ["train", str(data_path), "--output", str(artifact_path)])
    assert trained.exit_code == 0, trained.output
    pd.DataFrame({"wrong": [1.0]}).to_csv(invalid_path, index=False)

    result = runner.invoke(app, ["predict", str(artifact_path), str(invalid_path)])

    assert result.exit_code == 2
    assert "Input schema does not match" in result.stderr


def test_predict_supports_threshold_override(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=51).to_csv(
        data_path, index=False
    )

    baseline = runner.invoke(
        app,
        ["predict", str(trained_artifact), str(data_path), "--output", str(tmp_path / "b.csv")],
    )
    assert baseline.exit_code == 0, baseline.output
    baseline_summary = json.loads(baseline.stdout)
    assert baseline_summary["threshold_overridden"] is False
    assert baseline_summary["threshold"] == baseline_summary["model_threshold"]

    overridden = runner.invoke(
        app,
        [
            "predict",
            str(trained_artifact),
            str(data_path),
            "--output",
            str(tmp_path / "o.csv"),
            "--threshold",
            "0.0",
        ],
    )
    assert overridden.exit_code == 0, overridden.output
    override_summary = json.loads(overridden.stdout)
    assert override_summary["threshold"] == 0.0
    assert override_summary["model_threshold"] == baseline_summary["model_threshold"]
    assert override_summary["threshold_overridden"] is True
    assert override_summary["flagged"] == override_summary["rows"]
    assert override_summary["flagged"] >= baseline_summary["flagged"]


def test_predict_rejects_invalid_threshold_override(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    data_path = tmp_path / "transactions.csv"
    model_path.write_bytes(b"placeholder")
    data_path.write_text("x\n1\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["predict", str(model_path), str(data_path), "--threshold", "1.5"],
    )

    assert result.exit_code == 2
    assert "between 0 and 1" in result.stderr


def test_predict_supports_local_explanation(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    artifact_path = tmp_path / "artifact"
    output_path = tmp_path / "predictions.csv"
    runner.invoke(
        app,
        [
            "generate-data",
            "--output",
            str(data_path),
            "--rows",
            "300",
            "--fraud-rate",
            "0.1",
            "--seed",
            "42",
        ],
    )
    trained = runner.invoke(
        app, ["train", str(data_path), "--output", str(artifact_path), "--seed", "42"]
    )
    assert trained.exit_code == 0, trained.output

    predicted = runner.invoke(
        app,
        [
            "predict",
            str(artifact_path),
            str(data_path),
            "--output",
            str(output_path),
            "--explain",
            "--explain-llm",
        ],
    )
    assert predicted.exit_code == 0, predicted.output

    scored = pd.read_csv(output_path)
    assert {"fraud_probability", "is_fraud"}.issubset(scored.columns)
    contrib_cols = [c for c in scored.columns if c.startswith("contrib_")]
    assert len(contrib_cols) == 30
    assert scored["llm_explanation"].str.contains("Top contributing factors:").all()


def test_model_card_prints_compact_view(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=42).to_csv(
        data_path, index=False
    )

    result = runner.invoke(
        app,
        ["model-card", str(trained_artifact)],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    # model_version is the first 12 chars of dataset_fingerprint
    assert body["model_version"] == body["dataset_fingerprint"][:12]
    assert body["estimator"] == "CalibratedClassifierCV(LogisticRegression)"
    assert "threshold" in body
    assert "model_version" in body
    assert "dataset_fingerprint" in body
    assert body["lineage"]["content_hash"]


def test_model_card_verbose(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=42).to_csv(
        data_path, index=False
    )

    result = runner.invoke(app, ["model-card", str(trained_artifact), "--verbose"])

    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert "model_version" in body
    assert "dataset_fingerprint" in body
    assert "estimator" in body
    assert "threshold" in body
    assert "artifact_version" in body
    assert "training_config" in body
    assert "cost_policy" in body
    assert "drift_thresholds" in body
    assert "splits" in body
    assert "split_time_ranges" in body
    assert "test_metrics" in body
    assert "validation_metrics" in body
    assert "cost_policy" in body
    assert "feature_count" in body
    assert "row_count" in body
    assert "fraud_count" in body
    assert "fraud_rate" in body
    assert "feature_effects" in body
    assert "scaler_mean" in body
    assert "scaler_scale" in body
    assert body["lineage"]["dataset_fingerprint"] == body["dataset_fingerprint"]


def test_model_card_with_git_info(
    tmp_path: Path, trained_artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=42).to_csv(
        data_path, index=False
    )

    def fake_run(cmd: Any, **_kwargs: Any) -> Any:

        class Result:
            returncode = 0
            stdout = "abc123\n"

        class Result2:
            returncode = 0
            stdout = "2024-01-01 12:00:00 Initial commit\n"

        if cmd[1] == "rev-parse":
            return Result()
        return Result2()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = runner.invoke(
        app,
        [
            "model-card",
            str(trained_artifact),
            "--git-info",
        ],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert "git" in body
    assert body["git"]["commit"] == "abc123"
    assert body["git"]["last_commit"] == "2024-01-01 12:00:00 Initial commit"
    assert body["git"]["repository"] == "2024-01-01 12:00:00 Initial commit"


def test_model_card_without_git_info(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=200, random_state=42).to_csv(data_path, index=False)

    result = runner.invoke(app, ["model-card", str(trained_artifact), "--no-git-info"])

    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert "git" not in body


def test_git_info_returns_empty_for_non_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        returncode = 128
        stdout = ""

    monkeypatch.setattr("fraud_detection.cli.subprocess.run", lambda *_args, **_kwargs: Result())

    assert _git_info() == {}


def test_git_info_omits_failed_optional_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        def __init__(self, returncode: int, stdout: str) -> None:
            self.returncode = returncode
            self.stdout = stdout

    def fake_run(command: list[str], **_kwargs: object) -> Result:
        if command[1] == "rev-parse":
            return Result(0, "abc123\n")
        return Result(1, "")

    monkeypatch.setattr("fraud_detection.cli.subprocess.run", fake_run)

    assert _git_info() == {"commit": "abc123"}


def test_git_info_handles_unavailable_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_run(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr("fraud_detection.cli.subprocess.run", fail_run)

    assert _git_info() == {}


def test_model_card_reports_missing_metadata(tmp_path: Path, trained_model: FraudModel) -> None:
    artifact = tmp_path / "artifact"
    save_model(trained_model, artifact)
    metadata_path = artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["cost_policy"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = runner.invoke(app, ["model-card", str(artifact), "--verbose"])

    assert result.exit_code == 2
    assert "Artifact integrity check failed" in result.stderr


def test_train_cli_supports_temporal_gap(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=800, fraud_rate=0.1, random_state=42).to_csv(
        data_path, index=False
    )
    artifact_path = tmp_path / "artifact"

    result = runner.invoke(
        app,
        [
            "train",
            str(data_path),
            "--output",
            str(artifact_path),
            "--split-strategy",
            "temporal",
            "--temporal-gap",
            "3600",
            "--calibration-method",
            "none",
        ],
    )

    assert result.exit_code == 0, result.output
    metadata = json.loads((artifact_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["training_config"]["temporal_gap"] == 3600.0
    assert metadata["split_time_ranges"]["temporal_gap"] == 3600.0
    assert metadata["split_time_ranges"]["gaps"]["train_to_validation"] >= 3600.0
    assert metadata["split_time_ranges"]["gaps"]["validation_to_test"] >= 3600.0


def test_rolling_cli_supports_temporal_gap(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    generate_synthetic_data(rows=800, fraud_rate=0.1, random_state=42).to_csv(
        data_path, index=False
    )

    result = runner.invoke(
        app,
        [
            "rolling",
            str(data_path),
            "--origins",
            "2",
            "--temporal-gap",
            "1800",
            "--calibration-method",
            "none",
            "--estimator",
            "logistic_regression",
        ],
    )

    assert result.exit_code == 0, result.output
    results = json.loads(result.stdout)["results"]
    assert len(results) == 1


def test_validate_artifact_cli_success(tmp_path: Path, trained_model: FraudModel) -> None:
    artifact_path = tmp_path / "artifact"
    save_model(trained_model, artifact_path)

    result = runner.invoke(app, ["validate-artifact", str(artifact_path)])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["valid"] is True
    assert report["errors"] == []


def test_validate_artifact_cli_attestation_output(
    tmp_path: Path, trained_model: FraudModel
) -> None:
    artifact_path = tmp_path / "artifact"
    save_model(trained_model, artifact_path)
    att_path = tmp_path / "attestation.json"

    # Generate attestation with custom signer
    result = runner.invoke(
        app,
        [
            "validate-artifact",
            str(artifact_path),
            "--attestation-output",
            str(att_path),
            "--signer",
            "release-bot-v1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert att_path.is_file()

    # Verify content and digest
    is_valid, msg = verify_attestation(att_path)
    assert is_valid is True
    assert "successfully" in msg
    att = json.loads(att_path.read_text(encoding="utf-8"))
    assert att["verifier"]["signer"] == "release-bot-v1"
    assert att["status"] == "PASSED"

    # Test overwrite protection
    res_no_ow = runner.invoke(
        app,
        [
            "validate-artifact",
            str(artifact_path),
            "--attestation-output",
            str(att_path),
        ],
    )
    assert res_no_ow.exit_code == 2
    assert "already exists" in res_no_ow.output

    # Test overwrite permitted
    res_ow = runner.invoke(
        app,
        [
            "validate-artifact",
            str(artifact_path),
            "--attestation-output",
            str(att_path),
            "--overwrite",
            "--signer",
            "release-bot-v2",
        ],
    )
    assert res_ow.exit_code == 0, res_ow.output
    att2 = json.loads(att_path.read_text(encoding="utf-8"))
    assert att2["verifier"]["signer"] == "release-bot-v2"


def test_signed_attestation_cli_workflow_and_admission(
    tmp_path: Path,
    trained_model: FraudModel,
) -> None:
    private_key = tmp_path / "keys" / "private.pem"
    public_key = tmp_path / "keys" / "public.pem"
    second_private = tmp_path / "keys" / "second-private.pem"
    second_public = tmp_path / "keys" / "second-public.pem"
    bundle_path = tmp_path / "keys" / "trust-bundle.json"
    rotated_bundle_path = tmp_path / "keys" / "trust-bundle-rotated.json"
    artifact_path = tmp_path / "artifact"
    attestation_path = tmp_path / "attestation.json"
    unsigned_path = tmp_path / "unsigned.json"

    generated = runner.invoke(
        app,
        [
            "generate-signing-key",
            "--private-key",
            str(private_key),
            "--public-key",
            str(public_key),
        ],
    )
    assert generated.exit_code == 0, generated.output
    assert "PRIVATE KEY" not in generated.stdout
    assert private_key.is_file() and public_key.is_file()
    write_keypair(second_private, second_public)
    bundle_generated = runner.invoke(
        app,
        [
            "generate-trust-bundle",
            "--output",
            str(bundle_path),
            "--public-key",
            str(public_key),
        ],
    )
    assert bundle_generated.exit_code == 0, bundle_generated.output
    old_key_id = json.loads(bundle_generated.stdout)["active_key_ids"][0]

    signed_model = deepcopy(trained_model)
    signed_model.metadata["lineage"].update(
        {"git_commit": "a" * 40, "git_repository": "https://example.test/repo.git"}
    )
    save_model(signed_model, artifact_path)
    signed = runner.invoke(
        app,
        [
            "validate-artifact",
            str(artifact_path),
            "--strict",
            "--attestation-output",
            str(attestation_path),
            "--signing-key",
            str(private_key),
            "--signer",
            "release-key-1",
        ],
    )
    assert signed.exit_code == 0, signed.output
    assert "signature" in json.loads(attestation_path.read_text(encoding="utf-8"))

    verified = runner.invoke(
        app,
        ["verify-attestation", str(attestation_path), "--trust-bundle", str(bundle_path)],
    )
    assert verified.exit_code == 0, verified.output
    assert json.loads(verified.stdout)["valid"] is True

    rotated = runner.invoke(
        app,
        [
            "rotate-trust-bundle",
            str(bundle_path),
            "--output",
            str(rotated_bundle_path),
            "--add-public-key",
            str(second_public),
            "--revoke-key-id",
            old_key_id,
        ],
    )
    assert rotated.exit_code == 0, rotated.output
    revoked_result = runner.invoke(
        app,
        ["verify-attestation", str(attestation_path), "--trust-bundle", str(rotated_bundle_path)],
    )
    assert revoked_result.exit_code == 1
    assert json.loads(revoked_result.stdout)["signature"]["key_status"] == "revoked"

    wrong_key = runner.invoke(
        app, ["verify-attestation", str(attestation_path), str(second_public)]
    )
    assert wrong_key.exit_code == 1
    assert json.loads(wrong_key.stdout)["valid"] is False

    tampered = json.loads(attestation_path.read_text(encoding="utf-8"))
    tampered["status"] = "FAILED"
    attestation_path.write_text(json.dumps(tampered), encoding="utf-8")
    tampered_result = runner.invoke(
        app, ["verify-attestation", str(attestation_path), str(public_key)]
    )
    assert tampered_result.exit_code == 1
    assert json.loads(tampered_result.stdout)["valid"] is False

    unsigned = runner.invoke(
        app,
        [
            "validate-artifact",
            str(artifact_path),
            "--attestation-output",
            str(unsigned_path),
            "--overwrite",
        ],
    )
    assert unsigned.exit_code == 0, unsigned.output
    unsigned_result = runner.invoke(
        app,
        ["verify-attestation", str(unsigned_path), str(public_key), "--allow-unsigned"],
    )
    assert unsigned_result.exit_code == 0, unsigned_result.output


def test_validate_artifact_cli_failure_on_corrupted_file(
    tmp_path: Path, trained_model: FraudModel
) -> None:
    artifact_path = tmp_path / "artifact"
    save_model(trained_model, artifact_path)
    (artifact_path / MODEL_FILENAME).write_bytes(b"tampered_content")

    result = runner.invoke(app, ["validate-artifact", str(artifact_path)])
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["valid"] is False
    assert len(report["errors"]) > 0


def test_validate_artifact_cli_strict_fails_without_git(
    tmp_path: Path, trained_model: FraudModel
) -> None:
    artifact_path = tmp_path / "artifact"
    save_model(trained_model, artifact_path)
    # Strip git_commit from lineage
    meta_path = artifact_path / METADATA_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["lineage"].pop("git_commit", None)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    manifest_path = artifact_path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][METADATA_FILENAME] = sha256(meta_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Non-strict should pass with warning
    normal_result = runner.invoke(app, ["validate-artifact", str(artifact_path)])
    assert normal_result.exit_code == 0
    normal_report = json.loads(normal_result.stdout)
    assert normal_report["valid"] is True
    assert len(normal_report["warnings"]) > 0

    # Strict mode should fail
    strict_result = runner.invoke(app, ["validate-artifact", str(artifact_path), "--strict"])
    assert strict_result.exit_code == 1
    strict_report = json.loads(strict_result.stdout)
    assert strict_report["valid"] is False


def test_predict_cli_with_audit_log(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    output_path = tmp_path / "predictions.csv"
    audit_log = tmp_path / "audit" / "predict_audit.jsonl"
    generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=42).to_csv(
        data_path, index=False
    )

    result = runner.invoke(
        app,
        [
            "predict",
            str(trained_artifact),
            str(data_path),
            "--output",
            str(output_path),
            "--audit-log",
            str(audit_log),
        ],
    )
    assert result.exit_code == 0, result.output
    assert audit_log.is_file()
    lines = audit_log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event_type"] == "scoring"
    assert record["payload"]["batch_size"] == 200
    assert len(record["payload"]["predictions"]) == 200


def test_promote_cli_with_audit_log(tmp_path: Path, trained_artifact: Path) -> None:
    heldout_path = tmp_path / "heldout.csv"
    recent_path = tmp_path / "recent.csv"
    audit_log = tmp_path / "audit" / "promote_audit.jsonl"
    generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=81).to_csv(
        heldout_path, index=False
    )
    generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=82).to_csv(
        recent_path, index=False
    )

    result = runner.invoke(
        app,
        [
            "promote",
            str(trained_artifact),
            str(heldout_path),
            str(recent_path),
            "--audit-log",
            str(audit_log),
        ],
    )
    assert result.exit_code == 0, result.output
    assert audit_log.is_file()
    lines = audit_log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event_type"] == "promotion"
    assert "bundle" in record["payload"]
    assert "estimator" in record["payload"]["bundle"]


def test_replay_audit_cli(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    output_path = tmp_path / "predictions.csv"
    audit_log = tmp_path / "audit" / "predict_audit.jsonl"
    replay_out = tmp_path / "audit" / "replay_report.json"
    generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=42).to_csv(
        data_path, index=False
    )

    # 1. Run predict with audit log to generate audit events containing inline features
    pred_res = runner.invoke(
        app,
        [
            "predict",
            str(trained_artifact),
            str(data_path),
            "--output",
            str(output_path),
            "--audit-log",
            str(audit_log),
        ],
    )
    assert pred_res.exit_code == 0, pred_res.output

    # 2. Replay audit log against same model: should MATCH
    replay_res = runner.invoke(
        app,
        [
            "replay-audit",
            str(audit_log),
            str(trained_artifact),
            "--output",
            str(replay_out),
        ],
    )
    assert replay_res.exit_code == 0, replay_res.output
    assert replay_out.is_file()
    report = json.loads(replay_res.stdout)
    assert report["status"] == "MATCH"
    assert report["replayed_events"] == 1
    assert report["total_transactions"] == 200
    assert report["score_discrepancies"] == 0
    assert report["decision_flips"] == 0

    # 3. Replay with threshold override causing decision flips and --fail-on-divergence
    flip_res = runner.invoke(
        app,
        [
            "replay-audit",
            str(audit_log),
            str(trained_artifact),
            "--threshold",
            "0.9999",
            "--fail-on-divergence",
        ],
    )
    assert flip_res.exit_code == 1
    flip_report = json.loads(flip_res.stdout)
    assert flip_report["status"] == "DIVERGENT"
    assert flip_report["decision_flips"] > 0

    dup_res = runner.invoke(
        app,
        [
            "replay-audit",
            str(audit_log),
            str(trained_artifact),
            "--output",
            str(replay_out),
        ],
    )
    assert dup_res.exit_code != 0
    assert "Output already exists" in dup_res.output


def test_retrain_cli(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "fresh_data.csv"
    challenger_path = tmp_path / "challenger_model"
    report_path = tmp_path / "retrain_report.json"
    generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=123).to_csv(
        data_path, index=False
    )

    # 1. Successful retrain run with promotion
    res = runner.invoke(
        app,
        [
            "retrain",
            str(trained_artifact),
            str(data_path),
            "--output",
            str(challenger_path),
            "--report-output",
            str(report_path),
            "--min-gain",
            "-1.0",  # Allow non-regressive / acceptable promotion
            "--temporal-gap",
            "0.0",
        ],
    )
    assert res.exit_code == 0, res.output
    assert challenger_path.is_dir()
    assert report_path.is_file()
    report = json.loads(res.stdout)
    assert report["decision"] == "PROMOTED"
    assert report["primary_metric"] == "auprc"
    assert "champion" in report
    assert "challenger" in report
    assert report["dataset"]["total_rows"] == 200

    # 2. Rejection with --fail-on-rejection
    fail_res = runner.invoke(
        app,
        [
            "retrain",
            str(trained_artifact),
            str(data_path),
            "--output",
            str(tmp_path / "challenger_fail"),
            "--min-gain",
            "0.999",  # Unachievable gain requirement
            "--fail-on-rejection",
        ],
    )
    assert fail_res.exit_code == 1
    fail_report = json.loads(fail_res.stdout)
    assert fail_report["decision"] == "REJECTED"

    # 3. Retrain with --metric expected_cost
    cost_res = runner.invoke(
        app,
        [
            "retrain",
            str(trained_artifact),
            str(data_path),
            "--output",
            str(tmp_path / "challenger_cost"),
            "--metric",
            "expected_cost",
            "--min-gain",
            "-100.0",
        ],
    )
    assert cost_res.exit_code == 0, cost_res.output
    cost_report = json.loads(cost_res.stdout)
    assert cost_report["primary_metric"] == "expected_cost"

    # 4. Invalid metric
    bad_metric_res = runner.invoke(
        app,
        [
            "retrain",
            str(trained_artifact),
            str(data_path),
            "--metric",
            "nonexistent_metric",
        ],
    )
    assert bad_metric_res.exit_code == 2

    # 5. Retrain with --promote flag
    champ_copy = tmp_path / "champ_copy"
    shutil.copytree(trained_artifact, champ_copy)
    promote_res = runner.invoke(
        app,
        [
            "retrain",
            str(champ_copy),
            str(data_path),
            "--output",
            str(tmp_path / "challenger_promote"),
            "--min-gain",
            "-1.0",
            "--promote",
        ],
    )
    assert promote_res.exit_code == 0, promote_res.output
    promote_report = json.loads(promote_res.stdout)
    assert promote_report["decision"] == "PROMOTED"
    assert promote_report["promoted_to_champion"] is True


def test_retrain_cli_promote_rejects_file_champion(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "fresh_data.csv"
    generate_synthetic_data(rows=200, random_state=42).to_csv(data_path, index=False)
    file_champion = trained_artifact / "model.joblib"

    res = runner.invoke(
        app,
        [
            "retrain",
            str(file_champion),
            str(data_path),
            "--output",
            str(tmp_path / "challenger_out"),
            "--promote",
        ],
    )
    assert res.exit_code == 2
    assert "--promote requires champion to be an artifact directory" in res.output


def test_drift_cli_success(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "drift_data.csv"
    generate_synthetic_data(rows=250, random_state=42).to_csv(data_path, index=False)

    result = runner.invoke(app, ["drift", str(trained_artifact), str(data_path)])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["overall_status"] in ("stable", "warning", "drifted")
    assert "mean_psi" in report


def test_drift_cli_surveillance_tripped_and_webhooks(
    tmp_path: Path, trained_artifact: Path
) -> None:
    data_path = tmp_path / "drift_data.csv"
    generate_synthetic_data(rows=250, random_state=42).to_csv(data_path, index=False)

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "drift",
                str(trained_artifact),
                str(data_path),
                "--fail-on",
                "stable",
                "--webhook-slack",
                "https://hooks.slack.com/services/test/123",
                "--webhook-pagerduty",
                "pd_key_abc",
            ],
        )
        assert result.exit_code == 1
        assert "Drift surveillance tripped" in result.output
        assert mock_urlopen.call_count == 2


def test_drift_cli_webhook_failure(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "drift_data.csv"
    generate_synthetic_data(rows=250, random_state=42).to_csv(data_path, index=False)

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        result = runner.invoke(
            app,
            [
                "drift",
                str(trained_artifact),
                str(data_path),
                "--fail-on",
                "stable",
                "--webhook-slack",
                "https://hooks.slack.com/services/test/123",
            ],
        )
        assert result.exit_code == 1
        assert "Warning: Failed to send webhook" in result.output


def test_replay_audit_cli_invalid_model_error(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    log_path.write_text("{}", encoding="utf-8")
    bad_model = tmp_path / "bad_model"
    bad_model.mkdir()
    result = runner.invoke(
        app,
        [
            "replay-audit",
            str(log_path),
            str(bad_model),
        ],
    )
    assert result.exit_code == 2
    assert "Error:" in result.output


def test_multi_window_drift_cli_success(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "drift_data.csv"
    generate_synthetic_data(rows=250, random_state=42).to_csv(data_path, index=False)
    out_file = tmp_path / "mw_report.json"

    result = runner.invoke(
        app,
        [
            "multi-window-drift",
            str(trained_artifact),
            str(data_path),
            "--short-window-rows",
            "50",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["short_window_rows"] == 50
    assert report["long_window_rows"] == 250
    assert report["overall_status"] in ("stable", "warning", "drifted")
    assert "max_velocity" in report
    assert out_file.exists()


def test_multi_window_drift_cli_surveillance_tripped_and_webhooks(
    tmp_path: Path, trained_artifact: Path
) -> None:
    data_path = tmp_path / "drift_data.csv"
    generate_synthetic_data(rows=250, random_state=42).to_csv(data_path, index=False)

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "multi-window-drift",
                str(trained_artifact),
                str(data_path),
                "--short-window-rows",
                "50",
                "--fail-on",
                "stable",
                "--webhook-slack",
                "https://hooks.slack.com/services/test/123",
                "--webhook-pagerduty",
                "pd_key_abc",
            ],
        )
        assert result.exit_code == 1
        assert "Multi-window drift surveillance tripped" in result.output
        assert mock_urlopen.call_count == 2


def test_multi_window_drift_cli_webhook_slack_only(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "drift_data.csv"
    generate_synthetic_data(rows=250, random_state=42).to_csv(data_path, index=False)

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "multi-window-drift",
                str(trained_artifact),
                str(data_path),
                "--short-window-rows",
                "50",
                "--fail-on",
                "stable",
                "--webhook-slack",
                "https://hooks.slack.com/services/test/123",
            ],
        )
        assert result.exit_code == 1
        assert mock_urlopen.call_count == 1


def test_multi_window_drift_cli_webhook_pagerduty_critical(
    tmp_path: Path, trained_artifact: Path
) -> None:
    data_path = tmp_path / "drifted_heavy.csv"
    df = generate_synthetic_data(rows=250, random_state=42)
    df.loc[df.index[-50:], "Amount"] = 999999.0
    df.to_csv(data_path, index=False)

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "multi-window-drift",
                str(trained_artifact),
                str(data_path),
                "--short-window-rows",
                "50",
                "--fail-on",
                "drifted",
                "--webhook-pagerduty",
                "pd_routing_key_123",
            ],
        )
        assert result.exit_code == 1
        assert mock_urlopen.call_count == 1


def test_multi_window_drift_cli_error_handling(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "short_data.csv"
    generate_synthetic_data(rows=250, random_state=42).iloc[:10].to_csv(data_path, index=False)

    result = runner.invoke(
        app,
        [
            "multi-window-drift",
            str(trained_artifact),
            str(data_path),
            "--short-window-rows",
            "50",
        ],
    )
    assert result.exit_code == 2
    assert "less than short_window_rows" in result.output


def test_stream_profile_cli_csv(tmp_path: Path, trained_artifact: Path) -> None:
    csv_path = tmp_path / "stream_data.csv"
    generate_synthetic_data(rows=250, random_state=42).to_csv(csv_path, index=False)
    out_profile = tmp_path / "profile.json"
    chk_path = tmp_path / "checkpoint.json"

    result = runner.invoke(
        app,
        [
            "stream-profile",
            str(csv_path),
            "-o",
            str(out_profile),
            "--model-path",
            str(trained_artifact),
            "--checkpoint-output",
            str(chk_path),
            "--batch-size",
            "100",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Successfully updated streaming profile" in result.output

    data = json.loads(out_profile.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "Amount" in data
    assert "proportions" in data["Amount"]

    chk = json.loads(chk_path.read_text(encoding="utf-8"))
    assert "features" in chk
    assert chk["features"]["Amount"]["total_count"] == 250


def test_stream_profile_cli_resume_from_checkpoint(tmp_path: Path, trained_artifact: Path) -> None:
    csv_path = tmp_path / "stream_data.csv"
    generate_synthetic_data(rows=250, random_state=42).to_csv(csv_path, index=False)
    out_profile1 = tmp_path / "p1.json"
    chk_path = tmp_path / "checkpoint.json"

    r1 = runner.invoke(
        app,
        [
            "stream-profile",
            str(csv_path),
            "-o",
            str(out_profile1),
            "--model-path",
            str(trained_artifact),
            "--checkpoint-output",
            str(chk_path),
        ],
    )
    assert r1.exit_code == 0, r1.output

    # Resume from checkpoint on another chunk
    out_profile2 = tmp_path / "p2.json"
    r2 = runner.invoke(
        app,
        [
            "stream-profile",
            str(csv_path),
            "-o",
            str(out_profile2),
            "--state-input",
            str(chk_path),
            "--checkpoint-output",
            str(chk_path),
            "--overwrite",
        ],
    )
    assert r2.exit_code == 0, r2.output
    chk2 = json.loads(chk_path.read_text(encoding="utf-8"))
    assert chk2["features"]["Amount"]["total_count"] == 500


def test_stream_profile_cli_jsonl(tmp_path: Path, trained_artifact: Path) -> None:
    jsonl_path = tmp_path / "audit.jsonl"
    frame = generate_synthetic_data(rows=250, random_state=42)
    records = frame.drop(columns="Class").to_dict(orient="records")

    with jsonl_path.open("w", encoding="utf-8") as f:
        # Write some scoring events
        event1 = {
            "event_type": "scoring",
            "payload": {"features": records[:100]},
        }
        event2 = {
            "event_type": "scoring",
            "payload": {"features": records[100:]},
        }
        f.write(json.dumps(event1) + "\n")
        f.write(json.dumps(event2) + "\n")

    out_profile = tmp_path / "profile_jsonl.json"
    result = runner.invoke(
        app,
        [
            "stream-profile",
            str(jsonl_path),
            "-o",
            str(out_profile),
            "--model-path",
            str(trained_artifact),
            "--batch-size",
            "50",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_profile.read_text(encoding="utf-8"))
    assert "Amount" in data


def test_stream_profile_cli_error_empty_data(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    out_file = tmp_path / "out.json"

    result = runner.invoke(
        app,
        [
            "stream-profile",
            str(empty_file),
            "-o",
            str(out_file),
        ],
    )
    assert result.exit_code == 2


def test_stream_profile_cli_csv_auto_init(tmp_path: Path) -> None:
    csv_path = tmp_path / "stream_data.csv"
    generate_synthetic_data(rows=200, random_state=42).to_csv(csv_path, index=False)
    out_profile = tmp_path / "auto_profile.json"

    result = runner.invoke(
        app,
        [
            "stream-profile",
            str(csv_path),
            "-o",
            str(out_profile),
            "--batch-size",
            "50",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_profile.read_text(encoding="utf-8"))
    assert "Amount" in data
    assert "proportions" in data["Amount"]


def test_stream_profile_cli_jsonl_auto_init_and_filtering(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "stream.jsonl"
    frame = generate_synthetic_data(rows=200, random_state=42)
    records = frame.drop(columns="Class").to_dict(orient="records")

    with jsonl_path.open("w", encoding="utf-8") as f:
        f.write("\n")
        f.write("not-a-valid-json\n")
        f.write('"just-a-string"\n')
        f.write(json.dumps({"event_type": "ignored"}) + "\n")
        f.write(json.dumps({"payload": {"features": records[:100]}}) + "\n")
        f.write(json.dumps({"payload": {"features": records[100:]}}) + "\n")

    out_profile = tmp_path / "auto_jsonl.json"
    result = runner.invoke(
        app,
        [
            "stream-profile",
            str(jsonl_path),
            "-o",
            str(out_profile),
            "--batch-size",
            "50",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_profile.read_text(encoding="utf-8"))
    assert "Amount" in data


def test_stream_profile_cli_errors(tmp_path: Path, trained_artifact: Path) -> None:
    csv_path = tmp_path / "stream_data.csv"
    generate_synthetic_data(rows=200, random_state=42).to_csv(csv_path, index=False)
    out_profile = tmp_path / "err_profile.json"

    res1 = runner.invoke(
        app,
        [
            "stream-profile",
            str(csv_path),
            "-o",
            str(out_profile),
            "--state-input",
            str(tmp_path / "non_existent_state.json"),
        ],
    )
    assert res1.exit_code == 2
    assert "State input file not found" in res1.output

    model = load_model(trained_artifact)
    model.metadata.pop("reference_profile", None)
    corrupted_model_path = tmp_path / "no_profile_model.joblib"
    save_model(model, corrupted_model_path)

    res2 = runner.invoke(
        app,
        [
            "stream-profile",
            str(csv_path),
            "-o",
            str(out_profile),
            "--model-path",
            str(corrupted_model_path),
        ],
    )
    assert res2.exit_code == 2
    assert "reference profile" in res2.output

    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text('{"empty": true}\n', encoding="utf-8")
    res3 = runner.invoke(
        app,
        [
            "stream-profile",
            str(bad_jsonl),
            "-o",
            str(out_profile),
        ],
    )
    assert res3.exit_code == 2
    assert "No valid transaction records" in res3.output


def test_simulate_drift_cli_success(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    generate_synthetic_data(rows=250, random_state=42).to_csv(csv_path, index=False)
    drifted_path = tmp_path / "drifted.csv"

    result = runner.invoke(
        app,
        [
            "simulate-drift",
            str(csv_path),
            "-o",
            str(drifted_path),
            "--mean-offset",
            "10.0",
            "--variance-scale",
            "1.5",
            "--sample-fraction",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Successfully injected drift" in result.output
    assert drifted_path.is_file()

    drifted_df = pd.read_csv(drifted_path)
    assert len(drifted_df) == 250


def test_simulate_drift_cli_specific_features(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    orig_df = generate_synthetic_data(rows=250, random_state=42)
    orig_df.to_csv(csv_path, index=False)
    drifted_path = tmp_path / "drifted_amount.csv"

    result = runner.invoke(
        app,
        [
            "simulate-drift",
            str(csv_path),
            "-o",
            str(drifted_path),
            "--features",
            "Amount",
            "--mean-offset",
            "100.0",
        ],
    )
    assert result.exit_code == 0, result.output
    drifted_df = pd.read_csv(drifted_path)
    assert drifted_df["Amount"].mean() > orig_df["Amount"].mean() + 80.0
    pd.testing.assert_series_equal(drifted_df["V1"], orig_df["V1"])


def test_simulate_drift_cli_validation_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    generate_synthetic_data(rows=250, random_state=42).to_csv(csv_path, index=False)
    drifted_path = tmp_path / "drifted.csv"

    result = runner.invoke(
        app,
        [
            "simulate-drift",
            str(csv_path),
            "-o",
            str(drifted_path),
            "--features",
            "non_existent",
        ],
    )
    assert result.exit_code == 2
    assert "Target features not found" in result.output


def test_simulate_drift_e2e_with_multi_window_surveillance(
    tmp_path: Path, trained_artifact: Path
) -> None:
    csv_path = tmp_path / "baseline.csv"
    generate_synthetic_data(rows=300, random_state=42).to_csv(csv_path, index=False)
    drifted_path = tmp_path / "chaos_drifted.csv"

    # Inject extreme tail drift in the last 50 rows
    sim_res = runner.invoke(
        app,
        [
            "simulate-drift",
            str(csv_path),
            "-o",
            str(drifted_path),
            "--mean-offset",
            "20.0",
            "--sample-fraction",
            "0.166",
        ],
    )
    assert sim_res.exit_code == 0, sim_res.output

    # Multi-window surveillance on the shifted dataset
    mw_res = runner.invoke(
        app,
        [
            "multi-window-drift",
            str(trained_artifact),
            str(drifted_path),
            "--short-window-rows",
            "50",
            "--fail-on",
            "warning",
        ],
    )
    assert mw_res.exit_code == 1
    assert "Multi-window drift surveillance tripped" in mw_res.output


def _sample_compliance_bundle() -> dict[str, Any]:
    return {
        "model_version": "abc123456789",
        "model": {
            "estimator": "LogisticRegression",
            "threshold": 0.42,
            "created_at": "2026-09-14T10:00:00+00:00",
            "cost_policy": {
                "name": "strict-recall",
                "false_positive_cost": 1.0,
                "false_negative_cost": 25.0,
            },
            "test_metrics": {
                "roc_auc": 0.91,
                "average_precision": 0.62,
                "brier_score": 0.04,
                "precision": 0.5,
                "recall": 0.8,
                "f1": 0.62,
                "balanced_accuracy": 0.88,
                "expected_cost_per_transaction": 0.12,
            },
        },
        "calibration": {
            "rows": 100,
            "bins": 2,
            "brier_score": 0.04,
            "expected_calibration_error": 0.03,
            "max_calibration_error": 0.08,
            "reliability": 0.01,
            "resolution": 0.12,
            "uncertainty": 0.09,
            "detail": [
                {
                    "bin_index": 0,
                    "lower": 0.0,
                    "upper": 0.5,
                    "count": 80,
                    "mean_predicted": 0.1,
                    "fraction_positive": 0.08,
                }
            ],
        },
        "thresholds": {
            "model_threshold_metrics": {
                "threshold": 0.42,
                "precision": 0.5,
                "recall": 0.8,
                "f1": 0.62,
                "expected_cost_per_transaction": 0.12,
                "flagged": 20,
                "flagged_rate": 0.2,
            },
            "detail": [
                {
                    "threshold": 0.5,
                    "precision": 0.6,
                    "recall": 0.7,
                    "f1": 0.65,
                    "expected_cost_per_transaction": 0.11,
                    "flagged": 15,
                    "flagged_rate": 0.15,
                }
            ],
        },
        "drift": {
            "rows": 100,
            "overall_status": "drifted",
            "mean_psi": 0.13,
            "max_psi": 0.31,
            "thresholds": {"warning_at": 0.1, "drift_at": 0.25},
            "features": [
                {"feature": "Amount", "psi": 0.31, "status": "drifted"},
                {"feature": "Time", "psi": 0.12, "status": "warning"},
            ],
        },
        "benchmark": {
            "results": [
                {
                    "batch_size": 1,
                    "median_ms": 1.2,
                    "ms_per_transaction": 1.2,
                    "transactions_per_second": 833.3,
                }
            ]
        },
    }


def test_compliance_cli_success_and_errors(tmp_path: Path, trained_artifact: Path) -> None:
    bundle_data = _sample_compliance_bundle()
    bundle_path = tmp_path / "promotion_bundle.json"
    bundle_path.write_text(json.dumps(bundle_data), encoding="utf-8")
    output_html = tmp_path / "compliance.html"

    # 1. Success without optional flags
    res = runner.invoke(app, ["compliance", str(bundle_path), "--output", str(output_html)])
    assert res.exit_code == 0, res.output
    assert output_html.is_file()
    assert "<!DOCTYPE html>" in output_html.read_text(encoding="utf-8")

    # 2. Success with stability and artifact manifest
    stab_path = tmp_path / "stability.json"
    stab_data = {
        "results": [
            {
                "start_date": "2026-01-01",
                "end_date": "2026-01-07",
                "train_rows": 100,
                "test_rows": 50,
                "test_fraud_rate": 0.05,
                "roc_auc": 0.92,
                "average_precision": 0.85,
                "brier_score": 0.04,
                "expected_cost_per_transaction": 0.12,
            }
        ]
    }
    stab_path.write_text(json.dumps(stab_data), encoding="utf-8")
    res_full = runner.invoke(
        app,
        [
            "compliance",
            str(bundle_path),
            "--stability",
            str(stab_path),
            "--artifact",
            str(trained_artifact),
        ],
    )
    assert res_full.exit_code == 0, res_full.output
    assert "<!DOCTYPE html>" in res_full.output

    # 3. Invalid JSON in bundle
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    res_bad = runner.invoke(app, ["compliance", str(bad_json)])
    assert res_bad.exit_code == 2
    assert "Invalid promotion bundle JSON" in res_bad.output

    # 4. Non-dict JSON in bundle
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    res_list = runner.invoke(app, ["compliance", str(list_json)])
    assert res_list.exit_code == 2
    assert "The promotion bundle must be a JSON object" in res_list.output


def test_export_edge_cli_max_error_exceeded(tmp_path: Path) -> None:
    uncalibrated_model = train_model(
        validate_frame(generate_synthetic_data(rows=300, fraud_rate=0.08, random_state=42)),
        config=TrainingConfig(calibration_method=CalibrationMethod.NONE),
    )
    uncal_art = tmp_path / "art_uncal"
    save_model(uncalibrated_model, uncal_art)

    data_path = tmp_path / "val.csv"
    generate_synthetic_data(rows=300, fraud_rate=0.08, random_state=42).to_csv(
        data_path, index=False
    )
    out_edge = tmp_path / "edge.json"

    res = runner.invoke(
        app,
        [
            "export-edge",
            str(uncal_art),
            "--output",
            str(out_edge),
            "--validation-data",
            str(data_path),
            "--max-error",
            "0.0",
        ],
    )
    assert res.exit_code == 2
    assert "exceeds the probability error tolerance" in res.output


def test_trust_bundle_cli_failure_paths(tmp_path: Path) -> None:
    # 1. generate-trust-bundle without --public-key
    res1 = runner.invoke(app, ["generate-trust-bundle"])
    assert res1.exit_code == 2
    assert "At least one --public-key is required" in res1.output

    # 2. generate-trust-bundle with invalid/corrupt public key
    corrupt_key = tmp_path / "corrupt_pub.pem"
    corrupt_key.write_text("not a pem", encoding="utf-8")
    res2 = runner.invoke(app, ["generate-trust-bundle", "--public-key", str(corrupt_key)])
    assert res2.exit_code == 2

    # 3. rotate-trust-bundle without --add-public-key or --revoke-key-id
    bundle_path = tmp_path / "bundle.json"
    valid_key = tmp_path / "key.pem"
    runner.invoke(
        app,
        [
            "generate-signing-key",
            "--private-key",
            str(tmp_path / "priv.pem"),
            "--public-key",
            str(valid_key),
        ],
    )
    res_gen = runner.invoke(
        app,
        ["generate-trust-bundle", "--output", str(bundle_path), "--public-key", str(valid_key)],
    )
    assert res_gen.exit_code == 0

    res3 = runner.invoke(app, ["rotate-trust-bundle", str(bundle_path)])
    assert res3.exit_code == 2
    assert "Provide --add-public-key or --revoke-key-id" in res3.output

    # 4. rotate-trust-bundle revoking an unknown key ID
    res4 = runner.invoke(
        app,
        [
            "rotate-trust-bundle",
            str(bundle_path),
            "--revoke-key-id",
            "unknown-key-1234567890abcdef",
        ],
    )
    assert res4.exit_code == 2


def test_verify_attestation_cli_failure_paths(tmp_path: Path) -> None:
    att_path = tmp_path / "att.json"
    att_path.write_text("{}", encoding="utf-8")
    pub_key = tmp_path / "pub.pem"
    pub_key.write_text("dummy", encoding="utf-8")
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text("{}", encoding="utf-8")

    # 1. Neither or both public_key and trust_bundle
    res_neither = runner.invoke(app, ["verify-attestation", str(att_path)])
    assert res_neither.exit_code == 2
    assert "Provide exactly one" in res_neither.output

    res_both = runner.invoke(
        app,
        ["verify-attestation", str(att_path), str(pub_key), "--trust-bundle", str(bundle_path)],
    )
    assert res_both.exit_code == 2
    assert "Provide exactly one" in res_both.output

    # 2. Corrupt JSON in attestation
    bad_att = tmp_path / "bad_att.json"
    bad_att.write_text("{corrupt", encoding="utf-8")
    res_corrupt = runner.invoke(app, ["verify-attestation", str(bad_att), str(pub_key)])
    assert res_corrupt.exit_code == 1
    assert "valid" in res_corrupt.output

    # 3. Non-dict root in attestation
    list_att = tmp_path / "list_att.json"
    list_att.write_text("[]", encoding="utf-8")
    keypair_priv = tmp_path / "k_priv.pem"
    keypair_pub = tmp_path / "k_pub.pem"
    runner.invoke(
        app,
        [
            "generate-signing-key",
            "--private-key",
            str(keypair_priv),
            "--public-key",
            str(keypair_pub),
        ],
    )
    res_list = runner.invoke(app, ["verify-attestation", str(list_att), str(keypair_pub)])
    assert res_list.exit_code == 1
    assert "Attestation root must be a JSON object" in res_list.output

    # 4. Attestation missing signature when require_signature is True
    no_sig_att = tmp_path / "no_sig_att.json"
    no_sig_att.write_text(json.dumps({"status": "PASSED"}), encoding="utf-8")
    res_no_sig = runner.invoke(app, ["verify-attestation", str(no_sig_att), str(keypair_pub)])
    assert res_no_sig.exit_code == 1
    assert "Attestation has no signature" in res_no_sig.output


def test_validate_artifact_cli_signing_flags(tmp_path: Path, trained_artifact: Path) -> None:
    priv_key = tmp_path / "sign_priv.pem"
    pub_key = tmp_path / "sign_pub.pem"
    runner.invoke(
        app,
        ["generate-signing-key", "--private-key", str(priv_key), "--public-key", str(pub_key)],
    )

    # 1. --signing-key without --attestation-output
    res1 = runner.invoke(
        app,
        ["validate-artifact", str(trained_artifact), "--strict", "--signing-key", str(priv_key)],
    )
    assert res1.exit_code == 2
    assert "--signing-key requires --attestation-output" in res1.output

    # 2. --signing-key without --strict
    res2 = runner.invoke(
        app,
        [
            "validate-artifact",
            str(trained_artifact),
            "--signing-key",
            str(priv_key),
            "--attestation-output",
            str(tmp_path / "att.json"),
        ],
    )
    assert res2.exit_code == 2
    assert "--signing-key requires --strict validation" in res2.output

    # 3. Corrupt private key
    corrupt_priv = tmp_path / "corrupt_priv.pem"
    corrupt_priv.write_text("bad key", encoding="utf-8")
    res3 = runner.invoke(
        app,
        [
            "validate-artifact",
            str(trained_artifact),
            "--strict",
            "--signing-key",
            str(corrupt_priv),
            "--attestation-output",
            str(tmp_path / "att.json"),
        ],
    )
    assert res3.exit_code == 2


def test_retrain_cli_more_branches(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "fresh_data.csv"
    generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=42).to_csv(
        data_path, index=False
    )

    # 1. Retrain with --metric f1
    res_f1 = runner.invoke(
        app,
        [
            "retrain",
            str(trained_artifact),
            str(data_path),
            "--output",
            str(tmp_path / "chall_f1"),
            "--metric",
            "f1",
            "--min-gain",
            "-1.0",
        ],
    )
    assert res_f1.exit_code == 0, res_f1.output
    report_f1 = json.loads(res_f1.stdout)
    assert report_f1["primary_metric"] == "f1"

    # 2. Invalid training data causes DataValidationError / abort
    bad_data = tmp_path / "bad_data.csv"
    bad_data.write_text("invalid,csv\n1\n", encoding="utf-8")
    res_bad = runner.invoke(
        app,
        [
            "retrain",
            str(trained_artifact),
            str(bad_data),
            "--output",
            str(tmp_path / "chall_bad"),
        ],
    )
    assert res_bad.exit_code == 2


def test_stream_profile_cli_jsonl_fewer_than_batch_size(tmp_path: Path) -> None:
    frame = generate_synthetic_data(rows=200, random_state=42)
    jsonl_path = tmp_path / "small.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in frame.to_dict(orient="records"):
            f.write(json.dumps({"payload": {"features": [record]}}) + "\n")

    out_prof = tmp_path / "stream_out.json"
    res = runner.invoke(
        app,
        [
            "stream-profile",
            str(jsonl_path),
            "--output",
            str(out_prof),
            "--batch-size",
            "500",
        ],
    )
    assert res.exit_code == 0, res.output
    assert out_prof.is_file()


def test_multi_window_drift_missing_reference_profile(
    tmp_path: Path, trained_artifact: Path
) -> None:
    m = load_model(trained_artifact)
    m.metadata.pop("reference_profile", None)
    standalone_joblib = tmp_path / "model.joblib"
    joblib.dump(m, standalone_joblib)

    data_path = tmp_path / "data.csv"
    generate_synthetic_data(rows=200, random_state=42).to_csv(data_path, index=False)

    res = runner.invoke(app, ["multi-window-drift", str(standalone_joblib), str(data_path)])
    assert res.exit_code == 2
    assert "does not contain a reference profile" in res.output


def test_cli_main_module_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["fraud-detect", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("fraud_detection.cli", run_name="__main__")
    assert exc_info.value.code == 0


def test_retrain_cli_output_already_exists(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "data.csv"
    generate_synthetic_data(rows=200, random_state=42).to_csv(data_path, index=False)
    existing_out = tmp_path / "existing_challenger"
    existing_out.mkdir()
    (existing_out / "file.txt").write_text("exists", encoding="utf-8")

    res = runner.invoke(
        app,
        [
            "retrain",
            str(trained_artifact),
            str(data_path),
            "--output",
            str(existing_out),
        ],
    )
    assert res.exit_code == 2
    assert "Output already exists" in res.output


def test_retrain_cli_fallback_champion_metadata(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "data.csv"
    data = generate_synthetic_data(rows=300, fraud_rate=0.08, random_state=42)
    data.to_csv(data_path, index=False)

    # 1. Champion with invalid config values triggers fallbacks
    m = load_model(trained_artifact)
    m.metadata["training_config"] = {
        "estimator": "invalid_est",
        "threshold_strategy": "invalid_thresh",
        "calibration_method": "invalid_calib",
        "split_strategy": "invalid_split",
    }
    champ_invalid = tmp_path / "champ_invalid.joblib"
    joblib.dump(m, champ_invalid)

    res1 = runner.invoke(
        app,
        [
            "retrain",
            str(champ_invalid),
            str(data_path),
            "--output",
            str(tmp_path / "out1"),
        ],
    )
    assert res1.exit_code == 0, res1.output

    # 2. Champion with no training_config triggers estimator name inference (e.g. 'forest')
    m2 = load_model(trained_artifact)
    m2.metadata.pop("training_config", None)
    m2.metadata["estimator"] = "RandomForestClassifier"
    champ_legacy = tmp_path / "champ_legacy.joblib"
    joblib.dump(m2, champ_legacy)

    res2 = runner.invoke(
        app,
        [
            "retrain",
            str(champ_legacy),
            str(data_path),
            "--output",
            str(tmp_path / "out2"),
        ],
    )
    assert res2.exit_code == 0, res2.output


def test_retrain_cli_recall_guardrail_failure(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "data.csv"
    data = generate_synthetic_data(rows=300, fraud_rate=0.08, random_state=42)
    data.to_csv(data_path, index=False)

    from fraud_detection.evaluation import ClassificationMetrics

    mock_champ_eval = ClassificationMetrics(
        threshold=0.5,
        roc_auc=0.9,
        average_precision=0.8,
        brier_score=0.05,
        precision=0.7,
        recall=0.8,
        f1=0.75,
        balanced_accuracy=0.85,
        true_positives=8,
        false_positives=3,
        true_negatives=87,
        false_negatives=2,
    )
    mock_chall_eval = ClassificationMetrics(
        threshold=0.5,
        roc_auc=0.9,
        average_precision=0.9,
        brier_score=0.04,
        precision=0.9,
        recall=0.01,
        f1=0.02,
        balanced_accuracy=0.5,
        true_positives=1,
        false_positives=0,
        true_negatives=90,
        false_negatives=9,
    )

    with patch(
        "fraud_detection.cli.evaluate_predictions",
        side_effect=[mock_chall_eval, mock_champ_eval],
    ):
        res = runner.invoke(
            app,
            [
                "retrain",
                str(trained_artifact),
                str(data_path),
                "--output",
                str(tmp_path / "out_guardrail"),
            ],
        )
    assert res.exit_code == 0
    report = json.loads(res.stdout)
    assert report["decision"] == "REJECTED"
    assert "collapsed below 0.05 guardrail" in report["decision_reason"]


def test_replay_audit_log_cli_write_error(tmp_path: Path, trained_artifact: Path) -> None:
    audit_file = tmp_path / "audit.jsonl"
    audit_file.write_text('{"event": "score"}\n', encoding="utf-8")
    rep_out = tmp_path / "report.json"

    with patch.object(Path, "write_text", side_effect=OSError("Disk write error")):
        res = runner.invoke(
            app,
            [
                "replay-audit",
                str(audit_file),
                str(trained_artifact),
                "--output",
                str(rep_out),
            ],
        )
    assert res.exit_code == 2
    assert "Failed to write replay report" in res.output


def test_generate_signing_key_cli_signing_error(tmp_path: Path) -> None:
    from fraud_detection.signing import SigningError

    with patch("fraud_detection.cli.write_keypair", side_effect=SigningError("Keygen failed")):
        res = runner.invoke(
            app,
            [
                "generate-signing-key",
                "--private-key",
                str(tmp_path / "priv.pem"),
                "--public-key",
                str(tmp_path / "pub.pem"),
            ],
        )
    assert res.exit_code == 2
    assert "Keygen failed" in res.output


def test_export_edge_cli_without_validation_data(tmp_path: Path) -> None:
    uncalibrated_model = train_model(
        validate_frame(generate_synthetic_data(rows=300, fraud_rate=0.08, random_state=42)),
        config=TrainingConfig(calibration_method=CalibrationMethod.NONE),
    )
    uncal_art = tmp_path / "art_uncal2"
    save_model(uncalibrated_model, uncal_art)
    out_edge = tmp_path / "edge_noval.json"

    res = runner.invoke(
        app,
        [
            "export-edge",
            str(uncal_art),
            "--output",
            str(out_edge),
        ],
    )
    assert res.exit_code == 0, res.output
    assert out_edge.is_file()


def test_predict_command_with_tiered_thresholds_and_audit(
    tmp_path: Path, trained_artifact: Path
) -> None:
    data_path = tmp_path / "data.csv"
    data = generate_synthetic_data(rows=200, random_state=12)
    data.to_csv(data_path, index=False)
    out_csv = tmp_path / "preds.csv"
    audit_log = tmp_path / "audit.jsonl"

    res = runner.invoke(
        app,
        [
            "predict",
            str(trained_artifact),
            str(data_path),
            "--output",
            str(out_csv),
            "--review-threshold",
            "0.15",
            "--deny-threshold",
            "0.85",
            "--audit-log",
            str(audit_log),
        ],
    )
    assert res.exit_code == 0, res.output
    summary = json.loads(res.output)
    assert summary["review_threshold"] == 0.15
    assert summary["deny_threshold"] == 0.85
    assert "decision_counts" in summary
    assert "ALLOW" in summary["decision_counts"]
    assert "CHALLENGE" in summary["decision_counts"]
    assert "DENY" in summary["decision_counts"]

    df = pd.read_csv(out_csv)
    assert "decision" in df.columns
    assert set(df["decision"].unique()).issubset({"ALLOW", "CHALLENGE", "DENY"})

    audit_lines = audit_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(audit_lines) == 1
    event = json.loads(audit_lines[0])
    assert event["event_type"] == "scoring"
    assert event["payload"]["review_threshold"] == 0.15
    assert event["payload"]["deny_threshold"] == 0.85
    assert "decision" in event["payload"]["predictions"][0]


def test_predict_command_tiered_threshold_validation(
    tmp_path: Path, trained_artifact: Path
) -> None:
    data_path = tmp_path / "data.csv"
    data = generate_synthetic_data(rows=200, random_state=1)
    data.to_csv(data_path, index=False)

    # Missing one threshold
    res1 = runner.invoke(
        app,
        [
            "predict",
            str(trained_artifact),
            str(data_path),
            "--review-threshold",
            "0.2",
        ],
    )
    assert res1.exit_code == 2
    assert "Both --review-threshold and --deny-threshold must be supplied together" in res1.output

    # review > deny
    res2 = runner.invoke(
        app,
        [
            "predict",
            str(trained_artifact),
            str(data_path),
            "--review-threshold",
            "0.8",
            "--deny-threshold",
            "0.2",
        ],
    )
    assert res2.exit_code == 2
    assert "--review-threshold cannot exceed --deny-threshold" in res2.output

    # out of bounds
    res3 = runner.invoke(
        app,
        [
            "predict",
            str(trained_artifact),
            str(data_path),
            "--review-threshold",
            "-0.1",
            "--deny-threshold",
            "0.5",
        ],
    )
    assert res3.exit_code == 2
    assert "Tiered thresholds must fall between 0.0 and 1.0" in res3.output


def test_score_drift_command_csv_and_reporting(tmp_path: Path, trained_model: FraudModel) -> None:
    art_path = tmp_path / "model_art"
    save_model(trained_model, art_path)

    # Generate predictions CSV
    df = pd.DataFrame({"fraud_probability": [0.01, 0.05, 0.02, 0.95, 0.12]})
    csv_path = tmp_path / "preds.csv"
    df.to_csv(csv_path, index=False)

    rep_out = tmp_path / "score_drift.json"
    res = runner.invoke(
        app,
        [
            "score-drift",
            str(art_path),
            str(csv_path),
            "--output",
            str(rep_out),
        ],
    )
    assert res.exit_code == 0, res.output
    assert rep_out.is_file()
    data = json.loads(rep_out.read_text(encoding="utf-8"))
    assert "psi" in data
    assert "status" in data
    assert "score_shift" in data


def test_score_drift_command_formats(tmp_path: Path, trained_model: FraudModel) -> None:
    art_path = tmp_path / "model_art"
    save_model(trained_model, art_path)

    # TSV format
    tsv_path = tmp_path / "preds.tsv"
    tsv_df = pd.DataFrame({"score": [0.02, 0.04, 0.03, 0.01]})
    tsv_df.to_csv(tsv_path, sep="\t", index=False)

    res_tsv = runner.invoke(
        app,
        [
            "score-drift",
            str(art_path),
            str(tsv_path),
            "--score-column",
            "score",
        ],
    )
    assert res_tsv.exit_code == 0, res_tsv.output

    # JSONL format with dicts
    jsonl_path = tmp_path / "preds.jsonl"
    jsonl_path.write_text(
        json.dumps({"fraud_probability": 0.05})
        + "\n"
        + json.dumps({"fraud_probability": 0.02})
        + "\n",
        encoding="utf-8",
    )
    res_jsonl = runner.invoke(
        app,
        [
            "score-drift",
            str(art_path),
            str(jsonl_path),
        ],
    )
    assert res_jsonl.exit_code == 0, res_jsonl.output

    # JSONL format with raw numbers
    jsonl_nums_path = tmp_path / "preds_nums.jsonl"
    jsonl_nums_path.write_text("0.1\n0.2\n0.05\n", encoding="utf-8")
    res_nums = runner.invoke(
        app,
        [
            "score-drift",
            str(art_path),
            str(jsonl_nums_path),
        ],
    )
    assert res_nums.exit_code == 0, res_nums.output

    # Plain text format
    txt_path = tmp_path / "preds.txt"
    txt_path.write_text("0.05\n0.12\n0.01\n", encoding="utf-8")
    res_txt = runner.invoke(
        app,
        [
            "score-drift",
            str(art_path),
            str(txt_path),
        ],
    )
    assert res_txt.exit_code == 0, res_txt.output


def test_score_drift_command_fail_on_and_alert_webhook(
    tmp_path: Path, trained_model: FraudModel
) -> None:
    art_path = tmp_path / "model_art"
    save_model(trained_model, art_path)

    # Heavily shifted predictions -> should cause drift
    shifted_df = pd.DataFrame({"fraud_probability": [0.99] * 50})
    csv_path = tmp_path / "shifted.csv"
    shifted_df.to_csv(csv_path, index=False)

    # Invalid fail-on
    res_inv = runner.invoke(
        app,
        [
            "score-drift",
            str(art_path),
            str(csv_path),
            "--fail-on",
            "critical",
        ],
    )
    assert res_inv.exit_code == 2
    assert "Invalid --fail-on 'critical'" in res_inv.output

    # Tripped fail-on with webhook
    with patch("fraud_detection.cli._post_webhook") as mock_post:
        res_trip = runner.invoke(
            app,
            [
                "score-drift",
                str(art_path),
                str(csv_path),
                "--fail-on",
                "warning",
                "--alert-webhook-url",
                "https://alerts.internal/drift",
            ],
        )
        assert res_trip.exit_code == 1
        assert "Score drift surveillance tripped" in res_trip.output
        mock_post.assert_called_once()
        call_args = mock_post.call_args[0]
        assert call_args[0] == "https://alerts.internal/drift"
        assert call_args[1]["alert"] == "SCORE_DRIFT"


def test_score_drift_command_errors(tmp_path: Path, trained_model: FraudModel) -> None:
    # Model without score profile
    meta = dict(trained_model.metadata)
    meta.pop("score_profile", None)
    model_no_prof = FraudModel(
        estimator=trained_model.estimator,
        threshold=trained_model.threshold,
        feature_names=trained_model.feature_names,
        metadata=meta,
        artifact_version=trained_model.artifact_version,
    )
    art_path = tmp_path / "no_prof"
    save_model(model_no_prof, art_path)

    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"fraud_probability": [0.1, 0.2]}).to_csv(csv_path, index=False)

    res_no_prof = runner.invoke(
        app,
        [
            "score-drift",
            str(art_path),
            str(csv_path),
        ],
    )
    assert res_no_prof.exit_code == 2
    assert "does not contain a baseline score_profile" in res_no_prof.output

    # Missing column in CSV
    art_with_prof = tmp_path / "with_prof"
    save_model(trained_model, art_with_prof)
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"unrelated_col": [1, 2]}).to_csv(bad_csv, index=False)

    res_bad_col = runner.invoke(
        app,
        [
            "score-drift",
            str(art_with_prof),
            str(bad_csv),
        ],
    )
    assert res_bad_col.exit_code == 2
    assert "Could not identify probability column" in res_bad_col.output

    # Empty scores
    empty_csv = tmp_path / "empty.csv"
    pd.DataFrame({"fraud_probability": []}).to_csv(empty_csv, index=False)
    res_empty = runner.invoke(
        app,
        [
            "score-drift",
            str(art_with_prof),
            str(empty_csv),
        ],
    )
    assert res_empty.exit_code == 2
    assert "No valid scores found" in res_empty.output


def test_score_edge_command(tmp_path: Path) -> None:
    dataset = validate_frame(generate_synthetic_data(rows=300, random_state=42))
    model = train_model(
        dataset,
        config=TrainingConfig(calibration_method=CalibrationMethod.NONE),
    )
    art_dir = tmp_path / "uncal_model"
    save_model(model, art_dir)

    edge_file = tmp_path / "edge_artifact.json"
    res_export = runner.invoke(
        app,
        [
            "export-edge",
            str(art_dir),
            "--output",
            str(edge_file),
        ],
    )
    assert res_export.exit_code == 0, res_export.output

    # Create input CSV
    in_csv = tmp_path / "input.csv"
    test_data = generate_synthetic_data(rows=200, random_state=99).head(20)
    test_data = test_data.drop(columns=["Class"], errors="ignore")
    test_data.to_csv(in_csv, index=False)

    out_jsonl = tmp_path / "edge_out.jsonl"
    res_score = runner.invoke(
        app,
        [
            "score-edge",
            str(edge_file),
            str(in_csv),
            "--output",
            str(out_jsonl),
            "--explain",
            "--top-k",
            "3",
            "--review-threshold",
            "0.2",
            "--deny-threshold",
            "0.8",
        ],
    )
    assert res_score.exit_code == 0, res_score.output
    summary = json.loads(res_score.output)
    assert summary["rows_processed"] == 20
    assert "decision_counts" in summary

    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 20
    first_record = json.loads(lines[0])
    assert "probability" in first_record
    assert "decision" in first_record
    assert first_record["decision"] in ("ALLOW", "CHALLENGE", "DENY")
    assert "explanation" in first_record
    assert "top_contributions" in first_record

    # Output already exists without overwrite
    res_dup = runner.invoke(
        app,
        [
            "score-edge",
            str(edge_file),
            str(in_csv),
            "--output",
            str(out_jsonl),
        ],
    )
    assert res_dup.exit_code == 2
    assert "Output already exists" in res_dup.output

    # With overwrite
    res_over = runner.invoke(
        app,
        [
            "score-edge",
            str(edge_file),
            str(in_csv),
            "--output",
            str(out_jsonl),
            "--overwrite",
        ],
    )
    assert res_over.exit_code == 0, res_over.output


def test_train_cli_tune_tiered_cost_minimization(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=42).to_csv(
        data_path, index=False
    )
    out_dir = tmp_path / "model_tuned_cost"

    res = runner.invoke(
        app,
        [
            "train",
            str(data_path),
            "--output",
            str(out_dir),
            "--tune-tiered",
            "--tiered-objective",
            "cost_minimization",
            "--tiered-manual-review-cost",
            "3.0",
        ],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert "tiered_thresholds" in payload
    assert "tiered_tuning" in payload
    assert payload["tiered_tuning"]["objective"] == "cost_minimization"
    assert (
        payload["tiered_thresholds"]["review_threshold"]
        <= payload["tiered_thresholds"]["deny_threshold"]
    )

    loaded = load_model(out_dir)
    assert loaded.tiered_thresholds is not None
    assert loaded.tiered_tuning is not None


def test_train_cli_tune_tiered_capacity_constrained(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=42).to_csv(
        data_path, index=False
    )
    out_dir = tmp_path / "model_tuned_cap"

    res = runner.invoke(
        app,
        [
            "train",
            str(data_path),
            "--output",
            str(out_dir),
            "--tune-tiered",
            "--tiered-objective",
            "capacity_constrained",
            "--tiered-max-review-rate",
            "0.15",
            "--tiered-min-deny-precision",
            "0.8",
        ],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert "tiered_thresholds" in payload
    assert payload["tiered_tuning"]["objective"] == "capacity_constrained"

    # Missing max-review-rate for capacity_constrained should fail
    res_err = runner.invoke(
        app,
        [
            "train",
            str(data_path),
            "--output",
            str(tmp_path / "fail"),
            "--tune-tiered",
            "--tiered-objective",
            "capacity_constrained",
        ],
    )
    assert res_err.exit_code == 2
    assert "tiered_max_review_rate is required" in res_err.output


def test_backtest_policy_cli_success_and_json(tmp_path: Path) -> None:
    log_file = tmp_path / "audit.jsonl"
    events = [
        json.dumps(
            {
                "event_type": "scoring",
                "payload": {
                    "review_threshold": 0.25,
                    "deny_threshold": 0.75,
                    "predictions": [
                        {"fraud_probability": 0.1, "is_fraud": False},
                        {"fraud_probability": 0.5, "is_fraud": False},
                        {"fraud_probability": 0.9, "is_fraud": True},
                    ],
                },
            }
        )
    ]
    log_file.write_text("\n".join(events) + "\n", encoding="utf-8")

    # 1. Text output
    res_text = runner.invoke(
        app,
        [
            "backtest-policy",
            str(log_file),
            "--candidate-review",
            "0.2",
            "--candidate-deny",
            "0.8",
        ],
    )
    assert res_text.exit_code == 0, res_text.output
    assert "Policy Backtest: 3 records evaluated" in res_text.output
    assert "Transition Matrix (Baseline -> Candidate):" in res_text.output

    # 2. JSON output and file saving
    out_file = tmp_path / "backtest_report.json"
    res_json = runner.invoke(
        app,
        [
            "backtest-policy",
            str(log_file),
            "--candidate-review",
            "0.2",
            "--candidate-deny",
            "0.8",
            "--json",
            "--output",
            str(out_file),
        ],
    )
    assert res_json.exit_code == 0, res_json.output
    data = json.loads(res_json.output)
    assert data["rows"] == 3
    assert data["candidate_thresholds"]["review_threshold"] == 0.2
    assert data["candidate_thresholds"]["deny_threshold"] == 0.8
    assert out_file.exists()
    assert json.loads(out_file.read_text(encoding="utf-8")) == data

    # 3. Explicit baseline options
    res_base = runner.invoke(
        app,
        [
            "backtest-policy",
            str(log_file),
            "--candidate-review",
            "0.3",
            "--candidate-deny",
            "0.7",
            "--baseline-review",
            "0.1",
            "--baseline-deny",
            "0.9",
        ],
    )
    assert res_base.exit_code == 0, res_base.output
    assert "Baseline: review=0.1000, deny=0.9000" in res_base.output


def test_backtest_policy_cli_errors(tmp_path: Path) -> None:
    log_file = tmp_path / "valid.jsonl"
    log_file.write_text(
        json.dumps(
            {
                "event_type": "scoring",
                "payload": {"predictions": [{"fraud_probability": 0.5}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Inverted candidate thresholds
    res_inv = runner.invoke(
        app,
        [
            "backtest-policy",
            str(log_file),
            "--candidate-review",
            "0.8",
            "--candidate-deny",
            "0.2",
        ],
    )
    assert res_inv.exit_code == 2
    assert "cannot exceed" in res_inv.output

    # Partial baseline thresholds (only one provided)
    res_part = runner.invoke(
        app,
        [
            "backtest-policy",
            str(log_file),
            "--candidate-review",
            "0.2",
            "--candidate-deny",
            "0.8",
            "--baseline-review",
            "0.3",
        ],
    )
    assert res_part.exit_code == 2
    assert "must be provided together" in res_part.output

    # Non-existent audit log
    res_missing = runner.invoke(
        app,
        [
            "backtest-policy",
            str(tmp_path / "nonexistent.jsonl"),
            "--candidate-review",
            "0.2",
            "--candidate-deny",
            "0.8",
        ],
    )
    assert res_missing.exit_code == 2

    # Output already exists without overwrite
    out_file = tmp_path / "out.json"
    out_file.write_text("{}", encoding="utf-8")
    res_exist = runner.invoke(
        app,
        [
            "backtest-policy",
            str(log_file),
            "--candidate-review",
            "0.2",
            "--candidate-deny",
            "0.8",
            "--output",
            str(out_file),
        ],
    )
    assert res_exist.exit_code == 2
    assert "Output already exists" in res_exist.output


def test_slice_metrics_cli_success_and_formatting(tmp_path: Path, trained_artifact: Path) -> None:
    data = generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=42)
    data["channel"] = ["web" if i % 2 == 0 else "mobile" for i in range(len(data))]
    csv_path = tmp_path / "slice_data.csv"
    data.to_csv(csv_path, index=False)

    # 1. Text table output
    res_text = runner.invoke(
        app,
        [
            "slice-metrics",
            str(trained_artifact),
            str(csv_path),
            "--slice-column",
            "channel",
        ],
    )
    assert res_text.exit_code == 0, res_text.output
    assert "Slice Disparity Analysis: 200 records across 2 slices" in res_text.output
    assert "web" in res_text.output
    assert "mobile" in res_text.output

    # 2. JSON output and file saving
    out_file = tmp_path / "slice_report.json"
    res_json = runner.invoke(
        app,
        [
            "slice-metrics",
            str(trained_artifact),
            str(csv_path),
            "--slice-column",
            "channel",
            "--json",
            "--output",
            str(out_file),
        ],
    )
    assert res_json.exit_code == 0, res_json.output
    report_dict = json.loads(res_json.output)
    assert report_dict["total_records"] == 200
    assert len(report_dict["slices"]) == 2
    assert out_file.exists()
    assert json.loads(out_file.read_text(encoding="utf-8")) == report_dict

    # 3. With tiered model / --use-tiered
    res_tiered = runner.invoke(
        app,
        [
            "slice-metrics",
            str(trained_artifact),
            str(csv_path),
            "--slice-column",
            "channel",
            "--use-tiered",
        ],
    )
    assert res_tiered.exit_code == 0, res_tiered.output


def test_slice_metrics_cli_disparity_flagging_and_errors(
    tmp_path: Path, trained_artifact: Path
) -> None:
    data = generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=42)
    data["channel"] = ["web" if i < 180 else "mobile" for i in range(len(data))]
    csv_path = tmp_path / "slice_data.csv"
    data.to_csv(csv_path, index=False)

    # Force failure with strict disparity bounds
    res_fail = runner.invoke(
        app,
        [
            "slice-metrics",
            str(trained_artifact),
            str(csv_path),
            "--slice-column",
            "channel",
            "--min-recall-disparity",
            "1.5",  # Impossible ratio -> forces underperforming slice
            "--fail-on-disparity",
        ],
    )
    assert res_fail.exit_code == 1

    # Missing slice column
    res_no_col = runner.invoke(
        app,
        [
            "slice-metrics",
            str(trained_artifact),
            str(csv_path),
            "--slice-column",
            "nonexistent_column",
        ],
    )
    assert res_no_col.exit_code == 2
    assert "Slice column 'nonexistent_column' not found" in res_no_col.output

    # Missing target column
    res_no_target = runner.invoke(
        app,
        [
            "slice-metrics",
            str(trained_artifact),
            str(csv_path),
            "--slice-column",
            "channel",
            "--target",
            "nonexistent_target",
        ],
    )
    assert res_no_target.exit_code == 2
    assert "Target column 'nonexistent_target' not found" in res_no_target.output

    # Missing features in CSV
    bad_data = data[["channel", "Class"]].copy()
    bad_csv = tmp_path / "missing_features.csv"
    bad_data.to_csv(bad_csv, index=False)
    res_missing_feat = runner.invoke(
        app,
        [
            "slice-metrics",
            str(trained_artifact),
            str(bad_csv),
            "--slice-column",
            "channel",
        ],
    )
    assert res_missing_feat.exit_code == 2
    assert "Missing required model features" in res_missing_feat.output

    # Output already exists without overwrite
    out_file = tmp_path / "existing.json"
    out_file.write_text("{}", encoding="utf-8")
    res_exist = runner.invoke(
        app,
        [
            "slice-metrics",
            str(trained_artifact),
            str(csv_path),
            "--slice-column",
            "channel",
            "--output",
            str(out_file),
        ],
    )
    assert res_exist.exit_code == 2
    assert "Output already exists" in res_exist.output


def test_slice_metrics_alert_webhook(tmp_path: Path, trained_artifact: Path) -> None:
    data = generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=42)
    data["channel"] = ["web" if i < 180 else "mobile" for i in range(len(data))]
    csv_path = tmp_path / "slice_data.csv"
    data.to_csv(csv_path, index=False)

    # 1. Underperforming slice detected with webhook configured
    with patch("fraud_detection.cli._post_webhook") as mock_post:
        res = runner.invoke(
            app,
            [
                "slice-metrics",
                str(trained_artifact),
                str(csv_path),
                "--slice-column",
                "channel",
                "--min-recall-disparity",
                "1.5",
                "--alert-webhook-url",
                "https://alerts.internal/disparity",
            ],
        )
        assert res.exit_code == 0
        mock_post.assert_called_once()
        call_args = mock_post.call_args[0]
        assert call_args[0] == "https://alerts.internal/disparity"
        payload = call_args[1]
        assert payload["alert"] == "SLICE_DISPARITY"
        assert payload["slice_column"] == "channel"
        assert len(payload["underperforming_slices"]) > 0

    # 2. No underperforming slices -> webhook not called
    with patch("fraud_detection.cli._post_webhook") as mock_post_clean:
        res_clean = runner.invoke(
            app,
            [
                "slice-metrics",
                str(trained_artifact),
                str(csv_path),
                "--slice-column",
                "channel",
                "--min-recall-disparity",
                "0.01",
                "--max-fpr-disparity",
                "100.0",
                "--alert-webhook-url",
                "https://alerts.internal/disparity",
            ],
        )
        assert res_clean.exit_code == 0
        mock_post_clean.assert_not_called()


def test_serve_passes_rules_options_to_create_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = tmp_path / "artifact"
    artifact_path.mkdir()
    rules_path = tmp_path / "rules.json"
    rules_path.write_text("{}", encoding="utf-8")
    created: dict[str, Any] = {}

    def fake_load_model(_path: Path | str) -> object:
        return object()

    def fake_create_app(**kwargs: Any) -> object:
        created.update(kwargs)
        return object()

    def fake_run(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("fraud_detection.cli.load_model", fake_load_model)
    monkeypatch.setattr("fraud_detection.api.create_app", fake_create_app)
    monkeypatch.setattr("fraud_detection.cli.uvicorn.run", fake_run)

    served = runner.invoke(
        app,
        [
            "serve",
            str(artifact_path),
            "--rules",
            str(rules_path),
            "--rule-precedence",
            "model_overrides_rules",
        ],
    )

    assert served.exit_code == 0, served.output
    assert created["rules_path"] == rules_path
    assert created["rule_precedence"] == RulePrecedence.MODEL_OVERRIDES_RULES


def test_rules_cli_success_and_json_output(tmp_path: Path) -> None:
    data = generate_synthetic_data(rows=200, fraud_rate=0.1, random_state=42)
    data["Amount"] = [600.0 if i < 20 else 20.0 for i in range(200)]
    csv_path = tmp_path / "data.csv"
    data.to_csv(csv_path, index=False)

    rule1 = DecisionRule(
        rule_id="R_HIGH",
        name="High Amount Deny",
        priority=10,
        action=RuleAction.DENY,
        conditions=(
            RuleCondition(field="Amount", operator=RuleOperator.GREATER_THAN, value=500.0),
        ),
    )
    rule2 = DecisionRule(
        rule_id="R_LOW",
        name="Low Amount Allow",
        priority=20,
        action=RuleAction.ALLOW,
        conditions=(
            RuleCondition(field="Amount", operator=RuleOperator.LESS_THAN_OR_EQUAL, value=50.0),
        ),
    )
    rules_path = tmp_path / "rules.json"
    RuleSet(rules=(rule1, rule2)).save_file(rules_path)

    # 1. Plain text table output
    res_table = runner.invoke(app, ["test-rules", str(rules_path), str(csv_path)])
    assert res_table.exit_code == 0
    assert "Rule Evaluation: 2 rules evaluated across 200 records" in res_table.output
    assert "R_HIGH" in res_table.output
    assert "R_LOW" in res_table.output

    # 2. JSON output with output file destination
    out_file = tmp_path / "out_report.json"
    res_json = runner.invoke(
        app,
        ["test-rules", str(rules_path), str(csv_path), "--json", "--output", str(out_file)],
    )
    assert res_json.exit_code == 0
    parsed = json.loads(res_json.output)
    assert parsed["total_records"] == 200
    assert parsed["total_rules"] == 2
    assert parsed["matched_records"] == 200
    assert parsed["match_rate"] == 1.0

    file_parsed = json.loads(out_file.read_text(encoding="utf-8"))
    assert file_parsed["matched_records"] == 200


def test_rules_cli_errors(tmp_path: Path) -> None:
    data_path = tmp_path / "dummy.csv"
    pd.DataFrame({"Amount": [1.0, 2.0]}).to_csv(data_path, index=False)

    bad_rules = tmp_path / "bad_rules.json"
    bad_rules.write_text("{invalid json", encoding="utf-8")

    # Syntax error in rules file
    res_bad_rules = runner.invoke(app, ["test-rules", str(bad_rules), str(data_path)])
    assert res_bad_rules.exit_code == 2
    assert "Failed to load rules" in res_bad_rules.output

    # Nonexistent rules file
    res_missing_rules = runner.invoke(
        app, ["test-rules", str(tmp_path / "missing.json"), str(data_path)]
    )
    assert res_missing_rules.exit_code == 2

    # Nonexistent data file
    good_rules = tmp_path / "good.json"
    RuleSet(rules=()).save_file(good_rules)
    res_missing_data = runner.invoke(
        app, ["test-rules", str(good_rules), str(tmp_path / "missing.csv")]
    )
    assert res_missing_data.exit_code == 2

    # Output already exists without overwrite
    existing_out = tmp_path / "existing.json"
    existing_out.write_text("{}", encoding="utf-8")
    res_exist = runner.invoke(
        app,
        ["test-rules", str(good_rules), str(data_path), "--output", str(existing_out)],
    )
    assert res_exist.exit_code == 2
    assert "Output already exists" in res_exist.output


def test_compute_velocity_cli_success(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    df = pd.DataFrame(
        {
            "card_id": ["c1", "c2", "c1", "c1"],
            "Time": [10.0, 20.0, 50.0, 100.0],
            "Amount": [100.0, 50.0, 200.0, 150.0],
            "V1": [0.1, 0.2, 0.3, 0.4],
        }
    )
    df.to_csv(data_path, index=False)

    out_csv = tmp_path / "enriched.csv"
    result = runner.invoke(
        app,
        [
            "compute-velocity",
            "--data",
            str(data_path),
            "--output",
            str(out_csv),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["rows_processed"] == 4
    assert summary["distinct_entities"] == 2
    assert summary["features_generated_count"] > 0
    assert out_csv.exists()

    enriched_df = pd.read_csv(out_csv)
    assert len(enriched_df) == 4
    assert "velocity_count_5m" in enriched_df.columns
    assert "velocity_sum_5m" in enriched_df.columns

    # Test table output
    out_table_csv = tmp_path / "enriched_table.csv"
    table_result = runner.invoke(
        app,
        [
            "compute-velocity",
            "--data",
            str(data_path),
            "--output",
            str(out_table_csv),
        ],
    )
    assert table_result.exit_code == 0
    assert "Velocity Feature Enrichment Complete" in table_result.output
    assert "Rows processed:       4" in table_result.output


def test_compute_velocity_cli_custom_config_and_windows(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    df = pd.DataFrame(
        {
            "card_id": ["c1", "c1"],
            "Time": [10.0, 50.0],
            "Amount": [100.0, 200.0],
        }
    )
    df.to_csv(data_path, index=False)

    config_path = tmp_path / "custom_velocity.json"
    cfg = VelocityConfig(
        windows=(VelocityWindow(duration_seconds=120.0, name="2m"),),
    )
    cfg.save_file(config_path)

    out_csv = tmp_path / "custom_enriched.csv"
    res = runner.invoke(
        app,
        [
            "compute-velocity",
            "--data",
            str(data_path),
            "--output",
            str(out_csv),
            "--config",
            str(config_path),
            "--windows",
            "60,300",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    summary = json.loads(res.stdout)
    assert summary["windows"] == ["1m", "5m"]
    enriched = pd.read_csv(out_csv)
    assert "velocity_count_1m" in enriched.columns
    assert "velocity_count_5m" in enriched.columns


def test_compute_velocity_cli_errors(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    df = pd.DataFrame(
        {
            "Time": [10.0, 20.0],
            "Amount": [50.0, 100.0],
        }
    )
    df.to_csv(data_path, index=False)

    out_csv = tmp_path / "enriched.csv"
    out_csv.write_text("existing", encoding="utf-8")

    # Output already exists without overwrite
    res_exist = runner.invoke(
        app,
        [
            "compute-velocity",
            "--data",
            str(data_path),
            "--output",
            str(out_csv),
        ],
    )
    assert res_exist.exit_code == 2
    assert "Output already exists" in res_exist.output

    # Malformed config file
    bad_cfg = tmp_path / "malformed.json"
    bad_cfg.write_text("{invalid json", encoding="utf-8")
    res_bad_cfg = runner.invoke(
        app,
        [
            "compute-velocity",
            "--data",
            str(data_path),
            "--output",
            str(tmp_path / "out2.csv"),
            "--config",
            str(bad_cfg),
        ],
    )
    assert res_bad_cfg.exit_code == 2
    assert "Failed to load velocity configuration" in res_bad_cfg.output

    # Invalid windows string
    res_bad_win = runner.invoke(
        app,
        [
            "compute-velocity",
            "--data",
            str(data_path),
            "--output",
            str(tmp_path / "out3.csv"),
            "--windows",
            "not_numbers",
        ],
    )
    assert res_bad_win.exit_code == 2
    assert "Invalid window specification" in res_bad_win.output

    # Empty data
    empty_csv = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(empty_csv, index=False)
    res_empty = runner.invoke(
        app,
        [
            "compute-velocity",
            "--data",
            str(empty_csv),
            "--output",
            str(tmp_path / "out4.csv"),
        ],
    )
    assert res_empty.exit_code == 2

    # Missing required timestamp column
    bad_cols_csv = tmp_path / "bad_cols.csv"
    pd.DataFrame({"V1": [1.0, 2.0]}).to_csv(bad_cols_csv, index=False)
    res_bad_cols = runner.invoke(
        app,
        [
            "compute-velocity",
            "--data",
            str(bad_cols_csv),
            "--output",
            str(tmp_path / "out5.csv"),
        ],
    )
    assert res_bad_cols.exit_code == 2
    assert "Velocity computation failed" in res_bad_cols.output


def test_cli_calibrate_eval_basic_and_json(
    trained_artifact: Path,
    tmp_path: Path,
) -> None:
    data_csv = tmp_path / "eval_data.csv"
    generate_synthetic_data(rows=200, random_state=42).to_csv(data_csv, index=False)

    # Human-readable table
    res_table = runner.invoke(
        app,
        ["calibrate-eval", str(trained_artifact), str(data_csv)],
    )
    assert res_table.exit_code == 0
    assert "Calibration Surveillance Diagnostics" in res_table.output
    assert "Expected Calibration Error:" in res_table.output
    assert "Brier Score:" in res_table.output

    # JSON output and output file saving
    report_file = tmp_path / "cal_report.json"
    res_json = runner.invoke(
        app,
        [
            "calibrate-eval",
            str(trained_artifact),
            str(data_csv),
            "--bins",
            "5",
            "--strategy",
            "quantile",
            "--output",
            str(report_file),
            "--json",
        ],
    )
    assert res_json.exit_code == 0
    payload = json.loads(res_json.output)
    assert payload["strategy"] == "quantile"
    assert payload["bins"] >= 2
    assert len(payload["detail"]) == payload["bins"]
    assert "expected_calibration_error" in payload
    assert report_file.exists()

    # Re-running with existing output without overwrite fails
    res_no_ovw = runner.invoke(
        app,
        ["calibrate-eval", str(trained_artifact), str(data_csv), "--output", str(report_file)],
    )
    assert res_no_ovw.exit_code == 2

    # Re-running with --overwrite succeeds
    res_ovw = runner.invoke(
        app,
        [
            "calibrate-eval",
            str(trained_artifact),
            str(data_csv),
            "--output",
            str(report_file),
            "--overwrite",
        ],
    )
    assert res_ovw.exit_code == 0


def test_cli_calibrate_eval_baseline_and_drift(
    trained_artifact: Path,
    tmp_path: Path,
) -> None:
    data_csv = tmp_path / "eval_data.csv"
    generate_synthetic_data(rows=200, random_state=42).to_csv(data_csv, index=False)

    baseline_file = tmp_path / "baseline.json"
    res_base = runner.invoke(
        app,
        ["calibrate-eval", str(trained_artifact), str(data_csv), "--output", str(baseline_file)],
    )
    assert res_base.exit_code == 0

    # Compare against identical baseline -> no drift
    res_comp = runner.invoke(
        app,
        [
            "calibrate-eval",
            str(trained_artifact),
            str(data_csv),
            "--baseline",
            str(baseline_file),
            "--fail-on-drift",
        ],
    )
    assert res_comp.exit_code == 0
    assert "Calibration Drift:          STABLE" in res_comp.output

    # Compare against an altered baseline that trips drift
    base_data = json.loads(baseline_file.read_text(encoding="utf-8"))
    base_data["expected_calibration_error"] = 0.99
    base_data["max_calibration_error"] = 0.99
    drifted_base_file = tmp_path / "drifted_baseline.json"
    drifted_base_file.write_text(json.dumps(base_data), encoding="utf-8")

    res_drift = runner.invoke(
        app,
        [
            "calibrate-eval",
            str(trained_artifact),
            str(data_csv),
            "--baseline",
            str(drifted_base_file),
            "--fail-on-drift",
        ],
    )
    assert res_drift.exit_code == 2
    assert "Calibration drift detected" in res_drift.output


def test_cli_calibrate_eval_error_handling(
    trained_artifact: Path,
    tmp_path: Path,
) -> None:
    data_csv = tmp_path / "eval_data.csv"
    generate_synthetic_data(rows=200, random_state=42).to_csv(data_csv, index=False)

    # Missing model
    res_no_mod = runner.invoke(
        app,
        ["calibrate-eval", str(tmp_path / "missing_model"), str(data_csv)],
    )
    assert res_no_mod.exit_code != 0

    # Corrupt model directory
    corrupt_model = tmp_path / "corrupt_model"
    corrupt_model.mkdir()
    (corrupt_model / "model.joblib").write_bytes(b"garbage")
    res_corrupt_mod = runner.invoke(app, ["calibrate-eval", str(corrupt_model), str(data_csv)])
    assert res_corrupt_mod.exit_code == 2
    assert "Failed to load model" in res_corrupt_mod.output

    # Missing data
    res_no_dat = runner.invoke(
        app,
        ["calibrate-eval", str(trained_artifact), str(tmp_path / "missing.csv")],
    )
    assert res_no_dat.exit_code != 0

    # Corrupt CSV
    bad_csv = tmp_path / "bad_data.csv"
    pd.DataFrame({"V1": [1.0, 2.0], "Class": [5, 6]}).to_csv(bad_csv, index=False)
    res_bad_csv = runner.invoke(app, ["calibrate-eval", str(trained_artifact), str(bad_csv)])
    assert res_bad_csv.exit_code == 2
    assert "Failed to load dataset" in res_bad_csv.output

    # Bad baseline file
    bad_base = tmp_path / "bad_baseline.json"
    bad_base.write_text("invalid json", encoding="utf-8")
    res_bad_base = runner.invoke(
        app,
        [
            "calibrate-eval",
            str(trained_artifact),
            str(data_csv),
            "--baseline",
            str(bad_base),
        ],
    )
    assert res_bad_base.exit_code == 2

    # Baseline with nested "diagnostics" payload
    base_file = tmp_path / "nested_base.json"
    base_diag = CalibrationDiagnostics(
        rows=100,
        bins=2,
        strategy="uniform",
        brier_score=0.1,
        expected_calibration_error=0.05,
        max_calibration_error=0.05,
        root_mean_squared_error=0.05,
        reliability=0.01,
        resolution=0.01,
        uncertainty=0.01,
        monotonicity_violations=0,
        detail=(),
    )
    base_file.write_text(json.dumps({"diagnostics": base_diag.to_dict()}), encoding="utf-8")
    res_nested = runner.invoke(
        app,
        [
            "calibrate-eval",
            str(trained_artifact),
            str(data_csv),
            "--baseline",
            str(base_file),
        ],
    )
    assert res_nested.exit_code == 0


def test_cli_recalibrate_basic_and_json(
    trained_artifact: Path,
    tmp_path: Path,
) -> None:
    cal_data_csv = tmp_path / "cal_data.csv"
    generate_synthetic_data(rows=250, random_state=123).to_csv(cal_data_csv, index=False)

    out_artifact = tmp_path / "recal_model"
    res_recal = runner.invoke(
        app,
        [
            "recalibrate",
            str(trained_artifact),
            str(cal_data_csv),
            "--output",
            str(out_artifact),
            "--method",
            "sigmoid",
            "--json",
        ],
    )
    assert res_recal.exit_code == 0
    summary = json.loads(res_recal.output)
    assert summary["method"] == "sigmoid"
    assert "pre_calibration" in summary
    assert "post_calibration" in summary
    assert (out_artifact / "model.joblib").exists()

    # Load and verify recalibrated artifact
    loaded_recal = load_model(out_artifact)
    assert loaded_recal.recalibrator is not None
    assert loaded_recal.recalibrator.method.value == "sigmoid"

    # Overwrite protection
    res_no_ovw = runner.invoke(
        app,
        [
            "recalibrate",
            str(trained_artifact),
            str(cal_data_csv),
            "--output",
            str(out_artifact),
        ],
    )
    assert res_no_ovw.exit_code == 2

    # Overwrite succeeds
    res_ovw = runner.invoke(
        app,
        [
            "recalibrate",
            str(trained_artifact),
            str(cal_data_csv),
            "--output",
            str(out_artifact),
            "--method",
            "isotonic",
            "--overwrite",
        ],
    )
    assert res_ovw.exit_code == 0
    assert "Post-Hoc Model Recalibration Complete" in res_ovw.output
    loaded_iso = load_model(out_artifact)
    assert loaded_iso.recalibrator is not None
    assert loaded_iso.recalibrator.method.value == "isotonic"

    # Temperature and no retune threshold
    out_temp = tmp_path / "temp_model"
    res_temp = runner.invoke(
        app,
        [
            "recalibrate",
            str(trained_artifact),
            str(cal_data_csv),
            "--output",
            str(out_temp),
            "--method",
            "temperature",
            "--no-retune-threshold",
            "--json",
        ],
    )
    assert res_temp.exit_code == 0
    loaded_temp = load_model(out_temp)
    assert loaded_temp.recalibrator is not None
    assert loaded_temp.recalibrator.method.value == "temperature"


def test_cli_recalibrate_error_handling(
    trained_artifact: Path,
    tmp_path: Path,
) -> None:
    cal_data_csv = tmp_path / "cal_data.csv"
    generate_synthetic_data(rows=250, random_state=123).to_csv(cal_data_csv, index=False)

    # Missing model
    res_bad_mod = runner.invoke(
        app,
        [
            "recalibrate",
            str(tmp_path / "nonexistent_model"),
            str(cal_data_csv),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert res_bad_mod.exit_code != 0

    # Corrupt model directory
    corrupt_model = tmp_path / "corrupt_model"
    corrupt_model.mkdir()
    (corrupt_model / "model.joblib").write_bytes(b"garbage")
    res_corrupt_mod = runner.invoke(
        app,
        [
            "recalibrate",
            str(corrupt_model),
            str(cal_data_csv),
            "--output",
            str(tmp_path / "out_corrupt"),
        ],
    )
    assert res_corrupt_mod.exit_code == 2
    assert "Failed to load model" in res_corrupt_mod.output

    # Missing data
    res_bad_dat = runner.invoke(
        app,
        [
            "recalibrate",
            str(trained_artifact),
            str(tmp_path / "nonexistent.csv"),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert res_bad_dat.exit_code != 0

    # Corrupt dataset
    bad_csv = tmp_path / "bad_data.csv"
    pd.DataFrame({"V1": [1.0, 2.0], "Class": [5, 6]}).to_csv(bad_csv, index=False)
    res_corrupt_csv = runner.invoke(
        app,
        [
            "recalibrate",
            str(trained_artifact),
            str(bad_csv),
            "--output",
            str(tmp_path / "out_corrupt_csv"),
        ],
    )
    assert res_corrupt_csv.exit_code == 2
    assert "Failed to load dataset" in res_corrupt_csv.output

    # Recalibration failure
    with patch(
        "fraud_detection.cli.recalibrate_model",
        side_effect=ValueError("Simulated calibration failure"),
    ):
        res_fail = runner.invoke(
            app,
            [
                "recalibrate",
                str(trained_artifact),
                str(cal_data_csv),
                "--output",
                str(tmp_path / "out_fail"),
            ],
        )
        assert res_fail.exit_code == 2
        assert "Model recalibration failed" in res_fail.output

    # Save failure
    with patch(
        "fraud_detection.cli.save_model", side_effect=ModelArtifactError("Simulated save failure")
    ):
        res_save_fail = runner.invoke(
            app,
            [
                "recalibrate",
                str(trained_artifact),
                str(cal_data_csv),
                "--output",
                str(tmp_path / "out_save_fail"),
            ],
        )
        assert res_save_fail.exit_code == 2
        assert "Failed to save recalibrated model" in res_save_fail.output


def test_cli_route_eval_help() -> None:
    res = runner.invoke(app, ["route-eval", "--help"])
    assert res.exit_code == 0
    assert "--champion" in res.output
    assert "--challenger" in res.output
    assert "--data" in res.output
    assert "Evaluate decision discrepancies" in res.output
    assert "--json" in res.output
    assert "--output" in res.output


def test_cli_route_eval_csv_and_json(
    tmp_path: Path,
    trained_artifact: Path,
) -> None:
    # Prepare challenger model
    base_model = load_model(trained_artifact)
    challenger = deepcopy(base_model)
    challenger.metadata["dataset_fingerprint"] = "c" * 64
    chall_path = tmp_path / "challenger_artifact"
    save_model(challenger, chall_path)

    # Prepare evaluation data CSV
    eval_df = generate_synthetic_data(rows=200, random_state=42)
    data_csv = tmp_path / "eval_data.csv"
    eval_df.to_csv(data_csv, index=False)

    # 1. Text output
    res_text = runner.invoke(
        app,
        [
            "route-eval",
            "--champion",
            str(trained_artifact),
            "--challenger",
            str(chall_path),
            "--data",
            str(data_csv),
        ],
    )
    assert res_text.exit_code == 0
    assert "Canary Multi-Model Divergence Evaluation Report" in res_text.output
    assert "Champion Version:" in res_text.output
    assert "Challenger Version:" in res_text.output
    assert "Decision Flips:" in res_text.output
    assert "Rollout Recommendation:" in res_text.output

    # 2. JSON output and file
    out_json = tmp_path / "report.json"
    res_json = runner.invoke(
        app,
        [
            "route-eval",
            "--champion",
            str(trained_artifact),
            "--challenger",
            str(chall_path),
            "--data",
            str(data_csv),
            "--json",
            "--output",
            str(out_json),
        ],
    )
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert data["total_samples"] == 200
    assert "flip_rate" in data
    assert "mean_probability_divergence" in data
    assert out_json.exists()


def test_cli_route_eval_jsonl_audit_input(
    tmp_path: Path,
    trained_artifact: Path,
) -> None:
    base_model = load_model(trained_artifact)
    challenger = deepcopy(base_model)
    challenger.metadata["dataset_fingerprint"] = "c" * 64
    chall_path = tmp_path / "challenger_audit_eval"
    save_model(challenger, chall_path)

    # Generate synthetic audit JSONL
    eval_df = generate_synthetic_data(rows=200, random_state=99)
    audit_file = tmp_path / "audit_events.jsonl"
    records = eval_df.drop(columns="Class", errors="ignore").to_dict(orient="records")
    with audit_file.open("w", encoding="utf-8") as f:
        for rec in records:
            event = {
                "event_type": "scoring",
                "payload": {"features": [rec]},
            }
            f.write(json.dumps(event) + "\n")

    res = runner.invoke(
        app,
        [
            "route-eval",
            "--champion",
            str(trained_artifact),
            "--challenger",
            str(chall_path),
            "--data",
            str(audit_file),
            "--json",
        ],
    )
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["total_samples"] == 200


def test_cli_route_eval_fail_on_discrepancy(
    tmp_path: Path,
    trained_artifact: Path,
) -> None:
    base_model = load_model(trained_artifact)
    divergent = deepcopy(base_model)
    divergent.metadata["dataset_fingerprint"] = "d" * 64
    divergent.threshold = 0.01  # will trigger flips against normal threshold
    div_path = tmp_path / "divergent_artifact"
    save_model(divergent, div_path)

    eval_df = generate_synthetic_data(rows=200, random_state=12)
    data_csv = tmp_path / "eval_div.csv"
    eval_df.to_csv(data_csv, index=False)

    res = runner.invoke(
        app,
        [
            "route-eval",
            "--champion",
            str(trained_artifact),
            "--challenger",
            str(div_path),
            "--data",
            str(data_csv),
            "--max-discrepancy-rate",
            "0.01",
            "--min-evaluations",
            "5",
            "--fail-on-discrepancy",
        ],
    )
    assert res.exit_code == 1
    assert "Safeguard Tripped:           True" in res.output


def test_cli_route_eval_errors(
    tmp_path: Path,
    trained_artifact: Path,
) -> None:
    # 1. Invalid champion
    res1 = runner.invoke(
        app,
        [
            "route-eval",
            "--champion",
            str(tmp_path / "nonexistent"),
            "--challenger",
            str(trained_artifact),
            "--data",
            str(tmp_path / "data.csv"),
        ],
    )
    assert res1.exit_code != 0

    # 2. Corrupt data
    bad_data = tmp_path / "bad.csv"
    bad_data.write_text("not,a,valid,csv\n1\n", encoding="utf-8")
    res2 = runner.invoke(
        app,
        [
            "route-eval",
            "--champion",
            str(trained_artifact),
            "--challenger",
            str(trained_artifact),
            "--data",
            str(bad_data),
        ],
    )
    assert res2.exit_code == 2

    # 3. Empty JSONL
    empty_jsonl = tmp_path / "empty.jsonl"
    empty_jsonl.write_text("\n\n", encoding="utf-8")
    res3 = runner.invoke(
        app,
        [
            "route-eval",
            "--champion",
            str(trained_artifact),
            "--challenger",
            str(trained_artifact),
            "--data",
            str(empty_jsonl),
        ],
    )
    assert res3.exit_code == 2
    assert "No valid transaction features" in res3.output

    # 4. Output exists without overwrite
    existing_out = tmp_path / "exists.json"
    existing_out.write_text("{}", encoding="utf-8")
    valid_data = tmp_path / "valid.csv"
    generate_synthetic_data(rows=200).to_csv(valid_data, index=False)
    res4 = runner.invoke(
        app,
        [
            "route-eval",
            "--champion",
            str(trained_artifact),
            "--challenger",
            str(trained_artifact),
            "--data",
            str(valid_data),
            "--output",
            str(existing_out),
        ],
    )
    assert res4.exit_code == 2
    assert "Output already exists" in res4.output


def test_cli_serve_routing_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = tmp_path / "artifact"
    artifact_path.mkdir()
    chall_path = tmp_path / "challenger"
    chall_path.mkdir()

    created: dict[str, Any] = {}

    def fake_load_model(_path: Path | str) -> object:
        return object()

    def fake_create_app(**kwargs: Any) -> object:
        created.update(kwargs)
        return object()

    def fake_run(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("fraud_detection.cli.load_model", fake_load_model)
    monkeypatch.setattr("fraud_detection.api.create_app", fake_create_app)
    monkeypatch.setattr("fraud_detection.cli.uvicorn.run", fake_run)

    # Valid routing options
    served = runner.invoke(
        app,
        [
            "serve",
            str(artifact_path),
            "--challenger-model",
            str(chall_path),
            "--routing-strategy",
            "canary",
            "--challenger-weight",
            "0.15",
            "--routing-entity-key",
            "card_id",
            "--canary-max-discrepancy",
            "0.10",
            "--canary-max-divergence",
            "0.20",
        ],
    )
    assert served.exit_code == 0, served.output
    assert created["challenger_model_path"] == chall_path
    assert created["routing_strategy"].value == "canary"
    assert created["challenger_weight"] == 0.15
    assert created["routing_entity_key"] == "card_id"
    assert created["canary_max_discrepancy"] == 0.10
    assert created["canary_max_divergence"] == 0.20

    # Invalid routing strategy
    invalid = runner.invoke(
        app,
        [
            "serve",
            str(artifact_path),
            "--routing-strategy",
            "nonexistent_strategy",
        ],
    )
    assert invalid.exit_code == 2
    assert "Invalid --routing-strategy" in invalid.output


def test_cli_feature_join_command(tmp_path: Path) -> None:
    obs_df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u1"],
            "Time": [10.0, 20.0, 30.0],
            "Amount": [100.0, 50.0, 75.0],
        }
    )
    obs_file = tmp_path / "obs.csv"
    obs_df.to_csv(obs_file, index=False)

    view = FeatureView(
        name="user_view",
        entity_key="user_id",
        features=(
            FeatureDefinition(
                name="user_risk",
                feature_type=FeatureType.FLOAT,
                default_value=0.1,
            ),
        ),
    )
    store = FileFeatureStore(views=[view])
    store.put_snapshot(
        "user_view",
        FeatureSnapshot(entity_id="u1", timestamp=5.0, values={"user_risk": 0.9}),
    )
    store_file = tmp_path / "store.json"
    store.save_to_file(store_file)

    out_file = tmp_path / "enriched.csv"
    result = runner.invoke(
        app,
        [
            "feature-join",
            str(obs_file),
            "--store",
            str(store_file),
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Point-in-Time Feature Join Completed" in result.output
    assert out_file.exists()
    enriched = pd.read_csv(out_file)
    assert "user_risk" in enriched.columns
    assert enriched.loc[0, "user_risk"] == 0.9
    assert enriched.loc[1, "user_risk"] == 0.1

    # Overwrite protection
    blocked = runner.invoke(
        app,
        [
            "feature-join",
            str(obs_file),
            "--store",
            str(store_file),
            "--output",
            str(out_file),
        ],
    )
    assert blocked.exit_code == 2

    # Invalid file
    bad = runner.invoke(
        app,
        [
            "feature-join",
            str(tmp_path / "missing.csv"),
            "--store",
            str(store_file),
        ],
    )
    assert bad.exit_code == 2


def test_cli_feature_check_command(tmp_path: Path) -> None:
    import numpy as np

    np.random.seed(42)
    ref_df = pd.DataFrame(
        {
            "feat_a": np.random.normal(0, 1, 200),
            "feat_b": np.random.normal(5, 2, 200),
        }
    )
    curr_df = pd.DataFrame(
        {
            "feat_a": np.random.normal(0, 1, 200),
            "feat_b": np.random.normal(25, 2, 200),
        }
    )
    ref_file = tmp_path / "ref.csv"
    curr_file = tmp_path / "curr.csv"
    ref_df.to_csv(ref_file, index=False)
    curr_df.to_csv(curr_file, index=False)

    # Standard human-readable output
    res = runner.invoke(
        app,
        [
            "feature-check",
            "--reference",
            str(ref_file),
            "--current",
            str(curr_file),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Training-Serving Feature Skew Report" in res.output
    assert "feat_b" in res.output

    # JSON output with file export
    json_out = tmp_path / "skew_report.json"
    res_json = runner.invoke(
        app,
        [
            "feature-check",
            "--reference",
            str(ref_file),
            "--current",
            str(curr_file),
            "--json",
            "--output",
            str(json_out),
        ],
    )
    assert res_json.exit_code == 0, res_json.output
    report = json.loads(res_json.output)
    assert report["overall_status"] in ("drifted", "warning")
    assert json_out.exists()

    # Fail on skew option when critical drift occurs
    res_fail = runner.invoke(
        app,
        [
            "feature-check",
            "--reference",
            str(ref_file),
            "--current",
            str(curr_file),
            "--fail-on-skew",
        ],
    )
    assert res_fail.exit_code == 1


def test_cli_serve_feature_store_option(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = tmp_path / "artifact"
    artifact_path.mkdir()
    store_path = tmp_path / "store.json"
    store_path.write_text("{}", encoding="utf-8")
    created: dict[str, Any] = {}

    def fake_load_model(_p: Path | str) -> object:
        return object()

    def fake_create_app(**kwargs: Any) -> object:
        created.update(kwargs)
        return object()

    def fake_run(*_a: Any, **_kw: Any) -> None:
        return None

    monkeypatch.setattr("fraud_detection.cli.load_model", fake_load_model)
    monkeypatch.setattr("fraud_detection.api.create_app", fake_create_app)
    monkeypatch.setattr("fraud_detection.cli.uvicorn.run", fake_run)

    res = runner.invoke(
        app,
        [
            "serve",
            str(artifact_path),
            "--feature-store",
            str(store_path),
        ],
    )
    assert res.exit_code == 0, res.output
    assert created["feature_store_path"] == store_path


def test_cli_pipeline_eval_success(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "eval_data.csv"
    generate_synthetic_data(rows=250, random_state=42).to_csv(data_path, index=False)

    result = runner.invoke(
        app,
        [
            "pipeline-eval",
            "--model",
            str(trained_artifact),
            "--input",
            str(data_path),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["total_evaluated"] == 250
    assert report["pipeline_name"] == "production_fraud_pipeline"
    assert report["node_count"] == 5
    assert "decisions" in report
    assert "ALLOW" in report["decisions"]
    assert "latency_ms" in report
    assert "mean" in report["latency_ms"]
    assert "stage_latency_ms" in report
    assert "inference" in report["stage_latency_ms"]
    assert "action" in report["stage_latency_ms"]


def test_cli_pipeline_eval_table_output_and_file(
    tmp_path: Path, trained_artifact: Path
) -> None:
    data_path = tmp_path / "eval_data.csv"
    generate_synthetic_data(rows=250, random_state=42).to_csv(data_path, index=False)
    out_file = tmp_path / "pipeline_report.json"

    result = runner.invoke(
        app,
        [
            "pipeline-eval",
            "--model",
            str(trained_artifact),
            "--input",
            str(data_path),
            "--max-records",
            "50",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Decision Graph Pipeline Evaluation Report:" in result.output
    assert "Evaluated: 50 transactions" in result.output
    assert "Stage Latencies" in result.output
    assert out_file.exists()

    saved_data = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved_data["total_evaluated"] == 50


def test_cli_pipeline_eval_with_rules_and_velocity(
    tmp_path: Path, trained_artifact: Path
) -> None:
    data_path = tmp_path / "eval_data.csv"
    generate_synthetic_data(rows=250, random_state=42).to_csv(data_path, index=False)

    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "rule_id": "rule_high_amount",
                        "name": "High Amount Deny",
                        "description": "Deny amounts over 5000",
                        "action": "DENY",
                        "priority": 100,
                        "conditions": [
                            {"field": "Amount", "operator": ">", "value": 5000.0}
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    vel_path = tmp_path / "velocity.json"
    vel_path.write_text(
        json.dumps(
            {
                "entity_key": "card_id",
                "windows": [
                    {
                        "duration_seconds": 3600.0,
                        "aggregations": ["count", "sum"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "pipeline-eval",
            "--model",
            str(trained_artifact),
            "--input",
            str(data_path),
            "--rules",
            str(rules_path),
            "--velocity-config",
            str(vel_path),
            "--json",
            "--fail-on-degraded",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["total_evaluated"] == 250
    assert data["degraded_count"] == 0


def test_cli_pipeline_eval_missing_model_or_input(tmp_path: Path) -> None:
    missing_model = tmp_path / "non_existent_model"
    data_path = tmp_path / "data.csv"
    data_path.write_text("Time,Amount\n0,10.0\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "pipeline-eval",
            "--model",
            str(missing_model),
            "--input",
            str(data_path),
        ],
    )
    assert result.exit_code != 0


def test_cli_pipeline_eval_corrupt_configs(tmp_path: Path, trained_artifact: Path) -> None:
    data_path = tmp_path / "data.csv"
    generate_synthetic_data(rows=250, random_state=42).to_csv(data_path, index=False)
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{corrupt json", encoding="utf-8")

    # Bad rules
    res_r = runner.invoke(
        app,
        ["pipeline-eval", "-m", str(trained_artifact), "-i", str(data_path), "-r", str(bad_file)],
    )
    assert res_r.exit_code == 2
    assert "Failed to load rules" in res_r.output

    # Bad feature store
    res_fs = runner.invoke(
        app,
        ["pipeline-eval", "-m", str(trained_artifact), "-i", str(data_path), "-fs", str(bad_file)],
    )
    assert res_fs.exit_code == 2
    assert "Failed to load feature store" in res_fs.output

    # Bad velocity
    res_vc = runner.invoke(
        app,
        ["pipeline-eval", "-m", str(trained_artifact), "-i", str(data_path), "-vc", str(bad_file)],
    )
    assert res_vc.exit_code == 2
    assert "Failed to load velocity config" in res_vc.output


def test_cli_pipeline_eval_with_cache_and_speculation(
    tmp_path: Path, trained_artifact: Path
) -> None:
    data_path = tmp_path / "data.csv"
    generate_synthetic_data(rows=250, random_state=42).to_csv(data_path, index=False)

    # 1. JSON report with cache and speculative evaluation enabled
    res_json = runner.invoke(
        app,
        [
            "pipeline-eval",
            "--model",
            str(trained_artifact),
            "--input",
            str(data_path),
            "--enable-cache",
            "--cache-size",
            "200",
            "--cache-ttl",
            "120.0",
            "--speculative",
            "--json",
        ],
    )
    assert res_json.exit_code == 0, res_json.output
    report = json.loads(res_json.stdout)
    assert report["total_evaluated"] == 250
    assert "cache" in report
    assert report["cache"]["enabled"] is True
    assert report["cache"]["max_size"] == 200
    assert report["cache"]["misses"] >= 1
    assert "hit_ratio" in report["cache"]
    assert "speculative" in report
    assert report["speculative"]["enabled"] is True
    assert "executed" in report["speculative"]
    assert "hits" in report["speculative"]

    # 2. Text table report displaying cache and speculative telemetry
    res_text = runner.invoke(
        app,
        [
            "pipeline-eval",
            "--model",
            str(trained_artifact),
            "--input",
            str(data_path),
            "--max-records",
            "20",
            "--enable-cache",
            "--speculative",
        ],
    )
    assert res_text.exit_code == 0, res_text.output
    assert "Cache: Hits=" in res_text.output
    assert "Speculative Inference: Executed=" in res_text.output





