from __future__ import annotations

import json
import subprocess
import urllib.error
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

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
from fraud_detection.model import (
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    MODEL_FILENAME,
    FraudModel,
    save_model,
    train_model,
    verify_attestation,
)

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


def test_predict_cli_with_audit_log(
    tmp_path: Path, trained_artifact: Path
) -> None:
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


def test_promote_cli_with_audit_log(
    tmp_path: Path, trained_artifact: Path
) -> None:
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
    import shutil

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




