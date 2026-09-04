from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from fraud_detection import __version__
from fraud_detection.cli import app
from fraud_detection.data import generate_synthetic_data, validate_frame
from fraud_detection.model import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    FraudModel,
    save_model,
    train_model,
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
        ],
    )
    assert predicted.exit_code == 0, predicted.output

    scored = pd.read_csv(output_path)
    assert {"fraud_probability", "is_fraud"}.issubset(scored.columns)
    contrib_cols = [c for c in scored.columns if c.startswith("contrib_")]
    assert len(contrib_cols) == 30
