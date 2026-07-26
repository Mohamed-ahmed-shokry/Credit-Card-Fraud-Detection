from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from fraud_detection.cli import app
from fraud_detection.model import METADATA_FILENAME, MODEL_FILENAME

runner = CliRunner()


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
    assert metadata["calibration"] == {"method": "sigmoid", "folds": 4}
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
