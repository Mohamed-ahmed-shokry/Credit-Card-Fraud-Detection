"""Command-line workflows for the fraud detection system."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, NoReturn

import pandas as pd
import typer
import uvicorn

from fraud_detection.data import (
    DEFAULT_TARGET,
    DataValidationError,
    generate_synthetic_data,
    load_csv,
)
from fraud_detection.drift import DriftError, assess_drift
from fraud_detection.model import (
    CalibrationMethod,
    ModelArtifactError,
    SplitStrategy,
    ThresholdStrategy,
    TrainingConfig,
    load_model,
    save_model,
    train_model,
)

app = typer.Typer(
    name="fraud-detect",
    help="Train, inspect, and run a credit-card fraud detector.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.command("generate-data")
def generate_data_command(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination CSV file."),
    ] = Path("data/synthetic_transactions.csv"),
    rows: Annotated[int, typer.Option(min=200, help="Number of transactions.")] = 5_000,
    fraud_rate: Annotated[
        float,
        typer.Option(min=0.005, max=0.5, help="Approximate fraction of fraudulent rows."),
    ] = 0.02,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing output file.")] = False,
) -> None:
    """Create deterministic synthetic data for demos and smoke tests."""
    if output.exists() and not overwrite:
        _abort(f"Output already exists: {output}. Pass --overwrite to replace it.")

    try:
        frame = generate_synthetic_data(rows=rows, fraud_rate=fraud_rate, random_state=seed)
    except ValueError as exc:
        _abort(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "rows": len(frame),
                "fraud_rows": int(frame[DEFAULT_TARGET].sum()),
            }
        )
    )


@app.command("train")
def train_command(
    data: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Training CSV."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Artifact output directory."),
    ] = Path("artifacts/model"),
    target: Annotated[
        str,
        typer.Option(help="Binary target column containing 0 and 1."),
    ] = DEFAULT_TARGET,
    test_size: Annotated[
        float,
        typer.Option(min=0.05, max=0.4, help="Untouched test-set fraction."),
    ] = 0.2,
    validation_size: Annotated[
        float,
        typer.Option(min=0.05, max=0.4, help="Threshold-tuning validation fraction."),
    ] = 0.2,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
    threshold_strategy: Annotated[
        ThresholdStrategy,
        typer.Option(help="Validation objective: maximize F1 or minimize weighted mistake cost."),
    ] = ThresholdStrategy.F1,
    false_positive_cost: Annotated[
        float,
        typer.Option(min=0.000001, help="Relative cost of flagging a legitimate transaction."),
    ] = 1.0,
    false_negative_cost: Annotated[
        float,
        typer.Option(min=0.000001, help="Relative cost of missing a fraudulent transaction."),
    ] = 10.0,
    calibration_method: Annotated[
        CalibrationMethod,
        typer.Option(help="Probability calibration policy fitted within training data."),
    ] = CalibrationMethod.SIGMOID,
    calibration_folds: Annotated[
        int,
        typer.Option(min=2, max=10, help="Cross-validation folds used for calibration."),
    ] = 3,
    split_strategy: Annotated[
        SplitStrategy,
        typer.Option(help="Random stratified or chronological dataset partitioning."),
    ] = SplitStrategy.STRATIFIED,
    time_column: Annotated[
        str,
        typer.Option(help="Ordering feature used when --split-strategy temporal."),
    ] = "Time",
) -> None:
    """Train, tune on validation data, evaluate on test data, and save."""
    try:
        dataset = load_csv(data, target_column=target)
        config = TrainingConfig(
            test_size=test_size,
            validation_size=validation_size,
            random_state=seed,
            threshold_strategy=threshold_strategy,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
            calibration_method=calibration_method,
            calibration_folds=calibration_folds,
            split_strategy=split_strategy,
            time_column=time_column,
        )
        model = train_model(dataset, config=config)
        model_path = save_model(model, output)
    except (DataValidationError, ModelArtifactError, ValueError) as exc:
        _abort(str(exc))

    typer.echo(
        json.dumps(
            {
                "model": str(model_path),
                "threshold": model.threshold,
                "test_metrics": model.metadata["test_metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("predict")
def predict_command(
    model_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Model file or artifact directory."),
    ],
    data: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Transactions CSV."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Scored CSV destination."),
    ] = Path("predictions.csv"),
    target: Annotated[
        str,
        typer.Option(help="Optional label column to exclude from model features."),
    ] = DEFAULT_TARGET,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing output file.")] = False,
) -> None:
    """Batch-score transactions and write probabilities plus binary decisions."""
    if output.exists() and not overwrite:
        _abort(f"Output already exists: {output}. Pass --overwrite to replace it.")

    try:
        model = load_model(model_path)
        frame = pd.read_csv(data)
        features = frame.drop(columns=target, errors="ignore")
        probabilities = model.predict_probabilities(features)
        predictions = (probabilities >= model.threshold).astype("int8")
    except (OSError, pd.errors.ParserError, ModelArtifactError) as exc:
        _abort(str(exc))

    scored = frame.copy()
    scored["fraud_probability"] = probabilities
    scored["is_fraud"] = predictions
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output, index=False)
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "rows": len(scored),
                "flagged": int(predictions.sum()),
            }
        )
    )


@app.command("inspect")
def inspect_command(
    model_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Model file or artifact directory."),
    ],
) -> None:
    """Print the persisted model card as JSON."""
    try:
        model = load_model(model_path)
    except ModelArtifactError as exc:
        _abort(str(exc))
    typer.echo(json.dumps(model.metadata, indent=2, sort_keys=True))


@app.command("explain")
def explain_command(
    model_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Model file or artifact directory."),
    ],
    top: Annotated[
        int,
        typer.Option(min=1, max=100, help="Number of ranked feature effects to show."),
    ] = 10,
) -> None:
    """Show the strongest global standardized feature effects."""
    try:
        model = load_model(model_path)
        effects = model.metadata.get("feature_effects")
        if not isinstance(effects, list):
            raise ModelArtifactError("Model artifact does not contain feature effects.")
    except ModelArtifactError as exc:
        _abort(str(exc))

    typer.echo(
        json.dumps(
            {
                "model_version": str(model.metadata["dataset_fingerprint"])[:12],
                "effects": effects[:top],
            },
            indent=2,
        )
    )


@app.command("drift")
def drift_command(
    model_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Model file or artifact directory."),
    ],
    data: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Current transactions."),
    ],
    target: Annotated[
        str,
        typer.Option(help="Optional label column to exclude from feature analysis."),
    ] = DEFAULT_TARGET,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional JSON report destination."),
    ] = None,
) -> None:
    """Compare current feature distributions with the training baseline."""
    try:
        model = load_model(model_path)
        frame = pd.read_csv(data).drop(columns=target, errors="ignore")
        features = model.validate_features(frame)
        profile = model.metadata.get("reference_profile")
        if not isinstance(profile, dict):
            raise DriftError("Model artifact does not contain a reference profile.")
        report_json = json.dumps(assess_drift(profile, features).to_dict(), indent=2)
    except (OSError, pd.errors.ParserError, ModelArtifactError, DriftError) as exc:
        _abort(str(exc))

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report_json + "\n", encoding="utf-8")
    typer.echo(report_json)


@app.command("serve")
def serve_command(
    model_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Model file or artifact directory."),
    ],
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65_535, help="TCP port.")] = 8000,
) -> None:
    """Run the versioned HTTP prediction service."""
    from fraud_detection.api import create_app

    try:
        model = load_model(model_path)
    except ModelArtifactError as exc:
        _abort(str(exc))
    uvicorn.run(create_app(model=model), host=host, port=port)


def _abort(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
