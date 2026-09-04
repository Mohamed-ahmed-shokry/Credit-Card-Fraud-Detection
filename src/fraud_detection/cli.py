"""Command-line workflows for the fraud detection system."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Annotated, Any, NoReturn
from uuid import uuid4

import pandas as pd
import typer
import uvicorn

from fraud_detection import __version__
from fraud_detection.data import (
    DEFAULT_TARGET,
    DataValidationError,
    generate_synthetic_data,
    load_csv,
)
from fraud_detection.drift import DriftError, assess_drift
from fraud_detection.evaluation import calibration_report
from fraud_detection.model import (
    CalibrationMethod,
    EstimatorType,
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


def _version_callback(show_version: bool) -> None:
    if show_version:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = False,
) -> None:
    """Train, inspect, and run a credit-card fraud detector."""


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
        _atomic_write_csv(frame, output)
    except (OSError, ValueError) as exc:
        _abort(str(exc))
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
    estimator: Annotated[
        EstimatorType,
        typer.Option(
            help="Base classifier: the interpretable logistic-regression baseline, "
            "or an opt-in random forest or histogram gradient boosting model."
        ),
    ] = EstimatorType.LOGISTIC_REGRESSION,
    regularization: Annotated[
        float,
        typer.Option(min=0.000001, help="Inverse regularization strength (logistic regression)."),
    ] = 1.0,
    max_iterations: Annotated[
        int,
        typer.Option(min=100, help="Solver iterations (logistic regression) or trees (boosting)."),
    ] = 1_000,
    n_estimators: Annotated[
        int,
        typer.Option(min=10, help="Number of trees (random forest)."),
    ] = 100,
    max_depth: Annotated[
        int | None,
        typer.Option(help="Maximum tree depth for forest/boosting models; omit for unlimited."),
    ] = None,
    learning_rate: Annotated[
        float,
        typer.Option(min=0.000001, help="Shrinkage step size (histogram gradient boosting)."),
    ] = 0.1,
    l2_regularization: Annotated[
        float,
        typer.Option(min=0.0, help="L2 regularization (histogram gradient boosting)."),
    ] = 0.0,
    max_bins: Annotated[
        int,
        typer.Option(min=2, help="Feature bin count (histogram gradient boosting)."),
    ] = 255,
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
    calibration_jobs: Annotated[
        int,
        typer.Option(help="Calibration workers: 1 is conservative; -1 uses all processors."),
    ] = 1,
    split_strategy: Annotated[
        SplitStrategy,
        typer.Option(help="Random stratified or chronological dataset partitioning."),
    ] = SplitStrategy.STRATIFIED,
    time_column: Annotated[
        str,
        typer.Option(help="Ordering feature used when --split-strategy temporal."),
    ] = "Time",
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace an existing model artifact."),
    ] = False,
) -> None:
    """Train, tune on validation data, evaluate on test data, and save."""
    output_has_content = output.is_file() or (
        output.is_dir() and next(output.iterdir(), None) is not None
    )
    if output_has_content and not overwrite:
        _abort(f"Output already exists: {output}. Pass --overwrite to replace it.")

    try:
        dataset = load_csv(data, target_column=target)
        config = TrainingConfig(
            test_size=test_size,
            validation_size=validation_size,
            random_state=seed,
            estimator=estimator,
            max_iterations=max_iterations,
            regularization=regularization,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            max_bins=max_bins,
            threshold_strategy=threshold_strategy,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
            calibration_method=calibration_method,
            calibration_folds=calibration_folds,
            calibration_jobs=calibration_jobs,
            split_strategy=split_strategy,
            time_column=time_column,
        )
        model = train_model(dataset, config=config)
        model_path = save_model(model, output)
    except (OSError, DataValidationError, ModelArtifactError, ValueError) as exc:
        _abort(str(exc))

    typer.echo(
        json.dumps(
            {
                "model": str(model_path),
                "estimator": model.metadata["estimator"],
                "threshold": model.threshold,
                "test_metrics": model.metadata["test_metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("compare")
def compare_command(
    data: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Training CSV."),
    ],
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
    estimators: Annotated[
        list[EstimatorType] | None,
        typer.Option(
            "--estimator",
            help="Estimator to include; repeat to compare specific choices. "
            "Defaults to comparing every supported estimator.",
        ),
    ] = None,
    regularization: Annotated[
        float,
        typer.Option(min=0.000001, help="Inverse regularization strength (logistic regression)."),
    ] = 1.0,
    max_iterations: Annotated[
        int,
        typer.Option(min=100, help="Solver iterations (logistic regression) or trees (boosting)."),
    ] = 1_000,
    n_estimators: Annotated[
        int,
        typer.Option(min=10, help="Number of trees (random forest)."),
    ] = 100,
    max_depth: Annotated[
        int | None,
        typer.Option(help="Maximum tree depth for forest/boosting models; omit for unlimited."),
    ] = None,
    learning_rate: Annotated[
        float,
        typer.Option(min=0.000001, help="Shrinkage step size (histogram gradient boosting)."),
    ] = 0.1,
    l2_regularization: Annotated[
        float,
        typer.Option(min=0.0, help="L2 regularization (histogram gradient boosting)."),
    ] = 0.0,
    max_bins: Annotated[
        int,
        typer.Option(min=2, help="Feature bin count (histogram gradient boosting)."),
    ] = 255,
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
    calibration_jobs: Annotated[
        int,
        typer.Option(help="Calibration workers: 1 is conservative; -1 uses all processors."),
    ] = 1,
    split_strategy: Annotated[
        SplitStrategy,
        typer.Option(help="Random stratified or chronological dataset partitioning."),
    ] = SplitStrategy.STRATIFIED,
    time_column: Annotated[
        str,
        typer.Option(help="Ordering feature used when --split-strategy temporal."),
    ] = "Time",
    param_name: Annotated[
        str | None,
        typer.Option(
            "--param-name",
            help="Hyperparameter name to sweep (e.g., regularization, max_iterations, max_depth). "
            "Requires --param-values.",
        ),
    ] = None,
    param_values: Annotated[
        str | None,
        typer.Option(
            "--param-values",
            help="Comma-separated values for the hyperparameter sweep (e.g., 0.1,1.0,10.0). "
            "Requires --param-name.",
        ),
    ] = None,
) -> None:
    """Train each estimator (or hyperparameter configuration) on the same split
    and report metrics side by side.

    Nothing is saved; this is for evidence-based estimator/hyperparameter
    selection before a real `train` run. Every candidate shares the same split,
    calibration policy, and threshold-selection objective.

    For hyperparameter sweeps, specify a single estimator with --estimator and
    use --param-name/--param-values to define the sweep.
    """
    chosen_estimators = list(dict.fromkeys(estimators or list(EstimatorType)))

    sweepable_parameters: dict[str, tuple[str, set[EstimatorType]]] = {
        "regularization": ("float", {EstimatorType.LOGISTIC_REGRESSION}),
        "max_iterations": (
            "int",
            {EstimatorType.LOGISTIC_REGRESSION, EstimatorType.HIST_GRADIENT_BOOSTING},
        ),
        "n_estimators": ("int", {EstimatorType.RANDOM_FOREST}),
        "max_depth": (
            "int",
            {EstimatorType.RANDOM_FOREST, EstimatorType.HIST_GRADIENT_BOOSTING},
        ),
        "learning_rate": ("float", {EstimatorType.HIST_GRADIENT_BOOSTING}),
        "l2_regularization": ("float", {EstimatorType.HIST_GRADIENT_BOOSTING}),
        "max_bins": ("int", {EstimatorType.HIST_GRADIENT_BOOSTING}),
        "calibration_folds": (
            "int",
            {
                EstimatorType.LOGISTIC_REGRESSION,
                EstimatorType.RANDOM_FOREST,
                EstimatorType.HIST_GRADIENT_BOOSTING,
            },
        ),
    }

    if param_name is not None and param_values is not None:
        if len(chosen_estimators) != 1:
            _abort("--param-name/--param-values requires exactly one estimator via --estimator")
        if param_name not in sweepable_parameters:
            _abort(
                f"Unknown hyperparameter: {param_name}. "
                f"Supported: {', '.join(sorted(sweepable_parameters))}"
            )
        _, supported_estimators = sweepable_parameters[param_name]
        if chosen_estimators[0] not in supported_estimators:
            _abort(
                f"Hyperparameter {param_name!r} does not apply to {chosen_estimators[0].value!r}."
            )
        param_values_list = [v.strip() for v in param_values.split(",") if v.strip()]
        if not param_values_list:
            _abort("--param-values must contain at least one value")
    elif param_name is not None or param_values is not None:
        _abort("Both --param-name and --param-values must be provided together")
    else:
        param_values_list = None

    try:
        dataset = load_csv(data, target_column=target)
        results = []

        if param_values_list is not None:
            # Hyperparameter sweep mode
            if param_name is None:
                _abort("Internal error: param_name should not be None in sweep mode")
            candidate = chosen_estimators[0]
            for value_str in param_values_list:
                # Parse value based on parameter type
                kind, _ = sweepable_parameters[param_name]
                try:
                    value: int | float = int(value_str) if kind == "int" else float(value_str)
                except ValueError as exc:
                    _abort(f"Invalid value {value_str!r} for hyperparameter {param_name!r}: {exc}")

                # Build config with the hyperparameter value
                config_kwargs: dict[str, Any] = {
                    "test_size": test_size,
                    "validation_size": validation_size,
                    "random_state": seed,
                    "estimator": candidate,
                    "max_iterations": max_iterations,
                    "regularization": regularization,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "l2_regularization": l2_regularization,
                    "max_bins": max_bins,
                    "threshold_strategy": threshold_strategy,
                    "false_positive_cost": false_positive_cost,
                    "false_negative_cost": false_negative_cost,
                    "calibration_method": calibration_method,
                    "calibration_folds": calibration_folds,
                    "calibration_jobs": calibration_jobs,
                    "split_strategy": split_strategy,
                    "time_column": time_column,
                }
                config_kwargs[param_name] = value
                config = TrainingConfig(**config_kwargs)

                model = train_model(dataset, config=config)
                results.append(
                    {
                        "estimator": model.metadata["estimator"],
                        "hyperparameter": {param_name: value},
                        "threshold": model.threshold,
                        "validation_metrics": model.metadata["validation_metrics"],
                        "test_metrics": model.metadata["test_metrics"],
                    }
                )
        else:
            # Estimator comparison mode
            for candidate in chosen_estimators:
                config = TrainingConfig(
                    test_size=test_size,
                    validation_size=validation_size,
                    random_state=seed,
                    estimator=candidate,
                    max_iterations=max_iterations,
                    regularization=regularization,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    l2_regularization=l2_regularization,
                    max_bins=max_bins,
                    threshold_strategy=threshold_strategy,
                    false_positive_cost=false_positive_cost,
                    false_negative_cost=false_negative_cost,
                    calibration_method=calibration_method,
                    calibration_folds=calibration_folds,
                    calibration_jobs=calibration_jobs,
                    split_strategy=split_strategy,
                    time_column=time_column,
                )
                model = train_model(dataset, config=config)
                results.append(
                    {
                        "estimator": model.metadata["estimator"],
                        "threshold": model.threshold,
                        "validation_metrics": model.metadata["validation_metrics"],
                        "test_metrics": model.metadata["test_metrics"],
                    }
                )
    except (OSError, DataValidationError, ModelArtifactError, ValueError) as exc:
        _abort(str(exc))

    typer.echo(json.dumps({"results": results}, indent=2, sort_keys=True))


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
    explain: Annotated[
        bool,
        typer.Option(help="Include per-transaction feature contributions in the output."),
    ] = False,
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
        scored = frame.copy()
        scored["fraud_probability"] = probabilities
        scored["is_fraud"] = predictions
        if explain:
            explanations = model.explain_local(features)
            for idx, expl in enumerate(explanations):
                for feature, contribution in expl.items():
                    scored.loc[scored.index[idx], f"contrib_{feature}"] = contribution
        _atomic_write_csv(scored, output)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ModelArtifactError,
    ) as exc:
        _abort(str(exc))
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "rows": len(scored),
                "flagged": int(predictions.sum()),
            }
        )
    )


@app.command("benchmark")
def benchmark_command(
    model_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Model file or artifact directory."),
    ],
    data: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Transactions CSV."),
    ],
    target: Annotated[
        str,
        typer.Option(help="Optional label column to exclude from model features."),
    ] = DEFAULT_TARGET,
    batch_sizes: Annotated[
        str,
        typer.Option(help="Comma-separated scoring batch sizes, e.g. '1,10,100,1000'."),
    ] = "1,10,100,1000",
    repeat: Annotated[
        int,
        typer.Option(min=1, max=10, help="Timed runs per batch size; the median is reported."),
    ] = 3,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional JSON report destination."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace an existing report file."),
    ] = False,
) -> None:
    """Time offline batch scoring for capacity planning.

    Measures this host only; production throughput also depends on the serving
    stack, concurrency, and hardware. Nothing is saved except the report.
    """
    if output is not None and output.exists() and not overwrite:
        _abort(f"Output already exists: {output}. Pass --overwrite to replace it.")

    try:
        sizes = _parse_batch_sizes(batch_sizes)
        model = load_model(model_path)
        frame = pd.read_csv(data)
        features = model.validate_features(frame.drop(columns=target, errors="ignore"))
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ModelArtifactError,
        ValueError,
    ) as exc:
        _abort(str(exc))

    rows = len(features)
    results = []
    for size in sizes:
        batch = features.iloc[[index % rows for index in range(size)]]
        model.predict_probabilities(batch)  # warmup, excluded from timing
        samples = []
        for _ in range(repeat):
            started = time.perf_counter()
            model.predict_probabilities(batch)
            samples.append((time.perf_counter() - started) * 1_000)
        median_ms = max(statistics.median(samples), 1e-9)
        results.append(
            {
                "batch_size": size,
                "median_ms": median_ms,
                "ms_per_transaction": median_ms / size,
                "transactions_per_second": 1_000.0 / (median_ms / size),
            }
        )

    report_json = json.dumps(
        {
            "model_version": str(model.metadata["dataset_fingerprint"])[:12],
            "rows_available": rows,
            "repeat": repeat,
            "results": results,
        },
        indent=2,
    )
    try:
        if output is not None:
            _atomic_write_text(report_json + "\n", output)
    except OSError as exc:
        _abort(str(exc))

    typer.echo(report_json)


def _parse_batch_sizes(raw: str) -> list[int]:
    try:
        sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid batch_sizes {raw!r}: every entry must be an integer.") from exc
    if not sizes:
        raise ValueError("batch_sizes must contain at least one batch size.")
    if any(size < 1 for size in sizes):
        raise ValueError("batch_sizes must contain only positive batch sizes.")
    if any(size > 100_000 for size in sizes):
        raise ValueError("batch_sizes must not exceed 100,000 per batch.")
    return sizes


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
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace an existing report file."),
    ] = False,
) -> None:
    """Compare current feature distributions with the training baseline."""
    if output is not None and output.exists() and not overwrite:
        _abort(f"Output already exists: {output}. Pass --overwrite to replace it.")

    try:
        model = load_model(model_path)
        frame = pd.read_csv(data).drop(columns=target, errors="ignore")
        features = model.validate_features(frame)
        profile = model.metadata.get("reference_profile")
        if not isinstance(profile, dict):
            raise DriftError("Model artifact does not contain a reference profile.")
        report = assess_drift(profile, features, thresholds=model.metadata.get("drift_thresholds"))
        report_json = json.dumps(report.to_dict(), indent=2)
        if output is not None:
            _atomic_write_text(report_json + "\n", output)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ModelArtifactError,
        DriftError,
    ) as exc:
        _abort(str(exc))

    typer.echo(report_json)


@app.command("calibration")
def calibration_command(
    model_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Model file or artifact directory."),
    ],
    data: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="Labeled transactions CSV."
        ),
    ],
    target: Annotated[
        str,
        typer.Option(help="Binary label column containing 0 and 1."),
    ] = DEFAULT_TARGET,
    bins: Annotated[
        int,
        typer.Option(min=2, max=20, help="Equal-width reliability bins."),
    ] = 10,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional JSON report destination."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace an existing report file."),
    ] = False,
) -> None:
    """Report reliability bins and Brier decomposition for labeled transactions.

    Use held-out labeled data (never the threshold-tuning validation split) to
    judge whether predicted probabilities mean what they say.
    """
    if output is not None and output.exists() and not overwrite:
        _abort(f"Output already exists: {output}. Pass --overwrite to replace it.")

    try:
        model = load_model(model_path)
        dataset = load_csv(data, target_column=target)
        probabilities = model.predict_probabilities(dataset.features)
        report = calibration_report(dataset.target.to_numpy(), probabilities, bins=bins).to_dict()
        report_json = json.dumps(
            {
                "model_version": str(model.metadata["dataset_fingerprint"])[:12],
                **report,
            },
            indent=2,
        )
        if output is not None:
            _atomic_write_text(report_json + "\n", output)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ModelArtifactError,
        DataValidationError,
        ValueError,
    ) as exc:
        _abort(str(exc))

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


def _atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(content: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    app()
