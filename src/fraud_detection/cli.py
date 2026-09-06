"""Command-line workflows for the fraud detection system."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Annotated, Any, NoReturn
from uuid import uuid4

import numpy as np
import pandas as pd
import typer
import uvicorn

from fraud_detection import __version__
from fraud_detection.data import (
    DEFAULT_TARGET,
    DataValidationError,
    ValidatedDataset,
    generate_synthetic_data,
    load_csv,
)
from fraud_detection.drift import DriftError, assess_drift
from fraud_detection.evaluation import (
    ThresholdRow,
    calibration_report,
    evaluate_predictions,
    expected_classification_cost,
    summarize_thresholds,
)
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

TestSizeOption = Annotated[
    float, typer.Option(min=0.05, max=0.4, help="Untouched test-set fraction.")
]
ValidationSizeOption = Annotated[
    float, typer.Option(min=0.05, max=0.4, help="Threshold-tuning validation fraction.")
]
SeedOption = Annotated[int, typer.Option(help="Random seed.")]
RegularizationOption = Annotated[
    float,
    typer.Option(min=0.000001, help="Inverse regularization strength (logistic regression)."),
]
MaxIterationsOption = Annotated[
    int,
    typer.Option(min=100, help="Solver iterations (logistic regression) or trees (boosting)."),
]
NEstimatorsOption = Annotated[int, typer.Option(min=10, help="Number of trees (random forest).")]
MaxDepthOption = Annotated[
    int | None,
    typer.Option(help="Maximum tree depth for forest/boosting models; omit for unlimited."),
]
LearningRateOption = Annotated[
    float,
    typer.Option(min=0.000001, help="Shrinkage step size (histogram gradient boosting)."),
]
L2RegularizationOption = Annotated[
    float, typer.Option(min=0.0, help="L2 regularization (histogram gradient boosting).")
]
MaxBinsOption = Annotated[
    int, typer.Option(min=2, help="Feature bin count (histogram gradient boosting).")
]
ThresholdStrategyOption = Annotated[
    ThresholdStrategy,
    typer.Option(help="Validation objective: maximize F1 or minimize weighted mistake cost."),
]
CostPolicyOption = Annotated[
    str, typer.Option(help="Name of the business-cost policy recorded in the model card.")
]
FalsePositiveCostOption = Annotated[
    float,
    typer.Option(min=0.000001, help="Relative cost of flagging a legitimate transaction."),
]
FalseNegativeCostOption = Annotated[
    float,
    typer.Option(min=0.000001, help="Relative cost of missing a fraudulent transaction."),
]
CalibrationMethodOption = Annotated[
    CalibrationMethod,
    typer.Option(help="Probability calibration policy fitted within training data."),
]
CalibrationFoldsOption = Annotated[
    int, typer.Option(min=2, max=10, help="Cross-validation folds used for calibration.")
]
CalibrationJobsOption = Annotated[
    int, typer.Option(help="Calibration workers: 1 is conservative; -1 uses all processors.")
]
SplitStrategyOption = Annotated[
    SplitStrategy,
    typer.Option(help="Random stratified or chronological dataset partitioning."),
]
TimeColumnOption = Annotated[
    str, typer.Option(help="Ordering feature used when --split-strategy temporal.")
]


def _training_config(
    *,
    test_size: float,
    validation_size: float,
    random_state: int,
    estimator: EstimatorType,
    max_iterations: int,
    regularization: float,
    n_estimators: int,
    max_depth: int | None,
    learning_rate: float,
    l2_regularization: float,
    max_bins: int,
    threshold_strategy: ThresholdStrategy,
    cost_policy: str,
    false_positive_cost: float,
    false_negative_cost: float,
    calibration_method: CalibrationMethod,
    calibration_folds: int,
    calibration_jobs: int,
    split_strategy: SplitStrategy,
    time_column: str,
    overrides: dict[str, Any] | None = None,
) -> TrainingConfig:
    """Assemble the shared training configuration for train, compare, and stability."""
    kwargs: dict[str, Any] = {
        "test_size": test_size,
        "validation_size": validation_size,
        "random_state": random_state,
        "estimator": estimator,
        "max_iterations": max_iterations,
        "regularization": regularization,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "l2_regularization": l2_regularization,
        "max_bins": max_bins,
        "threshold_strategy": threshold_strategy,
        "cost_policy": cost_policy,
        "false_positive_cost": false_positive_cost,
        "false_negative_cost": false_negative_cost,
        "calibration_method": calibration_method,
        "calibration_folds": calibration_folds,
        "calibration_jobs": calibration_jobs,
        "split_strategy": split_strategy,
        "time_column": time_column,
    }
    if overrides:
        kwargs.update(overrides)
    return TrainingConfig(**kwargs)


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
    test_size: TestSizeOption = 0.2,
    validation_size: ValidationSizeOption = 0.2,
    seed: SeedOption = 42,
    estimator: Annotated[
        EstimatorType,
        typer.Option(
            help="Base classifier: the interpretable logistic-regression baseline, "
            "or an opt-in random forest or histogram gradient boosting model."
        ),
    ] = EstimatorType.LOGISTIC_REGRESSION,
    regularization: RegularizationOption = 1.0,
    max_iterations: MaxIterationsOption = 1_000,
    n_estimators: NEstimatorsOption = 100,
    max_depth: MaxDepthOption = None,
    learning_rate: LearningRateOption = 0.1,
    l2_regularization: L2RegularizationOption = 0.0,
    max_bins: MaxBinsOption = 255,
    threshold_strategy: ThresholdStrategyOption = ThresholdStrategy.F1,
    cost_policy: CostPolicyOption = "default",
    false_positive_cost: FalsePositiveCostOption = 1.0,
    false_negative_cost: FalseNegativeCostOption = 10.0,
    calibration_method: CalibrationMethodOption = CalibrationMethod.SIGMOID,
    calibration_folds: CalibrationFoldsOption = 3,
    calibration_jobs: CalibrationJobsOption = 1,
    split_strategy: SplitStrategyOption = SplitStrategy.STRATIFIED,
    time_column: TimeColumnOption = "Time",
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
        config = _training_config(
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
            cost_policy=cost_policy,
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
    test_size: TestSizeOption = 0.2,
    validation_size: ValidationSizeOption = 0.2,
    seed: SeedOption = 42,
    estimators: Annotated[
        list[EstimatorType] | None,
        typer.Option(
            "--estimator",
            help="Estimator to include; repeat to compare specific choices. "
            "Defaults to comparing every supported estimator.",
        ),
    ] = None,
    regularization: RegularizationOption = 1.0,
    max_iterations: MaxIterationsOption = 1_000,
    n_estimators: NEstimatorsOption = 100,
    max_depth: MaxDepthOption = None,
    learning_rate: LearningRateOption = 0.1,
    l2_regularization: L2RegularizationOption = 0.0,
    max_bins: MaxBinsOption = 255,
    threshold_strategy: ThresholdStrategyOption = ThresholdStrategy.F1,
    cost_policy: CostPolicyOption = "default",
    false_positive_cost: FalsePositiveCostOption = 1.0,
    false_negative_cost: FalseNegativeCostOption = 10.0,
    calibration_method: CalibrationMethodOption = CalibrationMethod.SIGMOID,
    calibration_folds: CalibrationFoldsOption = 3,
    calibration_jobs: CalibrationJobsOption = 1,
    split_strategy: SplitStrategyOption = SplitStrategy.STRATIFIED,
    time_column: TimeColumnOption = "Time",
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
                config = _training_config(
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
                    cost_policy=cost_policy,
                    false_positive_cost=false_positive_cost,
                    false_negative_cost=false_negative_cost,
                    calibration_method=calibration_method,
                    calibration_folds=calibration_folds,
                    calibration_jobs=calibration_jobs,
                    split_strategy=split_strategy,
                    time_column=time_column,
                    overrides={param_name: value},
                )

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
                config = _training_config(
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
                    cost_policy=cost_policy,
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


_STABILITY_METRICS = (
    "roc_auc",
    "average_precision",
    "brier_score",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
)


@app.command("stability")
def stability_command(
    data: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Training CSV."),
    ],
    target: Annotated[
        str,
        typer.Option(help="Binary target column containing 0 and 1."),
    ] = DEFAULT_TARGET,
    test_size: TestSizeOption = 0.2,
    validation_size: ValidationSizeOption = 0.2,
    seed: Annotated[int, typer.Option(help="Base random seed; repeat N uses seed+N.")] = 42,
    repeats: Annotated[
        int,
        typer.Option(min=2, max=10, help="Repeated train/validate/test runs."),
    ] = 5,
    estimators: Annotated[
        list[EstimatorType] | None,
        typer.Option(
            "--estimator",
            help="Estimator to include; repeat to assess specific choices. "
            "Defaults to assessing every supported estimator.",
        ),
    ] = None,
    regularization: RegularizationOption = 1.0,
    max_iterations: MaxIterationsOption = 1_000,
    n_estimators: NEstimatorsOption = 100,
    max_depth: MaxDepthOption = None,
    learning_rate: LearningRateOption = 0.1,
    l2_regularization: L2RegularizationOption = 0.0,
    max_bins: MaxBinsOption = 255,
    threshold_strategy: ThresholdStrategyOption = ThresholdStrategy.F1,
    cost_policy: CostPolicyOption = "default",
    false_positive_cost: FalsePositiveCostOption = 1.0,
    false_negative_cost: FalseNegativeCostOption = 10.0,
    calibration_method: CalibrationMethodOption = CalibrationMethod.SIGMOID,
    calibration_folds: CalibrationFoldsOption = 3,
    calibration_jobs: CalibrationJobsOption = 1,
) -> None:
    """Repeat training with successive seeds and report test-metric stability.

    Nothing is saved; this measures how sensitive holdout metrics are to the
    random split before trusting a single `train` run. Every repeat shares the
    same configuration and differs only in random seed. Repeats require the
    stratified split because chronological windows are deterministic.
    """
    chosen_estimators = list(dict.fromkeys(estimators or list(EstimatorType)))

    try:
        dataset = load_csv(data, target_column=target)
        results = []
        for candidate in chosen_estimators:
            runs = []
            estimator_label = ""
            for offset in range(repeats):
                config = _training_config(
                    test_size=test_size,
                    validation_size=validation_size,
                    random_state=seed + offset,
                    estimator=candidate,
                    max_iterations=max_iterations,
                    regularization=regularization,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    l2_regularization=l2_regularization,
                    max_bins=max_bins,
                    threshold_strategy=threshold_strategy,
                    cost_policy=cost_policy,
                    false_positive_cost=false_positive_cost,
                    false_negative_cost=false_negative_cost,
                    calibration_method=calibration_method,
                    calibration_folds=calibration_folds,
                    calibration_jobs=calibration_jobs,
                    split_strategy=SplitStrategy.STRATIFIED,
                    time_column="Time",
                )
                model = train_model(dataset, config=config)
                estimator_label = str(model.metadata["estimator"])
                runs.append(
                    {
                        "seed": seed + offset,
                        "threshold": model.threshold,
                        "test_metrics": model.metadata["test_metrics"],
                    }
                )
            means = {
                metric: statistics.mean(run["test_metrics"][metric] for run in runs)
                for metric in _STABILITY_METRICS
            }
            deviations = {
                metric: statistics.stdev(run["test_metrics"][metric] for run in runs)
                for metric in _STABILITY_METRICS
            }
            results.append(
                {
                    "estimator": estimator_label,
                    "repeats": repeats,
                    "test_metrics_mean": means,
                    "test_metrics_std": deviations,
                    "runs": runs,
                }
            )
    except (OSError, DataValidationError, ModelArtifactError, ValueError) as exc:
        _abort(str(exc))

    typer.echo(json.dumps({"results": results}, indent=2, sort_keys=True))


@app.command("rolling")
def rolling_command(
    data: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Training CSV."),
    ],
    target: Annotated[
        str,
        typer.Option(help="Binary target column containing 0 and 1."),
    ] = DEFAULT_TARGET,
    time_column: Annotated[
        str,
        typer.Option(help="Ordering feature for chronological prefixes."),
    ] = "Time",
    origins: Annotated[
        int,
        typer.Option(min=2, max=5, help="Rolling time prefixes to evaluate."),
    ] = 3,
    test_size: TestSizeOption = 0.2,
    validation_size: ValidationSizeOption = 0.2,
    seed: SeedOption = 42,
    estimators: Annotated[
        list[EstimatorType] | None,
        typer.Option(
            "--estimator",
            help="Estimator to include; repeat to assess specific choices. "
            "Defaults to assessing every supported estimator.",
        ),
    ] = None,
    regularization: RegularizationOption = 1.0,
    max_iterations: MaxIterationsOption = 1_000,
    n_estimators: NEstimatorsOption = 100,
    max_depth: MaxDepthOption = None,
    learning_rate: LearningRateOption = 0.1,
    l2_regularization: L2RegularizationOption = 0.0,
    max_bins: MaxBinsOption = 255,
    threshold_strategy: ThresholdStrategyOption = ThresholdStrategy.F1,
    cost_policy: CostPolicyOption = "default",
    false_positive_cost: FalsePositiveCostOption = 1.0,
    false_negative_cost: FalseNegativeCostOption = 10.0,
    calibration_method: CalibrationMethodOption = CalibrationMethod.SIGMOID,
    calibration_folds: CalibrationFoldsOption = 3,
    calibration_jobs: CalibrationJobsOption = 1,
) -> None:
    """Evaluate rolling chronological prefixes and report metric spread.

    Nothing is saved; this measures how holdout metrics evolve as more history
    becomes available. Origin N trains, tunes, and tests on the first
    (N+3)/(origins+2) of time-ordered rows using the same temporal split as
    `train`, so later origins strictly extend earlier ones. Every time window
    must contain both classes or the run stops with an actionable error.
    """
    chosen_estimators = list(dict.fromkeys(estimators or list(EstimatorType)))

    try:
        dataset = load_csv(data, target_column=target)
        if time_column not in dataset.features.columns:
            raise ValueError(f"Rolling evaluation requires feature column {time_column!r}.")
        order = np.argsort(dataset.features[time_column].to_numpy(dtype=float), kind="stable")
        ordered_features = dataset.features.iloc[order].reset_index(drop=True)
        ordered_target = dataset.target.iloc[order].reset_index(drop=True)
        results = []
        for candidate in chosen_estimators:
            runs = []
            estimator_label = ""
            for index in range(origins):
                prefix_rows = max(1, round(len(ordered_target) * (index + 3) / (origins + 2)))
                prefix = ValidatedDataset(
                    features=ordered_features.iloc[:prefix_rows],
                    target=ordered_target.iloc[:prefix_rows],
                )
                config = _training_config(
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
                    cost_policy=cost_policy,
                    false_positive_cost=false_positive_cost,
                    false_negative_cost=false_negative_cost,
                    calibration_method=calibration_method,
                    calibration_folds=calibration_folds,
                    calibration_jobs=calibration_jobs,
                    split_strategy=SplitStrategy.TEMPORAL,
                    time_column=time_column,
                )
                model = train_model(prefix, config=config)
                estimator_label = str(model.metadata["estimator"])
                time_ranges = model.metadata["split_time_ranges"]
                runs.append(
                    {
                        "origin": index,
                        "rows": prefix_rows,
                        "threshold": model.threshold,
                        "test_metrics": model.metadata["test_metrics"],
                        "test_time_range": time_ranges["test"],
                    }
                )
            means = {
                metric: statistics.mean(run["test_metrics"][metric] for run in runs)
                for metric in _STABILITY_METRICS
            }
            deviations = {
                metric: statistics.stdev(run["test_metrics"][metric] for run in runs)
                for metric in _STABILITY_METRICS
            }
            results.append(
                {
                    "estimator": estimator_label,
                    "origins": origins,
                    "test_metrics_mean": means,
                    "test_metrics_std": deviations,
                    "runs": runs,
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
    threshold: Annotated[
        float | None,
        typer.Option(
            help="Override the model's tuned threshold for audit/backtest scoring. "
            "Both thresholds are reported in the summary."
        ),
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing output file.")] = False,
) -> None:
    """Batch-score transactions and write probabilities plus binary decisions."""
    if output.exists() and not overwrite:
        _abort(f"Output already exists: {output}. Pass --overwrite to replace it.")
    if threshold is not None and (isinstance(threshold, bool) or not 0.0 <= threshold <= 1.0):
        _abort(f"Invalid threshold {threshold!r}: must fall between 0 and 1.")

    try:
        model = load_model(model_path)
        frame = pd.read_csv(data)
        features = frame.drop(columns=target, errors="ignore")
        probabilities = model.predict_probabilities(features)
        applied_threshold = model.threshold if threshold is None else threshold
        predictions = (probabilities >= applied_threshold).astype("int8")
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
                "threshold": applied_threshold,
                "model_threshold": model.threshold,
                "threshold_overridden": threshold is not None,
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


@app.command("thresholds")
def thresholds_command(
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
    thresholds: Annotated[
        str,
        typer.Option(help="Comma-separated candidate thresholds, e.g. '0.2,0.5,0.8'."),
    ] = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
    false_positive_cost: Annotated[
        float | None,
        typer.Option(help="Override the model's false-positive cost weight."),
    ] = None,
    false_negative_cost: Annotated[
        float | None,
        typer.Option(help="Override the model's false-negative cost weight."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional JSON report destination."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace an existing report file."),
    ] = False,
) -> None:
    """Score candidate thresholds on labeled transactions.

    Reports precision, recall, F1, and expected cost per threshold alongside
    the model's tuned operating point. Use held-out labeled data (never the
    threshold-tuning validation split). Costs default to the model's training
    policy unless overridden.
    """
    if output is not None and output.exists() and not overwrite:
        _abort(f"Output already exists: {output}. Pass --overwrite to replace it.")

    try:
        model = load_model(model_path)
        dataset = load_csv(data, target_column=target)
        probabilities = model.predict_probabilities(dataset.features)
        candidates = _parse_thresholds(thresholds)
        policy_name, fp_cost, fn_cost = _resolve_report_costs(
            model, false_positive_cost, false_negative_cost
        )
        y_true = dataset.target.to_numpy()
        tradeoff = summarize_thresholds(
            y_true,
            probabilities,
            candidates,
            false_positive_cost=fp_cost,
            false_negative_cost=fn_cost,
        ).to_dict()
        at_tuned = evaluate_predictions(y_true, probabilities, threshold=model.threshold)
        tuned_cost = expected_classification_cost(
            y_true,
            probabilities,
            threshold=model.threshold,
            false_positive_cost=fp_cost,
            false_negative_cost=fn_cost,
        )
        tuned_row = ThresholdRow(
            threshold=model.threshold,
            precision=at_tuned.precision,
            recall=at_tuned.recall,
            f1=at_tuned.f1,
            expected_cost_per_transaction=tuned_cost,
            flagged=at_tuned.false_positives + at_tuned.true_positives,
            flagged_rate=(at_tuned.false_positives + at_tuned.true_positives) / y_true.size,
            true_positives=at_tuned.true_positives,
            false_positives=at_tuned.false_positives,
        ).to_dict()
        report_json = json.dumps(
            {
                "model_version": str(model.metadata["dataset_fingerprint"])[:12],
                "model_threshold": model.threshold,
                "cost_policy": {
                    "name": policy_name,
                    "false_positive_cost": fp_cost,
                    "false_negative_cost": fn_cost,
                },
                "false_positive_cost": fp_cost,
                "false_negative_cost": fn_cost,
                "model_threshold_metrics": tuned_row,
                **tradeoff,
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


def _parse_thresholds(raw: str) -> list[float]:
    try:
        return [float(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid thresholds {raw!r}: every entry must be a number.") from exc


def _resolve_report_costs(
    model: object,
    false_positive_cost: float | None,
    false_negative_cost: float | None,
) -> tuple[str, float, float]:
    """Resolve the effective named cost policy for a report.

    Without overrides the model's persisted policy applies; any override
    produces a policy named "custom" so ad-hoc weights are never confused
    with a recorded business-cost definition.
    """
    metadata = getattr(model, "metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Model artifact metadata is invalid.")
    policy = metadata.get("cost_policy", {})
    if policy is None:
        policy = {}
    if not isinstance(policy, dict):
        raise ValueError("Model artifact cost_policy is invalid.")
    training_config = metadata.get("training_config", {})
    if training_config is None:
        training_config = {}
    if not isinstance(training_config, dict):
        raise ValueError("Model artifact training_config is invalid.")
    policy_name = policy.get("name", "default")
    if not isinstance(policy_name, str) or not policy_name.strip():
        raise ValueError("Model artifact cost_policy name is invalid.")
    base_fp = policy.get("false_positive_cost", training_config.get("false_positive_cost", 1.0))
    base_fn = policy.get("false_negative_cost", training_config.get("false_negative_cost", 10.0))
    raw_fp: Any = base_fp if false_positive_cost is None else false_positive_cost
    raw_fn: Any = base_fn if false_negative_cost is None else false_negative_cost
    try:
        fp_cost = float(raw_fp)
        fn_cost = float(raw_fn)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid classification costs: {exc}") from exc
    if false_positive_cost is None and false_negative_cost is None:
        return (policy_name, fp_cost, fn_cost)
    return ("custom", fp_cost, fn_cost)


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
