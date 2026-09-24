"""Command-line workflows for the fraud detection system."""

from __future__ import annotations

import json
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Annotated, Any, NoReturn
from uuid import uuid4

import numpy as np
import pandas as pd
import typer
import uvicorn

from fraud_detection import __version__
from fraud_detection.audit import (
    JsonlAuditSink,
    build_promotion_audit_event,
    build_scoring_audit_event,
    replay_audit_log,
)
from fraud_detection.data import (
    DEFAULT_TARGET,
    DataValidationError,
    ValidatedDataset,
    generate_synthetic_data,
    inject_drift,
    load_csv,
)
from fraud_detection.drift import (
    DriftError,
    DriftReport,
    MultiWindowDriftReport,
    StreamingProfile,
    assess_drift,
    assess_multi_window_drift,
    build_reference_profile,
    surveillance_tripped,
)
from fraud_detection.edge import EdgeArtifactError, build_edge_model
from fraud_detection.evaluation import (
    ThresholdRow,
    calibration_report,
    evaluate_predictions,
    expected_classification_cost,
    summarize_thresholds,
)
from fraud_detection.explanations import ExplanationProvider
from fraud_detection.model import (
    MANIFEST_FILENAME,
    CalibrationMethod,
    EstimatorType,
    FraudModel,
    ModelArtifactError,
    SplitStrategy,
    ThresholdStrategy,
    TrainingConfig,
    load_model,
    save_model,
    split_dataset,
    train_model,
    validate_artifact,
    verify_attestation,
)
from fraud_detection.reporting import ComplianceReportError, render_compliance_report
from fraud_detection.signing import (
    SigningError,
    load_private_key,
    load_public_key,
    verify_attestation_signature,
    write_keypair,
)
from fraud_detection.trust import (
    TrustBundle,
    TrustBundleError,
    load_public_keys,
    load_trust_bundle,
    verify_attestation_with_bundle,
    write_trust_bundle,
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
TemporalGapOption = Annotated[
    float,
    typer.Option(
        min=0.0,
        help="Minimum temporal gap (in time units) between splits to enforce label delay.",
    ),
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
    temporal_gap: float = 0.0,
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
        "temporal_gap": temporal_gap,
    }
    if overrides:
        kwargs.update(overrides)
    return TrainingConfig(**kwargs)


def _version_callback(show_version: bool) -> None:
    if show_version:
        typer.echo(__version__)
        raise typer.Exit()


def _git_info() -> dict[str, str]:
    """Return best-effort Git information for model provenance and inspection."""
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path.cwd(),
            check=False,
        )
        if commit_result.returncode != 0 or not commit_result.stdout.strip():
            return {}

        info = {"commit": commit_result.stdout.strip()}
        log_result = subprocess.run(
            ["git", "log", "-1", "--format=%ci %s"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path.cwd(),
            check=False,
        )
        if log_result.returncode == 0 and log_result.stdout.strip():
            info["last_commit"] = log_result.stdout.strip()

        remote_result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path.cwd(),
            check=False,
        )
        if remote_result.returncode == 0 and remote_result.stdout.strip():
            info["repository"] = remote_result.stdout.strip()
        return info
    except (OSError, subprocess.SubprocessError):
        return {}


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
    temporal_gap: TemporalGapOption = 0.0,
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
            temporal_gap=temporal_gap,
        )
        git_info = _git_info()
        provenance = {
            key: git_info[source]
            for key, source in (
                ("git_commit", "commit"),
                ("git_repository", "repository"),
            )
            if source in git_info
        }
        model = train_model(dataset, config=config, provenance=provenance or None)
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


@app.command("retrain")
def retrain_command(
    champion: Annotated[
        Path,
        typer.Argument(
            exists=True, readable=True, help="Existing champion model file or directory."
        ),
    ],
    data: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="New labeled transaction dataset."
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination path for challenger model artifact."),
    ] = Path("artifacts/challenger"),
    target: Annotated[
        str,
        typer.Option(help="Binary target column containing 0 and 1."),
    ] = DEFAULT_TARGET,
    metric: Annotated[
        str,
        typer.Option(
            help="Primary evaluation metric for promotion: 'auprc', 'f1', or 'expected_cost'."
        ),
    ] = "auprc",
    min_gain: Annotated[
        float,
        typer.Option(help="Minimum required metric improvement to promote challenger."),
    ] = 0.0,
    test_size: TestSizeOption = 0.2,
    validation_size: ValidationSizeOption = 0.2,
    seed: SeedOption = 42,
    estimator: Annotated[
        EstimatorType | None,
        typer.Option(help="Estimator type for challenger; defaults to champion's estimator."),
    ] = None,
    threshold_strategy: Annotated[
        ThresholdStrategy | None,
        typer.Option(help="Threshold strategy; defaults to champion's strategy."),
    ] = None,
    cost_policy: CostPolicyOption = "default",
    false_positive_cost: Annotated[
        float | None,
        typer.Option(help="False positive cost weight; defaults to champion's cost weight."),
    ] = None,
    false_negative_cost: Annotated[
        float | None,
        typer.Option(help="False negative cost weight; defaults to champion's cost weight."),
    ] = None,
    calibration_method: Annotated[
        CalibrationMethod | None,
        typer.Option(help="Calibration method; defaults to champion's calibration method."),
    ] = None,
    calibration_folds: CalibrationFoldsOption = 3,
    calibration_jobs: CalibrationJobsOption = 1,
    split_strategy: Annotated[
        SplitStrategy | None,
        typer.Option(help="Split strategy; defaults to champion's split strategy or temporal."),
    ] = None,
    time_column: TimeColumnOption = "Time",
    temporal_gap: TemporalGapOption = 0.0,
    report_output: Annotated[
        Path | None,
        typer.Option("--report-output", "-r", help="Optional JSON path for retraining report."),
    ] = None,
    promote: Annotated[
        bool,
        typer.Option(
            help=(
                "Automatically promote challenger (saving to champion destination) "
                "if criteria pass."
            )
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace an existing challenger output."),
    ] = False,
    fail_on_rejection: Annotated[
        bool,
        typer.Option(help="Exit with non-zero code if challenger fails promotion criteria."),
    ] = False,
) -> None:
    """Train challenger, compare with champion on test data, and decide promotion."""
    valid_metrics = {"auprc", "f1", "expected_cost"}
    normalized_metric = metric.lower().strip()
    if normalized_metric not in valid_metrics:
        _abort(f"Invalid evaluation metric '{metric}'; must be one of {sorted(valid_metrics)}.")

    output_has_content = output.is_file() or (
        output.is_dir() and next(output.iterdir(), None) is not None
    )
    if output_has_content and not overwrite:
        _abort(f"Output already exists: {output}. Pass --overwrite to replace it.")
    _guard_output(report_output, overwrite)

    try:
        champion_model = load_model(champion)
        champ_meta = champion_model.metadata
        champ_tc = champ_meta.get("training_config", {})

        # Resolve configuration defaults from champion metadata
        champ_est_raw = champ_tc.get("estimator")
        if champ_est_raw is not None:
            try:
                default_est = EstimatorType(champ_est_raw)
            except ValueError:
                default_est = EstimatorType.LOGISTIC_REGRESSION
        else:
            est_desc = str(champ_meta.get("estimator", "")).lower()
            if "forest" in est_desc:
                default_est = EstimatorType.RANDOM_FOREST
            elif "boost" in est_desc:
                default_est = EstimatorType.HIST_GRADIENT_BOOSTING
            else:
                default_est = EstimatorType.LOGISTIC_REGRESSION
        resolved_estimator = estimator if estimator is not None else default_est

        champ_thresh_raw = champ_tc.get("threshold_strategy")
        try:
            default_thresh = (
                ThresholdStrategy(champ_thresh_raw)
                if champ_thresh_raw is not None
                else ThresholdStrategy.F1
            )
        except ValueError:
            default_thresh = ThresholdStrategy.F1
        resolved_thresh_strat = (
            threshold_strategy if threshold_strategy is not None else default_thresh
        )

        champ_calib_raw = champ_tc.get("calibration_method")
        try:
            default_calib = (
                CalibrationMethod(champ_calib_raw)
                if champ_calib_raw is not None
                else CalibrationMethod.SIGMOID
            )
        except ValueError:
            default_calib = CalibrationMethod.SIGMOID
        resolved_calib_method = (
            calibration_method if calibration_method is not None else default_calib
        )

        champ_split_raw = champ_tc.get("split_strategy")
        try:
            default_split = (
                SplitStrategy(champ_split_raw)
                if champ_split_raw is not None
                else SplitStrategy.STRATIFIED
            )
        except ValueError:
            default_split = SplitStrategy.STRATIFIED
        resolved_split_strat = split_strategy if split_strategy is not None else default_split

        resolved_fp_cost = (
            false_positive_cost
            if false_positive_cost is not None
            else float(champ_tc.get("false_positive_cost", 1.0))
        )
        resolved_fn_cost = (
            false_negative_cost
            if false_negative_cost is not None
            else float(champ_tc.get("false_negative_cost", 10.0))
        )
        resolved_max_iter = int(champ_tc.get("max_iterations", 1_000))
        resolved_regularization = float(champ_tc.get("regularization", 1.0))
        resolved_n_est = int(champ_tc.get("n_estimators", 100))
        raw_depth = champ_tc.get("max_depth")
        resolved_max_depth = int(raw_depth) if raw_depth is not None else None
        resolved_lr = float(champ_tc.get("learning_rate", 0.1))
        resolved_l2 = float(champ_tc.get("l2_regularization", 0.0))
        resolved_max_bins = int(champ_tc.get("max_bins", 255))

        dataset = load_csv(data, target_column=target)
        config = _training_config(
            test_size=test_size,
            validation_size=validation_size,
            random_state=seed,
            estimator=resolved_estimator,
            max_iterations=resolved_max_iter,
            regularization=resolved_regularization,
            n_estimators=resolved_n_est,
            max_depth=resolved_max_depth,
            learning_rate=resolved_lr,
            l2_regularization=resolved_l2,
            max_bins=resolved_max_bins,
            threshold_strategy=resolved_thresh_strat,
            cost_policy=cost_policy,
            false_positive_cost=resolved_fp_cost,
            false_negative_cost=resolved_fn_cost,
            calibration_method=resolved_calib_method,
            calibration_folds=calibration_folds,
            calibration_jobs=calibration_jobs,
            split_strategy=resolved_split_strat,
            time_column=time_column,
            temporal_gap=temporal_gap,
        )

        git_info = _git_info()
        provenance = {
            key: git_info[source]
            for key, source in (
                ("git_commit", "commit"),
                ("git_repository", "repository"),
            )
            if source in git_info
        }

        # Train challenger model
        challenger_model = train_model(dataset, config=config, provenance=provenance or None)
        challenger_path = save_model(challenger_model, output)

        # Get exact test split to evaluate both models on identical data
        (
            _,
            _,
            features_test,
            _,
            _,
            target_test,
        ) = split_dataset(dataset, config)
        y_test = target_test.to_numpy()

        # Evaluate challenger on test split
        chall_probs = challenger_model.predict_probabilities(features_test)
        chall_metrics = evaluate_predictions(
            y_test,
            chall_probs,
            threshold=challenger_model.threshold,
        )
        chall_cost = expected_classification_cost(
            y_test,
            chall_probs,
            threshold=challenger_model.threshold,
            false_positive_cost=resolved_fp_cost,
            false_negative_cost=resolved_fn_cost,
        )

        # Evaluate champion on test split
        champ_probs = champion_model.predict_probabilities(features_test)
        champ_metrics = evaluate_predictions(
            y_test,
            champ_probs,
            threshold=champion_model.threshold,
        )
        champ_cost = expected_classification_cost(
            y_test,
            champ_probs,
            threshold=champion_model.threshold,
            false_positive_cost=resolved_fp_cost,
            false_negative_cost=resolved_fn_cost,
        )

        # Compare primary metric
        if normalized_metric == "auprc":
            champ_metric_val = champ_metrics.average_precision
            chall_metric_val = chall_metrics.average_precision
            metric_gain = chall_metric_val - champ_metric_val
            meets_gain = metric_gain >= min_gain
        elif normalized_metric == "f1":
            champ_metric_val = champ_metrics.f1
            chall_metric_val = chall_metrics.f1
            metric_gain = chall_metric_val - champ_metric_val
            meets_gain = metric_gain >= min_gain
        else:  # expected_cost
            champ_metric_val = champ_cost
            chall_metric_val = chall_cost
            metric_gain = champ_metric_val - chall_metric_val
            meets_gain = metric_gain >= min_gain

        # Guardrail: ensure challenger recall has not collapsed
        guardrail_passed = True
        guardrail_reason: str | None = None
        if chall_metrics.recall < 0.05 and champ_metrics.recall >= 0.05:
            guardrail_passed = False
            guardrail_reason = (
                f"Challenger recall ({chall_metrics.recall:.4f}) collapsed below 0.05 guardrail."
            )

        is_promoted = meets_gain and guardrail_passed
        decision = "PROMOTED" if is_promoted else "REJECTED"
        if not meets_gain:
            decision_reason = (
                f"Challenger {normalized_metric} ({chall_metric_val:.4f}) failed to achieve "
                f"minimum gain {min_gain:+.4f} against champion ({champ_metric_val:.4f}); "
                f"actual gain: {metric_gain:+.4f}."
            )
        elif not guardrail_passed:
            decision_reason = f"Challenger failed guardrail: {guardrail_reason}"
        else:
            decision_reason = (
                f"Challenger {normalized_metric} ({chall_metric_val:.4f}) exceeded champion "
                f"({champ_metric_val:.4f}) with gain {metric_gain:+.4f} "
                f"(required: >={min_gain:+.4f})."
            )

        promoted_to_champion = False
        if is_promoted and promote and champion.is_dir():
            save_model(challenger_model, champion)
            promoted_to_champion = True

        champ_metrics_dict = champ_metrics.to_dict()
        champ_metrics_dict["expected_cost_per_transaction"] = champ_cost
        chall_metrics_dict = chall_metrics.to_dict()
        chall_metrics_dict["expected_cost_per_transaction"] = chall_cost

        report: dict[str, Any] = {
            "decision": decision,
            "decision_reason": decision_reason,
            "primary_metric": normalized_metric,
            "min_gain_required": min_gain,
            "metric_gain": round(metric_gain, 6),
            "promoted_to_champion": promoted_to_champion,
            "champion": {
                "path": str(champion),
                "model_version": str(champ_meta.get("dataset_fingerprint", ""))[:12],
                "estimator": champ_meta.get("estimator"),
                "threshold": champion_model.threshold,
                "test_metrics": champ_metrics_dict,
            },
            "challenger": {
                "path": str(challenger_path),
                "model_version": str(challenger_model.metadata.get("dataset_fingerprint", ""))[:12],
                "estimator": challenger_model.metadata.get("estimator"),
                "threshold": challenger_model.threshold,
                "test_metrics": chall_metrics_dict,
            },
            "dataset": {
                "path": str(data),
                "total_rows": len(dataset.target),
                "test_rows": len(y_test),
                "split_strategy": config.split_strategy.value,
                "temporal_gap": config.temporal_gap,
            },
        }

        report_json = json.dumps(report, indent=2, sort_keys=True)
        if report_output is not None:
            _atomic_write_text(report_json + "\n", report_output)

        typer.echo(report_json)

        if fail_on_rejection and not is_promoted:
            raise typer.Exit(code=1)

    except (OSError, DataValidationError, ModelArtifactError, ValueError) as exc:
        _abort(str(exc))


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
    temporal_gap: TemporalGapOption = 0.0,
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
                    temporal_gap=temporal_gap,
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


def _generate_llm_explanations(
    *,
    model: FraudModel,
    features: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    provider: ExplanationProvider | None = None,
) -> list[str]:
    """Generate natural-language explanations from model effects via provider."""
    return model.explain_local_natural_language(
        features,
        probabilities,
        threshold=threshold,
        provider=provider,
    )


@app.command("export-edge")
def export_edge_command(
    model_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Model file or artifact directory."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Edge JSON artifact destination."),
    ] = Path("edge-model.json"),
    validation_data: Annotated[
        Path | None,
        typer.Option(
            "--validation-data",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Optional CSV used to measure source-model probability error.",
        ),
    ] = None,
    target: Annotated[
        str,
        typer.Option(help="Optional label column to exclude from validation features."),
    ] = DEFAULT_TARGET,
    prune_epsilon: Annotated[
        float,
        typer.Option(min=0.0, help="Zero coefficients whose absolute value is below this value."),
    ] = 0.0,
    max_error: Annotated[
        float,
        typer.Option(min=0.0, help="Maximum allowed validation probability error."),
    ] = 0.01,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing edge artifact.")] = False,
) -> None:
    """Export a supported model to the dependency-light int8 edge runtime."""
    _guard_output(output, overwrite)
    try:
        model = load_model(model_path)
        edge_model = build_edge_model(model, prune_epsilon=prune_epsilon)
        validation: dict[str, Any] | None = None
        if validation_data is not None:
            frame = pd.read_csv(validation_data)
            features = frame.drop(columns=target, errors="ignore")
            source_probabilities = model.predict_probabilities(features)
            records = [
                {str(key): value for key, value in record.items()}
                for record in features.to_dict(orient="records")
            ]
            edge_probabilities = edge_model.score_records(records)
            errors = np.abs(source_probabilities - np.asarray(edge_probabilities))
            maximum_error = float(errors.max())
            mean_error = float(errors.mean())
            validation = {
                "rows": len(features),
                "max_absolute_probability_error": maximum_error,
                "mean_absolute_probability_error": mean_error,
                "max_error_tolerance": max_error,
            }
            if maximum_error > max_error:
                _abort(
                    "Edge export exceeds the probability error tolerance: "
                    f"max={maximum_error:.6f}, tolerance={max_error:.6f}."
                )

        payload = edge_model.to_dict()
        if validation is not None:
            payload["validation"] = validation
        _atomic_write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", output)
    except (
        EdgeArtifactError,
        ModelArtifactError,
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
    ) as exc:
        _abort(str(exc))

    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "feature_count": len(edge_model.feature_names),
                "quantization_bits": edge_model.quantization_bits,
                "pruned_features": list(edge_model.pruned_features),
                "validation": validation,
            }
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
    explain_llm: Annotated[
        bool,
        typer.Option(help="Include per-transaction LLM-generated natural language explanations."),
    ] = False,
    audit_log: Annotated[
        Path | None,
        typer.Option(
            "--audit-log",
            "-a",
            help="Optional JSONL destination for structured scoring audit events.",
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
        if explain_llm:
            llm_explanations = _generate_llm_explanations(
                model=model,
                features=features,
                probabilities=probabilities,
                threshold=applied_threshold,
            )
            for idx, explanation in enumerate(llm_explanations):
                scored.loc[scored.index[idx], "llm_explanation"] = explanation
        _atomic_write_csv(scored, output)

        if audit_log is not None:
            sink = JsonlAuditSink(audit_log)
            try:
                scoring_event = build_scoring_audit_event(
                    model_version=str(model.metadata.get("dataset_fingerprint", ""))[:12],
                    dataset_fingerprint=str(model.metadata.get("dataset_fingerprint", "")),
                    threshold=applied_threshold,
                    features=features.to_dict(orient="records"),
                    predictions=[
                        {
                            "index": int(idx),
                            "fraud_probability": float(prob),
                            "is_fraud": bool(pred),
                        }
                        for idx, prob, pred in zip(
                            scored.index,
                            probabilities,
                            predictions,
                            strict=True,
                        )
                    ],
                    client_metadata={"source_file": str(data), "output_file": str(output)},
                )
                sink.emit(scoring_event)
            finally:
                sink.close()
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
    _guard_output(output, overwrite)

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

    payload = _benchmark_payload(model, features, sizes, repeat)
    report_json = json.dumps(
        {
            "model_version": str(model.metadata["dataset_fingerprint"])[:12],
            **payload,
        },
        indent=2,
    )
    try:
        _emit_report(report_json, output)
    except OSError as exc:
        _abort(str(exc))


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


@app.command("replay-audit")
def replay_audit_command(
    audit_log: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Audit log JSONL file."),
    ],
    model_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Model file or artifact directory."),
    ],
    data: Annotated[
        Path | None,
        typer.Option(
            "--data",
            "-d",
            exists=True,
            readable=True,
            help="Optional CSV containing original transactions if features are not inline.",
        ),
    ] = None,
    threshold: Annotated[
        float | None,
        typer.Option(help="Override scoring decision threshold during replay."),
    ] = None,
    tolerance: Annotated[
        float,
        typer.Option(min=0.0, help="Score difference tolerance before flagging discrepancy."),
    ] = 1e-4,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional JSON destination for replay report."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace an existing output file."),
    ] = False,
    fail_on_divergence: Annotated[
        bool,
        typer.Option(help="Exit with non-zero code if scores or decisions diverge."),
    ] = False,
) -> None:
    """Replay scoring events from a JSONL audit log against a model to detect score divergence."""
    _guard_output(output, overwrite)

    try:
        report = replay_audit_log(
            audit_log,
            model_path,
            data_path=data,
            threshold=threshold,
            tolerance=tolerance,
        )
    except (FileNotFoundError, ModelArtifactError, OSError, ValueError) as exc:
        _abort(str(exc))

    report_dict = report.to_dict()
    if output is not None:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report_dict, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            _abort(f"Failed to write replay report: {exc}")

    typer.echo(json.dumps(report_dict, indent=2, sort_keys=True))

    if fail_on_divergence and report.status == "DIVERGENT":
        raise typer.Exit(code=1)


@app.command("generate-signing-key")
def generate_signing_key_command(
    private_key: Annotated[
        Path,
        typer.Option("--private-key", "-k", help="Private Ed25519 PEM output path."),
    ] = Path("signing-private.pem"),
    public_key: Annotated[
        Path,
        typer.Option("--public-key", "-p", help="Public Ed25519 PEM output path."),
    ] = Path("signing-public.pem"),
    overwrite: Annotated[bool, typer.Option(help="Replace existing key files.")] = False,
) -> None:
    """Generate an Ed25519 keypair for signing and verifying attestations."""
    _guard_output(private_key, overwrite)
    _guard_output(public_key, overwrite)
    try:
        fingerprint = write_keypair(private_key, public_key, overwrite=overwrite)
    except SigningError as exc:
        _abort(str(exc))
    typer.echo(
        json.dumps(
            {
                "private_key": str(private_key),
                "public_key": str(public_key),
                "public_key_fingerprint": fingerprint,
            }
        )
    )


@app.command("generate-trust-bundle")
def generate_trust_bundle_command(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Trust-bundle JSON output path."),
    ] = Path("trust-bundle.json"),
    public_keys: Annotated[
        list[Path] | None,
        typer.Option(
            "--public-key",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Public Ed25519 PEM key; repeat for rotation overlap.",
        ),
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing trust bundle.")] = False,
) -> None:
    """Create a validated trust bundle from one or more public keys."""
    selected_public_keys = public_keys or []
    if not selected_public_keys:
        _abort("At least one --public-key is required.")
    _guard_output(output, overwrite)
    try:
        bundle = TrustBundle.from_public_keys(load_public_keys(tuple(selected_public_keys)))
        write_trust_bundle(output, bundle, overwrite=overwrite)
    except TrustBundleError as exc:
        _abort(str(exc))
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "active_key_ids": list(bundle.active_key_ids()),
                "revoked_key_ids": [],
            }
        )
    )


@app.command("rotate-trust-bundle")
def rotate_trust_bundle_command(
    bundle_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Existing trust bundle."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Rotated trust-bundle JSON output path."),
    ] = Path("trust-bundle.json"),
    add_public_keys: Annotated[
        list[Path] | None,
        typer.Option(
            "--add-public-key",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Public key to add as active; repeat for multiple keys.",
        ),
    ] = None,
    revoke_key_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--revoke-key-id",
            help="Existing active key ID to revoke; repeat for multiple keys.",
        ),
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing trust bundle.")] = False,
) -> None:
    """Rotate a trust bundle by adding and/or revoking verification keys."""
    selected_additions = add_public_keys or []
    selected_revocations = revoke_key_ids or []
    if not selected_additions and not selected_revocations:
        _abort("Provide --add-public-key or --revoke-key-id.")
    _guard_output(output, overwrite)
    try:
        bundle = load_trust_bundle(bundle_path)
        rotated = bundle.rotate(
            additions=load_public_keys(tuple(selected_additions)),
            revoked_key_ids=tuple(selected_revocations),
        )
        write_trust_bundle(output, rotated, overwrite=overwrite)
    except TrustBundleError as exc:
        _abort(str(exc))
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "active_key_ids": list(rotated.active_key_ids()),
                "revoked_key_ids": [key.key_id for key in rotated.keys if key.status == "revoked"],
            }
        )
    )


@app.command("validate-artifact")
def validate_artifact_command(
    model_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Model file or artifact directory."),
    ],
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Enforce strict requirements including Git provenance and directory manifest.",
        ),
    ] = False,
    attestation_output: Annotated[
        Path | None,
        typer.Option(
            "--attestation-output",
            "-a",
            help="Optional JSON destination for signed attestation manifest.",
        ),
    ] = None,
    signer: Annotated[
        str | None,
        typer.Option(help="Identifier or entity signing the verification attestation."),
    ] = None,
    signing_key: Annotated[
        Path | None,
        typer.Option(
            "--signing-key",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Ed25519 private PEM key used to sign the attestation.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace an existing attestation output file."),
    ] = False,
) -> None:
    """Validate artifact integrity, runtime compatibility, lineage, and report readiness."""
    _guard_output(attestation_output, overwrite)
    if signing_key is not None and attestation_output is None:
        _abort("--signing-key requires --attestation-output.")
    if signing_key is not None and not strict:
        _abort("--signing-key requires --strict validation.")
    private_key = None
    if signing_key is not None:
        try:
            private_key = load_private_key(signing_key)
        except SigningError as exc:
            _abort(str(exc))
    report = validate_artifact(model_path, strict=strict)

    if attestation_output is not None:
        attestation = report.to_attestation(signer=signer, signing_key=private_key)
        _atomic_write_text(
            json.dumps(attestation, indent=2, sort_keys=True) + "\n",
            attestation_output,
        )

    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if not report.valid:
        raise typer.Exit(code=1)


@app.command("verify-attestation")
def verify_attestation_command(
    attestation: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Attestation JSON file."),
    ],
    public_key: Annotated[
        Path | None,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Trusted public PEM key; use --trust-bundle for rotation-aware admission.",
        ),
    ] = None,
    trust_bundle: Annotated[
        Path | None,
        typer.Option(
            "--trust-bundle",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Rotation-aware JSON trust bundle; mutually exclusive with public key.",
        ),
    ] = None,
    require_signature: Annotated[
        bool,
        typer.Option("--require-signature/--allow-unsigned", help="Require an Ed25519 signature."),
    ] = True,
    require_passed: Annotated[
        bool,
        typer.Option("--require-passed/--allow-failed", help="Require PASSED artifact status."),
    ] = True,
) -> None:
    """Verify attestation integrity, signature authenticity, and admission status."""
    if (public_key is None) == (trust_bundle is None):
        _abort("Provide exactly one trusted public key argument or --trust-bundle.")
    try:
        payload = json.loads(attestation.read_text(encoding="utf-8"))
        trusted_key = load_public_key(public_key) if public_key is not None else None
        loaded_bundle = load_trust_bundle(trust_bundle) if trust_bundle is not None else None
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        SigningError,
        TrustBundleError,
    ) as exc:
        typer.echo(json.dumps({"valid": False, "message": str(exc)}, sort_keys=True))
        raise typer.Exit(code=1) from exc

    if not isinstance(payload, dict):
        result = {"valid": False, "message": "Attestation root must be a JSON object."}
        typer.echo(json.dumps(result, sort_keys=True))
        raise typer.Exit(code=1)

    digest_valid, digest_message = verify_attestation(payload)
    status = payload.get("status")
    status_valid = not require_passed or status == "PASSED"
    signature = payload.get("signature")
    signature_key_id = signature.get("key_id") if isinstance(signature, dict) else None
    signature_key_status = "external"
    if signature is None and not require_signature:
        signature_valid, signature_message = True, "Unsigned attestation accepted."
    elif signature is None:
        signature_valid, signature_message = False, "Attestation has no signature."
    elif loaded_bundle is not None:
        signature_valid, signature_message, selected_key = verify_attestation_with_bundle(
            payload, loaded_bundle
        )
        if selected_key is not None:
            signature_key_status = selected_key.status
    else:
        if trusted_key is None:
            _abort("A trusted public key is required for single-key verification.")
        signature_valid, signature_message = verify_attestation_signature(payload, trusted_key)

    valid = digest_valid and status_valid and signature_valid
    result = {
        "valid": valid,
        "status": {"value": status, "valid": status_valid},
        "digest": {"message": digest_message, "valid": digest_valid},
        "signature": {
            "key_id": signature_key_id,
            "key_status": signature_key_status,
            "message": signature_message,
            "trust_source": "trust_bundle" if loaded_bundle is not None else "public_key",
            "valid": signature_valid,
        },
    }
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
    if not valid:
        raise typer.Exit(code=1)


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
    fail_on: Annotated[
        str | None,
        typer.Option(
            help="Exit 1 when the overall status reaches 'warning' or 'drifted', "
            "for scheduled surveillance."
        ),
    ] = None,
    webhook_slack: Annotated[
        str | None,
        typer.Option(help="Slack webhook URL for drift alerts."),
    ] = None,
    webhook_pagerduty: Annotated[
        str | None,
        typer.Option(help="PagerDuty Events API v2 integration key for drift alerts."),
    ] = None,
) -> None:
    """Compare current feature distributions with the training baseline."""
    _guard_output(output, overwrite)

    try:
        model = load_model(model_path)
        frame = pd.read_csv(data).drop(columns=target, errors="ignore")
        features = model.validate_features(frame)
        profile = model.metadata.get("reference_profile")
        if not isinstance(profile, dict):
            raise DriftError("Model artifact does not contain a reference profile.")
        report = assess_drift(profile, features, thresholds=model.metadata.get("drift_thresholds"))
        report_json = json.dumps(report.to_dict(), indent=2)
        _emit_report(report_json, output)
        tripped = surveillance_tripped(report.overall_status, fail_on)
        if tripped and (webhook_slack or webhook_pagerduty):
            _send_drift_alert(
                report=report,
                webhook_slack=webhook_slack,
                webhook_pagerduty=webhook_pagerduty,
                model_version=str(model.metadata["dataset_fingerprint"])[:12],
            )
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ModelArtifactError,
        DriftError,
    ) as exc:
        _abort(str(exc))

    if tripped:
        typer.echo(
            f"Drift surveillance tripped: overall_status={report.overall_status} "
            f"meets --fail-on {fail_on}.",
            err=True,
        )
        raise typer.Exit(code=1)


def _send_drift_alert(
    *,
    report: DriftReport,
    webhook_slack: str | None,
    webhook_pagerduty: str | None,
    model_version: str,
) -> None:
    """Send drift alert to configured webhooks (Slack and/or PagerDuty)."""

    # Build alert payload
    alert_payload: dict[str, Any] = {
        "model_version": model_version,
        "overall_status": report.overall_status,
        "mean_psi": report.mean_psi,
        "max_psi": report.max_psi,
        "drifted_features": [
            {"feature": f.feature, "psi": f.psi, "status": f.status}
            for f in report.features
            if f.status in ("warning", "drifted")
        ],
    }

    # Send to Slack
    if webhook_slack:
        slack_payload = {
            "text": f"Drift Alert: Model {model_version} - {report.overall_status.upper()}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"Drift Alert: {report.overall_status.upper()}",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Model:* {model_version}"},
                        {"type": "mrkdwn", "text": f"*Status:* {report.overall_status}"},
                        {"type": "mrkdwn", "text": f"*Mean PSI:* {report.mean_psi:.4f}"},
                        {"type": "mrkdwn", "text": f"*Max PSI:* {report.max_psi:.4f}"},
                    ],
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Drifted Features:\n"
                        + "\n".join(
                            f"• {f['feature']}: PSI={f['psi']:.4f} ({f['status']})"
                            for f in alert_payload["drifted_features"][:5]
                        ),
                    },
                },
            ],
        }
        _post_webhook(webhook_slack, slack_payload)

    # Send to PagerDuty
    if webhook_pagerduty:
        # PagerDuty Events API v2
        has_drifted = any(f["status"] == "drifted" for f in alert_payload["drifted_features"])
        severity = "critical" if has_drifted else "warning"
        pd_payload: dict[str, Any] = {
            "routing_key": webhook_pagerduty,
            "event_action": "trigger",
            "payload": {
                "summary": f"Drift Alert: Model {model_version} - {report.overall_status.upper()}",
                "source": "fraud-detection",
                "severity": severity,
                "custom_details": alert_payload,
            },
        }
        _post_webhook("https://events.pagerduty.com/v2/enqueue", pd_payload)


def _post_webhook(url: str, payload: dict[str, Any]) -> None:
    """Post JSON payload to a webhook URL with error handling."""
    req = urllib.request.Request(  # noqa: S310
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req, timeout=10)  # noqa: S310
    except urllib.error.URLError as exc:
        typer.echo(f"Warning: Failed to send webhook: {exc}", err=True)


@app.command("multi-window-drift")
def multi_window_drift_command(
    model_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Model file or artifact directory."),
    ],
    data: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Current transactions."),
    ],
    short_window_rows: Annotated[
        int,
        typer.Option(min=2, help="Number of recent rows for the short surveillance window."),
    ] = 100,
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
    fail_on: Annotated[
        str | None,
        typer.Option(
            help="Exit 1 when the overall status reaches 'warning' or 'drifted', "
            "for scheduled surveillance."
        ),
    ] = None,
    webhook_slack: Annotated[
        str | None,
        typer.Option(help="Slack webhook URL for drift alerts."),
    ] = None,
    webhook_pagerduty: Annotated[
        str | None,
        typer.Option(help="PagerDuty Events API v2 integration key for drift alerts."),
    ] = None,
) -> None:
    """Perform dual-window drift surveillance comparing recent and aggregate features."""
    _guard_output(output, overwrite)

    try:
        model = load_model(model_path)
        frame = pd.read_csv(data).drop(columns=target, errors="ignore")
        features = model.validate_features(frame)
        profile = model.metadata.get("reference_profile")
        if not isinstance(profile, dict):
            raise DriftError("Model artifact does not contain a reference profile.")
        report = assess_multi_window_drift(
            profile,
            features,
            short_window_rows=short_window_rows,
            thresholds=model.metadata.get("drift_thresholds"),
        )
        report_json = json.dumps(report.to_dict(), indent=2)
        _emit_report(report_json, output)
        tripped = surveillance_tripped(report.overall_status, fail_on)
        if tripped and (webhook_slack or webhook_pagerduty):
            _send_multi_window_drift_alert(
                report=report,
                webhook_slack=webhook_slack,
                webhook_pagerduty=webhook_pagerduty,
                model_version=str(model.metadata["dataset_fingerprint"])[:12],
            )
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ModelArtifactError,
        DriftError,
    ) as exc:
        _abort(str(exc))

    if tripped:
        typer.echo(
            f"Multi-window drift surveillance tripped: overall_status={report.overall_status} "
            f"meets --fail-on {fail_on}.",
            err=True,
        )
        raise typer.Exit(code=1)


def _send_multi_window_drift_alert(
    *,
    report: MultiWindowDriftReport,
    webhook_slack: str | None,
    webhook_pagerduty: str | None,
    model_version: str,
) -> None:
    """Send multi-window drift alert to configured webhooks (Slack and/or PagerDuty)."""
    alert_payload: dict[str, Any] = {
        "model_version": model_version,
        "overall_status": report.overall_status,
        "short_window_rows": report.short_window_rows,
        "long_window_rows": report.long_window_rows,
        "mean_short_psi": report.mean_short_psi,
        "max_short_psi": report.max_short_psi,
        "mean_long_psi": report.mean_long_psi,
        "max_long_psi": report.max_long_psi,
        "max_velocity": report.max_velocity,
        "drifted_features": [
            {
                "feature": f.feature,
                "short_psi": f.short_psi,
                "long_psi": f.long_psi,
                "velocity": f.velocity,
                "status": f.status,
            }
            for f in report.features
            if f.status in ("warning", "drifted")
        ],
    }

    if webhook_slack:
        slack_payload = {
            "text": (
                f"Multi-Window Drift Alert: Model {model_version} - {report.overall_status.upper()}"
            ),
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"Multi-Window Drift Alert: {report.overall_status.upper()}",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Model:* {model_version}"},
                        {"type": "mrkdwn", "text": f"*Status:* {report.overall_status}"},
                        {"type": "mrkdwn", "text": f"*Max Short PSI:* {report.max_short_psi:.4f}"},
                        {"type": "mrkdwn", "text": f"*Max Long PSI:* {report.max_long_psi:.4f}"},
                        {"type": "mrkdwn", "text": f"*Max Velocity:* {report.max_velocity:.4f}"},
                        {
                            "type": "mrkdwn",
                            "text": (
                                f"*Windows:* {report.short_window_rows}/"
                                f"{report.long_window_rows} rows"
                            ),
                        },
                    ],
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Drifted Features:\n"
                        + "\n".join(
                            f"• {f['feature']}: Short PSI={f['short_psi']:.4f}, "
                            f"Long PSI={f['long_psi']:.4f}, Vel={f['velocity']:+.4f} "
                            f"({f['status']})"
                            for f in alert_payload["drifted_features"][:5]
                        ),
                    },
                },
            ],
        }
        _post_webhook(webhook_slack, slack_payload)

    if webhook_pagerduty:
        has_drifted = any(f["status"] == "drifted" for f in alert_payload["drifted_features"])
        severity = "critical" if has_drifted else "warning"
        pd_payload: dict[str, Any] = {
            "routing_key": webhook_pagerduty,
            "event_action": "trigger",
            "payload": {
                "summary": (
                    f"Multi-Window Drift Alert: Model {model_version} - "
                    f"{report.overall_status.upper()}"
                ),
                "source": "fraud-detection",
                "severity": severity,
                "custom_details": alert_payload,
            },
        }
        _post_webhook("https://events.pagerduty.com/v2/enqueue", pd_payload)


@app.command("stream-profile")
def stream_profile_command(
    data: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="Streaming CSV or JSONL audit file."
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination JSON path for reference profile."),
    ],
    model_path: Annotated[
        Path | None,
        typer.Option(
            help="Optional trained model artifact to supply initial reference profile edges."
        ),
    ] = None,
    state_input: Annotated[
        Path | None,
        typer.Option(help="Optional saved StreamingProfile JSON checkpoint to resume from."),
    ] = None,
    checkpoint_output: Annotated[
        Path | None,
        typer.Option(help="Optional destination path to save updated StreamingProfile state."),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(min=1, help="Number of rows per streaming update chunk."),
    ] = 500,
    target: Annotated[
        str,
        typer.Option(help="Optional label column to exclude from feature profiling."),
    ] = DEFAULT_TARGET,
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace existing destination files."),
    ] = False,
) -> None:
    """Incrementally profile feature distributions from streaming CSV batches or JSONL logs."""
    _guard_output(output, overwrite)
    if checkpoint_output is not None:
        _guard_output(checkpoint_output, overwrite)

    try:
        profiler: StreamingProfile | None = None
        if state_input is not None:
            if not state_input.is_file():
                _abort(f"State input file not found: {state_input}")
            state_data = json.loads(state_input.read_text(encoding="utf-8"))
            profiler = StreamingProfile.from_dict(state_data)
        elif model_path is not None:
            model = load_model(model_path)
            profile = model.metadata.get("reference_profile")
            if not isinstance(profile, dict):
                raise DriftError("Model artifact does not contain a reference profile.")
            profiler = StreamingProfile.from_reference_profile(profile)

        total_rows_processed = 0
        is_jsonl = data.suffix.lower() == ".jsonl"

        if is_jsonl:
            batch_records: list[dict[str, Any]] = []
            with data.open("r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        event = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    payload = event.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    features_list = payload.get("features")
                    if isinstance(features_list, list):
                        for item in features_list:
                            if isinstance(item, dict):
                                batch_records.append(item)
                                if len(batch_records) >= batch_size:
                                    if profiler is None:
                                        init_frame = pd.DataFrame(batch_records).drop(
                                            columns=target, errors="ignore"
                                        )
                                        ref = build_reference_profile(init_frame)
                                        profiler = StreamingProfile.from_reference_profile(ref)
                                    profiler.update(batch_records)
                                    total_rows_processed += len(batch_records)
                                    batch_records = []
            if batch_records:
                if profiler is None:
                    init_frame = pd.DataFrame(batch_records).drop(columns=target, errors="ignore")
                    ref = build_reference_profile(init_frame)
                    profiler = StreamingProfile.from_reference_profile(ref)
                profiler.update(batch_records)
                total_rows_processed += len(batch_records)
        else:
            reader = pd.read_csv(data, chunksize=batch_size)
            for chunk in reader:
                features_chunk = chunk.drop(columns=target, errors="ignore")
                if profiler is None:
                    ref = build_reference_profile(features_chunk)
                    profiler = StreamingProfile.from_reference_profile(ref)
                profiler.update(features_chunk)
                total_rows_processed += len(features_chunk)

        if profiler is None or total_rows_processed == 0:
            _abort("No valid transaction records found to profile.")

        ref_profile = profiler.to_reference_profile()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(ref_profile, indent=2), encoding="utf-8")

        if checkpoint_output is not None:
            checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_output.write_text(json.dumps(profiler.to_dict(), indent=2), encoding="utf-8")

        typer.echo(
            f"Successfully updated streaming profile for {len(ref_profile)} features "
            f"across {total_rows_processed} transactions."
        )
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ModelArtifactError,
        DriftError,
    ) as exc:
        _abort(str(exc))


@app.command("simulate-drift")
def simulate_drift_command(
    data: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Input CSV dataset to inject drift into.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination CSV file for drifted dataset."),
    ],
    features: Annotated[
        list[str] | None,
        typer.Option(help="Specific features to drift (repeat flag; defaults to all non-target)."),
    ] = None,
    mean_offset: Annotated[
        float,
        typer.Option(help="Additive shift applied to feature distributions."),
    ] = 0.0,
    variance_scale: Annotated[
        float,
        typer.Option(min=0.0, help="Multiplicative scale applied to standard deviations."),
    ] = 1.0,
    anomaly_fraction: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Fraction of rows to inject anomaly spikes into."),
    ] = 0.0,
    anomaly_scale: Annotated[
        float,
        typer.Option(min=0.0, help="Magnitude multiplier for injected anomaly spikes."),
    ] = 5.0,
    sample_fraction: Annotated[
        float,
        typer.Option(min=0.01, max=1.0, help="Fraction of recent tail rows to modify."),
    ] = 1.0,
    seed: Annotated[
        int,
        typer.Option(help="Random seed for repeatable noise and sampling."),
    ] = 42,
    target: Annotated[
        str,
        typer.Option(help="Optional label column to preserve without modification."),
    ] = DEFAULT_TARGET,
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace existing destination file."),
    ] = False,
) -> None:
    """Inject synthetic distribution shifts and anomalies for chaos and surveillance testing."""
    _guard_output(output, overwrite)

    try:
        frame = pd.read_csv(data)
        drifted = inject_drift(
            frame,
            target_features=features,
            mean_offset=mean_offset,
            variance_scale=variance_scale,
            anomaly_fraction=anomaly_fraction,
            anomaly_scale=anomaly_scale,
            sample_fraction=sample_fraction,
            target_column=target,
            random_state=seed,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        drifted.to_csv(output, index=False)
        num_drifted_features = (
            len(features) if features else len([c for c in frame.columns if c != target])
        )
        affected_rows = max(1, round(len(frame) * sample_fraction))
        typer.echo(
            f"Successfully injected drift into {num_drifted_features} features across "
            f"{affected_rows}/{len(frame)} rows. Saved to {output}."
        )
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        DataValidationError,
    ) as exc:
        _abort(str(exc))


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
    _guard_output(output, overwrite)

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
        _emit_report(report_json, output)
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
    _guard_output(output, overwrite)

    try:
        model = load_model(model_path)
        dataset = load_csv(data, target_column=target)
        candidates = _parse_thresholds(thresholds)
        policy_name, fp_cost, fn_cost = _resolve_report_costs(
            model, false_positive_cost, false_negative_cost
        )
        payload = _thresholds_payload(model, dataset, candidates, policy_name, fp_cost, fn_cost)
        report_json = json.dumps(
            {
                "model_version": str(model.metadata["dataset_fingerprint"])[:12],
                **payload,
            },
            indent=2,
        )
        _emit_report(report_json, output)
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


def _thresholds_payload(
    model: FraudModel,
    dataset: ValidatedDataset,
    candidates: list[float],
    policy_name: str,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict[str, Any]:
    """Score candidate thresholds plus the tuned operating point on labeled data."""
    probabilities = model.predict_probabilities(dataset.features)
    y_true = dataset.target.to_numpy()
    tradeoff = summarize_thresholds(
        y_true,
        probabilities,
        candidates,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    ).to_dict()
    at_tuned = evaluate_predictions(y_true, probabilities, threshold=model.threshold)
    tuned_cost = expected_classification_cost(
        y_true,
        probabilities,
        threshold=model.threshold,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
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
    return {
        "model_threshold": model.threshold,
        "cost_policy": {
            "name": policy_name,
            "false_positive_cost": false_positive_cost,
            "false_negative_cost": false_negative_cost,
        },
        "false_positive_cost": false_positive_cost,
        "false_negative_cost": false_negative_cost,
        "model_threshold_metrics": tuned_row,
        **tradeoff,
    }


def _benchmark_payload(
    model: FraudModel,
    features: pd.DataFrame,
    sizes: list[int],
    repeat: int,
) -> dict[str, Any]:
    """Time offline batch scoring and report latency plus throughput."""
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
    return {"rows_available": rows, "repeat": repeat, "results": results}


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


@app.command("promote")
def promote_command(
    model_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Model file or artifact directory."),
    ],
    heldout: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="Held-out labeled transactions."
        ),
    ],
    recent: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="Recent transactions for drift."
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
    bins: Annotated[
        int,
        typer.Option(min=2, max=20, help="Equal-width reliability bins."),
    ] = 10,
    batch_sizes: Annotated[
        str,
        typer.Option(help="Comma-separated scoring batch sizes, e.g. '1,10,100'."),
    ] = "1,10,100",
    repeat: Annotated[
        int,
        typer.Option(min=1, max=10, help="Timed runs per batch size; the median is reported."),
    ] = 3,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional JSON bundle destination."),
    ] = None,
    audit_log: Annotated[
        Path | None,
        typer.Option(
            "--audit-log",
            "-a",
            help="Optional JSONL destination for structured promotion audit events.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace an existing bundle file."),
    ] = False,
) -> None:
    """Assemble the promotion evidence bundle for a challenger model.

    Runs calibration, threshold, drift, and benchmark evidence against held-out
    labeled data and recent traffic, and packs the results with the model card
    summary into one reviewable document. Assessment only: nothing is trained
    and the bundle states facts, not a promotion verdict.
    """
    _guard_output(output, overwrite)

    try:
        model = load_model(model_path)
        labeled = load_csv(heldout, target_column=target)
        recent_features = model.validate_features(
            pd.read_csv(recent).drop(columns=target, errors="ignore")
        )
        candidates = _parse_thresholds(thresholds)
        policy_name, fp_cost, fn_cost = _resolve_report_costs(
            model, false_positive_cost, false_negative_cost
        )
        sizes = _parse_batch_sizes(batch_sizes)
        profile = model.metadata.get("reference_profile")
        if not isinstance(profile, dict):
            raise DriftError("Model artifact does not contain a reference profile.")
        bundle: dict[str, Any] = {
            "model_version": str(model.metadata["dataset_fingerprint"])[:12],
            "model": {
                "estimator": model.metadata["estimator"],
                "threshold": model.threshold,
                "cost_policy": model.metadata.get("cost_policy"),
                "created_at": model.metadata["created_at"],
                "test_metrics": model.metadata["test_metrics"],
            },
            "calibration": calibration_report(
                labeled.target.to_numpy(),
                model.predict_probabilities(labeled.features),
                bins=bins,
            ).to_dict(),
            "thresholds": _thresholds_payload(
                model, labeled, candidates, policy_name, fp_cost, fn_cost
            ),
            "drift": assess_drift(
                profile, recent_features, thresholds=model.metadata.get("drift_thresholds")
            ).to_dict(),
            "benchmark": _benchmark_payload(model, labeled.features, sizes, repeat),
        }
        _emit_report(json.dumps(bundle, indent=2), output)

        if audit_log is not None:
            sink = JsonlAuditSink(audit_log)
            try:
                promotion_event = build_promotion_audit_event(
                    model_version=str(model.metadata.get("dataset_fingerprint", ""))[:12],
                    dataset_fingerprint=str(model.metadata.get("dataset_fingerprint", "")),
                    bundle_summary={
                        "model_version": bundle["model_version"],
                        "estimator": bundle["model"]["estimator"],
                        "threshold": bundle["model"]["threshold"],
                        "test_metrics": bundle["model"]["test_metrics"],
                        "drift_detected": bundle["drift"].get("drift_detected", False),
                        "warning_detected": bundle["drift"].get("warning_detected", False),
                    },
                    metadata={"heldout_path": str(heldout), "recent_path": str(recent)},
                )
                sink.emit(promotion_event)
            finally:
                sink.close()
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ModelArtifactError,
        DataValidationError,
        DriftError,
        ValueError,
    ) as exc:
        _abort(str(exc))


@app.command("compliance")
def compliance_command(
    bundle: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Promotion bundle JSON produced by the promote command.",
        ),
    ],
    stability: Annotated[
        Path | None,
        typer.Option(
            "--stability",
            help="Optional stability report JSON to include in the HTML report.",
        ),
    ] = None,
    artifact: Annotated[
        Path | None,
        typer.Option(
            "--artifact",
            help="Optional model artifact directory or manifest JSON for integrity evidence.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional HTML report destination."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace an existing report file."),
    ] = False,
) -> None:
    """Render promotion evidence as a self-contained HTML compliance report."""
    _guard_output(output, overwrite)
    try:
        bundle_payload = _read_json_mapping(bundle, "promotion bundle")
        stability_payload = (
            _read_json_mapping(stability, "stability report") if stability is not None else None
        )
        manifest_payload = None
        if artifact is not None:
            manifest_path = artifact / MANIFEST_FILENAME if artifact.is_dir() else artifact
            manifest_payload = _read_json_mapping(manifest_path, "artifact manifest")
        report = render_compliance_report(
            bundle_payload,
            stability=stability_payload,
            manifest=manifest_payload,
        )
        if output is not None:
            _atomic_write_text(report, output)
        typer.echo(report)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ComplianceReportError) as exc:
        _abort(str(exc))


def _read_json_mapping(path: Path, description: str) -> dict[str, Any]:
    """Read a JSON object used as compliance evidence."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ComplianceReportError(f"Invalid {description} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ComplianceReportError(f"The {description} must be a JSON object.")
    return payload


@app.command("model-card")
def model_card_command(
    model_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Model file or artifact directory."),
    ],
    verbose: Annotated[
        bool,
        typer.Option(help="Include full model card with all metadata."),
    ] = False,
    git_info_flag: Annotated[
        bool,
        typer.Option(
            "--git-info/--no-git-info", help="Include Git commit information if available."
        ),
    ] = True,
) -> None:
    """Display model card with versioning, lineage, and provenance information.

    Shows content-addressable identifiers (SHA-256 of training data,
    hyperparameters, and code version), training configuration, and
    performance metrics. Optionally includes Git commit information.
    """
    try:
        model = load_model(model_path)
        metadata = model.metadata

        git_info = _git_info() if git_info_flag else {}

        # Build model card
        card: dict[str, Any] = {
            "model_version": str(metadata.get("dataset_fingerprint", ""))[:12],
            "dataset_fingerprint": metadata.get("dataset_fingerprint"),
            "lineage": metadata.get("lineage"),
            "created_at": metadata.get("created_at"),
            "estimator": metadata.get("estimator"),
            "threshold": metadata.get("threshold"),
            "artifact_version": metadata.get("artifact_version"),
            "training_config": metadata.get("training_config"),
            "cost_policy": metadata.get("cost_policy"),
            "drift_thresholds": metadata.get("drift_thresholds"),
            "splits": metadata.get("splits"),
            "split_time_ranges": metadata.get("split_time_ranges"),
            "test_metrics": metadata.get("test_metrics"),
            "validation_metrics": metadata.get("validation_metrics"),
            "feature_count": metadata.get("feature_count"),
            "row_count": metadata.get("row_count"),
            "fraud_count": metadata.get("fraud_count"),
            "fraud_rate": metadata.get("fraud_rate"),
            "feature_effects": metadata.get("feature_effects") if verbose else None,
            "scaler_mean": metadata.get("scaler_mean") if verbose else None,
            "scaler_scale": metadata.get("scaler_scale") if verbose else None,
        }

        if git_info:
            card["git"] = git_info

        if not verbose:
            # Compact view
            compact = {
                "model_version": card["model_version"],
                "dataset_fingerprint": card["dataset_fingerprint"],
                "lineage": card["lineage"],
                "estimator": card["estimator"],
                "threshold": card["threshold"],
                "created_at": card["created_at"],
                "test_roc_auc": card.get("test_metrics", {}).get("roc_auc"),
                "test_f1": card.get("test_metrics", {}).get("f1"),
            }
            if git_info:
                compact["git"] = git_info
            typer.echo(json.dumps(compact, indent=2, sort_keys=True))
        else:
            typer.echo(json.dumps(card, indent=2, sort_keys=True))

    except (
        OSError,
        ModelArtifactError,
        ValueError,
    ) as exc:
        _abort(str(exc))


@app.command("serve")
def serve_command(
    model_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Model file or artifact directory."),
    ],
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65_535, help="TCP port.")] = 8000,
    otlp_endpoint: Annotated[
        str | None,
        typer.Option(help="Optional OTLP/HTTP traces endpoint; env fallback is supported."),
    ] = None,
    otlp_service_name: Annotated[
        str | None,
        typer.Option(help="Service name included in optional OTLP spans."),
    ] = None,
    otlp_timeout_seconds: Annotated[
        float,
        typer.Option(min=0.01, help="Optional OTLP collector request timeout in seconds."),
    ] = 0.5,
    attestation: Annotated[
        Path | None,
        typer.Option(
            "--attestation",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Optional signed attestation required before serving.",
        ),
    ] = None,
    trust_bundle: Annotated[
        Path | None,
        typer.Option(
            "--trust-bundle",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Optional rotation-aware trust bundle paired with --attestation.",
        ),
    ] = None,
    api_key: Annotated[
        list[str] | None,
        typer.Option(
            "--api-key",
            help="Valid X-API-Key value; repeat for multiple keys. Env fallback is FRAUD_API_KEYS.",
        ),
    ] = None,
    rate_limit_requests: Annotated[
        int,
        typer.Option(
            min=0,
            help=(
                "Max requests per client IP per window; 0 disables. "
                "Env fallback is FRAUD_RATE_LIMIT_REQUESTS."
            ),
        ),
    ] = 0,
    rate_limit_window_seconds: Annotated[
        float,
        typer.Option(
            min=0.01,
            help=("Rate-limit window in seconds. Env fallback is FRAUD_RATE_LIMIT_WINDOW_SECONDS."),
        ),
    ] = 60.0,
    max_concurrent_scoring: Annotated[
        int,
        typer.Option(
            min=0,
            help=(
                "Max scoring requests running at once; 0 disables. "
                "Env fallback is FRAUD_MAX_CONCURRENT_SCORING."
            ),
        ),
    ] = 0,
) -> None:
    """Run the versioned HTTP prediction service with optional trace export."""
    from fraud_detection.api import create_app

    try:
        model = load_model(model_path)
    except ModelArtifactError as exc:
        _abort(str(exc))
    uvicorn.run(
        create_app(
            model=model,
            api_keys=api_key,
            rate_limit_requests=rate_limit_requests,
            rate_limit_window_seconds=rate_limit_window_seconds,
            max_concurrent_scoring=max_concurrent_scoring,
            otlp_endpoint=otlp_endpoint,
            otlp_service_name=otlp_service_name,
            otlp_timeout_seconds=otlp_timeout_seconds,
            attestation_path=attestation,
            trust_bundle_path=trust_bundle,
        ),
        host=host,
        port=port,
    )


def _abort(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=2)


def _guard_output(output: Path | None, overwrite: bool) -> None:
    """Refuse to replace an existing report file without explicit opt-in."""
    if output is not None and output.exists() and not overwrite:
        _abort(f"Output already exists: {output}. Pass --overwrite to replace it.")


def _emit_report(report_json: str, output: Path | None) -> None:
    """Atomically persist a JSON report when requested, then print it."""
    if output is not None:
        _atomic_write_text(report_json + "\n", output)
    typer.echo(report_json)


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
