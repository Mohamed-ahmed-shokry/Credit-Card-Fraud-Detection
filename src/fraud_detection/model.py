"""Leakage-safe model training, prediction, and artifact persistence."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from importlib.metadata import version as distribution_version
from pathlib import Path
from platform import python_version
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection import __version__
from fraud_detection.data import ValidatedDataset
from fraud_detection.drift import build_reference_profile, default_thresholds
from fraud_detection.evaluation import (
    evaluate_predictions,
    expected_classification_cost,
    select_cost_threshold,
    select_f1_threshold,
)
from fraud_detection.explanations import (
    ExplanationProvider,
    ExplanationRequest,
    TemplateExplanationProvider,
)

ARTIFACT_VERSION = 2
MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
MANIFEST_FILENAME = "manifest.json"


class ModelArtifactError(ValueError):
    """Raised when a model artifact or inference request is invalid."""


class ThresholdStrategy(StrEnum):
    """Supported validation-set decision-threshold objectives."""

    F1 = "f1"
    COST = "cost"


class CalibrationMethod(StrEnum):
    """Supported probability-calibration policies."""

    NONE = "none"
    SIGMOID = "sigmoid"
    ISOTONIC = "isotonic"


class SplitStrategy(StrEnum):
    """Supported dataset partitioning policies."""

    STRATIFIED = "stratified"
    TEMPORAL = "temporal"


class EstimatorType(StrEnum):
    """Supported base classifiers.

    Logistic regression is the default: an interpretable, well-calibrated
    linear baseline. Random forest and histogram-based gradient boosting are
    opt-in, more expressive alternatives for callers who have validated them
    against the baseline on their own data.
    """

    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"


_ESTIMATOR_CLASS_NAMES: dict[EstimatorType, str] = {
    EstimatorType.LOGISTIC_REGRESSION: "LogisticRegression",
    EstimatorType.RANDOM_FOREST: "RandomForestClassifier",
    EstimatorType.HIST_GRADIENT_BOOSTING: "HistGradientBoostingClassifier",
}


@dataclass(frozen=True)
class TrainingConfig:
    """Training and holdout configuration."""

    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    estimator: EstimatorType = EstimatorType.LOGISTIC_REGRESSION
    max_iterations: int = 1_000
    regularization: float = 1.0
    threshold_strategy: ThresholdStrategy = ThresholdStrategy.F1
    cost_policy: str = "default"
    false_positive_cost: float = 1.0
    false_negative_cost: float = 10.0
    calibration_method: CalibrationMethod = CalibrationMethod.SIGMOID
    calibration_folds: int = 3
    calibration_jobs: int = 1
    split_strategy: SplitStrategy = SplitStrategy.STRATIFIED
    time_column: str = "Time"
    temporal_gap: float = 0.0
    # Tree-estimator hyperparameters (random forest uses n_estimators and
    # max_depth; histogram gradient boosting uses all four below).
    n_estimators: int = 100
    max_depth: int | None = None
    learning_rate: float = 0.1
    l2_regularization: float = 0.0
    max_bins: int = 255

    def __post_init__(self) -> None:
        if not 0.05 <= self.test_size <= 0.4:
            raise ValueError("test_size must be between 0.05 and 0.4")
        if not 0.05 <= self.validation_size <= 0.4:
            raise ValueError("validation_size must be between 0.05 and 0.4")
        if self.test_size + self.validation_size > 0.6:
            raise ValueError("test_size and validation_size must sum to at most 0.6")
        if not isinstance(self.estimator, EstimatorType):
            raise ValueError(
                "estimator must be 'logistic_regression', 'random_forest', "
                "or 'hist_gradient_boosting'"
            )
        if self.max_iterations < 100:
            raise ValueError("max_iterations must be at least 100")
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")
        if not isinstance(self.threshold_strategy, ThresholdStrategy):
            raise ValueError("threshold_strategy must be 'f1' or 'cost'")
        if not isinstance(self.cost_policy, str) or not self.cost_policy.strip():
            raise ValueError("cost_policy must be a non-empty name")
        if (
            not np.isfinite(self.false_positive_cost)
            or not np.isfinite(self.false_negative_cost)
            or self.false_positive_cost <= 0
            or self.false_negative_cost <= 0
        ):
            raise ValueError("classification costs must be finite and positive")
        if not isinstance(self.calibration_method, CalibrationMethod):
            raise ValueError("calibration_method must be 'none', 'sigmoid', or 'isotonic'")
        if not 2 <= self.calibration_folds <= 10:
            raise ValueError("calibration_folds must be between 2 and 10")
        if self.calibration_jobs == 0 or self.calibration_jobs < -1:
            raise ValueError("calibration_jobs must be -1 or a positive integer")
        if not isinstance(self.split_strategy, SplitStrategy):
            raise ValueError("split_strategy must be 'stratified' or 'temporal'")
        if not self.time_column.strip():
            raise ValueError("time_column must not be empty")
        if not np.isfinite(self.temporal_gap) or self.temporal_gap < 0:
            raise ValueError("temporal_gap must be non-negative and finite")
        if self.temporal_gap > 0 and self.split_strategy is not SplitStrategy.TEMPORAL:
            raise ValueError("temporal_gap requires split_strategy='temporal'")
        if self.n_estimators < 10:
            raise ValueError("n_estimators must be at least 10")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be positive or None")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.l2_regularization < 0:
            raise ValueError("l2_regularization must be non-negative")
        if self.max_bins < 2:
            raise ValueError("max_bins must be at least 2")


@dataclass
class FraudModel:
    """A fitted fraud classifier plus its decision policy and model card data."""

    estimator: Any
    threshold: float
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]
    artifact_version: int = ARTIFACT_VERSION

    def predict_probabilities(self, features: pd.DataFrame) -> np.ndarray:
        """Return fraud probabilities after enforcing the training schema."""
        ordered = self.validate_features(features)
        try:
            probability_matrix = np.asarray(
                self.estimator.predict_proba(ordered),
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise ModelArtifactError("Model returned non-numeric probabilities.") from exc
        if probability_matrix.shape != (len(ordered), 2):
            raise ModelArtifactError(
                "Model returned probabilities with an invalid binary-class shape."
            )
        probabilities = probability_matrix[:, 1]
        if not np.isfinite(probabilities).all() or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)
        ):
            raise ModelArtifactError(
                "Model returned fraud probabilities outside the finite range from 0 to 1."
            )
        return probabilities

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return binary decisions using the artifact's tuned threshold."""
        return (self.predict_probabilities(features) >= self.threshold).astype("int8")

    def validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Validate and reorder transaction features to the training schema."""
        if features.empty:
            raise ModelArtifactError("At least one transaction is required.")
        if not features.columns.is_unique:
            raise ModelArtifactError("Input feature names must be unique.")

        provided = set(features.columns)
        expected = set(self.feature_names)
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            raise ModelArtifactError("Input schema does not match the model: " + ", ".join(details))

        ordered = features.reindex(columns=list(self.feature_names))
        non_numeric = ordered.select_dtypes(exclude=np.number).columns.tolist()
        if non_numeric:
            raise ModelArtifactError(f"Input features must be numeric: {non_numeric}")
        values = ordered.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ModelArtifactError("Input features must not contain missing or infinite values.")
        return ordered

    def explain_local(self, features: pd.DataFrame) -> list[dict[str, Any]]:
        """Return per-transaction feature contributions to the fraud score.

        For logistic regression, contributions are standardized coefficients multiplied
        by the scaled feature values (centered by training mean). For tree
        estimators, a simplified approximation based on importance weights
        (native importances, or training-data permutation importance when the
        estimator exposes none) is used.
        """
        ordered = self.validate_features(features)
        estimator_type = self.metadata.get("training_config", {}).get(
            "estimator", "logistic_regression"
        )

        if estimator_type == "logistic_regression":
            return self._explain_local_logistic(ordered)
        return self._explain_local_importance(ordered)

    def explain_local_natural_language(
        self,
        features: pd.DataFrame,
        probabilities: np.ndarray,
        *,
        threshold: float | None = None,
        contributions: list[dict[str, Any]] | None = None,
        top_k: int = 3,
        provider: ExplanationProvider | None = None,
    ) -> list[str]:
        """Render natural-language explanations from local effects.

        Defaults to TemplateExplanationProvider for offline, reproducible, and
        auditable explanations, while supporting optional external providers
        behind an isolated provider boundary.
        """
        ordered = self.validate_features(features)
        if not 1 <= top_k <= len(self.feature_names):
            raise ValueError("top_k must be between 1 and the number of model features")
        applied_threshold = self.threshold if threshold is None else threshold
        if isinstance(applied_threshold, bool) or not 0.0 <= applied_threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        probability_values = np.asarray(probabilities, dtype=float)
        if (
            probability_values.shape != (len(ordered),)
            or not np.isfinite(probability_values).all()
            or np.any((probability_values < 0.0) | (probability_values > 1.0))
        ):
            raise ModelArtifactError("Natural-language explanations require valid probabilities.")

        local_contributions = contributions or self.explain_local(ordered)
        if len(local_contributions) != len(ordered):
            raise ModelArtifactError("Local explanations do not match the transaction count.")

        active_provider = provider or TemplateExplanationProvider()
        requests = [
            ExplanationRequest(
                probability=float(prob),
                threshold=applied_threshold,
                decision="FRAUD" if float(prob) >= applied_threshold else "LEGITIMATE",
                contributions={k: float(v) for k, v in row_contrib.items()},
                features={str(k): v for k, v in ordered.iloc[i].to_dict().items()}
                if len(ordered) > 0
                else {},
                top_k=top_k,
            )
            for i, (prob, row_contrib) in enumerate(
                zip(probability_values, local_contributions, strict=True)
            )
        ]
        results = active_provider.explain_batch(requests)
        return [res.explanation for res in results]

    def _explain_local_logistic(self, ordered: pd.DataFrame) -> list[dict[str, Any]]:
        """Compute local explanations for logistic regression using standardized coefficients."""
        coefficients = self.metadata.get("feature_effects", [])
        if not coefficients:
            raise ModelArtifactError("Feature effects not available for local explanation.")

        scaler_mean = self.metadata.get("scaler_mean")
        scaler_scale = self.metadata.get("scaler_scale")
        if scaler_mean is None or scaler_scale is None:
            raise ModelArtifactError("Scaler statistics not available for local explanation.")

        coef_dict = {eff["feature"]: eff["coefficient"] for eff in coefficients}
        feature_array = ordered.to_numpy(dtype=float)

        centered = feature_array - np.asarray(scaler_mean, dtype=float)
        scaled = centered / np.asarray(scaler_scale, dtype=float)

        results = []
        for row_idx in range(len(ordered)):
            row_scaled = scaled[row_idx]
            row_contributions = {}
            for feat_idx, feature in enumerate(self.feature_names):
                coef = coef_dict.get(feature, 0.0)
                row_contributions[feature] = float(coef * row_scaled[feat_idx])
            results.append(row_contributions)
        return results

    def _explain_local_importance(self, ordered: pd.DataFrame) -> list[dict[str, Any]]:
        """Compute local explanations for tree estimators from importance weights."""
        effects = self.metadata.get("feature_effects", [])
        if not effects:
            raise ModelArtifactError("Feature effects not available for local explanation.")

        importance_dict = {eff["feature"]: eff["coefficient"] for eff in effects}
        total_importance = sum(importance_dict.values()) or 1.0

        feature_array = ordered.to_numpy(dtype=float)
        feature_means = np.mean(feature_array, axis=0)

        results = []
        for row_idx in range(len(ordered)):
            row_values = feature_array[row_idx]
            row_contributions = {}
            for feat_idx, feature in enumerate(self.feature_names):
                importance = importance_dict.get(feature, 0.0)
                deviation = row_values[feat_idx] - feature_means[feat_idx]
                normalized_importance = importance / total_importance
                row_contributions[feature] = float(normalized_importance * deviation)
            results.append(row_contributions)
        return results


def train_model(
    dataset: ValidatedDataset,
    *,
    config: TrainingConfig | None = None,
    provenance: dict[str, str] | None = None,
) -> FraudModel:
    """Fit and evaluate a deterministic fraud model with untouched test data."""
    settings = config or TrainingConfig()
    _ensure_split_capacity(dataset.target, settings)
    (
        features_train,
        features_validation,
        features_test,
        target_train,
        target_validation,
        target_test,
    ) = _split_dataset(
        dataset,
        settings,
    )

    base_estimator = _build_base_estimator(settings)
    if settings.calibration_method is CalibrationMethod.NONE:
        estimator: Any = base_estimator
    else:
        minimum_training_class = int(target_train.value_counts().min())
        if minimum_training_class < settings.calibration_folds:
            raise ValueError(
                "Each training class needs at least as many rows as calibration_folds."
            )
        estimator = CalibratedClassifierCV(
            estimator=base_estimator,
            method=settings.calibration_method.value,
            cv=settings.calibration_folds,
            n_jobs=settings.calibration_jobs,
        )
    estimator.fit(features_train, target_train)

    validation_probabilities = np.asarray(
        estimator.predict_proba(features_validation)[:, 1],
        dtype=float,
    )
    validation_target_array = target_validation.to_numpy()
    if settings.threshold_strategy is ThresholdStrategy.COST:
        threshold = select_cost_threshold(
            validation_target_array,
            validation_probabilities,
            false_positive_cost=settings.false_positive_cost,
            false_negative_cost=settings.false_negative_cost,
        )
    else:
        threshold = select_f1_threshold(validation_target_array, validation_probabilities)
    validation_metrics = evaluate_predictions(
        validation_target_array,
        validation_probabilities,
        threshold=threshold,
    )
    test_probabilities = np.asarray(estimator.predict_proba(features_test)[:, 1], dtype=float)
    test_metrics = evaluate_predictions(
        target_test.to_numpy(),
        test_probabilities,
        threshold=threshold,
    )
    validation_metrics_payload = validation_metrics.to_dict()
    test_metrics_payload = test_metrics.to_dict()
    validation_metrics_payload["expected_cost_per_transaction"] = expected_classification_cost(
        validation_target_array,
        validation_probabilities,
        threshold=threshold,
        false_positive_cost=settings.false_positive_cost,
        false_negative_cost=settings.false_negative_cost,
    )
    test_metrics_payload["expected_cost_per_transaction"] = expected_classification_cost(
        target_test.to_numpy(),
        test_probabilities,
        threshold=threshold,
        false_positive_cost=settings.false_positive_cost,
        false_negative_cost=settings.false_negative_cost,
    )
    feature_effects = _extract_feature_effects(
        estimator,
        dataset.feature_names,
        calibration_method=settings.calibration_method,
        permutation_data=(features_train, target_train),
        random_state=settings.random_state,
    )

    scaler_mean = None
    scaler_scale = None
    if settings.estimator is EstimatorType.LOGISTIC_REGRESSION:
        if settings.calibration_method is CalibrationMethod.NONE:
            scaler = estimator.named_steps["scale"]
        else:
            scaler = estimator.calibrated_classifiers_[0].estimator.named_steps["scale"]
        scaler_mean = scaler.mean_.tolist()
        scaler_scale = scaler.scale_.tolist()

    dataset_fingerprint = _dataset_fingerprint(dataset)
    metadata: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "estimator": (
            _ESTIMATOR_CLASS_NAMES[settings.estimator]
            if settings.calibration_method is CalibrationMethod.NONE
            else f"CalibratedClassifierCV({_ESTIMATOR_CLASS_NAMES[settings.estimator]})"
        ),
        "calibration": {
            "method": settings.calibration_method.value,
            "folds": (
                settings.calibration_folds
                if settings.calibration_method is not CalibrationMethod.NONE
                else None
            ),
            "jobs": (
                settings.calibration_jobs
                if settings.calibration_method is not CalibrationMethod.NONE
                else None
            ),
        },
        "runtime_versions": {
            "python": python_version(),
            "joblib": distribution_version("joblib"),
            "numpy": distribution_version("numpy"),
            "pandas": distribution_version("pandas"),
            "scikit_learn": sklearn.__version__,
            "scipy": distribution_version("scipy"),
        },
        "scikit_learn_version": sklearn.__version__,
        "feature_names": list(dataset.feature_names),
        "feature_count": len(dataset.feature_names),
        "row_count": len(dataset.target),
        "fraud_count": int(dataset.target.sum()),
        "fraud_rate": float(dataset.target.mean()),
        "dataset_fingerprint": dataset_fingerprint,
        "splits": {
            "train": len(target_train),
            "validation": len(target_validation),
            "test": len(target_test),
        },
        "split_time_ranges": _split_time_ranges(
            features_train,
            features_validation,
            features_test,
            settings,
        ),
        "training_config": asdict(settings),
        "cost_policy": {
            "name": settings.cost_policy,
            "false_positive_cost": settings.false_positive_cost,
            "false_negative_cost": settings.false_negative_cost,
        },
        "reference_profile": build_reference_profile(features_train),
        "drift_thresholds": default_thresholds(),
        "feature_effects": feature_effects,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "validation_metrics": validation_metrics_payload,
        "test_metrics": test_metrics_payload,
    }
    metadata["lineage"] = _build_lineage(
        dataset_fingerprint=dataset_fingerprint,
        settings=settings,
        provenance=provenance,
    )
    return FraudModel(
        estimator=estimator,
        threshold=threshold,
        feature_names=dataset.feature_names,
        metadata=metadata,
    )


def save_model(model: FraudModel, output_directory: Path | str) -> Path:
    """Persist a model, metadata, and integrity manifest with atomic file swaps."""
    _validate_loaded_model(model)
    metadata_json = _serialize_metadata(model.metadata)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / MODEL_FILENAME
    metadata_path = destination / METADATA_FILENAME
    manifest_path = destination / MANIFEST_FILENAME
    temporary_model = destination / f".{MODEL_FILENAME}.tmp"
    temporary_metadata = destination / f".{METADATA_FILENAME}.tmp"
    temporary_manifest = destination / f".{MANIFEST_FILENAME}.tmp"

    try:
        joblib.dump(model, temporary_model)
        temporary_metadata.write_text(metadata_json, encoding="utf-8")
        manifest = {
            "artifact_version": model.artifact_version,
            "files": {
                MODEL_FILENAME: _file_sha256(temporary_model),
                METADATA_FILENAME: _file_sha256(temporary_metadata),
            },
            "hash_algorithm": "sha256",
        }
        temporary_manifest.write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_model.replace(model_path)
        temporary_metadata.replace(metadata_path)
        temporary_manifest.replace(manifest_path)
    finally:
        temporary_model.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    return model_path


def load_model(path: Path | str) -> FraudModel:
    """Load a trusted model artifact.

    Joblib artifacts can execute code while loading. Only load artifacts produced
    by a trusted training process.
    """
    requested_path = Path(path)
    artifact_path = requested_path
    persisted_metadata: dict[str, Any] | None = None
    if requested_path.is_dir():
        persisted_metadata = _verify_manifest(requested_path)
        _validate_sklearn_compatibility(persisted_metadata)
        artifact_path = requested_path / MODEL_FILENAME
    if not artifact_path.is_file():
        raise ModelArtifactError(f"Model artifact does not exist: {artifact_path}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            candidate = joblib.load(artifact_path)
    except InconsistentVersionWarning as exc:
        raise ModelArtifactError(
            "Model artifact was created by an incompatible scikit-learn version; "
            "retrain it in the current runtime."
        ) from exc
    except Exception as exc:
        raise ModelArtifactError(f"Could not load model artifact: {exc}") from exc
    if not isinstance(candidate, FraudModel):
        raise ModelArtifactError("Artifact does not contain a FraudModel.")
    _validate_loaded_model(candidate)
    embedded_metadata_json = _serialize_metadata(candidate.metadata)
    if persisted_metadata is not None:
        embedded_metadata = json.loads(embedded_metadata_json)
        if embedded_metadata != persisted_metadata:
            raise ModelArtifactError(
                "Persisted metadata does not match the metadata embedded in the model."
            )
    return candidate


def _validate_loaded_model(candidate: FraudModel) -> None:
    """Validate the runtime structure required by every artifact consumer."""
    if candidate.artifact_version != ARTIFACT_VERSION:
        raise ModelArtifactError(
            f"Unsupported artifact version {candidate.artifact_version}; "
            f"expected {ARTIFACT_VERSION}."
        )
    if (
        isinstance(candidate.threshold, bool)
        or not isinstance(candidate.threshold, int | float)
        or not np.isfinite(candidate.threshold)
        or not 0.0 <= candidate.threshold <= 1.0
    ):
        raise ModelArtifactError("Artifact contains an invalid decision threshold.")
    if (
        not isinstance(candidate.feature_names, tuple)
        or not candidate.feature_names
        or any(not isinstance(feature, str) or not feature for feature in candidate.feature_names)
        or len(set(candidate.feature_names)) != len(candidate.feature_names)
    ):
        raise ModelArtifactError("Artifact contains an invalid feature schema.")
    if not callable(getattr(candidate.estimator, "predict_proba", None)):
        raise ModelArtifactError("Artifact estimator does not support probability prediction.")
    estimator_classes = np.asarray(getattr(candidate.estimator, "classes_", None))
    if estimator_classes.shape != (2,) or not np.array_equal(
        estimator_classes,
        np.array([0, 1]),
    ):
        raise ModelArtifactError("Artifact estimator must use the binary class order [0, 1].")
    if not isinstance(candidate.metadata, dict):
        raise ModelArtifactError("Artifact metadata must be a mapping.")
    _validate_sklearn_compatibility(candidate.metadata)

    required_metadata = {
        "artifact_version": candidate.artifact_version,
        "feature_names": list(candidate.feature_names),
        "feature_count": len(candidate.feature_names),
    }
    for field, expected_value in required_metadata.items():
        if candidate.metadata.get(field) != expected_value:
            raise ModelArtifactError(
                f"Artifact metadata field {field!r} is missing or inconsistent."
            )
    created_at = candidate.metadata.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ModelArtifactError("Artifact metadata field 'created_at' is missing or invalid.")
    fingerprint = candidate.metadata.get("dataset_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ModelArtifactError(
            "Artifact metadata field 'dataset_fingerprint' is missing or invalid."
        )


def _validate_sklearn_compatibility(metadata: dict[str, Any]) -> None:
    artifact_version = metadata.get("scikit_learn_version")
    if not isinstance(artifact_version, str) or not artifact_version:
        raise ModelArtifactError(
            "Artifact metadata field 'scikit_learn_version' is missing or invalid."
        )
    if artifact_version != sklearn.__version__:
        raise ModelArtifactError(
            "Model artifact scikit-learn version mismatch: "
            f"artifact={artifact_version!r}, runtime={sklearn.__version__!r}. "
            "Retrain the model with the current runtime."
        )


def _serialize_metadata(metadata: dict[str, Any]) -> str:
    try:
        return json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True) + "\n"
    except (TypeError, ValueError) as exc:
        raise ModelArtifactError(
            "Artifact metadata must contain only finite JSON-compatible values."
        ) from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def _verify_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ModelArtifactError(f"Artifact integrity manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
        expected_files = manifest["files"]
        if (
            manifest["hash_algorithm"] != "sha256"
            or manifest["artifact_version"] != ARTIFACT_VERSION
            or not isinstance(expected_files, dict)
        ):
            raise ValueError("unsupported manifest")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ModelArtifactError(f"Artifact integrity manifest is invalid: {exc}") from exc

    for filename in (MODEL_FILENAME, METADATA_FILENAME):
        file_path = directory / filename
        expected_digest = expected_files.get(filename)
        if not file_path.is_file() or not isinstance(expected_digest, str):
            raise ModelArtifactError(f"Artifact integrity entry is missing for {filename}.")
        if _file_sha256(file_path) != expected_digest:
            raise ModelArtifactError(f"Artifact integrity check failed for {filename}.")

    metadata_path = directory / METADATA_FILENAME
    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, ValueError) as exc:
        raise ModelArtifactError(f"Artifact metadata is invalid: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ModelArtifactError("Artifact metadata must be a JSON object.")
    return metadata


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckResult:
    """Outcome of an individual artifact validation check."""

    name: str
    status: str
    details: str
    data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "details": self.details,
        }
        if self.data is not None:
            payload["data"] = self.data
        return payload


@dataclass(frozen=True)
class ArtifactValidationReport:
    """Comprehensive read-only validation report for a model artifact."""

    valid: bool
    artifact_path: str
    artifact_version: int | None
    checks: tuple[CheckResult, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    metadata_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "artifact_path": self.artifact_path,
            "artifact_version": self.artifact_version,
            "checks": [check.to_dict() for check in self.checks],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metadata_summary": self.metadata_summary,
        }


def validate_artifact(
    path: Path | str,
    *,
    strict: bool = False,
) -> ArtifactValidationReport:
    """Inspect and validate artifact integrity, runtime compatibility, lineage, and reports."""
    artifact_path = Path(path)
    checks: list[CheckResult] = []
    errors: list[str] = []
    warnings_list: list[str] = []
    metadata_summary: dict[str, Any] | None = None
    artifact_version: int | None = None

    if not artifact_path.exists():
        msg = f"Artifact path does not exist: {artifact_path}"
        errors.append(msg)
        checks.append(CheckResult(name="path_exists", status="failed", details=msg))
        return ArtifactValidationReport(
            valid=False,
            artifact_path=str(artifact_path),
            artifact_version=None,
            checks=tuple(checks),
            errors=tuple(errors),
            warnings=tuple(warnings_list),
        )

    is_dir = artifact_path.is_dir()
    manifest_file = artifact_path / MANIFEST_FILENAME if is_dir else None
    metadata_file = artifact_path / METADATA_FILENAME if is_dir else None
    model_file = artifact_path / MODEL_FILENAME if is_dir else artifact_path

    raw_metadata: dict[str, Any] | None = None
    can_load_model = False

    # 1. Manifest and file integrity check
    if not is_dir:
        warnings_list.append("Artifact is a standalone file without directory manifest.")
        if strict:
            errors.append("Strict mode requires a full artifact directory with manifest.json.")
        checks.append(
            CheckResult(
                name="integrity",
                status="failed" if strict else "warning",
                details="Standalone joblib file without manifest.json.",
            )
        )
        can_load_model = not strict and model_file.is_file()
    elif manifest_file is None or not manifest_file.is_file():
        errors.append(f"Missing {MANIFEST_FILENAME} in artifact directory.")
        checks.append(
            CheckResult(
                name="integrity",
                status="failed",
                details=f"Missing {MANIFEST_FILENAME}.",
            )
        )
    else:
        integrity_ok = True
        try:
            manifest = json.loads(
                manifest_file.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
            )
            artifact_version = manifest.get("artifact_version")
            if manifest.get("hash_algorithm") != "sha256":
                errors.append(
                    f"Unsupported manifest hash algorithm: {manifest.get('hash_algorithm')}"
                )
                integrity_ok = False
            if artifact_version != ARTIFACT_VERSION:
                errors.append(
                    f"Artifact version {artifact_version} does not match "
                    f"current {ARTIFACT_VERSION}."
                )
                integrity_ok = False

            files_map = manifest.get("files", {})
            for fname in (MODEL_FILENAME, METADATA_FILENAME):
                fpath = artifact_path / fname
                expected_hash = files_map.get(fname)
                if not fpath.is_file():
                    errors.append(f"Missing required artifact file: {fname}")
                    integrity_ok = False
                elif not expected_hash:
                    errors.append(f"Manifest missing entry for file: {fname}")
                    integrity_ok = False
                else:
                    actual_hash = _file_sha256(fpath)
                    if actual_hash != expected_hash:
                        errors.append(
                            f"Integrity digest mismatch for {fname}: "
                            f"expected {expected_hash}, got {actual_hash}"
                        )
                        integrity_ok = False

            can_load_model = integrity_ok and model_file.is_file()
            checks.append(
                CheckResult(
                    name="integrity",
                    status="passed" if integrity_ok else "failed",
                    details=(
                        "Manifest digests verified successfully."
                        if integrity_ok
                        else "Integrity verification failed."
                    ),
                    data={"manifest_files": list(files_map.keys())},
                )
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"Failed to read or parse manifest: {exc}")
            checks.append(
                CheckResult(name="integrity", status="failed", details=str(exc))
            )

    # Read metadata.json if present
    if metadata_file and metadata_file.is_file():
        try:
            raw_metadata = json.loads(
                metadata_file.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
            )
            if not isinstance(raw_metadata, dict):
                errors.append("metadata.json does not contain a JSON mapping.")
                raw_metadata = None
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"Failed to read or parse metadata.json: {exc}")

    # 2. Runtime compatibility check
    if raw_metadata is not None:
        art_sklearn = raw_metadata.get("scikit_learn_version")
        if art_sklearn != sklearn.__version__:
            errors.append(
                f"scikit-learn runtime mismatch: artifact was created with {art_sklearn}, "
                f"current runtime is {sklearn.__version__}."
            )
            checks.append(
                CheckResult(
                    name="runtime_compatibility",
                    status="failed",
                    details=f"scikit-learn mismatch: {art_sklearn} != {sklearn.__version__}",
                    data={"artifact_sklearn": art_sklearn, "runtime_sklearn": sklearn.__version__},
                )
            )
        else:
            checks.append(
                CheckResult(
                    name="runtime_compatibility",
                    status="passed",
                    details=f"Runtime scikit-learn version matches: {sklearn.__version__}",
                    data={"scikit_learn": sklearn.__version__},
                )
            )

    # 3. Lineage completeness check
    if raw_metadata is not None:
        lineage = raw_metadata.get("lineage")
        if not isinstance(lineage, dict):
            errors.append("Artifact metadata is missing 'lineage' block.")
            checks.append(
                CheckResult(
                    name="lineage_completeness",
                    status="failed",
                    details="Missing lineage block.",
                )
            )
        else:
            lineage_ok = True
            for req_key in ("dataset_fingerprint", "config_hash", "code_version", "content_hash"):
                if not lineage.get(req_key):
                    errors.append(f"Lineage is missing required key {req_key!r}.")
                    lineage_ok = False

            if lineage_ok:
                content_json = json.dumps(
                    {
                        "code_version": lineage["code_version"],
                        "config_hash": lineage["config_hash"],
                        "dataset_fingerprint": lineage["dataset_fingerprint"],
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                expected_content_hash = sha256(content_json.encode("utf-8")).hexdigest()
                if lineage["content_hash"] != expected_content_hash:
                    errors.append(
                        f"Lineage content_hash mismatch: expected {expected_content_hash}, "
                        f"got {lineage['content_hash']}"
                    )
                    lineage_ok = False

            has_git = bool(lineage.get("git_commit"))
            if not has_git:
                warn_msg = "Lineage does not include git commit provenance."
                warnings_list.append(warn_msg)
                if strict:
                    errors.append(warn_msg)
                    lineage_ok = False

            checks.append(
                CheckResult(
                    name="lineage_completeness",
                    status="passed" if lineage_ok else "failed",
                    details=(
                        "Lineage hashes verified successfully."
                        if lineage_ok
                        else "Lineage verification failed."
                    ),
                    data={
                        "dataset_fingerprint": lineage.get("dataset_fingerprint"),
                        "content_hash": lineage.get("content_hash"),
                        "has_git_provenance": has_git,
                    },
                )
            )

    # 4. Model and report compatibility check
    loaded_model: FraudModel | None = None
    if not can_load_model:
        errors.append("Model deserialization skipped due to integrity verification failure.")
    elif model_file.is_file():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                candidate = joblib.load(model_file)
            if not isinstance(candidate, FraudModel):
                errors.append("Deserialized file does not contain a FraudModel instance.")
            else:
                loaded_model = candidate
                _validate_loaded_model(candidate)
                if raw_metadata is not None:
                    embedded_meta = json.loads(_serialize_metadata(candidate.metadata))
                    if embedded_meta != raw_metadata:
                        errors.append("Persisted metadata.json does not match embedded metadata.")
        except (
            OSError,
            ModelArtifactError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            ImportError,
            EOFError,
        ) as exc:
            errors.append(f"Model loading or structural validation failed: {exc}")

    if loaded_model is not None:
        meta = loaded_model.metadata
        report_missing = [
            req_field
            for req_field in (
                "feature_effects",
                "reference_profile",
                "drift_thresholds",
                "cost_policy",
                "validation_metrics",
                "test_metrics",
            )
            if req_field not in meta or meta[req_field] is None
        ]

        if report_missing:
            errors.append(f"Model metadata is missing required report fields: {report_missing}")
            checks.append(
                CheckResult(
                    name="report_compatibility",
                    status="failed",
                    details=f"Missing report fields: {report_missing}",
                )
            )
        else:
            checks.append(
                CheckResult(
                    name="report_compatibility",
                    status="passed",
                    details="All required report structures and metrics are present.",
                )
            )

        test_metrics = meta.get("test_metrics")
        test_roc = test_metrics.get("roc_auc") if isinstance(test_metrics, dict) else None
        metadata_summary = {
            "estimator": str(meta.get("estimator")),
            "threshold": loaded_model.threshold,
            "feature_count": len(loaded_model.feature_names),
            "created_at": meta.get("created_at"),
            "dataset_fingerprint": meta.get("dataset_fingerprint"),
            "test_roc_auc": test_roc,
        }
    else:
        checks.append(
            CheckResult(
                name="report_compatibility",
                status="failed",
                details="Could not inspect model for report compatibility.",
            )
        )

    is_valid = len(errors) == 0
    return ArtifactValidationReport(
        valid=is_valid,
        artifact_path=str(artifact_path),
        artifact_version=artifact_version or ARTIFACT_VERSION,
        checks=tuple(checks),
        errors=tuple(errors),
        warnings=tuple(warnings_list),
        metadata_summary=metadata_summary,
    )



def _dataset_fingerprint(dataset: ValidatedDataset) -> str:
    digest = hashlib.sha256()
    digest.update("\0".join(dataset.feature_names).encode())
    feature_hash = pd.util.hash_pandas_object(dataset.features, index=True).to_numpy(dtype="uint64")
    target_hash = pd.util.hash_pandas_object(dataset.target, index=True).to_numpy(dtype="uint64")
    digest.update(cast(np.ndarray, feature_hash).tobytes())
    digest.update(cast(np.ndarray, target_hash).tobytes())
    return digest.hexdigest()


def _build_lineage(
    *,
    dataset_fingerprint: str,
    settings: TrainingConfig,
    provenance: dict[str, str] | None,
) -> dict[str, str]:
    """Build deterministic content lineage for a newly trained artifact."""
    config_json = json.dumps(
        asdict(settings),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    config_hash = sha256(config_json.encode("utf-8")).hexdigest()
    content_json = json.dumps(
        {
            "code_version": __version__,
            "config_hash": config_hash,
            "dataset_fingerprint": dataset_fingerprint,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    lineage = {
        "code_version": __version__,
        "config_hash": config_hash,
        "content_hash": sha256(content_json.encode("utf-8")).hexdigest(),
        "dataset_fingerprint": dataset_fingerprint,
    }
    if provenance:
        for key in ("git_commit", "git_repository"):
            value = provenance.get(key)
            if value is not None:
                if not value.strip():
                    raise ValueError(f"provenance field {key!r} must not be empty")
                lineage[key] = value
    return lineage


def _build_base_estimator(settings: TrainingConfig) -> Pipeline:
    if settings.estimator is EstimatorType.RANDOM_FOREST:
        return Pipeline(
            steps=[
                (
                    "classifier",
                    RandomForestClassifier(
                        class_weight="balanced",
                        random_state=settings.random_state,
                        n_estimators=settings.n_estimators,
                        max_depth=settings.max_depth,
                    ),
                ),
            ]
        )
    if settings.estimator is EstimatorType.HIST_GRADIENT_BOOSTING:
        return Pipeline(
            steps=[
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        class_weight="balanced",
                        random_state=settings.random_state,
                        max_iter=settings.max_iterations,
                        learning_rate=settings.learning_rate,
                        max_depth=settings.max_depth,
                        l2_regularization=settings.l2_regularization,
                        max_bins=settings.max_bins,
                    ),
                ),
            ]
        )
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=settings.regularization,
                    class_weight="balanced",
                    max_iter=settings.max_iterations,
                    random_state=settings.random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _extract_feature_effects(
    estimator: Any,
    feature_names: tuple[str, ...],
    *,
    calibration_method: CalibrationMethod,
    permutation_data: tuple[pd.DataFrame, pd.Series] | None = None,
    random_state: int = 42,
) -> list[dict[str, str | float | int | None]]:
    if calibration_method is CalibrationMethod.NONE:
        fitted_pipelines = [estimator]
    else:
        fitted_pipelines = [
            calibrated_classifier.estimator
            for calibrated_classifier in estimator.calibrated_classifiers_
        ]

    classifiers = [pipeline.named_steps["classifier"] for pipeline in fitted_pipelines]
    if hasattr(classifiers[0], "coef_"):
        method = "standardized_coefficient"
        rows = [np.asarray(classifier.coef_[0], dtype=float) for classifier in classifiers]
    elif hasattr(classifiers[0], "feature_importances_"):
        method = "feature_importance"
        rows = [
            np.asarray(classifier.feature_importances_, dtype=float) for classifier in classifiers
        ]
    else:
        if permutation_data is None:
            raise ValueError(
                "Permutation training data is required for estimators without "
                "native feature importance."
            )
        method = "permutation_importance"
        rows = [
            _permutation_effects(pipeline, permutation_data, random_state)
            for pipeline in fitted_pipelines
        ]

    mean_values = np.mean(np.vstack(rows), axis=0)
    ranked = sorted(
        zip(feature_names, mean_values, strict=True),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return [
        {
            "rank": rank,
            "feature": feature,
            "coefficient": float(value),
            "absolute_effect": float(abs(value)),
            "direction": (
                ("higher_fraud_risk" if value >= 0 else "lower_fraud_risk")
                if method == "standardized_coefficient"
                else None
            ),
            "method": method,
        }
        for rank, (feature, value) in enumerate(ranked, start=1)
    ]


def _permutation_effects(
    pipeline: Any,
    permutation_data: tuple[pd.DataFrame, pd.Series],
    random_state: int,
) -> np.ndarray:
    """Measure mean average-precision drop per feature on training data.

    Used only for estimators without native coefficients or importances.
    Training data is appropriate here because the result is descriptive, not
    used for threshold tuning or holdout evaluation.
    """
    features, target = permutation_data
    result = permutation_importance(
        pipeline,
        features,
        target,
        n_repeats=5,
        random_state=random_state,
        scoring="average_precision",
        n_jobs=1,
    )
    values = np.asarray(result.importances_mean, dtype=float)
    return np.asarray(np.clip(values, 0.0, None), dtype=float)


def _split_dataset(
    dataset: ValidatedDataset,
    settings: TrainingConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    if settings.split_strategy is SplitStrategy.TEMPORAL:
        if settings.time_column not in dataset.features.columns:
            raise ValueError(
                f"Temporal splitting requires feature column {settings.time_column!r}."
            )
        order = np.argsort(
            dataset.features[settings.time_column].to_numpy(dtype=float),
            kind="stable",
        )
        ordered_features = dataset.features.iloc[order].reset_index(drop=True)
        ordered_target = dataset.target.iloc[order].reset_index(drop=True)
        times = ordered_features[settings.time_column].to_numpy(dtype=float)
        test_count = max(1, round(len(ordered_target) * settings.test_size))
        validation_count = max(1, round(len(ordered_target) * settings.validation_size))
        if settings.temporal_gap == 0.0:
            train_end = len(ordered_target) - validation_count - test_count
            validation_start = train_end
            validation_end = len(ordered_target) - test_count
            test_start = validation_end
        else:
            test_start = len(ordered_target) - test_count
            test_min_time = times[test_start]
            val_end_cutoff = test_min_time - settings.temporal_gap
            validation_end = int(np.searchsorted(times, val_end_cutoff, side="right"))
            validation_start = max(0, validation_end - validation_count)
            if validation_end <= validation_start:
                raise ValueError(
                    f"Temporal split with temporal_gap={settings.temporal_gap} "
                    "leaves no validation rows."
                )
            val_min_time = times[validation_start]
            train_end_cutoff = val_min_time - settings.temporal_gap
            train_end = int(np.searchsorted(times, train_end_cutoff, side="right"))

        if train_end <= 0:
            raise ValueError(
                "Temporal split leaves no training rows."
                if settings.temporal_gap == 0.0
                else (
                    f"Temporal split with temporal_gap={settings.temporal_gap} "
                    "leaves no training rows."
                )
            )

        features_train = ordered_features.iloc[:train_end]
        features_validation = ordered_features.iloc[validation_start:validation_end]
        features_test = ordered_features.iloc[test_start:]
        target_train = ordered_target.iloc[:train_end]
        target_validation = ordered_target.iloc[validation_start:validation_end]
        target_test = ordered_target.iloc[test_start:]
        for split_name, split_target in (
            ("training", target_train),
            ("validation", target_validation),
            ("test", target_test),
        ):
            if set(split_target.unique().tolist()) != {0, 1}:
                raise ValueError(
                    f"Temporal {split_name} split must contain both target classes; "
                    "use more data or stratified splitting."
                )
        return (
            features_train,
            features_validation,
            features_test,
            target_train,
            target_validation,
            target_test,
        )

    features_train_validation, features_test, target_train_validation, target_test = (
        train_test_split(
            dataset.features,
            dataset.target,
            test_size=settings.test_size,
            random_state=settings.random_state,
            stratify=dataset.target,
        )
    )
    relative_validation_size = settings.validation_size / (1.0 - settings.test_size)
    features_train, features_validation, target_train, target_validation = train_test_split(
        features_train_validation,
        target_train_validation,
        test_size=relative_validation_size,
        random_state=settings.random_state,
        stratify=target_train_validation,
    )
    return (
        features_train,
        features_validation,
        features_test,
        target_train,
        target_validation,
        target_test,
    )


def _split_time_ranges(
    features_train: pd.DataFrame,
    features_validation: pd.DataFrame,
    features_test: pd.DataFrame,
    settings: TrainingConfig,
) -> dict[str, Any] | None:
    if settings.split_strategy is not SplitStrategy.TEMPORAL:
        return None
    train_min = float(features_train[settings.time_column].min())
    train_max = float(features_train[settings.time_column].max())
    val_min = float(features_validation[settings.time_column].min())
    val_max = float(features_validation[settings.time_column].max())
    test_min = float(features_test[settings.time_column].min())
    test_max = float(features_test[settings.time_column].max())
    return {
        "train": {
            "minimum": train_min,
            "maximum": train_max,
        },
        "validation": {
            "minimum": val_min,
            "maximum": val_max,
        },
        "test": {
            "minimum": test_min,
            "maximum": test_max,
        },
        "gaps": {
            "train_to_validation": val_min - train_max,
            "validation_to_test": test_min - val_max,
        },
        "temporal_gap": settings.temporal_gap,
    }


def _ensure_split_capacity(target: pd.Series, config: TrainingConfig) -> None:
    class_counts = target.value_counts()
    minimum_count = int(class_counts.min())
    if minimum_count < 6:
        raise ValueError(
            "Each class needs at least 6 rows for stratified train/validation/test splits."
        )

    expected_test_minority = minimum_count * config.test_size
    expected_validation_minority = minimum_count * config.validation_size
    if min(expected_test_minority, expected_validation_minority) < 1:
        raise ValueError(
            "The minority class is too small for the configured test and validation splits."
        )
