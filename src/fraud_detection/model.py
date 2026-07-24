"""Leakage-safe model training, prediction, and artifact persistence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.data import ValidatedDataset
from fraud_detection.drift import build_reference_profile
from fraud_detection.evaluation import evaluate_predictions, select_f1_threshold

ARTIFACT_VERSION = 1
MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
MANIFEST_FILENAME = "manifest.json"


class ModelArtifactError(ValueError):
    """Raised when a model artifact or inference request is invalid."""


@dataclass(frozen=True)
class TrainingConfig:
    """Training and holdout configuration."""

    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    max_iterations: int = 1_000
    regularization: float = 1.0

    def __post_init__(self) -> None:
        if not 0.05 <= self.test_size <= 0.4:
            raise ValueError("test_size must be between 0.05 and 0.4")
        if not 0.05 <= self.validation_size <= 0.4:
            raise ValueError("validation_size must be between 0.05 and 0.4")
        if self.test_size + self.validation_size > 0.6:
            raise ValueError("test_size and validation_size must sum to at most 0.6")
        if self.max_iterations < 100:
            raise ValueError("max_iterations must be at least 100")
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")


@dataclass
class FraudModel:
    """A fitted fraud classifier plus its decision policy and model card data."""

    pipeline: Pipeline
    threshold: float
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]
    artifact_version: int = ARTIFACT_VERSION

    def predict_probabilities(self, features: pd.DataFrame) -> np.ndarray:
        """Return fraud probabilities after enforcing the training schema."""
        ordered = self.validate_features(features)
        probabilities = cast(
            np.ndarray,
            np.asarray(self.pipeline.predict_proba(ordered)[:, 1], dtype=float),
        )
        if probabilities.shape != (len(ordered),):
            raise ModelArtifactError("Model returned probabilities with an invalid shape.")
        return probabilities

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return binary decisions using the artifact's tuned threshold."""
        return cast(
            np.ndarray,
            (self.predict_probabilities(features) >= self.threshold).astype("int8"),
        )

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


def train_model(
    dataset: ValidatedDataset,
    *,
    config: TrainingConfig | None = None,
) -> FraudModel:
    """Fit and evaluate a deterministic fraud model with untouched test data."""
    settings = config or TrainingConfig()
    _ensure_split_capacity(dataset.target, settings)

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

    pipeline = Pipeline(
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
    pipeline.fit(features_train, target_train)

    validation_probabilities = np.asarray(
        pipeline.predict_proba(features_validation)[:, 1],
        dtype=float,
    )
    threshold = select_f1_threshold(target_validation.to_numpy(), validation_probabilities)
    validation_metrics = evaluate_predictions(
        target_validation.to_numpy(),
        validation_probabilities,
        threshold=threshold,
    )
    test_probabilities = np.asarray(pipeline.predict_proba(features_test)[:, 1], dtype=float)
    test_metrics = evaluate_predictions(
        target_test.to_numpy(),
        test_probabilities,
        threshold=threshold,
    )

    metadata: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "estimator": "LogisticRegression",
        "scikit_learn_version": sklearn.__version__,
        "feature_names": list(dataset.feature_names),
        "feature_count": len(dataset.feature_names),
        "row_count": len(dataset.target),
        "fraud_count": int(dataset.target.sum()),
        "fraud_rate": float(dataset.target.mean()),
        "dataset_fingerprint": _dataset_fingerprint(dataset),
        "splits": {
            "train": len(target_train),
            "validation": len(target_validation),
            "test": len(target_test),
        },
        "training_config": asdict(settings),
        "reference_profile": build_reference_profile(features_train),
        "validation_metrics": validation_metrics.to_dict(),
        "test_metrics": test_metrics.to_dict(),
    }
    return FraudModel(
        pipeline=pipeline,
        threshold=threshold,
        feature_names=dataset.feature_names,
        metadata=metadata,
    )


def save_model(model: FraudModel, output_directory: Path | str) -> Path:
    """Persist a model, metadata, and integrity manifest with atomic file swaps."""
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
        temporary_metadata.write_text(
            json.dumps(model.metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "artifact_version": model.artifact_version,
            "files": {
                MODEL_FILENAME: _file_sha256(temporary_model),
                METADATA_FILENAME: _file_sha256(temporary_metadata),
            },
            "hash_algorithm": "sha256",
        }
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
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
    if requested_path.is_dir():
        _verify_manifest(requested_path)
        artifact_path = requested_path / MODEL_FILENAME
    if not artifact_path.is_file():
        raise ModelArtifactError(f"Model artifact does not exist: {artifact_path}")

    try:
        candidate = joblib.load(artifact_path)
    except Exception as exc:
        raise ModelArtifactError(f"Could not load model artifact: {exc}") from exc
    if not isinstance(candidate, FraudModel):
        raise ModelArtifactError("Artifact does not contain a FraudModel.")
    if candidate.artifact_version != ARTIFACT_VERSION:
        raise ModelArtifactError(
            f"Unsupported artifact version {candidate.artifact_version}; "
            f"expected {ARTIFACT_VERSION}."
        )
    if not 0.0 <= candidate.threshold <= 1.0:
        raise ModelArtifactError("Artifact contains an invalid decision threshold.")
    return candidate


def _verify_manifest(directory: Path) -> None:
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ModelArtifactError(f"Artifact integrity manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_fingerprint(dataset: ValidatedDataset) -> str:
    digest = hashlib.sha256()
    digest.update("\0".join(dataset.feature_names).encode())
    feature_hash = pd.util.hash_pandas_object(dataset.features, index=True).to_numpy(dtype="uint64")
    target_hash = pd.util.hash_pandas_object(dataset.target, index=True).to_numpy(dtype="uint64")
    digest.update(cast(np.ndarray, feature_hash).tobytes())
    digest.update(cast(np.ndarray, target_hash).tobytes())
    return digest.hexdigest()


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
