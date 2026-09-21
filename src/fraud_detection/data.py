"""Dataset loading, validation, and deterministic demo-data generation."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

DEFAULT_TARGET = "Class"


class DataValidationError(ValueError):
    """Raised when a dataset cannot safely be used for binary classification."""


@dataclass(frozen=True)
class ValidatedDataset:
    """Validated features and binary target."""

    features: pd.DataFrame
    target: pd.Series

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return feature names in model input order."""
        return tuple(self.features.columns)


def load_csv(path: Path | str, *, target_column: str = DEFAULT_TARGET) -> ValidatedDataset:
    """Load and validate a CSV dataset."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise DataValidationError(f"Dataset does not exist or is not a file: {csv_path}")

    try:
        with csv_path.open(encoding="utf-8-sig", newline="") as file_handle:
            header = next(csv.reader(file_handle), None)
        if header is not None:
            _reject_duplicate_names(header, context="CSV columns")
        frame = pd.read_csv(csv_path)
    except (
        OSError,
        csv.Error,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
    ) as exc:
        raise DataValidationError(f"Could not read dataset {csv_path}: {exc}") from exc

    return validate_frame(frame, target_column=target_column)


def validate_frame(
    frame: pd.DataFrame,
    *,
    target_column: str = DEFAULT_TARGET,
) -> ValidatedDataset:
    """Validate a tabular binary-classification dataset.

    Features must be uniquely named, numeric, finite, and non-null. The target
    must contain exactly the integer labels 0 and 1.
    """
    if frame.empty:
        raise DataValidationError("Dataset is empty.")
    if not frame.columns.is_unique:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise DataValidationError(f"Duplicate columns are not allowed: {duplicates}")
    if target_column not in frame.columns:
        raise DataValidationError(f"Target column {target_column!r} is missing.")

    features = frame.drop(columns=target_column).copy()
    if features.shape[1] == 0:
        raise DataValidationError("Dataset must contain at least one feature column.")
    feature_names = [str(column) for column in features.columns]
    if any(not name for name in feature_names):
        raise DataValidationError("Feature names must not be empty.")
    _reject_duplicate_names(feature_names, context="Feature names after string conversion")
    features.columns = feature_names

    non_numeric = features.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric:
        raise DataValidationError(f"All features must be numeric; invalid columns: {non_numeric}")
    if features.isna().any().any():
        invalid = features.columns[features.isna().any()].tolist()
        raise DataValidationError(f"Feature columns contain missing values: {invalid}")

    feature_values = features.to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        invalid = features.columns[~np.isfinite(feature_values).all(axis=0)].tolist()
        raise DataValidationError(f"Feature columns contain infinite values: {invalid}")

    raw_target = frame[target_column]
    if raw_target.isna().any():
        raise DataValidationError("Target column contains missing values.")
    if not pd.api.types.is_numeric_dtype(raw_target):
        raise DataValidationError("Target column must be numeric with labels 0 and 1.")

    unique_labels = set(raw_target.unique().tolist())
    if unique_labels != {0, 1}:
        raise DataValidationError(
            f"Target must contain exactly the labels 0 and 1; found {sorted(unique_labels)}"
        )

    target = raw_target.astype("int8").copy()
    target.name = target_column
    return ValidatedDataset(features=features, target=target)


def _reject_duplicate_names(names: list[str], *, context: str) -> None:
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise DataValidationError(f"{context} must be unique; duplicates: {duplicates}")


def generate_synthetic_data(
    *,
    rows: int = 5_000,
    fraud_rate: float = 0.02,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate realistic-enough demo data with the public dataset's schema.

    This data is intended for smoke tests and demonstrations, not benchmarking.
    """
    if rows < 200:
        raise ValueError("rows must be at least 200")
    if not 0.005 <= fraud_rate <= 0.5:
        raise ValueError("fraud_rate must be between 0.005 and 0.5")

    features, target = make_classification(
        n_samples=rows,
        n_features=28,
        n_informative=12,
        n_redundant=8,
        n_repeated=0,
        n_classes=2,
        weights=[1.0 - fraud_rate, fraud_rate],
        class_sep=1.5,
        flip_y=min(0.002, fraud_rate / 10),
        random_state=random_state,
    )
    rng = np.random.default_rng(random_state)
    frame = pd.DataFrame(features, columns=[f"V{number}" for number in range(1, 29)])
    frame.insert(0, "Time", np.sort(rng.uniform(0, 172_800, size=rows)))
    frame["Amount"] = rng.lognormal(mean=3.2, sigma=1.1, size=rows).round(2)
    frame[DEFAULT_TARGET] = target.astype("int8")
    return frame


def inject_drift(
    frame: pd.DataFrame,
    *,
    target_features: list[str] | None = None,
    mean_offset: float = 0.0,
    variance_scale: float = 1.0,
    anomaly_fraction: float = 0.0,
    anomaly_scale: float = 5.0,
    sample_fraction: float = 1.0,
    target_column: str = DEFAULT_TARGET,
    random_state: int = 42,
) -> pd.DataFrame:
    """Inject synthetic distribution shifts and anomalies for chaos and surveillance testing.

    Args:
        frame: Input tabular dataframe.
        target_features: List of feature names to drift; None drifts all non-target columns.
        mean_offset: Additive shift applied to feature distributions.
        variance_scale: Multiplicative scale applied to feature spread around the mean.
        anomaly_fraction: Fraction of rows to inject with extreme anomaly spikes.
        anomaly_scale: Multiplier for anomaly magnitude (in units of feature standard deviation).
        sample_fraction: Fraction of recent tail rows to modify (1.0 modifies all rows).
        target_column: Target column name to exclude from drift injection.
        random_state: Seed for reproducible noise generation.
    """
    if frame.empty:
        raise DataValidationError("Cannot inject drift into an empty dataset.")
    if variance_scale < 0.0:
        raise DataValidationError("variance_scale must be non-negative.")
    if not 0.0 <= anomaly_fraction <= 1.0:
        raise DataValidationError("anomaly_fraction must be between 0.0 and 1.0.")
    if not 0.0 < sample_fraction <= 1.0:
        raise DataValidationError("sample_fraction must be in (0.0, 1.0].")

    result = frame.copy()
    available_features = [col for col in result.columns if col != target_column]

    if target_features is not None:
        if not target_features:
            raise DataValidationError("target_features must not be empty if specified.")
        missing = [f for f in target_features if f not in result.columns]
        if missing:
            raise DataValidationError(f"Target features not found in dataset: {missing}")
        features_to_drift = target_features
    else:
        features_to_drift = available_features

    rng = np.random.default_rng(random_state)
    n_rows = len(result)
    n_affected = max(1, round(n_rows * sample_fraction))
    affected_idx = result.index[-n_affected:]

    for feature in features_to_drift:
        col_values = result.loc[affected_idx, feature].to_numpy(dtype=float)
        mean_val = float(np.mean(col_values))
        std_val = float(np.std(col_values))
        if not np.isfinite(std_val) or std_val == 0.0:
            std_val = 1.0

        shifted = (col_values - mean_val) * variance_scale + mean_val + mean_offset

        if anomaly_fraction > 0.0:
            n_anomalies = max(1, round(len(col_values) * anomaly_fraction))
            anomaly_pos = rng.choice(len(col_values), size=n_anomalies, replace=False)
            directions = rng.choice([-1.0, 1.0], size=n_anomalies)
            shifted[anomaly_pos] += directions * anomaly_scale * std_val

        result.loc[affected_idx, feature] = shifted

    return result

