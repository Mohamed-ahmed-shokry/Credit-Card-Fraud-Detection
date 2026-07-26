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
    except (OSError, csv.Error, pd.errors.ParserError, UnicodeDecodeError) as exc:
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
