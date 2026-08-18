from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import (
    DataValidationError,
    generate_synthetic_data,
    load_csv,
    validate_frame,
)


def test_generate_synthetic_data_is_deterministic_and_valid() -> None:
    first = generate_synthetic_data(rows=500, fraud_rate=0.05, random_state=7)
    second = generate_synthetic_data(rows=500, fraud_rate=0.05, random_state=7)

    pd.testing.assert_frame_equal(first, second)
    dataset = validate_frame(first)
    assert dataset.features.shape == (500, 30)
    assert dataset.target.sum() > 0
    assert dataset.feature_names[:2] == ("Time", "V1")
    assert dataset.feature_names[-1] == "Amount"


@pytest.mark.parametrize(
    ("rows", "fraud_rate", "message"),
    [
        (199, 0.02, "rows must be at least 200"),
        (500, 0.001, "fraud_rate must be between"),
        (500, 0.6, "fraud_rate must be between"),
    ],
)
def test_generate_synthetic_data_rejects_invalid_options(
    rows: int,
    fraud_rate: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        generate_synthetic_data(rows=rows, fraud_rate=fraud_rate)


def test_load_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "transactions.csv"
    frame = pd.DataFrame({"amount": [10.0, 20.0], "Class": [0, 1]})
    frame.to_csv(path, index=False)

    dataset = load_csv(path)

    pd.testing.assert_frame_equal(dataset.features, frame[["amount"]])
    assert dataset.target.tolist() == [0, 1]


def test_load_csv_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="does not exist"):
        load_csv(tmp_path / "missing.csv")


def test_load_csv_rejects_duplicate_headers_before_pandas_mangles_them(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate_headers.csv"
    path.write_text("amount,amount,Class\n10,20,0\n30,40,1\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="CSV columns must be unique"):
        load_csv(path)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame(), "empty"),
        (pd.DataFrame({"x": [1, 2]}), "Target column"),
        (pd.DataFrame({"Class": [0, 1]}), "at least one feature"),
        (pd.DataFrame({"x": ["a", "b"], "Class": [0, 1]}), "must be numeric"),
        (pd.DataFrame({"x": [1.0, np.nan], "Class": [0, 1]}), "missing values"),
        (pd.DataFrame({"x": [1.0, 2.0], "Class": [0, np.nan]}), "Target column contains"),
        (pd.DataFrame({"x": [1.0, np.inf], "Class": [0, 1]}), "infinite values"),
        (pd.DataFrame({"x": [1, 2], "Class": [0, 0]}), "exactly the labels"),
        (pd.DataFrame({"x": [1, 2], "Class": ["clean", "fraud"]}), "must be numeric"),
    ],
)
def test_validate_frame_rejects_unsafe_data(frame: pd.DataFrame, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        validate_frame(frame)


def test_validate_frame_rejects_duplicate_columns() -> None:
    frame = pd.DataFrame([[1, 2, 0], [3, 4, 1]], columns=["x", "x", "Class"])

    with pytest.raises(DataValidationError, match="Duplicate"):
        validate_frame(frame)


def test_validate_frame_rejects_feature_names_that_collide_as_strings() -> None:
    frame = pd.DataFrame([[1, 2, 0], [3, 4, 1]], columns=[1, "1", "Class"])

    with pytest.raises(DataValidationError, match="after string conversion"):
        validate_frame(frame)


def test_validate_frame_rejects_empty_feature_names() -> None:
    frame = pd.DataFrame([[1, 0], [2, 1]], columns=["", "Class"])

    with pytest.raises(DataValidationError, match="must not be empty"):
        validate_frame(frame)
