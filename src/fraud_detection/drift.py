"""Feature-distribution profiling and Population Stability Index reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

STABLE_THRESHOLD = 0.1
DRIFT_THRESHOLD = 0.25
_EPSILON = 1e-6


class DriftError(ValueError):
    """Raised when a drift report cannot be calculated."""


@dataclass(frozen=True)
class FeatureDrift:
    """Drift result for one model feature."""

    feature: str
    psi: float
    status: str


@dataclass(frozen=True)
class DriftReport:
    """Serializable drift report for a scored dataset."""

    rows: int
    overall_status: str
    mean_psi: float
    max_psi: float
    features: tuple[FeatureDrift, ...]
    warning_at: float = STABLE_THRESHOLD
    drift_at: float = DRIFT_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report mapping."""
        return {
            "rows": self.rows,
            "overall_status": self.overall_status,
            "mean_psi": self.mean_psi,
            "max_psi": self.max_psi,
            "thresholds": {"warning_at": self.warning_at, "drift_at": self.drift_at},
            "features": [asdict(item) for item in self.features],
        }


def default_thresholds() -> dict[str, float]:
    """Return the reference PSI cutoffs persisted with each model card."""
    return {"warning_at": STABLE_THRESHOLD, "drift_at": DRIFT_THRESHOLD}


def resolve_thresholds(source: object) -> tuple[float, float]:
    """Validate persisted or caller-supplied PSI cutoffs.

    `None` selects the reference defaults so artifacts trained before cutoffs
    were persisted keep working. Anything else must map `warning_at` and
    `drift_at` to finite numbers with `0 <= warning_at < drift_at`.
    """
    if source is None:
        return (STABLE_THRESHOLD, DRIFT_THRESHOLD)
    if not isinstance(source, dict):
        raise DriftError("Drift thresholds must be a mapping.")
    try:
        warning_at = float(source["warning_at"])
        drift_at = float(source["drift_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DriftError(
            "Drift thresholds must define numeric 'warning_at' and 'drift_at'."
        ) from exc
    if not np.isfinite(warning_at) or not np.isfinite(drift_at) or not 0.0 <= warning_at < drift_at:
        raise DriftError("Drift thresholds must satisfy 0 <= warning_at < drift_at.")
    return (warning_at, drift_at)


def build_reference_profile(
    features: pd.DataFrame,
    *,
    bins: int = 10,
) -> dict[str, dict[str, Any]]:
    """Build compact per-feature histogram baselines from training data."""
    if features.empty:
        raise DriftError("Reference features must not be empty.")
    if bins < 2 or bins > 50:
        raise DriftError("bins must be between 2 and 50")

    column_lookup = _normalized_column_lookup(features, context="Reference")
    profile: dict[str, dict[str, Any]] = {}
    for feature, column in column_lookup.items():
        values = _numeric_feature_values(features, column, context="Reference")
        if not np.isfinite(values).all():
            raise DriftError(f"Reference feature {feature!r} contains non-finite values.")

        quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
        with np.errstate(invalid="ignore", over="ignore"):
            interior_edges = np.unique(np.quantile(values, quantiles))
            mean = float(np.mean(values))
            standard_deviation = float(np.std(values))
        if (
            not np.isfinite(interior_edges).all()
            or not np.isfinite(mean)
            or not np.isfinite(standard_deviation)
        ):
            raise DriftError(
                f"Reference feature {feature!r} cannot be profiled without numeric overflow."
            )
        edges = np.concatenate(([-np.inf], interior_edges, [np.inf]))
        counts, _ = np.histogram(values, bins=edges)
        proportions = counts / counts.sum()
        profile[feature] = {
            "edges": [None, *[float(edge) for edge in interior_edges], None],
            "proportions": [float(value) for value in proportions],
            "mean": mean,
            "standard_deviation": standard_deviation,
        }
    return profile


def assess_drift(
    reference_profile: dict[str, dict[str, Any]],
    features: pd.DataFrame,
    *,
    thresholds: object = None,
) -> DriftReport:
    """Compare current features with the training profile using PSI.

    `thresholds` accepts the persisted `drift_thresholds` model-card mapping
    (or `None` for the reference defaults) so alerting cutoffs travel with the
    model instead of living only in code.
    """
    warning_at, drift_at = resolve_thresholds(thresholds)
    if features.empty:
        raise DriftError("Current features must not be empty.")

    if not reference_profile or any(
        not isinstance(feature, str) or not feature for feature in reference_profile
    ):
        raise DriftError("Reference profile must use non-empty string feature names.")
    column_lookup = _normalized_column_lookup(features, context="Current")
    expected = set(reference_profile)
    provided = set(column_lookup)
    if expected != provided:
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        raise DriftError(
            f"Current feature schema does not match the reference: "
            f"missing={missing}, unexpected={unexpected}"
        )

    results: list[FeatureDrift] = []
    for feature, baseline in reference_profile.items():
        values = _numeric_feature_values(
            features,
            column_lookup[feature],
            context="Current",
        )
        if not np.isfinite(values).all():
            raise DriftError(f"Current feature {feature!r} contains non-finite values.")
        try:
            serialized_edges = baseline["edges"]
            if (
                not isinstance(serialized_edges, list)
                or len(serialized_edges) < 2
                or serialized_edges[0] is not None
                or serialized_edges[-1] is not None
            ):
                raise ValueError("invalid open-ended bins")
            interior_edges: np.ndarray = np.asarray(
                serialized_edges[1:-1],
                dtype=float,
            )
            expected_proportions = np.asarray(baseline["proportions"], dtype=float)
            if not np.isfinite(interior_edges).all() or np.any(np.diff(interior_edges) <= 0):
                raise ValueError("bin edges must be finite and strictly increasing")
            if (
                expected_proportions.ndim != 1
                or expected_proportions.shape != (len(serialized_edges) - 1,)
                or not np.isfinite(expected_proportions).all()
                or np.any(expected_proportions < 0)
                or not np.isclose(float(expected_proportions.sum()), 1.0)
            ):
                raise ValueError("bin proportions must be finite, non-negative, and sum to 1")
            edges = np.concatenate(([-np.inf], interior_edges, [np.inf]))
        except (KeyError, TypeError, ValueError) as exc:
            raise DriftError(f"Reference profile for {feature!r} is invalid.") from exc

        actual_counts, _ = np.histogram(values, bins=edges)
        actual_proportions = actual_counts / actual_counts.sum()
        psi = max(
            0.0,
            float(
                np.sum(
                    (actual_proportions - expected_proportions)
                    * np.log((actual_proportions + _EPSILON) / (expected_proportions + _EPSILON))
                )
            ),
        )
        results.append(
            FeatureDrift(
                feature=feature,
                psi=psi,
                status=_status(psi, warning_at=warning_at, drift_at=drift_at),
            )
        )

    ordered = tuple(sorted(results, key=lambda item: item.psi, reverse=True))
    psi_values = np.asarray([item.psi for item in ordered], dtype=float)
    maximum = float(np.max(psi_values))
    return DriftReport(
        rows=len(features),
        overall_status=_status(maximum, warning_at=warning_at, drift_at=drift_at),
        mean_psi=float(np.mean(psi_values)),
        max_psi=maximum,
        features=ordered,
        warning_at=warning_at,
        drift_at=drift_at,
    )


def _status(
    psi: float, *, warning_at: float = STABLE_THRESHOLD, drift_at: float = DRIFT_THRESHOLD
) -> str:
    if psi < warning_at:
        return "stable"
    if psi < drift_at:
        return "warning"
    return "drifted"


def _normalized_column_lookup(
    features: pd.DataFrame,
    *,
    context: str,
) -> dict[str, object]:
    if not features.columns.is_unique:
        raise DriftError(f"{context} feature names must be unique.")
    names = [str(column) for column in features.columns]
    if any(not name for name in names):
        raise DriftError(f"{context} feature names must not be empty.")
    if len(set(names)) != len(names):
        raise DriftError(f"{context} feature names must remain unique after string conversion.")
    return dict(zip(names, features.columns, strict=True))


def _numeric_feature_values(
    features: pd.DataFrame,
    column: object,
    *,
    context: str,
) -> np.ndarray:
    try:
        return cast(np.ndarray, features[column].to_numpy(dtype=float))
    except (TypeError, ValueError) as exc:
        raise DriftError(f"{context} feature {column!r} must be numeric.") from exc
