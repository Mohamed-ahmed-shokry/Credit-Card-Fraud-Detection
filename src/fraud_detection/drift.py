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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report mapping."""
        return {
            "rows": self.rows,
            "overall_status": self.overall_status,
            "mean_psi": self.mean_psi,
            "max_psi": self.max_psi,
            "features": [asdict(item) for item in self.features],
        }


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

    profile: dict[str, dict[str, Any]] = {}
    for column in features.columns:
        values = features[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DriftError(f"Reference feature {column!r} contains non-finite values.")

        quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
        interior_edges = np.unique(np.quantile(values, quantiles))
        edges = np.concatenate(([-np.inf], interior_edges, [np.inf]))
        counts, _ = np.histogram(values, bins=edges)
        proportions = counts / counts.sum()
        profile[str(column)] = {
            "edges": [None, *[float(edge) for edge in interior_edges], None],
            "proportions": [float(value) for value in proportions],
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
        }
    return profile


def assess_drift(
    reference_profile: dict[str, dict[str, Any]],
    features: pd.DataFrame,
) -> DriftReport:
    """Compare current features with the training profile using PSI."""
    if features.empty:
        raise DriftError("Current features must not be empty.")

    expected = set(reference_profile)
    provided = set(features.columns)
    if expected != provided:
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        raise DriftError(
            f"Current feature schema does not match the reference: "
            f"missing={missing}, unexpected={unexpected}"
        )

    results: list[FeatureDrift] = []
    for feature, baseline in reference_profile.items():
        values = features[feature].to_numpy(dtype=float)
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
            edges = np.asarray(
                [-np.inf, *serialized_edges[1:-1], np.inf],
                dtype=float,
            )
            expected_proportions = np.asarray(baseline["proportions"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise DriftError(f"Reference profile for {feature!r} is invalid.") from exc

        actual_counts, _ = np.histogram(values, bins=edges)
        actual_proportions = actual_counts / actual_counts.sum()
        if actual_proportions.shape != expected_proportions.shape:
            raise DriftError(f"Reference profile for {feature!r} has inconsistent bins.")
        psi = float(
            np.sum(
                (actual_proportions - expected_proportions)
                * np.log((actual_proportions + _EPSILON) / (expected_proportions + _EPSILON))
            )
        )
        results.append(FeatureDrift(feature=feature, psi=psi, status=_status(psi)))

    ordered = tuple(sorted(results, key=lambda item: item.psi, reverse=True))
    psi_values = cast(np.ndarray, np.asarray([item.psi for item in ordered], dtype=float))
    maximum = float(np.max(psi_values))
    return DriftReport(
        rows=len(features),
        overall_status=_status(maximum),
        mean_psi=float(np.mean(psi_values)),
        max_psi=maximum,
        features=ordered,
    )


def _status(psi: float) -> str:
    if psi < STABLE_THRESHOLD:
        return "stable"
    if psi < DRIFT_THRESHOLD:
        return "warning"
    return "drifted"
