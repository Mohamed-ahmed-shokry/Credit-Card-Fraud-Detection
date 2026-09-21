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


@dataclass(frozen=True)
class MultiWindowFeatureDrift:
    """Drift result for one model feature across dual observation windows."""

    feature: str
    short_psi: float
    long_psi: float
    velocity: float
    status: str


@dataclass(frozen=True)
class MultiWindowDriftReport:
    """Serializable multi-window drift report for short and long surveillance windows."""

    short_window_rows: int
    long_window_rows: int
    overall_status: str
    mean_short_psi: float
    max_short_psi: float
    mean_long_psi: float
    max_long_psi: float
    max_velocity: float
    features: tuple[MultiWindowFeatureDrift, ...]
    warning_at: float = STABLE_THRESHOLD
    drift_at: float = DRIFT_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report mapping."""
        return {
            "short_window_rows": self.short_window_rows,
            "long_window_rows": self.long_window_rows,
            "overall_status": self.overall_status,
            "mean_short_psi": self.mean_short_psi,
            "max_short_psi": self.max_short_psi,
            "mean_long_psi": self.mean_long_psi,
            "max_long_psi": self.max_long_psi,
            "max_velocity": self.max_velocity,
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


class StreamingProfile:
    """Incremental online distribution and quantile profiler for streaming surveillance."""

    def __init__(self, feature_edges: dict[str, list[float]]) -> None:
        if not feature_edges:
            raise DriftError("feature_edges must not be empty.")
        self.feature_edges: dict[str, list[float]] = {}
        self.counts: dict[str, np.ndarray] = {}
        self.total_counts: dict[str, int] = {}
        self.means: dict[str, float] = {}
        self.m2s: dict[str, float] = {}

        for feature, edges in feature_edges.items():
            if not isinstance(feature, str) or not feature:
                raise DriftError("Feature names must be non-empty strings.")
            interior = np.asarray(edges, dtype=float)
            if not np.isfinite(interior).all() or np.any(np.diff(interior) <= 0):
                raise DriftError(
                    f"Bin edges for feature {feature!r} must be finite and strictly increasing."
                )
            self.feature_edges[feature] = [float(e) for e in interior]
            self.counts[feature] = np.zeros(len(interior) + 1, dtype=np.int64)
            self.total_counts[feature] = 0
            self.means[feature] = 0.0
            self.m2s[feature] = 0.0

    @classmethod
    def from_reference_profile(
        cls, reference_profile: dict[str, dict[str, Any]]
    ) -> StreamingProfile:
        """Initialize a streaming profiler using bin edges from a reference profile."""
        if not reference_profile:
            raise DriftError("Reference profile must not be empty.")
        feature_edges: dict[str, list[float]] = {}
        for feature, baseline in reference_profile.items():
            edges = baseline.get("edges")
            if not isinstance(edges, list) or len(edges) < 2:
                raise DriftError(f"Invalid edges in reference profile for feature {feature!r}.")
            interior = edges[1:-1]
            feature_edges[feature] = [float(e) for e in interior]
        return cls(feature_edges)

    def update(self, batch: pd.DataFrame | list[dict[str, Any]]) -> None:
        """Update streaming distribution statistics with a batch of observations."""
        if isinstance(batch, list):
            if not batch:
                return
            frame = pd.DataFrame(batch)
        elif isinstance(batch, pd.DataFrame):
            if batch.empty:
                return
            frame = batch
        else:
            raise DriftError("Batch must be a pandas DataFrame or list of dicts.")

        column_lookup = _normalized_column_lookup(frame, context="Streaming")
        expected = set(self.feature_edges)
        provided = set(column_lookup)
        if not expected.issubset(provided):
            missing = sorted(expected - provided)
            raise DriftError(f"Streaming batch is missing expected features: {missing}")

        for feature, edges_list in self.feature_edges.items():
            values = _numeric_feature_values(frame, column_lookup[feature], context="Streaming")
            if not np.isfinite(values).all():
                raise DriftError(f"Streaming feature {feature!r} contains non-finite values.")

            full_edges = np.concatenate(([-np.inf], edges_list, [np.inf]))
            batch_counts, _ = np.histogram(values, bins=full_edges)
            self.counts[feature] += batch_counts

            n_b = len(values)
            if n_b > 0:
                n_a = self.total_counts[feature]
                n = n_a + n_b
                mean_b = float(np.mean(values))
                m2_b = float(np.sum((values - mean_b) ** 2))

                delta = mean_b - self.means[feature]
                self.means[feature] += delta * (n_b / n)
                self.m2s[feature] += m2_b + (delta**2) * (n_a * n_b / n)
                self.total_counts[feature] = n

    def to_reference_profile(self) -> dict[str, dict[str, Any]]:
        """Export accumulated statistics to reference profile schema."""
        profile: dict[str, dict[str, Any]] = {}
        for feature, edges in self.feature_edges.items():
            total = self.total_counts[feature]
            if total == 0:
                num_bins = len(edges) + 1
                proportions = [1.0 / num_bins] * num_bins
                std = 0.0
            else:
                proportions = [float(c / total) for c in self.counts[feature]]
                std = float(np.sqrt(self.m2s[feature] / total))

            profile[feature] = {
                "edges": [None, *edges, None],
                "proportions": proportions,
                "mean": float(self.means[feature]),
                "standard_deviation": std,
            }
        return profile

    def to_dict(self) -> dict[str, Any]:
        """Serialize full internal profiler state for checkpointing."""
        return {
            "features": {
                feature: {
                    "edges": self.feature_edges[feature],
                    "counts": [int(c) for c in self.counts[feature]],
                    "total_count": self.total_counts[feature],
                    "mean": float(self.means[feature]),
                    "m2": float(self.m2s[feature]),
                }
                for feature in self.feature_edges
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamingProfile:
        """Restore profiler state from a serialized checkpoint."""
        if not isinstance(data, dict) or "features" not in data:
            raise DriftError("Invalid profiler state dictionary: missing 'features'.")
        raw_features = data["features"]
        if not isinstance(raw_features, dict) or not raw_features:
            raise DriftError("Profiler state 'features' must be a non-empty mapping.")

        feature_edges = {
            feature: [float(e) for e in spec["edges"]]
            for feature, spec in raw_features.items()
        }
        instance = cls(feature_edges)
        for feature, spec in raw_features.items():
            instance.counts[feature] = np.asarray(spec["counts"], dtype=np.int64)
            instance.total_counts[feature] = int(spec["total_count"])
            instance.means[feature] = float(spec["mean"])
            instance.m2s[feature] = float(spec["m2"])
        return instance



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


def assess_multi_window_drift(
    reference_profile: dict[str, dict[str, Any]],
    features: pd.DataFrame,
    *,
    short_window_rows: int = 100,
    thresholds: object = None,
) -> MultiWindowDriftReport:
    """Compare recent (short window) and aggregate (long window) features with baseline.

    The short window evaluates the last `short_window_rows` transactions in
    `features`, while the long window evaluates the entire feature set.
    Drift velocity is calculated as `(short_psi - long_psi)` to reveal rapid
    distribution shifts before long-term metrics trip.
    """
    if short_window_rows < 2:
        raise DriftError("short_window_rows must be at least 2.")
    if len(features) < short_window_rows:
        raise DriftError(
            f"Current features rows ({len(features)}) is less than "
            f"short_window_rows ({short_window_rows})."
        )

    warning_at, drift_at = resolve_thresholds(thresholds)
    long_report = assess_drift(reference_profile, features, thresholds=thresholds)
    short_features = features.iloc[-short_window_rows:]
    short_report = assess_drift(reference_profile, short_features, thresholds=thresholds)

    short_lookup = {item.feature: item for item in short_report.features}
    long_lookup = {item.feature: item for item in long_report.features}

    results: list[MultiWindowFeatureDrift] = []
    for feature, long_item in long_lookup.items():
        short_item = short_lookup[feature]
        s_psi = short_item.psi
        l_psi = long_item.psi
        vel = s_psi - l_psi
        feat_status = _status(max(s_psi, l_psi), warning_at=warning_at, drift_at=drift_at)
        results.append(
            MultiWindowFeatureDrift(
                feature=feature,
                short_psi=s_psi,
                long_psi=l_psi,
                velocity=vel,
                status=feat_status,
            )
        )

    ordered = tuple(sorted(results, key=lambda item: item.short_psi, reverse=True))
    short_psis = [item.short_psi for item in ordered]
    long_psis = [item.long_psi for item in ordered]
    velocities = [item.velocity for item in ordered]

    overall_max_psi = max(short_report.max_psi, long_report.max_psi)
    overall_status = _status(overall_max_psi, warning_at=warning_at, drift_at=drift_at)

    return MultiWindowDriftReport(
        short_window_rows=short_window_rows,
        long_window_rows=len(features),
        overall_status=overall_status,
        mean_short_psi=float(np.mean(short_psis)),
        max_short_psi=float(np.max(short_psis)),
        mean_long_psi=float(np.mean(long_psis)),
        max_long_psi=float(np.max(long_psis)),
        max_velocity=float(np.max(velocities)),
        features=ordered,
        warning_at=warning_at,
        drift_at=drift_at,
    )



_SURVEILLANCE_SEVERITY = {"stable": 0, "warning": 1, "drifted": 2}


def surveillance_tripped(overall_status: str, fail_on: str | None) -> bool:
    """Report whether a drift status meets a surveillance cutoff.

    `None` disables surveillance. Unknown levels raise so scheduled jobs fail
    loudly instead of silently watching nothing.
    """
    if fail_on is None:
        return False
    try:
        return _SURVEILLANCE_SEVERITY[overall_status] >= _SURVEILLANCE_SEVERITY[fail_on]
    except KeyError as exc:
        raise DriftError(f"Unknown drift surveillance level: {exc}") from exc


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
