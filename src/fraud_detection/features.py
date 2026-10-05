"""Feature store abstractions, point-in-time joins, and training-serving skew surveillance."""

from __future__ import annotations

import bisect
import json
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


class FeatureStoreError(ValueError):
    """Raised when feature store operations, schemas, or point-in-time joins fail."""


class FeatureType(StrEnum):
    """Supported data types for stored feature values."""

    FLOAT = "float"
    INT = "int"
    STRING = "string"
    BOOL = "bool"


class SkewStatus(StrEnum):
    """Operational drift status for training-serving feature skew."""

    STABLE = "stable"
    WARNING = "warning"
    DRIFTED = "drifted"


@dataclass(frozen=True)
class FeatureDefinition:
    """Specification for an individual feature within a feature view."""

    name: str
    feature_type: FeatureType | str = FeatureType.FLOAT
    description: str = ""
    default_value: Any = 0.0

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise FeatureStoreError("Feature name must be a non-empty string.")
        if not isinstance(self.feature_type, FeatureType):
            try:
                object.__setattr__(self, "feature_type", FeatureType(self.feature_type))
            except ValueError as err:
                raise FeatureStoreError(f"Unsupported feature type: {self.feature_type}") from err

    def validate_and_cast(self, val: Any) -> Any:
        """Validate and cast a raw value according to feature type."""
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return self.default_value

        try:
            if self.feature_type == FeatureType.FLOAT:
                float_val = float(val)
                if math.isnan(float_val) or math.isinf(float_val):
                    return self.default_value
                return float_val
            if self.feature_type == FeatureType.INT:
                return int(val)
            if self.feature_type == FeatureType.BOOL:
                if isinstance(val, str):
                    return val.strip().lower() in ("true", "1", "yes", "t")
                return bool(val)
            return str(val)
        except (ValueError, TypeError) as err:
            msg = (
                f"Failed to cast value {val!r} to feature '{self.name}' "
                f"({self.feature_type}): {err}"
            )
            raise FeatureStoreError(msg) from err

    def to_dict(self) -> dict[str, Any]:
        """Convert feature definition to serializable dictionary."""
        ft_val = (
            self.feature_type.value
            if isinstance(self.feature_type, FeatureType)
            else str(self.feature_type)
        )
        return {
            "name": self.name,
            "feature_type": ft_val,
            "description": self.description,
            "default_value": self.default_value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeatureDefinition:
        """Construct feature definition from dictionary."""
        return cls(
            name=str(data["name"]),
            feature_type=FeatureType(data.get("feature_type", FeatureType.FLOAT)),
            description=str(data.get("description", "")),
            default_value=data.get("default_value", 0.0),
        )


@dataclass(frozen=True)
class FeatureView:
    """Group of co-located features associated with an entity key."""

    name: str
    entity_key: str
    features: tuple[FeatureDefinition, ...]
    timestamp_col: str = "timestamp"
    ttl_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise FeatureStoreError("FeatureView name must be a non-empty string.")
        if not self.entity_key or not isinstance(self.entity_key, str):
            raise FeatureStoreError("FeatureView entity_key must be a non-empty string.")
        if not self.features:
            raise FeatureStoreError(f"FeatureView '{self.name}' must define at least one feature.")
        if self.ttl_seconds is not None and (
            not isinstance(self.ttl_seconds, (int, float))
            or math.isnan(self.ttl_seconds)
            or self.ttl_seconds <= 0
        ):
            raise FeatureStoreError(
                f"FeatureView ttl_seconds must be positive, got {self.ttl_seconds}"
            )

        names = [f.name for f in self.features]
        if len(names) != len(set(names)):
            raise FeatureStoreError(
                f"FeatureView '{self.name}' contains duplicate feature names: {names}"
            )

    def get_feature(self, name: str) -> FeatureDefinition | None:
        """Find feature definition by name."""
        for feat in self.features:
            if feat.name == name:
                return feat
        return None

    def feature_names(self) -> tuple[str, ...]:
        """Return tuple of all feature names in this view."""
        return tuple(f.name for f in self.features)

    def get_default_values(self) -> dict[str, Any]:
        """Return mapping of feature names to their default values."""
        return {f.name: f.default_value for f in self.features}

    def to_dict(self) -> dict[str, Any]:
        """Convert view definition to serializable dictionary."""
        return {
            "name": self.name,
            "entity_key": self.entity_key,
            "features": [f.to_dict() for f in self.features],
            "timestamp_col": self.timestamp_col,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeatureView:
        """Construct feature view from dictionary."""
        features_raw = data.get("features", [])
        features = tuple(FeatureDefinition.from_dict(f) for f in features_raw)
        return cls(
            name=str(data["name"]),
            entity_key=str(data["entity_key"]),
            features=features,
            timestamp_col=str(data.get("timestamp_col", "timestamp")),
            ttl_seconds=float(data["ttl_seconds"]) if data.get("ttl_seconds") is not None else None,
        )


@dataclass(frozen=True)
class FeatureSnapshot:
    """Historical point-in-time value snapshot for an entity."""

    entity_id: str
    timestamp: float
    values: dict[str, Any]

    def __post_init__(self) -> None:
        if not str(self.entity_id):
            raise FeatureStoreError("FeatureSnapshot entity_id must not be empty.")
        if (
            not isinstance(self.timestamp, (int, float))
            or math.isnan(self.timestamp)
            or math.isinf(self.timestamp)
        ):
            raise FeatureStoreError(f"Invalid timestamp in FeatureSnapshot: {self.timestamp}")

    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "entity_id": str(self.entity_id),
            "timestamp": float(self.timestamp),
            "values": dict(self.values),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeatureSnapshot:
        """Construct snapshot from dictionary."""
        return cls(
            entity_id=str(data["entity_id"]),
            timestamp=float(data["timestamp"]),
            values=dict(data.get("values", {})),
        )


@dataclass(frozen=True)
class FeatureLookupResult:
    """Result of an online or point-in-time feature lookup."""

    entity_key: str
    entity_id: str
    found: bool
    timestamp: float | None
    values: dict[str, Any]
    is_stale: bool = False
    view_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert lookup result to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class PointInTimeJoinSummary:
    """Diagnostic metrics from a point-in-time join operation."""

    total_observations: int
    matched_observations: int
    defaulted_observations: int
    stale_observations: int
    enriched_features: tuple[str, ...]
    match_rate: float
    views_applied: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "total_observations": self.total_observations,
            "matched_observations": self.matched_observations,
            "defaulted_observations": self.defaulted_observations,
            "stale_observations": self.stale_observations,
            "enriched_features": list(self.enriched_features),
            "match_rate": round(self.match_rate, 4),
            "views_applied": list(self.views_applied),
        }


class FeatureStoreProtocol(Protocol):
    """Protocol defining the feature store interface."""

    def get_view(self, name: str) -> FeatureView | None:
        """Retrieve view definition by name."""
        ...

    def list_views(self) -> list[str]:
        """List all registered view names."""
        ...

    def lookup_online(
        self,
        entity_key: str,
        entity_id: str,
        view_name: str | None = None,
        as_of_time: float | None = None,
    ) -> FeatureLookupResult:
        """Lookup current or as-of features for an entity."""
        ...

    def put_snapshot(self, view_name: str, snapshot: FeatureSnapshot) -> None:
        """Record a feature snapshot into the store."""
        ...

    def get_historical_snapshots(
        self,
        view_name: str,
        entity_id: str | None = None,
    ) -> list[FeatureSnapshot]:
        """Retrieve historical snapshots for a view."""
        ...

    def get_stats(self) -> dict[str, Any]:
        """Return diagnostic metrics about stored views and entities."""
        ...


class InMemoryFeatureStore:
    """Thread-safe in-memory feature store supporting point-in-time lookups."""

    def __init__(self, views: Sequence[FeatureView] | None = None) -> None:
        self._lock = threading.RLock()
        self._views: dict[str, FeatureView] = {}
        # Key: (view_name, entity_id) -> list of FeatureSnapshot, sorted by timestamp
        self._snapshots: dict[tuple[str, str], list[FeatureSnapshot]] = {}
        # Diagnostic counters
        self._lookup_count: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        if views:
            for v in views:
                self.add_view(v)

    def add_view(self, view: FeatureView) -> None:
        """Register a feature view."""
        with self._lock:
            if view.name in self._views:
                raise FeatureStoreError(f"FeatureView '{view.name}' already registered.")
            self._views[view.name] = view

    def get_view(self, name: str) -> FeatureView | None:
        """Retrieve view definition by name."""
        with self._lock:
            return self._views.get(name)

    def list_views(self) -> list[str]:
        """List all registered view names."""
        with self._lock:
            return sorted(self._views.keys())

    def put_snapshot(self, view_name: str, snapshot: FeatureSnapshot) -> None:
        """Record a validated feature snapshot into the store."""
        with self._lock:
            view = self._views.get(view_name)
            if view is None:
                raise FeatureStoreError(f"View '{view_name}' is not registered.")

            # Validate and cast feature values
            cast_values: dict[str, Any] = {}
            for feat in view.features:
                raw_val = snapshot.values.get(feat.name)
                cast_values[feat.name] = feat.validate_and_cast(raw_val)

            validated_snapshot = FeatureSnapshot(
                entity_id=str(snapshot.entity_id),
                timestamp=float(snapshot.timestamp),
                values=cast_values,
            )

            key = (view_name, validated_snapshot.entity_id)
            if key not in self._snapshots:
                self._snapshots[key] = []

            lst = self._snapshots[key]
            # Maintain sorted order by timestamp
            timestamps = [s.timestamp for s in lst]
            idx = bisect.bisect_right(timestamps, validated_snapshot.timestamp)
            lst.insert(idx, validated_snapshot)

    def put_snapshots(self, view_name: str, snapshots: Sequence[FeatureSnapshot]) -> None:
        """Bulk insert multiple snapshots."""
        for s in snapshots:
            self.put_snapshot(view_name, s)

    def lookup_online(
        self,
        entity_key: str,
        entity_id: str,
        view_name: str | None = None,
        as_of_time: float | None = None,
    ) -> FeatureLookupResult:
        """Perform an online or point-in-time entity lookup.

        If as_of_time is provided, finds the latest snapshot with timestamp <= as_of_time.
        Strictly guarantees that no snapshot with timestamp > as_of_time is returned.
        """
        with self._lock:
            self._lookup_count += 1
            str_entity_id = str(entity_id)

            # Resolve view
            target_view: FeatureView | None = None
            if view_name is not None:
                target_view = self._views.get(view_name)
                if target_view is None:
                    raise FeatureStoreError(f"View '{view_name}' not found.")
                if target_view.entity_key != entity_key:
                    msg = (
                        f"View '{view_name}' has entity_key '{target_view.entity_key}', "
                        f"expected '{entity_key}'."
                    )
                    raise FeatureStoreError(msg)
            else:
                for v in self._views.values():
                    if v.entity_key == entity_key:
                        target_view = v
                        break

            if target_view is None:
                self._cache_misses += 1
                return FeatureLookupResult(
                    entity_key=entity_key,
                    entity_id=str_entity_id,
                    found=False,
                    timestamp=None,
                    values={},
                    is_stale=False,
                    view_name=None,
                )

            key = (target_view.name, str_entity_id)
            snapshots = self._snapshots.get(key)
            defaults = target_view.get_default_values()

            if not snapshots:
                self._cache_misses += 1
                return FeatureLookupResult(
                    entity_key=entity_key,
                    entity_id=str_entity_id,
                    found=False,
                    timestamp=None,
                    values=defaults,
                    is_stale=False,
                    view_name=target_view.name,
                )

            # Point-in-time time-travel resolution: latest snapshot where t <= as_of_time
            if as_of_time is not None:
                timestamps = [s.timestamp for s in snapshots]
                idx = bisect.bisect_right(timestamps, as_of_time) - 1
                if idx < 0:
                    # All available snapshots are in the future relative to as_of_time!
                    self._cache_misses += 1
                    return FeatureLookupResult(
                        entity_key=entity_key,
                        entity_id=str_entity_id,
                        found=False,
                        timestamp=None,
                        values=defaults,
                        is_stale=False,
                        view_name=target_view.name,
                    )
                best_snapshot = snapshots[idx]
            else:
                best_snapshot = snapshots[-1]

            # Check staleness against TTL
            is_stale = False
            if target_view.ttl_seconds is not None:
                ref_time = as_of_time if as_of_time is not None else best_snapshot.timestamp
                staleness = ref_time - best_snapshot.timestamp
                if staleness > target_view.ttl_seconds:
                    is_stale = True

            self._cache_hits += 1
            merged_values = dict(defaults)
            merged_values.update(best_snapshot.values)

            return FeatureLookupResult(
                entity_key=entity_key,
                entity_id=str_entity_id,
                found=True,
                timestamp=best_snapshot.timestamp,
                values=merged_values,
                is_stale=is_stale,
                view_name=target_view.name,
            )

    def get_historical_snapshots(
        self,
        view_name: str,
        entity_id: str | None = None,
    ) -> list[FeatureSnapshot]:
        """Retrieve copies of stored historical snapshots."""
        with self._lock:
            result: list[FeatureSnapshot] = []
            if entity_id is not None:
                key = (view_name, str(entity_id))
                return list(self._snapshots.get(key, []))

            for (v_name, _), s_list in self._snapshots.items():
                if v_name == view_name:
                    result.extend(s_list)
            return sorted(result, key=lambda s: s.timestamp)

    def get_stats(self) -> dict[str, Any]:
        """Return diagnostic metrics about stored views and entities."""
        with self._lock:
            entities = {eid for (_, eid) in self._snapshots}
            total_snapshots = sum(len(lst) for lst in self._snapshots.values())
            hit_rate = (
                round(self._cache_hits / self._lookup_count, 4) if self._lookup_count > 0 else 0.0
            )
            return {
                "views_count": len(self._views),
                "views": list(self._views.keys()),
                "total_entities": len(entities),
                "total_snapshots": total_snapshots,
                "lookup_count": self._lookup_count,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_rate": hit_rate,
            }


class FileFeatureStore(InMemoryFeatureStore):
    """File-backed feature store reading and saving state from JSON bundles."""

    def save_to_file(self, path: Path | str) -> None:
        """Serialize registered views and snapshots to JSON file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            data = {
                "views": [v.to_dict() for v in self._views.values()],
                "snapshots": [
                    {
                        "view_name": v_name,
                        "snapshot": s.to_dict(),
                    }
                    for (v_name, _), s_list in self._snapshots.items()
                    for s in s_list
                ],
            }

        tmp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(target_path)

    @classmethod
    def from_file(cls, path: Path | str) -> FileFeatureStore:
        """Construct a feature store from an exported JSON bundle."""
        source_path = Path(path)
        if not source_path.exists():
            raise FeatureStoreError(f"Feature store file not found: {source_path}")

        try:
            with source_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as err:
            msg = f"Failed to parse feature store file {source_path}: {err}"
            raise FeatureStoreError(msg) from err

        views_raw = data.get("views", [])
        views = [FeatureView.from_dict(v) for v in views_raw]
        store = cls(views=views)

        snapshots_raw = data.get("snapshots", [])
        for entry in snapshots_raw:
            view_name = entry.get("view_name")
            snapshot_data = entry.get("snapshot")
            if view_name and snapshot_data:
                store.put_snapshot(view_name, FeatureSnapshot.from_dict(snapshot_data))

        return store


def point_in_time_join(
    observations: pd.DataFrame | Sequence[Mapping[str, Any]],
    store: FeatureStoreProtocol,
    view_names: Sequence[str] | None = None,
    timestamp_col: str = "Time",
    entity_col_mapping: Mapping[str, str] | None = None,
    override_ttl_seconds: float | None = None,
) -> tuple[pd.DataFrame, PointInTimeJoinSummary]:
    """Perform point-in-time historical as-of join between observations and feature views.

    Guarantees strict temporal causality: for every observation at timestamp t_obs,
    only feature snapshots with t_feature <= t_obs are considered. Future snapshots
    are never joined, completely eliminating target leakage and lookahead bias.
    """
    if isinstance(observations, pd.DataFrame):
        df = observations.copy()
    else:
        df = pd.DataFrame(list(observations))

    if df.empty:
        return df, PointInTimeJoinSummary(
            total_observations=0,
            matched_observations=0,
            defaulted_observations=0,
            stale_observations=0,
            enriched_features=(),
            match_rate=1.0,
            views_applied=(),
        )

    if timestamp_col not in df.columns:
        raise FeatureStoreError(f"Timestamp column '{timestamp_col}' not found in observations.")

    active_views: list[FeatureView] = []
    if view_names is not None:
        for vn in view_names:
            v = store.get_view(vn)
            if v is None:
                raise FeatureStoreError(f"FeatureView '{vn}' not found in store.")
            active_views.append(v)
    else:
        for vn in store.list_views():
            v = store.get_view(vn)
            if v is not None:
                active_views.append(v)

    if not active_views:
        return df, PointInTimeJoinSummary(
            total_observations=len(df),
            matched_observations=0,
            defaulted_observations=0,
            stale_observations=0,
            enriched_features=(),
            match_rate=0.0,
            views_applied=(),
        )

    col_mapping = dict(entity_col_mapping) if entity_col_mapping is not None else {}
    enriched_feature_names: list[str] = []
    total_matched = 0
    total_stale = 0
    total_defaulted = 0

    for view in active_views:
        entity_col = col_mapping.get(view.entity_key, view.entity_key)
        if entity_col not in df.columns:
            msg = (
                f"Entity column '{entity_col}' for view '{view.name}' "
                "not found in observations DataFrame."
            )
            raise FeatureStoreError(msg)

        # Pre-allocate column arrays
        feat_arrays: dict[str, list[Any]] = {f.name: [] for f in view.features}
        ttl = override_ttl_seconds if override_ttl_seconds is not None else view.ttl_seconds

        for _, row in df.iterrows():
            entity_id = str(row[entity_col])
            obs_time = float(row[timestamp_col])

            res = store.lookup_online(
                entity_key=view.entity_key,
                entity_id=entity_id,
                view_name=view.name,
                as_of_time=obs_time,
            )

            # Custom TTL override if provided
            is_stale = res.is_stale
            if (
                res.found
                and res.timestamp is not None
                and ttl is not None
                and (obs_time - res.timestamp) > ttl
            ):
                is_stale = True

            if res.found and not is_stale:
                total_matched += 1
                for f in view.features:
                    feat_arrays[f.name].append(res.values.get(f.name, f.default_value))
            elif is_stale:
                total_stale += 1
                total_defaulted += 1
                for f in view.features:
                    feat_arrays[f.name].append(f.default_value)
            else:
                total_defaulted += 1
                for f in view.features:
                    feat_arrays[f.name].append(f.default_value)

        for feat_name, arr in feat_arrays.items():
            df[feat_name] = arr
            enriched_feature_names.append(feat_name)

    total_obs = len(df)
    applied_count = total_obs * len(active_views)
    match_rate = total_matched / applied_count if applied_count > 0 else 0.0

    summary = PointInTimeJoinSummary(
        total_observations=total_obs,
        matched_observations=total_matched,
        defaulted_observations=total_defaulted,
        stale_observations=total_stale,
        enriched_features=tuple(enriched_feature_names),
        match_rate=match_rate,
        views_applied=tuple(v.name for v in active_views),
    )
    return df, summary


# ============================================================================
# Training-Serving Feature Skew Surveillance
# ============================================================================


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between two distributions."""
    ref_clean = reference[~np.isnan(reference) & ~np.isinf(reference)]
    curr_clean = current[~np.isnan(current) & ~np.isinf(current)]

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    # Determine quantile bins from reference distribution
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    bin_edges = np.percentile(ref_clean, quantiles * 100)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Ensure strictly monotonic bin edges to avoid collapsed bins
    for i in range(1, len(bin_edges) - 1):
        if bin_edges[i] <= bin_edges[i - 1]:
            bin_edges[i] = bin_edges[i - 1] + 1e-6

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    ref_pct = (ref_counts / len(ref_clean)) + epsilon
    curr_pct = (curr_counts / len(curr_clean)) + epsilon

    # Normalize after adding epsilon smoothing
    ref_pct = ref_pct / np.sum(ref_pct)
    curr_pct = curr_pct / np.sum(curr_pct)

    psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(max(0.0, psi_val))


def calculate_ks_statistic(
    reference: np.ndarray,
    current: np.ndarray,
) -> float:
    """Calculate 2-sample Kolmogorov-Smirnov test statistic D between two distributions."""
    ref_clean = np.sort(reference[~np.isnan(reference) & ~np.isinf(reference)])
    curr_clean = np.sort(current[~np.isnan(current) & ~np.isinf(current)])

    n1 = len(ref_clean)
    n2 = len(curr_clean)
    if n1 == 0 or n2 == 0:
        return 0.0

    all_vals = np.concatenate([ref_clean, curr_clean])
    cdf1 = np.searchsorted(ref_clean, all_vals, side="right") / n1
    cdf2 = np.searchsorted(curr_clean, all_vals, side="right") / n2

    return float(np.max(np.abs(cdf1 - cdf2)))


def calculate_wasserstein_distance(
    reference: np.ndarray,
    current: np.ndarray,
) -> float:
    """Calculate 1D Wasserstein-1 (Earth Mover's Distance) between two distributions."""
    ref_clean = np.sort(reference[~np.isnan(reference) & ~np.isinf(reference)])
    curr_clean = np.sort(current[~np.isnan(current) & ~np.isinf(current)])

    n1 = len(ref_clean)
    n2 = len(curr_clean)
    if n1 == 0 or n2 == 0:
        return 0.0

    all_vals = np.unique(np.concatenate([ref_clean, curr_clean]))
    cdf1 = np.searchsorted(ref_clean, all_vals, side="right") / n1
    cdf2 = np.searchsorted(curr_clean, all_vals, side="right") / n2

    deltas = np.diff(all_vals)
    abs_cdf_diff = np.abs(cdf1[:-1] - cdf2[:-1])
    return float(np.sum(abs_cdf_diff * deltas))


@dataclass(frozen=True)
class FeatureSkewMetric:
    """Skew evaluation metric for an individual feature."""

    feature_name: str
    feature_type: str
    psi: float
    ks_statistic: float | None
    wasserstein_distance: float | None
    null_rate_reference: float
    null_rate_current: float
    null_rate_delta: float
    mean_reference: float | None
    mean_current: float | None
    std_reference: float | None
    std_current: float | None
    status: SkewStatus

    def to_dict(self) -> dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "feature_name": self.feature_name,
            "feature_type": self.feature_type,
            "psi": round(self.psi, 4),
            "ks_statistic": round(self.ks_statistic, 4) if self.ks_statistic is not None else None,
            "wasserstein_distance": (
                round(self.wasserstein_distance, 4)
                if self.wasserstein_distance is not None
                else None
            ),
            "null_rate_reference": round(self.null_rate_reference, 4),
            "null_rate_current": round(self.null_rate_current, 4),
            "null_rate_delta": round(self.null_rate_delta, 4),
            "mean_reference": (
                round(self.mean_reference, 4) if self.mean_reference is not None else None
            ),
            "mean_current": round(self.mean_current, 4) if self.mean_current is not None else None,
            "std_reference": (
                round(self.std_reference, 4) if self.std_reference is not None else None
            ),
            "std_current": round(self.std_current, 4) if self.std_current is not None else None,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class FeatureSkewReport:
    """Comprehensive training-serving feature skew surveillance report."""

    overall_status: SkewStatus
    reference_count: int
    current_count: int
    warning_threshold_psi: float
    drift_threshold_psi: float
    metrics: dict[str, FeatureSkewMetric]

    def to_dict(self) -> dict[str, Any]:
        """Convert report to serializable dictionary."""
        return {
            "overall_status": self.overall_status.value,
            "reference_count": self.reference_count,
            "current_count": self.current_count,
            "warning_threshold_psi": self.warning_threshold_psi,
            "drift_threshold_psi": self.drift_threshold_psi,
            "metrics": {k: m.to_dict() for k, m in self.metrics.items()},
        }

    def summary_table(self) -> str:
        """Render a readable text summary table of the skew metrics."""
        thresh_info = (
            f"Thresholds: Warning PSI >= {self.warning_threshold_psi:.2f} | "
            f"Drift PSI >= {self.drift_threshold_psi:.2f}"
        )
        hdr = (
            f"{'Feature':<20} {'Type':<8} {'PSI':<8} {'KS':<8} "
            f"{'Wasserstein':<12} {'Null Δ':<8} {'Status':<10}"
        )
        lines = [
            f"Overall Status: {self.overall_status.value.upper()}",
            f"Reference Samples: {self.reference_count} | Current Samples: {self.current_count}",
            thresh_info,
            "-" * 80,
            hdr,
            "-" * 80,
        ]
        for name, m in sorted(self.metrics.items()):
            ks_str = f"{m.ks_statistic:.4f}" if m.ks_statistic is not None else "N/A"
            w_str = f"{m.wasserstein_distance:.4f}" if m.wasserstein_distance is not None else "N/A"
            row = (
                f"{name:<20} {m.feature_type:<8} {m.psi:<8.4f} {ks_str:<8} "
                f"{w_str:<12} {m.null_rate_delta:<8.4f} {m.status.value:<10}"
            )
            lines.append(row)
        lines.append("-" * 80)
        return "\n".join(lines)


class FeatureSkewAnalyzer:
    """Surveillance analyzer detecting feature distribution drift and training-serving skew."""

    def __init__(
        self,
        warning_threshold_psi: float = 0.10,
        drift_threshold_psi: float = 0.25,
        ks_warning_threshold: float = 0.10,
        ks_drift_threshold: float = 0.20,
    ) -> None:
        if warning_threshold_psi < 0 or drift_threshold_psi < warning_threshold_psi:
            raise FeatureStoreError("Invalid PSI thresholds: must satisfy 0 <= warning <= drift.")
        self.warning_threshold_psi = float(warning_threshold_psi)
        self.drift_threshold_psi = float(drift_threshold_psi)
        self.ks_warning_threshold = float(ks_warning_threshold)
        self.ks_drift_threshold = float(ks_drift_threshold)

    def analyze_skew(
        self,
        reference_data: pd.DataFrame | Sequence[Mapping[str, Any]],
        current_data: pd.DataFrame | Sequence[Mapping[str, Any]],
        feature_names: Sequence[str] | None = None,
    ) -> FeatureSkewReport:
        """Analyze statistical feature divergence between reference and current samples."""
        ref_df = (
            reference_data
            if isinstance(reference_data, pd.DataFrame)
            else pd.DataFrame(list(reference_data))
        )
        curr_df = (
            current_data
            if isinstance(current_data, pd.DataFrame)
            else pd.DataFrame(list(current_data))
        )

        if feature_names is not None:
            eval_cols = [c for c in feature_names if c in ref_df.columns and c in curr_df.columns]
        else:
            eval_cols = [
                c for c in ref_df.columns if c in curr_df.columns and c not in ("Time", "Class")
            ]

        metrics: dict[str, FeatureSkewMetric] = {}
        worst_status = SkewStatus.STABLE

        for col in eval_cols:
            ref_col = ref_df[col]
            curr_col = curr_df[col]

            null_ref = float(ref_col.isna().mean())
            null_curr = float(curr_col.isna().mean())
            null_delta = abs(null_curr - null_ref)

            is_numeric = pd.api.types.is_numeric_dtype(
                ref_col
            ) and pd.api.types.is_numeric_dtype(curr_col)

            if is_numeric:
                ref_arr = ref_col.dropna().to_numpy(dtype=float)
                curr_arr = curr_col.dropna().to_numpy(dtype=float)

                psi_val = calculate_psi(ref_arr, curr_arr)
                ks_val = calculate_ks_statistic(ref_arr, curr_arr)
                w_val = calculate_wasserstein_distance(ref_arr, curr_arr)

                mean_ref = float(np.mean(ref_arr)) if len(ref_arr) > 0 else None
                mean_curr = float(np.mean(curr_arr)) if len(curr_arr) > 0 else None
                std_ref = float(np.std(ref_arr)) if len(ref_arr) > 0 else None
                std_curr = float(np.std(curr_arr)) if len(curr_arr) > 0 else None

                # Status decision based on PSI and KS
                if (
                    psi_val >= self.drift_threshold_psi
                    or ks_val >= self.ks_drift_threshold
                    or null_delta >= 0.20
                ):
                    status = SkewStatus.DRIFTED
                elif (
                    psi_val >= self.warning_threshold_psi
                    or ks_val >= self.ks_warning_threshold
                    or null_delta >= 0.10
                ):
                    status = SkewStatus.WARNING
                else:
                    status = SkewStatus.STABLE

                metrics[col] = FeatureSkewMetric(
                    feature_name=col,
                    feature_type="numeric",
                    psi=psi_val,
                    ks_statistic=ks_val,
                    wasserstein_distance=w_val,
                    null_rate_reference=null_ref,
                    null_rate_current=null_curr,
                    null_rate_delta=null_delta,
                    mean_reference=mean_ref,
                    mean_current=mean_curr,
                    std_reference=std_ref,
                    std_current=std_curr,
                    status=status,
                )
            else:
                # Categorical / string feature evaluation
                ref_s = ref_col.fillna("__NULL__").astype(str)
                curr_s = curr_col.fillna("__NULL__").astype(str)

                all_cats = sorted(set(ref_s.unique()) | set(curr_s.unique()))
                ref_counts = ref_s.value_counts(normalize=True).to_dict()
                curr_counts = curr_s.value_counts(normalize=True).to_dict()

                eps = 1e-4
                ref_probs = np.array([ref_counts.get(c, 0.0) + eps for c in all_cats])
                curr_probs = np.array([curr_counts.get(c, 0.0) + eps for c in all_cats])
                ref_probs /= ref_probs.sum()
                curr_probs /= curr_probs.sum()

                psi_val = float(
                    max(0.0, np.sum((curr_probs - ref_probs) * np.log(curr_probs / ref_probs)))
                )

                if psi_val >= self.drift_threshold_psi or null_delta >= 0.20:
                    status = SkewStatus.DRIFTED
                elif psi_val >= self.warning_threshold_psi or null_delta >= 0.10:
                    status = SkewStatus.WARNING
                else:
                    status = SkewStatus.STABLE

                metrics[col] = FeatureSkewMetric(
                    feature_name=col,
                    feature_type="categorical",
                    psi=psi_val,
                    ks_statistic=None,
                    wasserstein_distance=None,
                    null_rate_reference=null_ref,
                    null_rate_current=null_curr,
                    null_rate_delta=null_delta,
                    mean_reference=None,
                    mean_current=None,
                    std_reference=None,
                    std_current=None,
                    status=status,
                )

            if status == SkewStatus.DRIFTED:
                worst_status = SkewStatus.DRIFTED
            elif status == SkewStatus.WARNING and worst_status != SkewStatus.DRIFTED:
                worst_status = SkewStatus.WARNING

        return FeatureSkewReport(
            overall_status=worst_status,
            reference_count=len(ref_df),
            current_count=len(curr_df),
            warning_threshold_psi=self.warning_threshold_psi,
            drift_threshold_psi=self.drift_threshold_psi,
            metrics=metrics,
        )
