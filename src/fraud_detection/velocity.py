"""Transaction velocity profiling, sliding feature windows, and stream enrichment."""

from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class VelocityError(ValueError):
    """Raised when velocity configuration, schema, or evaluation is invalid."""


class AggregationType(StrEnum):
    """Supported sliding-window mathematical aggregations."""

    COUNT = "count"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    STD = "std"
    EMA = "ema"


DEFAULT_AGGREGATIONS: tuple[AggregationType, ...] = (
    AggregationType.COUNT,
    AggregationType.SUM,
    AggregationType.MIN,
    AggregationType.MAX,
    AggregationType.MEAN,
    AggregationType.STD,
    AggregationType.EMA,
)


@dataclass(frozen=True)
class VelocityWindow:
    """Sliding time-window configuration defining duration and aggregations."""

    duration_seconds: float
    name: str | None = None
    aggregations: tuple[AggregationType, ...] = DEFAULT_AGGREGATIONS
    ema_alpha: float = 0.3

    def __post_init__(self) -> None:
        if (
            not isinstance(self.duration_seconds, (int, float))
            or math.isnan(self.duration_seconds)
            or math.isinf(self.duration_seconds)
            or self.duration_seconds <= 0.0
        ):
            raise VelocityError("VelocityWindow duration_seconds must be a positive finite number.")

        if self.name is None:
            dur = self.duration_seconds
            if dur >= 86400 and dur % 86400 == 0:
                calc_name = f"{int(dur // 86400)}d"
            elif dur >= 3600 and dur % 3600 == 0:
                calc_name = f"{int(dur // 3600)}h"
            elif dur >= 60 and dur % 60 == 0:
                calc_name = f"{int(dur // 60)}m"
            else:
                calc_name = f"{int(dur)}s" if dur.is_integer() else f"{dur}s"
            object.__setattr__(self, "name", calc_name)
        elif not isinstance(self.name, str) or not self.name.strip():
            raise VelocityError("VelocityWindow name must be a non-empty string.")

        if not self.aggregations:
            raise VelocityError("VelocityWindow must specify at least one aggregation.")

        coerced_aggs: list[AggregationType] = []
        for agg in self.aggregations:
            raw_agg: Any = agg
            if isinstance(raw_agg, AggregationType):
                coerced_aggs.append(raw_agg)
            else:
                try:
                    coerced_aggs.append(AggregationType(raw_agg))
                except ValueError as exc:
                    valid = [a.value for a in AggregationType]
                    raise VelocityError(
                        f"Unsupported aggregation {raw_agg!r}. Supported aggregations: {valid}."
                    ) from exc
        object.__setattr__(self, "aggregations", tuple(coerced_aggs))

        if (
            not isinstance(self.ema_alpha, (int, float))
            or math.isnan(self.ema_alpha)
            or not 0.0 < self.ema_alpha <= 1.0
        ):
            raise VelocityError("VelocityWindow ema_alpha must be a finite float in (0.0, 1.0].")

    def to_dict(self) -> dict[str, Any]:
        """Serialize window configuration to dictionary."""
        return {
            "duration_seconds": self.duration_seconds,
            "name": self.name,
            "aggregations": [a.value for a in self.aggregations],
            "ema_alpha": self.ema_alpha,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VelocityWindow:
        """Deserialize window configuration from mapping."""
        if not isinstance(data, Mapping) or "duration_seconds" not in data:
            raise VelocityError("VelocityWindow requires 'duration_seconds'.")
        aggs = data.get("aggregations", DEFAULT_AGGREGATIONS)
        return cls(
            duration_seconds=float(data["duration_seconds"]),
            name=str(data["name"]) if data.get("name") is not None else None,
            aggregations=tuple(aggs),
            ema_alpha=float(data.get("ema_alpha", 0.3)),
        )


DEFAULT_WINDOWS: tuple[VelocityWindow, ...] = (
    VelocityWindow(duration_seconds=300.0, name="5m"),
    VelocityWindow(duration_seconds=3600.0, name="1h"),
    VelocityWindow(duration_seconds=86400.0, name="24h"),
)


@dataclass(frozen=True)
class VelocityConfig:
    """Comprehensive specification for entity velocity tracking and feature enrichment."""

    entity_key: str = "card_id"
    timestamp_key: str = "Time"
    amount_key: str = "Amount"
    windows: tuple[VelocityWindow, ...] = DEFAULT_WINDOWS
    include_ratios: bool = True
    include_deltas: bool = True
    feature_prefix: str = "velocity_"
    max_events_per_entity: int = 1000
    max_entities: int = 100_000
    default_entity_id: str = "global"

    def __post_init__(self) -> None:
        if not isinstance(self.entity_key, str) or not self.entity_key.strip():
            raise VelocityError("VelocityConfig entity_key must be a non-empty string.")
        if not isinstance(self.timestamp_key, str) or not self.timestamp_key.strip():
            raise VelocityError("VelocityConfig timestamp_key must be a non-empty string.")
        if not isinstance(self.amount_key, str) or not self.amount_key.strip():
            raise VelocityError("VelocityConfig amount_key must be a non-empty string.")
        if not self.windows:
            raise VelocityError("VelocityConfig windows must contain at least one VelocityWindow.")

        # Ensure windows are sorted by duration ascending and names are unique
        sorted_windows = tuple(sorted(self.windows, key=lambda w: w.duration_seconds))
        names: set[str] = set()
        for w in sorted_windows:
            w_name = w.name or str(w.duration_seconds)
            if w_name in names:
                raise VelocityError(f"Duplicate window name {w_name!r} in VelocityConfig.")
            names.add(w_name)
        object.__setattr__(self, "windows", sorted_windows)

        if not isinstance(self.feature_prefix, str):
            raise VelocityError("VelocityConfig feature_prefix must be a string.")
        if (
            not isinstance(self.max_events_per_entity, int)
            or isinstance(self.max_events_per_entity, bool)
            or self.max_events_per_entity <= 0
        ):
            raise VelocityError("VelocityConfig max_events_per_entity must be a positive integer.")
        if (
            not isinstance(self.max_entities, int)
            or isinstance(self.max_entities, bool)
            or self.max_entities <= 0
        ):
            raise VelocityError("VelocityConfig max_entities must be a positive integer.")

    @property
    def max_window_duration(self) -> float:
        """Return the maximum duration in seconds across all configured windows."""
        return max(w.duration_seconds for w in self.windows)

    def feature_names(self) -> tuple[str, ...]:
        """Compute the deterministic ordered tuple of feature names generated by this config."""
        names: list[str] = []
        p = self.feature_prefix

        # Window-level aggregations
        for w in self.windows:
            w_name = w.name or str(w.duration_seconds)
            names.extend(f"{p}{agg.value}_{w_name}" for agg in w.aggregations)
            if self.include_ratios and AggregationType.MEAN in w.aggregations:
                names.append(f"{p}amount_to_mean_{w_name}")

        # Pairwise count ratios across consecutive window pairs
        if self.include_ratios and len(self.windows) > 1:
            for i in range(len(self.windows) - 1):
                w_short = self.windows[i].name or str(self.windows[i].duration_seconds)
                w_long = self.windows[i + 1].name or str(self.windows[i + 1].duration_seconds)
                names.append(f"{p}count_ratio_{w_short}_{w_long}")

        # Instantaneous delta features
        if self.include_deltas:
            names.append(f"{p}time_delta_prev")
            names.append(f"{p}amount_delta_prev")

        return tuple(names)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "entity_key": self.entity_key,
            "timestamp_key": self.timestamp_key,
            "amount_key": self.amount_key,
            "windows": [w.to_dict() for w in self.windows],
            "include_ratios": self.include_ratios,
            "include_deltas": self.include_deltas,
            "feature_prefix": self.feature_prefix,
            "max_events_per_entity": self.max_events_per_entity,
            "max_entities": self.max_entities,
            "default_entity_id": self.default_entity_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VelocityConfig:
        """Deserialize configuration from mapping."""
        if not isinstance(data, Mapping):
            raise VelocityError("VelocityConfig data must be a dictionary.")
        raw_windows = data.get("windows")
        windows: tuple[VelocityWindow, ...]
        if raw_windows is None:
            windows = DEFAULT_WINDOWS
        elif isinstance(raw_windows, (list, tuple)):
            windows = tuple(VelocityWindow.from_dict(w) for w in raw_windows)
        else:
            raise VelocityError("VelocityConfig windows must be a list of window definitions.")

        return cls(
            entity_key=str(data.get("entity_key", "card_id")),
            timestamp_key=str(data.get("timestamp_key", "Time")),
            amount_key=str(data.get("amount_key", "Amount")),
            windows=windows,
            include_ratios=bool(data.get("include_ratios", True)),
            include_deltas=bool(data.get("include_deltas", True)),
            feature_prefix=str(data.get("feature_prefix", "velocity_")),
            max_events_per_entity=int(data.get("max_events_per_entity", 1000)),
            max_entities=int(data.get("max_entities", 100_000)),
            default_entity_id=str(data.get("default_entity_id", "global")),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> VelocityConfig:
        """Deserialize configuration from JSON string."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise VelocityError(f"Invalid JSON for VelocityConfig: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def load_file(cls, path: str | Path) -> VelocityConfig:
        """Load and parse VelocityConfig from a JSON file."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Velocity config file not found: {file_path}")
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise VelocityError(f"Failed to read velocity config {file_path}: {exc}") from exc
        return cls.from_json(content)

    def save_file(self, path: str | Path) -> None:
        """Atomically persist VelocityConfig to a JSON file."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp_file = dest.with_name(f".{dest.name}.tmp")
        temp_file.write_text(self.to_json() + "\n", encoding="utf-8")
        temp_file.replace(dest)


@dataclass(frozen=True)
class TransactionEvent:
    """Historical transaction record stored in a sliding window."""

    timestamp: float
    amount: float


class SlidingWindow:
    """In-memory sliding event buffer for a single entity with $O(1)$ amortized eviction."""

    def __init__(self, max_events: int = 1000) -> None:
        self.max_events = max_events
        self.events: deque[TransactionEvent] = deque()

    def evict_older_than(self, cutoff_timestamp: float) -> int:
        """Evict events strictly older than cutoff timestamp. Returns number of evicted items."""
        evicted = 0
        while self.events and self.events[0].timestamp < cutoff_timestamp:
            self.events.popleft()
            evicted += 1
        return evicted

    def append(self, timestamp: float, amount: float, max_window: float) -> None:
        """Record an event, evicting expired items and enforcing capacity constraints."""
        cutoff = timestamp - max_window
        self.evict_older_than(cutoff)
        if len(self.events) >= self.max_events:
            self.events.popleft()
        self.events.append(TransactionEvent(timestamp=timestamp, amount=amount))

    def compute_features(
        self,
        current_timestamp: float,
        current_amount: float,
        config: VelocityConfig,
    ) -> dict[str, float]:
        """Compute all sliding window aggregations strictly before the current event.

        Strict temporal causality: Only events with timestamp <= current_timestamp that
        were recorded prior to the current transaction are incorporated.
        """
        features: dict[str, float] = {}
        p = config.feature_prefix

        # Instantaneous delta features from previous event
        if config.include_deltas:
            if self.events:
                last_event = self.events[-1]
                time_delta = max(0.0, current_timestamp - last_event.timestamp)
                amount_delta = current_amount - last_event.amount
            else:
                time_delta = 0.0
                amount_delta = 0.0
            features[f"{p}time_delta_prev"] = float(time_delta)
            features[f"{p}amount_delta_prev"] = float(amount_delta)

        # Window-level calculations
        window_counts: dict[str, int] = {}
        event_list = list(self.events)

        for window in config.windows:
            w_name = window.name or str(window.duration_seconds)
            cutoff = current_timestamp - window.duration_seconds

            # Collect amounts in window [cutoff, current_timestamp]
            # Since event_list is chronological, slice backwards
            window_amounts: list[float] = []
            for ev in reversed(event_list):
                if ev.timestamp >= cutoff:
                    window_amounts.append(ev.amount)
                else:
                    break

            count = len(window_amounts)
            window_counts[w_name] = count

            # Default zero/cold-start values
            if count == 0:
                for agg in window.aggregations:
                    features[f"{p}{agg.value}_{w_name}"] = 0.0
                if config.include_ratios and AggregationType.MEAN in window.aggregations:
                    features[f"{p}amount_to_mean_{w_name}"] = 1.0 if current_amount == 0.0 else 0.0
                continue

            sum_val = sum(window_amounts)
            mean_val = sum_val / count

            for agg in window.aggregations:
                col_name = f"{p}{agg.value}_{w_name}"
                if agg == AggregationType.COUNT:
                    features[col_name] = float(count)
                elif agg == AggregationType.SUM:
                    features[col_name] = float(sum_val)
                elif agg == AggregationType.MIN:
                    features[col_name] = float(min(window_amounts))
                elif agg == AggregationType.MAX:
                    features[col_name] = float(max(window_amounts))
                elif agg == AggregationType.MEAN:
                    features[col_name] = float(mean_val)
                elif agg == AggregationType.STD:
                    if count <= 1:
                        features[col_name] = 0.0
                    else:
                        var = sum((x - mean_val) ** 2 for x in window_amounts) / count
                        features[col_name] = float(math.sqrt(var))
                elif agg == AggregationType.EMA:
                    chronological_amounts = list(reversed(window_amounts))
                    ema = chronological_amounts[0]
                    alpha = window.ema_alpha
                    for x in chronological_amounts[1:]:
                        ema = alpha * x + (1.0 - alpha) * ema
                    features[col_name] = float(ema)

            if config.include_ratios and AggregationType.MEAN in window.aggregations:
                ratio = (current_amount / mean_val) if mean_val > 0.0 else 1.0
                features[f"{p}amount_to_mean_{w_name}"] = float(ratio)

        # Pairwise count ratios between consecutive windows
        if config.include_ratios and len(config.windows) > 1:
            for i in range(len(config.windows) - 1):
                w_short = config.windows[i].name or str(config.windows[i].duration_seconds)
                w_long = config.windows[i + 1].name or str(config.windows[i + 1].duration_seconds)
                cnt_short = window_counts.get(w_short, 0)
                cnt_long = window_counts.get(w_long, 0)
                ratio = (cnt_short / cnt_long) if cnt_long > 0 else 0.0
                features[f"{p}count_ratio_{w_short}_{w_long}"] = float(ratio)

        return features


class VelocityWindowBuffer:
    """Thread-safe stateful buffer managing per-entity sliding transaction windows."""

    def __init__(self, config: VelocityConfig | None = None) -> None:
        self.config = config or VelocityConfig()
        self._lock = threading.Lock()
        self._windows: OrderedDict[str, SlidingWindow] = OrderedDict()
        self._last_seen: dict[str, float] = {}

    @property
    def entity_count(self) -> int:
        """Return the number of currently tracked entities."""
        with self._lock:
            return len(self._windows)

    def total_events(self) -> int:
        """Return the total number of events currently held across all entities."""
        with self._lock:
            return sum(len(w.events) for w in self._windows.values())

    def clear(self) -> None:
        """Reset all tracked state in the buffer."""
        with self._lock:
            self._windows.clear()
            self._last_seen.clear()

    def _prune_expired_entities_locked(self, current_timestamp: float) -> int:
        """Evict entities that have had no transactions inside the max window duration."""
        cutoff = current_timestamp - self.config.max_window_duration
        expired = [entity_id for entity_id, last_t in self._last_seen.items() if last_t < cutoff]
        for entity_id in expired:
            self._windows.pop(entity_id, None)
            self._last_seen.pop(entity_id, None)
        return len(expired)

    def _enforce_entity_limit_locked(self) -> None:
        """Evict the least recently used entity if entity count exceeds max_entities."""
        while len(self._windows) > self.config.max_entities:
            oldest_entity, _ = self._windows.popitem(last=False)
            self._last_seen.pop(oldest_entity, None)

    def record(self, entity_id: str, timestamp: float, amount: float) -> None:
        """Record a transaction for the given entity in thread-safe fashion."""
        entity = str(entity_id) if entity_id else self.config.default_entity_id
        with self._lock:
            if entity not in self._windows:
                self._prune_expired_entities_locked(timestamp)
                self._windows[entity] = SlidingWindow(max_events=self.config.max_events_per_entity)
            else:
                self._windows.move_to_end(entity)

            self._windows[entity].append(timestamp, amount, self.config.max_window_duration)
            self._last_seen[entity] = timestamp
            self._enforce_entity_limit_locked()

    def compute_features(
        self,
        entity_id: str,
        timestamp: float,
        amount: float,
    ) -> dict[str, float]:
        """Compute velocity features for entity without updating the sliding window."""
        entity = str(entity_id) if entity_id else self.config.default_entity_id
        with self._lock:
            window = self._windows.get(entity)
            if window is None:
                # Cold-start: entity has no history
                empty_window = SlidingWindow(max_events=self.config.max_events_per_entity)
                return empty_window.compute_features(timestamp, amount, self.config)
            return window.compute_features(timestamp, amount, self.config)

    def enrich_and_record(
        self,
        entity_id: str,
        timestamp: float,
        amount: float,
    ) -> dict[str, float]:
        """Atomically compute velocity features on prior history, then record this event."""
        entity = str(entity_id) if entity_id else self.config.default_entity_id
        with self._lock:
            if entity not in self._windows:
                self._prune_expired_entities_locked(timestamp)
                window = SlidingWindow(max_events=self.config.max_events_per_entity)
                self._windows[entity] = window
            else:
                self._windows.move_to_end(entity)
                window = self._windows[entity]

            # 1. Compute features strictly prior to recording this event
            features = window.compute_features(timestamp, amount, self.config)

            # 2. Append event to window
            window.append(timestamp, amount, self.config.max_window_duration)
            self._last_seen[entity] = timestamp
            self._enforce_entity_limit_locked()

            return features

    def get_entity_profile(
        self, entity_id: str, current_timestamp: float | None = None
    ) -> dict[str, Any]:
        """Inspect current state and active counts for a specific entity."""
        entity = str(entity_id) if entity_id else self.config.default_entity_id
        with self._lock:
            window = self._windows.get(entity)
            if window is None:
                return {
                    "entity_id": entity,
                    "tracked": False,
                    "events_count": 0,
                    "last_seen": None,
                    "windows": {w.name or str(w.duration_seconds): 0 for w in self.config.windows},
                }

            t = (
                current_timestamp
                if current_timestamp is not None
                else self._last_seen.get(entity, 0.0)
            )
            window_counts: dict[str, int] = {}
            for w in self.config.windows:
                w_name = w.name or str(w.duration_seconds)
                cutoff = t - w.duration_seconds
                cnt = sum(1 for ev in window.events if ev.timestamp >= cutoff)
                window_counts[w_name] = cnt

            return {
                "entity_id": entity,
                "tracked": True,
                "events_count": len(window.events),
                "last_seen": self._last_seen.get(entity),
                "windows": window_counts,
            }

    def stats(self) -> dict[str, Any]:
        """Summary statistics of the in-memory velocity buffer."""
        with self._lock:
            total_evs = sum(len(w.events) for w in self._windows.values())
            return {
                "entities_tracked": len(self._windows),
                "total_events": total_evs,
                "max_entities": self.config.max_entities,
                "max_events_per_entity": self.config.max_events_per_entity,
                "configured_windows": [
                    w.name or str(w.duration_seconds) for w in self.config.windows
                ],
            }


def enrich_record_velocity(
    record: Mapping[str, Any],
    buffer: VelocityWindowBuffer,
    config: VelocityConfig | None = None,
    *,
    update: bool = True,
) -> dict[str, Any]:
    """Enrich a single transaction record mapping with sliding window velocity features.

    Args:
        record: Inbound transaction dictionary containing feature keys.
        buffer: Stateful VelocityWindowBuffer tracking historical entity streams.
        config: Optional velocity configuration override (defaults to buffer.config).
        update: When True, records this transaction in the buffer after computing features.
    """
    cfg = config or buffer.config

    # Extract entity identifier
    entity_val = record.get(cfg.entity_key)
    entity_id = str(entity_val) if entity_val is not None else cfg.default_entity_id

    # Extract timestamp
    if cfg.timestamp_key not in record:
        raise VelocityError(
            f"Missing required timestamp key {cfg.timestamp_key!r} in transaction record."
        )
    raw_timestamp = record[cfg.timestamp_key]
    try:
        timestamp = float(raw_timestamp)
        if math.isnan(timestamp) or math.isinf(timestamp):
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise VelocityError(
            f"Invalid timestamp value {raw_timestamp!r}; must be a finite numeric float."
        ) from exc

    # Extract amount
    if cfg.amount_key not in record:
        raise VelocityError(
            f"Missing required amount key {cfg.amount_key!r} in transaction record."
        )
    raw_amount = record[cfg.amount_key]
    try:
        amount = float(raw_amount)
        if math.isnan(amount) or math.isinf(amount):
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise VelocityError(
            f"Invalid amount value {raw_amount!r}; must be a finite numeric float."
        ) from exc

    if update:
        velocity_features = buffer.enrich_and_record(entity_id, timestamp, amount)
    else:
        velocity_features = buffer.compute_features(entity_id, timestamp, amount)

    # Return merged dictionary
    enriched = dict(record)
    enriched.update(velocity_features)
    return enriched


def compute_batch_velocity(
    df: pd.DataFrame,
    config: VelocityConfig | None = None,
    *,
    timestamp_col: str | None = None,
    amount_col: str | None = None,
    entity_col: str | None = None,
) -> pd.DataFrame:
    """Compute sliding window behavioral features across an entire historical DataFrame.

    Guarantees:
    - Strict temporal causality: processes events in chronological sequence, guaranteeing
      that for every row $i$, features depend exclusively on prior rows $j < i$.
    - Preserves original DataFrame row indexing and ordering.
    - Cold-start handling for first appearances of new entities.
    - Appends all velocity feature columns as numeric float64.
    """
    if df.empty:
        return df.copy()

    cfg = config or VelocityConfig()
    t_col = timestamp_col or cfg.timestamp_key
    a_col = amount_col or cfg.amount_key
    e_col = entity_col or cfg.entity_key

    if t_col not in df.columns:
        raise VelocityError(f"Timestamp column {t_col!r} not found in DataFrame.")
    if a_col not in df.columns:
        raise VelocityError(f"Amount column {a_col!r} not found in DataFrame.")

    has_entity_col = e_col in df.columns

    # Verify numeric timestamps and amounts
    timestamps = pd.to_numeric(df[t_col], errors="coerce")
    if timestamps.isna().any():
        raise VelocityError(f"Timestamp column {t_col!r} contains non-numeric or NaN values.")
    amounts = pd.to_numeric(df[a_col], errors="coerce")
    if amounts.isna().any():
        raise VelocityError(f"Amount column {a_col!r} contains non-numeric or NaN values.")

    # Build chronological sort indices
    sort_order = np.argsort(timestamps.to_numpy(dtype=float), kind="stable")

    # Dedicated isolated buffer for batch execution
    buffer = VelocityWindowBuffer(config=cfg)

    # Pre-allocate feature columns dict
    feature_names = cfg.feature_names()
    computed_columns: dict[str, np.ndarray] = {
        fname: np.zeros(len(df), dtype=np.float64) for fname in feature_names
    }

    t_arr = timestamps.to_numpy(dtype=float)
    a_arr = amounts.to_numpy(dtype=float)

    if has_entity_col:
        e_arr = df[e_col].astype(str).to_numpy()
    else:
        e_arr = np.full(len(df), cfg.default_entity_id, dtype=object)

    for idx in sort_order:
        ent = str(e_arr[idx])
        t_val = float(t_arr[idx])
        a_val = float(a_arr[idx])

        features = buffer.enrich_and_record(ent, t_val, a_val)
        for fname, val in features.items():
            if fname in computed_columns:
                computed_columns[fname][idx] = val

    result = df.copy()
    for fname in feature_names:
        result[fname] = computed_columns[fname]

    return result
