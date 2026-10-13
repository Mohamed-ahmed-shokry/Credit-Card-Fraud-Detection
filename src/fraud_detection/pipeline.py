"""Real-time streaming decision graph and adaptive execution pipeline."""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import hashlib
import inspect
import json
import logging
import math
import statistics
import threading
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from time import monotonic, perf_counter, sleep
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

import numpy as np
import pandas as pd

from fraud_detection.evaluation import (
    DecisionAction,
    PolicyTransitionMatrix,
    TieredMetrics,
    TieredThresholds,
    assign_tiered_decisions,
    evaluate_tiered_policy,
)
from fraud_detection.features import FeatureStoreProtocol
from fraud_detection.model import FraudModel
from fraud_detection.recalibration import BaseRecalibrator
from fraud_detection.rules import RuleAction, RulePrecedence, RuleSet
from fraud_detection.velocity import (
    VelocityConfig,
    VelocityWindowBuffer,
    enrich_record_velocity,
)

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Base error for execution pipeline exceptions."""


class PipelineValidationError(PipelineError):
    """Raised when decision graph topology is invalid."""


class PipelineCycleError(PipelineValidationError):
    """Raised when decision graph contains circular dependencies."""


class PipelineExecutionError(PipelineError):
    """Raised when a required pipeline stage fails or cannot recover."""


class PipelineTimeoutError(PipelineExecutionError):
    """Raised when a required pipeline stage exceeds its latency budget."""


class PipelineCircuitOpenError(PipelineExecutionError):
    """Raised when a required pipeline stage is blocked by an open circuit breaker."""


class StageType(StrEnum):
    """Functional categorization for pipeline stages."""

    ENRICHMENT = "enrichment"
    RULES = "rules"
    INFERENCE = "inference"
    CALIBRATION = "calibration"
    ACTION = "action"
    CUSTOM = "custom"


class StageStatus(StrEnum):
    """Execution state of a single pipeline node."""

    PENDING = "pending"
    SUCCESS = "success"
    FALLBACK = "fallback"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    FAILED = "failed"
    CIRCUIT_OPEN = "circuit_open"


@dataclass(frozen=True)
class PipelineEdge:
    """Directed dependency link between two pipeline stages."""

    source: str
    target: str

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "target": self.target}


@dataclass
class StageExecutionResult:
    """Execution telemetry and output payload for a single pipeline stage."""

    node_name: str
    stage_type: StageType
    status: StageStatus
    latency_ms: float = 0.0
    output: Any = None
    error: str | None = None
    degraded: bool = False
    cached: bool = False
    speculative: bool = False
    short_circuited_by_breaker: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert stage telemetry to a serializable dictionary."""
        is_primitive = isinstance(self.output, (dict, list, str, int, float, bool, type(None)))
        safe_output = self.output if is_primitive else str(self.output)
        return {
            "node_name": self.node_name,
            "stage_type": self.stage_type.value,
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 3),
            "output": safe_output,
            "error": self.error,
            "degraded": self.degraded,
            "cached": self.cached,
            "speculative": self.speculative,
            "short_circuited_by_breaker": self.short_circuited_by_breaker,
        }


@dataclass
class PipelineContext:
    """Mutable runtime context shared across executing graph nodes."""

    transaction: dict[str, Any]
    frame: pd.DataFrame | None = None
    results: dict[str, StageExecutionResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_output(self, node_name: str, default: Any = None) -> Any:
        """Retrieve output of an upstream completed node."""
        res = self.results.get(node_name)
        if res is not None and res.output is not None:
            return res.output
        return default

    def set_output(self, node_name: str, value: Any) -> None:
        """Store output directly into results for a node."""
        if node_name in self.results:
            self.results[node_name].output = value


@runtime_checkable
class StageProcessorProtocol(Protocol):
    """Callable interface for asynchronous or synchronous stage processors."""

    def __call__(self, context: PipelineContext) -> Any: ...


@dataclass(frozen=True)
class NodeCachePolicy:
    """Configures deterministic memoization for a graph node."""

    enabled: bool = True
    ttl_seconds: float | None = 60.0
    max_size: int = 1000
    key_fields: tuple[str, ...] | None = None
    key_fn: Callable[[PipelineContext], str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ttl_seconds": self.ttl_seconds,
            "max_size": self.max_size,
            "key_fields": list(self.key_fields) if self.key_fields is not None else None,
            "has_custom_key_fn": self.key_fn is not None,
        }


@dataclass
class NodeCacheEntry:
    """Cached output entry with creation timestamp."""

    created_at: float
    output: Any
    key: str

    def is_expired(self, now: float, ttl_seconds: float | None) -> bool:
        if ttl_seconds is None or ttl_seconds <= 0:
            return False
        return (now - self.created_at) >= ttl_seconds


@dataclass
class NodeCacheStats:
    """Operational telemetry counters for stage node cache."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def total_lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_ratio(self) -> float:
        total = self.total_lookups
        return round(self.hits / total, 4) if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_lookups": self.total_lookups,
            "hit_ratio": self.hit_ratio,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "size": self.size,
            "max_size": self.max_size,
        }


class StageExecutionCache:
    """Thread-safe, TTL/LRU bounded in-memory cache for decision graph nodes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._caches: dict[str, OrderedDict[str, NodeCacheEntry]] = defaultdict(OrderedDict)
        self._policies: dict[str, NodeCachePolicy] = {}
        self._stats: dict[str, NodeCacheStats] = defaultdict(NodeCacheStats)

    def register_policy(self, node_name: str, policy: NodeCachePolicy) -> None:
        """Register or update caching policy for a specific node."""
        with self._lock:
            self._policies[node_name] = policy
            self._stats[node_name].max_size = policy.max_size

    def get_policy(self, node_name: str) -> NodeCachePolicy | None:
        with self._lock:
            return self._policies.get(node_name)

    def compute_key(self, node_name: str, context: PipelineContext) -> str:
        """Deterministically derive string cache key for the node and context."""
        policy = self._policies.get(node_name)
        if policy is not None and policy.key_fn is not None:
            return policy.key_fn(context)

        tx = context.transaction
        if policy is not None and policy.key_fields is not None:
            extracted = {k: tx.get(k) for k in policy.key_fields}
        else:
            extracted = tx

        canonical_json = json.dumps(extracted, sort_keys=True, default=str)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def get(self, node_name: str, key: str, now: float | None = None) -> tuple[bool, Any]:
        """Look up cached node output. Returns (hit, output)."""
        current_time = monotonic() if now is None else now
        with self._lock:
            policy = self._policies.get(node_name)
            ttl = policy.ttl_seconds if policy is not None else 60.0

            cache = self._caches[node_name]
            stats = self._stats[node_name]

            if key not in cache:
                stats.misses += 1
                return False, None

            entry = cache[key]
            if entry.is_expired(current_time, ttl):
                del cache[key]
                stats.expirations += 1
                stats.misses += 1
                stats.size = len(cache)
                return False, None

            cache.move_to_end(key)
            stats.hits += 1
            out = (
                copy.deepcopy(entry.output)
                if isinstance(entry.output, (dict, list))
                else entry.output
            )
            return True, out

    def put(self, node_name: str, key: str, output: Any, now: float | None = None) -> None:
        """Store node output in cache, evicting oldest item if exceeding max_size."""
        current_time = monotonic() if now is None else now
        with self._lock:
            policy = self._policies.get(node_name)
            max_size = policy.max_size if policy is not None else 1000

            cache = self._caches[node_name]
            stats = self._stats[node_name]
            stats.max_size = max_size

            if key in cache:
                del cache[key]
            elif len(cache) >= max_size and len(cache) > 0:
                cache.popitem(last=False)
                stats.evictions += 1

            stored_output = copy.deepcopy(output) if isinstance(output, (dict, list)) else output
            cache[key] = NodeCacheEntry(
                created_at=current_time,
                output=stored_output,
                key=key,
            )
            stats.size = len(cache)

    def clear(self, node_name: str | None = None) -> None:
        """Clear cache for a specific node or all nodes."""
        with self._lock:
            if node_name is not None:
                self._caches[node_name].clear()
                self._stats[node_name].size = 0
            else:
                for c in self._caches.values():
                    c.clear()
                for s in self._stats.values():
                    s.size = 0

    def get_stats(self, node_name: str) -> NodeCacheStats:
        """Retrieve copy of telemetry stats for a node."""
        with self._lock:
            st = self._stats[node_name]
            return NodeCacheStats(
                hits=st.hits,
                misses=st.misses,
                evictions=st.evictions,
                expirations=st.expirations,
                size=len(self._caches[node_name]),
                max_size=st.max_size,
            )

    def stats_dict(self) -> dict[str, Any]:
        """Aggregate summary of cache performance."""
        with self._lock:
            total_hits = sum(s.hits for s in self._stats.values())
            total_misses = sum(s.misses for s in self._stats.values())
            total_lookups = total_hits + total_misses
            total_evictions = sum(s.evictions for s in self._stats.values())
            total_expirations = sum(s.expirations for s in self._stats.values())
            total_size = sum(len(c) for c in self._caches.values())
            ratio = round(total_hits / total_lookups, 4) if total_lookups > 0 else 0.0

            node_details = {}
            for name, st in self._stats.items():
                node_details[name] = {
                    "hits": st.hits,
                    "misses": st.misses,
                    "total_lookups": st.total_lookups,
                    "hit_ratio": st.hit_ratio,
                    "evictions": st.evictions,
                    "expirations": st.expirations,
                    "size": len(self._caches[name]),
                    "max_size": st.max_size,
                }

            return {
                "total_hits": total_hits,
                "total_misses": total_misses,
                "total_lookups": total_lookups,
                "hit_ratio": ratio,
                "total_evictions": total_evictions,
                "total_expirations": total_expirations,
                "total_size": total_size,
                "nodes": node_details,
            }


class StageCircuitState(StrEnum):
    """Operational state of a pipeline stage circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class StageCircuitPolicy:
    """Configuration for per-stage circuit breaker protection."""

    enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    half_open_max_calls: int = 1
    count_timeouts: bool = True
    count_exceptions: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.failure_threshold, bool) or self.failure_threshold < 1:
            raise ValueError("failure_threshold must be an integer >= 1")
        if (
            isinstance(self.recovery_timeout_seconds, bool)
            or not math.isfinite(self.recovery_timeout_seconds)
            or self.recovery_timeout_seconds <= 0.0
        ):
            raise ValueError("recovery_timeout_seconds must be a positive finite number")
        if isinstance(self.half_open_max_calls, bool) or self.half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be an integer >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_seconds": self.recovery_timeout_seconds,
            "half_open_max_calls": self.half_open_max_calls,
            "count_timeouts": self.count_timeouts,
            "count_exceptions": self.count_exceptions,
        }


@dataclass
class StageCircuitStats:
    """Runtime telemetry counters for a single stage circuit breaker."""

    state: StageCircuitState = StageCircuitState.CLOSED
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    total_trips: int = 0
    short_circuited_count: int = 0
    last_failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "total_trips": self.total_trips,
            "short_circuited_count": self.short_circuited_count,
            "last_failure_reason": self.last_failure_reason,
        }


class StageCircuitBreaker:
    """Thread-safe per-stage circuit breaker tracking consecutive failures and recovery."""

    def __init__(self, policy: StageCircuitPolicy | None = None) -> None:
        self.policy = policy or StageCircuitPolicy()
        self._lock = threading.Lock()
        self._state: StageCircuitState = StageCircuitState.CLOSED
        self._consecutive_failures: int = 0
        self._total_failures: int = 0
        self._total_successes: int = 0
        self._total_trips: int = 0
        self._short_circuited_count: int = 0
        self._last_failure_reason: str | None = None
        self._opened_at: float | None = None
        self._half_open_in_flight: int = 0
        self._half_open_successes: int = 0

    @property
    def state(self) -> StageCircuitState:
        """Effective state of the circuit breaker."""
        return self.get_stats().state

    def allow_execution(self, now: float | None = None) -> bool:
        """Return True if the stage is allowed to execute, False if short-circuited."""
        if not self.policy.enabled:
            return True
        current_time = monotonic() if now is None else now
        with self._lock:
            if self._state == StageCircuitState.CLOSED:
                return True
            if self._state == StageCircuitState.OPEN:
                if (
                    self._opened_at is not None
                    and (current_time - self._opened_at) >= self.policy.recovery_timeout_seconds
                ):
                    self._state = StageCircuitState.HALF_OPEN
                    self._half_open_in_flight = 1
                    self._half_open_successes = 0
                    return True
                self._short_circuited_count += 1
                return False
            if self._half_open_in_flight < self.policy.half_open_max_calls:
                self._half_open_in_flight += 1
                return True
            self._short_circuited_count += 1
            return False

    def record_success(self) -> None:
        """Record a successful stage execution and advance half-open recovery."""
        if not self.policy.enabled:
            return
        with self._lock:
            self._total_successes += 1
            self._consecutive_failures = 0
            if self._state == StageCircuitState.HALF_OPEN:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._half_open_successes += 1
                if self._half_open_successes >= self.policy.half_open_max_calls:
                    self._state = StageCircuitState.CLOSED
                    self._opened_at = None
                    self._half_open_in_flight = 0
                    self._half_open_successes = 0

    def record_failure(
        self,
        *,
        reason: str = "error",
        is_timeout: bool = False,
        now: float | None = None,
    ) -> bool:
        """Record a stage failure or timeout. Returns True if circuit is now OPEN."""
        if not self.policy.enabled:
            return False
        if is_timeout and not self.policy.count_timeouts:
            return False
        if not is_timeout and not self.policy.count_exceptions:
            return False
        current_time = monotonic() if now is None else now
        with self._lock:
            self._total_failures += 1
            self._consecutive_failures += 1
            self._last_failure_reason = reason
            if self._state == StageCircuitState.HALF_OPEN or (
                self._state == StageCircuitState.CLOSED
                and self._consecutive_failures >= self.policy.failure_threshold
            ):
                self._state = StageCircuitState.OPEN
                self._opened_at = current_time
                self._total_trips += 1
                self._half_open_in_flight = 0
                self._half_open_successes = 0
                return True
            return self._state == StageCircuitState.OPEN

    def reset(self) -> None:
        """Manually reset the circuit breaker state back to CLOSED."""
        with self._lock:
            self._state = StageCircuitState.CLOSED
            self._consecutive_failures = 0
            self._opened_at = None
            self._half_open_in_flight = 0
            self._half_open_successes = 0

    def get_stats(self, now: float | None = None) -> StageCircuitStats:
        """Return a snapshot of current circuit breaker statistics."""
        current_time = monotonic() if now is None else now
        with self._lock:
            effective_state = self._state
            if (
                effective_state == StageCircuitState.OPEN
                and self._opened_at is not None
                and (current_time - self._opened_at) >= self.policy.recovery_timeout_seconds
            ):
                effective_state = StageCircuitState.HALF_OPEN
            return StageCircuitStats(
                state=effective_state,
                consecutive_failures=self._consecutive_failures,
                total_failures=self._total_failures,
                total_successes=self._total_successes,
                total_trips=self._total_trips,
                short_circuited_count=self._short_circuited_count,
                last_failure_reason=self._last_failure_reason,
            )


class StageCircuitRegistry:
    """Thread-safe registry managing per-node circuit breakers across a decision graph."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._breakers: dict[str, StageCircuitBreaker] = {}

    def register_policy(
        self,
        node_name: str,
        policy: StageCircuitPolicy,
    ) -> StageCircuitBreaker:
        """Register or replace a circuit breaker policy for a graph node."""
        with self._lock:
            breaker = StageCircuitBreaker(policy)
            self._breakers[node_name] = breaker
            return breaker

    def get_breaker(self, node_name: str) -> StageCircuitBreaker | None:
        """Retrieve registered circuit breaker for a node, if any."""
        with self._lock:
            return self._breakers.get(node_name)

    def allow_execution(self, node_name: str, now: float | None = None) -> bool:
        """Check whether a stage node is permitted to execute."""
        breaker = self.get_breaker(node_name)
        if breaker is None:
            return True
        return breaker.allow_execution(now=now)

    def record_success(self, node_name: str) -> None:
        """Record successful stage completion for a node."""
        breaker = self.get_breaker(node_name)
        if breaker is not None:
            breaker.record_success()

    def record_failure(
        self,
        node_name: str,
        *,
        reason: str = "error",
        is_timeout: bool = False,
        now: float | None = None,
    ) -> bool:
        """Record stage failure/timeout for a node. Returns True if circuit is OPEN."""
        breaker = self.get_breaker(node_name)
        if breaker is None:
            return False
        return breaker.record_failure(reason=reason, is_timeout=is_timeout, now=now)

    def reset(self, node_name: str | None = None) -> list[str]:
        """Reset one or all registered stage circuit breakers. Returns reset node names."""
        with self._lock:
            if node_name is not None:
                if node_name in self._breakers:
                    self._breakers[node_name].reset()
                    return [node_name]
                return []
            for b in self._breakers.values():
                b.reset()
            return sorted(self._breakers.keys())

    @property
    def breakers(self) -> dict[str, StageCircuitBreaker]:
        """Thread-safe snapshot mapping of node names to their circuit breakers."""
        with self._lock:
            return dict(self._breakers)

    def summary(self, now: float | None = None) -> dict[str, Any]:
        """Return aggregate and per-node circuit breaker telemetry."""
        return self.stats_dict(now=now)

    def stats_dict(self, now: float | None = None) -> dict[str, Any]:
        """Return aggregate and per-node circuit breaker telemetry."""
        with self._lock:
            items = list(self._breakers.items())
        total_trips = 0
        total_short_circuits = 0
        open_nodes: list[str] = []
        half_open_nodes: list[str] = []
        nodes_map: dict[str, dict[str, Any]] = {}
        for name, breaker in items:
            st = breaker.get_stats(now=now)
            total_trips += st.total_trips
            total_short_circuits += st.short_circuited_count
            if st.state == StageCircuitState.OPEN:
                open_nodes.append(name)
            elif st.state == StageCircuitState.HALF_OPEN:
                half_open_nodes.append(name)
            node_entry = st.to_dict()
            node_entry["policy"] = breaker.policy.to_dict()
            nodes_map[name] = node_entry
        return {
            "total_trips": total_trips,
            "total_short_circuits": total_short_circuits,
            "open_nodes": sorted(open_nodes),
            "half_open_nodes": sorted(half_open_nodes),
            "nodes": nodes_map,
        }


@dataclass
class PipelineNode:
    """Configured execution node in the decision graph."""

    name: str
    stage_type: StageType
    processor: Callable[[PipelineContext], Any] | StageProcessorProtocol
    dependencies: list[str] = field(default_factory=list)
    timeout_seconds: float | None = None
    fallback_value: Any = None
    required: bool = False
    condition: Callable[[PipelineContext], bool] | None = None
    cache_policy: NodeCachePolicy | None = None
    circuit_policy: StageCircuitPolicy | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert node definition to a serializable dictionary."""
        is_primitive = isinstance(
            self.fallback_value, (dict, list, str, int, float, bool, type(None))
        )
        safe_fallback = self.fallback_value if is_primitive else str(self.fallback_value)
        return {
            "name": self.name,
            "stage_type": self.stage_type.value,
            "dependencies": list(self.dependencies),
            "timeout_seconds": self.timeout_seconds,
            "fallback_value": safe_fallback,
            "required": self.required,
            "has_condition": self.condition is not None,
            "cache_policy": self.cache_policy.to_dict() if self.cache_policy is not None else None,
            "circuit_policy": (
                self.circuit_policy.to_dict() if self.circuit_policy is not None else None
            ),
        }


@dataclass(frozen=True)
class ExecutionPlan:
    """Topologically ordered execution plan organized into parallel waves."""

    waves: tuple[tuple[str, ...], ...]
    topological_order: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "waves": [list(wave) for wave in self.waves],
            "topological_order": list(self.topological_order),
        }


class DecisionGraph:
    """Directed Acyclic Graph (DAG) for fraud evaluation stage scheduling."""

    def __init__(self, name: str = "fraud_decision_graph") -> None:
        self.name = name
        self.nodes: dict[str, PipelineNode] = {}
        self.edges: list[PipelineEdge] = []
        self.metadata: dict[str, Any] = {}

    def add_node(self, node: PipelineNode) -> DecisionGraph:
        """Register a pipeline node with the decision graph."""
        if node.name in self.nodes:
            raise PipelineValidationError(f"Duplicate node name '{node.name}' in decision graph.")
        self.nodes[node.name] = node
        for dep in node.dependencies:
            self.edges.append(PipelineEdge(source=dep, target=node.name))
        return self

    def add_edge(self, source: str, target: str) -> DecisionGraph:
        """Explicitly add a dependency edge source -> target."""
        if target not in self.nodes:
            raise PipelineValidationError(f"Target node '{target}' does not exist.")
        if source not in self.nodes[target].dependencies:
            self.nodes[target].dependencies.append(source)
            self.edges.append(PipelineEdge(source=source, target=target))
        return self

    def validate(self) -> None:
        """Validate DAG integrity, missing dependencies, and absence of cycles."""
        for name, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise PipelineValidationError(
                        f"Node '{name}' references unknown dependency '{dep}'."
                    )
                if dep == name:
                    raise PipelineCycleError(f"Node '{name}' depends on itself.")

        # Cycle detection using 3-color DFS
        visited: dict[str, int] = dict.fromkeys(self.nodes, 0)

        def dfs(curr: str, path: list[str]) -> None:
            visited[curr] = 1
            path.append(curr)
            for dep in self.nodes[curr].dependencies:
                if visited[dep] == 1:
                    cycle = " -> ".join([*path, dep])
                    raise PipelineCycleError(f"Cycle detected in decision graph: {cycle}")
                if visited[dep] == 0:
                    dfs(dep, path)
            path.pop()
            visited[curr] = 2

        for node_name in self.nodes:
            if visited[node_name] == 0:
                dfs(node_name, [])

    def topological_sort(self) -> list[str]:
        """Compute a linear topological order of node names."""
        self.validate()

        in_degree: dict[str, int] = dict.fromkeys(self.nodes, 0)
        adjacency: dict[str, list[str]] = defaultdict(list)

        for name, node in self.nodes.items():
            in_degree[name] = len(node.dependencies)
            for dep in node.dependencies:
                adjacency[dep].append(name)

        queue: deque[str] = deque([name for name, deg in in_degree.items() if deg == 0])
        order: list[str] = []

        while queue:
            curr = queue.popleft()
            order.append(curr)
            for neighbor in adjacency[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.nodes):
            raise PipelineCycleError("Graph contains cycles or unresolvable dependencies.")

        return order

    def build_execution_plan(self) -> ExecutionPlan:
        """Compile nodes into parallel execution waves using level-by-level scheduling."""
        self.validate()

        adjacency: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {}

        for name, node in self.nodes.items():
            in_degree[name] = len(node.dependencies)
            for dep in node.dependencies:
                adjacency[dep].append(name)

        current_wave = [name for name, deg in in_degree.items() if deg == 0]
        waves: list[list[str]] = []
        topological_order: list[str] = []

        while current_wave:
            current_wave.sort()
            waves.append(current_wave)
            topological_order.extend(current_wave)
            next_wave: list[str] = []
            for name in current_wave:
                for neighbor in adjacency[name]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_wave.append(neighbor)
            current_wave = next_wave

        if len(topological_order) != len(self.nodes):
            raise PipelineCycleError("Could not build valid execution plan: circular dependency.")

        return ExecutionPlan(
            waves=tuple(tuple(wave) for wave in waves),
            topological_order=tuple(topological_order),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete decision graph topology."""
        self.validate()
        plan = self.build_execution_plan()
        return {
            "name": self.name,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "execution_plan": plan.to_dict(),
            "metadata": dict(self.metadata),
        }


# ============================================================================
# Pluggable Stage Processors
# ============================================================================


class EnrichmentStageProcessor:
    """Enriches transaction features using online feature store and velocity buffers."""

    def __init__(
        self,
        feature_store: FeatureStoreProtocol | None = None,
        velocity_buffer: VelocityWindowBuffer | None = None,
        velocity_config: VelocityConfig | None = None,
        expected_columns: Sequence[str] | None = None,
        update_velocity: bool = True,
    ) -> None:
        self.feature_store = feature_store
        self.velocity_buffer = velocity_buffer
        self.velocity_config = velocity_config
        self.expected_columns = set(expected_columns) if expected_columns else None
        self.update_velocity = update_velocity

    def __call__(self, context: PipelineContext) -> dict[str, Any]:
        tx = dict(context.transaction)
        enriched_features: list[str] = []
        feature_views_applied: list[str] = []
        velocity_features: list[str] = []

        if self.feature_store is not None:
            for view_name in self.feature_store.list_views():
                view = self.feature_store.get_view(view_name)
                if view is None:
                    continue
                eid = tx.get(view.entity_key)
                if eid is not None:
                    obs_t = tx.get("Time")
                    as_of = float(obs_t) if obs_t is not None else None
                    str_eid = (
                        str(int(eid))
                        if isinstance(eid, (int, float)) and float(eid).is_integer()
                        else str(eid)
                    )
                    res = self.feature_store.lookup_online(
                        entity_key=view.entity_key,
                        entity_id=str_eid,
                        view_name=view.name,
                        as_of_time=as_of,
                    )
                    if res.found:
                        if view.name not in feature_views_applied:
                            feature_views_applied.append(view.name)
                        for fname, fval in res.values.items():
                            if (
                                self.expected_columns is None
                                or fname in self.expected_columns
                                or fname in tx
                            ):
                                tx[fname] = float(fval)
                                if fname not in enriched_features:
                                    enriched_features.append(fname)

        if self.velocity_buffer is not None:
            prev_keys = set(tx.keys())
            tx = dict(
                enrich_record_velocity(
                    tx,
                    buffer=self.velocity_buffer,
                    config=self.velocity_config,
                    update=self.update_velocity,
                )
            )
            v_cols = [k for k in tx if k not in prev_keys]
            velocity_features.extend(v_cols)

        context.transaction = tx
        context.metadata["enriched_features"] = enriched_features
        context.metadata["feature_views_applied"] = feature_views_applied
        context.metadata["velocity_features"] = velocity_features

        return {
            "enriched_features": enriched_features,
            "feature_views_applied": feature_views_applied,
            "velocity_features": velocity_features,
        }


class RuleStageProcessor:
    """Evaluates declarative rules against current transaction features."""

    def __init__(
        self,
        rules: RuleSet | None = None,
        short_circuit_actions: Sequence[RuleAction] | None = None,
    ) -> None:
        self.rules = rules
        self.short_circuit_actions = (
            tuple(short_circuit_actions)
            if short_circuit_actions is not None
            else (RuleAction.DENY, RuleAction.ALLOW)
        )

    def __call__(self, context: PipelineContext) -> dict[str, Any]:
        if self.rules is None or not self.rules.rules:
            return {
                "matched": False,
                "matched_rule_id": None,
                "matched_rule_name": None,
                "action": None,
                "reason": "",
                "short_circuited": False,
            }

        eval_res = self.rules.evaluate_record(context.transaction)
        short_circuit = False
        if eval_res.matched and eval_res.action in self.short_circuit_actions:
            short_circuit = True
            context.metadata["short_circuit"] = True
            context.metadata["short_circuit_action"] = eval_res.action.value
            context.metadata["matched_rule_id"] = eval_res.matched_rule_id
            context.metadata["matched_rule_name"] = eval_res.matched_rule_name

        return {
            "matched": eval_res.matched,
            "matched_rule_id": eval_res.matched_rule_id,
            "matched_rule_name": eval_res.matched_rule_name,
            "action": eval_res.action.value if eval_res.action is not None else None,
            "reason": eval_res.reason,
            "short_circuited": short_circuit,
        }


class InferenceStageProcessor:
    """Computes fraud risk score from model given transaction features."""

    def __init__(
        self,
        model: FraudModel | None = None,
        explain: bool = False,
        skip_if_short_circuited: bool = True,
    ) -> None:
        self.model = model
        self.explain = explain
        self.skip_if_short_circuited = skip_if_short_circuited

    def __call__(self, context: PipelineContext) -> dict[str, Any]:
        if self.model is None:
            return {
                "skipped": False,
                "probability": None,
                "raw_probability": None,
                "contributions": None,
            }

        if self.skip_if_short_circuited and context.metadata.get("short_circuit"):
            return {
                "skipped": True,
                "probability": None,
                "raw_probability": None,
                "contributions": None,
                "model_threshold": self.model.threshold,
            }

        row_dict: dict[str, float] = {}
        for f in self.model.feature_names:
            v = context.transaction.get(f, 0.0)
            try:
                row_dict[f] = float(v)
            except (ValueError, TypeError):
                row_dict[f] = 0.0
        frame = pd.DataFrame([row_dict])
        context.frame = frame

        probs = self.model.predict_probabilities(frame, raw=True)
        prob = float(probs[0])

        contributions: dict[str, float] | None = None
        if self.explain:
            exps = self.model.explain_local(frame)
            if exps:
                contributions = exps[0]

        return {
            "skipped": False,
            "probability": prob,
            "raw_probability": prob,
            "model_threshold": self.model.threshold,
            "contributions": contributions,
            "model_version": getattr(self.model, "version", "primary"),
        }


class CalibrationStageProcessor:
    """Applies post-hoc probability recalibration to model output."""

    def __init__(
        self,
        recalibrator: BaseRecalibrator | None = None,
        inference_node_name: str = "inference",
    ) -> None:
        self.recalibrator = recalibrator
        self.inference_node_name = inference_node_name

    def __call__(self, context: PipelineContext) -> dict[str, Any]:
        inf_output = context.get_output(self.inference_node_name)
        if not inf_output or inf_output.get("skipped") or inf_output.get("probability") is None:
            return {
                "calibrated_probability": None,
                "raw_probability": None,
                "method": "none",
            }

        raw_prob = float(inf_output["probability"])
        if self.recalibrator is not None:
            calibrated_arr = self.recalibrator.transform(np.array([raw_prob], dtype=float))
            cal_prob = float(calibrated_arr[0])
            method = getattr(self.recalibrator, "method", "custom")
            method_str = method.value if hasattr(method, "value") else str(method)
        else:
            cal_prob = raw_prob
            method_str = "none"

        return {
            "calibrated_probability": cal_prob,
            "raw_probability": raw_prob,
            "method": method_str,
        }


class ActionAggregatorStageProcessor:
    """Arbitrates deterministic rules and calibrated probability into a final decision."""

    def __init__(
        self,
        threshold: float = 0.5,
        review_threshold: float | None = None,
        deny_threshold: float | None = None,
        rule_precedence: RulePrecedence = RulePrecedence.RULES_OVERRIDE_MODEL,
        rules_node_name: str = "rules",
        calibration_node_name: str = "calibration",
        inference_node_name: str = "inference",
    ) -> None:
        self.threshold = threshold
        self.review_threshold = review_threshold
        self.deny_threshold = deny_threshold
        self.rule_precedence = rule_precedence
        self.rules_node_name = rules_node_name
        self.calibration_node_name = calibration_node_name
        self.inference_node_name = inference_node_name

    def __call__(self, context: PipelineContext) -> dict[str, Any]:
        rules_out = context.get_output(self.rules_node_name) or {}
        cal_out = context.get_output(self.calibration_node_name) or {}
        inf_out = context.get_output(self.inference_node_name) or {}

        matched_rule = rules_out.get("matched_rule_id")
        rule_action = rules_out.get("action")
        contributions = inf_out.get("contributions")

        prob = cal_out.get("calibrated_probability")
        raw_prob = cal_out.get("raw_probability", inf_out.get("raw_probability"))

        if prob is None and inf_out.get("probability") is not None:
            prob = inf_out.get("probability")

        model_action: str
        if prob is not None:
            if self.review_threshold is not None and self.deny_threshold is not None:
                if prob >= self.deny_threshold:
                    model_action = DecisionAction.DENY.value
                elif prob >= self.review_threshold:
                    model_action = DecisionAction.CHALLENGE.value
                else:
                    model_action = DecisionAction.ALLOW.value
            else:
                model_action = (
                    DecisionAction.DENY.value
                    if prob >= self.threshold
                    else DecisionAction.ALLOW.value
                )
        else:
            model_action = DecisionAction.ALLOW.value

        final_action: str
        if matched_rule is not None and rule_action is not None:
            if (
                self.rule_precedence == RulePrecedence.MODEL_OVERRIDES_RULES
                and model_action == DecisionAction.DENY.value
            ):
                final_action = model_action
            else:
                final_action = str(rule_action)
        else:
            final_action = model_action

        is_fraud = final_action == DecisionAction.DENY.value
        final_prob = float(prob) if prob is not None else (1.0 if is_fraud else 0.0)

        return {
            "decision": final_action,
            "is_fraud": is_fraud,
            "fraud_probability": round(final_prob, 5),
            "raw_probability": round(float(raw_prob), 5) if raw_prob is not None else None,
            "matched_rule": matched_rule,
            "rule_action": rule_action,
            "contributions": contributions,
            "threshold": self.threshold,
            "review_threshold": self.review_threshold,
            "deny_threshold": self.deny_threshold,
        }


@dataclass
class PipelineExecutionResult:
    """Comprehensive output, trace, and telemetry for a pipeline execution."""

    execution_id: str
    success: bool
    total_latency_ms: float
    stage_results: dict[str, StageExecutionResult]
    execution_path: list[str]
    final_decision: dict[str, Any]
    degraded: bool = False
    degraded_nodes: list[str] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    speculative_executed: bool = False
    speculative_hit: bool = False
    circuit_short_circuits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert pipeline execution result to a serializable dictionary."""
        return {
            "execution_id": self.execution_id,
            "success": self.success,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "execution_path": self.execution_path,
            "stage_results": {k: v.to_dict() for k, v in self.stage_results.items()},
            "final_decision": self.final_decision,
            "degraded": self.degraded,
            "degraded_nodes": self.degraded_nodes,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "speculative_executed": self.speculative_executed,
            "speculative_hit": self.speculative_hit,
            "circuit_short_circuits": list(self.circuit_short_circuits),
        }

    @property
    def decision(self) -> str:
        return str(self.final_decision.get("decision", "ALLOW"))

    @property
    def is_fraud(self) -> bool:
        return bool(self.final_decision.get("is_fraud", False))

    @property
    def fraud_probability(self) -> float:
        return float(self.final_decision.get("fraud_probability", 0.0))

    @property
    def raw_probability(self) -> float | None:
        raw = self.final_decision.get("raw_probability")
        return float(raw) if raw is not None else None

    @property
    def matched_rule(self) -> str | None:
        res = self.final_decision.get("matched_rule")
        return str(res) if res is not None else None

    @property
    def rule_action(self) -> str | None:
        res = self.final_decision.get("rule_action")
        return str(res) if res is not None else None

    @property
    def contributions(self) -> dict[str, float] | None:
        res = self.final_decision.get("contributions")
        return dict(res) if isinstance(res, dict) else None


# ============================================================================
# Asynchronous Graph Executor & Pipeline Factory
# ============================================================================


class DecisionGraphExecutor:
    """Executes a DecisionGraph with concurrent waves, timeouts, and fail-soft fallback."""

    def __init__(
        self,
        graph: DecisionGraph,
        *,
        cache: StageExecutionCache | None = None,
        enable_cache: bool = True,
        enable_speculative: bool | None = None,
        speculative_nodes: Sequence[str] | None = None,
        circuit_registry: StageCircuitRegistry | None = None,
        enable_circuit_breaker: bool | None = None,
    ) -> None:
        self.graph = graph
        self.graph.validate()
        self.plan = self.graph.build_execution_plan()
        self.cache = cache or StageExecutionCache()
        self.enable_cache = enable_cache
        self.circuit_registry = circuit_registry or StageCircuitRegistry()
        self.enable_circuit_breaker = (
            enable_circuit_breaker
            if enable_circuit_breaker is not None
            else bool(self.graph.metadata.get("enable_circuit_breaker", False))
        )
        self.enable_speculative = (
            enable_speculative
            if enable_speculative is not None
            else bool(self.graph.metadata.get("speculative_inference", False))
        )

        if speculative_nodes is not None:
            self.speculative_nodes = tuple(speculative_nodes)
        elif self.enable_speculative:
            self.speculative_nodes = tuple(
                name
                for name, node in self.graph.nodes.items()
                if node.stage_type == StageType.INFERENCE and len(node.dependencies) > 0
            )
        else:
            self.speculative_nodes = ()

        for name, node in self.graph.nodes.items():
            if node.cache_policy is not None:
                self.cache.register_policy(name, node.cache_policy)
            if self.circuit_registry.get_breaker(name) is not None:
                continue
            if node.circuit_policy is not None:
                self.circuit_registry.register_policy(name, node.circuit_policy)
            elif self.enable_circuit_breaker:
                thresh = int(self.graph.metadata.get("circuit_failure_threshold", 5))
                rec_sec = float(self.graph.metadata.get("circuit_recovery_timeout_seconds", 30.0))
                self.circuit_registry.register_policy(
                    name,
                    StageCircuitPolicy(
                        enabled=True,
                        failure_threshold=thresh,
                        recovery_timeout_seconds=rec_sec,
                    ),
                )

    @property
    def circuits(self) -> StageCircuitRegistry:
        """Ergonomic accessor for the stage circuit registry."""
        return self.circuit_registry

    async def _execute_node(self, node_name: str, context: PipelineContext) -> None:
        node = self.graph.nodes[node_name]

        if not self.circuit_registry.allow_execution(node_name):
            logger.warning(
                "Pipeline stage '%s' short-circuited: circuit breaker is OPEN", node_name
            )
            if node.required:
                raise PipelineCircuitOpenError(
                    f"Required stage '{node_name}' cannot execute: circuit breaker is open."
                )
            context.results[node_name] = StageExecutionResult(
                node_name=node.name,
                stage_type=node.stage_type,
                status=StageStatus.CIRCUIT_OPEN,
                latency_ms=0.0,
                output=(
                    copy.deepcopy(node.fallback_value)
                    if isinstance(node.fallback_value, (dict, list))
                    else node.fallback_value
                ),
                error="Stage circuit breaker is open",
                degraded=True,
                short_circuited_by_breaker=True,
            )
            return

        if node.condition is not None:
            try:
                if not node.condition(context):
                    context.results[node_name] = StageExecutionResult(
                        node_name=node.name,
                        stage_type=node.stage_type,
                        status=StageStatus.SKIPPED,
                        output=node.fallback_value,
                    )
                    return
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as cond_exc:
                logger.warning("Condition for node '%s' raised: %s", node_name, cond_exc)

        cache_key: str | None = None
        if self.enable_cache:
            policy = self.cache.get_policy(node_name) or node.cache_policy
            if policy is not None and policy.enabled:
                cache_key = self.cache.compute_key(node_name, context)
                hit, cached_output = self.cache.get(node_name, cache_key)
                if hit:
                    self.circuit_registry.record_success(node_name)
                    context.results[node_name] = StageExecutionResult(
                        node_name=node.name,
                        stage_type=node.stage_type,
                        status=StageStatus.SUCCESS,
                        latency_ms=0.0,
                        output=cached_output,
                        degraded=False,
                        cached=True,
                    )
                    return

        start = perf_counter()
        try:
            is_async = inspect.iscoroutinefunction(node.processor)

            if is_async:
                proc_coro = node.processor(context)
            else:
                proc_coro = asyncio.to_thread(node.processor, context)

            if node.timeout_seconds is not None:
                output = await asyncio.wait_for(proc_coro, timeout=node.timeout_seconds)
            else:
                output = await proc_coro

            elapsed = (perf_counter() - start) * 1000.0
            self.circuit_registry.record_success(node_name)
            context.results[node_name] = StageExecutionResult(
                node_name=node.name,
                stage_type=node.stage_type,
                status=StageStatus.SUCCESS,
                latency_ms=elapsed,
                output=output,
                degraded=False,
                cached=False,
            )

            if self.enable_cache and cache_key is not None:
                self.cache.put(node_name, cache_key, output)

        except TimeoutError:
            elapsed = (perf_counter() - start) * 1000.0
            logger.warning(
                "Pipeline stage '%s' timed out after %.2f ms (budget %.2f s)",
                node_name,
                elapsed,
                node.timeout_seconds or 0.0,
            )
            self.circuit_registry.record_failure(node_name, reason="timeout", is_timeout=True)
            if node.required:
                raise PipelineTimeoutError(
                    f"Required stage '{node_name}' exceeded latency budget "
                    f"of {node.timeout_seconds}s."
                ) from None
            context.results[node_name] = StageExecutionResult(
                node_name=node.name,
                stage_type=node.stage_type,
                status=StageStatus.TIMEOUT,
                latency_ms=elapsed,
                output=node.fallback_value,
                error="Stage latency budget exceeded",
                degraded=True,
            )

        except Exception as exc:
            elapsed = (perf_counter() - start) * 1000.0
            logger.exception("Pipeline stage '%s' failed: %s", node_name, exc)
            self.circuit_registry.record_failure(node_name, reason=str(exc), is_timeout=False)
            if node.required:
                raise PipelineExecutionError(f"Required stage '{node_name}' failed: {exc}") from exc
            context.results[node_name] = StageExecutionResult(
                node_name=node.name,
                stage_type=node.stage_type,
                status=StageStatus.FALLBACK,
                latency_ms=elapsed,
                output=node.fallback_value,
                error=str(exc),
                degraded=True,
            )

    async def execute(
        self,
        transaction: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
        global_timeout_seconds: float | None = None,
    ) -> PipelineExecutionResult:
        """Execute decision graph across topologically scheduled parallel waves."""
        start_total = perf_counter()
        execution_id = uuid4().hex
        context = PipelineContext(
            transaction=dict(transaction),
            metadata=dict(metadata or {}),
        )

        initial_tx = dict(context.transaction)
        speculative_tasks: dict[str, tuple[asyncio.Task[None], PipelineContext]] = {}

        if self.enable_speculative and self.speculative_nodes:
            for s_name in self.speculative_nodes:
                if s_name in self.graph.nodes:
                    s_ctx = PipelineContext(
                        transaction=dict(initial_tx),
                        metadata=dict(context.metadata),
                    )
                    s_task = asyncio.create_task(self._execute_node(s_name, s_ctx))
                    speculative_tasks[s_name] = (s_task, s_ctx)
            if speculative_tasks:
                context.metadata["speculative_executed"] = True

        async def _execute_step(node_name: str) -> None:
            if node_name in speculative_tasks:
                s_task, s_ctx = speculative_tasks[node_name]
                node = self.graph.nodes[node_name]
                short_circuited = bool(context.metadata.get("short_circuit"))

                if (
                    isinstance(node.processor, InferenceStageProcessor)
                    and node.processor.model is not None
                ):
                    relevant_features = node.processor.model.feature_names
                else:
                    relevant_features = None

                features_changed = False
                if relevant_features is not None:
                    for f in relevant_features:
                        if initial_tx.get(f) != context.transaction.get(f):
                            features_changed = True
                            break
                else:
                    features_changed = initial_tx != context.transaction

                if not short_circuited and not features_changed:
                    try:
                        await s_task
                        s_res = s_ctx.results.get(node_name)
                        if s_res is not None and s_res.status == StageStatus.SUCCESS:
                            context.results[node_name] = StageExecutionResult(
                                node_name=s_res.node_name,
                                stage_type=s_res.stage_type,
                                status=s_res.status,
                                latency_ms=s_res.latency_ms,
                                output=s_res.output,
                                error=s_res.error,
                                degraded=s_res.degraded,
                                cached=s_res.cached,
                                speculative=True,
                            )
                            context.metadata["speculative_hit"] = True
                            return
                    except (
                        PipelineError,
                        ValueError,
                        TypeError,
                        KeyError,
                        AttributeError,
                        RuntimeError,
                        TimeoutError,
                        OSError,
                    ) as s_exc:
                        logger.debug(
                            "Speculative task for '%s' raised %s; falling back",
                            node_name,
                            s_exc,
                        )

                if not s_task.done():
                    s_task.cancel()
                context.metadata["speculative_discarded"] = True

            await self._execute_node(node_name, context)

        async def _run_waves() -> None:
            try:
                for wave in self.plan.waves:
                    tasks = [_execute_step(node_name) for node_name in wave]
                    await asyncio.gather(*tasks)
            finally:
                for s_task, _ in speculative_tasks.values():
                    if not s_task.done():
                        s_task.cancel()

        if global_timeout_seconds is not None:
            await asyncio.wait_for(_run_waves(), timeout=global_timeout_seconds)
        else:
            await _run_waves()

        total_latency = (perf_counter() - start_total) * 1000.0

        # Collect execution path in topological order
        execution_path = [
            node_name for node_name in self.plan.topological_order if node_name in context.results
        ]

        # Final decision is taken from ACTION stage node if present, else fallback
        action_node_name: str | None = None
        for name, node in self.graph.nodes.items():
            if node.stage_type == StageType.ACTION:
                action_node_name = name
                break
        if action_node_name is None and "action" in context.results:
            action_node_name = "action"

        final_decision: dict[str, Any] = {}
        if (
            action_node_name is not None
            and action_node_name in context.results
            and context.results[action_node_name].output
        ):
            final_decision = dict(context.results[action_node_name].output)
        else:
            final_decision = {
                "decision": "ALLOW",
                "is_fraud": False,
                "fraud_probability": 0.0,
                "raw_probability": None,
                "matched_rule": None,
                "rule_action": None,
                "contributions": None,
            }

        degraded_nodes = [node_name for node_name, res in context.results.items() if res.degraded]

        cache_hits = sum(1 for res in context.results.values() if res.cached)
        cache_misses = sum(
            1
            for name, res in context.results.items()
            if not res.cached
            and (
                self.cache.get_policy(name) is not None
                or getattr(self.graph.nodes.get(name), "cache_policy", None) is not None
            )
        )
        circuit_short_circuits = [
            node_name
            for node_name, res in context.results.items()
            if res.short_circuited_by_breaker
        ]

        return PipelineExecutionResult(
            execution_id=execution_id,
            success=True,
            total_latency_ms=total_latency,
            stage_results=context.results,
            execution_path=execution_path,
            final_decision=final_decision,
            degraded=len(degraded_nodes) > 0,
            degraded_nodes=degraded_nodes,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            speculative_executed=bool(context.metadata.get("speculative_executed", False)),
            speculative_hit=bool(context.metadata.get("speculative_hit", False)),
            circuit_short_circuits=circuit_short_circuits,
        )

    def execute_sync(
        self,
        transaction: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
        global_timeout_seconds: float | None = None,
    ) -> PipelineExecutionResult:
        """Synchronously execute graph, safely handling nested or ambient event loops."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self.execute(
                        transaction,
                        metadata=metadata,
                        global_timeout_seconds=global_timeout_seconds,
                    ),
                ).result()

        return asyncio.run(
            self.execute(
                transaction,
                metadata=metadata,
                global_timeout_seconds=global_timeout_seconds,
            )
        )


def create_default_fraud_pipeline(
    model: FraudModel,
    *,
    rules: RuleSet | None = None,
    rule_precedence: RulePrecedence = RulePrecedence.RULES_OVERRIDE_MODEL,
    feature_store: FeatureStoreProtocol | None = None,
    velocity_buffer: VelocityWindowBuffer | None = None,
    velocity_config: VelocityConfig | None = None,
    recalibrator: BaseRecalibrator | None = None,
    threshold: float | None = None,
    review_threshold: float | None = None,
    deny_threshold: float | None = None,
    enrichment_timeout_seconds: float | None = 0.25,
    inference_timeout_seconds: float | None = 0.25,
    calibration_timeout_seconds: float | None = 0.25,
    rules_timeout_seconds: float | None = 0.25,
    action_timeout_seconds: float | None = 0.25,
    explain: bool = False,
    enable_cache: bool = False,
    cache_rules: bool = True,
    cache_inference: bool = False,
    cache_ttl_seconds: float = 60.0,
    cache_max_size: int = 1000,
    speculative_inference: bool = False,
    enable_circuit_breaker: bool = False,
    circuit_failure_threshold: int = 5,
    circuit_recovery_timeout_seconds: float = 30.0,
) -> DecisionGraph:
    """Build a standard, production-ready fraud decision execution graph."""
    applied_threshold = model.threshold if threshold is None else threshold
    graph = DecisionGraph("production_fraud_pipeline")
    graph.metadata["speculative_inference"] = speculative_inference
    graph.metadata["enable_cache"] = enable_cache
    graph.metadata["enable_circuit_breaker"] = enable_circuit_breaker
    graph.metadata["circuit_failure_threshold"] = circuit_failure_threshold
    graph.metadata["circuit_recovery_timeout_seconds"] = circuit_recovery_timeout_seconds

    circuit_policy: StageCircuitPolicy | None = None
    if enable_circuit_breaker:
        circuit_policy = StageCircuitPolicy(
            enabled=True,
            failure_threshold=circuit_failure_threshold,
            recovery_timeout_seconds=circuit_recovery_timeout_seconds,
        )

    rules_cache_policy: NodeCachePolicy | None = None
    if enable_cache and cache_rules:
        rules_cache_policy = NodeCachePolicy(
            enabled=True,
            ttl_seconds=cache_ttl_seconds,
            max_size=cache_max_size,
        )

    inference_cache_policy: NodeCachePolicy | None = None
    if enable_cache and cache_inference:
        inference_cache_policy = NodeCachePolicy(
            enabled=True,
            ttl_seconds=cache_ttl_seconds,
            max_size=cache_max_size,
            key_fields=tuple(model.feature_names),
        )

    enrich_proc = EnrichmentStageProcessor(
        feature_store=feature_store,
        velocity_buffer=velocity_buffer,
        velocity_config=velocity_config,
        expected_columns=model.feature_names,
    )
    graph.add_node(
        PipelineNode(
            name="enrichment",
            stage_type=StageType.ENRICHMENT,
            processor=enrich_proc,
            timeout_seconds=enrichment_timeout_seconds,
            fallback_value={},
            required=False,
            circuit_policy=circuit_policy,
        )
    )

    rule_proc = RuleStageProcessor(rules=rules)
    graph.add_node(
        PipelineNode(
            name="rules",
            stage_type=StageType.RULES,
            processor=rule_proc,
            dependencies=["enrichment"],
            timeout_seconds=rules_timeout_seconds,
            fallback_value={"matched": False, "short_circuited": False},
            required=False,
            cache_policy=rules_cache_policy,
            circuit_policy=circuit_policy,
        )
    )

    inf_proc = InferenceStageProcessor(model=model, explain=explain)
    graph.add_node(
        PipelineNode(
            name="inference",
            stage_type=StageType.INFERENCE,
            processor=inf_proc,
            dependencies=["enrichment"],
            timeout_seconds=inference_timeout_seconds,
            fallback_value={
                "skipped": False,
                "probability": applied_threshold,
                "raw_probability": applied_threshold,
                "contributions": None,
            },
            required=False,
            cache_policy=inference_cache_policy,
            circuit_policy=circuit_policy,
        )
    )

    cal_proc = CalibrationStageProcessor(
        recalibrator=recalibrator,
        inference_node_name="inference",
    )
    graph.add_node(
        PipelineNode(
            name="calibration",
            stage_type=StageType.CALIBRATION,
            processor=cal_proc,
            dependencies=["inference"],
            timeout_seconds=calibration_timeout_seconds,
            fallback_value={
                "calibrated_probability": applied_threshold,
                "raw_probability": applied_threshold,
                "method": "fallback",
            },
            required=False,
            circuit_policy=circuit_policy,
        )
    )

    act_proc = ActionAggregatorStageProcessor(
        threshold=applied_threshold,
        review_threshold=review_threshold,
        deny_threshold=deny_threshold,
        rule_precedence=rule_precedence,
        rules_node_name="rules",
        calibration_node_name="calibration",
        inference_node_name="inference",
    )
    graph.add_node(
        PipelineNode(
            name="action",
            stage_type=StageType.ACTION,
            processor=act_proc,
            dependencies=["rules", "calibration"],
            timeout_seconds=action_timeout_seconds,
            required=True,
        )
    )

    return graph


# ============================================================================
# Dynamic Decision Graph Replay & Counterfactual Policy Optimization
# ============================================================================


@dataclass(frozen=True)
class PipelineDiscrepancyDetail:
    """Detailed discrepancy record between original audit log and replayed decision graph."""

    event_id: str
    transaction_index: int
    logged_probability: float
    replayed_probability: float
    score_difference: float
    logged_decision: str
    replayed_decision: str
    decision_flipped: bool
    replayed_rule: str | None = None
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert discrepancy detail to a serializable dictionary."""
        return {
            "event_id": self.event_id,
            "transaction_index": self.transaction_index,
            "logged_probability": self.logged_probability,
            "replayed_probability": self.replayed_probability,
            "score_difference": self.score_difference,
            "logged_decision": self.logged_decision,
            "replayed_decision": self.replayed_decision,
            "decision_flipped": self.decision_flipped,
            "replayed_rule": self.replayed_rule,
            "degraded": self.degraded,
        }


@dataclass(frozen=True)
class PipelineReplayReport:
    """Outcome and telemetry from replaying historical audit events through a decision graph."""

    total_events: int
    scoring_events: int
    replayed_events: int
    skipped_events: int
    total_transactions: int
    score_discrepancies: int
    decision_flips: int
    flip_rate: float
    max_absolute_difference: float
    mean_absolute_difference: float
    tolerance: float
    status: str  # "MATCH", "DIVERGENT", or "EMPTY"
    transition_matrix: PolicyTransitionMatrix
    baseline_decision_counts: dict[str, int]
    replayed_decision_counts: dict[str, int]
    mean_latency_ms: float
    p95_latency_ms: float
    degraded_executions: int
    stage_latencies: dict[str, dict[str, float]]
    discrepancies: tuple[PipelineDiscrepancyDetail, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Convert replay report to a serializable dictionary."""
        return {
            "total_events": self.total_events,
            "scoring_events": self.scoring_events,
            "replayed_events": self.replayed_events,
            "skipped_events": self.skipped_events,
            "total_transactions": self.total_transactions,
            "score_discrepancies": self.score_discrepancies,
            "decision_flips": self.decision_flips,
            "flip_rate": self.flip_rate,
            "max_absolute_difference": self.max_absolute_difference,
            "mean_absolute_difference": self.mean_absolute_difference,
            "tolerance": self.tolerance,
            "status": self.status,
            "transition_matrix": self.transition_matrix.to_dict(),
            "baseline_decision_counts": dict(self.baseline_decision_counts),
            "replayed_decision_counts": dict(self.replayed_decision_counts),
            "mean_latency_ms": self.mean_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "degraded_executions": self.degraded_executions,
            "stage_latencies": self.stage_latencies,
            "discrepancies": [d.to_dict() for d in self.discrepancies],
        }


def replay_audit_log_through_pipeline(
    audit_log_path: Path | str,
    pipeline: DecisionGraph | DecisionGraphExecutor,
    *,
    data_path: Path | str | None = None,
    tolerance: float = 1e-4,
    max_discrepancies_to_record: int = 100,
) -> PipelineReplayReport:
    """Replay historical scoring events from a JSONL audit log through a candidate decision graph.

    Compares logged predictions and decisions against replayed pipeline outcomes,
    computing probability discrepancies, decision turnover matrices, and stage-by-stage latencies.
    """
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise ValueError("tolerance must be a finite, non-negative float.")

    if (
        isinstance(max_discrepancies_to_record, bool)
        or not isinstance(max_discrepancies_to_record, int)
        or max_discrepancies_to_record < 0
    ):
        raise ValueError("max_discrepancies_to_record must be a non-negative integer.")

    audit_path = Path(audit_log_path)
    if not audit_path.is_file():
        raise FileNotFoundError(f"Audit log file not found: {audit_path}")

    external_df: pd.DataFrame | None = None
    if data_path is not None:
        data_p = Path(data_path)
        if not data_p.is_file():
            raise FileNotFoundError(f"External observations file not found: {data_p}")
        external_df = pd.read_csv(data_p)

    executor = (
        pipeline if isinstance(pipeline, DecisionGraphExecutor) else DecisionGraphExecutor(pipeline)
    )

    total_events = 0
    scoring_events = 0
    replayed_events = 0
    skipped_events = 0
    total_transactions = 0
    score_discrepancies = 0
    decision_flips = 0
    degraded_executions = 0

    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    latencies: list[float] = []
    stage_latencies_collector: dict[str, list[float]] = {name: [] for name in executor.graph.nodes}

    b_actions_list: list[str] = []
    r_actions_list: list[str] = []
    discrepancies_list: list[PipelineDiscrepancyDetail] = []

    external_offset = 0

    with audit_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            total_events += 1
            try:
                event_dict = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            if not isinstance(event_dict, dict) or event_dict.get("event_type") != "scoring":
                continue

            scoring_events += 1
            payload = event_dict.get("payload")
            if not isinstance(payload, dict):
                skipped_events += 1
                continue

            logged_predictions = payload.get("predictions")
            if not isinstance(logged_predictions, list) or not logged_predictions:
                skipped_events += 1
                continue

            inline_features = payload.get("features")
            records_to_score: list[dict[str, Any]] = []

            if isinstance(inline_features, list) and inline_features:
                if len(inline_features) != len(logged_predictions):
                    skipped_events += 1
                    continue
                records_to_score = [dict(f) for f in inline_features if isinstance(f, dict)]
                if len(records_to_score) != len(logged_predictions):
                    skipped_events += 1
                    continue
            elif external_df is not None:
                n_rows = len(logged_predictions)
                if external_offset + n_rows > len(external_df):
                    skipped_events += 1
                    continue
                chunk = external_df.iloc[external_offset : external_offset + n_rows]
                external_offset += n_rows
                records_to_score = [
                    {str(k): v for k, v in r.items()} for r in chunk.to_dict(orient="records")
                ]
            else:
                skipped_events += 1
                continue

            event_id = str(event_dict.get("event_id", f"event_{line_num}"))
            event_has_valid_tx = False

            for idx, (pred_dict, tx_record) in enumerate(
                zip(logged_predictions, records_to_score, strict=True)
            ):
                if not isinstance(pred_dict, dict):
                    continue

                raw_prob = pred_dict.get("fraud_probability")
                if raw_prob is None or isinstance(raw_prob, bool):
                    continue

                try:
                    logged_prob = float(raw_prob)
                    if not math.isfinite(logged_prob) or not (0.0 <= logged_prob <= 1.0):
                        continue
                except (TypeError, ValueError):
                    continue

                logged_dec = pred_dict.get("decision")
                if logged_dec is None:
                    is_f = bool(pred_dict.get("is_fraud", False))
                    logged_dec = DecisionAction.DENY.value if is_f else DecisionAction.ALLOW.value
                else:
                    logged_dec = str(logged_dec)

                try:
                    result = executor.execute_sync(tx_record)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Pipeline replay execution failed for event %s: %s",
                        event_id,
                        exc,
                    )
                    continue

                event_has_valid_tx = True
                total_transactions += 1
                replayed_prob = result.fraud_probability
                replayed_dec = result.decision

                latencies.append(result.total_latency_ms)
                if result.degraded:
                    degraded_executions += 1

                for s_name, s_res in result.stage_results.items():
                    if s_name in stage_latencies_collector:
                        stage_latencies_collector[s_name].append(s_res.latency_ms)

                diff = abs(replayed_prob - logged_prob)
                if diff > max_abs_diff:
                    max_abs_diff = diff
                sum_abs_diff += diff

                is_score_disc = diff > tolerance
                if is_score_disc:
                    score_discrepancies += 1

                flipped = replayed_dec != logged_dec
                if flipped:
                    decision_flips += 1

                b_actions_list.append(logged_dec)
                r_actions_list.append(replayed_dec)

                can_rec = (is_score_disc or flipped) and len(
                    discrepancies_list
                ) < max_discrepancies_to_record
                if can_rec:
                    discrepancies_list.append(
                        PipelineDiscrepancyDetail(
                            event_id=event_id,
                            transaction_index=idx,
                            logged_probability=logged_prob,
                            replayed_probability=replayed_prob,
                            score_difference=round(diff, 6),
                            logged_decision=logged_dec,
                            replayed_decision=replayed_dec,
                            decision_flipped=flipped,
                            replayed_rule=result.matched_rule,
                            degraded=result.degraded,
                        )
                    )

            if event_has_valid_tx:
                replayed_events += 1
            else:
                skipped_events += 1

    mean_abs_diff = (sum_abs_diff / total_transactions) if total_transactions > 0 else 0.0
    flip_rate = (decision_flips / total_transactions) if total_transactions > 0 else 0.0
    mean_lat = statistics.fmean(latencies) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

    stage_summary: dict[str, dict[str, float]] = {}
    for s_name, s_list in stage_latencies_collector.items():
        if s_list:
            stage_summary[s_name] = {
                "mean_ms": round(statistics.fmean(s_list), 3),
                "p95_ms": round(float(np.percentile(s_list, 95)), 3),
            }
        else:
            stage_summary[s_name] = {"mean_ms": 0.0, "p95_ms": 0.0}

    # Construct 3x3 PolicyTransitionMatrix
    def _to_action_code(a: str) -> str:
        upper = a.upper()
        if "DENY" in upper:
            return DecisionAction.DENY.value
        if "CHALLENGE" in upper or "REVIEW" in upper:
            return DecisionAction.CHALLENGE.value
        return DecisionAction.ALLOW.value

    norm_b = [_to_action_code(a) for a in b_actions_list]
    norm_r = [_to_action_code(a) for a in r_actions_list]

    b_arr = np.array(norm_b, dtype=object)
    r_arr = np.array(norm_r, dtype=object)

    b_allow = b_arr == DecisionAction.ALLOW.value
    b_review = b_arr == DecisionAction.CHALLENGE.value
    b_deny = b_arr == DecisionAction.DENY.value

    r_allow = r_arr == DecisionAction.ALLOW.value
    r_review = r_arr == DecisionAction.CHALLENGE.value
    r_deny = r_arr == DecisionAction.DENY.value

    turnover = int(np.sum(b_arr != r_arr)) if len(b_arr) > 0 else 0
    turnover_rate = float(turnover / total_transactions) if total_transactions > 0 else 0.0

    transition_matrix = PolicyTransitionMatrix(
        allow_to_allow=int(np.sum(b_allow & r_allow)),
        allow_to_challenge=int(np.sum(b_allow & r_review)),
        allow_to_deny=int(np.sum(b_allow & r_deny)),
        challenge_to_allow=int(np.sum(b_review & r_allow)),
        challenge_to_challenge=int(np.sum(b_review & r_review)),
        challenge_to_deny=int(np.sum(b_review & r_deny)),
        deny_to_allow=int(np.sum(b_deny & r_allow)),
        deny_to_challenge=int(np.sum(b_deny & r_review)),
        deny_to_deny=int(np.sum(b_deny & r_deny)),
        total_records=total_transactions,
        turnover_count=turnover,
        turnover_rate=turnover_rate,
    )

    baseline_counts = {
        "ALLOW": int(np.sum(b_allow)),
        "CHALLENGE": int(np.sum(b_review)),
        "DENY": int(np.sum(b_deny)),
    }
    replayed_counts = {
        "ALLOW": int(np.sum(r_allow)),
        "CHALLENGE": int(np.sum(r_review)),
        "DENY": int(np.sum(r_deny)),
    }

    if replayed_events == 0 or total_transactions == 0:
        status = "EMPTY"
    elif score_discrepancies > 0 or decision_flips > 0:
        status = "DIVERGENT"
    else:
        status = "MATCH"

    return PipelineReplayReport(
        total_events=total_events,
        scoring_events=scoring_events,
        replayed_events=replayed_events,
        skipped_events=skipped_events,
        total_transactions=total_transactions,
        score_discrepancies=score_discrepancies,
        decision_flips=decision_flips,
        flip_rate=round(flip_rate, 4),
        max_absolute_difference=round(max_abs_diff, 6),
        mean_absolute_difference=round(mean_abs_diff, 6),
        tolerance=tolerance,
        status=status,
        transition_matrix=transition_matrix,
        baseline_decision_counts=baseline_counts,
        replayed_decision_counts=replayed_counts,
        mean_latency_ms=round(mean_lat, 3),
        p95_latency_ms=round(p95_lat, 3),
        degraded_executions=degraded_executions,
        stage_latencies=stage_summary,
        discrepancies=tuple(discrepancies_list),
    )


@dataclass(frozen=True)
class PipelineCounterfactualEvaluation:
    """Counterfactual policy evaluation metrics and cost trade-offs for candidate graph settings."""

    rows: int
    baseline_thresholds: TieredThresholds
    candidate_thresholds: TieredThresholds
    transition_matrix: PolicyTransitionMatrix
    baseline_decision_counts: dict[str, int]
    candidate_decision_counts: dict[str, int]
    review_count_delta: int
    review_rate_delta: float
    deny_count_delta: int
    deny_rate_delta: float
    turnover_count: int
    turnover_rate: float
    baseline_metrics: TieredMetrics | None = None
    candidate_metrics: TieredMetrics | None = None
    cost_delta: float | None = None
    fraud_catch_delta: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert counterfactual evaluation result to a serializable dictionary."""
        return {
            "rows": self.rows,
            "baseline_thresholds": self.baseline_thresholds.to_dict(),
            "candidate_thresholds": self.candidate_thresholds.to_dict(),
            "transition_matrix": self.transition_matrix.to_dict(),
            "baseline_decision_counts": dict(self.baseline_decision_counts),
            "candidate_decision_counts": dict(self.candidate_decision_counts),
            "review_count_delta": self.review_count_delta,
            "review_rate_delta": self.review_rate_delta,
            "deny_count_delta": self.deny_count_delta,
            "deny_rate_delta": self.deny_rate_delta,
            "turnover_count": self.turnover_count,
            "turnover_rate": self.turnover_rate,
            "baseline_metrics": self.baseline_metrics.to_dict() if self.baseline_metrics else None,
            "candidate_metrics": (
                self.candidate_metrics.to_dict() if self.candidate_metrics else None
            ),
            "cost_delta": self.cost_delta,
            "fraud_catch_delta": self.fraud_catch_delta,
        }


def optimize_counterfactual_pipeline(
    probabilities: np.ndarray | Sequence[float],
    candidate_thresholds: TieredThresholds,
    *,
    baseline_thresholds: TieredThresholds | None = None,
    baseline_decisions: Sequence[str] | None = None,
    y_true: np.ndarray | Sequence[int] | None = None,
    manual_review_cost: float = 5.0,
    false_deny_cost: float = 50.0,
    missed_fraud_cost: float = 200.0,
) -> PipelineCounterfactualEvaluation:
    """Evaluate counterfactual policy shifts, decision turnover, and workload/cost impact.

    Parameters
    ----------
    probabilities:
        Scored transaction probabilities (from baseline model or replayed graph).
    candidate_thresholds:
        Target candidate tiered thresholds to evaluate.
    baseline_thresholds:
        Baseline tiered thresholds. Defaults to (0.5, 0.5) if not provided.
    baseline_decisions:
        Optional historical decision labels (e.g. from audit log). If omitted,
        inferred from baseline_thresholds.
    y_true:
        Optional ground truth binary labels matching the transactions.
    manual_review_cost:
        Unit friction cost for transactions routed to CHALLENGE.
    false_deny_cost:
        Unit loss cost for legitimate transactions routed to DENY.
    missed_fraud_cost:
        Unit loss cost for fraudulent transactions routed to ALLOW.
    """
    if not isinstance(candidate_thresholds, TieredThresholds):
        raise TypeError("candidate_thresholds must be a TieredThresholds instance")
    if baseline_thresholds is not None and not isinstance(baseline_thresholds, TieredThresholds):
        raise TypeError("baseline_thresholds must be a TieredThresholds instance")

    for cost_name, cost_val in [
        ("manual_review_cost", manual_review_cost),
        ("false_deny_cost", false_deny_cost),
        ("missed_fraud_cost", missed_fraud_cost),
    ]:
        if (
            isinstance(cost_val, bool)
            or not isinstance(cost_val, (int, float))
            or not math.isfinite(cost_val)
            or cost_val < 0.0
        ):
            raise ValueError(f"{cost_name} must be a non-negative finite number")

    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array")
    if not np.isfinite(probs).all() or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must be finite numbers between 0 and 1")

    effective_baseline = baseline_thresholds or TieredThresholds(0.5, 0.5)
    n_rows = len(probs)

    def _normalize_action(a: str) -> str:
        upper = a.upper()
        if "DENY" in upper:
            return DecisionAction.DENY.value
        if "CHALLENGE" in upper or "REVIEW" in upper:
            return DecisionAction.CHALLENGE.value
        return DecisionAction.ALLOW.value

    if baseline_decisions is not None:
        if len(baseline_decisions) != n_rows:
            raise ValueError("Length of baseline_decisions must match probabilities")
        b_actions = np.array([_normalize_action(str(d)) for d in baseline_decisions], dtype=object)
    else:
        b_actions = assign_tiered_decisions(
            probs,
            review_threshold=effective_baseline.review_threshold,
            deny_threshold=effective_baseline.deny_threshold,
        )

    c_actions = assign_tiered_decisions(
        probs,
        review_threshold=candidate_thresholds.review_threshold,
        deny_threshold=candidate_thresholds.deny_threshold,
    )

    b_allow = b_actions == DecisionAction.ALLOW.value
    b_review = b_actions == DecisionAction.CHALLENGE.value
    b_deny = b_actions == DecisionAction.DENY.value

    c_allow = c_actions == DecisionAction.ALLOW.value
    c_review = c_actions == DecisionAction.CHALLENGE.value
    c_deny = c_actions == DecisionAction.DENY.value

    turnover = int(np.sum(b_actions != c_actions))
    turnover_rate = float(turnover / n_rows) if n_rows > 0 else 0.0

    matrix = PolicyTransitionMatrix(
        allow_to_allow=int(np.sum(b_allow & c_allow)),
        allow_to_challenge=int(np.sum(b_allow & c_review)),
        allow_to_deny=int(np.sum(b_allow & c_deny)),
        challenge_to_allow=int(np.sum(b_review & c_allow)),
        challenge_to_challenge=int(np.sum(b_review & c_review)),
        challenge_to_deny=int(np.sum(b_review & c_deny)),
        deny_to_allow=int(np.sum(b_deny & c_allow)),
        deny_to_challenge=int(np.sum(b_deny & c_review)),
        deny_to_deny=int(np.sum(b_deny & c_deny)),
        total_records=n_rows,
        turnover_count=turnover,
        turnover_rate=turnover_rate,
    )

    b_counts = {
        DecisionAction.ALLOW.value: int(np.sum(b_allow)),
        DecisionAction.CHALLENGE.value: int(np.sum(b_review)),
        DecisionAction.DENY.value: int(np.sum(b_deny)),
    }
    c_counts = {
        DecisionAction.ALLOW.value: int(np.sum(c_allow)),
        DecisionAction.CHALLENGE.value: int(np.sum(c_review)),
        DecisionAction.DENY.value: int(np.sum(c_deny)),
    }

    review_delta = (
        c_counts[DecisionAction.CHALLENGE.value] - b_counts[DecisionAction.CHALLENGE.value]
    )
    review_rate_delta = float(review_delta / n_rows) if n_rows > 0 else 0.0
    deny_delta = c_counts[DecisionAction.DENY.value] - b_counts[DecisionAction.DENY.value]
    deny_rate_delta = float(deny_delta / n_rows) if n_rows > 0 else 0.0

    b_metrics: TieredMetrics | None = None
    c_metrics: TieredMetrics | None = None
    cost_delta: float | None = None
    fraud_catch_delta: int | None = None

    if y_true is not None:
        y_arr = np.asarray(y_true, dtype=int)
        if len(y_arr) != n_rows:
            raise ValueError(
                f"Length of y_true ({len(y_arr)}) does not match probabilities ({n_rows})"
            )
        b_metrics = evaluate_tiered_policy(
            y_arr,
            probs,
            review_threshold=effective_baseline.review_threshold,
            deny_threshold=effective_baseline.deny_threshold,
            manual_review_cost=manual_review_cost,
            false_deny_cost=false_deny_cost,
            missed_fraud_cost=missed_fraud_cost,
        )
        c_metrics = evaluate_tiered_policy(
            y_arr,
            probs,
            review_threshold=candidate_thresholds.review_threshold,
            deny_threshold=candidate_thresholds.deny_threshold,
            manual_review_cost=manual_review_cost,
            false_deny_cost=false_deny_cost,
            missed_fraud_cost=missed_fraud_cost,
        )
        cost_delta = float(
            c_metrics.expected_cost_per_transaction - b_metrics.expected_cost_per_transaction
        )
        b_caught = b_metrics.caught_fraud_review + b_metrics.caught_fraud_deny
        c_caught = c_metrics.caught_fraud_review + c_metrics.caught_fraud_deny
        fraud_catch_delta = int(c_caught - b_caught)

    return PipelineCounterfactualEvaluation(
        rows=n_rows,
        baseline_thresholds=effective_baseline,
        candidate_thresholds=candidate_thresholds,
        transition_matrix=matrix,
        baseline_decision_counts=b_counts,
        candidate_decision_counts=c_counts,
        review_count_delta=review_delta,
        review_rate_delta=round(review_rate_delta, 4),
        deny_count_delta=deny_delta,
        deny_rate_delta=round(deny_rate_delta, 4),
        turnover_count=turnover,
        turnover_rate=round(turnover_rate, 4),
        baseline_metrics=b_metrics,
        candidate_metrics=c_metrics,
        cost_delta=round(cost_delta, 4) if cost_delta is not None else None,
        fraud_catch_delta=fraud_catch_delta,
    )


# ============================================================================
# Deterministic Stage Chaos Fault Injection & Resilience Simulation
# ============================================================================


class StageFaultType(StrEnum):
    """Supported fault injection modes for stage resilience simulation."""

    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    LATENCY_SPIKE = "latency_spike"


@dataclass(frozen=True)
class StageFaultSpec:
    """Deterministic fault injection specification targeting a stage node."""

    node_name: str
    fault_type: StageFaultType | str
    start_index: int = 0
    end_index: int | None = None
    latency_ms: float = 0.0
    error_message: str = "Injected stage chaos fault"

    def __post_init__(self) -> None:
        if not self.node_name or not isinstance(self.node_name, str):
            raise ValueError("node_name must be a non-empty string")
        if not isinstance(self.fault_type, StageFaultType):
            try:
                object.__setattr__(self, "fault_type", StageFaultType(self.fault_type))
            except (ValueError, TypeError):
                raise ValueError(
                    f"Unsupported fault_type '{self.fault_type}'. "
                    f"Must be one of {[t.value for t in StageFaultType]}"
                ) from None
        if isinstance(self.start_index, bool) or self.start_index < 0:
            raise ValueError("start_index must be an integer >= 0")
        if self.end_index is not None and (
            isinstance(self.end_index, bool) or self.end_index < self.start_index
        ):
            raise ValueError("end_index must be an integer >= start_index")
        if (
            isinstance(self.latency_ms, bool)
            or not math.isfinite(self.latency_ms)
            or self.latency_ms < 0.0
        ):
            raise ValueError("latency_ms must be a non-negative finite number")

    def to_dict(self) -> dict[str, Any]:
        """Convert fault specification to a serializable dictionary."""
        ft_val = (
            self.fault_type.value
            if isinstance(self.fault_type, StageFaultType)
            else str(self.fault_type)
        )
        return {
            "node_name": self.node_name,
            "fault_type": ft_val,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "latency_ms": round(self.latency_ms, 3),
            "error_message": self.error_message,
        }


@dataclass
class StageChaosMetrics:
    """Per-stage fault injection counters and circuit trip status."""

    node_name: str
    fault_type: str
    injected_count: int = 0
    handled_count: int = 0
    tripped_circuit: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert stage chaos metrics to a serializable dictionary."""
        return {
            "node_name": self.node_name,
            "fault_type": self.fault_type,
            "injected_count": self.injected_count,
            "handled_count": self.handled_count,
            "tripped_circuit": self.tripped_circuit,
        }


@dataclass
class PipelineChaosReport:
    """Comprehensive telemetry report emitted by simulate_pipeline_chaos."""

    total_transactions: int
    fault_specs: list[StageFaultSpec]
    circuit_breaker_enabled: bool
    sla_budget_ms: float
    sla_breaches: int
    circuit_trips: int
    short_circuited_calls: int
    unhandled_errors: int
    degraded_transactions: int
    latencies_p50_ms: float
    latencies_p95_ms: float
    latencies_max_ms: float
    stage_metrics: dict[str, StageChaosMetrics]
    recovered: bool
    status: str

    def to_dict(self) -> dict[str, Any]:
        """Convert chaos report to a serializable dictionary."""
        return {
            "total_transactions": self.total_transactions,
            "fault_specs": [s.to_dict() for s in self.fault_specs],
            "circuit_breaker_enabled": self.circuit_breaker_enabled,
            "sla_budget_ms": round(self.sla_budget_ms, 3),
            "sla_breaches": self.sla_breaches,
            "circuit_trips": self.circuit_trips,
            "short_circuited_calls": self.short_circuited_calls,
            "unhandled_errors": self.unhandled_errors,
            "degraded_transactions": self.degraded_transactions,
            "latencies_p50_ms": round(self.latencies_p50_ms, 3),
            "latencies_p95_ms": round(self.latencies_p95_ms, 3),
            "latencies_max_ms": round(self.latencies_max_ms, 3),
            "stage_metrics": {k: v.to_dict() for k, v in self.stage_metrics.items()},
            "recovered": self.recovered,
            "status": self.status,
        }


def simulate_pipeline_chaos(
    records: Sequence[dict[str, Any]],
    pipeline: DecisionGraph | DecisionGraphExecutor,
    fault_specs: Sequence[StageFaultSpec | dict[str, Any]],
    *,
    sla_budget_ms: float = 100.0,
    enable_circuit_breaker: bool = True,
    circuit_failure_threshold: int = 5,
    circuit_recovery_timeout_seconds: float = 30.0,
    global_timeout_seconds: float | None = None,
) -> PipelineChaosReport:
    """Execute transactions through a decision graph under deterministic chaos fault injection.

    Injects stage-level exceptions, timeouts, and latency spikes across specified transaction
    index ranges, evaluating stage circuit breaker tripping, zero-latency fast-fail isolation,
    SLA budget compliance, and post-fault circuit recovery.
    """
    if isinstance(sla_budget_ms, bool) or not math.isfinite(sla_budget_ms) or sla_budget_ms <= 0.0:
        raise ValueError("sla_budget_ms must be a positive finite number")

    parsed_specs: list[StageFaultSpec] = []
    for s in fault_specs:
        if isinstance(s, StageFaultSpec):
            parsed_specs.append(s)
        elif isinstance(s, dict):
            parsed_specs.append(
                StageFaultSpec(
                    node_name=str(s["node_name"]),
                    fault_type=StageFaultType(s["fault_type"]),
                    start_index=int(s.get("start_index", 0)),
                    end_index=int(s["end_index"]) if s.get("end_index") is not None else None,
                    latency_ms=float(s.get("latency_ms", 0.0)),
                    error_message=str(s.get("error_message", "Injected stage chaos fault")),
                )
            )
        else:
            raise TypeError(f"fault_spec must be StageFaultSpec or dict, got {type(s)}")

    if isinstance(pipeline, DecisionGraphExecutor):
        graph = pipeline.graph
    elif isinstance(pipeline, DecisionGraph):
        graph = pipeline
    else:
        raise TypeError(
            f"pipeline must be DecisionGraph or DecisionGraphExecutor, got {type(pipeline)}"
        )

    for spec in parsed_specs:
        if spec.node_name not in graph.nodes:
            raise ValueError(f"Target node '{spec.node_name}' does not exist in decision graph")

    registry = StageCircuitRegistry()
    if enable_circuit_breaker:
        for name, node in graph.nodes.items():
            if node.circuit_policy is not None:
                pol = StageCircuitPolicy(
                    enabled=node.circuit_policy.enabled,
                    failure_threshold=circuit_failure_threshold,
                    recovery_timeout_seconds=circuit_recovery_timeout_seconds,
                    half_open_max_calls=node.circuit_policy.half_open_max_calls,
                    count_timeouts=node.circuit_policy.count_timeouts,
                    count_exceptions=node.circuit_policy.count_exceptions,
                )
            else:
                pol = StageCircuitPolicy(
                    enabled=True,
                    failure_threshold=circuit_failure_threshold,
                    recovery_timeout_seconds=circuit_recovery_timeout_seconds,
                )
            registry.register_policy(name, pol)

    executor = DecisionGraphExecutor(
        graph,
        circuit_registry=registry,
        enable_circuit_breaker=enable_circuit_breaker,
        enable_cache=(
            getattr(pipeline, "enable_cache", False)
            if isinstance(pipeline, DecisionGraphExecutor)
            else False
        ),
    )

    current_tx_index = 0
    stage_chaos_metrics: dict[str, StageChaosMetrics] = {
        spec.node_name: StageChaosMetrics(
            node_name=spec.node_name,
            fault_type=(
                spec.fault_type.value
                if isinstance(spec.fault_type, StageFaultType)
                else str(spec.fault_type)
            ),
        )
        for spec in parsed_specs
    }

    original_processors = {name: node.processor for name, node in graph.nodes.items()}

    def make_chaos_processor(node_name: str, orig_proc: Any) -> Any:
        node_specs = [s for s in parsed_specs if s.node_name == node_name]
        if not node_specs:
            return orig_proc

        if inspect.iscoroutinefunction(orig_proc):

            async def async_chaos_wrapper(ctx: PipelineContext) -> Any:
                for spec in node_specs:
                    is_active = spec.start_index <= current_tx_index and (
                        spec.end_index is None or current_tx_index < spec.end_index
                    )
                    if is_active:
                        m = stage_chaos_metrics[node_name]
                        m.injected_count += 1
                        ft = (
                            spec.fault_type
                            if isinstance(spec.fault_type, StageFaultType)
                            else StageFaultType(spec.fault_type)
                        )
                        if ft == StageFaultType.EXCEPTION:
                            raise RuntimeError(spec.error_message)
                        elif ft == StageFaultType.TIMEOUT:
                            await asyncio.sleep(0.005)
                            raise TimeoutError(spec.error_message)
                        elif ft == StageFaultType.LATENCY_SPIKE and spec.latency_ms > 0:
                            await asyncio.sleep(spec.latency_ms / 1000.0)
                return await orig_proc(ctx)

            return async_chaos_wrapper

        def sync_chaos_wrapper(ctx: PipelineContext) -> Any:
            for spec in node_specs:
                is_active = spec.start_index <= current_tx_index and (
                    spec.end_index is None or current_tx_index < spec.end_index
                )
                if is_active:
                    m = stage_chaos_metrics[node_name]
                    m.injected_count += 1
                    ft = (
                        spec.fault_type
                        if isinstance(spec.fault_type, StageFaultType)
                        else StageFaultType(spec.fault_type)
                    )
                    if ft == StageFaultType.EXCEPTION:
                        raise RuntimeError(spec.error_message)
                    elif ft == StageFaultType.TIMEOUT:
                        sleep(0.005)
                        raise TimeoutError(spec.error_message)
                    elif ft == StageFaultType.LATENCY_SPIKE and spec.latency_ms > 0:
                        sleep(spec.latency_ms / 1000.0)
            return orig_proc(ctx)

        return sync_chaos_wrapper

    for name, orig_proc in original_processors.items():
        graph.nodes[name].processor = make_chaos_processor(name, orig_proc)

    latencies: list[float] = []
    sla_breaches = 0
    degraded_transactions = 0
    unhandled_errors = 0

    try:
        for idx, record in enumerate(records):
            current_tx_index = idx
            try:
                res = executor.execute_sync(
                    record,
                    global_timeout_seconds=global_timeout_seconds,
                )
                latencies.append(res.total_latency_ms)
                if res.total_latency_ms > sla_budget_ms:
                    sla_breaches += 1
                if res.degraded:
                    degraded_transactions += 1
                for node_name in res.stage_results:
                    if node_name in stage_chaos_metrics and res.stage_results[node_name].degraded:
                        stage_chaos_metrics[node_name].handled_count += 1
            except (
                PipelineExecutionError,
                RuntimeError,
                TimeoutError,
                ValueError,
                KeyError,
            ) as exc:
                logger.warning("Unhandled error during chaos execution on tx %d: %s", idx, exc)
                unhandled_errors += 1
                latencies.append(sla_budget_ms * 2.0)
                sla_breaches += 1
    finally:
        for name, orig_proc in original_processors.items():
            graph.nodes[name].processor = orig_proc

    cb_stats = registry.stats_dict()
    total_trips = int(cb_stats["total_trips"])
    short_circuited_calls = int(cb_stats["total_short_circuits"])

    for node_name, m in stage_chaos_metrics.items():
        node_st = cb_stats["nodes"].get(node_name)
        if node_st and int(node_st.get("total_trips", 0)) > 0:
            m.tripped_circuit = True

    all_closed = len(cb_stats["open_nodes"]) == 0 and len(cb_stats["half_open_nodes"]) == 0

    if latencies:
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        max_lat = float(np.max(latencies))
    else:
        p50 = 0.0
        p95 = 0.0
        max_lat = 0.0

    if unhandled_errors > 0:
        overall_status = "FAILED"
    elif sla_breaches > 0:
        overall_status = "SLA_BREACHED"
    elif degraded_transactions > 0 or not all_closed:
        overall_status = "DEGRADED"
    else:
        overall_status = "RESILIENT"

    return PipelineChaosReport(
        total_transactions=len(records),
        fault_specs=parsed_specs,
        circuit_breaker_enabled=enable_circuit_breaker,
        sla_budget_ms=sla_budget_ms,
        sla_breaches=sla_breaches,
        circuit_trips=total_trips,
        short_circuited_calls=short_circuited_calls,
        unhandled_errors=unhandled_errors,
        degraded_transactions=degraded_transactions,
        latencies_p50_ms=p50,
        latencies_p95_ms=p95,
        latencies_max_ms=max_lat,
        stage_metrics=stage_chaos_metrics,
        recovered=all_closed,
        status=overall_status,
    )
