"""Real-time streaming decision graph and adaptive execution pipeline."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import pandas as pd

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
        # 0 = unvisited, 1 = visiting (in stack), 2 = visited
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
        }
