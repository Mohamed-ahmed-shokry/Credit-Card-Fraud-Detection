"""Real-time streaming decision graph and adaptive execution pipeline."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

import numpy as np
import pandas as pd

from fraud_detection.evaluation import DecisionAction
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
        model: FraudModel,
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

    def __init__(self, graph: DecisionGraph) -> None:
        self.graph = graph
        self.graph.validate()
        self.plan = self.graph.build_execution_plan()

    async def _execute_node(self, node_name: str, context: PipelineContext) -> None:
        node = self.graph.nodes[node_name]

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
            context.results[node_name] = StageExecutionResult(
                node_name=node.name,
                stage_type=node.stage_type,
                status=StageStatus.SUCCESS,
                latency_ms=elapsed,
                output=output,
                degraded=False,
            )

        except TimeoutError:
            elapsed = (perf_counter() - start) * 1000.0
            logger.warning(
                "Pipeline stage '%s' timed out after %.2f ms (budget %.2f s)",
                node_name,
                elapsed,
                node.timeout_seconds or 0.0,
            )
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
            if node.required:
                raise PipelineExecutionError(
                    f"Required stage '{node_name}' failed: {exc}"
                ) from exc
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

        async def _run_waves() -> None:
            for wave in self.plan.waves:
                tasks = [self._execute_node(node_name, context) for node_name in wave]
                await asyncio.gather(*tasks)

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

        degraded_nodes = [
            node_name for node_name, res in context.results.items() if res.degraded
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
    enrichment_timeout_seconds: float | None = 0.05,
    inference_timeout_seconds: float | None = 0.05,
    calibration_timeout_seconds: float | None = 0.02,
    rules_timeout_seconds: float | None = 0.02,
    action_timeout_seconds: float | None = 0.02,
    explain: bool = False,
) -> DecisionGraph:
    """Build a standard, production-ready fraud decision execution graph."""
    applied_threshold = model.threshold if threshold is None else threshold
    graph = DecisionGraph("production_fraud_pipeline")

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
