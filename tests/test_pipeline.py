"""Unit and integration tests for decision graph execution pipeline."""

from __future__ import annotations

import pytest

from fraud_detection.pipeline import (
    DecisionGraph,
    PipelineContext,
    PipelineCycleError,
    PipelineNode,
    PipelineValidationError,
    StageExecutionResult,
    StageStatus,
    StageType,
)


def dummy_processor(_context: PipelineContext) -> dict[str, str]:
    return {"status": "ok"}


def test_stage_execution_result_serialization() -> None:
    res = StageExecutionResult(
        node_name="enrichment",
        stage_type=StageType.ENRICHMENT,
        status=StageStatus.SUCCESS,
        latency_ms=1.2346,
        output={"v1": 1.0},
        error=None,
        degraded=False,
    )
    d = res.to_dict()
    assert d["node_name"] == "enrichment"
    assert d["stage_type"] == "enrichment"
    assert d["status"] == "success"
    assert d["latency_ms"] == 1.235
    assert d["output"] == {"v1": 1.0}
    assert d["degraded"] is False


def test_pipeline_context_helpers() -> None:
    ctx = PipelineContext(transaction={"Amount": 100.0})
    assert ctx.transaction["Amount"] == 100.0
    assert ctx.get_output("missing", default="default_val") == "default_val"

    ctx.results["stage1"] = StageExecutionResult(
        node_name="stage1",
        stage_type=StageType.ENRICHMENT,
        status=StageStatus.SUCCESS,
        output={"score": 0.8},
    )
    assert ctx.get_output("stage1") == {"score": 0.8}

    ctx.set_output("stage1", {"score": 0.9})
    assert ctx.get_output("stage1") == {"score": 0.9}


def test_decision_graph_construction_and_validation() -> None:
    graph = DecisionGraph("test_graph")
    n1 = PipelineNode(name="enrichment", stage_type=StageType.ENRICHMENT, processor=dummy_processor)
    n2 = PipelineNode(
        name="rules",
        stage_type=StageType.RULES,
        processor=dummy_processor,
        dependencies=["enrichment"],
    )
    n3 = PipelineNode(
        name="inference",
        stage_type=StageType.INFERENCE,
        processor=dummy_processor,
        dependencies=["enrichment"],
    )
    n4 = PipelineNode(
        name="action",
        stage_type=StageType.ACTION,
        processor=dummy_processor,
        dependencies=["rules", "inference"],
    )

    graph.add_node(n1).add_node(n2).add_node(n3).add_node(n4)
    graph.validate()

    order = graph.topological_sort()
    assert order.index("enrichment") < order.index("rules")
    assert order.index("enrichment") < order.index("inference")
    assert order.index("rules") < order.index("action")
    assert order.index("inference") < order.index("action")

    plan = graph.build_execution_plan()
    assert len(plan.waves) == 3
    assert plan.waves[0] == ("enrichment",)
    assert set(plan.waves[1]) == {"rules", "inference"}
    assert plan.waves[2] == ("action",)

    graph_dict = graph.to_dict()
    assert graph_dict["name"] == "test_graph"
    assert graph_dict["node_count"] == 4
    assert len(graph_dict["edges"]) == 4


def test_decision_graph_duplicate_node() -> None:
    graph = DecisionGraph()
    n1 = PipelineNode(name="node_a", stage_type=StageType.CUSTOM, processor=dummy_processor)
    graph.add_node(n1)
    with pytest.raises(PipelineValidationError, match="Duplicate node name"):
        graph.add_node(n1)


def test_decision_graph_missing_dependency() -> None:
    graph = DecisionGraph()
    n1 = PipelineNode(
        name="node_b",
        stage_type=StageType.CUSTOM,
        processor=dummy_processor,
        dependencies=["non_existent"],
    )
    graph.add_node(n1)
    with pytest.raises(PipelineValidationError, match="references unknown dependency"):
        graph.validate()


def test_decision_graph_self_dependency_cycle() -> None:
    graph = DecisionGraph()
    n1 = PipelineNode(
        name="self_loop",
        stage_type=StageType.CUSTOM,
        processor=dummy_processor,
        dependencies=["self_loop"],
    )
    graph.add_node(n1)
    with pytest.raises(PipelineCycleError, match="depends on itself"):
        graph.validate()


def test_decision_graph_multi_node_cycle() -> None:
    graph = DecisionGraph()
    n1 = PipelineNode(
        name="node_1",
        stage_type=StageType.CUSTOM,
        processor=dummy_processor,
        dependencies=["node_3"],
    )
    n2 = PipelineNode(
        name="node_2",
        stage_type=StageType.CUSTOM,
        processor=dummy_processor,
        dependencies=["node_1"],
    )
    n3 = PipelineNode(
        name="node_3",
        stage_type=StageType.CUSTOM,
        processor=dummy_processor,
        dependencies=["node_2"],
    )

    graph.add_node(n1).add_node(n2).add_node(n3)
    with pytest.raises(PipelineCycleError, match="Cycle detected"):
        graph.validate()


def test_decision_graph_add_edge() -> None:
    graph = DecisionGraph()
    n1 = PipelineNode(name="first", stage_type=StageType.CUSTOM, processor=dummy_processor)
    n2 = PipelineNode(name="second", stage_type=StageType.CUSTOM, processor=dummy_processor)
    graph.add_node(n1).add_node(n2)
    graph.add_edge("first", "second")

    assert "first" in graph.nodes["second"].dependencies
    order = graph.topological_sort()
    assert order == ["first", "second"]


def test_decision_graph_add_edge_unknown_target() -> None:
    graph = DecisionGraph()
    n1 = PipelineNode(name="first", stage_type=StageType.CUSTOM, processor=dummy_processor)
    graph.add_node(n1)
    with pytest.raises(PipelineValidationError, match="Target node 'unknown' does not exist"):
        graph.add_edge("first", "unknown")
