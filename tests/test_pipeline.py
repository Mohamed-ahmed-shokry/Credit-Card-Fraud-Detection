"""Unit and integration tests for decision graph execution pipeline."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from fraud_detection.data import generate_synthetic_data, validate_frame
from fraud_detection.features import (
    FeatureDefinition,
    FeatureSnapshot,
    FeatureType,
    FeatureView,
    InMemoryFeatureStore,
)
from fraud_detection.model import train_model
from fraud_detection.pipeline import (
    ActionAggregatorStageProcessor,
    CalibrationStageProcessor,
    DecisionGraph,
    DecisionGraphExecutor,
    EnrichmentStageProcessor,
    InferenceStageProcessor,
    PipelineContext,
    PipelineCycleError,
    PipelineExecutionError,
    PipelineNode,
    PipelineValidationError,
    RuleStageProcessor,
    StageExecutionResult,
    StageStatus,
    StageType,
    create_default_fraud_pipeline,
)
from fraud_detection.recalibration import PlattRecalibrator
from fraud_detection.rules import (
    DecisionRule,
    RuleAction,
    RuleCondition,
    RuleOperator,
    RulePrecedence,
    RuleSet,
)
from fraud_detection.velocity import VelocityConfig, VelocityWindow, VelocityWindowBuffer


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


def test_enrichment_stage_processor() -> None:
    # Set up feature store
    store = InMemoryFeatureStore()
    view = FeatureView(
        name="card_risk",
        entity_key="card_id",
        features=[FeatureDefinition(name="card_risk_score", feature_type=FeatureType.FLOAT)],
        ttl_seconds=3600.0,
    )
    store.add_view(view)
    store.put_snapshot(
        "card_risk",
        FeatureSnapshot(
            entity_id="card_999",
            timestamp=100.0,
            values={"card_risk_score": 0.75},
        ),
    )

    # Set up velocity buffer
    v_buf = VelocityWindowBuffer()
    v_conf = VelocityConfig(
        entity_key="card_id",
        windows=(VelocityWindow(duration_seconds=3600.0, name="1h"),),
    )

    processor = EnrichmentStageProcessor(
        feature_store=store,
        velocity_buffer=v_buf,
        velocity_config=v_conf,
    )

    ctx = PipelineContext(
        transaction={"card_id": "card_999", "Amount": 150.0, "Time": 150.0}
    )
    result = processor(ctx)

    assert "card_risk_score" in result["enriched_features"]
    assert "card_risk" in result["feature_views_applied"]
    assert ctx.transaction["card_risk_score"] == 0.75
    assert "velocity_count_1h" in ctx.transaction
    assert ctx.transaction["velocity_count_1h"] == 0.0

    # Second transaction for the same entity should see count=1.0
    ctx2 = PipelineContext(
        transaction={"card_id": "card_999", "Amount": 200.0, "Time": 200.0}
    )
    processor(ctx2)
    assert ctx2.transaction["velocity_count_1h"] == 1.0


def test_rule_stage_processor_short_circuit() -> None:
    rule = DecisionRule(
        rule_id="R001",
        name="High Amount Deny",
        action=RuleAction.DENY,
        conditions=(
            RuleCondition(field="Amount", operator=RuleOperator.GREATER_THAN, value=5000.0),
        ),
    )
    ruleset = RuleSet(rules=(rule,))
    processor = RuleStageProcessor(rules=ruleset)

    ctx = PipelineContext(transaction={"Amount": 6000.0})
    res = processor(ctx)
    assert res["matched"] is True
    assert res["action"] == "DENY"
    assert res["short_circuited"] is True
    assert ctx.metadata.get("short_circuit") is True


def test_inference_stage_processor_and_calibration() -> None:
    dataset = validate_frame(generate_synthetic_data(rows=300, fraud_rate=0.1, random_state=42))
    model = train_model(dataset)

    # Test normal inference
    inf_proc = InferenceStageProcessor(model=model, explain=True)
    ctx = PipelineContext(transaction={"Amount": 50.0, "V1": 0.5, "V2": -0.2})
    res = inf_proc(ctx)
    assert res["skipped"] is False
    assert 0.0 <= res["probability"] <= 1.0
    assert res["contributions"] is not None

    ctx.results["inference"] = StageExecutionResult(
        node_name="inference",
        stage_type=StageType.INFERENCE,
        status=StageStatus.SUCCESS,
        output=res,
    )

    # Test calibration processor
    recal = PlattRecalibrator(a=1.5, b=-0.2)
    cal_proc = CalibrationStageProcessor(recalibrator=recal)
    cal_res = cal_proc(ctx)
    assert 0.0 <= cal_res["calibrated_probability"] <= 1.0
    assert cal_res["method"] == "sigmoid"

    # Test inference skip on short-circuit
    ctx_skip = PipelineContext(
        transaction={"Amount": 50.0},
        metadata={"short_circuit": True},
    )
    skip_res = inf_proc(ctx_skip)
    assert skip_res["skipped"] is True
    assert skip_res["probability"] is None


def test_action_aggregator_stage_processor() -> None:
    act_proc = ActionAggregatorStageProcessor(
        threshold=0.5,
        review_threshold=0.3,
        deny_threshold=0.7,
        rule_precedence=RulePrecedence.RULES_OVERRIDE_MODEL,
    )

    # Test rule overrides model
    ctx = PipelineContext(
        transaction={},
        results={
            "rules": StageExecutionResult(
                node_name="rules",
                stage_type=StageType.RULES,
                status=StageStatus.SUCCESS,
                output={"matched": True, "matched_rule_id": "R001", "action": "ALLOW"},
            ),
            "calibration": StageExecutionResult(
                node_name="calibration",
                stage_type=StageType.CALIBRATION,
                status=StageStatus.SUCCESS,
                output={"calibrated_probability": 0.85, "raw_probability": 0.85},
            ),
        },
    )
    decision = act_proc(ctx)
    assert decision["decision"] == "ALLOW"
    assert decision["matched_rule"] == "R001"
    assert decision["is_fraud"] is False

    # Test model overrides rules when model action is DENY
    act_proc_model_first = ActionAggregatorStageProcessor(
        threshold=0.5,
        review_threshold=0.3,
        deny_threshold=0.7,
        rule_precedence=RulePrecedence.MODEL_OVERRIDES_RULES,
    )
    decision_model_first = act_proc_model_first(ctx)
    assert decision_model_first["decision"] == "DENY"
    assert decision_model_first["is_fraud"] is True


def test_decision_graph_executor_async_and_sync() -> None:
    graph = DecisionGraph("async_test")
    graph.add_node(
        PipelineNode(
            name="stage1",
            stage_type=StageType.ENRICHMENT,
            processor=lambda _ctx: {"v": 1},
        )
    )
    graph.add_node(
        PipelineNode(
            name="action",
            stage_type=StageType.ACTION,
            processor=lambda _ctx: {"decision": "ALLOW", "is_fraud": False},
            dependencies=["stage1"],
        )
    )

    executor = DecisionGraphExecutor(graph)

    # Async execute via asyncio.run
    res_async = asyncio.run(executor.execute({"Amount": 10.0}))
    assert res_async.success is True
    assert res_async.final_decision["decision"] == "ALLOW"
    assert res_async.degraded is False
    assert res_async.execution_path == ["stage1", "action"]

    # Sync execute
    res_sync = executor.execute_sync({"Amount": 10.0})
    assert res_sync.success is True
    assert res_sync.final_decision["decision"] == "ALLOW"


def test_decision_graph_executor_stage_timeout_fail_soft() -> None:
    def slow_processor(_ctx: PipelineContext) -> dict[str, Any]:
        time.sleep(0.15)
        return {"done": True}

    graph = DecisionGraph("timeout_graph")
    graph.add_node(
        PipelineNode(
            name="slow_node",
            stage_type=StageType.ENRICHMENT,
            processor=slow_processor,
            timeout_seconds=0.03,
            fallback_value={"fallback": True},
            required=False,
        )
    )
    graph.add_node(
        PipelineNode(
            name="action",
            stage_type=StageType.ACTION,
            processor=lambda ctx: {
                "decision": "ALLOW",
                "is_fraud": False,
                "used_fallback": ctx.get_output("slow_node"),
            },
            dependencies=["slow_node"],
        )
    )

    executor = DecisionGraphExecutor(graph)
    res = asyncio.run(executor.execute({"Amount": 10.0}))

    assert res.success is True
    assert res.degraded is True
    assert "slow_node" in res.degraded_nodes
    assert res.stage_results["slow_node"].status == StageStatus.TIMEOUT
    assert res.stage_results["slow_node"].output == {"fallback": True}
    assert res.final_decision["used_fallback"] == {"fallback": True}


def test_decision_graph_executor_required_stage_failure() -> None:
    def broken_processor(_ctx: PipelineContext) -> None:
        raise ValueError("Critical hardware fault")

    graph = DecisionGraph("required_graph")
    graph.add_node(
        PipelineNode(
            name="critical_node",
            stage_type=StageType.ACTION,
            processor=broken_processor,
            required=True,
        )
    )

    executor = DecisionGraphExecutor(graph)
    with pytest.raises(PipelineExecutionError, match="Critical hardware fault"):
        asyncio.run(executor.execute({"Amount": 10.0}))


def test_decision_graph_executor_node_condition_skipped() -> None:
    graph = DecisionGraph("condition_graph")
    graph.add_node(
        PipelineNode(
            name="cond_node",
            stage_type=StageType.ENRICHMENT,
            processor=lambda _ctx: {"processed": True},
            condition=lambda ctx: ctx.transaction.get("should_run", False),
            fallback_value={"processed": False},
        )
    )

    executor = DecisionGraphExecutor(graph)
    res = asyncio.run(executor.execute({"should_run": False}))

    assert res.stage_results["cond_node"].status == StageStatus.SKIPPED
    assert res.stage_results["cond_node"].output == {"processed": False}


def test_create_default_fraud_pipeline_e2e() -> None:
    dataset = validate_frame(generate_synthetic_data(rows=300, fraud_rate=0.1, random_state=42))
    model = train_model(dataset)

    rule = DecisionRule(
        rule_id="R99",
        name="Suspicious Amount Rule",
        action=RuleAction.DENY,
        conditions=(
            RuleCondition(field="Amount", operator=RuleOperator.GREATER_THAN, value=5000.0),
        ),
    )
    ruleset = RuleSet(rules=(rule,))

    graph = create_default_fraud_pipeline(
        model=model,
        rules=ruleset,
        threshold=0.5,
        review_threshold=0.2,
        deny_threshold=0.8,
        explain=True,
    )

    executor = DecisionGraphExecutor(graph)

    # Normal transaction below threshold
    res_normal = executor.execute_sync({"Amount": 25.0, "V1": 0.0, "V2": 0.0})
    assert res_normal.success is True
    assert res_normal.degraded is False
    assert res_normal.final_decision["decision"] in ("ALLOW", "CHALLENGE")
    assert res_normal.final_decision["matched_rule"] is None

    # Transaction matching rule
    res_flagged = executor.execute_sync({"Amount": 6000.0, "V1": 0.0, "V2": 0.0})
    assert res_flagged.success is True
    assert res_flagged.final_decision["decision"] == "DENY"
    assert res_flagged.final_decision["is_fraud"] is True
    assert res_flagged.final_decision["matched_rule"] == "R99"
