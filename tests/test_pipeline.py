"""Unit and integration tests for decision graph execution pipeline."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from fraud_detection.data import generate_synthetic_data, validate_frame
from fraud_detection.evaluation import TieredThresholds
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
    NodeCachePolicy,
    NodeCacheStats,
    PipelineChaosReport,
    PipelineCircuitOpenError,
    PipelineContext,
    PipelineCounterfactualEvaluation,
    PipelineCycleError,
    PipelineDiscrepancyDetail,
    PipelineExecutionError,
    PipelineNode,
    PipelineReplayReport,
    PipelineValidationError,
    RuleStageProcessor,
    StageChaosMetrics,
    StageCircuitBreaker,
    StageCircuitPolicy,
    StageCircuitRegistry,
    StageCircuitState,
    StageCircuitStats,
    StageExecutionCache,
    StageExecutionResult,
    StageFaultSpec,
    StageFaultType,
    StageStatus,
    StageType,
    create_default_fraud_pipeline,
    optimize_counterfactual_pipeline,
    replay_audit_log_through_pipeline,
    simulate_pipeline_chaos,
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


def test_create_default_fraud_pipeline_cache_and_speculative_options() -> None:
    dataset = validate_frame(generate_synthetic_data(rows=300, fraud_rate=0.1, random_state=42))
    model = train_model(dataset)

    graph = create_default_fraud_pipeline(
        model=model,
        enable_cache=True,
        cache_rules=True,
        cache_inference=True,
        speculative_inference=True,
    )
    assert graph.metadata["speculative_inference"] is True
    assert graph.metadata["enable_cache"] is True
    assert graph.nodes["rules"].cache_policy is not None
    assert graph.nodes["inference"].cache_policy is not None

    executor = DecisionGraphExecutor(graph)
    assert executor.enable_speculative is True

    # Execute first time: cache miss, speculative hit
    sample_tx = dict.fromkeys(model.feature_names, 0.2)
    sample_tx["Amount"] = 42.0
    res1 = executor.execute_sync(sample_tx)
    assert res1.speculative_executed is True
    assert res1.speculative_hit is True
    assert res1.cache_hits == 0

    # Execute second time: cache hit for rules and inference!
    res2 = executor.execute_sync(sample_tx)
    assert res2.cache_hits >= 1


def test_action_aggregator_precedence_and_thresholds() -> None:
    # 1. MODEL_OVERRIDES_RULES
    proc_override = ActionAggregatorStageProcessor(
        threshold=0.5,
        rule_precedence=RulePrecedence.MODEL_OVERRIDES_RULES,
    )
    ctx1 = PipelineContext(transaction={"Amount": 10.0})
    ctx1.results["rules"] = StageExecutionResult(
        node_name="rules",
        stage_type=StageType.RULES,
        status=StageStatus.SUCCESS,
        output={"matched": True, "matched_rule_id": "R1", "action": "ALLOW"},
    )
    ctx1.results["calibration"] = StageExecutionResult(
        node_name="calibration",
        stage_type=StageType.CALIBRATION,
        status=StageStatus.SUCCESS,
        output={"calibrated_probability": 0.95, "raw_probability": 0.95},
    )
    out1 = proc_override(ctx1)
    assert out1["decision"] == "DENY"
    assert out1["is_fraud"] is True

    # 2. Tiered Challenge Review Threshold
    proc_tiered = ActionAggregatorStageProcessor(
        threshold=0.5,
        review_threshold=0.3,
        deny_threshold=0.7,
    )
    ctx2 = PipelineContext(transaction={"Amount": 10.0})
    ctx2.results["calibration"] = StageExecutionResult(
        node_name="calibration",
        stage_type=StageType.CALIBRATION,
        status=StageStatus.SUCCESS,
        output={"calibrated_probability": 0.45, "raw_probability": 0.45},
    )
    out2 = proc_tiered(ctx2)
    assert out2["decision"] == "CHALLENGE"
    assert out2["is_fraud"] is False

    # 3. No probability provided
    ctx3 = PipelineContext(transaction={"Amount": 10.0})
    out3 = proc_tiered(ctx3)
    assert out3["decision"] == "ALLOW"
    assert out3["is_fraud"] is False


def test_decision_graph_executor_fail_soft_processor_error() -> None:
    def buggy_processor(_ctx: PipelineContext) -> None:
        raise RuntimeError("Transient store failure")

    graph = DecisionGraph("fail_soft_graph")
    graph.add_node(
        PipelineNode(
            name="optional_node",
            stage_type=StageType.ENRICHMENT,
            processor=buggy_processor,
            fallback_value={"recovered": True},
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
                "data": ctx.get_output("optional_node"),
            },
            dependencies=["optional_node"],
        )
    )

    executor = DecisionGraphExecutor(graph)
    res = asyncio.run(executor.execute({"Amount": 10.0}))
    assert res.success is True
    assert res.degraded is True
    assert "optional_node" in res.degraded_nodes
    assert res.stage_results["optional_node"].status == StageStatus.FALLBACK
    assert res.stage_results["optional_node"].output == {"recovered": True}
    assert res.final_decision["data"] == {"recovered": True}


def test_pipeline_context_set_output_and_inference_none_model() -> None:
    ctx = PipelineContext(transaction={"Amount": 5.0})
    ctx.results["stage1"] = StageExecutionResult(
        node_name="stage1",
        stage_type=StageType.CUSTOM,
        status=StageStatus.SUCCESS,
        output="old",
    )
    ctx.set_output("stage1", "new")
    assert ctx.get_output("stage1") == "new"

    # InferenceStageProcessor with model=None
    inf_proc = InferenceStageProcessor(model=None)
    inf_res = inf_proc(ctx)
    assert inf_res["probability"] is None


def test_node_cache_policy_and_stats_serialization() -> None:
    policy = NodeCachePolicy(
        enabled=True,
        ttl_seconds=30.0,
        max_size=500,
        key_fields=("card_id", "Amount"),
    )
    p_dict = policy.to_dict()
    assert p_dict["enabled"] is True
    assert p_dict["ttl_seconds"] == 30.0
    assert p_dict["max_size"] == 500
    assert p_dict["key_fields"] == ["card_id", "Amount"]
    assert p_dict["has_custom_key_fn"] is False

    stats = NodeCacheStats(hits=10, misses=5, evictions=2, expirations=1, size=20, max_size=500)
    assert stats.total_lookups == 15
    assert stats.hit_ratio == round(10 / 15, 4)
    s_dict = stats.to_dict()
    assert s_dict["hits"] == 10
    assert s_dict["misses"] == 5
    assert s_dict["hit_ratio"] == round(10 / 15, 4)


def test_stage_execution_cache_get_put_eviction_and_expiry() -> None:
    cache = StageExecutionCache()
    policy = NodeCachePolicy(ttl_seconds=10.0, max_size=2)
    cache.register_policy("rules", policy)
    assert cache.get_policy("rules") == policy

    # Miss
    hit, val = cache.get("rules", "key1", now=100.0)
    assert hit is False
    assert val is None

    # Put key1 and key2
    cache.put("rules", "key1", {"decision": "ALLOW"}, now=100.0)
    cache.put("rules", "key2", {"decision": "DENY"}, now=101.0)

    # Hit key1 (moves key1 to end)
    hit, val = cache.get("rules", "key1", now=102.0)
    assert hit is True
    assert val == {"decision": "ALLOW"}

    # Put key3 -> evicts oldest (key2, since key1 was accessed more recently)
    cache.put("rules", "key3", {"decision": "CHALLENGE"}, now=103.0)

    hit_k2, val_k2 = cache.get("rules", "key2", now=104.0)
    assert hit_k2 is False
    assert val_k2 is None

    hit_k1, val_k1 = cache.get("rules", "key1", now=105.0)
    assert hit_k1 is True
    assert val_k1 == {"decision": "ALLOW"}

    # Expiry test at now=120.0 (TTL 10.0 from 100.0/103.0)
    hit_expired, _ = cache.get("rules", "key1", now=120.0)
    assert hit_expired is False

    st = cache.get_stats("rules")
    assert st.hits == 2
    assert st.misses == 3
    assert st.evictions == 1
    assert st.expirations == 1


def test_stage_execution_cache_key_generation_and_clear() -> None:
    cache = StageExecutionCache()

    # Default key generation (all tx keys)
    ctx1 = PipelineContext(transaction={"b": 2, "a": 1})
    ctx2 = PipelineContext(transaction={"a": 1, "b": 2})
    k1 = cache.compute_key("stage", ctx1)
    k2 = cache.compute_key("stage", ctx2)
    assert k1 == k2  # Canonical sorted keys produce deterministic hash

    # Specific key fields
    policy_fields = NodeCachePolicy(key_fields=("card_id",))
    cache.register_policy("stage_fields", policy_fields)
    ctx_f1 = PipelineContext(transaction={"card_id": "c1", "Amount": 100.0})
    ctx_f2 = PipelineContext(transaction={"card_id": "c1", "Amount": 500.0})
    kf1 = cache.compute_key("stage_fields", ctx_f1)
    kf2 = cache.compute_key("stage_fields", ctx_f2)
    assert kf1 == kf2  # Ignores Amount since key_fields is only ('card_id',)

    # Custom key function
    policy_custom = NodeCachePolicy(key_fn=lambda c: f"custom_{c.transaction.get('user_id')}")
    cache.register_policy("stage_custom", policy_custom)
    ctx_c = PipelineContext(transaction={"user_id": "u42"})
    assert cache.compute_key("stage_custom", ctx_c) == "custom_u42"

    # Put and clear
    cache.put("stage", k1, "val1")
    cache.put("stage_fields", kf1, "val2")
    stats = cache.stats_dict()
    assert stats["total_size"] == 2

    cache.clear("stage")
    assert cache.get_stats("stage").size == 0
    assert cache.get_stats("stage_fields").size == 1

    cache.clear()
    assert cache.stats_dict()["total_size"] == 0


def test_decision_graph_executor_node_caching_hit_and_miss() -> None:
    call_count = 0

    def expensive_rules(ctx: PipelineContext) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        amt = ctx.transaction.get("Amount", 0.0)
        return {"decision": "DENY" if amt > 1000.0 else "ALLOW"}

    graph = DecisionGraph("cached_graph")
    graph.add_node(
        PipelineNode(
            name="rules",
            stage_type=StageType.RULES,
            processor=expensive_rules,
            cache_policy=NodeCachePolicy(enabled=True, ttl_seconds=60.0),
        )
    )
    graph.add_node(
        PipelineNode(
            name="action",
            stage_type=StageType.ACTION,
            processor=lambda ctx: ctx.get_output("rules"),
            dependencies=["rules"],
        )
    )

    executor = DecisionGraphExecutor(graph, enable_cache=True)

    # First run: Cache miss
    res1 = executor.execute_sync({"Amount": 50.0, "card_id": "c1"})
    assert call_count == 1
    assert res1.stage_results["rules"].cached is False
    assert res1.cache_hits == 0
    assert res1.cache_misses == 1
    assert res1.final_decision == {"decision": "ALLOW"}

    # Second run with same input: Cache hit!
    res2 = executor.execute_sync({"Amount": 50.0, "card_id": "c1"})
    assert call_count == 1  # Processor was NOT called again!
    assert res2.stage_results["rules"].cached is True
    assert res2.cache_hits == 1
    assert res2.cache_misses == 0
    assert res2.final_decision == {"decision": "ALLOW"}

    # Third run with different input: Cache miss
    res3 = executor.execute_sync({"Amount": 2000.0, "card_id": "c2"})
    assert call_count == 2
    assert res3.stage_results["rules"].cached is False
    assert res3.cache_hits == 0
    assert res3.cache_misses == 1
    assert res3.final_decision == {"decision": "DENY"}

    # Check stats dict
    stats = executor.cache.stats_dict()
    assert stats["total_hits"] == 1
    assert stats["total_misses"] == 2
    assert stats["nodes"]["rules"]["hits"] == 1
    assert stats["nodes"]["rules"]["misses"] == 2

    # Disable cache
    executor_no_cache = DecisionGraphExecutor(graph, enable_cache=False)
    res4 = executor_no_cache.execute_sync({"Amount": 50.0, "card_id": "c1"})
    assert call_count == 3
    assert res4.stage_results["rules"].cached is False
    assert res4.cache_hits == 0


def test_speculative_execution_hit_when_features_unchanged() -> None:
    dataset = validate_frame(generate_synthetic_data(rows=300, fraud_rate=0.1, random_state=42))
    model = train_model(dataset)

    def dummy_enrich(ctx: PipelineContext) -> dict[str, Any]:
        # Modifies only a metadata column not in model.feature_names
        ctx.transaction["unrelated_tag"] = "enrich_ok"
        return {"enriched": True}

    graph = DecisionGraph("spec_graph")
    graph.add_node(
        PipelineNode(
            name="enrichment",
            stage_type=StageType.ENRICHMENT,
            processor=dummy_enrich,
        )
    )
    graph.add_node(
        PipelineNode(
            name="inference",
            stage_type=StageType.INFERENCE,
            processor=InferenceStageProcessor(model),
            dependencies=["enrichment"],
        )
    )
    graph.add_node(
        PipelineNode(
            name="action",
            stage_type=StageType.ACTION,
            processor=lambda ctx: ctx.get_output("inference"),
            dependencies=["inference"],
            required=True,
        )
    )

    executor = DecisionGraphExecutor(graph, enable_speculative=True)
    sample_tx = dict.fromkeys(model.feature_names, 0.1)
    res = executor.execute_sync(sample_tx)

    assert res.speculative_executed is True
    assert res.speculative_hit is True
    assert res.stage_results["inference"].speculative is True
    assert res.stage_results["inference"].output["probability"] is not None


def test_speculative_execution_discard_when_features_modified() -> None:
    dataset = validate_frame(generate_synthetic_data(rows=300, fraud_rate=0.1, random_state=42))
    model = train_model(dataset)

    def modifying_enrich(ctx: PipelineContext) -> dict[str, Any]:
        # Modifies Amount which IS in model.feature_names
        ctx.transaction["Amount"] = 99999.0
        return {"modified": True}

    graph = DecisionGraph("spec_mod_graph")
    graph.add_node(
        PipelineNode(
            name="enrichment",
            stage_type=StageType.ENRICHMENT,
            processor=modifying_enrich,
        )
    )
    graph.add_node(
        PipelineNode(
            name="inference",
            stage_type=StageType.INFERENCE,
            processor=InferenceStageProcessor(model),
            dependencies=["enrichment"],
        )
    )
    graph.add_node(
        PipelineNode(
            name="action",
            stage_type=StageType.ACTION,
            processor=lambda ctx: ctx.get_output("inference"),
            dependencies=["inference"],
            required=True,
        )
    )

    executor = DecisionGraphExecutor(graph, enable_speculative=True)
    sample_tx = dict.fromkeys(model.feature_names, 0.1)
    sample_tx["Amount"] = 1.0
    res = executor.execute_sync(sample_tx)

    assert res.speculative_executed is True
    assert res.speculative_hit is False
    assert res.stage_results["inference"].speculative is False
    assert res.stage_results["inference"].output["probability"] is not None


def test_speculative_execution_discard_on_short_circuit() -> None:
    dataset = validate_frame(generate_synthetic_data(rows=300, fraud_rate=0.1, random_state=42))
    model = train_model(dataset)

    rule = DecisionRule(
        rule_id="r1",
        name="Deny All",
        conditions=[RuleCondition(field="Amount", operator=RuleOperator.GREATER_THAN, value=0.0)],
        action=RuleAction.DENY,
    )
    rule_set = RuleSet(rules=(rule,))

    graph = DecisionGraph("spec_short_circuit_graph")
    graph.add_node(
        PipelineNode(
            name="rules",
            stage_type=StageType.RULES,
            processor=RuleStageProcessor(rules=rule_set),
        )
    )
    graph.add_node(
        PipelineNode(
            name="inference",
            stage_type=StageType.INFERENCE,
            processor=InferenceStageProcessor(model),
            dependencies=["rules"],
        )
    )
    graph.add_node(
        PipelineNode(
            name="action",
            stage_type=StageType.ACTION,
            processor=lambda ctx: ctx.get_output("rules"),
            dependencies=["inference"],
        )
    )

    executor = DecisionGraphExecutor(graph, enable_speculative=True)

    sample_tx = dict.fromkeys(model.feature_names, 0.1)
    sample_tx["Amount"] = 100.0
    res = executor.execute_sync(sample_tx)

    assert res.speculative_executed is True
    # Rule short-circuited: inference should be skipped, speculative output discarded
    assert res.speculative_hit is False
    assert res.stage_results["inference"].output["skipped"] is True






def test_speculative_custom_node_and_global_timeout() -> None:
    graph = DecisionGraph("spec_custom_graph")
    graph.add_node(
        PipelineNode(
            name="enrichment",
            stage_type=StageType.ENRICHMENT,
            processor=lambda _ctx: {"enriched": True},
        )
    )
    graph.add_node(
        PipelineNode(
            name="scoring",
            stage_type=StageType.INFERENCE,
            processor=lambda _ctx: {"probability": 0.2},
            dependencies=["enrichment"],
        )
    )
    graph.add_node(
        PipelineNode(
            name="action",
            stage_type=StageType.ACTION,
            processor=lambda ctx: {
                "decision": "ALLOW",
                "is_fraud": False,
                "fraud_probability": ctx.get_output("scoring")["probability"],
            },
            dependencies=["scoring"],
            required=True,
        )
    )

    executor = DecisionGraphExecutor(
        graph, enable_speculative=True, speculative_nodes=["scoring"]
    )
    res = asyncio.run(executor.execute({"Amount": 1.0}, global_timeout_seconds=5.0))

    assert res.speculative_executed is True
    assert res.speculative_hit is True
    assert res.stage_results["scoring"].speculative is True
    assert res.decision == "ALLOW"


def _create_simple_replay_graph(prob_fn: Any = None, decision_fn: Any = None) -> DecisionGraph:
    graph = DecisionGraph("replay_test_graph")
    graph.add_node(
        PipelineNode(
            name="enrichment",
            stage_type=StageType.ENRICHMENT,
            processor=lambda ctx: dict(ctx.transaction),
        )
    )
    graph.add_node(
        PipelineNode(
            name="inference",
            stage_type=StageType.INFERENCE,
            processor=lambda ctx: {
                "probability": (
                    prob_fn(ctx.transaction)
                    if prob_fn is not None
                    else float(ctx.transaction.get("prob", 0.1))
                )
            },
            dependencies=["enrichment"],
        )
    )
    graph.add_node(
        PipelineNode(
            name="action",
            stage_type=StageType.ACTION,
            processor=lambda ctx: {
                "decision": (
                    decision_fn(ctx.get_output("inference")["probability"])
                    if decision_fn is not None
                    else ("DENY" if ctx.get_output("inference")["probability"] >= 0.5 else "ALLOW")
                ),
                "is_fraud": ctx.get_output("inference")["probability"] >= 0.5,
                "fraud_probability": ctx.get_output("inference")["probability"],
                "matched_rule": ctx.transaction.get("rule"),
            },
            dependencies=["inference"],
            required=True,
        )
    )
    return graph


def test_replay_audit_log_through_pipeline_inline_and_divergent(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    events = [
        {
            "event_id": "evt_1",
            "event_type": "scoring",
            "payload": {
                "predictions": [
                    {"fraud_probability": 0.1, "decision": "ALLOW", "is_fraud": False},
                    {"fraud_probability": 0.8, "decision": "DENY", "is_fraud": True},
                ],
                "features": [
                    {"prob": 0.1, "Amount": 10.0},
                    {"prob": 0.8, "Amount": 500.0},
                ],
            },
        },
        {
            "event_id": "evt_2",
            "event_type": "scoring",
            "payload": {
                "predictions": [
                    {"fraud_probability": 0.2, "is_fraud": False},
                    {"fraud_probability": 0.4, "decision": "CHALLENGE"},
                ],
                "features": [
                    {"prob": 0.9, "rule": "high_risk"},
                    {"prob": 0.4, "Amount": 100.0},
                ],
            },
        },
        {"event_id": "evt_promote", "event_type": "promotion", "payload": {}},
        {"event_id": "evt_no_payload", "event_type": "scoring", "payload": "invalid"},
        {"event_id": "evt_no_preds", "event_type": "scoring", "payload": {"predictions": []}},
        {
            "event_id": "evt_mismatch_len",
            "event_type": "scoring",
            "payload": {
                "predictions": [{"fraud_probability": 0.1}],
                "features": [{"prob": 0.1}, {"prob": 0.2}],
            },
        },
        {
            "event_id": "evt_bad_feature_item",
            "event_type": "scoring",
            "payload": {
                "predictions": [{"fraud_probability": 0.1}],
                "features": ["not_a_dict"],
            },
        },
        {
            "event_id": "evt_no_features_no_csv",
            "event_type": "scoring",
            "payload": {"predictions": [{"fraud_probability": 0.1}]},
        },
    ]

    with audit_path.open("w", encoding="utf-8") as f:
        f.write("\n")
        f.write("{invalid json\n")
        f.write("[]\n")
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    graph = _create_simple_replay_graph()
    report = replay_audit_log_through_pipeline(
        audit_path,
        graph,
        tolerance=1e-4,
        max_discrepancies_to_record=10,
    )

    assert isinstance(report, PipelineReplayReport)
    assert report.scoring_events == 7
    assert report.replayed_events == 2
    assert report.skipped_events == 5
    assert report.total_transactions == 4
    assert report.score_discrepancies == 1
    assert report.decision_flips == 2
    assert report.status == "DIVERGENT"
    assert len(report.discrepancies) == 2
    assert isinstance(report.discrepancies[0], PipelineDiscrepancyDetail)

    d_dict = report.to_dict()
    assert d_dict["status"] == "DIVERGENT"
    assert d_dict["transition_matrix"]["turnover_count"] == 2
    assert len(d_dict["discrepancies"]) == 2
    assert "enrichment" in d_dict["stage_latencies"]


def test_replay_audit_log_through_pipeline_external_csv_and_match(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit_ext.jsonl"
    csv_path = tmp_path / "obs.csv"
    csv_path.write_text("prob,Amount\n0.15,20.0\n0.85,200.0\n", encoding="utf-8")

    ev1 = {
        "event_id": "ext_1",
        "event_type": "scoring",
        "payload": {
            "predictions": [
                {"fraud_probability": 0.15, "decision": "ALLOW"},
                {"fraud_probability": 0.85, "decision": "DENY"},
            ]
        },
    }
    ev2 = {
        "event_id": "ext_2",
        "event_type": "scoring",
        "payload": {
            "predictions": [
                {"fraud_probability": 0.5, "decision": "DENY"},
            ]
        },
    }
    audit_path.write_text(json.dumps(ev1) + "\n" + json.dumps(ev2) + "\n", encoding="utf-8")

    executor = DecisionGraphExecutor(_create_simple_replay_graph())
    report = replay_audit_log_through_pipeline(
        audit_path,
        executor,
        data_path=csv_path,
        tolerance=1e-4,
    )

    assert report.status == "MATCH"
    assert report.replayed_events == 1
    assert report.skipped_events == 1
    assert report.total_transactions == 2
    assert report.score_discrepancies == 0
    assert report.decision_flips == 0


def test_replay_audit_log_edge_cases_and_validation(tmp_path: Path) -> None:
    graph = _create_simple_replay_graph()
    audit_path = tmp_path / "empty.jsonl"
    audit_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="tolerance"):
        replay_audit_log_through_pipeline(audit_path, graph, tolerance=-0.1)

    with pytest.raises(ValueError, match="tolerance"):
        replay_audit_log_through_pipeline(audit_path, graph, tolerance=True)

    with pytest.raises(ValueError, match="max_discrepancies_to_record"):
        replay_audit_log_through_pipeline(audit_path, graph, max_discrepancies_to_record=-1)

    with pytest.raises(FileNotFoundError, match="Audit log file not found"):
        replay_audit_log_through_pipeline(tmp_path / "missing.jsonl", graph)

    with pytest.raises(FileNotFoundError, match="External observations file not found"):
        replay_audit_log_through_pipeline(
            audit_path, graph, data_path=tmp_path / "missing.csv"
        )

    empty_rep = replay_audit_log_through_pipeline(audit_path, graph)
    assert empty_rep.status == "EMPTY"
    assert empty_rep.total_transactions == 0

    bad_ev = {
        "event_id": "bad_tx",
        "event_type": "scoring",
        "payload": {
            "predictions": [
                "not_a_dict",
                {"fraud_probability": None},
                {"fraud_probability": True},
                {"fraud_probability": "not_a_float"},
                {"fraud_probability": 1.5},
                {"fraud_probability": 0.5, "decision": "DENY"},
            ],
            "features": [
                {"prob": 0.1},
                {"prob": 0.1},
                {"prob": 0.1},
                {"prob": 0.1},
                {"prob": 0.1},
                {"raise_err": True},
            ],
        },
    }
    audit_path.write_text(json.dumps(bad_ev) + "\n", encoding="utf-8")

    def _err_prob(tx: dict[str, Any]) -> float:
        if tx.get("raise_err"):
            raise RuntimeError("boom")
        return 0.1

    err_graph = _create_simple_replay_graph(prob_fn=_err_prob)
    err_graph.nodes["inference"].required = True
    rep2 = replay_audit_log_through_pipeline(audit_path, err_graph)
    assert rep2.status == "EMPTY"
    assert rep2.skipped_events == 1


def test_optimize_counterfactual_pipeline_metrics_and_validation() -> None:
    probs = [0.1, 0.35, 0.6, 0.85]
    y_true = [0, 0, 1, 1]
    base_thresh = TieredThresholds(review_threshold=0.5, deny_threshold=0.8)
    cand_thresh = TieredThresholds(review_threshold=0.3, deny_threshold=0.7)

    eval_res = optimize_counterfactual_pipeline(
        probabilities=probs,
        candidate_thresholds=cand_thresh,
        baseline_thresholds=base_thresh,
        y_true=y_true,
        manual_review_cost=5.0,
        false_deny_cost=50.0,
        missed_fraud_cost=200.0,
    )
    assert isinstance(eval_res, PipelineCounterfactualEvaluation)
    assert eval_res.rows == 4
    assert eval_res.turnover_count > 0
    assert eval_res.baseline_metrics is not None
    assert eval_res.candidate_metrics is not None
    assert eval_res.cost_delta is not None
    assert eval_res.fraud_catch_delta is not None

    d = eval_res.to_dict()
    assert d["rows"] == 4
    assert d["baseline_metrics"] is not None

    eval_with_dec = optimize_counterfactual_pipeline(
        probabilities=probs,
        candidate_thresholds=cand_thresh,
        baseline_decisions=["ALLOW", "REVIEW", "DENY", "DENY"],
    )
    assert eval_with_dec.baseline_decision_counts["ALLOW"] == 1
    assert eval_with_dec.baseline_decision_counts["CHALLENGE"] == 1
    assert eval_with_dec.baseline_decision_counts["DENY"] == 2
    assert eval_with_dec.baseline_metrics is None

    with pytest.raises(TypeError, match="candidate_thresholds"):
        optimize_counterfactual_pipeline(probs, "invalid")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="baseline_thresholds"):
        optimize_counterfactual_pipeline(
            probs, cand_thresh, baseline_thresholds="invalid"  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="manual_review_cost"):
        optimize_counterfactual_pipeline(probs, cand_thresh, manual_review_cost=-1.0)

    with pytest.raises(ValueError, match="non-empty"):
        optimize_counterfactual_pipeline([], cand_thresh)

    with pytest.raises(ValueError, match="between 0 and 1"):
        optimize_counterfactual_pipeline([0.2, 1.5], cand_thresh)

    with pytest.raises(ValueError, match="baseline_decisions"):
        optimize_counterfactual_pipeline(probs, cand_thresh, baseline_decisions=["ALLOW"])

    with pytest.raises(ValueError, match="Length of y_true"):
        optimize_counterfactual_pipeline(probs, cand_thresh, y_true=[0, 1])


def test_stage_execution_result_circuit_open_serialization() -> None:
    err = PipelineCircuitOpenError("stage circuit is open")
    assert isinstance(err, PipelineExecutionError)

    res = StageExecutionResult(
        node_name="enrichment",
        stage_type=StageType.ENRICHMENT,
        status=StageStatus.CIRCUIT_OPEN,
        latency_ms=0.0,
        output={"fallback": True},
        error="Stage circuit breaker is open",
        degraded=True,
        short_circuited_by_breaker=True,
    )
    payload = res.to_dict()
    assert payload["status"] == "circuit_open"
    assert payload["degraded"] is True
    assert payload["short_circuited_by_breaker"] is True
    assert payload["output"] == {"fallback": True}


def test_stage_circuit_policy_and_stats_validation() -> None:
    policy = StageCircuitPolicy(
        enabled=True,
        failure_threshold=3,
        recovery_timeout_seconds=15.0,
        half_open_max_calls=2,
        count_timeouts=True,
        count_exceptions=False,
    )
    assert policy.to_dict() == {
        "enabled": True,
        "failure_threshold": 3,
        "recovery_timeout_seconds": 15.0,
        "half_open_max_calls": 2,
        "count_timeouts": True,
        "count_exceptions": False,
    }

    with pytest.raises(ValueError, match="failure_threshold"):
        StageCircuitPolicy(failure_threshold=0)
    with pytest.raises(ValueError, match="failure_threshold"):
        StageCircuitPolicy(failure_threshold=True)

    with pytest.raises(ValueError, match="recovery_timeout_seconds"):
        StageCircuitPolicy(recovery_timeout_seconds=0.0)
    with pytest.raises(ValueError, match="recovery_timeout_seconds"):
        StageCircuitPolicy(recovery_timeout_seconds=float("inf"))
    with pytest.raises(ValueError, match="recovery_timeout_seconds"):
        StageCircuitPolicy(recovery_timeout_seconds=True)

    with pytest.raises(ValueError, match="half_open_max_calls"):
        StageCircuitPolicy(half_open_max_calls=0)
    with pytest.raises(ValueError, match="half_open_max_calls"):
        StageCircuitPolicy(half_open_max_calls=False)

    stats = StageCircuitStats(
        state=StageCircuitState.OPEN,
        consecutive_failures=3,
        total_failures=5,
        total_successes=10,
        total_trips=2,
        short_circuited_count=4,
        last_failure_reason="timeout",
    )
    assert stats.to_dict() == {
        "state": "open",
        "consecutive_failures": 3,
        "total_failures": 5,
        "total_successes": 10,
        "total_trips": 2,
        "short_circuited_count": 4,
        "last_failure_reason": "timeout",
    }


def test_stage_circuit_breaker_state_machine() -> None:
    policy = StageCircuitPolicy(
        enabled=True,
        failure_threshold=2,
        recovery_timeout_seconds=10.0,
        half_open_max_calls=2,
    )
    breaker = StageCircuitBreaker(policy)
    assert breaker.allow_execution(now=100.0) is True
    breaker.record_success()
    assert breaker.get_stats(now=100.0).total_successes == 1

    # First failure stays CLOSED
    assert breaker.record_failure(reason="boom1", now=101.0) is False
    assert breaker.get_stats(now=101.0).state == StageCircuitState.CLOSED

    # Second consecutive failure trips to OPEN
    assert breaker.record_failure(reason="boom2", now=102.0) is True
    st_open = breaker.get_stats(now=102.0)
    assert st_open.state == StageCircuitState.OPEN
    assert st_open.total_trips == 1
    assert st_open.last_failure_reason == "boom2"

    # Additional failure while already OPEN stays OPEN
    assert breaker.record_failure(reason="boom3", now=103.0) is True

    # Before recovery timeout: short-circuited
    assert breaker.allow_execution(now=105.0) is False
    assert breaker.get_stats(now=105.0).short_circuited_count == 1

    # After recovery timeout: get_stats reports HALF_OPEN
    assert breaker.get_stats(now=113.0).state == StageCircuitState.HALF_OPEN

    # Half-open allows up to half_open_max_calls (2)
    assert breaker.allow_execution(now=113.0) is True
    assert breaker.allow_execution(now=113.1) is True
    assert breaker.allow_execution(now=113.2) is False
    assert breaker.get_stats(now=113.2).short_circuited_count == 2

    # First half-open success keeps HALF_OPEN until 2nd success
    breaker.record_success()
    assert breaker.get_stats(now=113.5).state == StageCircuitState.HALF_OPEN
    breaker.record_success()
    assert breaker.get_stats(now=114.0).state == StageCircuitState.CLOSED

    # Trip again, move to HALF_OPEN, then fail during probe -> re-trips to OPEN
    breaker.record_failure(reason="err1", now=200.0)
    breaker.record_failure(reason="err2", now=201.0)
    assert breaker.allow_execution(now=215.0) is True
    assert breaker.record_failure(reason="probe_fail", now=215.5) is True
    assert breaker.get_stats(now=216.0).state == StageCircuitState.OPEN
    assert breaker.get_stats(now=216.0).total_trips == 3

    # Manual reset returns to CLOSED
    breaker.reset()
    assert breaker.get_stats(now=216.0).state == StageCircuitState.CLOSED

    # Test disabled policy and selective counting
    disabled = StageCircuitBreaker(StageCircuitPolicy(enabled=False, failure_threshold=1))
    assert disabled.allow_execution() is True
    disabled.record_success()
    assert disabled.record_failure(reason="ignored") is False

    selective = StageCircuitBreaker(
        StageCircuitPolicy(
            failure_threshold=1,
            count_timeouts=False,
            count_exceptions=False,
        )
    )
    assert selective.record_failure(reason="timeout", is_timeout=True) is False
    assert selective.record_failure(reason="exc", is_timeout=False) is False


def test_stage_circuit_registry_operations() -> None:
    registry = StageCircuitRegistry()
    # Unregistered nodes always allow and no-op
    assert registry.allow_execution("unregistered") is True
    registry.record_success("unregistered")
    assert registry.record_failure("unregistered") is False
    assert registry.reset("unregistered") == []

    policy_a = StageCircuitPolicy(failure_threshold=1, recovery_timeout_seconds=10.0)
    policy_b = StageCircuitPolicy(failure_threshold=1, recovery_timeout_seconds=5.0)
    registry.register_policy("node_a", policy_a)
    registry.register_policy("node_b", policy_b)

    assert registry.allow_execution("node_a", now=10.0) is True
    registry.record_success("node_a")
    assert registry.record_failure("node_a", reason="err_a", now=11.0) is True
    assert registry.record_failure("node_b", reason="err_b", now=11.0) is True
    assert registry.allow_execution("node_a", now=12.0) is False

    # At now=17.0, node_a is still OPEN (10s timeout), node_b is HALF_OPEN (5s timeout)
    stats = registry.stats_dict(now=17.0)
    assert stats["total_trips"] == 2
    assert stats["total_short_circuits"] == 1
    assert stats["open_nodes"] == ["node_a"]
    assert stats["half_open_nodes"] == ["node_b"]
    assert stats["nodes"]["node_a"]["policy"]["failure_threshold"] == 1

    assert registry.reset("node_a") == ["node_a"]
    assert registry.get_breaker("node_a") is not None
    assert registry.reset() == ["node_a", "node_b"]


def test_executor_stage_circuit_breaking_fast_fail() -> None:
    fail_enrichment = True

    def flakey_enrichment(_ctx: PipelineContext) -> dict[str, Any]:
        if fail_enrichment:
            raise RuntimeError("Enrichment service unavailable")
        return {"account_age": 100}

    graph = DecisionGraph("cb_graph")
    graph.add_node(
        PipelineNode(
            name="enrichment",
            stage_type=StageType.ENRICHMENT,
            processor=flakey_enrichment,
            fallback_value={"account_age": 0},
            required=False,
            circuit_policy=StageCircuitPolicy(
                failure_threshold=2,
                recovery_timeout_seconds=0.05,
                half_open_max_calls=1,
            ),
        )
    )
    graph.add_node(
        PipelineNode(
            name="action",
            stage_type=StageType.ACTION,
            processor=lambda ctx: {
                "decision": "ALLOW",
                "age": ctx.get_output("enrichment").get("account_age", 0),
            },
            dependencies=["enrichment"],
            required=True,
        )
    )

    executor = DecisionGraphExecutor(graph)

    # 1st run fails (fallback, degraded, breaker consecutive failures = 1, CLOSED)
    res1 = executor.execute_sync({"tx_id": "1"})
    assert res1.stage_results["enrichment"].status == StageStatus.FALLBACK
    assert res1.stage_results["enrichment"].degraded is True
    assert res1.stage_results["enrichment"].short_circuited_by_breaker is False
    assert res1.circuit_short_circuits == []

    # 2nd run fails (fallback, trips breaker to OPEN)
    res2 = executor.execute_sync({"tx_id": "2"})
    assert res2.stage_results["enrichment"].status == StageStatus.FALLBACK
    assert executor.circuit_registry.get_breaker("enrichment") is not None
    assert (
        executor.circuit_registry.get_breaker("enrichment").get_stats().state  # type: ignore[union-attr]
        == StageCircuitState.OPEN
    )

    # 3rd run short-circuits in O(1) immediately!
    res3 = executor.execute_sync({"tx_id": "3"})
    assert res3.stage_results["enrichment"].status == StageStatus.CIRCUIT_OPEN
    assert res3.stage_results["enrichment"].short_circuited_by_breaker is True
    assert res3.stage_results["enrichment"].degraded is True
    assert res3.circuit_short_circuits == ["enrichment"]
    assert res3.final_decision["age"] == 0
    assert res3.to_dict()["circuit_short_circuits"] == ["enrichment"]

    # Wait for recovery timeout, fix service, verify half-open recovery to CLOSED
    time.sleep(0.06)
    fail_enrichment = False
    res4 = executor.execute_sync({"tx_id": "4"})
    assert res4.stage_results["enrichment"].status == StageStatus.SUCCESS
    assert res4.stage_results["enrichment"].short_circuited_by_breaker is False
    assert res4.final_decision["age"] == 100
    assert (
        executor.circuit_registry.get_breaker("enrichment").get_stats().state  # type: ignore[union-attr]
        == StageCircuitState.CLOSED
    )


def test_executor_required_stage_circuit_open_raises() -> None:
    graph = DecisionGraph("req_cb_graph")
    graph.add_node(
        PipelineNode(
            name="required_step",
            stage_type=StageType.ACTION,
            processor=lambda _ctx: (_ for _ in ()).throw(RuntimeError("action failed")),
            required=True,
            circuit_policy=StageCircuitPolicy(failure_threshold=1, recovery_timeout_seconds=10.0),
        )
    )
    executor = DecisionGraphExecutor(graph)

    # 1st run fails and raises PipelineExecutionError, tripping breaker to OPEN
    with pytest.raises(PipelineExecutionError, match="action failed"):
        executor.execute_sync({})

    # 2nd run short-circuits and raises PipelineCircuitOpenError
    with pytest.raises(PipelineCircuitOpenError, match="circuit breaker is open"):
        executor.execute_sync({})


def test_default_fraud_pipeline_circuit_breakers() -> None:
    dataset = validate_frame(generate_synthetic_data(rows=250, fraud_rate=0.1, random_state=42))
    model = train_model(dataset)

    pipe = create_default_fraud_pipeline(
        model,
        enable_circuit_breaker=True,
        circuit_failure_threshold=3,
        circuit_recovery_timeout_seconds=15.0,
    )
    assert pipe.metadata["enable_circuit_breaker"] is True
    assert pipe.metadata["circuit_failure_threshold"] == 3
    assert pipe.metadata["circuit_recovery_timeout_seconds"] == 15.0

    enrichment_node = pipe.nodes["enrichment"]
    assert enrichment_node.circuit_policy is not None
    assert enrichment_node.circuit_policy.failure_threshold == 3
    assert enrichment_node.circuit_policy.recovery_timeout_seconds == 15.0
    assert enrichment_node.to_dict()["circuit_policy"] is not None

    executor = DecisionGraphExecutor(pipe)
    stats = executor.circuit_registry.stats_dict()
    assert "enrichment" in stats["nodes"]
    assert "rules" in stats["nodes"]
    assert "inference" in stats["nodes"]
    assert "calibration" in stats["nodes"]


def test_stage_fault_spec_and_metrics_validation() -> None:
    spec = StageFaultSpec(
        node_name="enrichment",
        fault_type="exception",
        start_index=1,
        end_index=5,
        latency_ms=12.5,
    )
    assert spec.fault_type == StageFaultType.EXCEPTION
    d = spec.to_dict()
    assert d["node_name"] == "enrichment"
    assert d["fault_type"] == "exception"
    assert d["start_index"] == 1
    assert d["end_index"] == 5
    assert d["latency_ms"] == 12.5

    with pytest.raises(ValueError, match="node_name"):
        StageFaultSpec(node_name="", fault_type=StageFaultType.EXCEPTION)

    with pytest.raises(ValueError, match="fault_type"):
        StageFaultSpec(node_name="test", fault_type="unsupported")

    with pytest.raises(ValueError, match="start_index"):
        StageFaultSpec(node_name="test", fault_type=StageFaultType.EXCEPTION, start_index=-1)

    with pytest.raises(ValueError, match="end_index"):
        StageFaultSpec(
            node_name="test", fault_type=StageFaultType.EXCEPTION, start_index=5, end_index=2
        )

    with pytest.raises(ValueError, match="latency_ms"):
        StageFaultSpec(node_name="test", fault_type=StageFaultType.LATENCY_SPIKE, latency_ms=-1.0)

    with pytest.raises(ValueError, match="latency_ms"):
        StageFaultSpec(
            node_name="test", fault_type=StageFaultType.LATENCY_SPIKE, latency_ms=float("inf")
        )

    metrics = StageChaosMetrics(
        node_name="inference",
        fault_type="timeout",
        injected_count=3,
        handled_count=3,
        tripped_circuit=True,
    )
    assert metrics.to_dict() == {
        "node_name": "inference",
        "fault_type": "timeout",
        "injected_count": 3,
        "handled_count": 3,
        "tripped_circuit": True,
    }


def test_simulate_pipeline_chaos_exception_and_recovery() -> None:
    graph = DecisionGraph("chaos_sim_graph")
    graph.add_node(
        PipelineNode(
            name="enrichment",
            stage_type=StageType.ENRICHMENT,
            processor=lambda _ctx: {"account_tier": "gold"},
            fallback_value={"account_tier": "basic"},
            required=False,
            circuit_policy=StageCircuitPolicy(
                failure_threshold=2,
                recovery_timeout_seconds=0.02,
                half_open_max_calls=1,
            ),
        )
    )
    graph.add_node(
        PipelineNode(
            name="action",
            stage_type=StageType.ACTION,
            processor=lambda ctx: {
                "decision": "ALLOW",
                "tier": ctx.get_output("enrichment").get("account_tier", "basic"),
            },
            dependencies=["enrichment"],
            required=True,
        )
    )

    records = [{"id": i} for i in range(10)]
    faults = [
        StageFaultSpec(
            node_name="enrichment",
            fault_type=StageFaultType.EXCEPTION,
            start_index=0,
            end_index=3,
        )
    ]

    report = simulate_pipeline_chaos(
        records,
        graph,
        faults,
        sla_budget_ms=500.0,
        enable_circuit_breaker=True,
        circuit_failure_threshold=2,
        circuit_recovery_timeout_seconds=0.02,
    )

    assert isinstance(report, PipelineChaosReport)
    assert report.total_transactions == 10
    assert report.circuit_trips >= 1
    assert report.short_circuited_calls >= 1
    assert report.unhandled_errors == 0
    assert report.degraded_transactions >= 3
    assert report.stage_metrics["enrichment"].injected_count >= 2
    assert report.stage_metrics["enrichment"].handled_count >= 2
    assert report.stage_metrics["enrichment"].tripped_circuit is True
    assert report.latencies_p50_ms >= 0.0

    d = report.to_dict()
    assert d["total_transactions"] == 10
    assert d["circuit_breaker_enabled"] is True
    assert "enrichment" in d["stage_metrics"]


def test_simulate_pipeline_chaos_sla_breach_and_timeout() -> None:
    graph = DecisionGraph("chaos_timeout_graph")
    graph.add_node(
        PipelineNode(
            name="inference",
            stage_type=StageType.INFERENCE,
            processor=lambda _ctx: {"prob": 0.1},
            fallback_value={"prob": 0.5},
            required=False,
            timeout_seconds=0.01,
        )
    )
    graph.add_node(
        PipelineNode(
            name="action",
            stage_type=StageType.ACTION,
            processor=lambda _ctx: {"decision": "ALLOW"},
            dependencies=["inference"],
            required=True,
        )
    )

    records = [{"id": 0}, {"id": 1}]
    faults = [
        StageFaultSpec(
            node_name="inference",
            fault_type=StageFaultType.TIMEOUT,
            start_index=0,
            end_index=2,
        )
    ]

    report = simulate_pipeline_chaos(
        records,
        graph,
        faults,
        sla_budget_ms=0.001,  # Guaranteed breach
        enable_circuit_breaker=False,
    )
    assert report.sla_breaches > 0
    assert report.status == "SLA_BREACHED"


def test_simulate_pipeline_chaos_unhandled_failure() -> None:
    graph = DecisionGraph("chaos_fail_graph")
    graph.add_node(
        PipelineNode(
            name="required_action",
            stage_type=StageType.ACTION,
            processor=lambda _ctx: {"done": True},
            required=True,
        )
    )
    records = [{"id": 0}]
    faults = [
        StageFaultSpec(
            node_name="required_action",
            fault_type=StageFaultType.EXCEPTION,
            start_index=0,
            end_index=1,
            error_message="fatal action crash",
        )
    ]
    report = simulate_pipeline_chaos(records, graph, faults, enable_circuit_breaker=False)
    assert report.unhandled_errors == 1
    assert report.status == "FAILED"


def test_simulate_pipeline_chaos_edge_cases_and_validation() -> None:
    graph = DecisionGraph("chaos_edge_graph")
    graph.add_node(
        PipelineNode(
            name="step",
            stage_type=StageType.ACTION,
            processor=lambda _ctx: {"ok": True},
            required=True,
        )
    )

    with pytest.raises(ValueError, match="sla_budget_ms"):
        simulate_pipeline_chaos([{}], graph, [], sla_budget_ms=-5.0)

    with pytest.raises(TypeError, match="pipeline must be"):
        simulate_pipeline_chaos([{}], "invalid", [])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="fault_spec must be"):
        simulate_pipeline_chaos([{}], graph, ["not_a_spec"])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="does not exist"):
        simulate_pipeline_chaos(
            [{}],
            graph,
            [StageFaultSpec(node_name="unknown", fault_type=StageFaultType.EXCEPTION)],
        )

    # Empty records
    empty_rep = simulate_pipeline_chaos([], graph, [])
    assert empty_rep.total_transactions == 0
    assert empty_rep.latencies_p50_ms == 0.0
    assert empty_rep.status == "RESILIENT"

    # Dict format for fault_specs
    dict_spec = {
        "node_name": "step",
        "fault_type": "latency_spike",
        "start_index": 0,
        "end_index": 1,
        "latency_ms": 1.0,
    }
    rep_dict = simulate_pipeline_chaos([{}], graph, [dict_spec])
    assert rep_dict.total_transactions == 1
    assert rep_dict.status in ("RESILIENT", "DEGRADED", "SLA_BREACHED")



