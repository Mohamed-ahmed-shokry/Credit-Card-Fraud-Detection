from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from fraud_detection.model import FraudModel
from fraud_detection.routing import (
    CanaryConfig,
    CanaryStatus,
    ChampionChallengerRouter,
    DivergenceReport,
    ModelRole,
    RouteDecision,
    RoutingError,
    RoutingMetricsTracker,
    RoutingPolicy,
    TrafficSplitStrategy,
    compute_entity_hash_bucket,
    evaluate_model_divergence,
    resolve_entity_key_value,
)


def test_enums() -> None:
    assert TrafficSplitStrategy.CHAMPION_ONLY == "champion_only"
    assert TrafficSplitStrategy.CHALLENGER_ONLY == "challenger_only"
    assert TrafficSplitStrategy.HASH == "hash"
    assert TrafficSplitStrategy.PERCENTAGE == "percentage"
    assert TrafficSplitStrategy.SHADOW == "shadow"
    assert TrafficSplitStrategy.CANARY == "canary"

    assert ModelRole.CHAMPION == "champion"
    assert ModelRole.CHALLENGER == "challenger"

    assert CanaryStatus.HEALTHY == "healthy"
    assert CanaryStatus.DEGRADED == "degraded"
    assert CanaryStatus.ROLLED_BACK == "rolled_back"


def test_route_decision_properties() -> None:
    rd = RouteDecision(
        role=ModelRole.CHAMPION,
        model_version="v1",
        reason="test",
        is_shadow_candidate=True,
        entity_key_value="card_1",
        hash_bucket=0.25,
    )
    assert rd.role == ModelRole.CHAMPION
    assert rd.model_version == "v1"
    assert rd.reason == "test"
    assert rd.is_shadow_candidate is True
    assert rd.entity_key_value == "card_1"
    assert rd.hash_bucket == 0.25


def test_canary_config_validation_and_serialization() -> None:
    cfg = CanaryConfig()
    assert cfg.max_discrepancy_rate == 0.15
    assert cfg.max_probability_divergence == 0.25
    assert cfg.min_evaluations == 10
    assert cfg.auto_rollback is True

    # Valid custom
    custom = CanaryConfig(
        max_discrepancy_rate=0.05,
        max_probability_divergence=0.10,
        min_evaluations=5,
        auto_rollback=False,
    )
    d = custom.to_dict()
    assert d["max_discrepancy_rate"] == 0.05
    assert d["max_probability_divergence"] == 0.10
    assert d["min_evaluations"] == 5
    assert d["auto_rollback"] is False

    restored = CanaryConfig.from_dict(d)
    assert restored == custom

    # Invalid validations
    with pytest.raises(RoutingError, match="max_discrepancy_rate"):
        CanaryConfig(max_discrepancy_rate=-0.1)
    with pytest.raises(RoutingError, match="max_discrepancy_rate"):
        CanaryConfig(max_discrepancy_rate=1.5)
    with pytest.raises(RoutingError, match="max_probability_divergence"):
        CanaryConfig(max_probability_divergence=-0.01)
    with pytest.raises(RoutingError, match="max_probability_divergence"):
        CanaryConfig(max_probability_divergence=2.0)
    with pytest.raises(RoutingError, match="min_evaluations"):
        CanaryConfig(min_evaluations=0)


def test_routing_policy_validation_and_serialization() -> None:
    policy = RoutingPolicy()
    assert policy.strategy == TrafficSplitStrategy.CHAMPION_ONLY
    assert policy.challenger_weight == 0.0
    assert policy.entity_key == "card_id"
    assert policy.shadow_challenger is False
    assert policy.salt == "fraud_routing_v1"

    # From string strategy
    p2 = RoutingPolicy(strategy="canary", challenger_weight=0.1)  # type: ignore[arg-type]
    assert p2.strategy == TrafficSplitStrategy.CANARY
    assert p2.challenger_weight == 0.1

    # Serialization roundtrip
    d = p2.to_dict()
    assert d["strategy"] == "canary"
    assert d["challenger_weight"] == 0.1
    restored = RoutingPolicy.from_dict(d)
    assert restored.strategy == TrafficSplitStrategy.CANARY
    assert restored.challenger_weight == 0.1

    # Validation errors
    with pytest.raises(RoutingError, match="Unsupported routing strategy"):
        RoutingPolicy(strategy="invalid_strat")  # type: ignore[arg-type]
    with pytest.raises(RoutingError, match="challenger_weight"):
        RoutingPolicy(challenger_weight=-0.1)
    with pytest.raises(RoutingError, match="challenger_weight"):
        RoutingPolicy(challenger_weight=1.2)
    with pytest.raises(RoutingError, match="entity_key"):
        RoutingPolicy(entity_key="")
    with pytest.raises(RoutingError, match="salt"):
        RoutingPolicy(salt="")


def test_routing_metrics_tracker_and_safeguards() -> None:
    tracker = RoutingMetricsTracker()
    tracker.record_route(ModelRole.CHAMPION, 5)
    tracker.record_route(ModelRole.CHALLENGER, 3)

    summary = tracker.get_summary()
    assert summary["champion_requests"] == 5
    assert summary["challenger_requests"] == 3
    assert summary["total_routed"] == 8
    assert summary["canary_status"] == "healthy"

    canary_cfg = CanaryConfig(
        max_discrepancy_rate=0.2,
        max_probability_divergence=0.3,
        min_evaluations=4,
        auto_rollback=True,
    )

    # Record comparisons below min_evaluations
    flip1 = tracker.record_comparison(0.8, True, 0.2, False, canary_cfg)
    assert flip1 is True
    assert tracker.canary_status == CanaryStatus.HEALTHY

    flip2 = tracker.record_comparison(0.7, True, 0.75, True, canary_cfg)
    assert flip2 is False
    assert tracker.canary_status == CanaryStatus.HEALTHY

    # Batch comparison pushing over threshold
    # 2 evaluations: both flips, large divergence
    batch_flips = tracker.record_batch_comparison(
        champion_probs=[0.9, 0.85],
        champion_decs=[True, True],
        challenger_probs=[0.1, 0.15],
        challenger_decs=[False, False],
        canary_config=canary_cfg,
    )
    assert batch_flips == 2

    # Now total evals = 4, discrepancies = 3 / 4 = 0.75 > 0.2 -> ROLLED_BACK!
    assert tracker.canary_status == CanaryStatus.ROLLED_BACK
    assert tracker.rollback_reason is not None
    assert "Canary safeguard tripped" in tracker.rollback_reason

    # Further comparisons do not change ROLLED_BACK state
    tracker.record_comparison(0.5, False, 0.5, False, canary_cfg)
    assert tracker.canary_status == CanaryStatus.ROLLED_BACK

    # Reset canary
    tracker.reset_canary()
    assert tracker.canary_status == CanaryStatus.HEALTHY
    assert tracker.rollback_reason is None
    assert tracker.discrepancy_count == 0

    # Manual rollback
    tracker.trigger_rollback("Manual intervention")
    assert tracker.canary_status == CanaryStatus.ROLLED_BACK
    assert tracker.rollback_reason == "Manual intervention"

    # Batch mismatch error
    with pytest.raises(RoutingError, match="must match"):
        tracker.record_batch_comparison([0.1], [False], [0.1, 0.2], [False, True])


def test_routing_metrics_degraded_when_auto_rollback_disabled() -> None:
    tracker = RoutingMetricsTracker()
    canary_cfg = CanaryConfig(
        max_discrepancy_rate=0.1,
        max_probability_divergence=0.2,
        min_evaluations=2,
        auto_rollback=False,
    )

    tracker.record_comparison(0.9, True, 0.1, False, canary_cfg)
    tracker.record_comparison(0.95, True, 0.05, False, canary_cfg)

    assert tracker.canary_status == CanaryStatus.DEGRADED
    assert tracker.rollback_reason is not None
    assert "Canary divergence warning" in tracker.rollback_reason


def test_entity_hash_bucket_and_resolution() -> None:
    # Determinism
    b1 = compute_entity_hash_bucket("user_123", salt="saltA")
    b2 = compute_entity_hash_bucket("user_123", salt="saltA")
    assert b1 == b2
    assert 0.0 <= b1 < 1.0

    b3 = compute_entity_hash_bucket("user_456", salt="saltA")
    assert b1 != b3

    # Salt changes distribution
    b_diff_salt = compute_entity_hash_bucket("user_123", salt="saltB")
    assert b1 != b_diff_salt

    # Key resolution
    rec = {"card_id": "card_999", "amount": 42.5}
    val, is_explicit = resolve_entity_key_value(rec, "card_id")
    assert val == "card_999"
    assert is_explicit is True

    # None or whitespace key value
    rec_none = {"card_id": None, "amount": 10.0}
    val_none, is_exp_none = resolve_entity_key_value(rec_none, "card_id")
    assert is_exp_none is False
    assert "amount=10.0" in val_none

    rec_ws = {"card_id": "   ", "amount": 10.0}
    _val_ws, is_exp_ws = resolve_entity_key_value(rec_ws, "card_id")
    assert is_exp_ws is False

    # Missing key fallback
    rec_missing = {"amount": 42.5, "v1": 0.12, "nested": {"a": 1}}
    val_fb, is_explicit_fb = resolve_entity_key_value(rec_missing, "card_id")
    assert "amount=42.5" in val_fb
    assert "v1=0.12" in val_fb
    assert is_explicit_fb is False

    # Empty record fallback
    val_empty, _ = resolve_entity_key_value({}, "card_id")
    assert val_empty == "empty_record"


def test_tracker_branch_coverage() -> None:
    tracker = RoutingMetricsTracker()
    # 1. Smaller diff after larger diff
    tracker.record_comparison(0.9, False, 0.1, False)  # diff = 0.8
    assert tracker.max_probability_diff == 0.8
    tracker.record_comparison(0.5, False, 0.4, False)  # diff = 0.1 < 0.8
    assert tracker.max_probability_diff == 0.8

    # 2. Batch with smaller diffs
    tracker.record_batch_comparison(
        champion_probs=[0.6, 0.55],
        champion_decs=[False, False],
        challenger_probs=[0.5, 0.5],
        challenger_decs=[False, False],
    )
    assert tracker.max_probability_diff == 0.8

    # 3. Safeguard: divergence only breached (flip_rate below max)
    tracker2 = RoutingMetricsTracker()
    canary_cfg = CanaryConfig(
        max_discrepancy_rate=0.5,  # High flip tolerance
        max_probability_divergence=0.1,  # Low divergence tolerance
        min_evaluations=2,
        auto_rollback=True,
    )
    tracker2.record_comparison(0.8, True, 0.55, True, canary_cfg)  # diff 0.25, no flip
    tracker2.record_comparison(0.8, True, 0.55, True, canary_cfg)  # diff 0.25, no flip
    assert tracker2.canary_status == CanaryStatus.ROLLED_BACK
    assert "mean_divergence" in str(tracker2.rollback_reason)
    assert "flip_rate" not in str(tracker2.rollback_reason)


def test_champion_challenger_router_routing_decisions() -> None:
    champ = MagicMock(spec=FraudModel)
    champ.metadata = {"dataset_fingerprint": "champ_fingerprint_123", "version": "v1.0.0"}
    champ.threshold = 0.5

    chall = MagicMock(spec=FraudModel)
    chall.metadata = {"dataset_fingerprint": "chall_fingerprint_456", "version": "v2.0.0"}
    chall.threshold = 0.6

    # 1. No challenger model configured
    router_single = ChampionChallengerRouter(champion_model=champ)
    assert router_single.champion_version == "v1.0.0"
    assert router_single.challenger_version is None

    decision = router_single.route_record({"card_id": "c1"})
    assert decision.role == ModelRole.CHAMPION
    assert decision.model_version == "v1.0.0"
    assert decision.reason == "no_challenger_configured"
    assert decision.is_shadow_candidate is False

    # 2. Strategy: CHAMPION_ONLY with shadow_challenger=True
    policy_champ_only = RoutingPolicy(
        strategy=TrafficSplitStrategy.CHAMPION_ONLY,
        shadow_challenger=True,
    )
    router_co = ChampionChallengerRouter(
        champion_model=champ, challenger_model=chall, policy=policy_champ_only
    )
    d_co = router_co.route_record({"card_id": "c1"})
    assert d_co.role == ModelRole.CHAMPION
    assert d_co.is_shadow_candidate is True

    # 3. Strategy: CHALLENGER_ONLY
    policy_chall_only = RoutingPolicy(strategy=TrafficSplitStrategy.CHALLENGER_ONLY)
    router_cho = ChampionChallengerRouter(
        champion_model=champ, challenger_model=chall, policy=policy_chall_only
    )
    d_cho = router_cho.route_record({"card_id": "c1"})
    assert d_cho.role == ModelRole.CHALLENGER
    assert d_cho.model_version == "v2.0.0"
    assert d_cho.is_shadow_candidate is False

    # 4. Strategy: SHADOW
    policy_shadow = RoutingPolicy(strategy=TrafficSplitStrategy.SHADOW)
    router_sh = ChampionChallengerRouter(
        champion_model=champ, challenger_model=chall, policy=policy_shadow
    )
    d_sh = router_sh.route_record({"card_id": "c1"})
    assert d_sh.role == ModelRole.CHAMPION
    assert d_sh.is_shadow_candidate is True
    assert d_sh.reason == "shadow_mirror_policy"

    # 5. Strategy: HASH deterministic sticky partitioning
    policy_hash = RoutingPolicy(
        strategy=TrafficSplitStrategy.HASH,
        challenger_weight=0.5,
        entity_key="card_id",
        salt="test_salt",
    )
    router_h = ChampionChallengerRouter(
        champion_model=champ, challenger_model=chall, policy=policy_hash
    )

    card_a = {"card_id": "card_AAA"}
    card_b = {"card_id": "card_BBB"}

    # Repeated calls must yield identical decisions for the same entity
    d_a1 = router_h.route_record(card_a)
    d_a2 = router_h.route_record(card_a)
    assert d_a1.role == d_a2.role
    assert d_a1.hash_bucket == d_a2.hash_bucket

    # Batch routing
    batch_records = [card_a, card_b, card_a]
    batch_decisions = router_h.route_batch(batch_records)
    assert len(batch_decisions) == 3
    assert batch_decisions[0].role == batch_decisions[2].role

    # 6. Strategy: PERCENTAGE
    policy_pct = RoutingPolicy(
        strategy=TrafficSplitStrategy.PERCENTAGE,
        challenger_weight=1.0,  # 100% challenger
    )
    router_pct = ChampionChallengerRouter(
        champion_model=champ, challenger_model=chall, policy=policy_pct
    )
    d_pct = router_pct.route_record(card_a)
    assert d_pct.role == ModelRole.CHALLENGER

    policy_pct0 = RoutingPolicy(
        strategy=TrafficSplitStrategy.PERCENTAGE,
        challenger_weight=0.0,  # 0% challenger
        shadow_challenger=True,
    )
    router_pct0 = ChampionChallengerRouter(
        champion_model=champ, challenger_model=chall, policy=policy_pct0
    )
    d_pct0 = router_pct0.route_record(card_a)
    assert d_pct0.role == ModelRole.CHAMPION
    assert d_pct0.is_shadow_candidate is True

    # 7. Strategy: CANARY with Rollback Safeguard
    policy_canary = RoutingPolicy(
        strategy=TrafficSplitStrategy.CANARY,
        challenger_weight=0.99,  # Would normally route to challenger
    )
    tracker = RoutingMetricsTracker()
    router_canary = ChampionChallengerRouter(
        champion_model=champ,
        challenger_model=chall,
        policy=policy_canary,
        metrics_tracker=tracker,
    )
    # When healthy, routes to challenger
    d_canary_healthy = router_canary.route_record(card_a)
    assert d_canary_healthy.role == ModelRole.CHALLENGER

    # Trigger rollback
    tracker.trigger_rollback("High false positive surge")
    d_canary_rolled_back = router_canary.route_record(card_a)
    assert d_canary_rolled_back.role == ModelRole.CHAMPION
    assert d_canary_rolled_back.reason == "canary_rolled_back"
    assert d_canary_rolled_back.is_shadow_candidate is False


def test_evaluate_divergence_and_safeguards() -> None:
    champ = MagicMock(spec=FraudModel)
    champ.metadata = {"dataset_fingerprint": "c" * 32}
    champ.threshold = 0.5

    chall = MagicMock(spec=FraudModel)
    chall.metadata = {"dataset_fingerprint": "d" * 32}
    chall.threshold = 0.5

    router = ChampionChallengerRouter(champion_model=champ, challenger_model=chall)

    # Identical predictions -> perfect alignment
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    report_identical = router.evaluate_divergence(
        champion_probabilities=probs,
        challenger_probabilities=probs,
        champion_threshold=0.5,
        challenger_threshold=0.5,
    )
    assert report_identical.total_samples == 4
    assert report_identical.discrepancies == 0
    assert report_identical.flip_rate == 0.0
    assert report_identical.mean_probability_divergence == 0.0
    assert report_identical.safeguard_tripped is False
    assert report_identical.recommendation == "PROCEED_ROLLOUT"

    # Divergent predictions
    c_p = np.array([0.2, 0.3, 0.8, 0.9, 0.1, 0.2, 0.8, 0.9, 0.2, 0.8])
    ch_p = np.array([0.8, 0.7, 0.2, 0.1, 0.1, 0.2, 0.8, 0.9, 0.2, 0.8])
    report_div = router.evaluate_divergence(
        champion_probabilities=c_p,
        challenger_probabilities=ch_p,
        champion_threshold=0.5,
        challenger_threshold=0.5,
        canary_config=CanaryConfig(
            max_discrepancy_rate=0.2,
            max_probability_divergence=0.2,
            min_evaluations=5,
            auto_rollback=True,
        ),
    )
    assert report_div.total_samples == 10
    assert report_div.discrepancies == 4  # 4 flips
    assert report_div.flip_rate == 0.4
    assert report_div.safeguard_tripped is True
    assert report_div.canary_status == CanaryStatus.ROLLED_BACK
    assert report_div.recommendation == "ROLLBACK_RECOMMENDED"

    # Serialization roundtrip
    d = report_div.to_dict()
    assert d["flip_rate"] == 0.4
    restored = DivergenceReport.from_dict(d)
    assert restored.total_samples == report_div.total_samples
    assert restored.discrepancies == report_div.discrepancies
    assert restored.canary_status == CanaryStatus.ROLLED_BACK

    # Validation errors
    with pytest.raises(RoutingError, match="cannot be empty"):
        router.evaluate_divergence(np.array([]), np.array([]), 0.5, 0.5)

    with pytest.raises(RoutingError, match="Shape mismatch"):
        router.evaluate_divergence(np.array([0.1]), np.array([0.1, 0.2]), 0.5, 0.5)


def test_evaluate_model_divergence_dataframe() -> None:
    champ = MagicMock(spec=FraudModel)
    champ.metadata = {"dataset_fingerprint": "c" * 32}
    champ.threshold = 0.5
    champ.predict_probabilities.return_value = np.array([0.1, 0.8])

    chall = MagicMock(spec=FraudModel)
    chall.metadata = {"dataset_fingerprint": "d" * 32}
    chall.threshold = 0.5
    chall.predict_probabilities.return_value = np.array([0.2, 0.7])

    df = pd.DataFrame({"V1": [1.0, 2.0], "Amount": [10.0, 20.0]})
    report = evaluate_model_divergence(champ, chall, df)
    assert report.total_samples == 2
    assert report.discrepancies == 0

    # Empty df raises
    with pytest.raises(RoutingError, match="empty dataset"):
        evaluate_model_divergence(champ, chall, pd.DataFrame())
