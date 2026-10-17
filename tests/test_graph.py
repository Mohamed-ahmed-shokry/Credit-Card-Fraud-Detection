"""Unit tests for real-time entity link graph, features, and fraud ring detection."""

import pytest

from fraud_detection.graph import (
    EntityGraph,
    EntityNode,
    EntityType,
    FraudRingCluster,
    FraudRingDetectorConfig,
    FraudRingReport,
    GraphConfig,
    GraphEdge,
    GraphError,
    GraphFeatureConfig,
    GraphFeatureExtractor,
    GraphMetrics,
    RingAction,
    RingPatternType,
    RingSeverity,
    TransactionNode,
)


def test_entity_type_enum_values() -> None:
    assert EntityType.CARD_ID == "card_id"
    assert EntityType.DEVICE_ID == "device_id"
    assert EntityType.IP_ADDRESS == "ip_address"
    assert EntityType.MERCHANT_ID == "merchant_id"
    assert EntityType.USER_ID == "user_id"


def test_entity_node_valid() -> None:
    node = EntityNode(
        entity_type=EntityType.CARD_ID,
        entity_id="card_1234",
        first_seen_timestamp=100.0,
        last_seen_timestamp=200.0,
        risk_score=0.45,
    )
    assert node.entity_id == "card_1234"
    assert node.entity_type == EntityType.CARD_ID
    assert node.first_seen_timestamp == 100.0
    assert node.last_seen_timestamp == 200.0
    assert node.risk_score == 0.45


def test_entity_node_invalid_validations() -> None:
    with pytest.raises(GraphError, match="entity_id must be a non-empty string"):
        EntityNode(
            entity_type=EntityType.CARD_ID,
            entity_id="",
            first_seen_timestamp=100.0,
            last_seen_timestamp=200.0,
        )

    with pytest.raises(GraphError, match="first_seen_timestamp must be a finite float"):
        EntityNode(
            entity_type=EntityType.CARD_ID,
            entity_id="c1",
            first_seen_timestamp=float("nan"),
            last_seen_timestamp=200.0,
        )

    with pytest.raises(GraphError, match="last_seen_timestamp must be >= first_seen_timestamp"):
        EntityNode(
            entity_type=EntityType.CARD_ID,
            entity_id="c1",
            first_seen_timestamp=200.0,
            last_seen_timestamp=100.0,
        )

    with pytest.raises(GraphError, match="risk_score must be a finite float"):
        EntityNode(
            entity_type=EntityType.CARD_ID,
            entity_id="c1",
            first_seen_timestamp=100.0,
            last_seen_timestamp=200.0,
            risk_score=1.5,
        )


def test_transaction_node_valid_and_invalid() -> None:
    tx = TransactionNode(
        transaction_id="tx_001",
        timestamp=1000.0,
        amount=50.25,
        entities={EntityType.CARD_ID: "c1", EntityType.DEVICE_ID: "d1"},
        is_fraud=False,
        risk_score=0.1,
    )
    assert tx.transaction_id == "tx_001"
    assert tx.amount == 50.25
    assert tx.entities[EntityType.CARD_ID] == "c1"

    with pytest.raises(GraphError, match="transaction_id must be a non-empty string"):
        TransactionNode(transaction_id="", timestamp=100.0, amount=10.0)

    with pytest.raises(GraphError, match="amount must be a non-negative finite float"):
        TransactionNode(transaction_id="tx_002", timestamp=100.0, amount=-5.0)

    with pytest.raises(GraphError, match="risk_score must be a finite float"):
        TransactionNode(transaction_id="tx_002", timestamp=100.0, amount=5.0, risk_score=-0.1)


def test_graph_edge_valid_and_invalid() -> None:
    edge = GraphEdge(
        transaction_id="tx_1",
        entity_type=EntityType.IP_ADDRESS,
        entity_id="192.168.1.1",
        timestamp=500.0,
        weight=2.0,
    )
    assert edge.transaction_id == "tx_1"
    assert edge.weight == 2.0

    with pytest.raises(GraphError, match="weight must be a positive finite float"):
        GraphEdge(
            transaction_id="tx_1",
            entity_type=EntityType.IP_ADDRESS,
            entity_id="ip_1",
            timestamp=500.0,
            weight=0.0,
        )


def test_graph_metrics() -> None:
    metrics = GraphMetrics(
        total_transactions=10,
        total_entities=5,
        total_edges=20,
        entities_by_type={"card_id": 3, "device_id": 2},
        window_seconds=3600.0,
        oldest_timestamp=100.0,
        newest_timestamp=500.0,
    )
    assert metrics.total_transactions == 10
    assert metrics.entities_by_type["card_id"] == 3


def test_graph_config_validation() -> None:
    config = GraphConfig(window_seconds=3600.0, max_transactions=1000)
    assert config.window_seconds == 3600.0
    assert config.max_transactions == 1000

    with pytest.raises(GraphError, match="window_seconds must be a positive finite float"):
        GraphConfig(window_seconds=-10.0)

    with pytest.raises(GraphError, match="max_transactions must be a positive integer"):
        GraphConfig(max_transactions=0)

    with pytest.raises(GraphError, match="max_entities must be a positive integer"):
        GraphConfig(max_entities=-5)

    with pytest.raises(
        GraphError, match="prune_interval_seconds must be a non-negative finite float"
    ):
        GraphConfig(prune_interval_seconds=-1.0)


def test_entity_graph_add_and_query() -> None:
    graph = EntityGraph()
    tx = graph.add_transaction(
        transaction_id="tx_101",
        timestamp=100.0,
        amount=150.0,
        entities={
            EntityType.CARD_ID: "card_A",
            EntityType.DEVICE_ID: "dev_1",
            EntityType.IP_ADDRESS: "10.0.0.1",
        },
        is_fraud=False,
        risk_score=0.2,
    )
    assert tx.transaction_id == "tx_101"
    assert graph.get_transaction("tx_101") == tx

    # Check entity indexing
    card_node = graph.get_entity(EntityType.CARD_ID, "card_A")
    assert card_node is not None
    assert card_node.first_seen_timestamp == 100.0
    assert card_node.risk_score == 0.2

    # Query transaction entities
    entities = graph.get_transaction_entities("tx_101")
    assert len(entities) == 3
    assert entities[EntityType.DEVICE_ID].entity_id == "dev_1"

    # Query entity transactions
    txs = graph.get_entity_transactions(EntityType.DEVICE_ID, "dev_1")
    assert len(txs) == 1
    assert txs[0].transaction_id == "tx_101"


def test_entity_graph_risk_propagation() -> None:
    graph = EntityGraph()
    graph.add_transaction(
        transaction_id="tx_normal",
        timestamp=100.0,
        amount=20.0,
        entities={EntityType.CARD_ID: "card_B", EntityType.DEVICE_ID: "dev_2"},
        is_fraud=False,
        risk_score=0.1,
    )
    dev = graph.get_entity(EntityType.DEVICE_ID, "dev_2")
    assert dev is not None
    assert dev.risk_score == 0.1

    # Now add fraud transaction on same device
    graph.add_transaction(
        transaction_id="tx_fraud",
        timestamp=200.0,
        amount=500.0,
        entities={EntityType.CARD_ID: "card_C", EntityType.DEVICE_ID: "dev_2"},
        is_fraud=True,
    )
    dev = graph.get_entity(EntityType.DEVICE_ID, "dev_2")
    assert dev is not None
    assert dev.risk_score == 1.0


def test_entity_graph_2hop_neighbors() -> None:
    graph = EntityGraph()
    # tx1 links card1 and device1
    graph.add_transaction(
        transaction_id="tx_1",
        timestamp=10.0,
        amount=30.0,
        entities={EntityType.CARD_ID: "card_1", EntityType.DEVICE_ID: "dev_shared"},
    )
    # tx2 links card2 and device1
    graph.add_transaction(
        transaction_id="tx_2",
        timestamp=20.0,
        amount=40.0,
        entities={EntityType.CARD_ID: "card_2", EntityType.DEVICE_ID: "dev_shared"},
    )
    # tx3 links card3 and device_other
    graph.add_transaction(
        transaction_id="tx_3",
        timestamp=30.0,
        amount=50.0,
        entities={EntityType.CARD_ID: "card_3", EntityType.DEVICE_ID: "dev_other"},
    )

    # 2-hop transactions from tx_1 (shares dev_shared with tx_2)
    peer_txs = graph.get_2hop_transactions("tx_1")
    assert len(peer_txs) == 1
    # Co-occurring entities from card_1 (directly in tx_1: dev_shared)
    co_entities = graph.get_cooccurring_entities(EntityType.CARD_ID, "card_1")
    assert len(co_entities) == 1
    assert co_entities[0].entity_id == "dev_shared"

    # 2-hop entities from card_1 (connected to card_2 via shared dev_shared)
    peer_cards = graph.get_2hop_entities(
        EntityType.CARD_ID, "card_1", target_type=EntityType.CARD_ID
    )
    assert len(peer_cards) == 1
    assert peer_cards[0].entity_id == "card_2"


def test_entity_graph_temporal_eviction_and_metrics() -> None:
    config = GraphConfig(window_seconds=100.0, prune_interval_seconds=1000.0)
    graph = EntityGraph(config=config)

    graph.add_transaction(
        transaction_id="tx_old",
        timestamp=50.0,
        amount=10.0,
        entities={EntityType.CARD_ID: "card_old"},
    )
    graph.add_transaction(
        transaction_id="tx_new",
        timestamp=200.0,
        amount=25.0,
        entities={EntityType.CARD_ID: "card_new"},
    )

    metrics_before = graph.get_metrics()
    assert metrics_before.total_transactions == 2
    assert metrics_before.total_entities == 2

    # Evict relative to ts=200 (cutoff = 100). tx_old (ts=50) should be evicted.
    evicted = graph.evict_expired(current_timestamp=200.0)
    assert evicted == 1
    assert graph.get_transaction("tx_old") is None
    assert graph.get_entity(EntityType.CARD_ID, "card_old") is None
    assert graph.get_transaction("tx_new") is not None

    metrics_after = graph.get_metrics()
    assert metrics_after.total_transactions == 1
    assert metrics_after.total_entities == 1


def test_entity_graph_capacity_and_reset() -> None:
    config = GraphConfig(max_transactions=2)
    graph = EntityGraph(config=config)

    graph.add_transaction(
        transaction_id="t1", timestamp=1.0, amount=1.0, entities={EntityType.CARD_ID: "c1"}
    )
    graph.add_transaction(
        transaction_id="t2", timestamp=2.0, amount=2.0, entities={EntityType.CARD_ID: "c2"}
    )
    graph.add_transaction(
        transaction_id="t3", timestamp=3.0, amount=3.0, entities={EntityType.CARD_ID: "c3"}
    )

    metrics_cap = graph.get_metrics()
    assert metrics_cap.total_transactions == 2
    assert graph.get_transaction("t1") is None
    assert graph.get_transaction("t2") is not None
    assert graph.get_transaction("t3") is not None

    graph.reset()
    metrics_empty = graph.get_metrics()
    assert metrics_empty.total_transactions == 0
    assert metrics_empty.total_entities == 0


def test_graph_feature_config_validation() -> None:
    cfg = GraphFeatureConfig(windows_seconds=(1800.0, 7200.0), max_hops=3, diffusion_damping=0.8)
    assert cfg.windows_seconds == (1800.0, 7200.0)
    assert cfg.max_hops == 3
    assert cfg.diffusion_damping == 0.8
    assert cfg.diffusion_iterations == 5

    with pytest.raises(
        GraphError, match="windows_seconds must be a non-empty sequence of positive floats"
    ):
        GraphFeatureConfig(windows_seconds=())

    with pytest.raises(
        GraphError, match="Each window in windows_seconds must be a positive finite float"
    ):
        GraphFeatureConfig(windows_seconds=(-100.0,))

    with pytest.raises(GraphError, match="max_hops must be an integer between 1 and 5"):
        GraphFeatureConfig(max_hops=10)

    with pytest.raises(
        GraphError, match=r"diffusion_damping must be a float between 0\.0 and 1\.0"
    ):
        GraphFeatureConfig(diffusion_damping=1.5)

    with pytest.raises(
        GraphError, match="diffusion_iterations must be an integer between 1 and 20"
    ):
        GraphFeatureConfig(diffusion_iterations=0)


def test_graph_feature_extractor_degree_and_sharing() -> None:
    graph = EntityGraph()
    # Add historical transaction 1
    graph.add_transaction(
        transaction_id="tx_h1",
        timestamp=1000.0,
        amount=100.0,
        entities={
            EntityType.CARD_ID: "card_1",
            EntityType.DEVICE_ID: "dev_1",
            EntityType.IP_ADDRESS: "1.2.3.4",
        },
        risk_score=0.1,
    )
    # Add historical transaction 2 on same device with different card
    graph.add_transaction(
        transaction_id="tx_h2",
        timestamp=1200.0,
        amount=200.0,
        entities={
            EntityType.CARD_ID: "card_2",
            EntityType.DEVICE_ID: "dev_1",
            EntityType.IP_ADDRESS: "5.6.7.8",
        },
        risk_score=0.9,
    )

    extractor = GraphFeatureExtractor(
        graph,
        config=GraphFeatureConfig(windows_seconds=(3600.0,), diffusion_damping=0.5),
    )

    # Candidate transaction using card_1 on dev_1 at timestamp 1500.0
    features = extractor.extract_features(
        timestamp=1500.0,
        entities={
            EntityType.CARD_ID: "card_1",
            EntityType.DEVICE_ID: "dev_1",
        },
    )

    # Device dev_1 was used by 2 distinct cards in the last 1h (card_1, card_2)
    assert features["graph_device_id_tx_count_3600s"] == 2.0
    assert features["graph_device_id_monetary_sum_3600s"] == 300.0
    assert features["graph_device_id_distinct_cards_3600s"] == 2.0

    # Card_1 had 1 transaction in the last 1h
    assert features["graph_card_id_tx_count_3600s"] == 1.0
    assert features["graph_card_id_monetary_sum_3600s"] == 100.0

    # 2-hop shared cards via dev_1: card_1 is connected to card_2
    assert features["graph_shared_device_cards_count"] == 1.0

    # Risk diffusion: dev_1 risk is 0.9 (from card_2 tx_h2).
    # Since dev_1 is in candidate entities, its risk (0.9) propagates.
    assert features["graph_risk_diffusion"] >= 0.9


def test_graph_feature_extractor_temporal_causality() -> None:
    graph = EntityGraph()
    # Past transaction at t=500
    graph.add_transaction(
        transaction_id="tx_past",
        timestamp=500.0,
        amount=50.0,
        entities={EntityType.CARD_ID: "card_X"},
    )
    # Future transaction at t=1500
    graph.add_transaction(
        transaction_id="tx_future",
        timestamp=1500.0,
        amount=500.0,
        entities={EntityType.CARD_ID: "card_X"},
    )

    extractor = GraphFeatureExtractor(
        graph,
        config=GraphFeatureConfig(windows_seconds=(3600.0,)),
    )

    # Evaluate at t=1000: future transaction at t=1500 MUST NOT leak!
    features = extractor.extract_features(
        timestamp=1000.0,
        entities={EntityType.CARD_ID: "card_X"},
    )
    assert features["graph_card_id_tx_count_3600s"] == 1.0
    assert features["graph_card_id_monetary_sum_3600s"] == 50.0

    # Features for empty entities
    empty_features = extractor.extract_features(timestamp=1000.0, entities=None)
    assert empty_features["graph_shared_device_cards_count"] == 0.0
    assert empty_features["graph_risk_diffusion"] == 0.0


def test_fraud_ring_enums_and_cluster_validation() -> None:
    assert RingSeverity.LOW == "low"
    assert RingSeverity.CRITICAL == "critical"
    assert RingAction.ALLOW == "allow"
    assert RingAction.DENY == "deny"
    assert RingPatternType.BICLIQUE == "biclique"
    assert RingPatternType.STAR_CARD_TESTING == "star_card_testing"
    assert RingPatternType.CYCLE == "cycle"

    cluster = FraudRingCluster(
        cluster_id="ring_1",
        pattern_type=RingPatternType.BICLIQUE,
        severity=RingSeverity.HIGH,
        recommended_action=RingAction.DENY,
        entities={EntityType.CARD_ID: ("c1", "c2"), EntityType.DEVICE_ID: ("d1",)},
        transaction_ids=("tx_1", "tx_2"),
        transaction_count=2,
        total_amount=500.0,
        density=0.8,
        risk_score=0.9,
        description="Dense card-device cluster",
    )
    assert cluster.cluster_id == "ring_1"
    assert cluster.severity == RingSeverity.HIGH
    assert cluster.recommended_action == RingAction.DENY

    with pytest.raises(GraphError, match="cluster_id must be a non-empty string"):
        FraudRingCluster(
            cluster_id="",
            pattern_type=RingPatternType.BICLIQUE,
            severity=RingSeverity.HIGH,
            recommended_action=RingAction.DENY,
            entities={},
            transaction_ids=(),
            transaction_count=0,
            total_amount=0.0,
            density=0.0,
            risk_score=0.0,
            description="",
        )

    with pytest.raises(GraphError, match="Invalid pattern_type"):
        FraudRingCluster(
            cluster_id="r",
            pattern_type="unknown",  # type: ignore[arg-type]
            severity=RingSeverity.HIGH,
            recommended_action=RingAction.DENY,
            entities={},
            transaction_ids=(),
            transaction_count=0,
            total_amount=0.0,
            density=0.0,
            risk_score=0.0,
            description="",
        )


def test_fraud_ring_report_validation() -> None:
    report = FraudRingReport(
        generated_at=1000.0,
        total_clusters=0,
        clusters=(),
        critical_count=0,
        high_count=0,
        medium_count=0,
        low_count=0,
    )
    assert report.total_clusters == 0

    with pytest.raises(GraphError, match="generated_at must be a finite float"):
        FraudRingReport(
            generated_at=float("nan"),
            total_clusters=0,
            clusters=(),
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
        )


def test_fraud_ring_detector_config_validation() -> None:
    cfg = FraudRingDetectorConfig()
    assert cfg.min_cards_per_device == 3
    assert cfg.min_devices_per_card == 2

    with pytest.raises(GraphError, match="min_cards_per_device must be an integer >= 2"):
        FraudRingDetectorConfig(min_cards_per_device=1)

    with pytest.raises(GraphError, match="min_devices_per_card must be an integer >= 1"):
        FraudRingDetectorConfig(min_devices_per_card=0)

    with pytest.raises(
        GraphError, match="card_testing_window_seconds must be a positive finite float"
    ):
        FraudRingDetectorConfig(card_testing_window_seconds=0.0)

    with pytest.raises(GraphError, match="card_testing_min_cards must be an integer >= 2"):
        FraudRingDetectorConfig(card_testing_min_cards=1)

    with pytest.raises(GraphError, match="cycle_max_depth must be an integer between 2 and 10"):
        FraudRingDetectorConfig(cycle_max_depth=20)
