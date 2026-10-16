"""Unit tests for real-time entity link graph, features, and fraud ring detection."""

import pytest

from fraud_detection.graph import (
    EntityGraph,
    EntityNode,
    EntityType,
    GraphConfig,
    GraphEdge,
    GraphError,
    GraphMetrics,
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
