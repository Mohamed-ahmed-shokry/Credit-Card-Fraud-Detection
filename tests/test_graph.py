"""Unit tests for real-time entity link graph, features, and fraud ring detection."""

import pytest

from fraud_detection.graph import (
    EntityNode,
    EntityType,
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
