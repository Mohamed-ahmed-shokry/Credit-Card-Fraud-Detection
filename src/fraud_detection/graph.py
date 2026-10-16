"""Real-time entity link analysis, fraud ring graph detection, and graph feature aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum


class GraphError(ValueError):
    """Raised when graph data, entity configuration, or ring detection fails."""


class EntityType(StrEnum):
    """Types of entities linked to transactions in the bipartite link graph."""

    CARD_ID = "card_id"
    DEVICE_ID = "device_id"
    IP_ADDRESS = "ip_address"
    MERCHANT_ID = "merchant_id"
    USER_ID = "user_id"


@dataclass(frozen=True)
class EntityNode:
    """Represents a unique entity vertex in the link graph."""

    entity_type: EntityType
    entity_id: str
    first_seen_timestamp: float
    last_seen_timestamp: float
    risk_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.entity_id or not isinstance(self.entity_id, str):
            raise GraphError("entity_id must be a non-empty string.")
        if not isinstance(self.entity_type, EntityType):
            raise GraphError(f"Invalid entity_type: {self.entity_type}")
        if (
            not isinstance(self.first_seen_timestamp, (int, float))
            or math.isnan(self.first_seen_timestamp)
            or math.isinf(self.first_seen_timestamp)
        ):
            raise GraphError("first_seen_timestamp must be a finite float.")
        if (
            not isinstance(self.last_seen_timestamp, (int, float))
            or math.isnan(self.last_seen_timestamp)
            or math.isinf(self.last_seen_timestamp)
            or self.last_seen_timestamp < self.first_seen_timestamp
        ):
            raise GraphError("last_seen_timestamp must be >= first_seen_timestamp.")
        if (
            not isinstance(self.risk_score, (int, float))
            or math.isnan(self.risk_score)
            or math.isinf(self.risk_score)
            or not (0.0 <= self.risk_score <= 1.0)
        ):
            raise GraphError("risk_score must be a finite float between 0.0 and 1.0.")


@dataclass(frozen=True)
class TransactionNode:
    """Represents a transaction vertex in the bipartite link graph."""

    transaction_id: str
    timestamp: float
    amount: float
    entities: dict[EntityType, str] = field(default_factory=dict)
    is_fraud: bool | None = None
    risk_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.transaction_id or not isinstance(self.transaction_id, str):
            raise GraphError("transaction_id must be a non-empty string.")
        if (
            not isinstance(self.timestamp, (int, float))
            or math.isnan(self.timestamp)
            or math.isinf(self.timestamp)
        ):
            raise GraphError("timestamp must be a finite float.")
        if (
            not isinstance(self.amount, (int, float))
            or math.isnan(self.amount)
            or math.isinf(self.amount)
            or self.amount < 0.0
        ):
            raise GraphError("amount must be a non-negative finite float.")
        if not isinstance(self.entities, dict):
            raise GraphError("entities must be a dictionary.")
        if (
            not isinstance(self.risk_score, (int, float))
            or math.isnan(self.risk_score)
            or math.isinf(self.risk_score)
            or not (0.0 <= self.risk_score <= 1.0)
        ):
            raise GraphError("risk_score must be a finite float between 0.0 and 1.0.")


@dataclass(frozen=True)
class GraphEdge:
    """Directed or undirected edge between a transaction and an entity."""

    transaction_id: str
    entity_type: EntityType
    entity_id: str
    timestamp: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.transaction_id:
            raise GraphError("transaction_id must not be empty.")
        if not self.entity_id:
            raise GraphError("entity_id must not be empty.")
        if not isinstance(self.entity_type, EntityType):
            raise GraphError(f"Invalid entity_type: {self.entity_type}")
        if (
            not isinstance(self.timestamp, (int, float))
            or math.isnan(self.timestamp)
            or math.isinf(self.timestamp)
        ):
            raise GraphError("timestamp must be a finite float.")
        if (
            not isinstance(self.weight, (int, float))
            or math.isnan(self.weight)
            or math.isinf(self.weight)
            or self.weight <= 0.0
        ):
            raise GraphError("weight must be a positive finite float.")


@dataclass(frozen=True)
class GraphMetrics:
    """Summary metrics of the link graph at a point in time."""

    total_transactions: int
    total_entities: int
    total_edges: int
    entities_by_type: dict[str, int]
    window_seconds: float
    oldest_timestamp: float | None
    newest_timestamp: float | None
