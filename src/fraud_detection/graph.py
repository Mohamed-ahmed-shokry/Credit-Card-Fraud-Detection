"""Real-time entity link analysis, fraud ring graph detection, and graph feature aggregation."""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
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


@dataclass(frozen=True)
class GraphConfig:
    """Configuration for EntityGraph memory, sliding window, and pruning."""

    window_seconds: float = 86400.0
    max_transactions: int = 50000
    max_entities: int = 100000
    prune_interval_seconds: float = 60.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.window_seconds, (int, float))
            or math.isnan(self.window_seconds)
            or math.isinf(self.window_seconds)
            or self.window_seconds <= 0.0
        ):
            raise GraphError("window_seconds must be a positive finite float.")
        if not isinstance(self.max_transactions, int) or self.max_transactions <= 0:
            raise GraphError("max_transactions must be a positive integer.")
        if not isinstance(self.max_entities, int) or self.max_entities <= 0:
            raise GraphError("max_entities must be a positive integer.")
        if (
            not isinstance(self.prune_interval_seconds, (int, float))
            or math.isnan(self.prune_interval_seconds)
            or math.isinf(self.prune_interval_seconds)
            or self.prune_interval_seconds < 0.0
        ):
            raise GraphError("prune_interval_seconds must be a non-negative finite float.")


class EntityGraph:
    """Thread-safe bipartite transaction-entity graph with temporal sliding window eviction."""

    def __init__(self, config: GraphConfig | None = None) -> None:
        self.config = config or GraphConfig()
        self._lock = threading.RLock()
        self._transactions: OrderedDict[str, TransactionNode] = OrderedDict()
        self._entities: dict[tuple[EntityType, str], EntityNode] = {}
        self._tx_to_entities: dict[str, set[tuple[EntityType, str]]] = {}
        self._entity_to_tx: dict[tuple[EntityType, str], set[str]] = {}
        self._edges: dict[tuple[str, EntityType, str], GraphEdge] = {}
        self._last_prune_timestamp: float = 0.0

    def add_transaction(
        self,
        transaction_id: str,
        timestamp: float,
        amount: float,
        entities: dict[EntityType, str],
        is_fraud: bool | None = None,
        risk_score: float = 0.0,
    ) -> TransactionNode:
        """Add a transaction and connect it to its associated entities."""
        with self._lock:
            # Auto-prune if interval elapsed
            if timestamp - self._last_prune_timestamp >= self.config.prune_interval_seconds:
                self.evict_expired(timestamp)
                self._last_prune_timestamp = timestamp

            # If transaction already exists, remove its old edges first
            if transaction_id in self._transactions:
                self._remove_transaction_edges(transaction_id)

            tx_node = TransactionNode(
                transaction_id=transaction_id,
                timestamp=timestamp,
                amount=amount,
                entities=entities,
                is_fraud=is_fraud,
                risk_score=risk_score,
            )
            self._transactions[transaction_id] = tx_node
            self._tx_to_entities[transaction_id] = set()

            for entity_type, entity_id in entities.items():
                if not entity_id or not isinstance(entity_id, str):
                    continue
                entity_key = (entity_type, entity_id)
                self._tx_to_entities[transaction_id].add(entity_key)

                if entity_key not in self._entity_to_tx:
                    self._entity_to_tx[entity_key] = set()
                self._entity_to_tx[entity_key].add(transaction_id)

                # Create or update EntityNode
                existing_entity = self._entities.get(entity_key)
                if existing_entity is None:
                    node_risk = risk_score if (is_fraud is not True) else 1.0
                    self._entities[entity_key] = EntityNode(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        first_seen_timestamp=timestamp,
                        last_seen_timestamp=timestamp,
                        risk_score=node_risk,
                    )
                else:
                    new_risk = max(
                        existing_entity.risk_score,
                        1.0 if (is_fraud is True) else risk_score,
                    )
                    self._entities[entity_key] = EntityNode(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        first_seen_timestamp=min(existing_entity.first_seen_timestamp, timestamp),
                        last_seen_timestamp=max(existing_entity.last_seen_timestamp, timestamp),
                        risk_score=new_risk,
                    )

                edge = GraphEdge(
                    transaction_id=transaction_id,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    timestamp=timestamp,
                )
                self._edges[(transaction_id, entity_type, entity_id)] = edge

            # Enforce max capacity
            self._enforce_capacity()
            return tx_node

    def _remove_transaction_edges(self, transaction_id: str) -> None:
        """Remove edges and associations for a transaction (internal helper)."""
        entity_keys = self._tx_to_entities.pop(transaction_id, set())
        for entity_key in entity_keys:
            if entity_key in self._entity_to_tx:
                self._entity_to_tx[entity_key].discard(transaction_id)
                if not self._entity_to_tx[entity_key]:
                    del self._entity_to_tx[entity_key]
                    self._entities.pop(entity_key, None)
            edge_key = (transaction_id, entity_key[0], entity_key[1])
            self._edges.pop(edge_key, None)

    def _enforce_capacity(self) -> None:
        """Prune oldest transactions if capacity exceeded."""
        while len(self._transactions) > self.config.max_transactions:
            oldest_tx_id, _ = self._transactions.popitem(last=False)
            self._remove_transaction_edges(oldest_tx_id)

    def evict_expired(self, current_timestamp: float | None = None) -> int:
        """Evict transactions older than window_seconds from the given or latest timestamp."""
        with self._lock:
            if not self._transactions:
                return 0

            ref_time = current_timestamp
            if ref_time is None:
                # Use newest transaction timestamp
                newest_tx = next(reversed(self._transactions.values()))
                ref_time = newest_tx.timestamp

            cutoff = ref_time - self.config.window_seconds
            evicted_count = 0
            # Collect expired transaction IDs
            expired_ids = [
                tx_id for tx_id, tx_node in self._transactions.items() if tx_node.timestamp < cutoff
            ]
            for tx_id in expired_ids:
                if tx_id in self._transactions:
                    del self._transactions[tx_id]
                    self._remove_transaction_edges(tx_id)
                    evicted_count += 1

            return evicted_count

    def get_entity(self, entity_type: EntityType, entity_id: str) -> EntityNode | None:
        """Retrieve an EntityNode by type and identifier."""
        with self._lock:
            return self._entities.get((entity_type, entity_id))

    def get_transaction(self, transaction_id: str) -> TransactionNode | None:
        """Retrieve a TransactionNode by identifier."""
        with self._lock:
            return self._transactions.get(transaction_id)

    def get_entity_transactions(
        self,
        entity_type: EntityType,
        entity_id: str,
        since_timestamp: float | None = None,
    ) -> list[TransactionNode]:
        """Return all transactions associated with the entity, optionally filtered by timestamp."""
        with self._lock:
            entity_key = (entity_type, entity_id)
            tx_ids = self._entity_to_tx.get(entity_key, set())
            result = []
            for tx_id in tx_ids:
                tx_node = self._transactions.get(tx_id)
                if tx_node is not None and (
                    since_timestamp is None or tx_node.timestamp >= since_timestamp
                ):
                    result.append(tx_node)
            return sorted(result, key=lambda t: t.timestamp)

    def get_transaction_entities(
        self,
        transaction_id: str,
    ) -> dict[EntityType, EntityNode]:
        """Return all entities connected to the given transaction."""
        with self._lock:
            entity_keys = self._tx_to_entities.get(transaction_id, set())
            result: dict[EntityType, EntityNode] = {}
            for et, eid in entity_keys:
                node = self._entities.get((et, eid))
                if node is not None:
                    result[et] = node
            return result

    def get_2hop_transactions(
        self,
        transaction_id: str,
        via_entity_type: EntityType | None = None,
    ) -> list[TransactionNode]:
        """Find transactions sharing at least one entity with the given transaction."""
        with self._lock:
            entity_keys = self._tx_to_entities.get(transaction_id, set())
            connected_tx_ids: set[str] = set()
            for et, eid in entity_keys:
                if via_entity_type is not None and et != via_entity_type:
                    continue
                peer_tx_ids = self._entity_to_tx.get((et, eid), set())
                connected_tx_ids.update(peer_tx_ids)

            connected_tx_ids.discard(transaction_id)
            result = [
                self._transactions[tid] for tid in connected_tx_ids if tid in self._transactions
            ]
            return sorted(result, key=lambda t: t.timestamp)

    def get_cooccurring_entities(
        self,
        entity_type: EntityType,
        entity_id: str,
        target_type: EntityType | None = None,
    ) -> list[EntityNode]:
        """Find entities directly co-occurring in the same transactions as the source entity."""
        with self._lock:
            entity_key = (entity_type, entity_id)
            tx_ids = self._entity_to_tx.get(entity_key, set())
            target_keys: set[tuple[EntityType, str]] = set()
            for tid in tx_ids:
                for pt in self._tx_to_entities.get(tid, set()):
                    if pt == entity_key:
                        continue
                    if target_type is not None and pt[0] != target_type:
                        continue
                    target_keys.add(pt)
            result = [self._entities[k] for k in target_keys if k in self._entities]
            return sorted(result, key=lambda e: (e.entity_type, e.entity_id))

    def get_2hop_entities(
        self,
        entity_type: EntityType,
        entity_id: str,
        target_type: EntityType | None = None,
    ) -> list[EntityNode]:
        """Find entities connected to the source entity through shared intermediate entities."""
        with self._lock:
            entity_key = (entity_type, entity_id)
            tx_ids = self._entity_to_tx.get(entity_key, set())
            # Find intermediate entities appearing in these transactions
            intermediate_entities: set[tuple[EntityType, str]] = set()
            for tid in tx_ids:
                for pt in self._tx_to_entities.get(tid, set()):
                    if pt != entity_key:
                        intermediate_entities.add(pt)

            # Find all transactions touching any of these intermediate entities
            connected_tx_ids: set[str] = set()
            for i_ent in intermediate_entities:
                connected_tx_ids.update(self._entity_to_tx.get(i_ent, set()))

            # Find all entities appearing in those connected transactions
            target_keys: set[tuple[EntityType, str]] = set()
            for c_tid in connected_tx_ids:
                for pt in self._tx_to_entities.get(c_tid, set()):
                    if pt == entity_key:
                        continue
                    if target_type is not None and pt[0] != target_type:
                        continue
                    target_keys.add(pt)

            result = [self._entities[k] for k in target_keys if k in self._entities]
            return sorted(result, key=lambda e: (e.entity_type, e.entity_id))

    def get_metrics(self) -> GraphMetrics:
        """Return graph structural metrics."""
        with self._lock:
            entities_by_type: dict[str, int] = {}
            for et, _ in self._entities:
                entities_by_type[et.value] = entities_by_type.get(et.value, 0) + 1

            oldest_ts = None
            newest_ts = None
            if self._transactions:
                first_tx = next(iter(self._transactions.values()))
                last_tx = next(reversed(self._transactions.values()))
                oldest_ts = first_tx.timestamp
                newest_ts = last_tx.timestamp

            return GraphMetrics(
                total_transactions=len(self._transactions),
                total_entities=len(self._entities),
                total_edges=len(self._edges),
                entities_by_type=entities_by_type,
                window_seconds=self.config.window_seconds,
                oldest_timestamp=oldest_ts,
                newest_timestamp=newest_ts,
            )

    def reset(self) -> None:
        """Clear all transactions, entities, and edges from the graph."""
        with self._lock:
            self._transactions.clear()
            self._entities.clear()
            self._tx_to_entities.clear()
            self._entity_to_tx.clear()
            self._edges.clear()
            self._last_prune_timestamp = 0.0
