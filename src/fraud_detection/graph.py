"""Real-time entity link analysis, fraud ring graph detection, and graph feature aggregation."""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


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

    def get_entities(
        self,
        entity_type: EntityType | None = None,
    ) -> list[EntityNode]:
        """Return all entities currently in the graph, optionally filtered by type."""
        with self._lock:
            if entity_type is None:
                return list(self._entities.values())
            return [node for (et, _), node in self._entities.items() if et == entity_type]

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


@dataclass(frozen=True)
class GraphFeatureConfig:
    """Configuration for graph feature extraction and risk diffusion."""

    windows_seconds: tuple[float, ...] = (3600.0, 86400.0)
    max_hops: int = 2
    diffusion_damping: float = 0.85
    diffusion_iterations: int = 5

    def __post_init__(self) -> None:
        if not self.windows_seconds:
            raise GraphError("windows_seconds must be a non-empty sequence of positive floats.")
        for w in self.windows_seconds:
            if not isinstance(w, (int, float)) or math.isnan(w) or math.isinf(w) or w <= 0.0:
                raise GraphError("Each window in windows_seconds must be a positive finite float.")
        if not isinstance(self.max_hops, int) or not (1 <= self.max_hops <= 5):
            raise GraphError("max_hops must be an integer between 1 and 5.")
        if (
            not isinstance(self.diffusion_damping, (int, float))
            or math.isnan(self.diffusion_damping)
            or math.isinf(self.diffusion_damping)
            or not (0.0 <= self.diffusion_damping <= 1.0)
        ):
            raise GraphError("diffusion_damping must be a float between 0.0 and 1.0.")
        if not isinstance(self.diffusion_iterations, int) or not (
            1 <= self.diffusion_iterations <= 20
        ):
            raise GraphError("diffusion_iterations must be an integer between 1 and 20.")


class GraphFeatureExtractor:
    """Extracts structural graph features and risk diffusion scores from EntityGraph."""

    def __init__(self, graph: EntityGraph, config: GraphFeatureConfig | None = None) -> None:
        self.graph = graph
        self.config = config or GraphFeatureConfig()

    def extract_features(
        self,
        transaction_id: str | None = None,
        timestamp: float | None = None,
        entities: dict[EntityType, str] | None = None,
    ) -> dict[str, float]:
        """Extract graph features for a candidate or existing transaction without lookahead bias."""
        if entities is None and transaction_id is not None:
            tx = self.graph.get_transaction(transaction_id)
            if tx is not None:
                entities = tx.entities
                if timestamp is None:
                    timestamp = tx.timestamp

        if entities is None:
            entities = {}

        if timestamp is None:
            timestamp = 0.0

        features: dict[str, float] = {}

        # 1. Degree Centrality across configured windows
        for w in self.config.windows_seconds:
            w_suffix = f"_{int(w)}s"
            since_ts = max(0.0, timestamp - w)

            # Degree features for each entity type in the transaction
            for et in (
                EntityType.CARD_ID,
                EntityType.DEVICE_ID,
                EntityType.IP_ADDRESS,
                EntityType.MERCHANT_ID,
                EntityType.USER_ID,
            ):
                eid = entities.get(et)
                if eid:
                    txs = self.graph.get_entity_transactions(et, eid, since_timestamp=since_ts)
                    # Filter out transactions strictly after timestamp to avoid lookahead bias
                    valid_txs = [
                        t
                        for t in txs
                        if t.timestamp <= timestamp and t.transaction_id != transaction_id
                    ]
                    features[f"graph_{et.value}_tx_count{w_suffix}"] = float(len(valid_txs))
                    features[f"graph_{et.value}_monetary_sum{w_suffix}"] = float(
                        sum(t.amount for t in valid_txs)
                    )

                    # Distinct peer cards connected to this device or IP
                    if et in (EntityType.DEVICE_ID, EntityType.IP_ADDRESS):
                        distinct_cards = {
                            t.entities[EntityType.CARD_ID]
                            for t in valid_txs
                            if EntityType.CARD_ID in t.entities
                        }
                        features[f"graph_{et.value}_distinct_cards{w_suffix}"] = float(
                            len(distinct_cards)
                        )

                    # Distinct devices or IPs connected to this card
                    if et == EntityType.CARD_ID:
                        distinct_devs = {
                            t.entities[EntityType.DEVICE_ID]
                            for t in valid_txs
                            if EntityType.DEVICE_ID in t.entities
                        }
                        distinct_ips = {
                            t.entities[EntityType.IP_ADDRESS]
                            for t in valid_txs
                            if EntityType.IP_ADDRESS in t.entities
                        }
                        features[f"graph_card_distinct_devices{w_suffix}"] = float(
                            len(distinct_devs)
                        )
                        features[f"graph_card_distinct_ips{w_suffix}"] = float(len(distinct_ips))
                else:
                    features[f"graph_{et.value}_tx_count{w_suffix}"] = 0.0
                    features[f"graph_{et.value}_monetary_sum{w_suffix}"] = 0.0
                    if et in (EntityType.DEVICE_ID, EntityType.IP_ADDRESS):
                        features[f"graph_{et.value}_distinct_cards{w_suffix}"] = 0.0
                    if et == EntityType.CARD_ID:
                        features[f"graph_card_distinct_devices{w_suffix}"] = 0.0
                        features[f"graph_card_distinct_ips{w_suffix}"] = 0.0

        # 2. 2-Hop Network Sharing & Connected Syndicate Size
        card_id = entities.get(EntityType.CARD_ID)
        if card_id:
            peer_cards = self.graph.get_2hop_entities(
                EntityType.CARD_ID, card_id, target_type=EntityType.CARD_ID
            )
            features["graph_shared_device_cards_count"] = float(len(peer_cards))
        else:
            features["graph_shared_device_cards_count"] = 0.0

        device_id = entities.get(EntityType.DEVICE_ID)
        if device_id:
            peer_devices = self.graph.get_2hop_entities(
                EntityType.DEVICE_ID, device_id, target_type=EntityType.DEVICE_ID
            )
            features["graph_shared_card_devices_count"] = float(len(peer_devices))
        else:
            features["graph_shared_card_devices_count"] = 0.0

        # 3. Localized Personalized Risk Diffusion Score
        risk_scores: list[float] = []
        for et, eid in entities.items():
            enode = self.graph.get_entity(et, eid)
            if enode is not None:
                risk_scores.append(enode.risk_score)
                # also consider 2-hop connected entities
                risk_scores.extend(
                    peer.risk_score * (1.0 - self.config.diffusion_damping)
                    for peer in self.graph.get_2hop_entities(et, eid)
                )

        if risk_scores:
            features["graph_risk_diffusion"] = float(min(1.0, max(0.0, max(risk_scores))))
            features["graph_neighborhood_avg_risk"] = float(
                min(1.0, max(0.0, sum(risk_scores) / len(risk_scores)))
            )
        else:
            features["graph_risk_diffusion"] = 0.0
            features["graph_neighborhood_avg_risk"] = 0.0

        return features


class RingSeverity(StrEnum):
    """Severity classification for detected fraud rings."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RingAction(StrEnum):
    """Recommended decision policy action for entities in a detected fraud ring."""

    ALLOW = "allow"
    CHALLENGE = "challenge"
    DENY = "deny"


class RingPatternType(StrEnum):
    """Topological pattern archetype for fraud syndicate rings."""

    BICLIQUE = "biclique"
    STAR_CARD_TESTING = "star_card_testing"
    CYCLE = "cycle"


@dataclass(frozen=True)
class FraudRingCluster:
    """Detected fraud syndicate cluster or attack ring."""

    cluster_id: str
    pattern_type: RingPatternType
    severity: RingSeverity
    recommended_action: RingAction
    entities: dict[EntityType, tuple[str, ...]]
    transaction_ids: tuple[str, ...]
    transaction_count: int
    total_amount: float
    density: float
    risk_score: float
    description: str

    def __post_init__(self) -> None:
        if not self.cluster_id or not isinstance(self.cluster_id, str):
            raise GraphError("cluster_id must be a non-empty string.")
        if not isinstance(self.pattern_type, RingPatternType):
            raise GraphError(f"Invalid pattern_type: {self.pattern_type}")
        if not isinstance(self.severity, RingSeverity):
            raise GraphError(f"Invalid severity: {self.severity}")
        if not isinstance(self.recommended_action, RingAction):
            raise GraphError(f"Invalid recommended_action: {self.recommended_action}")
        if not isinstance(self.entities, dict):
            raise GraphError("entities must be a dictionary.")
        if not isinstance(self.transaction_ids, tuple):
            raise GraphError("transaction_ids must be a tuple.")
        if not isinstance(self.transaction_count, int) or self.transaction_count < 0:
            raise GraphError("transaction_count must be a non-negative integer.")
        if (
            not isinstance(self.total_amount, (int, float))
            or math.isnan(self.total_amount)
            or math.isinf(self.total_amount)
            or self.total_amount < 0.0
        ):
            raise GraphError("total_amount must be a non-negative finite float.")
        if (
            not isinstance(self.density, (int, float))
            or math.isnan(self.density)
            or math.isinf(self.density)
            or not (0.0 <= self.density <= 1.0)
        ):
            raise GraphError("density must be a float between 0.0 and 1.0.")
        if (
            not isinstance(self.risk_score, (int, float))
            or math.isnan(self.risk_score)
            or math.isinf(self.risk_score)
            or not (0.0 <= self.risk_score <= 1.0)
        ):
            raise GraphError("risk_score must be a float between 0.0 and 1.0.")
        if not isinstance(self.description, str):
            raise GraphError("description must be a string.")


@dataclass(frozen=True)
class FraudRingReport:
    """Comprehensive report summarizing detected active fraud rings."""

    generated_at: float
    total_clusters: int
    clusters: tuple[FraudRingCluster, ...]
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.generated_at, (int, float))
            or math.isnan(self.generated_at)
            or math.isinf(self.generated_at)
        ):
            raise GraphError("generated_at must be a finite float.")
        if not isinstance(self.total_clusters, int) or self.total_clusters < 0:
            raise GraphError("total_clusters must be a non-negative integer.")
        if not isinstance(self.clusters, tuple):
            raise GraphError("clusters must be a tuple of FraudRingCluster objects.")


@dataclass(frozen=True)
class FraudRingDetectorConfig:
    """Configuration thresholds for automated fraud ring detection."""

    min_cards_per_device: int = 3
    min_devices_per_card: int = 2
    card_testing_window_seconds: float = 300.0
    card_testing_min_cards: int = 4
    cycle_max_depth: int = 4
    min_cluster_amount: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.min_cards_per_device, int) or self.min_cards_per_device < 2:
            raise GraphError("min_cards_per_device must be an integer >= 2.")
        if not isinstance(self.min_devices_per_card, int) or self.min_devices_per_card < 1:
            raise GraphError("min_devices_per_card must be an integer >= 1.")
        if (
            not isinstance(self.card_testing_window_seconds, (int, float))
            or math.isnan(self.card_testing_window_seconds)
            or math.isinf(self.card_testing_window_seconds)
            or self.card_testing_window_seconds <= 0.0
        ):
            raise GraphError("card_testing_window_seconds must be a positive finite float.")
        if not isinstance(self.card_testing_min_cards, int) or self.card_testing_min_cards < 2:
            raise GraphError("card_testing_min_cards must be an integer >= 2.")
        if not isinstance(self.cycle_max_depth, int) or not (2 <= self.cycle_max_depth <= 10):
            raise GraphError("cycle_max_depth must be an integer between 2 and 10.")
        if (
            not isinstance(self.min_cluster_amount, (int, float))
            or math.isnan(self.min_cluster_amount)
            or math.isinf(self.min_cluster_amount)
            or self.min_cluster_amount < 0.0
        ):
            raise GraphError("min_cluster_amount must be a non-negative finite float.")


class FraudRingDetector:
    """Detects fraud syndicates, star card-testing botnets, and transaction cycles."""

    def __init__(self, graph: EntityGraph, config: FraudRingDetectorConfig | None = None) -> None:
        self.graph = graph
        self.config = config or FraudRingDetectorConfig()

    def detect_rings(self, current_timestamp: float | None = None) -> FraudRingReport:
        """Analyze graph structure and detect all active fraud rings and attack clusters."""
        clusters: list[FraudRingCluster] = []
        clusters.extend(self._detect_card_testing_attacks(current_timestamp))
        clusters.extend(self._detect_dense_bicliques())
        clusters.extend(self._detect_cycles())

        severity_order = {
            RingSeverity.CRITICAL: 0,
            RingSeverity.HIGH: 1,
            RingSeverity.MEDIUM: 2,
            RingSeverity.LOW: 3,
        }
        clusters.sort(
            key=lambda c: (severity_order.get(c.severity, 4), -c.risk_score, -c.total_amount)
        )

        ref_time = current_timestamp if current_timestamp is not None else 0.0
        if current_timestamp is None:
            metrics = self.graph.get_metrics()
            ref_time = metrics.newest_timestamp or 0.0

        crit = sum(1 for c in clusters if c.severity == RingSeverity.CRITICAL)
        high = sum(1 for c in clusters if c.severity == RingSeverity.HIGH)
        med = sum(1 for c in clusters if c.severity == RingSeverity.MEDIUM)
        low = sum(1 for c in clusters if c.severity == RingSeverity.LOW)

        return FraudRingReport(
            generated_at=ref_time,
            total_clusters=len(clusters),
            clusters=tuple(clusters),
            critical_count=crit,
            high_count=high,
            medium_count=med,
            low_count=low,
            summary={
                "total_clusters": len(clusters),
                "critical": crit,
                "high": high,
                "medium": med,
                "low": low,
            },
        )

    def evaluate_transaction(
        self,
        transaction_id: str | None = None,
        entities: dict[EntityType, str] | None = None,
        timestamp: float | None = None,
    ) -> list[FraudRingCluster]:
        """Check if a specific transaction or entity set intersects any detected fraud ring."""
        report = self.detect_rings(current_timestamp=timestamp)
        if not report.clusters:
            return []

        if entities is None and transaction_id is not None:
            tx = self.graph.get_transaction(transaction_id)
            if tx is not None:
                entities = tx.entities

        matched: list[FraudRingCluster] = []
        for cluster in report.clusters:
            is_match = False
            if transaction_id and transaction_id in cluster.transaction_ids:
                is_match = True
            elif entities:
                for et, eid in entities.items():
                    if et in cluster.entities and eid in cluster.entities[et]:
                        is_match = True
                        break
            if is_match:
                matched.append(cluster)
        return matched

    def _detect_card_testing_attacks(
        self, current_timestamp: float | None
    ) -> list[FraudRingCluster]:
        """Detect rapid star-topology attacks (single device/IP hitting many cards)."""
        clusters: list[FraudRingCluster] = []
        window = self.config.card_testing_window_seconds
        target_entities = self.graph.get_entities(EntityType.DEVICE_ID) + self.graph.get_entities(
            EntityType.IP_ADDRESS
        )

        for enode in target_entities:
            et = enode.entity_type
            eid = enode.entity_id

            ref_time = (
                current_timestamp if current_timestamp is not None else enode.last_seen_timestamp
            )
            cutoff = ref_time - window

            txs = self.graph.get_entity_transactions(et, eid, since_timestamp=cutoff)
            valid_txs = [t for t in txs if t.timestamp <= ref_time]
            distinct_cards = {
                t.entities[EntityType.CARD_ID]
                for t in valid_txs
                if EntityType.CARD_ID in t.entities
            }

            if len(distinct_cards) >= self.config.card_testing_min_cards:
                total_amt = sum(t.amount for t in valid_txs)
                if total_amt < self.config.min_cluster_amount:
                    continue

                if len(distinct_cards) >= 8 or total_amt >= 5000.0 or enode.risk_score >= 0.8:
                    severity = RingSeverity.CRITICAL
                    action = RingAction.DENY
                    risk = 0.95
                elif len(distinct_cards) >= 4:
                    severity = RingSeverity.HIGH
                    action = RingAction.DENY
                    risk = 0.85
                else:
                    severity = RingSeverity.MEDIUM
                    action = RingAction.CHALLENGE
                    risk = 0.65

                tx_ids = tuple(sorted(t.transaction_id for t in valid_txs))
                c_id = f"star_{et.value}_{eid}"
                cluster_entities = {
                    et: (eid,),
                    EntityType.CARD_ID: tuple(sorted(distinct_cards)),
                }
                desc = (
                    f"High-velocity card-testing attack on {et.value} '{eid}' "
                    f"with {len(distinct_cards)} distinct cards across {len(valid_txs)} txs."
                )
                clusters.append(
                    FraudRingCluster(
                        cluster_id=c_id,
                        pattern_type=RingPatternType.STAR_CARD_TESTING,
                        severity=severity,
                        recommended_action=action,
                        entities=cluster_entities,
                        transaction_ids=tx_ids,
                        transaction_count=len(valid_txs),
                        total_amount=total_amt,
                        density=1.0,
                        risk_score=risk,
                        description=desc,
                    )
                )
        return clusters

    def _detect_dense_bicliques(self) -> list[FraudRingCluster]:
        """Detect dense bipartite card-device clusters (syndicate rings)."""
        clusters: list[FraudRingCluster] = []
        card_entities = self.graph.get_entities(EntityType.CARD_ID)
        card_to_devices: dict[str, set[str]] = {}

        for enode in card_entities:
            eid = enode.entity_id
            devices = {
                node.entity_id
                for node in self.graph.get_cooccurring_entities(
                    EntityType.CARD_ID, eid, target_type=EntityType.DEVICE_ID
                )
            }
            if len(devices) >= self.config.min_devices_per_card:
                card_to_devices[eid] = devices

        visited_cards: set[str] = set()
        for card1, dev1 in card_to_devices.items():
            if card1 in visited_cards:
                continue
            syndicate_cards: set[str] = {card1}
            shared_devices: set[str] = set(dev1)

            for card2, dev2 in card_to_devices.items():
                if card1 == card2:
                    continue
                common = dev1.intersection(dev2)
                if len(common) >= self.config.min_devices_per_card:
                    syndicate_cards.add(card2)
                    shared_devices.update(common)

            if (
                len(syndicate_cards) >= self.config.min_cards_per_device
                and len(shared_devices) >= self.config.min_devices_per_card
            ):
                visited_cards.update(syndicate_cards)
                all_tx_ids: set[str] = set()
                total_amt = 0.0
                for c_id in syndicate_cards:
                    for tx in self.graph.get_entity_transactions(EntityType.CARD_ID, c_id):
                        all_tx_ids.add(tx.transaction_id)
                        total_amt += tx.amount

                if total_amt < self.config.min_cluster_amount:
                    continue

                actual_edges = sum(
                    len(card_to_devices.get(c, set()).intersection(shared_devices))
                    for c in syndicate_cards
                )
                possible_edges = len(syndicate_cards) * len(shared_devices)
                density = float(actual_edges / possible_edges) if possible_edges > 0 else 0.0

                if len(syndicate_cards) >= 5 or total_amt >= 5000.0:
                    sev = RingSeverity.CRITICAL
                    act = RingAction.DENY
                    risk = 0.92
                elif len(syndicate_cards) >= 3:
                    sev = RingSeverity.HIGH
                    act = RingAction.DENY
                    risk = 0.82
                else:
                    sev = RingSeverity.MEDIUM
                    act = RingAction.CHALLENGE
                    risk = 0.65

                sorted_cards = tuple(sorted(syndicate_cards))
                sorted_devices = tuple(sorted(shared_devices))
                cluster_id = f"biclique_cards_{sorted_cards[0]}"
                desc = (
                    f"Dense card-device syndicate ring with {len(syndicate_cards)} cards "
                    f"operating across {len(shared_devices)} devices."
                )
                clusters.append(
                    FraudRingCluster(
                        cluster_id=cluster_id,
                        pattern_type=RingPatternType.BICLIQUE,
                        severity=sev,
                        recommended_action=act,
                        entities={
                            EntityType.CARD_ID: sorted_cards,
                            EntityType.DEVICE_ID: sorted_devices,
                        },
                        transaction_ids=tuple(sorted(all_tx_ids)),
                        transaction_count=len(all_tx_ids),
                        total_amount=total_amt,
                        density=density,
                        risk_score=risk,
                        description=desc,
                    )
                )
        return clusters

    def _detect_cycles(self) -> list[FraudRingCluster]:
        """Detect circular loops in entity relationships."""
        clusters: list[FraudRingCluster] = []
        card_entities = self.graph.get_entities(EntityType.CARD_ID)
        card_adj: dict[str, set[str]] = {}

        for enode in card_entities:
            eid = enode.entity_id
            peer_cards = {
                peer.entity_id
                for peer in self.graph.get_2hop_entities(
                    EntityType.CARD_ID, eid, target_type=EntityType.CARD_ID
                )
            }
            if peer_cards:
                card_adj[eid] = peer_cards

        visited: set[str] = set()
        for start_card in card_adj:
            if start_card in visited:
                continue
            path = [start_card]
            stack = [(start_card, iter(card_adj.get(start_card, set())))]
            while stack:
                _curr, neighbors = stack[-1]
                try:
                    nxt = next(neighbors)
                    if nxt == start_card and len(path) >= 3:
                        cycle_cards = tuple(sorted(set(path)))
                        visited.update(cycle_cards)
                        all_tx_ids: set[str] = set()
                        total_amt = 0.0
                        for c in cycle_cards:
                            for tx in self.graph.get_entity_transactions(EntityType.CARD_ID, c):
                                all_tx_ids.add(tx.transaction_id)
                                total_amt += tx.amount

                        desc = f"Closed circular transaction cycle of length {len(cycle_cards)}."
                        clusters.append(
                            FraudRingCluster(
                                cluster_id=f"cycle_{cycle_cards[0]}",
                                pattern_type=RingPatternType.CYCLE,
                                severity=RingSeverity.HIGH,
                                recommended_action=RingAction.DENY,
                                entities={EntityType.CARD_ID: cycle_cards},
                                transaction_ids=tuple(sorted(all_tx_ids)),
                                transaction_count=len(all_tx_ids),
                                total_amount=total_amt,
                                density=1.0,
                                risk_score=0.88,
                                description=desc,
                            )
                        )
                        break
                    if nxt not in path and len(path) < self.config.cycle_max_depth:
                        path.append(nxt)
                        stack.append((nxt, iter(card_adj.get(nxt, set()))))
                except StopIteration:
                    stack.pop()
                    path.pop()

        return clusters
