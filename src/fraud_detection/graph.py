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
