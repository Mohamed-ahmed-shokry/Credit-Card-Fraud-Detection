"""Anti-Money Laundering (AML) Transaction Monitoring, Typology Detection & Regulatory SAR Engine.

Provides domain models, sanctions/PEP fuzzy watchlist screening, typology anomaly detectors
(structuring/smurfing, layering flow-through, circular fund movement, FATF high-risk corridors),
composite AML risk decisioning, and regulatory Suspicious Activity Report (SAR) dossier management.
"""

from __future__ import annotations

import re
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from fraud_detection.cases import CaseManager
    from fraud_detection.events import EventBus


class AmlTypology(StrEnum):
    """Specific money laundering and financial crime typologies."""

    STRUCTURING_SMURFING = "structuring_smurfing"
    RAPID_MOVEMENT_LAYERING = "rapid_movement_layering"
    ROUND_TRIP_CIRCULAR = "round_trip_circular"
    SANCTION_PEP_MATCH = "sanction_pep_match"
    HIGH_RISK_JURISDICTION = "high_risk_jurisdiction"
    HIGH_VELOCITY_BURST = "high_velocity_burst"


class AmlRiskLevel(StrEnum):
    """Calibrated AML risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AmlAction(StrEnum):
    """Policy actions triggered by AML assessment."""

    PASS = "pass"  # noqa: S105
    MONITOR = "monitor"
    ENHANCED_DUE_DILIGENCE = "enhanced_due_diligence"
    FILE_SAR_BLOCK = "file_sar_block"


class SarStatus(StrEnum):
    """Lifecycle review states for Suspicious Activity Reports (SAR)."""

    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    FILED = "filed"
    REJECTED = "rejected"


class WatchlistType(StrEnum):
    """Categories of regulatory watchlists."""

    OFAC_SDN = "ofac_sdn"
    PEP = "pep"
    FATF_HIGH_RISK = "fatf_high_risk"
    LAW_ENFORCEMENT = "law_enforcement"


class TransactionDirection(StrEnum):
    """Direction of money movement."""

    INFLOW = "inflow"
    OUTFLOW = "outflow"


@dataclass(frozen=True)
class WatchlistEntry:
    """An entry on a sanctions, PEP, or law enforcement watchlist."""

    name: str
    watchlist_type: WatchlistType
    country: str
    entity_id: str = field(default_factory=lambda: str(uuid4()))
    aliases: tuple[str, ...] = field(default_factory=tuple)
    listed_date: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize watchlist entry to dictionary."""
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "watchlist_type": self.watchlist_type.value,
            "country": self.country,
            "aliases": list(self.aliases),
            "listed_date": self.listed_date,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WatchlistEntry:
        """Deserialize watchlist entry from dictionary."""
        return cls(
            entity_id=str(data.get("entity_id", uuid4())),
            name=str(data["name"]),
            watchlist_type=WatchlistType(data["watchlist_type"]),
            country=str(data.get("country", "")),
            aliases=tuple(str(a) for a in data.get("aliases", ())),
            listed_date=str(data.get("listed_date", "")),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class ScreeningMatch:
    """Detailed report of a watchlist screening hit."""

    entity_name: str
    matched_entry: WatchlistEntry
    similarity_score: float
    match_algorithm: str
    risk_level: AmlRiskLevel
    match_id: str = field(default_factory=lambda: str(uuid4()))
    screened_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize screening match to dictionary."""
        return {
            "match_id": self.match_id,
            "entity_name": self.entity_name,
            "matched_entry": self.matched_entry.to_dict(),
            "similarity_score": round(self.similarity_score, 4),
            "match_algorithm": self.match_algorithm,
            "risk_level": self.risk_level.value,
            "screened_at": self.screened_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScreeningMatch:
        """Deserialize screening match from dictionary."""
        screened_at_raw = data.get("screened_at")
        screened_at = (
            datetime.fromisoformat(screened_at_raw)
            if isinstance(screened_at_raw, str)
            else datetime.now(UTC)
        )

        return cls(
            match_id=str(data.get("match_id", uuid4())),
            entity_name=str(data["entity_name"]),
            matched_entry=WatchlistEntry.from_dict(data["matched_entry"]),
            similarity_score=float(data["similarity_score"]),
            match_algorithm=str(data["match_algorithm"]),
            risk_level=AmlRiskLevel(data["risk_level"]),
            screened_at=screened_at,
        )


@dataclass(frozen=True)
class AmlTransactionEvent:
    """Financial transaction event ingested for AML analysis."""

    transaction_id: str
    user_id: str
    amount: float
    currency: str = "USD"
    direction: TransactionDirection = TransactionDirection.OUTFLOW
    counterparty_id: str = ""
    country: str = "US"
    payment_method: str = "card"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    tenant_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize transaction event to dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "user_id": self.user_id,
            "amount": round(self.amount, 2),
            "currency": self.currency,
            "direction": self.direction.value,
            "counterparty_id": self.counterparty_id,
            "country": self.country,
            "payment_method": self.payment_method,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AmlTransactionEvent:
        """Deserialize transaction event from dictionary."""
        ts_raw = data.get("timestamp")
        ts = datetime.fromisoformat(ts_raw) if isinstance(ts_raw, str) else datetime.now(UTC)

        return cls(
            transaction_id=str(data["transaction_id"]),
            user_id=str(data["user_id"]),
            amount=float(data["amount"]),
            currency=str(data.get("currency", "USD")),
            direction=TransactionDirection(
                data.get("direction", TransactionDirection.OUTFLOW.value)
            ),
            counterparty_id=str(data.get("counterparty_id", "")),
            country=str(data.get("country", "US")),
            payment_method=str(data.get("payment_method", "card")),
            timestamp=ts,
            tenant_id=str(data.get("tenant_id", "default")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class StructuringAlert:
    """Alert triggered by rapid sub-threshold cash/card structuring."""

    user_id: str
    transaction_ids: tuple[str, ...]
    total_amount: float
    transaction_count: int
    window_seconds: float
    average_amount: float
    threshold_proximity: float
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize structuring alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "user_id": self.user_id,
            "transaction_ids": list(self.transaction_ids),
            "total_amount": round(self.total_amount, 2),
            "transaction_count": self.transaction_count,
            "window_seconds": self.window_seconds,
            "average_amount": round(self.average_amount, 2),
            "threshold_proximity": round(self.threshold_proximity, 4),
            "detected_at": self.detected_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuringAlert:
        """Deserialize structuring alert from dictionary."""
        detected_at_raw = data.get("detected_at")
        detected_at = (
            datetime.fromisoformat(detected_at_raw)
            if isinstance(detected_at_raw, str)
            else datetime.now(UTC)
        )
        return cls(
            alert_id=str(data.get("alert_id", uuid4())),
            user_id=str(data["user_id"]),
            transaction_ids=tuple(str(t) for t in data.get("transaction_ids", ())),
            total_amount=float(data["total_amount"]),
            transaction_count=int(data["transaction_count"]),
            window_seconds=float(data["window_seconds"]),
            average_amount=float(data["average_amount"]),
            threshold_proximity=float(data["threshold_proximity"]),
            detected_at=detected_at,
        )


@dataclass(frozen=True)
class AmlAlert:
    """Comprehensive alert record for suspected money laundering."""

    user_id: str
    typology: AmlTypology
    risk_level: AmlRiskLevel
    recommended_action: AmlAction
    risk_score: float
    trigger_details: dict[str, Any] = field(default_factory=dict)
    transaction_ids: tuple[str, ...] = field(default_factory=tuple)
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    tenant_id: str = "default"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize AML alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "user_id": self.user_id,
            "typology": self.typology.value,
            "risk_level": self.risk_level.value,
            "recommended_action": self.recommended_action.value,
            "risk_score": round(self.risk_score, 4),
            "trigger_details": dict(self.trigger_details),
            "transaction_ids": list(self.transaction_ids),
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AmlAlert:
        """Deserialize AML alert from dictionary."""
        created_at_raw = data.get("created_at")
        created_at = (
            datetime.fromisoformat(created_at_raw)
            if isinstance(created_at_raw, str)
            else datetime.now(UTC)
        )
        return cls(
            alert_id=str(data.get("alert_id", uuid4())),
            user_id=str(data["user_id"]),
            typology=AmlTypology(data["typology"]),
            risk_level=AmlRiskLevel(data["risk_level"]),
            recommended_action=AmlAction(data["recommended_action"]),
            risk_score=float(data["risk_score"]),
            trigger_details=dict(data.get("trigger_details", {})),
            transaction_ids=tuple(str(t) for t in data.get("transaction_ids", ())),
            tenant_id=str(data.get("tenant_id", "default")),
            created_at=created_at,
        )


@dataclass(frozen=True)
class SarDossier:
    """Suspicious Activity Report (SAR) regulatory filing package."""

    user_id: str
    narrative: str
    alert_ids: tuple[str, ...] = field(default_factory=tuple)
    status: SarStatus = SarStatus.DRAFT
    suspect_details: dict[str, Any] = field(default_factory=dict)
    summary_financials: dict[str, Any] = field(default_factory=dict)
    jurisdictions: tuple[str, ...] = field(default_factory=tuple)
    sar_id: str = field(default_factory=lambda: str(uuid4()))
    filing_date: str = ""
    tenant_id: str = "default"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    audit_trail: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialize SAR dossier to dictionary."""
        return {
            "sar_id": self.sar_id,
            "user_id": self.user_id,
            "alert_ids": list(self.alert_ids),
            "status": self.status.value,
            "narrative": self.narrative,
            "suspect_details": dict(self.suspect_details),
            "summary_financials": dict(self.summary_financials),
            "jurisdictions": list(self.jurisdictions),
            "filing_date": self.filing_date,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "audit_trail": [dict(entry) for entry in self.audit_trail],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SarDossier:
        """Deserialize SAR dossier from dictionary."""
        created_at_raw = data.get("created_at")
        created_at = (
            datetime.fromisoformat(created_at_raw)
            if isinstance(created_at_raw, str)
            else datetime.now(UTC)
        )
        updated_at_raw = data.get("updated_at")
        updated_at = (
            datetime.fromisoformat(updated_at_raw)
            if isinstance(updated_at_raw, str)
            else datetime.now(UTC)
        )
        return cls(
            sar_id=str(data.get("sar_id", uuid4())),
            user_id=str(data["user_id"]),
            narrative=str(data.get("narrative", "")),
            alert_ids=tuple(str(a) for a in data.get("alert_ids", ())),
            status=SarStatus(data.get("status", SarStatus.DRAFT.value)),
            suspect_details=dict(data.get("suspect_details", {})),
            summary_financials=dict(data.get("summary_financials", {})),
            jurisdictions=tuple(str(j) for j in data.get("jurisdictions", ())),
            filing_date=str(data.get("filing_date", "")),
            tenant_id=str(data.get("tenant_id", "default")),
            created_at=created_at,
            updated_at=updated_at,
            audit_trail=tuple(dict(entry) for entry in data.get("audit_trail", ())),
        )


def soundex(text: str) -> str:
    """Compute the American Soundex code for an entity name.

    Returns a 4-character phonetic code (e.g., 'Robert' -> 'R163', 'Rupert' -> 'R163').
    """
    cleaned = re.sub(r"[^A-Za-z]", "", text).upper()
    if not cleaned:
        return ""

    first_char = cleaned[0]
    mapping = {
        "B": "1",
        "F": "1",
        "P": "1",
        "V": "1",
        "C": "2",
        "G": "2",
        "J": "2",
        "K": "2",
        "Q": "2",
        "S": "2",
        "X": "2",
        "Z": "2",
        "D": "3",
        "T": "3",
        "L": "4",
        "M": "5",
        "N": "5",
        "R": "6",
        "A": "0",
        "E": "0",
        "I": "0",
        "O": "0",
        "U": "0",
        "Y": "0",
        "H": "0",
        "W": "0",
    }

    codes: list[str] = [first_char]
    prev_code = mapping.get(first_char, "0")

    for char in cleaned[1:]:
        code = mapping.get(char, "0")
        if char in ("H", "W"):
            continue
        if code != "0" and code != prev_code:
            codes.append(code)
        prev_code = code

    # Remove 0s and pad to 4 characters
    soundex_digits = [c for c in codes[1:] if c != "0"]
    return (first_char + "".join(soundex_digits) + "000")[:4]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute minimum edit distance (insertions, deletions, substitutions)."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity ratio between 0.0 and 1.0."""
    if s1 == s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    return max(0.0, 1.0 - (dist / max_len))


def jaro_similarity(s1: str, s2: str) -> float:
    """Compute the Jaro similarity metric between two strings (0.0 to 1.0)."""
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(0, (max(len1, len2) // 2) - 1)

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    t = transpositions / 2.0
    return (matches / len1 + matches / len2 + (matches - t) / matches) / 3.0


def jaro_winkler_similarity(
    s1: str,
    s2: str,
    prefix_weight: float = 0.1,
    max_prefix: int = 4,
) -> float:
    """Compute Jaro-Winkler similarity giving higher weight to common prefix."""
    jaro_sim = jaro_similarity(s1, s2)
    if jaro_sim == 1.0 or jaro_sim == 0.0:
        return jaro_sim

    prefix_len = 0
    for c1, c2 in zip(s1, s2, strict=False):
        if c1 == c2:
            prefix_len += 1
            if prefix_len >= max_prefix:
                break
        else:
            break

    return min(1.0, jaro_sim + prefix_len * prefix_weight * (1.0 - jaro_sim))


class WatchlistScreeningEngine:
    """Thread-safe sanctions, PEP, and law enforcement fuzzy screening engine."""

    def __init__(self, default_threshold: float = 0.85) -> None:
        self.default_threshold = default_threshold
        self._lock = threading.RLock()
        self._entries: dict[str, WatchlistEntry] = {}

    def add_entry(self, entry: WatchlistEntry) -> None:
        """Add or update an entry in the watchlist."""
        with self._lock:
            self._entries[entry.entity_id] = entry

    def remove_entry(self, entity_id: str) -> bool:
        """Remove an entry from the watchlist. Returns True if removed."""
        with self._lock:
            return self._entries.pop(entity_id, None) is not None

    def get_entry(self, entity_id: str) -> WatchlistEntry | None:
        """Retrieve a watchlist entry by entity ID."""
        with self._lock:
            return self._entries.get(entity_id)

    def list_entries(
        self,
        watchlist_type: WatchlistType | None = None,
    ) -> list[WatchlistEntry]:
        """List all registered watchlist entries, optionally filtered by type."""
        with self._lock:
            if watchlist_type is None:
                return list(self._entries.values())
            return [e for e in self._entries.values() if e.watchlist_type == watchlist_type]

    def count(self) -> int:
        """Return total number of entries in watchlist."""
        with self._lock:
            return len(self._entries)

    def screen_entity(
        self,
        name: str,
        threshold: float | None = None,
        country: str | None = None,
    ) -> list[ScreeningMatch]:
        """Screen an individual or organization name against registered watchlist entries."""
        cutoff = self.default_threshold if threshold is None else threshold
        normalized_query = " ".join(re.sub(r"[^\w\s]", "", name).strip().upper().split())
        if not normalized_query:
            return []

        query_soundex = soundex(normalized_query)
        query_tokens = sorted(normalized_query.split())
        query_sorted_str = " ".join(query_tokens)

        matches: list[ScreeningMatch] = []

        with self._lock:
            candidates = list(self._entries.values())

        for entry in candidates:
            # If country filter specified and does not match entry country, skip
            if country and entry.country and entry.country.upper() != country.upper():
                continue

            target_names = [entry.name, *entry.aliases]
            best_score = 0.0
            best_algo = "none"

            for target in target_names:
                norm_target = " ".join(re.sub(r"[^\w\s]", "", target).strip().upper().split())
                if not norm_target:
                    continue

                # 1. Exact match
                if normalized_query == norm_target:
                    best_score = 1.0
                    best_algo = "exact"
                    break

                # 2. Token-sorted match (e.g. "Bout Viktor" vs "Viktor Bout")
                target_tokens = sorted(norm_target.split())
                target_sorted_str = " ".join(target_tokens)
                if query_sorted_str == target_sorted_str:
                    best_score = max(best_score, 0.98)
                    best_algo = "token_sort_exact"

                # 3. Jaro-Winkler similarity
                jw_score = jaro_winkler_similarity(normalized_query, norm_target)
                jw_sorted_score = jaro_winkler_similarity(query_sorted_str, target_sorted_str)
                top_jw = max(jw_score, jw_sorted_score)
                if top_jw > best_score:
                    best_score = top_jw
                    best_algo = "jaro_winkler"

                # 4. Levenshtein normalized similarity
                lev_score = levenshtein_similarity(normalized_query, norm_target)
                if lev_score > best_score:
                    best_score = lev_score
                    best_algo = "levenshtein"

                # 5. Phonetic Soundex match boost
                target_soundex = soundex(norm_target)
                if (
                    query_soundex
                    and target_soundex
                    and query_soundex == target_soundex
                    and lev_score >= 0.70
                ):
                    soundex_boosted = min(1.0, max(best_score, 0.88))
                    if soundex_boosted > best_score:
                        best_score = soundex_boosted
                        best_algo = "soundex_phonetic"

            if best_score >= cutoff:
                if best_score >= 0.95:
                    risk_level = AmlRiskLevel.CRITICAL
                elif best_score >= 0.88:
                    risk_level = AmlRiskLevel.HIGH
                elif best_score >= 0.80:
                    risk_level = AmlRiskLevel.MEDIUM
                else:
                    risk_level = AmlRiskLevel.LOW

                matches.append(
                    ScreeningMatch(
                        entity_name=name,
                        matched_entry=entry,
                        similarity_score=best_score,
                        match_algorithm=best_algo,
                        risk_level=risk_level,
                    )
                )

        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches


class StructuringDetector:
    """Sliding-window detector for currency/card transaction structuring and smurfing."""

    def __init__(
        self,
        reporting_threshold: float = 10000.0,
        sub_threshold_ratio: float = 0.80,
        window_seconds: float = 86400.0,
        min_sub_threshold_count: int = 2,
    ) -> None:
        self.reporting_threshold = reporting_threshold
        self.sub_threshold_ratio = sub_threshold_ratio
        self.window_seconds = window_seconds
        self.min_sub_threshold_count = min_sub_threshold_count
        self._lock = threading.RLock()
        self._user_events: dict[str, deque[AmlTransactionEvent]] = defaultdict(deque)

    def record_and_evaluate(
        self,
        event: AmlTransactionEvent,
    ) -> StructuringAlert | None:
        """Record a transaction and evaluate whether structuring/smurfing patterns match."""
        with self._lock:
            queue = self._user_events[event.user_id]
            queue.append(event)
            cutoff = event.timestamp - timedelta(seconds=self.window_seconds)
            while queue and queue[0].timestamp < cutoff:
                queue.popleft()

            events_in_window = list(queue)

        sub_floor = self.reporting_threshold * self.sub_threshold_ratio
        sub_ceiling = self.reporting_threshold

        near_threshold_events = [e for e in events_in_window if sub_floor <= e.amount < sub_ceiling]
        total_sum = sum(e.amount for e in events_in_window)

        is_structuring = False
        flagged_events: list[AmlTransactionEvent] = []

        if len(near_threshold_events) >= self.min_sub_threshold_count:
            is_structuring = True
            flagged_events = near_threshold_events
        elif (
            len(events_in_window) >= 3
            and total_sum >= self.reporting_threshold
            and all(e.amount < self.reporting_threshold for e in events_in_window)
        ):
            is_structuring = True
            flagged_events = events_in_window

        if not is_structuring:
            return None

        flagged_sum = sum(e.amount for e in flagged_events)
        avg_amt = flagged_sum / len(flagged_events)
        proximity = min(1.0, avg_amt / self.reporting_threshold)

        return StructuringAlert(
            user_id=event.user_id,
            transaction_ids=tuple(e.transaction_id for e in flagged_events),
            total_amount=flagged_sum,
            transaction_count=len(flagged_events),
            window_seconds=self.window_seconds,
            average_amount=avg_amt,
            threshold_proximity=proximity,
            detected_at=event.timestamp,
        )

    def clear(self) -> None:
        """Clear all sliding window transaction histories."""
        with self._lock:
            self._user_events.clear()


@dataclass(frozen=True)
class LayeringAlert:
    """Alert triggered by rapid fund pass-through with minimal account dwell time."""

    user_id: str
    inflow_tx_id: str
    outflow_tx_id: str
    inflow_amount: float
    outflow_amount: float
    turnover_ratio: float
    dwell_minutes: float
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize layering alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "user_id": self.user_id,
            "inflow_tx_id": self.inflow_tx_id,
            "outflow_tx_id": self.outflow_tx_id,
            "inflow_amount": round(self.inflow_amount, 2),
            "outflow_amount": round(self.outflow_amount, 2),
            "turnover_ratio": round(self.turnover_ratio, 4),
            "dwell_minutes": round(self.dwell_minutes, 2),
            "detected_at": self.detected_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayeringAlert:
        """Deserialize layering alert from dictionary."""
        detected_at_raw = data.get("detected_at")
        detected_at = (
            datetime.fromisoformat(detected_at_raw)
            if isinstance(detected_at_raw, str)
            else datetime.now(UTC)
        )
        return cls(
            alert_id=str(data.get("alert_id", uuid4())),
            user_id=str(data["user_id"]),
            inflow_tx_id=str(data["inflow_tx_id"]),
            outflow_tx_id=str(data["outflow_tx_id"]),
            inflow_amount=float(data["inflow_amount"]),
            outflow_amount=float(data["outflow_amount"]),
            turnover_ratio=float(data["turnover_ratio"]),
            dwell_minutes=float(data["dwell_minutes"]),
            detected_at=detected_at,
        )


class LayeringDetector:
    """Detects rapid pass-through / layering flow of funds with near-zero dwell time."""

    def __init__(
        self,
        max_dwell_hours: float = 4.0,
        min_flow_amount: float = 3000.0,
        min_turnover_ratio: float = 0.85,
        window_seconds: float = 86400.0,
    ) -> None:
        self.max_dwell_hours = max_dwell_hours
        self.min_flow_amount = min_flow_amount
        self.min_turnover_ratio = min_turnover_ratio
        self.window_seconds = window_seconds
        self._lock = threading.RLock()
        self._inflows: dict[str, deque[AmlTransactionEvent]] = defaultdict(deque)

    def record_and_evaluate(
        self,
        event: AmlTransactionEvent,
    ) -> LayeringAlert | None:
        """Record an inflow or outflow and evaluate if funds are rapidly stripped."""
        with self._lock:
            inflows_queue = self._inflows[event.user_id]
            cutoff = event.timestamp - timedelta(seconds=self.window_seconds)
            while inflows_queue and inflows_queue[0].timestamp < cutoff:
                inflows_queue.popleft()

            if event.direction == TransactionDirection.INFLOW:
                inflows_queue.append(event)
                return None

            if event.amount < self.min_flow_amount:
                return None

            for inflow in reversed(inflows_queue):
                if inflow.timestamp > event.timestamp:
                    continue
                dwell = (event.timestamp - inflow.timestamp).total_seconds() / 60.0
                if dwell > (self.max_dwell_hours * 60.0):
                    break
                if inflow.amount <= 0:
                    continue
                ratio = event.amount / inflow.amount
                if ratio >= self.min_turnover_ratio:
                    return LayeringAlert(
                        user_id=event.user_id,
                        inflow_tx_id=inflow.transaction_id,
                        outflow_tx_id=event.transaction_id,
                        inflow_amount=inflow.amount,
                        outflow_amount=event.amount,
                        turnover_ratio=ratio,
                        dwell_minutes=dwell,
                        detected_at=event.timestamp,
                    )
            return None

    def clear(self) -> None:
        """Clear all inflow queues."""
        with self._lock:
            self._inflows.clear()


@dataclass(frozen=True)
class CircularFlowAlert:
    """Alert triggered by circular / round-trip transfer loops."""

    origin_user_id: str
    counterparty_id: str
    forward_tx_id: str
    return_tx_id: str
    forward_amount: float
    return_amount: float
    cycle_hours: float
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize circular flow alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "origin_user_id": self.origin_user_id,
            "counterparty_id": self.counterparty_id,
            "forward_tx_id": self.forward_tx_id,
            "return_tx_id": self.return_tx_id,
            "forward_amount": round(self.forward_amount, 2),
            "return_amount": round(self.return_amount, 2),
            "cycle_hours": round(self.cycle_hours, 2),
            "detected_at": self.detected_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CircularFlowAlert:
        """Deserialize circular flow alert from dictionary."""
        detected_at_raw = data.get("detected_at")
        detected_at = (
            datetime.fromisoformat(detected_at_raw)
            if isinstance(detected_at_raw, str)
            else datetime.now(UTC)
        )
        return cls(
            alert_id=str(data.get("alert_id", uuid4())),
            origin_user_id=str(data["origin_user_id"]),
            counterparty_id=str(data["counterparty_id"]),
            forward_tx_id=str(data["forward_tx_id"]),
            return_tx_id=str(data["return_tx_id"]),
            forward_amount=float(data["forward_amount"]),
            return_amount=float(data["return_amount"]),
            cycle_hours=float(data["cycle_hours"]),
            detected_at=detected_at,
        )


class CircularFlowDetector:
    """Detects round-trip / circular laundering flows between users and counterparties."""

    def __init__(
        self,
        window_seconds: float = 172800.0,
        min_amount: float = 1000.0,
        amount_tolerance: float = 0.20,
    ) -> None:
        self.window_seconds = window_seconds
        self.min_amount = min_amount
        self.amount_tolerance = amount_tolerance
        self._lock = threading.RLock()
        self._transfers: dict[tuple[str, str], deque[tuple[str, float, datetime]]] = defaultdict(
            deque
        )

    def record_and_evaluate(
        self,
        event: AmlTransactionEvent,
    ) -> CircularFlowAlert | None:
        """Record transfer and evaluate whether reverse flow creates a circular loop."""
        if not event.counterparty_id or event.amount < self.min_amount:
            return None

        with self._lock:
            if event.direction == TransactionDirection.OUTFLOW:
                sender = event.user_id
                receiver = event.counterparty_id
            else:
                sender = event.counterparty_id
                receiver = event.user_id

            cutoff = event.timestamp - timedelta(seconds=self.window_seconds)

            reverse_pair = (receiver, sender)
            reverse_queue = self._transfers[reverse_pair]
            while reverse_queue and reverse_queue[0][2] < cutoff:
                reverse_queue.popleft()

            alert: CircularFlowAlert | None = None
            for prior_tx_id, prior_amt, prior_ts in reversed(reverse_queue):
                if prior_ts > event.timestamp:
                    continue
                diff = abs(event.amount - prior_amt) / max(prior_amt, 1.0)
                if diff <= self.amount_tolerance:
                    cycle_hrs = (event.timestamp - prior_ts).total_seconds() / 3600.0
                    alert = CircularFlowAlert(
                        origin_user_id=sender,
                        counterparty_id=receiver,
                        forward_tx_id=prior_tx_id,
                        return_tx_id=event.transaction_id,
                        forward_amount=prior_amt,
                        return_amount=event.amount,
                        cycle_hours=cycle_hrs,
                        detected_at=event.timestamp,
                    )
                    break

            pair = (sender, receiver)
            self._transfers[pair].append((event.transaction_id, event.amount, event.timestamp))
            return alert

    def clear(self) -> None:
        """Clear all transfer tracking queues."""
        with self._lock:
            self._transfers.clear()


@dataclass(frozen=True)
class JurisdictionRiskAssessment:
    """Evaluation of geographic corridor risk under AML regulations."""

    country: str
    is_high_risk: bool
    fatf_tier: str
    risk_score: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize jurisdiction risk assessment to dictionary."""
        return {
            "country": self.country,
            "is_high_risk": self.is_high_risk,
            "fatf_tier": self.fatf_tier,
            "risk_score": round(self.risk_score, 4),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JurisdictionRiskAssessment:
        """Deserialize jurisdiction risk assessment from dictionary."""
        return cls(
            country=str(data["country"]),
            is_high_risk=bool(data["is_high_risk"]),
            fatf_tier=str(data["fatf_tier"]),
            risk_score=float(data["risk_score"]),
            description=str(data.get("description", "")),
        )


class HighRiskJurisdictionDetector:
    """Detects transactions flowing through FATF non-cooperative or monitored jurisdictions."""

    DEFAULT_BLACK_LIST: frozenset[str] = frozenset({"KP", "IR", "MM"})
    DEFAULT_GRAY_LIST: frozenset[str] = frozenset(
        {"SY", "YE", "SS", "HT", "NG", "ML", "BF", "CD", "MZ"}
    )

    def __init__(
        self,
        custom_black_list: set[str] | None = None,
        custom_gray_list: set[str] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._black_list = set(custom_black_list or self.DEFAULT_BLACK_LIST)
        self._gray_list = set(custom_gray_list or self.DEFAULT_GRAY_LIST)

    def evaluate(self, country: str) -> JurisdictionRiskAssessment:
        """Evaluate a country code against FATF non-cooperative and monitored lists."""
        code = country.strip().upper()
        with self._lock:
            if code in self._black_list:
                return JurisdictionRiskAssessment(
                    country=code,
                    is_high_risk=True,
                    fatf_tier="black_list",
                    risk_score=1.0,
                    description=(
                        f"Jurisdiction {code} is on the FATF High-Risk Call for Action (Blacklist)."
                    ),
                )
            if code in self._gray_list:
                return JurisdictionRiskAssessment(
                    country=code,
                    is_high_risk=True,
                    fatf_tier="gray_list",
                    risk_score=0.60,
                    description=(
                        f"Jurisdiction {code} is on the FATF Increased Monitoring (Graylist)."
                    ),
                )
            return JurisdictionRiskAssessment(
                country=code,
                is_high_risk=False,
                fatf_tier="standard",
                risk_score=0.0,
                description=f"Jurisdiction {code} has standard regulatory compliance standing.",
            )

    def add_jurisdiction(self, country: str, tier: str = "gray_list") -> None:
        """Register an additional high-risk jurisdiction."""
        code = country.strip().upper()
        with self._lock:
            if tier == "black_list":
                self._black_list.add(code)
                self._gray_list.discard(code)
            else:
                self._gray_list.add(code)
                self._black_list.discard(code)

    def remove_jurisdiction(self, country: str) -> bool:
        """Remove a jurisdiction from high-risk monitoring."""
        code = country.strip().upper()
        with self._lock:
            removed_black = code in self._black_list
            removed_gray = code in self._gray_list
            self._black_list.discard(code)
            self._gray_list.discard(code)
            return removed_black or removed_gray


class AmlStore:
    """Thread-safe concurrent storage for AML alerts, SAR dossiers, and audit trails."""

    def __init__(self, max_capacity: int = 10_000) -> None:
        self.max_capacity = max_capacity
        self._lock = threading.RLock()
        self._alerts: dict[str, AmlAlert] = {}
        self._user_alerts: dict[str, list[str]] = defaultdict(list)
        self._sars: dict[str, SarDossier] = {}
        self._user_sars: dict[str, list[str]] = defaultdict(list)

    def record_alert(self, alert: AmlAlert) -> None:
        """Record an AML alert and index by alert ID and user ID."""
        with self._lock:
            if len(self._alerts) >= self.max_capacity:
                oldest_id = next(iter(self._alerts))
                evicted = self._alerts.pop(oldest_id)
                user_list = self._user_alerts.get(evicted.user_id, [])
                if oldest_id in user_list:
                    user_list.remove(oldest_id)

            self._alerts[alert.alert_id] = alert
            self._user_alerts[alert.user_id].append(alert.alert_id)

    def get_alert(self, alert_id: str) -> AmlAlert | None:
        """Retrieve an AML alert by ID."""
        with self._lock:
            return self._alerts.get(alert_id)

    def list_alerts(
        self,
        user_id: str | None = None,
        typology: AmlTypology | None = None,
        risk_level: AmlRiskLevel | None = None,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[AmlAlert]:
        """Query alerts matching optional filters, sorted by creation time descending."""
        with self._lock:
            if user_id:
                alert_ids = self._user_alerts.get(user_id, [])
                candidates = [self._alerts[aid] for aid in alert_ids if aid in self._alerts]
            else:
                candidates = list(self._alerts.values())

        filtered: list[AmlAlert] = []
        for alert in candidates:
            if typology and alert.typology != typology:
                continue
            if risk_level and alert.risk_level != risk_level:
                continue
            if tenant_id and alert.tenant_id != tenant_id:
                continue
            filtered.append(alert)

        filtered.sort(key=lambda a: a.created_at, reverse=True)
        return filtered[:limit]

    def record_sar(self, sar: SarDossier) -> None:
        """Record a SAR filing dossier."""
        with self._lock:
            if len(self._sars) >= self.max_capacity:
                oldest_id = next(iter(self._sars))
                evicted = self._sars.pop(oldest_id)
                user_list = self._user_sars.get(evicted.user_id, [])
                if oldest_id in user_list:
                    user_list.remove(oldest_id)

            self._sars[sar.sar_id] = sar
            if sar.sar_id not in self._user_sars[sar.user_id]:
                self._user_sars[sar.user_id].append(sar.sar_id)

    def get_sar(self, sar_id: str) -> SarDossier | None:
        """Retrieve a SAR filing dossier by ID."""
        with self._lock:
            return self._sars.get(sar_id)

    def list_sars(
        self,
        user_id: str | None = None,
        status: SarStatus | None = None,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[SarDossier]:
        """Query SAR dossiers matching filters, sorted by updated time descending."""
        with self._lock:
            if user_id:
                sar_ids = self._user_sars.get(user_id, [])
                candidates = [self._sars[sid] for sid in sar_ids if sid in self._sars]
            else:
                candidates = list(self._sars.values())

        filtered: list[SarDossier] = []
        for s in candidates:
            if status and s.status != status:
                continue
            if tenant_id and s.tenant_id != tenant_id:
                continue
            filtered.append(s)

        filtered.sort(key=lambda s: s.updated_at, reverse=True)
        return filtered[:limit]

    def update_sar_status(
        self,
        sar_id: str,
        new_status: SarStatus,
        notes: str = "",
        actor: str = "compliance_officer",
    ) -> SarDossier | None:
        """Transition the review or filing status of a SAR with an audit record."""
        with self._lock:
            sar = self._sars.get(sar_id)
            if sar is None:
                return None

            now = datetime.now(UTC)
            filing_date = (
                now.strftime("%Y-%m-%d") if new_status == SarStatus.FILED else sar.filing_date
            )
            audit_entry = {
                "timestamp": now.isoformat(),
                "from_status": sar.status.value,
                "to_status": new_status.value,
                "actor": actor,
                "notes": notes,
            }
            new_trail = (*sar.audit_trail, audit_entry)

            updated = SarDossier(
                sar_id=sar.sar_id,
                user_id=sar.user_id,
                narrative=sar.narrative,
                alert_ids=sar.alert_ids,
                status=new_status,
                suspect_details=sar.suspect_details,
                summary_financials=sar.summary_financials,
                jurisdictions=sar.jurisdictions,
                filing_date=filing_date,
                tenant_id=sar.tenant_id,
                created_at=sar.created_at,
                updated_at=now,
                audit_trail=new_trail,
            )
            self._sars[sar_id] = updated
            return updated

    def count_alerts(self) -> int:
        """Return total count of recorded alerts."""
        with self._lock:
            return len(self._alerts)

    def count_sars(self) -> int:
        """Return total count of recorded SAR dossiers."""
        with self._lock:
            return len(self._sars)

    def clear(self) -> None:
        """Clear all stored alerts and dossiers."""
        with self._lock:
            self._alerts.clear()
            self._user_alerts.clear()
            self._sars.clear()
            self._user_sars.clear()


@dataclass(frozen=True)
class AmlAssessment:
    """Consolidated assessment of AML risk, policy routing, and triggered typologies."""

    transaction_id: str
    user_id: str
    risk_score: float
    risk_level: AmlRiskLevel
    recommended_action: AmlAction
    triggered_typologies: tuple[AmlTypology, ...]
    score_breakdown: dict[str, float]
    screening_matches: tuple[ScreeningMatch, ...] = field(default_factory=tuple)
    alert: AmlAlert | None = None
    case_id: str | None = None
    tenant_id: str = "default"
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize AML assessment to dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "user_id": self.user_id,
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level.value,
            "recommended_action": self.recommended_action.value,
            "triggered_typologies": [t.value for t in self.triggered_typologies],
            "score_breakdown": {k: round(v, 4) for k, v in self.score_breakdown.items()},
            "screening_matches": [m.to_dict() for m in self.screening_matches],
            "alert": self.alert.to_dict() if self.alert else None,
            "case_id": self.case_id,
            "tenant_id": self.tenant_id,
            "evaluated_at": self.evaluated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AmlAssessment:
        """Deserialize AML assessment from dictionary."""
        eval_at_raw = data.get("evaluated_at")
        eval_at = (
            datetime.fromisoformat(eval_at_raw)
            if isinstance(eval_at_raw, str)
            else datetime.now(UTC)
        )
        alert_data = data.get("alert")
        alert = AmlAlert.from_dict(alert_data) if alert_data else None

        return cls(
            transaction_id=str(data["transaction_id"]),
            user_id=str(data["user_id"]),
            risk_score=float(data["risk_score"]),
            risk_level=AmlRiskLevel(data["risk_level"]),
            recommended_action=AmlAction(data["recommended_action"]),
            triggered_typologies=tuple(
                AmlTypology(t) for t in data.get("triggered_typologies", ())
            ),
            score_breakdown={k: float(v) for k, v in data.get("score_breakdown", {}).items()},
            screening_matches=tuple(
                ScreeningMatch.from_dict(m) for m in data.get("screening_matches", ())
            ),
            alert=alert,
            case_id=data.get("case_id"),
            tenant_id=str(data.get("tenant_id", "default")),
            evaluated_at=eval_at,
        )


class AmlEngine:
    """Multi-typology AML evaluation, calibrated risk scoring, and policy enforcement engine."""

    def __init__(
        self,
        screening_engine: WatchlistScreeningEngine | None = None,
        structuring_detector: StructuringDetector | None = None,
        layering_detector: LayeringDetector | None = None,
        circular_detector: CircularFlowDetector | None = None,
        jurisdiction_detector: HighRiskJurisdictionDetector | None = None,
        store: AmlStore | None = None,
        case_manager: CaseManager | None = None,
        event_bus: EventBus | None = None,
        pass_threshold: float = 0.30,
        monitor_threshold: float = 0.70,
        edd_threshold: float = 0.85,
    ) -> None:
        self.screening_engine = screening_engine or WatchlistScreeningEngine()
        self.structuring_detector = structuring_detector or StructuringDetector()
        self.layering_detector = layering_detector or LayeringDetector()
        self.circular_detector = circular_detector or CircularFlowDetector()
        self.jurisdiction_detector = jurisdiction_detector or HighRiskJurisdictionDetector()
        self.store = store or AmlStore()
        self.case_manager = case_manager
        self.event_bus = event_bus
        self.pass_threshold = pass_threshold
        self.monitor_threshold = monitor_threshold
        self.edd_threshold = edd_threshold
        self._lock = threading.RLock()

    def evaluate_transaction(
        self,
        event: AmlTransactionEvent,
        entity_name: str | None = None,
    ) -> AmlAssessment:
        """Evaluate financial transaction against all AML typologies, sanctions, and corridors."""
        triggered_typologies: list[AmlTypology] = []
        score_breakdown: dict[str, float] = {}
        trigger_details: dict[str, Any] = {}
        tx_ids: list[str] = [event.transaction_id]

        # 1. Sanctions / PEP Screening
        matches: list[ScreeningMatch] = []
        target_name = entity_name or event.counterparty_id
        if target_name:
            matches = self.screening_engine.screen_entity(target_name)
            if matches:
                top_match = matches[0]
                if top_match.similarity_score >= 0.85:
                    triggered_typologies.append(AmlTypology.SANCTION_PEP_MATCH)
                    match_weight = 1.0 if top_match.similarity_score >= 0.95 else 0.85
                    score_breakdown["sanction_pep_match"] = match_weight
                    trigger_details["top_match"] = top_match.to_dict()

        # 2. Structuring / Smurfing
        structuring_alert = self.structuring_detector.record_and_evaluate(event)
        if structuring_alert:
            triggered_typologies.append(AmlTypology.STRUCTURING_SMURFING)
            score_breakdown["structuring_smurfing"] = 0.45
            trigger_details["structuring"] = structuring_alert.to_dict()
            tx_ids.extend(list(structuring_alert.transaction_ids))

        # 3. Layering / Rapid Flow-Through
        layering_alert = self.layering_detector.record_and_evaluate(event)
        if layering_alert:
            triggered_typologies.append(AmlTypology.RAPID_MOVEMENT_LAYERING)
            score_breakdown["rapid_movement_layering"] = 0.40
            trigger_details["layering"] = layering_alert.to_dict()
            tx_ids.append(layering_alert.inflow_tx_id)

        # 4. Circular / Round-Trip Flow
        circular_alert = self.circular_detector.record_and_evaluate(event)
        if circular_alert:
            triggered_typologies.append(AmlTypology.ROUND_TRIP_CIRCULAR)
            score_breakdown["round_trip_circular"] = 0.35
            trigger_details["circular_flow"] = circular_alert.to_dict()
            tx_ids.append(circular_alert.forward_tx_id)

        # 5. High-Risk Jurisdiction Corridor
        jur_assessment = self.jurisdiction_detector.evaluate(event.country)
        if jur_assessment.is_high_risk:
            triggered_typologies.append(AmlTypology.HIGH_RISK_JURISDICTION)
            jur_weight = 0.50 if jur_assessment.fatf_tier == "black_list" else 0.30
            score_breakdown["high_risk_jurisdiction"] = jur_weight
            trigger_details["jurisdiction"] = jur_assessment.to_dict()

        # Calculate composite calibrated risk score [0.0, 1.0]
        if (
            "sanction_pep_match" in score_breakdown
            and score_breakdown["sanction_pep_match"] >= 0.95
        ):
            composite_score = 1.0
        else:
            composite_score = min(1.0, sum(score_breakdown.values()))

        # Policy Action Routing
        if composite_score >= self.edd_threshold or (
            matches and matches[0].similarity_score >= 0.95
        ):
            recommended_action = AmlAction.FILE_SAR_BLOCK
            risk_level = AmlRiskLevel.CRITICAL
        elif composite_score >= self.monitor_threshold:
            recommended_action = AmlAction.ENHANCED_DUE_DILIGENCE
            risk_level = AmlRiskLevel.HIGH
        elif composite_score >= self.pass_threshold:
            recommended_action = AmlAction.MONITOR
            risk_level = AmlRiskLevel.MEDIUM
        else:
            recommended_action = AmlAction.PASS
            risk_level = AmlRiskLevel.LOW

        alert: AmlAlert | None = None
        unique_tx_ids = tuple(sorted(set(tx_ids)))
        if recommended_action != AmlAction.PASS or triggered_typologies:
            primary_typology = (
                triggered_typologies[0] if triggered_typologies else AmlTypology.HIGH_VELOCITY_BURST
            )
            alert = AmlAlert(
                user_id=event.user_id,
                typology=primary_typology,
                risk_level=risk_level,
                recommended_action=recommended_action,
                risk_score=composite_score,
                trigger_details=trigger_details,
                transaction_ids=unique_tx_ids,
                tenant_id=event.tenant_id,
                created_at=event.timestamp,
            )
            self.store.record_alert(alert)

        # Case Escalation
        case_id: str | None = None
        if self.case_manager is not None and recommended_action in (
            AmlAction.ENHANCED_DUE_DILIGENCE,
            AmlAction.FILE_SAR_BLOCK,
        ):
            from fraud_detection.cases import CasePriority

            priority = (
                CasePriority.CRITICAL
                if recommended_action == AmlAction.FILE_SAR_BLOCK
                else CasePriority.HIGH
            )
            case_record = self.case_manager.create_case(
                transaction_id=event.transaction_id,
                fraud_score=composite_score,
                amount=event.amount,
                currency=event.currency,
                tenant_id=event.tenant_id,
                priority=priority,
                metadata={
                    "source": "aml_engine",
                    "action": recommended_action.value,
                    "typologies": [t.value for t in triggered_typologies],
                    "alert_id": alert.alert_id if alert else None,
                },
            )
            case_id = case_record.case_id

        # Streaming EventBus Notification
        if self.event_bus is not None:
            self._publish_events(
                event, composite_score, recommended_action, triggered_typologies, alert
            )

        return AmlAssessment(
            transaction_id=event.transaction_id,
            user_id=event.user_id,
            risk_score=composite_score,
            risk_level=risk_level,
            recommended_action=recommended_action,
            triggered_typologies=tuple(triggered_typologies),
            score_breakdown=score_breakdown,
            screening_matches=tuple(matches),
            alert=alert,
            case_id=case_id,
            tenant_id=event.tenant_id,
            evaluated_at=event.timestamp,
        )

    def _publish_events(
        self,
        event: AmlTransactionEvent,
        score: float,
        action: AmlAction,
        typologies: list[AmlTypology],
        alert: AmlAlert | None,
    ) -> None:
        if self.event_bus is None:
            return
        from fraud_detection.events import EventType, FraudDomainEvent

        payload = {
            "transaction_id": event.transaction_id,
            "user_id": event.user_id,
            "amount": event.amount,
            "score": round(score, 4),
            "action": action.value,
            "typologies": [t.value for t in typologies],
            "alert_id": alert.alert_id if alert else None,
            "tenant_id": event.tenant_id,
        }
        self.event_bus.publish(
            FraudDomainEvent(
                event_type=EventType.AML_TRANSACTION_MONITORED,
                payload=payload,
                tenant_id=event.tenant_id,
            )
        )
        if alert is not None:
            self.event_bus.publish(
                FraudDomainEvent(
                    event_type=EventType.AML_ALERT_TRIGGERED,
                    payload=alert.to_dict(),
                    tenant_id=event.tenant_id,
                )
            )
        if AmlTypology.SANCTION_PEP_MATCH in typologies:
            self.event_bus.publish(
                FraudDomainEvent(
                    event_type=EventType.AML_SANCTION_MATCH_DETECTED,
                    payload=payload,
                    tenant_id=event.tenant_id,
                )
            )
