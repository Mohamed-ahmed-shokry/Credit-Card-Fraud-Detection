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
from typing import Any
from uuid import uuid4


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
