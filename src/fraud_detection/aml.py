"""Anti-Money Laundering (AML) Transaction Monitoring, Typology Detection & Regulatory SAR Engine.

Provides domain models, sanctions/PEP fuzzy watchlist screening, typology anomaly detectors
(structuring/smurfing, layering flow-through, circular fund movement, FATF high-risk corridors),
composite AML risk decisioning, and regulatory Suspicious Activity Report (SAR) dossier management.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
