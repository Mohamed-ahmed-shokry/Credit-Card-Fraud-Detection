"""Anti-Money Laundering (AML) Transaction Monitoring, Typology Detection & Regulatory SAR Engine.

Provides domain models, sanctions/PEP fuzzy watchlist screening, typology anomaly detectors
(structuring/smurfing, layering flow-through, circular fund movement, FATF high-risk corridors),
composite AML risk decisioning, and regulatory Suspicious Activity Report (SAR) dossier management.
"""

from __future__ import annotations

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
