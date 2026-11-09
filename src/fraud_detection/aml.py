"""Anti-Money Laundering (AML) Transaction Monitoring, Typology Detection & Regulatory SAR Engine.

Provides domain models, sanctions/PEP fuzzy watchlist screening, typology anomaly detectors
(structuring/smurfing, layering flow-through, circular fund movement, FATF high-risk corridors),
composite AML risk decisioning, and regulatory Suspicious Activity Report (SAR) dossier management.
"""

from __future__ import annotations

from enum import StrEnum


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
