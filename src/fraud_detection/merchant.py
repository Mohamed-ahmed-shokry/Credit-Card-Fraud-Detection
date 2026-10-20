"""Real-time merchant and MCC risk profiling, chargeback surveillance, and anomaly detection."""

from __future__ import annotations

from enum import StrEnum


class MerchantRiskError(ValueError):
    """Raised when merchant attributes, MCC codes, or risk tracking parameters are invalid."""


class MCCCategory(StrEnum):
    """High-level ISO 18245 Merchant Category Code risk taxonomy."""

    GAMBLING_CRYPTO = "gambling_crypto"
    MONEY_TRANSFER = "money_transfer"
    DIGITAL_GOODS = "digital_goods"
    LUXURY_ELECTRONICS = "luxury_electronics"
    TRAVEL_HOSPITALITY = "travel_hospitality"
    RETAIL_COMMERCE = "retail_commerce"
    GROCERY_UTILITIES = "grocery_utilities"
    UNKNOWN = "unknown"
