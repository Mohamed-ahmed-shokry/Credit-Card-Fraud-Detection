"""Real-time merchant and MCC risk profiling, chargeback surveillance, and anomaly detection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


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


@dataclass(frozen=True)
class MCCRiskProfile:
    """ISO 18245 Merchant Category Code classification and baseline risk priors."""

    mcc: str
    category: MCCCategory
    baseline_fraud_rate: float
    risk_score: float
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.mcc, str) or not self.mcc.strip():
            raise MerchantRiskError("mcc must be a non-empty string.")
        cleaned = self.mcc.strip()
        if not (cleaned.isdigit() and len(cleaned) == 4):
            raise MerchantRiskError("mcc must be a 4-digit numeric string (ISO 18245).")
        object.__setattr__(self, "mcc", cleaned)

        raw_cat: Any = self.category
        if not isinstance(raw_cat, MCCCategory):
            try:
                object.__setattr__(self, "category", MCCCategory(str(raw_cat)))
            except ValueError as exc:
                raise MerchantRiskError(f"Invalid MCCCategory: {raw_cat!r}") from exc

        if (
            isinstance(self.baseline_fraud_rate, bool)
            or not isinstance(self.baseline_fraud_rate, (int, float))
            or not math.isfinite(self.baseline_fraud_rate)
            or not (0.0 <= self.baseline_fraud_rate <= 1.0)
        ):
            raise MerchantRiskError("baseline_fraud_rate must be a finite float in [0.0, 1.0].")

        if (
            isinstance(self.risk_score, bool)
            or not isinstance(self.risk_score, (int, float))
            or not math.isfinite(self.risk_score)
            or not (0.0 <= self.risk_score <= 1.0)
        ):
            raise MerchantRiskError("risk_score must be a finite float in [0.0, 1.0].")

        if not isinstance(self.description, str):
            raise MerchantRiskError("description must be a string.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize MCCRiskProfile to dictionary."""
        return {
            "mcc": self.mcc,
            "category": self.category.value,
            "baseline_fraud_rate": float(self.baseline_fraud_rate),
            "risk_score": float(self.risk_score),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MCCRiskProfile:
        """Construct MCCRiskProfile from mapping."""
        if not isinstance(data, Mapping):
            raise MerchantRiskError("MCCRiskProfile data must be a mapping.")
        try:
            return cls(
                mcc=str(data["mcc"]),
                category=MCCCategory(str(data["category"])),
                baseline_fraud_rate=float(data["baseline_fraud_rate"]),
                risk_score=float(data["risk_score"]),
                description=str(data.get("description", "")),
            )
        except (KeyError, ValueError, TypeError) as exc:
            if isinstance(exc, MerchantRiskError):
                raise
            raise MerchantRiskError(f"Invalid MCCRiskProfile data: {exc}") from exc


DEFAULT_MCC_PROFILES: dict[str, MCCRiskProfile] = {
    "7995": MCCRiskProfile(
        "7995", MCCCategory.GAMBLING_CRYPTO, 0.045, 0.90, "Betting and Casino Gambling"
    ),
    "6051": MCCRiskProfile(
        "6051", MCCCategory.GAMBLING_CRYPTO, 0.040, 0.85, "Quasi-Cash and Cryptocurrency"
    ),
    "4829": MCCRiskProfile("4829", MCCCategory.MONEY_TRANSFER, 0.035, 0.80, "Wire Money Orders"),
    "6012": MCCRiskProfile("6012", MCCCategory.MONEY_TRANSFER, 0.025, 0.70, "Financial Merchandise"),
    "5816": MCCRiskProfile("5816", MCCCategory.DIGITAL_GOODS, 0.028, 0.75, "Digital Games"),
    "5815": MCCRiskProfile("5815", MCCCategory.DIGITAL_GOODS, 0.022, 0.65, "Digital Media"),
    "5944": MCCRiskProfile(
        "5944", MCCCategory.LUXURY_ELECTRONICS, 0.020, 0.60, "Jewelry and Watches"
    ),
    "5732": MCCRiskProfile("5732", MCCCategory.LUXURY_ELECTRONICS, 0.018, 0.55, "Electronics Stores"),
    "4511": MCCRiskProfile("4511", MCCCategory.TRAVEL_HOSPITALITY, 0.012, 0.40, "Airlines"),
    "7011": MCCRiskProfile("7011", MCCCategory.TRAVEL_HOSPITALITY, 0.010, 0.35, "Hotels and Lodging"),
    "5311": MCCRiskProfile("5311", MCCCategory.RETAIL_COMMERCE, 0.005, 0.20, "Department Stores"),
    "5812": MCCRiskProfile("5812", MCCCategory.RETAIL_COMMERCE, 0.004, 0.15, "Restaurants"),
    "5411": MCCRiskProfile("5411", MCCCategory.GROCERY_UTILITIES, 0.002, 0.08, "Grocery Stores"),
    "4900": MCCRiskProfile("4900", MCCCategory.GROCERY_UTILITIES, 0.001, 0.05, "Electric Utilities"),
}


def resolve_mcc_profile(mcc: str | int | None) -> MCCRiskProfile:
    """Resolve an ISO 18245 4-digit MCC to a risk profile using known codes or range heuristics."""
    if mcc is None or isinstance(mcc, bool):
        return MCCRiskProfile("0000", MCCCategory.UNKNOWN, 0.010, 0.30, "Unknown or Unspecified MCC")
    raw = str(mcc).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    if not raw:
        return MCCRiskProfile("0000", MCCCategory.UNKNOWN, 0.010, 0.30, "Unknown or Unspecified MCC")
    if raw.isdigit() and len(raw) < 4:
        raw = raw.zfill(4)
    if not (raw.isdigit() and len(raw) == 4):
        return MCCRiskProfile("0000", MCCCategory.UNKNOWN, 0.010, 0.30, "Unknown or Unspecified MCC")

    if raw in DEFAULT_MCC_PROFILES:
        return DEFAULT_MCC_PROFILES[raw]

    code_num = int(raw)
    if 3000 <= code_num <= 4799 or 7000 <= code_num <= 7099:
        return MCCRiskProfile(raw, MCCCategory.TRAVEL_HOSPITALITY, 0.010, 0.35, "Travel & Lodging")
    if 4800 <= code_num <= 4999:
        return MCCRiskProfile(raw, MCCCategory.GROCERY_UTILITIES, 0.002, 0.10, "Utilities & Telecom")
    if 5000 <= code_num <= 5999:
        return MCCRiskProfile(raw, MCCCategory.RETAIL_COMMERCE, 0.005, 0.20, "Retail & Wholesale")
    if 6000 <= code_num <= 6999:
        return MCCRiskProfile(raw, MCCCategory.MONEY_TRANSFER, 0.025, 0.70, "Financial Services")
    if 7800 <= code_num <= 7999:
        return MCCRiskProfile(raw, MCCCategory.GAMBLING_CRYPTO, 0.035, 0.80, "Amusement & Gaming")
    return MCCRiskProfile(raw, MCCCategory.UNKNOWN, 0.008, 0.25, "General Commercial Services")
