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
