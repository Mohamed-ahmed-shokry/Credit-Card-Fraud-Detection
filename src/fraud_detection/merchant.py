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
    "6012": MCCRiskProfile(
        "6012", MCCCategory.MONEY_TRANSFER, 0.025, 0.70, "Financial Merchandise"
    ),
    "5816": MCCRiskProfile("5816", MCCCategory.DIGITAL_GOODS, 0.028, 0.75, "Digital Games"),
    "5815": MCCRiskProfile("5815", MCCCategory.DIGITAL_GOODS, 0.022, 0.65, "Digital Media"),
    "5944": MCCRiskProfile(
        "5944", MCCCategory.LUXURY_ELECTRONICS, 0.020, 0.60, "Jewelry and Watches"
    ),
    "5732": MCCRiskProfile(
        "5732", MCCCategory.LUXURY_ELECTRONICS, 0.018, 0.55, "Electronics Stores"
    ),
    "4511": MCCRiskProfile("4511", MCCCategory.TRAVEL_HOSPITALITY, 0.012, 0.40, "Airlines"),
    "7011": MCCRiskProfile(
        "7011", MCCCategory.TRAVEL_HOSPITALITY, 0.010, 0.35, "Hotels and Lodging"
    ),
    "5311": MCCRiskProfile("5311", MCCCategory.RETAIL_COMMERCE, 0.005, 0.20, "Department Stores"),
    "5812": MCCRiskProfile("5812", MCCCategory.RETAIL_COMMERCE, 0.004, 0.15, "Restaurants"),
    "5411": MCCRiskProfile("5411", MCCCategory.GROCERY_UTILITIES, 0.002, 0.08, "Grocery Stores"),
    "4900": MCCRiskProfile(
        "4900", MCCCategory.GROCERY_UTILITIES, 0.001, 0.05, "Electric Utilities"
    ),
}


def resolve_mcc_profile(mcc: str | int | None) -> MCCRiskProfile:
    """Resolve an ISO 18245 4-digit MCC to a risk profile using known codes or range heuristics."""
    if mcc is None or isinstance(mcc, bool):
        return MCCRiskProfile(
            "0000", MCCCategory.UNKNOWN, 0.010, 0.30, "Unknown or Unspecified MCC"
        )
    raw = str(mcc).strip().removesuffix(".0")
    if not raw:
        return MCCRiskProfile(
            "0000", MCCCategory.UNKNOWN, 0.010, 0.30, "Unknown or Unspecified MCC"
        )
    if raw.isdigit() and len(raw) < 4:
        raw = raw.zfill(4)
    if not (raw.isdigit() and len(raw) == 4):
        return MCCRiskProfile(
            "0000", MCCCategory.UNKNOWN, 0.010, 0.30, "Unknown or Unspecified MCC"
        )

    if raw in DEFAULT_MCC_PROFILES:
        return DEFAULT_MCC_PROFILES[raw]

    code_num = int(raw)
    if 3000 <= code_num <= 4799 or 7000 <= code_num <= 7099:
        return MCCRiskProfile(raw, MCCCategory.TRAVEL_HOSPITALITY, 0.010, 0.35, "Travel & Lodging")
    if 4800 <= code_num <= 4999:
        return MCCRiskProfile(
            raw, MCCCategory.GROCERY_UTILITIES, 0.002, 0.10, "Utilities & Telecom"
        )
    if 5000 <= code_num <= 5999:
        return MCCRiskProfile(raw, MCCCategory.RETAIL_COMMERCE, 0.005, 0.20, "Retail & Wholesale")
    if 6000 <= code_num <= 6999:
        return MCCRiskProfile(raw, MCCCategory.MONEY_TRANSFER, 0.025, 0.70, "Financial Services")
    if 7800 <= code_num <= 7999:
        return MCCRiskProfile(raw, MCCCategory.GAMBLING_CRYPTO, 0.035, 0.80, "Amusement & Gaming")
    return MCCRiskProfile(raw, MCCCategory.UNKNOWN, 0.008, 0.25, "General Commercial Services")


def calculate_empirical_bayes_fraud_rate(
    fraud_count: int,
    total_count: int,
    prior_rate: float = 0.01,
    prior_strength: float = 20.0,
) -> float:
    """Compute Beta-Binomial Empirical Bayes credibility-smoothed merchant fraud rate."""
    if isinstance(fraud_count, bool) or not isinstance(fraud_count, int) or fraud_count < 0:
        raise MerchantRiskError("fraud_count must be a non-negative integer.")
    if isinstance(total_count, bool) or not isinstance(total_count, int) or total_count < 0:
        raise MerchantRiskError("total_count must be a non-negative integer.")
    if fraud_count > total_count:
        raise MerchantRiskError("fraud_count cannot exceed total_count.")
    if (
        isinstance(prior_rate, bool)
        or not isinstance(prior_rate, (int, float))
        or not math.isfinite(prior_rate)
        or not (0.0 <= prior_rate <= 1.0)
    ):
        raise MerchantRiskError("prior_rate must be a finite float in [0.0, 1.0].")
    if (
        isinstance(prior_strength, bool)
        or not isinstance(prior_strength, (int, float))
        or not math.isfinite(prior_strength)
        or prior_strength <= 0.0
    ):
        raise MerchantRiskError("prior_strength must be a positive finite float.")

    numerator = float(fraud_count) + float(prior_strength) * float(prior_rate)
    denominator = float(total_count) + float(prior_strength)
    return min(1.0, max(0.0, numerator / denominator))


def calculate_wilson_upper_bound(
    fraud_count: int,
    total_count: int,
    z_value: float = 1.96,
) -> float:
    """Compute the Wilson score upper confidence bound for merchant chargeback rate."""
    if isinstance(fraud_count, bool) or not isinstance(fraud_count, int) or fraud_count < 0:
        raise MerchantRiskError("fraud_count must be a non-negative integer.")
    if isinstance(total_count, bool) or not isinstance(total_count, int) or total_count < 0:
        raise MerchantRiskError("total_count must be a non-negative integer.")
    if fraud_count > total_count:
        raise MerchantRiskError("fraud_count cannot exceed total_count.")
    if (
        isinstance(z_value, bool)
        or not isinstance(z_value, (int, float))
        or not math.isfinite(z_value)
        or z_value <= 0.0
    ):
        raise MerchantRiskError("z_value must be a positive finite float.")
    if total_count == 0:
        return 0.0

    n = float(total_count)
    p = float(fraud_count) / n
    z2 = float(z_value) ** 2
    center = p + z2 / (2.0 * n)
    margin = float(z_value) * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    denom = 1.0 + z2 / n
    return min(1.0, max(0.0, (center + margin) / denom))


@dataclass(frozen=True)
class MerchantTransactionEvent:
    """Single merchant transaction observation for sliding-window risk tracking."""

    merchant_id: str
    timestamp: float
    amount: float
    mcc: str = "0000"
    card_id: str | None = None
    is_fraud: bool = False
    is_chargeback: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.merchant_id, str) or not self.merchant_id.strip():
            raise MerchantRiskError("merchant_id must be a non-empty string.")
        object.__setattr__(self, "merchant_id", self.merchant_id.strip())
        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, (int, float))
            or not math.isfinite(self.timestamp)
        ):
            raise MerchantRiskError("timestamp must be a finite number.")
        if (
            isinstance(self.amount, bool)
            or not isinstance(self.amount, (int, float))
            or not math.isfinite(self.amount)
            or self.amount < 0.0
        ):
            raise MerchantRiskError("amount must be a non-negative finite number.")

        mcc_prof = resolve_mcc_profile(self.mcc)
        object.__setattr__(self, "mcc", mcc_prof.mcc)

        if self.card_id is not None:
            if not isinstance(self.card_id, str) or not self.card_id.strip():
                raise MerchantRiskError("card_id must be a non-empty string or None.")
            object.__setattr__(self, "card_id", self.card_id.strip())

        if not isinstance(self.is_fraud, bool):
            raise MerchantRiskError("is_fraud must be a boolean.")
        if not isinstance(self.is_chargeback, bool):
            raise MerchantRiskError("is_chargeback must be a boolean.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize MerchantTransactionEvent to dictionary."""
        return {
            "merchant_id": self.merchant_id,
            "timestamp": float(self.timestamp),
            "amount": float(self.amount),
            "mcc": self.mcc,
            "card_id": self.card_id,
            "is_fraud": self.is_fraud,
            "is_chargeback": self.is_chargeback,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MerchantTransactionEvent:
        """Construct MerchantTransactionEvent from mapping."""
        if not isinstance(data, Mapping):
            raise MerchantRiskError("MerchantTransactionEvent data must be a mapping.")
        try:
            raw_card = data.get("card_id")
            card_str = (
                str(raw_card).strip() if raw_card is not None and str(raw_card).strip() else None
            )
            return cls(
                merchant_id=str(data["merchant_id"]),
                timestamp=float(data.get("timestamp", 0.0)),
                amount=float(data.get("amount", 0.0)),
                mcc=str(data.get("mcc", "0000")),
                card_id=card_str,
                is_fraud=bool(data.get("is_fraud", False)),
                is_chargeback=bool(data.get("is_chargeback", False)),
            )
        except (KeyError, ValueError, TypeError) as exc:
            if isinstance(exc, MerchantRiskError):
                raise
            raise MerchantRiskError(f"Invalid MerchantTransactionEvent data: {exc}") from exc


@dataclass(frozen=True)
class MerchantTrackerConfig:
    """Configuration for sliding-window merchant risk tracking and anomaly thresholds."""

    window_seconds: float = 86400.0
    max_merchants: int = 50000
    prior_strength: float = 20.0
    chargeback_rate_threshold: float = 0.015
    min_chargeback_count: int = 3
    bust_out_zscore_threshold: float = 3.0
    min_history_for_zscore: int = 5
    micro_tx_amount_cutoff: float = 5.0
    micro_tx_ratio_threshold: float = 0.50
    min_distinct_cards_for_testing: int = 5

    def __post_init__(self) -> None:
        if (
            isinstance(self.window_seconds, bool)
            or not isinstance(self.window_seconds, (int, float))
            or not math.isfinite(self.window_seconds)
            or self.window_seconds <= 0.0
        ):
            raise MerchantRiskError("window_seconds must be a positive finite float.")
        if (
            isinstance(self.max_merchants, bool)
            or not isinstance(self.max_merchants, int)
            or self.max_merchants <= 0
        ):
            raise MerchantRiskError("max_merchants must be a positive integer.")
        if (
            isinstance(self.prior_strength, bool)
            or not isinstance(self.prior_strength, (int, float))
            or not math.isfinite(self.prior_strength)
            or self.prior_strength <= 0.0
        ):
            raise MerchantRiskError("prior_strength must be a positive finite float.")
        if (
            isinstance(self.chargeback_rate_threshold, bool)
            or not isinstance(self.chargeback_rate_threshold, (int, float))
            or not math.isfinite(self.chargeback_rate_threshold)
            or not (0.0 < self.chargeback_rate_threshold <= 1.0)
        ):
            raise MerchantRiskError("chargeback_rate_threshold must be in (0.0, 1.0].")
        if (
            isinstance(self.min_chargeback_count, bool)
            or not isinstance(self.min_chargeback_count, int)
            or self.min_chargeback_count <= 0
        ):
            raise MerchantRiskError("min_chargeback_count must be a positive integer.")
        if (
            isinstance(self.bust_out_zscore_threshold, bool)
            or not isinstance(self.bust_out_zscore_threshold, (int, float))
            or not math.isfinite(self.bust_out_zscore_threshold)
            or self.bust_out_zscore_threshold <= 0.0
        ):
            raise MerchantRiskError("bust_out_zscore_threshold must be a positive finite float.")
        if (
            isinstance(self.min_history_for_zscore, bool)
            or not isinstance(self.min_history_for_zscore, int)
            or self.min_history_for_zscore < 2
        ):
            raise MerchantRiskError("min_history_for_zscore must be an integer >= 2.")
        if (
            isinstance(self.micro_tx_amount_cutoff, bool)
            or not isinstance(self.micro_tx_amount_cutoff, (int, float))
            or not math.isfinite(self.micro_tx_amount_cutoff)
            or self.micro_tx_amount_cutoff < 0.0
        ):
            raise MerchantRiskError("micro_tx_amount_cutoff must be a non-negative finite float.")
        if (
            isinstance(self.micro_tx_ratio_threshold, bool)
            or not isinstance(self.micro_tx_ratio_threshold, (int, float))
            or not math.isfinite(self.micro_tx_ratio_threshold)
            or not (0.0 < self.micro_tx_ratio_threshold <= 1.0)
        ):
            raise MerchantRiskError("micro_tx_ratio_threshold must be in (0.0, 1.0].")
        if (
            isinstance(self.min_distinct_cards_for_testing, bool)
            or not isinstance(self.min_distinct_cards_for_testing, int)
            or self.min_distinct_cards_for_testing <= 0
        ):
            raise MerchantRiskError("min_distinct_cards_for_testing must be a positive integer.")


@dataclass(frozen=True)
class MerchantProfile:
    """Snapshot of a merchant's sliding-window risk and ticket statistics."""

    merchant_id: str
    mcc: str
    mcc_category: str
    first_seen_timestamp: float
    last_seen_timestamp: float
    transaction_count: int
    fraud_count: int
    chargeback_count: int
    total_amount: float
    mean_amount: float
    std_amount: float
    distinct_cards_count: int
    micro_tx_count: int
    smoothed_fraud_rate: float
    wilson_upper_bound: float
    mcc_risk_score: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize MerchantProfile to dictionary."""
        return {
            "merchant_id": self.merchant_id,
            "mcc": self.mcc,
            "mcc_category": self.mcc_category,
            "first_seen_timestamp": float(self.first_seen_timestamp),
            "last_seen_timestamp": float(self.last_seen_timestamp),
            "transaction_count": int(self.transaction_count),
            "fraud_count": int(self.fraud_count),
            "chargeback_count": int(self.chargeback_count),
            "total_amount": round(float(self.total_amount), 4),
            "mean_amount": round(float(self.mean_amount), 4),
            "std_amount": round(float(self.std_amount), 4),
            "distinct_cards_count": int(self.distinct_cards_count),
            "micro_tx_count": int(self.micro_tx_count),
            "smoothed_fraud_rate": round(float(self.smoothed_fraud_rate), 6),
            "wilson_upper_bound": round(float(self.wilson_upper_bound), 6),
            "mcc_risk_score": round(float(self.mcc_risk_score), 4),
        }


@dataclass(frozen=True)
class MerchantAnomalyReport:
    """Result of evaluating a merchant transaction against chargeback and bust-out heuristics."""

    merchant_id: str
    mcc: str
    is_high_chargeback: bool
    is_bust_out_surge: bool
    is_card_testing: bool
    is_ticket_outlier: bool
    smoothed_fraud_rate: float
    amount_zscore: float
    micro_tx_ratio: float
    distinct_cards_count: int
    transaction_count: int
    severity: str
    reasons: tuple[str, ...] = ()

    @property
    def has_anomaly(self) -> bool:
        """Return True if any merchant anomaly heuristic fired."""
        return (
            self.is_high_chargeback
            or self.is_bust_out_surge
            or self.is_card_testing
            or self.is_ticket_outlier
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize MerchantAnomalyReport to dictionary."""
        return {
            "merchant_id": self.merchant_id,
            "mcc": self.mcc,
            "has_anomaly": self.has_anomaly,
            "is_high_chargeback": self.is_high_chargeback,
            "is_bust_out_surge": self.is_bust_out_surge,
            "is_card_testing": self.is_card_testing,
            "is_ticket_outlier": self.is_ticket_outlier,
            "smoothed_fraud_rate": round(float(self.smoothed_fraud_rate), 6),
            "amount_zscore": round(float(self.amount_zscore), 4),
            "micro_tx_ratio": round(float(self.micro_tx_ratio), 4),
            "distinct_cards_count": int(self.distinct_cards_count),
            "transaction_count": int(self.transaction_count),
            "severity": self.severity,
            "reasons": list(self.reasons),
        }
