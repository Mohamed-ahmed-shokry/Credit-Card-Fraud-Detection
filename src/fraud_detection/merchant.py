"""Real-time merchant and MCC risk profiling, chargeback surveillance, and anomaly detection."""

from __future__ import annotations

import math
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
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


@dataclass
class _MerchantState:
    """Internal mutable state for a tracked merchant."""

    merchant_id: str
    mcc: str
    first_seen_timestamp: float
    last_seen_timestamp: float
    transaction_count: int = 0
    fraud_count: int = 0
    chargeback_count: int = 0
    total_amount: float = 0.0
    mean_amount: float = 0.0
    m2_amount: float = 0.0
    events: deque[MerchantTransactionEvent] = field(default_factory=deque)

    @property
    def std_amount(self) -> float:
        """Population standard deviation of observed transaction amounts."""
        if self.transaction_count < 2 or self.m2_amount <= 0.0:
            return 0.0
        return math.sqrt(self.m2_amount / float(self.transaction_count))


class MerchantTracker:
    """Stateful merchant risk and chargeback surveillance tracker with LRU bounding."""

    def __init__(self, config: MerchantTrackerConfig | None = None) -> None:
        self.config = config or MerchantTrackerConfig()
        self._merchants: OrderedDict[str, _MerchantState] = OrderedDict()
        self._total_evaluations: int = 0
        self._total_anomalies: int = 0

    def _evict_expired(self, state: _MerchantState, current_timestamp: float) -> None:
        """Remove events outside the sliding window."""
        cutoff = current_timestamp - self.config.window_seconds
        while state.events and state.events[0].timestamp < cutoff:
            state.events.popleft()

    def _update_welford(self, state: _MerchantState, amount: float) -> None:
        """Update running Welford mean and variance accumulator for ticket sizes."""
        state.transaction_count += 1
        state.total_amount += amount
        delta = amount - state.mean_amount
        state.mean_amount += delta / float(state.transaction_count)
        delta2 = amount - state.mean_amount
        state.m2_amount += delta * delta2

    def _build_profile(self, state: _MerchantState) -> MerchantProfile:
        """Construct an immutable MerchantProfile snapshot from internal state."""
        mcc_profile = resolve_mcc_profile(state.mcc)
        distinct_cards = len({ev.card_id for ev in state.events if ev.card_id})
        micro_count = sum(
            1 for ev in state.events if ev.amount <= self.config.micro_tx_amount_cutoff
        )
        adverse_count = max(state.fraud_count, state.chargeback_count)
        smoothed_rate = calculate_empirical_bayes_fraud_rate(
            fraud_count=adverse_count,
            total_count=state.transaction_count,
            prior_rate=mcc_profile.baseline_fraud_rate,
            prior_strength=self.config.prior_strength,
        )
        wilson_upper = calculate_wilson_upper_bound(
            fraud_count=adverse_count,
            total_count=state.transaction_count,
        )
        return MerchantProfile(
            merchant_id=state.merchant_id,
            mcc=mcc_profile.mcc,
            mcc_category=str(mcc_profile.category),
            first_seen_timestamp=state.first_seen_timestamp,
            last_seen_timestamp=state.last_seen_timestamp,
            transaction_count=state.transaction_count,
            fraud_count=state.fraud_count,
            chargeback_count=state.chargeback_count,
            total_amount=state.total_amount,
            mean_amount=state.mean_amount,
            std_amount=state.std_amount,
            distinct_cards_count=distinct_cards,
            micro_tx_count=micro_count,
            smoothed_fraud_rate=smoothed_rate,
            wilson_upper_bound=wilson_upper,
            mcc_risk_score=mcc_profile.risk_score,
        )

    def get_profile(self, merchant_id: str) -> MerchantProfile | None:
        """Return the current profile for merchant_id, or None if not tracked."""
        normalized_id = str(merchant_id).strip()
        state = self._merchants.get(normalized_id)
        if state is None:
            return None
        return self._build_profile(state)

    def get_tracked_merchant_count(self) -> int:
        """Return the number of currently tracked merchants."""
        return len(self._merchants)

    def get_stats(self) -> dict[str, Any]:
        """Return operational telemetry statistics for the merchant tracker."""
        return {
            "tracked_merchants": len(self._merchants),
            "max_merchants": self.config.max_merchants,
            "window_seconds": self.config.window_seconds,
            "total_evaluations": self._total_evaluations,
            "total_anomalies": self._total_anomalies,
        }

    def reset(self) -> int:
        """Clear all tracked merchant state and return the count of cleared merchants."""
        cleared = len(self._merchants)
        self._merchants.clear()
        self._total_evaluations = 0
        self._total_anomalies = 0
        return cleared

    def assess_event(self, event: MerchantTransactionEvent) -> MerchantAnomalyReport:
        """Evaluate a merchant transaction event without mutating state."""
        state = self._merchants.get(event.merchant_id)
        cutoff = event.timestamp - self.config.window_seconds

        if state is not None:
            active_events = [ev for ev in state.events if ev.timestamp >= cutoff]
            prior_tx_count = state.transaction_count
            prior_fraud_count = state.fraud_count
            prior_cb_count = state.chargeback_count
        else:
            active_events = []
            prior_tx_count = 0
            prior_fraud_count = 0
            prior_cb_count = 0

        amount_zscore = 0.0
        if state is not None and prior_tx_count >= self.config.min_history_for_zscore:
            if state.std_amount > 1e-6:
                amount_zscore = (event.amount - state.mean_amount) / state.std_amount
            elif state.mean_amount > 0.0 and event.amount > state.mean_amount:
                denom = max(state.mean_amount * 0.1, 1.0)
                amount_zscore = (event.amount - state.mean_amount) / denom

        window_tx_count = len(active_events) + 1
        card_set = {ev.card_id for ev in active_events if ev.card_id}
        if event.card_id:
            card_set.add(event.card_id)
        distinct_cards_count = len(card_set)

        micro_tx_count = sum(
            1 for ev in active_events if ev.amount <= self.config.micro_tx_amount_cutoff
        ) + (1 if event.amount <= self.config.micro_tx_amount_cutoff else 0)
        micro_tx_ratio = float(micro_tx_count) / float(window_tx_count)

        total_tx = prior_tx_count + 1
        total_fraud = prior_fraud_count + (1 if event.is_fraud else 0)
        total_cb = prior_cb_count + (1 if event.is_chargeback else 0)
        adverse_count = max(total_fraud, total_cb)

        mcc_profile = resolve_mcc_profile(event.mcc)
        smoothed_fraud_rate = calculate_empirical_bayes_fraud_rate(
            fraud_count=adverse_count,
            total_count=total_tx,
            prior_rate=mcc_profile.baseline_fraud_rate,
            prior_strength=self.config.prior_strength,
        )

        is_high_chargeback = (
            adverse_count >= self.config.min_chargeback_count
            and smoothed_fraud_rate >= self.config.chargeback_rate_threshold
        )
        is_ticket_outlier = amount_zscore >= self.config.bust_out_zscore_threshold
        is_card_testing = (
            distinct_cards_count >= self.config.min_distinct_cards_for_testing
            and micro_tx_ratio >= self.config.micro_tx_ratio_threshold
        )
        is_bust_out_surge = is_ticket_outlier and (
            distinct_cards_count >= 3 or is_high_chargeback or mcc_profile.risk_score >= 0.70
        )

        reasons: list[str] = []
        if is_high_chargeback:
            reasons.append(f"high_chargeback_rate:{smoothed_fraud_rate:.4f}")
        if is_bust_out_surge:
            reasons.append(f"bust_out_surge:zscore={amount_zscore:.2f}")
        if is_card_testing:
            reasons.append(
                f"card_testing_micro_tx:cards={distinct_cards_count},ratio={micro_tx_ratio:.2f}"
            )
        if is_ticket_outlier and not is_bust_out_surge:
            reasons.append(f"ticket_size_outlier:zscore={amount_zscore:.2f}")

        flag_count = sum(
            [is_high_chargeback, is_bust_out_surge, is_card_testing, is_ticket_outlier]
        )
        if flag_count == 0:
            severity = "LOW"
        elif (is_high_chargeback and is_bust_out_surge) or flag_count >= 3:
            severity = "CRITICAL"
        elif is_high_chargeback or is_bust_out_surge or is_card_testing:
            severity = "HIGH"
        else:
            severity = "MEDIUM"

        return MerchantAnomalyReport(
            merchant_id=event.merchant_id,
            mcc=mcc_profile.mcc,
            is_high_chargeback=is_high_chargeback,
            is_bust_out_surge=is_bust_out_surge,
            is_card_testing=is_card_testing,
            is_ticket_outlier=is_ticket_outlier,
            smoothed_fraud_rate=smoothed_fraud_rate,
            amount_zscore=amount_zscore,
            micro_tx_ratio=micro_tx_ratio,
            distinct_cards_count=distinct_cards_count,
            transaction_count=total_tx,
            severity=severity,
            reasons=tuple(reasons),
        )

    def record_event(
        self, event: MerchantTransactionEvent
    ) -> tuple[MerchantProfile, MerchantAnomalyReport]:
        """Evaluate and record a merchant transaction event into sliding-window state."""
        report = self.assess_event(event)
        self._total_evaluations += 1
        if report.has_anomaly:
            self._total_anomalies += 1

        state = self._merchants.get(event.merchant_id)
        if state is None:
            if len(self._merchants) >= self.config.max_merchants:
                self._merchants.popitem(last=False)
            state = _MerchantState(
                merchant_id=event.merchant_id,
                mcc=event.mcc,
                first_seen_timestamp=event.timestamp,
                last_seen_timestamp=event.timestamp,
            )
            self._merchants[event.merchant_id] = state
        else:
            self._merchants.move_to_end(event.merchant_id)
            state.mcc = event.mcc
            state.first_seen_timestamp = min(state.first_seen_timestamp, event.timestamp)
            state.last_seen_timestamp = max(state.last_seen_timestamp, event.timestamp)

        self._evict_expired(state, state.last_seen_timestamp)
        state.events.append(event)
        self._update_welford(state, event.amount)
        if event.is_fraud:
            state.fraud_count += 1
        if event.is_chargeback:
            state.chargeback_count += 1

        profile = self._build_profile(state)
        return profile, report
