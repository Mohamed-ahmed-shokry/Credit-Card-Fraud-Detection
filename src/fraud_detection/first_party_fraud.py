"""First-party fraud, refund abuse, and friendly fraud detection engine.

Detects legitimate cardholders and account owners exploiting merchant refund,
chargeback, and promotional policies via serial friendly-fraud disputes,
high-ratio wardrobing/return abuse, Item-Not-Received (INR) claim bursts,
and multi-account promo/referral cycling.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
import threading
from typing import Any

__all__ = [
    "WARDROBING_CATEGORIES",
    "ClaimOutcome",
    "FirstPartyAccountProfile",
    "FirstPartyAccountStore",
    "FirstPartyAction",
    "FirstPartyAnomalyType",
    "FirstPartyAssessment",
    "FirstPartyEventRecord",
    "FirstPartyRiskLevel",
    "FriendlyFraudAnalyzer",
    "FriendlyFraudMetrics",
    "OrderClaimType",
    "PromoAbuseDetector",
    "PromoAbuseMetrics",
    "RefundAbuseAnalyzer",
    "RefundAbuseMetrics",
]


class FirstPartyRiskLevel(StrEnum):
    """Risk severity tier for first-party fraud and refund abuse."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FirstPartyAction(StrEnum):
    """Recommended policy action for first-party abuse assessments."""

    ALLOW = "allow"
    REQUIRE_SIGNATURE_DELIVERY = "require_signature_delivery"
    RESTRICT_INSTANT_REFUND = "restrict_instant_refund"
    DENY_REFUND_MANUAL_REVIEW = "deny_refund_manual_review"
    REVOKE_PRIVILEGES_AND_BLOCK = "revoke_privileges_and_block"


class FirstPartyAnomalyType(StrEnum):
    """Specific behavioral indicators of first-party abuse."""

    SERIAL_FRIENDLY_FRAUD = "serial_friendly_fraud"
    HIGH_RETURN_RATIO = "high_return_ratio"
    WARDROBING_PATTERN = "wardrobing_pattern"
    INR_CLAIM_BURST = "inr_claim_burst"
    HIGH_VALUE_REFUND_SKEW = "high_value_refund_skew"
    PROMO_BONUS_ABUSE = "promo_bonus_abuse"
    LINKED_ACCOUNT_CYCLING = "linked_account_cycling"
    PRE_DISPUTE_VELOCITY = "pre_dispute_velocity"


class OrderClaimType(StrEnum):
    """Lifecycle event types tracked for first-party abuse analysis."""

    PURCHASE = "purchase"
    REFUND_RETURN = "refund_return"
    INR_CLAIM = "inr_claim"
    FRIENDLY_DISPUTE = "friendly_dispute"
    PROMO_REDEMPTION = "promo_redemption"


class ClaimOutcome(StrEnum):
    """Resolution status of a refund, INR claim, or dispute."""

    APPROVED = "approved"
    PENDING = "pending"
    DENIED = "denied"
    REVERSED = "reversed"


WARDROBING_CATEGORIES: frozenset[str] = frozenset(
    {
        "apparel",
        "luxury_fashion",
        "electronics",
        "cameras",
        "jewelry",
        "footwear",
    }
)


def _coerce_utc_datetime(value: datetime | str) -> datetime:
    """Parse an ISO-8601 string or normalize a datetime to UTC."""
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class FirstPartyEventRecord:
    """Single order, refund, INR claim, dispute, or promo event for an account."""

    event_id: str
    account_id: str
    claim_type: OrderClaimType
    amount: float
    timestamp: datetime
    order_id: str = ""
    currency: str = "USD"
    merchant_category: str = "general"
    outcome: ClaimOutcome = ClaimOutcome.APPROVED
    promo_code: str = ""
    shipping_address_hash: str = ""
    payment_instrument_hash: str = ""
    device_id: str = ""
    avs_cvv_matched: bool = False
    three_ds_authenticated: bool = False
    delivery_confirmed: bool = False
    signature_captured: bool = False
    tenant_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must be non-empty.")
        if not self.account_id:
            raise ValueError("account_id must be non-empty.")
        if self.amount < 0:
            raise ValueError("amount must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "account_id": self.account_id,
            "claim_type": self.claim_type.value,
            "amount": float(self.amount),
            "timestamp": self.timestamp.astimezone(UTC).isoformat(),
            "order_id": self.order_id,
            "currency": self.currency,
            "merchant_category": self.merchant_category,
            "outcome": self.outcome.value,
            "promo_code": self.promo_code,
            "shipping_address_hash": self.shipping_address_hash,
            "payment_instrument_hash": self.payment_instrument_hash,
            "device_id": self.device_id,
            "avs_cvv_matched": self.avs_cvv_matched,
            "three_ds_authenticated": self.three_ds_authenticated,
            "delivery_confirmed": self.delivery_confirmed,
            "signature_captured": self.signature_captured,
            "tenant_id": self.tenant_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FirstPartyEventRecord:
        return cls(
            event_id=str(payload["event_id"]),
            account_id=str(payload["account_id"]),
            claim_type=OrderClaimType(payload["claim_type"]),
            amount=float(payload["amount"]),
            timestamp=_coerce_utc_datetime(payload["timestamp"]),
            order_id=str(payload.get("order_id", "")),
            currency=str(payload.get("currency", "USD")),
            merchant_category=str(payload.get("merchant_category", "general")),
            outcome=ClaimOutcome(payload.get("outcome", ClaimOutcome.APPROVED.value)),
            promo_code=str(payload.get("promo_code", "")),
            shipping_address_hash=str(payload.get("shipping_address_hash", "")),
            payment_instrument_hash=str(payload.get("payment_instrument_hash", "")),
            device_id=str(payload.get("device_id", "")),
            avs_cvv_matched=bool(payload.get("avs_cvv_matched", False)),
            three_ds_authenticated=bool(payload.get("three_ds_authenticated", False)),
            delivery_confirmed=bool(payload.get("delivery_confirmed", False)),
            signature_captured=bool(payload.get("signature_captured", False)),
            tenant_id=str(payload.get("tenant_id", "default")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class FirstPartyAccountProfile:
    """Mutable behavioral state for an account under first-party fraud surveillance."""

    account_id: str
    tenant_id: str = "default"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    events: list[FirstPartyEventRecord] = field(default_factory=list)
    linked_accounts: set[str] = field(default_factory=set)
    risk_level: FirstPartyRiskLevel = FirstPartyRiskLevel.LOW
    last_action: FirstPartyAction = FirstPartyAction.ALLOW
    last_assessed_at: datetime | None = None
    last_assessment_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.astimezone(UTC).isoformat(),
            "events": [e.to_dict() for e in self.events],
            "linked_accounts": sorted(self.linked_accounts),
            "risk_level": self.risk_level.value,
            "last_action": self.last_action.value,
            "last_assessed_at": (
                self.last_assessed_at.astimezone(UTC).isoformat()
                if self.last_assessed_at is not None
                else None
            ),
            "last_assessment_id": self.last_assessment_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FirstPartyAccountProfile:
        last_assessed_raw = payload.get("last_assessed_at")
        return cls(
            account_id=str(payload["account_id"]),
            tenant_id=str(payload.get("tenant_id", "default")),
            created_at=_coerce_utc_datetime(payload["created_at"]),
            events=[FirstPartyEventRecord.from_dict(item) for item in payload.get("events", [])],
            linked_accounts={str(a) for a in payload.get("linked_accounts", [])},
            risk_level=FirstPartyRiskLevel(
                payload.get("risk_level", FirstPartyRiskLevel.LOW.value)
            ),
            last_action=FirstPartyAction(payload.get("last_action", FirstPartyAction.ALLOW.value)),
            last_assessed_at=(
                _coerce_utc_datetime(last_assessed_raw) if last_assessed_raw is not None else None
            ),
            last_assessment_id=payload.get("last_assessment_id"),
        )


@dataclass(frozen=True, slots=True)
class FirstPartyAssessment:
    """Risk assessment output produced by FirstPartyFraudEngine."""

    assessment_id: str
    account_id: str
    tenant_id: str
    risk_score: float
    risk_level: FirstPartyRiskLevel
    action: FirstPartyAction
    anomalies: tuple[FirstPartyAnomalyType, ...]
    return_ratio: float
    purchase_volume: float
    refund_volume: float
    dispute_count: int
    inr_claim_count: int
    promo_redemption_count: int
    linked_account_count: int
    case_id: str | None = None
    reasons: tuple[str, ...] = ()
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "account_id": self.account_id,
            "tenant_id": self.tenant_id,
            "risk_score": round(float(self.risk_score), 6),
            "risk_level": self.risk_level.value,
            "action": self.action.value,
            "anomalies": [a.value for a in self.anomalies],
            "return_ratio": round(float(self.return_ratio), 6),
            "purchase_volume": round(float(self.purchase_volume), 4),
            "refund_volume": round(float(self.refund_volume), 4),
            "dispute_count": self.dispute_count,
            "inr_claim_count": self.inr_claim_count,
            "promo_redemption_count": self.promo_redemption_count,
            "linked_account_count": self.linked_account_count,
            "case_id": self.case_id,
            "reasons": list(self.reasons),
            "evaluated_at": self.evaluated_at.astimezone(UTC).isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FirstPartyAssessment:
        return cls(
            assessment_id=str(payload["assessment_id"]),
            account_id=str(payload["account_id"]),
            tenant_id=str(payload.get("tenant_id", "default")),
            risk_score=float(payload["risk_score"]),
            risk_level=FirstPartyRiskLevel(payload["risk_level"]),
            action=FirstPartyAction(payload["action"]),
            anomalies=tuple(FirstPartyAnomalyType(a) for a in payload.get("anomalies", [])),
            return_ratio=float(payload["return_ratio"]),
            purchase_volume=float(payload["purchase_volume"]),
            refund_volume=float(payload["refund_volume"]),
            dispute_count=int(payload["dispute_count"]),
            inr_claim_count=int(payload["inr_claim_count"]),
            promo_redemption_count=int(payload["promo_redemption_count"]),
            linked_account_count=int(payload["linked_account_count"]),
            case_id=payload.get("case_id"),
            reasons=tuple(str(r) for r in payload.get("reasons", [])),
            evaluated_at=_coerce_utc_datetime(payload["evaluated_at"]),
        )


@dataclass(frozen=True, slots=True)
class RefundAbuseMetrics:
    """Computed refund, wardrobing, and INR claim metrics for an account."""

    purchase_volume: float
    refund_volume: float
    return_ratio: float
    inr_claim_count: int
    wardrobing_count: int
    high_value_skew: bool
    anomalies: tuple[FirstPartyAnomalyType, ...]
    reasons: tuple[str, ...]


class RefundAbuseAnalyzer:
    """Detects serial return abuse, wardrobing, high-value refund skew, and INR bursts."""

    def __init__(
        self,
        *,
        window_days: int = 90,
        return_ratio_threshold: float = 0.50,
        min_refund_volume: float = 100.0,
        wardrobing_threshold: int = 2,
        inr_claim_threshold: int = 2,
        skew_multiplier: float = 1.5,
    ) -> None:
        self.window_days = window_days
        self.return_ratio_threshold = return_ratio_threshold
        self.min_refund_volume = min_refund_volume
        self.wardrobing_threshold = wardrobing_threshold
        self.inr_claim_threshold = inr_claim_threshold
        self.skew_multiplier = skew_multiplier

    def analyze(
        self,
        events: list[FirstPartyEventRecord],
        *,
        now: datetime | None = None,
    ) -> RefundAbuseMetrics:
        if not events:
            return RefundAbuseMetrics(
                purchase_volume=0.0,
                refund_volume=0.0,
                return_ratio=0.0,
                inr_claim_count=0,
                wardrobing_count=0,
                high_value_skew=False,
                anomalies=(),
                reasons=(),
            )
        ref_time = now or max(e.timestamp for e in events)
        cutoff = ref_time - timedelta(days=self.window_days)
        window_events = [e for e in events if e.timestamp >= cutoff]

        purchases = [e for e in window_events if e.claim_type == OrderClaimType.PURCHASE]
        refunds = [
            e
            for e in window_events
            if e.claim_type in {OrderClaimType.REFUND_RETURN, OrderClaimType.INR_CLAIM}
            and e.outcome != ClaimOutcome.DENIED
        ]
        inr_claims = [e for e in window_events if e.claim_type == OrderClaimType.INR_CLAIM]

        purchase_volume = sum(e.amount for e in purchases)
        refund_volume = sum(e.amount for e in refunds)
        if purchase_volume > 0:
            return_ratio = min(refund_volume / purchase_volume, 2.0)
        else:
            return_ratio = 1.0 if refund_volume > 0 else 0.0

        wardrobing_count = sum(
            1
            for e in window_events
            if e.claim_type == OrderClaimType.REFUND_RETURN
            and e.merchant_category.lower() in WARDROBING_CATEGORIES
        )

        avg_purchase = purchase_volume / len(purchases) if purchases else 0.0
        avg_refund = refund_volume / len(refunds) if refunds else 0.0
        high_value_skew = (
            avg_purchase > 0
            and avg_refund >= self.skew_multiplier * avg_purchase
            and len(refunds) >= 2
        )

        anomalies: list[FirstPartyAnomalyType] = []
        reasons: list[str] = []

        if return_ratio >= self.return_ratio_threshold and refund_volume >= self.min_refund_volume:
            anomalies.append(FirstPartyAnomalyType.HIGH_RETURN_RATIO)
            reasons.append(
                f"Return-to-purchase ratio {return_ratio:.2f} exceeds "
                f"{self.return_ratio_threshold:.2f}"
            )
        if wardrobing_count >= self.wardrobing_threshold:
            anomalies.append(FirstPartyAnomalyType.WARDROBING_PATTERN)
            reasons.append(f"{wardrobing_count} returns in wardrobing-prone categories")
        if len(inr_claims) >= self.inr_claim_threshold:
            anomalies.append(FirstPartyAnomalyType.INR_CLAIM_BURST)
            reasons.append(
                f"{len(inr_claims)} Item-Not-Received (INR) claims within {self.window_days}d"
            )
        if high_value_skew:
            anomalies.append(FirstPartyAnomalyType.HIGH_VALUE_REFUND_SKEW)
            reasons.append(
                f"Average refund (${avg_refund:.2f}) >= {self.skew_multiplier:.1f}x "
                f"avg purchase (${avg_purchase:.2f})"
            )

        return RefundAbuseMetrics(
            purchase_volume=purchase_volume,
            refund_volume=refund_volume,
            return_ratio=return_ratio,
            inr_claim_count=len(inr_claims),
            wardrobing_count=wardrobing_count,
            high_value_skew=high_value_skew,
            anomalies=tuple(anomalies),
            reasons=tuple(reasons),
        )


@dataclass(frozen=True, slots=True)
class FriendlyFraudMetrics:
    """Computed friendly-fraud dispute and pre-dispute velocity metrics."""

    dispute_count: int
    verified_dispute_count: int
    recent_dispute_count: int
    anomalies: tuple[FirstPartyAnomalyType, ...]
    reasons: tuple[str, ...]


class FriendlyFraudAnalyzer:
    """Detects serial friendly-fraud disputes on authenticated or delivered orders."""

    def __init__(
        self,
        *,
        window_days: int = 180,
        pre_dispute_window_days: int = 14,
        serial_dispute_threshold: int = 2,
        pre_dispute_velocity_threshold: int = 2,
    ) -> None:
        self.window_days = window_days
        self.pre_dispute_window_days = pre_dispute_window_days
        self.serial_dispute_threshold = serial_dispute_threshold
        self.pre_dispute_velocity_threshold = pre_dispute_velocity_threshold

    def analyze(
        self,
        events: list[FirstPartyEventRecord],
        *,
        now: datetime | None = None,
    ) -> FriendlyFraudMetrics:
        disputes = [e for e in events if e.claim_type == OrderClaimType.FRIENDLY_DISPUTE]
        if not disputes:
            return FriendlyFraudMetrics(
                dispute_count=0,
                verified_dispute_count=0,
                recent_dispute_count=0,
                anomalies=(),
                reasons=(),
            )
        ref_time = now or max(e.timestamp for e in events)
        cutoff = ref_time - timedelta(days=self.window_days)
        recent_cutoff = ref_time - timedelta(days=self.pre_dispute_window_days)

        window_disputes = [d for d in disputes if d.timestamp >= cutoff]
        verified_disputes = [
            d
            for d in window_disputes
            if (
                d.three_ds_authenticated
                or d.delivery_confirmed
                or d.signature_captured
                or d.avs_cvv_matched
            )
        ]
        recent_disputes = [d for d in window_disputes if d.timestamp >= recent_cutoff]

        anomalies: list[FirstPartyAnomalyType] = []
        reasons: list[str] = []
        if len(verified_disputes) >= self.serial_dispute_threshold:
            anomalies.append(FirstPartyAnomalyType.SERIAL_FRIENDLY_FRAUD)
            reasons.append(
                f"{len(verified_disputes)} friendly-fraud disputes on "
                "authenticated/delivered orders"
            )
        if len(recent_disputes) >= self.pre_dispute_velocity_threshold:
            anomalies.append(FirstPartyAnomalyType.PRE_DISPUTE_VELOCITY)
            reasons.append(
                f"{len(recent_disputes)} disputes filed within {self.pre_dispute_window_days}d"
            )

        return FriendlyFraudMetrics(
            dispute_count=len(window_disputes),
            verified_dispute_count=len(verified_disputes),
            recent_dispute_count=len(recent_disputes),
            anomalies=tuple(anomalies),
            reasons=tuple(reasons),
        )


@dataclass(frozen=True, slots=True)
class PromoAbuseMetrics:
    """Computed promotional/referral abuse and linked-account cycling metrics."""

    promo_redemption_count: int
    unique_promo_codes: int
    linked_account_count: int
    anomalies: tuple[FirstPartyAnomalyType, ...]
    reasons: tuple[str, ...]


class PromoAbuseDetector:
    """Detects promotional/referral bonus abuse and linked-account cycling."""

    def __init__(
        self,
        *,
        promo_threshold: int = 3,
        linked_account_threshold: int = 2,
    ) -> None:
        self.promo_threshold = promo_threshold
        self.linked_account_threshold = linked_account_threshold

    def analyze(
        self,
        events: list[FirstPartyEventRecord],
        *,
        linked_accounts: set[str] | None = None,
    ) -> PromoAbuseMetrics:
        promos = [e for e in events if e.claim_type == OrderClaimType.PROMO_REDEMPTION]
        purchases = [e for e in events if e.claim_type == OrderClaimType.PURCHASE]
        unique_codes = {e.promo_code for e in promos if e.promo_code}
        linked_count = len(linked_accounts) if linked_accounts else 0

        anomalies: list[FirstPartyAnomalyType] = []
        reasons: list[str] = []

        if len(promos) >= self.promo_threshold or (len(promos) >= 2 and not purchases):
            anomalies.append(FirstPartyAnomalyType.PROMO_BONUS_ABUSE)
            reasons.append(
                f"{len(promos)} promo redemptions across {max(len(unique_codes), 1)} code(s)"
            )
        if linked_count >= self.linked_account_threshold:
            anomalies.append(FirstPartyAnomalyType.LINKED_ACCOUNT_CYCLING)
            reasons.append(
                f"Account shares payment/shipping identifiers with {linked_count} linked accounts"
            )

        return PromoAbuseMetrics(
            promo_redemption_count=len(promos),
            unique_promo_codes=len(unique_codes),
            linked_account_count=linked_count,
            anomalies=tuple(anomalies),
            reasons=tuple(reasons),
        )


class FirstPartyAccountStore:
    """Thread-safe in-memory store for first-party account profiles and assessments."""

    def __init__(
        self,
        *,
        max_accounts: int = 10_000,
        max_assessments: int = 20_000,
        max_events_per_account: int = 500,
    ) -> None:
        self._lock = threading.RLock()
        self._max_accounts = max_accounts
        self._max_assessments = max_assessments
        self._max_events_per_account = max_events_per_account
        self._profiles: OrderedDict[str, FirstPartyAccountProfile] = OrderedDict()
        self._assessments: OrderedDict[str, FirstPartyAssessment] = OrderedDict()
        self._tenant_index: dict[str, set[str]] = {}
        self._shipping_index: dict[str, set[str]] = {}
        self._payment_index: dict[str, set[str]] = {}

    def record_event(self, record: FirstPartyEventRecord) -> FirstPartyAccountProfile:
        with self._lock:
            profile = self._profiles.get(record.account_id)
            if profile is None:
                while len(self._profiles) >= self._max_accounts:
                    evicted_id, evicted = self._profiles.popitem(last=False)
                    self._tenant_index.get(evicted.tenant_id, set()).discard(evicted_id)
                profile = FirstPartyAccountProfile(
                    account_id=record.account_id,
                    tenant_id=record.tenant_id,
                    created_at=record.timestamp,
                )
            else:
                self._profiles.move_to_end(record.account_id)

            profile.events.append(record)
            if len(profile.events) > self._max_events_per_account:
                profile.events = profile.events[-self._max_events_per_account :]

            self._profiles[record.account_id] = profile
            self._tenant_index.setdefault(record.tenant_id, set()).add(record.account_id)

            if record.shipping_address_hash:
                peers = self._shipping_index.setdefault(record.shipping_address_hash, set())
                peers.add(record.account_id)
                for peer in peers:
                    if peer != record.account_id:
                        profile.linked_accounts.add(peer)
                        peer_profile = self._profiles.get(peer)
                        if peer_profile is not None:
                            peer_profile.linked_accounts.add(record.account_id)

            if record.payment_instrument_hash:
                peers = self._payment_index.setdefault(record.payment_instrument_hash, set())
                peers.add(record.account_id)
                for peer in peers:
                    if peer != record.account_id:
                        profile.linked_accounts.add(peer)
                        peer_profile = self._profiles.get(peer)
                        if peer_profile is not None:
                            peer_profile.linked_accounts.add(record.account_id)

            return profile

    def get_profile(self, account_id: str) -> FirstPartyAccountProfile | None:
        with self._lock:
            return self._profiles.get(account_id)

    def save_assessment(self, assessment: FirstPartyAssessment) -> None:
        with self._lock:
            while len(self._assessments) >= self._max_assessments:
                self._assessments.popitem(last=False)
            self._assessments[assessment.assessment_id] = assessment

    def get_assessment(self, assessment_id: str) -> FirstPartyAssessment | None:
        with self._lock:
            return self._assessments.get(assessment_id)

    def list_assessments(
        self,
        *,
        account_id: str | None = None,
        risk_level: FirstPartyRiskLevel | None = None,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[FirstPartyAssessment]:
        with self._lock:
            items = list(self._assessments.values())
            items.reverse()
            filtered = [
                item
                for item in items
                if (account_id is None or item.account_id == account_id)
                and (risk_level is None or item.risk_level == risk_level)
                and (tenant_id is None or item.tenant_id == tenant_id)
            ]
            return filtered[:limit]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            risk_counts: dict[str, int] = {level.value: 0 for level in FirstPartyRiskLevel}
            for profile in self._profiles.values():
                risk_counts[profile.risk_level.value] += 1
            return {
                "total_accounts": len(self._profiles),
                "total_assessments": len(self._assessments),
                "risk_distribution": risk_counts,
            }

    def export_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "profiles": [p.to_dict() for p in self._profiles.values()],
                "assessments": [a.to_dict() for a in self._assessments.values()],
            }

    def load_state(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._profiles.clear()
            self._assessments.clear()
            self._tenant_index.clear()
            self._shipping_index.clear()
            self._payment_index.clear()
            for p_raw in payload.get("profiles", []):
                profile = FirstPartyAccountProfile.from_dict(p_raw)
                self._profiles[profile.account_id] = profile
                self._tenant_index.setdefault(profile.tenant_id, set()).add(profile.account_id)
                for ev in profile.events:
                    if ev.shipping_address_hash:
                        self._shipping_index.setdefault(ev.shipping_address_hash, set()).add(
                            profile.account_id
                        )
                    if ev.payment_instrument_hash:
                        self._payment_index.setdefault(ev.payment_instrument_hash, set()).add(
                            profile.account_id
                        )
            for a_raw in payload.get("assessments", []):
                assessment = FirstPartyAssessment.from_dict(a_raw)
                self._assessments[assessment.assessment_id] = assessment
