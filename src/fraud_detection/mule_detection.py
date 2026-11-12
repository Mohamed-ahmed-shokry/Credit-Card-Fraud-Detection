"""Money Mule Account & Rapid Fund Exfiltration Detection Engine.

Detects pass-through conduit accounts, rapid dwell-time compression,
fan-in/fan-out counterparty concentration, dormancy-to-burst transitions,
and high-risk off-ramp cash-out channels with zero external dependencies.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from statistics import median
from threading import RLock
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from fraud_detection.cases import CaseManager


class MuleRiskLevel(StrEnum):
    """Categorical risk tiers for mule account assessments."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MuleAction(StrEnum):
    """Policy actions determined by mule risk evaluation."""

    ALLOW = "allow"
    STEP_UP_VERIFICATION = "step_up_verification"
    HOLD_FUNDS = "hold_funds"
    FREEZE_ACCOUNT = "freeze_account"
    ESCALATE_AML_SAR = "escalate_aml_sar"


class MuleAnomalyType(StrEnum):
    """Specific behavioral and topological anomalies indicative of mule activity."""

    HIGH_PASS_THROUGH_RATIO = "high_pass_through_ratio"  # noqa: S105
    RAPID_DWELL_COMPRESSION = "rapid_dwell_compression"
    FAN_IN_COLLECTION = "fan_in_collection"
    FAN_OUT_DISPERSION = "fan_out_dispersion"
    DORMANCY_BURST = "dormancy_burst"
    HIGH_RISK_OFF_RAMP = "high_risk_off_ramp"
    NEW_COUNTERPARTY_FLOOD = "new_counterparty_flood"


class TransferDirection(StrEnum):
    """Direction of fund movement relative to the monitored account."""

    INBOUND = "inbound"
    OUTBOUND = "outbound"


class ExfiltrationChannel(StrEnum):
    """Payment rail or off-ramp channel used for fund movement."""

    ACH = "ach"
    CARD = "card"
    P2P_WALLET = "p2p_wallet"
    CRYPTO_EXCHANGE = "crypto_exchange"
    WIRE_INTERNATIONAL = "wire_international"
    PREPAID_LOAD = "prepaid_load"
    ATM_CASH_OUT = "atm_cash_out"


HIGH_RISK_OFF_RAMP_CHANNELS: frozenset[ExfiltrationChannel] = frozenset(
    {
        ExfiltrationChannel.CRYPTO_EXCHANGE,
        ExfiltrationChannel.WIRE_INTERNATIONAL,
        ExfiltrationChannel.PREPAID_LOAD,
        ExfiltrationChannel.ATM_CASH_OUT,
    }
)


def _parse_dt(raw: Any, default: datetime | None = None) -> datetime:
    """Parse ISO-8601 datetime strings or return UTC default."""
    if isinstance(raw, datetime):
        return raw if raw.tzinfo is not None else raw.replace(tzinfo=UTC)
    if isinstance(raw, str) and raw.strip():
        parsed = datetime.fromisoformat(raw)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return default if default is not None else datetime.now(UTC)


@dataclass(frozen=True)
class MuleTransferRecord:
    """Single inbound or outbound transfer event associated with a monitored account."""

    transfer_id: str
    account_id: str
    counterparty_id: str
    direction: TransferDirection
    amount: float
    channel: ExfiltrationChannel = ExfiltrationChannel.ACH
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    device_id: str | None = None
    tenant_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transfer_id": self.transfer_id,
            "account_id": self.account_id,
            "counterparty_id": self.counterparty_id,
            "direction": self.direction.value,
            "amount": float(self.amount),
            "channel": self.channel.value,
            "timestamp": self.timestamp.isoformat(),
            "device_id": self.device_id,
            "tenant_id": self.tenant_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MuleTransferRecord:
        return cls(
            transfer_id=str(data.get("transfer_id") or f"xfer_{uuid4().hex[:12]}"),
            account_id=str(data.get("account_id", "")),
            counterparty_id=str(data.get("counterparty_id", "")),
            direction=TransferDirection(data.get("direction", "inbound")),
            amount=float(data.get("amount", 0.0)),
            channel=ExfiltrationChannel(data.get("channel", "ach")),
            timestamp=_parse_dt(data.get("timestamp")),
            device_id=data.get("device_id"),
            tenant_id=str(data.get("tenant_id", "default")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class MuleAccountProfile:
    """Aggregated behavioral and flow state for an account monitored for mule activity."""

    account_id: str
    tenant_id: str = "default"
    account_opened_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    transfers: list[MuleTransferRecord] = field(default_factory=list)
    last_assessed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "tenant_id": self.tenant_id,
            "account_opened_at": self.account_opened_at.isoformat(),
            "transfers": [t.to_dict() for t in self.transfers],
            "last_assessed_at": (
                self.last_assessed_at.isoformat() if self.last_assessed_at else None
            ),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MuleAccountProfile:
        last_raw = data.get("last_assessed_at")
        return cls(
            account_id=str(data.get("account_id", "")),
            tenant_id=str(data.get("tenant_id", "default")),
            account_opened_at=_parse_dt(data.get("account_opened_at")),
            transfers=[
                MuleTransferRecord.from_dict(t) if isinstance(t, dict) else t
                for t in data.get("transfers", [])
            ],
            last_assessed_at=_parse_dt(last_raw) if last_raw else None,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class MuleAssessment:
    """Composite money-mule risk assessment report for a monitored account."""

    assessment_id: str = field(default_factory=lambda: f"mule_{uuid4().hex[:12]}")
    account_id: str = ""
    tenant_id: str = "default"
    risk_score: float = 0.0
    risk_level: MuleRiskLevel = MuleRiskLevel.LOW
    action: MuleAction = MuleAction.ALLOW
    triggered_anomalies: list[MuleAnomalyType] = field(default_factory=list)
    anomaly_breakdown: dict[str, float] = field(default_factory=dict)
    pass_through_ratio: float = 0.0
    median_dwell_hours: float | None = None
    unique_inbound_counterparties: int = 0
    unique_outbound_counterparties: int = 0
    inbound_volume: float = 0.0
    outbound_volume: float = 0.0
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "account_id": self.account_id,
            "tenant_id": self.tenant_id,
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level.value,
            "action": self.action.value,
            "triggered_anomalies": [a.value for a in self.triggered_anomalies],
            "anomaly_breakdown": dict(self.anomaly_breakdown),
            "pass_through_ratio": round(self.pass_through_ratio, 4),
            "median_dwell_hours": (
                round(self.median_dwell_hours, 4) if self.median_dwell_hours is not None else None
            ),
            "unique_inbound_counterparties": self.unique_inbound_counterparties,
            "unique_outbound_counterparties": self.unique_outbound_counterparties,
            "inbound_volume": round(self.inbound_volume, 2),
            "outbound_volume": round(self.outbound_volume, 2),
            "evaluated_at": self.evaluated_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MuleAssessment:
        dwell_raw = data.get("median_dwell_hours")
        return cls(
            assessment_id=str(data.get("assessment_id") or f"mule_{uuid4().hex[:12]}"),
            account_id=str(data.get("account_id", "")),
            tenant_id=str(data.get("tenant_id", "default")),
            risk_score=float(data.get("risk_score", 0.0)),
            risk_level=MuleRiskLevel(data.get("risk_level", "low")),
            action=MuleAction(data.get("action", "allow")),
            triggered_anomalies=[MuleAnomalyType(a) for a in data.get("triggered_anomalies", [])],
            anomaly_breakdown=dict(data.get("anomaly_breakdown", {})),
            pass_through_ratio=float(data.get("pass_through_ratio", 0.0)),
            median_dwell_hours=float(dwell_raw) if dwell_raw is not None else None,
            unique_inbound_counterparties=int(data.get("unique_inbound_counterparties", 0)),
            unique_outbound_counterparties=int(data.get("unique_outbound_counterparties", 0)),
            inbound_volume=float(data.get("inbound_volume", 0.0)),
            outbound_volume=float(data.get("outbound_volume", 0.0)),
            evaluated_at=_parse_dt(data.get("evaluated_at")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class PassThroughMetrics:
    """Computed pass-through velocity, dwell-time compression, and off-ramp exposure."""

    inbound_volume: float
    outbound_volume: float
    pass_through_ratio: float
    median_dwell_hours: float | None
    high_risk_off_ramp_ratio: float
    anomalies: list[MuleAnomalyType]
    risk_score: float


class PassThroughAnalyzer:
    """Analyzes inbound-to-outbound fund velocity, dwell compression, and off-ramps."""

    def __init__(
        self,
        window: timedelta = timedelta(hours=72),
        min_inbound_volume: float = 500.0,
        pass_through_threshold: float = 0.85,
        rapid_dwell_hours: float = 6.0,
        high_risk_off_ramp_threshold: float = 0.50,
    ) -> None:
        self.window = window
        self.min_inbound_volume = min_inbound_volume
        self.pass_through_threshold = pass_through_threshold
        self.rapid_dwell_hours = rapid_dwell_hours
        self.high_risk_off_ramp_threshold = high_risk_off_ramp_threshold

    def analyze(
        self,
        transfers: list[MuleTransferRecord],
        now: datetime | None = None,
    ) -> PassThroughMetrics:
        """Compute pass-through ratio, dwell time compression, and off-ramp risk."""
        if not transfers:
            return PassThroughMetrics(
                inbound_volume=0.0,
                outbound_volume=0.0,
                pass_through_ratio=0.0,
                median_dwell_hours=None,
                high_risk_off_ramp_ratio=0.0,
                anomalies=[],
                risk_score=0.0,
            )

        ref_time = now if now is not None else max(t.timestamp for t in transfers)
        cutoff = ref_time - self.window
        recent = sorted(
            (t for t in transfers if t.timestamp >= cutoff),
            key=lambda t: t.timestamp,
        )

        inbound_xfers = [t for t in recent if t.direction == TransferDirection.INBOUND]
        outbound_xfers = [t for t in recent if t.direction == TransferDirection.OUTBOUND]

        inbound_vol = sum(t.amount for t in inbound_xfers)
        outbound_vol = sum(t.amount for t in outbound_xfers)

        pass_through_ratio = (
            min(outbound_vol, inbound_vol) / inbound_vol if inbound_vol > 0.0 else 0.0
        )

        # Pair each outbound transfer with the most recent preceding inbound transfer
        dwell_hours_list: list[float] = []
        last_in_ts: datetime | None = None
        for t in recent:
            if t.direction == TransferDirection.INBOUND:
                last_in_ts = t.timestamp
            elif t.direction == TransferDirection.OUTBOUND and last_in_ts is not None:
                delta_h = max(0.0, (t.timestamp - last_in_ts).total_seconds() / 3600.0)
                dwell_hours_list.append(delta_h)

        median_dwell = median(dwell_hours_list) if dwell_hours_list else None

        high_risk_out = sum(
            t.amount for t in outbound_xfers if t.channel in HIGH_RISK_OFF_RAMP_CHANNELS
        )
        off_ramp_ratio = high_risk_out / outbound_vol if outbound_vol > 0.0 else 0.0

        anomalies: list[MuleAnomalyType] = []
        score = 0.0

        if (
            inbound_vol >= self.min_inbound_volume
            and pass_through_ratio >= self.pass_through_threshold
        ):
            anomalies.append(MuleAnomalyType.HIGH_PASS_THROUGH_RATIO)
            score += 0.45

        if (
            median_dwell is not None
            and inbound_vol >= self.min_inbound_volume
            and outbound_vol > 0.0
            and median_dwell <= self.rapid_dwell_hours
        ):
            anomalies.append(MuleAnomalyType.RAPID_DWELL_COMPRESSION)
            score += 0.35

        if (
            outbound_vol >= self.min_inbound_volume
            and off_ramp_ratio >= self.high_risk_off_ramp_threshold
        ):
            anomalies.append(MuleAnomalyType.HIGH_RISK_OFF_RAMP)
            score += 0.30

        return PassThroughMetrics(
            inbound_volume=inbound_vol,
            outbound_volume=outbound_vol,
            pass_through_ratio=pass_through_ratio,
            median_dwell_hours=median_dwell,
            high_risk_off_ramp_ratio=off_ramp_ratio,
            anomalies=anomalies,
            risk_score=min(1.0, score),
        )


@dataclass(frozen=True)
class MuleNetworkMetrics:
    """Counterparty fan-in, fan-out, and new-counterparty concentration metrics."""

    unique_inbound_counterparties: int
    unique_outbound_counterparties: int
    new_counterparties_24h: int
    anomalies: list[MuleAnomalyType]
    risk_score: float


class MuleNetworkAnalyzer:
    """Detects fan-in collection, fan-out dispersion, and rapid counterparty floods."""

    def __init__(
        self,
        window: timedelta = timedelta(hours=72),
        fan_in_threshold: int = 4,
        fan_out_threshold: int = 4,
        new_counterparty_threshold: int = 5,
    ) -> None:
        self.window = window
        self.fan_in_threshold = fan_in_threshold
        self.fan_out_threshold = fan_out_threshold
        self.new_counterparty_threshold = new_counterparty_threshold

    def analyze(
        self,
        transfers: list[MuleTransferRecord],
        now: datetime | None = None,
    ) -> MuleNetworkMetrics:
        """Evaluate counterparty fan-in/fan-out topology over the sliding window."""
        if not transfers:
            return MuleNetworkMetrics(
                unique_inbound_counterparties=0,
                unique_outbound_counterparties=0,
                new_counterparties_24h=0,
                anomalies=[],
                risk_score=0.0,
            )

        ref_time = now if now is not None else max(t.timestamp for t in transfers)
        cutoff = ref_time - self.window
        recent_24h = ref_time - timedelta(hours=24)

        prior_cps = {t.counterparty_id for t in transfers if t.timestamp < recent_24h}
        recent_xfers = [t for t in transfers if t.timestamp >= cutoff]

        inbound_cps = {
            t.counterparty_id
            for t in recent_xfers
            if t.direction == TransferDirection.INBOUND and t.counterparty_id
        }
        outbound_cps = {
            t.counterparty_id
            for t in recent_xfers
            if t.direction == TransferDirection.OUTBOUND and t.counterparty_id
        }
        new_24h_cps = {
            t.counterparty_id
            for t in recent_xfers
            if t.timestamp >= recent_24h
            and t.counterparty_id
            and t.counterparty_id not in prior_cps
        }

        anomalies: list[MuleAnomalyType] = []
        score = 0.0

        if len(inbound_cps) >= self.fan_in_threshold:
            anomalies.append(MuleAnomalyType.FAN_IN_COLLECTION)
            score += 0.40
        if len(outbound_cps) >= self.fan_out_threshold:
            anomalies.append(MuleAnomalyType.FAN_OUT_DISPERSION)
            score += 0.35
        if len(new_24h_cps) >= self.new_counterparty_threshold:
            anomalies.append(MuleAnomalyType.NEW_COUNTERPARTY_FLOOD)
            score += 0.30

        return MuleNetworkMetrics(
            unique_inbound_counterparties=len(inbound_cps),
            unique_outbound_counterparties=len(outbound_cps),
            new_counterparties_24h=len(new_24h_cps),
            anomalies=anomalies,
            risk_score=min(1.0, score),
        )


class DormancyBurstDetector:
    """Detects dormant accounts waking up into sudden high-volume transfer bursts."""

    def __init__(
        self,
        dormancy_days: int = 30,
        burst_window: timedelta = timedelta(hours=24),
        burst_min_transfers: int = 3,
        burst_min_volume: float = 2000.0,
    ) -> None:
        self.dormancy_days = dormancy_days
        self.burst_window = burst_window
        self.burst_min_transfers = burst_min_transfers
        self.burst_min_volume = burst_min_volume

    def evaluate(
        self,
        profile: MuleAccountProfile,
        now: datetime | None = None,
    ) -> tuple[bool, float, str]:
        """Return (is_burst, risk_score, reason) if dormancy-to-burst is detected."""
        transfers = profile.transfers
        if not transfers:
            return (False, 0.0, "")

        ref_time = now if now is not None else max(t.timestamp for t in transfers)
        burst_cutoff = ref_time - self.burst_window
        dormancy_delta = timedelta(days=self.dormancy_days)

        burst_xfers = [t for t in transfers if t.timestamp >= burst_cutoff]
        prior_xfers = [t for t in transfers if t.timestamp < burst_cutoff]

        if len(burst_xfers) < self.burst_min_transfers:
            return (False, 0.0, "")

        burst_volume = sum(t.amount for t in burst_xfers)
        if burst_volume < self.burst_min_volume:
            return (False, 0.0, "")

        earliest_burst = min(t.timestamp for t in burst_xfers)
        if prior_xfers:
            last_prior = max(t.timestamp for t in prior_xfers)
            dormant_gap = earliest_burst - last_prior
        else:
            dormant_gap = earliest_burst - profile.account_opened_at

        if dormant_gap >= dormancy_delta:
            days = int(dormant_gap.total_seconds() // 86400)
            reason = (
                f"Account dormant for {days} days followed by {len(burst_xfers)} "
                f"transfers totaling {burst_volume:.2f}"
            )
            return (True, 0.45, reason)

        return (False, 0.0, "")


class MuleAccountStore:
    """Thread-safe in-memory store for monitored mule accounts and assessments."""

    def __init__(self, max_records: int = 10_000) -> None:
        self.max_records = max_records
        self._lock = RLock()
        self._profiles: dict[str, MuleAccountProfile] = {}
        self._profiles_by_tenant: dict[str, list[str]] = defaultdict(list)
        self._profiles_by_counterparty: dict[str, list[str]] = defaultdict(list)
        self._assessments: dict[str, MuleAssessment] = {}
        self._assessments_by_account: dict[str, list[str]] = defaultdict(list)
        self._assessments_by_tenant: dict[str, list[str]] = defaultdict(list)

    def upsert_profile(self, profile: MuleAccountProfile) -> None:
        """Store or update an account profile and maintain secondary indexes."""
        with self._lock:
            acct_id = profile.account_id
            if len(self._profiles) >= self.max_records and acct_id not in self._profiles:
                oldest = next(iter(self._profiles))
                self.remove_profile(oldest)

            self._profiles[acct_id] = profile
            if acct_id not in self._profiles_by_tenant[profile.tenant_id]:
                self._profiles_by_tenant[profile.tenant_id].append(acct_id)
            for xfer in profile.transfers:
                if (
                    xfer.counterparty_id
                    and acct_id not in self._profiles_by_counterparty[xfer.counterparty_id]
                ):
                    self._profiles_by_counterparty[xfer.counterparty_id].append(acct_id)

    def remove_profile(self, account_id: str) -> None:
        """Remove an account profile and clean up secondary index entries."""
        with self._lock:
            prof = self._profiles.pop(account_id, None)
            if prof is None:
                return
            if account_id in self._profiles_by_tenant[prof.tenant_id]:
                self._profiles_by_tenant[prof.tenant_id].remove(account_id)
            for xfer in prof.transfers:
                if (
                    xfer.counterparty_id
                    and account_id in self._profiles_by_counterparty[xfer.counterparty_id]
                ):
                    self._profiles_by_counterparty[xfer.counterparty_id].remove(account_id)

    def record_transfer(self, transfer: MuleTransferRecord) -> MuleAccountProfile:
        """Append a transfer to an existing or newly initialized account profile."""
        with self._lock:
            prof = self._profiles.get(transfer.account_id)
            if prof is None:
                prof = MuleAccountProfile(
                    account_id=transfer.account_id,
                    tenant_id=transfer.tenant_id,
                    account_opened_at=transfer.timestamp,
                )
            prof.transfers.append(transfer)
            self.upsert_profile(prof)
            return prof

    def get_profile(self, account_id: str) -> MuleAccountProfile | None:
        with self._lock:
            return self._profiles.get(account_id)

    def list_profiles(
        self, tenant_id: str | None = None, limit: int = 100
    ) -> list[MuleAccountProfile]:
        with self._lock:
            if tenant_id:
                ids = self._profiles_by_tenant.get(tenant_id, [])
                return [self._profiles[aid] for aid in reversed(ids) if aid in self._profiles][
                    :limit
                ]
            return list(reversed(list(self._profiles.values())))[:limit]

    def find_accounts_by_counterparty(self, counterparty_id: str) -> list[MuleAccountProfile]:
        with self._lock:
            ids = self._profiles_by_counterparty.get(counterparty_id, [])
            return [self._profiles[aid] for aid in ids if aid in self._profiles]

    def add_assessment(self, assessment: MuleAssessment) -> None:
        with self._lock:
            aid = assessment.assessment_id
            if len(self._assessments) >= self.max_records and aid not in self._assessments:
                oldest = next(iter(self._assessments))
                self._assessments.pop(oldest, None)
            self._assessments[aid] = assessment
            self._assessments_by_account[assessment.account_id].append(aid)
            self._assessments_by_tenant[assessment.tenant_id].append(aid)

    def get_assessment(self, assessment_id: str) -> MuleAssessment | None:
        with self._lock:
            return self._assessments.get(assessment_id)

    def list_assessments(
        self, tenant_id: str | None = None, limit: int = 100
    ) -> list[MuleAssessment]:
        with self._lock:
            if tenant_id:
                aids = self._assessments_by_tenant.get(tenant_id, [])
                return [
                    self._assessments[aid] for aid in reversed(aids) if aid in self._assessments
                ][:limit]
            return list(reversed(list(self._assessments.values())))[:limit]

    @property
    def profile_count(self) -> int:
        with self._lock:
            return len(self._profiles)

    @property
    def assessment_count(self) -> int:
        with self._lock:
            return len(self._assessments)

    def clear(self) -> None:
        with self._lock:
            self._profiles.clear()
            self._profiles_by_tenant.clear()
            self._profiles_by_counterparty.clear()
            self._assessments.clear()
            self._assessments_by_account.clear()
            self._assessments_by_tenant.clear()

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_records": self.max_records,
                "profiles": [p.to_dict() for p in self._profiles.values()],
                "assessments": [a.to_dict() for a in self._assessments.values()],
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MuleAccountStore:
        store = cls(max_records=int(data.get("max_records", 10_000)))
        for p in data.get("profiles", []):
            store.upsert_profile(MuleAccountProfile.from_dict(p))
        for a in data.get("assessments", []):
            store.add_assessment(MuleAssessment.from_dict(a))
        return store


class MuleDetectionEngine:
    """Composite scoring and policy routing engine for money mule detection."""

    def __init__(
        self,
        store: MuleAccountStore | None = None,
        pass_through_analyzer: PassThroughAnalyzer | None = None,
        network_analyzer: MuleNetworkAnalyzer | None = None,
        dormancy_detector: DormancyBurstDetector | None = None,
        case_manager: CaseManager | None = None,
        step_up_threshold: float = 0.35,
        hold_funds_threshold: float = 0.60,
        freeze_threshold: float = 0.80,
        escalate_sar_threshold: float = 0.90,
    ) -> None:
        self.store = store or MuleAccountStore()
        self.pass_through_analyzer = pass_through_analyzer or PassThroughAnalyzer()
        self.network_analyzer = network_analyzer or MuleNetworkAnalyzer()
        self.dormancy_detector = dormancy_detector or DormancyBurstDetector()
        self.case_manager = case_manager
        self.step_up_threshold = step_up_threshold
        self.hold_funds_threshold = hold_funds_threshold
        self.freeze_threshold = freeze_threshold
        self.escalate_sar_threshold = escalate_sar_threshold

    def ingest_and_evaluate(
        self,
        transfer: MuleTransferRecord,
        now: datetime | None = None,
    ) -> MuleAssessment:
        """Record a transfer and immediately evaluate the account's mule risk."""
        profile = self.store.record_transfer(transfer)
        eval_time = now if now is not None else transfer.timestamp
        return self.evaluate_account(profile.account_id, now=eval_time)

    def evaluate_account(
        self,
        account_id: str,
        now: datetime | None = None,
    ) -> MuleAssessment:
        """Evaluate an existing monitored account for mule activity and route policy."""
        profile = self.store.get_profile(account_id)
        if profile is None:
            profile = MuleAccountProfile(account_id=account_id)
            self.store.upsert_profile(profile)
        return self.evaluate_profile(profile, now=now)

    def evaluate_profile(
        self,
        profile: MuleAccountProfile,
        now: datetime | None = None,
    ) -> MuleAssessment:
        """Compute composite mule risk score, route policy action, and persist."""
        eval_time = now if now is not None else datetime.now(UTC)
        pt = self.pass_through_analyzer.analyze(profile.transfers, now=eval_time)
        net = self.network_analyzer.analyze(profile.transfers, now=eval_time)
        is_burst, burst_score, burst_reason = self.dormancy_detector.evaluate(
            profile, now=eval_time
        )

        anomalies: list[MuleAnomalyType] = list(pt.anomalies)
        for a in net.anomalies:
            if a not in anomalies:
                anomalies.append(a)
        if is_burst and MuleAnomalyType.DORMANCY_BURST not in anomalies:
            anomalies.append(MuleAnomalyType.DORMANCY_BURST)

        breakdown: dict[str, float] = {}
        if pt.risk_score > 0.0:
            breakdown["pass_through_velocity"] = round(pt.risk_score, 4)
        if net.risk_score > 0.0:
            breakdown["network_topology"] = round(net.risk_score, 4)
        if burst_score > 0.0:
            breakdown["dormancy_burst"] = round(burst_score, 4)

        risk_score = min(
            1.0,
            max(0.0, pt.risk_score * 0.45 + net.risk_score * 0.35 + burst_score * 0.30),
        )
        if (
            MuleAnomalyType.HIGH_PASS_THROUGH_RATIO in anomalies
            and MuleAnomalyType.RAPID_DWELL_COMPRESSION in anomalies
        ):
            risk_score = max(risk_score, 0.70)
        if (
            MuleAnomalyType.HIGH_PASS_THROUGH_RATIO in anomalies
            and MuleAnomalyType.HIGH_RISK_OFF_RAMP in anomalies
            and (
                MuleAnomalyType.FAN_IN_COLLECTION in anomalies
                or MuleAnomalyType.DORMANCY_BURST in anomalies
            )
        ):
            risk_score = max(risk_score, 0.90)

        if risk_score >= self.escalate_sar_threshold:
            action = MuleAction.ESCALATE_AML_SAR
            risk_level = MuleRiskLevel.CRITICAL
        elif risk_score >= self.freeze_threshold:
            action = MuleAction.FREEZE_ACCOUNT
            risk_level = MuleRiskLevel.CRITICAL
        elif risk_score >= self.hold_funds_threshold:
            action = MuleAction.HOLD_FUNDS
            risk_level = MuleRiskLevel.HIGH
        elif risk_score >= self.step_up_threshold:
            action = MuleAction.STEP_UP_VERIFICATION
            risk_level = MuleRiskLevel.MEDIUM
        else:
            action = MuleAction.ALLOW
            risk_level = MuleRiskLevel.LOW

        assessment = MuleAssessment(
            account_id=profile.account_id,
            tenant_id=profile.tenant_id,
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            action=action,
            triggered_anomalies=anomalies,
            anomaly_breakdown=breakdown,
            pass_through_ratio=round(pt.pass_through_ratio, 4),
            median_dwell_hours=pt.median_dwell_hours,
            unique_inbound_counterparties=net.unique_inbound_counterparties,
            unique_outbound_counterparties=net.unique_outbound_counterparties,
            inbound_volume=round(pt.inbound_volume, 2),
            outbound_volume=round(pt.outbound_volume, 2),
            evaluated_at=eval_time,
            metadata={"dormancy_reason": burst_reason} if burst_reason else {},
        )

        profile.last_assessed_at = eval_time
        self.store.upsert_profile(profile)
        self.store.add_assessment(assessment)

        if self.case_manager is not None and action in (
            MuleAction.FREEZE_ACCOUNT,
            MuleAction.ESCALATE_AML_SAR,
        ):
            self.case_manager.create_case(
                transaction_id=assessment.assessment_id,
                fraud_score=assessment.risk_score,
                amount=assessment.outbound_volume or assessment.inbound_volume,
                tenant_id=profile.tenant_id,
                metadata={
                    "source": "mule_detection",
                    "account_id": profile.account_id,
                    "action": action.value,
                    "anomalies": [a.value for a in anomalies],
                },
            )

        return assessment
