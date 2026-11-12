"""Money Mule Account & Rapid Fund Exfiltration Detection Engine.

Detects pass-through conduit accounts, rapid dwell-time compression,
fan-in/fan-out counterparty concentration, dormancy-to-burst transitions,
and high-risk off-ramp cash-out channels with zero external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from statistics import median
from typing import Any
from uuid import uuid4


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
