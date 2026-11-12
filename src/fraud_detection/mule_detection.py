"""Money Mule Account & Rapid Fund Exfiltration Detection Engine.

Detects pass-through conduit accounts, rapid dwell-time compression,
fan-in/fan-out counterparty concentration, dormancy-to-burst transitions,
and high-risk off-ramp cash-out channels with zero external dependencies.
"""

from __future__ import annotations

from enum import StrEnum


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
