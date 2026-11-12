"""Unit and integration tests for Mule Account & Rapid Exfiltration Detection Engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fraud_detection.mule_detection import (
    HIGH_RISK_OFF_RAMP_CHANNELS,
    DormancyBurstDetector,
    ExfiltrationChannel,
    MuleAccountProfile,
    MuleAction,
    MuleAnomalyType,
    MuleAssessment,
    MuleNetworkAnalyzer,
    MuleRiskLevel,
    MuleTransferRecord,
    PassThroughAnalyzer,
    TransferDirection,
)


def test_mule_enums_and_high_risk_channels() -> None:
    assert MuleRiskLevel.LOW == "low"
    assert MuleRiskLevel.CRITICAL == "critical"
    assert MuleAction.HOLD_FUNDS == "hold_funds"
    assert MuleAction.ESCALATE_AML_SAR == "escalate_aml_sar"
    assert MuleAnomalyType.HIGH_PASS_THROUGH_RATIO.value == "high_pass_through_ratio"
    assert TransferDirection.INBOUND == "inbound"
    assert ExfiltrationChannel.CRYPTO_EXCHANGE in HIGH_RISK_OFF_RAMP_CHANNELS
    assert ExfiltrationChannel.ACH not in HIGH_RISK_OFF_RAMP_CHANNELS


def test_mule_models_roundtrip_serialization() -> None:
    ts = datetime(2026, 11, 12, 10, 0, 0, tzinfo=UTC)
    xfer = MuleTransferRecord(
        transfer_id="xfer_01",
        account_id="acct_100",
        counterparty_id="cp_ext_1",
        direction=TransferDirection.INBOUND,
        amount=4500.0,
        channel=ExfiltrationChannel.P2P_WALLET,
        timestamp=ts,
        device_id="dev_1",
        tenant_id="tenant_a",
        metadata={"note": "inbound funding"},
    )
    xfer_rt = MuleTransferRecord.from_dict(xfer.to_dict())
    assert xfer_rt.transfer_id == "xfer_01"
    assert xfer_rt.direction == TransferDirection.INBOUND
    assert xfer_rt.channel == ExfiltrationChannel.P2P_WALLET
    assert xfer_rt.timestamp == ts

    profile = MuleAccountProfile(
        account_id="acct_100",
        tenant_id="tenant_a",
        account_opened_at=ts,
        transfers=[xfer],
        last_assessed_at=ts,
        metadata={"tier": "standard"},
    )
    prof_rt = MuleAccountProfile.from_dict(profile.to_dict())
    assert prof_rt.account_id == "acct_100"
    assert len(prof_rt.transfers) == 1
    assert prof_rt.last_assessed_at == ts

    assessment = MuleAssessment(
        assessment_id="mule_01",
        account_id="acct_100",
        tenant_id="tenant_a",
        risk_score=0.78,
        risk_level=MuleRiskLevel.HIGH,
        action=MuleAction.FREEZE_ACCOUNT,
        triggered_anomalies=[MuleAnomalyType.HIGH_PASS_THROUGH_RATIO],
        anomaly_breakdown={"pass_through": 0.45},
        pass_through_ratio=0.92,
        median_dwell_hours=1.5,
        unique_inbound_counterparties=4,
        unique_outbound_counterparties=2,
        inbound_volume=9000.0,
        outbound_volume=8280.0,
        evaluated_at=ts,
    )
    ass_rt = MuleAssessment.from_dict(assessment.to_dict())
    assert ass_rt.assessment_id == "mule_01"
    assert ass_rt.risk_level == MuleRiskLevel.HIGH
    assert ass_rt.action == MuleAction.FREEZE_ACCOUNT
    assert ass_rt.triggered_anomalies == [MuleAnomalyType.HIGH_PASS_THROUGH_RATIO]
    assert ass_rt.median_dwell_hours == 1.5


def test_pass_through_analyzer_rapid_cash_out_and_empty() -> None:
    analyzer = PassThroughAnalyzer()
    empty = analyzer.analyze([])
    assert empty.pass_through_ratio == 0.0
    assert empty.median_dwell_hours is None
    assert empty.anomalies == []

    t0 = datetime(2026, 11, 12, 12, 0, 0, tzinfo=UTC)
    transfers = [
        MuleTransferRecord(
            transfer_id="in_1",
            account_id="acct_mule",
            counterparty_id="victim_1",
            direction=TransferDirection.INBOUND,
            amount=5000.0,
            channel=ExfiltrationChannel.ACH,
            timestamp=t0,
        ),
        MuleTransferRecord(
            transfer_id="out_1",
            account_id="acct_mule",
            counterparty_id="crypto_1",
            direction=TransferDirection.OUTBOUND,
            amount=4800.0,
            channel=ExfiltrationChannel.CRYPTO_EXCHANGE,
            timestamp=t0 + timedelta(minutes=45),
        ),
    ]
    metrics = analyzer.analyze(transfers, now=t0 + timedelta(hours=1))
    assert metrics.inbound_volume == 5000.0
    assert metrics.outbound_volume == 4800.0
    assert metrics.pass_through_ratio == 0.96
    assert metrics.median_dwell_hours == 0.75
    assert metrics.high_risk_off_ramp_ratio == 1.0
    assert MuleAnomalyType.HIGH_PASS_THROUGH_RATIO in metrics.anomalies
    assert MuleAnomalyType.RAPID_DWELL_COMPRESSION in metrics.anomalies
    assert MuleAnomalyType.HIGH_RISK_OFF_RAMP in metrics.anomalies
    assert metrics.risk_score == 1.0


def test_mule_network_analyzer_fan_in_fan_out_and_new_counterparties() -> None:
    net = MuleNetworkAnalyzer(fan_in_threshold=3, fan_out_threshold=3, new_counterparty_threshold=4)
    empty = net.analyze([])
    assert empty.unique_inbound_counterparties == 0
    assert empty.anomalies == []

    t0 = datetime(2026, 11, 12, 10, 0, 0, tzinfo=UTC)
    transfers = [
        MuleTransferRecord(
            transfer_id=f"in_{i}",
            account_id="acct_hub",
            counterparty_id=f"src_{i}",
            direction=TransferDirection.INBOUND,
            amount=1000.0,
            timestamp=t0 + timedelta(minutes=i * 10),
        )
        for i in range(3)
    ] + [
        MuleTransferRecord(
            transfer_id=f"out_{i}",
            account_id="acct_hub",
            counterparty_id=f"dst_{i}",
            direction=TransferDirection.OUTBOUND,
            amount=950.0,
            timestamp=t0 + timedelta(hours=1, minutes=i * 10),
        )
        for i in range(3)
    ]
    res = net.analyze(transfers, now=t0 + timedelta(hours=2))
    assert res.unique_inbound_counterparties == 3
    assert res.unique_outbound_counterparties == 3
    assert res.new_counterparties_24h == 6
    assert MuleAnomalyType.FAN_IN_COLLECTION in res.anomalies
    assert MuleAnomalyType.FAN_OUT_DISPERSION in res.anomalies
    assert MuleAnomalyType.NEW_COUNTERPARTY_FLOOD in res.anomalies
    assert res.risk_score == 1.0


def test_dormancy_burst_detector() -> None:
    detector = DormancyBurstDetector(
        dormancy_days=30, burst_min_transfers=3, burst_min_volume=2000.0
    )
    opened = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    empty_prof = MuleAccountProfile(account_id="acct_dormant", account_opened_at=opened)
    assert detector.evaluate(empty_prof) == (False, 0.0, "")

    burst_time = datetime(2026, 11, 12, 12, 0, 0, tzinfo=UTC)
    burst_transfers = [
        MuleTransferRecord(
            transfer_id=f"b_{i}",
            account_id="acct_dormant",
            counterparty_id=f"cp_{i}",
            direction=TransferDirection.INBOUND if i < 2 else TransferDirection.OUTBOUND,
            amount=1200.0,
            timestamp=burst_time + timedelta(minutes=i * 15),
        )
        for i in range(3)
    ]
    prof = MuleAccountProfile(
        account_id="acct_dormant",
        account_opened_at=opened,
        transfers=burst_transfers,
    )
    is_burst, score, reason = detector.evaluate(prof, now=burst_time + timedelta(hours=1))
    assert is_burst is True
    assert score == 0.45
    assert "dormant" in reason.lower()
