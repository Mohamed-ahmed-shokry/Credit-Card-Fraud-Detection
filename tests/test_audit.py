"""Tests for structured audit event export and redaction."""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any

import pytest

from fraud_detection import __version__
from fraud_detection.audit import (
    AuditEvent,
    AuditReplayReport,
    DiscrepancyDetail,
    JsonlAuditSink,
    NullAuditSink,
    _is_luhn_valid,
    _mask_pans_in_string,
    backtest_audit_policy,
    build_challenger_retrained_audit_event,
    build_delayed_label_audit_event,
    build_device_anomaly_audit_event,
    build_fraud_ring_detected_audit_event,
    build_impossible_travel_audit_event,
    build_merchant_anomaly_audit_event,
    build_model_promoted_audit_event,
    build_pipeline_chaos_audit_event,
    build_pipeline_replay_audit_event,
    build_promotion_audit_event,
    build_retraining_trigger_audit_event,
    build_scoring_audit_event,
    build_shadow_scoring_audit_event,
    build_tenant_audit_event,
    redact_data,
    replay_audit_log,
)
from fraud_detection.device import DeviceAnomalyReport
from fraud_detection.evaluation import TieredThresholds
from fraud_detection.feedback import (
    DelayedLabelRecord,
    DisputeType,
    RealizedMetricsReport,
    RetrainingTriggerEvent,
)
from fraud_detection.geolocation import (
    GeoCoordinates,
    GeoLocationEvent,
    TravelVelocity,
)
from fraud_detection.graph import (
    EntityType,
    FraudRingCluster,
    RingAction,
    RingPatternType,
    RingSeverity,
)
from fraud_detection.merchant import MerchantAnomalyReport


def test_is_luhn_valid() -> None:
    # Standard test card numbers (Luhn valid)
    assert _is_luhn_valid("4532015112830366") is True
    assert _is_luhn_valid("4532-0151-1283-0366") is True
    assert _is_luhn_valid("4532 0151 1283 0366") is True
    # Invalid check digit
    assert _is_luhn_valid("4532015112830367") is False
    # Length outside 13-19 digits
    assert _is_luhn_valid("123456789") is False
    assert _is_luhn_valid("1" * 25) is False


def test_mask_pans_in_string() -> None:
    text = "Transaction processed for card 4532 0151 1283 0366 with status OK"
    masked = _mask_pans_in_string(text)
    assert "4532 0151 1283 0366" not in masked
    assert "[REDACTED_PAN]" in masked

    non_pan = "Transaction ID 1234567890123456 amount 99.99"
    assert _mask_pans_in_string(non_pan) == non_pan


def test_redact_data_dictionary() -> None:
    payload = {
        "amount": 100.5,
        "token": "secret_session_token_123",
        "nested": {
            "credit_card": "4532015112830366",
            "cvv": "123",
            "safe_feature": 42,
        },
        "items": [
            {"password_hash": "abc", "status": "active"},
            "Note with PAN 4532-0151-1283-0366 inside",
        ],
        "nan_metric": float("nan"),
        "inf_metric": float("inf"),
    }
    redacted = redact_data(payload)

    assert redacted["amount"] == 100.5
    assert redacted["token"] == "[REDACTED]"  # noqa: S105
    assert redacted["nested"]["credit_card"] == "[REDACTED]"
    assert redacted["nested"]["cvv"] == "[REDACTED]"
    assert redacted["nested"]["safe_feature"] == 42
    assert redacted["items"][0]["password_hash"] == "[REDACTED]"  # noqa: S105
    assert redacted["items"][0]["status"] == "active"
    assert "[REDACTED_PAN]" in redacted["items"][1]
    assert redacted["nan_metric"] is None
    assert redacted["inf_metric"] is None


def test_audit_event_serialization() -> None:
    event = AuditEvent(
        event_type="scoring",
        model_version="abc123456789",
        dataset_fingerprint="def987654321",
        payload={
            "api_key": "super_secret",
            "probability": 0.85,
        },
    )
    serialized = event.to_json()
    parsed = json.loads(serialized)

    assert parsed["event_type"] == "scoring"
    assert parsed["model_version"] == "abc123456789"
    assert parsed["dataset_fingerprint"] == "def987654321"
    assert parsed["code_version"] == __version__
    assert parsed["payload"]["probability"] == 0.85
    assert parsed["payload"]["api_key"] == "[REDACTED]"

    # Unredacted mode
    raw_dict = event.to_dict(redact=False)
    assert raw_dict["payload"]["api_key"] == "super_secret"


def test_null_audit_sink() -> None:
    sink = NullAuditSink()
    event = AuditEvent(event_type="test")
    # Must not raise
    sink.emit(event)
    sink.close()


def test_jsonl_audit_sink(tmp_path: Path) -> None:
    sink_path = tmp_path / "logs" / "audit.jsonl"
    sink = JsonlAuditSink(sink_path)

    events = [
        AuditEvent(event_type="scoring", payload={"index": i, "token": f"sec_{i}"})
        for i in range(5)
    ]
    for e in events:
        sink.emit(e)
    sink.close()

    lines = sink_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 5
    for idx, line in enumerate(lines):
        record = json.loads(line)
        assert record["event_type"] == "scoring"
        assert record["payload"]["index"] == idx
        assert record["payload"]["token"] == "[REDACTED]"  # noqa: S105


def test_jsonl_audit_sink_thread_safety(tmp_path: Path) -> None:
    sink_path = tmp_path / "concurrent_audit.jsonl"
    sink = JsonlAuditSink(sink_path)
    total_events = 50

    def write_event(idx: int) -> None:
        sink.emit(AuditEvent(event_type="concurrent", payload={"idx": idx}))

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(write_event, i) for i in range(total_events)]
        concurrent.futures.wait(futures)

    lines = sink_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == total_events
    indices = {json.loads(line)["payload"]["idx"] for line in lines}
    assert indices == set(range(total_events))


def test_build_scoring_audit_event() -> None:
    event = build_scoring_audit_event(
        model_version="mod1",
        dataset_fingerprint="fp1",
        threshold=0.5,
        predictions=[
            {"fraud_probability": 0.9, "is_fraud": True},
            {"fraud_probability": 0.1, "is_fraud": False},
        ],
        request_id="req_999",
        client_metadata={"client_ip": "127.0.0.1"},
    )
    assert event.event_type == "scoring"
    assert event.payload["batch_size"] == 2
    assert event.payload["threshold"] == 0.5
    assert event.payload["fraud_count"] == 1
    assert event.payload["request_id"] == "req_999"
    assert event.payload["client_metadata"] == {"client_ip": "127.0.0.1"}


def test_build_scoring_audit_event_multi_model_routing() -> None:
    event = build_scoring_audit_event(
        model_version="champ_v1",
        dataset_fingerprint="fp1",
        threshold=0.5,
        predictions=[{"fraud_probability": 0.8, "is_fraud": True}],
        model_role="challenger",
        routed_model_version="chall_v2",
        shadow_predictions=[{"shadow_probability": 0.2, "shadow_decision": False}],
    )
    assert event.event_type == "scoring"
    assert event.payload["model_role"] == "challenger"
    assert event.payload["routed_model_version"] == "chall_v2"
    assert len(event.payload["shadow_predictions"]) == 1
    assert event.payload["shadow_predictions"][0]["shadow_probability"] == 0.2


def test_build_shadow_scoring_audit_event() -> None:
    event = build_shadow_scoring_audit_event(
        shadow_model_version="chall_v2",
        shadow_dataset_fingerprint="fp_chall",
        evaluated_count=10,
        discrepancy_count=2,
        discrepancies=[{"index": 0, "primary_decision": True, "shadow_decision": False}],
        request_id="req_shadow_123",
        mean_probability_divergence=0.08,
        max_probability_divergence=0.25,
        primary_model_version="champ_v1",
    )
    assert event.event_type == "shadow_scoring"
    assert event.model_version == "chall_v2"
    assert event.dataset_fingerprint == "fp_chall"
    assert event.payload["evaluated_count"] == 10
    assert event.payload["discrepancy_count"] == 2
    assert len(event.payload["discrepancies"]) == 1
    assert event.payload["request_id"] == "req_shadow_123"
    assert event.payload["mean_probability_divergence"] == 0.08
    assert event.payload["max_probability_divergence"] == 0.25
    assert event.payload["primary_model_version"] == "champ_v1"


def test_build_promotion_audit_event() -> None:
    event = build_promotion_audit_event(
        model_version="mod1",
        dataset_fingerprint="fp1",
        bundle_summary={"status": "promoted", "gate_count": 3},
        metadata={"operator": "ci-bot"},
    )
    assert event.event_type == "promotion"
    assert event.payload["bundle"]["status"] == "promoted"
    assert event.payload["metadata"]["operator"] == "ci-bot"

    # With metadata=None
    event_no_meta = build_promotion_audit_event(
        model_version="mod2",
        dataset_fingerprint="fp2",
        bundle_summary={"status": "rejected"},
    )
    assert "metadata" not in event_no_meta.payload


def test_redact_data_without_pan_masking() -> None:
    text = "Card 4532-0151-1283-0366 present"
    result = redact_data(text, mask_pan_values=False)
    assert result == text


class DummyFraudModel:
    """Mock model for fast, deterministic audit replay unit testing."""

    def __init__(
        self,
        threshold: float = 0.5,
        probs: list[float] | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.threshold = threshold
        self.probs = probs or [0.1, 0.9]
        self.feature_names = feature_names or ["V1", "V2", "Amount"]

    def predict_probabilities(self, frame: Any) -> list[float]:
        n = len(frame)
        if len(self.probs) < n:
            return (self.probs * ((n // len(self.probs)) + 1))[:n]
        return self.probs[:n]


def test_build_scoring_audit_event_features_and_fallback() -> None:
    event = build_scoring_audit_event(
        model_version="mod1",
        dataset_fingerprint="fp1",
        threshold=0.5,
        predictions=[{"fraud_probability": 0.9, "is_fraud": True}],
        features=[{"V1": 0.1, "V2": 0.2, "Amount": 50.0, "card_number": "4532015112830366"}],
        fallback_applied=True,
        fallback_reason="Estimator degraded",
    )
    assert event.payload["fallback_applied"] is True
    assert event.payload["fallback_reason"] == "Estimator degraded"
    assert "features" in event.payload
    redacted = event.to_dict(redact=True)
    assert redacted["payload"]["features"][0]["card_number"] == "[REDACTED]"
    assert redacted["payload"]["features"][0]["Amount"] == 50.0


def test_replay_audit_log_match(tmp_path: Path) -> None:
    log_file = tmp_path / "audit.jsonl"
    model = DummyFraudModel(threshold=0.5, probs=[0.05, 0.95])

    event = build_scoring_audit_event(
        model_version="v1",
        dataset_fingerprint="fp1",
        threshold=0.5,
        predictions=[
            {"fraud_probability": 0.05, "is_fraud": False},
            {"fraud_probability": 0.95, "is_fraud": True},
        ],
        features=[
            {"V1": 1.0, "V2": 2.0, "Amount": 10.0},
            {"V1": -1.0, "V2": -2.0, "Amount": 100.0},
        ],
    )
    log_file.write_text(event.to_json() + "\n", encoding="utf-8")

    report = replay_audit_log(log_file, model)
    assert isinstance(report, AuditReplayReport)
    assert report.status == "MATCH"
    assert report.total_events == 1
    assert report.scoring_events == 1
    assert report.replayed_events == 1
    assert report.skipped_events == 0
    assert report.total_transactions == 2
    assert report.score_discrepancies == 0
    assert report.decision_flips == 0
    assert report.max_absolute_difference < 1e-4
    assert len(report.discrepancies) == 0

    as_dict = report.to_dict()
    assert as_dict["status"] == "MATCH"


def test_replay_audit_log_divergence(tmp_path: Path) -> None:
    log_file = tmp_path / "audit.jsonl"
    # Challenger model predicts different scores
    model = DummyFraudModel(threshold=0.5, probs=[0.85, 0.10])

    event = build_scoring_audit_event(
        model_version="v1",
        dataset_fingerprint="fp1",
        threshold=0.5,
        predictions=[
            {"fraud_probability": 0.05, "is_fraud": False},
            {"fraud_probability": 0.95, "is_fraud": True},
        ],
        features=[
            {"V1": 1.0, "V2": 2.0, "Amount": 10.0, "extra_col": 999},
            {"V1": -1.0, "V2": -2.0, "Amount": 100.0, "extra_col": 888},
        ],
    )
    log_file.write_text(event.to_json() + "\n", encoding="utf-8")

    report = replay_audit_log(log_file, model, tolerance=0.01)
    assert isinstance(report, AuditReplayReport)
    assert report.status == "DIVERGENT"
    assert report.score_discrepancies == 2
    assert report.decision_flips == 2
    assert report.max_absolute_difference > 0.7
    assert len(report.discrepancies) == 2
    first_disc = report.discrepancies[0]
    assert isinstance(first_disc, DiscrepancyDetail)
    assert first_disc.decision_flipped is True
    assert first_disc.to_dict()["decision_flipped"] is True


def test_replay_audit_log_external_data(tmp_path: Path) -> None:
    log_file = tmp_path / "audit.jsonl"
    data_file = tmp_path / "tx.csv"
    data_file.write_text("V1,V2,Amount\n1.0,2.0,10.0\n-1.0,-2.0,100.0\n", encoding="utf-8")

    model = DummyFraudModel(threshold=0.5, probs=[0.05, 0.95])
    # Audit event without inline features
    event = build_scoring_audit_event(
        model_version="v1",
        dataset_fingerprint="fp1",
        threshold=0.5,
        predictions=[
            {"fraud_probability": 0.05, "is_fraud": False},
            {"fraud_probability": 0.95, "is_fraud": True},
        ],
    )
    log_file.write_text(event.to_json() + "\n", encoding="utf-8")

    report = replay_audit_log(log_file, model, data_path=data_file)
    assert report.status == "MATCH"
    assert report.replayed_events == 1
    assert report.total_transactions == 2


def test_replay_audit_log_empty_and_skipped(tmp_path: Path) -> None:
    log_file = tmp_path / "empty_audit.jsonl"
    # Write a promotion event (not scoring) and an event without features
    promo = build_promotion_audit_event(
        model_version="v1", dataset_fingerprint="fp1", bundle_summary={"status": "ok"}
    )
    no_feat = build_scoring_audit_event(
        model_version="v1",
        dataset_fingerprint="fp1",
        threshold=0.5,
        predictions=[{"fraud_probability": 0.1, "is_fraud": False}],
    )
    log_file.write_text(promo.to_json() + "\n\n" + no_feat.to_json() + "\n", encoding="utf-8")

    model = DummyFraudModel()
    report = replay_audit_log(log_file, model)
    assert report.status == "EMPTY"
    assert report.scoring_events == 1
    assert report.skipped_events == 1
    assert report.replayed_events == 0


def test_replay_audit_log_missing_file(tmp_path: Path) -> None:
    import pytest

    missing_log = tmp_path / "does_not_exist.jsonl"
    model = DummyFraudModel()
    with pytest.raises(FileNotFoundError, match="Audit log file not found"):
        replay_audit_log(missing_log, model)

    log_file = tmp_path / "valid.jsonl"
    log_file.write_text("", encoding="utf-8")
    missing_data = tmp_path / "missing_data.csv"
    with pytest.raises(FileNotFoundError, match="Transaction data file not found"):
        replay_audit_log(log_file, model, data_path=missing_data)


def test_audit_event_fallback_without_reason() -> None:
    event = build_scoring_audit_event(
        model_version="v1",
        dataset_fingerprint="fp1",
        threshold=0.5,
        predictions=[{"fraud_probability": 0.1, "is_fraud": False}],
        fallback_applied=True,
        fallback_reason=None,
    )
    assert event.payload["fallback_applied"] is True
    assert "fallback_reason" not in event.payload


def test_replay_audit_log_skip_edge_cases(tmp_path: Path) -> None:
    log_file = tmp_path / "skip_edge_cases.jsonl"
    lines = [
        "not-valid-json",
        json.dumps(["not", "a", "dict"]),
        json.dumps({"event_type": "promotion", "payload": {}}),
        json.dumps({"event_type": "scoring", "payload": "not-a-dict"}),
        json.dumps({"event_type": "scoring", "payload": {"predictions": []}}),
        json.dumps(
            {
                "event_type": "scoring",
                "payload": {
                    "features": [{"V1": 1.0}],
                    "predictions": ["not-a-dict-prediction"],
                },
            }
        ),
    ]
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    model = DummyFraudModel(threshold=0.5, probs=[0.1])
    report = replay_audit_log(log_file, model)
    assert report.total_events == 6
    assert report.scoring_events == 3
    assert report.skipped_events == 2
    assert report.replayed_events == 1
    assert report.total_transactions == 0

    # Test external offset exhausted
    data_file = tmp_path / "one_row.csv"
    data_file.write_text("V1,Amount\n1.0,10.0\n", encoding="utf-8")
    two_preds_log = tmp_path / "two_preds.jsonl"
    two_preds_log.write_text(
        json.dumps(
            {
                "event_type": "scoring",
                "payload": {
                    "predictions": [
                        {"fraud_probability": 0.1, "is_fraud": False},
                        {"fraud_probability": 0.9, "is_fraud": True},
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report_exhausted = replay_audit_log(two_preds_log, model, data_path=data_file)
    assert report_exhausted.skipped_events == 1

    # Test model prediction exception during replay
    class FailingModel:
        def __init__(self) -> None:
            self.feature_names = ["V1"]

        def predict_probabilities(self, _df: Any) -> Any:
            raise RuntimeError("Inference breakdown")

    fail_log = tmp_path / "fail.jsonl"
    fail_log.write_text(
        json.dumps(
            {
                "event_type": "scoring",
                "payload": {
                    "features": [{"V1": 1.0}],
                    "predictions": [{"fraud_probability": 0.1, "is_fraud": False}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report_fail = replay_audit_log(fail_log, FailingModel())
    assert report_fail.skipped_events == 1


def test_replay_audit_log_validation_and_robustness(tmp_path: Path) -> None:
    log_file = tmp_path / "robust.jsonl"
    log_file.write_text("", encoding="utf-8")
    model = DummyFraudModel(threshold=0.5, probs=[0.2])

    # Validation errors on inputs
    with pytest.raises(ValueError, match="tolerance must be a finite, non-negative float"):
        replay_audit_log(log_file, model, tolerance=float("nan"))
    with pytest.raises(ValueError, match="tolerance must be a finite, non-negative float"):
        replay_audit_log(log_file, model, tolerance=-0.01)
    with pytest.raises(ValueError, match="tolerance must be a finite, non-negative float"):
        replay_audit_log(log_file, model, tolerance=True)

    with pytest.raises(ValueError, match="max_discrepancies_to_record must be"):
        replay_audit_log(log_file, model, max_discrepancies_to_record=-1)
    with pytest.raises(ValueError, match="max_discrepancies_to_record must be"):
        replay_audit_log(log_file, model, max_discrepancies_to_record=True)

    with pytest.raises(ValueError, match=r"threshold must be a finite float between 0\.0 and 1\.0"):
        replay_audit_log(log_file, model, threshold=1.5)
    with pytest.raises(ValueError, match=r"threshold must be a finite float between 0\.0 and 1\.0"):
        replay_audit_log(log_file, model, threshold=-0.1)

    # Payload with null threshold and non-numeric probabilities
    events = [
        # Null threshold in payload, valid predictions
        json.dumps(
            {
                "event_type": "scoring",
                "payload": {
                    "threshold": None,
                    "features": [{"V1": 1.0}],
                    "predictions": [{"fraud_probability": 0.2, "is_fraud": False}],
                },
            }
        ),
        # Non-numeric probability should be skipped gracefully
        json.dumps(
            {
                "event_type": "scoring",
                "payload": {
                    "threshold": 0.5,
                    "features": [{"V1": 1.0}],
                    "predictions": [{"fraud_probability": "not-numeric", "is_fraud": False}],
                },
            }
        ),
        # Length mismatch between features (2) and predictions (1) should skip event
        json.dumps(
            {
                "event_type": "scoring",
                "payload": {
                    "threshold": 0.5,
                    "features": [{"V1": 1.0}, {"V1": 2.0}],
                    "predictions": [{"fraud_probability": 0.2, "is_fraud": False}],
                },
            }
        ),
    ]
    log_file.write_text("\n".join(events) + "\n", encoding="utf-8")
    report = replay_audit_log(log_file, model)
    assert report.total_events == 3
    assert report.scoring_events == 3
    assert report.replayed_events == 2
    assert report.skipped_events == 1  # length mismatch skipped
    assert report.total_transactions == 1  # only first event had 1 valid transaction
    assert report.status == "MATCH"

    # When zero transactions are evaluated, status must be EMPTY
    empty_log = tmp_path / "empty_tx.jsonl"
    empty_log.write_text(
        json.dumps(
            {
                "event_type": "scoring",
                "payload": {
                    "features": [{"V1": 1.0}],
                    "predictions": [{"fraud_probability": "corrupt", "is_fraud": False}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report_empty = replay_audit_log(empty_log, model)
    assert report_empty.total_transactions == 0
    assert report_empty.status == "EMPTY"


def test_build_scoring_audit_event_tiered_thresholds_and_decisions() -> None:
    # 1. Automatic decision_counts derivation
    event = build_scoring_audit_event(
        model_version="v1",
        dataset_fingerprint="fp1",
        threshold=0.5,
        review_threshold=0.2,
        deny_threshold=0.7,
        predictions=[
            {"fraud_probability": 0.1, "is_fraud": False, "decision": "ALLOW"},
            {"fraud_probability": 0.4, "is_fraud": False, "decision": "CHALLENGE"},
            {"fraud_probability": 0.8, "is_fraud": True, "decision": "DENY"},
        ],
    )
    assert event.payload["review_threshold"] == 0.2
    assert event.payload["deny_threshold"] == 0.7
    assert event.payload["decision_counts"] == {"ALLOW": 1, "CHALLENGE": 1, "DENY": 1}

    # 2. Explicit decision_counts override
    event_explicit = build_scoring_audit_event(
        model_version="v1",
        dataset_fingerprint="fp1",
        threshold=0.5,
        predictions=[],
        decision_counts={"ALLOW": 5, "DENY": 2},
    )
    assert event_explicit.payload["decision_counts"] == {"ALLOW": 5, "DENY": 2}


def test_backtest_audit_policy_explicit_and_inferred_baseline(tmp_path: Path) -> None:
    log_file = tmp_path / "audit_backtest.jsonl"
    events = [
        # Scoring event 1 with tiered thresholds in payload
        json.dumps(
            {
                "event_type": "scoring",
                "payload": {
                    "review_threshold": 0.25,
                    "deny_threshold": 0.75,
                    "predictions": [
                        {"fraud_probability": 0.1, "is_fraud": False},
                        {"fraud_probability": 0.5, "is_fraud": False},
                    ],
                },
            }
        ),
        # Non-scoring event should be skipped
        json.dumps({"event_type": "promotion", "payload": {}}),
        # Corrupt JSON line should be skipped
        "{invalid json",
        # Scoring event 2
        json.dumps(
            {
                "event_type": "scoring",
                "payload": {
                    "predictions": [
                        {"fraud_probability": 0.85, "is_fraud": True},
                    ],
                },
            }
        ),
    ]
    log_file.write_text("\n".join(events) + "\n", encoding="utf-8")

    candidate = TieredThresholds(0.3, 0.7)

    # 1. Inferred baseline from audit log (should find 0.25, 0.75 from event 1)
    report_inferred = backtest_audit_policy(log_file, candidate)
    assert report_inferred.rows == 3
    assert report_inferred.baseline_thresholds == TieredThresholds(0.25, 0.75)
    assert report_inferred.candidate_thresholds == candidate
    assert report_inferred.transition_matrix.total_records == 3

    # 2. Explicit baseline override
    explicit_baseline = TieredThresholds(0.2, 0.8)
    report_explicit = backtest_audit_policy(
        log_file,
        candidate,
        baseline_thresholds=explicit_baseline,
    )
    assert report_explicit.baseline_thresholds == explicit_baseline
    assert report_explicit.candidate_thresholds == candidate

    # 3. Ground truth evaluation
    y_true = [0, 0, 1]
    report_with_labels = backtest_audit_policy(
        log_file,
        candidate,
        baseline_thresholds=explicit_baseline,
        y_true=y_true,
    )
    assert report_with_labels.cost_delta is not None
    assert report_with_labels.fraud_catch_delta is not None


def test_backtest_audit_policy_inferred_binary_threshold(tmp_path: Path) -> None:
    log_file = tmp_path / "binary_audit.jsonl"
    event = json.dumps(
        {
            "event_type": "scoring",
            "payload": {
                "threshold": 0.6,
                "predictions": [
                    {"fraud_probability": 0.4},
                    {"fraud_probability": 0.7},
                ],
            },
        }
    )
    log_file.write_text(event + "\n", encoding="utf-8")

    candidate = TieredThresholds(0.3, 0.8)
    report = backtest_audit_policy(log_file, candidate)
    assert report.baseline_thresholds == TieredThresholds(0.6, 0.6)
    assert report.rows == 2


def test_backtest_audit_policy_default_fallback_baseline(tmp_path: Path) -> None:
    log_file = tmp_path / "fallback_audit.jsonl"
    event = json.dumps(
        {
            "event_type": "scoring",
            "payload": {
                "predictions": [
                    {"fraud_probability": 0.3},
                    {"fraud_probability": 0.7},
                ],
            },
        }
    )
    log_file.write_text(event + "\n", encoding="utf-8")

    candidate = TieredThresholds(0.2, 0.8)
    report = backtest_audit_policy(log_file, candidate)
    assert report.baseline_thresholds == TieredThresholds(0.5, 0.5)


def test_backtest_audit_policy_validation_errors(tmp_path: Path) -> None:
    candidate = TieredThresholds(0.2, 0.8)

    # File not found
    with pytest.raises(FileNotFoundError, match="Audit log file not found"):
        backtest_audit_policy(tmp_path / "nonexistent.jsonl", candidate)

    # Candidate not TieredThresholds
    with pytest.raises(TypeError, match="candidate_thresholds must be a TieredThresholds instance"):
        backtest_audit_policy(tmp_path / "dummy.jsonl", (0.2, 0.8))  # type: ignore[arg-type]

    # Baseline not TieredThresholds
    log_file = tmp_path / "valid.jsonl"
    log_file.write_text(
        json.dumps(
            {
                "event_type": "scoring",
                "payload": {"predictions": [{"fraud_probability": 0.5}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="baseline_thresholds must be a TieredThresholds instance"):
        backtest_audit_policy(log_file, candidate, baseline_thresholds=0.5)  # type: ignore[arg-type]

    # Empty log / no valid scoring transactions
    empty_log = tmp_path / "empty.jsonl"
    empty_log.write_text(
        json.dumps({"event_type": "promotion", "payload": {}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No valid scoring transactions found in audit log"):
        backtest_audit_policy(empty_log, candidate)

    # y_true length mismatch
    with pytest.raises(
        ValueError,
        match=r"Length of y_true \(2\) does not match number of scored transactions \(1\)",
    ):
        backtest_audit_policy(log_file, candidate, y_true=[0, 1])


def test_build_scoring_audit_event_feature_store_enrichment() -> None:
    event = build_scoring_audit_event(
        model_version="v1",
        dataset_fingerprint="fp1",
        threshold=0.5,
        predictions=[{"fraud_probability": 0.3, "is_fraud": False}],
        enriched_features=["user_risk_score", "card_velocity_1h"],
        feature_views_applied=["user_profile", "card_profile"],
    )
    assert event.payload["enriched_features"] == ["user_risk_score", "card_velocity_1h"]
    assert event.payload["feature_views_applied"] == ["user_profile", "card_profile"]


def test_build_scoring_audit_event_pipeline_execution_trace() -> None:
    stages = [
        {"node_name": "enrichment", "status": "success", "latency_ms": 1.2},
        {"node_name": "inference", "status": "timeout", "latency_ms": 50.0, "degraded": True},
    ]
    event = build_scoring_audit_event(
        model_version="v1",
        dataset_fingerprint="fp1",
        threshold=0.5,
        predictions=[{"fraud_probability": 0.3, "is_fraud": False}],
        execution_trace={"execution_path": ["enrichment", "inference"], "total_latency_ms": 51.2},
        pipeline_stages=stages,
        degraded_nodes=["inference"],
    )
    assert event.payload["execution_trace"]["execution_path"] == ["enrichment", "inference"]
    assert len(event.payload["pipeline_stages"]) == 2
    assert event.payload["degraded_nodes"] == ["inference"]

    # Verify serialization and JSON redaction safety
    serialized = event.to_json(redact=True)
    parsed = json.loads(serialized)
    assert parsed["payload"]["degraded_nodes"] == ["inference"]


def test_build_scoring_audit_event_cache_and_speculative() -> None:
    event = build_scoring_audit_event(
        model_version="v2",
        dataset_fingerprint="fp2",
        threshold=0.5,
        predictions=[{"fraud_probability": 0.1, "is_fraud": False}],
        cache_hits=2,
        cache_misses=1,
        speculative_executed=True,
        speculative_hit=True,
    )
    assert event.payload["cache_hits"] == 2
    assert event.payload["cache_misses"] == 1
    assert event.payload["speculative_executed"] is True
    assert event.payload["speculative_hit"] is True

    serialized = event.to_json(redact=True)
    parsed = json.loads(serialized)
    assert parsed["payload"]["cache_hits"] == 2
    assert parsed["payload"]["speculative_hit"] is True


def test_build_pipeline_replay_audit_event() -> None:
    event = build_pipeline_replay_audit_event(
        pipeline_name="custom_graph",
        audit_source="/path/to/audit.jsonl",
        replayed_events=100,
        total_transactions=100,
        score_discrepancies=5,
        decision_flips=2,
        flip_rate=0.02,
        status="diverged",
        transition_matrix={"approve_to_review": 2},
        mean_latency_ms=1.45,
        p95_latency_ms=3.12,
        metadata={"env": "staging", "api_key": "secret_key_12345"},
    )
    assert event.event_type == "pipeline_replay"
    assert event.model_version == "custom_graph"
    assert event.payload["replayed_events"] == 100
    assert event.payload["flip_rate"] == 0.02
    assert event.payload["status"] == "diverged"
    assert event.payload["transition_matrix"] == {"approve_to_review": 2}

    # Verify redaction of sensitive metadata
    serialized = event.to_json(redact=True)
    parsed = json.loads(serialized)
    assert parsed["payload"]["metadata"]["api_key"] == "[REDACTED]"
    assert parsed["payload"]["metadata"]["env"] == "staging"

    minimal_event = build_pipeline_replay_audit_event(
        pipeline_name="min_graph",
        audit_source="/path/to/min.jsonl",
        replayed_events=10,
        total_transactions=10,
        score_discrepancies=0,
        decision_flips=0,
        flip_rate=0.0,
        status="MATCH",
    )
    assert "transition_matrix" not in minimal_event.payload
    assert "mean_latency_ms" not in minimal_event.payload
    assert "p95_latency_ms" not in minimal_event.payload
    assert "metadata" not in minimal_event.payload


def test_build_scoring_audit_event_with_circuit_short_circuits() -> None:
    event = build_scoring_audit_event(
        model_version="v2",
        dataset_fingerprint="fp123",
        threshold=0.5,
        predictions=[{"is_fraud": False, "decision": "ALLOW"}],
        circuit_short_circuits=["enrichment", "rules"],
    )
    assert event.payload["circuit_short_circuits"] == ["enrichment", "rules"]
    assert "enrichment" in event.to_dict()["payload"]["circuit_short_circuits"]


def test_build_pipeline_chaos_audit_event() -> None:
    raw_report = {
        "status": "RESILIENT",
        "total_transactions": 50,
        "circuit_breaker_enabled": True,
        "sla_budget_ms": 100.0,
        "sla_breaches": 0,
        "circuit_trips": 1,
        "short_circuited_calls": 5,
        "unhandled_errors": 0,
        "degraded_transactions": 5,
        "latencies_p50_ms": 1.2,
        "latencies_p95_ms": 3.4,
        "latencies_max_ms": 5.6,
        "recovered": True,
        "fault_specs": [{"node_name": "enrichment", "fault_type": "exception"}],
        "stage_metrics": {
            "enrichment": {
                "node_name": "enrichment",
                "fault_type": "exception",
                "injected_count": 2,
                "handled_count": 2,
                "tripped_circuit": True,
            }
        },
    }

    event = build_pipeline_chaos_audit_event(
        chaos_report=raw_report,
        model_version="model_chaos_v1",
        dataset_fingerprint="ds_chaos_fp",
        metadata={"environment": "canary", "api_key": "dummy_key_12345"},
    )

    assert event.event_type == "pipeline_chaos"
    assert event.model_version == "model_chaos_v1"
    assert event.dataset_fingerprint == "ds_chaos_fp"
    assert event.payload["simulation_type"] == "pipeline_chaos"
    assert event.payload["status"] == "RESILIENT"
    assert event.payload["total_transactions"] == 50
    assert event.payload["circuit_breaker_enabled"] is True
    assert event.payload["circuit_trips"] == 1
    assert event.payload["short_circuited_calls"] == 5
    assert event.payload["recovered"] is True

    # Check redaction of metadata
    d = event.to_dict(redact=True)
    assert d["payload"]["metadata"]["environment"] == "canary"
    assert d["payload"]["metadata"]["api_key"] == "[REDACTED]"


def test_build_delayed_label_audit_event() -> None:
    rec = DelayedLabelRecord(
        transaction_id="tx_audit_100",
        event_timestamp=1000.0,
        label_timestamp=87400.0,
        dispute_type=DisputeType.CHARGEBACK,
        is_fraud=True,
        amount=199.99,
        entity_key="card_12345",
        notes="Network report from issuer",
    )

    event = build_delayed_label_audit_event(
        record=rec,
        model_version="v2.1.0",
        dataset_fingerprint="ds_fp_123",
        metadata={"operator": "fraud_analyst", "user_id": "usr_999"},
    )

    assert event.event_type == "delayed_label_ingestion"
    assert event.model_version == "v2.1.0"
    assert event.dataset_fingerprint == "ds_fp_123"
    assert event.payload["transaction_id"] == "tx_audit_100"
    assert event.payload["event_timestamp"] == 1000.0
    assert event.payload["label_timestamp"] == 87400.0
    assert event.payload["delay_days"] == 1.0
    assert event.payload["dispute_type"] == "chargeback"
    assert event.payload["is_fraud"] is True
    assert event.payload["amount"] == 199.99
    assert event.payload["ingestion_status"] == "ingested"

    # Redaction verification
    d = event.to_dict(redact=True)
    assert d["payload"]["metadata"]["operator"] == "fraud_analyst"
    assert d["payload"]["metadata"]["user_id"] == "[REDACTED]"

    # Test with dict record without metadata
    event_dict = build_delayed_label_audit_event(
        record={
            "transaction_id": "tx_audit_101",
            "event_timestamp": 2000.0,
            "label_timestamp": 90000.0,
            "dispute_type": "fraud_dispute",
            "is_fraud": False,
            "amount": 50.0,
        }
    )
    assert event_dict.payload["transaction_id"] == "tx_audit_101"
    assert event_dict.payload["is_fraud"] is False
    assert "metadata" not in event_dict.payload


def test_build_retraining_trigger_audit_event() -> None:
    report = RealizedMetricsReport(
        total_transactions=20,
        actual_fraud_count=5,
        actual_fraud_rate=0.25,
        predicted_positive_count=6,
        true_positives=3,
        false_positives=3,
        true_negatives=12,
        false_negatives=2,
        realized_precision=0.5,
        realized_recall=0.6,
        realized_f1=0.5455,
        realized_fpr=0.2,
        realized_auprc=0.48,
        total_fraud_loss=500.0,
        prevented_fraud_amount=300.0,
        missed_fraud_amount=200.0,
        false_positive_cost=45.0,
        net_financial_savings=255.0,
        decision_counts={"deny": 6, "allow": 14},
        cohort_id="c_2026_w41",
    )
    trig = RetrainingTriggerEvent(
        trigger_id="trig_retrain_1",
        triggered_at=1700000000.0,
        breached_metrics={"precision": {"observed": 0.5, "threshold": 0.7}},
        realized_metrics=report,
        recommended_retrain_start=1680000000.0,
        recommended_retrain_end=1698000000.0,
        recommended_temporal_gap_seconds=1209600.0,
        reason="precision breached (0.5 vs threshold 0.7)",
        cohort_id="c_2026_w41",
    )

    event = build_retraining_trigger_audit_event(
        trigger_event=trig,
        model_version="v2.1.0",
        metadata={"initiator": "surveillance_cron", "token": "cron_tok_xyz"},
    )

    assert event.event_type == "retraining_trigger"
    assert event.model_version == "v2.1.0"
    assert event.payload["trigger_id"] == "trig_retrain_1"
    assert event.payload["triggered_at"] == 1700000000.0
    assert event.payload["cohort_id"] == "c_2026_w41"
    assert event.payload["breached_metrics"]["precision"]["observed"] == 0.5
    assert event.payload["recommended_temporal_gap_seconds"] == 1209600.0
    assert event.payload["realized_metrics"]["total_transactions"] == 20

    # Redaction verification
    d = event.to_dict(redact=True)
    assert d["payload"]["metadata"]["initiator"] == "surveillance_cron"
    assert d["payload"]["metadata"]["token"] == "[REDACTED]"  # noqa: S105

    # Test with dict trigger_event without metadata
    event_dict = build_retraining_trigger_audit_event(
        trigger_event={
            "trigger_id": "trig_dict_2",
            "triggered_at": 1700000000.0,
            "cohort_id": "c_dict",
            "reason": "fpr breached",
            "breached_metrics": {"fpr": {"observed": 0.1, "threshold": 0.05}},
            "recommended_retrain_start": 1600.0,
            "recommended_retrain_end": 1700.0,
            "recommended_temporal_gap_seconds": 100.0,
            "realized_metrics": {"total_transactions": 15},
        }
    )
    assert event_dict.payload["trigger_id"] == "trig_dict_2"
    assert "metadata" not in event_dict.payload


def test_build_challenger_retrained_audit_event_passed_and_rejected() -> None:
    """Test challenger retrained audit event creation for both passing and rejected runs."""
    cand_dict = {
        "model_version": "cand_v2",
        "dataset_fingerprint": "fp_retrain_123",
        "training_duration_seconds": 12.34,
        "validation_metrics": {"pr_auc": 0.88, "f1": 0.82},
        "test_metrics": {"pr_auc": 0.87, "f1": 0.81},
        "config": {"hyperparameters": {"regularization": 0.5}},
    }
    pass_report = {
        "status": "passed",
        "candidate_version": "cand_v2",
        "champion_version": "champ_v1",
        "summary": "Passed all safety gates",
    }

    # 1. Successful retraining run with attestation
    event_pass = build_challenger_retrained_audit_event(
        candidate_result=cand_dict,
        verification_report=pass_report,
        attestation_digest="sha256:abc1234567890",
        metadata={"user": "mlops_agent", "api_key": "secret_key_123"},
    )
    assert event_pass.event_type == "fraud.retraining.completed"
    assert event_pass.model_version == "cand_v2"
    assert event_pass.dataset_fingerprint == "fp_retrain_123"
    assert event_pass.payload["training_duration_seconds"] == 12.34
    assert event_pass.payload["verification_status"] == "passed"
    assert event_pass.payload["is_certified"] is True
    assert event_pass.payload["attestation_digest"] == "sha256:abc1234567890"

    # Redaction check
    redacted = event_pass.to_dict(redact=True)
    assert redacted["payload"]["metadata"]["user"] == "mlops_agent"
    assert redacted["payload"]["metadata"]["api_key"] == "[REDACTED]"

    # 2. Rejected retraining run
    fail_report = {
        "status": "failed",
        "candidate_version": "cand_v3",
        "champion_version": "champ_v1",
        "summary": "Degraded recall",
    }
    event_fail = build_challenger_retrained_audit_event(
        candidate_result=cand_dict,
        verification_report=fail_report,
    )
    assert event_fail.event_type == "fraud.retraining.rejected"
    assert event_fail.payload["verification_status"] == "failed"
    assert event_fail.payload["is_certified"] is False


def test_build_model_promoted_audit_event_full_and_canary() -> None:
    """Test model promotion audit event for full rollout and canary shift."""
    # 1. Full promotion
    event_full = build_model_promoted_audit_event(
        previous_champion_version="champ_v1",
        new_champion_version="cand_v2",
        promotion_type="full",
        canary_percentage=100.0,
        attestation_digest="digest_v2",
        promoted_by="release_pipeline",
        dataset_fingerprint="fp_promoted_999",
        metadata={"actor": "deployer", "auth_token": "token_abc"},
    )
    assert event_full.event_type == "fraud.model.promoted"
    assert event_full.model_version == "cand_v2"
    assert event_full.dataset_fingerprint == "fp_promoted_999"
    assert event_full.payload["previous_champion_version"] == "champ_v1"
    assert event_full.payload["new_champion_version"] == "cand_v2"
    assert event_full.payload["canary_percentage"] == 100.0
    assert event_full.payload["promoted_by"] == "release_pipeline"

    redacted = event_full.to_dict(redact=True)
    assert redacted["payload"]["metadata"]["actor"] == "deployer"
    assert redacted["payload"]["metadata"]["auth_token"] == "[REDACTED]"  # noqa: S105

    # 2. Canary update
    event_canary = build_model_promoted_audit_event(
        previous_champion_version="champ_v1",
        new_champion_version="cand_v2",
        promotion_type="canary",
        canary_percentage=20.0,
    )
    assert event_canary.event_type == "fraud.canary.updated"
    assert event_canary.payload["promotion_type"] == "canary"
    assert event_canary.payload["canary_percentage"] == 20.0


def test_build_fraud_ring_detected_audit_event_redaction() -> None:
    # 1. Cluster object with Luhn-valid PANs
    cluster = FraudRingCluster(
        cluster_id="ring_star_123",
        pattern_type=RingPatternType.STAR_CARD_TESTING,
        severity=RingSeverity.HIGH,
        recommended_action=RingAction.DENY,
        entities={
            EntityType.CARD_ID: ("4111111111111111", "5500000000000004"),
            EntityType.DEVICE_ID: ("dev_botnet_1",),
        },
        transaction_ids=("tx_1", "tx_2"),
        transaction_count=2,
        total_amount=150.0,
        density=1.0,
        risk_score=0.88,
        description="High velocity card testing on dev_botnet_1",
    )

    event = build_fraud_ring_detected_audit_event(
        cluster=cluster,
        transaction_id="tx_test_trigger",
        trigger_action="DENY",
        model_version="v2",
        metadata={"operator": "secops", "api_key": "secret_key_123"},
    )

    assert event.event_type == "fraud.graph.ring_detected"
    assert event.model_version == "v2"
    assert event.payload["cluster_id"] == "ring_star_123"
    assert event.payload["severity"] == "high"
    assert event.payload["recommended_action"] == "deny"
    assert event.payload["trigger_action"] == "DENY"
    assert event.payload["transaction_id"] == "tx_test_trigger"

    # Redaction checks: sensitive card entities and api_key must be masked
    d = event.to_dict(redact=True)
    assert d["payload"]["metadata"]["operator"] == "secops"
    assert d["payload"]["metadata"]["api_key"] == "[REDACTED]"
    # Card entities under 'card_id' key should be redacted
    assert d["payload"]["entities"]["card_id"] == "[REDACTED]"
    assert d["payload"]["entities"]["device_id"] == ["dev_botnet_1"]

    # 2. Test building from Mapping
    mapping_cluster = {
        "cluster_id": "ring_dict_456",
        "pattern_type": "biclique",
        "severity": "critical",
        "recommended_action": "deny",
        "entities": {"device_id": ["d1", "d2"]},
        "transaction_ids": ["tx_a", "tx_b"],
        "transaction_count": 2,
        "total_amount": 5000.0,
        "density": 0.9,
        "risk_score": 0.95,
        "description": "Dense syndicate ring",
    }
    event_dict = build_fraud_ring_detected_audit_event(
        cluster=mapping_cluster,
        transaction_id="tx_a",
    )
    assert event_dict.payload["cluster_id"] == "ring_dict_456"
    assert event_dict.payload["severity"] == "critical"
    assert event_dict.payload["total_amount"] == 5000.0


def test_build_impossible_travel_audit_event() -> None:
    # 1. Test building from typed objects and Luhn-valid PAN masking
    pan_entity = "4532015112830366"  # Luhn valid
    prev_loc = GeoCoordinates(latitude=51.5074, longitude=-0.1278, country_code="GB", city="London")
    curr_loc = GeoCoordinates(
        latitude=40.7128, longitude=-74.0060, country_code="US", city="New York"
    )

    prev_ev = GeoLocationEvent(
        event_id="ev_lon",
        entity_id=pan_entity,
        timestamp=1000.0,
        coordinates=prev_loc,
        ip_address="198.51.100.1",
    )
    curr_ev = GeoLocationEvent(
        event_id="ev_nyc",
        entity_id=pan_entity,
        timestamp=1600.0,
        coordinates=curr_loc,
        ip_address="198.51.100.2",
    )

    velocity = TravelVelocity(
        distance_km=5570.0,
        time_elapsed_seconds=600.0,
        velocity_kmh=33420.0,
        is_impossible_travel=True,
        is_cross_border=True,
        origin_country="GB",
        destination_country="US",
        origin_city="London",
        destination_city="New York",
    )

    event = build_impossible_travel_audit_event(
        velocity=velocity,
        transaction_id="tx_nyc",
        prev_event=prev_ev,
        curr_event=curr_ev,
        trigger_action="DENY",
        model_version="v2",
        metadata={"operator": "secops", "auth_token": "secret_jwt_xyz"},
    )

    assert event.event_type == "fraud.geo.impossible_travel"
    assert event.model_version == "v2"
    assert event.payload["transaction_id"] == "tx_nyc"
    assert event.payload["distance_km"] == 5570.0
    assert event.payload["velocity_kmh"] == 33420.0
    assert event.payload["is_impossible_travel"] is True
    assert event.payload["is_cross_border"] is True
    assert event.payload["trigger_action"] == "DENY"

    # Redaction checks: PAN in entity_id must be redacted
    assert event.payload["entity_id"] == "[REDACTED_PAN]"
    # Nested event entity IDs (Luhn valid) must also be redacted
    assert event.payload["prev_event"]["entity_id"] == "[REDACTED_PAN]"
    assert event.payload["curr_event"]["entity_id"] == "[REDACTED_PAN]"

    # Metadata token redaction
    d = event.to_dict(redact=True)
    assert d["payload"]["metadata"]["operator"] == "secops"
    assert d["payload"]["metadata"]["auth_token"] == "[REDACTED]"  # noqa: S105

    # 2. Test building from Mapping with non-PAN entity
    mapping_velocity = {
        "distance_km": 1200.0,
        "time_elapsed_seconds": 1800.0,
        "velocity_kmh": 2400.0,
        "is_impossible_travel": True,
        "is_cross_border": False,
        "origin_country": "US",
        "destination_country": "US",
        "origin_city": "Chicago",
        "destination_city": "New York",
    }
    mapping_event = build_impossible_travel_audit_event(
        velocity=mapping_velocity,
        entity_id="user_chicago_nyc_999",
        transaction_id="tx_map_1",
    )
    assert mapping_event.payload["entity_id"] == "user_chicago_nyc_999"
    assert mapping_event.payload["distance_km"] == 1200.0
    assert mapping_event.payload["velocity_kmh"] == 2400.0
    assert mapping_event.payload["is_impossible_travel"] is True


def test_build_device_anomaly_audit_event() -> None:
    pan_card = "4532015112830366"  # Luhn-valid PAN
    rep = DeviceAnomalyReport(
        device_id="device_attack_cluster_99",
        card_id=pan_card,
        is_emulator=True,
        is_card_cycling=True,
        is_device_cycling=False,
        is_fingerprint_drift=False,
        distinct_cards_count=4,
        distinct_devices_count=1,
        similarity_to_baseline=1.0,
        severity="CRITICAL",
        reasons=(
            "Device exhibits automated webdriver or emulator signatures.",
            f"Device associated with card {pan_card}.",
        ),
    )

    event = build_device_anomaly_audit_event(
        report=rep,
        transaction_id="tx_dev_anomaly_1",
        trigger_action="DENY",
        model_version="v2",
        metadata={"operator": "fraud_ops", "api_key": "secret_key_123"},
    )

    assert event.event_type == "fraud.device.anomaly_detected"
    assert event.model_version == "v2"
    assert event.payload["device_id"] == "device_attack_cluster_99"
    assert event.payload["transaction_id"] == "tx_dev_anomaly_1"
    assert event.payload["is_emulator"] is True
    assert event.payload["is_card_cycling"] is True
    assert event.payload["distinct_cards_count"] == 4
    assert event.payload["severity"] == "CRITICAL"
    assert event.payload["trigger_action"] == "DENY"

    # Redaction checks: PAN in card_id and reasons must be masked
    assert event.payload["card_id"] == "[REDACTED_PAN]"
    assert "[REDACTED_PAN]" in event.payload["reasons"][1]
    assert pan_card not in event.payload["reasons"][1]

    # Redaction of secret api_key in metadata
    d = event.to_dict(redact=True)
    assert d["payload"]["metadata"]["api_key"] == "[REDACTED]"
    assert d["payload"]["metadata"]["operator"] == "fraud_ops"

    # Mapping report test
    mapping_rep = {
        "device_id": "dev_map",
        "card_id": "plain_user_1",
        "is_emulator": False,
        "is_card_cycling": True,
        "distinct_cards_count": 5,
        "severity": "HIGH",
    }
    mapping_event = build_device_anomaly_audit_event(
        report=mapping_rep,
        transaction_id="tx_map_dev",
    )
    assert mapping_event.payload["device_id"] == "dev_map"
    assert mapping_event.payload["card_id"] == "plain_user_1"
    assert mapping_event.payload["is_card_cycling"] is True
    assert mapping_event.payload["distinct_cards_count"] == 5
    assert mapping_event.payload["severity"] == "HIGH"


def test_build_merchant_anomaly_audit_event() -> None:
    pan_card = "4532015112830366"  # Luhn-valid PAN
    rep = MerchantAnomalyReport(
        merchant_id="m_collusion_99",
        mcc="7995",
        is_high_chargeback=True,
        is_bust_out_surge=True,
        is_card_testing=False,
        is_ticket_outlier=True,
        smoothed_fraud_rate=0.085,
        amount_zscore=4.25,
        micro_tx_ratio=0.10,
        distinct_cards_count=6,
        transaction_count=12,
        severity="CRITICAL",
        reasons=(
            "high_chargeback_rate:0.0850",
            f"bust_out_surge on card {pan_card}",
        ),
    )

    event = build_merchant_anomaly_audit_event(
        report=rep,
        card_id=pan_card,
        transaction_id="tx_merch_anomaly_1",
        trigger_action="DENY",
        model_version="v2",
        metadata={"operator": "fraud_ops", "api_key": "secret_key_123"},
    )

    assert event.event_type == "fraud.merchant.anomaly_detected"
    assert event.model_version == "v2"
    assert event.payload["merchant_id"] == "m_collusion_99"
    assert event.payload["mcc"] == "7995"
    assert event.payload["transaction_id"] == "tx_merch_anomaly_1"
    assert event.payload["is_high_chargeback"] is True
    assert event.payload["is_bust_out_surge"] is True
    assert event.payload["distinct_cards_count"] == 6
    assert event.payload["severity"] == "CRITICAL"
    assert event.payload["trigger_action"] == "DENY"

    # Redaction checks: PAN in card_id and reasons must be masked
    assert event.payload["card_id"] == "[REDACTED_PAN]"
    assert "[REDACTED_PAN]" in event.payload["reasons"][1]
    assert pan_card not in event.payload["reasons"][1]

    # Redaction of secret api_key in metadata
    d = event.to_dict(redact=True)
    assert d["payload"]["metadata"]["api_key"] == "[REDACTED]"
    assert d["payload"]["metadata"]["operator"] == "fraud_ops"

    # Mapping report test
    mapping_rep = {
        "merchant_id": "m_map",
        "mcc": "5817",
        "is_high_chargeback": False,
        "is_card_testing": True,
        "distinct_cards_count": 7,
        "transaction_count": 8,
        "severity": "HIGH",
    }
    mapping_event = build_merchant_anomaly_audit_event(
        report=mapping_rep,
        card_id="plain_user_1",
        transaction_id="tx_map_merch",
    )
    assert mapping_event.payload["merchant_id"] == "m_map"
    assert mapping_event.payload["mcc"] == "5817"
    assert mapping_event.payload["card_id"] == "plain_user_1"
    assert mapping_event.payload["is_card_testing"] is True
    assert mapping_event.payload["distinct_cards_count"] == 7
    assert mapping_event.payload["severity"] == "HIGH"


def test_build_tenant_audit_event() -> None:
    """Validate construction and sanitization of tenant audit event."""
    pan_id = "4532015112830366"  # Valid Luhn PAN inside transaction id
    event = build_tenant_audit_event(
        tenant_id="tenant_fintech_enterprise",
        tier="enterprise",
        status="active",
        transaction_id=pan_id,
        action="DENY",
        matched_rule_id="RULE_TENANT_HIGH_RISK",
        matched_rule_name="Tenant High Risk Block",
        is_suspended=False,
        thresholds={"decline": 0.90, "challenge": 0.65, "review": 0.40},
        metadata={"operator": "fraud_lead", "api_key": "secret_key_123"},
    )

    assert event.event_type == "fraud.tenant.policy_evaluated"
    assert event.payload["tenant_id"] == "tenant_fintech_enterprise"
    assert event.payload["tier"] == "enterprise"
    assert event.payload["status"] == "active"
    assert event.payload["action"] == "DENY"
    assert event.payload["matched_rule_id"] == "RULE_TENANT_HIGH_RISK"
    assert event.payload["matched_rule_name"] == "Tenant High Risk Block"
    assert event.payload["is_suspended"] is False
    assert event.payload["thresholds"]["decline"] == 0.90

    # Ensure PAN in transaction_id was redacted
    assert event.payload["transaction_id"] == "[REDACTED_PAN]"

    # Redaction test through to_dict
    d = event.to_dict(redact=True)
    assert d["payload"]["metadata"]["api_key"] == "[REDACTED]"
    assert d["payload"]["metadata"]["operator"] == "fraud_lead"
