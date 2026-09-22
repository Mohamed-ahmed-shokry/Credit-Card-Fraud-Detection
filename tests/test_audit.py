"""Tests for structured audit event export and redaction."""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any

from fraud_detection import __version__
from fraud_detection.audit import (
    AuditEvent,
    AuditReplayReport,
    DiscrepancyDetail,
    JsonlAuditSink,
    NullAuditSink,
    _is_luhn_valid,
    _mask_pans_in_string,
    build_promotion_audit_event,
    build_scoring_audit_event,
    redact_data,
    replay_audit_log,
)


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
