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
    build_promotion_audit_event,
    build_scoring_audit_event,
    build_shadow_scoring_audit_event,
    redact_data,
    replay_audit_log,
)
from fraud_detection.evaluation import TieredThresholds


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


