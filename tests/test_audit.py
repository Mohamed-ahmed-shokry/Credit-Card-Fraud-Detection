"""Tests for structured audit event export and redaction."""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

from fraud_detection import __version__
from fraud_detection.audit import (
    AuditEvent,
    JsonlAuditSink,
    NullAuditSink,
    _is_luhn_valid,
    _mask_pans_in_string,
    build_promotion_audit_event,
    build_scoring_audit_event,
    redact_data,
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

