from __future__ import annotations

import json
import threading
from typing import Any
from unittest.mock import patch

import pytest

from fraud_detection.telemetry import (
    OTLPHttpTraceExporter,
    TraceContext,
    TraceContextError,
    TraceSpan,
    parse_traceparent,
)


def test_trace_context_parses_and_creates_child() -> None:
    incoming = TraceContext.from_traceparent(
        "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    )
    child = incoming.child()

    assert child.trace_id == incoming.trace_id
    assert child.span_id != incoming.span_id
    assert parse_traceparent(child.traceparent) == child


@pytest.mark.parametrize(
    "value",
    [
        "00-00000000000000000000000000000000-0123456789abcdef-01",
        "00-0123456789abcdef0123456789abcdef-0000000000000000-01",
        "01-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
        "malformed",
    ],
)
def test_trace_context_rejects_invalid_values(value: str) -> None:
    with pytest.raises(TraceContextError):
        TraceContext.from_traceparent(value)
    assert parse_traceparent(value) is None


def test_otlp_payload_contains_http_span_and_service_attributes() -> None:
    context = TraceContext.new()
    span = TraceSpan(
        name="HTTP /v1/score",
        context=context,
        parent_span_id="0123456789abcdef",
        start_time_unix_nano=10,
        end_time_unix_nano=20,
        attributes={"http.request.method": "POST", "http.response.status_code": 200},
        status_code=200,
    )
    exporter = OTLPHttpTraceExporter("http://collector:4318", service_name="fraud-api")
    captured: dict[str, Any] = {}

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_urlopen(request: Any, timeout: float) -> _Response:
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data)
        return _Response()

    with patch("fraud_detection.telemetry.urllib.request.urlopen", fake_urlopen):
        exporter._send(span)  # noqa: SLF001

    assert captured["url"] == "http://collector:4318/v1/traces"
    assert captured["payload"]["resourceSpans"][0]["resource"]["attributes"][0] == {
        "key": "service.name",
        "value": {"stringValue": "fraud-api"},
    }
    exported_span = captured["payload"]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert exported_span["traceId"] == context.trace_id
    assert exported_span["parentSpanId"] == "0123456789abcdef"
    assert exported_span["status"]["code"] == 1


def test_otlp_export_is_non_blocking_and_isolates_failures() -> None:
    exporter = OTLPHttpTraceExporter("http://collector:4318")
    completed = threading.Event()

    def fake_send(_span: TraceSpan) -> None:
        completed.set()

    exporter._send = fake_send  # noqa: SLF001
    exporter.export(
        TraceSpan(
            name="HTTP /health",
            context=TraceContext.new(),
            parent_span_id=None,
            start_time_unix_nano=1,
            end_time_unix_nano=2,
            attributes={},
            status_code=200,
        )
    )

    assert completed.wait(timeout=1.0)


def test_otlp_endpoint_requires_http_url() -> None:
    with pytest.raises(ValueError, match="absolute HTTP"):
        OTLPHttpTraceExporter("collector:4318")
