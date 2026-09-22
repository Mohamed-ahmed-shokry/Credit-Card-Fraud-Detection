"""Optional dependency-free distributed trace propagation and OTLP export."""

from __future__ import annotations

import json
import logging
import secrets
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

TRACEPARENT_HEADER = "traceparent"
TRACESTATE_HEADER = "tracestate"
OTLP_ENDPOINT_ENVIRONMENT_VARIABLE = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
OTLP_SERVICE_NAME_ENVIRONMENT_VARIABLE = "OTEL_SERVICE_NAME"
DEFAULT_SERVICE_NAME = "fraud-detection-api"
DEFAULT_OTLP_TIMEOUT_SECONDS = 0.5


class TraceContextError(ValueError):
    """Raised when a W3C trace context is invalid."""


@dataclass(frozen=True)
class TraceContext:
    """A valid W3C trace context suitable for propagating to a child span."""

    trace_id: str
    span_id: str
    trace_flags: str = "01"

    def __post_init__(self) -> None:
        if len(self.trace_id) != 32 or not _is_lower_hex(self.trace_id):
            raise TraceContextError("trace_id must be 32 lowercase hexadecimal characters.")
        if self.trace_id == "0" * 32:
            raise TraceContextError("trace_id must not be all zeroes.")
        if len(self.span_id) != 16 or not _is_lower_hex(self.span_id):
            raise TraceContextError("span_id must be 16 lowercase hexadecimal characters.")
        if self.span_id == "0" * 16:
            raise TraceContextError("span_id must not be all zeroes.")
        if len(self.trace_flags) != 2 or not _is_lower_hex(self.trace_flags):
            raise TraceContextError("trace_flags must be two lowercase hexadecimal characters.")

    @classmethod
    def new(cls) -> TraceContext:
        """Create a fresh sampled trace context."""
        return cls(trace_id=secrets.token_hex(16), span_id=secrets.token_hex(8))

    @classmethod
    def from_traceparent(cls, value: str) -> TraceContext:
        """Parse a W3C traceparent header, rejecting malformed or unsampled versions."""
        parts = value.split("-")
        if len(parts) != 4:
            raise TraceContextError("traceparent must contain four hyphen-separated fields.")
        version, trace_id, span_id, trace_flags = parts
        if version != "00":
            raise TraceContextError("Only traceparent version 00 is supported.")
        return cls(trace_id=trace_id, span_id=span_id, trace_flags=trace_flags)

    def child(self) -> TraceContext:
        """Create a new span context within the same trace."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=secrets.token_hex(8),
            trace_flags=self.trace_flags,
        )

    @property
    def traceparent(self) -> str:
        """Return the context in W3C traceparent wire format."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags}"


@dataclass(frozen=True)
class TraceSpan:
    """Completed HTTP server span represented in OTLP-compatible fields."""

    name: str
    context: TraceContext
    parent_span_id: str | None
    start_time_unix_nano: int
    end_time_unix_nano: int
    attributes: Mapping[str, str | int | float | bool]
    status_code: int

    def to_otlp(self, service_name: str) -> dict[str, Any]:
        """Return the OTLP/HTTP JSON representation for one span."""
        attributes = [
            {"key": key, "value": _otlp_value(value)} for key, value in self.attributes.items()
        ]
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": service_name}}
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "fraud_detection.telemetry"},
                            "spans": [
                                {
                                    "traceId": self.context.trace_id,
                                    "spanId": self.context.span_id,
                                    "parentSpanId": self.parent_span_id or "",
                                    "name": self.name,
                                    "kind": 2,
                                    "startTimeUnixNano": str(self.start_time_unix_nano),
                                    "endTimeUnixNano": str(self.end_time_unix_nano),
                                    "attributes": attributes,
                                    "status": {
                                        "code": 1 if self.status_code < 500 else 2,
                                    },
                                }
                            ],
                        }
                    ],
                }
            ]
        }


class TraceExporter(Protocol):
    """Exporter interface used by the API request middleware."""

    def export(self, span: TraceSpan) -> None:
        """Export one completed span without affecting the request result."""


class NullTraceExporter:
    """Default exporter that intentionally performs no work."""

    def export(self, span: TraceSpan) -> None:
        del span


class OTLPHttpTraceExporter:
    """Best-effort non-blocking OTLP/HTTP JSON exporter."""

    def __init__(
        self,
        endpoint: str,
        *,
        service_name: str = DEFAULT_SERVICE_NAME,
        timeout_seconds: float = DEFAULT_OTLP_TIMEOUT_SECONDS,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("OTLP endpoint must be an absolute HTTP or HTTPS URL.")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("OTLP endpoint must not contain embedded credentials.")
        if not service_name.strip():
            raise ValueError("service_name must not be empty.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        self.endpoint = endpoint.rstrip("/")
        if not self.endpoint.endswith("/v1/traces"):
            self.endpoint += "/v1/traces"
        self.service_name = service_name
        self.timeout_seconds = timeout_seconds

    def export(self, span: TraceSpan) -> None:
        """Queue a span on a daemon thread so collector failures cannot block scoring."""
        thread = threading.Thread(target=self._send, args=(span,), daemon=True)
        thread.start()

    def _send(self, span: TraceSpan) -> None:
        payload = json.dumps(span.to_otlp(self.service_name)).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds):  # noqa: S310
                return
        except (OSError, urllib.error.URLError, ValueError) as exc:
            logger.warning("OTLP trace export failed: %s", exc)


def parse_traceparent(value: str | None) -> TraceContext | None:
    """Parse an incoming traceparent, returning None for absent or invalid input."""
    if value is None:
        return None
    try:
        return TraceContext.from_traceparent(value)
    except TraceContextError:
        return None


def _is_lower_hex(value: str) -> bool:
    return all(character in "0123456789abcdef" for character in value)


def _otlp_value(value: str | int | float | bool) -> dict[str, str | int | float | bool]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": value}
