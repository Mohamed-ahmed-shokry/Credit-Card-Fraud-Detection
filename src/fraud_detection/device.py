"""Real-time adaptive device and browser fingerprinting with anomaly clustering."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class DeviceFingerprintError(ValueError):
    """Raised when device fingerprint attributes, parsing, or tracking operations are invalid."""


@dataclass(frozen=True)
class ScreenGeometry:
    """Normalized client screen dimensions, color depth, and pixel ratio."""

    width: int
    height: int
    color_depth: int = 24
    pixel_ratio: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.width, int) or isinstance(self.width, bool) or self.width <= 0:
            raise DeviceFingerprintError("width must be a positive integer.")
        if not isinstance(self.height, int) or isinstance(self.height, bool) or self.height <= 0:
            raise DeviceFingerprintError("height must be a positive integer.")
        if (
            not isinstance(self.color_depth, int)
            or isinstance(self.color_depth, bool)
            or self.color_depth <= 0
        ):
            raise DeviceFingerprintError("color_depth must be a positive integer.")
        if (
            not isinstance(self.pixel_ratio, (int, float))
            or math.isnan(self.pixel_ratio)
            or math.isinf(self.pixel_ratio)
            or self.pixel_ratio <= 0.0
        ):
            raise DeviceFingerprintError("pixel_ratio must be a positive finite float.")

    def to_dict(self) -> dict[str, Any]:
        """Convert screen geometry to dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "color_depth": self.color_depth,
            "pixel_ratio": float(self.pixel_ratio),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScreenGeometry:
        """Create ScreenGeometry instance from mapping."""
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            color_depth=int(data.get("color_depth", 24)),
            pixel_ratio=float(data.get("pixel_ratio", 1.0)),
        )


@dataclass(frozen=True)
class DeviceFingerprint:
    """Canonical attributes representing client browser and hardware runtime."""

    device_id: str | None = None
    user_agent: str = ""
    os_platform: str = ""
    browser_family: str = ""
    screen: ScreenGeometry | None = None
    language: str = "en"
    timezone_offset_minutes: int = 0
    hardware_concurrency: int = 4
    device_memory_gb: float = 8.0
    webgl_vendor: str = ""
    webgl_renderer: str = ""
    canvas_hash: str = ""
    audio_hash: str = ""
    is_touch_capable: bool = False
    has_webdriver: bool = False
    plugins_count: int = 0

    def __post_init__(self) -> None:
        if self.device_id is not None and (
            not isinstance(self.device_id, str) or not self.device_id.strip()
        ):
            raise DeviceFingerprintError("device_id must be a non-empty string or None.")
        if not isinstance(self.user_agent, str):
            raise DeviceFingerprintError("user_agent must be a string.")
        if not isinstance(self.os_platform, str):
            raise DeviceFingerprintError("os_platform must be a string.")
        if not isinstance(self.browser_family, str):
            raise DeviceFingerprintError("browser_family must be a string.")
        if self.screen is not None and not isinstance(self.screen, ScreenGeometry):
            raise DeviceFingerprintError("screen must be a ScreenGeometry instance or None.")
        if not isinstance(self.language, str):
            raise DeviceFingerprintError("language must be a string.")
        if (
            not isinstance(self.timezone_offset_minutes, int)
            or isinstance(self.timezone_offset_minutes, bool)
            or not (-720 <= self.timezone_offset_minutes <= 840)
        ):
            raise DeviceFingerprintError(
                "timezone_offset_minutes must be an integer between -720 and 840."
            )
        if (
            not isinstance(self.hardware_concurrency, int)
            or isinstance(self.hardware_concurrency, bool)
            or self.hardware_concurrency < 0
        ):
            raise DeviceFingerprintError("hardware_concurrency must be a non-negative integer.")
        if (
            not isinstance(self.device_memory_gb, (int, float))
            or math.isnan(self.device_memory_gb)
            or math.isinf(self.device_memory_gb)
            or self.device_memory_gb < 0.0
        ):
            raise DeviceFingerprintError("device_memory_gb must be a non-negative finite float.")
        if not isinstance(self.webgl_vendor, str):
            raise DeviceFingerprintError("webgl_vendor must be a string.")
        if not isinstance(self.webgl_renderer, str):
            raise DeviceFingerprintError("webgl_renderer must be a string.")
        if not isinstance(self.canvas_hash, str):
            raise DeviceFingerprintError("canvas_hash must be a string.")
        if not isinstance(self.audio_hash, str):
            raise DeviceFingerprintError("audio_hash must be a string.")
        if not isinstance(self.is_touch_capable, bool):
            raise DeviceFingerprintError("is_touch_capable must be a boolean.")
        if not isinstance(self.has_webdriver, bool):
            raise DeviceFingerprintError("has_webdriver must be a boolean.")
        if (
            not isinstance(self.plugins_count, int)
            or isinstance(self.plugins_count, bool)
            or self.plugins_count < 0
        ):
            raise DeviceFingerprintError("plugins_count must be a non-negative integer.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize fingerprint to dictionary."""
        return {
            "device_id": self.device_id,
            "user_agent": self.user_agent,
            "os_platform": self.os_platform,
            "browser_family": self.browser_family,
            "screen": self.screen.to_dict() if self.screen is not None else None,
            "language": self.language,
            "timezone_offset_minutes": self.timezone_offset_minutes,
            "hardware_concurrency": self.hardware_concurrency,
            "device_memory_gb": float(self.device_memory_gb),
            "webgl_vendor": self.webgl_vendor,
            "webgl_renderer": self.webgl_renderer,
            "canvas_hash": self.canvas_hash,
            "audio_hash": self.audio_hash,
            "is_touch_capable": self.is_touch_capable,
            "has_webdriver": self.has_webdriver,
            "plugins_count": self.plugins_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DeviceFingerprint:
        """Construct DeviceFingerprint from mapping."""
        screen_data = data.get("screen")
        screen_obj: ScreenGeometry | None = None
        if isinstance(screen_data, Mapping):
            screen_obj = ScreenGeometry.from_dict(screen_data)
        elif isinstance(screen_data, ScreenGeometry):
            screen_obj = screen_data

        return cls(
            device_id=data.get("device_id"),
            user_agent=str(data.get("user_agent", "")),
            os_platform=str(data.get("os_platform", "")),
            browser_family=str(data.get("browser_family", "")),
            screen=screen_obj,
            language=str(data.get("language", "en")),
            timezone_offset_minutes=int(data.get("timezone_offset_minutes", 0)),
            hardware_concurrency=int(data.get("hardware_concurrency", 4)),
            device_memory_gb=float(data.get("device_memory_gb", 8.0)),
            webgl_vendor=str(data.get("webgl_vendor", "")),
            webgl_renderer=str(data.get("webgl_renderer", "")),
            canvas_hash=str(data.get("canvas_hash", "")),
            audio_hash=str(data.get("audio_hash", "")),
            is_touch_capable=bool(data.get("is_touch_capable", False)),
            has_webdriver=bool(data.get("has_webdriver", False)),
            plugins_count=int(data.get("plugins_count", 0)),
        )
