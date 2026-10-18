"""Unit and integration tests for adaptive device fingerprinting and clustering engine."""

from __future__ import annotations

import pytest

from fraud_detection.device import (
    DeviceFingerprint,
    DeviceFingerprintError,
    ScreenGeometry,
)


def test_screen_geometry_valid() -> None:
    sg = ScreenGeometry(width=1920, height=1080, color_depth=24, pixel_ratio=1.0)
    assert sg.width == 1920
    assert sg.height == 1080
    assert sg.color_depth == 24
    assert sg.pixel_ratio == 1.0

    d = sg.to_dict()
    assert d == {"width": 1920, "height": 1080, "color_depth": 24, "pixel_ratio": 1.0}
    rebuilt = ScreenGeometry.from_dict(d)
    assert rebuilt == sg


@pytest.mark.parametrize(
    ("width", "height", "color_depth", "pixel_ratio", "err_msg"),
    [
        (0, 1080, 24, 1.0, "width must be a positive integer"),
        (-100, 1080, 24, 1.0, "width must be a positive integer"),
        (1920, 0, 24, 1.0, "height must be a positive integer"),
        (1920, 1080, 0, 1.0, "color_depth must be a positive integer"),
        (1920, 1080, 24, 0.0, "pixel_ratio must be a positive finite float"),
        (1920, 1080, 24, -1.0, "pixel_ratio must be a positive finite float"),
        (1920, 1080, 24, float("nan"), "pixel_ratio must be a positive finite float"),
    ],
)
def test_screen_geometry_invalid(
    width: int, height: int, color_depth: int, pixel_ratio: float, err_msg: str
) -> None:
    with pytest.raises(DeviceFingerprintError, match=err_msg):
        ScreenGeometry(
            width=width,
            height=height,
            color_depth=color_depth,
            pixel_ratio=pixel_ratio,
        )


def test_device_fingerprint_valid_and_serde() -> None:
    sg = ScreenGeometry(width=2560, height=1440, color_depth=30, pixel_ratio=2.0)
    fp = DeviceFingerprint(
        device_id="dev-123",
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        os_platform="macOS",
        browser_family="Chrome",
        screen=sg,
        language="en-US",
        timezone_offset_minutes=-300,
        hardware_concurrency=8,
        device_memory_gb=16.0,
        webgl_vendor="Apple",
        webgl_renderer="Apple M2 Pro",
        canvas_hash="canvas_hash_abc",
        audio_hash="audio_hash_xyz",
        is_touch_capable=False,
        has_webdriver=False,
        plugins_count=3,
    )
    assert fp.device_id == "dev-123"
    assert fp.screen == sg
    assert fp.hardware_concurrency == 8

    d = fp.to_dict()
    assert d["device_id"] == "dev-123"
    assert d["screen"] == sg.to_dict()

    rebuilt = DeviceFingerprint.from_dict(d)
    assert rebuilt == fp


@pytest.mark.parametrize(
    ("kwargs", "err_msg"),
    [
        ({"device_id": ""}, "device_id must be a non-empty string or None"),
        ({"device_id": "   "}, "device_id must be a non-empty string or None"),
        ({"screen": "not-a-screen"}, "screen must be a ScreenGeometry instance or None"),
        ({"timezone_offset_minutes": -1000}, "timezone_offset_minutes must be an integer between"),
        ({"timezone_offset_minutes": 1000}, "timezone_offset_minutes must be an integer between"),
        ({"hardware_concurrency": -1}, "hardware_concurrency must be a non-negative integer"),
        ({"device_memory_gb": -2.0}, "device_memory_gb must be a non-negative finite float"),
        ({"plugins_count": -5}, "plugins_count must be a non-negative integer"),
    ],
)
def test_device_fingerprint_invalid(kwargs: dict[str, object], err_msg: str) -> None:
    with pytest.raises(DeviceFingerprintError, match=err_msg):
        DeviceFingerprint(**kwargs)  # type: ignore[arg-type]
