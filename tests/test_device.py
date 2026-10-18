"""Unit and integration tests for adaptive device fingerprinting and clustering engine."""

from __future__ import annotations

import pytest

from fraud_detection.device import (
    DeviceFingerprint,
    DeviceFingerprintError,
    DeviceHasher,
    DeviceTracker,
    DeviceTrackerConfig,
    ScreenGeometry,
    calculate_fingerprint_similarity,
    calculate_hamming_distance,
    calculate_jaccard_similarity,
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


def test_device_hasher_canonical_hash() -> None:
    sg = ScreenGeometry(width=1920, height=1080)
    fp1 = DeviceFingerprint(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        os_platform="Windows",
        browser_family="Chrome",
        screen=sg,
        hardware_concurrency=8,
        device_memory_gb=16.0,
        webgl_vendor="Google Inc. (NVIDIA)",
        webgl_renderer="ANGLE (NVIDIA GeForce RTX 3080)",
        canvas_hash="hash_123",
        audio_hash="audio_456",
    )
    fp2 = DeviceFingerprint(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        os_platform="windows",  # casing difference should normalize
        browser_family="chrome",
        screen=sg,
        hardware_concurrency=8,
        device_memory_gb=16.0,
        webgl_vendor="Google Inc. (NVIDIA)",
        webgl_renderer="ANGLE (NVIDIA GeForce RTX 3080)",
        canvas_hash="hash_123",
        audio_hash="audio_456",
    )
    hash1 = DeviceHasher.canonical_hash(fp1)
    hash2 = DeviceHasher.canonical_hash(fp2)
    assert len(hash1) == 64
    assert hash1 == hash2

    # Different canvas hash should yield different canonical hash
    fp_diff = DeviceFingerprint(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        os_platform="Windows",
        browser_family="Chrome",
        screen=sg,
        canvas_hash="different_canvas",
    )
    assert DeviceHasher.canonical_hash(fp_diff) != hash1


def test_device_hasher_simhash() -> None:
    sg = ScreenGeometry(width=1920, height=1080)
    fp1 = DeviceFingerprint(
        os_platform="Windows",
        browser_family="Chrome",
        screen=sg,
        hardware_concurrency=8,
        device_memory_gb=16.0,
    )
    # fp2 has minor change: hardware_concurrency=6
    fp2 = DeviceFingerprint(
        os_platform="Windows",
        browser_family="Chrome",
        screen=sg,
        hardware_concurrency=6,
        device_memory_gb=16.0,
    )
    # fp3 is completely different: macOS Safari
    fp3 = DeviceFingerprint(
        os_platform="macOS",
        browser_family="Safari",
        screen=ScreenGeometry(width=1440, height=900),
        hardware_concurrency=4,
        device_memory_gb=8.0,
        canvas_hash="diff_canv",
    )

    sim1 = DeviceHasher.simhash(fp1, hash_bits=64)
    sim2 = DeviceHasher.simhash(fp2, hash_bits=64)
    sim3 = DeviceHasher.simhash(fp3, hash_bits=64)

    assert len(sim1) == 16
    assert len(sim2) == 16
    dist_1_2 = calculate_hamming_distance(sim1, sim2)
    dist_1_3 = calculate_hamming_distance(sim1, sim3)
    assert dist_1_2 <= dist_1_3

    # 128-bit simhash
    sim_128 = DeviceHasher.simhash(fp1, hash_bits=128)
    assert len(sim_128) == 32

    with pytest.raises(DeviceFingerprintError, match="hash_bits must be either 64 or 128"):
        DeviceHasher.simhash(fp1, hash_bits=32)


def test_calculate_jaccard_similarity() -> None:
    assert calculate_jaccard_similarity([], []) == 1.0
    assert calculate_jaccard_similarity(["a", "b"], ["a", "b"]) == 1.0
    assert calculate_jaccard_similarity(["a", "b"], ["b", "c"]) == 1.0 / 3.0
    assert calculate_jaccard_similarity(["a"], ["b"]) == 0.0


def test_calculate_hamming_distance_errors() -> None:
    with pytest.raises(DeviceFingerprintError, match="hex1 and hex2 must be strings"):
        calculate_hamming_distance(123, "abc")  # type: ignore[arg-type]
    with pytest.raises(DeviceFingerprintError, match="equal length"):
        calculate_hamming_distance("a", "abc")
    with pytest.raises(DeviceFingerprintError, match="Invalid hexadecimal string"):
        calculate_hamming_distance("zz", "aa")


def test_calculate_fingerprint_similarity() -> None:
    sg1 = ScreenGeometry(width=1920, height=1080, pixel_ratio=1.0)
    fp1 = DeviceFingerprint(
        os_platform="Windows",
        browser_family="Chrome",
        screen=sg1,
        hardware_concurrency=8,
        device_memory_gb=16.0,
        webgl_vendor="NVIDIA",
        webgl_renderer="RTX 3080",
        canvas_hash="canvas_1",
        audio_hash="audio_1",
        language="en-US",
        timezone_offset_minutes=-300,
    )
    # Identical should give 1.0
    assert calculate_fingerprint_similarity(fp1, fp1) == 1.0

    # Slight variation in timezone
    fp_tz = DeviceFingerprint(
        os_platform="Windows",
        browser_family="Chrome",
        screen=sg1,
        hardware_concurrency=8,
        device_memory_gb=16.0,
        webgl_vendor="NVIDIA",
        webgl_renderer="RTX 3080",
        canvas_hash="canvas_1",
        audio_hash="audio_1",
        language="en-US",
        timezone_offset_minutes=-360,
    )
    sim_tz = calculate_fingerprint_similarity(fp1, fp_tz)
    assert 0.95 <= sim_tz < 1.0

    # Different device (macOS Safari on MacBook)
    fp_mac = DeviceFingerprint(
        os_platform="macOS",
        browser_family="Safari",
        screen=ScreenGeometry(width=1440, height=900, pixel_ratio=2.0),
        hardware_concurrency=8,
        device_memory_gb=16.0,
        webgl_vendor="Apple",
        webgl_renderer="Apple M1",
        canvas_hash="canvas_mac",
        audio_hash="audio_mac",
        language="fr-FR",
        timezone_offset_minutes=60,
    )
    sim_mac = calculate_fingerprint_similarity(fp1, fp_mac)
    assert sim_mac < 0.50


def test_device_tracker_config_valid_and_invalid() -> None:
    cfg = DeviceTrackerConfig(
        window_seconds=3600.0,
        max_devices=1000,
        card_cycling_threshold=5,
        device_cycling_threshold=4,
        min_similarity_threshold=0.75,
    )
    assert cfg.window_seconds == 3600.0
    assert cfg.max_devices == 1000
    assert cfg.card_cycling_threshold == 5
    assert cfg.device_cycling_threshold == 4
    assert cfg.min_similarity_threshold == 0.75

    with pytest.raises(DeviceFingerprintError, match="window_seconds must be a positive"):
        DeviceTrackerConfig(window_seconds=0.0)
    with pytest.raises(DeviceFingerprintError, match="max_devices must be a positive"):
        DeviceTrackerConfig(max_devices=-1)
    with pytest.raises(DeviceFingerprintError, match="card_cycling_threshold must be a positive"):
        DeviceTrackerConfig(card_cycling_threshold=0)
    with pytest.raises(DeviceFingerprintError, match="device_cycling_threshold must be a positive"):
        DeviceTrackerConfig(device_cycling_threshold=0)
    with pytest.raises(DeviceFingerprintError, match="min_similarity_threshold must be a finite"):
        DeviceTrackerConfig(min_similarity_threshold=1.5)


def test_device_tracker_card_cycling_detection() -> None:
    cfg = DeviceTrackerConfig(card_cycling_threshold=3, window_seconds=3600.0)
    tracker = DeviceTracker(config=cfg)

    # 1st card on device-A
    rep1 = tracker.record_event(device_id="dev-A", card_id="card-1", timestamp=1000.0)
    assert not rep1.has_anomaly
    assert rep1.distinct_cards_count == 1
    assert rep1.severity == "LOW"

    # 2nd card on device-A
    rep2 = tracker.record_event(device_id="dev-A", card_id="card-2", timestamp=1010.0)
    assert not rep2.has_anomaly
    assert rep2.distinct_cards_count == 2

    # 3rd card on device-A -> trips card cycling threshold
    rep3 = tracker.record_event(device_id="dev-A", card_id="card-3", timestamp=1020.0)
    assert rep3.has_anomaly
    assert rep3.is_card_cycling
    assert rep3.distinct_cards_count == 3
    assert rep3.severity == "HIGH"
    assert "Device associated with 3 cards" in rep3.reasons[0]


def test_device_tracker_device_cycling_detection() -> None:
    cfg = DeviceTrackerConfig(device_cycling_threshold=3, window_seconds=3600.0)
    tracker = DeviceTracker(config=cfg)

    # card-1 on dev-A
    tracker.record_event(device_id="dev-A", card_id="card-1", timestamp=1000.0)
    # card-1 on dev-B
    tracker.record_event(device_id="dev-B", card_id="card-1", timestamp=1010.0)
    # card-1 on dev-C -> trips device cycling threshold
    rep3 = tracker.record_event(device_id="dev-C", card_id="card-1", timestamp=1020.0)
    assert rep3.has_anomaly
    assert rep3.is_device_cycling
    assert rep3.distinct_devices_count == 3
    assert rep3.severity == "HIGH"


def test_device_tracker_webdriver_emulator_and_critical_severity() -> None:
    cfg = DeviceTrackerConfig(card_cycling_threshold=2)
    tracker = DeviceTracker(config=cfg)

    fp_bot = DeviceFingerprint(has_webdriver=True)
    # 1st card with bot -> emulator flagged, severity MEDIUM
    rep1 = tracker.record_event(
        device_id="dev-bot", card_id="card-1", timestamp=1000.0, fingerprint=fp_bot
    )
    assert rep1.has_anomaly
    assert rep1.is_emulator
    assert not rep1.is_card_cycling
    assert rep1.severity == "MEDIUM"

    # 2nd card with bot -> card cycling AND emulator -> CRITICAL
    rep2 = tracker.record_event(
        device_id="dev-bot", card_id="card-2", timestamp=1010.0, fingerprint=fp_bot
    )
    assert rep2.has_anomaly
    assert rep2.is_card_cycling
    assert rep2.is_emulator
    assert rep2.severity == "CRITICAL"


def test_device_tracker_assess_event_is_pure_read() -> None:
    tracker = DeviceTracker()
    rep = tracker.assess_event(device_id="dev-read", card_id="card-1", timestamp=1000.0)
    assert not rep.has_anomaly
    # State should remain empty
    stats = tracker.get_stats()
    assert stats["tracked_devices_count"] == 0
    assert stats["total_evaluations"] == 0
    assert tracker.get_profile("dev-read") is None


def test_device_tracker_lru_capacity_and_reset() -> None:
    cfg = DeviceTrackerConfig(max_devices=2)
    tracker = DeviceTracker(config=cfg)

    tracker.record_event(device_id="dev-1", card_id="card-1", timestamp=100.0)
    tracker.record_event(device_id="dev-2", card_id="card-2", timestamp=101.0)
    assert tracker.get_stats()["tracked_devices_count"] == 2

    # dev-3 causes dev-1 to be evicted
    tracker.record_event(device_id="dev-3", card_id="card-3", timestamp=102.0)
    assert tracker.get_stats()["tracked_devices_count"] == 2
    assert tracker.get_profile("dev-1") is None
    assert tracker.get_profile("dev-2") is not None
    assert tracker.get_profile("dev-3") is not None

    tracker.reset()
    assert tracker.get_stats()["tracked_devices_count"] == 0
    assert tracker.get_stats()["total_evaluations"] == 0
