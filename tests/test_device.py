"""Unit and integration tests for adaptive device fingerprinting and clustering engine."""

from __future__ import annotations

import pytest

from fraud_detection.device import (
    DEVICE_FEATURE_NAMES,
    DeviceFeatureExtractor,
    DeviceFingerprint,
    DeviceFingerprintError,
    DeviceHasher,
    DeviceTracker,
    DeviceTrackerConfig,
    ScreenGeometry,
    calculate_fingerprint_entropy,
    calculate_fingerprint_similarity,
    calculate_hamming_distance,
    calculate_jaccard_similarity,
    detect_emulator_signatures,
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


def test_detect_emulator_signatures() -> None:
    # Clean legitimate device
    fp_clean = DeviceFingerprint(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        os_platform="Windows",
        browser_family="Chrome",
        screen=ScreenGeometry(width=1920, height=1080),
        hardware_concurrency=8,
        device_memory_gb=16.0,
        webgl_vendor="NVIDIA",
        webgl_renderer="GeForce",
        canvas_hash="canv_1",
    )
    is_emu, reasons = detect_emulator_signatures(fp_clean)
    assert not is_emu
    assert reasons == ()

    # Headless Chrome bot
    fp_bot = DeviceFingerprint(
        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 HeadlessChrome/119.0.0.0",
        os_platform="Linux",
        has_webdriver=True,
    )
    is_emu_bot, bot_reasons = detect_emulator_signatures(fp_bot)
    assert is_emu_bot
    assert len(bot_reasons) >= 2
    assert any("webdriver" in r.lower() for r in bot_reasons)
    assert any("headless" in r.lower() for r in bot_reasons)

    # Suspicious screen dimensions
    fp_tiny_screen = DeviceFingerprint(screen=ScreenGeometry(width=100, height=100))
    is_emu_screen, screen_reasons = detect_emulator_signatures(fp_tiny_screen)
    assert is_emu_screen
    assert any("screen dimensions" in r.lower() for r in screen_reasons)

    # 0 hardware concurrency
    fp_zero_cores = DeviceFingerprint(hardware_concurrency=0)
    is_emu_cores, core_reasons = detect_emulator_signatures(fp_zero_cores)
    assert is_emu_cores
    assert any("0 logical cores" in r.lower() for r in core_reasons)


def test_calculate_fingerprint_entropy() -> None:
    fp_rich = DeviceFingerprint(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0"
        ),
        os_platform="Windows",
        browser_family="Firefox",
        screen=ScreenGeometry(width=2560, height=1440, color_depth=24, pixel_ratio=1.25),
        language="en-US",
        timezone_offset_minutes=-300,
        hardware_concurrency=12,
        device_memory_gb=32.0,
        webgl_vendor="AMD",
        webgl_renderer="Radeon RX 6800",
        canvas_hash="hash_canvas_complex_12345",
        audio_hash="hash_audio_complex_67890",
    )
    entropy = calculate_fingerprint_entropy(fp_rich)
    assert entropy > 3.0

    # Degenerate empty fingerprint has lower entropy
    fp_empty = DeviceFingerprint()
    empty_entropy = calculate_fingerprint_entropy(fp_empty)
    assert 0.0 <= empty_entropy < entropy


def test_device_feature_extractor() -> None:
    tracker = DeviceTracker()
    extractor = DeviceFeatureExtractor(tracker=tracker)

    # Raw transaction record without fingerprint attributes
    rec_plain = {"card_id": "c1", "timestamp": 1000.0}
    feats_plain = extractor.extract_features(rec_plain)
    assert set(feats_plain.keys()) == set(DEVICE_FEATURE_NAMES)
    assert feats_plain["device_seen_count"] == 1.0
    assert feats_plain["device_distinct_cards_24h"] == 1.0
    assert feats_plain["device_is_emulator"] == 0.0

    # Transaction with headless Chrome bot
    rec_bot = {
        "device_id": "bot_device",
        "card_id": "c2",
        "timestamp": 1010.0,
        "user_agent": "Mozilla/5.0 HeadlessChrome/120.0",
        "has_webdriver": True,
    }
    feats_bot = extractor.extract_features(rec_bot)
    assert feats_bot["device_is_emulator"] == 1.0

    # Repeat card cycling on same device
    extractor.extract_features({"device_id": "bot_device", "card_id": "c3", "timestamp": 1020.0})
    feats_cycle = extractor.extract_features(
        {"device_id": "bot_device", "card_id": "c4", "timestamp": 1030.0}
    )
    assert feats_cycle["device_is_card_cycling"] == 1.0
    assert feats_cycle["device_distinct_cards_24h"] == 3.0


def test_device_validation_and_edge_cases() -> None:
    # ScreenGeometry from_dict with ScreenGeometry instance
    sg = ScreenGeometry(1920, 1080)
    fp_with_sg = DeviceFingerprint.from_dict({"device_id": "d1", "screen": sg})
    assert fp_with_sg.screen == sg

    # DeviceFingerprint invalid types
    with pytest.raises(
        DeviceFingerprintError, match="device_id must be a non-empty string or None"
    ):
        DeviceFingerprint(device_id="")
    with pytest.raises(DeviceFingerprintError, match="user_agent must be a string"):
        DeviceFingerprint(user_agent=123)  # type: ignore[arg-type]
    with pytest.raises(DeviceFingerprintError, match="os_platform must be a string"):
        DeviceFingerprint(os_platform=123)  # type: ignore[arg-type]
    with pytest.raises(DeviceFingerprintError, match="browser_family must be a string"):
        DeviceFingerprint(browser_family=123)  # type: ignore[arg-type]
    with pytest.raises(
        DeviceFingerprintError, match="screen must be a ScreenGeometry instance or None"
    ):
        DeviceFingerprint(screen="invalid")  # type: ignore[arg-type]
    with pytest.raises(DeviceFingerprintError, match="language must be a string"):
        DeviceFingerprint(language=123)  # type: ignore[arg-type]
    with pytest.raises(DeviceFingerprintError, match="webgl_vendor must be a string"):
        DeviceFingerprint(webgl_vendor=123)  # type: ignore[arg-type]
    with pytest.raises(DeviceFingerprintError, match="webgl_renderer must be a string"):
        DeviceFingerprint(webgl_renderer=123)  # type: ignore[arg-type]
    with pytest.raises(DeviceFingerprintError, match="canvas_hash must be a string"):
        DeviceFingerprint(canvas_hash=123)  # type: ignore[arg-type]
    with pytest.raises(DeviceFingerprintError, match="audio_hash must be a string"):
        DeviceFingerprint(audio_hash=123)  # type: ignore[arg-type]
    with pytest.raises(DeviceFingerprintError, match="is_touch_capable must be a boolean"):
        DeviceFingerprint(is_touch_capable="yes")  # type: ignore[arg-type]
    with pytest.raises(DeviceFingerprintError, match="has_webdriver must be a boolean"):
        DeviceFingerprint(has_webdriver="yes")  # type: ignore[arg-type]

    # DeviceHasher type checks
    with pytest.raises(DeviceFingerprintError, match="fp must be an instance of DeviceFingerprint"):
        DeviceHasher.canonical_hash("not_fp")  # type: ignore[arg-type]
    with pytest.raises(DeviceFingerprintError, match="fp must be an instance of DeviceFingerprint"):
        DeviceHasher.simhash("not_fp")  # type: ignore[arg-type]
    with pytest.raises(DeviceFingerprintError, match="hash_bits must be either 64 or 128"):
        DeviceHasher.simhash(DeviceFingerprint(), hash_bits=32)

    # Jaccard / similarity / hamming type checks
    with pytest.raises(DeviceFingerprintError, match="hex1 and hex2 must be strings"):
        calculate_hamming_distance(123, "abc")  # type: ignore[arg-type]
    with pytest.raises(DeviceFingerprintError, match="items1 and items2 must be iterable"):
        calculate_jaccard_similarity(123, "abc")  # type: ignore[arg-type]
    with pytest.raises(
        DeviceFingerprintError, match="fp1 and fp2 must be instances of DeviceFingerprint"
    ):
        calculate_fingerprint_similarity("fp1", DeviceFingerprint())  # type: ignore[arg-type]


def test_device_similarity_branch_coverage() -> None:
    # 1. Both screens None
    fp1 = DeviceFingerprint(
        hardware_concurrency=4,
        device_memory_gb=8.0,
        screen=None,
        webgl_vendor="Google Inc.",
        webgl_renderer="ANGLE",
        canvas_hash="canvas_1",
        audio_hash="audio_1",
        language="en-US",
        timezone_offset_minutes=-300,
    )
    fp2 = DeviceFingerprint(
        hardware_concurrency=8,
        device_memory_gb=16.0,
        screen=None,
        webgl_vendor="Google Inc.",
        webgl_renderer="ANGLE",
        canvas_hash="canvas_1",
        audio_hash="audio_1",
        language="en-GB",
        timezone_offset_minutes=-360,
    )
    sim1 = calculate_fingerprint_similarity(fp1, fp2)
    assert 0.0 < sim1 < 1.0

    # 2. Timezone diff between 60 and 180
    fp3 = DeviceFingerprint(timezone_offset_minutes=0)
    fp4 = DeviceFingerprint(timezone_offset_minutes=120)
    sim_tz = calculate_fingerprint_similarity(fp3, fp4)
    assert sim_tz > 0.0

    # 3. Differing canvas/audio/renderer
    fp5 = DeviceFingerprint(
        webgl_vendor="VendorA",
        webgl_renderer="RendererA",
        canvas_hash="canvas_A",
        audio_hash="audio_A",
    )
    fp6 = DeviceFingerprint(
        webgl_vendor="VendorB",
        webgl_renderer="RendererB",
        canvas_hash="canvas_B",
        audio_hash="audio_B",
    )
    sim_diff = calculate_fingerprint_similarity(fp5, fp6)
    assert sim_diff <= 0.70


def test_device_tracker_eviction_and_drift() -> None:
    cfg = DeviceTrackerConfig(
        window_seconds=10.0,
        card_cycling_threshold=2,
        device_cycling_threshold=2,
        min_similarity_threshold=0.8,
    )
    tracker = DeviceTracker(config=cfg)

    # Invalid input arguments
    with pytest.raises(DeviceFingerprintError, match="device_id must be a non-empty string"):
        tracker.assess_event("  ", "card1", 100.0)
    with pytest.raises(DeviceFingerprintError, match="card_id must be a non-empty string or None"):
        tracker.assess_event("dev1", "  ", 100.0)
    with pytest.raises(DeviceFingerprintError, match="timestamp must be a finite float"):
        tracker.assess_event("dev1", "card1", float("nan"))

    # Initial record without card
    fp_base = DeviceFingerprint(
        device_id="dev_drift",
        hardware_concurrency=8,
        device_memory_gb=16.0,
        canvas_hash="canvas_orig",
        audio_hash="audio_orig",
    )
    r1 = tracker.record_event("dev_drift", None, 100.0, fp_base)
    assert r1.severity == "LOW"

    # Profile to_dict and AnomalyReport to_dict
    prof = tracker.get_profile("dev_drift")
    assert prof is not None
    p_dict = prof.to_dict()
    assert p_dict["device_id"] == "dev_drift"
    rep_dict = r1.to_dict()
    assert rep_dict["device_id"] == "dev_drift"

    # Second event with radically different fingerprint -> drift
    fp_drifted = DeviceFingerprint(
        device_id="dev_drift",
        hardware_concurrency=2,
        device_memory_gb=2.0,
        canvas_hash="canvas_other",
        audio_hash="audio_other",
        os_platform="FreeBSD",
        language="zh-CN",
    )
    r2 = tracker.record_event("dev_drift", "c1", 105.0, fp_drifted)
    assert r2.is_fingerprint_drift is True
    assert "regressed below" in r2.reasons[0]

    # Time expiration outside window_seconds (t=125 > 105 + 10)
    r3 = tracker.record_event("dev_drift", "c2", 125.0)
    assert r3.distinct_cards_count == 1  # previous card expired from sliding window


def test_emulator_heuristics_all_branches() -> None:
    # Invalid argument
    with pytest.raises(DeviceFingerprintError, match="fp must be an instance of DeviceFingerprint"):
        detect_emulator_signatures("not_a_fp")  # type: ignore[arg-type]

    # 1. Screen size suspicious
    fp_zero_screen = DeviceFingerprint(
        user_agent="Mozilla/5.0",
        screen=ScreenGeometry(width=10, height=10),
    )
    is_emu, reasons = detect_emulator_signatures(fp_zero_screen)
    assert is_emu is True
    assert any("Suspicious screen dimensions" in r for r in reasons)

    # 2. Concurrency == 0
    fp_concurr = DeviceFingerprint(
        user_agent="Mozilla/5.0",
        hardware_concurrency=0,
    )
    is_emu, reasons = detect_emulator_signatures(fp_concurr)
    assert is_emu is True
    assert any("Hardware concurrency reported as 0" in r for r in reasons)

    # 3. Memory == 0.0 GB
    fp_mem = DeviceFingerprint(
        user_agent="Mozilla/5.0",
        device_memory_gb=0.0,
    )
    is_emu, reasons = detect_emulator_signatures(fp_mem)
    assert is_emu is True
    assert any("Device memory reported as 0 GB RAM" in r for r in reasons)

    # 4. Known automation keywords in UA
    for kw in (
        "phantomjs",
        "selenium",
        "puppeteer",
        "playwright",
        "electron",
        "nightwatch",
    ):
        fp_kw = DeviceFingerprint(user_agent=f"Mozilla/5.0 ({kw}/1.0)")
        is_emu, reasons = detect_emulator_signatures(fp_kw)
        assert is_emu is True
        assert any("Automated browser/emulator signature" in r for r in reasons)

    # 5. Mobile user agent without touch
    fp_mob = DeviceFingerprint(
        user_agent="Mozilla/5.0 (Android 12; Mobile; rv:68.0)",
        is_touch_capable=False,
    )
    is_emu, reasons = detect_emulator_signatures(fp_mob)
    assert is_emu is True
    assert any("Mobile user agent reported without touch capability" in r for r in reasons)

    # 6. Missing WebGL / Canvas with Desktop UA and screen
    fp_missing_render = DeviceFingerprint(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        screen=ScreenGeometry(width=1920, height=1080),
        canvas_hash="",
        webgl_vendor="",
        webgl_renderer="",
    )
    is_emu, reasons = detect_emulator_signatures(fp_missing_render)
    assert is_emu is True
    assert any("Desktop browser completely missing WebGL and Canvas" in r for r in reasons)


def test_device_entropy_and_extractor_branches() -> None:
    # Entropy type check
    with pytest.raises(DeviceFingerprintError, match="fp must be an instance of DeviceFingerprint"):
        calculate_fingerprint_entropy("not_fp")  # type: ignore[arg-type]

    # Extractor invalid record
    extractor = DeviceFeatureExtractor()
    with pytest.raises(DeviceFingerprintError, match="record must be a Mapping"):
        extractor.extract_features("not_a_mapping")  # type: ignore[arg-type]

    # Extractor with corrupted device subdict
    rec_bad_dev = {"device": "not_a_dict", "timestamp": 1000.0}
    feats = extractor.extract_features(rec_bad_dev)
    assert feats["device_seen_count"] == 1.0

    # Batch extraction
    records = [
        {"device_id": "d1", "card_id": "c1", "timestamp": 1000.0},
        {"device_id": "d2", "card_id": "c2", "timestamp": 1005.0},
    ]
    batch_feats = extractor.extract_batch(records)
    assert len(batch_feats) == 2


def test_device_comprehensive_branch_coverage() -> None:
    # 1. from_dict with no screen / screen None
    fp_no_screen = DeviceFingerprint.from_dict({"device_id": "dev_no_screen", "screen": None})
    assert fp_no_screen.screen is None

    # 2. simhash with screen
    fp_with_screen = DeviceFingerprint(
        device_id="dev_scr",
        screen=ScreenGeometry(1920, 1080),
    )
    sh = DeviceHasher.simhash(fp_with_screen)
    assert len(sh) == 16

    # 3. calculate_fingerprint_similarity branches
    # 3a. device_memory_gb == 0 for both
    fp_mem0_a = DeviceFingerprint(device_memory_gb=0.0, screen=ScreenGeometry(1920, 1080))
    fp_mem0_b = DeviceFingerprint(device_memory_gb=0.0, screen=None)
    sim_mem0 = calculate_fingerprint_similarity(fp_mem0_a, fp_mem0_b)
    assert 0.0 < sim_mem0 < 1.0

    # 3b. Differing webgl_vendor, webgl_renderer, canvas, audio, os, browser, touch
    fp_mismatch_a = DeviceFingerprint(
        webgl_vendor="Vendor1",
        webgl_renderer="Renderer1",
        canvas_hash="canvas1",
        audio_hash="audio1",
        os_platform="Windows",
        browser_family="Chrome",
        is_touch_capable=True,
    )
    fp_mismatch_b = DeviceFingerprint(
        webgl_vendor="",
        webgl_renderer="",
        canvas_hash="",
        audio_hash="",
        os_platform="",
        browser_family="",
        is_touch_capable=False,
    )
    sim_mismatch = calculate_fingerprint_similarity(fp_mismatch_a, fp_mismatch_b)
    assert sim_mismatch < 0.5

    # 4. Tracker baseline update & card deque expiration & tracked device count
    tracker = DeviceTracker(config=DeviceTrackerConfig(window_seconds=10.0))
    assert tracker.get_tracked_device_count() == 0
    # First record without fingerprint
    tracker.record_event(device_id="d_base", card_id="c_base", timestamp=100.0, fingerprint=None)
    assert tracker.get_tracked_device_count() == 1
    # Second record with fingerprint -> baseline set
    tracker.record_event(
        device_id="d_base", card_id="c_base", timestamp=105.0, fingerprint=fp_with_screen
    )
    prof = tracker.get_profile("d_base")
    assert prof is not None and prof.baseline_fingerprint is not None

    # Card deque expiration (t=120 > 100 + 10)
    tracker.record_event(device_id="d_other", card_id="c_base", timestamp=120.0)

    # 5. DeviceFeatureExtractor branches
    ext = DeviceFeatureExtractor(tracker=tracker)
    # 5a. raw_fp is DeviceFingerprint instance
    f_fp = ext.extract_features({"fingerprint": fp_with_screen, "timestamp": 130.0})
    assert f_fp["device_seen_count"] >= 1.0

    # 5b. raw_fp is Mapping
    f_map = ext.extract_features(
        {"device_fingerprint": {"user_agent": "Mozilla/5.0"}, "timestamp": 131.0}
    )
    assert "device_is_emulator" in f_map

    # 5c. Flat screen dimensions in record
    f_flat_screen = ext.extract_features(
        {
            "screen_width": 1920,
            "screen_height": 1080,
            "screen_color_depth": 24,
            "screen_pixel_ratio": 1.0,
            "user_agent": "Mozilla/5.0",
            "unix_time": 132.0,
        }
    )
    assert f_flat_screen["device_seen_count"] >= 1.0

    # 5d. Malformed screen dimensions & malformed fp fields
    f_bad_scr = ext.extract_features(
        {
            "screen_width": -10,
            "screen_height": "not_int",
            "user_agent": 123,
            "trans_date_trans_time": "invalid_date_format",
        }
    )
    assert f_bad_scr["device_seen_count"] >= 1.0

    # 5e. mutate_tracker=False
    f_readonly = ext.extract_features(
        {"device_id": "d_readonly", "card_id": "c_ro", "timestamp": 140.0},
        mutate_tracker=False,
    )
    assert f_readonly["device_seen_count"] == 1.0

    # 5f. Auto-generated device_id from canonical hash
    fp_auto_id = DeviceFingerprint(user_agent="AutoID/1.0")
    f_auto = ext.extract_features({"fingerprint": fp_auto_id, "timestamp": 150.0})
    assert f_auto["device_seen_count"] >= 1.0

    # 5g. Emulator signature detected by extractor check
    fp_emu = DeviceFingerprint(user_agent="HeadlessChrome/90.0", device_id="d_emu_test")
    f_emu = ext.extract_features({"fingerprint": fp_emu, "timestamp": 160.0})
    assert f_emu["device_is_emulator"] == 1.0
