"""Real-time adaptive device and browser fingerprinting with anomaly clustering."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, field
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


class DeviceHasher:
    """Computes deterministic and fuzzy hashes of client device fingerprints."""

    @staticmethod
    def canonical_hash(fp: DeviceFingerprint) -> str:
        """Compute deterministic SHA-256 canonical hash of normalized stable attributes."""
        if not isinstance(fp, DeviceFingerprint):
            raise DeviceFingerprintError("fp must be an instance of DeviceFingerprint.")
        screen_repr = (
            f"{fp.screen.width}x{fp.screen.height}x{fp.screen.color_depth}@{fp.screen.pixel_ratio:.1f}"
            if fp.screen is not None
            else "none"
        )
        canonical_components = [
            f"os:{fp.os_platform.strip().lower()}",
            f"browser:{fp.browser_family.strip().lower()}",
            f"screen:{screen_repr}",
            f"concurrency:{fp.hardware_concurrency}",
            f"memory:{fp.device_memory_gb:.1f}",
            f"webgl_v:{fp.webgl_vendor.strip().lower()}",
            f"webgl_r:{fp.webgl_renderer.strip().lower()}",
            f"canvas:{fp.canvas_hash.strip().lower()}",
            f"audio:{fp.audio_hash.strip().lower()}",
            f"touch:{int(fp.is_touch_capable)}",
            f"webdriver:{int(fp.has_webdriver)}",
            f"plugins:{fp.plugins_count}",
        ]
        raw_repr = "|".join(canonical_components)
        return hashlib.sha256(raw_repr.encode("utf-8")).hexdigest()

    @staticmethod
    def simhash(fp: DeviceFingerprint, hash_bits: int = 64) -> str:
        """Compute locality-sensitive simhash bit-vector hex string for fuzzy clustering."""
        if not isinstance(fp, DeviceFingerprint):
            raise DeviceFingerprintError("fp must be an instance of DeviceFingerprint.")
        if hash_bits not in (64, 128):
            raise DeviceFingerprintError("hash_bits must be either 64 or 128.")

        tokens = [
            (f"os:{fp.os_platform.strip().lower()}", 3),
            (f"browser:{fp.browser_family.strip().lower()}", 2),
            (f"concurrency:{fp.hardware_concurrency}", 4),
            (f"memory:{int(fp.device_memory_gb)}", 3),
            (f"canvas:{fp.canvas_hash.strip().lower()}", 5),
            (f"audio:{fp.audio_hash.strip().lower()}", 4),
            (f"webgl:{fp.webgl_vendor.strip().lower()}:{fp.webgl_renderer.strip().lower()}", 4),
            (f"touch:{int(fp.is_touch_capable)}", 2),
        ]
        if fp.screen is not None:
            tokens.append((f"screen:{fp.screen.width}x{fp.screen.height}", 4))

        v = [0] * hash_bits
        for token, weight in tokens:
            token_digest = hashlib.md5(token.encode("utf-8")).digest()  # noqa: S324
            num_bytes = hash_bits // 8
            token_int = int.from_bytes(token_digest[:num_bytes], byteorder="big")
            for i in range(hash_bits):
                bit = (token_int >> i) & 1
                if bit:
                    v[i] += weight
                else:
                    v[i] -= weight

        fingerprint = 0
        for i in range(hash_bits):
            if v[i] > 0:
                fingerprint |= 1 << i

        hex_len = hash_bits // 4
        return f"{fingerprint:0{hex_len}x}"


def calculate_jaccard_similarity(
    items1: Sequence[str] | Set[str],
    items2: Sequence[str] | Set[str],
) -> float:
    """Calculate Jaccard similarity coefficient between two token collections in [0.0, 1.0]."""
    try:
        set1 = set(items1)
        set2 = set(items2)
    except (TypeError, ValueError) as exc:
        raise DeviceFingerprintError(
            f"items1 and items2 must be iterable collections: {exc}"
        ) from exc
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return float(intersection / union)


def calculate_hamming_distance(hex1: str, hex2: str) -> int:
    """Calculate bitwise Hamming distance between two hexadecimal strings of equal length."""
    if not isinstance(hex1, str) or not isinstance(hex2, str):
        raise DeviceFingerprintError("hex1 and hex2 must be strings.")
    if len(hex1) != len(hex2) or len(hex1) == 0:
        raise DeviceFingerprintError("hex1 and hex2 must be non-empty strings of equal length.")
    try:
        val1 = int(hex1, 16)
        val2 = int(hex2, 16)
    except ValueError as exc:
        raise DeviceFingerprintError(f"Invalid hexadecimal string: {exc}") from exc

    xor_val = val1 ^ val2
    return xor_val.bit_count()


def calculate_fingerprint_similarity(fp1: DeviceFingerprint, fp2: DeviceFingerprint) -> float:
    """Calculate multi-attribute stability-weighted composite similarity in [0.0, 1.0]."""
    if not isinstance(fp1, DeviceFingerprint) or not isinstance(fp2, DeviceFingerprint):
        raise DeviceFingerprintError("fp1 and fp2 must be instances of DeviceFingerprint.")
    if fp1 == fp2:
        return 1.0

    score = 0.0

    # 1. Hardware attributes (weight: 0.30)
    hw_score = 0.0
    hw_components = 0
    if fp1.hardware_concurrency == fp2.hardware_concurrency:
        hw_score += 1.0
    hw_components += 1

    if fp1.device_memory_gb > 0 and fp2.device_memory_gb > 0:
        mem_ratio = min(fp1.device_memory_gb, fp2.device_memory_gb) / max(
            fp1.device_memory_gb, fp2.device_memory_gb
        )
        hw_score += mem_ratio
    elif fp1.device_memory_gb == fp2.device_memory_gb:
        hw_score += 1.0
    hw_components += 1

    if fp1.screen is not None and fp2.screen is not None:
        screen_sim = 0.0
        if fp1.screen.width == fp2.screen.width and fp1.screen.height == fp2.screen.height:
            screen_sim += 0.7
        if fp1.screen.pixel_ratio == fp2.screen.pixel_ratio:
            screen_sim += 0.3
        hw_score += screen_sim
    elif fp1.screen is None and fp2.screen is None:
        hw_score += 1.0
    hw_components += 1
    score += (hw_score / hw_components) * 0.30

    # 2. Rendering / canvas / audio (weight: 0.35)
    render_score = 0.0
    render_components = 0
    if fp1.webgl_vendor and fp2.webgl_vendor:
        if fp1.webgl_vendor.strip().lower() == fp2.webgl_vendor.strip().lower():
            render_score += 0.5
    elif fp1.webgl_vendor == fp2.webgl_vendor:
        render_score += 0.5
    if fp1.webgl_renderer and fp2.webgl_renderer:
        if fp1.webgl_renderer.strip().lower() == fp2.webgl_renderer.strip().lower():
            render_score += 0.5
    elif fp1.webgl_renderer == fp2.webgl_renderer:
        render_score += 0.5
    render_components += 1

    if fp1.canvas_hash and fp2.canvas_hash:
        render_score += 1.0 if fp1.canvas_hash == fp2.canvas_hash else 0.0
    elif fp1.canvas_hash == fp2.canvas_hash:
        render_score += 1.0
    render_components += 1

    if fp1.audio_hash and fp2.audio_hash:
        render_score += 1.0 if fp1.audio_hash == fp2.audio_hash else 0.0
    elif fp1.audio_hash == fp2.audio_hash:
        render_score += 1.0
    render_components += 1
    score += (render_score / render_components) * 0.35

    # 3. Browser & platform (weight: 0.20)
    bp_score = 0.0
    bp_components = 0
    if fp1.os_platform and fp2.os_platform:
        bp_score += 1.0 if fp1.os_platform.lower() == fp2.os_platform.lower() else 0.0
    elif fp1.os_platform == fp2.os_platform:
        bp_score += 1.0
    bp_components += 1

    if fp1.browser_family and fp2.browser_family:
        bp_score += 1.0 if fp1.browser_family.lower() == fp2.browser_family.lower() else 0.0
    elif fp1.browser_family == fp2.browser_family:
        bp_score += 1.0
    bp_components += 1

    if fp1.is_touch_capable == fp2.is_touch_capable:
        bp_score += 1.0
    bp_components += 1
    score += (bp_score / bp_components) * 0.20

    # 4. Locale & environment (weight: 0.15)
    env_score = 0.0
    env_components = 0
    lang1 = fp1.language.strip().lower()
    lang2 = fp2.language.strip().lower()
    if lang1 == lang2:
        env_score += 1.0
    elif lang1.split("-")[0] == lang2.split("-")[0]:
        env_score += 0.7
    env_components += 1

    tz_diff = abs(fp1.timezone_offset_minutes - fp2.timezone_offset_minutes)
    if tz_diff == 0:
        env_score += 1.0
    elif tz_diff <= 60:
        env_score += 0.8
    elif tz_diff <= 180:
        env_score += 0.5
    env_components += 1
    score += (env_score / env_components) * 0.15

    return max(0.0, min(1.0, float(score)))


DEFAULT_DEVICE_WINDOW_SECONDS: float = 86400.0  # 24 hours
DEFAULT_MAX_DEVICES: int = 50_000
DEFAULT_CARD_CYCLING_THRESHOLD: int = 3
DEFAULT_DEVICE_CYCLING_THRESHOLD: int = 3
DEFAULT_MIN_SIMILARITY_THRESHOLD: float = 0.60


@dataclass(frozen=True)
class DeviceTrackerConfig:
    """Configuration settings for streaming device tracking and anomaly clustering."""

    window_seconds: float = DEFAULT_DEVICE_WINDOW_SECONDS
    max_devices: int = DEFAULT_MAX_DEVICES
    card_cycling_threshold: int = DEFAULT_CARD_CYCLING_THRESHOLD
    device_cycling_threshold: int = DEFAULT_DEVICE_CYCLING_THRESHOLD
    min_similarity_threshold: float = DEFAULT_MIN_SIMILARITY_THRESHOLD

    def __post_init__(self) -> None:
        if (
            not isinstance(self.window_seconds, (int, float))
            or math.isnan(self.window_seconds)
            or math.isinf(self.window_seconds)
            or self.window_seconds <= 0.0
        ):
            raise DeviceFingerprintError("window_seconds must be a positive finite float.")
        if (
            not isinstance(self.max_devices, int)
            or isinstance(self.max_devices, bool)
            or self.max_devices <= 0
        ):
            raise DeviceFingerprintError("max_devices must be a positive integer.")
        if (
            not isinstance(self.card_cycling_threshold, int)
            or isinstance(self.card_cycling_threshold, bool)
            or self.card_cycling_threshold <= 0
        ):
            raise DeviceFingerprintError("card_cycling_threshold must be a positive integer.")
        if (
            not isinstance(self.device_cycling_threshold, int)
            or isinstance(self.device_cycling_threshold, bool)
            or self.device_cycling_threshold <= 0
        ):
            raise DeviceFingerprintError("device_cycling_threshold must be a positive integer.")
        if (
            not isinstance(self.min_similarity_threshold, (int, float))
            or math.isnan(self.min_similarity_threshold)
            or math.isinf(self.min_similarity_threshold)
            or not (0.0 <= self.min_similarity_threshold <= 1.0)
        ):
            raise DeviceFingerprintError(
                "min_similarity_threshold must be a finite float between 0.0 and 1.0."
            )


@dataclass(frozen=True)
class DeviceAnomalyReport:
    """Report summarizing detected device anomalies, risk scores, and attack indicators."""

    device_id: str
    card_id: str | None = None
    is_emulator: bool = False
    is_card_cycling: bool = False
    is_device_cycling: bool = False
    is_fingerprint_drift: bool = False
    distinct_cards_count: int = 0
    distinct_devices_count: int = 0
    similarity_to_baseline: float = 1.0
    severity: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    reasons: tuple[str, ...] = ()

    @property
    def has_anomaly(self) -> bool:
        """Return True if any anomaly flag is raised."""
        return (
            self.is_emulator
            or self.is_card_cycling
            or self.is_device_cycling
            or self.is_fingerprint_drift
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert anomaly report to dictionary."""
        return {
            "device_id": self.device_id,
            "card_id": self.card_id,
            "is_emulator": self.is_emulator,
            "is_card_cycling": self.is_card_cycling,
            "is_device_cycling": self.is_device_cycling,
            "is_fingerprint_drift": self.is_fingerprint_drift,
            "distinct_cards_count": self.distinct_cards_count,
            "distinct_devices_count": self.distinct_devices_count,
            "similarity_to_baseline": float(self.similarity_to_baseline),
            "severity": self.severity,
            "reasons": list(self.reasons),
            "has_anomaly": self.has_anomaly,
        }


@dataclass
class DeviceProfile:
    """Aggregated historical device state within the sliding observation window."""

    device_id: str
    first_seen_timestamp: float
    last_seen_timestamp: float
    transaction_count: int = 0
    cards_seen: set[str] = field(default_factory=set)
    canonical_hashes: set[str] = field(default_factory=set)
    baseline_fingerprint: DeviceFingerprint | None = None
    is_emulator: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert device profile to serializable dictionary."""
        return {
            "device_id": self.device_id,
            "first_seen_timestamp": self.first_seen_timestamp,
            "last_seen_timestamp": self.last_seen_timestamp,
            "transaction_count": self.transaction_count,
            "distinct_cards_count": len(self.cards_seen),
            "distinct_hashes_count": len(self.canonical_hashes),
            "is_emulator": self.is_emulator,
        }


class DeviceTracker:
    """Thread-safe streaming sliding-window device association tracker and anomaly clusterer."""

    def __init__(self, config: DeviceTrackerConfig | None = None) -> None:
        self.config = config if config is not None else DeviceTrackerConfig()
        self._lock = threading.RLock()
        self._device_profiles: OrderedDict[str, DeviceProfile] = OrderedDict()
        self._device_card_events: OrderedDict[str, deque[tuple[float, str]]] = OrderedDict()
        self._card_device_events: OrderedDict[str, deque[tuple[float, str]]] = OrderedDict()
        self._total_evaluations: int = 0
        self._total_anomalies: int = 0

    def assess_event(
        self,
        device_id: str,
        card_id: str | None,
        timestamp: float,
        fingerprint: DeviceFingerprint | None = None,
    ) -> DeviceAnomalyReport:
        """Evaluate a device/card event for anomalies without mutating tracker state."""
        if not isinstance(device_id, str) or not device_id.strip():
            raise DeviceFingerprintError("device_id must be a non-empty string.")
        if card_id is not None and (not isinstance(card_id, str) or not card_id.strip()):
            raise DeviceFingerprintError("card_id must be a non-empty string or None.")
        if (
            not isinstance(timestamp, (int, float))
            or math.isnan(timestamp)
            or math.isinf(timestamp)
        ):
            raise DeviceFingerprintError("timestamp must be a finite float.")

        clean_device = device_id.strip()
        clean_card = card_id.strip() if card_id is not None else None
        cutoff = timestamp - self.config.window_seconds

        with self._lock:
            device_events = self._device_card_events.get(clean_device)
            active_cards = {c for (t, c) in (device_events or ()) if t >= cutoff and c is not None}
            if clean_card is not None:
                active_cards.add(clean_card)
            distinct_cards = len(active_cards)

            distinct_devices = 0
            if clean_card is not None:
                card_events = self._card_device_events.get(clean_card)
                active_devices = {d for (t, d) in (card_events or ()) if t >= cutoff}
                active_devices.add(clean_device)
                distinct_devices = len(active_devices)

            profile = self._device_profiles.get(clean_device)
            is_emulator = profile.is_emulator if profile is not None else False
            similarity_to_baseline = 1.0
            if fingerprint is not None:
                if fingerprint.has_webdriver:
                    is_emulator = True
                if profile is not None and profile.baseline_fingerprint is not None:
                    similarity_to_baseline = calculate_fingerprint_similarity(
                        profile.baseline_fingerprint, fingerprint
                    )

            is_card_cycling = distinct_cards >= self.config.card_cycling_threshold
            is_device_cycling = (
                distinct_devices >= self.config.device_cycling_threshold
                if clean_card is not None
                else False
            )
            is_drift = similarity_to_baseline < self.config.min_similarity_threshold

            reasons: list[str] = []
            if is_emulator:
                reasons.append("Device exhibits automated webdriver or emulator signatures.")
            if is_card_cycling:
                thresh_c = self.config.card_cycling_threshold
                reasons.append(
                    f"Device associated with {distinct_cards} cards (threshold: {thresh_c})."
                )
            if is_device_cycling:
                thresh_d = self.config.device_cycling_threshold
                reasons.append(
                    f"Card associated with {distinct_devices} devices (threshold: {thresh_d})."
                )
            if is_drift:
                min_sim = self.config.min_similarity_threshold
                sim_val = f"{similarity_to_baseline:.2f}"
                reasons.append(f"Fingerprint similarity {sim_val} regressed below {min_sim:.2f}.")

            if is_card_cycling and is_emulator:
                severity = "CRITICAL"
            elif is_card_cycling or is_device_cycling:
                severity = "HIGH"
            elif is_emulator or is_drift:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            return DeviceAnomalyReport(
                device_id=clean_device,
                card_id=clean_card,
                is_emulator=is_emulator,
                is_card_cycling=is_card_cycling,
                is_device_cycling=is_device_cycling,
                is_fingerprint_drift=is_drift,
                distinct_cards_count=distinct_cards,
                distinct_devices_count=distinct_devices,
                similarity_to_baseline=similarity_to_baseline,
                severity=severity,
                reasons=tuple(reasons),
            )

    def record_event(
        self,
        device_id: str,
        card_id: str | None,
        timestamp: float,
        fingerprint: DeviceFingerprint | None = None,
    ) -> DeviceAnomalyReport:
        """Record an event into sliding windows and return anomaly report."""
        report = self.assess_event(
            device_id=device_id,
            card_id=card_id,
            timestamp=timestamp,
            fingerprint=fingerprint,
        )

        clean_device = device_id.strip()
        clean_card = card_id.strip() if card_id is not None else None
        cutoff = timestamp - self.config.window_seconds

        with self._lock:
            self._total_evaluations += 1
            if report.has_anomaly:
                self._total_anomalies += 1

            profile = self._device_profiles.get(clean_device)
            if profile is None:
                while len(self._device_profiles) >= self.config.max_devices:
                    self._device_profiles.popitem(last=False)

                profile = DeviceProfile(
                    device_id=clean_device,
                    first_seen_timestamp=timestamp,
                    last_seen_timestamp=timestamp,
                    transaction_count=1,
                    cards_seen={clean_card} if clean_card else set(),
                    canonical_hashes=set(),
                    baseline_fingerprint=fingerprint,
                    is_emulator=report.is_emulator,
                )
                if fingerprint is not None:
                    profile.canonical_hashes.add(DeviceHasher.canonical_hash(fingerprint))
                self._device_profiles[clean_device] = profile
            else:
                profile.last_seen_timestamp = max(profile.last_seen_timestamp, timestamp)
                profile.transaction_count += 1
                if clean_card:
                    profile.cards_seen.add(clean_card)
                if fingerprint is not None:
                    if profile.baseline_fingerprint is None:
                        profile.baseline_fingerprint = fingerprint
                    profile.canonical_hashes.add(DeviceHasher.canonical_hash(fingerprint))
                if report.is_emulator:
                    profile.is_emulator = True
                self._device_profiles.move_to_end(clean_device)

            if clean_device not in self._device_card_events:
                self._device_card_events[clean_device] = deque()
            dq_dev = self._device_card_events[clean_device]
            if clean_card:
                dq_dev.append((timestamp, clean_card))
            while dq_dev and dq_dev[0][0] < cutoff:
                dq_dev.popleft()
            self._device_card_events.move_to_end(clean_device)

            if clean_card:
                if clean_card not in self._card_device_events:
                    self._card_device_events[clean_card] = deque()
                dq_card = self._card_device_events[clean_card]
                dq_card.append((timestamp, clean_device))
                while dq_card and dq_card[0][0] < cutoff:
                    dq_card.popleft()
                self._card_device_events.move_to_end(clean_card)

            return report

    def get_profile(self, device_id: str) -> DeviceProfile | None:
        """Retrieve the current DeviceProfile for a device ID if known."""
        clean = device_id.strip()
        with self._lock:
            return self._device_profiles.get(clean)

    def get_tracked_device_count(self) -> int:
        """Return the number of currently tracked devices."""
        with self._lock:
            return len(self._device_profiles)

    def get_stats(self) -> dict[str, Any]:
        """Return operational summary metrics for monitored devices and anomalies."""
        with self._lock:
            return {
                "tracked_devices_count": len(self._device_profiles),
                "tracked_cards_count": len(self._card_device_events),
                "total_evaluations": self._total_evaluations,
                "total_anomalies": self._total_anomalies,
                "max_devices_capacity": self.config.max_devices,
                "window_seconds": self.config.window_seconds,
            }

    def reset(self) -> None:
        """Reset all in-memory tracker state and metrics."""
        with self._lock:
            self._device_profiles.clear()
            self._device_card_events.clear()
            self._card_device_events.clear()
            self._total_evaluations = 0
            self._total_anomalies = 0


EMULATOR_USER_AGENT_PATTERNS: tuple[str, ...] = (
    "headlesschrome",
    "phantomjs",
    "selenium",
    "puppeteer",
    "playwright",
    "electron",
    "nightwatch",
    "webdriver",
)


def detect_emulator_signatures(fp: DeviceFingerprint) -> tuple[bool, tuple[str, ...]]:
    """Detect bot runtime, automated webdriver, and emulator anomalies in a fingerprint."""
    if not isinstance(fp, DeviceFingerprint):
        raise DeviceFingerprintError("fp must be an instance of DeviceFingerprint.")

    reasons: list[str] = []

    if fp.has_webdriver:
        reasons.append("Webdriver automation flag is set to true.")

    ua_lower = fp.user_agent.strip().lower()
    for pattern in EMULATOR_USER_AGENT_PATTERNS:
        if pattern in ua_lower:
            reasons.append(f"Automated browser/emulator signature '{pattern}' found in user-agent.")
            break

    if fp.screen is not None and (fp.screen.width < 300 or fp.screen.height < 300):
        dims = f"{fp.screen.width}x{fp.screen.height}"
        reasons.append(f"Suspicious screen dimensions {dims} below realistic bounds.")

    if fp.hardware_concurrency == 0:
        reasons.append("Hardware concurrency reported as 0 logical cores.")
    if fp.device_memory_gb == 0.0:
        reasons.append("Device memory reported as 0 GB RAM.")

    # Mismatch check: mobile user-agent but is_touch_capable is False
    if (
        any(m in ua_lower for m in ("android", "iphone", "ipad", "mobile"))
        and not fp.is_touch_capable
    ):
        reasons.append("Mobile user agent reported without touch capability.")

    # Desktop client with reported screen but completely missing WebGL and Canvas
    if (
        fp.screen is not None
        and any(d in ua_lower for d in ("windows", "macintosh", "x11", "linux"))
        and not any(m in ua_lower for m in ("android", "iphone", "mobile"))
        and not fp.canvas_hash
        and not fp.webgl_vendor
        and not fp.webgl_renderer
    ):
        reasons.append("Desktop browser completely missing WebGL and Canvas rendering context.")

    is_emulator = len(reasons) > 0
    return is_emulator, tuple(reasons)


def calculate_fingerprint_entropy(fp: DeviceFingerprint) -> float:
    """Calculate Shannon entropy over normalized fingerprint attribute representations."""
    if not isinstance(fp, DeviceFingerprint):
        raise DeviceFingerprintError("fp must be an instance of DeviceFingerprint.")

    screen_str = f"{fp.screen.width}x{fp.screen.height}" if fp.screen is not None else "none"
    tokens = [
        fp.user_agent.strip().lower(),
        fp.os_platform.strip().lower(),
        fp.browser_family.strip().lower(),
        screen_str,
        str(fp.hardware_concurrency),
        f"{fp.device_memory_gb:.1f}",
        fp.webgl_vendor.strip().lower(),
        fp.webgl_renderer.strip().lower(),
        fp.canvas_hash.strip().lower(),
        fp.audio_hash.strip().lower(),
        fp.language.strip().lower(),
        str(fp.timezone_offset_minutes),
        str(int(fp.is_touch_capable)),
        str(int(fp.has_webdriver)),
        str(fp.plugins_count),
    ]

    combined = "|".join(tokens)
    char_counts: dict[str, int] = {}

    for ch in combined:
        char_counts[ch] = char_counts.get(ch, 0) + 1

    total = len(combined)
    entropy = 0.0
    for count in char_counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return max(0.0, float(entropy))


DEVICE_FEATURE_NAMES: tuple[str, ...] = (
    "device_seen_count",
    "device_distinct_cards_24h",
    "device_card_distinct_devices_24h",
    "device_is_emulator",
    "device_entropy_score",
    "device_is_card_cycling",
    "device_is_device_cycling",
)


class DeviceFeatureExtractor:
    """Extracts structured device fingerprinting and anomaly features from transactions."""

    def __init__(
        self,
        tracker: DeviceTracker | None = None,
        min_entropy_threshold: float = 1.5,
    ) -> None:
        self.tracker = tracker if tracker is not None else DeviceTracker()
        self.min_entropy_threshold = float(min_entropy_threshold)

    def extract_features(
        self,
        record: Mapping[str, Any],
        mutate_tracker: bool = True,
    ) -> dict[str, float]:
        """Extract structured features from transaction record dictionary."""
        if not isinstance(record, Mapping):
            raise DeviceFingerprintError("record must be a Mapping.")

        fp: DeviceFingerprint | None = None
        raw_fp = record.get("fingerprint", record.get("device_fingerprint"))
        if isinstance(raw_fp, DeviceFingerprint):
            fp = raw_fp
        elif isinstance(raw_fp, Mapping):
            fp = DeviceFingerprint.from_dict(raw_fp)
        elif any(
            k in record for k in ("user_agent", "os_platform", "browser_family", "screen_width")
        ):
            screen_obj: ScreenGeometry | None = None
            if "screen_width" in record and "screen_height" in record:
                try:
                    w = int(record["screen_width"])
                    h = int(record["screen_height"])
                    if w > 0 and h > 0:
                        screen_obj = ScreenGeometry(
                            width=w,
                            height=h,
                            color_depth=int(record.get("screen_color_depth", 24)),
                            pixel_ratio=float(record.get("screen_pixel_ratio", 1.0)),
                        )
                except (ValueError, TypeError, DeviceFingerprintError):
                    screen_obj = None

            try:
                raw_dev_id = record.get("device_id")
                dev_id_str = str(raw_dev_id) if raw_dev_id else None
                fp = DeviceFingerprint(
                    device_id=dev_id_str,
                    user_agent=str(record.get("user_agent", "")),
                    os_platform=str(record.get("os_platform", "")),
                    browser_family=str(record.get("browser_family", "")),
                    screen=screen_obj,
                    language=str(record.get("language", "en")),
                    timezone_offset_minutes=int(record.get("timezone_offset_minutes", 0)),
                    hardware_concurrency=int(record.get("hardware_concurrency", 4)),
                    device_memory_gb=float(record.get("device_memory_gb", 8.0)),
                    webgl_vendor=str(record.get("webgl_vendor", "")),
                    webgl_renderer=str(record.get("webgl_renderer", "")),
                    canvas_hash=str(record.get("canvas_hash", "")),
                    audio_hash=str(record.get("audio_hash", "")),
                    is_touch_capable=bool(record.get("is_touch_capable", False)),
                    has_webdriver=bool(record.get("has_webdriver", False)),
                    plugins_count=int(record.get("plugins_count", 0)),
                )
            except (ValueError, TypeError, DeviceFingerprintError):
                fp = None

        device_id = str(record.get("device_id", "")).strip()
        if not device_id and fp is not None:
            device_id = fp.device_id or DeviceHasher.canonical_hash(fp)
        if not device_id:
            device_id = "device_unknown"

        card_id: str | None = None
        for key in ("card_id", "cc_num", "account_id", "user_id"):
            val = record.get(key)
            if val is not None and str(val).strip():
                card_id = str(val).strip()
                break

        timestamp = time.time()
        for t_key in ("timestamp", "trans_date_trans_time", "unix_time", "event_time"):
            if t_key in record:
                try:
                    val = float(record[t_key])
                    if not math.isnan(val) and not math.isinf(val) and val > 0:
                        timestamp = val
                        break
                except (ValueError, TypeError):
                    pass

        if mutate_tracker:
            report = self.tracker.record_event(
                device_id=device_id,
                card_id=card_id,
                timestamp=timestamp,
                fingerprint=fp,
            )
        else:
            report = self.tracker.assess_event(
                device_id=device_id,
                card_id=card_id,
                timestamp=timestamp,
                fingerprint=fp,
            )

        is_emu = report.is_emulator
        if fp is not None and not is_emu:
            detected_emu, _ = detect_emulator_signatures(fp)
            if detected_emu:
                is_emu = True

        entropy_score = calculate_fingerprint_entropy(fp) if fp is not None else 0.0

        profile = self.tracker.get_profile(device_id)
        seen_count = float(profile.transaction_count) if profile is not None else 1.0

        return {
            "device_seen_count": seen_count,
            "device_distinct_cards_24h": float(report.distinct_cards_count),
            "device_card_distinct_devices_24h": float(report.distinct_devices_count),
            "device_is_emulator": 1.0 if is_emu else 0.0,
            "device_entropy_score": float(entropy_score),
            "device_is_card_cycling": 1.0 if report.is_card_cycling else 0.0,
            "device_is_device_cycling": 1.0 if report.is_device_cycling else 0.0,
        }

    def extract_batch(
        self,
        records: Sequence[Mapping[str, Any]],
        mutate_tracker: bool = True,
    ) -> list[dict[str, float]]:
        """Extract structured device features for a sequence of transaction records."""
        return [self.extract_features(r, mutate_tracker=mutate_tracker) for r in records]
