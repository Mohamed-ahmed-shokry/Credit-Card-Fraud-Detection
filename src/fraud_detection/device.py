"""Real-time adaptive device and browser fingerprinting with anomaly clustering."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence, Set
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
    set1 = set(items1)
    set2 = set(items2)
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union == 0:
        return 1.0
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
