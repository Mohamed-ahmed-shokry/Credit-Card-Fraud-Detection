"""Real-time geographic geolocation anomaly detection, speed-of-travel velocity,
and IP geocoding.
"""

from __future__ import annotations

import ipaddress
import math
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any


class GeoLocationError(ValueError):
    """Raised when coordinates, trajectory events, or geolocation configurations are invalid."""


@dataclass(frozen=True)
class GeoCoordinates:
    """Geographic point coordinates with optional civic location metadata."""

    latitude: float
    longitude: float
    accuracy_radius_km: float = 0.0
    country_code: str | None = None
    city: str | None = None
    postal_code: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.latitude, (int, float))
            or math.isnan(self.latitude)
            or math.isinf(self.latitude)
            or not (-90.0 <= self.latitude <= 90.0)
        ):
            raise GeoLocationError("latitude must be a finite float between -90.0 and 90.0.")
        if (
            not isinstance(self.longitude, (int, float))
            or math.isnan(self.longitude)
            or math.isinf(self.longitude)
            or not (-180.0 <= self.longitude <= 180.0)
        ):
            raise GeoLocationError("longitude must be a finite float between -180.0 and 180.0.")
        if (
            not isinstance(self.accuracy_radius_km, (int, float))
            or math.isnan(self.accuracy_radius_km)
            or math.isinf(self.accuracy_radius_km)
            or self.accuracy_radius_km < 0.0
        ):
            raise GeoLocationError("accuracy_radius_km must be a non-negative finite float.")
        if self.country_code is not None:
            if (
                not isinstance(self.country_code, str)
                or len(self.country_code) != 2
                or not self.country_code.isalpha()
            ):
                raise GeoLocationError("country_code must be an ISO 3166-1 alpha-2 2-letter code.")
            object.__setattr__(self, "country_code", self.country_code.upper())
        if self.city is not None and (not isinstance(self.city, str) or not self.city.strip()):
            raise GeoLocationError("city must be a non-empty string or None.")
        if self.postal_code is not None and (
            not isinstance(self.postal_code, str) or not self.postal_code.strip()
        ):
            raise GeoLocationError("postal_code must be a non-empty string or None.")


EARTH_RADIUS_KM: float = 6371.0088


def haversine_distance(coord1: GeoCoordinates, coord2: GeoCoordinates) -> float:
    """Calculate the great-circle distance between two geographic coordinates in kilometers.

    Uses the spherical law of haversines with WGS 84 mean Earth radius (6,371.0088 km).
    Numerically bounded to prevent domain errors on antipodal or identical coordinates.
    """
    if not isinstance(coord1, GeoCoordinates) or not isinstance(coord2, GeoCoordinates):
        raise GeoLocationError("coord1 and coord2 must be instances of GeoCoordinates.")

    if coord1.latitude == coord2.latitude and coord1.longitude == coord2.longitude:
        return 0.0

    lat1_rad = math.radians(coord1.latitude)
    lat2_rad = math.radians(coord2.latitude)
    dlat_rad = math.radians(coord2.latitude - coord1.latitude)
    dlon_rad = math.radians(coord2.longitude - coord1.longitude)

    sin_half_dlat = math.sin(dlat_rad / 2.0)
    sin_half_dlon = math.sin(dlon_rad / 2.0)

    a = sin_half_dlat * sin_half_dlat + math.cos(lat1_rad) * math.cos(lat2_rad) * (
        sin_half_dlon * sin_half_dlon
    )
    # Clip numerical precision drift to strictly valid domain [0.0, 1.0]
    a = max(0.0, min(1.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))

    return EARTH_RADIUS_KM * c


@dataclass(frozen=True)
class GeoLocationEvent:
    """Represents a timestamped geolocation event tied to a transaction or entity."""

    event_id: str
    entity_id: str
    timestamp: float
    coordinates: GeoCoordinates
    ip_address: str | None = None
    asn: int | None = None
    is_proxy: bool | None = None
    is_vpn: bool | None = None
    is_tor: bool | None = None

    def __post_init__(self) -> None:
        if not self.event_id or not isinstance(self.event_id, str):
            raise GeoLocationError("event_id must be a non-empty string.")
        if not self.entity_id or not isinstance(self.entity_id, str):
            raise GeoLocationError("entity_id must be a non-empty string.")
        if (
            not isinstance(self.timestamp, (int, float))
            or math.isnan(self.timestamp)
            or math.isinf(self.timestamp)
        ):
            raise GeoLocationError("timestamp must be a finite float.")
        if not isinstance(self.coordinates, GeoCoordinates):
            raise GeoLocationError("coordinates must be an instance of GeoCoordinates.")
        if self.ip_address is not None and (
            not isinstance(self.ip_address, str) or not self.ip_address.strip()
        ):
            raise GeoLocationError("ip_address must be a non-empty string or None.")
        if self.asn is not None and (
            not isinstance(self.asn, int) or isinstance(self.asn, bool) or self.asn < 0
        ):
            raise GeoLocationError("asn must be a non-negative integer or None.")
        if self.is_proxy is not None and not isinstance(self.is_proxy, bool):
            raise GeoLocationError("is_proxy must be a boolean or None.")
        if self.is_vpn is not None and not isinstance(self.is_vpn, bool):
            raise GeoLocationError("is_vpn must be a boolean or None.")
        if self.is_tor is not None and not isinstance(self.is_tor, bool):
            raise GeoLocationError("is_tor must be a boolean or None.")


DEFAULT_MAX_SPEED_KMH: float = 900.0


@dataclass(frozen=True)
class TravelVelocity:
    """Calculated velocity, geodesic displacement, and impossible travel assessment
    between two events.
    """

    distance_km: float
    time_elapsed_seconds: float
    velocity_kmh: float
    is_impossible_travel: bool
    is_cross_border: bool
    origin_country: str | None = None
    destination_country: str | None = None
    origin_city: str | None = None
    destination_city: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.distance_km, (int, float))
            or math.isnan(self.distance_km)
            or math.isinf(self.distance_km)
            or self.distance_km < 0.0
        ):
            raise GeoLocationError("distance_km must be a non-negative finite float.")
        if (
            not isinstance(self.time_elapsed_seconds, (int, float))
            or math.isnan(self.time_elapsed_seconds)
            or math.isinf(self.time_elapsed_seconds)
            or self.time_elapsed_seconds < 0.0
        ):
            raise GeoLocationError("time_elapsed_seconds must be a non-negative finite float.")
        if (
            not isinstance(self.velocity_kmh, (int, float))
            or math.isnan(self.velocity_kmh)
            or math.isinf(self.velocity_kmh)
            or self.velocity_kmh < 0.0
        ):
            raise GeoLocationError("velocity_kmh must be a non-negative finite float.")
        if not isinstance(self.is_impossible_travel, bool):
            raise GeoLocationError("is_impossible_travel must be a boolean.")
        if not isinstance(self.is_cross_border, bool):
            raise GeoLocationError("is_cross_border must be a boolean.")


def calculate_travel_velocity(
    prev_event: GeoLocationEvent,
    curr_event: GeoLocationEvent,
    max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH,
) -> TravelVelocity:
    """Compute displacement, elapsed time, and speed of travel between two sequential events.

    Raises GeoLocationError if curr_event.timestamp < prev_event.timestamp.
    """
    if not isinstance(prev_event, GeoLocationEvent) or not isinstance(curr_event, GeoLocationEvent):
        raise GeoLocationError("prev_event and curr_event must be instances of GeoLocationEvent.")

    if curr_event.timestamp < prev_event.timestamp:
        raise GeoLocationError(
            f"Temporal causality violation: curr_event timestamp ({curr_event.timestamp}) "
            f"is earlier than prev_event timestamp ({prev_event.timestamp})."
        )

    if (
        not isinstance(max_speed_kmh, (int, float))
        or math.isnan(max_speed_kmh)
        or math.isinf(max_speed_kmh)
        or max_speed_kmh <= 0.0
    ):
        raise GeoLocationError("max_speed_kmh must be a positive finite float.")

    distance = haversine_distance(prev_event.coordinates, curr_event.coordinates)
    time_elapsed = float(curr_event.timestamp - prev_event.timestamp)

    if time_elapsed <= 0.0:
        velocity = 999_999.0 if distance > 0.0 else 0.0
        is_impossible = distance > 0.0
    else:
        hours = time_elapsed / 3600.0
        velocity = distance / hours
        is_impossible = velocity > max_speed_kmh

    origin_country = prev_event.coordinates.country_code
    dest_country = curr_event.coordinates.country_code
    is_cross_border = (
        origin_country is not None and dest_country is not None and origin_country != dest_country
    )

    return TravelVelocity(
        distance_km=distance,
        time_elapsed_seconds=time_elapsed,
        velocity_kmh=velocity,
        is_impossible_travel=is_impossible,
        is_cross_border=is_cross_border,
        origin_country=origin_country,
        destination_country=dest_country,
        origin_city=prev_event.coordinates.city,
        destination_city=curr_event.coordinates.city,
    )


@dataclass(frozen=True)
class IPGeoResult:
    """Resolved geolocation and network risk properties of an IP address."""

    ip_address: str
    is_private: bool
    coordinates: GeoCoordinates
    asn: int | None = None
    is_proxy: bool = False
    is_vpn: bool = False
    is_tor: bool = False

    def __post_init__(self) -> None:
        if not self.ip_address or not isinstance(self.ip_address, str):
            raise GeoLocationError("ip_address must be a non-empty string.")
        if not isinstance(self.is_private, bool):
            raise GeoLocationError("is_private must be a boolean.")
        if not isinstance(self.coordinates, GeoCoordinates):
            raise GeoLocationError("coordinates must be an instance of GeoCoordinates.")
        if self.asn is not None and (
            not isinstance(self.asn, int) or isinstance(self.asn, bool) or self.asn < 0
        ):
            raise GeoLocationError("asn must be a non-negative integer or None.")
        if not isinstance(self.is_proxy, bool):
            raise GeoLocationError("is_proxy must be a boolean.")
        if not isinstance(self.is_vpn, bool):
            raise GeoLocationError("is_vpn must be a boolean.")
        if not isinstance(self.is_tor, bool):
            raise GeoLocationError("is_tor must be a boolean.")


@dataclass(frozen=True)
class SubnetRecord:
    """Internal representation of a registered IP network range."""

    network: ipaddress.IPv4Network | ipaddress.IPv6Network
    coordinates: GeoCoordinates
    asn: int | None = None
    is_proxy: bool = False
    is_vpn: bool = False
    is_tor: bool = False


class IPGeoResolver:
    """Zero-dependency streaming IP geocoder and proxy/VPN risk classifier.

    Uses longest-prefix subnet matching against registered CIDR ranges with thread-safe updates.
    """

    def __init__(self, include_defaults: bool = True) -> None:
        self._lock = threading.RLock()
        self._subnets: list[SubnetRecord] = []
        if include_defaults:
            self._load_reference_subnets()

    def register_subnet(
        self,
        cidr: str,
        coordinates: GeoCoordinates,
        asn: int | None = None,
        is_proxy: bool = False,
        is_vpn: bool = False,
        is_tor: bool = False,
    ) -> None:
        """Register a CIDR block with designated coordinates and risk flags."""
        if not isinstance(cidr, str) or not cidr.strip():
            raise GeoLocationError("cidr must be a non-empty string.")
        if not isinstance(coordinates, GeoCoordinates):
            raise GeoLocationError("coordinates must be an instance of GeoCoordinates.")
        try:
            net = ipaddress.ip_network(cidr.strip(), strict=False)
        except ValueError as exc:
            raise GeoLocationError(f"Invalid CIDR notation {cidr!r}: {exc}") from exc

        rec = SubnetRecord(
            network=net,
            coordinates=coordinates,
            asn=asn,
            is_proxy=is_proxy,
            is_vpn=is_vpn,
            is_tor=is_tor,
        )
        with self._lock:
            self._subnets.append(rec)
            self._subnets.sort(key=lambda r: r.network.prefixlen, reverse=True)

    def resolve(self, ip_str: str) -> IPGeoResult:
        """Resolve an IPv4 or IPv6 address string to coordinates and risk classification."""
        if not isinstance(ip_str, str) or not ip_str.strip():
            raise GeoLocationError("ip_str must be a non-empty string.")
        try:
            ip = ipaddress.ip_address(ip_str.strip())
        except ValueError as exc:
            raise GeoLocationError(f"Invalid IP address {ip_str!r}: {exc}") from exc

        with self._lock:
            for rec in self._subnets:
                if ip in rec.network:
                    return IPGeoResult(
                        ip_address=str(ip),
                        is_private=ip.is_private or ip.is_loopback or ip.is_link_local,
                        coordinates=rec.coordinates,
                        asn=rec.asn,
                        is_proxy=rec.is_proxy,
                        is_vpn=rec.is_vpn,
                        is_tor=rec.is_tor,
                    )

        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return IPGeoResult(
                ip_address=str(ip),
                is_private=True,
                coordinates=GeoCoordinates(
                    latitude=0.0,
                    longitude=0.0,
                    country_code="ZZ",
                    city="Private Network",
                ),
                asn=None,
                is_proxy=False,
                is_vpn=False,
                is_tor=False,
            )

        return IPGeoResult(
            ip_address=str(ip),
            is_private=False,
            coordinates=GeoCoordinates(
                latitude=0.0,
                longitude=0.0,
                country_code="XX",
                city="Unknown Public",
            ),
            asn=None,
            is_proxy=False,
            is_vpn=False,
            is_tor=False,
        )

    def _load_reference_subnets(self) -> None:
        """Seed reference test/demo subnets covering key international financial centers."""
        self.register_subnet(
            "185.86.151.0/24",
            GeoCoordinates(51.5074, -0.1278, country_code="GB", city="London"),
            asn=54113,
        )
        self.register_subnet(
            "198.51.100.0/24",
            GeoCoordinates(40.7128, -74.0060, country_code="US", city="New York"),
            asn=13335,
        )
        self.register_subnet(
            "194.109.6.0/24",
            GeoCoordinates(50.1109, 8.6821, country_code="DE", city="Frankfurt"),
            asn=3265,
        )
        self.register_subnet(
            "203.0.113.0/24",
            GeoCoordinates(35.6762, 139.6503, country_code="JP", city="Tokyo"),
            asn=2516,
        )
        self.register_subnet(
            "139.130.4.0/24",
            GeoCoordinates(-33.8688, 151.2093, country_code="AU", city="Sydney"),
            asn=1221,
        )
        self.register_subnet(
            "192.0.2.0/26",
            GeoCoordinates(37.7749, -122.4194, country_code="US", city="San Francisco"),
            asn=9009,
            is_proxy=True,
            is_vpn=True,
        )
        self.register_subnet(
            "192.0.2.64/28",
            GeoCoordinates(52.3676, 4.9041, country_code="NL", city="Amsterdam"),
            asn=60729,
            is_proxy=True,
            is_tor=True,
        )


DEFAULT_GEO_WINDOW_SECONDS: float = 86400.0  # 24 hours
DEFAULT_MAX_ENTITIES: int = 100_000


class GeoLocationTracker:
    """Thread-safe streaming geolocation trajectory tracker with sliding temporal eviction."""

    def __init__(
        self,
        window_seconds: float = DEFAULT_GEO_WINDOW_SECONDS,
        max_entities: int = DEFAULT_MAX_ENTITIES,
        max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH,
    ) -> None:
        if (
            not isinstance(window_seconds, (int, float))
            or math.isnan(window_seconds)
            or math.isinf(window_seconds)
            or window_seconds <= 0.0
        ):
            raise GeoLocationError("window_seconds must be a positive finite float.")
        if not isinstance(max_entities, int) or isinstance(max_entities, bool) or max_entities <= 0:
            raise GeoLocationError("max_entities must be a positive integer.")
        if (
            not isinstance(max_speed_kmh, (int, float))
            or math.isnan(max_speed_kmh)
            or math.isinf(max_speed_kmh)
            or max_speed_kmh <= 0.0
        ):
            raise GeoLocationError("max_speed_kmh must be a positive finite float.")

        self.window_seconds = float(window_seconds)
        self.max_entities = max_entities
        self.max_speed_kmh = float(max_speed_kmh)

        self._lock = threading.RLock()
        self._trajectories: OrderedDict[str, deque[GeoLocationEvent]] = OrderedDict()

    def record_event(self, event: GeoLocationEvent) -> TravelVelocity | None:
        """Record an event into the entity trajectory and return velocity from previous event."""
        if not isinstance(event, GeoLocationEvent):
            raise GeoLocationError("event must be an instance of GeoLocationEvent.")

        with self._lock:
            history = self._trajectories.get(event.entity_id)
            if history is None:
                if len(self._trajectories) >= self.max_entities:
                    self._trajectories.popitem(last=False)
                history = deque()
                self._trajectories[event.entity_id] = history
            else:
                self._trajectories.move_to_end(event.entity_id)

            prev_event: GeoLocationEvent | None = None
            if history:
                prev_event = history[-1]

            velocity: TravelVelocity | None = None
            if prev_event is not None:
                velocity = calculate_travel_velocity(
                    prev_event=prev_event,
                    curr_event=event,
                    max_speed_kmh=self.max_speed_kmh,
                )

            history.append(event)
            self._trim_entity_history(history, event.timestamp)
            return velocity

    def get_trajectory(self, entity_id: str) -> list[GeoLocationEvent]:
        """Return the sequence of events recorded for the entity within the sliding window."""
        if not isinstance(entity_id, str) or not entity_id.strip():
            raise GeoLocationError("entity_id must be a non-empty string.")

        with self._lock:
            history = self._trajectories.get(entity_id)
            if not history:
                return []
            return list(history)

    def get_last_known_event(self, entity_id: str) -> GeoLocationEvent | None:
        """Return the most recent recorded event for the entity, or None if unknown."""
        if not isinstance(entity_id, str) or not entity_id.strip():
            raise GeoLocationError("entity_id must be a non-empty string.")

        with self._lock:
            history = self._trajectories.get(entity_id)
            if not history:
                return None
            return history[-1]

    def assess_travel(
        self,
        entity_id: str,
        candidate_event: GeoLocationEvent,
    ) -> TravelVelocity | None:
        """Compute prospective velocity against last event without mutating trajectory state."""
        if not isinstance(candidate_event, GeoLocationEvent):
            raise GeoLocationError("candidate_event must be an instance of GeoLocationEvent.")

        with self._lock:
            last_event = self.get_last_known_event(entity_id)
            if last_event is None:
                return None
            return calculate_travel_velocity(
                prev_event=last_event,
                curr_event=candidate_event,
                max_speed_kmh=self.max_speed_kmh,
            )

    def evict_expired(self, current_timestamp: float) -> int:
        """Purge entries older than the sliding window relative to current_timestamp."""
        if (
            not isinstance(current_timestamp, (int, float))
            or math.isnan(current_timestamp)
            or math.isinf(current_timestamp)
        ):
            raise GeoLocationError("current_timestamp must be a finite float.")

        evicted_count = 0
        cutoff = current_timestamp - self.window_seconds
        with self._lock:
            expired_keys: list[str] = []
            for entity_id, history in self._trajectories.items():
                before_len = len(history)
                while history and history[0].timestamp < cutoff:
                    history.popleft()
                evicted_count += before_len - len(history)
                if not history:
                    expired_keys.append(entity_id)
            for k in expired_keys:
                del self._trajectories[k]
        return evicted_count

    def reset(self) -> None:
        """Clear all tracked trajectories."""
        with self._lock:
            self._trajectories.clear()

    def stats(self) -> dict[str, Any]:
        """Return operational capacity and usage telemetry."""
        with self._lock:
            total_events = sum(len(h) for h in self._trajectories.values())
            return {
                "total_entities": len(self._trajectories),
                "total_events": total_events,
                "window_seconds": self.window_seconds,
                "max_entities": self.max_entities,
                "max_speed_kmh": self.max_speed_kmh,
            }

    def _trim_entity_history(self, history: deque[GeoLocationEvent], current_ts: float) -> None:
        cutoff = current_ts - self.window_seconds
        while history and history[0].timestamp < cutoff:
            history.popleft()
