"""Real-time geographic geolocation anomaly detection, speed-of-travel velocity,
and IP geocoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


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
