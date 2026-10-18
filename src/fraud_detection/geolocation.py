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
