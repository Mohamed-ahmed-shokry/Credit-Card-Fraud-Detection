"""Real-time geographic geolocation anomaly detection, speed-of-travel velocity,
and IP geocoding.
"""

from __future__ import annotations


class GeoLocationError(ValueError):
    """Raised when coordinates, trajectory events, or geolocation configurations are invalid."""
