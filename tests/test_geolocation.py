"""Unit and integration tests for geolocation anomaly detection and speed-of-travel engine."""

from __future__ import annotations

import pytest

from fraud_detection.geolocation import GeoCoordinates, GeoLocationError


def test_geo_coordinates_valid() -> None:
    coord = GeoCoordinates(
        latitude=51.5074,
        longitude=-0.1278,
        accuracy_radius_km=0.5,
        country_code="gb",
        city="London",
        postal_code="SW1A",
    )
    assert coord.latitude == 51.5074
    assert coord.longitude == -0.1278
    assert coord.accuracy_radius_km == 0.5
    assert coord.country_code == "GB"
    assert coord.city == "London"
    assert coord.postal_code == "SW1A"


@pytest.mark.parametrize(
    ("lat", "lon", "radius", "country", "city", "postal", "err_msg"),
    [
        (91.0, 0.0, 0.0, None, None, None, "latitude must be a finite float"),
        (-90.1, 0.0, 0.0, None, None, None, "latitude must be a finite float"),
        (float("nan"), 0.0, 0.0, None, None, None, "latitude must be a finite float"),
        (0.0, 180.1, 0.0, None, None, None, "longitude must be a finite float"),
        (0.0, -180.1, 0.0, None, None, None, "longitude must be a finite float"),
        (0.0, 0.0, -1.0, None, None, None, "accuracy_radius_km must be a non-negative"),
        (0.0, 0.0, 0.0, "USA", None, None, "country_code must be an ISO 3166-1 alpha-2"),
        (0.0, 0.0, 0.0, "12", None, None, "country_code must be an ISO 3166-1 alpha-2"),
        (0.0, 0.0, 0.0, None, "", None, "city must be a non-empty string"),
        (0.0, 0.0, 0.0, None, None, "   ", "postal_code must be a non-empty string"),
    ],
)
def test_geo_coordinates_validation_errors(
    lat: float,
    lon: float,
    radius: float,
    country: str | None,
    city: str | None,
    postal: str | None,
    err_msg: str,
) -> None:
    with pytest.raises(GeoLocationError, match=err_msg):
        GeoCoordinates(
            latitude=lat,
            longitude=lon,
            accuracy_radius_km=radius,
            country_code=country,
            city=city,
            postal_code=postal,
        )
