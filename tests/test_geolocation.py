"""Unit and integration tests for geolocation anomaly detection and speed-of-travel engine."""

from __future__ import annotations

import pytest

from fraud_detection.geolocation import GeoCoordinates, GeoLocationError, haversine_distance


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


def test_haversine_distance_identical_points() -> None:
    pt = GeoCoordinates(latitude=40.7128, longitude=-74.0060)
    assert haversine_distance(pt, pt) == 0.0


def test_haversine_distance_known_routes() -> None:
    # London (51.5074, -0.1278) to New York (40.7128, -74.0060) ~ 5570 km
    london = GeoCoordinates(latitude=51.5074, longitude=-0.1278)
    nyc = GeoCoordinates(latitude=40.7128, longitude=-74.0060)
    dist_lon_nyc = haversine_distance(london, nyc)
    assert 5560.0 <= dist_lon_nyc <= 5585.0

    # Symmetry check: d(A, B) == d(B, A)
    assert haversine_distance(nyc, london) == pytest.approx(dist_lon_nyc, rel=1e-6)

    # North Pole to South Pole ~ 20015 km
    north_pole = GeoCoordinates(latitude=90.0, longitude=0.0)
    south_pole = GeoCoordinates(latitude=-90.0, longitude=0.0)
    dist_poles = haversine_distance(north_pole, south_pole)
    assert 20000.0 <= dist_poles <= 20030.0

    # Antipodal points on equator
    eq1 = GeoCoordinates(latitude=0.0, longitude=0.0)
    eq2 = GeoCoordinates(latitude=0.0, longitude=180.0)
    dist_anti = haversine_distance(eq1, eq2)
    assert 20000.0 <= dist_anti <= 20030.0


def test_haversine_distance_invalid_type() -> None:
    pt = GeoCoordinates(latitude=0.0, longitude=0.0)
    with pytest.raises(GeoLocationError, match="must be instances of GeoCoordinates"):
        haversine_distance(pt, "not-a-coord")  # type: ignore[arg-type]
