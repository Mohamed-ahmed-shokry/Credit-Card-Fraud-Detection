"""Unit and integration tests for geolocation anomaly detection and speed-of-travel engine."""

from __future__ import annotations

from typing import Any

import pytest

from fraud_detection.geolocation import (
    DEFAULT_GEO_WINDOW_SECONDS,
    DEFAULT_MAX_ENTITIES,
    DEFAULT_MAX_SPEED_KMH,
    GeoCoordinates,
    GeoLocationError,
    GeoLocationEvent,
    GeoLocationTracker,
    IPGeoResolver,
    IPGeoResult,
    TravelVelocity,
    calculate_travel_velocity,
    haversine_distance,
)


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


def test_geo_location_event_valid() -> None:
    coord = GeoCoordinates(latitude=51.5074, longitude=-0.1278, country_code="GB")
    event = GeoLocationEvent(
        event_id="evt-101",
        entity_id="card-999",
        timestamp=1000.0,
        coordinates=coord,
        ip_address="198.51.100.1",
        asn=13335,
        is_proxy=False,
        is_vpn=False,
        is_tor=False,
    )
    assert event.event_id == "evt-101"
    assert event.entity_id == "card-999"
    assert event.timestamp == 1000.0
    assert event.coordinates == coord
    assert event.ip_address == "198.51.100.1"
    assert event.asn == 13335
    assert event.is_proxy is False
    assert event.is_vpn is False
    assert event.is_tor is False


@pytest.mark.parametrize(
    ("evt_id", "ent_id", "ts", "coords", "ip", "asn", "proxy", "vpn", "tor", "err_msg"),
    [
        ("", "e1", 100.0, GeoCoordinates(0.0, 0.0), None, None, None, None, None, "event_id"),
        ("ev1", "", 100.0, GeoCoordinates(0.0, 0.0), None, None, None, None, None, "entity_id"),
        (
            "ev1",
            "e1",
            float("nan"),
            GeoCoordinates(0.0, 0.0),
            None,
            None,
            None,
            None,
            None,
            "timestamp",
        ),
        ("ev1", "e1", 100.0, "not_coord", None, None, None, None, None, "coordinates"),
        ("ev1", "e1", 100.0, GeoCoordinates(0.0, 0.0), "", None, None, None, None, "ip_address"),
        ("ev1", "e1", 100.0, GeoCoordinates(0.0, 0.0), None, -5, None, None, None, "asn"),
        (
            "ev1",
            "e1",
            100.0,
            GeoCoordinates(0.0, 0.0),
            None,
            None,
            "not_bool",
            None,
            None,
            "is_proxy",
        ),
        (
            "ev1",
            "e1",
            100.0,
            GeoCoordinates(0.0, 0.0),
            None,
            None,
            None,
            "not_bool",
            None,
            "is_vpn",
        ),
        (
            "ev1",
            "e1",
            100.0,
            GeoCoordinates(0.0, 0.0),
            None,
            None,
            None,
            None,
            "not_bool",
            "is_tor",
        ),
    ],
)
def test_geo_location_event_invalid(
    evt_id: str,
    ent_id: str,
    ts: float,
    coords: Any,
    ip: str | None,
    asn: int | None,
    proxy: Any,
    vpn: Any,
    tor: Any,
    err_msg: str,
) -> None:
    with pytest.raises(GeoLocationError, match=err_msg):
        GeoLocationEvent(
            event_id=evt_id,
            entity_id=ent_id,
            timestamp=ts,
            coordinates=coords,
            ip_address=ip,
            asn=asn,
            is_proxy=proxy,
            is_vpn=vpn,
            is_tor=tor,
        )


def test_calculate_travel_velocity_normal() -> None:
    assert DEFAULT_MAX_SPEED_KMH == 900.0
    # 50 km in 1 hour (3600 seconds)
    c1 = GeoCoordinates(latitude=51.5074, longitude=-0.1278, country_code="GB", city="London")
    # Point ~50km away
    c2 = GeoCoordinates(latitude=51.8500, longitude=-0.1278, country_code="GB", city="Luton")
    ev1 = GeoLocationEvent("ev1", "card-1", 1000.0, c1)
    ev2 = GeoLocationEvent("ev2", "card-1", 4600.0, c2)

    tv = calculate_travel_velocity(ev1, ev2)
    assert tv.time_elapsed_seconds == 3600.0
    assert 35.0 <= tv.distance_km <= 45.0
    assert tv.velocity_kmh == pytest.approx(tv.distance_km, rel=1e-3)
    assert tv.is_impossible_travel is False
    assert tv.is_cross_border is False
    assert tv.origin_country == "GB"
    assert tv.destination_country == "GB"
    assert tv.origin_city == "London"
    assert tv.destination_city == "Luton"


def test_calculate_travel_velocity_impossible() -> None:
    # London to NYC in 1800 seconds (30 minutes)
    c_lon = GeoCoordinates(latitude=51.5074, longitude=-0.1278, country_code="GB", city="London")
    c_nyc = GeoCoordinates(latitude=40.7128, longitude=-74.0060, country_code="US", city="New York")
    ev1 = GeoLocationEvent("ev1", "card-1", 1000.0, c_lon)
    ev2 = GeoLocationEvent("ev2", "card-1", 2800.0, c_nyc)

    tv = calculate_travel_velocity(ev1, ev2)
    assert tv.time_elapsed_seconds == 1800.0
    assert 5560.0 <= tv.distance_km <= 5585.0
    # ~5570 km in 0.5h = ~11140 km/h > 900 km/h
    assert tv.velocity_kmh > 900.0
    assert tv.is_impossible_travel is True
    assert tv.is_cross_border is True
    assert tv.origin_country == "GB"
    assert tv.destination_country == "US"


def test_calculate_travel_velocity_zero_time_elapsed() -> None:
    c1 = GeoCoordinates(latitude=51.5074, longitude=-0.1278)
    c2 = GeoCoordinates(latitude=40.7128, longitude=-74.0060)
    ev1 = GeoLocationEvent("ev1", "card-1", 1000.0, c1)
    ev2 = GeoLocationEvent("ev2", "card-1", 1000.0, c2)

    # Simultaneous event at different location = instantaneous teleportation
    tv = calculate_travel_velocity(ev1, ev2)
    assert tv.time_elapsed_seconds == 0.0
    assert tv.is_impossible_travel is True

    # Simultaneous event at identical location = valid 0 velocity
    ev3 = GeoLocationEvent("ev3", "card-1", 1000.0, c1)
    tv_same = calculate_travel_velocity(ev1, ev3)
    assert tv_same.distance_km == 0.0
    assert tv_same.velocity_kmh == 0.0
    assert tv_same.is_impossible_travel is False


def test_calculate_travel_velocity_causality_violation() -> None:
    c1 = GeoCoordinates(latitude=0.0, longitude=0.0)
    ev1 = GeoLocationEvent("ev1", "card-1", 2000.0, c1)
    ev2 = GeoLocationEvent("ev2", "card-1", 1000.0, c1)

    with pytest.raises(GeoLocationError, match="Temporal causality violation"):
        calculate_travel_velocity(ev1, ev2)


def test_calculate_travel_velocity_invalid_inputs() -> None:
    c1 = GeoCoordinates(latitude=0.0, longitude=0.0)
    ev1 = GeoLocationEvent("ev1", "card-1", 1000.0, c1)

    with pytest.raises(GeoLocationError, match="must be instances of GeoLocationEvent"):
        calculate_travel_velocity(ev1, "not-an-event")  # type: ignore[arg-type]

    with pytest.raises(GeoLocationError, match="max_speed_kmh must be a positive finite float"):
        calculate_travel_velocity(ev1, ev1, max_speed_kmh=-10.0)


@pytest.mark.parametrize(
    ("dist", "t_elapsed", "vel", "impossible", "cross_border", "err_msg"),
    [
        (-1.0, 10.0, 0.0, False, False, "distance_km must be a non-negative"),
        (10.0, -5.0, 0.0, False, False, "time_elapsed_seconds must be a non-negative"),
        (10.0, 10.0, -1.0, False, False, "velocity_kmh must be a non-negative"),
        (10.0, 10.0, 10.0, "not_bool", False, "is_impossible_travel must be a boolean"),
        (10.0, 10.0, 10.0, False, "not_bool", "is_cross_border must be a boolean"),
    ],
)
def test_travel_velocity_validation_errors(
    dist: float,
    t_elapsed: float,
    vel: float,
    impossible: Any,
    cross_border: Any,
    err_msg: str,
) -> None:
    with pytest.raises(GeoLocationError, match=err_msg):
        TravelVelocity(
            distance_km=dist,
            time_elapsed_seconds=t_elapsed,
            velocity_kmh=vel,
            is_impossible_travel=impossible,
            is_cross_border=cross_border,
        )


def test_ip_geo_resolver_private_ips() -> None:
    resolver = IPGeoResolver()
    for ip in ["127.0.0.1", "10.1.2.3", "172.16.50.1", "192.168.1.100", "::1"]:
        res = resolver.resolve(ip)
        assert res.is_private is True
        assert res.coordinates.country_code == "ZZ"
        assert res.asn is None
        assert res.is_proxy is False


def test_ip_geo_resolver_reference_subnets() -> None:
    resolver = IPGeoResolver()

    # London
    res_lon = resolver.resolve("185.86.151.42")
    assert res_lon.is_private is False
    assert res_lon.coordinates.country_code == "GB"
    assert res_lon.coordinates.city == "London"
    assert res_lon.asn == 54113
    assert res_lon.is_proxy is False

    # New York
    res_nyc = resolver.resolve("198.51.100.12")
    assert res_nyc.coordinates.country_code == "US"
    assert res_nyc.coordinates.city == "New York"
    assert res_nyc.asn == 13335

    # Proxy / VPN
    res_vpn = resolver.resolve("192.0.2.15")
    assert res_vpn.is_proxy is True
    assert res_vpn.is_vpn is True
    assert res_vpn.is_tor is False

    # Tor exit node
    res_tor = resolver.resolve("192.0.2.70")
    assert res_tor.is_proxy is True
    assert res_tor.is_tor is True
    assert res_tor.is_vpn is False

    # Unknown public IP
    res_unk = resolver.resolve("8.8.8.8")
    assert res_unk.is_private is False
    assert res_unk.coordinates.country_code == "XX"


def test_ip_geo_resolver_custom_registration_and_errors() -> None:
    resolver = IPGeoResolver(include_defaults=False)
    custom_coord = GeoCoordinates(48.8566, 2.3522, country_code="FR", city="Paris")
    resolver.register_subnet("195.154.0.0/16", custom_coord, asn=12876)

    res = resolver.resolve("195.154.20.10")
    assert res.coordinates.city == "Paris"
    assert res.coordinates.country_code == "FR"
    assert res.asn == 12876

    with pytest.raises(GeoLocationError, match="cidr must be a non-empty string"):
        resolver.register_subnet("", custom_coord)

    with pytest.raises(GeoLocationError, match="coordinates must be an instance of GeoCoordinates"):
        resolver.register_subnet("1.2.3.0/24", "invalid")  # type: ignore[arg-type]

    with pytest.raises(GeoLocationError, match="Invalid CIDR notation"):
        resolver.register_subnet("invalid-cidr", custom_coord)

    with pytest.raises(GeoLocationError, match="ip_str must be a non-empty string"):
        resolver.resolve("")

    with pytest.raises(GeoLocationError, match="Invalid IP address"):
        resolver.resolve("999.999.999.999")


def test_ip_geo_result_validation() -> None:
    c = GeoCoordinates(0.0, 0.0)
    with pytest.raises(GeoLocationError, match="ip_address must be a non-empty string"):
        IPGeoResult("", False, c)
    with pytest.raises(GeoLocationError, match="is_private must be a boolean"):
        IPGeoResult("1.1.1.1", "not_bool", c)  # type: ignore[arg-type]
    with pytest.raises(GeoLocationError, match="coordinates must be an instance of GeoCoordinates"):
        IPGeoResult("1.1.1.1", False, "not_coord")  # type: ignore[arg-type]
    with pytest.raises(GeoLocationError, match="asn must be a non-negative integer"):
        IPGeoResult("1.1.1.1", False, c, asn=-10)
    with pytest.raises(GeoLocationError, match="is_proxy must be a boolean"):
        IPGeoResult("1.1.1.1", False, c, is_proxy="not_bool")  # type: ignore[arg-type]
    with pytest.raises(GeoLocationError, match="is_vpn must be a boolean"):
        IPGeoResult("1.1.1.1", False, c, is_vpn="not_bool")  # type: ignore[arg-type]
    with pytest.raises(GeoLocationError, match="is_tor must be a boolean"):
        IPGeoResult("1.1.1.1", False, c, is_tor="not_bool")  # type: ignore[arg-type]


def test_geo_location_tracker_lifecycle() -> None:
    assert DEFAULT_GEO_WINDOW_SECONDS == 86400.0
    assert DEFAULT_MAX_ENTITIES == 100_000

    tracker = GeoLocationTracker(window_seconds=3600.0, max_entities=10)
    c_lon = GeoCoordinates(51.5074, -0.1278, country_code="GB", city="London")
    c_nyc = GeoCoordinates(40.7128, -74.0060, country_code="US", city="New York")

    ev1 = GeoLocationEvent("ev1", "card-1", 1000.0, c_lon)
    vel1 = tracker.record_event(ev1)
    assert vel1 is None  # First event has no prior event

    ev2 = GeoLocationEvent("ev2", "card-1", 2000.0, c_nyc)
    vel2 = tracker.record_event(ev2)
    assert vel2 is not None
    assert vel2.is_impossible_travel is True
    assert vel2.time_elapsed_seconds == 1000.0

    # Test trajectory retrieval
    traj = tracker.get_trajectory("card-1")
    assert len(traj) == 2
    assert traj[0].event_id == "ev1"
    assert traj[1].event_id == "ev2"

    assert tracker.get_trajectory("unknown-card") == []

    # Test last known event
    last_ev = tracker.get_last_known_event("card-1")
    assert last_ev is not None
    assert last_ev.event_id == "ev2"
    assert tracker.get_last_known_event("unknown-card") is None

    # Test assess_travel without mutating
    c_sf = GeoCoordinates(37.7749, -122.4194, country_code="US", city="San Francisco")
    ev3_candidate = GeoLocationEvent("ev3", "card-1", 3000.0, c_sf)
    vel_assess = tracker.assess_travel("card-1", ev3_candidate)
    assert vel_assess is not None
    assert vel_assess.destination_city == "San Francisco"

    # Trajectory should still have only 2 events
    assert len(tracker.get_trajectory("card-1")) == 2

    assert tracker.assess_travel("unknown-card", ev3_candidate) is None

    # Test stats
    st = tracker.stats()
    assert st["total_entities"] == 1
    assert st["total_events"] == 2
    assert st["window_seconds"] == 3600.0

    # Test eviction
    evicted = tracker.evict_expired(current_timestamp=5000.0)
    assert evicted == 1  # ev1 (ts=1000.0 < 5000 - 3600 = 1400) evicted
    assert len(tracker.get_trajectory("card-1")) == 1

    tracker.reset()
    assert tracker.stats()["total_entities"] == 0
    assert tracker.stats()["total_events"] == 0


def test_geo_location_tracker_capacity_eviction() -> None:
    # Max entities = 2
    tracker = GeoLocationTracker(window_seconds=3600.0, max_entities=2)
    coord = GeoCoordinates(0.0, 0.0)

    tracker.record_event(GeoLocationEvent("ev1", "card-1", 100.0, coord))
    tracker.record_event(GeoLocationEvent("ev2", "card-2", 101.0, coord))
    assert tracker.stats()["total_entities"] == 2

    # Adding third entity evicts card-1
    tracker.record_event(GeoLocationEvent("ev3", "card-3", 102.0, coord))
    assert tracker.stats()["total_entities"] == 2
    assert tracker.get_trajectory("card-1") == []
    assert len(tracker.get_trajectory("card-2")) == 1
    assert len(tracker.get_trajectory("card-3")) == 1


def test_geo_location_tracker_validation_errors() -> None:
    with pytest.raises(GeoLocationError, match="window_seconds must be a positive finite float"):
        GeoLocationTracker(window_seconds=-1.0)
    with pytest.raises(GeoLocationError, match="max_entities must be a positive integer"):
        GeoLocationTracker(max_entities=0)
    with pytest.raises(GeoLocationError, match="max_speed_kmh must be a positive finite float"):
        GeoLocationTracker(max_speed_kmh=0.0)

    tracker = GeoLocationTracker()
    with pytest.raises(GeoLocationError, match="event must be an instance of GeoLocationEvent"):
        tracker.record_event("not_event")  # type: ignore[arg-type]

    with pytest.raises(GeoLocationError, match="entity_id must be a non-empty string"):
        tracker.get_trajectory("")

    with pytest.raises(GeoLocationError, match="entity_id must be a non-empty string"):
        tracker.get_last_known_event("")

    with pytest.raises(
        GeoLocationError, match="candidate_event must be an instance of GeoLocationEvent"
    ):
        tracker.assess_travel("c1", "not_event")  # type: ignore[arg-type]

    with pytest.raises(GeoLocationError, match="current_timestamp must be a finite float"):
        tracker.evict_expired(float("nan"))
