"""Unit and integration tests for geolocation anomaly detection and speed-of-travel engine."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from fraud_detection.geolocation import (
    DEFAULT_GEO_WINDOW_SECONDS,
    DEFAULT_MAX_ENTITIES,
    DEFAULT_MAX_SPEED_KMH,
    GEO_FEATURE_NAMES,
    GeoAnomalyReport,
    GeoCoordinates,
    GeoFeatureConfig,
    GeoFeatureExtractor,
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


def test_geo_feature_config_validation() -> None:
    cfg = GeoFeatureConfig()
    assert cfg.max_speed_kmh == 900.0
    assert cfg.entity_id_field == "card_id"

    with pytest.raises(GeoLocationError, match="max_speed_kmh must be a positive finite float"):
        GeoFeatureConfig(max_speed_kmh=-10.0)

    with pytest.raises(GeoLocationError, match="entity_id_field must be a non-empty string"):
        GeoFeatureConfig(entity_id_field="")


def test_geo_feature_extractor_streaming() -> None:
    extractor = GeoFeatureExtractor()

    # Record 1: London transaction via latitude/longitude
    r1 = {
        "transaction_id": "tx-1",
        "card_id": "card-100",
        "timestamp": 1000.0,
        "latitude": 51.5074,
        "longitude": -0.1278,
        "country_code": "GB",
        "billing_country": "GB",
    }
    f1 = extractor.extract_features(r1)
    assert f1["geo_distance_km"] == 0.0
    assert f1["geo_speed_kmh"] == 0.0
    assert f1["geo_is_impossible_travel"] == 0.0
    assert f1["geo_is_cross_border"] == 0.0
    assert f1["geo_is_proxy"] == 0.0
    assert f1["geo_country_mismatch"] == 0.0

    # Record 2: Impossible travel to NYC in 1800s via IP
    r2 = {
        "transaction_id": "tx-2",
        "card_id": "card-100",
        "timestamp": 2800.0,
        "ip_address": "198.51.100.25",  # Resolves to NYC (US)
        "billing_country": "GB",
    }
    f2 = extractor.extract_features(r2)
    assert 5560.0 <= f2["geo_distance_km"] <= 5585.0
    assert f2["geo_speed_kmh"] > 900.0
    assert f2["geo_is_impossible_travel"] == 1.0
    assert f2["geo_is_cross_border"] == 1.0
    assert f2["geo_country_mismatch"] == 1.0  # US transaction vs GB billing

    # Record 3: Proxy IP detection
    r3 = {
        "transaction_id": "tx-3",
        "card_id": "card-200",
        "timestamp": 3000.0,
        "ip_address": "192.0.2.10",  # Commercial VPN proxy
        "billing_country": "US",
    }
    f3 = extractor.extract_features(r3)
    assert f3["geo_is_proxy"] == 1.0


def test_geo_feature_extractor_batch_and_anomalies() -> None:
    assert len(GEO_FEATURE_NAMES) == 7
    extractor = GeoFeatureExtractor()

    records = [
        {
            "transaction_id": "tx-10",
            "card_id": "card-500",
            "timestamp": 1000.0,
            "latitude": 51.5074,
            "longitude": -0.1278,
            "billing_country": "GB",
        },
        {
            "transaction_id": "tx-11",
            "card_id": "card-500",
            "timestamp": 1600.0,  # 600 seconds later in NYC = impossible
            "latitude": 40.7128,
            "longitude": -74.0060,
            "billing_country": "GB",
        },
        {
            "transaction_id": "tx-12",
            "card_id": "card-600",
            "timestamp": 2000.0,
            "ip_address": "192.0.2.70",  # Tor exit node
            "billing_country": "NL",
        },
        {
            "transaction_id": "tx-13",
            "card_id": "",  # Missing entity ID
            "timestamp": 2100.0,
        },
    ]

    df = pd.DataFrame(records)
    enriched = extractor.extract_batch(df)
    assert len(enriched) == 4
    for feat in GEO_FEATURE_NAMES:
        assert feat in enriched.columns

    assert (
        enriched.loc[enriched["transaction_id"] == "tx-11", "geo_is_impossible_travel"].iloc[0]
        == 1.0
    )
    assert enriched.loc[enriched["transaction_id"] == "tx-12", "geo_is_proxy"].iloc[0] == 1.0

    # Test analyze_anomalies
    report = extractor.analyze_anomalies(records[:3])
    assert isinstance(report, GeoAnomalyReport)
    assert report.total_events == 3
    assert report.impossible_travel_events == 1
    assert report.proxy_events == 1
    assert "tx-11" in report.anomalous_event_ids
    assert "tx-12" in report.anomalous_event_ids


def test_geo_tracker_advanced_branches() -> None:
    tracker = GeoLocationTracker(window_seconds=100.0, max_entities=5)
    c1 = GeoCoordinates(51.5074, -0.1278)
    c2 = GeoCoordinates(40.7128, -74.0060)

    # 1. Insertion out of order (all within window_seconds=100)
    ev_first = GeoLocationEvent("ev1", "card-out-of-order", 650.0, c1)
    ev_third = GeoLocationEvent("ev3", "card-out-of-order", 700.0, c2)
    ev_second = GeoLocationEvent("ev2", "card-out-of-order", 680.0, c1)

    assert tracker.record_event(ev_first) is None
    assert tracker.record_event(ev_third) is not None
    # This triggers lines 540-545 (inserting event between existing events)
    tracker.record_event(ev_second)
    traj = tracker.get_trajectory("card-out-of-order")
    assert len(traj) == 3
    assert [e.event_id for e in traj] == ["ev1", "ev2", "ev3"]

    # 2. Check property and velocity inspection
    assert tracker.entity_count == 1
    assert tracker.get_stats()["total_events"] == 3
    latest_vel = tracker.assess_latest_velocity("card-out-of-order")
    assert latest_vel is not None
    vel_dict = latest_vel.to_dict()
    assert vel_dict["distance_km"] > 0.0

    ev_dict = ev_first.to_dict()
    assert ev_dict["event_id"] == "ev1"

    # 3. assess_travel when candidate timestamp is before all history events (prev_event is None)
    ev_early = GeoLocationEvent("ev0", "card-out-of-order", 100.0, c1)
    assert tracker.assess_travel("card-out-of-order", ev_early) is None
    cand_vel = tracker.assess_candidate_velocity(ev_early)
    assert cand_vel is None

    # 4. Validation errors
    with pytest.raises(GeoLocationError, match="candidate_event must be an instance"):
        tracker.assess_candidate_velocity("not-event")  # type: ignore[arg-type]

    with pytest.raises(GeoLocationError, match="entity_id must be a non-empty string"):
        tracker.assess_latest_velocity("")

    with pytest.raises(GeoLocationError, match="current_timestamp must be a finite float"):
        tracker.evict_expired(float("nan"))

    with pytest.raises(GeoLocationError, match="entity_id must be a non-empty string"):
        tracker.get_trajectory("   ")

    with pytest.raises(GeoLocationError, match="entity_id must be a non-empty string"):
        tracker.get_last_known_event("")

    # 5. Trimming and full eviction (lines 617-619 and 677)
    evict_tracker = GeoLocationTracker(window_seconds=50.0)
    evict_tracker.record_event(GeoLocationEvent("ev_old", "card_evict", 10.0, c1))
    evict_tracker.record_event(GeoLocationEvent("ev_new", "card_evict", 70.0, c2))
    assert len(evict_tracker.get_trajectory("card_evict")) == 1

    evicted = evict_tracker.evict_expired(current_timestamp=200.0)
    assert evicted == 1
    assert evict_tracker.entity_count == 0


def test_geo_feature_extractor_coverage_edge_cases() -> None:
    extractor = GeoFeatureExtractor()

    # 1. Invalid timestamp (NaN / infinite / string)
    res_nan = extractor.extract_features({"card_id": "c1", "timestamp": float("nan")})
    assert res_nan["geo_distance_km"] == 0.0

    res_str = extractor.extract_features({"card_id": "c1", "timestamp": "not-numeric"})
    assert res_str["geo_distance_km"] == 0.0

    res_no_ts = extractor.extract_features({"card_id": "c1"})
    assert res_no_ts["geo_distance_km"] == 0.0

    # 2. Coordinates error handling (latitude out of bounds)
    res_bad_lat = extractor.extract_features(
        {"card_id": "c1", "timestamp": 100.0, "latitude": 999.0, "longitude": 0.0}
    )
    assert res_bad_lat["geo_distance_km"] == 0.0

    # 3. IP resolver error handling
    res_bad_ip = extractor.extract_features(
        {"card_id": "c1", "timestamp": 100.0, "ip_address": "invalid-ip"}
    )
    assert res_bad_ip["geo_distance_km"] == 0.0

    # 4. Non-mutating feature extraction (mutate_tracker=False)
    rec1 = {"card_id": "c_nomutate", "timestamp": 100.0, "latitude": 51.5074, "longitude": -0.1278}
    rec2 = {"card_id": "c_nomutate", "timestamp": 200.0, "latitude": 40.7128, "longitude": -74.0060}
    extractor.extract_features(rec1, mutate_tracker=True)
    res_nomut = extractor.extract_features(rec2, mutate_tracker=False)
    assert res_nomut["geo_is_impossible_travel"] == 1.0
    assert len(extractor.tracker.get_trajectory("c_nomutate")) == 1

    # 5. extract_batch without timestamp column (else branch)
    no_ts_df = pd.DataFrame(
        [
            {"card_id": "c_nots", "latitude": 51.5074, "longitude": -0.1278},
            {"card_id": "c_nots", "latitude": 40.7128, "longitude": -74.0060},
        ]
    )
    batch_res = extractor.extract_batch(no_ts_df)
    assert "geo_distance_km" in batch_res.columns

    # 6. analyze_anomalies with reset_tracker=False and without transaction_id column
    anom_records = [
        {"card_id": "c_anom", "timestamp": 100.0, "latitude": 51.5074, "longitude": -0.1278},
        {"card_id": "c_anom", "timestamp": 200.0, "latitude": 40.7128, "longitude": -74.0060},
    ]
    anom_rep = extractor.analyze_anomalies(anom_records, reset_tracker=False)
    assert anom_rep.impossible_travel_events == 1
    assert "1" in anom_rep.anomalous_event_ids
    rep_dict = anom_rep.to_dict()
    assert rep_dict["impossible_travel_events"] == 1

    # 7. Coordinates without country enriched from IP subnet (lines 830-840)
    rec_ip_enrich = {
        "entity_id": "ent_ip_enrich",
        "Time": 100.0,
        "latitude": 51.5074,
        "longitude": -0.1278,
        "ip_address": "185.86.151.10",
        "billing_country": "US",
    }
    feats_enriched = extractor.extract_features(rec_ip_enrich)
    assert feats_enriched["geo_country_mismatch"] == 1.0

    # 8. User ID fallback
    feats_user = extractor.extract_features(
        {"user_id": "user_fallback", "Time": 500.0, "latitude": 40.7128, "longitude": -74.0060}
    )
    assert "geo_distance_km" in feats_user

    # 9. GeoLocationError
    err = GeoLocationError("test geolocation error")
    assert str(err) == "test geolocation error"
