"""Tests for feature store abstractions, point-in-time historical joins, and skew surveillance."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.features import (
    FeatureDefinition,
    FeatureSkewAnalyzer,
    FeatureSnapshot,
    FeatureStoreError,
    FeatureType,
    FeatureView,
    FileFeatureStore,
    InMemoryFeatureStore,
    SkewStatus,
    calculate_ks_statistic,
    calculate_psi,
    calculate_wasserstein_distance,
    point_in_time_join,
)


def test_feature_type_and_skew_status_enums() -> None:
    assert FeatureType.FLOAT.value == "float"
    assert FeatureType.INT.value == "int"
    assert FeatureType.STRING.value == "string"
    assert FeatureType.BOOL.value == "bool"

    assert SkewStatus.STABLE.value == "stable"
    assert SkewStatus.WARNING.value == "warning"
    assert SkewStatus.DRIFTED.value == "drifted"


def test_feature_definition_validation_and_casting() -> None:
    # Validation errors
    with pytest.raises(FeatureStoreError, match="non-empty string"):
        FeatureDefinition(name="")
    with pytest.raises(FeatureStoreError, match="Unsupported feature type"):
        FeatureDefinition(name="test", feature_type="unsupported")

    # Float casting
    f_float = FeatureDefinition(name="score", feature_type=FeatureType.FLOAT, default_value=0.0)
    assert f_float.validate_and_cast(12.5) == 12.5
    assert f_float.validate_and_cast("3.14") == 3.14
    assert f_float.validate_and_cast(None) == 0.0
    assert f_float.validate_and_cast(float("nan")) == 0.0
    assert f_float.validate_and_cast(float("inf")) == 0.0
    with pytest.raises(FeatureStoreError, match="Failed to cast"):
        f_float.validate_and_cast("not_a_number")

    # Int casting
    f_int = FeatureDefinition(name="count", feature_type=FeatureType.INT, default_value=1)
    assert f_int.validate_and_cast(42) == 42
    assert f_int.validate_and_cast("7") == 7
    assert f_int.validate_and_cast(None) == 1
    with pytest.raises(FeatureStoreError, match="Failed to cast"):
        f_int.validate_and_cast("invalid_int")

    # Bool casting
    f_bool = FeatureDefinition(
        name="is_active", feature_type=FeatureType.BOOL, default_value=False
    )
    assert f_bool.validate_and_cast(True) is True
    assert f_bool.validate_and_cast("true") is True
    assert f_bool.validate_and_cast("YES") is True
    assert f_bool.validate_and_cast("0") is False
    assert f_bool.validate_and_cast(None) is False

    # String casting
    f_str = FeatureDefinition(name="country", feature_type=FeatureType.STRING, default_value="US")
    assert f_str.validate_and_cast("CA") == "CA"
    assert f_str.validate_and_cast(123) == "123"
    assert f_str.validate_and_cast(None) == "US"

    # Serialization
    d = f_float.to_dict()
    assert d["name"] == "score"
    assert d["feature_type"] == "float"
    f_restored = FeatureDefinition.from_dict(d)
    assert f_restored == f_float


def test_feature_view_validation_and_methods() -> None:
    f1 = FeatureDefinition(name="f1", feature_type=FeatureType.FLOAT)
    f2 = FeatureDefinition(name="f2", feature_type=FeatureType.INT, default_value=10)

    # Valid initialization
    view = FeatureView(
        name="user_view",
        entity_key="user_id",
        features=(f1, f2),
        timestamp_col="created_at",
        ttl_seconds=3600.0,
    )
    assert view.name == "user_view"
    assert view.entity_key == "user_id"
    assert view.feature_names() == ("f1", "f2")
    assert view.get_feature("f1") == f1
    assert view.get_feature("missing") is None
    assert view.get_default_values() == {"f1": 0.0, "f2": 10}

    # Serialization roundtrip
    view_dict = view.to_dict()
    restored = FeatureView.from_dict(view_dict)
    assert restored.name == view.name
    assert restored.entity_key == view.entity_key
    assert restored.ttl_seconds == 3600.0
    assert len(restored.features) == 2

    # Validation errors
    with pytest.raises(FeatureStoreError, match="non-empty string"):
        FeatureView(name="", entity_key="user_id", features=(f1,))
    with pytest.raises(FeatureStoreError, match="non-empty string"):
        FeatureView(name="v", entity_key="", features=(f1,))
    with pytest.raises(FeatureStoreError, match="at least one feature"):
        FeatureView(name="v", entity_key="user_id", features=())
    with pytest.raises(FeatureStoreError, match="ttl_seconds must be positive"):
        FeatureView(name="v", entity_key="user_id", features=(f1,), ttl_seconds=-10.0)
    with pytest.raises(FeatureStoreError, match="duplicate feature names"):
        FeatureView(name="v", entity_key="user_id", features=(f1, f1))


def test_feature_snapshot_validation_and_serialization() -> None:
    snap = FeatureSnapshot(
        entity_id="card_123",
        timestamp=1000.0,
        values={"score": 0.85, "count": 3},
    )
    assert snap.entity_id == "card_123"
    assert snap.timestamp == 1000.0

    d = snap.to_dict()
    restored = FeatureSnapshot.from_dict(d)
    assert restored == snap

    with pytest.raises(FeatureStoreError, match="entity_id must not be empty"):
        FeatureSnapshot(entity_id="", timestamp=10.0, values={})
    with pytest.raises(FeatureStoreError, match="Invalid timestamp"):
        FeatureSnapshot(entity_id="123", timestamp=float("nan"), values={})


def test_in_memory_feature_store_crud_and_lookups() -> None:
    f_risk = FeatureDefinition(
        name="risk_score", feature_type=FeatureType.FLOAT, default_value=0.1
    )
    f_tier = FeatureDefinition(
        name="tier", feature_type=FeatureType.STRING, default_value="standard"
    )
    view = FeatureView(
        name="card_profile",
        entity_key="card_id",
        features=(f_risk, f_tier),
        ttl_seconds=300.0,
    )

    store = InMemoryFeatureStore(views=[view])
    assert store.list_views() == ["card_profile"]
    assert store.get_view("card_profile") == view
    assert store.get_view("missing") is None

    # Duplicate view registration error
    with pytest.raises(FeatureStoreError, match="already registered"):
        store.add_view(view)

    # Insert snapshot for unregistered view
    with pytest.raises(FeatureStoreError, match="is not registered"):
        store.put_snapshot(
            "nonexistent",
            FeatureSnapshot(entity_id="c1", timestamp=10.0, values={}),
        )

    # Put snapshots with time ordering
    store.put_snapshots(
        "card_profile",
        [
            FeatureSnapshot(
                entity_id="c1",
                timestamp=100.0,
                values={"risk_score": 0.25, "tier": "silver"},
            ),
            FeatureSnapshot(
                entity_id="c1",
                timestamp=50.0,
                values={"risk_score": 0.15, "tier": "bronze"},
            ),
            FeatureSnapshot(
                entity_id="c1",
                timestamp=200.0,
                values={"risk_score": 0.75, "tier": "gold"},
            ),
        ],
    )

    # Lookup latest (no as_of_time)
    latest_res = store.lookup_online(entity_key="card_id", entity_id="c1")
    assert latest_res.found is True
    assert latest_res.timestamp == 200.0
    assert latest_res.values["risk_score"] == 0.75
    assert latest_res.values["tier"] == "gold"
    assert latest_res.is_stale is False

    # Lookup as_of_time = 75 (should match t=50)
    as_of_75 = store.lookup_online(entity_key="card_id", entity_id="c1", as_of_time=75.0)
    assert as_of_75.found is True
    assert as_of_75.timestamp == 50.0
    assert as_of_75.values["risk_score"] == 0.15
    assert as_of_75.values["tier"] == "bronze"

    # Lookup as_of_time = 150 (should match t=100)
    as_of_150 = store.lookup_online(entity_key="card_id", entity_id="c1", as_of_time=150.0)
    assert as_of_150.found is True
    assert as_of_150.timestamp == 100.0
    assert as_of_150.values["risk_score"] == 0.25

    # Lookup before any snapshot existed (t=20 -> strictly no leakage!)
    as_of_20 = store.lookup_online(entity_key="card_id", entity_id="c1", as_of_time=20.0)
    assert as_of_20.found is False
    assert as_of_20.timestamp is None
    assert as_of_20.values["risk_score"] == 0.1  # default value returned

    # Entity not found
    not_found = store.lookup_online(entity_key="card_id", entity_id="unknown_card")
    assert not_found.found is False
    assert not_found.values["risk_score"] == 0.1

    # Entity key with no view found
    no_view = store.lookup_online(entity_key="unknown_key", entity_id="123")
    assert no_view.found is False

    # View name specified explicitly
    with pytest.raises(FeatureStoreError, match="View 'missing' not found"):
        store.lookup_online(entity_key="card_id", entity_id="c1", view_name="missing")

    with pytest.raises(FeatureStoreError, match="has entity_key 'card_id', expected 'user_id'"):
        store.lookup_online(entity_key="user_id", entity_id="c1", view_name="card_profile")

    # Check TTL staleness: snapshot at t=200, query at t=600 (gap 400 > TTL 300)
    stale_lookup = store.lookup_online(entity_key="card_id", entity_id="c1", as_of_time=600.0)
    assert stale_lookup.found is True
    assert stale_lookup.is_stale is True

    # Historical snapshots retrieval
    hist_all = store.get_historical_snapshots("card_profile")
    assert len(hist_all) == 3
    assert [s.timestamp for s in hist_all] == [50.0, 100.0, 200.0]

    hist_c1 = store.get_historical_snapshots("card_profile", entity_id="c1")
    assert len(hist_c1) == 3

    hist_empty = store.get_historical_snapshots("card_profile", entity_id="c999")
    assert hist_empty == []

    # Diagnostic stats
    stats = store.get_stats()
    assert stats["views_count"] == 1
    assert stats["total_entities"] == 1
    assert stats["total_snapshots"] == 3
    assert stats["lookup_count"] > 0
    assert stats["cache_hits"] > 0
    assert 0.0 <= stats["hit_rate"] <= 1.0


def test_in_memory_feature_store_thread_safety() -> None:
    f = FeatureDefinition(name="count", feature_type=FeatureType.INT, default_value=0)
    view = FeatureView(name="thread_view", entity_key="user_id", features=(f,))
    store = InMemoryFeatureStore(views=[view])

    errors: list[Exception] = []

    def writer(thread_id: int) -> None:
        try:
            for i in range(50):
                store.put_snapshot(
                    "thread_view",
                    FeatureSnapshot(
                        entity_id=f"u_{thread_id}",
                        timestamp=float(i * 10),
                        values={"count": i},
                    ),
                )
        except FeatureStoreError as e:
            errors.append(e)

    def reader(thread_id: int) -> None:
        try:
            for i in range(50):
                store.lookup_online(
                    entity_key="user_id",
                    entity_id=f"u_{thread_id}",
                    as_of_time=float(i * 10),
                )
        except FeatureStoreError as e:
            errors.append(e)

    threads: list[threading.Thread] = []
    for tid in range(4):
        threads.append(threading.Thread(target=writer, args=(tid,)))
        threads.append(threading.Thread(target=reader, args=(tid,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    stats = store.get_stats()
    assert stats["total_snapshots"] == 200
    assert stats["total_entities"] == 4


def test_file_feature_store_save_and_load(tmp_path: Path) -> None:
    f_risk = FeatureDefinition(name="risk", feature_type=FeatureType.FLOAT, default_value=0.0)
    view = FeatureView(name="v1", entity_key="card_id", features=(f_risk,))
    store = FileFeatureStore(views=[view])
    store.put_snapshot(
        "v1", FeatureSnapshot(entity_id="c1", timestamp=10.0, values={"risk": 0.42})
    )

    file_path = tmp_path / "sub" / "store.json"
    store.save_to_file(file_path)
    assert file_path.exists()

    loaded = FileFeatureStore.from_file(file_path)
    assert loaded.list_views() == ["v1"]
    res = loaded.lookup_online(entity_key="card_id", entity_id="c1")
    assert res.found is True
    assert res.values["risk"] == 0.42

    # Error handling
    with pytest.raises(FeatureStoreError, match="not found"):
        FileFeatureStore.from_file(tmp_path / "nonexistent.json")

    bad_file = tmp_path / "bad.json"
    bad_file.write_text("invalid json", encoding="utf-8")
    with pytest.raises(FeatureStoreError, match="Failed to parse"):
        FileFeatureStore.from_file(bad_file)


def test_point_in_time_join_causality_and_enrichment() -> None:
    f_v1 = FeatureDefinition(
        name="user_fraud_rate", feature_type=FeatureType.FLOAT, default_value=0.01
    )
    f_v2 = FeatureDefinition(name="user_kyc_level", feature_type=FeatureType.INT, default_value=1)
    view = FeatureView(
        name="user_features",
        entity_key="user_id",
        features=(f_v1, f_v2),
        ttl_seconds=100.0,
    )
    store = InMemoryFeatureStore(views=[view])

    # Add historical snapshots for user_A:
    # t=50: fraud_rate=0.05, kyc=2
    # t=150: fraud_rate=0.20, kyc=3
    store.put_snapshots(
        "user_features",
        [
            FeatureSnapshot(
                entity_id="user_A",
                timestamp=50.0,
                values={"user_fraud_rate": 0.05, "user_kyc_level": 2},
            ),
            FeatureSnapshot(
                entity_id="user_A",
                timestamp=150.0,
                values={"user_fraud_rate": 0.20, "user_kyc_level": 3},
            ),
        ],
    )

    # Observations at different points in time
    observations = pd.DataFrame(
        [
            {"Time": 25.0, "user_id": "user_A", "Amount": 10.0},
            {"Time": 75.0, "user_id": "user_A", "Amount": 20.0},
            {"Time": 120.0, "user_id": "user_A", "Amount": 30.0},
            {"Time": 149.0, "user_id": "user_A", "Amount": 40.0},
            {"Time": 155.0, "user_id": "user_A", "Amount": 50.0},
            {"Time": 300.0, "user_id": "user_A", "Amount": 60.0},
            {"Time": 75.0, "user_id": "user_B", "Amount": 70.0},
        ]
    )

    enriched_df, summary = point_in_time_join(
        observations=observations,
        store=store,
        view_names=["user_features"],
        timestamp_col="Time",
    )

    assert "user_fraud_rate" in enriched_df.columns
    assert "user_kyc_level" in enriched_df.columns
    assert summary.total_observations == 7
    assert summary.views_applied == ("user_features",)
    assert summary.enriched_features == ("user_fraud_rate", "user_kyc_level")

    # Row 0: t=25 -> defaults
    assert enriched_df.iloc[0]["user_fraud_rate"] == 0.01
    assert enriched_df.iloc[0]["user_kyc_level"] == 1

    # Row 1: t=75 -> joined t=50
    assert enriched_df.iloc[1]["user_fraud_rate"] == 0.05
    assert enriched_df.iloc[1]["user_kyc_level"] == 2

    # Row 2: t=120 -> joined t=50
    assert enriched_df.iloc[2]["user_fraud_rate"] == 0.05
    assert enriched_df.iloc[2]["user_kyc_level"] == 2

    # Row 3: t=149 -> joined t=50, NOT t=150 (strict future-leakage prevention!)
    assert enriched_df.iloc[3]["user_fraud_rate"] == 0.05
    assert enriched_df.iloc[3]["user_kyc_level"] == 2

    # Row 4: t=155 -> joined t=150
    assert enriched_df.iloc[4]["user_fraud_rate"] == 0.20
    assert enriched_df.iloc[4]["user_kyc_level"] == 3

    # Row 5: t=300 -> stale (gap 150 > 100 TTL) -> defaulted
    assert enriched_df.iloc[5]["user_fraud_rate"] == 0.01
    assert enriched_df.iloc[5]["user_kyc_level"] == 1

    # Row 6: user_B -> defaulted
    assert enriched_df.iloc[6]["user_fraud_rate"] == 0.01

    assert summary.matched_observations == 4
    assert summary.stale_observations == 1
    assert summary.to_dict()["total_observations"] == 7


def test_point_in_time_join_edge_cases() -> None:
    f = FeatureDefinition(name="x", feature_type=FeatureType.FLOAT)
    v = FeatureView(name="v", entity_key="card_id", features=(f,))
    store = InMemoryFeatureStore(views=[v])

    # Empty observations
    empty_df, s_empty = point_in_time_join(pd.DataFrame(), store)
    assert empty_df.empty
    assert s_empty.total_observations == 0

    # No views in store
    empty_store = InMemoryFeatureStore()
    df_in = pd.DataFrame([{"Time": 1.0, "card_id": "c1"}])
    _, s_no_views = point_in_time_join(df_in, empty_store)
    assert s_no_views.match_rate == 0.0

    # Missing timestamp col
    with pytest.raises(FeatureStoreError, match="Timestamp column 'missing' not found"):
        point_in_time_join(df_in, store, timestamp_col="missing")

    # Missing view name
    with pytest.raises(FeatureStoreError, match="FeatureView 'invalid' not found"):
        point_in_time_join(df_in, store, view_names=["invalid"])

    # Missing entity column in dataframe
    df_no_entity = pd.DataFrame([{"Time": 1.0, "other_col": "c1"}])
    with pytest.raises(FeatureStoreError, match="Entity column 'card_id' for view 'v' not found"):
        point_in_time_join(df_no_entity, store)

    # Custom entity column mapping
    df_custom = pd.DataFrame([{"Time": 10.0, "my_card": "c1"}])
    store.put_snapshot("v", FeatureSnapshot(entity_id="c1", timestamp=5.0, values={"x": 9.9}))
    df_res, _ = point_in_time_join(
        df_custom,
        store,
        entity_col_mapping={"card_id": "my_card"},
    )
    assert df_res.iloc[0]["x"] == 9.9


def test_distribution_distance_metrics() -> None:
    np.random.seed(42)
    ref = np.random.normal(0.0, 1.0, 1000)
    curr_same = np.random.normal(0.0, 1.0, 1000)
    curr_shifted = np.random.normal(1.5, 1.0, 1000)

    # PSI
    psi_same = calculate_psi(ref, curr_same)
    psi_shifted = calculate_psi(ref, curr_shifted)
    assert psi_same < 0.10
    assert psi_shifted > 0.25

    # Empty arrays
    assert calculate_psi(np.array([]), curr_same) == 0.0

    # KS statistic
    ks_same = calculate_ks_statistic(ref, curr_same)
    ks_shifted = calculate_ks_statistic(ref, curr_shifted)
    assert ks_same < 0.10
    assert ks_shifted > 0.30
    assert calculate_ks_statistic(np.array([]), curr_same) == 0.0

    # Wasserstein distance
    w_same = calculate_wasserstein_distance(ref, curr_same)
    w_shifted = calculate_wasserstein_distance(ref, curr_shifted)
    assert w_same < 0.20
    assert w_shifted > 1.0
    assert calculate_wasserstein_distance(np.array([]), curr_same) == 0.0


def test_feature_skew_analyzer_and_reporting() -> None:
    # Invalid thresholds
    with pytest.raises(FeatureStoreError, match="Invalid PSI thresholds"):
        FeatureSkewAnalyzer(warning_threshold_psi=0.5, drift_threshold_psi=0.2)

    analyzer = FeatureSkewAnalyzer(
        warning_threshold_psi=0.10,
        drift_threshold_psi=0.25,
        ks_warning_threshold=0.10,
        ks_drift_threshold=0.20,
    )

    np.random.seed(123)
    n = 500
    ref_df = pd.DataFrame(
        {
            "Time": np.linspace(0, 100, n),
            "feat_stable": np.random.normal(0.0, 1.0, n),
            "feat_drifted": np.random.normal(0.0, 1.0, n),
            "cat_stable": np.random.choice(["A", "B", "C"], size=n, p=[0.7, 0.2, 0.1]),
            "cat_drifted": np.random.choice(["A", "B", "C"], size=n, p=[0.7, 0.2, 0.1]),
        }
    )

    curr_df = pd.DataFrame(
        {
            "Time": np.linspace(100, 200, n),
            "feat_stable": np.random.normal(0.02, 1.0, n),  # stable
            "feat_drifted": np.random.normal(2.0, 1.0, n),  # severe drift
            "cat_stable": np.random.choice(["A", "B", "C"], size=n, p=[0.7, 0.2, 0.1]),
            "cat_drifted": np.random.choice(["A", "B", "C"], size=n, p=[0.1, 0.2, 0.7]),
        }
    )

    report = analyzer.analyze_skew(ref_df, curr_df)
    assert report.overall_status == SkewStatus.DRIFTED
    assert report.reference_count == n
    assert report.current_count == n

    m_stable = report.metrics["feat_stable"]
    assert m_stable.status == SkewStatus.STABLE
    assert m_stable.mean_reference is not None

    m_drift = report.metrics["feat_drifted"]
    assert m_drift.status == SkewStatus.DRIFTED
    assert m_drift.ks_statistic is not None and m_drift.ks_statistic > 0.20

    m_cat_stable = report.metrics["cat_stable"]
    assert m_cat_stable.status == SkewStatus.STABLE

    m_cat_drift = report.metrics["cat_drifted"]
    assert m_cat_drift.status == SkewStatus.DRIFTED

    # Test serialization and table rendering
    report_dict = report.to_dict()
    assert report_dict["overall_status"] == "drifted"
    assert "metrics" in report_dict

    table_text = report.summary_table()
    assert "Overall Status: DRIFTED" in table_text
    assert "feat_stable" in table_text
    assert "feat_drifted" in table_text

    # Test feature name filtering
    filtered_report = analyzer.analyze_skew(ref_df, curr_df, feature_names=["feat_stable"])
    assert len(filtered_report.metrics) == 1
    assert filtered_report.overall_status == SkewStatus.STABLE


def test_feature_lookup_to_dict_and_list_of_dicts_join() -> None:
    f = FeatureDefinition(name="v_score", feature_type=FeatureType.FLOAT, default_value=1.0)
    v = FeatureView(name="v", entity_key="card_id", features=(f,))
    store = InMemoryFeatureStore(views=[v])
    store.put_snapshot("v", FeatureSnapshot(entity_id="c1", timestamp=5.0, values={"v_score": 2.5}))

    lookup = store.lookup_online(entity_key="card_id", entity_id="c1")
    d = lookup.to_dict()
    assert d["found"] is True
    assert d["values"]["v_score"] == 2.5

    # Pass observations as list of mappings
    obs_list = [{"Time": 10.0, "card_id": "c1"}]
    df_res, summary = point_in_time_join(obs_list, store)
    assert len(df_res) == 1
    assert df_res.iloc[0]["v_score"] == 2.5
    assert summary.matched_observations == 1


def test_feature_skew_warning_status() -> None:
    analyzer = FeatureSkewAnalyzer(
        warning_threshold_psi=0.05,
        drift_threshold_psi=0.50,
        ks_warning_threshold=0.05,
        ks_drift_threshold=0.50,
    )
    np.random.seed(42)
    n = 500
    ref_df = pd.DataFrame(
        {
            "feat_num": np.random.normal(0.0, 1.0, n),
            "feat_cat": np.random.choice(["A", "B"], size=n, p=[0.6, 0.4]),
        }
    )
    curr_df = pd.DataFrame(
        {
            "feat_num": np.random.normal(0.2, 1.0, n),
            "feat_cat": np.random.choice(["A", "B"], size=n, p=[0.45, 0.55]),
        }
    )

    report = analyzer.analyze_skew(ref_df, curr_df)
    assert report.overall_status == SkewStatus.WARNING
