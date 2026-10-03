"""Comprehensive test suite for transaction velocity profiling and sliding feature windows."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fraud_detection.velocity import (
    AggregationType,
    SlidingWindow,
    VelocityConfig,
    VelocityError,
    VelocityWindow,
    VelocityWindowBuffer,
    compute_batch_velocity,
    enrich_record_velocity,
)


class TestVelocityWindow:
    """Tests for individual VelocityWindow configuration and validation."""

    def test_default_name_generation(self) -> None:
        assert VelocityWindow(duration_seconds=45.0).name == "45s"
        assert VelocityWindow(duration_seconds=10.5).name == "10.5s"
        assert VelocityWindow(duration_seconds=300.0).name == "5m"
        assert VelocityWindow(duration_seconds=7200.0).name == "2h"
        assert VelocityWindow(duration_seconds=172800.0).name == "2d"

    def test_custom_name_and_aggregations(self) -> None:
        window = VelocityWindow(
            duration_seconds=60.0,
            name="1m_custom",
            aggregations=(AggregationType.COUNT, AggregationType.SUM),
            ema_alpha=0.5,
        )
        assert window.duration_seconds == 60.0
        assert window.name == "1m_custom"
        assert window.aggregations == (AggregationType.COUNT, AggregationType.SUM)
        assert window.ema_alpha == 0.5

    def test_aggregation_string_coercion(self) -> None:
        # Pass string literals that get coerced to AggregationType enum
        window = VelocityWindow(
            duration_seconds=120.0,
            aggregations=("count", "mean"),  # type: ignore[arg-type]
        )
        assert window.aggregations == (AggregationType.COUNT, AggregationType.MEAN)

    def test_invalid_duration(self) -> None:
        with pytest.raises(VelocityError, match="duration_seconds must be a positive"):
            VelocityWindow(duration_seconds=0.0)
        with pytest.raises(VelocityError, match="duration_seconds must be a positive"):
            VelocityWindow(duration_seconds=-10.0)
        with pytest.raises(VelocityError, match="duration_seconds must be a positive"):
            VelocityWindow(duration_seconds=float("nan"))
        with pytest.raises(VelocityError, match="duration_seconds must be a positive"):
            VelocityWindow(duration_seconds=float("inf"))

    def test_invalid_name(self) -> None:
        with pytest.raises(VelocityError, match="name must be a non-empty string"):
            VelocityWindow(duration_seconds=60.0, name="   ")

    def test_empty_aggregations(self) -> None:
        with pytest.raises(VelocityError, match="specify at least one aggregation"):
            VelocityWindow(duration_seconds=60.0, aggregations=())

    def test_unsupported_aggregation(self) -> None:
        with pytest.raises(VelocityError, match="Unsupported aggregation"):
            VelocityWindow(duration_seconds=60.0, aggregations=("median",))  # type: ignore[arg-type]

    def test_invalid_ema_alpha(self) -> None:
        with pytest.raises(VelocityError, match="ema_alpha must be a finite float in"):
            VelocityWindow(duration_seconds=60.0, ema_alpha=0.0)
        with pytest.raises(VelocityError, match="ema_alpha must be a finite float in"):
            VelocityWindow(duration_seconds=60.0, ema_alpha=1.5)
        with pytest.raises(VelocityError, match="ema_alpha must be a finite float in"):
            VelocityWindow(duration_seconds=60.0, ema_alpha=float("nan"))

    def test_serialization_roundtrip(self) -> None:
        window = VelocityWindow(
            duration_seconds=600.0,
            name="10m",
            aggregations=(AggregationType.COUNT, AggregationType.MAX),
            ema_alpha=0.25,
        )
        as_dict = window.to_dict()
        assert as_dict["duration_seconds"] == 600.0
        assert as_dict["name"] == "10m"
        assert as_dict["aggregations"] == ["count", "max"]
        assert as_dict["ema_alpha"] == 0.25

        recovered = VelocityWindow.from_dict(as_dict)
        assert recovered == window

    def test_from_dict_validation(self) -> None:
        with pytest.raises(VelocityError, match="requires 'duration_seconds'"):
            VelocityWindow.from_dict({})


class TestVelocityConfig:
    """Tests for VelocityConfig initialization, validation, and feature generation."""

    def test_default_config(self) -> None:
        cfg = VelocityConfig()
        assert cfg.entity_key == "card_id"
        assert cfg.timestamp_key == "Time"
        assert cfg.amount_key == "Amount"
        assert len(cfg.windows) == 3
        assert cfg.include_ratios is True
        assert cfg.include_deltas is True
        assert cfg.feature_prefix == "velocity_"
        assert cfg.max_window_duration == 86400.0

    def test_feature_names_order(self) -> None:
        cfg = VelocityConfig(
            windows=(
                VelocityWindow(
                    duration_seconds=300.0,
                    name="5m",
                    aggregations=(AggregationType.COUNT, AggregationType.SUM),
                ),
                VelocityWindow(
                    duration_seconds=3600.0,
                    name="1h",
                    aggregations=(AggregationType.COUNT, AggregationType.MEAN),
                ),
            ),
            include_ratios=True,
            include_deltas=True,
            feature_prefix="v_",
        )
        expected = (
            "v_count_5m",
            "v_sum_5m",
            "v_count_1h",
            "v_mean_1h",
            "v_amount_to_mean_1h",
            "v_count_ratio_5m_1h",
            "v_time_delta_prev",
            "v_amount_delta_prev",
        )
        assert cfg.feature_names() == expected

    def test_sorting_windows_by_duration(self) -> None:
        w_long = VelocityWindow(duration_seconds=86400.0, name="24h")
        w_short = VelocityWindow(duration_seconds=300.0, name="5m")
        cfg = VelocityConfig(windows=(w_long, w_short))
        assert cfg.windows[0].duration_seconds == 300.0
        assert cfg.windows[1].duration_seconds == 86400.0

    def test_duplicate_window_name_error(self) -> None:
        w1 = VelocityWindow(duration_seconds=300.0, name="5m")
        w2 = VelocityWindow(duration_seconds=600.0, name="5m")
        with pytest.raises(VelocityError, match="Duplicate window name '5m'"):
            VelocityConfig(windows=(w1, w2))

    def test_validation_errors(self) -> None:
        with pytest.raises(VelocityError, match="entity_key must be a non-empty string"):
            VelocityConfig(entity_key="")
        with pytest.raises(VelocityError, match="timestamp_key must be a non-empty string"):
            VelocityConfig(timestamp_key=" ")
        with pytest.raises(VelocityError, match="amount_key must be a non-empty string"):
            VelocityConfig(amount_key="")
        with pytest.raises(VelocityError, match="windows must contain at least one"):
            VelocityConfig(windows=())
        with pytest.raises(VelocityError, match="feature_prefix must be a string"):
            VelocityConfig(feature_prefix=123)  # type: ignore[arg-type]
        with pytest.raises(VelocityError, match="max_events_per_entity must be a positive"):
            VelocityConfig(max_events_per_entity=0)
        with pytest.raises(VelocityError, match="max_entities must be a positive"):
            VelocityConfig(max_entities=-5)

    def test_serialization_roundtrip(self, tmp_path: Path) -> None:
        cfg = VelocityConfig(
            entity_key="user_id",
            timestamp_key="ts",
            amount_key="val",
            windows=(VelocityWindow(duration_seconds=60.0, name="1m"),),
            include_ratios=False,
            include_deltas=False,
            feature_prefix="vel_",
            max_events_per_entity=500,
            max_entities=50_000,
        )
        as_dict = cfg.to_dict()
        recovered = VelocityConfig.from_dict(as_dict)
        assert recovered == cfg

        json_str = cfg.to_json()
        assert VelocityConfig.from_json(json_str) == cfg

        file_path = tmp_path / "velocity_config.json"
        cfg.save_file(file_path)
        assert file_path.exists()
        loaded = VelocityConfig.load_file(file_path)
        assert loaded == cfg

    def test_from_dict_and_load_file_errors(self, tmp_path: Path) -> None:
        with pytest.raises(VelocityError, match="data must be a dictionary"):
            VelocityConfig.from_dict([])  # type: ignore[arg-type]
        with pytest.raises(VelocityError, match="windows must be a list"):
            VelocityConfig.from_dict({"windows": "invalid"})
        with pytest.raises(VelocityError, match="Invalid JSON for VelocityConfig"):
            VelocityConfig.from_json("{invalid json")
        with pytest.raises(FileNotFoundError):
            VelocityConfig.load_file(tmp_path / "nonexistent.json")

    def test_from_dict_defaults_and_oserror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = VelocityConfig.from_dict({})
        assert len(cfg.windows) == 3

        test_file = tmp_path / "read_err.json"
        test_file.write_text("{}", encoding="utf-8")

        def _raise_oserror(*_args: Any, **_kwargs: Any) -> str:
            raise OSError("disk failure")

        monkeypatch.setattr(Path, "read_text", _raise_oserror)
        with pytest.raises(VelocityError, match="Failed to read velocity config"):
            VelocityConfig.load_file(test_file)

    def test_disabled_ratios_and_deltas(self) -> None:
        cfg = VelocityConfig(
            windows=(VelocityWindow(duration_seconds=60.0, name="1m"),),
            include_ratios=False,
            include_deltas=False,
        )
        names = cfg.feature_names()
        assert not any("ratio" in n for n in names)
        assert not any("delta" in n for n in names)

        win = SlidingWindow(max_events=5)
        win.append(10.0, 100.0, max_window=60.0)
        feats = win.compute_features(20.0, 120.0, cfg)
        assert "velocity_time_delta_prev" not in feats
        assert "velocity_amount_delta_prev" not in feats
        assert "velocity_amount_to_mean_1m" not in feats


class TestSlidingWindowCalculations:
    """Mathematical verification of sliding window aggregations."""

    def test_cold_start_metrics(self) -> None:
        win = SlidingWindow(max_events=10)
        cfg = VelocityConfig(
            windows=(VelocityWindow(duration_seconds=60.0, name="1m"),),
            include_ratios=True,
            include_deltas=True,
            feature_prefix="v_",
        )
        feats = win.compute_features(current_timestamp=100.0, current_amount=50.0, config=cfg)
        assert feats["v_count_1m"] == 0.0
        assert feats["v_sum_1m"] == 0.0
        assert feats["v_min_1m"] == 0.0
        assert feats["v_max_1m"] == 0.0
        assert feats["v_mean_1m"] == 0.0
        assert feats["v_std_1m"] == 0.0
        assert feats["v_ema_1m"] == 0.0
        assert feats["v_amount_to_mean_1m"] == 0.0
        assert feats["v_time_delta_prev"] == 0.0
        assert feats["v_amount_delta_prev"] == 0.0

        # When current_amount is 0.0, amount_to_mean is 1.0
        feats_zero = win.compute_features(current_timestamp=100.0, current_amount=0.0, config=cfg)
        assert feats_zero["v_amount_to_mean_1m"] == 1.0

    def test_single_prior_event(self) -> None:
        win = SlidingWindow(max_events=10)
        win.append(timestamp=100.0, amount=25.0, max_window=300.0)

        cfg = VelocityConfig(
            windows=(VelocityWindow(duration_seconds=300.0, name="5m"),),
            include_ratios=True,
            include_deltas=True,
            feature_prefix="v_",
        )
        feats = win.compute_features(current_timestamp=130.0, current_amount=50.0, config=cfg)
        assert feats["v_count_5m"] == 1.0
        assert feats["v_sum_5m"] == 25.0
        assert feats["v_min_5m"] == 25.0
        assert feats["v_max_5m"] == 25.0
        assert feats["v_mean_5m"] == 25.0
        assert feats["v_std_5m"] == 0.0
        assert feats["v_ema_5m"] == 25.0
        assert feats["v_amount_to_mean_5m"] == 2.0  # 50.0 / 25.0
        assert feats["v_time_delta_prev"] == 30.0
        assert feats["v_amount_delta_prev"] == 25.0

    def test_multi_event_exact_math(self) -> None:
        win = SlidingWindow(max_events=10)
        # Add events at t=10 (amount=10), t=20 (amount=20), t=30 (amount=30)
        win.append(10.0, 10.0, max_window=100.0)
        win.append(20.0, 20.0, max_window=100.0)
        win.append(30.0, 30.0, max_window=100.0)

        cfg = VelocityConfig(
            windows=(
                VelocityWindow(
                    duration_seconds=15.0, name="15s", ema_alpha=0.5
                ),  # covers t=20, t=30
                VelocityWindow(
                    duration_seconds=50.0, name="50s", ema_alpha=0.5
                ),  # covers t=10, t=20, t=30
            ),
            include_ratios=True,
            include_deltas=True,
            feature_prefix="v_",
        )
        feats = win.compute_features(current_timestamp=35.0, current_amount=60.0, config=cfg)

        # 15s window (cutoff = 35 - 15 = 20) -> events at t=20 (20.0) and t=30 (30.0)
        assert feats["v_count_15s"] == 2.0
        assert feats["v_sum_15s"] == 50.0
        assert feats["v_min_15s"] == 20.0
        assert feats["v_max_15s"] == 30.0
        assert feats["v_mean_15s"] == 25.0
        # Population std for [20, 30]: mean=25, diffs=[-5, 5], sq=[25, 25], var=25, std=5.0
        assert pytest.approx(feats["v_std_15s"], abs=1e-5) == 5.0
        # EMA: first=20.0, second: 0.5 * 30 + 0.5 * 20 = 25.0
        assert pytest.approx(feats["v_ema_15s"], abs=1e-5) == 25.0
        assert pytest.approx(feats["v_amount_to_mean_15s"], abs=1e-5) == 2.4  # 60.0 / 25.0

        # 50s window (cutoff = 35 - 50 = -15) -> events at t=10, 20, 30
        assert feats["v_count_50s"] == 3.0
        assert feats["v_sum_50s"] == 60.0
        assert feats["v_min_50s"] == 10.0
        assert feats["v_max_50s"] == 30.0
        assert feats["v_mean_50s"] == 20.0
        # EMA: t=10 -> 10.0; t=20 -> 0.5 * 20 + 0.5 * 10 = 15.0; t=30 -> 0.5 * 30 + 0.5 * 15 = 22.5
        assert pytest.approx(feats["v_ema_50s"], abs=1e-5) == 22.5

        # Pairwise count ratio: count_15s / count_50s = 2 / 3
        assert pytest.approx(feats["v_count_ratio_15s_50s"], abs=1e-5) == 2.0 / 3.0

        # Deltas
        assert feats["v_time_delta_prev"] == 5.0  # 35 - 30
        assert feats["v_amount_delta_prev"] == 30.0  # 60 - 30

    def test_eviction_and_capacity_bound(self) -> None:
        win = SlidingWindow(max_events=3)
        win.append(10.0, 100.0, max_window=50.0)
        win.append(20.0, 200.0, max_window=50.0)
        win.append(30.0, 300.0, max_window=50.0)
        assert len(win.events) == 3

        # Exceed capacity: should drop oldest (t=10)
        win.append(40.0, 400.0, max_window=50.0)
        assert len(win.events) == 3
        assert win.events[0].timestamp == 20.0

        # Evict expired: window cutoff drops events older than t=100 - 50 = 50
        win.append(100.0, 500.0, max_window=50.0)
        assert len(win.events) == 1
        assert win.events[0].timestamp == 100.0


class TestVelocityWindowBuffer:
    """Tests for multi-entity thread-safe state management."""

    def test_entity_isolation_and_causality(self) -> None:
        buffer = VelocityWindowBuffer()
        # Entity 1 at t=10
        feats1 = buffer.enrich_and_record("card_A", timestamp=10.0, amount=100.0)
        assert feats1["velocity_count_5m"] == 0.0  # strictly prior
        assert feats1["velocity_sum_5m"] == 0.0

        # Entity 2 at t=15 (should not see card_A)
        feats2 = buffer.enrich_and_record("card_B", timestamp=15.0, amount=50.0)
        assert feats2["velocity_count_5m"] == 0.0
        assert feats2["velocity_sum_5m"] == 0.0

        # Entity 1 again at t=20
        feats1_next = buffer.enrich_and_record("card_A", timestamp=20.0, amount=150.0)
        assert feats1_next["velocity_count_5m"] == 1.0
        assert feats1_next["velocity_sum_5m"] == 100.0
        assert feats1_next["velocity_time_delta_prev"] == 10.0
        assert feats1_next["velocity_amount_delta_prev"] == 50.0

        # compute_features without update
        preview = buffer.compute_features("card_A", timestamp=25.0, amount=200.0)
        assert preview["velocity_count_5m"] == 2.0
        assert preview["velocity_sum_5m"] == 250.0

        # Still only 2 events in card_A because preview didn't update
        prof = buffer.get_entity_profile("card_A", current_timestamp=25.0)
        assert prof["tracked"] is True
        assert prof["events_count"] == 2
        assert prof["windows"]["5m"] == 2

    def test_default_entity_fallback(self) -> None:
        buffer = VelocityWindowBuffer()
        buffer.record("", timestamp=10.0, amount=50.0)
        prof = buffer.get_entity_profile("global")
        assert prof["tracked"] is True
        assert prof["events_count"] == 1

        untracked = buffer.get_entity_profile("nonexistent")
        assert untracked["tracked"] is False
        assert untracked["events_count"] == 0

    def test_clear_and_stats(self) -> None:
        buffer = VelocityWindowBuffer()
        buffer.record("A", 1.0, 10.0)
        buffer.record("B", 2.0, 20.0)
        assert buffer.entity_count == 2
        assert buffer.total_events() == 2

        stats = buffer.stats()
        assert stats["entities_tracked"] == 2
        assert stats["total_events"] == 2

        buffer.clear()
        assert buffer.entity_count == 0
        assert buffer.total_events() == 0

    def test_lru_entity_eviction(self) -> None:
        cfg = VelocityConfig(max_entities=2)
        buffer = VelocityWindowBuffer(config=cfg)
        buffer.record("A", 1.0, 10.0)
        buffer.record("B", 2.0, 20.0)
        assert buffer.entity_count == 2

        # Access A so B is older
        buffer.record("A", 3.0, 15.0)

        # Add C: should evict B
        buffer.record("C", 4.0, 30.0)
        assert buffer.entity_count == 2
        assert buffer.get_entity_profile("A")["tracked"] is True
        assert buffer.get_entity_profile("C")["tracked"] is True
        assert buffer.get_entity_profile("B")["tracked"] is False

    def test_expired_entity_pruning(self) -> None:
        cfg = VelocityConfig(
            windows=(VelocityWindow(duration_seconds=100.0, name="100s"),),
        )
        buffer = VelocityWindowBuffer(config=cfg)
        buffer.record("A", 10.0, 10.0)

        # Add new entity B at t=200 -> cutoff is 200 - 100 = 100 -> A (t=10) should be pruned
        buffer.record("B", 200.0, 20.0)
        assert buffer.get_entity_profile("A")["tracked"] is False
        assert buffer.get_entity_profile("B")["tracked"] is True

    def test_concurrency_thread_safety(self) -> None:
        buffer = VelocityWindowBuffer()
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(50):
                    entity = f"thread_{thread_id}"
                    t = float(i * 10)
                    buffer.enrich_and_record(entity, timestamp=t, amount=float(i + 1))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(5)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors
        assert buffer.entity_count == 5
        assert buffer.total_events() == 250


class TestEnrichRecordVelocity:
    """Tests for streaming record enrichment."""

    def test_enrich_record_success(self) -> None:
        buffer = VelocityWindowBuffer()
        rec1 = {"card_id": "c1", "Time": 100.0, "Amount": 50.0, "V1": 1.23}
        out1 = enrich_record_velocity(rec1, buffer, update=True)
        assert out1["card_id"] == "c1"
        assert out1["V1"] == 1.23
        assert out1["velocity_count_5m"] == 0.0

        rec2 = {"card_id": "c1", "Time": 150.0, "Amount": 75.0, "V1": 2.34}
        out2 = enrich_record_velocity(rec2, buffer, update=False)
        assert out2["velocity_count_5m"] == 1.0
        assert out2["velocity_sum_5m"] == 50.0

        # Since update was False for rec2, buffer still only has rec1
        out3 = enrich_record_velocity(rec2, buffer, update=True)
        assert out3["velocity_count_5m"] == 1.0

    def test_missing_timestamp_or_amount(self) -> None:
        buffer = VelocityWindowBuffer()
        with pytest.raises(VelocityError, match="Missing required timestamp key"):
            enrich_record_velocity({"Amount": 10.0}, buffer)
        with pytest.raises(VelocityError, match="Missing required amount key"):
            enrich_record_velocity({"Time": 10.0}, buffer)

    def test_invalid_numeric_values(self) -> None:
        buffer = VelocityWindowBuffer()
        with pytest.raises(VelocityError, match="Invalid timestamp value"):
            enrich_record_velocity({"Time": "not_a_number", "Amount": 10.0}, buffer)
        with pytest.raises(VelocityError, match="Invalid timestamp value"):
            enrich_record_velocity({"Time": float("nan"), "Amount": 10.0}, buffer)
        with pytest.raises(VelocityError, match="Invalid amount value"):
            enrich_record_velocity({"Time": 10.0, "Amount": "bad"}, buffer)
        with pytest.raises(VelocityError, match="Invalid amount value"):
            enrich_record_velocity({"Time": 10.0, "Amount": float("inf")}, buffer)


class TestComputeBatchVelocity:
    """Tests for offline dataset batch velocity computation."""

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame()
        res = compute_batch_velocity(df)
        assert res.empty

    def test_missing_columns_validation(self) -> None:
        df = pd.DataFrame({"Amount": [10.0, 20.0]})
        with pytest.raises(VelocityError, match="Timestamp column 'Time' not found"):
            compute_batch_velocity(df)

        df2 = pd.DataFrame({"Time": [1.0, 2.0]})
        with pytest.raises(VelocityError, match="Amount column 'Amount' not found"):
            compute_batch_velocity(df2)

    def test_non_numeric_validation(self) -> None:
        df = pd.DataFrame({"Time": ["abc", "def"], "Amount": [10.0, 20.0]})
        with pytest.raises(VelocityError, match="Timestamp column 'Time' contains non-numeric"):
            compute_batch_velocity(df)

        df2 = pd.DataFrame({"Time": [1.0, 2.0], "Amount": [10.0, "bad"]})
        with pytest.raises(VelocityError, match="Amount column 'Amount' contains non-numeric"):
            compute_batch_velocity(df2)

    def test_batch_computation_parity_with_streaming(self) -> None:
        # Create a test dataframe with multiple cards and timestamps
        data = {
            "card_id": ["c1", "c2", "c1", "c1", "c2"],
            "Time": [10.0, 20.0, 50.0, 100.0, 120.0],
            "Amount": [100.0, 50.0, 200.0, 150.0, 80.0],
            "V1": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
        df = pd.DataFrame(data)
        cfg = VelocityConfig(
            windows=(
                VelocityWindow(duration_seconds=60.0, name="1m"),
                VelocityWindow(duration_seconds=300.0, name="5m"),
            ),
            include_ratios=True,
            include_deltas=True,
        )

        batch_result = compute_batch_velocity(df, config=cfg)

        # Compute streamingly row by row
        stream_buffer = VelocityWindowBuffer(config=cfg)
        stream_rows: list[dict[str, Any]] = []
        for raw_rec in df.to_dict(orient="records"):
            rec: dict[str, Any] = {str(k): v for k, v in raw_rec.items()}
            enriched = enrich_record_velocity(rec, stream_buffer, config=cfg, update=True)
            stream_rows.append(enriched)
        stream_df = pd.DataFrame(stream_rows)

        # Check exact equality across all feature columns
        for fname in cfg.feature_names():
            assert np.allclose(batch_result[fname].to_numpy(), stream_df[fname].to_numpy())

    def test_out_of_chronological_order_preservation(self) -> None:
        # DataFrame where rows are NOT in chronological order
        df = pd.DataFrame(
            {
                "card_id": ["c1", "c1", "c1"],
                "Time": [300.0, 100.0, 200.0],  # out of order
                "Amount": [30.0, 10.0, 20.0],
            }
        )
        cfg = VelocityConfig(
            windows=(VelocityWindow(duration_seconds=600.0, name="10m"),),
            include_deltas=True,
            include_ratios=True,
        )
        res = compute_batch_velocity(df, config=cfg)

        # Row 1 (t=100.0, amount=10.0) is chronologically first -> cold start
        assert res.loc[1, "velocity_count_10m"] == 0.0
        # Row 2 (t=200.0, amount=20.0) is chronologically second -> sees row 1 (10.0)
        assert res.loc[2, "velocity_count_10m"] == 1.0
        assert res.loc[2, "velocity_sum_10m"] == 10.0
        assert res.loc[2, "velocity_time_delta_prev"] == 100.0
        # Row 0 (t=300.0, amount=30.0) is chronologically third -> sees row 1 and 2
        assert res.loc[0, "velocity_count_10m"] == 2.0
        assert res.loc[0, "velocity_sum_10m"] == 30.0
        assert res.loc[0, "velocity_time_delta_prev"] == 100.0

    def test_single_stream_without_entity_col(self) -> None:
        # No entity column -> all events treated as single entity
        df = pd.DataFrame(
            {
                "Time": [10.0, 20.0, 30.0],
                "Amount": [5.0, 10.0, 15.0],
            }
        )
        cfg = VelocityConfig(
            windows=(VelocityWindow(duration_seconds=100.0, name="100s"),),
            include_deltas=True,
        )
        res = compute_batch_velocity(df, config=cfg)
        assert res["velocity_count_100s"].tolist() == [0.0, 1.0, 2.0]
        assert res["velocity_sum_100s"].tolist() == [0.0, 5.0, 15.0]
