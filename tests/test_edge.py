from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fraud_detection.data import generate_synthetic_data, validate_frame
from fraud_detection.edge import (
    EdgeArtifactError,
    EdgeDecisionAction,
    EdgeModel,
    _sigmoid,
    build_edge_model,
    load_edge_model,
    score_batch_file,
)
from fraud_detection.model import (
    CalibrationMethod,
    EstimatorType,
    FraudModel,
    TrainingConfig,
    train_model,
)


@pytest.fixture(scope="module")
def edge_inputs():
    dataset = validate_frame(generate_synthetic_data(rows=600, fraud_rate=0.1))
    model = train_model(
        dataset,
        config=TrainingConfig(calibration_method=CalibrationMethod.NONE),
    )
    return model, dataset


def _minimal_edge_model() -> EdgeModel:
    return EdgeModel(
        model_version="model",
        feature_names=("first", "second"),
        threshold=0.5,
        scaler_mean=(0.0, 0.0),
        scaler_scale=(1.0, 1.0),
        quantized_weights=(10, -10),
        weight_scale=0.1,
        intercept=0.0,
        quantization_bits=8,
        prune_epsilon=0.0,
        pruned_features=(),
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": 2}, "schema version"),
        ({"quantization_bits": 4}, "8-bit"),
        ({"feature_names": ()}, "feature names"),
        ({"feature_names": ("first", "first")}, "feature names"),
        ({"scaler_mean": (0.0,)}, "arrays"),
        ({"threshold": 2.0}, "threshold"),
        ({"weight_scale": 0.0}, "weight_scale"),
        ({"prune_epsilon": -1.0}, "prune_epsilon"),
        ({"scaler_mean": (float("nan"), 0.0)}, "scaler values"),
        ({"scaler_scale": (0.0, 1.0)}, "scaler scales"),
        ({"quantized_weights": (128, 0)}, "int8"),
    ],
)
def test_edge_artifact_validation(changes: dict[str, object], message: str) -> None:
    with pytest.raises(EdgeArtifactError, match=message):
        replace(_minimal_edge_model(), **changes)


def test_edge_runtime_rejects_empty_records_and_invalid_values() -> None:
    edge_model = _minimal_edge_model()

    with pytest.raises(EdgeArtifactError, match="At least one"):
        edge_model.score_records([])
    with pytest.raises(EdgeArtifactError, match="unexpected"):
        edge_model.score_record({"first": 0.0, "second": 0.0, "extra": 1.0})
    with pytest.raises(EdgeArtifactError, match="numeric"):
        edge_model.score_record({"first": True, "second": 0.0})


def test_edge_loader_rejects_unreadable_json_and_non_object(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    with pytest.raises(EdgeArtifactError, match="Could not read"):
        load_edge_model(missing_path)

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("not json", encoding="utf-8")
    with pytest.raises(EdgeArtifactError, match="Could not read"):
        load_edge_model(invalid_path)

    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")
    with pytest.raises(EdgeArtifactError, match="root"):
        load_edge_model(list_path)


def test_edge_export_rejects_invalid_pruning_and_pipeline_contracts(edge_inputs) -> None:
    model, _dataset = edge_inputs
    with pytest.raises(EdgeArtifactError, match="prune_epsilon"):
        build_edge_model(model, prune_epsilon=float("nan"))

    metadata = {
        "training_config": {"estimator": "logistic_regression", "calibration_method": "none"}
    }
    base = FraudModel(SimpleNamespace(), 0.5, ("first",), metadata)
    with pytest.raises(EdgeArtifactError, match="scale/classifier"):
        build_edge_model(base)

    base.estimator = SimpleNamespace(
        named_steps={
            "scale": SimpleNamespace(mean_=[0.0], scale_=[1.0]),
            "classifier": SimpleNamespace(),
        }
    )
    with pytest.raises(EdgeArtifactError, match="not fitted"):
        build_edge_model(base)

    base.estimator.named_steps["classifier"] = SimpleNamespace(
        coef_=[[1.0]], intercept_=[0.0], classes_=[0, 2]
    )
    with pytest.raises(EdgeArtifactError, match="binary"):
        build_edge_model(base)


def test_edge_runtime_matches_source_model_and_round_trips(edge_inputs, tmp_path: Path) -> None:
    model, dataset = edge_inputs
    edge_model = build_edge_model(model)
    records = dataset.features.iloc[:25].to_dict(orient="records")

    edge_probabilities = edge_model.score_records(records)
    source_probabilities = model.predict_probabilities(dataset.features.iloc[:25])

    assert (
        max(
            abs(edge - source)
            for edge, source in zip(edge_probabilities, source_probabilities, strict=True)
        )
        < 0.005
    )
    assert [edge_model.predict_record(record) for record in records] == [
        bool(value >= model.threshold) for value in source_probabilities
    ]

    artifact_path = tmp_path / "edge-model.json"
    artifact_path.write_text(json.dumps(edge_model.to_dict()), encoding="utf-8")
    restored = load_edge_model(artifact_path)
    np.testing.assert_allclose(restored.score_records(records), edge_probabilities)


def test_edge_export_records_pruned_features(edge_inputs) -> None:
    model, _dataset = edge_inputs
    edge_model = build_edge_model(model, prune_epsilon=1_000.0)

    assert edge_model.pruned_features == model.feature_names
    assert edge_model.quantized_weights == (0,) * len(model.feature_names)


def test_edge_runtime_rejects_schema_and_non_finite_values(edge_inputs) -> None:
    model, dataset = edge_inputs
    edge_model = build_edge_model(model)
    record = dataset.features.iloc[0].to_dict()

    with pytest.raises(EdgeArtifactError, match="missing"):
        edge_model.score_record({key: value for key, value in record.items() if key != "V1"})
    record["V1"] = float("nan")
    with pytest.raises(EdgeArtifactError, match="finite"):
        edge_model.score_record(record)


def test_edge_export_rejects_calibrated_and_tree_models() -> None:
    dataset = validate_frame(generate_synthetic_data(rows=400, fraud_rate=0.1))

    calibrated = train_model(dataset)
    with pytest.raises(EdgeArtifactError, match="calibration_method='none'"):
        build_edge_model(calibrated)

    tree_model = train_model(
        dataset,
        config=TrainingConfig(
            estimator=EstimatorType.RANDOM_FOREST,
            calibration_method=CalibrationMethod.NONE,
        ),
    )
    with pytest.raises(EdgeArtifactError, match="logistic_regression"):
        build_edge_model(tree_model)


def test_edge_loader_rejects_malformed_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "malformed.json"
    artifact_path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")

    with pytest.raises(EdgeArtifactError, match="Invalid edge artifact"):
        load_edge_model(artifact_path)


def test_edge_explain_record_and_records(edge_inputs) -> None:
    model, dataset = edge_inputs
    edge_model = build_edge_model(model)
    records = dataset.features.iloc[:5].to_dict(orient="records")

    # 1. explain_record single
    res = edge_model.explain_record(records[0], top_k=3)
    assert "probability" in res
    assert "intercept" in res
    assert "contributions" in res
    assert "top_contributions" in res
    assert "summary" in res
    assert len(res["top_contributions"]) == 3

    # Check linear reconstruction
    linear_score = res["intercept"] + sum(res["contributions"].values())
    reconstructed_prob = 1.0 / (1.0 + np.exp(-linear_score))
    np.testing.assert_allclose(res["probability"], reconstructed_prob, atol=1e-6)
    np.testing.assert_allclose(res["probability"], edge_model.score_record(records[0]), atol=1e-6)

    # Check top_contributions ordering (abs descending)
    contribs = [abs(c["contribution"]) for c in res["top_contributions"]]
    assert contribs == sorted(contribs, reverse=True)

    # Direction check
    for item in res["top_contributions"]:
        assert item["direction"] in ("increases_risk", "decreases_risk")
        if item["contribution"] >= 0:
            assert item["direction"] == "increases_risk"
        else:
            assert item["direction"] == "decreases_risk"

    # 2. explain_records batch
    batch_res = edge_model.explain_records(records, top_k=2)
    assert len(batch_res) == len(records)
    assert all(len(r["top_contributions"]) == 2 for r in batch_res)

    # 3. Validation errors
    with pytest.raises(EdgeArtifactError, match="At least one edge transaction"):
        edge_model.explain_records([])
    with pytest.raises(EdgeArtifactError, match="top_k must be an integer between 1 and"):
        edge_model.explain_record(records[0], top_k=0)
    with pytest.raises(EdgeArtifactError, match="top_k must be an integer between 1 and"):
        edge_model.explain_record(records[0], top_k=len(edge_model.feature_names) + 1)
    with pytest.raises(EdgeArtifactError, match="top_k must be an integer between 1 and"):
        edge_model.explain_record(records[0], top_k=True)  # type: ignore[arg-type]


def test_edge_predict_decision_record_and_records(edge_inputs) -> None:
    model, dataset = edge_inputs
    edge_model = build_edge_model(model)
    records = dataset.features.iloc[:10].to_dict(orient="records")

    # 1. Default thresholds (binary fallback to threshold)
    default_dec = edge_model.predict_decision_record(records[0])
    assert default_dec["decision"] in (
        EdgeDecisionAction.ALLOW.value,
        EdgeDecisionAction.DENY.value,
    )
    assert default_dec["review_threshold"] == edge_model.threshold
    assert default_dec["deny_threshold"] == edge_model.threshold
    assert default_dec["is_fraud"] == (default_dec["probability"] >= edge_model.threshold)

    # 2. Tiered decisions with distinct thresholds
    # Set thresholds such that we force CHALLENGE
    prob = edge_model.score_record(records[0])
    low_th = max(0.01, prob - 0.1)
    high_th = min(0.99, prob + 0.1)
    tiered_dec = edge_model.predict_decision_record(
        records[0], review_threshold=low_th, deny_threshold=high_th
    )
    assert tiered_dec["decision"] == EdgeDecisionAction.CHALLENGE.value
    assert tiered_dec["is_fraud"] is False

    # Force ALLOW
    allow_dec = edge_model.predict_decision_record(
        records[0], review_threshold=0.99, deny_threshold=0.99
    )
    assert allow_dec["decision"] == EdgeDecisionAction.ALLOW.value
    assert allow_dec["is_fraud"] is False

    # Force DENY
    deny_dec = edge_model.predict_decision_record(
        records[0], review_threshold=0.0, deny_threshold=0.0
    )
    assert deny_dec["decision"] == EdgeDecisionAction.DENY.value
    assert deny_dec["is_fraud"] is True

    # 3. Batch predict_decision_records
    batch_dec = edge_model.predict_decision_records(
        records, review_threshold=0.2, deny_threshold=0.8
    )
    assert len(batch_dec) == 10
    assert all(r["decision"] in ("ALLOW", "CHALLENGE", "DENY") for r in batch_dec)

    # 4. Validation errors
    with pytest.raises(EdgeArtifactError, match="must both be provided or both omitted"):
        edge_model.predict_decision_record(records[0], review_threshold=0.2)
    with pytest.raises(EdgeArtifactError, match="cannot exceed"):
        edge_model.predict_decision_record(records[0], review_threshold=0.8, deny_threshold=0.2)
    with pytest.raises(EdgeArtifactError, match="review_threshold must be a finite float"):
        edge_model.predict_decision_record(records[0], review_threshold=-0.1, deny_threshold=0.5)
    with pytest.raises(EdgeArtifactError, match="deny_threshold must be a finite float"):
        edge_model.predict_decision_record(records[0], review_threshold=0.1, deny_threshold=1.5)
    with pytest.raises(EdgeArtifactError, match="review_threshold must be a finite float"):
        edge_model.predict_decision_record(
            records[0],
            review_threshold=True,
            deny_threshold=0.5,  # type: ignore[arg-type]
        )
    with pytest.raises(EdgeArtifactError, match="At least one edge transaction"):
        edge_model.predict_decision_records([])


def test_score_batch_file_csv_and_jsonl(edge_inputs, tmp_path: Path) -> None:
    model, dataset = edge_inputs
    edge_model = build_edge_model(model)
    records = dataset.features.iloc[:15].to_dict(orient="records")

    # 1. Create CSV input
    csv_in = tmp_path / "input.csv"
    dataset.features.iloc[:15].to_csv(csv_in, index=False)

    csv_out = tmp_path / "output.csv"
    res_csv = edge_model.score_batch_file(csv_in, csv_out, explain=True, top_k=2)
    assert res_csv["rows_processed"] == 15
    assert res_csv["explained"] is True
    assert csv_out.is_file()

    # Read back CSV output and verify columns
    lines = csv_out.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    assert "row_id" in header
    assert "probability" in header
    assert "decision" in header
    assert "is_fraud" in header
    assert "summary" in header
    assert "top_factors" in header
    assert len(lines) == 16  # header + 15 rows

    # 2. Create JSONL input and output
    jsonl_in = tmp_path / "input.jsonl"
    with jsonl_in.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    jsonl_out = tmp_path / "output.jsonl"
    res_jsonl = score_batch_file(
        jsonl_in,
        jsonl_out,
        model=edge_model,
        explain=True,
        review_threshold=0.1,
        deny_threshold=0.9,
    )
    assert res_jsonl["rows_processed"] == 15
    assert jsonl_out.is_file()

    out_records = [json.loads(line) for line in jsonl_out.read_text(encoding="utf-8").splitlines()]
    assert len(out_records) == 15
    assert "explanation" in out_records[0]
    assert "top_contributions" in out_records[0]
    assert "decision" in out_records[0]

    # 3. JSON array input and JSON output
    json_in = tmp_path / "input.json"
    json_in.write_text(json.dumps(records), encoding="utf-8")

    json_out = tmp_path / "output.json"
    res_json = score_batch_file(edge_model, json_in, json_out)
    assert res_json["rows_processed"] == 15
    parsed_json = json.loads(json_out.read_text(encoding="utf-8"))
    assert isinstance(parsed_json, list)
    assert len(parsed_json) == 15

    # 4. Polymorphic model path call
    art_path = tmp_path / "edge_art.json"
    art_path.write_text(json.dumps(edge_model.to_dict()), encoding="utf-8")
    tsv_in = tmp_path / "input.tsv"
    dataset.features.iloc[:5].to_csv(tsv_in, sep="\t", index=False)
    tsv_out = tmp_path / "output.tsv"
    res_tsv = score_batch_file(art_path, tsv_in, tsv_out)
    assert res_tsv["rows_processed"] == 5

    # 5. Validation errors
    with pytest.raises(EdgeArtifactError, match="Input file not found"):
        score_batch_file(edge_model, tmp_path / "nonexistent.csv", csv_out)

    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    with pytest.raises(EdgeArtifactError, match="no records to score"):
        score_batch_file(edge_model, empty_csv, tmp_path / "empty_out.csv")

    bad_in = tmp_path / "input.parquet"
    bad_in.write_text("dummy", encoding="utf-8")
    with pytest.raises(EdgeArtifactError, match="Unsupported input file format"):
        score_batch_file(edge_model, bad_in, tmp_path / "out.csv")

    with pytest.raises(EdgeArtifactError, match="Unsupported output file format"):
        score_batch_file(edge_model, csv_in, tmp_path / "out.parquet")

    # 6. Additional caller signatures and error branches
    with pytest.raises(EdgeArtifactError, match="output_path is required"):
        score_batch_file(edge_model, csv_in)

    with pytest.raises(EdgeArtifactError, match="model must be provided"):
        score_batch_file(csv_in, csv_out)

    # JSON output with explain=True
    json_expl_out = tmp_path / "out_expl.json"
    res_json_expl = score_batch_file(edge_model, json_in, json_expl_out, explain=True)
    assert res_json_expl["explained"] is True
    parsed_expl = json.loads(json_expl_out.read_text(encoding="utf-8"))
    assert "explanation" in parsed_expl[0]

    # JSONL output with explain=False
    jsonl_noexpl_out = tmp_path / "out_noexpl.jsonl"
    score_batch_file(edge_model, jsonl_in, jsonl_noexpl_out, explain=False)
    line1 = json.loads(jsonl_noexpl_out.read_text(encoding="utf-8").splitlines()[0])
    assert "explanation" not in line1

    # Malformed JSONL line
    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text("not json\n", encoding="utf-8")
    with pytest.raises(EdgeArtifactError, match="Line 1 is not valid JSON"):
        score_batch_file(edge_model, bad_jsonl, tmp_path / "bad_out.jsonl")

    non_obj_jsonl = tmp_path / "non_obj.jsonl"
    non_obj_jsonl.write_text("123\n", encoding="utf-8")
    with pytest.raises(EdgeArtifactError, match="Line 1 must be a JSON object"):
        score_batch_file(edge_model, non_obj_jsonl, tmp_path / "bad_out.jsonl")

    # Malformed JSON array
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("not json", encoding="utf-8")
    with pytest.raises(EdgeArtifactError, match="Invalid JSON"):
        score_batch_file(edge_model, bad_json, tmp_path / "bad_out.json")

    dict_json = tmp_path / "dict.json"
    dict_json.write_text(json.dumps({"key": "value"}), encoding="utf-8")
    with pytest.raises(EdgeArtifactError, match="list of records"):
        score_batch_file(edge_model, dict_json, tmp_path / "bad_out.json")

    item_not_dict_json = tmp_path / "item_not_dict.json"
    item_not_dict_json.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(EdgeArtifactError, match="Item 1 must be a JSON object"):
        score_batch_file(edge_model, item_not_dict_json, tmp_path / "bad_out.json")


def test_edge_sigmoid_branches() -> None:
    # Test positive, zero, and negative values
    pos = _sigmoid(5.0)
    neg = _sigmoid(-5.0)
    zero = _sigmoid(0.0)
    assert 0.99 < pos < 1.0
    assert 0.0 < neg < 0.01
    assert zero == 0.5
