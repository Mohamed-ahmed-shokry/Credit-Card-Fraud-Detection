from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from fraud_detection.cli import app
from fraud_detection.reporting import ComplianceReportError, render_compliance_report

runner = CliRunner()


def _bundle() -> dict[str, Any]:
    return {
        "model_version": "abc123456789",
        "model": {
            "estimator": "LogisticRegression",
            "threshold": 0.42,
            "created_at": "2026-09-14T10:00:00+00:00",
            "cost_policy": {
                "name": "strict-recall",
                "false_positive_cost": 1.0,
                "false_negative_cost": 25.0,
            },
            "test_metrics": {
                "roc_auc": 0.91,
                "average_precision": 0.62,
                "brier_score": 0.04,
                "precision": 0.5,
                "recall": 0.8,
                "f1": 0.62,
                "balanced_accuracy": 0.88,
                "expected_cost_per_transaction": 0.12,
            },
        },
        "calibration": {
            "rows": 100,
            "bins": 2,
            "brier_score": 0.04,
            "expected_calibration_error": 0.03,
            "max_calibration_error": 0.08,
            "reliability": 0.01,
            "resolution": 0.12,
            "uncertainty": 0.09,
            "detail": [
                {
                    "bin_index": 0,
                    "lower": 0.0,
                    "upper": 0.5,
                    "count": 80,
                    "mean_predicted": 0.1,
                    "fraction_positive": 0.08,
                }
            ],
        },
        "thresholds": {
            "model_threshold_metrics": {
                "threshold": 0.42,
                "precision": 0.5,
                "recall": 0.8,
                "f1": 0.62,
                "expected_cost_per_transaction": 0.12,
                "flagged": 20,
                "flagged_rate": 0.2,
            },
            "detail": [
                {
                    "threshold": 0.5,
                    "precision": 0.6,
                    "recall": 0.7,
                    "f1": 0.65,
                    "expected_cost_per_transaction": 0.11,
                    "flagged": 15,
                    "flagged_rate": 0.15,
                }
            ],
        },
        "drift": {
            "rows": 100,
            "overall_status": "drifted",
            "mean_psi": 0.13,
            "max_psi": 0.31,
            "thresholds": {"warning_at": 0.1, "drift_at": 0.25},
            "features": [
                {"feature": "Amount", "psi": 0.31, "status": "drifted"},
                {"feature": "Time", "psi": 0.12, "status": "warning"},
            ],
        },
        "benchmark": {
            "results": [
                {
                    "batch_size": 1,
                    "median_ms": 1.2,
                    "ms_per_transaction": 1.2,
                    "transactions_per_second": 833.3,
                }
            ]
        },
    }


def test_render_compliance_report_includes_all_evidence_sections() -> None:
    html = render_compliance_report(
        _bundle(),
        stability={
            "results": [
                {
                    "estimator": "LogisticRegression",
                    "test_metrics_mean": {"f1": 0.6, "roc_auc": 0.9},
                    "test_metrics_std": {"f1": 0.02, "roc_auc": 0.01},
                }
            ]
        },
        manifest={
            "artifact_version": 2,
            "hash_algorithm": "sha256",
            "files": {"model.joblib": "a" * 64},
        },
    )

    assert '<section id="identity">' in html
    assert '<section id="metrics">' in html
    assert '<section id="calibration">' in html
    assert '<section id="thresholds">' in html
    assert '<section id="drift">' in html
    assert '<section id="benchmark">' in html
    assert '<section id="stability">' in html
    assert '<section id="integrity">' in html
    assert "drifted" in html
    assert "model.joblib" in html


def test_render_compliance_report_escapes_untrusted_text() -> None:
    bundle = _bundle()
    bundle["model"]["estimator"] = "<script>alert('x')</script>"
    bundle["drift"]["features"][0]["feature"] = "<unsafe>"

    html = render_compliance_report(bundle)

    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in html
    assert "&lt;unsafe&gt;" in html
    assert "<script>alert" not in html


def test_render_compliance_report_handles_optional_and_incomplete_sections() -> None:
    bundle = _bundle()
    del bundle["model"]["test_metrics"]["f1"]
    html = render_compliance_report(bundle)
    assert '<section id="metrics">' not in html

    invalid_bundles = []
    for section in ("model", "calibration", "thresholds", "drift", "benchmark"):
        invalid = _bundle()
        invalid[section] = None
        invalid_bundles.append(invalid)
    for invalid in invalid_bundles:
        with pytest.raises(ComplianceReportError, match="must be a JSON object"):
            render_compliance_report(invalid)


def test_render_compliance_report_rejects_malformed_detail() -> None:
    mutations = []
    calibration = _bundle()
    calibration["calibration"]["detail"] = "invalid"
    mutations.append((calibration, "calibration.detail"))
    thresholds = _bundle()
    thresholds["thresholds"]["detail"] = "invalid"
    mutations.append((thresholds, "thresholds.detail"))
    drift = _bundle()
    drift["drift"]["features"] = [1]
    mutations.append((drift, "drift.features"))
    benchmark = _bundle()
    benchmark["benchmark"]["results"] = []
    mutations.append((benchmark, "benchmark.results"))

    for invalid, message in mutations:
        with pytest.raises(ComplianceReportError, match=message):
            render_compliance_report(invalid)

    with pytest.raises(ComplianceReportError, match="Stability report"):
        render_compliance_report(_bundle(), stability={})
    with pytest.raises(ComplianceReportError, match="manifest"):
        render_compliance_report(_bundle(), manifest={})


def test_compliance_command_renders_bundle_and_optional_evidence(tmp_path: Path) -> None:
    bundle_path = tmp_path / "promotion.json"
    stability_path = tmp_path / "stability.json"
    artifact_path = tmp_path / "artifact"
    output_path = tmp_path / "compliance.html"
    artifact_path.mkdir()
    (artifact_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_version": 2,
                "hash_algorithm": "sha256",
                "files": {"model.joblib": "a" * 64},
            }
        ),
        encoding="utf-8",
    )
    bundle_path.write_text(json.dumps(_bundle()), encoding="utf-8")
    stability_path.write_text(
        '{"results": [{"estimator": "baseline", "test_metrics_mean": {}, "test_metrics_std": {}}]}',
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "compliance",
            str(bundle_path),
            "--stability",
            str(stability_path),
            "--artifact",
            str(artifact_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == output_path.read_text(encoding="utf-8") + "\n"
    assert "Model compliance report" in result.stdout
    assert "Artifact integrity" in result.stdout


def test_compliance_command_rejects_non_object_bundle(tmp_path: Path) -> None:
    bundle_path = tmp_path / "invalid.json"
    bundle_path.write_text("[]", encoding="utf-8")

    result = runner.invoke(app, ["compliance", str(bundle_path)])

    assert result.exit_code == 2
    assert "must be a JSON object" in result.stderr
