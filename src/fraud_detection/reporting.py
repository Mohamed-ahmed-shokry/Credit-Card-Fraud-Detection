"""Audit-ready HTML compliance reporting built from promotion evidence.

The ``compliance`` CLI command renders the JSON bundle produced by ``promote``
(plus an optional ``stability`` report and artifact manifest) into a single
self-contained HTML document for auditors and regulators. Rendering uses only
the Python standard library so the runtime and packaging stay dependency-free.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from html import escape
from typing import Any

from fraud_detection import __version__

_REQUIRED_SECTIONS = ("model", "calibration", "thresholds", "drift", "benchmark")

MENU_METRIC_LABELS: dict[str, str] = {
    "roc_auc": "ROC AUC",
    "average_precision": "Average precision",
    "brier_score": "Brier score",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "balanced_accuracy": "Balanced accuracy",
    "expected_cost_per_transaction": "Expected cost / transaction",
}

_STABILITY_METRICS = (
    "roc_auc",
    "average_precision",
    "f1",
    "precision",
    "recall",
    "balanced_accuracy",
    "brier_score",
)

_CSS = """
:root { color-scheme: light; }
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       margin: 2rem auto; max-width: 64rem; padding: 0 1rem; color: #1a1a1a; }
header h1 { margin-bottom: 0.25rem; }
.meta { color: #555; font-size: 0.9rem; margin: 0.25rem 0; }
section { border-top: 2px solid #ddd; margin-top: 2rem; padding-top: 1rem; }
h2 { font-size: 1.25rem; }
h3 { font-size: 1.05rem; margin-bottom: 0.25rem; }
dl { display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr));
     gap: 0.5rem 1.5rem; }
dt { font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.03em; }
dd { margin: 0; font-size: 0.95rem; font-weight: 500; }
table { border-collapse: collapse; margin: 0.75rem 0 1rem; width: 100%; font-size: 0.9rem; }
th, td { border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }
th { background: #f4f4f4; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.tuned td { background: #fff8e1; }
tr.drifted td { background: #fdecea; }
tr.warning td { background: #fff4e5; }
.status-alert { color: #d32f2f; font-weight: 600; }
.status-ok { color: #2e7d32; font-weight: 600; }
.badge { display: inline-block; padding: 0.15rem 0.4rem; border-radius: 4px;
         font-size: 0.8rem; font-weight: 600; }
.badge-allow { background: #e8f5e9; color: #2e7d32; }
.badge-challenge { background: #fff3e0; color: #e65100; }
.badge-deny { background: #ffebee; color: #c62828; }
code { background: #f2f2f2; padding: 0.05rem 0.3rem; border-radius: 3px; }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ddd;
         color: #777; font-size: 0.8rem; }
.footnote { color: #666; font-size: 0.85rem; }
"""

_MISSING_VALUE = "-"


class ComplianceReportError(ValueError):
    """Raised when compliance evidence is incomplete or invalid."""


def _fmt(value: object, *, digits: int = 4) -> str:
    """Format a JSON value for display, escaping strings and rendering cleanly."""
    if value is None:
        return _MISSING_VALUE
    if isinstance(value, str):
        return escape(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return escape(str(value))
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.{digits}f}"
    return escape(str(value))


def _require_mapping(section: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComplianceReportError(f"Compliance bundle section {section!r} must be a JSON object.")
    return value


def _require_list(section: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise ComplianceReportError(
            f"Compliance bundle section {section!r} must contain a JSON array."
        )
    return value


def _table(headers: list[str], rows: list[list[str]], *, numeric: set[str] | None = None) -> str:
    """Render an HTML table; every cell is already escaped by the caller."""
    numeric = numeric or set()
    columns = "".join(
        f'<th class="num">{_cap(header)}</th>' if header in numeric else f"<th>{_cap(header)}</th>"
        for header in headers
    ).join(("<tr>", "</tr>"))
    body = ""
    for row in rows:
        cells = "".join(
            f'<td class="num">{cell}</td>' if header in numeric else f"<td>{cell}</td>"
            for header, cell in zip(headers, row, strict=True)
        )
        body += f"<tr>{cells}</tr>"
    return f"<table><thead>{columns}</thead><tbody>{body}</tbody></table>"


def _cap(value: str) -> str:
    return value if not value else value[0].upper() + value[1:]


def _metric_rows(metrics: Mapping[str, Any]) -> list[list[str]]:
    rows = []
    for key, label in MENU_METRIC_LABELS.items():
        if key in metrics:
            rows.append([label, _fmt(metrics[key])])
    return rows


def render_compliance_report(
    bundle: Mapping[str, Any],
    *,
    stability: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
    slice_disparity: Mapping[str, Any] | None = None,
    tiered_policy: Mapping[str, Any] | None = None,
    score_profile: Mapping[str, Any] | None = None,
) -> str:
    """Render a promotion bundle into a self-contained HTML audit report.

    Args:
        bundle: The JSON document produced by the ``promote`` command, with the
            ``model``, ``calibration``, ``thresholds``, ``drift``, and
            ``benchmark`` sections.
        stability: An optional stability report (as written by the ``stability``
            command) merged into the report as a reproducibility section.
        manifest: An optional artifact integrity manifest, recorded with its
            SHA-256 file digests for artifact provenance.
        slice_disparity: An optional sub-population slice disparity report mapping.
        tiered_policy: An optional tiered threshold tuning result or policy mapping.
        score_profile: An optional validation score distribution profile mapping.

    Returns:
        The complete HTML document as a string.
    """
    model_mapping = _require_mapping("model", bundle.get("model"))
    _require_mapping("calibration", bundle.get("calibration"))
    _require_mapping("thresholds", bundle.get("thresholds"))
    _require_mapping("drift", bundle.get("drift"))
    _require_mapping("benchmark", bundle.get("benchmark"))

    model_version = _fmt(bundle.get("model_version") or _missing_version(bundle))
    document_title = f"Fraud detection compliance report — {model_version}"

    sections = [
        _identity_section(bundle),
        _metrics_section(bundle),
        _calibration_section(bundle),
        _thresholds_section(bundle),
    ]

    t_policy = (
        tiered_policy
        or bundle.get("tiered_policy")
        or model_mapping.get("tiered_tuning")
        or model_mapping.get("tiered_thresholds")
    )
    if isinstance(t_policy, Mapping):
        sections.append(_tiered_policy_section(t_policy))

    s_profile = score_profile or bundle.get("score_profile") or model_mapping.get("score_profile")
    if isinstance(s_profile, Mapping):
        sections.append(_score_profile_section(s_profile))

    s_disparity = slice_disparity or bundle.get("slice_disparity")
    if isinstance(s_disparity, Mapping):
        sections.append(_slice_disparity_section(s_disparity))

    sections.extend(
        [
            _drift_section(bundle),
            _benchmark_section(bundle),
        ]
    )

    content = "".join(sections)
    if stability is not None:
        content += _stability_section(stability)
    if manifest is not None:
        content += _manifest_section(manifest)
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{document_title}</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        "<header>\n"
        "<h1>Model compliance report</h1>\n"
        f'<p class="meta">Model version <code>{escape(model_version)}</code></p>\n'
        f'<p class="meta">Generated {generated_at} by credit-card-fraud-detection '
        f"{escape(__version__)}</p>\n"
        "</header>\n"
        f"<main>{content}</main>\n"
        "<footer>This document records the facts assembled during promotion "
        "review. It states evidence, not a promotion or deployment decision.</footer>\n"
        "</body>\n"
        "</html>\n"
    )


def _missing_version(bundle: Mapping[str, Any]) -> str:
    model = bundle.get("model")
    if isinstance(model, Mapping) and isinstance(model.get("created_at"), str):
        return str(model["created_at"])[:12]
    return "unknown"


def _identity_section(bundle: Mapping[str, Any]) -> str:
    model = _require_mapping("model", bundle.get("model"))
    policy = model.get("cost_policy")
    policy_name = "default"
    fp_cost = _MISSING_VALUE
    fn_cost = _MISSING_VALUE
    if isinstance(policy, Mapping):
        policy_name = str(policy.get("name", "default"))
        fp_cost = _fmt(policy.get("false_positive_cost"))
        fn_cost = _fmt(policy.get("false_negative_cost"))
    model_version = _fmt(bundle.get("model_version") or _missing_version(bundle))
    created_at = _fmt(model.get("created_at"))
    estimator = _fmt(model.get("estimator"))
    threshold = _fmt(model.get("threshold"))
    description = (
        f"<dd>{estimator}</dd>"
        f"<dt>Decision threshold</dt><dd>{threshold}</dd>"
        f"<dt>Model version</dt><dd><code>{escape(model_version)}</code></dd>"
        f"<dt>Created</dt><dd>{created_at}</dd>"
        f"<dt>Cost policy</dt><dd>{escape(policy_name)}</dd>"
        f"<dt>False-positive cost</dt><dd>{fp_cost}</dd>"
        f"<dt>False-negative cost</dt><dd>{fn_cost}</dd>"
    )
    return f'<section id="identity"><h2>Model identity</h2><dl>{description}</dl></section>'


def _metrics_section(bundle: Mapping[str, Any]) -> str:
    model = _require_mapping("model", bundle.get("model"))
    test_metrics = model.get("test_metrics")
    if not isinstance(test_metrics, Mapping):
        raise ComplianceReportError(
            "Compliance bundle section 'model.test_metrics' must be a JSON object."
        )
    rows = _metric_rows(test_metrics)
    missing = [label for label in MENU_METRIC_LABELS.values() if label not in [r[0] for r in rows]]
    if missing:
        return ""
    table = _table(["Metric", "Value"], rows)
    return f'<section id="metrics"><h2>Holdout test metrics</h2>{table}</section>'


def _calibration_section(bundle: Mapping[str, Any]) -> str:
    calibration = _require_mapping("calibration", bundle.get("calibration"))
    summary = _table(
        ["Rows", "Bins"],
        [
            [
                _fmt(calibration.get("rows")),
                _fmt(calibration.get("bins")),
            ]
        ],
    )
    detail = [
        [
            _fmt(calibration.get("brier_score")),
            _fmt(calibration.get("expected_calibration_error")),
            _fmt(calibration.get("max_calibration_error")),
            _fmt(calibration.get("reliability")),
            _fmt(calibration.get("resolution")),
            _fmt(calibration.get("uncertainty")),
        ]
    ]
    decomposition = _table(
        [
            "Brier score",
            "Expected cal. error",
            "Max cal. error",
            "Reliability",
            "Resolution",
            "Uncertainty",
        ],
        detail,
        numeric={
            "Brier score",
            "Expected cal. error",
            "Max cal. error",
            "Reliability",
            "Resolution",
            "Uncertainty",
        },
    )
    bins = _require_list("calibration.detail", calibration.get("detail"))
    bin_rows = []
    for item in bins:
        if not isinstance(item, Mapping):
            raise ComplianceReportError(
                "Compliance bundle section 'calibration.detail' must contain JSON objects."
            )
        bin_rows.append(
            [
                _fmt(item.get("bin_index")),
                f"{_fmt(item.get('lower'))} - {_fmt(item.get('upper'))}",
                _fmt(item.get("count")),
                _fmt(item.get("mean_predicted")),
                _fmt(item.get("fraction_positive")),
            ]
        )
    reliability_table = _table(
        ["Bin", "Range", "Count", "Mean predicted", "Fraction positive"],
        bin_rows,
        numeric={"Count", "Mean predicted", "Fraction positive"},
    )
    return (
        f'<section id="calibration"><h2>Probability calibration</h2>'
        f"<h3>Summary</h3>{summary}<h3>Decomposition</h3>{decomposition}"
        f"<h3>Reliability bins</h3>{reliability_table}"
        '<p class="footnote">Lower Brier score, calibration errors, and '
        "reliability are better; higher resolution is better.</p></section>"
    )


def _thresholds_section(bundle: Mapping[str, Any]) -> str:
    thresholds = _require_mapping("thresholds", bundle.get("thresholds"))
    tuned = thresholds.get("model_threshold_metrics")
    tuned_row: list[str] = []
    if isinstance(tuned, Mapping):
        tuned_row = [
            _fmt(tuned.get("threshold")),
            _fmt(tuned.get("precision")),
            _fmt(tuned.get("recall")),
            _fmt(tuned.get("f1")),
            _fmt(tuned.get("expected_cost_per_transaction")),
            _fmt(tuned.get("flagged")),
            _fmt(tuned.get("flagged_rate")),
        ]
    headers = [
        "Threshold",
        "Precision",
        "Recall",
        "F1",
        "Expected cost",
        "Flagged",
        "Flagged rate",
    ]
    numeric = set(headers)
    rows: list[list[str]] = []
    if tuned_row:
        rows.append(tuned_row)
    candidates = thresholds.get("detail")
    if candidates is not None:
        for item in _require_list("thresholds.detail", candidates):
            if not isinstance(item, Mapping):
                raise ComplianceReportError(
                    "Compliance bundle section 'thresholds.detail' must contain JSON objects."
                )
            rows.append(
                [
                    _fmt(item.get("threshold")),
                    _fmt(item.get("precision")),
                    _fmt(item.get("recall")),
                    _fmt(item.get("f1")),
                    _fmt(item.get("expected_cost_per_transaction")),
                    _fmt(item.get("flagged")),
                    _fmt(item.get("flagged_rate")),
                ]
            )
    body = ""
    for position, row in enumerate(rows):
        row_class = ' class="tuned"' if tuned_row and position == 0 else ""
        cells = "".join(
            f'<td class="num">{cell}</td>' if header in numeric else f"<td>{cell}</td>"
            for header, cell in zip(headers, row, strict=True)
        )
        body += f"<tr{row_class}>{cells}</tr>"
    table = (
        f"<table><thead><tr>{
            ''.join(
                f'<th class="num">{header}</th>' if header in numeric else f'<th>{header}</th>'
                for header in headers
            )
        }"
        f"</tr></thead><tbody>{body}</tbody></table>"
    )
    return (
        f'<section id="thresholds"><h2>Threshold policy</h2>{table}'
        '<p class="footnote">The shaded first row is the model tune; the '
        "remaining rows are candidate thresholds scored on shared labeled "
        "data.</p></section>"
    )


def _tiered_policy_section(policy: Mapping[str, Any]) -> str:
    raw_thresholds = policy.get("best_thresholds")
    thresholds: Mapping[str, Any] = (
        raw_thresholds if isinstance(raw_thresholds, Mapping) else policy
    )
    raw_metrics = policy.get("best_metrics")
    metrics: Mapping[str, Any] = (
        raw_metrics if isinstance(raw_metrics, Mapping) else policy
    )

    r_th = _fmt(thresholds.get("review_threshold"))
    d_th = _fmt(thresholds.get("deny_threshold"))
    objective = _fmt(policy.get("objective")) if "objective" in policy else None

    dl_items = [
        f"<dt>Review threshold</dt><dd>{r_th}</dd>",
        f"<dt>Deny threshold</dt><dd>{d_th}</dd>",
    ]
    if objective:
        dl_items.append(f"<dt>Optimization objective</dt><dd>{objective}</dd>")
    if "expected_cost_per_transaction" in metrics:
        cost_val = _fmt(metrics.get("expected_cost_per_transaction"))
        dl_items.append(f"<dt>Expected cost / tx</dt><dd>{cost_val}</dd>")
    if "total_catch_rate" in metrics:
        dl_items.append(
            f"<dt>Total fraud catch rate</dt><dd>{_fmt(metrics.get('total_catch_rate'))}</dd>"
        )

    tier_rows: list[list[str]] = []
    if "allow_count" in metrics and "deny_count" in metrics:
        tier_rows = [
            [
                '<span class="badge badge-allow">ALLOW</span>',
                f"[0.0000, {r_th})",
                _fmt(metrics.get("allow_count")),
                _fmt(metrics.get("allow_rate")),
                _fmt(metrics.get("missed_fraud", _MISSING_VALUE)),
                _MISSING_VALUE,
            ],
            [
                '<span class="badge badge-challenge">CHALLENGE</span>',
                f"[{r_th}, {d_th})",
                _fmt(metrics.get("review_count")),
                _fmt(metrics.get("review_rate")),
                _fmt(metrics.get("caught_fraud_review")),
                _fmt(metrics.get("review_precision")),
            ],
            [
                '<span class="badge badge-deny">DENY</span>',
                f"[{d_th}, 1.0000]",
                _fmt(metrics.get("deny_count")),
                _fmt(metrics.get("deny_rate")),
                _fmt(metrics.get("caught_fraud_deny")),
                _fmt(metrics.get("deny_precision")),
            ],
        ]

    table_html = ""
    if tier_rows:
        table_html = _table(
            ["Tier action", "Score range", "Transactions", "Rate", "Caught fraud", "Precision"],
            tier_rows,
            numeric={"Transactions", "Rate", "Caught fraud", "Precision"},
        )

    return (
        f'<section id="tiered-policy"><h2>Decision policy &amp; tiered thresholds</h2>'
        f"<dl>{''.join(dl_items)}</dl>{table_html}"
        '<p class="footnote">Transactions routed to CHALLENGE undergo step-up '
        "authentication or manual review; DENY blocks immediately.</p></section>"
    )


def _score_profile_section(profile: Mapping[str, Any]) -> str:
    rows_cnt = _fmt(profile.get("rows"))
    mean_val = _fmt(profile.get("mean"))
    std_val = _fmt(profile.get("std"))
    min_val = _fmt(profile.get("min"))
    max_val = _fmt(profile.get("max"))

    dl_items = (
        f"<dt>Evaluated rows</dt><dd>{rows_cnt}</dd>"
        f"<dt>Mean risk score</dt><dd>{mean_val}</dd>"
        f"<dt>Std deviation</dt><dd>{std_val}</dd>"
        f"<dt>Min score</dt><dd>{min_val}</dd>"
        f"<dt>Max score</dt><dd>{max_val}</dd>"
    )

    quantiles = profile.get("quantiles")
    q_table = ""
    if isinstance(quantiles, Mapping):
        q_rows = [[escape(str(q)), _fmt(val)] for q, val in quantiles.items()]
        q_table = "<h3>Score quantiles</h3>" + _table(
            ["Quantile", "Score threshold"], q_rows, numeric={"Score threshold"}
        )

    return (
        f'<section id="score-profile"><h2>Prediction score profile</h2>'
        f"<dl>{dl_items}</dl>{q_table}</section>"
    )


def _slice_disparity_section(report: Mapping[str, Any]) -> str:
    tot = _fmt(report.get("total_records"))
    g_fr = _fmt(report.get("global_fraud_rate"))
    g_rec = _fmt(report.get("global_recall"))
    g_fpr = _fmt(report.get("global_false_positive_rate"))
    min_rec_d = _fmt(report.get("min_recall_disparity"))
    max_fp_d = _fmt(report.get("max_fpr_disparity"))

    dl_items = (
        f"<dt>Total evaluated</dt><dd>{tot}</dd>"
        f"<dt>Global fraud rate</dt><dd>{g_fr}</dd>"
        f"<dt>Global recall</dt><dd>{g_rec}</dd>"
        f"<dt>Global false positive rate</dt><dd>{g_fpr}</dd>"
        f"<dt>Min recall disparity</dt><dd>{min_rec_d}</dd>"
        f"<dt>Max FPR disparity</dt><dd>{max_fp_d}</dd>"
    )

    slices_list = report.get("slices")
    rows: list[list[str]] = []
    headers = [
        "Slice",
        "Records",
        "Share",
        "Fraud rate",
        "Recall",
        "Recall disp.",
        "FPR",
        "FPR disp.",
        "Status",
    ]
    numeric = {"Records", "Share", "Fraud rate", "Recall", "Recall disp.", "FPR", "FPR disp."}

    row_classes: list[str] = []
    if isinstance(slices_list, list):
        for item in slices_list:
            if not isinstance(item, Mapping):
                continue
            is_under = bool(item.get("is_underperforming", False))
            status_html = (
                '<span class="status-alert">UNDERPERFORMING</span>'
                if is_under
                else '<span class="status-ok">PASS</span>'
            )
            row_classes.append(' class="drifted"' if is_under else "")
            rows.append(
                [
                    escape(str(item.get("slice_name", _MISSING_VALUE))),
                    _fmt(item.get("count")),
                    _fmt(item.get("percentage")),
                    _fmt(item.get("fraud_rate")),
                    _fmt(item.get("recall")),
                    _fmt(item.get("recall_disparity")),
                    _fmt(item.get("false_positive_rate")),
                    _fmt(item.get("fpr_disparity")),
                    status_html,
                ]
            )

    body = ""
    for r_class, row in zip(row_classes, rows, strict=True):
        cells = "".join(
            f'<td class="num">{cell}</td>' if header in numeric else f"<td>{cell}</td>"
            for header, cell in zip(headers, row, strict=True)
        )
        body += f"<tr{r_class}>{cells}</tr>"

    headers_th = "".join(
        f'<th class="num">{h}</th>' if h in numeric else f"<th>{h}</th>" for h in headers
    )
    table_html = f"<table><thead><tr>{headers_th}</tr></thead><tbody>{body}</tbody></table>"

    under_slices = report.get("underperforming_slices")
    alert_note = ""
    if isinstance(under_slices, (list, tuple)) and under_slices:
        escaped_names = ", ".join(f"<code>{escape(str(s))}</code>" for s in under_slices)
        alert_note = (
            f'<p class="meta" style="color: #d32f2f;">'
            f"<strong>Warning:</strong> Slices violating disparity bounds: {escaped_names}.</p>"
        )

    return (
        f'<section id="slice-disparity"><h2>Sub-population slice disparity profiling</h2>'
        f"<dl>{dl_items}</dl>{alert_note}{table_html}"
        '<p class="footnote">Evaluates fairness and reliability across transaction '
        "sub-populations. Slices with recall disparity below threshold or FPR "
        "disparity above threshold are flagged.</p></section>"
    )


def _drift_section(bundle: Mapping[str, Any]) -> str:
    drift = _require_mapping("drift", bundle.get("drift"))
    status = str(drift.get("overall_status", "unknown"))
    thresholds = drift.get("thresholds")
    cutoff_note = ""
    if isinstance(thresholds, Mapping):
        cutoff_note = (
            f" cutoffs at PSI {_fmt(thresholds.get('warning_at'))} "
            f"(warning) and {_fmt(thresholds.get('drift_at'))} (drifted)"
        )
    summary = _table(
        ["Rows", "Overall status", "Mean PSI", "Max PSI"],
        [
            [
                _fmt(drift.get("rows")),
                escape(status),
                _fmt(drift.get("mean_psi")),
                _fmt(drift.get("max_psi")),
            ]
        ],
    )
    features = drift.get("features")
    feature_rows: list[list[str]] = []
    feature_classes: list[str] = []
    if features is not None:
        for item in _require_list("drift.features", features):
            if not isinstance(item, Mapping):
                raise ComplianceReportError(
                    "Compliance bundle section 'drift.features' must contain JSON objects."
                )
            feature_status = str(item.get("status", "stable"))
            feature_rows.append(
                [
                    _fmt(item.get("feature")),
                    _fmt(item.get("psi")),
                    escape(feature_status),
                ]
            )
            safe_cls = re.sub(r"[^a-zA-Z0-9_\-]", "", feature_status)
            feature_classes.append(safe_cls)
    features_table = _table(
        ["Feature", "PSI", "Status"],
        [
            [cell, row[1], f'<span class="{cls}">{row[2]}</span>']
            for cell, row, cls in zip(
                [row[0] for row in feature_rows], feature_rows, feature_classes, strict=True
            )
        ],
        numeric={"PSI"},
    )
    return (
        f'<section id="drift"><h2>Feature drift</h2>{summary}'
        f'<p class="meta">Status{cutoff_note}.</p>'
        f"{features_table}</section>"
    )


def _benchmark_section(bundle: Mapping[str, Any]) -> str:
    benchmark = _require_mapping("benchmark", bundle.get("benchmark"))
    results = benchmark.get("results")
    rows = []
    if results is not None:
        for item in _require_list("benchmark.results", results):
            if not isinstance(item, Mapping):
                raise ComplianceReportError(
                    "Compliance bundle section 'benchmark.results' must contain JSON objects."
                )
            rows.append(
                [
                    _fmt(item.get("batch_size")),
                    _fmt(item.get("median_ms")),
                    _fmt(item.get("ms_per_transaction")),
                    _fmt(item.get("transactions_per_second")),
                ]
            )
    if not rows:
        raise ComplianceReportError(
            "Compliance bundle section 'benchmark.results' must contain at least one row."
        )
    table = _table(
        ["Batch size", "Median ms", "ms / transaction", "Transactions / second"],
        rows,
        numeric={
            "Batch size",
            "Median ms",
            "ms / transaction",
            "Transactions / second",
        },
    )
    return (
        f'<section id="benchmark"><h2>Serving benchmark</h2>{table}'
        '<p class="footnote">Measured on the validation host; production '
        "throughput also depends on concurrency and hardware.</p></section>"
    )


def _stability_section(stability: Mapping[str, Any]) -> str:
    results = stability.get("results")
    sections = []
    if results is None:
        raise ComplianceReportError("Stability report must contain a 'results' array.")
    for item in _require_list("stability.results", results):
        if not isinstance(item, Mapping):
            raise ComplianceReportError("Stability report 'results' must contain JSON objects.")
        estimator = _fmt(item.get("estimator"))
        means = item.get("test_metrics_mean")
        deviations = item.get("test_metrics_std")
        rows: list[list[str]] = []
        if isinstance(means, Mapping) and isinstance(deviations, Mapping):
            for metric in _STABILITY_METRICS:
                if metric in means or metric in deviations:
                    rows.extend(
                        [
                            [
                                MENU_METRIC_LABELS.get(metric, metric),
                                _fmt(means.get(metric)),
                                _fmt(deviations.get(metric)),
                            ]
                        ]
                    )
        sections.append(
            f"<h3>{estimator}</h3>"
            + _table(
                ["Metric", "Mean", "Std dev"],
                rows,
                numeric={"Mean", "Std dev"},
            )
        )
    return f'<section id="stability"><h2>Retraining stability</h2>{"".join(sections)}</section>'


def _manifest_section(manifest: Mapping[str, Any]) -> str:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ComplianceReportError("Artifact manifest must contain a 'files' object.")
    rows = [[_fmt(name), f"<code>{escape(str(digest))}</code>"] for name, digest in files.items()]
    return (
        f'<section id="integrity"><h2>Artifact integrity</h2>'
        f'<p class="meta">Artifact format {_fmt(manifest.get("artifact_version"))}, '
        f"hash algorithm {_fmt(manifest.get('hash_algorithm'))}.</p>"
        f"{_table(['File', 'SHA-256 digest'], rows)}"
        '<p class="footnote">Digests come from the model artifact\'s integrity '
        "manifest; load the artifact in-package to verify them.</p></section>"
    )
