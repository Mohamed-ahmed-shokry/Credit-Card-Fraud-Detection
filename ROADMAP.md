# Roadmap

This is a living plan for where the project goes after `v0.1.0`. It exists
because "reference implementation" is not a finished state — it is a starting
point that [Responsible use and limitations](README.md#responsible-use-and-limitations)
already names honestly: representative temporal data, validated business-cost
inputs, ongoing calibration analysis, and monitoring for drift are all things
a real deployment needs beyond what ships here today.

Each phase is scoped to stay inside this project's actual mission — a
reproducible, leakage-safe reference implementation and its tooling — not a
general-purpose production security stack. Authentication, rate limiting, and
network isolation remain the deploying operator's responsibility, as
[SECURITY.md](SECURITY.md) already states; where this roadmap touches that
territory, it is to ship an optional, clearly-labeled reference pattern, not a
replacement for a real gateway.

Status legend: `Proposed` (not started), `In progress`, `Done`.

## Phase 1 — Distribution

The package already builds cleanly and passes `twine check` in CI. Nothing
publishes it anywhere yet.

- **Automated PyPI publishing on release** (`In progress`) — a GitHub Actions
  workflow triggered by publishing a GitHub Release, using PyPI's trusted
  publishing (OIDC) so no long-lived API token has to live in repository
  secrets. Requires the maintainer to link the trusted publisher on pypi.org
  before it can actually run; see the workflow file for the exact steps.
- **TestPyPI dry run** (`Done`) — publish to TestPyPI on every push to
  `main` so a metadata or packaging regression is visible before it ever
  reaches a real release.

## Phase 2 — Model flexibility

The trained model is a class-balanced logistic regression by design — an
honest, interpretable baseline. That should stay the default, but power users
training on their own data may want more expressive options.

- **Opt-in alternative estimators** (`Done`) — `--estimator random_forest`
  and `--estimator hist_gradient_boosting` alongside the default logistic
  regression, sharing the same leakage-safe split, calibration, and
  threshold-selection pipeline. `explain` reports native feature importances
  for random forest, and deterministic training-data permutation importance
  (new `permutation_importance` method label) for histogram gradient boosting,
  which exposes no native importances in the pinned scikit-learn runtime.
- **Model comparison** (`Done`) — `compare` trains every requested estimator
  against the same split and reports validation and test metrics side by
  side, so a choice between estimators is evidence-based rather than a single
  trained artifact taken on faith.
- **Hyperparameter comparison within one estimator** (`Done`) — `compare`
  accepts `--param-name`/`--param-values` to sweep one hyperparameter for a
  single estimator on the same split (e.g. `--estimator random_forest
  --param-name n_estimators --param-values 50,100,200`). Sweepable parameters
  are allow-listed per estimator and validated before training, and every
  swept parameter is also settable on `train`, so a sweep winner can be
  reproduced in a real artifact.

## Phase 3 — Explainability and observability depth

`explain` already reports global, standardized feature effects. Two natural
extensions:

- **Per-transaction local explanation** (`Done`) — extend `predict` (CLI
  and API) with an optional per-transaction contribution breakdown, so a
  flagged transaction's score is traceable to specific feature values, not
  just the global ranking. The CLI uses `--explain` flag; the API accepts
  `"explain": true` in the request body. Contributions are included in the
  output CSV (`contrib_<feature>` columns) or as `contributions` in each
  prediction object.
- **Prometheus-compatible metrics endpoint** (`Done`) — `GET /metrics` reports
  request counts, request duration, and scored-transaction decision counts,
  built from data the existing request-logging middleware already computed.
  Each app instance owns an isolated registry rather than sharing
  `prometheus_client`'s global default, which would otherwise raise on the
  second `create_app()` call in the same process.

## Phase 4 — Optional reference hardening patterns

Explicitly optional and off by default. SECURITY.md is correct that a
production operator owns authentication and rate limiting; these exist so
someone standing this project up has a documented, tested starting point
instead of building it from nothing.

- **Optional API key middleware** (`Done`) — a reference implementation,
  disabled unless configured, documented as defense in depth rather than a
  substitute for a real authentication layer. Enabled via `api_keys` parameter
  in `create_app()`. Validates the `X-API-Key` header.
- **Optional rate-limiting middleware** (`Done`) — same framing as above,
  for request-rate limiting ahead of the existing body-size limit. Enabled via
  `rate_limit_requests` and `rate_limit_window_seconds` parameters in
  `create_app()`. Uses a fixed-window in-memory algorithm keyed by client IP.

## Phase 5 — Calibration and operations depth (next)

Scoped, proposed next steps that stay inside the reference-implementation
mission:

- **Calibration analysis report** (`Proposed`) — a `calibrate`-style report
  (reliability curve bins plus Brier-score decomposition) computed from the
  untouched test split, so operators can judge whether predicted probabilities
  mean what they say before wiring them to thresholds.
- **Drift-alert thresholds in the model card** (`Proposed`) — persist the PSI
  warning/drift cutoffs alongside the reference profile so future tooling can
  share one source of truth instead of hardcoding `0.10`/`0.25` in two places.
- **Serving latency benchmark** (`Proposed`) — an offline `benchmark` command
  that times batch scoring for representative batch sizes and reports
  throughput, giving operators evidence for capacity planning without touching
  production traffic.
- **First PyPI release** (`Proposed`) — link the trusted publisher on pypi.org
  (see Phase 1), publish a release candidate to TestPyPI, then cut the first
  real release once the dry-run workflow is green.

## Contributing to the roadmap

Open an issue or a pull request that references the relevant phase item.
Proposing a new item is welcome; keep it scoped to this project's stated
mission rather than general production-readiness concerns already assigned
to the deploying operator in SECURITY.md.
