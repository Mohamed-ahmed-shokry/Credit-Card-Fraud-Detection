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

## Current state

The project is a production-grade reference implementation with the implementation work in
Phases 2 through 15 complete. Phase 1's workflows are in place, but the first
real PyPI release still requires maintainer-side trusted-publisher setup. The
shipped system covers leakage-safe training, multiple calibrated estimators,
threshold and calibration analysis, label-delay temporal gaps, drift surveillance,
promotion evidence, artifact integrity and lineage validation, signed release
artifacts, online scoring, structured audit event export, an isolated explanation
provider boundary with deterministic fallback, persisted lineage, HTML compliance
reports, historical audit log replay, degraded serving fallback guardrails,
an automated champion-challenger retraining pipeline, and machine-readable
cryptographic lineage attestation manifests.

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

- **Calibration analysis report** (`Done`) — a `calibration` command reporting
  reliability bins plus a Brier-score decomposition for held-out labeled data,
  so operators can judge whether predicted probabilities mean what they say
  before wiring them to thresholds.
- **Drift-alert thresholds in the model card** (`Done`) — the PSI
  warning/drift cutoffs travel in `drift_thresholds` next to the reference
  profile, and drift reports honor (and echo) them instead of hardcoding
  `0.10`/`0.25` in two places.
- **Serving latency benchmark** (`Done`) — an offline `benchmark` command
  that times batch scoring for representative batch sizes and reports
  throughput, giving operators evidence for capacity planning without touching
  production traffic.
- **First PyPI release** (`In progress`) — the TestPyPI dry run is live and a
  release checklist is documented in `CONTRIBUTING.md`; remaining is a
  maintainer action: link the trusted publisher on pypi.org (see Phase 1),
  then cut the first real release once the dry-run workflow is green.

## Phase 6 — Evaluation rigor and serving polish

- **Repeated-seed stability report** (`Done`) — a `stability` command that
  repeats training with successive seeds and reports mean/standard-deviation
  per test metric, so split luck is visible before trusting one `train` run.
- **Rate-limit client headers** (`Done`) — `Retry-After`, `X-RateLimit-Limit`,
  and `X-RateLimit-Remaining` on `429` responses (plus limit/remaining on
  allowed responses) so clients can back off before hitting the cap.
- **Threshold tradeoff report** (`Done`) — a `thresholds` command scoring
  candidate thresholds on held-out labeled data, reporting precision/recall/F1
  and expected cost per candidate next to the model's tuned operating point,
  with costs defaulting to the training policy.
- **Artifact retention policy** (`Done`) — documented in the README: keep the
  serving artifact plus the two most recent predecessors so `compare`,
  `stability`, and `drift` stay reproducible.

## Phase 7 — Decision-policy flexibility

- **Predict-time threshold override** (`Done`) — an opt-in `--threshold`
  flag on `predict` (and matching API field) for audit and backtest scenarios,
  recorded next to the tuned `model_threshold` in the CLI summary and API
  response so the two are never confused.
- **Cost-policy presets** (`Done`) — a named cost policy (`--cost-policy`)
  recorded as a `cost_policy` block in the model card, so `train`, `compare`,
  `stability`, and `thresholds` can reference one shared business-cost
  definition instead of repeating raw weights. Ad-hoc overrides are labeled
  `"custom"` in reports.

## Phase 8 — Operations runbook and temporal rigor

- **Retraining runbook** (`Done`) — documented in the README: drift and
  calibration triggers, the promotion checklist (`compare` → `stability` or
  `rolling` → `calibration` → `thresholds` → `benchmark`), and rollback to
  retained artifacts.
- **Rolling-origin temporal evaluation** (`Done`) — a `rolling` command that
  trains, tunes, and tests on expanding chronological prefixes, reporting
  metric spread across origins the way `stability` does for random splits.

## Phase 9 — Promotion evidence and surveillance

- **Promotion bundle** (`Done`) — a `promote` command assembling the
  `calibration`, `thresholds`, `drift`, and `benchmark` evidence plus the
  model card summary into a single reviewable JSON document, matching the
  runbook's promotion checklist. Built on shared payload builders so the
  bundle and the standalone commands cannot drift apart.
- **Drift surveillance exit codes** (`Done`) — an opt-in `drift --fail-on`
  flag returning exit status 1 when the overall status reaches `warning` or
  `drifted`, for cron and scheduled-job alerting without parsing JSON (the
  report is still printed first).

## Phase 10 — Release hardening

- **Release SBOM** (`Done`) — the publish workflow installs the built wheel,
  inventories the shipping environment with `cyclonedx-bom`, and uploads the
  validated CycloneDX SBOM as a workflow artifact; the TestPyPI dry run
  exercises the same mechanism on every push so tool drift surfaces early.
- **Signed container images** (`Done`) — the release workflow publishes the
  API image to GHCR and signs it keylessly with Sigstore/cosign (no stored
  keys; least-privilege job permissions), with README verification
  instructions. Like every release-gated workflow here, the first real
  release exercises it.

## Phase 11 — Serving and data frontiers

- **Streaming prediction endpoint** (`Done`) — a single-transaction `POST /v1/score`
  route alongside the batch endpoint, returning the applied threshold, the
  model's tuned threshold, and the per-transaction prediction with optional
  explanations. Shares the scoring helper with the batch endpoint so both
  paths stay consistent.
- **Reference label-delay analysis** (`Done`) — documented in the README: how
  to reason about chargeback label delays when assembling held-out evaluation
  sets, since fraud labels arrive late and naive random splits overstate
  quality. The `stability` and `rolling` commands help quantify how much
  performance fluctuates across time windows.

## Phase 12 — Prompt engineering and governance

- **Per-transaction natural-language explanations** (`Done`) — `predict` and
  the API accept the historical `--explain-llm`/`explain_llm` flag and render a
  deterministic, offline rationale from local model contributions, citing the
  top contributing features and their direction. The response records the
  applied threshold and remains auditable without an external provider.
- **Model card drift alerts** (`Done`) — extend `drift` with an optional
  `--fail-on warning|drifted` flag and Slack/PagerDuty webhook options. Alerts
  include the top drifted features and their PSI values; deployment-specific
  minimum-feature policies remain an operator concern.

## Phase 13 — Model governance and compliance

- **Model versioning and lineage** (`Done`) — newly trained artifacts persist
  SHA-256 dataset and configuration fingerprints, package version, a combined
  content hash, and best-effort Git commit/origin provenance from CLI training.
  The `model-card` command exposes compact and verbose views while preserving
  loading compatibility for older artifacts.
- **Automated compliance reporting** (`Done`) — `compliance` renders a
  self-contained, escaped HTML report from a `promote` bundle, with optional
  stability evidence and artifact manifest hashes. It records evidence and
  explicitly does not make an automated promotion decision.

## Phase 14 — Evaluation realism and operational integration

The next phase closes the remaining gap between this reference workflow
and a deployment team's data/operations process without turning the project
into a general-purpose platform:

- **Label-delay-aware temporal gaps** (`Done`) — add a configured gap
  between training, validation, and test windows so chargeback latency is
  enforced by the split implementation rather than documented only in the
  runbook.
- **Artifact lineage validation command** (`Done`) — add a read-only
  validator that checks artifact integrity, runtime compatibility, lineage
  completeness, and report compatibility before a deployment job consumes an
  artifact.
- **Structured audit event export** (`Done`) — provide an opt-in JSONL
  sink for scoring and promotion events with redaction guarantees, leaving
  durable storage, retention, and access control to the deploying operator.
- **Optional explanation provider interface** (`Done`) — if external
  language models are added, isolate them behind an explicit provider boundary
  with timeouts, redaction, cost controls, and a deterministic fallback; the
  current offline template remains the default.

## Phase 15 — Continuous Surveillance, Audit Replay, and Serving Guardrails

This phase strengthens operational reliability, auditability, and automated model lifecycles:

- **Audit log replay and divergence backtesting** (`Done`) — a `replay-audit`
  command and replay engine to stream historical JSONL scoring audit logs through a
  model, reporting decision flips, score divergence rates, and maximum discrepancy metrics.
- **Serving guardrails and degraded-state fallback** (`Done`) — resilient runtime
  fallback policies (rule-based heuristic or constant-score fallback) in the FastAPI serving
  layer when estimators encounter runtime exceptions or when degraded operations are signaled.
- **Automated champion-challenger retraining pipeline** (`Done`) — a `retrain`
  command that ingests fresh labeled transactions with temporal gaps, trains a challenger,
  evaluates both champion and challenger on identical held-out test data, and assesses
  metric improvements against strict promotion guardrails.
- **Machine-readable lineage attestation export** (`Done`) — extend
  `validate-artifact` with `--attestation-output` to emit cryptographically verifiable,
  tamper-evident JSON attestation manifests for CI/CD gates and deployment admission controllers.

## Phase 16 — Dual-Window Streaming Surveillance, Traffic Shadowing, and Operational Circuit Breakers (Done)

This phase strengthens online operational safety, real-time diagnostic surveillance, and canary shadowing:

- **Dual-window streaming drift surveillance** (`Done`) — an `assess_multi_window_drift()`
  engine and `multi-window-drift` CLI command that computes short-window PSI (recent transactions)
  and long-window PSI against reference profiles, calculating drift velocity and acceleration to
  catch sudden distribution shocks before aggregate batch metrics trip.
- **Traffic shadowing & latency circuit breaker in FastAPI serving** (`Done`) — asynchronous
  challenger traffic shadowing via `shadow_model_path` without adding latency to primary response
  paths, paired with an automated `CircuitBreaker` that tracks consecutive failures and latency SLA
  breaches, automatically tripping to safe fallback modes (`constant` or `rule`) when models degrade.
- **Streaming distribution and quantile profiler** (`Done`) — an incremental
  `StreamingProfile` class and `stream-profile` CLI command to update reference profile bins and
  distribution statistics online from streaming CSV batches or JSONL audit logs without keeping
  historical transactions in memory.
- **Surveillance simulation and chaos harness** (`Done`) — a `simulate-drift` CLI command
  to inject synthetic distribution shifts (mean offsets, variance scaling, anomaly spikes) into
  validation datasets to test alerting webhooks, fallback behavior, and circuit breakers in staging.

## Phase 17 — Edge Runtime Optimization and Distributed Telemetry (Done)

### Objective

Make a validated model usable at constrained edge boundaries and make online
requests traceable across gateway and scoring services without imposing a new
mandatory runtime dependency or changing the default serving path.

### Scope

- **Quantized and pruned edge runtime export** — add an `export-edge` command
  and versioned JSON runtime format for the supported logistic-regression
  artifact contract. The format stores int8 weights, a quantization scale,
  optional coefficient pruning, the feature schema, scaler statistics, and
  decision threshold. A small runtime scorer validates the format without
  importing scikit-learn and reports source-model agreement when validation
  data is supplied.
- **Distributed OTLP trace propagation** — add W3C `traceparent` parsing and
  response propagation plus optional OTLP/HTTP span export for API request
  lifecycles. Export is disabled by default, injectable for tests, and must
  never make scoring fail when a collector is unavailable.
- **Integration and documentation** — expose configuration through
  `create_app`, `serve`, and environment variables; document the edge format,
  approximation limits, trace configuration, collector expectations, and
  security boundaries.

### Acceptance criteria

- `export-edge` rejects unsupported estimator/calibration contracts with an
  actionable error and produces a schema-versioned artifact for supported
  logistic models.
- The edge runtime validates schemas, scores finite numeric records, applies
  the persisted threshold, and records quantization/pruning metadata.
- Export can measure maximum and mean probability error against supplied
  validation data and refuses output when the configured error tolerance is
  exceeded.
- Valid incoming `traceparent` headers are continued, invalid headers create a
  fresh trace, and responses return a valid trace context.
- Optional OTLP export emits request spans with timing, route, status, and
  model attributes; exporter/network failures are isolated from requests.
- Focused unit/API/CLI tests cover success, unsupported contracts, malformed
  edge artifacts, trace propagation, exporter payloads, and collector failure.
- Full project quality gates, package build, documentation, and roadmap status
  are updated before the phase is marked `Done`.

### Explicit exclusions

This phase does not add ONNX/TensorFlow Lite conversion, automatic model
architecture rewriting, a mandatory OpenTelemetry SDK, a collector deployment,
or durable trace storage. Those require deployment-specific runtime and
infrastructure decisions and remain later work.

### Delivery record

- Delivered `fraud-detect export-edge`, the schema-versioned dependency-light int8
  runtime, source-model validation error reporting, pruning metadata, and strict
  contract rejection for calibrated and tree artifacts.
- Delivered W3C `traceparent` continuation/fresh-trace behavior, response headers,
  injectable exporters, optional asynchronous OTLP/HTTP JSON spans, `create_app`
  configuration, `serve` options, and environment configuration.
- Added focused edge, telemetry, API, and CLI tests covering success, malformed
  artifacts, unsupported contracts, schema failures, trace payloads, propagation,
  and collector failure isolation.
- Updated `README.md`, `ARCHITECTURE.md`, and `CHANGELOG.md` with usage, limits,
  security boundaries, and deployment expectations.
- Final validation: 451 tests passed with 97.90% branch coverage; Ruff formatting
  and linting passed; strict mypy passed; package build and Twine metadata checks
  passed; isolated project dependency audit passed after upgrading the audit
  environment's pip to 26.2.1.
- Docker validation could not run in this Windows environment because the Docker
  executable is unavailable; CI remains responsible for Compose, image, and
  container smoke validation.

## Contributing to the roadmap

Open an issue or a pull request that references the relevant phase item.
Proposing a new item is welcome; keep it scoped to this project's stated
mission rather than general production-readiness concerns already assigned
to the deploying operator in SECURITY.md.
