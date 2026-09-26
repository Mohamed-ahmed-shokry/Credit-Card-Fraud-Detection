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
Phases 2 through 22 complete. Phase 1's workflows are in place, but the first
real PyPI release still requires maintainer-side trusted-publisher setup. The
shipped system covers leakage-safe training, multiple calibrated estimators,
threshold and calibration analysis, label-delay temporal gaps, drift surveillance,
promotion evidence, artifact integrity and lineage validation, signed release
artifacts and rotation-aware admission, online scoring, structured audit event
export, an isolated explanation provider boundary with deterministic fallback,
persisted lineage, HTML compliance reports, historical audit log replay, degraded
serving fallback guardrails, an automated champion-challenger retraining pipeline,
edge runtime export, distributed telemetry, continuous operational
surveillance, and an operational serving contract with liveness/readiness probes,
deployable reference middleware, and an optional scoring concurrency cap.

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

## Phase 18 — Artifact Trust and Deployment Admission (Done)

### Objective

Turn the existing integrity and validation reports into verifiable deployment
evidence. Operators must be able to sign a validated artifact attestation with
an Ed25519 key, verify it against an independently trusted public key, and use
that result as an explicit admission decision without changing model scoring.

### Scope

- **Signing primitive** — add a small PyNaCl-backed Ed25519 boundary with
  self-describing PEM key loading, raw public-key encoding, canonical JSON
  signing, and actionable failures. Private keys are accepted only as explicit
  signing inputs and are never written into artifacts or logs.
- **Attestation signatures** — extend the existing digest attestation with a
  versioned signature envelope covering the complete digest-bearing payload,
  while preserving verification of unsigned legacy attestations.
- **CLI workflows** — add `generate-signing-key` for controlled keypair
  creation, add signing options to `validate-artifact`, and add
  `verify-attestation` with required public-key and passed-status checks for
  deployment admission.
- **Deployment guidance** — document key custody, rotation, public-key pinning,
  CI admission behavior, failure modes, and the distinction between artifact
  integrity hashes and authenticity signatures.

### Acceptance criteria

- Ed25519 keypairs can be generated into caller-selected files with overwrite
  protection and never expose private key material in command output.
- A valid strict artifact validation can emit a signed attestation whose
  signature covers the canonical payload and whose embedded public key is
  independently checkable.
- Signature verification rejects modified payloads, wrong public keys,
  malformed envelopes, unsupported algorithms, and unsigned attestations when
  `--require-signature` is supplied.
- `verify-attestation` exits non-zero for failed validation status, invalid
  digest, invalid signature, or an untrusted public key, and emits a concise
  machine-readable result.
- Existing unsigned attestation verification and all model/API/edge behavior
  remain backward compatible.
- Focused unit/CLI tests cover key generation, permissions-safe output,
  canonical signing, tampering, wrong-key rejection, legacy compatibility, and
  admission exit codes.
- README, SECURITY, ARCHITECTURE, CHANGELOG, and this roadmap document the
  trust model and deployment workflow.
- Full quality gates, package build, dependency audit, and available smoke
  checks pass before this phase is marked `Done`.

### Explicit exclusions

This phase does not provide a hosted key-management service, automatic key
rotation, remote signature transparency logs, hardware-backed signing, or
automatic deployment orchestration. Operators remain responsible for protecting
private keys, distributing and pinning public keys, and integrating the CLI
admission result into their deployment system.

### Delivery record

- Added PyNaCl-backed Ed25519 key generation, PEM loading, canonical JSON
  signing, trusted public-key matching, and actionable verification failures.
- Extended validation attestations with an optional signature envelope while
  preserving digest verification for unsigned legacy attestations.
- Added `generate-signing-key`, signed `validate-artifact`, and fail-closed
  `verify-attestation` CLI workflows with overwrite protection and machine-readable
  admission results.
- Added focused signing, model, and CLI tests for key generation, tampering,
  wrong keys, malformed envelopes, legacy compatibility, and admission exits.
- Updated README, SECURITY, ARCHITECTURE, and CHANGELOG with key custody,
  trust-anchor, migration, and deployment guidance.
- Final validation: 461 tests passed with 97.45% branch coverage; Ruff formatting
  and linting passed; strict mypy passed; package build and Twine checks passed;
  isolated project dependency audit passed after upgrading the temporary audit
  environment's pip to 26.2.1.
- Docker validation remains unavailable in this environment because the Docker
  executable is not installed; CI remains responsible for container smoke checks.

## Phase 19 — Key Rotation and Automated Admission Enforcement (Done)

### Objective

Make signed artifact trust operable beyond a single pinned key. Operators must
be able to introduce and revoke verification keys without rewriting attestations,
and configured API/container startup must reject an artifact before model
deserialization when its signed validation evidence is absent or untrusted.

### Scope

- **Versioned trust bundles** — add a validated JSON trust-bundle format holding
  multiple Ed25519 public keys, stable key IDs, active/revoked status, and
  rotation metadata. Add CLI workflows to create a bundle and atomically add or
  revoke keys without exposing private material.
- **Rotation-aware signatures** — include the signing key ID in new signature
  envelopes, preserve verification of Phase 18 envelopes without key IDs, and
  select the trusted bundle key by ID. Revoked, unknown, mismatched, and
  ambiguous keys must fail closed.
- **Serving admission gate** — add optional `create_app`, `serve`, and
  environment configuration for an attestation plus trust bundle. Verify the
  digest, `PASSED` status, signature, and key status before loading the model;
  reject partial configuration and startup verification failures.
- **CI and deployment integration** — add a GitHub Actions admission smoke job,
  Compose/Docker configuration examples, and operator documentation for staged
  rotation: add new key, deploy trust bundle, sign with new key, revoke old key.

### Acceptance criteria

- Trust bundles reject malformed schemas, duplicate IDs, duplicate public keys,
  invalid key material, unknown statuses, and bundles with no active key.
- A bundle can be generated, atomically rotated by adding a key, and rotated by
  revoking an existing key; output overwrite protection remains explicit.
- New attestations identify their signing key; verification accepts active bundle
  keys and rejects revoked or unknown keys, wrong embedded IDs, and Phase 18
  signature envelopes only when the required legacy single-key mode is used.
- `verify-attestation` supports either the existing single public key or a trust
  bundle and emits machine-readable key ID/status details with non-zero admission
  exits on failure.
- Configured `create_app`/`serve` startup verifies admission before model loading,
  requires attestation and trust-bundle paths together, and leaves existing
  unconfigured development behavior unchanged.
- CI executes a real generate, sign, rotate, and verify admission workflow on a
  supported Python version; Compose documents the read-only attestation and
  trust-bundle mounts.
- Focused unit/API/CLI/workflow tests cover rotation, revocation, legacy mode,
  startup failure, startup success, and model-load ordering.
- README, SECURITY, ARCHITECTURE, CHANGELOG, and roadmap progress document key
  rotation and enforcement boundaries.
- Full quality gates, package build, dependency audit, and available smoke checks
  pass before this phase is marked `Done`.

### Explicit exclusions

This phase does not implement a hosted KMS, remote trust-bundle distribution,
hardware-backed keys, transparency logs, or automatic key generation in
production deployments. Private-key custody, bundle distribution, and rotation
authorization remain operator responsibilities.

### Delivery record

- Added schema-validated, atomic Ed25519 trust bundles with stable key IDs,
  active/revoked lifecycle status, duplicate detection, and staged add/revoke
  rotation operations.
- Added key IDs to new signature envelopes while preserving the Phase 18
  single-public-key verification path; bundle verification fails closed for
  unknown, revoked, mismatched, and legacy-without-key-ID signatures.
- Added `generate-trust-bundle`, `rotate-trust-bundle`, bundle-aware
  `verify-attestation`, and startup admission enforcement before API model
  deserialization, including `serve` and environment configuration.
- Added Compose read-only mount guidance, a GitHub Actions generate/sign/rotate/
  revoke admission smoke job, and operator documentation across README,
  SECURITY, ARCHITECTURE, and CHANGELOG.
- Added trust, CLI, API, and workflow coverage for rotation, revocation, malformed
  bundles, legacy mode, startup success/failure, and model-load ordering.
- Final validation: 476 tests passed with 97.29% branch coverage; Ruff checks and
  formatting passed; strict mypy passed; package build and Twine checks passed;
  isolated project dependency audit reported no known vulnerabilities; the local
  generate/sign/rotate/revoke admission smoke sequence passed.
- Docker validation remains unavailable in this environment because the Docker
  executable is not installed; CI remains responsible for container smoke checks.

## Phase 20 — Reproducible Release Engineering (Done)

### Objective

Make the package distribution path reliable enough for every verified commit and
for the first real release. Release workflows must build the same artifacts they
publish, produce their promised SBOM, tolerate the repository's immutable
`0.1.0` development version on repeated TestPyPI runs, and prove that the wheel
works outside the source checkout.

### Scope

- **Release workflow repair** — fix the currently failing TestPyPI and release
  SBOM steps, make missing SBOM output fail clearly, and keep the TestPyPI dry run
  idempotent when an unchanged version already exists.
- **Action dependency maintenance** — consume the verified Dependabot updates for
  `actions/download-artifact` and `actions/upload-artifact`, keeping the release
  workflow on the current supported major versions.
- **Wheel installation smoke test** — install the built wheel into a clean virtual
  environment in CI, run the installed `fraud-detect --version` entry point from
  outside the repository, and fail if the package is only working because of the
  source checkout.
- **Release documentation** — align README, CONTRIBUTING, SECURITY, CHANGELOG,
  and this roadmap with the repaired workflow, SBOM location, repeat-upload
  behavior, and maintainer-only trusted-publisher setup.

### Acceptance criteria

- The TestPyPI workflow creates and validates its SBOM output before invoking the
  publisher; the release workflow applies the same guarantee and uploads a
  non-empty SBOM artifact.
- Re-running the TestPyPI workflow for an already published `0.1.0` artifact does
  not fail solely because the file exists, while new build and metadata failures
  remain fatal.
- The release workflow uses `actions/download-artifact@v8` and
  `actions/upload-artifact@v7`, and both existing Dependabot action updates are
  resolved without bypassing CI.
- CI installs the wheel into a clean environment outside the checkout and the
  installed CLI reports the package version successfully.
- Build, Twine metadata, SBOM generation, and package-install smoke behavior are
  covered by workflow checks or deterministic local equivalents.
- README, CONTRIBUTING, SECURITY, CHANGELOG, and this roadmap accurately describe
  the release path and identify trusted-publisher account linking as the only
  maintainer-side prerequisite for an external PyPI/TestPyPI upload.
- Full project quality gates pass and every implementation/documentation change is
  committed and pushed before this phase is marked `Done`.

### Explicit exclusions

This phase does not create or configure PyPI/TestPyPI accounts, trusted publishers,
GitHub environments, or release credentials. Those require maintainer ownership of
external services. It does not change the package version, publish a real release,
or introduce a dependency lockfile; dependency freshness remains Dependabot's job.

### Delivery record

- Repaired TestPyPI and release SBOM generation by creating the output directory,
  requiring a non-empty CycloneDX file, and failing before publication when the
  inventory is missing. TestPyPI repeated uploads now use `skip-existing` for the
  immutable development version.
- Updated release artifact actions to `download-artifact@v8` and
  `upload-artifact@v7`; the two superseded Dependabot PRs were resolved after the
  changes passed CI.
- Added a CI wheel-install smoke gate that builds the package, installs the wheel
  into a clean environment outside the checkout, and verifies the installed
  `fraud-detect --version` entry point against package metadata.
- Updated README, CONTRIBUTING, SECURITY, ARCHITECTURE, CHANGELOG, and the source
  module map with SBOM, repeat-upload, wheel verification, and trusted-publisher
  guidance.
- Final validation: 476 tests passed with 97.29% branch coverage; Ruff checks and
  formatting passed; strict mypy passed; package build and Twine checks passed;
  isolated project dependency audit reported no known vulnerabilities; the local
  wheel-install smoke and 133402-byte CycloneDX SBOM generation checks passed; and
  remote CI run `35804224031` passed Python 3.12/3.14 quality, admission, container,
  and wheel smoke jobs.
- The pushed TestPyPI run reached and passed build, metadata, wheel installation,
  and SBOM generation, then stopped at the external OIDC publisher with
  `invalid-publisher`; configuring the TestPyPI trusted publisher remains the
  documented maintainer-only prerequisite.

## Phase 21 — Operational Serving Contract (Done)

### Objective

Make the serving HTTP surface match what SECURITY.md, README, and container
tooling already claim. Operational probes and scrapes must work without gateway
credentials, dedicated liveness/readiness endpoints must distinguish process
health from service readiness, the optional API-key and rate-limit middleware
must be usable from `serve` and the container without Python embeds, and every
documented run command must actually work.

### Scope

- **Operational endpoint exemption** — exempt `GET /health`, `GET /metrics`,
  `GET /live`, and `GET /ready` from the optional API-key middleware and from
  per-client rate limiting so probes, scrapes, and healthchecks are never
  rejected with `401`/`429` when those defenses are enabled, matching
  [SECURITY.md](SECURITY.md)'s statement that operational endpoints are
  unauthenticated.
- **Liveness and readiness probes** — add `GET /live` (process is accepting
  requests) and `GET /ready` (model loaded and scoring path able to serve);
  readiness returns `503` when the model is absent or the circuit is open with
  `fallback_mode=raise`, and `200` with health detail otherwise (including
  degraded-but-serving states). Keep `/health` behavior for existing callers.
- **Deployable middleware configuration** — resolve optional API keys and rate
  limits from `FRAUD_API_KEYS`, `FRAUD_RATE_LIMIT_REQUESTS`, and
  `FRAUD_RATE_LIMIT_WINDOW_SECONDS`, and expose matching `serve` options, so
  container and CLI deployments can enable the reference defenses without
  calling `create_app` from Python.
- **Container and CI probes** — point the Docker `HEALTHCHECK` at readiness,
  add a Compose healthcheck, and extend the CI container smoke to verify
  liveness and readiness endpoints.
- **Run-command and documentation repair** — replace the broken
  `uvicorn fraud_detection.api:app` README example with the working factory
  invocation, and update README, SECURITY, ARCHITECTURE, CHANGELOG, and this
  roadmap for probes, exemptions, and configuration.

### Acceptance criteria

- With `api_keys` configured, unauthenticated `GET /health`, `/metrics`,
  `/live`, and `/ready` are not rejected with `401`, while `POST /v1/predict`
  without a key still returns `401`.
- With rate limiting configured, operational endpoints do not consume the
  client budget and never return `429`; scoring endpoints still return `429`
  over the limit with `Retry-After` and `X-RateLimit-*` headers.
- `GET /live` returns `200` with the service version whenever the app accepts
  requests.
- `GET /ready` returns `200` with model/health detail when scoring can proceed
  (including degraded fallback serving), and `503` when the model is missing or
  the circuit is open with `fallback_mode=raise`.
- `FRAUD_API_KEYS` and rate-limit environment variables, and the corresponding
  `serve` options, enable the middleware end to end; invalid configuration is
  rejected clearly rather than silently ignored.
- The README documents working `uvicorn --factory
  fraud_detection.api:app_from_environment` startup and the probe endpoints;
  SECURITY.md remains accurate for operational endpoint exposure.
- Docker `HEALTHCHECK` and Compose healthcheck probe readiness; CI container
  smoke verifies `/live` and `/ready`.
- Full test suite, Ruff format/check, strict mypy, package build, and Twine
  checks pass; every change is committed and pushed before the phase is marked
  `Done`.

### Explicit exclusions

This phase does not add TLS, replace the gateway with a real auth product,
publicize OpenAPI/docs endpoints, change scoring request/response contracts,
bump the package version, publish a release, or configure external publishers.
Authentication, durable rate limiting, and network isolation remain the
deploying operator's responsibility as SECURITY.md already states.

### Delivery record

- Exempted `/health`, `/metrics`, `/live`, and `/ready` from the optional
  API-key and per-client rate-limit middleware, matching SECURITY.md's statement
  that operational endpoints are unauthenticated and keeping probes/scrapes from
  receiving `401`/`429` when those defenses are enabled.
- Added `GET /live` (always `200` while serving) and `GET /ready` (`200` with
  model/health detail when scoring can proceed, `503` when the model is missing
  or the circuit is open with `fallback_mode=raise`), with OpenAPI responses and
  failure-path tests; `/health` behavior is unchanged for existing callers.
- Made the reference middleware deployable without Python embeds: API keys and
  rate limits resolve from `FRAUD_API_KEYS`, `FRAUD_RATE_LIMIT_REQUESTS`, and
  `FRAUD_RATE_LIMIT_WINDOW_SECONDS` with clear rejection of invalid values, and
  `fraud-detect serve` gained `--api-key`, `--rate-limit-requests`, and
  `--rate-limit-window-seconds` options that flow into `create_app`.
- Pointed the Docker `HEALTHCHECK` at `/ready`, added a matching Compose
  healthcheck, and extended the CI container smoke to verify `/ready`, `/live`,
  `/health`, and `/metrics`.
- Fixed the broken `uvicorn fraud_detection.api:app` README example to the
  working `uvicorn --factory fraud_detection.api:app_from_environment` form and
  updated README, SECURITY, ARCHITECTURE, and CHANGELOG for probes, exemptions,
  and configuration.
- Final validation: 492 tests passed with 97.36% branch coverage; Ruff format
  and check passed; strict mypy passed on 14 source files; package build and
  Twine checks passed; remote CI run `35953894857` succeeded. The TestPyPI dry
  run still stops only at the external trusted-publisher step (`environment`
  missing), the documented maintainer-only prerequisite.

## Phase 22 — Serving Concurrency and Credential Hardening (Done)

### Objective

Keep the async scoring service honest under load. Scoring is CPU- and
I/O-bound work that currently runs directly on the event loop inside
`async def` handlers, so one slow batch freezes probes, middleware, and every
other in-flight request. Verification of that path, add an optional reference
backpressure guard so operators can shed overload instead of queueing
unboundedly, and make API-key verification timing-safe.

### Scope

- **Event-loop offload** — execute `score_frame` (model inference, fallback,
  explanation providers) and the synchronous audit-sink emit through Starlette's
  threadpool in both `POST /v1/predict` and `POST /v1/score`, so liveness,
  readiness, health, metrics, and concurrent scoring requests stay responsive
  while a batch is in flight. Shadow evaluation stays in background tasks.
- **Timing-safe API-key verification** — replace set membership (`in`) with a
  `secrets.compare_digest` check across all configured keys without early exit,
  preserving current accept/reject semantics including missing or empty headers.
- **Optional concurrency guard** — a `max_concurrent_scoring` cap (default
  disabled), configurable through `create_app`, `FRAUD_MAX_CONCURRENT_SCORING`,
  and `fraud-detect serve --max-concurrent-scoring`. When the cap is reached,
  scoring endpoints return `503` with `Retry-After` while operational endpoints
  remain unaffected. This is defense in depth, not a substitute for gateway
  concurrency limits.
- **Documentation** — README (serving/concurrency notes and configuration),
  SECURITY (timing-safe comparison and backpressure framing), ARCHITECTURE
  (serving boundary), CHANGELOG, and this roadmap.

### Acceptance criteria

- While a scoring request is blocked in a deliberately slow model, `GET /live`
  (and `GET /ready`) complete promptly, proving scoring no longer blocks the
  event loop.
- Two scoring requests run concurrently through the threadpool (their
  slow-model sections overlap) instead of serializing on the event loop.
- API-key middleware still accepts valid keys and rejects invalid or missing
  keys; verification goes through `secrets.compare_digest` over every
  configured key.
- With `max_concurrent_scoring=1` (or `FRAUD_MAX_CONCURRENT_SCORING=1`), a
  second concurrent scoring request receives `503` with `Retry-After` while
  `/health`/`/live` still return `200`; the default configuration caps nothing
  and behaves as before.
- The cap is settable from `serve` and the environment with clear validation
  of invalid values.
- Scoring response contracts, audit event content, circuit-breaker behavior,
  and rate-limit/API-key exemptions for operational endpoints are unchanged.
- Full test suite, Ruff format/check, strict mypy, package build, and Twine
  checks pass; every change is committed and pushed before the phase is marked
  `Done`.

### Explicit exclusions

This phase does not add TLS, replace the gateway with a real auth product,
change scoring request/response schemas, introduce per-feature CPU quotas or
autoscaling, or publish a release. Gateway authentication, durable rate
limiting, and network-level concurrency limits remain the deploying
operator's responsibility as SECURITY.md already states.

### Delivery record

- Moved `score_frame` and the synchronous audit-sink `emit` off the event loop in
  `POST /v1/predict` and `POST /v1/score` via Starlette's threadpool, so probes,
  health, metrics, and concurrent scoring requests stay responsive while a batch
  is in flight; shadow evaluation still runs as a background task.
- Replaced API-key set membership with a `secrets.compare_digest` check across
  every configured key with no early exit, preserving accept/reject semantics for
  valid, invalid, missing, and empty headers, and added a test asserting the
  comparison runs once per configured key.
- Added the optional `max_concurrent_scoring` overload guard, settable through
  `create_app`, `FRAUD_MAX_CONCURRENT_SCORING`, and
  `fraud-detect serve --max-concurrent-scoring`, with clear rejection of invalid
  values. When the cap is reached, scoring endpoints return `503` with
  `Retry-After` while `/health`, `/live`, and `/ready` are unaffected; the
  default configuration caps nothing and behaves exactly as before.
- Added tests for probe responsiveness during blocked scoring, threadpool
  overlap of two concurrent scoring requests, a slow audit emit, the `503` shed
  path, environment resolution, and invalid-cap rejection.
- Documented the threadpool boundary, the timing-safe comparison, and the
  overload-shedding framing in README, SECURITY (defense in depth, not a
  substitute for gateway concurrency limits), ARCHITECTURE, and CHANGELOG.
- Final validation: 500 tests passed with 97.32% total branch coverage; Ruff
  format and check passed; strict mypy passed on 14 source files; package build
  and Twine checks passed.

## Phase 23 — Serving Correctness and Defense-in-Depth Hardening (In progress)

### Objective

Make the serving surface report failures honestly and keep the optional defenses
effective under load, failure, and hostile input. An audit of the Phase 21/22
serving layer found, and reproduced against the running app, that several failure
paths currently fabricate results, hide from operators, or contradict documented
contracts:

- A request whose feature schema does not match the artifact is treated as a
  *model* failure. It records a circuit-breaker failure and, under any non-`raise`
  fallback, returns `200` with a fabricated constant score instead of `422`.
- A model that returns no probabilities raises `500` *after* the fallback block
  and records a breaker *success*, so the fallback policy is bypassed while the
  breaker looks healthy.
- Unhandled `500` responses carry no `X-Request-ID`, `X-Process-Time-Ms`, or
  `traceparent`, contradicting the README's "every HTTP response" guarantee and
  making the `request_failed` log line uncorrelatable with the client.
- Shadow evaluation runs as a `BackgroundTasks` entry, which still holds the ASGI
  request open: a 1s shadow model makes a 0s primary request take 1.017s,
  contradicting ARCHITECTURE's and README's "never blocks" / "zero impact" claims.
- Rate limiting runs inside API-key verification, so unauthenticated traffic
  (including key brute force) consumes no budget at all.
- The chaos header `X-Simulate-Degraded` is compiled in and unauthenticated: any
  client can force fallback scoring (`0.3335` model score → `0.5` fabricated) and
  there is no switch to disable it.
- Prometheus `path` labels come from the raw request path, so any client can
  create unbounded series (`/nope-<random>`), which is both a memory-growth path
  and a caller-controlled label.
- The circuit breaker is mutated from threadpool workers without a lock, an
  injected breaker's `on_trip` callback is silently replaced, and
  `latency_budget_ms` is silently dropped when a breaker instance is injected.
- `degraded_mode=True` with `fallback_mode="raise"` is a no-op that still reports
  `status: degraded` on `/health` while scoring the primary model.
- `/health` raises `500` when the model is absent while `/ready` correctly `503`s.
- The in-memory rate limiter never evicts idle client keys, so memory grows with
  the number of distinct client IPs seen for the process lifetime.

### Scope

- **Shared scoring path** — extract the duplicated `POST /v1/predict` /
  `POST /v1/score` guardrail sequence (state reads, concurrency cap, threadpool
  scoring, audit event build/emit) into one internal coroutine and share the
  health/readiness status computation, with no behavior change.
- **Caller-input errors are not model failures** — validate request features
  against the artifact schema before the circuit breaker is consulted; a
  `ModelArtifactError` caused by caller input is re-raised (existing `422`
  handler) and never records a breaker failure or activates the fallback policy.
- **Model contract violations use the fallback policy** — missing or
  length-mismatched probabilities are detected inside the guarded block, so they
  record a breaker failure and route through the configured fallback instead of
  raising `500` after the fallback decision was made.
- **Circuit-breaker hardening** — guard breaker state with a lock now that
  scoring runs concurrently, chain rather than replace a caller-supplied
  `on_trip`, honor `latency_budget_ms` with an injected breaker, and parse the
  circuit-breaker environment variables through named validation errors like
  every other resolver.
- **Honest error responses** — unhandled `500`s return a sanitized JSON body
  carrying `X-Request-ID`, `X-Process-Time-Ms`, and `traceparent`, so the
  documented correlation contract holds for failures too.
- **Chaos header opt-in** — `X-Simulate-Degraded` is honored only when
  `FRAUD_ENABLE_CHAOS_HEADER` is enabled (default off), and `degraded_mode=True`
  combined with `fallback_mode="raise"` is rejected at startup instead of
  silently doing nothing.
- **Shadow evaluation off the request lifecycle** — run shadow evaluation on a
  detached, reference-held task so it cannot extend the request or hold the
  connection, matching the documented guarantee.
- **Rate limiting that meters unauthenticated traffic** — order the rate-limit
  middleware outside API-key verification, and evict idle client keys so the
  in-memory window cannot grow without bound.
- **Bounded, useful metrics** — label request metrics with the matched route
  template instead of the raw path, and add counters for shed and rate-limited
  requests.
- **Operational endpoint agreement** — `/health` returns `503` with a reason
  instead of raising when the model is absent, and shadow-model metadata is read
  defensively.
- **CI container smoke** — exercise `/v1/predict` and `/v1/score` in the image
  smoke test, not only the operational endpoints.
- **Documentation** — README (correlation contract, chaos-header switch, rate
  limiting scope, metric labels, the previously undocumented
  `FRAUD_CIRCUIT_BREAKER_RECOVERY_TIMEOUT`), SECURITY, ARCHITECTURE, CHANGELOG,
  and this roadmap.

### Acceptance criteria

- A request with an unknown, missing, or extra feature returns `422` and leaves
  the circuit breaker untouched (no failure recorded, no fallback activated),
  while a genuine model exception still trips the breaker as before.
- With `fallback_mode="constant"`, a model returning no probabilities (or a
  wrong-length array) activates the fallback with `fallback_applied=true` and
  records a breaker failure instead of returning `500`.
- An unhandled `500` response carries `X-Request-ID` (echoing a caller-supplied
  one), `X-Process-Time-Ms`, and `traceparent`.
- A scoring request whose primary model returns immediately completes in well
  under the duration of a deliberately slow shadow model.
- `X-Simulate-Degraded` is ignored unless `FRAUD_ENABLE_CHAOS_HEADER` is enabled;
  with it enabled the documented simulation still works; `degraded_mode=True`
  plus `fallback_mode="raise"` fails fast with a clear configuration error.
- With `api_keys` and `rate_limit_requests=1` configured, unauthenticated
  requests consume the client budget and eventually return `429` with
  `Retry-After`, while `/health`, `/live`, and `/ready` remain exempt.
- A request to an undeclared path does not create a new `path` label series; all
  `404`s share one label. `fraud_scoring_rejected_total` counts shed and
  rate-limited requests.
- The circuit breaker records state transitions safely under concurrent scoring;
  an injected breaker's `on_trip` still runs alongside the metrics counter; and
  `latency_budget_ms` applies to an injected breaker.
- `GET /health` returns `503` with `{"detail": "Model not loaded."}` when the
  model is absent, matching `/ready`.
- Scoring request/response schemas, audit event content, `/live`, `/ready`,
  `/metrics` success shape, and the Phase 22 concurrency cap are unchanged.
- Full test suite, Ruff format/check, strict mypy, package build, and Twine checks
  pass; every change is committed and pushed before the phase is marked `Done`.

### Explicit exclusions

This phase does not move the request-body size limit ahead of authentication
(the outermost middleware buffers before verifying credentials; that needs a
pure-ASGI middleware rewrite), add TLS, replace the gateway with a real auth
product, change scoring request/response schemas, add durable or distributed
rate limiting, change the package version, or publish a release. Findings from
the same audit outside the serving layer (compliance-report escaping, audit
replay validation, explanation-provider timeouts, artifact validation
robustness, and CLI documentation errors) are recorded as Phase 24.

### Delivery record

Filled in when the phase completes.

## Phase 24 — Evidence Rendering, Replay, and Documentation Correctness (Proposed)

Recorded from the same audit that scoped Phase 23, so the verified findings are
not lost. Every item below was reproduced against the current code.

- **Compliance report escaping** — `reporting.py` interpolates drift
  `overall_status` and the per-feature status class into HTML unescaped, so a
  bundle carrying crafted status text injects script or event-handler markup,
  contradicting README's "escapes evidence values before rendering".
- **Audit replay validation** — `replay_audit_log` does not validate `tolerance`
  or `max_discrepancies_to_record` (`tolerance=nan` silently disables score
  comparison), crashes on a non-numeric `fraud_probability` or a null threshold,
  silently truncates mismatched feature/prediction lists, and reports
  `status="MATCH"` when zero transactions were compared.
- **Explanation-provider timeouts** — `timeout_seconds` does not bound latency
  because the threadpool context manager waits on shutdown, contradicting
  README's "strict timeouts"; the `CostController` token budget is never
  enforced, and `ExplanationRequest.decision` is ignored by the template
  provider.
- **Artifact validation robustness** — `validate_artifact` can raise instead of
  reporting on a corrupted `model.joblib` with a matching manifest digest, and
  raises `AttributeError` when `manifest["files"]` is a list.
- **Streaming profiler validation** — `StreamingProfile.from_dict` performs no
  shape checks (missing `edges`, mismatched `counts` length, non-finite `mean`),
  and `update` commits earlier features when a later feature fails validation.
- **Local explanation consistency** — `explain_local` is batch-dependent for
  estimators without native importances, and for the default sigmoid-calibrated
  model the contributions do not sum to the served score's log-odds.
- **Documentation errors** — the README `retrain`, and `simulate-drift` examples
  use options that do not exist; `generate-trust-bundle` is undocumented, so the
  key-rotation runbook cannot be followed; `retrain --promote` reports
  `PROMOTED` while silently not promoting when the champion is a file.
- **Test debt** — `cli.py` sits at 92% coverage with untested failure paths in
  `retrain`, `export-edge --max-error`, and the signing/attestation commands,
  against CONTRIBUTING's rule that tests must cover failure behavior.

## Contributing to the roadmap

Open an issue or a pull request that references the relevant phase item.
Proposing a new item is welcome; keep it scoped to this project's stated
mission rather than general production-readiness concerns already assigned
to the deploying operator in SECURITY.md.
