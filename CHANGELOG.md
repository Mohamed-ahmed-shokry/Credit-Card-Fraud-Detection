# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Operational serving probes: added `GET /live` and `GET /ready` endpoints,
  exempted `/health`, `/live`, `/ready`, and `/metrics` from the optional
  API-key and rate-limit middleware, and wired Compose and Docker healthchecks
  plus the CI container smoke to readiness.
- Deployable reference middleware configuration: API keys and rate limits can
  now be set through `FRAUD_API_KEYS`, `FRAUD_RATE_LIMIT_REQUESTS`,
  `FRAUD_RATE_LIMIT_WINDOW_SECONDS`, and matching `fraud-detect serve` options.
- Release workflow hardening: TestPyPI uploads tolerate repeated immutable-version
  runs, release and dry-run SBOM output is created and validated explicitly, and
  CI smoke-tests the built wheel from an isolated environment outside the checkout.
- Rotation-aware deployment admission: added validated trust bundles, add/revoke
  key rotation workflows, key IDs in new signatures, API/serve startup verification
  before model loading, Compose mount guidance, and a CI admission smoke job.
- Deployment artifact trust: added Ed25519 keypair generation, canonical signed
  validation attestations, and fail-closed `verify-attestation` admission checks
  for trusted public keys, passed artifact status, digest integrity, and signature
  authenticity.
- Dependency-light edge runtime export: added `fraud-detect export-edge` and a
  schema-versioned int8 JSON scorer for supported uncalibrated logistic artifacts,
  with optional coefficient pruning, validation-set error measurement, and strict
  output refusal when the configured probability tolerance is exceeded.
- Optional distributed tracing: added W3C `traceparent` propagation and best-effort
  asynchronous OTLP/HTTP request spans, configurable through `create_app`,
  `fraud-detect serve`, and OTEL/FRAUD environment variables while remaining disabled
  by default and isolated from scoring failures.
- Dual-window streaming drift surveillance: added `assess_multi_window_drift()` engine,
  `MultiWindowDriftReport` / `WindowFeatureDrift` data models, and `fraud-detect multi-window-drift`
  CLI command computing short-window PSI, long-window PSI, and drift velocity/acceleration
  with automated alerting webhooks for Slack and PagerDuty.
- Traffic shadowing and operational circuit breakers in FastAPI serving: added stateful
  `CircuitBreaker` (tracking consecutive failures and latency SLA budgets across CLOSED,
  HALF_OPEN, and OPEN states) with automated fallback execution, plus background non-blocking
  challenger traffic shadowing via `shadow_model_path`, `/health` telemetry, and Prometheus
  metrics (`fraud_shadow_evaluations_total`, `fraud_circuit_breaker_state`).
- Streaming distribution and quantile profiler: added online incremental `StreamingProfile`
  (Welford's algorithm for running means/variances, streaming histogram binning) and
  `fraud-detect stream-profile` CLI command supporting batch-by-batch updating from CSV chunks
  or JSONL audit logs without retaining raw historical transactions.
- Surveillance simulation and chaos testing harness: added `inject_drift()` engine and
  `fraud-detect simulate-drift` CLI command to inject deterministic distribution shifts
  (mean offsets, variance scaling, anomaly spikes) into transaction datasets for testing
  surveillance webhooks, fallback policies, and circuit breakers.
- Audit log replay and divergence backtesting: added `replay_audit_log()` engine
  and `fraud-detect replay-audit` CLI command, enabling stream processing of
  historical JSONL scoring audit logs, tolerance thresholding, decision-flip detection,
  and export of divergence reports.
- Serving guardrails and degraded-state fallback: implemented safe fallback policies
  in FastAPI serving (`constant`, `rule`, `raise`), degraded-mode runtime toggle via
  environment variable or `X-Simulate-Degraded` header, and Prometheus fallback counter
  metrics (`fraud_fallback_predictions_total`).
- Automated champion-challenger retraining pipeline: added `fraud-detect retrain`
  CLI command to train challengers on fresh data with temporal gaps, evaluate
  champion vs challenger on identical holdout data, verify minimum metric gains, and
  optionally promote winners to champion.
- Machine-readable lineage attestation manifests: added `generate_attestation()`,
  `verify_attestation()`, and `fraud-detect validate-artifact --attestation-output`
  with `--signer` support to export and verify tamper-evident, cryptographically
  digested SHA-256 JSON attestation manifests for CI/CD gates.
- Label-delay-aware temporal gaps: added `--temporal-gap` CLI option and
  `temporal_gap` parameter to `TrainingConfig`, `train_model`, and
  `split_dataset_temporally`, enforcing holdout window separation to reflect
  chargeback maturation latency.
- Read-only artifact lineage validation command (`fraud-detect validate-artifact`)
  and `validate_artifact()` helper, verifying manifest digests, runtime
  compatibility, lineage completeness, and report readiness, with an optional
  `--strict` flag to require Git commit provenance.
- Structured audit event export: opt-in thread-safe JSONL audit sink
  (`JsonlAuditSink`, `AuditEvent`, `NullAuditSink`) with strict payment card
  (Luhn-validated PAN) and sensitive key redaction guarantees, supported via
  `--audit-log` in `predict` and `promote` CLI commands and
  `FRAUD_AUDIT_LOG_PATH` in FastAPI serving.
- Isolated explanation provider interface (`ExplanationProvider` protocol,
  `TemplateExplanationProvider`, `ExternalExplanationProvider`) supporting
  timeouts, prompt redaction, cost controls (`CostController`), and
  deterministic fallback, with the offline template remaining default across
  the model, API, and CLI.
- Persisted model lineage with deterministic configuration and content hashes,
  package version, and best-effort Git commit/origin provenance from CLI
  training. The new `model-card` command exposes compact and verbose lineage
  views while keeping older artifacts loadable.
- A dependency-free `compliance` CLI command that renders promotion evidence,
  optional stability results, and artifact manifest hashes as escaped,
  self-contained HTML for human review.
- API support for the documented `explain_llm` request flag, returning an
  `explanation` field alongside the existing local contributions. The
  historical flag now uses deterministic model-derived contribution templates,
  not an unconfigured external service.
- Keyless-signed container images: the release workflow publishes the API
  image to GHCR with Sigstore/cosign signing (no stored keys), plus README
  verification instructions and a release-checklist confirmation step.
- CycloneDX SBOM generation for releases: the publish workflow installs the
  built wheel, inventories the shipping environment with `cyclonedx-bom`,
  and uploads the validated SBOM as a workflow artifact; the TestPyPI
  dry-run workflow exercises the same mechanism on every push.
- A `promote` CLI command assembling calibration, threshold, drift, and
  benchmark evidence plus the model card summary into one reviewable
  promotion bundle (assessment only: it states facts, not a verdict).
- A `drift --fail-on warning|drifted` flag exiting 1 when the overall status
  reaches the cutoff, for cron and scheduled-job alerting without parsing
  JSON (the report is still printed first).
- Shared thresholds/benchmark payload builders used by both the standalone
  commands and `promote`, removing a duplicated scoring pass.
- A `predict --threshold` override (CLI) and `threshold` request field (API)
  for audit/backtest scoring, always recorded next to the tuned
  `model_threshold` so overridden decisions are never confused with it.
- A `POST /v1/score` single-transaction endpoint (API) returning the applied
  threshold, the model threshold, and the per-transaction prediction with
  optional explanations, sharing the scoring logic with the batch endpoint.
- A `fraud-detect predict --threshold` override (CLI) recording the applied
  threshold, the model threshold, and whether it was overridden in the output
  summary.
- An optional `--explain-llm` flag on `predict` (and `"explain_llm": true`
  in the API) that adds per-transaction natural-language rationales citing
  the top contributing features and their direction.
- Drift surveillance with `--fail-on warning|drifted` plus optional
  `--webhook-slack` / `--webhook-pagerduty` flags to push real-time alerts
  to Slack or PagerDuty, including the drifted features and their PSI values.
- Reference label-delay analysis documentation in the README, explaining how
  chargeback label delays can leak into training data when using random splits
  and how the `stability` and `rolling` commands help quantify performance
  fluctuations across time windows.
- Named business-cost policies: `train`/`compare`/`stability` accept
  `--cost-policy`, persisted as a `cost_policy` block in the model card and
  echoed by the `thresholds` report (`"custom"` when overridden ad hoc).
- An opt-in `--estimator hist_gradient_boosting` training option alongside
  logistic regression and random forest, sharing the same leakage-safe split,
  calibration, and threshold-selection pipeline. Its global effects use
  deterministic training-data permutation importance (new
  `permutation_importance` method label) because the pinned scikit-learn
  runtime exposes no native importance for it.
- A `compare --param-name/--param-values` hyperparameter sweep for a single
  estimator on the exact same split, with an allow-list validated per
  estimator before training (unknown, inapplicable, empty, or unparsable
  values fail with actionable errors).
- Settable hyperparameters on `train` and `compare`: `--regularization` and
  `--max-iterations` (logistic regression), `--n-estimators` and `--max-depth`
  (random forest), and `--max-iterations`, `--learning-rate`,
  `--l2-regularization`, `--max-bins`, plus `--max-depth` (histogram gradient
  boosting), so sweep winners can be reproduced in a real artifact.
- Per-transaction local explanations: `predict --explain` adds
  `contrib_<feature>` columns to batch CSV output, and `POST /v1/predict`
  accepts `"explain": true` to include per-transaction `contributions`.
- Optional, off-by-default API hardening references: API-key validation via
  the `X-API-Key` header and per-client-IP fixed-window rate limiting, both
  configured on `create_app()` and documented as defense in depth, not a
  replacement for a real gateway.
- A TestPyPI dry-run publishing workflow on every push to `main`, so
  packaging regressions surface before a real release.
- A `rolling` CLI command evaluating expanding chronological prefixes and
  reporting test-metric spread across origins, for ordered data where
  `stability` does not apply.
- A retraining runbook in the README: drift/calibration triggers, a promotion
  checklist (`compare` → `stability`/`rolling` → `calibration` →
  `thresholds` → `benchmark`), and rollback to retained artifacts.
- Shared output-guard/report-emit helpers across the `drift`, `calibration`,
  `thresholds`, and `benchmark` commands, removing four copies of the
  protect-write-print sequence.
- A `thresholds` CLI command scoring candidate thresholds on held-out labeled
  data, reporting precision, recall, F1, and expected cost per candidate next
  to the model's tuned operating point (costs default to the training policy).
- An artifact retention policy in the README: keep the serving artifact plus
  the two most recent predecessors so comparisons stay reproducible.
- A `stability` CLI command repeating training with successive seeds and
  reporting mean and standard deviation per test metric, so split luck is
  visible before trusting a single `train` run.
- `Retry-After`, `X-RateLimit-Limit`, and `X-RateLimit-Remaining` headers on
  rate-limited (`429`) responses, plus limit/remaining headers on allowed
  responses when the optional limiter is enabled.
- A `calibration` CLI command reporting equal-width reliability bins,
  expected/maximum calibration errors, and a Brier-score decomposition
  (reliability, resolution, uncertainty) for held-out labeled transactions.
- An offline `benchmark` CLI command reporting median batch latency and
  throughput per batch size for capacity planning.
- Persisted PSI alert cutoffs (`drift_thresholds`) in every model card, honored
  by drift reports (older artifacts fall back to the reference defaults).
- A trusted-publishing release checklist in `CONTRIBUTING.md`.
- A `GET /metrics` endpoint reporting Prometheus-format request counts, request
  duration, and scored-transaction decision counts. Each app instance gets its own
  isolated metrics registry.
- An opt-in `--estimator random_forest` training option alongside the default
  logistic regression, sharing the same leakage-safe split, calibration, and
  threshold-selection pipeline. `explain` reports feature importances instead of
  standardized coefficients for it, distinguished by a new `method` field on each
  `feature_effects` entry.
- A `compare` CLI command that trains every requested estimator on the exact same
  split and reports validation/test metrics side by side, without persisting
  anything, for evidence-based estimator selection before a real `train` run.
- Validation-only cost-sensitive threshold selection with configurable false-positive
  and false-negative weights plus holdout expected-cost reporting.
- Cross-validated sigmoid or isotonic probability calibration, optional raw scores,
  and holdout Brier-score reporting.
- Persisted, ranked standardized feature effects and an `explain` CLI for global
  model interpretation.
- Optional chronological train/validation/test evaluation with persisted time ranges
  and safeguards against single-class windows.
- API request correlation, processing-time response headers, and structured
  completion/failure log fields.
- CLI overwrite protection for non-empty model artifact directories.
- CI dependency vulnerability auditing and a patched pytest 9 requirement.
- CI now trains a demo model and runs the built container image against a live
  `/health` check instead of only building it, catching startup regressions in the
  serving path.
- Dependabot now tracks the Docker base image alongside Python dependencies and
  GitHub Actions.
- CI enforces several additional zero-finding ruff rule categories and stricter
  mypy error codes as permanent quality gates, and the required branch coverage
  floor is raised from 90% to 97% to match what the test suite now actually
  achieves.
- A `.python-version` file pins local development to the same Python version as
  the Dockerfile's runtime.
- Enabled GitHub private vulnerability reporting, secret scanning with push
  protection, Dependabot security updates, and branch protection on `main`
  requiring the CI status checks to pass before a pull request can merge.
- A `CODE_OF_CONDUCT.md` and a bug-report issue template (with a security-advisory
  contact link instead of a public-issue path), closing the gaps in GitHub's
  community-standards checklist.
- A `--version` CLI flag.
- CI validates the built wheel and sdist with `twine check` so a PyPI-metadata
  regression (for example, a malformed long description) fails the build instead
  of only surfacing at publish time.
- `ROADMAP.md`, laying out phased next steps beyond `v0.1.0`.
- A PyPI publishing workflow, scaffolded to run on GitHub Release via trusted
  publishing; dormant until the maintainer links the trusted publisher on
  pypi.org.

### Changed

- The development HTTP test client constraint now requires `httpx2>=2.12`,
  avoiding the vulnerable 2.9.x/httpcore2 dependency line reported by
  `pip-audit`.
- Model artifacts now use format version 2 for the calibrated estimator contract.
- Model saving and loading validate the estimator, feature schema, decision threshold,
  finite JSON metadata, and agreement between the embedded and readable model cards.
- Dataset validation rejects duplicate CSV headers and feature names that become
  ambiguous when normalized to the persisted string schema.
- Inference rejects estimators with reversed class labels, malformed binary output,
  non-numeric scores, or non-finite and out-of-range probabilities.
- The API requires strict finite JSON numbers and sanitizes validation responses so
  rejected transaction values are not echoed or rendered as invalid JSON.
- Generated datasets, predictions, and drift reports now use atomic replacement;
  drift reports also require explicit `--overwrite` before replacing a file.
- CLI training reports artifact-destination filesystem failures without a traceback.
- Drift calculation validates normalized feature names, numeric profile statistics,
  bin ordering, and baseline probability distributions before calculating PSI.
- Artifact loading rejects mismatched scikit-learn runtimes before directory-based
  deserialization and converts scikit-learn inconsistency warnings into model errors.
- Calibration uses one worker by default, persists its worker policy, and supports
  explicit `--calibration-jobs -1` opt-in for all processors.
- API requests are bounded to 2 MiB for both declared and streamed bodies while
  preserving correlation and timing headers on `413` responses.
- Model cards record Python, joblib, NumPy, pandas, scikit-learn, and SciPy versions
  needed to reconstruct the training runtime.
- The API now configures root logging on startup so request correlation and timing
  logs are emitted by default under `fraud-detect serve` and the container entrypoint,
  not only when a caller separately configures logging.
- The container base image moves from `python:3.12-slim` to `python:3.14-slim`.

### Fixed

- Random-forest training now actually applies the configured `n_estimators` and
  `max_depth` instead of silently using library defaults.
- `compare` no longer leaks an unhandled `TypeError` traceback for unknown
  sweep parameters; they are rejected upfront with the supported list.
- `load_csv`, `fraud-detect predict`, and `fraud-detect drift` now report an empty
  transactions CSV as an actionable error instead of leaking an unhandled
  `pandas.errors.EmptyDataError` traceback.
- CI now actually enforces flake8-bandit (`S`) lint rules; the rule set was never
  selected, so a pre-existing `S101` per-file ignore for tests was silently a no-op.
- The package version is now derived from a single source (`fraud_detection.__version__`)
  instead of also being hardcoded separately in `pyproject.toml`, where the two could
  previously drift out of agreement.
- mypy no longer skips numpy imports; the skip was a stale workaround, and removing
  it surfaced (and let us remove) an unnecessary `cast()` in `FraudModel.predict()`.
- SECURITY.md's instruction to use GitHub's private vulnerability reporting now
  actually works; the feature was disabled on the repository itself.
- The container CI job's `setup-python` step now matches the version used by the
  main quality job.

## [0.1.0] - 2026-07-24

### Added

- Strict numeric CSV validation and deterministic synthetic demo data.
- Stratified train/validation/test workflow with a class-balanced logistic model.
- Validation-only F1 threshold tuning and imbalance-aware holdout metrics.
- Reproducible model cards with data fingerprints and training feature profiles.
- Atomic model persistence with SHA-256 integrity manifests.
- Batch generation, training, inspection, prediction, drift, and serving commands.
- Versioned FastAPI batch prediction endpoint, readiness endpoint, and OpenAPI docs.
- Population Stability Index feature-drift reports.
- Non-root, read-only Docker Compose deployment profile.
- Python 3.12/3.14 CI, strict typing, linting, formatting, branch coverage, package
  builds, and automated dependency updates.
- Usage, architecture, security, responsible-use, and contribution documentation.

[Unreleased]: https://github.com/Mohamed-ahmed-shokry/Credit-Card-Fraud-Detection/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Mohamed-ahmed-shokry/Credit-Card-Fraud-Detection/releases/tag/v0.1.0
