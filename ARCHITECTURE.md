# Architecture

This document describes the boundaries that make the project reproducible and
safe to extend. The project is a reference implementation, not a complete
financial-services platform.

## Runtime flow

1. `data.py` loads CSV input, rejects ambiguous schemas and unsafe values, and
   exposes a `ValidatedDataset` containing numeric features and a binary target.
2. `model.py` performs stratified or temporal splitting, fits the selected
   estimator and calibration policy, chooses a validation-only threshold, and
   evaluates the untouched test split.
3. The fitted `FraudModel` owns feature ordering, probability validation,
   thresholded decisions, local explanations, metadata, and artifact loading.
4. `cli.py` orchestrates offline and operational workflows without duplicating model logic:
   training, comparison, scoring, monitoring, promotion evidence, model-card
   inspection, compliance rendering, historical audit replay, challenger retraining,
   machine-readable attestation export, multi-window drift surveillance, streaming
   profiling, and chaos simulation.
5. `api.py` adapts the same `FraudModel` to bounded FastAPI requests. Batch and
   single-transaction endpoints share one scoring helper with resilient serving
   guardrails (`fallback_mode`), degraded-state handling, an automated stateful
   `CircuitBreaker` (tracking failure counts and latency budgets), and non-blocking
   asynchronous challenger traffic shadowing (`shadow_model_path`).
6. `drift.py` compares current numeric distributions with the training profile,
   provides multi-window surveillance computing drift velocity and acceleration,
   and implements an online incremental `StreamingProfile` via Welford's algorithm.
   `evaluation.py` computes holdout, calibration, and threshold reports;
   `reporting.py` renders promotion evidence as escaped, self-contained HTML.
7. `audit.py` exports thread-safe, structured JSONL audit events for scoring,
   shadowing, and promotion decisions with automated Luhn-validated PAN and sensitive key
   redaction guarantees, and provides `replay_audit_log` for historical replay
   and divergence backtesting.
8. `explanations.py` defines the `ExplanationProvider` boundary, keeping
   deterministic offline template explanations as default while wrapping external
   LLM calls with timeouts, prompt redaction, cost controls, and fallback.
9. `model.py` provides `generate_attestation` and `verify_attestation` to export
   and verify tamper-evident, canonical SHA-256 digested attestation manifests.

## Artifact contract

An artifact directory contains:

```text
manifest.json   SHA-256 digests for model.joblib and metadata.json
metadata.json   JSON model card, schema, metrics, profile, and lineage
model.joblib    trusted-process pickle containing the fitted FraudModel
```

`save_model` writes temporary files and replaces the final files only after the
metadata and manifest are complete. `load_model` verifies the manifest before
deserialization, checks the scikit-learn runtime, validates the embedded model,
and confirms that embedded and readable metadata agree.

`validate_artifact` (and the `validate-artifact` CLI command) provides a
safe, read-only validation pass verifying manifest integrity, runtime
compatibility, lineage completeness, and report readiness prior to deployment.

The lineage block on newly trained artifacts contains:

- `dataset_fingerprint`: SHA-256-derived fingerprint of feature and target data;
- `config_hash`: SHA-256 of the canonical training configuration;
- `code_version`: package version used by the training process;
- `content_hash`: deterministic hash over dataset, configuration, and code
  version;
- optional `git_commit` and `git_repository` values from CLI training.

Artifact validation reports can be exported as signed machine-readable JSON
attestation manifests via `generate_attestation()` / `validate-artifact --attestation-output`,
providing a verifiable SHA-256 cryptographic digest over the full verification state for admission controllers.

Lineage is additive. Artifacts created before lineage was introduced remain
loadable, but their model cards identify the missing provenance.

## Serving boundary

`create_app` receives an already trusted model or resolves one from
`FRAUD_MODEL_PATH` at startup. It applies request-size limits, strict finite
numeric validation, optional API-key and in-memory rate limiting middleware,
request correlation headers, structured logging, opt-in JSONL audit event export
(`FRAUD_AUDIT_LOG_PATH`), pluggable explanation providers, and isolated Prometheus
metrics. Serving guardrails support degraded-state fallback policies (`constant`,
`rule`, `raise`) configurable via environment variables or header simulation
(`X-Simulate-Degraded`), with fallback executions tracked by Prometheus counters
(`fraud_fallback_predictions_total`). Authentication, network policy, durable
rate limiting, secret management, and audit-log retention remain deployment
responsibilities.

For operational resiliency and safe challenger evaluation:
- `CircuitBreaker` maintains scoring stability. Consecutive failures exceeding
  `failure_threshold` or execution times exceeding `latency_budget_ms` trip the
  breaker from `CLOSED` to `OPEN`, failing over immediately to fallback policies
  until `recovery_timeout` elapses, after which a `HALF_OPEN` probe test determines
  recovery. Breaker state is tracked via `/health` telemetry and Prometheus
  (`fraud_circuit_breaker_state`).
- Asynchronous traffic shadowing evaluates transactions against `shadow_model_path`
  in the background without blocking or delaying primary client responses, recording
  prediction discrepancies and latency metrics (`fraud_shadow_evaluations_total`)
  to structured audit logs.

The API does not retrain or mutate a model. Threshold overrides are explicit in
the request and response, while the tuned artifact threshold remains visible as
`model_threshold`.

## Evaluation invariants

- The validation split owns threshold and model-selection decisions.
- The test split is untouched until final evaluation.
- Temporal evaluation sorts by the configured time feature and rejects
  single-class windows.
- Temporal gaps enforce holdout separation to mirror label delay (chargeback maturation)
  without unconfirmed labels leaking into training windows.
- Calibration is fit inside training data and is never fitted on holdout data.
- Drift is a diagnostic signal; it is not treated as proof of changed model
  quality without labels and business context.
- Explanations isolate external model calls behind timeouts, redactions, cost
  controls, and deterministic fallbacks.
- Compliance reports assemble evidence and never make an automatic promotion
  decision.

## Operational surveillance & safety invariants

- Multi-window drift analyzes both recent short windows and baseline long windows
  to compute velocity ($\Delta \text{PSI}$) and acceleration ($\Delta^2 \text{PSI}$),
  detecting distribution shifts early without waiting for full batch cycles.
- Streaming profile accumulation updates feature distributions incrementally
  using Welford's algorithm, keeping memory constant regardless of stream length.
- Traffic shadowing never blocks, delays, or fails primary scoring responses.
- Circuit breakers fail fast to safe deterministic fallback policies when upstream
  or model degradations occur.

## Extension guidance

New estimators should be added through `EstimatorType` and `_build_base_estimator`
so calibration, splitting, evaluation, artifact validation, and explanations
remain shared. New reports should expose a serializable payload builder in the
relevant domain module and keep rendering/orchestration in `cli.py` or
`reporting.py`. New API fields should be additive where possible and covered by
request-validation and response tests.
