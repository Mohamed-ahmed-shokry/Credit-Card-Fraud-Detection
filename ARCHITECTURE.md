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
   tracks output prediction score drift via `ScoreProfile` and `calculate_score_drift`,
   and implements an online incremental `StreamingProfile` via Welford's algorithm.
   `evaluation.py` defines multi-tier decision routing (`DecisionAction`, `TieredThresholds`,
   `evaluate_tiered_policy`), automated decision tier calibration (`tune_tiered_thresholds`),
   counterfactual policy backtesting (`PolicyTransitionMatrix`, `backtest_policy_transition`),
   sub-population slice disparity diagnostics (`SliceMetricRow`, `evaluate_slices`),
   calibration analysis, and threshold reports;
   `reporting.py` renders promotion evidence as escaped, self-contained HTML.
7. `audit.py` exports thread-safe, structured JSONL audit events for scoring,
   shadowing, and promotion decisions with automated Luhn-validated PAN and sensitive key
   redaction guarantees, provides `replay_audit_log` for historical replay
   and divergence backtesting, and `backtest_audit_policy` for streaming counterfactual
   policy backtesting against audit logs.
8. `explanations.py` defines the `ExplanationProvider` boundary, keeping
   deterministic offline template explanations as default while wrapping external
   LLM calls with timeouts, prompt redaction, cost controls, and fallback.
9. `model.py` provides `generate_attestation` and `verify_attestation` to export
   and verify tamper-evident, canonical SHA-256 digested attestation manifests.
10. `edge.py` defines a versioned, dependency-light int8 runtime for the supported
    uncalibrated logistic artifact contract, providing quantized linear feature explanations
    (`explain_record`), multi-tier decision routing (`predict_decision_record`), and
    pure-Python streaming batch file scoring (`score_batch_file`). `cli.py` builds and
    validates this representation with `export-edge`; the runtime scorer itself does not
    import scikit-learn or pandas.
11. `telemetry.py` parses W3C `traceparent` contexts and provides an injectable
    trace exporter plus best-effort OTLP/HTTP JSON export. `api.py` creates one
    server span per HTTP request and isolates exporter failures from scoring.
12. `signing.py` owns Ed25519 key loading, canonical JSON signing, and trusted
    public-key verification. The CLI uses it to turn validation reports into
    deployment admission evidence without coupling signatures to inference.
13. `trust.py` validates rotation-aware public-key bundles, performs add/revoke
    operations, and exposes the admission check used by API startup. Revocation
    is evaluated before the model artifact is loaded.
14. `rules.py` defines declarative decision rule conditions, operators (`==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not_in`), actions (`ALLOW`, `CHALLENGE`, `DENY`), priority evaluation, and serialization for `RuleSet`. `model.py` and `api.py` integrate rule sets with configurable precedence (`rules_override_model` vs `model_overrides_rules`), Prometheus instrumentation (`fraud_rules_triggered_total`), and structured audit logging.
15. `velocity.py` provides in-memory sliding window accumulators (`VelocityWindowBuffer`, `SlidingWindow`, `VelocityConfig`) tracking count, sum, min, max, mean, std, and exponential moving averages over configurable intervals with bounded $O(1)$ amortized insertion and eviction latency. It guarantees strict temporal causality without future-data leakage in both real-time streaming enrichment (`enrich_record_velocity`) and vectorized offline dataset transformations (`compute_batch_velocity`). `api.py` attaches velocity buffers to `create_app` with dedicated operational endpoints (`/v1/velocity/stats`, `/v1/velocity/profile/{entity_id}`) and Prometheus instrumentation (`fraud_velocity_enrichments_total`, `fraud_velocity_entities_active`).
16. `recalibration.py` provides model calibration surveillance and post-hoc recalibration diagnostics. It computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Root Mean Squared Calibration Error (RMSCE), Brier score decomposition (reliability, resolution, uncertainty), reliability curve binning under uniform, quantile (equal frequency for imbalanced fraud data), and tiered partitioning strategies, and calibration drift detection (`CalibrationDriftReport`) with warning/alarm thresholds against baseline references. It implements three post-hoc recalibrators: Platt scaling (`PlattRecalibrator`) with monotonic $A \ge 0$ constraints and Laplace prior smoothing, isotonic regression (`IsotonicRecalibrator`) with monotonic step clipping, and temperature scaling (`TemperatureRecalibrator`) with rank-preserving probability sharpening/softening. `model.py` integrates recalibrators into `FraudModel`, offering `predict_probabilities(raw=True/False)` and `recalibrate_model` with automatic decision threshold retuning. `api.py` surfaces calibration status in `/health` and `/ready` probes, tracks `fraud_recalibrated_predictions_total` Prometheus metrics, and passes uncalibrated `raw_probability` through prediction results and audit events. `cli.py` exposes `calibrate-eval` and `recalibrate` commands.
17. `routing.py` defines champion/challenger multi-model routing (`ChampionChallengerRouter`, `RoutingPolicy`, `CanaryConfig`, `TrafficSplitStrategy`), deterministic entity hash stickiness across configurable entity keys (`card_id`, `user_id`, `transaction_id`), and automated canary safeguard surveillance (`RoutingMetricsTracker`, `DivergenceReport`) with automatic rollback. `api.py` integrates the router into `create_app`, exposing operational endpoints (`/v1/routing/status`, `/v1/routing/rollback`, `/v1/routing/reset`) and Prometheus metrics (`fraud_routing_predictions_total`, `fraud_shadow_discrepancy_total`, `fraud_canary_status`). Inferences run asynchronously in threadpool workers, and `cli.py` provides `route-eval` for offline and audit log divergence evaluation and safeguard threshold verification.
18. `features.py` defines typed feature store abstractions (`FeatureDefinition`, `FeatureType`, `FeatureView`, `FeatureSnapshot`), thread-safe storage engines (`InMemoryFeatureStore`, `FileFeatureStore`), point-in-time time-travel as-of joins (`point_in_time_join`) with zero lookahead bias guarantee ($t_{\text{feature}} \le t_{\text{observation}}$), and statistical training-serving skew surveillance (`FeatureSkewAnalyzer`, `FeatureSkewReport`, `FeatureSkewMetric`, `SkewStatus`) evaluating Population Stability Index (PSI), Kolmogorov-Smirnov (KS) statistic, Wasserstein distance, and null rate disparity. `audit.py` records enriched features and applied views in scoring audit events. `api.py` enriches incoming prediction requests from the feature store, exposes operational telemetry endpoints (`/v1/features/stats`, `/v1/features/lookup/{entity_key}/{entity_id}`), and exports Prometheus counters (`fraud_feature_lookups_total`). `cli.py` provides `feature-join` and `feature-check` commands and adds `--feature-store` to `serve`.
19. `pipeline.py` defines DAG-based execution of scoring decision workflows (`DecisionPipeline`, `PipelineStage`, `StageProcessor`), organizing stage dependencies into topologically ordered parallel execution waves via `asyncio.gather()`, with per-stage latency budgets, circuit breakers, node caching, speculative model inference, and fail-soft degradation.
20. `feedback.py` defines delayed ground truth label ingestion (`DelayedLabelRecord`, `LabelIngestionStore`), point-in-time audit log matching (`MatchedFeedbackRecord`, `match_feedback_with_audit_log`, `partition_feedback_cohorts`), empirical label maturation curve modeling (`LabelMaturationCurve`, `fit_maturation_curve`) to eliminate survivor bias and immature cohort distortions, realized performance and financial cost surveillance (`RealizedMetricsReport`, `compute_realized_metrics`), and automated retraining trigger policy evaluation (`RetrainingTriggerPolicy`, `FeedbackSurveillanceAnalyzer`). `audit.py` records structured delayed label and retraining trigger audit events. `api.py` exposes `/v1/feedback/labels`, `/v1/feedback/maturation`, and `/v1/feedback/surveillance` endpoints with Prometheus counters (`fraud_feedback_labels_ingested_total`, `fraud_feedback_matched_total`, `fraud_feedback_unmatched_total`, `fraud_feedback_retraining_triggers_total`) and gauges (`fraud_feedback_realized_recall`, `fraud_feedback_realized_pr_auc`, `fraud_feedback_realized_net_financial_loss`). `cli.py` provides `ingest-labels` and `feedback-surveillance` commands.
21. `geolocation.py` implements real-time geographic anomaly detection, spherical Haversine geodesic displacement math ($R = 6371.0\text{ km}$), speed-of-travel calculations (`TravelVelocity`, `calculate_travel_velocity`), streaming entity trajectory tracking (`GeoLocationTracker`) with thread-safe sliding-window temporal TTL eviction and capacity bounding, and in-process IP geocoding with CIDR prefix resolution and proxy/VPN/Tor classification (`IPGeoResolver`). `GeoFeatureExtractor` extracts 7 spatial features (`geo_distance_km`, `geo_speed_kmh`, `geo_time_delta_seconds`, `geo_is_impossible_travel`, `geo_is_cross_border`, `geo_is_proxy`, `geo_country_mismatch`). `rules.py` provides declarative decision heuristics (`build_impossible_travel_rule`, `build_geo_proxy_rule`), `pipeline.py` wires `GeoEnrichmentStageProcessor` into the decision DAG, `audit.py` captures structured PCI-DSS masked audit events (`build_impossible_travel_audit_event`), `api.py` exposes `/v1/geo/velocity/{entity_id}`, `/v1/geo/stats`, `/v1/geo/assess`, and `/v1/geo/reset` endpoints with Prometheus metrics (`fraud_geo_impossible_travel_total`, `fraud_geo_events_total`, `fraud_geo_distance_kilometers`), and `cli.py` provides `geo-velocity` and `geo-enrich` workflows.

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

An edge export is a separate JSON artifact and is not a replacement for the trusted
`model.joblib` artifact:

```text
schema_version       Edge format version (currently 1)
feature_names        Ordered numeric input contract
scaler_mean/scale    Standardization statistics
quantized_weights    Signed int8 logistic coefficients
weight_scale         Coefficient dequantization scale
intercept/threshold  Decision policy values
pruned_features      Coefficients removed by the export policy
```

Only `logistic_regression` artifacts with `calibration_method=none` are exportable.
The edge loader validates the schema, exact feature mapping, finite numeric values,
positive scaler scales, and int8 bounds. The exporter can compare edge and source
probabilities against validation data and refuses output when the configured error
tolerance is exceeded. The JSON runtime is suitable for a constrained scorer, but
the source model remains authoritative for retraining, audit replay, and calibrated
probability reporting.

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

Artifact validation reports can be exported as machine-readable JSON attestation
manifests via `generate_attestation()` / `validate-artifact --attestation-output`,
providing a verifiable SHA-256 digest over the full verification state for admission
controllers. The digest detects payload modification but does not establish who
produced the attestation.

The optional authenticity layer works as follows:

- `generate-signing-key` writes a self-describing Ed25519 private PEM and a public PEM key
  using atomic replacement. Private output is never printed or included in an
  attestation.
- `validate-artifact --strict --signing-key ... --attestation-output ...` adds a
  `signature` envelope. The signature covers the complete payload after the
  SHA-256 `attestation_digest` is attached.
- `verify-attestation` independently loads a pinned public key, validates the
  digest, verifies the Ed25519 signature, and requires `status=PASSED` by default.
  It exits `1` for any failed admission condition.

Unsigned attestations remain digest-verifiable for backward compatibility, but
deployment admission must keep the required-signature default. Public keys are
trust anchors and must come from access-controlled deployment configuration
rather than from the attestation being verified.

Lineage is additive. Artifacts created before lineage was introduced remain
loadable, but their model cards identify the missing provenance.

## Serving boundary

`create_app` receives an already trusted model or resolves one from
`FRAUD_MODEL_PATH` at startup. It applies request-size limits, strict finite
numeric validation, optional API-key and in-memory rate limiting middleware
(configurable through `FRAUD_API_KEYS`, `FRAUD_RATE_LIMIT_REQUESTS`,
`FRAUD_RATE_LIMIT_WINDOW_SECONDS`, and matching `serve` options) that exempt
`/health`, `/live`, `/ready`, and `/metrics`, request correlation headers,
structured logging, opt-in JSONL audit event export
(`FRAUD_AUDIT_LOG_PATH`), pluggable explanation providers, and isolated Prometheus
metrics. Model inference and audit-sink writes run through Starlette's
threadpool so CPU- and I/O-bound scoring never blocks the event loop that
serves probes and other requests, and API keys are compared with
`secrets.compare_digest` across every configured value. An optional
`max_concurrent_scoring` semaphore (default off, also settable via
`FRAUD_MAX_CONCURRENT_SCORING` and `serve`) returns `503` with `Retry-After`
when the cap is reached instead of queueing unbounded work. Dedicated probes
expose process liveness (`/live`) and scoring-path
readiness (`/ready`, HTTP 503 when the model is missing or the circuit is open
with `fallback_mode=raise`). It also returns a W3C `traceparent` response header and can export request
spans through an injected `TraceExporter` or optional OTLP/HTTP collector configured
with `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, `OTEL_SERVICE_NAME`, and
`FRAUD_OTLP_TIMEOUT_SECONDS`. Trace export is disabled by default, sends only
operational attributes (method, path, status, request ID, and model version), and
contains no transaction features. Exporter/network failures are isolated from
requests. Serving guardrails support degraded-state fallback policies (`constant`,
`rule`, `raise`) configurable via environment variables or header simulation
(`X-Simulate-Degraded`), with fallback executions tracked by Prometheus counters
(`fraud_fallback_predictions_total`). Authentication, network policy, durable
rate limiting, collector authentication, secret management, trace retention, and
audit-log retention remain deployment responsibilities. When both
`FRAUD_ATTESTATION_PATH` and `FRAUD_TRUST_BUNDLE_PATH` are configured, startup
first verifies the attestation digest, `PASSED` status, signing key ID, active
bundle status, and Ed25519 signature; only then does it deserialize the model.
Partial admission configuration or failed verification stops startup.

## Distribution boundary

GitHub Actions builds the wheel and source distribution once per release workflow.
The release candidate wheel is installed before CycloneDX generates the
`sbom/sbom.cdx.json` artifact, so the recorded dependency inventory describes
what is being published rather than an unrelated source-only environment. CI
also installs that wheel from outside the checkout and runs the versioned CLI.
TestPyPI uses `skip-existing` only for immutable already-uploaded development
files; build, metadata, and SBOM failures remain fatal. PyPI and TestPyPI
publishing authenticate through OIDC trusted publishers, which are deployment
configuration rather than application code.

For operational resiliency and safe challenger evaluation:
- `CircuitBreaker` maintains scoring stability. Consecutive failures exceeding
  `failure_threshold` or execution times exceeding `latency_budget_ms` trip the
  breaker from `CLOSED` to `OPEN`, failing over immediately to fallback policies
  until `recovery_timeout` elapses, after which a `HALF_OPEN` probe test determines
  recovery. State transitions are synchronized with a lock across concurrent threadpool
  workers. Breaker state is tracked via `/health` telemetry and Prometheus
  (`fraud_circuit_breaker_state`).
- Asynchronous traffic shadowing evaluates transactions against `shadow_model_path`
  on detached background tasks outside the ASGI request lifecycle without blocking,
  delaying primary client responses, or holding connections open, recording
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
- Traffic shadowing runs on detached tasks outside the ASGI request lifecycle,
  never blocking, delaying, or failing primary scoring responses.
- Caller-input schema mismatches return `422` immediately and do not affect
  circuit-breaker health or trip fallback policies.
- Prometheus request metrics are labeled with matched route templates, collapsing
  undeclared paths into `'unmatched'` to prevent label cardinality explosion.
- Circuit breakers fail fast to safe deterministic fallback policies when upstream
  or model degradations occur.
- Trace propagation preserves a valid incoming W3C trace ID while creating a new
  server span ID; malformed contexts start a fresh trace.
- OTLP collection is optional and best effort. It must never become a scoring
  availability dependency.
- Output score drift surveillance evaluates prediction probabilities against baseline
  validation profiles using Population Stability Index (PSI) without requiring labels,
  tripping alerting webhooks when distributions deviate.
- Multi-tier decision routing adheres strictly to $0 \le review\_threshold \le deny\_threshold \le 1.0$,
  ensuring transactions cannot simultaneously qualify for both ALLOW and DENY actions.
- Post-hoc probability recalibrators enforce monotonicity ($A \ge 0$ for Platt scaling, non-decreasing step functions for isotonic regression, and rank-preserving temperature scaling), guaranteeing that calibrated risk scores never invert predicted class rankings.
- Recalibration surveillance monitors reliability calibration drift against baseline curves, alerting operational teams when ECE degradation exceeds configured warning or alarm tolerances.
- Multi-model canary routing protects production scoring via automated discrepancy and probability divergence safeguards; exceeding safety thresholds automatically halts Challenger traffic and executes zero-latency rollback to Champion inference.
- Entity hash routing guarantees deterministic model stickiness across transactions for the same entity key, preventing alternating model decisions during rolling canary deployments.
- Point-in-time feature store as-of joins strictly enforce temporal causality ($t_{\text{feature}} \le t_{\text{observation}}$), eliminating future target leakage in offline training set generation.
- Training-serving feature skew surveillance monitors feature distribution drift across serving traffic against training baselines, evaluating PSI, Kolmogorov-Smirnov test, and Wasserstein distance with configurable drift thresholds and automated safeguard exit codes.
- Decision graph DAG execution organizes stage dependencies into topologically ordered parallel execution waves, executing concurrent non-interdependent stages (e.g. feature store lookups and heuristic rules) via `asyncio.gather()`.
- Per-stage latency budgets and fail-soft degradation guarantee pipeline resilience; when non-required stages (enrichment, recalibration, explainability) time out or fail, they fall back to pre-configured safe outputs and flag `degraded=True`, allowing final action arbitration to complete within SLA.
- Full execution tracing records complete pipeline DAG lineage, stage durations, and degraded node lists in structured audit events and Prometheus telemetry (`fraud_pipeline_stage_duration_seconds`, `fraud_pipeline_executions_total`).
- Stage node caching keeps a bounded in-memory LRU per node with TTL expiry; cache keys are derived deterministically from the transaction (or configured key columns), so a hit returns exactly what the processor would have produced for identical inputs. Only pure stages are cached by default (rules); model inference caching is opt-in.
- Speculative model inference starts the inference stage concurrently with upstream enrichment on the pre-enrichment transaction. The speculative result is adopted only when the model's feature values are unchanged after enrichment and no rule short-circuit occurred; otherwise it is cancelled and the stage is recomputed, so speculation never changes the decision, only latency. Outcomes are recorded as `speculative_executed` / `speculative_hit` in results, audit events, and Prometheus counters.
- Dynamic decision graph replay (`replay_audit_log_through_pipeline`) evaluates historical JSONL scoring audit logs against candidate DAG configurations (incorporating updated rules, rule precedence, recalibrators, and tiered thresholds), tracking probability divergence, 3×3 decision turnover matrices (`ALLOW`, `CHALLENGE`, `DENY`), and stage-by-stage execution latency distributions (`mean_ms`, `p95_ms`).
- Counterfactual policy optimization (`optimize_counterfactual_pipeline`) quantifies decision turnover, review queue workload shifts, deny volume deltas, and expected financial cost differentials across candidate tiered policies prior to production rollout.
- Decision graph stage circuit breakers provide thread-safe fast-fail isolation per stage. Each stage circuit breaker implements a three-state finite state machine (`CLOSED`, `OPEN`, `HALF_OPEN`) tracking consecutive failure thresholds and recovery timeout cooldowns. When a stage trips `OPEN`, calls short-circuit immediately without executing processor logic, returning pre-configured safe fallbacks with `StageStatus.CIRCUIT_OPEN`, flagging `degraded=True`, and recording the short-circuit in audit logs and Prometheus telemetry (`fraud_pipeline_circuit_short_circuits_total`).
- Deterministic stage chaos fault injection (`simulate_pipeline_chaos`) enables resilience testing by injecting controlled faults (latency spikes, transient exceptions, payload corruption, and timeouts) into target pipeline stages using seedable pseudo-random distributions. Evaluates degraded execution rates, circuit trips, stage short-circuits, and strict SLA breach budgets (`sla_budget_ms`) while generating structured audit events and Prometheus counters (`fraud_pipeline_chaos_injections_total`).
- Delayed ground-truth feedback ingestion operates thread-safely with append-only stores (`LabelIngestionStore`), supporting multi-source label aggregation (chargebacks, merchant disputes, manual fraud analyst reviews, confirmed non-fraud clears) and point-in-time join matching against immutable JSONL scoring audit logs via transaction ID.
- Empirical label maturation curve modeling (`LabelMaturationCurve`) estimates cumulative delay CDFs and quantiles ($p_{50}, p_{75}, p_{90}, p_{95}, p_{99}$) across chargeback arrival intervals. Cohorts younger than the configured maturity threshold (e.g. $p_{90}$ or 90% label maturity fraction) are filtered from primary performance evaluation to prevent optimistic survivor bias and premature false alarm triggers.
- Realized surveillance calculates performance metrics (precision, recall, F1, PR-AUC, ROC-AUC, Brier score) and business financial metrics (net financial loss, fraud loss avoided, false positive friction cost, net ROI) exclusively on mature transaction cohorts.
- Automated retraining trigger policies enforce rigorous composite criteria (requiring minimum mature transaction counts and minimum confirmed fraud incidents before evaluation) and threshold checks across recall degradation, PR-AUC drops, cost surges, and sustained feature drift, emitting structured audit events and non-zero CLI exit codes (`EXIT_CODE_RETRAIN_TRIGGERED = 3`) to initiate downstream automated re-training pipelines.
- Automated closed-loop retraining dataset materialization (`RetrainingDatasetBuilder`) joins mature ground-truth dispute feedback with immutable production scoring audit logs, enforcing temporal ordering, holdout purge buffers, capacity assertions, and multi-class verification without future data leakage.
- Automated challenger training and tuning (`ChallengerTrainer`) maps hyperparameter configurations, executes cross-validation splitting, optimizes decision thresholds for cost minimization or F1, and captures complete training duration and lineage metadata.
- Multi-faceted challenger safety verification (`ChallengerSafetyVerifier`) evaluates challenger candidates against active production champions on identical holdout data across composite safety gates: minimum PR-AUC delta ($\ge -0.02$), recall non-regression ($\ge -0.03$), maximum calibration error increase ($\Delta \text{ECE} \le 0.03$), financial cost differential ($\le 5\%$), and sub-population slice disparity ($\le 0.25$).
- Cryptographic candidate attestation and lineage signing (`export_certified_candidate`, `verify_certified_candidate`) seals candidate artifacts with canonical JSON manifests, SHA-256 digests, and Ed25519 digital signatures, preventing tampered or uncertified models from loading into production.
- Zero-downtime hot-reloading (`POST /v1/models/reload`) and dynamic canary promotion (`POST /v1/models/promote`) perform thread-safe in-memory model swaps, adjusting canary traffic splits or promoting candidates to primary champion without process restarts, connection drops, or scoring interruption.
- Automated retraining CLI orchestration (`fraud-detect auto-retrain`) provides end-to-end execution with deterministic exit code gating ($0=\text{Success}$, $1=\text{Runtime Error}$, $2=\text{Config Error}$, $3=\text{Candidate Rejected by Safety Gates}$) to safely integrate into automated orchestrators (Airflow, Kubeflow, GitHub Actions).
- Real-time bipartite entity-transaction link graph (`EntityGraph`) maintains an in-memory streaming bipartite topology mapping discrete entities (`card_id`, `device_id`, `ip_address`, `merchant_id`, `user_id`, `session_id`, `terminal_id`) to transaction nodes with $O(1)$ adjacency lookups, sliding temporal window eviction, thread safety (`RLock`), and hard capacity capping.
- Graph feature extraction (`GraphFeatureExtractor`) strictly enforces zero lookahead bias by extracting structural centrality, degree, and 2-hop entity sharing counts computed strictly before candidate transaction ingestion.
- Personalized PageRank risk diffusion propagates empirical fraud risk scores through the 2-hop bipartite entity network using bounded power iteration with configurable damping ($\alpha=0.85$) and iterations ($k=5$), surfacing hidden synthetic identity links and collusion syndicates.
- Automated topological fraud ring detection (`FraudRingDetector`) discovers dense bicliques (cards cycling across shared devices/IPs), high-frequency card-testing botnet attacks, and circular flow cycles, assigning severity ratings (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`) and recommended mitigation actions (`ALLOW`, `CHALLENGE`, `DENY`).
- Pipeline integration (`GraphEnrichmentStageProcessor`) enriches real-time decision graph transactions with 26+ link features, enabling downstream rules (`build_fraud_ring_rule`) to trigger immediate challenges or denies.
- Redacted audit trail logging (`build_fraud_ring_detected_audit_event`) emits structured audit records with automated PCI-DSS compliant PAN masking for all credit card identifiers.
- Prometheus graph telemetry tracks active entity counts by type (`fraud_graph_nodes_total`), total edges (`fraud_graph_edges_total`), detected syndicate counts by severity (`fraud_graph_rings_detected_total`), and ring detection latency histograms (`fraud_graph_ring_detection_seconds`).
- Real-time geographic geolocation anomaly detection and velocity speed-of-travel tracking (`GeoLocationTracker`, `calculate_travel_velocity`) enforces zero-lookahead temporal causality and spherical Haversine geodesic displacement math, detecting impossible physical travel anomalies ($v > 900\text{ km/h}$) and high-risk proxy/VPN/Tor egress nodes.
- Thread-safe sliding temporal window trajectory tracking evicts stale locations beyond the configured TTL window (default 24h) and bounds maximum entity capacity, preventing memory leaks in high-throughput streaming environments.
- In-process IP geocoding (`IPGeoResolver`) performs deterministic longest-prefix CIDR matching against registered subnets without external network calls or binary database dependencies, resolving geographic coordinates and threat flags (`is_proxy`, `is_vpn`, `is_tor`).
- Structured audit event emission (`build_impossible_travel_audit_event`) guarantees PCI-DSS compliant Luhn-validated PAN masking and sensitive data redaction across impossible travel incident records.
- Pipeline integration (`GeoEnrichmentStageProcessor`) enriches real-time decision graph transactions with 7 spatial features, allowing downstream rules (`build_impossible_travel_rule`, `build_geo_proxy_rule`) to trigger immediate transaction challenges or denies.
- Prometheus geolocation metrics track impossible travel incidents (`fraud_geo_impossible_travel_total`), processed spatial events (`fraud_geo_events_total`), and geodesic displacement histograms (`fraud_geo_distance_kilometers`).
- Real-time adaptive device and browser fingerprinting subsystem (`DeviceFingerprint`, `ScreenGeometry`, `DeviceHasher`, `DeviceTracker`) provides deterministic canonical SHA-256 and locality-sensitive SimHash computation for fuzzy device clustering and drift detection.
- Multi-attribute composite similarity calculation (`calculate_fingerprint_similarity`) weights hardware concurrency, memory ratios, screen dimensions, WebGL vendor/renderer strings, Canvas/Audio hashes, and OS/browser families into a calibrated stability score in $[0.0, 1.0]$.
- Thread-safe sliding temporal window device tracking (`DeviceTracker`) detects card cycling (multiple cards on one device) and device cycling (multiple devices on one card) across a configurable lookback window with bounded LRU eviction.
- Automated emulator and bot detection heuristics (`detect_emulator_signatures`) inspect user agents for headless automation frameworks (HeadlessChrome, Puppeteer, Selenium, Playwright), explicit webdriver flags, unrealistic hardware configurations, and desktop environments lacking WebGL/Canvas contexts.
- Shannon entropy scoring (`calculate_fingerprint_entropy`) calculates information-theoretic diversity scores over normalized attribute distributions, identifying low-entropy synthetic or automated browser profiles.
- Pipeline graph integration (`DeviceEnrichmentStageProcessor`) enriches real-time decision graph transactions with 7 device features, enabling downstream rules (`build_device_emulator_rule`, `build_device_card_cycling_rule`) to trigger immediate challenges or denies.
- Redacted audit trail logging (`build_device_anomaly_audit_event`) emits structured audit records with automated PCI-DSS compliant PAN masking for all credit card numbers.
- Prometheus device telemetry tracks total device evaluations (`fraud_device_evaluations_total`), detected anomalies by severity (`fraud_device_anomalies_total`), and composite fingerprint similarity histograms (`fraud_device_similarity_score`).
- Real-time merchant and ISO 18245 MCC risk profiling (`MCCRiskProfile`, `resolve_mcc_profile`, `MerchantTracker`) computes Empirical Bayes (Beta-Binomial) credibility-smoothed fraud rates ($\hat{p} = \frac{f + m \cdot p_0}{n + m}$) and Wilson score upper confidence bounds, preventing zero-volume division variance on newly onboarded terminals while rapidly escalating established high-chargeback merchants.
- Online Welford ticket-size statistics (`_update_welford`) maintain running mean and variance accumulators per merchant in $O(1)$ space, detecting sudden terminal bust-out surges ($Z > 3.0$) and micro-transaction card-testing enumeration ($\ge 50\%$ micro-transactions across $\ge 5$ distinct cards).
- Pipeline graph integration (`MerchantEnrichmentStageProcessor`) enriches real-time decision graph transactions with 8 merchant features (`merchant_tx_count_24h`, `merchant_smoothed_fraud_rate`, `merchant_mcc_risk_score`, `merchant_amount_zscore`, `merchant_distinct_cards_24h`, `merchant_micro_tx_ratio`, `merchant_is_high_chargeback`, `merchant_is_bust_out_surge`), enabling downstream rules (`build_merchant_chargeback_rule`, `build_merchant_bust_out_rule`) to trigger immediate challenges or denies.
- Redacted audit trail logging (`build_merchant_anomaly_audit_event`) emits structured `fraud.merchant.anomaly_detected` records with automated PCI-DSS compliant Luhn PAN masking, and Prometheus merchant telemetry tracks evaluations (`fraud_merchant_evaluations_total`), anomalies (`fraud_merchant_anomalies_total`), and active tracked merchants (`fraud_merchant_tracked_merchants_total`).
- Multi-tenant policy partitioning (`TenantContext`, `TenantRiskProfile`, `TenantRegistry`) enables secure multi-organization payment processing and SaaS fraud gateway hosting with strict account isolation, tenant-specific risk profiles (`decline_threshold`, `challenge_threshold`, `review_threshold`, `max_amount_per_tx`), custom velocity limits, and metadata tags.
- Dynamic rule federation (`resolve_tenant_ruleset`) dynamically merges baseline system rules with organization-level declarative rules across three inheritance modes: `OVERRIDE_ONLY`, `INHERIT_AND_OVERRIDE`, and `ISOLATED`, resolving priority order and rule collisions deterministically.
- Thread-safe tenant registry (`TenantRegistry`) manages runtime tenant onboarding and lifecycle states (`ACTIVE`, `SUSPENDED`, `DEPRECATED`), supporting bounded LRU memory eviction, strict isolation enforcement, and atomic manifest export/import.
- Pipeline graph integration (`TenantPartitionStageProcessor`) binds active tenant context to Decision DAG evaluation without cross-tenant state leakage, dynamically injecting tenant thresholds and rule actions into `ActionAggregatorStageProcessor`.
- Redacted audit trail logging (`build_tenant_audit_event`) emits structured `fraud.tenant.policy_evaluated` audit records with PCI-DSS compliant Luhn-validated PAN masking and sensitive credential redaction.
- REST API and Prometheus observability provide tenant resolution (`X-Tenant-ID` header), tenant management endpoints (`/v1/tenants/*`), suspended 403 enforcement, and tenant telemetry (`fraud_tenant_evaluations_total`, `fraud_tenant_rule_overrides_total`).



## Real-Time Geolocation & Velocity Travel Subsystem

```mermaid
flowchart TD
    subgraph Ingestion ["Transaction Ingestion"]
        TX["Transaction Record\n(lat, lon, IP, card_id, ts)"]
    end

    subgraph Resolver ["In-Process IP Geocoding"]
        IPR["IPGeoResolver\n(CIDR Longest-Prefix Matching)"]
        SUBNET["Registered Subnets &\nThreat Flags (Proxy/VPN/Tor)"]
        IPR --> SUBNET
    end

    subgraph Tracker ["Trajectory Tracking Engine"]
        GLT["GeoLocationTracker\n(Thread-Safe Sliding Window)"]
        HIST["Entity Trajectory History\n(Bounded Capacity & 24h TTL)"]
        GLT --> HIST
    end

    subgraph Velocity ["Geodesic Math Engine"]
        HAV["Haversine Geodesic Math\n(R = 6371.0 km)"]
        VEL["calculate_travel_velocity()\n(Distance, dt, Velocity km/h)"]
        HAV --> VEL
    end

    subgraph Features ["Feature Extraction & Rules"]
        GFE["GeoFeatureExtractor\n(7 Spatial Features)"]
        RULES["Rules Engine\n(Impossible Travel & Proxy Rules)"]
        GFE --> RULES
    end

    subgraph Serving ["Decision Serving & Telemetry"]
        PIPE["Decision Pipeline DAG\n(GeoEnrichmentStageProcessor)"]
        AUDIT["Structured Audit Trail\n(PCI-DSS Luhn Masked)"]
        PROM["Prometheus Telemetry\n(Travel Metrics & Counters)"]
    end

    TX --> IPR
    TX --> GLT
    IPR --> GFE
    GLT --> VEL
    VEL --> GFE
    GFE --> PIPE
    RULES --> AUDIT
    PIPE --> PROM
```


## Adaptive Device & Browser Fingerprinting Subsystem

```mermaid
flowchart TD
    subgraph Ingestion ["Client Ingestion"]
        FP_IN["Transaction & Device Payload\n(UA, Canvas, WebGL, Screen, Audio, Cores, Memory)"]
    end

    subgraph Hasher ["Hashing & Normalization Engine"]
        DH["DeviceHasher\n(SHA-256 Canonical & 64/128-bit SimHash)"]
        ENTROPY["calculate_fingerprint_entropy()\n(Shannon Diversity Metric)"]
        EMU["detect_emulator_signatures()\n(Automation & Bot Heuristics)"]
        FP_IN --> DH
        FP_IN --> ENTROPY
        FP_IN --> EMU
    end

    subgraph Tracker ["Streaming Device & Cycling Tracker"]
        DT["DeviceTracker\n(Thread-Safe Sliding Windows)"]
        PROF["DeviceProfile & Baselines\n(Multi-Attribute Similarity Scoring)"]
        DEV_WIN["Device-to-Card Sliding Window\n(Card-Cycling Ring Detection)"]
        CARD_WIN["Card-to-Device Sliding Window\n(Device-Cycling Detection)"]
        DT --> PROF
        DT --> DEV_WIN
        DT --> CARD_WIN
    end

    subgraph Pipeline ["Decision Pipeline DAG"]
        STAGE["DeviceEnrichmentStageProcessor\n(7 Structured Device Features)"]
        RULES["Rules Engine\n(Emulator & Card-Cycling Rules)"]
        STAGE --> RULES
    end

    subgraph Telemetry ["Observability & Serving"]
        API["REST API\n(/v1/device/stats, /assess, /profile)"]
        AUDIT["Structured Audit Trail\n(PCI-DSS Redacted PANs)"]
        PROM["Prometheus Telemetry\n(Evaluations, Anomalies, Drift Scores)"]
    end

    DH --> DT
    ENTROPY --> STAGE
    EMU --> DT
    DT --> STAGE
    STAGE --> API
    RULES --> AUDIT
    DT --> PROM
```


## Real-Time Merchant & MCC Risk Profiling Subsystem

```mermaid
flowchart TD
    subgraph Ingestion ["Transaction & Merchant Ingestion"]
        M_IN["Transaction Event\n(merchant_id, mcc, amount, card_id, is_fraud, is_chargeback)"]
    end

    subgraph Taxonomy ["ISO 18245 MCC Risk Taxonomy & Bayes Prior"]
        MCC_RES["resolve_mcc_profile()\n(High-Risk Gambling/Crypto/Wire vs Retail/Grocery)"]
        EB["Empirical Bayes Beta-Binomial Smoothing\n& Wilson Upper Confidence Bound"]
        M_IN --> MCC_RES
        MCC_RES --> EB
    end

    subgraph Tracker ["Streaming Merchant Surveillance Tracker"]
        MT["MerchantTracker\n(Sliding 24h Window & Bounded LRU)"]
        WEL["Online Welford Ticket Statistics\n(Running Mean, Std, Z-Score)"]
        CB["Chargeback Breach & Micro-Tx\nCard-Testing Heuristics"]
        MT --> WEL
        MT --> CB
    end

    subgraph Pipeline ["Decision Pipeline DAG"]
        M_STAGE["MerchantEnrichmentStageProcessor\n(8 Structured Merchant Features)"]
        M_RULES["Rules Engine\n(Chargeback Breach & Bust-Out Surge Rules)"]
        M_STAGE --> M_RULES
    end

    subgraph Telemetry ["Observability & Serving"]
        M_API["REST API\n(/v1/merchant/stats, /assess, /profile, /reset)"]
        M_AUDIT["Structured Audit Trail\n(PCI-DSS Luhn Redacted PANs)"]
        M_PROM["Prometheus Telemetry\n(Evaluations, Anomalies, Tracked Merchants)"]
    end

    EB --> MT
    MT --> M_STAGE
    M_STAGE --> M_API
    M_RULES --> M_AUDIT
    MT --> M_PROM
```

## Multi-Tenant Policy Partitioning & Dynamic Rule Federation Subsystem

```mermaid
flowchart TD
    subgraph Client ["Multi-Tenant Client Request"]
        REQ["Payment Transaction\n+ Header: X-Tenant-ID"]
    end

    subgraph Resolution ["Tenant Resolution & Validation"]
        T_REG["TenantRegistry\n(LRU Bounded & Thread-Safe)"]
        T_CTX["TenantContext\n(ID, Tier, Status, RiskProfile)"]
        T_GUARD{"Status Check\n(Active vs Suspended)"}
        T_ERR["403 Forbidden\n(Suspended Account)"]
    end

    subgraph Federation ["Dynamic Rule Federation Engine"]
        SYS_RULES["Global System Rules"]
        TENANT_RULES["Tenant Custom Rules"]
        FED_ENGINE["resolve_tenant_ruleset\n(OVERRIDE_ONLY | INHERIT_AND_OVERRIDE | ISOLATED)"]
        COMPILED_RULES["Federated RuleSet\n(Priority Sorted)"]
    end

    subgraph Execution ["Decision DAG Pipeline Execution"]
        T_STAGE["TenantPartitionStageProcessor\n(Context-Bound Stage Node)"]
        DAG["DecisionGraph\n(TOPOLOGICAL WAVE)"]
        AGG["ActionAggregatorStageProcessor\n(Tenant-Specific Score Thresholds)"]
    end

    subgraph Observability ["Observability & Audit"]
        AUDIT["Structured Audit Event\n(fraud.tenant.policy_evaluated + Luhn Masking)"]
        PROM["Prometheus Metrics\n(fraud_tenant_evaluations_total, overrides)"]
    end

    REQ --> T_REG
    T_REG --> T_CTX
    T_CTX --> T_GUARD
    T_GUARD -- Suspended --> T_ERR
    T_GUARD -- Active --> T_STAGE
    SYS_RULES --> FED_ENGINE
    TENANT_RULES --> FED_ENGINE
    FED_ENGINE --> COMPILED_RULES
    COMPILED_RULES --> T_STAGE
    T_STAGE --> DAG
    DAG --> AGG
    AGG --> AUDIT
    AGG --> PROM
```


## Asynchronous Real-Time Event Bus & Streaming Webhook Dispatcher

The fraud detection engine includes a zero-dependency, in-process asynchronous streaming event bus and resilient webhook dispatch system (`events.py`). This allows multi-tenant payment gateways, downstream fraud analyst portals, and external alerting systems to subscribe to transaction verdicts, graph fraud rings, merchant anomalies, and model promotions without introducing latency overhead into online scoring paths.

```mermaid
flowchart TD
    subgraph Publisher ["Event Sources & Ingestion"]
        DAG["DecisionGraph / Pipeline\n(ActionAggregatorStageProcessor)"]
        API["FastAPI Engine\n(/v1/score)"]
        GRAPH["Graph / Merchant / Model Engine"]
    end

    subgraph EventBusSub ["EventBus & Subscription Registry"]
        BUS["EventBus\n(Thread-Safe Broadcaster)"]
        REG["SubscriptionRegistry\n(Tenant-Filtered Subscriptions)"]
        WORKERS["ThreadPoolExecutor\n(Non-Blocking Asynchronous Dispatch)"]
    end

    subgraph DispatcherSub ["Webhook Dispatch Engine"]
        DISP["WebhookDispatcher\n(HMAC-SHA256 Signing + Luhn Masking)"]
        CB{"Circuit Breaker\n(Failures >= Threshold?)"}
        RETRY["Exponential Backoff\n(Full Jittered Sleep)"]
        HTTP["HTTP POST Delivery\n(X-Fraud-Signature: t=...,v1=...)"]
    end

    subgraph DLQSub ["Dead Letter Retention & Observability"]
        DLQ["DeadLetterQueue\n(Bounded FIFO Queue)"]
        PROM["Prometheus Telemetry\n(fraud_event_bus_published_total\nfraud_webhook_dispatches_total)"]
    end

    DAG --> BUS
    API --> BUS
    GRAPH --> BUS
    BUS --> REG
    BUS --> WORKERS
    WORKERS --> DISP
    DISP --> CB
    CB -- Closed --> HTTP
    CB -- Open / Tripped --> DLQ
    HTTP -- Failure / 5xx --> RETRY
    RETRY -- Attempts Exhausted --> DLQ
    HTTP -- Success 2xx --> PROM
    DLQ --> PROM
```

### Key Components

- **Domain Event Taxonomy (`EventType`)**: Typed domain events across the fraud detection lifecycle (`fraud.transaction.scored`, `fraud.decision.denied`, `fraud.graph.ring_detected`, `fraud.merchant.anomaly_detected`, `fraud.geo.anomaly_detected`, `fraud.model.promoted`, etc.).
- **HMAC-SHA256 Signature Verification**: All webhook notifications include an `X-Fraud-Signature: t={timestamp},v1={hex_digest}` header computed over the serialized payload to ensure integrity and prevent replay attacks within a 300-second window.
- **PCI-DSS PAN Redaction**: Domain events and webhook envelopes automatically pass through Luhn PAN masking and credential redaction, preventing cardholder data leakage.
- **Circuit Breaker & Exponential Backoff**: Webhook destinations experiencing repeated server errors are quarantined by an automatic circuit breaker (`circuit_breaker_failure_threshold=5`, `recovery_seconds=30.0`), with intermediate attempts scheduled via exponential backoff with full jitter.
- **Dead-Letter Queue (DLQ)**: Undeliverable notifications are preserved in a thread-safe, bounded in-memory buffer (`DeadLetterQueue`) for inspection and on-demand replay via CLI or API.


## Risk-Based Step-Up Authentication & Challenge Orchestration (Phase 44)

The platform incorporates an in-process, zero-dependency Strong Customer Authentication (SCA) step-up challenge and 3-D Secure (3DS 2.3) orchestration engine (`challenge.py`). When transaction risk evaluation falls into review/friction thresholds (`RuleAction.CHALLENGE`), step-up authentication sessions are automatically provisioned with cryptographically secure OTP tokens, channel destination masking, constant-time verification, brute-force lockout, and rate-limited resends.

```mermaid
flowchart TD
    subgraph Decision Engine
        DAG["Decision DAG Pipeline"] -->|Evaluate Rules & Model| Action["ActionAggregatorStageProcessor"]
        Action -->|Action == CHALLENGE| CM["ChallengeManager"]
    end

    subgraph Challenge Lifecycle
        CM --> Gen["Cryptographic OTP Generator\n(secrets.SystemRandom)"]
        Gen --> Hash["HMAC-SHA256 Token Hash + Salt"]
        Hash --> Mask["PII Masking Filter\n(Email/SMS Phone Masking)"]
        Mask --> Store[("ChallengeStore\n(Thread-Safe Multi-Index)")]
        Store --> State{"Session State Machine\n(PENDING / VERIFIED / LOCKED_OUT / EXPIRED)"}
    end

    subgraph Verification & Protection
        API["FastAPI /v1/challenges/*\nCLI challenge-*"] -->|Verify Token| Comp["Constant-Time Verification\nhmac.compare_digest"]
        Comp -->|Match| Verified["Status: VERIFIED\nPrometheus: fraud_challenges_verified_total"]
        Comp -->|Mismatch| Attempts{"Attempts >= Max?"}
        Attempts -->|No| Invalid["Status: PENDING\nRemaining Attempts Decremented"]
        Attempts -->|Yes| Lockout["Status: LOCKED_OUT\nUser Lockout Timer Triggered"]
    end

    subgraph Event Broadcasting
        CM -->|Emit CHALLENGE_ISSUED| Bus["EventBus"]
        Comp -->|Emit CHALLENGE_VERIFIED / FAILED| Bus
    end
```

### Key Components

- **Domain Models & Enums (`ChallengeSession`, `ChallengeMethod`, `ChallengeStatus`)**: Immutable records tracking multi-channel challenge methods (`sms_otp`, `email_otp`, `totp`, `biometric_push`, `kba_security_questions`, `out_of_band`) and lifecycle states.
- **Cryptographic Security & Constant-Time Verification**: Generates cryptographically secure nonces via `secrets.SystemRandom`, salted with high-entropy bytes and hashed using HMAC-SHA256. Verified using `hmac.compare_digest` to prevent side-channel timing attacks.
- **PII Destination Redaction**: Automatically redacts sensitive phone numbers and emails (e.g., `+1 ***-***-1234` and `a***e@example.com`) to prevent exposure in logs, API responses, and audit records.
- **Brute-Force Lockout & Cooldown Rate Limiting**: Enforces configurable max attempts per session (default: 3). Failed attempts exceeding thresholds trigger automated user lockouts (`lockout_duration_seconds=900`). Resend operations enforce a cooldown period (`resend_cooldown_seconds=30`) to protect against SMS pumping and denial-of-service.
- **Pipeline & EventBus Synergy**: Seamlessly wired into the Decision DAG and broadcasts `fraud.challenge.issued`, `fraud.challenge.verified`, `fraud.challenge.failed`, and `fraud.challenge.expired` across the streaming EventBus.


## Fraud Case Management, Investigation SLA Tracking & Dispute Evidence Packaging Engine (Phase 45)

The platform incorporates an in-process, zero-dependency Fraud Case Management, Analyst Workflow, SLA Tracking, and Dispute Evidence Packaging Engine (`cases.py`). Suspicious transactions flagged for manual review (`final_action == "REVIEW"` in `pipeline.py`) or charged-back transactions are tracked through a formalized investigation lifecycle with SLA timers, audit trails, and cryptographically fingerprinted dispute dossiers suitable for card network representment.

```mermaid
flowchart TD
    subgraph Trigger & Creation
        DAG["Decision DAG Pipeline\n(Action == REVIEW)"] -->|Auto-Create Case| CM["CaseManager"]
        Manual["Analyst API / CLI\nPOST /v1/cases"] -->|Create Case| CM
    end

    subgraph Case Lifecycle & SLA
        CM --> Store[("CaseStore\n(Thread-Safe Multi-Index)")]
        Store --> Prioritization["compute_priority\n(Score x Amount Matrix)"]
        Prioritization --> Lifecycle["Lifecycle State Machine\nOPEN -> INVESTIGATING -> PENDING_CUSTOMER\n-> REPRESENTED -> RESOLVED -> CLOSED"]
        Lifecycle --> SLATimer["SLA Tracking & Breach Monitor\n(sla_deadline_hours, is_sla_breached)"]
    end

    subgraph Dispute Evidence Packaging
        CM --> Compiler["compile_evidence_package\n(DisputeEvidencePackage)"]
        Compiler --> Modals["Multi-Modal Evidence Aggregation\n- Transaction Context & Amount\n- Risk & Model Decision Score\n- Device Signals (Phase 40)\n- Geolocation Velocities (Phase 39)\n- 3DS Challenge Verification Proof (Phase 44)"]
        Modals --> Canonical["Deterministic Canonical JSON\n(rfc8785 strict keys & format)"]
        Canonical --> Fingerprint["SHA-256 Package Fingerprint\n(Visa CE3.0 / Mastercard First-Party Trust)"]
    end

    subgraph Event Broadcasting & Operations
        CM -->|Emit CASE_CREATED / ASSIGNED / RESOLVED / EVIDENCE_PACKAGED| Bus["EventBus"]
        Store --> API["FastAPI /v1/cases/*\nPrometheus Metrics"]
        Store --> CLI["CLI: case-create, case-list,\ncase-inspect, case-resolve, case-evidence"]
    end
```

### Key Components

- **Domain Models & Taxonomy (`cases.py`)**: `FraudCase`, `CasePriority` (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), `CaseStatus` (`OPEN`, `INVESTIGATING`, `PENDING_CUSTOMER`, `REPRESENTED`, `RESOLVED`, `CLOSED`), `CaseDisposition` (`CONFIRMED_FRAUD`, `FALSE_POSITIVE`, `MERCHANT_ERROR`, `ACCOUNT_TAKEOVER`, `FRIENDLY_FRAUD`, `UNRESOLVED`), and append-only `CaseNote`.
- **Dynamic Priority & Invariant Validation**: Automatically calculates priority using an empirical severity matrix combining fraud probability and monetary exposure. Enforces strict lifecycle state machines preventing invalid transitions (e.g. directly closing an uninvestigated case).
- **Tamper-Evident Dispute Evidence Dossiers**: `DisputeEvidencePackage` compiles multi-modal transaction context, risk scores, device anomalies, geolocation travel velocities, and 3DS step-up verification proofs into a canonicalized JSON payload hashed with SHA-256 for network chargeback representment.
- **SLA Countdown & Breach Surveillance**: Computes SLA deadlines and dynamically evaluates SLA breach status, supporting priority-based operational queues and automated breach escalation counters.
- **Thread-Safe Storage & Manager**: `CaseStore` provides multi-index querying (by `case_id`, `transaction_id`, `tenant_id`, `status`, `priority`, `assigned_analyst`, and SLA breach). `CaseManager` coordinates safe concurrent state updates with thread-level re-entrant locks.
- **API, CLI & Telemetry Integration**: Exposes REST endpoints (`/v1/cases/*`), CLI commands (`case-create`, `case-list`, `case-inspect`, `case-resolve`, `case-evidence`), and Prometheus counters (`fraud_cases_total`, `fraud_cases_resolved_total`, `fraud_case_sla_breaches_total`, `fraud_evidence_packages_generated_total`).


## Card Scheme Chargeback Ingestion, Reason Code Taxonomy & Automated Representment Gateway (Phase 46)

The platform incorporates an in-process, zero-dependency Card Scheme Chargeback Ingestion, Reason Code Taxonomy, Representment Viability Engine, and Representment Gateway (`chargeback.py`). Inbound post-authorization dispute notices across major networks (Visa, Mastercard, American Express, Discover) are classified, evaluated for liability shift and network deflection rules (Visa CE3.0 / Mastercard First-Party Trust), defended with cryptographic evidence packages, and forwarded into the continuous learning feedback pipeline.

```mermaid
flowchart TD
    subgraph Ingestion & Taxonomy
        Webhook["Gateway Webhook\nPOST /v1/chargebacks/webhook"] --> CM["ChargebackManager"]
        Manual["Analyst API / CLI\nPOST /v1/chargebacks"] --> CM
        CM --> Reg["ReasonCodeRegistry\n(Visa, Mastercard, Amex, Discover)"]
        Reg --> Cat["Normalized ChargebackCategory\n(Fraud CNP/CP, Friendly Fraud, Consumer, Auth)"]
    end

    subgraph Representment Viability Engine
        CM --> Viability["evaluate_representment_viability"]
        Viability --> ThreeDS["3DS Liability Shift Check\n(ECI 05/02/06 / CAVV Verification)"]
        Viability --> CE3["CE3.0 / First-Party Trust Rules\n(>=2 Qualifying Undisputed Prior Txns\nDevice / IP / Account Hash Match)"]
        Viability --> Score["Winnability Score & Action\n(REPRESENT, ACCEPT_LIABILITY, PRE_ARBITRATION)"]
    end

    subgraph Dispute Lifecycle & Store
        CM --> Store[("ChargebackStore\n(Thread-Safe Multi-Index)")]
        Store --> StateMachine["Dispute State Machine\nFIRST_CHARGEBACK -> REPRESENTED\n-> PRE_ARBITRATION -> ARBITRATION\n-> RESOLVED_WON / RESOLVED_LOST / EXPIRED"]
        Store --> ExpireTimer["Response Deadline Surveillance\n(is_expired, days_remaining)"]
    end

    subgraph Continuous Learning & Interfaces
        CM -->|Publish CHARGEBACK_INGESTED / REPRESENTED / RESOLVED| Bus["EventBus"]
        CM -->|Forward Realized Labels on Resolution| Feedback["Feedback LabelIngestionStore\n(DelayedLabelRecord)"]
        Store --> API["FastAPI /v1/chargebacks/*\nPrometheus Exposure & Win Metrics"]
        Store --> CLI["Typer CLI: chargeback-ingest, chargeback-list,\nchargeback-inspect, chargeback-represent, chargeback-resolve"]
    end
```

### Key Components

- **Card Scheme Reason Code Taxonomy (`chargeback.py`)**: `ReasonCodeRegistry` provides standard network taxonomies for Visa (e.g., 10.4 Fraud CNP, 13.1 Merchandise Not Received), Mastercard (e.g., 4837 Fraud CNP, 4853 Cardholder Dispute), American Express (e.g., FR2, C08), and Discover (e.g., 4837, 4853). Automatically infers network-specific response windows (e.g. 30 days Visa/MC, 20 days Amex) and maps reasons into normalized categories.
- **Representment Viability & Liability Shift Engine**: Calculates objective representment winnability scores ($0.0$ to $1.0$) based on protocol verification proofs. Analyzes 3D-Secure Electronic Commerce Indicators (ECI `05` / `02` for fully authenticated liability shift; `06` / `01` for merchant liability protection attempts) and generates representment recommendations.
- **Visa Compelling Evidence 3.0 (CE3.0) & Mastercard First-Party Trust Qualification**: Defends against first-party/friendly fraud claims by auditing prior transaction history for at least 2 qualifying historical undisputed purchases ($\ge 120$ days prior) matching IP address, device fingerprint, or customer credentials.
- **Thread-Safe Dispute Store & Lifecycle Gateway**: `ChargebackStore` manages secondary indexes by `dispute_id`, `transaction_id`, `scheme_dispute_id`, and `tenant_id` with bounded capacity eviction. `ChargebackManager` enforces network state transition constraints, deadline tracking, audit logging, and evidence attachment.
- **Continuous Learning Feedback Loop**: When chargebacks reach terminal resolution (`resolved_won`, `resolved_lost`, `accepted_liability`), `publish_feedback_label` converts outcomes into `DelayedLabelRecord` and submits them to `feedback.py` (`LabelIngestionStore`), updating realized precision/recall surveillance and triggering automated model retraining.
- **API, CLI & Telemetry Integration**: Exposes REST endpoints (`/v1/chargebacks/*`, `/v1/chargebacks/webhook`), Typer CLI commands (`chargeback-ingest`, `chargeback-list`, `chargeback-inspect`, `chargeback-represent`, `chargeback-resolve`), and Prometheus metrics (`fraud_chargebacks_received_total`, `fraud_chargebacks_represented_total`, `fraud_chargebacks_won_total`, `fraud_chargebacks_lost_total`, `fraud_chargeback_financial_exposure_amount`).


## Phase 47 — Account Takeover (ATO) Detection, Credential Stuffing Defense & Identity Session Anomaly Engine

Phase 47 introduces an in-process, zero-external-dependency Account Takeover (ATO) Detection, Credential Stuffing Defense, and Identity Session Anomaly Engine (`ato.py`). It analyzes authentication attempts, sensitive account profile mutations (password, email, phone, and MFA resets), device recognition, and impossible travel velocity to prevent account hijacking before fraudulent card charges occur.

```mermaid
flowchart TD
    subgraph Authentication & Telemetry Ingestion
        LoginReq["Login Event\nPOST /v1/ato/login / CLI"] --> Engine["AtoEngine"]
        ActionReq["Action / Txn Eval\nPOST /v1/ato/evaluate"] --> Engine
        MutationReq["Profile Mutation\nPOST /v1/ato/accounts/{id}/mutations"] --> Tracker["AccountProfileTracker"]
    end

    subgraph Multi-Factor ATO Scoring Engine
        Engine --> Stuffing["CredentialStuffingDetector\n(Sliding Window IP Spraying\nDistributed Brute Force / Subnet Bursts)"]
        Engine --> Geo["Impossible Travel Calculation\n(Haversine Distance > 800 km/h)"]
        Engine --> Device["Device Recognition\n(Known Device Fingerprints)"]
        Engine --> Mutation["Mutation Risk Decay\n(Password/MFA Reset < 1h: +0.40, < 24h: +0.20)"]
        Engine --> Dormant["Dormancy Spike Detection\n(Inactive > 90 Days Sudden Active)"]
        Stuffing --> Score["Composite Calibrated Risk Score\n[0.0, 1.0]"]
        Geo --> Score
        Device --> Score
        Mutation --> Score
        Dormant --> Score
    end

    subgraph Enforcement & Store
        Score --> Policy{"Policy Decision"}
        Policy -->|< 0.40| Allow["ALLOW\nUpdate Profile Baseline"]
        Policy -->|0.40 - 0.74| Challenge["CHALLENGE\nStep-Up MFA Trigger"]
        Policy -->|>= 0.75| Quarantine["QUARANTINE_ACCOUNT / BLOCK\nLock User Account & Audit"]
        Quarantine --> Store[("AtoStore\n(Thread-Safe Multi-Index)")]
        Allow --> Store
        Challenge --> Store
    end

    subgraph Interfaces & Continuous Surveillance
        Engine -->|Publish ATO_LOGIN_EVALUATED / QUARANTINED| Bus["EventBus"]
        Store --> API["FastAPI /v1/ato/*\nPrometheus ATO Counters"]
        Store --> CLI["Typer CLI: ato-login, ato-eval,\nato-inspect, ato-quarantine, ato-release, ato-mutation"]
    end
```

### Key Components

- **ATO Risk Factors & Decision Policies (`ato.py`)**: `AtoRiskFactor` categorizes compromise indicators (`NEW_DEVICE`, `IMPOSSIBLE_TRAVEL`, `RECENT_CREDENTIAL_CHANGE`, `RECENT_MFA_RESET`, `CREDENTIAL_STUFFING_BURST`, `PASSWORD_SPRAYING`, `DORMANT_ACCOUNT_REACTIVE`, `ACCOUNT_QUARANTINED`). `AtoAction` provides deterministic enforcement (`ALLOW`, `CHALLENGE`, `QUARANTINE_ACCOUNT`, `BLOCK`).
- **Sliding-Window Credential Stuffing & Spraying Detection (`CredentialStuffingDetector`)**: Maintains sliding windows of authentication failures across IPs, user accounts, and `/24` or `/48` subnet prefixes to detect distributed password spraying botnets and multi-IP brute force attacks.
- **Account Behavioral Baselines & Mutation Decay (`AccountProfileTracker`)**: Maintains known device fingerprints, geolocation centroids, dormancy history, and time-decayed penalties for sensitive mutations (e.g., password change within 1 hour adds a severe risk penalty).
- **Impossible Travel Velocity**: Calculates spherical Haversine distances between sequential user events and computes implied travel speed ($km/h$). Velocities exceeding commercial aircraft thresholds ($800\text{ km/h}$) automatically trigger `IMPOSSIBLE_TRAVEL`.
- **Thread-Safe ATO Store & EventBus Streaming**: `AtoStore` provides concurrent, multi-indexed storage with capacity eviction. Lifecycle transitions publish domain events (`ATO_LOGIN_EVALUATED`, `ATO_ACCOUNT_QUARANTINED`, `ATO_ACCOUNT_RELEASED`, `ATO_CREDENTIAL_STUFFING_BURST`) across `EventBus`.
- **API, CLI & Observability**: Exposes FastAPI endpoints (`/v1/ato/login`, `/v1/ato/evaluate`, `/v1/ato/accounts/{user_id}`, `/v1/ato/accounts/{user_id}/mutations`, `/v1/ato/accounts/{user_id}/quarantine`, `/v1/ato/accounts/{user_id}/release`, `/v1/ato/assessments`), Typer CLI subcommands (`ato-login`, `ato-eval`, `ato-inspect`, `ato-quarantine`, `ato-release`, `ato-mutation`), and Prometheus metrics (`fraud_ato_login_attempts_total`, `fraud_ato_risk_assessments_total`, `fraud_ato_quarantines_total`).


## Phase 48 — Anti-Money Laundering (AML) Transaction Monitoring, Structuring/Smurfing Detection & Regulatory SAR Engine

Phase 48 introduces an in-process, zero-external-dependency Anti-Money Laundering (AML) Transaction Monitoring, Structuring/Smurfing Detection, Sanction/PEP Screening, and Regulatory Suspicious Activity Report (SAR) Filing Engine (`aml.py`). It enforces Bank Secrecy Act (BSA), FinCEN, and FATF regulatory compliance across payments, card transactions, and account transfers.

```mermaid
flowchart TD
    subgraph Ingestion & Screening
        TxIn["Financial Transaction Event\nPOST /v1/aml/monitor / CLI"] --> AML["AmlEngine"]
        ScreenReq["Entity Name / PEP Query\nPOST /v1/aml/watchlist/screen"] --> ScreenEngine["WatchlistScreeningEngine\n(Levenshtein, Jaro-Winkler, Soundex)"]
        ScreenEngine --> AML
    end

    subgraph AML Typology Detectors
        AML --> Struct["StructuringDetector\n(Sub-CTR Smurfing Window\n$8,000-$9,999 Cumulative >= $10,000)"]
        AML --> Layer["LayeringDetector\n(Rapid Flow-Through Inflow/Outflow\nTurnover Ratio > 80% within 4h)"]
        AML --> Circ["CircularFlowDetector\n(Round-Trip Counterparty Loops\nA -> B -> A within 48h)"]
        AML --> Juris["HighRiskJurisdictionDetector\n(FATF Blacklist / Graylist Corridors)"]
        Struct --> Scoring["Calibrated Risk Scorer & Policy Matrix\n[0.0, 1.0]"]
        Layer --> Scoring
        Circ --> Scoring
        Juris --> Scoring
        ScreenEngine --> Scoring
    end

    subgraph Policy Routing & Case Escalation
        Scoring --> Policy{"AML Policy Decision"}
        Policy -->|< 0.30| Pass["PASS"]
        Policy -->|0.30 - 0.69| Mon["MONITOR\nLog Alert"]
        Policy -->|0.70 - 0.84| EDD["ENHANCED_DUE_DILIGENCE\nEscalate Case"]
        Policy -->|>= 0.85 or Sanction Match| Block["FILE_SAR_BLOCK\nEscalate Priority P1 Case & Draft SAR"]
        Block --> CaseEscalation["CaseManager\n(Auto-Escalate Cases)"]
        EDD --> CaseEscalation
    end

    subgraph Store & Regulatory SAR Packaging
        AML --> Store[("AmlStore\n(Thread-Safe Multi-Index)")]
        Store --> SAR["SarManager\n(FinCEN BSA Narrative Generator\nAudit Trail & Status State Machine)"]
        SAR --> XML["FinCEN XML Export\nGET /v1/aml/sar/{id}/xml"]
        SAR --> SARAPI["SAR Lifecycle REST & CLI\n(draft -> pending_review -> approved -> filed)"]
    end

    subgraph Streaming & Observability
        AML -->|Publish AML Domain Events| Bus["EventBus"]
        Store --> API["FastAPI /v1/aml/*\n5x Prometheus AML Metrics"]
        Store --> CLI["Typer CLI: aml-monitor, aml-screen,\naml-alerts, aml-sar-*"]
    end
```

### Key Components

- **AML Typologies & Taxonomy (`aml.py`)**: `AmlTypology` models regulatory financial crime patterns (`STRUCTURING_SMURFING`, `RAPID_MOVEMENT_LAYERING`, `ROUND_TRIP_CIRCULAR`, `HIGH_RISK_JURISDICTION`, `SANCTION_PEP_MATCH`). `AmlAction` provides deterministic compliance actions (`PASS`, `MONITOR`, `ENHANCED_DUE_DILIGENCE`, `FILE_SAR_BLOCK`).
- **Phonetic & Fuzzy Watchlist Screening (`WatchlistScreeningEngine`)**: Zero-dependency string matching using Soundex phonetic encoding, Levenshtein edit distance, Jaro, and Jaro-Winkler prefix-weighted similarity against OFAC SDN, PEP, and FATF watchlists.
- **Typology Detectors**:
  - `StructuringDetector`: Detects intentional sub-threshold structuring (e.g. transactions between \$8,000 and \$9,999 aggregating over the \$10,000 Currency Transaction Report threshold in sliding time windows).
  - `LayeringDetector`: Detects rapid movement of funds where inbound deposits are promptly disbursed to third parties with high turnover ratios (>80%) within short dwell windows (<4h).
  - `CircularFlowDetector`: Identifies circular round-trip money movement loops between matching counterparties ($A \to B \to A$) within a 48-hour window.
  - `HighRiskJurisdictionDetector`: Flags transaction corridors involving FATF High-Risk Jurisdictions subject to a Call for Action (Blacklist) and Increased Monitoring (Graylist).
- **AML Engine & Autonomous Case Escalation (`AmlEngine`)**: Evaluates transactions against all active typologies, aggregates weighted scores into a calibrated composite risk metric, routes policy actions, and auto-escalates high-risk alerts into `CaseManager` investigations.
- **Regulatory SAR Packaging & Lifecycle (`SarManager`)**: Formulates comprehensive FinCEN/FATF Suspicious Activity Report narratives, maintains strict audit-trailed state transitions (`DRAFT` $\to$ `PENDING_REVIEW` $\to$ `APPROVED` $\to$ `FILED` / `REJECTED`), and exports compliant FinCEN BSA XML representations.
- **REST API, CLI & Observability**: Exposes FastAPI endpoints (`/v1/aml/monitor`, `/v1/aml/watchlist/screen`, `/v1/aml/alerts`, `/v1/aml/sar`, `/v1/aml/sar/{id}/xml`, `/v1/aml/sar/{id}/status`), Typer CLI subcommands (`aml-monitor`, `aml-screen`, `aml-alerts`, `aml-sar-generate`, `aml-sar-list`, `aml-sar-status`), and Prometheus metrics (`fraud_aml_transactions_monitored_total`, `fraud_aml_alerts_total`, `fraud_aml_sanction_hits_total`, `fraud_aml_sars_created_total`, `fraud_aml_sars_filed_total`).


## Phase 49 — Synthetic Identity Fraud (SIF) Detection, PII Inconsistency & Identity Verification Engine

Phase 49 delivers an in-process, zero-external-dependency Synthetic Identity Fraud (SIF) Detection, PII Inconsistency, and Identity Verification Engine (`synthetic_identity.py`). It protects credit card originations and account lifecycle against synthetic identities fabricated by splicing authentic PII (e.g., dormant/minor SSNs) with fabricated personal details, burner phones, and drop addresses.

```mermaid
flowchart TD
    subgraph Ingestion
        AppIn["Credit Applicant Profile\nPOST /v1/sif/evaluate / CLI"] --> SIF["SyntheticIdentityEngine"]
    end

    subgraph Verification Modules
        SIF --> SSN["SsnValidator\n(SSA Area Allocations, Group Progression,\nDOB Consistency, Death Master File)"]
        SIF --> PII["PiiInconsistencyEngine\n(Cross-Profile PII Fragmentation,\nPairwise Fuzzy Similarity)"]
        SIF --> Trk["SyntheticIdentityTracker\n(Drop Address, Burner Phone, Piggybacking,\nSliding-Window Application Velocity)"]
        SSN --> Scorer["Calibrated Composite Risk Scorer\n[0.0, 1.0]"]
        PII --> Scorer
        Trk --> Scorer
    end

    subgraph Policy Routing & Ecosystem Integration
        Scorer --> Policy{"SIF Policy Decision"}
        Policy -->|< 0.35| Appr["APPROVE (Low Risk)"]
        Policy -->|0.35 - 0.64| StepUp["STEP_UP_IDENTITY_PROOFING\n(Medium Risk)"]
        Policy -->|0.65 - 0.84| Review["MANUAL_REVIEW\n(High Risk)"]
        Policy -->|>= 0.85 or Deceased| Block["DECLINE_BLOCK\n(Critical Risk)"]
        StepUp --> Challenge["ChallengeManager\n(Issue OTP Challenge)"]
        Review --> CaseEscalation["CaseManager\n(Create Investigation Case)"]
    end

    subgraph Store & Observability
        SIF --> Store[("SyntheticIdentityStore\n(Thread-Safe Multi-Index)")]
        SIF -->|SIF Domain Events| Bus["EventBus"]
        Store --> API["FastAPI /v1/sif/*\n4x Prometheus SIF Metrics"]
        Store --> CLI["Typer CLI: sif-eval, sif-check-ssn,\nsif-inspect, sif-summary"]
    end
```

### Key Components

- **SIF Taxonomies & Domain Models (`synthetic_identity.py`)**: `SifRiskLevel` (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), `SifAction` (`APPROVE`, `STEP_UP_IDENTITY_PROOFING`, `MANUAL_REVIEW`, `DECLINE_BLOCK`), `SifAnomalyType` (`SSN_DOB_MISMATCH`, `DECEASED_SSN_MATCH`, `INVALID_SSN_STRUCTURE`, `UNISSUED_SSN_AREA`, `SYNTHETIC_VELOCITY_BURST`, `PII_ELEMENT_FRAGMENTATION`, `AUTHORIZED_USER_ABUSE`, `HIGH_RISK_ADDRESS_DROP`, `BURNER_PHONE_DETECTED`), `IdentityProfile`, and `SifAssessment`.
- **SSA Historical Allocation & Death Master File (`SsnValidator`)**: Implements pre-June 25, 2011 SSA geographical area allocations (001-728), group number issuance progression sequences, Railroad Retirement board cessation (1963) verification, post-2011 randomization checks, and simulated Death Master File (DMF) matching.
- **PII Element Fragmentation & Fuzzy Inconsistency (`PiiInconsistencyEngine`)**: Evaluates cross-application PII element fragmentation (e.g., stolen SSN paired with conflicting names/DOBs, identical name/DOB paired with differing SSNs, shared contact info across disparate identities) using Jaro-Winkler and Levenshtein similarity distance algorithms.
- **Velocity, Burner Phone & Piggybacking Surveillance (`SyntheticIdentityTracker`)**: Detects commercial mail drops and PO Boxes, disposable burner and toll-free numbers, authorized user abuse ("credit piggybacking" with multiple unrelated tradelines), and multi-application velocity bursts in sliding time windows.
- **Synthetic Identity Engine & Policy Decisioning (`SyntheticIdentityEngine`)**: Computes calibrated multi-factor composite risk scores, maps policies to actions, auto-escalates `MANUAL_REVIEW` alerts to `CaseManager`, and routes `STEP_UP_IDENTITY_PROOFING` to `ChallengeManager`.
- **Thread-Safe Store (`SyntheticIdentityStore`)**: Multi-index storage with capacity eviction supporting lookups by `applicant_id`, `ssn_hash`, `phone`, `email`, and `tenant_id`.
- **REST API, CLI & Observability**: Exposes FastAPI endpoints (`/v1/sif/evaluate`, `/v1/sif/assessments/{id}`, `/v1/sif/assessments`, `/v1/sif/profiles`, `/v1/sif/applicants/{id}`, `/v1/sif/metrics`), Typer CLI subcommands (`sif-eval`, `sif-check-ssn`, `sif-inspect`, `sif-summary`), and Prometheus metrics (`fraud_sif_evaluations_total`, `fraud_sif_anomalies_total`, `fraud_sif_high_risk_total`, `fraud_sif_blocks_total`).


## Phase 50 — Mule Account & Rapid Fund Exfiltration Detection Engine

Phase 50 delivers an in-process, zero-external-dependency Money Mule & Rapid Fund Exfiltration Detection Engine (`mule_detection.py`) that identifies pass-through conduit accounts, dwell-time compression, fan-in/fan-out counterparty hubs, dormancy-to-burst transitions, and high-risk cash-out off-ramps.

```mermaid
flowchart TD
    subgraph Ingestion
        XferIn["Transfer Event\nPOST /v1/mule/transfers / CLI"] --> Engine["MuleDetectionEngine"]
    end

    subgraph Analyzers
        Engine --> PT["PassThroughAnalyzer\n(Inflow/Outflow Ratio & Dwell Time)"]
        Engine --> Net["MuleNetworkAnalyzer\n(Fan-In / Fan-Out & New Counterparties)"]
        Engine --> Dorm["DormancyBurstDetector\n(Dormancy Wake-Up Spikes)"]
    end

    subgraph Policy & Escalation
        PT & Net & Dorm --> Score["Composite Mule Risk Score"]
        Score -->|"< 0.35"| Allow["ALLOW"]
        Score -->|"0.35 - 0.59"| StepUp["STEP_UP_VERIFICATION"]
        Score -->|"0.60 - 0.79"| Hold["HOLD_FUNDS\n(Event: MULE_FUNDS_HELD)"]
        Score -->|"0.80 - 0.89"| Freeze["FREEZE_ACCOUNT\n(CaseManager + MULE_ACCOUNT_FLAGGED)"]
        Score -->|">= 0.90"| SAR["ESCALATE_AML_SAR\n(CaseManager + MULE_ACCOUNT_FLAGGED)"]
    end
```

- **Pass-Through & Off-Ramp Analysis (`PassThroughAnalyzer`)**: Tracks inbound vs. outbound volume, pass-through ratio (`min(outbound, inbound) / inbound`), median dwell time between funding and liquidation, and high-risk off-ramp channels (`crypto_exchange`, `wire_international`, `prepaid_load`, `atm_cash_out`).
- **Network Topology & Dormancy (`MuleNetworkAnalyzer`, `DormancyBurstDetector`)**: Flags fan-in collection, fan-out dispersion, 24h new-counterparty floods, and dormant accounts waking up into sudden high-volume transfer bursts.
- **Store, EventBus, API & CLI**: `MuleAccountStore` provides thread-safe secondary indexing (`tenant_id`, `counterparty_id`) with capacity eviction; `MuleDetectionEngine` emits `MULE_ACTIVITY_EVALUATED`, `MULE_FUNDS_HELD`, `MULE_ACCOUNT_FLAGGED`, and `MULE_NETWORK_DETECTED`; FastAPI exposes `/v1/mule/*` endpoints and Prometheus counters (`fraud_mule_evaluations_total`, `fraud_mule_anomalies_total`, `fraud_mule_flagged_total`); Typer CLI exposes `mule-ingest`, `mule-eval`, `mule-inspect`, and `mule-summary`.


## Phase 51 — First-Party Fraud, Refund Abuse & Friendly Fraud Detection Engine

Phase 51 delivers an in-process, zero-external-dependency First-Party Fraud, Refund Abuse & Friendly Fraud Detection Engine (`first_party_fraud.py`) that detects cardholders and account owners exploiting merchant refund, chargeback, and promotional policies via serial friendly-fraud disputes, wardrobing/return abuse, Item-Not-Received (INR) claim bursts, and linked-account promo cycling.

```mermaid
flowchart TD
    subgraph Ingestion
        EvtIn["Order / Refund / INR / Dispute / Promo\nPOST /v1/first-party/events / CLI"] --> Engine["FirstPartyFraudEngine"]
    end

    subgraph Analyzers
        Engine --> RA["RefundAbuseAnalyzer\n(Return Ratio, Wardrobing, INR, Skew)"]
        Engine --> FF["FriendlyFraudAnalyzer\n(Verified-Delivery Disputes & Velocity)"]
        Engine --> PA["PromoAbuseDetector\n(Bonus Abuse & Linked Account Cycling)"]
    end

    subgraph Policy & Escalation
        RA & FF & PA --> Score["Composite First-Party Risk Score"]
        Score -->|"< 0.25"| Allow["ALLOW"]
        Score -->|"0.25 - 0.44"| Sig["REQUIRE_SIGNATURE_DELIVERY"]
        Score -->|"0.45 - 0.64"| Restrict["RESTRICT_INSTANT_REFUND\n(Event: REFUND_RESTRICTED)"]
        Score -->|"0.65 - 0.84"| Deny["DENY_REFUND_MANUAL_REVIEW\n(CaseManager + REFUND_RESTRICTED)"]
        Score -->|">= 0.85"| Revoke["REVOKE_PRIVILEGES_AND_BLOCK\n(CaseManager + ACCOUNT_PRIVILEGES_REVOKED)"]
    end
```

- **Refund & Return Abuse (`RefundAbuseAnalyzer`)**: Tracks return-to-purchase ratios, high-value refund skew, wardrobing-prone category returns (`apparel`, `luxury_fashion`, `electronics`, `cameras`, `jewelry`, `footwear`), and INR claim bursts.
- **Friendly Fraud & Promo Cycling (`FriendlyFraudAnalyzer`, `PromoAbuseDetector`)**: Flags serial disputes on 3DS/AVS/CVV/delivery-confirmed orders, pre-dispute velocity, promo bonus abuse, and accounts linked via shared shipping address or payment instrument hashes.
- **Store, EventBus, API & CLI**: `FirstPartyAccountStore` provides thread-safe secondary indexing (`tenant_id`, `shipping_address_hash`, `payment_instrument_hash`) with capacity eviction; `FirstPartyFraudEngine` emits `FIRST_PARTY_EVALUATED`, `FIRST_PARTY_ABUSE_DETECTED`, `REFUND_RESTRICTED`, and `ACCOUNT_PRIVILEGES_REVOKED`; FastAPI exposes `/v1/first-party/*` endpoints and Prometheus counters (`fraud_first_party_evaluations_total`, `fraud_first_party_anomalies_total`, `fraud_first_party_restricted_total`); Typer CLI exposes `first-party-ingest`, `first-party-eval`, `first-party-inspect`, and `first-party-summary`.


## Extension guidance


New estimators should be added through `EstimatorType` and `_build_base_estimator`
so calibration, splitting, evaluation, artifact validation, and explanations
remain shared. New reports should expose a serializable payload builder in the
relevant domain module and keep rendering/orchestration in `cli.py` or
`reporting.py`. New API fields should be additive where possible and covered by
request-validation and response tests.
