# Phase 41 Progress Record

### Overview
- **Phase**: Phase 41 — Real-Time Merchant & MCC Risk Profiling, Chargeback Surveillance & Bust-Out Anomaly Detection
- **Branch**: `phase/merchant-risk-surveillance`
- **Status**: In Progress

## Task Status
- [x] Task 1: Create `src/fraud_detection/merchant.py` with `MerchantRiskError`, `MCCCategory`, `MCCRiskProfile`, `resolve_mcc_profile`, and unit tests in `tests/test_merchant.py`.
- [x] Task 2: Implement `calculate_empirical_bayes_fraud_rate`, `calculate_wilson_upper_bound`, and `MerchantTransactionEvent` in `merchant.py` with tests in `tests/test_merchant.py`.
- [x] Task 3: Implement `MerchantTrackerConfig`, `MerchantProfile`, `MerchantAnomalyReport`, and `MerchantTracker` in `merchant.py` with tests in `tests/test_merchant.py`.
- [ ] Task 4: Implement `MerchantFeatureExtractor` in `merchant.py` with sequential batch/single extraction and tests in `tests/test_merchant.py`.
- [ ] Task 5: Integrate merchant rules (`build_merchant_chargeback_rule`, `build_merchant_bust_out_rule`) in `rules.py` and `MerchantEnrichmentStageProcessor` in `pipeline.py` with tests.
- [ ] Task 6: Implement `build_merchant_anomaly_audit_event` in `audit.py` with tests in `tests/test_audit.py`.
- [ ] Task 7: Implement `/v1/merchant/*` API endpoints and Prometheus telemetry in `api.py` with tests in `tests/test_api.py`.
- [ ] Task 8: Implement `merchant-surveillance` and `merchant-enrich` CLI commands in `cli.py` with tests in `tests/test_cli.py`.
- [ ] Task 9: Update documentation (`ARCHITECTURE.md`, `README.md`, `ROADMAP.md`, `docs/PROGRESS.md`) and verify Completion Gate.

## Acceptance Criteria Status
| Criterion | Status | Evidence |
|-----------|--------|----------|
| AC-1 (MCC Taxonomy & Empirical Bayes Smoothing) | PASSED | Unit tests in `tests/test_merchant.py` (4 passed) |
| AC-2 (Streaming Merchant Tracker & Anomaly Heuristics) | PASSED | Unit tests in `tests/test_merchant.py` (6 passed) |
| AC-3 (Zero-Lookahead Feature Extraction) | PENDING | |
| AC-4 (Pipeline & Decision Rule Integration) | PENDING | |
| AC-5 (Structured Audit Trail & Redaction) | PENDING | |
| AC-6 (API Serving & Operational Telemetry) | PENDING | |
| AC-7 (CLI Workflows & Gating) | PENDING | |
| AC-8 (Quality Standards & 100% Green CI) | PENDING | |

## Decision Log
| Date | Decision | Alternatives Considered | Rationale |
|------|----------|-------------------------|-----------|
| 2026-10-21 | Select Phase 41: Real-Time Merchant & MCC Risk Profiling, Chargeback Surveillance & Bust-Out Anomaly Detection | External commercial KYB/merchant risk APIs | Self-contained, zero-dependency Empirical Bayes smoothing, ISO 18245 MCC risk taxonomy, Welford ticket-size Z-scores, and sliding-window velocity surveillance fit the leakage-safe reference architecture. |
| 2026-10-21 | Create branch `phase/merchant-risk-surveillance` from `origin/main` | Reuse old merged branch `phase/device-fingerprint-clustering` | Starts cleanly from merged `origin/main` (`1cc7812`) and isolates Phase 41 changes for continuous remote CI and PR delivery. |

## Validation Commands & Results
- Baseline `ruff format --check src tests`: PASSED (49 files formatted)
- Baseline `ruff check src tests`: PASSED
- Baseline `mypy src`: PASSED (0 issues in 25 source files)
- Baseline `build` & `twine check dist/*`: PASSED
