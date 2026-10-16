# Phase 37 Progress Record

## Overview
- **Phase**: Phase 37 — Automated Closed-Loop Retraining, Candidate Verification & Zero-Downtime Safe Promotion
- **Branch**: `phase/retraining-ci-stabilization`
- **Status**: Complete

## Task Status
- [x] Task 1: Create `src/fraud_detection/retraining.py` with domain data models (`RetrainingConfig`, `ChallengerVerificationReport`, `RetrainingStatus`, `RetrainingError`) and `RetrainingDatasetBuilder` with unit tests in `tests/test_retraining.py`.
- [x] Task 2: Implement `ChallengerTrainer` with hyperparameter configuration and threshold tuning in `retraining.py` with tests in `tests/test_retraining.py`.
- [x] Task 3: Implement `ChallengerSafetyVerifier` evaluating PR-AUC, recall, ECE, slice disparity, and cost vs. champion with tests in `tests/test_retraining.py`.
- [x] Task 4: Implement certified artifact export and attestation signing (`export_certified_candidate`) in `retraining.py` with tests in `tests/test_retraining.py`.
- [x] Task 5: Implement audit event builders (`build_challenger_retrained_audit_event`, `build_model_promoted_audit_event`) in `audit.py` with tests in `tests/test_audit.py`.
- [x] Task 6: Implement zero-downtime hot-reload and dynamic canary promotion endpoints and telemetry in `api.py` with tests in `tests/test_api.py`.
- [x] Task 7: Implement CLI command `auto-retrain` in `cli.py` with tests in `tests/test_cli.py`.
- [x] Task 8: Update documentation (`ARCHITECTURE.md`, `README.md`, `ROADMAP.md`, `docs/PROGRESS.md`) and verify Completion Gate.

## Acceptance Criteria Status
| Criterion | Status | Evidence |
|-----------|--------|----------|
| AC-1 (Retraining Dataset Materialization) | **Verified** | Verified in `tests/test_retraining.py`: `RetrainingDatasetBuilder` builds `ValidatedDataset` from `RetrainingRecord` and audit logs + labels with temporal filtering, capacity checks, and multi-class assertions. |
| AC-2 (Automated Challenger Training & Tuning) | **Verified** | Verified in `tests/test_retraining.py`: `ChallengerTrainer` maps hyperparameters, handles stratified/temporal CV splits, tunes decision thresholds, and records training duration and lineage metadata. |
| AC-3 (Multi-Faceted Challenger Verification & Safety Gating) | **Verified** | Verified in `tests/test_retraining.py`: `ChallengerSafetyVerifier` evaluates candidate vs champion across PR-AUC, recall parity, Expected Calibration Error (ECE), financial cost differential, and sub-population slice disparity. |
| AC-4 (Cryptographic Attestation & Lineage Signing) | **Verified** | Verified in `tests/test_retraining.py`: `export_certified_candidate` exports artifact bundles with Ed25519 signatures and `verify_certified_candidate` validates attestation digest, file SHA-256 integrity, and cryptographic signatures. |
| AC-5 (Zero-Downtime Hot-Reload & Safe Canary Promotion) | **Verified** | Verified in `tests/test_api.py`: `/v1/models/reload` provides zero-downtime hot-reload with signature verification; `/v1/models/promote` safely adjusts canary traffic percentage or promotes candidate to full champion. |
| AC-6 (Audit Trail, API Endpoints & Telemetry) | **Verified** | Verified in `tests/test_audit.py` and `tests/test_api.py`: Challenger retrained and model promoted audit events emitted; Prometheus metrics `fraud_retraining_runs_total`, `fraud_challenger_verification_status`, `fraud_model_promotions_total` exported. |
| AC-7 (CLI Workflow & Exit Code Gating) | **Verified** | Verified in `tests/test_cli.py`: `fraud-detect auto-retrain` coordinates dataset materialization, candidate training, safety verification gating, and promotion with exit code 0 on success, exit code 3 on verification rejection, exit code 2 on invalid config, and exit code 1 on pipeline error. |
| AC-8 (Quality Standards & Zero Regressions) | **Verified** | Verified across comprehensive test suites (881 passed, 97.04% branch coverage), ruff check/format clean, strict mypy 0 issues across 22 source files, pip-audit 0 vulnerabilities, build and twine validation passing. |

## Decision Log
| Date | Decision | Alternatives Considered | Rationale |
|------|----------|-------------------------|-----------|
| 2026-10-15 | Create dedicated branch `phase/closed-loop-retraining-promotion` from HEAD | Continue on old branch | Follows §4 protocol: Phase 36 was merged to main; creating a clean feature branch allows independent lifecycle and PR review. |
| 2026-10-15 | Introduce dedicated `retraining.py` module | Put logic into `feedback.py` or `model.py` | Maintains separation of concerns: `feedback.py` manages ground-truth labels and surveillance trigger events; `retraining.py` handles dataset materialization, model training, candidate multi-factor verification, and promotion. |
| 2026-10-15 | Preserve untracked `uv.lock` untouched | Stage or remove `uv.lock` | Strictly follows §1 hard rule ("Never discard, overwrite, stage, or commit changes you did not make"). |
| 2026-10-17 | Fix terminal ANSI escaping in route-eval CLI test, cast isotonic recalibration clip result to float ndarray for mypy strictness, parse string booleans in DelayedLabelRecord, expand test coverage to 97.04% across 881 tests | Skip or weaken tests | Strictly adheres to §1 hard rules: never weaken tests or linter checks; fix underlying types and ANSI formatting to ensure deterministic CI execution across platforms. |

## Validation Commands & Results
- `ruff format --check src tests`: PASSED (43 files cleanly formatted)
- `ruff check src tests`: PASSED (0 errors)
- `mypy src`: PASSED (0 issues in 22 source files)
- `pip-audit --skip-editable`: PASSED (0 known vulnerabilities found)
- `pytest --cov=fraud_detection --cov-report=term-missing`: PASSED (881 passed, 97.04% branch coverage)
- `build` & `twine check dist/*`: PASSED (sdist and wheel valid)



