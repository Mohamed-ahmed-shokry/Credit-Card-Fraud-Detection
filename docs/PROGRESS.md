# Phase 37 Progress Record

## Overview
- **Phase**: Phase 37 — Automated Closed-Loop Retraining, Candidate Verification & Zero-Downtime Safe Promotion
- **Branch**: `phase/closed-loop-retraining-promotion`
- **Status**: In progress

## Task Status
- [x] Task 1: Create `src/fraud_detection/retraining.py` with domain data models (`RetrainingConfig`, `ChallengerVerificationReport`, `RetrainingStatus`, `RetrainingError`) and `RetrainingDatasetBuilder` with unit tests in `tests/test_retraining.py`.
- [x] Task 2: Implement `ChallengerTrainer` with hyperparameter configuration and threshold tuning in `retraining.py` with tests in `tests/test_retraining.py`.
- [x] Task 3: Implement `ChallengerSafetyVerifier` evaluating PR-AUC, recall, ECE, slice disparity, and cost vs. champion with tests in `tests/test_retraining.py`.
- [x] Task 4: Implement certified artifact export and attestation signing (`export_certified_candidate`) in `retraining.py` with tests in `tests/test_retraining.py`.
- [x] Task 5: Implement audit event builders (`build_challenger_retrained_audit_event`, `build_model_promoted_audit_event`) in `audit.py` with tests in `tests/test_audit.py`.
- [ ] Task 6: Implement zero-downtime hot-reload and dynamic canary promotion endpoints and telemetry in `api.py` with tests in `tests/test_api.py`.
- [ ] Task 7: Implement CLI command `auto-retrain` in `cli.py` with tests in `tests/test_cli.py`.
- [ ] Task 8: Update documentation (`ARCHITECTURE.md`, `README.md`, `ROADMAP.md`, `docs/PROGRESS.md`) and verify Completion Gate.

## Acceptance Criteria Status
| Criterion | Status | Evidence |
|-----------|--------|----------|
| AC-1 (Retraining Dataset Materialization) | **Verified** | Verified in `tests/test_retraining.py`: `RetrainingDatasetBuilder` builds `ValidatedDataset` from `RetrainingRecord` and audit logs + labels with temporal filtering, capacity checks, and multi-class assertions. |
| AC-2 (Automated Challenger Training & Tuning) | **Verified** | Verified in `tests/test_retraining.py`: `ChallengerTrainer` maps hyperparameters, handles stratified/temporal CV splits, tunes decision thresholds, and records training duration and lineage metadata. |
| AC-3 (Multi-Faceted Challenger Verification & Safety Gating) | **Verified** | Verified in `tests/test_retraining.py`: `ChallengerSafetyVerifier` evaluates candidate vs champion across PR-AUC, recall parity, Expected Calibration Error (ECE), financial cost differential, and sub-population slice disparity. |
| AC-4 (Cryptographic Attestation & Lineage Signing) | **Verified** | Verified in `tests/test_retraining.py`: `export_certified_candidate` exports artifact bundles with Ed25519 signatures and `verify_certified_candidate` validates attestation digest, file SHA-256 integrity, and cryptographic signatures. |
| AC-5 (Zero-Downtime Hot-Reload & Safe Canary Promotion) | Pending | — |
| AC-6 (Audit Trail, API Endpoints & Telemetry) | Pending | — |
| AC-7 (CLI Workflow & Exit Code Gating) | Pending | — |
| AC-8 (Quality Standards & Zero Regressions) | Pending | — |

## Decision Log
| Date | Decision | Alternatives Considered | Rationale |
|------|----------|-------------------------|-----------|
| 2026-10-15 | Create dedicated branch `phase/closed-loop-retraining-promotion` from HEAD | Continue on old branch | Follows §4 protocol: Phase 36 was merged to main; creating a clean feature branch allows independent lifecycle and PR review. |
| 2026-10-15 | Introduce dedicated `retraining.py` module | Put logic into `feedback.py` or `model.py` | Maintains separation of concerns: `feedback.py` manages ground-truth labels and surveillance trigger events; `retraining.py` handles dataset materialization, model training, candidate multi-factor verification, and promotion. |
| 2026-10-15 | Preserve untracked `uv.lock` untouched | Stage or remove `uv.lock` | Strictly follows §1 hard rule ("Never discard, overwrite, stage, or commit changes you did not make"). |

## Validation Commands & Results
- Baseline `ruff format --check src tests`: PASSED (41 files cleanly formatted)
- Baseline `ruff check src tests`: PASSED (0 errors)
- Baseline `mypy src`: PASSED (0 issues in 21 source files)
- Baseline `pip-audit --skip-editable`: PASSED (0 known vulnerabilities found)
- Baseline `pytest --cov=fraud_detection --cov-report=term-missing`: PASSED (838 passed, 97.15% branch coverage)
- Baseline `build` & `twine check dist/*`: PASSED (sdist and wheel valid)


