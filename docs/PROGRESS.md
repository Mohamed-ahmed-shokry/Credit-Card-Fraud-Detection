# Phase 35 Progress Record

## Overview
- **Phase**: Phase 35 — Decision Graph Stage Circuit Breakers, Fast-Fail Isolation & Chaos Fault Injection
- **Branch**: `phase/graph-replay-counterfactual`
- **Status**: In progress

## Task Status
- [x] Task 1: Define stage circuit breaker primitives (`PipelineCircuitOpenError`, `StageCircuitState`, `StageCircuitPolicy`, `StageCircuitStats`, `StageCircuitBreaker`, `StageCircuitRegistry`) in `pipeline.py`.
- [x] Task 2: Integrate stage circuit breaking into `StageStatus`, `StageExecutionResult`, `PipelineNode`, `PipelineExecutionResult`, `DecisionGraphExecutor`, and `create_default_fraud_pipeline` in `pipeline.py`.
- [x] Task 3: Implement stage chaos fault-injection primitives (`StageFaultType`, `StageFaultSpec`, `StageChaosMetrics`, `PipelineChaosReport`) and `simulate_pipeline_chaos` in `pipeline.py`.
- [x] Task 4: Add `circuit_short_circuits` to `build_scoring_audit_event` and implement `build_pipeline_chaos_audit_event` in `audit.py` with unit tests in `tests/test_audit.py`.
- [x] Task 5: Expose `/v1/pipeline/circuits`, `/v1/pipeline/circuits/reset`, and `/v1/pipeline/chaos` endpoints and Prometheus metrics in `api.py` with tests in `tests/test_api.py`.
- [ ] Task 6: Add circuit breaker flags to `pipeline-eval` and implement `pipeline-chaos` CLI command in `cli.py` with tests in `tests/test_cli.py`.
- [ ] Task 7: Update documentation (`ARCHITECTURE.md`, `README.md`, `ROADMAP.md`, `docs/PROGRESS.md`) and verify Completion Gate.

## Acceptance Criteria Status
| Criterion | Status | Evidence |
|-----------|--------|----------|
| AC-1 (Stage Circuit Breaker State Machine & Fast-Fail Isolation) | **Verified** | Verified in `tests/test_pipeline.py`: state machine transitions, fast-fail 0ms short-circuit, half-open probe recovery, required node circuit open error. |
| AC-2 (Deterministic Stage Chaos Fault Injection & SLA Verification) | **Verified** | Verified in `tests/test_pipeline.py`: exception/timeout/latency spike simulation, SLA budget breach tracking, circuit tripping and recovery detection. |
| AC-3 (Audit Trail Fidelity & Redaction) | **Verified** | Verified in `tests/test_audit.py`: circuit_short_circuits in scoring event, pipeline_chaos event payload structure and sensitive metadata redaction. |
| AC-4 (API Serving & Prometheus Telemetry) | **Verified** | Verified in `tests/test_api.py`: GET `/v1/pipeline/circuits`, POST `/v1/pipeline/circuits/reset`, POST `/v1/pipeline/chaos`, circuit_short_circuits scoring telemetry, Prometheus metrics export. |
| AC-5 (CLI Evaluation & Chaos Workflows) | **Pending** | — |
| AC-6 (Quality Standards & Zero Regressions) | **Pending** | — |

## Decision Log
| Date | Decision | Alternatives Considered | Rationale |
|------|----------|-------------------------|-----------|
| 2026-10-13 | Continue on active feature branch `phase/graph-replay-counterfactual` with upstream and open draft PR #12 | Create a separate branch from `main` | Follows §4A rule ("if you are on a feature branch with an upstream, use it") and preserves unmerged Phase 34 commits on `HEAD`. |
| 2026-10-13 | Implement per-stage circuit breakers (`StageCircuitBreaker`, `StageCircuitRegistry`) and deterministic chaos simulation (`simulate_pipeline_chaos`) in `pipeline.py` | Service-level circuit breaker only | Service-level `CircuitBreaker` in `api.py` fails over the entire model response, whereas stage-level circuit breakers isolate slow/failing non-critical DAG nodes (e.g., enrichment, calibration) in $O(1)$ without waiting for stage timeouts on every request. |
| 2026-10-13 | Leave untracked `uv.lock` untouched | Stage or remove `uv.lock` | Follows §1 hard rule ("Never discard, overwrite, stage, or commit changes you did not make"). |

## Validation Commands & Results
- Baseline `ruff check src tests`: Passed (0 errors)
- Baseline `mypy src`: Passed (0 issues in 20 source files)
- Baseline `pytest --cov=fraud_detection --cov-report=term-missing -q`: Passed (798 passed, 97.05% branch coverage)
