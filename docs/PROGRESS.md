# Phase 34 Progress Record

## Overview
- **Phase**: Phase 34 — Dynamic Decision Graph Replay & Counterfactual Policy Optimization
- **Branch**: `phase/graph-replay-counterfactual`
- **Status**: In progress

## Task Status
- [ ] Task 1: Define `PipelineReplayReport`, `PipelineDiscrepancyDetail`, and `replay_audit_log_through_pipeline` in `pipeline.py`.
- [ ] Task 2: Implement counterfactual policy evaluation and threshold optimization helpers in `pipeline.py`.
- [ ] Task 3: Add `build_pipeline_replay_audit_event` in `audit.py`.
- [ ] Task 4: Expose `/v1/pipeline/replay` and `/v1/pipeline/counterfactual` endpoints with Prometheus telemetry in `api.py`.
- [ ] Task 5: Implement `pipeline-replay` and `pipeline-counterfactual` CLI commands in `cli.py`.
- [ ] Task 6: Add comprehensive unit and integration tests across pipeline, audit, API, and CLI suites.
- [ ] Task 7: Update documentation (`ARCHITECTURE.md`, `README.md`, `ROADMAP.md`) and verify Completion Gate.

## Acceptance Criteria Status
| Criterion | Status | Evidence |
|-----------|--------|----------|
| AC-1 (Audit Replay Fidelity) | In progress | Pending `replay_audit_log_through_pipeline` implementation. |
| AC-2 (Counterfactual Transition & Cost Evaluation) | In progress | Pending counterfactual simulation engine in `pipeline.py`. |
| AC-3 (Audit Trail Tracing) | In progress | Pending `build_pipeline_replay_audit_event` in `audit.py`. |
| AC-4 (API Serving Contract) | In progress | Pending `/v1/pipeline/replay` & `/v1/pipeline/counterfactual` endpoints. |
| AC-5 (CLI Evaluation Suite) | In progress | Pending `pipeline-replay` & `pipeline-counterfactual` CLI commands. |
| AC-6 (Quality Standards & Zero Regressions) | In progress | Full suite, strict mypy, ruff, branch coverage >= 97.0%. |

## Decision Log
| Date | Decision | Alternatives Considered | Rationale |
|------|----------|-------------------------|-----------|
| 2026-10-12 | Implement stream-based JSONL audit event replaying through `DecisionGraphExecutor` | Batch dataframe-only replay | Streaming JSONL mirrors real production audit logs, supports inline or external features, and keeps memory bounded. |
| 2026-10-12 | Reuse `PolicyTransitionMatrix` from `evaluation.py` for 3x3 migration matrix | Separate graph-specific transition structure | Standardizes transition metrics (ALLOW, CHALLENGE, DENY) and ensures consistency across audit backtesting and graph counterfactuals. |
| 2026-10-12 | Calculate counterfactual expected cost by joining ground-truth labels if provided | Label-free cost approximations only | Providing ground-truth labels enables exact false-deny and missed-fraud loss calculation alongside review friction costs. |

## Validation Commands
```bash
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m mypy
.venv/Scripts/python.exe -m pytest -ra
```
