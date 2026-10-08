# Phase 33 Progress Record

## Overview
- **Phase**: Phase 33 — Real-Time Graph Node Cache & Speculative Model Evaluation
- **Branch**: `phase/node-cache-speculation`
- **Status**: In progress

## Task Status
- [x] Task 1: Fix baseline mypy unreachable code in `InferenceStageProcessor` and tighten batch evaluation timeout robustness.
- [x] Task 2: Implement `NodeCachePolicy`, `NodeCacheStats`, and `StageExecutionCache` in `pipeline.py`.
- [x] Task 3: Integrate node caching into `DecisionGraphExecutor` and `StageExecutionResult`.
- [x] Task 4: Implement speculative asynchronous model inference execution in `DecisionGraphExecutor`.
- [x] Task 5: Enhance `create_default_fraud_pipeline` with cache and speculative evaluation options.
- [x] Task 6: Update audit event export in `audit.py` with cache and speculative execution metadata.
- [ ] Task 7: Integrate cache endpoints (`/v1/pipeline/cache/stats`, `/v1/pipeline/cache/clear`) and Prometheus metrics in `api.py`.
- [ ] Task 8: Add cache and speculative evaluation options to `pipeline-eval` CLI command in `cli.py`.
- [ ] Task 9: Implement comprehensive unit and integration tests across pipeline, audit, API, and CLI.
- [ ] Task 10: Update documentation (`ARCHITECTURE.md`, `README.md`, `ROADMAP.md`).

## Acceptance Criteria Status
| Criterion | Status | Evidence |
|-----------|--------|----------|
| AC-1 (Baseline resilience) | Met | Strict mypy passes with 0 unreachable statement errors; `pipeline-eval` tests execute reliably without spurious threadpool timeout degradation (5/5 passed). |
| AC-2 (Node cache semantics) | Met | `StageExecutionCache` verified with TTL expiration, LRU eviction, deterministic key generation, and telemetry counters in `tests/test_pipeline.py`. |
| AC-3 (Speculative inference correctness) | Met | Speculative inference hit verified when features unchanged; safe discard and re-computation verified on feature delta or rule short-circuit in `tests/test_pipeline.py`. |
| AC-4 (Audit & telemetry fidelity) | Met | Scoring audit events serialize and redact cache hits/misses, speculative executed, and speculative hit flags in `tests/test_audit.py`. |
| AC-5 (API serving integration) | Pending | Implementation pending |
| AC-6 (CLI workflow) | Pending | Implementation pending |
| AC-7 (Quality & coverage standards) | Pending | Full suite pending |

## Decision Log
| Date | Decision | Alternatives Considered | Rationale |
|------|----------|-------------------------|-----------|
| 2026-10-08 | Use in-memory LRU+TTL cache with bounded capacity for stage caching | External cache (Redis/Memcached), unbounded dict | Keeps project zero-dependency, self-contained, microsecond-latency oriented, and safe against memory exhaustion. |
| 2026-10-08 | Speculative inference evaluates on raw transaction features and verifies against post-enrichment delta | Blocking speculation until enrichment finishes, speculation without verification | Running inference concurrently with enrichment achieves real parallelism; validating feature delta guarantees zero accuracy divergence. |
| 2026-10-08 | Support optional per-node cache policies rather than global caching | Blanket cache for all nodes | Not all nodes are pure or beneficial to cache; allowing per-node policy allows operators to target expensive pure stages (e.g. rules, transforms). |

## Validation Commands
```bash
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m mypy
.venv/Scripts/python.exe -m pytest -ra
```
