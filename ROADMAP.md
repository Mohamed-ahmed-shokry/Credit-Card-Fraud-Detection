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

## Phase 1 — Distribution

The package already builds cleanly and passes `twine check` in CI. Nothing
publishes it anywhere yet.

- **Automated PyPI publishing on release** (`In progress`) — a GitHub Actions
  workflow triggered by publishing a GitHub Release, using PyPI's trusted
  publishing (OIDC) so no long-lived API token has to live in repository
  secrets. Requires the maintainer to link the trusted publisher on pypi.org
  before it can actually run; see the workflow file for the exact steps.
- **TestPyPI dry run** (`Proposed`) — publish to TestPyPI on every push to
  `main` so a metadata or packaging regression is visible before it ever
  reaches a real release.

## Phase 2 — Model flexibility

The trained model is a class-balanced logistic regression by design — an
honest, interpretable baseline. That should stay the default, but power users
training on their own data may want more expressive options.

- **Opt-in alternative estimators** (`Done`) — `--estimator random_forest`
  alongside the default logistic regression, sharing the same leakage-safe
  split, calibration, and threshold-selection pipeline. `explain` reports
  feature importances instead of standardized coefficients for it. Further
  estimators (for example, gradient-boosted trees) can follow the same
  pattern established here.
- **Model comparison** (`Done`) — `compare` trains every requested estimator
  against the same split and reports validation and test metrics side by
  side, so a choice between estimators is evidence-based rather than a single
  trained artifact taken on faith. Comparing hyperparameters within one
  estimator (not just across estimators) remains open if it turns out to be
  needed.

## Phase 3 — Explainability and observability depth

`explain` already reports global, standardized feature effects. Two natural
extensions:

- **Per-transaction local explanation** (`Proposed`) — extend `predict` (CLI
  and API) with an optional per-transaction contribution breakdown, so a
  flagged transaction's score is traceable to specific feature values, not
  just the global ranking.
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

- **Optional API key middleware** (`Proposed`) — a reference implementation,
  disabled unless configured, documented as defense in depth rather than a
  substitute for a real authentication layer.
- **Optional rate-limiting middleware** (`Proposed`) — same framing as above,
  for request-rate limiting ahead of the existing body-size limit.

## Contributing to the roadmap

Open an issue or a pull request that references the relevant phase item.
Proposing a new item is welcome; keep it scoped to this project's stated
mission rather than general production-readiness concerns already assigned
to the deploying operator in SECURITY.md.
