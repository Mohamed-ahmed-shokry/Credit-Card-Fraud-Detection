# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Validation-only cost-sensitive threshold selection with configurable false-positive
  and false-negative weights plus holdout expected-cost reporting.
- Cross-validated sigmoid or isotonic probability calibration, optional raw scores,
  and holdout Brier-score reporting.
- Persisted, ranked standardized feature effects and an `explain` CLI for global
  model interpretation.
- Optional chronological train/validation/test evaluation with persisted time ranges
  and safeguards against single-class windows.

### Changed

- Model artifacts now use format version 2 for the calibrated estimator contract.

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
