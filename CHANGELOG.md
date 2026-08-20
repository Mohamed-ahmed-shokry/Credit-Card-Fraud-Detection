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
- API request correlation, processing-time response headers, and structured
  completion/failure log fields.
- CLI overwrite protection for non-empty model artifact directories.
- CI dependency vulnerability auditing and a patched pytest 9 requirement.
- CI now trains a demo model and runs the built container image against a live
  `/health` check instead of only building it, catching startup regressions in the
  serving path.

### Changed

- Model artifacts now use format version 2 for the calibrated estimator contract.
- Model saving and loading validate the estimator, feature schema, decision threshold,
  finite JSON metadata, and agreement between the embedded and readable model cards.
- Dataset validation rejects duplicate CSV headers and feature names that become
  ambiguous when normalized to the persisted string schema.
- Inference rejects estimators with reversed class labels, malformed binary output,
  non-numeric scores, or non-finite and out-of-range probabilities.
- The API requires strict finite JSON numbers and sanitizes validation responses so
  rejected transaction values are not echoed or rendered as invalid JSON.
- Generated datasets, predictions, and drift reports now use atomic replacement;
  drift reports also require explicit `--overwrite` before replacing a file.
- CLI training reports artifact-destination filesystem failures without a traceback.
- Drift calculation validates normalized feature names, numeric profile statistics,
  bin ordering, and baseline probability distributions before calculating PSI.
- Artifact loading rejects mismatched scikit-learn runtimes before directory-based
  deserialization and converts scikit-learn inconsistency warnings into model errors.
- Calibration uses one worker by default, persists its worker policy, and supports
  explicit `--calibration-jobs -1` opt-in for all processors.
- API requests are bounded to 2 MiB for both declared and streamed bodies while
  preserving correlation and timing headers on `413` responses.
- Model cards record Python, joblib, NumPy, pandas, scikit-learn, and SciPy versions
  needed to reconstruct the training runtime.
- The API now configures root logging on startup so request correlation and timing
  logs are emitted by default under `fraud-detect serve` and the container entrypoint,
  not only when a caller separately configures logging.

### Fixed

- `load_csv`, `fraud-detect predict`, and `fraud-detect drift` now report an empty
  transactions CSV as an actionable error instead of leaking an unhandled
  `pandas.errors.EmptyDataError` traceback.
- CI now actually enforces flake8-bandit (`S`) lint rules; the rule set was never
  selected, so a pre-existing `S101` per-file ignore for tests was silently a no-op.

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
