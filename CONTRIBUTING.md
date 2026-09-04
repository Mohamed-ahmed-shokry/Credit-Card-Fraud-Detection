# Contributing

Thank you for improving the project. Contributions should keep the training and
inference paths reproducible, reviewable, and safe for imbalanced financial data.
Participation in this project is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Set up the development environment

Python 3.12 or newer is required.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS or
`.\.venv\Scripts\Activate.ps1` in PowerShell.

## Make a change

1. Create a focused branch from the latest `main`.
2. Add or update tests alongside behavior changes.
3. Keep raw datasets, predictions, and model artifacts out of Git.
4. Document any public CLI, API, schema, or artifact-format change.
5. Use small commits that each leave the repository in a verified state.

Do not tune a threshold or choose a model using the test split. The validation
split owns model-selection decisions; the test split is for final evaluation only.
Use precision/recall-oriented metrics for imbalanced behavior rather than accuracy
alone.

Looking for something to work on? [ROADMAP.md](ROADMAP.md) lists scoped, proposed
next steps by phase.

## Run the quality gates

```bash
python -m ruff format src tests
python -m ruff check src tests
python -m mypy src
python -m pip_audit --skip-editable
python -m pytest --cov=fraud_detection --cov-report=term-missing
python -m build
python -m twine check dist/*
```

Branch coverage must remain at or above 97%. Tests should cover failure behavior,
not only the successful path. The CI matrix runs on Python 3.12 and 3.14.

## Commit and pull request guidance

Use an imperative, scoped commit subject, for example:

```text
feat: add cost-sensitive threshold selection
fix: reject infinite prediction features
docs: explain artifact trust boundaries
```

Pull requests should state the problem, approach, verification commands, operational
or model-risk impact, and any migration needed for existing artifacts.

`main` requires both CI jobs (the Python quality matrix and the container smoke
test) to pass before a pull request can merge.

## Release checklist

The package version has a single source: `fraud_detection.__version__` in
`src/fraud_detection/__init__.py`. Publishing uses trusted publishing (OIDC),
so no API token is stored in repository secrets.

1. Update `__version__`, `CHANGELOG.md` (move `Unreleased` entries under the
   new version with today's date), and `ROADMAP.md` statuses.
2. Run the full quality gates above; all must pass, including the 97%
   branch-coverage floor and `twine check`.
3. Confirm the TestPyPI dry-run workflow is green on `main`.
4. Tag the release (`git tag vX.Y.Z && git push origin vX.Y.Z`) and publish a
   GitHub Release; the `publish.yml` workflow builds and uploads to PyPI.
5. Verify the release on PyPI and confirm the trusted-publisher link on
   pypi.org is still configured for this repository and workflow.

## Security

Report suspected vulnerabilities privately as described in
[SECURITY.md](SECURITY.md). Never include credentials, personal data, or real
cardholder information in issues, tests, or pull requests.
