# Contributing

Thank you for improving the project. Contributions should keep the training and
inference paths reproducible, reviewable, and safe for imbalanced financial data.

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

## Run the quality gates

```bash
python -m ruff format src tests
python -m ruff check src tests
python -m mypy src
python -m pip_audit --skip-editable
python -m pytest --cov=fraud_detection --cov-report=term-missing
python -m build
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

## Security

Report suspected vulnerabilities privately as described in
[SECURITY.md](SECURITY.md). Never include credentials, personal data, or real
cardholder information in issues, tests, or pull requests.
