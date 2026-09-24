# Security policy

## Supported versions

Security fixes are applied to the latest release and the `main` branch.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's
private vulnerability reporting feature on this repository and include:

- the affected version or commit;
- a minimal reproduction;
- the expected and observed impact; and
- any suggested mitigation.

Avoid including real cardholder, customer, credential, or transaction data. A
maintainer should acknowledge a complete report within seven days. Timelines for
validation, remediation, and coordinated disclosure depend on severity and scope.

## Security boundaries

This repository provides model logic and a reference API. A production operator is
responsible for authentication, authorization, TLS termination, rate limiting,
network isolation, secret management, audit retention, and regulatory controls.

### Model artifacts

`model.joblib` uses pickle-compatible serialization. Loading an untrusted artifact
can execute arbitrary code. Only load artifacts from a trusted training pipeline.

Directory-based loading checks SHA-256 hashes from `manifest.json` before unpickling.
These hashes detect accidental or unauthorized modification, but they are not a
digital signature. Store artifacts in access-controlled, immutable storage and
require a signed validation attestation when crossing a trust boundary.

Before publication and after deserialization, artifact handling also validates the
estimator interface, decision threshold, feature schema, finite standards-compliant
JSON metadata, and agreement between `metadata.json` and the model's embedded
metadata. This prevents a structurally inconsistent trusted artifact from becoming
a latent serving failure.

Directory artifacts are checked for an exact scikit-learn runtime match before
deserialization. Direct-file loading converts scikit-learn version warnings into
errors. Retrain artifacts after dependency upgrades; do not suppress compatibility
checks for production serving.

### Attestation signatures

`fraud-detect generate-signing-key` creates an Ed25519 keypair through the
audited PyNaCl backend. The private PEM
is unencrypted by design so it can be supplied by a controlled CI secret-file
mount; it must never be committed, logged, or placed in an artifact directory.
Protect it with the CI secret store or an external KMS workflow and rotate it
through an operator-controlled process. The public PEM is the deployment trust
anchor and should be distributed through access-controlled configuration or
image metadata with an explicit rotation policy.

Use strict validation to sign only a report that passed artifact checks:

```text
validate-artifact --strict --signing-key PRIVATE --attestation-output ATTESTATION
verify-attestation ATTESTATION TRUSTED_PUBLIC_KEY
```

Admission verification checks the canonical SHA-256 digest, Ed25519 signature,
trusted public-key match, and `PASSED` validation status. It fails closed for
modified payloads, wrong keys, malformed signatures, failed reports, and missing
signatures. `--allow-unsigned` exists only for legacy migration and must not be
used as the production admission policy. Signatures authenticate the validation
evidence; they do not make an untrusted `model.joblib` safe to deserialize.

### Key rotation and startup admission

Trust bundles contain the public key ID, raw public key, and lifecycle status for
each verification key. Rotate in two stages: add the replacement key while the
old key remains active, deploy the updated bundle, move signing to the replacement,
then revoke the old key with `rotate-trust-bundle`. Revocation is fail-closed and
does not rewrite existing attestations. Keep old bundles available for historical
verification, but do not use them for new deployment admission.

When `FRAUD_ATTESTATION_PATH` and `FRAUD_TRUST_BUNDLE_PATH` are both configured,
the API verifies the attestation before deserializing `model.joblib`. A failed,
missing, tampered, revoked, or unknown-key attestation prevents startup. The two
variables are intentionally all-or-nothing; leaving both unset is the explicit
local-development mode. Mount both files read-only and distribute the bundle
through an access-controlled deployment channel.

### Transaction data

Raw datasets, generated predictions, and trained artifacts are excluded by
`.gitignore`. Do not commit payment data or personally identifiable information.
Use encrypted storage and transport, minimize retention, and follow the rules that
apply in the deployment jurisdiction.

API validation responses report the failing field location and rule without echoing
the rejected transaction value.

### Repository safeguards

GitHub secret scanning and push protection are enabled on this repository and block
pushes that contain a recognizable credential or API key. `pip-audit` runs in CI
against every change, and Dependabot opens pull requests both for routine dependency
updates and for advisories against dependencies already in use.

Release workflows build the distributions before publishing, install the exact
wheel for a CycloneDX SBOM, and retain the non-empty `sbom/sbom.cdx.json` output
as a release artifact. TestPyPI reruns may skip an already uploaded immutable
development version, but they do not skip build, metadata, or SBOM failures.
Publishing uses OIDC trusted publishers, so PyPI/TestPyPI account and environment
configuration must be reviewed and controlled outside the repository.

### API deployment

The supplied container runs without root privileges or Linux capabilities and uses
a read-only filesystem. Those controls are defense in depth, not a substitute for
an authenticated gateway and a private service network.

The application rejects declared or streamed request bodies larger than 2 MiB.
Production gateways should still enforce their own request-size, rate, and
concurrency limits before traffic reaches the service.

The API ships optional, off-by-default API-key and rate-limiting middleware as
a documented reference starting point (see the README). They are defense in
depth, not a replacement for gateway authentication: responses carry
`Retry-After` and `X-RateLimit-*` headers, but key storage, rotation, and
perimeter enforcement remain the operator's job. Provided keys are checked
with `secrets.compare_digest` against every configured value so verification
time does not depend on which key matched.

An optional scoring concurrency cap (`max_concurrent_scoring` /
`FRAUD_MAX_CONCURRENT_SCORING`) sheds excess scoring load with `503` and
`Retry-After` before work is queued unboundedly. It is likewise defense in
depth; production gateways remain responsible for connection and concurrency
limits as stated above.

`GET /metrics`, `GET /health`, `GET /live`, and `GET /ready` are
unauthenticated even when the optional API-key middleware is enabled, and they
are never rate limited or counted against the scoring concurrency cap, like
most Prometheus and orchestrator probe endpoints. They report counts, labels,
timing, and readiness only, never transaction values, but should still be
reachable only from a trusted scrape or probe network rather than exposed
publicly, the same as any other operational endpoint.
