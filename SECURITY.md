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
digital signature. Store artifacts in access-controlled, immutable storage and add
signature verification when crossing a trust boundary.

Before publication and after deserialization, artifact handling also validates the
estimator interface, decision threshold, feature schema, finite standards-compliant
JSON metadata, and agreement between `metadata.json` and the model's embedded
metadata. This prevents a structurally inconsistent trusted artifact from becoming
a latent serving failure.

Directory artifacts are checked for an exact scikit-learn runtime match before
deserialization. Direct-file loading converts scikit-learn version warnings into
errors. Retrain artifacts after dependency upgrades; do not suppress compatibility
checks for production serving.

### Transaction data

Raw datasets, generated predictions, and trained artifacts are excluded by
`.gitignore`. Do not commit payment data or personally identifiable information.
Use encrypted storage and transport, minimize retention, and follow the rules that
apply in the deployment jurisdiction.

API validation responses report the failing field location and rule without echoing
the rejected transaction value.

### API deployment

The supplied container runs without root privileges or Linux capabilities and uses
a read-only filesystem. Those controls are defense in depth, not a substitute for
an authenticated gateway and a private service network.

The application rejects declared or streamed request bodies larger than 2 MiB.
Production gateways should still enforce their own request-size, rate, and
concurrency limits before traffic reaches the service.
