"""Validated Ed25519 trust bundles and rotation operations."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from nacl.signing import VerifyKey

from fraud_detection.signing import (
    SIGNATURE_ALGORITHM,
    SigningError,
    load_public_key,
    public_key_fingerprint,
    verify_attestation_signature,
)

TRUST_BUNDLE_SCHEMA = "1.0"
TRUSTED_KEY_STATUSES = frozenset({"active", "revoked"})


class TrustBundleError(ValueError):
    """Raised when a trust bundle is malformed or cannot be rotated safely."""


@dataclass(frozen=True)
class TrustedKey:
    """One public key and its lifecycle status in a trust bundle."""

    key_id: str
    public_key: str
    status: str
    added_at: str
    revoked_at: str | None = None

    @classmethod
    def from_verify_key(
        cls,
        public_key: VerifyKey,
        *,
        status: str = "active",
        added_at: str | None = None,
        revoked_at: str | None = None,
    ) -> TrustedKey:
        raw_key = public_key.encode()
        return cls(
            key_id=public_key_fingerprint(public_key),
            public_key=base64.b64encode(raw_key).decode("ascii"),
            status=status,
            added_at=added_at or _now(),
            revoked_at=revoked_at,
        )

    @classmethod
    def from_dict(cls, payload: object) -> TrustedKey:
        if not isinstance(payload, dict):
            raise TrustBundleError("Each trust-bundle key must be a JSON object.")
        expected_fields = {"key_id", "public_key", "status", "added_at", "revoked_at"}
        if set(payload) != expected_fields:
            raise TrustBundleError("Trust-bundle key fields are invalid.")
        key_id = payload["key_id"]
        encoded_key = payload["public_key"]
        status = payload["status"]
        added_at = payload["added_at"]
        revoked_at = payload["revoked_at"]
        if not isinstance(key_id, str) or len(key_id) != 16 or not _is_lower_hex(key_id):
            raise TrustBundleError("Trust-bundle key_id must be 16 lowercase hex characters.")
        if not isinstance(encoded_key, str):
            raise TrustBundleError("Trust-bundle public_key must be a base64 string.")
        if status not in TRUSTED_KEY_STATUSES:
            raise TrustBundleError(f"Unsupported trust-bundle key status: {status!r}.")
        if not isinstance(added_at, str) or not added_at:
            raise TrustBundleError("Trust-bundle added_at must be a non-empty string.")
        if revoked_at is not None and (not isinstance(revoked_at, str) or not revoked_at):
            raise TrustBundleError("Trust-bundle revoked_at must be null or a non-empty string.")
        try:
            raw_key = base64.b64decode(encoded_key, validate=True)
            verify_key = VerifyKey(raw_key)
        except (ValueError, TypeError) as exc:
            raise TrustBundleError(f"Invalid trust-bundle public key: {exc}") from exc
        if public_key_fingerprint(verify_key) != key_id:
            raise TrustBundleError("Trust-bundle key_id does not match public_key.")
        if status == "revoked" and revoked_at is None:
            raise TrustBundleError("Revoked trust-bundle keys require revoked_at.")
        if status == "active" and revoked_at is not None:
            raise TrustBundleError("Active trust-bundle keys must not have revoked_at.")
        return cls(key_id, encoded_key, status, added_at, revoked_at)

    def to_dict(self) -> dict[str, str | None]:
        return {
            "key_id": self.key_id,
            "public_key": self.public_key,
            "status": self.status,
            "added_at": self.added_at,
            "revoked_at": self.revoked_at,
        }

    def verify_key(self) -> VerifyKey:
        try:
            return VerifyKey(base64.b64decode(self.public_key, validate=True))
        except (ValueError, TypeError) as exc:
            raise TrustBundleError(f"Invalid trust-bundle public key {self.key_id}: {exc}") from exc


@dataclass(frozen=True)
class TrustBundle:
    """A versioned set of active and revoked verification keys."""

    keys: tuple[TrustedKey, ...]
    created_at: str
    updated_at: str
    schema: str = TRUST_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRUST_BUNDLE_SCHEMA:
            raise TrustBundleError(f"Unsupported trust-bundle schema: {self.schema!r}.")
        if not self.keys:
            raise TrustBundleError("Trust bundle must contain at least one key.")
        key_ids = [key.key_id for key in self.keys]
        public_keys = [key.public_key for key in self.keys]
        if len(set(key_ids)) != len(key_ids):
            raise TrustBundleError("Trust bundle contains duplicate key IDs.")
        if len(set(public_keys)) != len(public_keys):
            raise TrustBundleError("Trust bundle contains duplicate public keys.")
        if not any(key.status == "active" for key in self.keys):
            raise TrustBundleError("Trust bundle must contain at least one active key.")

    @classmethod
    def from_dict(cls, payload: object) -> TrustBundle:
        if not isinstance(payload, dict):
            raise TrustBundleError("Trust-bundle root must be a JSON object.")
        expected_fields = {"schema", "created_at", "updated_at", "keys"}
        if set(payload) != expected_fields:
            raise TrustBundleError("Trust-bundle root fields are invalid.")
        schema = payload["schema"]
        created_at = payload["created_at"]
        updated_at = payload["updated_at"]
        raw_keys = payload["keys"]
        if (
            not isinstance(schema, str)
            or not isinstance(created_at, str)
            or not isinstance(updated_at, str)
        ):
            raise TrustBundleError("Trust-bundle schema and timestamps must be strings.")
        if not isinstance(raw_keys, list):
            raise TrustBundleError("Trust-bundle keys must be a JSON list.")
        return cls(
            keys=tuple(TrustedKey.from_dict(key) for key in raw_keys),
            created_at=created_at,
            updated_at=updated_at,
            schema=schema,
        )

    @classmethod
    def from_public_keys(cls, public_keys: tuple[VerifyKey, ...]) -> TrustBundle:
        if not public_keys:
            raise TrustBundleError("At least one public key is required.")
        timestamp = _now()
        return cls(
            keys=tuple(
                TrustedKey.from_verify_key(public_key, added_at=timestamp)
                for public_key in public_keys
            ),
            created_at=timestamp,
            updated_at=timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "keys": [key.to_dict() for key in self.keys],
        }

    def active_key_ids(self) -> tuple[str, ...]:
        return tuple(key.key_id for key in self.keys if key.status == "active")

    def key_by_id(self, key_id: str) -> TrustedKey | None:
        return next((key for key in self.keys if key.key_id == key_id), None)

    def rotate(
        self,
        additions: tuple[VerifyKey, ...] = (),
        revoked_key_ids: tuple[str, ...] = (),
    ) -> TrustBundle:
        """Return a new bundle after adding keys and revoking active keys."""
        keys = list(self.keys)
        known_ids = {key.key_id for key in keys}
        known_public = {key.public_key for key in keys}
        timestamp = _now()
        for public_key in additions:
            new_key = TrustedKey.from_verify_key(public_key, added_at=timestamp)
            if new_key.key_id in known_ids:
                raise TrustBundleError(f"Trust-bundle key already exists: {new_key.key_id}.")
            if new_key.public_key in known_public:
                raise TrustBundleError("Trust-bundle public key already exists.")
            keys.append(new_key)
            known_ids.add(new_key.key_id)
            known_public.add(new_key.public_key)

        revoke_set = set(revoked_key_ids)
        if len(revoke_set) != len(revoked_key_ids):
            raise TrustBundleError("Trust-bundle revocation list contains duplicates.")
        for index, key in enumerate(keys):
            if key.key_id not in revoke_set:
                continue
            if key.status != "active":
                raise TrustBundleError(f"Trust-bundle key is not active: {key.key_id}.")
            keys[index] = TrustedKey(
                key_id=key.key_id,
                public_key=key.public_key,
                status="revoked",
                added_at=key.added_at,
                revoked_at=timestamp,
            )
        unknown = revoke_set - {key.key_id for key in keys}
        if unknown:
            raise TrustBundleError(f"Cannot revoke unknown trust-bundle keys: {sorted(unknown)}.")
        return TrustBundle(
            keys=tuple(keys),
            created_at=self.created_at,
            updated_at=timestamp,
            schema=self.schema,
        )


def load_trust_bundle(path: Path | str) -> TrustBundle:
    """Load and validate a trust bundle from JSON."""
    bundle_path = Path(path)
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrustBundleError(f"Could not read trust bundle {bundle_path}: {exc}") from exc
    return TrustBundle.from_dict(payload)


def write_trust_bundle(
    path: Path | str,
    bundle: TrustBundle,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write a validated trust bundle."""
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise TrustBundleError(
            f"Trust-bundle output already exists: {destination}. Pass --overwrite to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(bundle.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
    except OSError as exc:
        raise TrustBundleError(f"Failed to write trust bundle: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def verify_attestation_with_bundle(
    attestation: object,
    bundle: TrustBundle,
) -> tuple[bool, str, TrustedKey | None]:
    """Verify an attestation signature against an active bundle key."""
    if not isinstance(attestation, dict):
        return False, "Attestation root must be a JSON object.", None
    envelope = attestation.get("signature")
    if not isinstance(envelope, dict):
        return False, "Attestation has no signature envelope.", None
    if envelope.get("algorithm") != SIGNATURE_ALGORITHM:
        return False, "Attestation signature algorithm is unsupported.", None
    key_id = envelope.get("key_id")
    if not isinstance(key_id, str):
        return False, "Trust-bundle verification requires a signature key_id.", None
    trusted_key = bundle.key_by_id(key_id)
    if trusted_key is None:
        return False, f"Signing key is not trusted by the bundle: {key_id}.", None
    if trusted_key.status != "active":
        return False, f"Signing key is {trusted_key.status}: {key_id}.", trusted_key
    valid, message = verify_attestation_signature(attestation, trusted_key.verify_key())
    return valid, message, trusted_key


def verify_admission_files(
    attestation_path: Path | str,
    bundle_path: Path | str,
) -> tuple[bool, str]:
    """Verify a passed signed attestation before a deployment loads its model."""
    try:
        payload = json.loads(Path(attestation_path).read_text(encoding="utf-8"))
        bundle = load_trust_bundle(bundle_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TrustBundleError) as exc:
        return False, str(exc)
    if not isinstance(payload, dict):
        return False, "Attestation root must be a JSON object."

    from fraud_detection.model import verify_attestation

    digest_valid, digest_message = verify_attestation(payload)
    if not digest_valid:
        return False, digest_message
    if payload.get("status") != "PASSED":
        return False, f"Attestation status is not PASSED: {payload.get('status')!r}."
    signature_valid, signature_message, _selected_key = verify_attestation_with_bundle(
        payload, bundle
    )
    if not signature_valid:
        return False, signature_message
    return True, "Deployment attestation admission verified successfully."


def load_public_keys(paths: tuple[Path, ...]) -> tuple[VerifyKey, ...]:
    """Load public keys for trust-bundle construction."""
    try:
        return tuple(load_public_key(path) for path in paths)
    except SigningError as exc:
        raise TrustBundleError(str(exc)) from exc


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _is_lower_hex(value: str) -> bool:
    return all(character in "0123456789abcdef" for character in value)
