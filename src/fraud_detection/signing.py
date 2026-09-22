"""Ed25519 signing primitives for deployment attestations."""

from __future__ import annotations

import base64
import json
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

SIGNATURE_SCHEMA = "1.0"
SIGNATURE_ALGORITHM = "Ed25519"


class SigningError(ValueError):
    """Raised for invalid signing input or an unverifiable signature."""


def generate_keypair_bytes() -> tuple[bytes, bytes]:
    """Return a new Ed25519 private PEM and public PEM keypair."""
    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def write_keypair(
    private_path: Path | str,
    public_path: Path | str,
    *,
    overwrite: bool = False,
) -> str:
    """Generate and atomically write a keypair, returning its public key fingerprint."""
    private_destination = Path(private_path)
    public_destination = Path(public_path)
    if private_destination.resolve() == public_destination.resolve():
        raise SigningError("Private and public key paths must be different.")
    if not overwrite and (private_destination.exists() or public_destination.exists()):
        existing = private_destination if private_destination.exists() else public_destination
        raise SigningError(
            f"Key output already exists: {existing}. Pass --overwrite to replace it."
        )

    private_pem, public_pem = generate_keypair_bytes()
    try:
        _atomic_write(private_destination, private_pem, mode=0o600)
        _atomic_write(public_destination, public_pem, mode=0o644)
    except OSError as exc:
        raise SigningError(f"Failed to write signing keys: {exc}") from exc
    return public_key_fingerprint(load_public_key(public_destination))


def load_private_key(path: Path | str) -> Ed25519PrivateKey:
    """Load an Ed25519 private key from PEM without exposing its contents."""
    key_path = Path(path)
    try:
        loaded = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    except (OSError, ValueError, TypeError) as exc:
        raise SigningError(f"Could not load Ed25519 private key {key_path}: {exc}") from exc
    if not isinstance(loaded, Ed25519PrivateKey):
        raise SigningError(f"Signing key {key_path} is not an Ed25519 private key.")
    return loaded


def load_public_key(path: Path | str) -> Ed25519PublicKey:
    """Load an Ed25519 public key from PEM without trusting attestation metadata."""
    key_path = Path(path)
    try:
        loaded = serialization.load_pem_public_key(key_path.read_bytes())
    except (OSError, ValueError, TypeError) as exc:
        raise SigningError(f"Could not load Ed25519 public key {key_path}: {exc}") from exc
    if not isinstance(loaded, Ed25519PublicKey):
        raise SigningError(f"Verification key {key_path} is not an Ed25519 public key.")
    return loaded


def sign_payload(payload: dict[str, Any], private_key: Ed25519PrivateKey) -> dict[str, str]:
    """Sign canonical JSON and return a self-describing signature envelope."""
    signature = private_key.sign(canonical_json(payload))
    public_key = private_key.public_key()
    return {
        "schema": SIGNATURE_SCHEMA,
        "algorithm": SIGNATURE_ALGORITHM,
        "public_key": _encode(
            public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        ),
        "signature": _encode(signature),
    }


def verify_payload_signature(
    payload: dict[str, Any],
    envelope: object,
    public_key: Ed25519PublicKey,
) -> tuple[bool, str]:
    """Verify a signature envelope against an independently trusted public key."""
    if not isinstance(envelope, dict):
        return False, "Signature envelope must be a JSON object."
    if envelope.get("schema") != SIGNATURE_SCHEMA:
        return False, f"Unsupported signature schema: {envelope.get('schema')!r}."
    if envelope.get("algorithm") != SIGNATURE_ALGORITHM:
        return False, f"Unsupported signature algorithm: {envelope.get('algorithm')!r}."
    try:
        embedded_public = _decode(envelope["public_key"])
        signature = _decode(envelope["signature"])
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"Malformed signature envelope: {exc}"

    trusted_public = public_key.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    if embedded_public != trusted_public:
        return False, "Signature public key does not match the trusted verification key."
    if len(signature) != 64:
        return False, "Ed25519 signatures must contain 64 decoded bytes."
    try:
        public_key.verify(signature, canonical_json(payload))
    except (InvalidSignature, TypeError, ValueError) as exc:
        detail = str(exc) or "invalid signature"
        return False, f"Signature verification failed: {detail}"
    return True, "Ed25519 signature verified successfully."


def verify_attestation_signature(
    attestation: object,
    public_key: Ed25519PublicKey,
) -> tuple[bool, str]:
    """Verify the signature field on an attestation without trusting its embedded key."""
    if not isinstance(attestation, dict):
        return False, "Attestation root must be a JSON object."
    envelope = attestation.get("signature")
    payload = {key: value for key, value in attestation.items() if key != "signature"}
    return verify_payload_signature(payload, envelope, public_key)


def public_key_fingerprint(public_key: Ed25519PublicKey) -> str:
    """Return a short stable identifier for a public key."""
    raw_key = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return sha256(raw_key).hexdigest()[:16]


def canonical_json(payload: dict[str, Any]) -> bytes:
    """Serialize a JSON payload deterministically for signing and verification."""
    try:
        return json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise SigningError(f"Payload cannot be canonically serialized: {exc}") from exc


def _encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode(value: object) -> bytes:
    if not isinstance(value, str):
        raise TypeError("signature fields must be base64 strings")
    return base64.b64decode(value, validate=True)


def _atomic_write(destination: Path, content: bytes, *, mode: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.chmod(mode)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
