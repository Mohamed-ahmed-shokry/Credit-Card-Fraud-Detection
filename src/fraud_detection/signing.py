"""Ed25519 signing primitives for deployment attestations."""

from __future__ import annotations

import base64
import json
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey, VerifyKey

SIGNATURE_SCHEMA = "1.0"
SIGNATURE_ALGORITHM = "Ed25519"
PRIVATE_KEY_LABEL = "ED25519 PRIVATE KEY"
PUBLIC_KEY_LABEL = "ED25519 PUBLIC KEY"


class SigningError(ValueError):
    """Raised for invalid signing input or an unverifiable signature."""


def generate_keypair_bytes() -> tuple[bytes, bytes]:
    """Return a new Ed25519 private PEM and public PEM keypair."""
    private_key = SigningKey.generate()
    return _pem(PRIVATE_KEY_LABEL, private_key.encode()), _pem(
        PUBLIC_KEY_LABEL, private_key.verify_key.encode()
    )


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


def load_private_key(path: Path | str) -> SigningKey:
    """Load an Ed25519 private key from PEM without exposing its contents."""
    key_path = Path(path)
    try:
        loaded = SigningKey(_read_pem(key_path, PRIVATE_KEY_LABEL))
    except (OSError, ValueError, TypeError) as exc:
        raise SigningError(f"Could not load Ed25519 private key {key_path}: {exc}") from exc
    return loaded


def load_public_key(path: Path | str) -> VerifyKey:
    """Load an Ed25519 public key from PEM without trusting attestation metadata."""
    key_path = Path(path)
    try:
        loaded = VerifyKey(_read_pem(key_path, PUBLIC_KEY_LABEL))
    except (OSError, ValueError, TypeError) as exc:
        raise SigningError(f"Could not load Ed25519 public key {key_path}: {exc}") from exc
    return loaded


def sign_payload(payload: dict[str, Any], private_key: SigningKey) -> dict[str, str]:
    """Sign canonical JSON and return a self-describing signature envelope."""
    signature = private_key.sign(canonical_json(payload)).signature
    return {
        "schema": SIGNATURE_SCHEMA,
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": public_key_fingerprint(private_key.verify_key),
        "public_key": _encode(private_key.verify_key.encode()),
        "signature": _encode(signature),
    }


def verify_payload_signature(
    payload: dict[str, Any],
    envelope: object,
    public_key: VerifyKey,
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

    trusted_public = public_key.encode()
    if embedded_public != trusted_public:
        return False, "Signature public key does not match the trusted verification key."
    if len(signature) != 64:
        return False, "Ed25519 signatures must contain 64 decoded bytes."
    try:
        public_key.verify(canonical_json(payload), signature)
    except (BadSignatureError, TypeError, ValueError) as exc:
        detail = str(exc) or "invalid signature"
        return False, f"Signature verification failed: {detail}"
    return True, "Ed25519 signature verified successfully."


def verify_attestation_signature(
    attestation: object,
    public_key: VerifyKey,
) -> tuple[bool, str]:
    """Verify the signature field on an attestation without trusting its embedded key."""
    if not isinstance(attestation, dict):
        return False, "Attestation root must be a JSON object."
    envelope = attestation.get("signature")
    payload = {key: value for key, value in attestation.items() if key != "signature"}
    return verify_payload_signature(payload, envelope, public_key)


def public_key_fingerprint(public_key: VerifyKey) -> str:
    """Return a short stable identifier for a public key."""
    return sha256(public_key.encode()).hexdigest()[:16]


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


def _pem(label: str, value: bytes) -> bytes:
    encoded = base64.b64encode(value).decode("ascii")
    lines = [encoded[index : index + 64] for index in range(0, len(encoded), 64)]
    body = f"-----BEGIN {label}-----\n" + "".join(f"{line}\n" for line in lines)
    return f"{body}-----END {label}-----\n".encode("ascii")


def _read_pem(path: Path, label: str) -> bytes:
    content = path.read_text(encoding="ascii").strip().splitlines()
    begin = f"-----BEGIN {label}-----"
    end = f"-----END {label}-----"
    if len(content) < 3 or content[0] != begin or content[-1] != end:
        raise ValueError(f"PEM does not contain an {label} key.")
    decoded = base64.b64decode("".join(content[1:-1]), validate=True)
    if len(decoded) != 32:
        raise ValueError(f"Ed25519 keys must contain 32 decoded bytes, got {len(decoded)}.")
    return decoded


def _atomic_write(destination: Path, content: bytes, *, mode: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.chmod(mode)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
