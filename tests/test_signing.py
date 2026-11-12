from __future__ import annotations

import base64
from pathlib import Path

import pytest

from fraud_detection.signing import (
    SIGNATURE_ALGORITHM,
    SigningError,
    canonical_json,
    generate_keypair_bytes,
    load_private_key,
    load_public_key,
    public_key_fingerprint,
    sign_payload,
    verify_attestation_signature,
    verify_payload_signature,
    write_keypair,
)


def test_keypair_generation_writes_safe_outputs_and_round_trips(tmp_path: Path) -> None:
    private_path = tmp_path / "keys" / "signing.pem"
    public_path = tmp_path / "keys" / "verification.pem"

    fingerprint = write_keypair(private_path, public_path)

    assert private_path.is_file()
    assert public_path.is_file()
    assert "PRIVATE KEY" in private_path.read_text(encoding="ascii")
    assert "PUBLIC KEY" in public_path.read_text(encoding="ascii")
    assert fingerprint == public_key_fingerprint(load_public_key(public_path))
    assert len(fingerprint) == 16
    assert "PRIVATE" not in fingerprint


def test_signing_verification_rejects_tampering_and_wrong_key(tmp_path: Path) -> None:
    first_private, first_public = generate_keypair_bytes()
    second_private, second_public = generate_keypair_bytes()
    first_private_path = tmp_path / "first-private.pem"
    first_public_path = tmp_path / "first-public.pem"
    second_private_path = tmp_path / "second-private.pem"
    second_public_path = tmp_path / "second-public.pem"
    first_private_path.write_bytes(first_private)
    first_public_path.write_bytes(first_public)
    second_private_path.write_bytes(second_private)
    second_public_path.write_bytes(second_public)

    payload = {"status": "PASSED", "artifact_digest": "abc123", "rows": 3}
    envelope = sign_payload(payload, load_private_key(first_private_path))

    assert envelope["algorithm"] == SIGNATURE_ALGORITHM
    assert verify_payload_signature(payload, envelope, load_public_key(first_public_path)) == (
        True,
        "Ed25519 signature verified successfully.",
    )
    assert (
        verify_payload_signature(
            {**payload, "rows": 4}, envelope, load_public_key(first_public_path)
        )[0]
        is False
    )
    assert (
        verify_payload_signature(payload, envelope, load_public_key(second_public_path))[0] is False
    )


@pytest.mark.parametrize(
    "envelope",
    [
        None,
        {"schema": "2.0", "algorithm": SIGNATURE_ALGORITHM},
        {"schema": "1.0", "algorithm": "RSA"},
        {"schema": "1.0", "algorithm": SIGNATURE_ALGORITHM, "public_key": "!", "signature": "!"},
    ],
)
def test_signature_verification_rejects_malformed_envelopes(
    envelope: object,
    tmp_path: Path,
) -> None:
    private_pem, public_pem = generate_keypair_bytes()
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_path.write_bytes(private_pem)
    public_path.write_bytes(public_pem)

    result = verify_payload_signature({"status": "PASSED"}, envelope, load_public_key(public_path))

    assert result[0] is False


def test_key_output_protection_and_invalid_inputs(tmp_path: Path) -> None:
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    write_keypair(private_path, public_path)

    with pytest.raises(SigningError, match="already exists"):
        write_keypair(private_path, public_path)
    with pytest.raises(SigningError, match="different"):
        write_keypair(private_path, private_path, overwrite=True)
    with pytest.raises(SigningError, match="canonical"):
        canonical_json({"invalid": float("nan")})

    malformed = tmp_path / "malformed.pem"
    malformed.write_text("not a key", encoding="ascii")
    with pytest.raises(SigningError, match="private key"):
        load_private_key(malformed)
    with pytest.raises(SigningError, match="public key"):
        load_public_key(malformed)


def test_signature_envelope_rejects_invalid_base64_length(tmp_path: Path) -> None:
    private_pem, public_pem = generate_keypair_bytes()
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_path.write_bytes(private_pem)
    public_path.write_bytes(public_pem)
    envelope = sign_payload({"status": "PASSED"}, load_private_key(private_path))
    envelope["signature"] = base64.b64encode(b"short").decode("ascii")

    valid, message = verify_payload_signature(
        {"status": "PASSED"}, envelope, load_public_key(public_path)
    )

    assert valid is False
    assert "64 decoded bytes" in message


def test_attestation_signature_helper_requires_object_and_signature(tmp_path: Path) -> None:
    private_pem, public_pem = generate_keypair_bytes()
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_path.write_bytes(private_pem)
    public_path.write_bytes(public_pem)
    public_key = load_public_key(public_path)

    assert verify_attestation_signature([], public_key)[0] is False
    assert verify_attestation_signature({"status": "PASSED"}, public_key)[0] is False


def test_signing_coverage_edge_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fraud_detection.signing import _decode, _read_pem

    # 1. _decode with non-string raises TypeError
    with pytest.raises(TypeError, match="must be base64 strings"):
        _decode(12345)

    # 2. _read_pem with decoded len != 32 raises ValueError
    bad_pem = tmp_path / "bad_len.pem"
    bad_b64 = base64.b64encode(b"short_bytes").decode("ascii")
    bad_pem.write_text(
        f"-----BEGIN PUBLIC KEY-----\n{bad_b64}\n-----END PUBLIC KEY-----\n", encoding="ascii"
    )
    with pytest.raises(ValueError, match="must contain 32 decoded bytes"):
        _read_pem(bad_pem, "PUBLIC KEY")

    # 3. OSError when writing keypair raises SigningError
    def _raise_os_error(*_args: object, **_kwargs: object) -> None:
        raise OSError("write denied")

    monkeypatch.setattr("fraud_detection.signing._atomic_write", _raise_os_error)
    with pytest.raises(SigningError, match="Failed to write signing keys"):
        write_keypair(tmp_path / "k1.pem", tmp_path / "k2.pem")
