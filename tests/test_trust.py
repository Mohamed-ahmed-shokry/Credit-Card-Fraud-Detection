from __future__ import annotations

from pathlib import Path

import pytest

from fraud_detection.signing import (
    generate_keypair_bytes,
    load_private_key,
    load_public_key,
    sign_payload,
)
from fraud_detection.trust import (
    TrustBundle,
    TrustBundleError,
    TrustedKey,
    load_trust_bundle,
    verify_attestation_with_bundle,
    write_trust_bundle,
)


def _key_files(tmp_path: Path, name: str) -> tuple[Path, Path]:
    private_pem, public_pem = generate_keypair_bytes()
    private_path = tmp_path / f"{name}-private.pem"
    public_path = tmp_path / f"{name}-public.pem"
    private_path.write_bytes(private_pem)
    public_path.write_bytes(public_pem)
    return private_path, public_path


def test_trust_bundle_round_trips_and_rotates_keys(tmp_path: Path) -> None:
    _old_private_path, old_public_path = _key_files(tmp_path, "old")
    _new_private_path, new_public_path = _key_files(tmp_path, "new")
    old_public = load_public_key(old_public_path)
    new_public = load_public_key(new_public_path)
    bundle = TrustBundle.from_public_keys((old_public,))
    bundle_path = tmp_path / "trust.json"
    write_trust_bundle(bundle_path, bundle)

    restored = load_trust_bundle(bundle_path)
    old_id = restored.active_key_ids()[0]
    rotated = restored.rotate((new_public,), (old_id,))
    write_trust_bundle(bundle_path, rotated, overwrite=True)
    loaded = load_trust_bundle(bundle_path)

    assert loaded.key_by_id(old_id) is not None
    assert loaded.key_by_id(old_id).status == "revoked"  # type: ignore[union-attr]
    assert len(loaded.active_key_ids()) == 1
    assert loaded.active_key_ids()[0] != old_id

    assert loaded.key_by_id(old_id) is not None


def test_trust_bundle_verifies_active_and_rejects_revoked_key(tmp_path: Path) -> None:
    old_private_path, old_public_path = _key_files(tmp_path, "old")
    new_private_path, new_public_path = _key_files(tmp_path, "new")
    old_private = load_private_key(old_private_path)
    new_private = load_private_key(new_private_path)
    old_public = load_public_key(old_public_path)
    new_public = load_public_key(new_public_path)
    bundle = TrustBundle.from_public_keys((old_public, new_public))
    old_id = bundle.keys[0].key_id
    revoked = bundle.rotate((), (old_id,))
    payload = {"status": "PASSED", "attestation_digest": "digest"}

    valid_attestation = {**payload, "signature": sign_payload(payload, new_private)}
    valid, _message, selected = verify_attestation_with_bundle(valid_attestation, bundle)
    assert valid is True
    assert selected is not None and selected.status == "active"

    revoked_attestation = {**payload, "signature": sign_payload(payload, old_private)}
    valid_revoked, message, selected_revoked = verify_attestation_with_bundle(
        revoked_attestation, revoked
    )
    assert valid_revoked is False
    assert "revoked" in message
    assert selected_revoked is not None


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"schema": "2.0", "created_at": "now", "updated_at": "now", "keys": []},
        {"schema": "1.0", "created_at": "now", "updated_at": "now", "keys": "bad"},
        {"schema": "1.0", "created_at": "now", "updated_at": "now", "keys": []},
    ],
)
def test_trust_bundle_rejects_malformed_roots(payload: object) -> None:
    with pytest.raises(TrustBundleError):
        TrustBundle.from_dict(payload)


def test_trust_bundle_rejects_bad_key_lifecycle_and_rotation(tmp_path: Path) -> None:
    _private_path, public_path = _key_files(tmp_path, "key")
    public = load_public_key(public_path)
    key = TrustedKey.from_verify_key(public)
    raw = key.to_dict()

    bad_status = {**raw, "status": "unknown"}
    with pytest.raises(TrustBundleError, match="status"):
        TrustedKey.from_dict(bad_status)
    bad_id = {**raw, "key_id": "0" * 16}
    with pytest.raises(TrustBundleError, match="key_id"):
        TrustedKey.from_dict(bad_id)
    revoked_without_time = {**raw, "status": "revoked"}
    with pytest.raises(TrustBundleError, match="revoked_at"):
        TrustedKey.from_dict(revoked_without_time)

    bundle = TrustBundle.from_public_keys((public,))
    with pytest.raises(TrustBundleError, match="unknown"):
        bundle.rotate(revoked_key_ids=("0" * 16,))
    with pytest.raises(TrustBundleError, match="at least one active"):
        bundle.rotate(revoked_key_ids=(bundle.keys[0].key_id,))


def test_trust_bundle_output_protection_and_signature_legacy_failure(tmp_path: Path) -> None:
    _private_path, public_path = _key_files(tmp_path, "key")
    bundle = TrustBundle.from_public_keys((load_public_key(public_path),))
    path = tmp_path / "trust.json"
    write_trust_bundle(path, bundle)
    with pytest.raises(TrustBundleError, match="already exists"):
        write_trust_bundle(path, bundle)

    valid, message, _selected = verify_attestation_with_bundle({"status": "PASSED"}, bundle)
    assert valid is False
    assert "signature" in message
