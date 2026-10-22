"""Detached Ed25519 attestations for content-addressed research bundles."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from volforge.research.report import verify_research_bundle

_ATTESTATION_SCHEMA_VERSION = 1
_MAX_ATTESTATION_BYTES = 1024 * 1024
_MAX_KEY_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class SigningKeyPair:
    """PEM-encoded Ed25519 key material and its stable public fingerprint."""

    private_key_pem: bytes
    public_key_pem: bytes
    public_key_sha256: str
    private_key_encrypted: bool


@dataclass(frozen=True, slots=True)
class AttestationVerification:
    """Independent bundle, signature, binding, and signer-trust decisions."""

    bundle_valid: bool
    signature_valid: bool
    bundle_binding_valid: bool
    signer_trusted: bool
    accepted: bool
    trust_mode: str
    public_key_sha256: str | None
    signer: str | None
    issued_at: str | None
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible verification record."""
        return asdict(self)


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _load_json(payload: bytes) -> Any:
    return json.loads(payload, object_pairs_hook=_unique_object)


def _public_raw(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _public_fingerprint(public_key: Ed25519PublicKey) -> str:
    return hashlib.sha256(_public_raw(public_key)).hexdigest()


def _load_private_key(payload: bytes, password: bytes | None) -> Ed25519PrivateKey:
    if len(payload) > _MAX_KEY_BYTES:
        raise ValueError("private key exceeds 64 KiB")
    try:
        key = serialization.load_pem_private_key(payload, password=password)
    except (TypeError, ValueError, UnsupportedAlgorithm) as exc:
        raise ValueError(f"could not load Ed25519 private key: {exc}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("private key must use Ed25519")
    return key


def _load_public_key(payload: bytes) -> Ed25519PublicKey:
    if len(payload) > _MAX_KEY_BYTES:
        raise ValueError("public key exceeds 64 KiB")
    try:
        key = serialization.load_pem_public_key(payload)
    except (TypeError, ValueError, UnsupportedAlgorithm) as exc:
        raise ValueError(f"could not load Ed25519 public key: {exc}") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("public key must use Ed25519")
    return key


def _normalize_issued_at(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("issued_at must be non-empty")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("issued_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("issued_at must include a UTC offset")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _bundle_manifest_hash(bundle_payload: bytes) -> str:
    with zipfile.ZipFile(BytesIO(bundle_payload)) as bundle:
        manifest = _load_json(bundle.read("manifest.json"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("manifest_sha256"), str):
        raise ValueError("bundle manifest_sha256 is missing")
    return str(manifest["manifest_sha256"])


def generate_signing_key_pair(password: bytes | None = None) -> SigningKeyPair:
    """Generate an Ed25519 key pair using encrypted PKCS8 when a password is supplied."""
    if password == b"":
        raise ValueError("private-key password must be non-empty")
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    encryption: serialization.KeySerializationEncryption = (
        serialization.BestAvailableEncryption(password)
        if password is not None
        else serialization.NoEncryption()
    )
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=encryption,
    )
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return SigningKeyPair(
        private_key_pem=private_pem,
        public_key_pem=public_pem,
        public_key_sha256=_public_fingerprint(public_key),
        private_key_encrypted=password is not None,
    )


def sign_research_bundle(
    bundle_payload: bytes,
    private_key_pem: bytes,
    *,
    issued_at: str,
    signer: str | None = None,
    password: bytes | None = None,
) -> bytes:
    """Create a deterministic detached Ed25519 attestation for an intact bundle."""
    bundle_verification = verify_research_bundle(bundle_payload)
    if not bundle_verification.valid or bundle_verification.bundle_root_sha256 is None:
        detail = "; ".join(bundle_verification.errors) or "bundle integrity check failed"
        raise ValueError(f"cannot sign invalid research bundle: {detail}")
    normalized_signer = str(signer).strip() if signer is not None else None
    if normalized_signer == "":
        raise ValueError("signer must be non-empty when supplied")
    if normalized_signer is not None and len(normalized_signer) > 256:
        raise ValueError("signer must not exceed 256 characters")
    private_key = _load_private_key(private_key_pem, password)
    public_key = private_key.public_key()
    fingerprint = _public_fingerprint(public_key)
    statement = {
        "algorithm": "ed25519",
        "bundle_root_sha256": bundle_verification.bundle_root_sha256,
        "bundle_sha256": hashlib.sha256(bundle_payload).hexdigest(),
        "format": "volforge-research-bundle-attestation",
        "issued_at": _normalize_issued_at(issued_at),
        "manifest_sha256": _bundle_manifest_hash(bundle_payload),
        "public_key_sha256": fingerprint,
        "schema_version": _ATTESTATION_SCHEMA_VERSION,
        "signer": normalized_signer,
    }
    signature = private_key.sign(_canonical_json(statement))
    attestation = {
        "public_key_raw_base64": base64.b64encode(_public_raw(public_key)).decode("ascii"),
        "signature_base64": base64.b64encode(signature).decode("ascii"),
        "statement": statement,
    }
    return _canonical_json(attestation)


def verify_bundle_attestation(
    bundle_payload: bytes,
    attestation_payload: bytes,
    *,
    trusted_public_key_pem: bytes | None = None,
) -> AttestationVerification:
    """Verify bundle integrity, detached signature, exact binding, and optional signer trust."""
    errors: list[str] = []
    bundle_verification = verify_research_bundle(bundle_payload)
    bundle_valid = bundle_verification.valid
    if not bundle_valid:
        errors.extend(f"bundle: {error}" for error in bundle_verification.errors)
    if len(attestation_payload) > _MAX_ATTESTATION_BYTES:
        return AttestationVerification(
            bundle_valid,
            False,
            False,
            False,
            False,
            "trusted-key" if trusted_public_key_pem is not None else "self-asserted",
            None,
            None,
            None,
            (*errors, "attestation exceeds 1 MiB"),
        )
    try:
        attestation = _load_json(attestation_payload)
        if not isinstance(attestation, dict):
            raise ValueError("attestation root must be an object")
        statement = attestation.get("statement")
        if not isinstance(statement, dict):
            raise ValueError("attestation statement must be an object")
        public_raw = base64.b64decode(
            str(attestation.get("public_key_raw_base64", "")), validate=True
        )
        signature = base64.b64decode(str(attestation.get("signature_base64", "")), validate=True)
        public_key = Ed25519PublicKey.from_public_bytes(public_raw)
        fingerprint = _public_fingerprint(public_key)
    except (
        binascii.Error,
        json.JSONDecodeError,
        TypeError,
        UnsupportedAlgorithm,
        ValueError,
    ) as exc:
        return AttestationVerification(
            bundle_valid,
            False,
            False,
            False,
            False,
            "trusted-key" if trusted_public_key_pem is not None else "self-asserted",
            None,
            None,
            None,
            (*errors, f"invalid attestation: {exc}"),
        )

    signature_valid = True
    try:
        public_key.verify(signature, _canonical_json(statement))
    except InvalidSignature:
        signature_valid = False
        errors.append("attestation signature is invalid")

    expected_binding = {
        "algorithm": "ed25519",
        "bundle_root_sha256": bundle_verification.bundle_root_sha256,
        "bundle_sha256": hashlib.sha256(bundle_payload).hexdigest(),
        "format": "volforge-research-bundle-attestation",
        "manifest_sha256": _bundle_manifest_hash(bundle_payload) if bundle_valid else None,
        "public_key_sha256": fingerprint,
        "schema_version": _ATTESTATION_SCHEMA_VERSION,
    }
    binding_valid = bundle_valid and all(
        statement.get(key) == value for key, value in expected_binding.items()
    )
    try:
        normalized_issued_at = _normalize_issued_at(str(statement.get("issued_at", "")))
        if statement.get("issued_at") != normalized_issued_at:
            binding_valid = False
    except ValueError:
        binding_valid = False
    signer_value = statement.get("signer")
    if signer_value is not None and (not isinstance(signer_value, str) or not signer_value.strip()):
        binding_valid = False
    if not binding_valid:
        errors.append("attestation statement does not bind the supplied bundle and key")

    signer_trusted = False
    trust_mode = "self-asserted"
    if trusted_public_key_pem is not None:
        trust_mode = "trusted-key"
        try:
            trusted_key = _load_public_key(trusted_public_key_pem)
            signer_trusted = _public_raw(trusted_key) == public_raw
            if not signer_trusted:
                errors.append("attestation key does not match the trusted public key")
        except ValueError as exc:
            errors.append(str(exc))
    accepted = bundle_valid and signature_valid and binding_valid and signer_trusted
    return AttestationVerification(
        bundle_valid=bundle_valid,
        signature_valid=signature_valid,
        bundle_binding_valid=binding_valid,
        signer_trusted=signer_trusted,
        accepted=accepted,
        trust_mode=trust_mode,
        public_key_sha256=fingerprint,
        signer=str(signer_value) if signer_value is not None else None,
        issued_at=(
            str(statement.get("issued_at")) if statement.get("issued_at") is not None else None
        ),
        errors=tuple(errors),
    )
