from __future__ import annotations

import json
import zipfile
from io import BytesIO

import pytest

from volforge.benchmark import BenchmarkSettings, run_chronological_benchmark
from volforge.data import SyntheticReplayConfig, generate_synthetic_replay
from volforge.research import (
    build_benchmark_research_bundle,
    generate_signing_key_pair,
    sign_research_bundle,
    verify_bundle_attestation,
    verify_research_bundle,
)


@pytest.fixture(scope="module")
def benchmark_result():  # type: ignore[no-untyped-def]
    return run_chronological_benchmark(
        generate_synthetic_replay(SyntheticReplayConfig()),
        settings=BenchmarkSettings(
            training_window=2,
            heston_max_function_evaluations=60,
            cos_terms=64,
        ),
    )


def test_research_bundle_is_deterministic_and_fully_verified(benchmark_result) -> None:  # type: ignore[no-untyped-def]
    first = build_benchmark_research_bundle(benchmark_result)
    second = build_benchmark_research_bundle(benchmark_result)
    assert first == second
    verification = verify_research_bundle(first)
    assert verification.valid
    assert verification.artifact_count == 8
    assert verification.bundle_root_sha256 is not None
    with zipfile.ZipFile(BytesIO(first)) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["benchmark_final_audit_hash"] == benchmark_result.final_audit_hash
        assert manifest["bundle_root_sha256"] == verification.bundle_root_sha256
        comparison = json.loads(bundle.read("inference/model-comparison.json"))
        assert comparison["dm_method"].startswith("hln-corrected")
        assert comparison["sign_test_method"].startswith("two-sided-exact")


def test_research_bundle_detects_artifact_tampering(benchmark_result) -> None:  # type: ignore[no-untyped-def]
    original = build_benchmark_research_bundle(benchmark_result)
    output = BytesIO()
    with zipfile.ZipFile(BytesIO(original)) as source, zipfile.ZipFile(output, "w") as target:
        for name in source.namelist():
            payload = source.read(name)
            if name == "data/summary.csv":
                payload += b"tampered\n"
            target.writestr(name, payload)
    verification = verify_research_bundle(output.getvalue())
    assert not verification.valid
    assert any("artifact hash mismatch" in error for error in verification.errors)


def test_ed25519_attestation_separates_signature_from_trust(benchmark_result) -> None:  # type: ignore[no-untyped-def]
    bundle = build_benchmark_research_bundle(benchmark_result)
    key_pair = generate_signing_key_pair(b"correct horse battery staple")
    attestation = sign_research_bundle(
        bundle,
        key_pair.private_key_pem,
        issued_at="2026-10-16T15:00:00-04:00",
        signer="VolForge release test",
        password=b"correct horse battery staple",
    )
    assert attestation == sign_research_bundle(
        bundle,
        key_pair.private_key_pem,
        issued_at="2026-10-16T19:00:00Z",
        signer="VolForge release test",
        password=b"correct horse battery staple",
    )
    self_asserted = verify_bundle_attestation(bundle, attestation)
    assert self_asserted.bundle_valid
    assert self_asserted.signature_valid
    assert self_asserted.bundle_binding_valid
    assert not self_asserted.signer_trusted
    assert not self_asserted.accepted
    assert self_asserted.trust_mode == "self-asserted"

    trusted = verify_bundle_attestation(
        bundle,
        attestation,
        trusted_public_key_pem=key_pair.public_key_pem,
    )
    assert trusted.accepted
    assert trusted.signer_trusted
    assert trusted.issued_at == "2026-10-16T19:00:00Z"
    assert trusted.public_key_sha256 == key_pair.public_key_sha256


def test_attestation_rejects_wrong_trust_key_and_statement_tampering(benchmark_result) -> None:  # type: ignore[no-untyped-def]
    bundle = build_benchmark_research_bundle(benchmark_result)
    signing_key = generate_signing_key_pair()
    wrong_key = generate_signing_key_pair()
    attestation = sign_research_bundle(
        bundle,
        signing_key.private_key_pem,
        issued_at="2026-10-16T19:00:00Z",
    )
    wrong_trust = verify_bundle_attestation(
        bundle,
        attestation,
        trusted_public_key_pem=wrong_key.public_key_pem,
    )
    assert wrong_trust.signature_valid
    assert wrong_trust.bundle_binding_valid
    assert not wrong_trust.signer_trusted
    assert not wrong_trust.accepted

    changed = json.loads(attestation)
    changed["statement"]["signer"] = "forged label"
    tampered = (json.dumps(changed, sort_keys=True) + "\n").encode()
    tampered_result = verify_bundle_attestation(
        bundle,
        tampered,
        trusted_public_key_pem=signing_key.public_key_pem,
    )
    assert not tampered_result.signature_valid
    assert tampered_result.signer_trusted
    assert not tampered_result.accepted


def test_duplicate_json_fields_are_rejected(benchmark_result) -> None:  # type: ignore[no-untyped-def]
    bundle = build_benchmark_research_bundle(benchmark_result)
    rewritten = BytesIO()
    with zipfile.ZipFile(BytesIO(bundle)) as source, zipfile.ZipFile(rewritten, "w") as target:
        for name in source.namelist():
            payload = source.read(name)
            if name == "manifest.json":
                payload = payload.replace(b'"format":', b'"format":"duplicate","format":', 1)
            target.writestr(name, payload)
    bundle_verification = verify_research_bundle(rewritten.getvalue())
    assert not bundle_verification.valid
    assert any("duplicate JSON field" in error for error in bundle_verification.errors)

    attestation_verification = verify_bundle_attestation(
        bundle,
        b'{"statement":{},"statement":{}}',
    )
    assert not attestation_verification.signature_valid
    assert any("duplicate JSON field" in error for error in attestation_verification.errors)


@pytest.mark.parametrize("manifest_payload", [b"[]", b"\xff"])
def test_malformed_manifest_is_reported_without_escaping(
    benchmark_result, manifest_payload: bytes
) -> None:  # type: ignore[no-untyped-def]
    bundle = build_benchmark_research_bundle(benchmark_result)
    rewritten = BytesIO()
    with zipfile.ZipFile(BytesIO(bundle)) as source, zipfile.ZipFile(rewritten, "w") as target:
        for name in source.namelist():
            target.writestr(
                name,
                manifest_payload if name == "manifest.json" else source.read(name),
            )

    verification = verify_research_bundle(rewritten.getvalue())
    assert not verification.valid
    assert verification.errors
