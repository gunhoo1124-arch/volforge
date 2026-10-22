"""Content-addressed research artifacts."""

from volforge.research.attestation import (
    AttestationVerification,
    SigningKeyPair,
    generate_signing_key_pair,
    sign_research_bundle,
    verify_bundle_attestation,
)
from volforge.research.report import (
    ResearchBundleVerification,
    build_benchmark_research_bundle,
    verify_research_bundle,
)

__all__ = [
    "AttestationVerification",
    "ResearchBundleVerification",
    "SigningKeyPair",
    "build_benchmark_research_bundle",
    "generate_signing_key_pair",
    "sign_research_bundle",
    "verify_bundle_attestation",
    "verify_research_bundle",
]
