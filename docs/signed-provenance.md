# Signed research provenance

VolForge v1.1 can attach a detached Ed25519 attestation to an intact research bundle. Ed25519 is
the EdDSA instantiation specified in
[RFC 8032](https://www.rfc-editor.org/info/rfc8032/). VolForge delegates key generation,
PKCS8/SubjectPublicKeyInfo serialization, signing, and verification to the maintained
[`cryptography` implementation](https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ed25519/)
rather than implementing curve arithmetic.

## What is signed

The canonical signed statement contains:

- the exact ZIP byte SHA-256;
- the bundle-root and manifest SHA-256 values already verified inside the ZIP;
- the raw Ed25519 public-key SHA-256 fingerprint;
- the algorithm and attestation schema identifiers;
- a caller-supplied issue timestamp normalized to UTC; and
- an optional signer label.

The attestation is detached JSON. Signing does not rewrite the deterministic research ZIP. Ed25519
signatures are deterministic, so the same key, bundle, timestamp, and signer label produce the same
attestation bytes.

## Integrity is not identity

An attestation embeds its public key so anyone can check whether its signature is internally valid.
That key is self-asserted and does not establish who controls it. Verification therefore reports
five separate decisions:

1. `bundle_valid`: the ZIP manifest and every artifact still verify;
2. `signature_valid`: the embedded key verifies the signed statement;
3. `bundle_binding_valid`: the statement names this exact ZIP and its lineage hashes;
4. `signer_trusted`: the embedded key exactly matches a separately supplied trusted public key;
5. `accepted`: all four preceding checks pass.

The package API can inspect a self-asserted signature, but `accepted` remains false without a
trusted key. The `research-verify` CLI requires `--trusted-public-key` and exits successfully only
when `accepted` is true. Distribute the public key or its SHA-256 fingerprint through an independent
authenticated channel.

## Key workflow

Create a password file outside the repository, restrict it with operating-system access controls,
then generate an encrypted PKCS8 key:

```bash
volforge research-keygen keys/research-private.pem keys/research-public.pem \
  --password-file keys/private-key.password
```

Key generation refuses to overwrite either output. `--unencrypted` is available only as an explicit
opt-in for externally protected key stores. On POSIX, VolForge applies mode `0600` to a newly written
private key. Python cannot apply equivalent Windows ACL restrictions through `chmod`; on Windows,
the command reports `private_key_permissions_hardened=false` and the operator must configure ACLs.

Sign an already verified bundle. `--issued-at` is mandatory so wall-clock state never silently
enters a supposedly reproducible report:

```bash
volforge research-sign reports/model-benchmark.zip keys/research-private.pem \
  reports/model-benchmark.attestation.json \
  --password-file keys/private-key.password \
  --issued-at 2026-10-16T19:00:00Z --signer "VolForge research"
```

Verify against the independently obtained public key:

```bash
volforge research-verify reports/model-benchmark.zip \
  reports/model-benchmark.attestation.json \
  --trusted-public-key keys/research-public.pem
```

## Security limits

- The signer label and issue timestamp are signed assertions, not verified legal identity or a
  trusted timestamp-authority response.
- A valid signature does not establish that the underlying data, model assumptions, or conclusions
  are correct.
- VolForge does not provide a certificate authority, transparency log, hardware-security-module
  integration, remote signer, revocation list, or automatic key rotation.
- If a private key or its password is exposed, stop using it, distribute a new trusted public key,
  and communicate revocation of the old fingerprint out of band.
- The exact ZIP bytes are signed. Recompressing an otherwise equivalent archive intentionally
  invalidates the binding and requires a new attestation.

Never commit private keys or password files. Treat encrypted private keys as secrets even though
their contents are protected at rest.
