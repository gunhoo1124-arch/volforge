# Content-addressed research-report bundles

VolForge v1.0 turns a completed or blocked chronological model benchmark into one deterministic
ZIP. The bundle is intended to keep the exact row-level evidence, summaries, configuration, and
audit lineage together when a result is reviewed or archived.

## Contents

`build_benchmark_research_bundle` writes these deterministic artifacts:

- the full benchmark JSON audit;
- row-level BSM/Heston predictions and losses;
- chronological audit steps and daily fitted parameters;
- overall, evaluation-timeline, and maturity/moneyness loss tables; and
- the comparison-inference record, including block-bootstrap, DM/HAC, and date-sign diagnostics.

ZIP members are sorted, have a fixed metadata timestamp and permissions, and use deterministic CSV
and JSON encodings. Identical benchmark results under the same VolForge version therefore produce
identical bundle bytes in the supported environment.

## Manifest and verification

`manifest.json` records the schema and VolForge versions, exact benchmark settings, final audit
hash, and each artifact's byte count and SHA-256. `bundle_root_sha256` hashes that full lineage and
artifact map. `manifest_sha256` hashes the manifest fields that precede it.

`verify_research_bundle` rejects malformed archives, duplicate members, undeclared or missing
artifacts, byte-count or artifact-hash mismatches, changed manifest content, changed root lineage,
unsupported formats/versions, and archives whose compressed or expanded input exceeds the current
100 MiB safety limit. Verification reads members but never extracts paths to disk.

These controls make the report content-addressed, byte-reproducible, and tamper-evident. They do
not make the ZIP physically immutable, prove who created it, establish a trusted creation time, or
prove vendor provenance. Publish or sign the root hash independently when those guarantees matter.

## API and CLI

```python
from volforge.research import build_benchmark_research_bundle, verify_research_bundle

payload = build_benchmark_research_bundle(result)
verification = verify_research_bundle(payload)
assert verification.valid
```

```bash
volforge benchmark data/replay.csv reports/model-predictions.csv \
  --report reports/model-benchmark.json \
  --bundle reports/model-benchmark.zip
```

The Streamlit Model benchmark workspace provides the same bundle as a download after each run.

VolForge v1.1 can sign the exact bundle bytes with a detached Ed25519 attestation without changing
the deterministic ZIP. Signature validity and signer trust remain separate decisions. See
[Signed research provenance](signed-provenance.md) for the trust model, key workflow, and limits.
