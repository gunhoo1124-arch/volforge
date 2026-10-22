"""Deterministic, content-addressed research-report bundles."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any

from volforge import __version__
from volforge.benchmark import ChronologicalBenchmarkResult, benchmark_report_json
from volforge.reporting import dataframe_to_csv

_BUNDLE_SCHEMA_VERSION = 1
_MAX_BUNDLE_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ResearchBundleVerification:
    """Integrity result for a VolForge research bundle."""

    valid: bool
    bundle_root_sha256: str | None
    artifact_count: int
    errors: tuple[str, ...]


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _artifact_payloads(result: ChronologicalBenchmarkResult) -> dict[str, bytes]:
    return {
        "audit/benchmark.json": benchmark_report_json(result),
        "data/audit-steps.csv": dataframe_to_csv(result.audit_frame()),
        "data/fits.csv": dataframe_to_csv(result.fit_frame()),
        "data/maturity-moneyness.csv": dataframe_to_csv(result.panel_frame()),
        "data/predictions.csv": dataframe_to_csv(result.prediction_frame()),
        "data/summary.csv": dataframe_to_csv(result.summary_frame()),
        "data/timeline.csv": dataframe_to_csv(result.timeline_frame()),
        "inference/model-comparison.json": (
            json.dumps(asdict(result.comparison_inference()), indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    }


def _manifest_core(
    result: ChronologicalBenchmarkResult,
    artifacts: dict[str, dict[str, int | str]],
) -> dict[str, Any]:
    root_payload = {
        "schema_version": _BUNDLE_SCHEMA_VERSION,
        "volforge_version": __version__,
        "benchmark_final_audit_hash": result.final_audit_hash,
        "benchmark_settings": asdict(result.settings),
        "artifacts": artifacts,
    }
    return {
        "format": "volforge-benchmark-research-bundle",
        **root_payload,
        "bundle_root_sha256": _sha256(_canonical_json(root_payload)),
    }


def build_benchmark_research_bundle(result: ChronologicalBenchmarkResult) -> bytes:
    """Build a byte-for-byte reproducible ZIP with a content-addressed manifest."""
    payloads = _artifact_payloads(result)
    artifacts = {
        name: {"bytes": len(payload), "sha256": _sha256(payload)}
        for name, payload in sorted(payloads.items())
    }
    core = _manifest_core(result, artifacts)
    manifest = {**core, "manifest_sha256": _sha256(_canonical_json(core))}
    entries = {"manifest.json": _canonical_json(manifest), **payloads}
    output = BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as bundle:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            bundle.writestr(
                info, entries[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
            )
    return output.getvalue()


def verify_research_bundle(payload: bytes) -> ResearchBundleVerification:
    """Verify the manifest, every declared artifact, and the bundle root hash."""
    errors: list[str] = []
    if len(payload) > _MAX_BUNDLE_BYTES:
        return ResearchBundleVerification(False, None, 0, ("bundle exceeds 100 MiB",))
    try:
        with zipfile.ZipFile(BytesIO(payload)) as bundle:
            names = bundle.namelist()
            if len(names) != len(set(names)):
                errors.append("bundle contains duplicate member names")
            if sum(info.file_size for info in bundle.infolist()) > _MAX_BUNDLE_BYTES:
                return ResearchBundleVerification(
                    False, None, 0, ("expanded bundle exceeds 100 MiB",)
                )
            if "manifest.json" not in names:
                return ResearchBundleVerification(False, None, 0, ("manifest.json is missing",))
            manifest = json.loads(bundle.read("manifest.json"), object_pairs_hook=_unique_object)
            if not isinstance(manifest, dict):
                return ResearchBundleVerification(
                    False, None, 0, ("manifest root must be an object",)
                )
            artifacts = manifest.get("artifacts")
            if not isinstance(artifacts, dict):
                return ResearchBundleVerification(
                    False, None, 0, ("manifest artifacts are invalid",)
                )
            declared = set(artifacts)
            observed = set(names).difference({"manifest.json"})
            if declared != observed:
                errors.append("declared and archived artifact names differ")
            for name, metadata in artifacts.items():
                if name not in observed:
                    continue
                info = bundle.getinfo(name)
                if info.file_size > _MAX_BUNDLE_BYTES:
                    errors.append(f"artifact {name} exceeds 100 MiB")
                    continue
                artifact = bundle.read(name)
                if not isinstance(metadata, dict):
                    errors.append(f"artifact metadata is invalid: {name}")
                    continue
                if metadata.get("bytes") != len(artifact):
                    errors.append(f"artifact byte count mismatch: {name}")
                if metadata.get("sha256") != _sha256(artifact):
                    errors.append(f"artifact hash mismatch: {name}")
            manifest_hash = manifest.pop("manifest_sha256", None)
            if manifest_hash != _sha256(_canonical_json(manifest)):
                errors.append("manifest hash mismatch")
            root_hash = manifest.get("bundle_root_sha256")
            root_payload = {
                key: manifest.get(key)
                for key in (
                    "schema_version",
                    "volforge_version",
                    "benchmark_final_audit_hash",
                    "benchmark_settings",
                    "artifacts",
                )
            }
            if root_hash != _sha256(_canonical_json(root_payload)):
                errors.append("bundle root hash mismatch")
            if manifest.get("format") != "volforge-benchmark-research-bundle":
                errors.append("bundle format is not supported")
            if manifest.get("schema_version") != _BUNDLE_SCHEMA_VERSION:
                errors.append("bundle schema version is not supported")
            return ResearchBundleVerification(
                valid=not errors,
                bundle_root_sha256=str(root_hash) if root_hash is not None else None,
                artifact_count=len(artifacts),
                errors=tuple(errors),
            )
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        return ResearchBundleVerification(False, None, 0, (f"invalid research bundle: {exc}",))
