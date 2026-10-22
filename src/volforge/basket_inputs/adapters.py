"""Bounded source adapters for canonical basket-input records."""

from __future__ import annotations

import json
import os
import ssl
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

MAX_ADAPTER_RESPONSE_BYTES = 32 * 1024 * 1024


class BasketAdapterError(RuntimeError):
    """Raised when a basket-input source cannot be fetched or decoded safely."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BasketAdapterError(f"adapter response contains duplicate JSON field {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class BasketAdapterPayload:
    """One raw provider artifact and its normalized canonical records."""

    raw_bytes: bytes
    source_file_name: str
    frame: pd.DataFrame
    adapter_name: str
    metadata: dict[str, Any]


@runtime_checkable
class BasketInputAdapter(Protocol):
    """Protocol implemented by basket-input source adapters."""

    def fetch(self) -> BasketAdapterPayload:
        """Fetch one bounded artifact and normalize it into canonical records."""


@dataclass(frozen=True, slots=True)
class CsvBasketInputAdapter:
    """Read canonical basket-input rows from one local CSV artifact."""

    path: Path

    def fetch(self) -> BasketAdapterPayload:
        resolved = self.path.resolve()
        try:
            size = resolved.stat().st_size
            if size > MAX_ADAPTER_RESPONSE_BYTES:
                raise BasketAdapterError(
                    f"CSV adapter input exceeds {MAX_ADAPTER_RESPONSE_BYTES} bytes"
                )
            raw_bytes = resolved.read_bytes()
            frame = pd.read_csv(BytesIO(raw_bytes))
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            raise BasketAdapterError(f"could not read basket-input CSV: {exc}") from exc
        return BasketAdapterPayload(
            raw_bytes=raw_bytes,
            source_file_name=resolved.name,
            frame=frame,
            adapter_name="canonical-csv",
            metadata={"path_name": resolved.name, "size_bytes": len(raw_bytes)},
        )


@dataclass(frozen=True, slots=True)
class HttpsJsonBasketInputAdapter:
    """Fetch canonical JSON records over verified HTTPS with bounded resources.

    The endpoint must return either a JSON list of record objects or an object with
    exactly a ``records`` list and optional ``dataset_id`` string. Authentication is
    read from an environment variable at fetch time and is never retained in adapter
    metadata, raw manifests, exception text, or URLs.
    """

    endpoint: str
    authorization_env: str | None = None
    timeout_seconds: float = 15.0
    maximum_response_bytes: int = MAX_ADAPTER_RESPONSE_BYTES

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if parsed.scheme.lower() != "https" or not parsed.hostname:
            raise ValueError("basket HTTPS adapter requires an https:// endpoint with a host")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "basket HTTPS adapter forbids credentials, query strings, and fragments in URLs"
            )
        if not 0.0 < self.timeout_seconds <= 120.0:
            raise ValueError("timeout_seconds must lie in (0, 120]")
        if not 1 <= self.maximum_response_bytes <= MAX_ADAPTER_RESPONSE_BYTES:
            raise ValueError(
                f"maximum_response_bytes must lie in [1, {MAX_ADAPTER_RESPONSE_BYTES}]"
            )
        if self.authorization_env is not None and not self.authorization_env.strip():
            raise ValueError("authorization_env must be non-empty when supplied")

    @property
    def sanitized_endpoint(self) -> str:
        """Return an endpoint representation that cannot contain credentials."""
        parsed = urlsplit(self.endpoint)
        return urlunsplit(("https", parsed.netloc, parsed.path, "", ""))

    def fetch(self) -> BasketAdapterPayload:
        headers = {
            "Accept": "application/json",
            "User-Agent": "VolForge-basket-input-adapter/1.6",
        }
        if self.authorization_env is not None:
            token = os.environ.get(self.authorization_env)
            if token is None or not token.strip():
                raise BasketAdapterError(
                    f"authorization environment variable {self.authorization_env!r} is not set"
                )
            if "\r" in token or "\n" in token:
                raise BasketAdapterError("authorization token contains prohibited line breaks")
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(self.endpoint, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
                context=ssl.create_default_context(),
            ) as response:
                status = int(getattr(response, "status", 200))
                if status != 200:
                    raise BasketAdapterError(f"basket HTTPS adapter returned HTTP {status}")
                declared_length = response.headers.get("Content-Length")
                if (
                    declared_length is not None
                    and int(declared_length) > self.maximum_response_bytes
                ):
                    raise BasketAdapterError(
                        f"basket HTTPS response exceeds {self.maximum_response_bytes} bytes"
                    )
                raw_bytes = response.read(self.maximum_response_bytes + 1)
                if len(raw_bytes) > self.maximum_response_bytes:
                    raise BasketAdapterError(
                        f"basket HTTPS response exceeds {self.maximum_response_bytes} bytes"
                    )
                response_metadata = {
                    "endpoint": self.sanitized_endpoint,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "content_type": response.headers.get("Content-Type"),
                    "size_bytes": len(raw_bytes),
                }
        except BasketAdapterError:
            raise
        except (OSError, ValueError) as exc:
            raise BasketAdapterError(
                f"basket HTTPS adapter could not fetch {self.sanitized_endpoint}: {exc}"
            ) from exc
        try:
            decoded = raw_bytes.decode("utf-8")
            payload = json.loads(decoded, object_pairs_hook=_strict_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BasketAdapterError("basket HTTPS response must be valid UTF-8 JSON") from exc
        dataset_id: str | None = None
        records: object
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            unexpected = sorted(set(payload) - {"records", "dataset_id"})
            if unexpected:
                raise BasketAdapterError(
                    "basket HTTPS response has unsupported fields: " + ", ".join(unexpected)
                )
            records = payload.get("records")
            raw_dataset_id = payload.get("dataset_id")
            if raw_dataset_id is not None:
                if not isinstance(raw_dataset_id, str) or not raw_dataset_id.strip():
                    raise BasketAdapterError("dataset_id must be a non-empty string")
                dataset_id = raw_dataset_id.strip()
        else:
            raise BasketAdapterError("basket HTTPS response must be a JSON list or object")
        if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
            raise BasketAdapterError("basket HTTPS records must be a JSON list of objects")
        metadata = {**response_metadata, "dataset_id": dataset_id}
        return BasketAdapterPayload(
            raw_bytes=raw_bytes,
            source_file_name="basket-inputs.json",
            frame=pd.DataFrame(records),
            adapter_name="canonical-https-json",
            metadata=metadata,
        )


__all__ = [
    "MAX_ADAPTER_RESPONSE_BYTES",
    "BasketAdapterError",
    "BasketAdapterPayload",
    "BasketInputAdapter",
    "CsvBasketInputAdapter",
    "HttpsJsonBasketInputAdapter",
]
