"""Append-only storage and point-in-time resolution for basket model inputs."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from volforge.basket import BasketAssetMarket, BasketMarket, parse_basket_market
from volforge.basket_inputs.adapters import (
    MAX_ADAPTER_RESPONSE_BYTES,
    BasketAdapterPayload,
    BasketInputAdapter,
    CsvBasketInputAdapter,
)

BASKET_INPUT_SCHEMA_VERSION = 1
BASKET_INPUT_RECORD_TYPES = ("component", "correlation")
BASKET_INPUT_MODES = ("initial", "incremental", "backfill", "correction")
SYNTHETIC_BASKET_ID = "VOLFORGE-SYNTHETIC-ADJUSTED"
MAX_BASKET_INPUT_ROWS = 100_000
BASKET_INPUT_COLUMNS = (
    "record_type",
    "snapshot_id",
    "market_timestamp",
    "received_timestamp",
    "source_record_id",
    "revision",
    "record_action",
    "basket_id",
    "asset_id",
    "paired_asset_id",
    "spot",
    "volatility",
    "dividend_yield",
    "bid",
    "ask",
    "correlation",
)
_RECORD_PAYLOAD_COLUMNS = (
    "source",
    *BASKET_INPUT_COLUMNS,
)


class BasketInputSchemaError(ValueError):
    """Raised when canonical basket-input records are malformed."""


class BasketInputIntegrityError(RuntimeError):
    """Raised when append-only or point-in-time invariants fail."""


@dataclass(frozen=True, slots=True)
class BasketInputBatch:
    """One append-only basket-input ingestion and its chained manifest."""

    batch_sequence: int
    batch_id: str
    schema_version: int
    source: str
    mode: str
    ingested_timestamp: str
    source_file_name: str
    source_sha256: str
    source_size_bytes: int
    raw_archive_name: str
    row_count: int
    component_count: int
    correlation_count: int
    minimum_market_timestamp: str
    maximum_market_timestamp: str
    reason: str | None
    adapter_name: str
    adapter_metadata_json: str
    previous_manifest_hash: str | None
    manifest_hash: str


@dataclass(frozen=True, slots=True)
class BasketInputVerification:
    """Integrity result for a basket-input database and its raw archives."""

    valid: bool
    schema_version: int
    batch_count: int
    record_count: int
    final_manifest_hash: str | None
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BasketInputSnapshot:
    """Complete component/execution/correlation package known at formation."""

    basket_id: str
    source: str
    snapshot_id: str
    formation_timestamp: str
    market_timestamp: str
    received_timestamp: str
    dataset_known_at: str
    market: BasketMarket
    execution_json: str
    record_count: int
    source_record_ids: tuple[str, ...]
    record_payload_sha256s: tuple[str, ...]
    batch_ids: tuple[str, ...]
    manifest_hashes: tuple[str, ...]
    snapshot_sha256: str

    @property
    def market_json(self) -> str:
        """Return the canonical basket-market JSON consumed by the pricing engine."""
        return self.market.canonical_json

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible snapshot audit."""
        return {
            "basket_id": self.basket_id,
            "source": self.source,
            "snapshot_id": self.snapshot_id,
            "formation_timestamp": self.formation_timestamp,
            "market_timestamp": self.market_timestamp,
            "received_timestamp": self.received_timestamp,
            "dataset_known_at": self.dataset_known_at,
            "market_json": self.market_json,
            "market_sha256": self.market.sha256,
            "execution_json": self.execution_json,
            "record_count": self.record_count,
            "source_record_ids": list(self.source_record_ids),
            "record_payload_sha256s": list(self.record_payload_sha256s),
            "batch_ids": list(self.batch_ids),
            "manifest_hashes": list(self.manifest_hashes),
            "snapshot_sha256": self.snapshot_sha256,
        }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: bytes | str) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BasketInputSchemaError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _timestamp(value: object, name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise BasketInputSchemaError(f"{name} must be a valid timezone-aware timestamp") from exc
    if pd.isna(result) or result.tzinfo is None:
        raise BasketInputSchemaError(f"{name} must be a valid timezone-aware timestamp")
    return result.tz_convert("UTC")


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BasketInputSchemaError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, name: str) -> str | None:
    if value is None or pd.isna(value):
        return None
    if not isinstance(value, str):
        raise BasketInputSchemaError(f"{name} must be a string when supplied")
    stripped = value.strip()
    return stripped or None


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise BasketInputSchemaError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BasketInputSchemaError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise BasketInputSchemaError(f"{name} must be a finite number")
    return result


def _optional_finite(value: object, name: str) -> float | None:
    if value is None or pd.isna(value):
        return None
    return _finite(value, name)


def _revision(value: object, prefix: str) -> int:
    if isinstance(value, bool):
        raise BasketInputSchemaError(f"{prefix} revision must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise BasketInputSchemaError(f"{prefix} revision must be a non-negative integer") from exc
    if result < 0 or float(value) != result:
        raise BasketInputSchemaError(f"{prefix} revision must be a non-negative integer")
    return result


def _normalize_records(frame: pd.DataFrame, *, source: str) -> list[dict[str, Any]]:
    missing = sorted(set(BASKET_INPUT_COLUMNS) - set(frame.columns))
    if missing:
        raise BasketInputSchemaError("missing basket-input columns: " + ", ".join(missing))
    unexpected = sorted(set(frame.columns) - set(BASKET_INPUT_COLUMNS))
    if unexpected:
        raise BasketInputSchemaError("unsupported basket-input columns: " + ", ".join(unexpected))
    if not 1 <= len(frame) <= MAX_BASKET_INPUT_ROWS:
        raise BasketInputSchemaError(
            f"basket-input batch requires between 1 and {MAX_BASKET_INPUT_ROWS} rows"
        )
    records: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        prefix = f"row {position}"
        record_type = _required_text(row["record_type"], f"{prefix} record_type").lower()
        if record_type not in BASKET_INPUT_RECORD_TYPES:
            raise BasketInputSchemaError(
                f"{prefix} record_type must be one of {', '.join(BASKET_INPUT_RECORD_TYPES)}"
            )
        action = _required_text(row["record_action"], f"{prefix} record_action").lower()
        if action not in {"upsert", "cancel"}:
            raise BasketInputSchemaError(f"{prefix} record_action must be upsert or cancel")
        market = _timestamp(row["market_timestamp"], f"{prefix} market_timestamp")
        received = _timestamp(row["received_timestamp"], f"{prefix} received_timestamp")
        if market > received:
            raise BasketInputSchemaError(
                f"{prefix} requires market_timestamp <= received_timestamp"
            )
        asset_id = _required_text(row["asset_id"], f"{prefix} asset_id")
        paired_asset_id = _optional_text(row["paired_asset_id"], f"{prefix} paired_asset_id")
        spot = _optional_finite(row["spot"], f"{prefix} spot")
        volatility = _optional_finite(row["volatility"], f"{prefix} volatility")
        dividend_yield = _optional_finite(row["dividend_yield"], f"{prefix} dividend_yield")
        bid = _optional_finite(row["bid"], f"{prefix} bid")
        ask = _optional_finite(row["ask"], f"{prefix} ask")
        correlation = _optional_finite(row["correlation"], f"{prefix} correlation")
        if record_type == "component":
            if paired_asset_id is not None or correlation is not None:
                raise BasketInputSchemaError(
                    f"{prefix} component records cannot set paired_asset_id or correlation"
                )
            if action == "upsert":
                if any(value is None for value in (spot, volatility, dividend_yield, bid, ask)):
                    raise BasketInputSchemaError(
                        f"{prefix} component upsert requires spot, volatility, dividend_yield, bid, and ask"
                    )
                assert spot is not None and volatility is not None
                assert bid is not None and ask is not None
                if spot <= 0.0 or volatility < 0.0 or bid <= 0.0 or ask < bid:
                    raise BasketInputSchemaError(
                        f"{prefix} component requires positive spot/bid, non-negative volatility, and bid <= ask"
                    )
                if not bid <= spot <= ask:
                    raise BasketInputSchemaError(
                        f"{prefix} component spot must lie inside its executable bid/ask"
                    )
            else:
                spot = volatility = dividend_yield = bid = ask = None
        else:
            if paired_asset_id is None:
                raise BasketInputSchemaError(f"{prefix} correlation requires paired_asset_id")
            if paired_asset_id == asset_id:
                raise BasketInputSchemaError(
                    f"{prefix} correlation pair must contain two different assets"
                )
            if asset_id > paired_asset_id:
                raise BasketInputSchemaError(
                    f"{prefix} correlation pair must use lexicographic asset order"
                )
            if any(value is not None for value in (spot, volatility, dividend_yield, bid, ask)):
                raise BasketInputSchemaError(
                    f"{prefix} correlation records cannot set component market fields"
                )
            if action == "upsert":
                if correlation is None or not -1.0 <= correlation <= 1.0:
                    raise BasketInputSchemaError(f"{prefix} correlation must lie in [-1, 1]")
            else:
                correlation = None
        record = {
            "source": source,
            "record_type": record_type,
            "snapshot_id": _required_text(row["snapshot_id"], f"{prefix} snapshot_id"),
            "market_timestamp": market.isoformat(),
            "received_timestamp": received.isoformat(),
            "source_record_id": _required_text(
                row["source_record_id"], f"{prefix} source_record_id"
            ),
            "revision": _revision(row["revision"], prefix),
            "record_action": action,
            "basket_id": _required_text(row["basket_id"], f"{prefix} basket_id"),
            "asset_id": asset_id,
            "paired_asset_id": paired_asset_id,
            "spot": spot,
            "volatility": volatility,
            "dividend_yield": dividend_yield,
            "bid": bid,
            "ask": ask,
            "correlation": correlation,
        }
        payload = _canonical_json(record)
        record["payload_json"] = payload
        record["payload_sha256"] = _sha256(payload)
        records.append(record)
    identities = [(record["source_record_id"], record["revision"]) for record in records]
    if len(identities) != len(set(identities)):
        raise BasketInputSchemaError("batch contains duplicate source_record_id/revision pairs")
    return records


def _validate_new_packages(records: list[dict[str, Any]], *, mode: str) -> None:
    if mode == "correction":
        return
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    snapshot_identities: dict[str, tuple[str, str]] = {}
    for record in records:
        if record["record_action"] != "upsert":
            continue
        identity = (record["basket_id"], record["market_timestamp"])
        previous = snapshot_identities.setdefault(record["snapshot_id"], identity)
        if previous != identity:
            raise BasketInputSchemaError(
                f"snapshot_id {record['snapshot_id']!r} maps to multiple basket/time identities"
            )
        key = (record["basket_id"], record["snapshot_id"], record["market_timestamp"])
        grouped.setdefault(key, []).append(record)
    for (_, snapshot_id, _), package in grouped.items():
        components = [record for record in package if record["record_type"] == "component"]
        correlations = [record for record in package if record["record_type"] == "correlation"]
        assets = [record["asset_id"] for record in components]
        if not assets:
            raise BasketInputSchemaError(f"snapshot {snapshot_id!r} has no component records")
        if len(assets) != len(set(assets)):
            raise BasketInputSchemaError(f"snapshot {snapshot_id!r} has duplicate components")
        expected_pairs = {
            (left, right)
            for index, left in enumerate(sorted(assets))
            for right in sorted(assets)[index + 1 :]
        }
        actual_pairs = {(record["asset_id"], record["paired_asset_id"]) for record in correlations}
        if len(actual_pairs) != len(correlations):
            raise BasketInputSchemaError(
                f"snapshot {snapshot_id!r} has duplicate correlation pairs"
            )
        if actual_pairs != expected_pairs:
            missing = sorted(expected_pairs - actual_pairs)
            extra = sorted(actual_pairs - expected_pairs)
            reasons: list[str] = []
            if missing:
                reasons.append(f"missing pairs {missing}")
            if extra:
                reasons.append(f"unknown pairs {extra}")
            raise BasketInputSchemaError(
                f"snapshot {snapshot_id!r} has an incomplete correlation package: "
                + "; ".join(reasons)
            )


def _manifest_fields(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "schema_version",
            "source",
            "mode",
            "ingested_timestamp",
            "source_file_name",
            "source_sha256",
            "source_size_bytes",
            "raw_archive_name",
            "row_count",
            "component_count",
            "correlation_count",
            "minimum_market_timestamp",
            "maximum_market_timestamp",
            "reason",
            "adapter_name",
            "adapter_metadata_json",
            "previous_manifest_hash",
        )
    }


class BasketInputStore:
    """SQLite-backed append-only store for basket model and execution inputs."""

    def __init__(self, path: str | PathLike[str]) -> None:
        self.path = Path(path).resolve()
        self.raw_directory = self.path.with_suffix(self.path.suffix + ".raw")
        if not self.path.is_file():
            raise FileNotFoundError(f"basket-input store does not exist: {self.path}")
        if not self.raw_directory.is_dir():
            raise FileNotFoundError(
                f"basket-input raw archive directory does not exist: {self.raw_directory}"
            )

    @classmethod
    def create(cls, path: str | PathLike[str]) -> BasketInputStore:
        """Create a new store and append-only database triggers."""
        resolved = Path(path).resolve()
        if resolved.exists() or resolved.with_suffix(resolved.suffix + ".raw").exists():
            raise FileExistsError(f"refusing to overwrite basket-input store: {resolved}")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(resolved)
        try:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE basket_input_batches (
                    batch_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL UNIQUE,
                    schema_version INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    ingested_timestamp TEXT NOT NULL,
                    source_file_name TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    source_size_bytes INTEGER NOT NULL,
                    raw_archive_name TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    component_count INTEGER NOT NULL,
                    correlation_count INTEGER NOT NULL,
                    minimum_market_timestamp TEXT NOT NULL,
                    maximum_market_timestamp TEXT NOT NULL,
                    reason TEXT,
                    adapter_name TEXT NOT NULL,
                    adapter_metadata_json TEXT NOT NULL,
                    previous_manifest_hash TEXT,
                    manifest_hash TEXT NOT NULL UNIQUE,
                    UNIQUE(source, source_sha256)
                );
                CREATE TABLE basket_input_records (
                    record_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL REFERENCES basket_input_batches(batch_id),
                    ingested_timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    record_type TEXT NOT NULL CHECK(record_type IN ('component', 'correlation')),
                    snapshot_id TEXT NOT NULL,
                    market_timestamp TEXT NOT NULL,
                    received_timestamp TEXT NOT NULL,
                    source_record_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 0),
                    record_action TEXT NOT NULL CHECK(record_action IN ('upsert', 'cancel')),
                    basket_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    paired_asset_id TEXT,
                    spot REAL,
                    volatility REAL,
                    dividend_yield REAL,
                    bid REAL,
                    ask REAL,
                    correlation REAL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    UNIQUE(source, source_record_id, revision)
                );
                CREATE INDEX basket_input_knowledge_idx
                ON basket_input_records(
                    basket_id, source, market_timestamp, received_timestamp, ingested_timestamp
                );
                CREATE INDEX basket_input_revision_idx
                ON basket_input_records(source, source_record_id, revision);
                CREATE TRIGGER basket_input_batches_no_update
                BEFORE UPDATE ON basket_input_batches
                BEGIN SELECT RAISE(ABORT, 'basket_input_batches is append-only'); END;
                CREATE TRIGGER basket_input_batches_no_delete
                BEFORE DELETE ON basket_input_batches
                BEGIN SELECT RAISE(ABORT, 'basket_input_batches is append-only'); END;
                CREATE TRIGGER basket_input_records_no_update
                BEFORE UPDATE ON basket_input_records
                BEGIN SELECT RAISE(ABORT, 'basket_input_records is append-only'); END;
                CREATE TRIGGER basket_input_records_no_delete
                BEFORE DELETE ON basket_input_records
                BEGIN SELECT RAISE(ABORT, 'basket_input_records is append-only'); END;
                """
            )
            connection.commit()
        finally:
            connection.close()
        resolved.with_suffix(resolved.suffix + ".raw").mkdir(parents=True, exist_ok=False)
        return cls(resolved)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def ingest_csv(
        self,
        source_path: str | PathLike[str],
        *,
        source: str,
        mode: Literal["initial", "incremental", "backfill", "correction"],
        ingested_timestamp: str | pd.Timestamp,
        reason: str | None = None,
    ) -> BasketInputBatch:
        """Validate, archive, hash, and atomically append canonical CSV records."""
        return self.ingest_adapter(
            CsvBasketInputAdapter(Path(source_path)),
            source=source,
            mode=mode,
            ingested_timestamp=ingested_timestamp,
            reason=reason,
        )

    def ingest_adapter(
        self,
        adapter: BasketInputAdapter,
        *,
        source: str,
        mode: Literal["initial", "incremental", "backfill", "correction"],
        ingested_timestamp: str | pd.Timestamp,
        reason: str | None = None,
    ) -> BasketInputBatch:
        """Fetch once through an adapter and append its exact raw artifact."""
        payload = adapter.fetch()
        if not isinstance(payload, BasketAdapterPayload):
            raise TypeError("basket-input adapter must return BasketAdapterPayload")
        return self._ingest(
            payload.frame,
            raw_bytes=payload.raw_bytes,
            source_file_name=payload.source_file_name,
            source=source,
            mode=mode,
            ingested_timestamp=ingested_timestamp,
            reason=reason,
            adapter_name=payload.adapter_name,
            adapter_metadata=payload.metadata,
        )

    def ingest_frame(
        self,
        frame: pd.DataFrame,
        *,
        source_file_name: str,
        source: str,
        mode: Literal["initial", "incremental", "backfill", "correction"],
        ingested_timestamp: str | pd.Timestamp,
        reason: str | None = None,
        adapter_name: str = "canonical-dataframe",
        adapter_metadata: dict[str, Any] | None = None,
    ) -> BasketInputBatch:
        """Append a frame using a deterministic CSV representation as its raw artifact."""
        raw_bytes = frame.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode(
            "utf-8"
        )
        return self._ingest(
            frame,
            raw_bytes=raw_bytes,
            source_file_name=Path(source_file_name).name,
            source=source,
            mode=mode,
            ingested_timestamp=ingested_timestamp,
            reason=reason,
            adapter_name=adapter_name,
            adapter_metadata=adapter_metadata or {},
        )

    def _ingest(
        self,
        frame: pd.DataFrame,
        *,
        raw_bytes: bytes,
        source_file_name: str,
        source: str,
        mode: str,
        ingested_timestamp: str | pd.Timestamp,
        reason: str | None,
        adapter_name: str,
        adapter_metadata: dict[str, Any],
    ) -> BasketInputBatch:
        source_name = _required_text(source, "source")
        if mode not in BASKET_INPUT_MODES:
            raise ValueError(f"mode must be one of {', '.join(BASKET_INPUT_MODES)}")
        if not source_file_name or Path(source_file_name).name != source_file_name:
            raise ValueError("source_file_name must be a non-empty base name")
        if len(raw_bytes) > MAX_ADAPTER_RESPONSE_BYTES:
            raise BasketInputSchemaError(
                f"basket-input artifact exceeds {MAX_ADAPTER_RESPONSE_BYTES} bytes"
            )
        adapter_name = _required_text(adapter_name, "adapter_name")
        try:
            adapter_metadata_json = _canonical_json(adapter_metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("adapter_metadata must be JSON serializable") from exc
        reason_text = reason.strip() if reason is not None else None
        if mode in {"backfill", "correction"} and not reason_text:
            raise ValueError(f"{mode} ingestion requires a non-empty reason")
        ingested = _timestamp(ingested_timestamp, "ingested_timestamp")
        records = _normalize_records(frame, source=source_name)
        _validate_new_packages(records, mode=mode)
        if any(
            _timestamp(record["received_timestamp"], "received_timestamp") > ingested
            for record in records
        ):
            raise BasketInputIntegrityError(
                "received_timestamp cannot follow the batch ingested_timestamp"
            )
        market_times = [record["market_timestamp"] for record in records]
        source_digest = _sha256(raw_bytes)
        suffix = Path(source_file_name).suffix.lower()
        archive_suffix = suffix if suffix in {".csv", ".json"} else ".bin"
        archive_name = f"{source_digest}{archive_suffix}"
        archive_path = self.raw_directory / archive_name

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            duplicate_batch = connection.execute(
                """
                SELECT 1 FROM basket_input_batches
                WHERE source = ? AND source_sha256 = ?
                """,
                (source_name, source_digest),
            ).fetchone()
            if duplicate_batch is not None:
                raise BasketInputIntegrityError("this exact source artifact was already ingested")
            latest_ingestion = connection.execute(
                "SELECT MAX(ingested_timestamp) FROM basket_input_batches"
            ).fetchone()[0]
            if latest_ingestion is not None and ingested.isoformat() < latest_ingestion:
                raise BasketInputIntegrityError(
                    "ingested_timestamp cannot precede the store's latest ingestion"
                )
            source_state = connection.execute(
                """
                SELECT COUNT(*) AS count, MAX(market_timestamp) AS maximum_market_timestamp
                FROM basket_input_records WHERE source = ?
                """,
                (source_name,),
            ).fetchone()
            source_exists = int(source_state["count"]) > 0
            if mode == "initial" and source_exists:
                raise BasketInputIntegrityError("initial mode requires an empty source history")
            if mode != "initial" and not source_exists:
                raise BasketInputIntegrityError(f"{mode} mode requires an existing source history")
            if (
                mode == "incremental"
                and source_state["maximum_market_timestamp"] is not None
                and min(market_times) < source_state["maximum_market_timestamp"]
            ):
                raise BasketInputIntegrityError(
                    "incremental ingestion would silently backfill older snapshots; use backfill mode"
                )

            identifiers = sorted({record["source_record_id"] for record in records})
            existing_records: dict[str, sqlite3.Row] = {}
            if identifiers:
                placeholders = ",".join("?" for _ in identifiers)
                rows = connection.execute(
                    f"""
                    WITH ranked AS (
                        SELECT *, ROW_NUMBER() OVER (
                            PARTITION BY source_record_id ORDER BY revision DESC
                        ) AS revision_rank
                        FROM basket_input_records
                        WHERE source = ? AND source_record_id IN ({placeholders})
                    )
                    SELECT * FROM ranked WHERE revision_rank = 1
                    """,
                    (source_name, *identifiers),
                ).fetchall()
                existing_records = {str(row["source_record_id"]): row for row in rows}
            if mode == "correction":
                new_ids = sorted(set(identifiers) - set(existing_records))
                if new_ids:
                    raise BasketInputIntegrityError(
                        "correction mode requires every source_record_id to exist already"
                    )
            elif existing_records:
                raise BasketInputIntegrityError(
                    "existing source_record_id requires explicit correction mode"
                )

            grouped: dict[str, list[dict[str, Any]]] = {}
            for record in records:
                grouped.setdefault(record["source_record_id"], []).append(record)
            identity_columns = (
                "record_type",
                "snapshot_id",
                "market_timestamp",
                "basket_id",
                "asset_id",
                "paired_asset_id",
            )
            for identifier, revisions in grouped.items():
                revisions.sort(key=lambda item: int(item["revision"]))
                existing = existing_records.get(identifier)
                previous_revision = int(existing["revision"]) if existing is not None else -1
                previous_received = (
                    str(existing["received_timestamp"]) if existing is not None else None
                )
                active = existing is not None and existing["record_action"] == "upsert"
                for record in revisions:
                    if int(record["revision"]) <= previous_revision:
                        raise BasketInputIntegrityError(
                            f"revision for {identifier!r} must increase strictly"
                        )
                    if existing is not None and any(
                        record[column] != existing[column] for column in identity_columns
                    ):
                        raise BasketInputIntegrityError(
                            f"identity fields for {identifier!r} cannot change across revisions"
                        )
                    if (
                        previous_received is not None
                        and record["received_timestamp"] < previous_received
                    ):
                        raise BasketInputIntegrityError(
                            f"received_timestamp for {identifier!r} cannot move backward"
                        )
                    if record["record_action"] == "cancel" and not active:
                        raise BasketInputIntegrityError(
                            f"cannot cancel {identifier!r} without an active prior upsert"
                        )
                    previous_revision = int(record["revision"])
                    previous_received = record["received_timestamp"]
                    active = record["record_action"] == "upsert"

            snapshot_keys = sorted({record["snapshot_id"] for record in records})
            for snapshot_id in snapshot_keys:
                identities = connection.execute(
                    """
                    SELECT DISTINCT basket_id, market_timestamp
                    FROM basket_input_records WHERE source = ? AND snapshot_id = ?
                    """,
                    (source_name, snapshot_id),
                ).fetchall()
                expected = {
                    (record["basket_id"], record["market_timestamp"])
                    for record in records
                    if record["snapshot_id"] == snapshot_id
                }
                existing_identities = {
                    (str(row["basket_id"]), str(row["market_timestamp"])) for row in identities
                }
                if existing_identities and existing_identities != expected:
                    raise BasketInputIntegrityError(
                        f"snapshot_id {snapshot_id!r} cannot change basket/time identity"
                    )

            if archive_path.exists():
                if _sha256(archive_path.read_bytes()) != source_digest:
                    raise BasketInputIntegrityError(
                        "existing raw archive content does not match its content address"
                    )
            else:
                archive_path.write_bytes(raw_bytes)
            previous_row = connection.execute(
                """
                SELECT manifest_hash FROM basket_input_batches
                ORDER BY batch_sequence DESC LIMIT 1
                """
            ).fetchone()
            previous_hash = str(previous_row[0]) if previous_row is not None else None
            batch_fields: dict[str, Any] = {
                "schema_version": BASKET_INPUT_SCHEMA_VERSION,
                "source": source_name,
                "mode": mode,
                "ingested_timestamp": ingested.isoformat(),
                "source_file_name": source_file_name,
                "source_sha256": source_digest,
                "source_size_bytes": len(raw_bytes),
                "raw_archive_name": archive_name,
                "row_count": len(records),
                "component_count": sum(record["record_type"] == "component" for record in records),
                "correlation_count": sum(
                    record["record_type"] == "correlation" for record in records
                ),
                "minimum_market_timestamp": min(market_times),
                "maximum_market_timestamp": max(market_times),
                "reason": reason_text,
                "adapter_name": adapter_name,
                "adapter_metadata_json": adapter_metadata_json,
                "previous_manifest_hash": previous_hash,
            }
            manifest_hash = _sha256(_canonical_json(_manifest_fields(batch_fields)))
            batch_id = manifest_hash
            connection.execute(
                """
                INSERT INTO basket_input_batches(
                    batch_id, schema_version, source, mode, ingested_timestamp,
                    source_file_name, source_sha256, source_size_bytes, raw_archive_name,
                    row_count, component_count, correlation_count, minimum_market_timestamp,
                    maximum_market_timestamp, reason, adapter_name, adapter_metadata_json,
                    previous_manifest_hash, manifest_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    BASKET_INPUT_SCHEMA_VERSION,
                    source_name,
                    mode,
                    ingested.isoformat(),
                    source_file_name,
                    source_digest,
                    len(raw_bytes),
                    archive_name,
                    len(records),
                    batch_fields["component_count"],
                    batch_fields["correlation_count"],
                    min(market_times),
                    max(market_times),
                    reason_text,
                    adapter_name,
                    adapter_metadata_json,
                    previous_hash,
                    manifest_hash,
                ),
            )
            record_columns = (*_RECORD_PAYLOAD_COLUMNS, "payload_json", "payload_sha256")
            placeholders = ",".join("?" for _ in range(len(record_columns) + 2))
            for record in records:
                connection.execute(
                    f"""
                    INSERT INTO basket_input_records(
                        batch_id, ingested_timestamp, {", ".join(record_columns)}
                    ) VALUES ({placeholders})
                    """,
                    (
                        batch_id,
                        ingested.isoformat(),
                        *(record[column] for column in record_columns),
                    ),
                )
            sequence = int(
                connection.execute(
                    "SELECT batch_sequence FROM basket_input_batches WHERE batch_id = ?",
                    (batch_id,),
                ).fetchone()[0]
            )
            connection.commit()
        return BasketInputBatch(
            batch_sequence=sequence,
            batch_id=batch_id,
            **batch_fields,
            manifest_hash=manifest_hash,
        )

    def batches(self) -> tuple[BasketInputBatch, ...]:
        """Return ingestion manifests in append order."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM basket_input_batches ORDER BY batch_sequence"
            ).fetchall()
        return tuple(
            BasketInputBatch(
                **{field: row[field] for field in BasketInputBatch.__dataclass_fields__}
            )
            for row in rows
        )

    def records_as_known(
        self,
        formation_at: str | pd.Timestamp,
        *,
        basket_id: str | None = None,
        source: str | None = None,
        dataset_known_at: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return latest active revisions available by an explicit formation time."""
        formation = _timestamp(formation_at, "formation_at")
        dataset_cutoff = (
            _timestamp(dataset_known_at, "dataset_known_at").isoformat()
            if dataset_known_at is not None
            else "9999-12-31T23:59:59+00:00"
        )
        conditions = [
            "r.market_timestamp <= ?",
            "r.received_timestamp <= ?",
            "r.ingested_timestamp <= ?",
        ]
        parameters: list[Any] = [formation.isoformat(), formation.isoformat(), dataset_cutoff]
        if basket_id is not None:
            conditions.append("r.basket_id = ?")
            parameters.append(_required_text(basket_id, "basket_id"))
        if source is not None:
            conditions.append("r.source = ?")
            parameters.append(_required_text(source, "source"))
        query = f"""
            WITH eligible AS (
                SELECT r.*, b.batch_sequence, b.manifest_hash,
                       ROW_NUMBER() OVER (
                           PARTITION BY r.source, r.source_record_id
                           ORDER BY r.revision DESC, r.received_timestamp DESC,
                                    r.ingested_timestamp DESC
                       ) AS revision_rank
                FROM basket_input_records r
                JOIN basket_input_batches b ON b.batch_id = r.batch_id
                WHERE {" AND ".join(conditions)}
            )
            SELECT * FROM eligible
            WHERE revision_rank = 1 AND record_action = 'upsert'
            ORDER BY market_timestamp, source, snapshot_id, record_type,
                     asset_id, paired_asset_id
        """
        with self._connection() as connection:
            frame = pd.read_sql_query(query, connection, params=parameters)
        for column in ("market_timestamp", "received_timestamp", "ingested_timestamp"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True)
        return frame.drop(columns=["revision_rank"], errors="ignore")

    def resolve_snapshot(
        self,
        basket_id: str,
        formation_timestamp: str | pd.Timestamp,
        *,
        source: str | None = None,
        maximum_age: str | pd.Timedelta = "15min",
        dataset_known_at: str | pd.Timestamp | None = None,
        _store_verified: bool = False,
    ) -> BasketInputSnapshot:
        """Resolve the newest complete snapshot that was received by formation."""
        if not _store_verified:
            verification = self.verify()
            if not verification.valid:
                raise BasketInputIntegrityError(
                    "basket-input store failed integrity verification: "
                    + "; ".join(verification.failure_reasons)
                )
        formation = _timestamp(formation_timestamp, "formation_timestamp")
        age_limit = pd.to_timedelta(maximum_age)
        if age_limit < pd.Timedelta(0):
            raise ValueError("maximum_age must be non-negative")
        frame = self.records_as_known(
            formation,
            basket_id=basket_id,
            source=source,
            dataset_known_at=dataset_known_at,
        )
        if frame.empty:
            raise BasketInputIntegrityError(
                f"no basket-input records were known for {basket_id!r} at {formation.isoformat()}"
            )
        sources = sorted(frame["source"].unique().tolist())
        if source is None and len(sources) != 1:
            raise BasketInputIntegrityError(
                "snapshot resolution requires source when multiple sources are eligible"
            )
        selected_source = _required_text(source or sources[0], "source")
        frame = frame.loc[frame["source"] == selected_source].copy()
        newest_market = frame["market_timestamp"].max()
        newest = frame.loc[frame["market_timestamp"] == newest_market].copy()
        snapshot_ids = sorted(newest["snapshot_id"].unique().tolist())
        if len(snapshot_ids) != 1:
            raise BasketInputIntegrityError(
                "newest basket-input market time maps to multiple snapshot_id values"
            )
        snapshot_id = str(snapshot_ids[0])
        package = newest.loc[newest["snapshot_id"] == snapshot_id].copy()
        received = package["received_timestamp"].max()
        if received > formation:
            raise BasketInputIntegrityError("resolved basket package was received after formation")
        market_age = formation - newest_market
        if market_age > age_limit:
            raise BasketInputIntegrityError(
                f"basket package age {market_age.total_seconds():.6g}s exceeds "
                f"{age_limit.total_seconds():.6g}s"
            )
        components = package.loc[package["record_type"] == "component"].copy()
        correlations = package.loc[package["record_type"] == "correlation"].copy()
        if components.empty:
            raise BasketInputIntegrityError("resolved basket package has no components")
        if components["asset_id"].duplicated().any():
            raise BasketInputIntegrityError("resolved basket package has duplicate components")
        components = components.sort_values("asset_id")
        assets = tuple(
            BasketAssetMarket(
                asset_id=str(row["asset_id"]),
                spot=float(row["spot"]),
                volatility=float(row["volatility"]),
                dividend_yield=float(row["dividend_yield"]),
            )
            for _, row in components.iterrows()
        )
        asset_ids = tuple(asset.asset_id for asset in assets)
        expected_pairs = {
            (left, right)
            for index, left in enumerate(asset_ids)
            for right in asset_ids[index + 1 :]
        }
        pair_values: dict[tuple[str, str], float] = {}
        for _, row in correlations.iterrows():
            pair = (str(row["asset_id"]), str(row["paired_asset_id"]))
            if pair in pair_values:
                raise BasketInputIntegrityError(
                    "resolved basket package has duplicate correlation pairs"
                )
            pair_values[pair] = float(row["correlation"])
        if set(pair_values) != expected_pairs:
            raise BasketInputIntegrityError(
                "resolved basket package does not contain the complete pairwise correlation set"
            )
        matrix = np.eye(len(assets), dtype=float)
        asset_position = {asset_id: index for index, asset_id in enumerate(asset_ids)}
        for (left, right), value in pair_values.items():
            left_index = asset_position[left]
            right_index = asset_position[right]
            matrix[left_index, right_index] = value
            matrix[right_index, left_index] = value
        try:
            market = BasketMarket(
                assets=assets,
                correlation=tuple(tuple(float(value) for value in row) for row in matrix),
            )
        except ValueError as exc:
            raise BasketInputIntegrityError(f"resolved basket package is invalid: {exc}") from exc
        execution = [
            {
                "asset_id": str(row["asset_id"]),
                "bid": float(row["bid"]),
                "ask": float(row["ask"]),
            }
            for _, row in components.iterrows()
        ]
        execution_json = _canonical_json(execution)
        ordered = package.sort_values("source_record_id")
        source_record_ids = tuple(str(value) for value in ordered["source_record_id"])
        payload_hashes = tuple(str(value) for value in ordered["payload_sha256"])
        manifest_rows = package[["batch_sequence", "batch_id", "manifest_hash"]].drop_duplicates()
        manifest_rows = manifest_rows.sort_values("batch_sequence")
        batch_ids = tuple(str(value) for value in manifest_rows["batch_id"])
        manifest_hashes = tuple(str(value) for value in manifest_rows["manifest_hash"])
        latest_dataset = package["ingested_timestamp"].max()
        snapshot_payload = {
            "basket_id": basket_id,
            "source": selected_source,
            "snapshot_id": snapshot_id,
            "formation_timestamp": formation.isoformat(),
            "market_timestamp": newest_market.isoformat(),
            "received_timestamp": received.isoformat(),
            "dataset_known_at": latest_dataset.isoformat(),
            "market_json": market.canonical_json,
            "execution_json": execution_json,
            "source_record_ids": source_record_ids,
            "record_payload_sha256s": payload_hashes,
            "batch_ids": batch_ids,
            "manifest_hashes": manifest_hashes,
        }
        return BasketInputSnapshot(
            basket_id=basket_id,
            source=selected_source,
            snapshot_id=snapshot_id,
            formation_timestamp=formation.isoformat(),
            market_timestamp=newest_market.isoformat(),
            received_timestamp=received.isoformat(),
            dataset_known_at=latest_dataset.isoformat(),
            market=market,
            execution_json=execution_json,
            record_count=len(package),
            source_record_ids=source_record_ids,
            record_payload_sha256s=payload_hashes,
            batch_ids=batch_ids,
            manifest_hashes=manifest_hashes,
            snapshot_sha256=_sha256(_canonical_json(snapshot_payload)),
        )

    def materialize_replay(
        self,
        schedule: pd.DataFrame,
        *,
        basket_id: str,
        source: str | None = None,
        maximum_age: str | pd.Timedelta = "15min",
        dataset_known_at: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Resolve every schedule row into a provenance-bound basket replay table."""
        from volforge.basket_replay import (
            BASKET_REPLAY_COLUMNS,
            BASKET_REPLAY_PROVENANCE_COLUMNS,
        )

        required = {"formation_timestamp", "option_bid", "option_ask", "risk_free_rate"}
        missing = sorted(required - set(schedule.columns))
        if missing:
            raise BasketInputSchemaError(
                "missing basket replay schedule columns: " + ", ".join(missing)
            )
        verification = self.verify()
        if not verification.valid:
            raise BasketInputIntegrityError(
                "cannot materialize replay from an invalid basket-input store: "
                + "; ".join(verification.failure_reasons)
            )
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for position, (_, row) in enumerate(schedule.iterrows()):
            snapshot = self.resolve_snapshot(
                basket_id,
                row["formation_timestamp"],
                source=source,
                maximum_age=maximum_age,
                dataset_known_at=dataset_known_at,
                _store_verified=True,
            )
            source_key = (snapshot.source, snapshot.snapshot_id)
            if source_key in seen:
                raise BasketInputIntegrityError(
                    f"schedule row {position} would reuse basket snapshot {snapshot.snapshot_id!r}"
                )
            seen.add(source_key)
            rows.append(
                {
                    "formation_timestamp": snapshot.formation_timestamp,
                    "market_timestamp": snapshot.market_timestamp,
                    "received_timestamp": snapshot.received_timestamp,
                    "source": snapshot.source,
                    "source_record_id": snapshot.snapshot_id,
                    "option_bid": _finite(row["option_bid"], f"row {position} option_bid"),
                    "option_ask": _finite(row["option_ask"], f"row {position} option_ask"),
                    "risk_free_rate": _finite(
                        row["risk_free_rate"], f"row {position} risk_free_rate"
                    ),
                    "market_json": snapshot.market_json,
                    "execution_json": snapshot.execution_json,
                    "input_store_snapshot_sha256": snapshot.snapshot_sha256,
                    "input_manifest_hashes_json": _canonical_json(snapshot.manifest_hashes),
                }
            )
        return pd.DataFrame(
            rows,
            columns=(*BASKET_REPLAY_COLUMNS, *BASKET_REPLAY_PROVENANCE_COLUMNS),
        )

    def verify(self) -> BasketInputVerification:
        """Verify SQLite integrity, raw artifacts, record payloads, and manifest chain."""
        failures: list[str] = []
        with self._connection() as connection:
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            if integrity != "ok":
                failures.append(f"SQLite integrity check failed: {integrity}")
            batch_rows = connection.execute(
                "SELECT * FROM basket_input_batches ORDER BY batch_sequence"
            ).fetchall()
            record_rows = connection.execute(
                "SELECT * FROM basket_input_records ORDER BY record_sequence"
            ).fetchall()
        previous_hash: str | None = None
        records_by_batch: dict[str, list[sqlite3.Row]] = {}
        for row in record_rows:
            records_by_batch.setdefault(str(row["batch_id"]), []).append(row)
            payload = {column: row[column] for column in _RECORD_PAYLOAD_COLUMNS}
            canonical = _canonical_json(payload)
            if canonical != row["payload_json"]:
                failures.append(
                    f"record {row['source']}/{row['source_record_id']}/{row['revision']} payload mismatch"
                )
            if _sha256(str(row["payload_json"])) != row["payload_sha256"]:
                failures.append(
                    f"record {row['source']}/{row['source_record_id']}/{row['revision']} hash mismatch"
                )
        for row in batch_rows:
            batch = dict(row)
            if row["previous_manifest_hash"] != previous_hash:
                failures.append(f"batch {row['batch_sequence']} predecessor hash mismatch")
            expected_manifest = _sha256(_canonical_json(_manifest_fields(batch)))
            if row["manifest_hash"] != expected_manifest or row["batch_id"] != expected_manifest:
                failures.append(f"batch {row['batch_sequence']} manifest hash mismatch")
            archive = self.raw_directory / str(row["raw_archive_name"])
            if not archive.is_file():
                failures.append(f"batch {row['batch_sequence']} raw archive is missing")
            else:
                raw = archive.read_bytes()
                if len(raw) != int(row["source_size_bytes"]):
                    failures.append(f"batch {row['batch_sequence']} raw archive size mismatch")
                if _sha256(raw) != row["source_sha256"]:
                    failures.append(f"batch {row['batch_sequence']} raw archive hash mismatch")
            batch_records = records_by_batch.get(str(row["batch_id"]), [])
            component_count = sum(item["record_type"] == "component" for item in batch_records)
            correlation_count = sum(item["record_type"] == "correlation" for item in batch_records)
            if (
                len(batch_records) != int(row["row_count"])
                or component_count != int(row["component_count"])
                or correlation_count != int(row["correlation_count"])
            ):
                failures.append(f"batch {row['batch_sequence']} record counts mismatch")
            if batch_records:
                market_times = [str(item["market_timestamp"]) for item in batch_records]
                if (
                    min(market_times) != row["minimum_market_timestamp"]
                    or max(market_times) != row["maximum_market_timestamp"]
                ):
                    failures.append(f"batch {row['batch_sequence']} market-time range mismatch")
            previous_hash = str(row["manifest_hash"])
        known_archives = {str(row["raw_archive_name"]) for row in batch_rows}
        actual_archives = {path.name for path in self.raw_directory.iterdir() if path.is_file()}
        if actual_archives != known_archives:
            failures.append("raw archive directory contains missing or unreferenced artifacts")
        return BasketInputVerification(
            valid=not failures,
            schema_version=BASKET_INPUT_SCHEMA_VERSION,
            batch_count=len(batch_rows),
            record_count=len(record_rows),
            final_manifest_hash=previous_hash,
            failure_reasons=tuple(failures),
        )


def generate_synthetic_basket_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate deterministic canonical input records and a quote/rate replay schedule."""
    from volforge.basket_replay import generate_synthetic_basket_replay

    replay = generate_synthetic_basket_replay()
    records: list[dict[str, Any]] = []
    schedule = replay[["formation_timestamp", "option_bid", "option_ask", "risk_free_rate"]].copy()
    for _, row in replay.iterrows():
        snapshot_id = str(row["source_record_id"])
        market = parse_basket_market(row["market_json"])
        execution = {
            str(item["asset_id"]): item
            for item in json.loads(str(row["execution_json"]), object_pairs_hook=_strict_object)
        }
        for asset in market.assets:
            quote = execution[asset.asset_id]
            records.append(
                {
                    "record_type": "component",
                    "snapshot_id": snapshot_id,
                    "market_timestamp": row["market_timestamp"],
                    "received_timestamp": row["received_timestamp"],
                    "source_record_id": f"{snapshot_id}:component:{asset.asset_id}",
                    "revision": 0,
                    "record_action": "upsert",
                    "basket_id": SYNTHETIC_BASKET_ID,
                    "asset_id": asset.asset_id,
                    "paired_asset_id": None,
                    "spot": asset.spot,
                    "volatility": asset.volatility,
                    "dividend_yield": asset.dividend_yield,
                    "bid": quote["bid"],
                    "ask": quote["ask"],
                    "correlation": None,
                }
            )
        for left_index, left in enumerate(market.asset_ids):
            for right_index in range(left_index + 1, len(market.asset_ids)):
                right = market.asset_ids[right_index]
                records.append(
                    {
                        "record_type": "correlation",
                        "snapshot_id": snapshot_id,
                        "market_timestamp": row["market_timestamp"],
                        "received_timestamp": row["received_timestamp"],
                        "source_record_id": f"{snapshot_id}:correlation:{left}:{right}",
                        "revision": 0,
                        "record_action": "upsert",
                        "basket_id": SYNTHETIC_BASKET_ID,
                        "asset_id": left,
                        "paired_asset_id": right,
                        "spot": None,
                        "volatility": None,
                        "dividend_yield": None,
                        "bid": None,
                        "ask": None,
                        "correlation": market.correlation[left_index][right_index],
                    }
                )
    return pd.DataFrame(records, columns=BASKET_INPUT_COLUMNS), schedule


def basket_input_report_json(
    value: BasketInputBatch | BasketInputSnapshot | BasketInputVerification,
) -> bytes:
    """Serialize a deterministic basket-input batch, snapshot, or verification report."""
    payload = value.to_dict() if isinstance(value, BasketInputSnapshot) else asdict(value)
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


__all__ = [
    "BASKET_INPUT_COLUMNS",
    "BASKET_INPUT_MODES",
    "BASKET_INPUT_RECORD_TYPES",
    "BASKET_INPUT_SCHEMA_VERSION",
    "MAX_BASKET_INPUT_ROWS",
    "SYNTHETIC_BASKET_ID",
    "BasketInputBatch",
    "BasketInputIntegrityError",
    "BasketInputSchemaError",
    "BasketInputSnapshot",
    "BasketInputStore",
    "BasketInputVerification",
    "basket_input_report_json",
    "generate_synthetic_basket_inputs",
]
