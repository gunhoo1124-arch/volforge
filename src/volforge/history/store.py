"""Append-only, revision-aware storage for historical option market events."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from io import BytesIO
from itertools import pairwise
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from volforge.reference import ReferenceDataStore

HISTORY_SCHEMA_VERSION = 1
INGESTION_MODES = ("initial", "incremental", "backfill", "correction")
RECORD_TYPES = ("quote", "trade")
RECORD_ACTIONS = ("upsert", "cancel")

COMMON_COLUMNS = (
    "record_type",
    "event_timestamp",
    "received_timestamp",
    "source_record_id",
    "revision",
    "record_action",
    "underlying_symbol",
    "expiration",
    "strike",
    "option_type",
)
QUOTE_COLUMNS = (
    "bid",
    "ask",
    "underlying_spot",
    "risk_free_rate",
    "dividend_yield",
)
TRADE_COLUMNS = ("trade_price", "trade_size")
OPTIONAL_COLUMNS = (
    "contract_symbol",
    "bid_size",
    "ask_size",
    "trade_condition",
    "exchange",
    "source_sequence",
)


class HistoricalSchemaError(ValueError):
    """Raised when a historical event file violates the normalized schema."""


class HistoricalIntegrityError(RuntimeError):
    """Raised when append-only, revision, backfill, or hash invariants fail."""


@dataclass(frozen=True, slots=True)
class IngestionBatch:
    """One append-only historical ingestion and its chained manifest."""

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
    quote_count: int
    trade_count: int
    minimum_event_timestamp: str
    maximum_event_timestamp: str
    reason: str | None
    previous_manifest_hash: str | None
    manifest_hash: str


@dataclass(frozen=True, slots=True)
class HistoryVerification:
    """Integrity status for the database, raw archive, records, and manifests."""

    valid: bool
    schema_version: int | None
    batch_count: int
    event_count: int
    archived_file_count: int
    final_manifest_hash: str | None
    failure_reasons: tuple[str, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_scalar(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_scalar(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    try:
        is_missing = pd.isna(value)
    except (TypeError, ValueError):
        is_missing = False
    if isinstance(is_missing, (bool, np.bool_)) and bool(is_missing):
        return None
    return str(value)


def _utc_timestamp(value: object, name: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        raise HistoricalSchemaError(f"{name} must be a valid timestamp")
    return pd.Timestamp(timestamp)


def _optional_float(row: pd.Series, name: str) -> float | None:
    if name not in row or pd.isna(row[name]):
        return None
    value = pd.to_numeric(row[name], errors="coerce")
    if pd.isna(value) or not math.isfinite(float(value)):
        raise HistoricalSchemaError(f"{name} must be finite when supplied")
    return float(value)


def _required_float(row: pd.Series, name: str) -> float:
    value = _optional_float(row, name)
    if value is None:
        raise HistoricalSchemaError(f"{name} is required")
    return value


def _optional_text(row: pd.Series, name: str) -> str | None:
    if name not in row or pd.isna(row[name]):
        return None
    value = str(row[name]).strip()
    return value or None


def _normalize_events(
    frame: pd.DataFrame,
    *,
    source: str,
    ingested_timestamp: pd.Timestamp,
) -> list[dict[str, Any]]:
    missing = [column for column in COMMON_COLUMNS if column not in frame.columns]
    if missing:
        raise HistoricalSchemaError(
            f"historical input is missing required columns: {', '.join(missing)}"
        )
    if frame.empty:
        raise HistoricalSchemaError("historical input must contain at least one event")

    records: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    for position, row in frame.reset_index(drop=True).iterrows():
        prefix = f"row {position}"
        record_type = str(row["record_type"]).strip().lower()
        action = str(row["record_action"]).strip().lower()
        source_record_id = str(row["source_record_id"]).strip()
        symbol = str(row["underlying_symbol"]).strip().upper()
        option_type = str(row["option_type"]).strip().lower()
        if record_type not in RECORD_TYPES:
            raise HistoricalSchemaError(f"{prefix}: record_type must be quote or trade")
        if action not in RECORD_ACTIONS:
            raise HistoricalSchemaError(f"{prefix}: record_action must be upsert or cancel")
        if not source_record_id:
            raise HistoricalSchemaError(f"{prefix}: source_record_id must not be empty")
        if not symbol:
            raise HistoricalSchemaError(f"{prefix}: underlying_symbol must not be empty")
        if option_type not in {"call", "put"}:
            raise HistoricalSchemaError(f"{prefix}: option_type must be call or put")

        revision_value = pd.to_numeric(row["revision"], errors="coerce")
        if (
            pd.isna(revision_value)
            or not float(revision_value).is_integer()
            or int(revision_value) < 0
        ):
            raise HistoricalSchemaError(f"{prefix}: revision must be a non-negative integer")
        revision = int(revision_value)
        identity = (source_record_id, revision)
        if identity in identities:
            raise HistoricalSchemaError(
                f"{prefix}: duplicate source_record_id/revision within the batch"
            )
        identities.add(identity)

        event_time = _utc_timestamp(row["event_timestamp"], f"{prefix} event_timestamp")
        received_time = _utc_timestamp(row["received_timestamp"], f"{prefix} received_timestamp")
        expiration = _utc_timestamp(row["expiration"], f"{prefix} expiration")
        if received_time < event_time:
            raise HistoricalSchemaError(
                f"{prefix}: received_timestamp cannot precede event_timestamp"
            )
        if received_time > ingested_timestamp:
            raise HistoricalSchemaError(
                f"{prefix}: received_timestamp cannot follow ingested_timestamp"
            )
        if expiration <= event_time:
            raise HistoricalSchemaError(f"{prefix}: expiration must follow event_timestamp")

        strike = _required_float(row, "strike")
        if strike <= 0.0:
            raise HistoricalSchemaError(f"{prefix}: strike must be positive")
        normalized: dict[str, Any] = {
            "source": source,
            "record_type": record_type,
            "event_timestamp": event_time.isoformat(),
            "received_timestamp": received_time.isoformat(),
            "source_record_id": source_record_id,
            "revision": revision,
            "record_action": action,
            "underlying_symbol": symbol,
            "contract_symbol": _optional_text(row, "contract_symbol"),
            "expiration": expiration.isoformat(),
            "strike": strike,
            "option_type": option_type,
            "bid": None,
            "ask": None,
            "bid_size": _optional_float(row, "bid_size"),
            "ask_size": _optional_float(row, "ask_size"),
            "trade_price": None,
            "trade_size": None,
            "trade_condition": _optional_text(row, "trade_condition"),
            "underlying_spot": _optional_float(row, "underlying_spot"),
            "risk_free_rate": _optional_float(row, "risk_free_rate"),
            "dividend_yield": _optional_float(row, "dividend_yield"),
            "exchange": _optional_text(row, "exchange"),
            "source_sequence": _optional_text(row, "source_sequence"),
        }
        if action == "upsert" and record_type == "quote":
            missing_quote = [column for column in QUOTE_COLUMNS if column not in frame.columns]
            if missing_quote:
                raise HistoricalSchemaError(
                    f"quote input is missing columns: {', '.join(missing_quote)}"
                )
            bid = _required_float(row, "bid")
            ask = _required_float(row, "ask")
            spot = _required_float(row, "underlying_spot")
            rate = _required_float(row, "risk_free_rate")
            dividend = _required_float(row, "dividend_yield")
            if bid < 0.0 or ask < 0.0 or bid > ask:
                raise HistoricalSchemaError(f"{prefix}: quote must satisfy 0 <= bid <= ask")
            if spot <= 0.0:
                raise HistoricalSchemaError(f"{prefix}: underlying_spot must be positive")
            for size_name in ("bid_size", "ask_size"):
                size = normalized[size_name]
                if size is not None and size < 0.0:
                    raise HistoricalSchemaError(f"{prefix}: {size_name} must be non-negative")
            normalized.update(
                {
                    "bid": bid,
                    "ask": ask,
                    "underlying_spot": spot,
                    "risk_free_rate": rate,
                    "dividend_yield": dividend,
                }
            )
        elif action == "upsert" and record_type == "trade":
            missing_trade = [column for column in TRADE_COLUMNS if column not in frame.columns]
            if missing_trade:
                raise HistoricalSchemaError(
                    f"trade input is missing columns: {', '.join(missing_trade)}"
                )
            trade_price = _required_float(row, "trade_price")
            trade_size = _required_float(row, "trade_size")
            if trade_price < 0.0 or trade_size <= 0.0:
                raise HistoricalSchemaError(
                    f"{prefix}: trade_price must be non-negative and trade_size positive"
                )
            normalized.update({"trade_price": trade_price, "trade_size": trade_size})

        raw_payload = {str(name): _json_scalar(value) for name, value in row.items()}
        payload_json = _canonical_json({"normalized": normalized, "source_row": raw_payload})
        normalized["payload_json"] = payload_json
        normalized["payload_sha256"] = _sha256(payload_json.encode("utf-8"))
        records.append(normalized)
    return records


def _manifest_fields(batch: dict[str, Any]) -> dict[str, Any]:
    names = (
        "schema_version",
        "source",
        "mode",
        "ingested_timestamp",
        "source_file_name",
        "source_sha256",
        "source_size_bytes",
        "raw_archive_name",
        "row_count",
        "quote_count",
        "trade_count",
        "minimum_event_timestamp",
        "maximum_event_timestamp",
        "reason",
        "previous_manifest_hash",
    )
    return {name: batch[name] for name in names}


class HistoricalEventStore:
    """SQLite-backed append-only store with bitemporal revision queries."""

    def __init__(self, path: str | PathLike[str]) -> None:
        self.path = Path(path).resolve()
        self.raw_directory = self.path.with_suffix(self.path.suffix + ".raw")
        if not self.path.is_file():
            raise FileNotFoundError(f"historical store does not exist: {self.path}")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
        if row is None or int(row[0]) != HISTORY_SCHEMA_VERSION:
            raise HistoricalIntegrityError("unsupported or missing historical schema version")

    @classmethod
    def create(cls, path: str | PathLike[str]) -> HistoricalEventStore:
        """Create a new store without overwriting any existing path."""
        resolved = Path(path).resolve()
        if resolved.exists():
            raise FileExistsError(f"refusing to overwrite existing historical store: {resolved}")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(resolved)
        try:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE ingestion_batches (
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
                    quote_count INTEGER NOT NULL,
                    trade_count INTEGER NOT NULL,
                    minimum_event_timestamp TEXT NOT NULL,
                    maximum_event_timestamp TEXT NOT NULL,
                    reason TEXT,
                    previous_manifest_hash TEXT,
                    manifest_hash TEXT NOT NULL UNIQUE
                );
                CREATE TABLE market_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL REFERENCES ingestion_batches(batch_id),
                    source TEXT NOT NULL,
                    record_type TEXT NOT NULL,
                    event_timestamp TEXT NOT NULL,
                    received_timestamp TEXT NOT NULL,
                    ingested_timestamp TEXT NOT NULL,
                    source_record_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    record_action TEXT NOT NULL,
                    underlying_symbol TEXT NOT NULL,
                    contract_symbol TEXT,
                    expiration TEXT NOT NULL,
                    strike REAL NOT NULL,
                    option_type TEXT NOT NULL,
                    bid REAL,
                    ask REAL,
                    bid_size REAL,
                    ask_size REAL,
                    trade_price REAL,
                    trade_size REAL,
                    trade_condition TEXT,
                    underlying_spot REAL,
                    risk_free_rate REAL,
                    dividend_yield REAL,
                    exchange TEXT,
                    source_sequence TEXT,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    UNIQUE(source, source_record_id, revision)
                );
                CREATE INDEX idx_events_event_time
                    ON market_events(record_type, event_timestamp);
                CREATE INDEX idx_events_knowledge_time
                    ON market_events(received_timestamp, ingested_timestamp);
                CREATE INDEX idx_events_contract
                    ON market_events(source, underlying_symbol, expiration, strike, option_type);
                CREATE TRIGGER batches_no_update BEFORE UPDATE ON ingestion_batches
                BEGIN SELECT RAISE(ABORT, 'ingestion_batches is append-only'); END;
                CREATE TRIGGER batches_no_delete BEFORE DELETE ON ingestion_batches
                BEGIN SELECT RAISE(ABORT, 'ingestion_batches is append-only'); END;
                CREATE TRIGGER events_no_update BEFORE UPDATE ON market_events
                BEGIN SELECT RAISE(ABORT, 'market_events is append-only'); END;
                CREATE TRIGGER events_no_delete BEFORE DELETE ON market_events
                BEGIN SELECT RAISE(ABORT, 'market_events is append-only'); END;
                """
            )
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
                (str(HISTORY_SCHEMA_VERSION),),
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
    ) -> IngestionBatch:
        """Validate, archive, hash, and atomically append a CSV ingestion."""
        path = Path(source_path).resolve()
        try:
            raw_bytes = path.read_bytes()
            frame = pd.read_csv(BytesIO(raw_bytes))
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            raise HistoricalSchemaError(f"could not read historical CSV: {exc}") from exc
        return self._ingest(
            frame,
            raw_bytes=raw_bytes,
            source_file_name=path.name,
            source=source,
            mode=mode,
            ingested_timestamp=ingested_timestamp,
            reason=reason,
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
    ) -> IngestionBatch:
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
    ) -> IngestionBatch:
        source_name = source.strip()
        if not source_name:
            raise ValueError("source must not be empty")
        if mode not in INGESTION_MODES:
            raise ValueError(f"mode must be one of {', '.join(INGESTION_MODES)}")
        if not source_file_name:
            raise ValueError("source_file_name must not be empty")
        reason_text = reason.strip() if reason is not None else None
        if mode in {"backfill", "correction"} and not reason_text:
            raise ValueError(f"{mode} ingestion requires a non-empty reason")
        ingested_at = _utc_timestamp(ingested_timestamp, "ingested_timestamp")
        records = _normalize_events(
            frame,
            source=source_name,
            ingested_timestamp=ingested_at,
        )
        event_times = [record["event_timestamp"] for record in records]
        source_digest = _sha256(raw_bytes)
        archive_name = f"{source_digest}.csv"
        archive_path = self.raw_directory / archive_name

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            duplicate_batch = connection.execute(
                "SELECT 1 FROM ingestion_batches WHERE source = ? AND source_sha256 = ?",
                (source_name, source_digest),
            ).fetchone()
            if duplicate_batch is not None:
                raise HistoricalIntegrityError("this exact source file was already ingested")
            global_latest = connection.execute(
                "SELECT MAX(ingested_timestamp) FROM ingestion_batches"
            ).fetchone()[0]
            if global_latest is not None and ingested_at.isoformat() < global_latest:
                raise HistoricalIntegrityError(
                    "ingested_timestamp cannot precede the store's latest ingestion"
                )
            source_state = connection.execute(
                """
                SELECT COUNT(*) AS count, MAX(event_timestamp) AS maximum_event_timestamp
                FROM market_events WHERE source = ?
                """,
                (source_name,),
            ).fetchone()
            source_exists = int(source_state["count"]) > 0
            if mode == "initial" and source_exists:
                raise HistoricalIntegrityError("initial mode requires an empty source history")
            if mode != "initial" and not source_exists:
                raise HistoricalIntegrityError(f"{mode} mode requires an existing source history")
            minimum_event = min(event_times)
            if (
                mode == "incremental"
                and source_state["maximum_event_timestamp"] is not None
                and minimum_event < source_state["maximum_event_timestamp"]
            ):
                raise HistoricalIntegrityError(
                    "incremental ingestion would silently backfill older events; use backfill mode"
                )

            identifiers = sorted({record["source_record_id"] for record in records})
            existing_records: dict[str, sqlite3.Row] = {}
            if identifiers:
                placeholders = ",".join("?" for _ in identifiers)
                rows = connection.execute(
                    f"""
                    WITH ranked AS (
                        SELECT source_record_id, revision, record_type, record_action,
                               ROW_NUMBER() OVER (
                                   PARTITION BY source_record_id ORDER BY revision DESC
                               ) AS revision_rank
                        FROM market_events
                        WHERE source = ? AND source_record_id IN ({placeholders})
                    )
                    SELECT source_record_id, revision, record_type, record_action
                    FROM ranked WHERE revision_rank = 1
                    """,
                    (source_name, *identifiers),
                ).fetchall()
                existing_records = {str(row["source_record_id"]): row for row in rows}
            if mode == "correction":
                new_identifiers = sorted(set(identifiers).difference(existing_records))
                if new_identifiers:
                    raise HistoricalIntegrityError(
                        "correction mode requires every source_record_id to exist already"
                    )
            elif existing_records:
                raise HistoricalIntegrityError(
                    "existing source_record_id requires explicit correction mode"
                )

            grouped: dict[str, list[dict[str, Any]]] = {}
            for record in records:
                grouped.setdefault(record["source_record_id"], []).append(record)
            for identifier, revisions in grouped.items():
                revisions.sort(key=lambda record: int(record["revision"]))
                existing = existing_records.get(identifier)
                previous_revision = int(existing["revision"]) if existing is not None else -1
                active_record_type = str(existing["record_type"]) if existing is not None else None
                has_active_record = existing is not None and existing["record_action"] == "upsert"
                for record in revisions:
                    revision = int(record["revision"])
                    if revision <= previous_revision:
                        raise HistoricalIntegrityError(
                            f"revision for {identifier!r} must increase strictly"
                        )
                    if (
                        active_record_type is not None
                        and record["record_type"] != active_record_type
                    ):
                        raise HistoricalIntegrityError(
                            f"record_type for {identifier!r} cannot change across revisions"
                        )
                    if record["record_action"] == "cancel" and not has_active_record:
                        raise HistoricalIntegrityError(
                            f"cannot cancel {identifier!r} without an active prior upsert"
                        )
                    previous_revision = revision
                    active_record_type = record["record_type"]
                    has_active_record = record["record_action"] == "upsert"

            if archive_path.exists():
                if _sha256(archive_path.read_bytes()) != source_digest:
                    raise HistoricalIntegrityError(
                        "existing raw archive content does not match its name"
                    )
            else:
                archive_path.write_bytes(raw_bytes)

            previous_row = connection.execute(
                """
                SELECT manifest_hash FROM ingestion_batches
                ORDER BY batch_sequence DESC LIMIT 1
                """
            ).fetchone()
            previous_hash = str(previous_row[0]) if previous_row is not None else None
            batch_fields: dict[str, Any] = {
                "schema_version": HISTORY_SCHEMA_VERSION,
                "source": source_name,
                "mode": mode,
                "ingested_timestamp": ingested_at.isoformat(),
                "source_file_name": source_file_name,
                "source_sha256": source_digest,
                "source_size_bytes": len(raw_bytes),
                "raw_archive_name": archive_name,
                "row_count": len(records),
                "quote_count": sum(record["record_type"] == "quote" for record in records),
                "trade_count": sum(record["record_type"] == "trade" for record in records),
                "minimum_event_timestamp": minimum_event,
                "maximum_event_timestamp": max(event_times),
                "reason": reason_text,
                "previous_manifest_hash": previous_hash,
            }
            manifest_hash = _sha256(_canonical_json(_manifest_fields(batch_fields)).encode("utf-8"))
            batch_id = manifest_hash
            connection.execute(
                """
                INSERT INTO ingestion_batches(
                    batch_id, schema_version, source, mode, ingested_timestamp,
                    source_file_name, source_sha256, source_size_bytes, raw_archive_name,
                    row_count, quote_count, trade_count, minimum_event_timestamp,
                    maximum_event_timestamp, reason, previous_manifest_hash, manifest_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    HISTORY_SCHEMA_VERSION,
                    source_name,
                    mode,
                    ingested_at.isoformat(),
                    source_file_name,
                    source_digest,
                    len(raw_bytes),
                    archive_name,
                    len(records),
                    batch_fields["quote_count"],
                    batch_fields["trade_count"],
                    minimum_event,
                    max(event_times),
                    reason_text,
                    previous_hash,
                    manifest_hash,
                ),
            )
            columns = (
                "source",
                "record_type",
                "event_timestamp",
                "received_timestamp",
                "source_record_id",
                "revision",
                "record_action",
                "underlying_symbol",
                "contract_symbol",
                "expiration",
                "strike",
                "option_type",
                "bid",
                "ask",
                "bid_size",
                "ask_size",
                "trade_price",
                "trade_size",
                "trade_condition",
                "underlying_spot",
                "risk_free_rate",
                "dividend_yield",
                "exchange",
                "source_sequence",
                "payload_json",
                "payload_sha256",
            )
            placeholders = ", ".join("?" for _ in range(len(columns) + 2))
            for record in records:
                connection.execute(
                    f"""
                    INSERT INTO market_events(
                        batch_id, ingested_timestamp, {", ".join(columns)}
                    ) VALUES ({placeholders})
                    """,
                    (
                        batch_id,
                        ingested_at.isoformat(),
                        *(record[column] for column in columns),
                    ),
                )
            sequence = int(
                connection.execute(
                    "SELECT batch_sequence FROM ingestion_batches WHERE batch_id = ?",
                    (batch_id,),
                ).fetchone()[0]
            )
            connection.commit()
        return IngestionBatch(
            batch_sequence=sequence,
            batch_id=batch_id,
            schema_version=HISTORY_SCHEMA_VERSION,
            source=source_name,
            mode=mode,
            ingested_timestamp=ingested_at.isoformat(),
            source_file_name=source_file_name,
            source_sha256=source_digest,
            source_size_bytes=len(raw_bytes),
            raw_archive_name=archive_name,
            row_count=len(records),
            quote_count=int(batch_fields["quote_count"]),
            trade_count=int(batch_fields["trade_count"]),
            minimum_event_timestamp=minimum_event,
            maximum_event_timestamp=max(event_times),
            reason=reason_text,
            previous_manifest_hash=previous_hash,
            manifest_hash=manifest_hash,
        )

    def batches(self) -> tuple[IngestionBatch, ...]:
        """Return ingestion manifests in append order."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM ingestion_batches ORDER BY batch_sequence"
            ).fetchall()
        return tuple(
            IngestionBatch(**{field: row[field] for field in IngestionBatch.__dataclass_fields__})
            for row in rows
        )

    def events_as_known(
        self,
        record_type: Literal["quote", "trade"],
        *,
        event_start: str | pd.Timestamp | None = None,
        event_end: str | pd.Timestamp | None = None,
        received_at: str | pd.Timestamp | None = None,
        ingested_at: str | pd.Timestamp | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Return latest active revisions under explicit market and dataset cutoffs."""
        if record_type not in RECORD_TYPES:
            raise ValueError("record_type must be quote or trade")
        event_start_iso = (
            _utc_timestamp(event_start, "event_start").isoformat()
            if event_start is not None
            else None
        )
        event_end_iso = (
            _utc_timestamp(event_end, "event_end").isoformat() if event_end is not None else None
        )
        if (
            event_start_iso is not None
            and event_end_iso is not None
            and event_start_iso > event_end_iso
        ):
            raise ValueError("event_start cannot follow event_end")
        with self._connection() as connection:
            latest_ingested = connection.execute(
                "SELECT MAX(ingested_timestamp) FROM ingestion_batches"
            ).fetchone()[0]
            if latest_ingested is None:
                return pd.DataFrame()
            received_cutoff = (
                _utc_timestamp(received_at, "received_at").isoformat()
                if received_at is not None
                else "9999-12-31T23:59:59+00:00"
            )
            ingested_cutoff = (
                _utc_timestamp(ingested_at, "ingested_at").isoformat()
                if ingested_at is not None
                else str(latest_ingested)
            )
            conditions = ["record_type = ?", "received_timestamp <= ?", "ingested_timestamp <= ?"]
            parameters: list[Any] = [record_type, received_cutoff, ingested_cutoff]
            if source is not None:
                conditions.append("source = ?")
                parameters.append(source)
            outer_conditions = ["revision_rank = 1", "record_action = 'upsert'"]
            if event_start_iso is not None:
                outer_conditions.append("event_timestamp >= ?")
                parameters.append(event_start_iso)
            if event_end_iso is not None:
                outer_conditions.append("event_timestamp <= ?")
                parameters.append(event_end_iso)
            query = f"""
                WITH eligible AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY source, source_record_id
                        ORDER BY revision DESC, received_timestamp DESC, ingested_timestamp DESC
                    ) AS revision_rank
                    FROM market_events
                    WHERE {" AND ".join(conditions)}
                )
                SELECT * FROM eligible
                WHERE {" AND ".join(outer_conditions)}
                ORDER BY event_timestamp, source, source_record_id
            """
            frame = pd.read_sql_query(query, connection, params=parameters)
        for column in ("event_timestamp", "received_timestamp", "ingested_timestamp", "expiration"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
        return frame.drop(columns=["revision_rank"], errors="ignore")

    def chain_snapshot(
        self,
        snapshot_time: str | pd.Timestamp,
        *,
        source: str | None = None,
        maximum_quote_age: str | pd.Timedelta = "15min",
        dataset_known_at: str | pd.Timestamp | None = None,
        reference_store: ReferenceDataStore | None = None,
        reference_ingested_at: str | pd.Timestamp | None = None,
        require_reference: bool = False,
    ) -> pd.DataFrame:
        """Build a chain from received quotes and optionally require reference validation."""
        snapshot = _utc_timestamp(snapshot_time, "snapshot_time")
        quote_age = pd.to_timedelta(maximum_quote_age)
        if quote_age < pd.Timedelta(0):
            raise ValueError("maximum_quote_age must be non-negative")
        quotes = self.events_as_known(
            "quote",
            event_start=snapshot - quote_age,
            event_end=snapshot,
            received_at=snapshot,
            ingested_at=dataset_known_at,
            source=source,
        )
        if quotes.empty:
            return pd.DataFrame()
        sources = quotes["source"].drop_duplicates()
        if source is None and len(sources) > 1:
            raise ValueError("chain_snapshot requires source when multiple sources are present")
        contract_key = [
            "source",
            "underlying_symbol",
            "expiration",
            "strike",
            "option_type",
        ]
        quotes = (
            quotes.sort_values(
                [*contract_key, "event_timestamp", "received_timestamp", "source_record_id"]
            )
            .groupby(contract_key, sort=True, as_index=False)
            .tail(1)
            .sort_values(["expiration", "strike", "option_type"], ignore_index=True)
        )
        quotes["source_event_timestamp"] = quotes["event_timestamp"]
        quotes["quote_age_seconds"] = (
            snapshot - quotes["source_event_timestamp"]
        ).dt.total_seconds()
        quotes["quote_timestamp"] = snapshot
        if reference_store is None:
            if require_reference:
                raise HistoricalIntegrityError(
                    "a reference store is required for replay-ready snapshots"
                )
            quotes["reference_validated"] = False
        else:
            quotes = reference_store.enrich_snapshot(
                quotes,
                snapshot,
                reference_ingested_at=reference_ingested_at,
            )
        ordered = [
            "quote_timestamp",
            "expiration",
            "strike",
            "option_type",
            "bid",
            "ask",
            "underlying_spot",
            "risk_free_rate",
            "dividend_yield",
            "underlying_symbol",
            "contract_symbol",
            "source",
            "exchange",
            "source_record_id",
            "revision",
            "source_event_timestamp",
            "received_timestamp",
            "ingested_timestamp",
            "quote_age_seconds",
            "bid_size",
            "ask_size",
            "payload_sha256",
            "batch_id",
        ]
        reference_columns = [
            "stable_contract_id",
            "reference_product_id",
            "reference_source",
            "contract_multiplier",
            "deliverable_json",
            "exercise_style",
            "settlement_method",
            "settlement_currency",
            "settlement_style",
            "last_trade_timestamp",
            "settlement_timestamp",
            "expiration_rule",
            "last_trade_rule",
            "settlement_rule",
            "corporate_action_id",
            "reference_contract_revision",
            "reference_contract_payload_sha256",
            "reference_product_payload_sha256",
            "reference_action_payload_sha256",
            "event_session_date",
            "event_session_name",
            "formation_session_date",
            "formation_session_name",
            "session_timezone",
            "reference_session_payload_sha256",
            "reference_validated",
        ]
        return quotes.loc[
            :, [*ordered, *(column for column in reference_columns if column in quotes)]
        ]

    def replay_chain(
        self,
        snapshot_times: list[str | pd.Timestamp] | tuple[str | pd.Timestamp, ...],
        *,
        source: str | None = None,
        maximum_quote_age: str | pd.Timedelta = "15min",
        dataset_known_at: str | pd.Timestamp | None = None,
        reference_store: ReferenceDataStore | None = None,
        reference_ingested_at: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Build point-in-time snapshots gated by contract and session references."""
        if len(snapshot_times) < 2:
            raise ValueError("replay_chain requires at least two snapshot times")
        if reference_store is None:
            raise HistoricalIntegrityError(
                "replay_chain requires a reference store for contract/session validation"
            )
        normalized = [_utc_timestamp(value, "snapshot_time") for value in snapshot_times]
        if any(later <= earlier for earlier, later in pairwise(normalized)):
            raise ValueError("snapshot_times must be strictly increasing")
        snapshots = [
            self.chain_snapshot(
                timestamp,
                source=source,
                maximum_quote_age=maximum_quote_age,
                dataset_known_at=dataset_known_at,
                reference_store=reference_store,
                reference_ingested_at=reference_ingested_at,
                require_reference=True,
            )
            for timestamp in normalized
        ]
        empty_positions = [
            timestamp.isoformat()
            for timestamp, snapshot in zip(normalized, snapshots, strict=True)
            if snapshot.empty
        ]
        if empty_positions:
            raise HistoricalIntegrityError(
                "no eligible quotes for replay snapshot(s): " + ", ".join(empty_positions)
            )
        return pd.concat(snapshots, ignore_index=True)

    def verify(self) -> HistoryVerification:
        """Verify SQLite integrity, manifest linkage, raw files, and record hashes."""
        failures: list[str] = []
        batches = self.batches()
        archived: set[str] = set()
        previous_hash: str | None = None
        with self._connection() as connection:
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            if integrity != "ok":
                failures.append(f"SQLite integrity check failed: {integrity}")
            event_count = int(
                connection.execute("SELECT COUNT(*) FROM market_events").fetchone()[0]
            )
            for batch in batches:
                fields = asdict(batch)
                observed_hash = fields.pop("manifest_hash")
                fields.pop("batch_sequence")
                fields.pop("batch_id")
                expected_hash = _sha256(_canonical_json(_manifest_fields(fields)).encode("utf-8"))
                if batch.previous_manifest_hash != previous_hash:
                    failures.append(f"batch {batch.batch_sequence} has a broken manifest link")
                if observed_hash != expected_hash or batch.batch_id != observed_hash:
                    failures.append(f"batch {batch.batch_sequence} manifest hash mismatch")
                previous_hash = observed_hash
                count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM market_events WHERE batch_id = ?",
                        (batch.batch_id,),
                    ).fetchone()[0]
                )
                if count != batch.row_count:
                    failures.append(f"batch {batch.batch_sequence} row count mismatch")
                archive_path = self.raw_directory / batch.raw_archive_name
                archived.add(batch.raw_archive_name)
                if not archive_path.is_file():
                    failures.append(f"batch {batch.batch_sequence} raw archive is missing")
                else:
                    raw = archive_path.read_bytes()
                    if len(raw) != batch.source_size_bytes or _sha256(raw) != batch.source_sha256:
                        failures.append(f"batch {batch.batch_sequence} raw archive hash mismatch")

            for row in connection.execute(
                "SELECT event_id, payload_json, payload_sha256 FROM market_events"
            ):
                expected = _sha256(str(row["payload_json"]).encode("utf-8"))
                if expected != row["payload_sha256"]:
                    failures.append(f"event {row['event_id']} payload hash mismatch")
        return HistoryVerification(
            valid=not failures,
            schema_version=HISTORY_SCHEMA_VERSION,
            batch_count=len(batches),
            event_count=event_count,
            archived_file_count=len(archived),
            final_manifest_hash=previous_hash,
            failure_reasons=tuple(failures),
        )


def history_report_json(value: IngestionBatch | HistoryVerification) -> bytes:
    """Serialize an ingestion or verification result deterministically."""
    return (json.dumps(asdict(value), indent=2, sort_keys=True) + "\n").encode("utf-8")
