"""Append-only, point-in-time reference data for listed option contracts."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

REFERENCE_SCHEMA_VERSION = 1
REFERENCE_KINDS = ("product", "contract", "session", "corporate_action")
REFERENCE_INGESTION_MODES = ("initial", "incremental", "backfill", "correction")
REFERENCE_ACTIONS = ("upsert", "cancel")

COMMON_COLUMNS = (
    "source_record_id",
    "revision",
    "record_action",
    "published_timestamp",
)
PRODUCT_COLUMNS = (
    "effective_from",
    "product_id",
    "product_symbol",
    "exchange",
    "calendar_id",
    "underlying_asset_type",
    "exercise_style",
    "settlement_method",
    "settlement_style",
    "currency",
    "default_multiplier",
    "expiration_rule",
    "last_trade_rule",
    "settlement_rule",
    "status",
    "source_reference",
)
CONTRACT_COLUMNS = (
    "effective_from",
    "contract_id",
    "market_data_source",
    "source_contract_symbol",
    "product_id",
    "underlying_id",
    "underlying_symbol",
    "option_type",
    "strike",
    "expiration_timestamp",
    "last_trade_timestamp",
    "settlement_timestamp",
    "contract_multiplier",
    "deliverable_json",
    "status",
    "source_reference",
)
SESSION_COLUMNS = (
    "calendar_id",
    "session_date",
    "session_name",
    "status",
    "timezone",
    "source_reference",
)
CORPORATE_ACTION_COLUMNS = (
    "corporate_action_id",
    "underlying_id",
    "action_type",
    "effective_timestamp",
    "description",
    "source_reference",
)

ReferenceKind = Literal["product", "contract", "session", "corporate_action"]
ReferenceMode = Literal["initial", "incremental", "backfill", "correction"]


class ReferenceSchemaError(ValueError):
    """Raised when a reference-data row violates the normalized schema."""


class ReferenceIntegrityError(RuntimeError):
    """Raised when reference history or resolution is incomplete or ambiguous."""


@dataclass(frozen=True, slots=True)
class ReferenceBatch:
    """One append-only reference ingestion and its chained manifest."""

    batch_sequence: int
    batch_id: str
    schema_version: int
    kind: str
    source: str
    mode: str
    ingested_timestamp: str
    source_file_name: str
    source_sha256: str
    source_size_bytes: int
    raw_archive_name: str
    row_count: int
    minimum_effective_timestamp: str
    maximum_effective_timestamp: str
    reason: str | None
    previous_manifest_hash: str | None
    manifest_hash: str


@dataclass(frozen=True, slots=True)
class ReferenceVerification:
    """Integrity status for the reference database and archived inputs."""

    valid: bool
    schema_version: int | None
    batch_count: int
    record_count: int
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
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _utc_timestamp(value: object, name: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        raise ReferenceSchemaError(f"{name} must be a valid timestamp")
    return pd.Timestamp(timestamp)


def _optional_timestamp(row: pd.Series, name: str, prefix: str) -> str | None:
    if name not in row or pd.isna(row[name]) or not str(row[name]).strip():
        return None
    return _utc_timestamp(row[name], f"{prefix} {name}").isoformat()


def _required_text(row: pd.Series, name: str, prefix: str) -> str:
    if name not in row or pd.isna(row[name]):
        raise ReferenceSchemaError(f"{prefix}: {name} is required")
    value = str(row[name]).strip()
    if not value:
        raise ReferenceSchemaError(f"{prefix}: {name} must not be empty")
    return value


def _optional_text(row: pd.Series, name: str) -> str | None:
    if name not in row or pd.isna(row[name]):
        return None
    value = str(row[name]).strip()
    return value or None


def _required_float(row: pd.Series, name: str, prefix: str) -> float:
    if name not in row:
        raise ReferenceSchemaError(f"{prefix}: {name} is required")
    value = pd.to_numeric(row[name], errors="coerce")
    if pd.isna(value) or not math.isfinite(float(value)):
        raise ReferenceSchemaError(f"{prefix}: {name} must be finite")
    return float(value)


def _enum(value: str, choices: set[str], name: str, prefix: str) -> str:
    normalized = value.strip().lower()
    if normalized not in choices:
        raise ReferenceSchemaError(f"{prefix}: {name} must be one of {', '.join(sorted(choices))}")
    return normalized


def _deliverable_json(value: object, prefix: str) -> str:
    try:
        deliverable = json.loads(str(value)) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ReferenceSchemaError(f"{prefix}: deliverable_json is not valid JSON") from exc
    if not isinstance(deliverable, list) or not deliverable:
        raise ReferenceSchemaError(f"{prefix}: deliverable_json must be a non-empty JSON list")
    normalized: list[dict[str, Any]] = []
    for position, component in enumerate(deliverable):
        if not isinstance(component, dict):
            raise ReferenceSchemaError(
                f"{prefix}: deliverable component {position} must be an object"
            )
        asset_id = str(component.get("asset_id", "")).strip()
        asset_type = str(component.get("asset_type", "")).strip().lower()
        quantity = pd.to_numeric(component.get("quantity"), errors="coerce")
        if not asset_id or asset_type not in {
            "security",
            "cash",
            "cash_in_lieu",
            "cash_settlement",
            "other",
        }:
            raise ReferenceSchemaError(f"{prefix}: each deliverable needs asset_id and asset_type")
        if pd.isna(quantity) or not math.isfinite(float(quantity)) or float(quantity) <= 0.0:
            raise ReferenceSchemaError(
                f"{prefix}: each deliverable quantity must be positive and finite"
            )
        normalized.append(
            {
                "asset_id": asset_id,
                "asset_type": asset_type,
                "quantity": float(quantity),
                "currency": (
                    str(component["currency"]).strip().upper()
                    if component.get("currency") is not None
                    else None
                ),
                "description": (
                    str(component["description"]).strip()
                    if component.get("description") is not None
                    else None
                ),
            }
        )
    return _canonical_json(normalized)


def _normalize_product(row: pd.Series, prefix: str) -> tuple[dict[str, Any], str]:
    effective_from = _utc_timestamp(
        _required_text(row, "effective_from", prefix), f"{prefix} effective_from"
    )
    effective_to = _optional_timestamp(row, "effective_to", prefix)
    if effective_to is not None and effective_to <= effective_from.isoformat():
        raise ReferenceSchemaError(f"{prefix}: effective_to must follow effective_from")
    multiplier = _required_float(row, "default_multiplier", prefix)
    if multiplier <= 0.0:
        raise ReferenceSchemaError(f"{prefix}: default_multiplier must be positive")
    normalized = {
        "effective_from": effective_from.isoformat(),
        "effective_to": effective_to,
        "product_id": _required_text(row, "product_id", prefix),
        "product_symbol": _required_text(row, "product_symbol", prefix).upper(),
        "exchange": _required_text(row, "exchange", prefix).upper(),
        "calendar_id": _required_text(row, "calendar_id", prefix),
        "underlying_asset_type": _enum(
            _required_text(row, "underlying_asset_type", prefix),
            {"equity", "etf", "index", "future", "other"},
            "underlying_asset_type",
            prefix,
        ),
        "exercise_style": _enum(
            _required_text(row, "exercise_style", prefix),
            {"american", "european"},
            "exercise_style",
            prefix,
        ),
        "settlement_method": _enum(
            _required_text(row, "settlement_method", prefix),
            {"cash", "physical"},
            "settlement_method",
            prefix,
        ),
        "settlement_style": _enum(
            _required_text(row, "settlement_style", prefix),
            {"am", "pm", "intraday"},
            "settlement_style",
            prefix,
        ),
        "currency": _required_text(row, "currency", prefix).upper(),
        "default_multiplier": multiplier,
        "expiration_rule": _required_text(row, "expiration_rule", prefix),
        "last_trade_rule": _required_text(row, "last_trade_rule", prefix),
        "settlement_rule": _required_text(row, "settlement_rule", prefix),
        "status": _enum(
            _required_text(row, "status", prefix),
            {"active", "inactive"},
            "status",
            prefix,
        ),
        "source_reference": _required_text(row, "source_reference", prefix),
    }
    return normalized, effective_from.isoformat()


def _normalize_contract(row: pd.Series, prefix: str) -> tuple[dict[str, Any], str]:
    effective_from = _utc_timestamp(
        _required_text(row, "effective_from", prefix), f"{prefix} effective_from"
    )
    effective_to = _optional_timestamp(row, "effective_to", prefix)
    if effective_to is not None and effective_to <= effective_from.isoformat():
        raise ReferenceSchemaError(f"{prefix}: effective_to must follow effective_from")
    expiration = _utc_timestamp(
        _required_text(row, "expiration_timestamp", prefix),
        f"{prefix} expiration_timestamp",
    )
    last_trade = _utc_timestamp(
        _required_text(row, "last_trade_timestamp", prefix),
        f"{prefix} last_trade_timestamp",
    )
    settlement = _utc_timestamp(
        _required_text(row, "settlement_timestamp", prefix),
        f"{prefix} settlement_timestamp",
    )
    if expiration <= effective_from:
        raise ReferenceSchemaError(f"{prefix}: expiration must follow effective_from")
    if last_trade > expiration:
        raise ReferenceSchemaError(f"{prefix}: last_trade_timestamp cannot follow expiration")
    if settlement < last_trade:
        raise ReferenceSchemaError(f"{prefix}: settlement_timestamp cannot precede last trade")
    strike = _required_float(row, "strike", prefix)
    multiplier = _required_float(row, "contract_multiplier", prefix)
    if strike <= 0.0 or multiplier <= 0.0:
        raise ReferenceSchemaError(f"{prefix}: strike and contract_multiplier must be positive")
    normalized = {
        "effective_from": effective_from.isoformat(),
        "effective_to": effective_to,
        "contract_id": _required_text(row, "contract_id", prefix),
        "market_data_source": _required_text(row, "market_data_source", prefix),
        "source_contract_symbol": _required_text(row, "source_contract_symbol", prefix),
        "product_id": _required_text(row, "product_id", prefix),
        "underlying_id": _required_text(row, "underlying_id", prefix),
        "underlying_symbol": _required_text(row, "underlying_symbol", prefix).upper(),
        "option_type": _enum(
            _required_text(row, "option_type", prefix),
            {"call", "put"},
            "option_type",
            prefix,
        ),
        "strike": strike,
        "expiration_timestamp": expiration.isoformat(),
        "last_trade_timestamp": last_trade.isoformat(),
        "settlement_timestamp": settlement.isoformat(),
        "contract_multiplier": multiplier,
        "deliverable_json": _deliverable_json(row["deliverable_json"], prefix),
        "status": _enum(
            _required_text(row, "status", prefix),
            {"active", "inactive", "expired", "delisted"},
            "status",
            prefix,
        ),
        "corporate_action_id": _optional_text(row, "corporate_action_id"),
        "source_reference": _required_text(row, "source_reference", prefix),
    }
    return normalized, effective_from.isoformat()


def _normalize_session(row: pd.Series, prefix: str) -> tuple[dict[str, Any], str]:
    session_date_text = _required_text(row, "session_date", prefix)
    try:
        session_date = date.fromisoformat(session_date_text)
    except ValueError as exc:
        raise ReferenceSchemaError(f"{prefix}: session_date must use YYYY-MM-DD") from exc
    timezone_name = _required_text(row, "timezone", prefix)
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ReferenceSchemaError(f"{prefix}: timezone must be a valid IANA name") from exc
    status = _enum(
        _required_text(row, "status", prefix),
        {"open", "closed"},
        "status",
        prefix,
    )
    open_timestamp = _optional_timestamp(row, "open_timestamp", prefix)
    close_timestamp = _optional_timestamp(row, "close_timestamp", prefix)
    if status == "open":
        if open_timestamp is None or close_timestamp is None:
            raise ReferenceSchemaError(
                f"{prefix}: open sessions require open_timestamp and close_timestamp"
            )
        if close_timestamp <= open_timestamp:
            raise ReferenceSchemaError(f"{prefix}: session close must follow open")
        effective_key = open_timestamp
    else:
        if open_timestamp is not None or close_timestamp is not None:
            raise ReferenceSchemaError(
                f"{prefix}: closed sessions cannot contain open or close timestamps"
            )
        effective_key = f"{session_date.isoformat()}T00:00:00+00:00"
    normalized = {
        "calendar_id": _required_text(row, "calendar_id", prefix),
        "session_date": session_date.isoformat(),
        "session_name": _required_text(row, "session_name", prefix).upper(),
        "status": status,
        "timezone": timezone_name,
        "open_timestamp": open_timestamp,
        "close_timestamp": close_timestamp,
        "reason": _optional_text(row, "reason"),
        "source_reference": _required_text(row, "source_reference", prefix),
    }
    return normalized, effective_key


def _normalize_corporate_action(row: pd.Series, prefix: str) -> tuple[dict[str, Any], str]:
    effective = _utc_timestamp(
        _required_text(row, "effective_timestamp", prefix),
        f"{prefix} effective_timestamp",
    )
    normalized = {
        "corporate_action_id": _required_text(row, "corporate_action_id", prefix),
        "underlying_id": _required_text(row, "underlying_id", prefix),
        "action_type": _enum(
            _required_text(row, "action_type", prefix),
            {
                "split",
                "reverse_split",
                "merger",
                "spin_off",
                "special_dividend",
                "symbol_change",
                "delisting",
                "other",
            },
            "action_type",
            prefix,
        ),
        "effective_timestamp": effective.isoformat(),
        "old_symbol": _optional_text(row, "old_symbol"),
        "new_symbol": _optional_text(row, "new_symbol"),
        "description": _required_text(row, "description", prefix),
        "source_reference": _required_text(row, "source_reference", prefix),
    }
    return normalized, effective.isoformat()


def _normalize_records(
    frame: pd.DataFrame,
    *,
    kind: str,
    source: str,
    ingested_timestamp: pd.Timestamp,
) -> list[dict[str, Any]]:
    kind_columns = {
        "product": PRODUCT_COLUMNS,
        "contract": CONTRACT_COLUMNS,
        "session": SESSION_COLUMNS,
        "corporate_action": CORPORATE_ACTION_COLUMNS,
    }[kind]
    missing = [column for column in (*COMMON_COLUMNS, *kind_columns) if column not in frame]
    if missing:
        raise ReferenceSchemaError(
            f"{kind} input is missing required columns: {', '.join(missing)}"
        )
    if frame.empty:
        raise ReferenceSchemaError("reference input must contain at least one record")

    normalizers = {
        "product": _normalize_product,
        "contract": _normalize_contract,
        "session": _normalize_session,
        "corporate_action": _normalize_corporate_action,
    }
    records: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    for position, row in frame.reset_index(drop=True).iterrows():
        prefix = f"row {position}"
        source_record_id = _required_text(row, "source_record_id", prefix)
        revision_value = pd.to_numeric(row["revision"], errors="coerce")
        if (
            pd.isna(revision_value)
            or not float(revision_value).is_integer()
            or int(revision_value) < 0
        ):
            raise ReferenceSchemaError(f"{prefix}: revision must be a non-negative integer")
        revision = int(revision_value)
        identity = (source_record_id, revision)
        if identity in identities:
            raise ReferenceSchemaError(
                f"{prefix}: duplicate source_record_id/revision within the batch"
            )
        identities.add(identity)
        action = _enum(
            _required_text(row, "record_action", prefix),
            set(REFERENCE_ACTIONS),
            "record_action",
            prefix,
        )
        published = _utc_timestamp(
            _required_text(row, "published_timestamp", prefix),
            f"{prefix} published_timestamp",
        )
        if published > ingested_timestamp:
            raise ReferenceSchemaError(
                f"{prefix}: published_timestamp cannot follow ingested_timestamp"
            )
        normalized, effective_key = normalizers[kind](row, prefix)
        common = {
            "kind": kind,
            "source": source,
            "source_record_id": source_record_id,
            "revision": revision,
            "record_action": action,
            "published_timestamp": published.isoformat(),
            "effective_key": effective_key,
            **normalized,
        }
        raw_payload = {str(name): _json_scalar(value) for name, value in row.items()}
        payload_json = _canonical_json({"normalized": common, "source_row": raw_payload})
        common["payload_json"] = payload_json
        common["payload_sha256"] = _sha256(payload_json.encode("utf-8"))
        records.append(common)
    return records


def _manifest_fields(batch: dict[str, Any]) -> dict[str, Any]:
    names = (
        "schema_version",
        "kind",
        "source",
        "mode",
        "ingested_timestamp",
        "source_file_name",
        "source_sha256",
        "source_size_bytes",
        "raw_archive_name",
        "row_count",
        "minimum_effective_timestamp",
        "maximum_effective_timestamp",
        "reason",
        "previous_manifest_hash",
    )
    return {name: batch[name] for name in names}


def _identity_fields(kind: str) -> tuple[str, ...]:
    return {
        "product": ("product_id",),
        "contract": ("contract_id", "effective_from"),
        "session": ("calendar_id", "session_date", "session_name"),
        "corporate_action": ("corporate_action_id",),
    }[kind]


class ReferenceDataStore:
    """SQLite-backed bitemporal reference master for listed options."""

    def __init__(self, path: str | PathLike[str]) -> None:
        self.path = Path(path).resolve()
        self.raw_directory = self.path.with_suffix(self.path.suffix + ".raw")
        if not self.path.is_file():
            raise FileNotFoundError(f"reference store does not exist: {self.path}")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
        if row is None or int(row[0]) != REFERENCE_SCHEMA_VERSION:
            raise ReferenceIntegrityError("unsupported or missing reference schema version")

    @classmethod
    def create(cls, path: str | PathLike[str]) -> ReferenceDataStore:
        """Create a new reference store without overwriting existing artifacts."""
        resolved = Path(path).resolve()
        raw_directory = resolved.with_suffix(resolved.suffix + ".raw")
        if resolved.exists() or raw_directory.exists():
            raise FileExistsError(f"refusing to overwrite existing reference store: {resolved}")
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
                CREATE TABLE reference_batches (
                    batch_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL UNIQUE,
                    schema_version INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    source TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    ingested_timestamp TEXT NOT NULL,
                    source_file_name TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    source_size_bytes INTEGER NOT NULL,
                    raw_archive_name TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    minimum_effective_timestamp TEXT NOT NULL,
                    maximum_effective_timestamp TEXT NOT NULL,
                    reason TEXT,
                    previous_manifest_hash TEXT,
                    manifest_hash TEXT NOT NULL UNIQUE
                );
                CREATE TABLE reference_records (
                    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL REFERENCES reference_batches(batch_id),
                    kind TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_record_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    record_action TEXT NOT NULL,
                    published_timestamp TEXT NOT NULL,
                    ingested_timestamp TEXT NOT NULL,
                    effective_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    UNIQUE(source, kind, source_record_id, revision)
                );
                CREATE INDEX idx_reference_knowledge
                    ON reference_records(kind, published_timestamp, ingested_timestamp);
                CREATE INDEX idx_reference_effective
                    ON reference_records(kind, source, effective_key);
                CREATE TRIGGER reference_batches_no_update
                    BEFORE UPDATE ON reference_batches
                    BEGIN SELECT RAISE(ABORT, 'reference_batches is append-only'); END;
                CREATE TRIGGER reference_batches_no_delete
                    BEFORE DELETE ON reference_batches
                    BEGIN SELECT RAISE(ABORT, 'reference_batches is append-only'); END;
                CREATE TRIGGER reference_records_no_update
                    BEFORE UPDATE ON reference_records
                    BEGIN SELECT RAISE(ABORT, 'reference_records is append-only'); END;
                CREATE TRIGGER reference_records_no_delete
                    BEFORE DELETE ON reference_records
                    BEGIN SELECT RAISE(ABORT, 'reference_records is append-only'); END;
                """
            )
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
                (str(REFERENCE_SCHEMA_VERSION),),
            )
            connection.commit()
        finally:
            connection.close()
        raw_directory.mkdir(parents=True, exist_ok=False)
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
        kind: ReferenceKind,
        source: str,
        mode: ReferenceMode,
        ingested_timestamp: str | pd.Timestamp,
        reason: str | None = None,
    ) -> ReferenceBatch:
        """Validate, archive, hash, and atomically append a reference CSV."""
        path = Path(source_path).resolve()
        try:
            raw_bytes = path.read_bytes()
            frame = pd.read_csv(BytesIO(raw_bytes))
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            raise ReferenceSchemaError(f"could not read reference CSV: {exc}") from exc
        return self._ingest(
            frame,
            raw_bytes=raw_bytes,
            source_file_name=path.name,
            kind=kind,
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
        kind: ReferenceKind,
        source: str,
        mode: ReferenceMode,
        ingested_timestamp: str | pd.Timestamp,
        reason: str | None = None,
    ) -> ReferenceBatch:
        """Append a frame using a deterministic CSV representation as its raw artifact."""
        raw_bytes = frame.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode(
            "utf-8"
        )
        return self._ingest(
            frame,
            raw_bytes=raw_bytes,
            source_file_name=Path(source_file_name).name,
            kind=kind,
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
        kind: str,
        source: str,
        mode: str,
        ingested_timestamp: str | pd.Timestamp,
        reason: str | None,
    ) -> ReferenceBatch:
        if kind not in REFERENCE_KINDS:
            raise ValueError(f"kind must be one of {', '.join(REFERENCE_KINDS)}")
        if mode not in REFERENCE_INGESTION_MODES:
            raise ValueError(f"mode must be one of {', '.join(REFERENCE_INGESTION_MODES)}")
        source_name = source.strip()
        if not source_name:
            raise ValueError("source must not be empty")
        if not source_file_name:
            raise ValueError("source_file_name must not be empty")
        reason_text = reason.strip() if reason is not None else None
        if mode in {"backfill", "correction"} and not reason_text:
            raise ValueError(f"{mode} ingestion requires a non-empty reason")
        ingested_at = _utc_timestamp(ingested_timestamp, "ingested_timestamp")
        records = _normalize_records(
            frame,
            kind=kind,
            source=source_name,
            ingested_timestamp=ingested_at,
        )
        effective_keys = [str(record["effective_key"]) for record in records]
        source_digest = _sha256(raw_bytes)
        archive_name = f"{source_digest}.csv"
        archive_path = self.raw_directory / archive_name

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            duplicate = connection.execute(
                """
                SELECT 1 FROM reference_batches
                WHERE kind = ? AND source = ? AND source_sha256 = ?
                """,
                (kind, source_name, source_digest),
            ).fetchone()
            if duplicate is not None:
                raise ReferenceIntegrityError(
                    "this exact reference source file was already ingested"
                )
            global_latest = connection.execute(
                "SELECT MAX(ingested_timestamp) FROM reference_batches"
            ).fetchone()[0]
            if global_latest is not None and ingested_at.isoformat() < global_latest:
                raise ReferenceIntegrityError(
                    "ingested_timestamp cannot precede the store's latest ingestion"
                )
            state = connection.execute(
                """
                SELECT COUNT(*) AS count, MAX(effective_key) AS maximum_effective_key
                FROM reference_records WHERE kind = ? AND source = ?
                """,
                (kind, source_name),
            ).fetchone()
            source_exists = int(state["count"]) > 0
            if mode == "initial" and source_exists:
                raise ReferenceIntegrityError(
                    "initial mode requires empty history for this source and kind"
                )
            if mode != "initial" and not source_exists:
                raise ReferenceIntegrityError(
                    f"{mode} mode requires existing history for this source and kind"
                )
            if (
                mode == "incremental"
                and state["maximum_effective_key"] is not None
                and min(effective_keys) < state["maximum_effective_key"]
            ):
                raise ReferenceIntegrityError(
                    "incremental ingestion would silently backfill older reference data; "
                    "use backfill mode"
                )

            identifiers = sorted({str(record["source_record_id"]) for record in records})
            existing_records: dict[str, sqlite3.Row] = {}
            if identifiers:
                placeholders = ",".join("?" for _ in identifiers)
                rows = connection.execute(
                    f"""
                    WITH ranked AS (
                        SELECT source_record_id, revision, record_action, payload_json,
                               ROW_NUMBER() OVER (
                                   PARTITION BY source_record_id ORDER BY revision DESC
                               ) AS revision_rank
                        FROM reference_records
                        WHERE kind = ? AND source = ?
                          AND source_record_id IN ({placeholders})
                    )
                    SELECT source_record_id, revision, record_action, payload_json
                    FROM ranked WHERE revision_rank = 1
                    """,
                    (kind, source_name, *identifiers),
                ).fetchall()
                existing_records = {str(row["source_record_id"]): row for row in rows}
            if mode == "correction":
                missing_ids = sorted(set(identifiers).difference(existing_records))
                if missing_ids:
                    raise ReferenceIntegrityError(
                        "correction mode requires every source_record_id to exist already"
                    )
            elif existing_records:
                raise ReferenceIntegrityError(
                    "existing source_record_id requires explicit correction mode"
                )

            grouped: dict[str, list[dict[str, Any]]] = {}
            for record in records:
                grouped.setdefault(str(record["source_record_id"]), []).append(record)
            for identifier, revisions in grouped.items():
                revisions.sort(key=lambda record: int(record["revision"]))
                existing = existing_records.get(identifier)
                previous_revision = int(existing["revision"]) if existing is not None else -1
                has_active_record = existing is not None and existing["record_action"] == "upsert"
                identity: tuple[Any, ...] | None = None
                if existing is not None:
                    payload = json.loads(str(existing["payload_json"]))["normalized"]
                    identity = tuple(payload[field] for field in _identity_fields(kind))
                for record in revisions:
                    revision = int(record["revision"])
                    if revision <= previous_revision:
                        raise ReferenceIntegrityError(
                            f"revision for {identifier!r} must increase strictly"
                        )
                    record_identity = tuple(record[field] for field in _identity_fields(kind))
                    if identity is not None and record_identity != identity:
                        raise ReferenceIntegrityError(
                            f"business identity for {identifier!r} cannot change across revisions"
                        )
                    if record["record_action"] == "cancel" and not has_active_record:
                        raise ReferenceIntegrityError(
                            f"cannot cancel {identifier!r} without an active prior upsert"
                        )
                    identity = record_identity
                    previous_revision = revision
                    has_active_record = record["record_action"] == "upsert"

            if archive_path.exists():
                if _sha256(archive_path.read_bytes()) != source_digest:
                    raise ReferenceIntegrityError(
                        "existing raw archive content does not match its name"
                    )
            else:
                archive_path.write_bytes(raw_bytes)
            previous_row = connection.execute(
                "SELECT manifest_hash FROM reference_batches ORDER BY batch_sequence DESC LIMIT 1"
            ).fetchone()
            previous_hash = str(previous_row[0]) if previous_row is not None else None
            batch_fields: dict[str, Any] = {
                "schema_version": REFERENCE_SCHEMA_VERSION,
                "kind": kind,
                "source": source_name,
                "mode": mode,
                "ingested_timestamp": ingested_at.isoformat(),
                "source_file_name": source_file_name,
                "source_sha256": source_digest,
                "source_size_bytes": len(raw_bytes),
                "raw_archive_name": archive_name,
                "row_count": len(records),
                "minimum_effective_timestamp": min(effective_keys),
                "maximum_effective_timestamp": max(effective_keys),
                "reason": reason_text,
                "previous_manifest_hash": previous_hash,
            }
            manifest_hash = _sha256(_canonical_json(_manifest_fields(batch_fields)).encode("utf-8"))
            batch_id = manifest_hash
            connection.execute(
                """
                INSERT INTO reference_batches(
                    batch_id, schema_version, kind, source, mode, ingested_timestamp,
                    source_file_name, source_sha256, source_size_bytes, raw_archive_name,
                    row_count, minimum_effective_timestamp, maximum_effective_timestamp,
                    reason, previous_manifest_hash, manifest_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    REFERENCE_SCHEMA_VERSION,
                    kind,
                    source_name,
                    mode,
                    ingested_at.isoformat(),
                    source_file_name,
                    source_digest,
                    len(raw_bytes),
                    archive_name,
                    len(records),
                    min(effective_keys),
                    max(effective_keys),
                    reason_text,
                    previous_hash,
                    manifest_hash,
                ),
            )
            for record in records:
                connection.execute(
                    """
                    INSERT INTO reference_records(
                        batch_id, kind, source, source_record_id, revision, record_action,
                        published_timestamp, ingested_timestamp, effective_key,
                        payload_json, payload_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        kind,
                        source_name,
                        record["source_record_id"],
                        record["revision"],
                        record["record_action"],
                        record["published_timestamp"],
                        ingested_at.isoformat(),
                        record["effective_key"],
                        record["payload_json"],
                        record["payload_sha256"],
                    ),
                )
            sequence = int(
                connection.execute(
                    "SELECT batch_sequence FROM reference_batches WHERE batch_id = ?",
                    (batch_id,),
                ).fetchone()[0]
            )
            connection.commit()
        return ReferenceBatch(
            batch_sequence=sequence,
            batch_id=batch_id,
            schema_version=REFERENCE_SCHEMA_VERSION,
            kind=kind,
            source=source_name,
            mode=mode,
            ingested_timestamp=ingested_at.isoformat(),
            source_file_name=source_file_name,
            source_sha256=source_digest,
            source_size_bytes=len(raw_bytes),
            raw_archive_name=archive_name,
            row_count=len(records),
            minimum_effective_timestamp=min(effective_keys),
            maximum_effective_timestamp=max(effective_keys),
            reason=reason_text,
            previous_manifest_hash=previous_hash,
            manifest_hash=manifest_hash,
        )

    def batches(self) -> tuple[ReferenceBatch, ...]:
        """Return ingestion manifests in append order."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM reference_batches ORDER BY batch_sequence"
            ).fetchall()
        return tuple(
            ReferenceBatch(**{field: row[field] for field in ReferenceBatch.__dataclass_fields__})
            for row in rows
        )

    def records_as_known(
        self,
        kind: ReferenceKind,
        *,
        effective_at: str | pd.Timestamp | None = None,
        published_at: str | pd.Timestamp | None = None,
        ingested_at: str | pd.Timestamp | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Return latest active revisions under knowledge and effective-time cutoffs."""
        if kind not in REFERENCE_KINDS:
            raise ValueError(f"kind must be one of {', '.join(REFERENCE_KINDS)}")
        with self._connection() as connection:
            latest_ingested = connection.execute(
                "SELECT MAX(ingested_timestamp) FROM reference_batches"
            ).fetchone()[0]
            if latest_ingested is None:
                return pd.DataFrame()
            published_cutoff = (
                _utc_timestamp(published_at, "published_at").isoformat()
                if published_at is not None
                else "9999-12-31T23:59:59+00:00"
            )
            ingested_cutoff = (
                _utc_timestamp(ingested_at, "ingested_at").isoformat()
                if ingested_at is not None
                else str(latest_ingested)
            )
            conditions = [
                "kind = ?",
                "published_timestamp <= ?",
                "ingested_timestamp <= ?",
            ]
            parameters: list[Any] = [kind, published_cutoff, ingested_cutoff]
            if source is not None:
                conditions.append("source = ?")
                parameters.append(source)
            rows = connection.execute(
                f"""
                WITH eligible AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY source, kind, source_record_id
                        ORDER BY revision DESC, published_timestamp DESC,
                                 ingested_timestamp DESC
                    ) AS revision_rank
                    FROM reference_records
                    WHERE {" AND ".join(conditions)}
                )
                SELECT * FROM eligible
                WHERE revision_rank = 1 AND record_action = 'upsert'
                ORDER BY effective_key, source, source_record_id
                """,
                parameters,
            ).fetchall()
        effective = (
            _utc_timestamp(effective_at, "effective_at") if effective_at is not None else None
        )
        output: list[dict[str, Any]] = []
        for row in rows:
            normalized = json.loads(str(row["payload_json"]))["normalized"]
            if effective is not None and kind in {"product", "contract"}:
                start = _utc_timestamp(normalized["effective_from"], "effective_from")
                end = (
                    _utc_timestamp(normalized["effective_to"], "effective_to")
                    if normalized.get("effective_to") is not None
                    else None
                )
                if effective < start or (end is not None and effective >= end):
                    continue
            if (
                effective is not None
                and kind == "corporate_action"
                and effective
                < _utc_timestamp(normalized["effective_timestamp"], "effective_timestamp")
            ):
                continue
            output.append(
                {
                    **normalized,
                    "_reference_source": row["source"],
                    "_source_record_id": row["source_record_id"],
                    "_revision": row["revision"],
                    "_published_timestamp": row["published_timestamp"],
                    "_ingested_timestamp": row["ingested_timestamp"],
                    "_payload_sha256": row["payload_sha256"],
                    "_batch_id": row["batch_id"],
                }
            )
        return pd.DataFrame(output)

    @staticmethod
    def _one_match(frame: pd.DataFrame, description: str) -> pd.Series:
        if frame.empty:
            raise ReferenceIntegrityError(f"missing {description}")
        if len(frame) > 1:
            raise ReferenceIntegrityError(f"ambiguous {description}")
        return frame.iloc[0]

    @staticmethod
    def _session_for(
        sessions: pd.DataFrame,
        *,
        reference_source: str,
        calendar_id: str,
        timestamp: pd.Timestamp,
        description: str,
    ) -> pd.Series:
        candidates = sessions[
            (sessions["_reference_source"] == reference_source)
            & (sessions["calendar_id"] == calendar_id)
        ]
        if candidates.empty:
            raise ReferenceIntegrityError(
                f"missing calendar coverage for {description}: {calendar_id}"
            )
        timezones = candidates["timezone"].drop_duplicates().tolist()
        if len(timezones) != 1:
            raise ReferenceIntegrityError(f"calendar {calendar_id!r} has ambiguous timezones")
        local_date = timestamp.tz_convert(ZoneInfo(str(timezones[0]))).date().isoformat()
        open_rows = candidates[candidates["status"] == "open"].copy()
        if not open_rows.empty:
            opens = pd.to_datetime(open_rows["open_timestamp"], utc=True)
            closes = pd.to_datetime(open_rows["close_timestamp"], utc=True)
            matching = open_rows[(opens <= timestamp) & (timestamp <= closes)]
        else:
            matching = open_rows
        if len(matching) == 1:
            return matching.iloc[0]
        if len(matching) > 1:
            raise ReferenceIntegrityError(
                f"multiple sessions overlap {description} at {timestamp.isoformat()}"
            )
        closed = candidates[
            (candidates["session_date"] == local_date) & (candidates["status"] == "closed")
        ]
        if not closed.empty:
            reason = str(closed.iloc[0].get("reason") or "scheduled closure")
            raise ReferenceIntegrityError(f"{description} falls on a closed session ({reason})")
        raise ReferenceIntegrityError(
            f"{description} at {timestamp.isoformat()} is outside an open session"
        )

    def enrich_snapshot(
        self,
        frame: pd.DataFrame,
        snapshot_time: str | pd.Timestamp,
        *,
        reference_published_at: str | pd.Timestamp | None = None,
        reference_ingested_at: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Resolve and validate every historical quote against point-in-time references."""
        snapshot = _utc_timestamp(snapshot_time, "snapshot_time")
        published_cutoff = (
            _utc_timestamp(reference_published_at, "reference_published_at")
            if reference_published_at is not None
            else snapshot
        )
        if published_cutoff > snapshot:
            raise ReferenceIntegrityError(
                "reference_published_at cannot follow the formation snapshot"
            )
        required = {
            "source",
            "contract_symbol",
            "underlying_symbol",
            "expiration",
            "strike",
            "option_type",
            "source_event_timestamp",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ReferenceIntegrityError(
                "snapshot is missing reference keys: " + ", ".join(missing)
            )
        if frame.empty:
            return frame.copy()
        query_args = {
            "published_at": published_cutoff,
            "ingested_at": reference_ingested_at,
        }
        contracts = self.records_as_known("contract", effective_at=snapshot, **query_args)
        products = self.records_as_known("product", effective_at=snapshot, **query_args)
        sessions = self.records_as_known("session", **query_args)
        actions = self.records_as_known("corporate_action", effective_at=snapshot, **query_args)
        if contracts.empty or products.empty or sessions.empty:
            raise ReferenceIntegrityError(
                "reference store lacks eligible contracts, products, or sessions"
            )

        enriched_rows: list[dict[str, Any]] = []
        for position, row in frame.reset_index(drop=True).iterrows():
            source = str(row["source"])
            symbol = str(row["contract_symbol"]).strip()
            if not symbol:
                raise ReferenceIntegrityError(
                    f"row {position}: contract_symbol is required for reference resolution"
                )
            contract_candidates = contracts[
                (contracts["market_data_source"] == source)
                & (contracts["source_contract_symbol"] == symbol)
                & (contracts["status"] == "active")
            ]
            contract = self._one_match(
                contract_candidates,
                f"active contract mapping for {source!r}/{symbol!r}",
            )
            reference_source = str(contract["_reference_source"])
            product_candidates = products[
                (products["_reference_source"] == reference_source)
                & (products["product_id"] == contract["product_id"])
                & (products["status"] == "active")
            ]
            product = self._one_match(
                product_candidates,
                f"active product {contract['product_id']!r}",
            )
            event_time = _utc_timestamp(
                row["source_event_timestamp"], f"row {position} source_event_timestamp"
            )
            effective_from = _utc_timestamp(contract["effective_from"], "effective_from")
            effective_to = (
                _utc_timestamp(contract["effective_to"], "effective_to")
                if contract.get("effective_to") is not None
                and not pd.isna(contract.get("effective_to"))
                else None
            )
            if event_time < effective_from or (
                effective_to is not None and event_time >= effective_to
            ):
                raise ReferenceIntegrityError(
                    f"row {position}: quote crosses a contract-term effective boundary"
                )
            observed_expiration = _utc_timestamp(row["expiration"], f"row {position} expiration")
            expected_expiration = _utc_timestamp(
                contract["expiration_timestamp"], "expiration_timestamp"
            )
            term_mismatches: list[str] = []
            if str(row["underlying_symbol"]).strip().upper() != contract["underlying_symbol"]:
                term_mismatches.append("underlying_symbol")
            if str(row["option_type"]).strip().lower() != contract["option_type"]:
                term_mismatches.append("option_type")
            if not math.isclose(
                float(row["strike"]), float(contract["strike"]), rel_tol=0.0, abs_tol=1e-9
            ):
                term_mismatches.append("strike")
            if observed_expiration != expected_expiration:
                term_mismatches.append("expiration")
            observed_exchange = row.get("exchange")
            if (
                observed_exchange is not None
                and not pd.isna(observed_exchange)
                and str(observed_exchange).strip()
                and str(observed_exchange).strip().upper() != product["exchange"]
            ):
                term_mismatches.append("exchange")
            if term_mismatches:
                raise ReferenceIntegrityError(
                    f"row {position}: event/reference term mismatch: " + ", ".join(term_mismatches)
                )
            last_trade = _utc_timestamp(contract["last_trade_timestamp"], "last_trade_timestamp")
            if snapshot > last_trade:
                raise ReferenceIntegrityError(
                    f"row {position}: formation snapshot follows the contract's last trade"
                )
            event_session = self._session_for(
                sessions,
                reference_source=reference_source,
                calendar_id=str(product["calendar_id"]),
                timestamp=event_time,
                description=f"row {position} quote",
            )
            formation_session = self._session_for(
                sessions,
                reference_source=reference_source,
                calendar_id=str(product["calendar_id"]),
                timestamp=snapshot,
                description="formation snapshot",
            )
            corporate_action_id = contract.get("corporate_action_id")
            action_hash: str | None = None
            if corporate_action_id is not None and not pd.isna(corporate_action_id):
                if actions.empty:
                    raise ReferenceIntegrityError(
                        f"missing announced corporate action {corporate_action_id!r}"
                    )
                action_candidates = actions[
                    (actions["_reference_source"] == reference_source)
                    & (actions["corporate_action_id"] == corporate_action_id)
                ]
                action = self._one_match(
                    action_candidates,
                    f"announced corporate action {corporate_action_id!r}",
                )
                if action["underlying_id"] != contract["underlying_id"]:
                    raise ReferenceIntegrityError(
                        f"row {position}: corporate action underlying does not match contract"
                    )
                action_hash = str(action["_payload_sha256"])
            enriched_rows.append(
                {
                    **row.to_dict(),
                    "stable_contract_id": contract["contract_id"],
                    "reference_product_id": product["product_id"],
                    "reference_source": reference_source,
                    "contract_multiplier": contract["contract_multiplier"],
                    "deliverable_json": contract["deliverable_json"],
                    "exercise_style": product["exercise_style"],
                    "settlement_method": product["settlement_method"],
                    "settlement_currency": product["currency"],
                    "settlement_style": product["settlement_style"],
                    "last_trade_timestamp": contract["last_trade_timestamp"],
                    "settlement_timestamp": contract["settlement_timestamp"],
                    "expiration_rule": product["expiration_rule"],
                    "last_trade_rule": product["last_trade_rule"],
                    "settlement_rule": product["settlement_rule"],
                    "corporate_action_id": (
                        None if pd.isna(corporate_action_id) else corporate_action_id
                    ),
                    "reference_contract_revision": int(contract["_revision"]),
                    "reference_contract_payload_sha256": contract["_payload_sha256"],
                    "reference_product_payload_sha256": product["_payload_sha256"],
                    "reference_action_payload_sha256": action_hash,
                    "event_session_date": event_session["session_date"],
                    "event_session_name": event_session["session_name"],
                    "formation_session_date": formation_session["session_date"],
                    "formation_session_name": formation_session["session_name"],
                    "session_timezone": formation_session["timezone"],
                    "reference_session_payload_sha256": formation_session["_payload_sha256"],
                    "reference_validated": True,
                }
            )
        enriched = pd.DataFrame(enriched_rows)
        duplicates = enriched["stable_contract_id"].duplicated(keep=False)
        if duplicates.any():
            duplicate_ids = sorted(enriched.loc[duplicates, "stable_contract_id"].unique())
            raise ReferenceIntegrityError(
                "multiple quotes resolve to the same stable contract: "
                + ", ".join(str(value) for value in duplicate_ids)
            )
        for column in (
            "last_trade_timestamp",
            "settlement_timestamp",
        ):
            enriched[column] = pd.to_datetime(enriched[column], utc=True)
        return enriched

    def verify(self) -> ReferenceVerification:
        """Verify SQLite, manifests, archived inputs, counts, and record hashes."""
        failures: list[str] = []
        batches = self.batches()
        archived: set[str] = set()
        previous_hash: str | None = None
        with self._connection() as connection:
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            if integrity != "ok":
                failures.append(f"SQLite integrity check failed: {integrity}")
            record_count = int(
                connection.execute("SELECT COUNT(*) FROM reference_records").fetchone()[0]
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
                        "SELECT COUNT(*) FROM reference_records WHERE batch_id = ?",
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
                "SELECT record_id, payload_json, payload_sha256 FROM reference_records"
            ):
                expected = _sha256(str(row["payload_json"]).encode("utf-8"))
                if expected != row["payload_sha256"]:
                    failures.append(f"record {row['record_id']} payload hash mismatch")
        return ReferenceVerification(
            valid=not failures,
            schema_version=REFERENCE_SCHEMA_VERSION,
            batch_count=len(batches),
            record_count=record_count,
            archived_file_count=len(archived),
            final_manifest_hash=previous_hash,
            failure_reasons=tuple(failures),
        )


def reference_report_json(value: ReferenceBatch | ReferenceVerification) -> bytes:
    """Serialize a reference ingestion or verification result deterministically."""
    return (json.dumps(asdict(value), indent=2, sort_keys=True) + "\n").encode("utf-8")
