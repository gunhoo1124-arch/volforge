"""Chronological, point-in-time option-chain calibration and hedge replay."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from volforge.calibration import calibrate_heston
from volforge.data.chain import REQUIRED_COLUMNS, ChainSchemaError, enrich_option_chain
from volforge.deliverables import parse_deliverable_terms
from volforge.hedging import (
    HEDGE_STRATEGIES,
    HedgeBacktestSettings,
    HestonRiskMetrics,
    heston_risk_metrics,
)
from volforge.models.common import OptionType
from volforge.models.heston import COSSettings, HestonParameters
from volforge.surfaces import (
    calibrate_ssvi,
    calibrate_svi_surface,
    prepare_surface_quotes,
)

_SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0


@dataclass(frozen=True, slots=True)
class ListedContract:
    """Stable identity of one listed option used throughout a replay."""

    expiration: str
    strike: float
    option_type: str
    stable_contract_id: str | None = None

    def __post_init__(self) -> None:
        expiration = pd.to_datetime(self.expiration, errors="coerce", utc=True)
        if pd.isna(expiration):
            raise ValueError("listed-contract expiration must be a valid timestamp")
        if not math.isfinite(self.strike) or self.strike <= 0.0:
            raise ValueError("listed-contract strike must be positive and finite")
        kind = OptionType.parse(self.option_type)
        stable_contract_id = self.stable_contract_id
        if stable_contract_id is not None and not str(stable_contract_id).strip():
            raise ValueError("stable_contract_id must be non-empty when supplied")
        object.__setattr__(self, "expiration", expiration.isoformat())
        object.__setattr__(self, "option_type", kind.value)
        object.__setattr__(
            self,
            "stable_contract_id",
            str(stable_contract_id).strip() if stable_contract_id is not None else None,
        )


@dataclass(frozen=True, slots=True)
class ReplaySettings:
    """Surface, Heston, hedge, cost, and audit controls for replay."""

    surface_model: str = "ssvi"
    strategy: str = "delta_gamma_vega"
    stock_cost_bps: float = 1.0
    option_cost_bps: float = 5.0
    max_option_position: float = 50.0
    condition_limit: float = 1e6
    surface_grid_points: int = 81
    heston_moneyness_min: float = -0.3
    heston_moneyness_max: float = 0.3
    heston_moneyness_points: int = 5
    heston_max_function_evaluations: int = 120
    require_feller: bool = False
    cos_terms: int = 96
    cos_truncation_width: float = 16.0
    accounting_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.surface_model not in {"svi", "ssvi"}:
            raise ValueError("surface_model must be 'svi' or 'ssvi'")
        if self.strategy not in HEDGE_STRATEGIES:
            raise ValueError(f"strategy must be one of {', '.join(HEDGE_STRATEGIES)}")
        costs = (self.stock_cost_bps, self.option_cost_bps)
        if any(not math.isfinite(value) or not 0.0 <= value <= 10_000.0 for value in costs):
            raise ValueError("transaction costs must lie between zero and 10,000 bps")
        if not math.isfinite(self.max_option_position) or self.max_option_position <= 0.0:
            raise ValueError("max_option_position must be positive and finite")
        if not math.isfinite(self.condition_limit) or self.condition_limit <= 1.0:
            raise ValueError("condition_limit must be finite and greater than one")
        if not isinstance(self.surface_grid_points, int) or self.surface_grid_points < 21:
            raise ValueError("surface_grid_points must be an integer of at least 21")
        if (
            not isinstance(self.heston_moneyness_points, int)
            or self.heston_moneyness_points < 5
            or self.heston_moneyness_min >= self.heston_moneyness_max
        ):
            raise ValueError("Heston moneyness grid requires at least five increasing points")
        if (
            not isinstance(self.heston_max_function_evaluations, int)
            or self.heston_max_function_evaluations <= 0
        ):
            raise ValueError("heston_max_function_evaluations must be positive")
        if not isinstance(self.cos_terms, int) or self.cos_terms < 32:
            raise ValueError("cos_terms must be an integer of at least 32")
        if not math.isfinite(self.cos_truncation_width) or self.cos_truncation_width <= 0.0:
            raise ValueError("cos_truncation_width must be positive and finite")
        if not math.isfinite(self.accounting_tolerance) or self.accounting_tolerance < 0.0:
            raise ValueError("accounting_tolerance must be non-negative and finite")


@dataclass(frozen=True, slots=True)
class ReplaySnapshotAudit:
    """Tamper-evident audit record for one processed point-in-time snapshot."""

    sequence: int
    timestamp: str
    action: str
    quote_rows: int
    snapshot_hash: str
    previous_audit_hash: str | None
    audit_hash: str
    no_lookahead_passed: bool
    source_timestamp_min: str
    source_timestamp_max: str
    all_quotes_valid: bool
    surface_downstream_eligible: bool | None
    heston_calibration_eligible: bool | None
    heston_parameters: HestonParameters | None
    target_multiplier: float | None
    target_deliverable_hash: str | None
    hedge_low_multiplier: float | None
    hedge_low_deliverable_hash: str | None
    hedge_high_multiplier: float | None
    hedge_high_deliverable_hash: str | None
    spot: float | None
    maturity_years: float | None
    target_mark: float | None
    target_bid: float | None
    target_ask: float | None
    target_fill: float | None
    hedge_low_mark: float | None
    hedge_low_bid: float | None
    hedge_low_ask: float | None
    hedge_low_fill: float | None
    hedge_high_mark: float | None
    hedge_high_bid: float | None
    hedge_high_ask: float | None
    hedge_high_fill: float | None
    target_position: float
    stock_position: float
    hedge_low_position: float
    hedge_high_position: float
    target_trade: float
    stock_trade: float
    hedge_low_trade: float
    hedge_high_trade: float
    cash_carry: float
    dividend_cashflow: float
    spread_cost: float
    cumulative_spread_cost: float
    transaction_cost: float
    cumulative_transaction_cost: float
    turnover: float
    cumulative_turnover: float
    cash: float
    portfolio_value: float | None
    incremental_pnl: float | None
    cumulative_pnl: float | None
    accounting_gap: float | None
    hedge_condition_number: float | None
    hedge_clipped: bool
    hedge_fallback_used: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChronologicalReplayResult:
    """Full point-in-time replay with contract identity and audit chain."""

    settings: ReplaySettings
    input_snapshot_count: int
    target_contract: ListedContract | None
    hedge_contracts: tuple[ListedContract, ListedContract] | None
    snapshots: tuple[ReplaySnapshotAudit, ...]
    completed: bool
    downstream_eligible: bool
    total_pnl: float | None
    total_spread_cost: float
    total_transaction_cost: float
    total_turnover: float
    maximum_absolute_accounting_gap: float | None
    final_audit_hash: str | None
    failure_reasons: tuple[str, ...]

    def ledger_frame(self) -> pd.DataFrame:
        """Return the chronological audit records as a flat table."""
        rows: list[dict[str, Any]] = []
        for snapshot in self.snapshots:
            row = asdict(snapshot)
            parameters = row.pop("heston_parameters")
            if parameters is not None:
                row.update({f"heston_{name}": value for name, value in parameters.items()})
            row["failure_reasons"] = "; ".join(row["failure_reasons"])
            rows.append(row)
        return pd.DataFrame(rows)


_DEFAULT_REPLAY_SETTINGS = ReplaySettings()


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
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


def snapshot_sha256(snapshot: pd.DataFrame) -> str:
    """Hash a snapshot after deterministic column, row, and scalar normalization."""
    columns = sorted(str(column) for column in snapshot.columns if column != "_parsed_timestamp")
    records = [
        {column: _json_value(row[column]) for column in columns}
        for _, row in snapshot.loc[:, columns].iterrows()
    ]
    canonical_records = sorted(
        records,
        key=lambda record: json.dumps(
            record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ),
    )
    payload = json.dumps(
        canonical_records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _audit_digest(previous_hash: str | None, fields: dict[str, Any]) -> str:
    payload = json.dumps(
        {"previous_audit_hash": previous_hash, "record": _json_value(fields)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _make_audit(**fields: Any) -> ReplaySnapshotAudit:
    previous_hash = fields["previous_audit_hash"]
    audit_hash = _audit_digest(previous_hash, fields)
    return ReplaySnapshotAudit(audit_hash=audit_hash, **fields)


def verify_replay_hash_chain(result: ChronologicalReplayResult) -> bool:
    """Verify ordering, linkage, and content hashes of an audit chain."""
    previous: str | None = None
    for expected_sequence, snapshot in enumerate(result.snapshots):
        if snapshot.sequence != expected_sequence or snapshot.previous_audit_hash != previous:
            return False
        fields = asdict(snapshot)
        observed = str(fields.pop("audit_hash"))
        if _audit_digest(previous, fields) != observed:
            return False
        previous = observed
    return result.final_audit_hash == previous


def _contract_rows(chain: pd.DataFrame, contract: ListedContract) -> pd.DataFrame:
    expiration = pd.to_datetime(contract.expiration, utc=True)
    terms_match = (
        (chain["expiration"] == expiration)
        & np.isclose(chain["strike"], contract.strike, rtol=0.0, atol=1e-8)
        & (chain["option_type"] == contract.option_type)
    )
    if contract.stable_contract_id is None:
        identity_match = terms_match
    else:
        if "stable_contract_id" not in chain:
            raise ValueError("stable contract identity is unavailable in this snapshot")
        identity_match = chain["stable_contract_id"].astype("string") == contract.stable_contract_id
        mismatched_terms = chain.loc[chain["is_valid"] & identity_match & ~terms_match]
        if not mismatched_terms.empty:
            raise ValueError(
                f"stable contract {contract.stable_contract_id} changed strike, expiry, or type; "
                "adjusted terms are not supported by the current pricing engine"
            )
    return chain.loc[chain["is_valid"] & identity_match]


def _canonical_deliverable(value: object, multiplier: float, row: pd.Series) -> tuple[str, str]:
    """Validate the single-security deliverable supported by the pricing ledger."""
    deliverable = value
    if deliverable is not None:
        try:
            is_missing = pd.isna(deliverable)
        except (TypeError, ValueError):
            is_missing = False
        if isinstance(is_missing, (bool, np.bool_)) and bool(is_missing):
            deliverable = None
    underlying_id = row.get("underlying_id")
    if underlying_id is not None and pd.isna(underlying_id):
        underlying_id = None
    settlement_method = row.get("settlement_method", "physical")
    if settlement_method is None or pd.isna(settlement_method):
        settlement_method = "physical"
    currency = row.get("settlement_currency", row.get("currency", "USD"))
    if currency is None or pd.isna(currency):
        currency = "USD"
    try:
        terms = parse_deliverable_terms(
            deliverable,
            settlement_method=str(settlement_method),
            contract_multiplier=multiplier,
            currency=str(currency),
            underlying_id=str(underlying_id) if underlying_id is not None else None,
        )
    except ValueError as exc:
        raise ValueError(f"unsupported structured deliverable: {exc}") from exc
    if not terms.single_factor_pricing_compatible:
        raise ValueError(
            "unsupported structured deliverable: replay Heston pricing requires a standard "
            "single-security physical deliverable"
        )
    return terms.canonical_json, terms.sha256


def _contract_economics(row: pd.Series) -> tuple[float, str]:
    has_multiplier = "contract_multiplier" in row.index
    has_deliverable = "deliverable_json" in row.index
    if has_multiplier != has_deliverable:
        raise ValueError("contract_multiplier and deliverable_json must be supplied together")
    raw_multiplier = row["contract_multiplier"] if has_multiplier else 1.0
    multiplier = pd.to_numeric(raw_multiplier, errors="coerce")
    if pd.isna(multiplier) or not math.isfinite(float(multiplier)) or float(multiplier) <= 0.0:
        raise ValueError("listed-contract multiplier must be positive and finite")
    raw_deliverable = row["deliverable_json"] if has_deliverable else None
    _, deliverable_hash = _canonical_deliverable(raw_deliverable, float(multiplier), row)
    return float(multiplier), deliverable_hash


def _contract_quote(
    chain: pd.DataFrame,
    contract: ListedContract,
) -> tuple[float, float, float, float, str, pd.Series]:
    rows = _contract_rows(chain, contract)
    if len(rows) != 1:
        raise ValueError(
            "snapshot must contain exactly one valid quote for "
            f"{contract.option_type} K={contract.strike:g} expiring {contract.expiration}"
        )
    row = rows.iloc[0]
    bid = float(row["bid"])
    ask = float(row["ask"])
    mark = float(row["midpoint"])
    if (
        not math.isfinite(bid)
        or not math.isfinite(ask)
        or not math.isfinite(mark)
        or bid < 0.0
        or bid > ask
    ):
        raise ValueError("listed-contract bid, ask, and midpoint must form a valid quote")
    multiplier, deliverable_hash = _contract_economics(row)
    return bid, ask, mark, multiplier, deliverable_hash, row


def _listed_contract(row: pd.Series) -> ListedContract:
    stable_contract_id = row.get("stable_contract_id")
    if stable_contract_id is not None and pd.isna(stable_contract_id):
        stable_contract_id = None
    return ListedContract(
        pd.Timestamp(row["expiration"]).isoformat(),
        float(row["strike"]),
        str(row["option_type"]),
        str(stable_contract_id) if stable_contract_id is not None else None,
    )


def _execution_price(trade: float, bid: float, ask: float, mark: float) -> float:
    """Return the immediately executable quote side for a signed trade."""
    if trade > 0.0:
        return ask
    if trade < 0.0:
        return bid
    return mark


def _select_contracts(
    chain: pd.DataFrame,
    requested_target: ListedContract | None,
    target_option_type: str,
) -> tuple[ListedContract, tuple[ListedContract, ListedContract]]:
    kind = OptionType.parse(target_option_type)
    valid = chain.loc[chain["is_valid"]].copy()
    if requested_target is None:
        candidates: list[tuple[pd.Timestamp, pd.DataFrame]] = []
        for expiration, group in valid.groupby("expiration", sort=True):
            call_strikes = group.loc[group["option_type"] == "call", "strike"].nunique()
            target_rows = group.loc[group["option_type"] == kind.value]
            if call_strikes >= 3 and not target_rows.empty:
                candidates.append((expiration, target_rows))
        if not candidates:
            raise ValueError("first snapshot has no expiry with a target and three listed calls")
        expiration, target_rows = candidates[0]
        spot = float(target_rows["underlying_spot"].median())
        target_row = target_rows.iloc[
            int(np.argmin(np.abs(target_rows["strike"].to_numpy(dtype=float) - spot)))
        ]
        target = _listed_contract(target_row)
    else:
        target = requested_target
        _contract_quote(chain, target)

    expiry = pd.to_datetime(target.expiration, utc=True)
    call_strikes = np.sort(
        valid.loc[(valid["expiration"] == expiry) & (valid["option_type"] == "call"), "strike"]
        .drop_duplicates()
        .to_numpy(dtype=float)
    )
    below = call_strikes[call_strikes < target.strike - 1e-8]
    above = call_strikes[call_strikes > target.strike + 1e-8]
    if below.size == 0 or above.size == 0:
        raise ValueError("listed hedge calls must bracket the target strike")
    hedge_rows = []
    for strike in (float(below[-1]), float(above[0])):
        rows = valid.loc[
            (valid["expiration"] == expiry)
            & (valid["option_type"] == "call")
            & np.isclose(valid["strike"], strike, rtol=0.0, atol=1e-8)
        ]
        if len(rows) != 1:
            raise ValueError("first snapshot must identify each listed hedge contract uniquely")
        hedge_rows.append(rows.iloc[0])
    hedges = (_listed_contract(hedge_rows[0]), _listed_contract(hedge_rows[1]))
    return target, hedges


def _hedge_positions(
    strategy: str,
    target: HestonRiskMetrics,
    low: HestonRiskMetrics,
    high: HestonRiskMetrics,
    settings: ReplaySettings,
    target_multiplier: float = 1.0,
    hedge_multipliers: tuple[float, float] = (1.0, 1.0),
) -> tuple[float, np.ndarray, float | None, bool, bool]:
    options = np.zeros(2, dtype=float)
    condition: float | None = None
    fallback = False
    if strategy == "unhedged":
        return 0.0, options, condition, False, fallback
    if strategy == "delta":
        return target.delta * target_multiplier, options, condition, False, fallback
    target_gamma = target.gamma * target_multiplier
    target_vega = target.instantaneous_vega * target_multiplier
    low_gamma = low.gamma * hedge_multipliers[0]
    high_gamma = high.gamma * hedge_multipliers[1]
    low_vega = low.instantaneous_vega * hedge_multipliers[0]
    high_vega = high.instantaneous_vega * hedge_multipliers[1]
    if strategy == "delta_gamma":
        if abs(low_gamma) <= 1e-12:
            fallback = True
        else:
            options[0] = target_gamma / low_gamma
    elif strategy == "delta_vega":
        if abs(high_vega) <= 1e-12:
            fallback = True
        else:
            options[1] = target_vega / high_vega
    else:
        gamma_scale = max(abs(target_gamma), abs(low_gamma), abs(high_gamma), 1e-12)
        vega_scale = max(
            abs(target_vega),
            abs(low_vega),
            abs(high_vega),
            1e-12,
        )
        system = np.asarray(
            [
                [low_gamma / gamma_scale, high_gamma / gamma_scale],
                [low_vega / vega_scale, high_vega / vega_scale],
            ]
        )
        exposure = np.asarray([target_gamma / gamma_scale, target_vega / vega_scale])
        condition = float(np.linalg.cond(system))
        if math.isfinite(condition) and condition <= settings.condition_limit:
            try:
                options = np.linalg.solve(system, exposure)
            except np.linalg.LinAlgError:
                fallback = True
        else:
            fallback = True
        if fallback:
            options = np.linalg.lstsq(system, exposure, rcond=1.0 / settings.condition_limit)[0]
    clipped_options = np.clip(options, -settings.max_option_position, settings.max_option_position)
    clipped = bool(np.any(clipped_options != options))
    stock = target.delta * target_multiplier - (
        clipped_options[0] * low.delta * hedge_multipliers[0]
        + clipped_options[1] * high.delta * hedge_multipliers[1]
    )
    return float(stock), clipped_options, condition, clipped, fallback


def _blocked_fields(
    *,
    sequence: int,
    timestamp: pd.Timestamp,
    snapshot: pd.DataFrame,
    snapshot_hash: str,
    previous_hash: str | None,
    reasons: list[str],
    no_lookahead: bool,
    all_valid: bool,
    surface_eligible: bool | None,
    heston_eligible: bool | None,
    heston_parameters: HestonParameters | None,
    cash: float,
    stock_position: float,
    option_positions: np.ndarray,
    cumulative_cost: float,
    cumulative_spread_cost: float,
    cumulative_turnover: float,
    target_multiplier: float | None = None,
    target_deliverable_hash: str | None = None,
    low_multiplier: float | None = None,
    low_deliverable_hash: str | None = None,
    high_multiplier: float | None = None,
    high_deliverable_hash: str | None = None,
) -> dict[str, Any]:
    timestamps = pd.to_datetime(snapshot["quote_timestamp"], errors="coerce", utc=True)
    return {
        "sequence": sequence,
        "timestamp": timestamp.isoformat(),
        "action": "blocked",
        "quote_rows": len(snapshot),
        "snapshot_hash": snapshot_hash,
        "previous_audit_hash": previous_hash,
        "no_lookahead_passed": no_lookahead,
        "source_timestamp_min": timestamps.min().isoformat(),
        "source_timestamp_max": timestamps.max().isoformat(),
        "all_quotes_valid": all_valid,
        "surface_downstream_eligible": surface_eligible,
        "heston_calibration_eligible": heston_eligible,
        "heston_parameters": heston_parameters,
        "target_multiplier": target_multiplier,
        "target_deliverable_hash": target_deliverable_hash,
        "hedge_low_multiplier": low_multiplier,
        "hedge_low_deliverable_hash": low_deliverable_hash,
        "hedge_high_multiplier": high_multiplier,
        "hedge_high_deliverable_hash": high_deliverable_hash,
        "spot": None,
        "maturity_years": None,
        "target_mark": None,
        "target_bid": None,
        "target_ask": None,
        "target_fill": None,
        "hedge_low_mark": None,
        "hedge_low_bid": None,
        "hedge_low_ask": None,
        "hedge_low_fill": None,
        "hedge_high_mark": None,
        "hedge_high_bid": None,
        "hedge_high_ask": None,
        "hedge_high_fill": None,
        "target_position": -1.0 if sequence > 0 else 0.0,
        "stock_position": stock_position,
        "hedge_low_position": float(option_positions[0]),
        "hedge_high_position": float(option_positions[1]),
        "target_trade": 0.0,
        "stock_trade": 0.0,
        "hedge_low_trade": 0.0,
        "hedge_high_trade": 0.0,
        "cash_carry": 0.0,
        "dividend_cashflow": 0.0,
        "spread_cost": 0.0,
        "cumulative_spread_cost": cumulative_spread_cost,
        "transaction_cost": 0.0,
        "cumulative_transaction_cost": cumulative_cost,
        "turnover": 0.0,
        "cumulative_turnover": cumulative_turnover,
        "cash": cash,
        "portfolio_value": None,
        "incremental_pnl": None,
        "cumulative_pnl": None,
        "accounting_gap": None,
        "hedge_condition_number": None,
        "hedge_clipped": False,
        "hedge_fallback_used": False,
        "failure_reasons": tuple(reasons),
    }


def run_chronological_replay(
    frame: pd.DataFrame,
    *,
    settings: ReplaySettings = _DEFAULT_REPLAY_SETTINGS,
    target_contract: ListedContract | None = None,
    target_option_type: str = "call",
) -> ChronologicalReplayResult:
    """Replay only contemporaneous snapshots and stop on the first unsafe day.

    The Heston model supplies hedge sensitivities. Buys execute at observed
    asks, sells execute at observed bids, and open positions remain marked at
    observed midpoints. Model prices never enter the cash ledger.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ChainSchemaError(f"missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("replay input must contain at least one quote")
    kind = OptionType.parse(target_option_type)
    work = frame.copy(deep=True)
    if "reference_validated" in work:
        reference_passed = work["reference_validated"].map(
            lambda value: value is True or str(value).strip().lower() == "true"
        )
        if not reference_passed.all():
            raise ValueError(
                "replay input contains rows that failed contract/session reference validation"
            )
        reference_columns = {"stable_contract_id", "contract_multiplier", "deliverable_json"}
        missing_reference = sorted(reference_columns.difference(work.columns))
        if missing_reference:
            raise ChainSchemaError(
                "reference-validated replay input requires " + ", ".join(missing_reference)
            )
    work["_parsed_timestamp"] = pd.to_datetime(work["quote_timestamp"], errors="coerce", utc=True)
    if work["_parsed_timestamp"].isna().any():
        raise ValueError("every replay quote_timestamp must be valid")
    timestamps = tuple(sorted(work["_parsed_timestamp"].unique()))
    if len(timestamps) < 2:
        raise ValueError("chronological replay requires at least two snapshots")

    target = target_contract
    hedges: tuple[ListedContract, ListedContract] | None = None
    audits: list[ReplaySnapshotAudit] = []
    replay_failures: list[str] = []
    previous_hash: str | None = None
    cash = 0.0
    stock_position = 0.0
    option_positions = np.zeros(2, dtype=float)
    cumulative_cost = 0.0
    cumulative_spread_cost = 0.0
    cumulative_turnover = 0.0
    previous_portfolio = 0.0
    previous_timestamp: pd.Timestamp | None = None
    previous_rate = 0.0
    previous_dividend_yield = 0.0
    previous_spot = 0.0
    accounting_gaps: list[float] = []

    cos_settings = COSSettings(settings.cos_terms, settings.cos_truncation_width)
    risk_settings = HedgeBacktestSettings(
        paths=2,
        rebalancing_steps=1,
        stock_cost_bps=settings.stock_cost_bps,
        option_cost_bps=settings.option_cost_bps,
        max_option_position=settings.max_option_position,
        condition_limit=settings.condition_limit,
    )
    diagnostic_grid = np.linspace(-1.0, 1.0, settings.surface_grid_points)
    heston_grid = np.linspace(
        settings.heston_moneyness_min,
        settings.heston_moneyness_max,
        settings.heston_moneyness_points,
    )

    for sequence, raw_timestamp in enumerate(timestamps):
        timestamp = pd.Timestamp(raw_timestamp)
        snapshot = work.loc[work["_parsed_timestamp"] == raw_timestamp].drop(
            columns="_parsed_timestamp"
        )
        content_hash = snapshot_sha256(snapshot)
        parsed_source = pd.to_datetime(snapshot["quote_timestamp"], errors="coerce", utc=True)
        no_lookahead = bool((parsed_source == timestamp).all())
        reasons: list[str] = []
        surface_eligible: bool | None = None
        heston_eligible: bool | None = None
        parameters: HestonParameters | None = None
        enriched = enrich_option_chain(
            snapshot,
            model="bsm",
            valuation_time=timestamp,
            stale_after="0s",
        )
        all_valid = bool(enriched["is_valid"].all())
        if not no_lookahead:
            reasons.append("snapshot contains a quote outside its valuation timestamp")
        if not all_valid:
            invalid = int((~enriched["is_valid"]).sum())
            reasons.append(f"snapshot contains {invalid} invalid quote row(s)")

        if not reasons and sequence == 0:
            try:
                target, hedges = _select_contracts(enriched, target, kind.value)
            except ValueError as exc:
                reasons.append(str(exc))

        target_mark: float | None = None
        target_bid: float | None = None
        target_ask: float | None = None
        low_mark: float | None = None
        low_bid: float | None = None
        low_ask: float | None = None
        high_mark: float | None = None
        high_bid: float | None = None
        high_ask: float | None = None
        target_multiplier: float | None = None
        target_deliverable_hash: str | None = None
        low_multiplier: float | None = None
        low_deliverable_hash: str | None = None
        high_multiplier: float | None = None
        high_deliverable_hash: str | None = None
        target_row: pd.Series | None = None
        if not reasons and target is not None and hedges is not None:
            try:
                (
                    target_bid,
                    target_ask,
                    target_mark,
                    target_multiplier,
                    target_deliverable_hash,
                    target_row,
                ) = _contract_quote(enriched, target)
                (
                    low_bid,
                    low_ask,
                    low_mark,
                    low_multiplier,
                    low_deliverable_hash,
                    _,
                ) = _contract_quote(enriched, hedges[0])
                (
                    high_bid,
                    high_ask,
                    high_mark,
                    high_multiplier,
                    high_deliverable_hash,
                    _,
                ) = _contract_quote(enriched, hedges[1])
            except ValueError as exc:
                reasons.append(str(exc))

        surface = None
        if not reasons:
            try:
                quotes = prepare_surface_quotes(enriched)
                surface = (
                    calibrate_svi_surface(quotes, diagnostic_grid=diagnostic_grid)
                    if settings.surface_model == "svi"
                    else calibrate_ssvi(quotes, diagnostic_grid=diagnostic_grid)
                )
                surface_eligible = surface.downstream_eligible
                if not surface_eligible:
                    reasons.extend(surface.failure_reasons)
            except (ArithmeticError, RuntimeError, ValueError) as exc:
                surface_eligible = False
                reasons.append(f"surface calibration failed: {exc}")

        calibration = None
        if not reasons and surface is not None:
            try:
                calibration = calibrate_heston(
                    surface,
                    log_moneyness=heston_grid,
                    settings=cos_settings,
                    require_feller=settings.require_feller,
                    max_function_evaluations=settings.heston_max_function_evaluations,
                )
                heston_eligible = calibration.calibration_eligible
                parameters = calibration.parameters
                if not heston_eligible:
                    reasons.extend(calibration.failure_reasons)
            except (ArithmeticError, RuntimeError, ValueError) as exc:
                heston_eligible = False
                reasons.append(f"Heston calibration failed: {exc}")

        if reasons:
            audit = _make_audit(
                **_blocked_fields(
                    sequence=sequence,
                    timestamp=timestamp,
                    snapshot=snapshot,
                    snapshot_hash=content_hash,
                    previous_hash=previous_hash,
                    reasons=reasons,
                    no_lookahead=no_lookahead,
                    all_valid=all_valid,
                    surface_eligible=surface_eligible,
                    heston_eligible=heston_eligible,
                    heston_parameters=parameters,
                    cash=cash,
                    stock_position=stock_position,
                    option_positions=option_positions,
                    cumulative_cost=cumulative_cost,
                    cumulative_spread_cost=cumulative_spread_cost,
                    cumulative_turnover=cumulative_turnover,
                    target_multiplier=target_multiplier,
                    target_deliverable_hash=target_deliverable_hash,
                    low_multiplier=low_multiplier,
                    low_deliverable_hash=low_deliverable_hash,
                    high_multiplier=high_multiplier,
                    high_deliverable_hash=high_deliverable_hash,
                )
            )
            audits.append(audit)
            previous_hash = audit.audit_hash
            replay_failures.extend(
                f"snapshot {sequence} ({timestamp.isoformat()}): {reason}" for reason in reasons
            )
            break

        if (
            target is None
            or hedges is None
            or target_mark is None
            or target_bid is None
            or target_ask is None
            or low_mark is None
            or low_bid is None
            or low_ask is None
            or high_mark is None
            or high_bid is None
            or high_ask is None
            or target_multiplier is None
            or target_deliverable_hash is None
            or low_multiplier is None
            or low_deliverable_hash is None
            or high_multiplier is None
            or high_deliverable_hash is None
            or target_row is None
            or calibration is None
            or parameters is None
        ):
            raise RuntimeError("replay reached an incomplete calibrated state")

        spot = float(target_row["underlying_spot"])
        maturity = float(target_row["maturity_years"])
        rate = float(target_row["risk_free_rate"])
        dividend_yield = float(target_row["dividend_yield"])
        try:
            target_metrics = heston_risk_metrics(
                target.option_type,
                spot,
                np.asarray([target.strike]),
                maturity,
                parameters,
                rate,
                dividend_yield,
                settings=risk_settings,
                cos_settings=cos_settings,
            )[0]
            hedge_metrics = heston_risk_metrics(
                "call",
                spot,
                np.asarray([hedges[0].strike, hedges[1].strike]),
                maturity,
                parameters,
                rate,
                dividend_yield,
                settings=risk_settings,
                cos_settings=cos_settings,
            )
            desired_stock, desired_options, condition, clipped, fallback = _hedge_positions(
                settings.strategy,
                target_metrics,
                hedge_metrics[0],
                hedge_metrics[1],
                settings,
                target_multiplier,
                (low_multiplier, high_multiplier),
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            reasons = [f"hedge sizing failed: {exc}"]
            audit = _make_audit(
                **_blocked_fields(
                    sequence=sequence,
                    timestamp=timestamp,
                    snapshot=snapshot,
                    snapshot_hash=content_hash,
                    previous_hash=previous_hash,
                    reasons=reasons,
                    no_lookahead=no_lookahead,
                    all_valid=all_valid,
                    surface_eligible=surface_eligible,
                    heston_eligible=heston_eligible,
                    heston_parameters=parameters,
                    cash=cash,
                    stock_position=stock_position,
                    option_positions=option_positions,
                    cumulative_cost=cumulative_cost,
                    cumulative_spread_cost=cumulative_spread_cost,
                    cumulative_turnover=cumulative_turnover,
                    target_multiplier=target_multiplier,
                    target_deliverable_hash=target_deliverable_hash,
                    low_multiplier=low_multiplier,
                    low_deliverable_hash=low_deliverable_hash,
                    high_multiplier=high_multiplier,
                    high_deliverable_hash=high_deliverable_hash,
                )
            )
            audits.append(audit)
            previous_hash = audit.audit_hash
            replay_failures.append(f"snapshot {sequence} ({timestamp.isoformat()}): {reasons[0]}")
            break

        cash_carry = 0.0
        dividend_cashflow = 0.0
        if previous_timestamp is not None:
            elapsed = (timestamp - previous_timestamp).total_seconds() / _SECONDS_PER_YEAR
            if elapsed <= 0.0:
                raise RuntimeError("replay timestamps must be strictly increasing")
            cash_carry = cash * math.expm1(previous_rate * elapsed)
            dividend_cashflow = (
                stock_position * previous_spot * math.expm1(previous_dividend_yield * elapsed)
            )
            cash += cash_carry + dividend_cashflow

        marks = np.asarray([low_mark, high_mark], dtype=float)
        option_multipliers = np.asarray([low_multiplier, high_multiplier], dtype=float)
        pre_trade_value = (
            cash
            + stock_position * spot
            + float(option_positions @ (marks * option_multipliers))
            - target_mark * target_multiplier
        )
        final_snapshot = sequence == len(timestamps) - 1
        if sequence == 0:
            target_trade = -1.0
            stock_trade = desired_stock
            option_trade = desired_options.copy()
            action = "open_rebalance"
            pre_trade_value = 0.0
        elif final_snapshot:
            target_trade = 1.0
            stock_trade = -stock_position
            option_trade = -option_positions
            action = "liquidate"
        else:
            target_trade = 0.0
            stock_trade = desired_stock - stock_position
            option_trade = desired_options - option_positions
            action = "rebalance"

        target_fill = _execution_price(target_trade, target_bid, target_ask, target_mark)
        hedge_bids = np.asarray([low_bid, high_bid], dtype=float)
        hedge_asks = np.asarray([low_ask, high_ask], dtype=float)
        hedge_fills = np.asarray(
            [
                _execution_price(trade, bid, ask, mark)
                for trade, bid, ask, mark in zip(
                    option_trade, hedge_bids, hedge_asks, marks, strict=True
                )
            ]
        )
        stock_notional = abs(stock_trade) * spot
        hedge_notional = float(np.abs(option_trade) @ (hedge_fills * option_multipliers))
        target_notional = abs(target_trade) * target_fill * target_multiplier
        turnover = stock_notional + hedge_notional + target_notional
        spread_cost = float(
            np.abs(option_trade) @ (((hedge_asks - hedge_bids) / 2.0) * option_multipliers)
        )
        spread_cost += abs(target_trade) * (target_ask - target_bid) / 2.0 * target_multiplier
        transaction_cost = (
            stock_notional * settings.stock_cost_bps * 1e-4
            + (hedge_notional + target_notional) * settings.option_cost_bps * 1e-4
        )
        cash -= (
            stock_trade * spot
            + float(option_trade @ (hedge_fills * option_multipliers))
            + target_trade * target_fill * target_multiplier
            + transaction_cost
        )
        stock_position += stock_trade
        option_positions = option_positions + option_trade
        target_position = -1.0 + target_trade if final_snapshot else -1.0
        if final_snapshot:
            portfolio_value = cash
        else:
            portfolio_value = (
                cash
                + stock_position * spot
                + float(option_positions @ (marks * option_multipliers))
                + target_position * target_mark * target_multiplier
            )
        accounting_gap = portfolio_value - (pre_trade_value - spread_cost - transaction_cost)
        incremental_pnl = portfolio_value - previous_portfolio
        cumulative_spread_cost += spread_cost
        cumulative_cost += transaction_cost
        cumulative_turnover += turnover
        cumulative_pnl = portfolio_value
        accounting_gaps.append(accounting_gap)
        if abs(accounting_gap) > settings.accounting_tolerance:
            replay_failures.append(
                f"snapshot {sequence} ({timestamp.isoformat()}): accounting gap "
                f"{accounting_gap:.6g} exceeds tolerance"
            )

        fields = {
            "sequence": sequence,
            "timestamp": timestamp.isoformat(),
            "action": action,
            "quote_rows": len(snapshot),
            "snapshot_hash": content_hash,
            "previous_audit_hash": previous_hash,
            "no_lookahead_passed": no_lookahead,
            "source_timestamp_min": parsed_source.min().isoformat(),
            "source_timestamp_max": parsed_source.max().isoformat(),
            "all_quotes_valid": all_valid,
            "surface_downstream_eligible": surface_eligible,
            "heston_calibration_eligible": heston_eligible,
            "heston_parameters": parameters,
            "target_multiplier": target_multiplier,
            "target_deliverable_hash": target_deliverable_hash,
            "hedge_low_multiplier": low_multiplier,
            "hedge_low_deliverable_hash": low_deliverable_hash,
            "hedge_high_multiplier": high_multiplier,
            "hedge_high_deliverable_hash": high_deliverable_hash,
            "spot": spot,
            "maturity_years": maturity,
            "target_mark": target_mark,
            "target_bid": target_bid,
            "target_ask": target_ask,
            "target_fill": target_fill,
            "hedge_low_mark": low_mark,
            "hedge_low_bid": low_bid,
            "hedge_low_ask": low_ask,
            "hedge_low_fill": float(hedge_fills[0]),
            "hedge_high_mark": high_mark,
            "hedge_high_bid": high_bid,
            "hedge_high_ask": high_ask,
            "hedge_high_fill": float(hedge_fills[1]),
            "target_position": target_position,
            "stock_position": stock_position,
            "hedge_low_position": float(option_positions[0]),
            "hedge_high_position": float(option_positions[1]),
            "target_trade": target_trade,
            "stock_trade": stock_trade,
            "hedge_low_trade": float(option_trade[0]),
            "hedge_high_trade": float(option_trade[1]),
            "cash_carry": cash_carry,
            "dividend_cashflow": dividend_cashflow,
            "spread_cost": spread_cost,
            "cumulative_spread_cost": cumulative_spread_cost,
            "transaction_cost": transaction_cost,
            "cumulative_transaction_cost": cumulative_cost,
            "turnover": turnover,
            "cumulative_turnover": cumulative_turnover,
            "cash": cash,
            "portfolio_value": portfolio_value,
            "incremental_pnl": incremental_pnl,
            "cumulative_pnl": cumulative_pnl,
            "accounting_gap": accounting_gap,
            "hedge_condition_number": condition,
            "hedge_clipped": clipped,
            "hedge_fallback_used": fallback,
            "failure_reasons": (
                ("accounting reconciliation failed",)
                if abs(accounting_gap) > settings.accounting_tolerance
                else ()
            ),
        }
        audit = _make_audit(**fields)
        audits.append(audit)
        previous_hash = audit.audit_hash
        previous_portfolio = portfolio_value
        previous_timestamp = timestamp
        previous_rate = rate
        previous_dividend_yield = dividend_yield
        previous_spot = spot
        if abs(accounting_gap) > settings.accounting_tolerance:
            break

    completed = (
        len(audits) == len(timestamps)
        and bool(audits)
        and audits[-1].action == "liquidate"
        and not replay_failures
    )
    maximum_gap = max((abs(value) for value in accounting_gaps), default=None)
    result = ChronologicalReplayResult(
        settings=settings,
        input_snapshot_count=len(timestamps),
        target_contract=target,
        hedge_contracts=hedges,
        snapshots=tuple(audits),
        completed=completed,
        downstream_eligible=False,
        total_pnl=(audits[-1].cumulative_pnl if completed else None),
        total_spread_cost=cumulative_spread_cost,
        total_transaction_cost=cumulative_cost,
        total_turnover=cumulative_turnover,
        maximum_absolute_accounting_gap=maximum_gap,
        final_audit_hash=previous_hash,
        failure_reasons=tuple(replay_failures),
    )
    chain_verified = verify_replay_hash_chain(result)
    if not chain_verified:
        result = replace(
            result,
            failure_reasons=(*result.failure_reasons, "audit hash-chain verification failed"),
        )
    return replace(result, downstream_eligible=completed and chain_verified)


def replay_summary(result: ChronologicalReplayResult) -> dict[str, Any]:
    """Return a deterministic JSON-compatible replay audit."""
    return {
        "engine": "point-in-time-chain-replay",
        "settings": asdict(result.settings),
        "input_snapshot_count": result.input_snapshot_count,
        "processed_snapshot_count": len(result.snapshots),
        "target_contract": (
            asdict(result.target_contract) if result.target_contract is not None else None
        ),
        "hedge_contracts": (
            [asdict(contract) for contract in result.hedge_contracts]
            if result.hedge_contracts is not None
            else None
        ),
        "completed": result.completed,
        "downstream_eligible": result.downstream_eligible,
        "total_pnl": result.total_pnl,
        "total_spread_cost": result.total_spread_cost,
        "total_transaction_cost": result.total_transaction_cost,
        "total_turnover": result.total_turnover,
        "maximum_absolute_accounting_gap": result.maximum_absolute_accounting_gap,
        "final_audit_hash": result.final_audit_hash,
        "hash_chain_verified": verify_replay_hash_chain(result),
        "failure_reasons": list(result.failure_reasons),
        "snapshots": [asdict(snapshot) for snapshot in result.snapshots],
    }


def replay_report_json(result: ChronologicalReplayResult) -> bytes:
    """Serialize a deterministic human-readable chronological replay audit."""
    return (json.dumps(replay_summary(result), indent=2, sort_keys=True) + "\n").encode("utf-8")
