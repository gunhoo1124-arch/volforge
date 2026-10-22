"""Executable multi-instrument basket hedge replay with optional smile risks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from volforge.basket import (
    BasketAssetMarket,
    BasketMarket,
    BasketMonteCarloSettings,
    parse_basket_market,
    price_basket_option,
)
from volforge.basket_covariance import (
    COVARIANCE_SELECTION_COLUMNS,
    verify_basket_covariance_selection,
)
from volforge.basket_estimation import verify_basket_input_estimate
from volforge.basket_fixtures import (
    synthetic_hedge_instruments_json,
    synthetic_smile_hedge_instruments_json,
)
from volforge.basket_option_inputs import (
    BASKET_OPTION_QUOTE_PROVENANCE_COLUMNS,
    verify_option_quote_snapshot,
)
from volforge.basket_replay import (
    BASKET_REPLAY_COLUMNS,
    BASKET_REPLAY_PROVENANCE_COLUMNS,
)
from volforge.deliverables import DeliverableTerms, parse_deliverable_terms, value_option_settlement
from volforge.models.common import OptionType

_SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0
BASE_HEDGE_INSTRUMENTS = 3
SMILE_HEDGE_INSTRUMENTS = 5
MAX_HEDGE_JSON_BYTES = 1_048_576
ADVANCED_BASKET_REPLAY_COLUMNS = (*BASKET_REPLAY_COLUMNS, "hedge_instruments_json")
ADVANCED_BASKET_ESTIMATE_COLUMNS = (
    *BASKET_REPLAY_PROVENANCE_COLUMNS,
    "input_estimate_sha256",
    "input_estimate_json",
)
ADVANCED_BASKET_COVARIANCE_COLUMNS = COVARIANCE_SELECTION_COLUMNS
ADVANCED_BASKET_OPTION_QUOTE_COLUMNS = BASKET_OPTION_QUOTE_PROVENANCE_COLUMNS


class BasketRiskHedgeSchemaError(ValueError):
    """Raised when an advanced hedge replay input is malformed."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _timestamp(value: object, name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise BasketRiskHedgeSchemaError(f"{name} must be timezone-aware") from exc
    if pd.isna(result) or result.tzinfo is None:
        raise BasketRiskHedgeSchemaError(f"{name} must be timezone-aware")
    return result.tz_convert("UTC")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BasketRiskHedgeSchemaError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class BasketRiskHedgeSettings:
    """Risk bumps, solver limits, execution costs, and accounting controls."""

    paths: int = 8_000
    seed: int = 1701
    antithetic: bool = True
    control_variate: bool = True
    confidence_level: float = 0.95
    short_option_contracts: float = 1.0
    component_cost_bps: float = 1.0
    hedge_option_cost_bps: float = 5.0
    target_option_cost_bps: float = 5.0
    spot_scale_bump: float = 0.005
    volatility_bump: float = 0.0025
    correlation_shrink_bump: float = 0.02
    include_smile_risks: bool = False
    smile_slope_bump: float = 0.01
    smile_curvature_bump: float = 0.01
    maximum_absolute_smile_log_moneyness: float = 0.75
    minimum_normalized_singular_value: float = 1e-8
    maximum_condition_number: float = 1e6
    maximum_absolute_option_position: float = 100.0
    exposure_tolerance: float = 1e-6
    maximum_market_age_seconds: float = 900.0
    accounting_tolerance: float = 1e-7
    require_estimate_provenance: bool = True
    require_covariance_selection: bool = False
    require_option_quote_provenance: bool = False

    def __post_init__(self) -> None:
        BasketMonteCarloSettings(
            paths=self.paths,
            seed=self.seed,
            antithetic=self.antithetic,
            control_variate=self.control_variate,
            confidence_level=self.confidence_level,
        )
        positive = {
            "short_option_contracts": self.short_option_contracts,
            "spot_scale_bump": self.spot_scale_bump,
            "volatility_bump": self.volatility_bump,
            "correlation_shrink_bump": self.correlation_shrink_bump,
            "smile_slope_bump": self.smile_slope_bump,
            "smile_curvature_bump": self.smile_curvature_bump,
            "maximum_absolute_smile_log_moneyness": (self.maximum_absolute_smile_log_moneyness),
            "minimum_normalized_singular_value": self.minimum_normalized_singular_value,
            "maximum_condition_number": self.maximum_condition_number,
            "maximum_absolute_option_position": self.maximum_absolute_option_position,
            "exposure_tolerance": self.exposure_tolerance,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.spot_scale_bump >= 1.0:
            raise ValueError("spot_scale_bump must be below one")
        if self.correlation_shrink_bump > 1.0:
            raise ValueError("correlation_shrink_bump must be at most one")
        if self.minimum_normalized_singular_value > 1.0:
            raise ValueError("minimum_normalized_singular_value must be at most one")
        for value in (
            self.component_cost_bps,
            self.hedge_option_cost_bps,
            self.target_option_cost_bps,
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 10_000.0:
                raise ValueError("transaction costs must lie between zero and 10,000 bps")
        if (
            not math.isfinite(self.maximum_market_age_seconds)
            or self.maximum_market_age_seconds < 0.0
        ):
            raise ValueError("maximum_market_age_seconds must be non-negative and finite")
        if not math.isfinite(self.accounting_tolerance) or self.accounting_tolerance < 0.0:
            raise ValueError("accounting_tolerance must be non-negative and finite")
        if not isinstance(self.require_estimate_provenance, bool):
            raise ValueError("require_estimate_provenance must be boolean")
        if not isinstance(self.require_covariance_selection, bool):
            raise ValueError("require_covariance_selection must be boolean")
        if not isinstance(self.require_option_quote_provenance, bool):
            raise ValueError("require_option_quote_provenance must be boolean")
        if not isinstance(self.include_smile_risks, bool):
            raise ValueError("include_smile_risks must be boolean")
        if self.require_covariance_selection and not self.require_estimate_provenance:
            raise ValueError("covariance selection requires estimate provenance")

    @property
    def nonlinear_risk_factors(self) -> tuple[str, ...]:
        """Return the ordered nonlinear rows used by the exact hedge solve."""
        base = ("spot_scale_gamma", "parallel_vega")
        if self.include_smile_risks:
            return (*base, "smile_skew", "smile_curvature", "correlation_strength")
        return (*base, "correlation_strength")

    @property
    def hedge_instrument_count(self) -> int:
        """Return the exact instrument count required for the declared risk system."""
        return SMILE_HEDGE_INSTRUMENTS if self.include_smile_risks else BASE_HEDGE_INSTRUMENTS

    def monte_carlo(self, sequence: int) -> BasketMonteCarloSettings:
        """Return common-random-number settings for all risks in one snapshot."""
        return BasketMonteCarloSettings(
            paths=self.paths,
            seed=self.seed + sequence,
            antithetic=self.antithetic,
            control_variate=self.control_variate,
            confidence_level=self.confidence_level,
        )


@dataclass(frozen=True, slots=True)
class BasketRiskVector:
    """Component delta and aggregate level, smile, and correlation risks."""

    component_deltas: tuple[float, ...]
    spot_scale_gamma: float
    parallel_vega: float
    forward_log_moneyness: float | None
    smile_skew_sensitivity: float
    smile_curvature_sensitivity: float
    correlation_strength_sensitivity: float


@dataclass(frozen=True, slots=True)
class HedgeInstrument:
    """One explicit quoted basket option available as a hedge."""

    instrument_id: str
    option_type: str
    strike: float
    expiry_timestamp: str
    bid: float
    ask: float
    terms: DeliverableTerms

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


@dataclass(frozen=True, slots=True)
class HedgeInstrumentAudit:
    """Model risk, solver target, and execution for one quoted hedge option."""

    instrument_id: str
    option_type: str
    strike: float
    expiry_timestamp: str
    deliverable_sha256: str
    bid: float
    ask: float
    mid: float
    model_price: float
    component_deltas: tuple[float, ...]
    spot_scale_gamma: float
    parallel_vega: float
    forward_log_moneyness: float | None
    smile_skew_sensitivity: float
    smile_curvature_sensitivity: float
    correlation_strength_sensitivity: float
    prior_position: float
    target_position: float
    trade: float
    execution_price: float
    trade_cashflow: float
    spread_cost: float
    transaction_cost: float


@dataclass(frozen=True, slots=True)
class AdvancedComponentAudit:
    """Residual delta hedge and execution for one basket component."""

    asset_id: str
    spot: float
    bid: float
    ask: float
    net_option_delta_before_component: float
    prior_position: float
    target_position: float
    trade: float
    execution_price: float
    trade_cashflow: float
    spread_cost: float
    transaction_cost: float


@dataclass(frozen=True, slots=True)
class BasketRiskHedgeSnapshotAudit:
    """One hash-chained advanced hedge and accounting snapshot."""

    sequence: int
    formation_timestamp: str
    market_timestamp: str
    received_timestamp: str
    action: str
    source: str
    source_record_id: str
    snapshot_sha256: str
    market_sha256: str
    instrument_quote_sha256: str
    input_store_snapshot_sha256: str | None
    input_manifest_hashes: tuple[str, ...]
    input_estimate_sha256: str | None
    input_covariance_selection_sha256: str | None
    input_option_quote_snapshot_sha256: str | None
    configuration_sha256: str
    previous_audit_hash: str | None
    audit_hash: str
    no_lookahead_passed: bool
    maturity_years: float
    rate: float
    target_bid: float
    target_ask: float
    target_mid: float
    target_model_price: float
    target_component_deltas: tuple[float, ...]
    risk_factor_ids: tuple[str, ...]
    target_spot_scale_gamma: float
    target_parallel_vega: float
    target_forward_log_moneyness: float | None
    target_smile_skew_sensitivity: float
    target_smile_curvature_sensitivity: float
    target_correlation_strength_sensitivity: float
    settlement_payoff: float | None
    solver_rank: int | None
    solver_minimum_singular_value: float | None
    solver_condition_number: float | None
    normalized_solver_residual: float
    residual_component_deltas: tuple[float, ...]
    residual_spot_scale_gamma: float
    residual_parallel_vega: float
    residual_smile_skew_sensitivity: float
    residual_smile_curvature_sensitivity: float
    residual_correlation_strength_sensitivity: float
    target_option_cashflow: float
    hedge_option_trade_cashflow: float
    component_trade_cashflow: float
    cash_before: float
    cash_carry: float
    dividend_cashflow: float
    spread_cost: float
    cumulative_spread_cost: float
    transaction_cost: float
    cumulative_transaction_cost: float
    turnover: float
    cumulative_turnover: float
    cash: float
    portfolio_value: float
    incremental_pnl: float
    cumulative_pnl: float
    accounting_gap: float
    hedge_instruments: tuple[HedgeInstrumentAudit, ...]
    components: tuple[AdvancedComponentAudit, ...]


@dataclass(frozen=True, slots=True)
class BasketRiskHedgeResult:
    """Completed or blocked multi-instrument basket hedge replay."""

    settings: BasketRiskHedgeSettings
    option_type: str
    strike: float
    expiry_timestamp: str
    settlement_method: str
    classification: str
    contract_multiplier: float
    currency: str
    underlying_id: str | None
    deliverable_sha256: str
    configuration_sha256: str
    input_snapshot_count: int
    snapshots: tuple[BasketRiskHedgeSnapshotAudit, ...]
    completed: bool
    downstream_eligible: bool
    risk_factor_ids: tuple[str, ...]
    estimate_provenance_bound: bool
    covariance_selection_provenance_bound: bool
    option_quote_provenance_bound: bool
    smile_risk_coverage_bound: bool
    total_pnl: float | None
    total_spread_cost: float
    total_transaction_cost: float
    total_turnover: float
    maximum_absolute_accounting_gap: float | None
    maximum_normalized_exposure_residual: float | None
    final_audit_hash: str | None
    failure_reasons: tuple[str, ...]

    def ledger_frame(self) -> pd.DataFrame:
        """Return one flat row per processed formation snapshot."""
        rows: list[dict[str, Any]] = []
        for snapshot in self.snapshots:
            row = asdict(snapshot)
            row["hedge_instruments_json"] = _canonical_json(row.pop("hedge_instruments"))
            row["components_json"] = _canonical_json(row.pop("components"))
            rows.append(row)
        return pd.DataFrame(rows)

    def instrument_frame(self) -> pd.DataFrame:
        """Return one row per snapshot and explicit option hedge instrument."""
        return pd.DataFrame(
            [
                {
                    "sequence": snapshot.sequence,
                    "formation_timestamp": snapshot.formation_timestamp,
                    "action": snapshot.action,
                    **asdict(instrument),
                }
                for snapshot in self.snapshots
                for instrument in snapshot.hedge_instruments
            ]
        )

    def component_frame(self) -> pd.DataFrame:
        """Return one row per snapshot and residual component hedge."""
        return pd.DataFrame(
            [
                {
                    "sequence": snapshot.sequence,
                    "formation_timestamp": snapshot.formation_timestamp,
                    "action": snapshot.action,
                    **asdict(component),
                }
                for snapshot in self.snapshots
                for component in snapshot.components
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible result."""
        payload = asdict(self)
        payload["settings"] = asdict(self.settings)
        payload["snapshots"] = [asdict(snapshot) for snapshot in self.snapshots]
        return payload


def _market_with(
    market: BasketMarket,
    *,
    spot_scale: float = 1.0,
    volatility_shift: float = 0.0,
    correlation: np.ndarray | None = None,
) -> BasketMarket:
    return BasketMarket(
        assets=tuple(
            BasketAssetMarket(
                asset_id=asset.asset_id,
                spot=asset.spot * spot_scale,
                volatility=max(asset.volatility + volatility_shift, 0.0),
                dividend_yield=asset.dividend_yield,
            )
            for asset in market.assets
        ),
        correlation=tuple(
            tuple(float(value) for value in row)
            for row in (
                np.asarray(market.correlation, dtype=float) if correlation is None else correlation
            )
        ),
    )


def basket_option_risk(
    terms: DeliverableTerms,
    market: BasketMarket,
    *,
    option_type: str | OptionType,
    strike: float,
    maturity: float,
    rate: float,
    settings: BasketRiskHedgeSettings,
    sequence: int,
) -> tuple[float, BasketRiskVector]:
    """Calculate common-random-number level, smile, and correlation risks."""
    mc = settings.monte_carlo(sequence)

    def price(candidate: BasketMarket) -> float:
        result = price_basket_option(
            terms,
            candidate,
            option_type=option_type,
            strike=strike,
            maturity=maturity,
            rate=rate,
            settings=mc,
        )
        if not result.bounds_passed:
            raise ArithmeticError("basket price failed discounted bounds during risk estimation")
        return result.price

    base_result = price_basket_option(
        terms,
        market,
        option_type=option_type,
        strike=strike,
        maturity=maturity,
        rate=rate,
        settings=mc,
    )
    if not base_result.bounds_passed:
        raise ArithmeticError("basket price failed discounted bounds during risk estimation")
    strike_amount = strike * terms.contract_multiplier
    forward_deliverable = base_result.expected_discounted_deliverable * math.exp(rate * maturity)
    forward_log_moneyness = (
        math.log(strike_amount / forward_deliverable)
        if strike_amount > 0.0 and forward_deliverable > 0.0
        else None
    )
    if settings.include_smile_risks and forward_log_moneyness is None:
        raise ArithmeticError("smile risks require positive strike and forward deliverable")
    if (
        settings.include_smile_risks
        and abs(float(forward_log_moneyness)) > settings.maximum_absolute_smile_log_moneyness
    ):
        raise ArithmeticError("option exceeds maximum absolute smile log-moneyness")
    spot_bump = settings.spot_scale_bump
    spot_up = price(_market_with(market, spot_scale=1.0 + spot_bump))
    spot_down = price(_market_with(market, spot_scale=1.0 - spot_bump))
    gamma = (spot_up - 2.0 * base_result.price + spot_down) / spot_bump**2
    minimum_volatility = min(asset.volatility for asset in market.assets)
    vol_bump = min(settings.volatility_bump, max(0.5 * minimum_volatility, 1e-8))
    vol_up = price(_market_with(market, volatility_shift=vol_bump))
    vol_down = price(_market_with(market, volatility_shift=-vol_bump))
    vega = (vol_up - vol_down) / (2.0 * vol_bump)
    smile_skew = 0.0
    smile_curvature = 0.0
    if settings.include_smile_risks:
        minimum_volatility = min(asset.volatility for asset in market.assets)

        def smile_sensitivity(coefficient_bump: float, shape: float) -> float:
            if abs(shape) <= 1e-12:
                return 0.0
            effective_bump = min(
                coefficient_bump,
                max(0.5 * minimum_volatility / abs(shape), 1e-8),
            )
            shift = effective_bump * shape
            up = price(_market_with(market, volatility_shift=shift))
            down = price(_market_with(market, volatility_shift=-shift))
            return (up - down) / (2.0 * effective_bump)

        smile_skew = smile_sensitivity(settings.smile_slope_bump, float(forward_log_moneyness))
        smile_curvature = smile_sensitivity(
            settings.smile_curvature_bump, float(forward_log_moneyness) ** 2
        )
    correlation = np.asarray(market.correlation, dtype=float)
    shrink = settings.correlation_shrink_bump
    shrunk = (1.0 - shrink) * correlation + shrink * np.eye(len(correlation))
    correlation_sensitivity = (
        base_result.price - price(_market_with(market, correlation=shrunk))
    ) / shrink
    risk = BasketRiskVector(
        component_deltas=tuple(delta.delta for delta in base_result.deltas),
        spot_scale_gamma=float(gamma),
        parallel_vega=float(vega),
        forward_log_moneyness=forward_log_moneyness,
        smile_skew_sensitivity=float(smile_skew),
        smile_curvature_sensitivity=float(smile_curvature),
        correlation_strength_sensitivity=float(correlation_sensitivity),
    )
    return base_result.price, risk


def _nonlinear_risk_values(
    risk: BasketRiskVector,
    *,
    include_smile_risks: bool,
) -> tuple[float, ...]:
    values = (risk.spot_scale_gamma, risk.parallel_vega)
    if include_smile_risks:
        return (
            *values,
            risk.smile_skew_sensitivity,
            risk.smile_curvature_sensitivity,
            risk.correlation_strength_sensitivity,
        )
    return (*values, risk.correlation_strength_sensitivity)


def _parse_execution_quotes(value: object, market: BasketMarket) -> dict[str, tuple[float, float]]:
    if not isinstance(value, str) or len(value.encode("utf-8")) > MAX_HEDGE_JSON_BYTES:
        raise BasketRiskHedgeSchemaError("execution_json must be a bounded JSON string")
    try:
        parsed = json.loads(value, object_pairs_hook=_strict_object)
    except json.JSONDecodeError as exc:
        raise BasketRiskHedgeSchemaError("execution_json is not valid JSON") from exc
    if not isinstance(parsed, list):
        raise BasketRiskHedgeSchemaError("execution_json must be a JSON list")
    quotes: dict[str, tuple[float, float]] = {}
    for position, item in enumerate(parsed):
        if not isinstance(item, dict) or set(item) != {"asset_id", "bid", "ask"}:
            raise BasketRiskHedgeSchemaError(f"execution quote {position} has invalid fields")
        asset_id = str(item["asset_id"]).strip()
        bid = _finite(item["bid"], f"execution quote {position} bid")
        ask = _finite(item["ask"], f"execution quote {position} ask")
        if not asset_id or asset_id in quotes or bid <= 0.0 or ask < bid:
            raise BasketRiskHedgeSchemaError(f"execution quote {position} is invalid")
        quotes[asset_id] = (bid, ask)
    if tuple(sorted(quotes)) != market.asset_ids:
        raise BasketRiskHedgeSchemaError("execution quotes must match basket assets exactly")
    for asset in market.assets:
        bid, ask = quotes[asset.asset_id]
        if not bid <= asset.spot <= ask:
            raise BasketRiskHedgeSchemaError("component spot must lie inside executable bid/ask")
    return quotes


def _parse_hedge_instruments(
    value: object,
    *,
    expected_count: int,
) -> tuple[HedgeInstrument, ...]:
    if not isinstance(value, str) or len(value.encode("utf-8")) > MAX_HEDGE_JSON_BYTES:
        raise BasketRiskHedgeSchemaError("hedge_instruments_json must be a bounded JSON string")
    try:
        parsed = json.loads(value, object_pairs_hook=_strict_object)
    except json.JSONDecodeError as exc:
        raise BasketRiskHedgeSchemaError("hedge_instruments_json is not valid JSON") from exc
    if not isinstance(parsed, list) or len(parsed) != expected_count:
        raise BasketRiskHedgeSchemaError(f"exactly {expected_count} hedge instruments are required")
    expected = {
        "instrument_id",
        "option_type",
        "strike",
        "expiry_timestamp",
        "bid",
        "ask",
        "settlement_method",
        "contract_multiplier",
        "currency",
        "underlying_id",
        "deliverable_json",
    }
    instruments: list[HedgeInstrument] = []
    seen: set[str] = set()
    for position, item in enumerate(parsed):
        if not isinstance(item, dict) or set(item) != expected:
            raise BasketRiskHedgeSchemaError(f"hedge instrument {position} has invalid fields")
        instrument_id = str(item["instrument_id"]).strip()
        if not instrument_id or instrument_id in seen:
            raise BasketRiskHedgeSchemaError("hedge instrument identifiers must be unique")
        seen.add(instrument_id)
        kind = OptionType.parse(str(item["option_type"]))
        strike = _finite(item["strike"], f"hedge instrument {position} strike")
        bid = _finite(item["bid"], f"hedge instrument {position} bid")
        ask = _finite(item["ask"], f"hedge instrument {position} ask")
        if strike < 0.0 or bid < 0.0 or ask < bid:
            raise BasketRiskHedgeSchemaError(f"hedge instrument {position} quote is invalid")
        expiry = _timestamp(item["expiry_timestamp"], f"hedge instrument {position} expiry")
        terms = parse_deliverable_terms(
            item["deliverable_json"],
            settlement_method=str(item["settlement_method"]),
            contract_multiplier=_finite(
                item["contract_multiplier"], f"hedge instrument {position} multiplier"
            ),
            currency=str(item["currency"]),
            underlying_id=(None if item["underlying_id"] is None else str(item["underlying_id"])),
        )
        instruments.append(
            HedgeInstrument(
                instrument_id=instrument_id,
                option_type=kind.value,
                strike=strike,
                expiry_timestamp=expiry.isoformat(),
                bid=bid,
                ask=ask,
                terms=terms,
            )
        )
    return tuple(instruments)


def _terminal_payoff(
    terms: DeliverableTerms,
    market: BasketMarket,
    *,
    option_type: OptionType,
    strike: float,
) -> float:
    if terms.settlement_method == "cash":
        return value_option_settlement(
            terms,
            option_type=option_type,
            strike=strike,
            settlement_level=market.assets[0].spot,
        ).net_payoff
    return value_option_settlement(
        terms,
        option_type=option_type,
        strike=strike,
        security_marks={asset.asset_id: asset.spot for asset in market.assets},
    ).net_payoff


def _estimate_provenance(
    row: pd.Series,
    *,
    supplied: bool,
) -> tuple[str | None, tuple[str, ...], str | None]:
    if not supplied:
        return None, (), None
    snapshot_hash = str(row["input_store_snapshot_sha256"])
    estimate_hash = str(row["input_estimate_sha256"])
    if len(snapshot_hash) != 64 or len(estimate_hash) != 64:
        raise ValueError("estimate provenance hashes must be 64-character digests")
    try:
        int(snapshot_hash, 16)
        int(estimate_hash, 16)
        manifests = json.loads(str(row["input_manifest_hashes_json"]))
        estimate_payload = json.loads(str(row["input_estimate_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("estimate provenance contains invalid JSON or hashes") from exc
    if (
        not isinstance(manifests, list)
        or not manifests
        or len(manifests) != len(set(manifests))
        or any(not isinstance(item, str) or len(item) != 64 for item in manifests)
    ):
        raise ValueError("input_manifest_hashes_json must contain unique SHA-256 hashes")
    if not verify_basket_input_estimate(estimate_payload):
        raise ValueError("input estimate failed its self-hash verification")
    if estimate_payload.get("estimate_sha256") != estimate_hash:
        raise ValueError("input estimate hash does not match its provenance column")
    if estimate_payload.get("input_store_snapshot_sha256") != snapshot_hash:
        raise ValueError("input estimate store snapshot identity does not match")
    if estimate_payload.get("input_manifest_hashes") != manifests:
        raise ValueError("input estimate manifest identities do not match")
    return snapshot_hash, tuple(manifests), estimate_hash


def _covariance_selection_provenance(
    row: pd.Series,
    *,
    supplied: bool,
) -> tuple[str | None, dict[str, Any] | None]:
    if not supplied:
        return None, None
    selection_hash = str(row["input_covariance_selection_sha256"])
    if len(selection_hash) != 64:
        raise ValueError("covariance selection hash must be a 64-character digest")
    try:
        int(selection_hash, 16)
        payload = json.loads(str(row["input_covariance_selection_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("covariance selection provenance contains invalid JSON or hash") from exc
    if not verify_basket_covariance_selection(payload):
        raise ValueError("covariance selection failed its self-hash verification")
    if payload.get("selection_sha256") != selection_hash:
        raise ValueError("covariance selection hash does not match its provenance column")
    if not payload.get("downstream_eligible"):
        raise ValueError("covariance selection is not downstream eligible")
    return selection_hash, payload


def _option_quote_provenance(
    row: pd.Series,
    *,
    supplied: bool,
    formation: pd.Timestamp,
    option_type: str,
    strike: float,
    expiry: pd.Timestamp,
    terms: DeliverableTerms,
    target_bid: float,
    target_ask: float,
) -> str | None:
    if not supplied:
        return None
    snapshot_hash = str(row["input_option_quote_snapshot_sha256"])
    if len(snapshot_hash) != 64:
        raise ValueError("option-quote snapshot hash must be a 64-character digest")
    try:
        int(snapshot_hash, 16)
        payload = json.loads(str(row["input_option_quote_snapshot_json"]))
        manifests = json.loads(str(row["input_option_quote_manifest_hashes_json"]))
        hedge_payload = json.loads(str(row["hedge_instruments_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("option-quote provenance contains invalid JSON or hashes") from exc
    if not verify_option_quote_snapshot(payload):
        raise ValueError("option-quote snapshot failed its self-hash verification")
    if payload.get("snapshot_sha256") != snapshot_hash:
        raise ValueError("option-quote snapshot hash does not match its provenance column")
    if (
        not isinstance(manifests, list)
        or not manifests
        or len(manifests) != len(set(manifests))
        or any(not isinstance(item, str) or len(item) != 64 for item in manifests)
        or payload.get("manifest_hashes") != manifests
    ):
        raise ValueError("option-quote manifest identities are invalid or do not match")
    if (
        payload.get("formation_timestamp") != formation.isoformat()
        or payload.get("target_option_type") != option_type
        or not math.isclose(
            _finite(payload.get("target_strike"), "target strike"), strike, abs_tol=1e-12
        )
        or payload.get("target_expiry_timestamp") != expiry.isoformat()
        or payload.get("target_settlement_method") != terms.settlement_method
        or not math.isclose(
            _finite(payload.get("target_contract_multiplier"), "target contract multiplier"),
            terms.contract_multiplier,
            abs_tol=1e-12,
        )
        or payload.get("target_currency") != terms.currency
        or payload.get("target_underlying_id") != terms.underlying_id
        or payload.get("target_deliverable_sha256") != terms.sha256
        or not math.isclose(
            _finite(payload.get("target_bid"), "target bid"), target_bid, abs_tol=1e-12
        )
        or not math.isclose(
            _finite(payload.get("target_ask"), "target ask"), target_ask, abs_tol=1e-12
        )
    ):
        raise ValueError("option-quote snapshot does not bind the target contract and quote")
    try:
        snapshot_hedges = json.loads(str(payload.get("hedge_instruments_json")))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("option-quote snapshot hedge payload is invalid") from exc
    if _canonical_json(snapshot_hedges) != _canonical_json(hedge_payload):
        raise ValueError("option-quote snapshot does not bind the hedge quotes")
    if payload.get("quote_count") != len(hedge_payload) + 1:
        raise ValueError("option-quote snapshot quote count does not match the replay")
    return snapshot_hash


def _configuration_hash(
    *,
    settings: BasketRiskHedgeSettings,
    option_type: str,
    strike: float,
    expiry_timestamp: str,
    terms: DeliverableTerms,
    input_snapshot_count: int,
) -> str:
    return _sha256(
        {
            "settings": asdict(settings),
            "option_type": option_type,
            "strike": strike,
            "expiry_timestamp": expiry_timestamp,
            "settlement_method": terms.settlement_method,
            "classification": terms.classification,
            "contract_multiplier": terms.contract_multiplier,
            "currency": terms.currency,
            "underlying_id": terms.underlying_id,
            "deliverable_sha256": terms.sha256,
            "input_snapshot_count": input_snapshot_count,
        }
    )


def _audit_hash(snapshot: BasketRiskHedgeSnapshotAudit) -> str:
    payload = asdict(snapshot)
    payload.pop("audit_hash")
    return _sha256(payload)


def verify_basket_risk_hedge_hash_chain(result: BasketRiskHedgeResult) -> bool:
    """Verify configuration identity, predecessor links, summaries, and audit hashes."""
    expected_configuration = _sha256(
        {
            "settings": asdict(result.settings),
            "option_type": result.option_type,
            "strike": result.strike,
            "expiry_timestamp": result.expiry_timestamp,
            "settlement_method": result.settlement_method,
            "classification": result.classification,
            "contract_multiplier": result.contract_multiplier,
            "currency": result.currency,
            "underlying_id": result.underlying_id,
            "deliverable_sha256": result.deliverable_sha256,
            "input_snapshot_count": result.input_snapshot_count,
        }
    )
    if result.configuration_sha256 != expected_configuration:
        return False
    previous: str | None = None
    for sequence, snapshot in enumerate(result.snapshots):
        if (
            snapshot.sequence != sequence
            or snapshot.previous_audit_hash != previous
            or snapshot.configuration_sha256 != expected_configuration
            or snapshot.audit_hash != _audit_hash(snapshot)
        ):
            return False
        previous = snapshot.audit_hash
    expected_completed = (
        len(result.snapshots) == result.input_snapshot_count and not result.failure_reasons
    )
    if result.completed != expected_completed or result.downstream_eligible != expected_completed:
        return False
    if result.risk_factor_ids != result.settings.nonlinear_risk_factors:
        return False
    estimate_bound = bool(result.snapshots) and all(
        snapshot.input_estimate_sha256 is not None for snapshot in result.snapshots
    )
    if result.estimate_provenance_bound != estimate_bound or result.final_audit_hash != previous:
        return False
    covariance_bound = bool(result.snapshots) and all(
        snapshot.input_covariance_selection_sha256 is not None for snapshot in result.snapshots
    )
    if result.covariance_selection_provenance_bound != covariance_bound:
        return False
    option_quote_bound = bool(result.snapshots) and all(
        snapshot.input_option_quote_snapshot_sha256 is not None for snapshot in result.snapshots
    )
    if result.option_quote_provenance_bound != option_quote_bound:
        return False
    expected_smile_bound = bool(
        expected_completed
        and result.settings.include_smile_risks
        and all(
            snapshot.risk_factor_ids == result.settings.nonlinear_risk_factors
            and snapshot.solver_rank
            == (None if snapshot.action == "settle" else SMILE_HEDGE_INSTRUMENTS)
            for snapshot in result.snapshots
        )
    )
    if result.smile_risk_coverage_bound != expected_smile_bound:
        return False
    expected_spread = result.snapshots[-1].cumulative_spread_cost if result.snapshots else 0.0
    expected_cost = result.snapshots[-1].cumulative_transaction_cost if result.snapshots else 0.0
    expected_turnover = result.snapshots[-1].cumulative_turnover if result.snapshots else 0.0
    expected_gap = (
        max(abs(snapshot.accounting_gap) for snapshot in result.snapshots)
        if result.snapshots
        else None
    )
    expected_residual = (
        max(snapshot.normalized_solver_residual for snapshot in result.snapshots)
        if result.snapshots
        else None
    )
    return (
        math.isclose(result.total_spread_cost, expected_spread, abs_tol=1e-12)
        and math.isclose(result.total_transaction_cost, expected_cost, abs_tol=1e-12)
        and math.isclose(result.total_turnover, expected_turnover, abs_tol=1e-12)
        and result.maximum_absolute_accounting_gap == expected_gap
        and result.maximum_normalized_exposure_residual == expected_residual
        and result.total_pnl
        == (result.snapshots[-1].portfolio_value if expected_completed else None)
    )


def run_basket_risk_hedge_replay(
    frame: pd.DataFrame,
    terms: DeliverableTerms,
    *,
    option_type: str | OptionType,
    strike: float,
    expiry_timestamp: str | pd.Timestamp,
    settings: BasketRiskHedgeSettings | None = None,
) -> BasketRiskHedgeResult:
    """Replay a self-financing delta/gamma/vega/correlation hedge portfolio."""
    settings = settings or BasketRiskHedgeSettings()
    kind = OptionType.parse(option_type)
    normalized_strike = _finite(strike, "strike")
    if normalized_strike < 0.0:
        raise ValueError("strike must be non-negative")
    expiry = _timestamp(expiry_timestamp, "expiry_timestamp")
    missing = sorted(set(ADVANCED_BASKET_REPLAY_COLUMNS) - set(frame.columns))
    if missing:
        raise BasketRiskHedgeSchemaError(
            "missing advanced basket replay columns: " + ", ".join(missing)
        )
    if not 2 <= len(frame) <= 10_000:
        raise BasketRiskHedgeSchemaError("advanced basket replay requires 2 to 10,000 rows")
    supplied_estimate_columns = set(ADVANCED_BASKET_ESTIMATE_COLUMNS).intersection(frame.columns)
    if supplied_estimate_columns and supplied_estimate_columns != set(
        ADVANCED_BASKET_ESTIMATE_COLUMNS
    ):
        missing_provenance = sorted(
            set(ADVANCED_BASKET_ESTIMATE_COLUMNS) - supplied_estimate_columns
        )
        raise BasketRiskHedgeSchemaError(
            "incomplete estimated-input provenance columns: " + ", ".join(missing_provenance)
        )
    has_estimate_provenance = bool(supplied_estimate_columns)
    if settings.require_estimate_provenance and not has_estimate_provenance:
        raise BasketRiskHedgeSchemaError("estimated-input provenance is required")
    supplied_covariance_columns = set(ADVANCED_BASKET_COVARIANCE_COLUMNS).intersection(
        frame.columns
    )
    if supplied_covariance_columns and supplied_covariance_columns != set(
        ADVANCED_BASKET_COVARIANCE_COLUMNS
    ):
        missing_provenance = sorted(
            set(ADVANCED_BASKET_COVARIANCE_COLUMNS) - supplied_covariance_columns
        )
        raise BasketRiskHedgeSchemaError(
            "incomplete covariance-selection provenance columns: " + ", ".join(missing_provenance)
        )
    has_covariance_provenance = bool(supplied_covariance_columns)
    if settings.require_covariance_selection and not has_covariance_provenance:
        raise BasketRiskHedgeSchemaError("covariance-selection provenance is required")
    if has_covariance_provenance and not has_estimate_provenance:
        raise BasketRiskHedgeSchemaError(
            "covariance-selection provenance requires estimated-input provenance"
        )
    supplied_option_quote_columns = set(ADVANCED_BASKET_OPTION_QUOTE_COLUMNS).intersection(
        frame.columns
    )
    if supplied_option_quote_columns and supplied_option_quote_columns != set(
        ADVANCED_BASKET_OPTION_QUOTE_COLUMNS
    ):
        missing_provenance = sorted(
            set(ADVANCED_BASKET_OPTION_QUOTE_COLUMNS) - supplied_option_quote_columns
        )
        raise BasketRiskHedgeSchemaError(
            "incomplete option-quote provenance columns: " + ", ".join(missing_provenance)
        )
    has_option_quote_provenance = bool(supplied_option_quote_columns)
    if settings.require_option_quote_provenance and not has_option_quote_provenance:
        raise BasketRiskHedgeSchemaError("option-quote provenance is required")
    formations = [
        _timestamp(value, f"formation_timestamp row {position}")
        for position, value in enumerate(frame["formation_timestamp"])
    ]
    if any(right <= left for left, right in pairwise(formations)):
        raise BasketRiskHedgeSchemaError("formation timestamps must be strictly increasing")
    if formations[-1] != expiry or formations[0] >= expiry:
        raise BasketRiskHedgeSchemaError(
            "the final formation timestamp must equal expiry after a pre-expiry opening"
        )
    configuration_hash = _configuration_hash(
        settings=settings,
        option_type=kind.value,
        strike=normalized_strike,
        expiry_timestamp=expiry.isoformat(),
        terms=terms,
        input_snapshot_count=len(frame),
    )

    snapshots: list[BasketRiskHedgeSnapshotAudit] = []
    failures: list[str] = []
    component_positions: dict[str, float] = {}
    option_positions: dict[str, float] = {}
    cash = 0.0
    cumulative_spread = 0.0
    cumulative_cost = 0.0
    cumulative_turnover = 0.0
    previous_value = 0.0
    previous_market: BasketMarket | None = None
    previous_formation: pd.Timestamp | None = None
    previous_rate = 0.0
    previous_audit_hash: str | None = None
    expected_assets: tuple[str, ...] | None = None
    expected_instruments: tuple[tuple[str, str, float, str, str], ...] | None = None

    for sequence, (_, row) in enumerate(frame.iterrows()):
        formation = formations[sequence]
        try:
            market_timestamp = _timestamp(row["market_timestamp"], "market_timestamp")
            received = _timestamp(row["received_timestamp"], "received_timestamp")
            if not market_timestamp <= received <= formation:
                raise ValueError(
                    "market_timestamp <= received_timestamp <= formation_timestamp is required"
                )
            age = (formation - market_timestamp).total_seconds()
            if age > settings.maximum_market_age_seconds:
                raise ValueError("market package exceeds maximum_market_age_seconds")
            source = str(row["source"]).strip()
            source_record_id = str(row["source_record_id"]).strip()
            if not source or not source_record_id:
                raise ValueError("source identities must be non-empty")
            market = parse_basket_market(row["market_json"])
            execution_quotes = _parse_execution_quotes(row["execution_json"], market)
            instruments = _parse_hedge_instruments(
                row["hedge_instruments_json"],
                expected_count=settings.hedge_instrument_count,
            )
            if expected_assets is None:
                expected_assets = market.asset_ids
            elif market.asset_ids != expected_assets:
                raise ValueError("basket risk-factor identities cannot change during replay")
            instrument_identity = tuple(
                (
                    instrument.instrument_id,
                    instrument.option_type,
                    instrument.strike,
                    instrument.expiry_timestamp,
                    instrument.terms.sha256,
                )
                for instrument in instruments
            )
            if expected_instruments is None:
                expected_instruments = instrument_identity
            elif instrument_identity != expected_instruments:
                raise ValueError("hedge instrument definitions cannot change during replay")
            for instrument in instruments:
                if _timestamp(instrument.expiry_timestamp, "instrument expiry") <= expiry:
                    raise ValueError("hedge instruments must expire after the target option")
                if instrument.terms.required_security_marks != market.asset_ids:
                    raise ValueError("hedge instrument risk factors must match the basket market")
            snapshot_hash, manifest_hashes, estimate_hash = _estimate_provenance(
                row, supplied=has_estimate_provenance
            )
            covariance_selection_hash, covariance_payload = _covariance_selection_provenance(
                row, supplied=has_covariance_provenance
            )
            if has_estimate_provenance:
                estimate_payload = json.loads(str(row["input_estimate_json"]))
                if (
                    estimate_payload.get("market_json") != market.canonical_json
                    or estimate_payload.get("formation_timestamp") != formation.isoformat()
                ):
                    raise ValueError("input estimate does not bind this market and formation")
                if covariance_payload is not None and (
                    covariance_payload.get("formation_timestamp") != formation.isoformat()
                    or covariance_payload.get("basket_id") != estimate_payload.get("basket_id")
                    or covariance_payload.get("source") != estimate_payload.get("source")
                    or covariance_payload.get("selected_estimator")
                    != estimate_payload.get("settings", {}).get("estimator")
                ):
                    raise ValueError(
                        "covariance selection does not bind this estimate and formation"
                    )
            target_bid = _finite(row["option_bid"], "option_bid")
            target_ask = _finite(row["option_ask"], "option_ask")
            if target_bid < 0.0 or target_ask < target_bid:
                raise ValueError("target option requires 0 <= bid <= ask")
            option_quote_snapshot_hash = _option_quote_provenance(
                row,
                supplied=has_option_quote_provenance,
                formation=formation,
                option_type=kind.value,
                strike=normalized_strike,
                expiry=expiry,
                terms=terms,
                target_bid=target_bid,
                target_ask=target_ask,
            )
            target_mid = 0.5 * (target_bid + target_ask)
            rate = _finite(row["risk_free_rate"], "risk_free_rate")
            maturity = (expiry - formation).total_seconds() / _SECONDS_PER_YEAR
            final = formation == expiry
            settlement_payoff = (
                _terminal_payoff(
                    terms,
                    market,
                    option_type=kind,
                    strike=normalized_strike,
                )
                if final
                else None
            )
            if final and (
                abs(target_bid - float(settlement_payoff)) > settings.accounting_tolerance
                or abs(target_ask - float(settlement_payoff)) > settings.accounting_tolerance
            ):
                raise ValueError("expiry target quote must equal explicit settlement payoff")

            if final:
                target_model_price = float(settlement_payoff)
                target_risk = BasketRiskVector(
                    component_deltas=tuple(0.0 for _ in market.assets),
                    spot_scale_gamma=0.0,
                    parallel_vega=0.0,
                    forward_log_moneyness=None,
                    smile_skew_sensitivity=0.0,
                    smile_curvature_sensitivity=0.0,
                    correlation_strength_sensitivity=0.0,
                )
            else:
                target_model_price, target_risk = basket_option_risk(
                    terms,
                    market,
                    option_type=kind,
                    strike=normalized_strike,
                    maturity=maturity,
                    rate=rate,
                    settings=settings,
                    sequence=sequence,
                )

            instrument_models: list[tuple[float, BasketRiskVector]] = []
            for instrument in instruments:
                instrument_maturity = (
                    _timestamp(instrument.expiry_timestamp, "instrument expiry") - formation
                ).total_seconds() / _SECONDS_PER_YEAR
                instrument_models.append(
                    basket_option_risk(
                        instrument.terms,
                        market,
                        option_type=instrument.option_type,
                        strike=instrument.strike,
                        maturity=instrument_maturity,
                        rate=rate,
                        settings=settings,
                        sequence=sequence,
                    )
                )

            solver_rank: int | None = None
            minimum_singular_value: float | None = None
            condition_number: float | None = None
            normalized_residual = 0.0
            if final:
                target_option_positions = {
                    instrument.instrument_id: 0.0 for instrument in instruments
                }
                residual_gamma = residual_vega = residual_correlation = 0.0
                residual_skew = residual_curvature = 0.0
                residual_deltas = tuple(0.0 for _ in market.assets)
            else:
                matrix = np.asarray(
                    [
                        _nonlinear_risk_values(
                            model[1],
                            include_smile_risks=settings.include_smile_risks,
                        )
                        for model in instrument_models
                    ],
                    dtype=float,
                ).T
                target = settings.short_option_contracts * np.asarray(
                    _nonlinear_risk_values(
                        target_risk,
                        include_smile_risks=settings.include_smile_risks,
                    ),
                    dtype=float,
                )
                scales = np.maximum(np.max(np.abs(matrix), axis=1), np.abs(target))
                if np.any(scales <= 1e-12):
                    raise ArithmeticError("nonlinear hedge risk row is numerically empty")
                normalized_matrix = matrix / scales[:, None]
                normalized_target = target / scales
                singular_values = np.linalg.svd(normalized_matrix, compute_uv=False)
                minimum_singular_value = float(singular_values[-1])
                solver_rank = int(
                    np.sum(singular_values >= settings.minimum_normalized_singular_value)
                )
                if solver_rank != len(settings.nonlinear_risk_factors):
                    raise ArithmeticError(
                        f"hedge risk matrix rank {solver_rank} does not span "
                        f"{len(settings.nonlinear_risk_factors)} declared risks"
                    )
                condition_number = float(singular_values[0] / singular_values[-1])
                if (
                    not math.isfinite(condition_number)
                    or condition_number > settings.maximum_condition_number
                ):
                    raise ArithmeticError(
                        f"hedge risk matrix condition number {condition_number:.6g} exceeds limit"
                    )
                option_targets = np.linalg.solve(normalized_matrix, normalized_target)
                if np.max(np.abs(option_targets)) > settings.maximum_absolute_option_position:
                    raise ArithmeticError("hedge solution exceeds the option position limit")
                residual = matrix @ option_targets - target
                normalized_residual = float(np.max(np.abs(residual / scales)))
                if normalized_residual > settings.exposure_tolerance:
                    raise ArithmeticError("nonlinear hedge residual exceeds tolerance")
                target_option_positions = {
                    instrument.instrument_id: float(option_targets[index])
                    for index, instrument in enumerate(instruments)
                }
                residual_gamma = float(residual[0])
                residual_vega = float(residual[1])
                if settings.include_smile_risks:
                    residual_skew = float(residual[2])
                    residual_curvature = float(residual[3])
                    residual_correlation = float(residual[4])
                else:
                    residual_skew = residual_curvature = 0.0
                    residual_correlation = float(residual[2])
                residual_deltas = tuple(
                    -settings.short_option_contracts * target_risk.component_deltas[index]
                    + sum(
                        option_targets[instrument_index]
                        * instrument_models[instrument_index][1].component_deltas[index]
                        for instrument_index in range(len(instruments))
                    )
                    for index in range(len(market.assets))
                )

            cash_before = cash
            cash_carry = 0.0
            dividend_cashflow = 0.0
            if previous_formation is not None and previous_market is not None:
                elapsed = (formation - previous_formation).total_seconds() / _SECONDS_PER_YEAR
                cash_carry = cash * math.expm1(previous_rate * elapsed)
                previous_assets = {asset.asset_id: asset for asset in previous_market.assets}
                dividend_cashflow = sum(
                    component_positions.get(asset_id, 0.0)
                    * asset.spot
                    * math.expm1(asset.dividend_yield * elapsed)
                    for asset_id, asset in previous_assets.items()
                )
                cash += cash_carry + dividend_cashflow

            target_cashflow = 0.0
            target_spread = 0.0
            target_transaction_cost = 0.0
            if sequence == 0:
                target_cashflow = settings.short_option_contracts * target_bid
                target_spread = settings.short_option_contracts * (target_mid - target_bid)
                target_transaction_cost = (
                    settings.short_option_contracts
                    * target_bid
                    * settings.target_option_cost_bps
                    / 10_000.0
                )
            elif final:
                target_cashflow = -settings.short_option_contracts * float(settlement_payoff)

            instrument_audits: list[HedgeInstrumentAudit] = []
            hedge_cashflow = hedge_spread = hedge_transaction_cost = hedge_turnover = 0.0
            for instrument, (model_price, risk) in zip(instruments, instrument_models, strict=True):
                prior = option_positions.get(instrument.instrument_id, 0.0)
                target_position = target_option_positions[instrument.instrument_id]
                trade = target_position - prior
                execution_price = (
                    instrument.ask
                    if trade > 0.0
                    else instrument.bid
                    if trade < 0.0
                    else instrument.mid
                )
                trade_cashflow = -trade * execution_price
                spread_cost = -(-trade * execution_price + trade * instrument.mid)
                transaction_cost = (
                    abs(trade * execution_price) * settings.hedge_option_cost_bps / 10_000.0
                )
                hedge_cashflow += trade_cashflow
                hedge_spread += spread_cost
                hedge_transaction_cost += transaction_cost
                hedge_turnover += abs(trade * execution_price)
                instrument_audits.append(
                    HedgeInstrumentAudit(
                        instrument_id=instrument.instrument_id,
                        option_type=instrument.option_type,
                        strike=instrument.strike,
                        expiry_timestamp=instrument.expiry_timestamp,
                        deliverable_sha256=instrument.terms.sha256,
                        bid=instrument.bid,
                        ask=instrument.ask,
                        mid=instrument.mid,
                        model_price=model_price,
                        component_deltas=risk.component_deltas,
                        spot_scale_gamma=risk.spot_scale_gamma,
                        parallel_vega=risk.parallel_vega,
                        forward_log_moneyness=risk.forward_log_moneyness,
                        smile_skew_sensitivity=risk.smile_skew_sensitivity,
                        smile_curvature_sensitivity=risk.smile_curvature_sensitivity,
                        correlation_strength_sensitivity=risk.correlation_strength_sensitivity,
                        prior_position=prior,
                        target_position=target_position,
                        trade=trade,
                        execution_price=execution_price,
                        trade_cashflow=trade_cashflow,
                        spread_cost=spread_cost,
                        transaction_cost=transaction_cost,
                    )
                )

            component_audits: list[AdvancedComponentAudit] = []
            component_cashflow = component_spread = component_cost = component_turnover = 0.0
            component_targets = {
                asset.asset_id: (0.0 if final else -residual_deltas[index])
                for index, asset in enumerate(market.assets)
            }
            final_residual_deltas: list[float] = []
            for index, asset in enumerate(market.assets):
                bid, ask = execution_quotes[asset.asset_id]
                prior = component_positions.get(asset.asset_id, 0.0)
                target_position = component_targets[asset.asset_id]
                trade = target_position - prior
                execution_price = ask if trade > 0.0 else bid if trade < 0.0 else asset.spot
                trade_cashflow = -trade * execution_price
                spread_cost = -(-trade * execution_price + trade * asset.spot)
                transaction_cost = (
                    abs(trade * execution_price) * settings.component_cost_bps / 10_000.0
                )
                component_cashflow += trade_cashflow
                component_spread += spread_cost
                component_cost += transaction_cost
                component_turnover += abs(trade * execution_price)
                final_residual_deltas.append(
                    0.0 if final else residual_deltas[index] + target_position
                )
                component_audits.append(
                    AdvancedComponentAudit(
                        asset_id=asset.asset_id,
                        spot=asset.spot,
                        bid=bid,
                        ask=ask,
                        net_option_delta_before_component=(
                            0.0 if final else residual_deltas[index]
                        ),
                        prior_position=prior,
                        target_position=target_position,
                        trade=trade,
                        execution_price=execution_price,
                        trade_cashflow=trade_cashflow,
                        spread_cost=spread_cost,
                        transaction_cost=transaction_cost,
                    )
                )

            target_mark = settings.short_option_contracts * (
                float(settlement_payoff) if final else target_mid
            )
            pretrade_value = (
                cash
                + sum(
                    component_positions.get(asset.asset_id, 0.0) * asset.spot
                    for asset in market.assets
                )
                + sum(
                    option_positions.get(instrument.instrument_id, 0.0) * instrument.mid
                    for instrument in instruments
                )
                - (0.0 if sequence == 0 else target_mark)
            )
            spread_cost = target_spread + hedge_spread + component_spread
            transaction_cost = target_transaction_cost + hedge_transaction_cost + component_cost
            turnover = (
                hedge_turnover
                + component_turnover
                + (settings.short_option_contracts * target_bid if sequence == 0 else 0.0)
            )
            cash += target_cashflow + hedge_cashflow + component_cashflow - transaction_cost
            option_positions = target_option_positions
            component_positions = component_targets
            portfolio_value = (
                cash
                + sum(
                    component_positions.get(asset.asset_id, 0.0) * asset.spot
                    for asset in market.assets
                )
                + sum(
                    option_positions.get(instrument.instrument_id, 0.0) * instrument.mid
                    for instrument in instruments
                )
                - (0.0 if final else target_mark)
            )
            accounting_gap = portfolio_value - (pretrade_value - spread_cost - transaction_cost)
            if abs(accounting_gap) > settings.accounting_tolerance:
                raise ArithmeticError("advanced hedge self-financing accounting failed")
            cumulative_spread += spread_cost
            cumulative_cost += transaction_cost
            cumulative_turnover += turnover
            incremental_pnl = portfolio_value - previous_value
            instrument_quote_payload = [
                {
                    "instrument_id": instrument.instrument_id,
                    "bid": instrument.bid,
                    "ask": instrument.ask,
                    "definition": instrument_identity[index],
                }
                for index, instrument in enumerate(instruments)
            ]
            input_hash = _sha256(
                {
                    "formation_timestamp": formation.isoformat(),
                    "market_timestamp": market_timestamp.isoformat(),
                    "received_timestamp": received.isoformat(),
                    "source": source,
                    "source_record_id": source_record_id,
                    "target_bid": target_bid,
                    "target_ask": target_ask,
                    "rate": rate,
                    "market_json": market.canonical_json,
                    "execution_json": row["execution_json"],
                    "hedge_instruments": instrument_quote_payload,
                    "input_store_snapshot_sha256": snapshot_hash,
                    "input_manifest_hashes": manifest_hashes,
                    "input_estimate_sha256": estimate_hash,
                    "input_covariance_selection_sha256": covariance_selection_hash,
                    "input_option_quote_snapshot_sha256": option_quote_snapshot_hash,
                }
            )
            snapshot = BasketRiskHedgeSnapshotAudit(
                sequence=sequence,
                formation_timestamp=formation.isoformat(),
                market_timestamp=market_timestamp.isoformat(),
                received_timestamp=received.isoformat(),
                action="settle" if final else "open" if sequence == 0 else "rebalance",
                source=source,
                source_record_id=source_record_id,
                snapshot_sha256=input_hash,
                market_sha256=market.sha256,
                instrument_quote_sha256=_sha256(instrument_quote_payload),
                input_store_snapshot_sha256=snapshot_hash,
                input_manifest_hashes=manifest_hashes,
                input_estimate_sha256=estimate_hash,
                input_covariance_selection_sha256=covariance_selection_hash,
                input_option_quote_snapshot_sha256=option_quote_snapshot_hash,
                configuration_sha256=configuration_hash,
                previous_audit_hash=previous_audit_hash,
                audit_hash="",
                no_lookahead_passed=True,
                maturity_years=maturity,
                rate=rate,
                target_bid=target_bid,
                target_ask=target_ask,
                target_mid=target_mid,
                target_model_price=target_model_price,
                target_component_deltas=target_risk.component_deltas,
                risk_factor_ids=settings.nonlinear_risk_factors,
                target_spot_scale_gamma=target_risk.spot_scale_gamma,
                target_parallel_vega=target_risk.parallel_vega,
                target_forward_log_moneyness=target_risk.forward_log_moneyness,
                target_smile_skew_sensitivity=target_risk.smile_skew_sensitivity,
                target_smile_curvature_sensitivity=(target_risk.smile_curvature_sensitivity),
                target_correlation_strength_sensitivity=(
                    target_risk.correlation_strength_sensitivity
                ),
                settlement_payoff=settlement_payoff,
                solver_rank=solver_rank,
                solver_minimum_singular_value=minimum_singular_value,
                solver_condition_number=condition_number,
                normalized_solver_residual=normalized_residual,
                residual_component_deltas=tuple(final_residual_deltas),
                residual_spot_scale_gamma=residual_gamma,
                residual_parallel_vega=residual_vega,
                residual_smile_skew_sensitivity=residual_skew,
                residual_smile_curvature_sensitivity=residual_curvature,
                residual_correlation_strength_sensitivity=residual_correlation,
                target_option_cashflow=target_cashflow,
                hedge_option_trade_cashflow=hedge_cashflow,
                component_trade_cashflow=component_cashflow,
                cash_before=cash_before,
                cash_carry=cash_carry,
                dividend_cashflow=dividend_cashflow,
                spread_cost=spread_cost,
                cumulative_spread_cost=cumulative_spread,
                transaction_cost=transaction_cost,
                cumulative_transaction_cost=cumulative_cost,
                turnover=turnover,
                cumulative_turnover=cumulative_turnover,
                cash=cash,
                portfolio_value=portfolio_value,
                incremental_pnl=incremental_pnl,
                cumulative_pnl=portfolio_value,
                accounting_gap=accounting_gap,
                hedge_instruments=tuple(instrument_audits),
                components=tuple(component_audits),
            )
            snapshot = replace(snapshot, audit_hash=_audit_hash(snapshot))
            snapshots.append(snapshot)
            previous_audit_hash = snapshot.audit_hash
            previous_value = portfolio_value
            previous_market = market
            previous_formation = formation
            previous_rate = rate
        except (ArithmeticError, BasketRiskHedgeSchemaError, ValueError) as exc:
            failures.append(f"snapshot {sequence} ({formation.isoformat()}): {exc}")
            break

    completed = len(snapshots) == len(frame) and not failures
    result = BasketRiskHedgeResult(
        settings=settings,
        option_type=kind.value,
        strike=normalized_strike,
        expiry_timestamp=expiry.isoformat(),
        settlement_method=terms.settlement_method,
        classification=terms.classification,
        contract_multiplier=terms.contract_multiplier,
        currency=terms.currency,
        underlying_id=terms.underlying_id,
        deliverable_sha256=terms.sha256,
        configuration_sha256=configuration_hash,
        input_snapshot_count=len(frame),
        snapshots=tuple(snapshots),
        completed=completed,
        downstream_eligible=completed,
        risk_factor_ids=settings.nonlinear_risk_factors,
        estimate_provenance_bound=bool(snapshots)
        and all(snapshot.input_estimate_sha256 is not None for snapshot in snapshots),
        covariance_selection_provenance_bound=bool(snapshots)
        and all(snapshot.input_covariance_selection_sha256 is not None for snapshot in snapshots),
        option_quote_provenance_bound=bool(snapshots)
        and all(snapshot.input_option_quote_snapshot_sha256 is not None for snapshot in snapshots),
        smile_risk_coverage_bound=bool(completed and settings.include_smile_risks),
        total_pnl=snapshots[-1].portfolio_value if completed else None,
        total_spread_cost=cumulative_spread,
        total_transaction_cost=cumulative_cost,
        total_turnover=cumulative_turnover,
        maximum_absolute_accounting_gap=(
            max(abs(snapshot.accounting_gap) for snapshot in snapshots) if snapshots else None
        ),
        maximum_normalized_exposure_residual=(
            max(snapshot.normalized_solver_residual for snapshot in snapshots)
            if snapshots
            else None
        ),
        final_audit_hash=previous_audit_hash,
        failure_reasons=tuple(failures),
    )
    if not verify_basket_risk_hedge_hash_chain(result):
        return replace(
            result,
            downstream_eligible=False,
            failure_reasons=(*result.failure_reasons, "advanced hedge audit chain failed"),
        )
    return result


def basket_risk_hedge_report_json(result: BasketRiskHedgeResult) -> bytes:
    """Serialize a human-readable deterministic advanced hedge report."""
    return (json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")


__all__ = [
    "ADVANCED_BASKET_COVARIANCE_COLUMNS",
    "ADVANCED_BASKET_ESTIMATE_COLUMNS",
    "ADVANCED_BASKET_REPLAY_COLUMNS",
    "AdvancedComponentAudit",
    "BasketRiskHedgeResult",
    "BasketRiskHedgeSchemaError",
    "BasketRiskHedgeSettings",
    "BasketRiskHedgeSnapshotAudit",
    "BasketRiskVector",
    "HedgeInstrumentAudit",
    "basket_option_risk",
    "basket_risk_hedge_report_json",
    "run_basket_risk_hedge_replay",
    "synthetic_hedge_instruments_json",
    "synthetic_smile_hedge_instruments_json",
    "verify_basket_risk_hedge_hash_chain",
]
