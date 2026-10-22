"""CSV option-chain schema, validation, and implied-volatility enrichment."""

from __future__ import annotations

import math
from os import PathLike
from typing import IO, Literal

import numpy as np
import pandas as pd

from volforge.implied_vol import implied_volatility
from volforge.models import bachelier, black_scholes
from volforge.models.common import OptionType

REQUIRED_COLUMNS = (
    "quote_timestamp",
    "expiration",
    "strike",
    "option_type",
    "bid",
    "ask",
    "underlying_spot",
    "risk_free_rate",
    "dividend_yield",
)
DAY_COUNT_BASIS = 365.0


class ChainSchemaError(ValueError):
    """Raised when a chain omits required columns entirely."""


def _append_reason(reasons: list[list[str]], position: int, reason: str) -> None:
    if reason not in reasons[position]:
        reasons[position].append(reason)


def _numeric_columns() -> tuple[str, ...]:
    return ("strike", "bid", "ask", "underlying_spot", "risk_free_rate", "dividend_yield")


def enrich_option_chain(
    frame: pd.DataFrame,
    *,
    model: Literal["bsm", "bachelier"] = "bsm",
    valuation_time: str | pd.Timestamp | None = None,
    stale_after: str | pd.Timedelta = "24h",
    bound_tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Validate and enrich every row without silently dropping observations.

    Maturity follows ACT/365F from ``quote_timestamp`` to ``expiration``.
    When ``valuation_time`` is omitted, the newest valid quote timestamp in
    the file is the deterministic staleness reference.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ChainSchemaError(f"missing required columns: {', '.join(missing)}")
    if model not in ("bsm", "bachelier"):
        raise ValueError("model must be 'bsm' or 'bachelier'")
    if bound_tolerance < 0:
        raise ValueError("bound_tolerance must be non-negative")

    enriched = frame.copy(deep=True).reset_index(drop=True)
    row_count = len(enriched)
    reasons: list[list[str]] = [[] for _ in range(row_count)]

    enriched["quote_timestamp"] = pd.to_datetime(
        enriched["quote_timestamp"], errors="coerce", utc=True
    )
    enriched["expiration"] = pd.to_datetime(enriched["expiration"], errors="coerce", utc=True)
    for column in _numeric_columns():
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
    enriched["option_type"] = enriched["option_type"].astype("string").str.strip().str.lower()

    invalid_timestamp = enriched["quote_timestamp"].isna()
    invalid_expiration = enriched["expiration"].isna()
    for position in np.flatnonzero(invalid_timestamp.to_numpy()):
        _append_reason(reasons, int(position), "invalid quote timestamp")
    for position in np.flatnonzero(invalid_expiration.to_numpy()):
        _append_reason(reasons, int(position), "invalid expiration")

    for column in _numeric_columns():
        invalid = enriched[column].isna() | ~np.isfinite(enriched[column])
        for position in np.flatnonzero(invalid.to_numpy()):
            _append_reason(reasons, int(position), f"invalid {column}")

    invalid_type = ~enriched["option_type"].isin([OptionType.CALL.value, OptionType.PUT.value])
    for position in np.flatnonzero(invalid_type.to_numpy()):
        _append_reason(reasons, int(position), "option_type must be call or put")

    maturity = (enriched["expiration"] - enriched["quote_timestamp"]).dt.total_seconds()
    enriched["maturity_years"] = maturity / (DAY_COUNT_BASIS * 24.0 * 60.0 * 60.0)
    negative_maturity = enriched["maturity_years"] < 0
    for position in np.flatnonzero(negative_maturity.fillna(False).to_numpy()):
        _append_reason(reasons, int(position), "expiration precedes quote timestamp")

    invalid_contract = (enriched["strike"] <= 0) | (enriched["underlying_spot"] <= 0)
    for position in np.flatnonzero(invalid_contract.fillna(False).to_numpy()):
        _append_reason(reasons, int(position), "strike and underlying_spot must be positive")

    negative_quote = (enriched["bid"] < 0) | (enriched["ask"] < 0)
    crossed_quote = enriched["bid"] > enriched["ask"]
    enriched["flag_negative_quote"] = negative_quote.fillna(False)
    enriched["flag_crossed_quote"] = crossed_quote.fillna(False)
    for position in np.flatnonzero(enriched["flag_negative_quote"].to_numpy()):
        _append_reason(reasons, int(position), "negative bid or ask")
    for position in np.flatnonzero(enriched["flag_crossed_quote"].to_numpy()):
        _append_reason(reasons, int(position), "crossed quote: bid exceeds ask")

    enriched["midpoint"] = (enriched["bid"] + enriched["ask"]) / 2.0
    invalid_quote = enriched["bid"].isna() | enriched["ask"].isna() | negative_quote | crossed_quote
    enriched.loc[invalid_quote, "midpoint"] = np.nan

    duplicate_subset = ["quote_timestamp", "expiration", "strike", "option_type"]
    enriched["flag_duplicate_contract"] = enriched.duplicated(duplicate_subset, keep=False)
    for position in np.flatnonzero(enriched["flag_duplicate_contract"].to_numpy()):
        _append_reason(reasons, int(position), "duplicate contract")

    valid_quotes = enriched["quote_timestamp"].dropna()
    if valuation_time is None:
        reference_time = valid_quotes.max() if not valid_quotes.empty else pd.NaT
    else:
        reference_time = pd.to_datetime(valuation_time, errors="coerce", utc=True)
        if pd.isna(reference_time):
            raise ValueError("valuation_time must be a valid timestamp")
    stale_delta = pd.to_timedelta(stale_after)
    if stale_delta < pd.Timedelta(0):
        raise ValueError("stale_after must be non-negative")
    enriched["flag_stale"] = False
    if not pd.isna(reference_time):
        enriched["flag_stale"] = (reference_time - enriched["quote_timestamp"]) > stale_delta
        future_quotes = enriched["quote_timestamp"] > reference_time
        for position in np.flatnonzero(future_quotes.fillna(False).to_numpy()):
            _append_reason(reasons, int(position), "quote timestamp is after valuation time")
    for position in np.flatnonzero(enriched["flag_stale"].fillna(False).to_numpy()):
        _append_reason(reasons, int(position), "stale quote")

    enriched["lower_bound"] = np.nan
    enriched["upper_bound"] = np.nan
    enriched["flag_below_bound"] = False
    enriched["flag_above_bound"] = False
    module = black_scholes if model == "bsm" else bachelier
    for position, row in enriched.iterrows():
        basic_values = (
            row["strike"],
            row["underlying_spot"],
            row["maturity_years"],
            row["risk_free_rate"],
            row["dividend_yield"],
        )
        if row["option_type"] not in ("call", "put") or any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in basic_values
        ):
            continue
        try:
            lower, upper = module.bounds(
                row["option_type"],
                float(row["underlying_spot"]),
                float(row["strike"]),
                float(row["maturity_years"]),
                float(row["risk_free_rate"]),
                float(row["dividend_yield"]),
            )
        except ValueError as exc:
            _append_reason(reasons, position, str(exc))
            continue
        enriched.at[position, "lower_bound"] = lower
        enriched.at[position, "upper_bound"] = upper
        midpoint = row["midpoint"]
        if not isinstance(midpoint, (int, float)) or not math.isfinite(float(midpoint)):
            continue
        tolerance = bound_tolerance * max(1.0, abs(lower), abs(upper))
        if midpoint < lower - tolerance:
            enriched.at[position, "flag_below_bound"] = True
            _append_reason(reasons, position, "midpoint below no-arbitrage bound")
        if midpoint > upper + tolerance:
            enriched.at[position, "flag_above_bound"] = True
            _append_reason(reasons, position, "midpoint above no-arbitrage bound")

    enriched["implied_volatility"] = np.nan
    enriched["iv_converged"] = False
    enriched["iv_iterations"] = pd.Series([pd.NA] * row_count, dtype="Int64")
    enriched["iv_residual"] = np.nan
    enriched["iv_failure_reason"] = pd.Series([pd.NA] * row_count, dtype="string")
    for position, row in enriched.iterrows():
        if reasons[position]:
            continue
        result = implied_volatility(
            float(row["midpoint"]),
            str(row["option_type"]),
            float(row["underlying_spot"]),
            float(row["strike"]),
            float(row["maturity_years"]),
            float(row["risk_free_rate"]),
            float(row["dividend_yield"]),
            model=model,
        )
        enriched.at[position, "iv_converged"] = result.converged
        enriched.at[position, "iv_iterations"] = result.iterations
        enriched.at[position, "iv_residual"] = result.residual
        if result.converged and result.volatility is not None:
            enriched.at[position, "implied_volatility"] = result.volatility
        else:
            failure = result.failure_reason or "unknown implied-volatility solver failure"
            enriched.at[position, "iv_failure_reason"] = failure
            _append_reason(reasons, position, f"implied-volatility failure: {failure}")

    enriched["validation_reasons"] = ["; ".join(row_reasons) for row_reasons in reasons]
    enriched["is_valid"] = [not row_reasons for row_reasons in reasons]
    return enriched


def load_csv(
    source: str | PathLike[str] | IO[str] | IO[bytes],
    **kwargs: object,
) -> pd.DataFrame:
    """Load, validate, and enrich an option-chain CSV source."""
    try:
        frame = pd.read_csv(source)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ChainSchemaError(f"could not read CSV: {exc}") from exc
    return enrich_option_chain(frame, **kwargs)  # type: ignore[arg-type]
