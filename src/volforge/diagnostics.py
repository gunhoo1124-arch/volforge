"""Put-call parity and option-chain quality diagnostics."""

from __future__ import annotations

import math

import pandas as pd


def put_call_parity_residuals(chain: pd.DataFrame) -> pd.DataFrame:
    """Pair calls and puts and calculate C-P-(S*e^-qT-K*e^-rT)."""
    required = {
        "quote_timestamp",
        "expiration",
        "strike",
        "option_type",
        "midpoint",
        "underlying_spot",
        "risk_free_rate",
        "dividend_yield",
        "maturity_years",
    }
    missing = required.difference(chain.columns)
    if missing:
        raise ValueError(f"chain is missing diagnostic columns: {', '.join(sorted(missing))}")
    usable = chain.copy()
    if "flag_duplicate_contract" in usable:
        usable = usable.loc[~usable["flag_duplicate_contract"]]
    keys = ["quote_timestamp", "expiration", "strike"]
    values = [
        *keys,
        "midpoint",
        "underlying_spot",
        "risk_free_rate",
        "dividend_yield",
        "maturity_years",
    ]
    calls = usable.loc[usable["option_type"] == "call", values].rename(
        columns={"midpoint": "call_midpoint"}
    )
    puts = usable.loc[usable["option_type"] == "put", [*keys, "midpoint"]].rename(
        columns={"midpoint": "put_midpoint"}
    )
    paired = calls.merge(puts, on=keys, how="inner", validate="one_to_one")
    if paired.empty:
        paired["parity_theoretical"] = pd.Series(dtype="float64")
        paired["parity_residual"] = pd.Series(dtype="float64")
        return paired
    discounted_spot = paired["underlying_spot"] * (
        -paired["dividend_yield"] * paired["maturity_years"]
    ).map(math.exp)
    discounted_strike = paired["strike"] * (
        -paired["risk_free_rate"] * paired["maturity_years"]
    ).map(math.exp)
    paired["parity_theoretical"] = discounted_spot - discounted_strike
    paired["parity_residual"] = (
        paired["call_midpoint"] - paired["put_midpoint"] - paired["parity_theoretical"]
    )
    return paired


def quality_summary(chain: pd.DataFrame) -> dict[str, int | float]:
    """Return deterministic aggregate data-quality metrics."""
    total = len(chain)

    def count(column: str) -> int:
        return int(chain[column].fillna(False).astype(bool).sum()) if column in chain else 0

    parity = put_call_parity_residuals(chain)
    residuals = parity["parity_residual"].dropna().abs()
    valid_rows = int(chain["is_valid"].sum()) if "is_valid" in chain else 0
    return {
        "total_rows": total,
        "valid_rows": valid_rows,
        "invalid_rows": total - valid_rows,
        "stale_rows": count("flag_stale"),
        "negative_quotes": count("flag_negative_quote"),
        "crossed_quotes": count("flag_crossed_quote"),
        "below_bound_rows": count("flag_below_bound"),
        "above_bound_rows": count("flag_above_bound"),
        "duplicate_rows": count("flag_duplicate_contract"),
        "iv_failures": int(chain["iv_failure_reason"].notna().sum())
        if "iv_failure_reason" in chain.columns
        else 0,
        "parity_pairs": len(parity),
        "mean_absolute_parity_residual": float(residuals.mean())
        if not residuals.empty
        else math.nan,
        "max_absolute_parity_residual": float(residuals.max()) if not residuals.empty else math.nan,
    }
