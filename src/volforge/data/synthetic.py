"""Deterministic synthetic option-chain generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from volforge.models.black_scholes import price
from volforge.surfaces.ssvi import ssvi_total_variance


@dataclass(frozen=True, slots=True)
class SyntheticChainConfig:
    """Configuration for a reproducible Black--Scholes smile chain."""

    quote_timestamp: str = "2025-01-02T15:30:00Z"
    spot: float = 100.0
    risk_free_rate: float = 0.03
    dividend_yield: float = 0.01
    base_volatility: float = 0.22
    skew: float = -0.12
    curvature: float = 0.35
    expiration_days: tuple[int, ...] = (30, 90, 180, 365)
    strike_multipliers: tuple[float, ...] = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)
    bid_ask_spread: float = 0.10
    quote_noise_std: float = 0.0
    seed: int = 7
    inject_quality_issues: bool = False


@dataclass(frozen=True, slots=True)
class SyntheticReplayConfig:
    """Configuration for deterministic point-in-time SSVI chain snapshots."""

    start_timestamp: str = "2025-01-02T15:30:00Z"
    spot: float = 100.0
    spot_multipliers: tuple[float, ...] = (1.0, 1.012, 0.996, 1.018)
    snapshot_offsets_days: tuple[int, ...] = (0, 1, 2, 3)
    expiration_days: tuple[int, ...] = (60, 120, 240)
    strike_multipliers: tuple[float, ...] = (0.75, 0.85, 0.925, 1.0, 1.075, 1.15, 1.25)
    risk_free_rate: float = 0.03
    dividend_yield: float = 0.01
    atm_volatility: float = 0.22
    ssvi_rho: float = -0.45
    ssvi_eta: float = 0.45
    ssvi_gamma: float = 0.5
    bid_ask_spread: float = 0.04


def generate_synthetic_chain(config: SyntheticChainConfig | None = None) -> pd.DataFrame:
    """Generate calls and puts across expirations and strikes reproducibly."""
    config = config or SyntheticChainConfig()
    if config.spot <= 0 or config.base_volatility <= 0:
        raise ValueError("spot and base_volatility must be positive")
    if any(day <= 0 for day in config.expiration_days):
        raise ValueError("expiration_days must contain only positive values")
    if any(multiplier <= 0 for multiplier in config.strike_multipliers):
        raise ValueError("strike_multipliers must contain only positive values")
    if config.bid_ask_spread < 0 or config.quote_noise_std < 0:
        raise ValueError("bid_ask_spread and quote_noise_std must be non-negative")

    quote_time = pd.to_datetime(config.quote_timestamp, utc=True)
    if pd.isna(quote_time):
        raise ValueError("quote_timestamp must be valid")
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, object]] = []
    for days in config.expiration_days:
        maturity = days / 365.0
        expiration = quote_time + pd.Timedelta(days=days)
        forward = config.spot * math.exp((config.risk_free_rate - config.dividend_yield) * maturity)
        for multiplier in config.strike_multipliers:
            strike = config.spot * multiplier
            log_moneyness = math.log(strike / forward)
            volatility = max(
                0.01,
                config.base_volatility
                + config.skew * log_moneyness
                + config.curvature * log_moneyness**2,
            )
            for option_type in ("call", "put"):
                fair_value = price(
                    option_type,
                    config.spot,
                    strike,
                    maturity,
                    volatility,
                    config.risk_free_rate,
                    config.dividend_yield,
                )
                noisy_mid = max(0.0, fair_value + rng.normal(0.0, config.quote_noise_std))
                half_spread = min(config.bid_ask_spread / 2.0, noisy_mid)
                bid = noisy_mid - half_spread
                ask = noisy_mid + half_spread
                rows.append(
                    {
                        "quote_timestamp": quote_time.isoformat(),
                        "expiration": expiration.isoformat(),
                        "strike": round(strike, 8),
                        "option_type": option_type,
                        "bid": round(bid, 8),
                        "ask": round(ask, 8),
                        "underlying_spot": config.spot,
                        "risk_free_rate": config.risk_free_rate,
                        "dividend_yield": config.dividend_yield,
                        "synthetic_true_volatility": volatility,
                    }
                )
    frame = pd.DataFrame(rows)
    if config.inject_quality_issues and len(frame) >= 5:
        frame.loc[0, "bid"] = frame.loc[0, "ask"] + 0.25
        frame.loc[1, "bid"] = -0.01
        frame.loc[2, ["bid", "ask"]] = (config.spot * 1.5, config.spot * 1.51)
        old_timestamp = quote_time - pd.Timedelta(days=3)
        frame.loc[3, "quote_timestamp"] = old_timestamp.isoformat()
        frame = pd.concat([frame, frame.iloc[[4]]], ignore_index=True)
    return frame


def generate_synthetic_replay(config: SyntheticReplayConfig | None = None) -> pd.DataFrame:
    """Generate fixed listed contracts across chronological quote snapshots.

    Each snapshot is priced from the information available at its own quote
    timestamp. Expirations and strikes remain fixed, which makes the output a
    realistic deterministic fixture for point-in-time hedge replay.
    """
    config = config or SyntheticReplayConfig()
    if config.spot <= 0.0 or config.atm_volatility <= 0.0:
        raise ValueError("spot and atm_volatility must be positive")
    if len(config.snapshot_offsets_days) != len(config.spot_multipliers):
        raise ValueError("snapshot offsets and spot multipliers must have matching lengths")
    if len(config.snapshot_offsets_days) < 2:
        raise ValueError("synthetic replay requires at least two snapshots")
    if any(
        later <= earlier
        for earlier, later in zip(
            config.snapshot_offsets_days[:-1],
            config.snapshot_offsets_days[1:],
            strict=True,
        )
    ):
        raise ValueError("snapshot offsets must be strictly increasing")
    if any(day <= config.snapshot_offsets_days[-1] for day in config.expiration_days):
        raise ValueError("all expirations must occur after the final snapshot")
    if len(config.expiration_days) < 2 or any(day <= 0 for day in config.expiration_days):
        raise ValueError("synthetic replay requires at least two positive expirations")
    if len(config.strike_multipliers) < 5 or any(
        multiplier <= 0.0 for multiplier in config.strike_multipliers
    ):
        raise ValueError("synthetic replay requires at least five positive strike multipliers")
    if any(multiplier <= 0.0 for multiplier in config.spot_multipliers):
        raise ValueError("spot multipliers must be positive")
    if config.bid_ask_spread < 0.0:
        raise ValueError("bid_ask_spread must be non-negative")

    start = pd.to_datetime(config.start_timestamp, errors="coerce", utc=True)
    if pd.isna(start):
        raise ValueError("start_timestamp must be valid")
    expirations = tuple(start + pd.Timedelta(days=days) for days in config.expiration_days)
    strikes = tuple(config.spot * multiplier for multiplier in config.strike_multipliers)
    rows: list[dict[str, object]] = []
    for offset, spot_multiplier in zip(
        config.snapshot_offsets_days, config.spot_multipliers, strict=True
    ):
        quote_time = start + pd.Timedelta(days=offset)
        snapshot_spot = config.spot * spot_multiplier
        for expiration in expirations:
            maturity = (expiration - quote_time).total_seconds() / (365.0 * 24.0 * 60.0 * 60.0)
            forward = snapshot_spot * math.exp(
                (config.risk_free_rate - config.dividend_yield) * maturity
            )
            theta = config.atm_volatility**2 * maturity
            for strike in strikes:
                log_moneyness = math.log(strike / forward)
                total_variance = float(
                    ssvi_total_variance(
                        np.asarray([log_moneyness]),
                        theta,
                        config.ssvi_rho,
                        config.ssvi_eta,
                        config.ssvi_gamma,
                    )[0]
                )
                volatility = math.sqrt(total_variance / maturity)
                for option_type in ("call", "put"):
                    fair_value = price(
                        option_type,
                        snapshot_spot,
                        strike,
                        maturity,
                        volatility,
                        config.risk_free_rate,
                        config.dividend_yield,
                    )
                    half_spread = min(config.bid_ask_spread / 2.0, fair_value)
                    rows.append(
                        {
                            "quote_timestamp": quote_time.isoformat(),
                            "expiration": expiration.isoformat(),
                            "strike": round(strike, 8),
                            "option_type": option_type,
                            "bid": round(fair_value - half_spread, 10),
                            "ask": round(fair_value + half_spread, 10),
                            "underlying_spot": round(snapshot_spot, 10),
                            "risk_free_rate": config.risk_free_rate,
                            "dividend_yield": config.dividend_yield,
                            "synthetic_true_volatility": volatility,
                        }
                    )
    return pd.DataFrame(rows)
