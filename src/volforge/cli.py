"""Command-line interface for VolForge v1.11.0."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volforge.american import (
    BinomialSettings,
    LSMSettings,
    american_binomial_price,
    heston_american_pde_price,
    heston_lsm_price,
)
from volforge.basket import (
    BasketMonteCarloSettings,
    parse_basket_market,
    price_basket_option,
)
from volforge.basket_covariance import (
    FORECAST_DISTRIBUTIONS,
    SELECTION_OBJECTIVES,
    BasketCovarianceSelectionSettings,
    basket_covariance_selection_report_json,
    materialize_selected_covariance_replay,
    select_basket_covariance,
)
from volforge.basket_estimation import (
    COVARIANCE_ESTIMATORS,
    BasketEstimationSettings,
    generate_synthetic_smile_basket_inputs,
    generate_synthetic_statistical_basket_inputs,
    materialize_estimated_replay,
)
from volforge.basket_inputs import (
    BASKET_INPUT_MODES,
    BASKET_INPUT_SCHEMA_VERSION,
    SYNTHETIC_BASKET_ID,
    BasketInputStore,
    HttpsJsonBasketInputAdapter,
    basket_input_report_json,
    generate_synthetic_basket_inputs,
)
from volforge.basket_option_inputs import (
    BASKET_OPTION_QUOTE_MODES,
    BASKET_OPTION_QUOTE_SCHEMA_VERSION,
    SYNTHETIC_TARGET_INSTRUMENT_ID,
    BasketOptionQuoteStore,
    basket_option_quote_report_json,
    generate_synthetic_option_quote_inputs,
)
from volforge.basket_replay import (
    SYNTHETIC_BASKET_DELIVERABLE,
    SYNTHETIC_BASKET_EXPIRY,
    BasketReplaySettings,
    basket_replay_report_json,
    generate_synthetic_basket_replay,
    run_basket_hedge_replay,
)
from volforge.basket_risk_hedging import (
    BasketRiskHedgeSettings,
    basket_risk_hedge_report_json,
    run_basket_risk_hedge_replay,
)
from volforge.benchmark import (
    COMPARISON_LOSSES,
    BenchmarkSettings,
    benchmark_report_json,
    run_chronological_benchmark,
)
from volforge.calibration import (
    calibrate_heston,
    heston_calibration_report_json,
    heston_calibration_summary,
)
from volforge.data.chain import load_csv
from volforge.data.synthetic import (
    SyntheticChainConfig,
    SyntheticReplayConfig,
    generate_synthetic_chain,
    generate_synthetic_replay,
)
from volforge.deliverables import (
    MAX_DELIVERABLE_BYTES,
    parse_deliverable_terms,
    value_option_settlement,
)
from volforge.diagnostics import quality_summary
from volforge.greeks import (
    FiniteDifferenceSettings,
    MonteCarloGreekSettings,
    heston_finite_difference_greeks,
    heston_monte_carlo_greeks,
)
from volforge.hedging import HEDGE_STRATEGIES, HedgeBacktestSettings, run_heston_hedge_backtest
from volforge.history import (
    HISTORY_SCHEMA_VERSION,
    INGESTION_MODES,
    RECORD_TYPES,
    HistoricalEventStore,
    history_report_json,
)
from volforge.implied_vol import implied_volatility
from volforge.models import HestonParameters, bachelier, black_scholes
from volforge.models.heston import COSSettings, price_with_diagnostics
from volforge.numerics import (
    ConvergenceSettings,
    MonteCarloSettings,
    PDEGridSettings,
    assess_heston_convergence,
    convergence_report_json,
    heston_monte_carlo,
    heston_pde_price,
)
from volforge.reference import (
    REFERENCE_INGESTION_MODES,
    REFERENCE_KINDS,
    REFERENCE_SCHEMA_VERSION,
    ReferenceDataStore,
    reference_report_json,
)
from volforge.replay import (
    ListedContract,
    ReplaySettings,
    replay_report_json,
    run_chronological_replay,
)
from volforge.reporting import dataframe_to_csv, quality_report_markdown
from volforge.research import (
    build_benchmark_research_bundle,
    generate_signing_key_pair,
    sign_research_bundle,
    verify_bundle_attestation,
)
from volforge.robustness import (
    HedgeRobustnessSettings,
    hedge_robustness_report_json,
    run_hedge_robustness_gate,
)
from volforge.surfaces import calibrate_ssvi, calibrate_svi_surface, prepare_surface_quotes
from volforge.surfaces.reporting import calibration_report_json, calibration_summary


def _add_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--option-type", choices=("call", "put"), required=True)
    parser.add_argument("--spot", type=float, required=True)
    parser.add_argument("--strike", type=float, required=True)
    parser.add_argument("--maturity", type=float, required=True, help="ACT/365F years")
    parser.add_argument("--rate", type=float, default=0.0, help="continuous annual rate")
    parser.add_argument("--dividend-yield", type=float, default=0.0, help="continuous annual yield")


def _add_heston_model_parameters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--kappa", type=float, required=True, help="variance mean reversion")
    parser.add_argument("--theta", type=float, required=True, help="long-run variance")
    parser.add_argument("--vol-of-vol", type=float, required=True)
    parser.add_argument("--rho", type=float, required=True, help="spot/variance correlation")
    parser.add_argument("--initial-variance", type=float, required=True)


def _add_cos_settings(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cos-terms", type=int, default=256)
    parser.add_argument("--truncation-width", type=float, default=16.0)


def build_parser() -> argparse.ArgumentParser:
    """Create the public CLI parser."""
    parser = argparse.ArgumentParser(prog="volforge", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    price_parser = subparsers.add_parser("price", help="price one European option")
    _add_contract_arguments(price_parser)
    price_parser.add_argument("--model", choices=("bsm", "bachelier"), default="bsm")
    price_parser.add_argument("--volatility", type=float, required=True)

    iv_parser = subparsers.add_parser("iv", help="solve one implied volatility")
    _add_contract_arguments(iv_parser)
    iv_parser.add_argument("--model", choices=("bsm", "bachelier"), default="bsm")
    iv_parser.add_argument("--price", type=float, required=True)
    iv_parser.add_argument("--method", choices=("brent", "newton"), default="brent")
    iv_parser.add_argument("--initial-guess", type=float)

    generate_parser = subparsers.add_parser(
        "generate", help="write a deterministic synthetic chain"
    )
    generate_parser.add_argument("output", type=Path)
    generate_parser.add_argument("--seed", type=int, default=7)
    generate_parser.add_argument("--spot", type=float, default=100.0)
    generate_parser.add_argument("--base-volatility", type=float, default=0.22)
    generate_parser.add_argument("--skew", type=float, default=-0.12)
    generate_parser.add_argument("--curvature", type=float, default=0.35)
    generate_parser.add_argument("--spread", type=float, default=0.10)
    generate_parser.add_argument("--noise", type=float, default=0.0)

    generate_replay_parser = subparsers.add_parser(
        "generate-replay", help="write deterministic point-in-time chain snapshots"
    )
    generate_replay_parser.add_argument("output", type=Path)
    generate_replay_parser.add_argument("--spot", type=float, default=100.0)
    generate_replay_parser.add_argument("--atm-volatility", type=float, default=0.22)
    generate_replay_parser.add_argument("--spread", type=float, default=0.04)

    generate_basket_replay_parser = subparsers.add_parser(
        "generate-basket-replay",
        help="write a deterministic adjusted-option replay fixture",
    )
    generate_basket_replay_parser.add_argument("output", type=Path, help="snapshot CSV")
    generate_basket_replay_parser.add_argument(
        "deliverable_output", type=Path, help="matching deliverable JSON"
    )
    generate_basket_replay_parser.add_argument("--paths", type=int, default=4_000)
    generate_basket_replay_parser.add_argument("--seed", type=int, default=101)

    validate_parser = subparsers.add_parser("validate", help="validate and enrich a chain CSV")
    validate_parser.add_argument("input", type=Path)
    validate_parser.add_argument("output", type=Path)
    validate_parser.add_argument("--model", choices=("bsm", "bachelier"), default="bsm")
    validate_parser.add_argument("--valuation-time")
    validate_parser.add_argument("--stale-after", default="24h")
    validate_parser.add_argument("--report", type=Path)

    surface_parser = subparsers.add_parser(
        "surface", help="calibrate and validate an SVI or SSVI total-variance surface"
    )
    surface_parser.add_argument("input", type=Path, help="raw option-chain CSV")
    surface_parser.add_argument("output", type=Path, help="evaluated surface CSV")
    surface_parser.add_argument("--model", choices=("svi", "ssvi"), default="ssvi")
    surface_parser.add_argument("--report", type=Path, help="JSON calibration audit")
    surface_parser.add_argument("--grid-min", type=float, default=-1.0)
    surface_parser.add_argument("--grid-max", type=float, default=1.0)
    surface_parser.add_argument("--grid-points", type=int, default=101)

    heston_price_parser = subparsers.add_parser(
        "heston-price", help="price one European option with Heston COS"
    )
    _add_contract_arguments(heston_price_parser)
    _add_heston_model_parameters(heston_price_parser)
    _add_cos_settings(heston_price_parser)

    heston_mc_parser = subparsers.add_parser(
        "heston-mc", help="price one European option with seeded Heston QE-M Monte Carlo"
    )
    _add_contract_arguments(heston_mc_parser)
    _add_heston_model_parameters(heston_mc_parser)
    heston_mc_parser.add_argument("--paths", type=int, default=100_000)
    heston_mc_parser.add_argument("--steps", type=int, default=252)
    heston_mc_parser.add_argument("--seed", type=int, default=7)
    heston_mc_parser.add_argument("--no-antithetic", action="store_true")
    heston_mc_parser.add_argument("--no-control-variate", action="store_true")

    heston_pde_parser = subparsers.add_parser(
        "heston-pde", help="price one European option with the Heston finite-difference PDE"
    )
    _add_contract_arguments(heston_pde_parser)
    _add_heston_model_parameters(heston_pde_parser)
    heston_pde_parser.add_argument("--log-price-points", type=int, default=121)
    heston_pde_parser.add_argument("--variance-points", type=int, default=61)
    heston_pde_parser.add_argument("--time-steps", type=int, default=180)
    heston_pde_parser.add_argument("--log-price-half-width", type=float, default=4.0)
    heston_pde_parser.add_argument("--variance-max", type=float, default=1.0)
    heston_pde_parser.add_argument("--rannacher-steps", type=int, default=2)

    binomial_parser = subparsers.add_parser(
        "american-binomial", help="price one American option with a CRR tree"
    )
    _add_contract_arguments(binomial_parser)
    binomial_parser.add_argument("--volatility", type=float, required=True)
    binomial_parser.add_argument("--steps", type=int, default=1_000)

    lsm_parser = subparsers.add_parser(
        "american-lsm", help="price one American option with Heston LSM"
    )
    _add_contract_arguments(lsm_parser)
    _add_heston_model_parameters(lsm_parser)
    lsm_parser.add_argument("--paths", type=int, default=20_000)
    lsm_parser.add_argument("--steps", type=int, default=64)
    lsm_parser.add_argument("--seed", type=int, default=7)
    lsm_parser.add_argument("--no-antithetic", action="store_true")

    american_pde_parser = subparsers.add_parser(
        "american-pde", help="price one American option with the Heston obstacle PDE"
    )
    _add_contract_arguments(american_pde_parser)
    _add_heston_model_parameters(american_pde_parser)
    american_pde_parser.add_argument("--log-price-points", type=int, default=121)
    american_pde_parser.add_argument("--variance-points", type=int, default=61)
    american_pde_parser.add_argument("--time-steps", type=int, default=180)
    american_pde_parser.add_argument("--log-price-half-width", type=float, default=4.0)
    american_pde_parser.add_argument("--variance-max", type=float, default=1.0)
    american_pde_parser.add_argument("--rannacher-steps", type=int, default=2)

    greeks_parser = subparsers.add_parser(
        "heston-greeks", help="estimate Heston Greeks with COS, PDE, or Monte Carlo"
    )
    _add_contract_arguments(greeks_parser)
    _add_heston_model_parameters(greeks_parser)
    greeks_parser.add_argument("--engine", choices=("cos", "pde", "mc"), default="cos")
    _add_cos_settings(greeks_parser)
    greeks_parser.add_argument("--log-price-points", type=int, default=81)
    greeks_parser.add_argument("--variance-points", type=int, default=41)
    greeks_parser.add_argument("--pde-time-steps", type=int, default=100)
    greeks_parser.add_argument("--log-price-half-width", type=float, default=4.0)
    greeks_parser.add_argument("--variance-max", type=float, default=1.0)
    greeks_parser.add_argument("--rannacher-steps", type=int, default=2)
    greeks_parser.add_argument("--paths", type=int, default=20_000)
    greeks_parser.add_argument("--mc-steps", type=int, default=64)
    greeks_parser.add_argument("--seed", type=int, default=7)
    greeks_parser.add_argument("--no-antithetic", action="store_true")
    greeks_parser.add_argument("--spot-bump", type=float, default=1e-3)
    greeks_parser.add_argument("--volatility-bump", type=float, default=1e-2)
    greeks_parser.add_argument("--maturity-bump", type=float, default=1e-2)
    greeks_parser.add_argument("--rate-bump", type=float, default=1e-4)

    hedge_parser = subparsers.add_parser(
        "heston-hedge", help="compare self-financing Heston hedge strategies"
    )
    _add_contract_arguments(hedge_parser)
    _add_heston_model_parameters(hedge_parser)
    _add_cos_settings(hedge_parser)
    hedge_parser.add_argument("--paths", type=int, default=100)
    hedge_parser.add_argument("--rebalancing-steps", type=int, default=12)
    hedge_parser.add_argument("--seed", type=int, default=7)
    hedge_parser.add_argument("--no-antithetic", action="store_true")
    hedge_parser.add_argument("--stock-cost-bps", type=float, default=1.0)
    hedge_parser.add_argument("--option-cost-bps", type=float, default=5.0)
    hedge_parser.add_argument("--hedge-strike-low", type=float, default=0.9)
    hedge_parser.add_argument("--hedge-strike-high", type=float, default=1.1)
    hedge_parser.add_argument("--max-option-position", type=float, default=50.0)
    hedge_parser.add_argument("--condition-limit", type=float, default=1e6)
    hedge_parser.add_argument("--report", type=Path, help="JSON hedging audit")

    robustness_parser = subparsers.add_parser(
        "heston-robustness",
        help="gate Heston hedges across seeds, budgets, and misspecification scenarios",
    )
    _add_contract_arguments(robustness_parser)
    _add_heston_model_parameters(robustness_parser)
    _add_cos_settings(robustness_parser)
    robustness_parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17])
    robustness_parser.add_argument("--path-counts", type=int, nargs="+", default=[10, 20, 40])
    robustness_parser.add_argument("--rebalancing-steps", type=int, nargs="+", default=[4, 6, 8])
    robustness_parser.add_argument("--no-antithetic", action="store_true")
    robustness_parser.add_argument("--stock-cost-bps", type=float, default=1.0)
    robustness_parser.add_argument("--option-cost-bps", type=float, default=5.0)
    robustness_parser.add_argument("--cost-stress-multiplier", type=float, default=3.0)
    robustness_parser.add_argument("--hedge-strike-low", type=float, default=0.9)
    robustness_parser.add_argument("--hedge-strike-high", type=float, default=1.1)
    robustness_parser.add_argument("--max-option-position", type=float, default=50.0)
    robustness_parser.add_argument("--condition-limit", type=float, default=1e6)
    robustness_parser.add_argument("--physical-annual-return", type=float, default=0.08)
    robustness_parser.add_argument("--physical-kappa-multiplier", type=float, default=0.75)
    robustness_parser.add_argument("--physical-theta-multiplier", type=float, default=1.25)
    robustness_parser.add_argument("--physical-vol-of-vol-multiplier", type=float, default=1.20)
    robustness_parser.add_argument("--physical-rho-shift", type=float, default=-0.10)
    robustness_parser.add_argument("--absolute-tolerance", type=float, default=0.25)
    robustness_parser.add_argument("--relative-tolerance", type=float, default=0.05)
    robustness_parser.add_argument("--minimum-volatility-reduction", type=float, default=0.02)
    robustness_parser.add_argument("--maximum-seed-range-z", type=float, default=4.0)
    robustness_parser.add_argument("--maximum-clipped-fraction", type=float, default=0.05)
    robustness_parser.add_argument("--maximum-fallback-fraction", type=float, default=0.10)
    robustness_parser.add_argument("--attribution-tolerance", type=float, default=1e-8)
    robustness_parser.add_argument("--report", type=Path, help="JSON robustness audit")

    convergence_parser = subparsers.add_parser(
        "heston-converge",
        help="refine COS, Monte Carlo, and PDE prices and enforce a consensus gate",
    )
    _add_contract_arguments(convergence_parser)
    _add_heston_model_parameters(convergence_parser)
    convergence_parser.add_argument("--absolute-tolerance", type=float, default=0.03)
    convergence_parser.add_argument("--relative-tolerance", type=float, default=0.01)
    convergence_parser.add_argument("--max-cos-terms", type=int, default=512)
    convergence_parser.add_argument("--max-cos-width", type=float, default=20.0)
    convergence_parser.add_argument("--max-paths", type=int, default=40_000)
    convergence_parser.add_argument("--max-mc-steps", type=int, default=128)
    convergence_parser.add_argument("--seed", type=int, default=7)
    convergence_parser.add_argument("--no-antithetic", action="store_true")
    convergence_parser.add_argument("--no-control-variate", action="store_true")
    convergence_parser.add_argument("--max-log-price-points", type=int, default=121)
    convergence_parser.add_argument("--max-variance-points", type=int, default=61)
    convergence_parser.add_argument("--max-pde-time-steps", type=int, default=180)
    convergence_parser.add_argument("--max-log-price-half-width", type=float, default=5.0)
    convergence_parser.add_argument("--max-variance", type=float, default=1.25)
    convergence_parser.add_argument("--report", type=Path, help="JSON convergence audit")

    heston_calibrate_parser = subparsers.add_parser(
        "heston-calibrate", help="calibrate Heston to an arbitrage-gated SVI/SSVI surface"
    )
    heston_calibrate_parser.add_argument("input", type=Path, help="raw option-chain CSV")
    heston_calibrate_parser.add_argument("output", type=Path, help="fit comparison CSV")
    heston_calibrate_parser.add_argument("--surface-model", choices=("svi", "ssvi"), default="ssvi")
    heston_calibrate_parser.add_argument("--report", type=Path, help="JSON calibration audit")
    heston_calibrate_parser.add_argument("--moneyness-min", type=float, default=-0.4)
    heston_calibrate_parser.add_argument("--moneyness-max", type=float, default=0.4)
    heston_calibrate_parser.add_argument("--moneyness-points", type=int, default=5)
    heston_calibrate_parser.add_argument("--cos-terms", type=int, default=192)
    heston_calibrate_parser.add_argument("--truncation-width", type=float, default=16.0)
    heston_calibrate_parser.add_argument("--max-function-evaluations", type=int, default=350)
    heston_calibrate_parser.add_argument("--require-feller", action="store_true")

    deliverable_parser = subparsers.add_parser(
        "deliverable-value",
        help="value explicit physical or cash-settled contract deliverables",
    )
    deliverable_parser.add_argument("deliverable", type=Path, help="deliverable JSON list")
    deliverable_parser.add_argument("--marks", type=Path, help="security-mark JSON object")
    deliverable_parser.add_argument(
        "--settlement-method", choices=("physical", "cash"), required=True
    )
    deliverable_parser.add_argument("--option-type", choices=("call", "put"), required=True)
    deliverable_parser.add_argument("--strike", type=float, required=True)
    deliverable_parser.add_argument("--contract-multiplier", type=float, required=True)
    deliverable_parser.add_argument("--currency", required=True)
    deliverable_parser.add_argument("--underlying-id")
    deliverable_parser.add_argument("--underlying-mark", type=float)
    deliverable_parser.add_argument("--settlement-level", type=float)

    basket_parser = subparsers.add_parser(
        "basket-price",
        help="price a European adjusted-deliverable option under correlated GBM",
    )
    basket_parser.add_argument("deliverable", type=Path, help="deliverable JSON list")
    basket_parser.add_argument("market", type=Path, help="asset and correlation JSON object")
    basket_parser.add_argument("--settlement-method", choices=("physical", "cash"), required=True)
    basket_parser.add_argument("--option-type", choices=("call", "put"), required=True)
    basket_parser.add_argument("--strike", type=float, required=True)
    basket_parser.add_argument("--contract-multiplier", type=float, required=True)
    basket_parser.add_argument("--currency", required=True)
    basket_parser.add_argument("--underlying-id")
    basket_parser.add_argument("--maturity", type=float, required=True, help="ACT/365F years")
    basket_parser.add_argument("--rate", type=float, default=0.0, help="continuous annual rate")
    basket_parser.add_argument("--paths", type=int, default=100_000)
    basket_parser.add_argument("--seed", type=int, default=7)
    basket_parser.add_argument("--no-antithetic", action="store_true")
    basket_parser.add_argument("--no-control-variate", action="store_true")
    basket_parser.add_argument("--confidence-level", type=float, default=0.95)

    basket_replay_parser = subparsers.add_parser(
        "basket-replay",
        help="run point-in-time adjusted-option delta-hedge replay",
    )
    basket_replay_parser.add_argument("input", type=Path, help="chronological basket snapshot CSV")
    basket_replay_parser.add_argument("deliverable", type=Path, help="deliverable JSON list")
    basket_replay_parser.add_argument("output", type=Path, help="chronological ledger CSV")
    basket_replay_parser.add_argument("--report", type=Path, help="hash-chained JSON audit")
    basket_replay_parser.add_argument(
        "--settlement-method", choices=("physical", "cash"), required=True
    )
    basket_replay_parser.add_argument("--option-type", choices=("call", "put"), required=True)
    basket_replay_parser.add_argument("--strike", type=float, required=True)
    basket_replay_parser.add_argument("--contract-multiplier", type=float, required=True)
    basket_replay_parser.add_argument("--currency", required=True)
    basket_replay_parser.add_argument("--underlying-id")
    basket_replay_parser.add_argument("--expiry", required=True)
    basket_replay_parser.add_argument("--paths", type=int, default=20_000)
    basket_replay_parser.add_argument("--seed", type=int, default=31)
    basket_replay_parser.add_argument("--no-antithetic", action="store_true")
    basket_replay_parser.add_argument("--no-control-variate", action="store_true")
    basket_replay_parser.add_argument("--confidence-level", type=float, default=0.95)
    basket_replay_parser.add_argument("--short-option-contracts", type=float, default=1.0)
    basket_replay_parser.add_argument("--component-cost-bps", type=float, default=1.0)
    basket_replay_parser.add_argument("--option-cost-bps", type=float, default=5.0)
    basket_replay_parser.add_argument("--maximum-market-age-seconds", type=float, default=900.0)
    basket_replay_parser.add_argument("--require-store-provenance", action="store_true")
    basket_replay_parser.add_argument("--require-estimated-inputs", action="store_true")

    generate_basket_inputs_parser = subparsers.add_parser(
        "generate-basket-inputs",
        help="generate canonical component/correlation records and a replay schedule",
    )
    generate_basket_inputs_parser.add_argument("records_output", type=Path)
    generate_basket_inputs_parser.add_argument("schedule_output", type=Path)

    generate_statistical_inputs_parser = subparsers.add_parser(
        "generate-basket-statistical-inputs",
        help="generate daily basket history and an advanced hedge schedule",
    )
    generate_statistical_inputs_parser.add_argument("records_output", type=Path)
    generate_statistical_inputs_parser.add_argument("schedule_output", type=Path)

    generate_smile_inputs_parser = subparsers.add_parser(
        "generate-basket-smile-inputs",
        help="generate daily basket history and a five-risk smile hedge schedule",
    )
    generate_smile_inputs_parser.add_argument("records_output", type=Path)
    generate_smile_inputs_parser.add_argument("schedule_output", type=Path)

    generate_option_inputs_parser = subparsers.add_parser(
        "generate-basket-option-inputs",
        help="generate independently ingestible target and hedge-option quote history",
    )
    generate_option_inputs_parser.add_argument("records_output", type=Path)
    generate_option_inputs_parser.add_argument("schedule_output", type=Path)

    basket_option_init_parser = subparsers.add_parser(
        "basket-option-init", help="create an append-only basket-option quote store"
    )
    basket_option_init_parser.add_argument("store", type=Path)

    basket_option_ingest_parser = subparsers.add_parser(
        "basket-option-ingest", help="validate and append canonical basket-option quote CSV"
    )
    basket_option_ingest_parser.add_argument("store", type=Path)
    basket_option_ingest_parser.add_argument("input", type=Path)
    basket_option_ingest_parser.add_argument("--source", required=True)
    basket_option_ingest_parser.add_argument(
        "--mode", choices=BASKET_OPTION_QUOTE_MODES, required=True
    )
    basket_option_ingest_parser.add_argument("--ingested-at", required=True)
    basket_option_ingest_parser.add_argument("--reason")
    basket_option_ingest_parser.add_argument("--report", type=Path)

    basket_option_materialize_parser = subparsers.add_parser(
        "basket-option-materialize",
        help="resolve formation/rate rows into provenance-bound executable option quotes",
    )
    basket_option_materialize_parser.add_argument("store", type=Path)
    basket_option_materialize_parser.add_argument("schedule", type=Path)
    basket_option_materialize_parser.add_argument("output", type=Path)
    basket_option_materialize_parser.add_argument("--basket-id", required=True)
    basket_option_materialize_parser.add_argument("--source")
    basket_option_materialize_parser.add_argument("--target-instrument-id", required=True)
    basket_option_materialize_parser.add_argument(
        "--hedge-instrument-id", action="append", required=True
    )
    basket_option_materialize_parser.add_argument("--maximum-age", default="15min")
    basket_option_materialize_parser.add_argument("--dataset-known-at")

    basket_option_verify_parser = subparsers.add_parser(
        "basket-option-verify", help="verify quote records, archives, and manifest chain"
    )
    basket_option_verify_parser.add_argument("store", type=Path)
    basket_option_verify_parser.add_argument("--report", type=Path)

    basket_input_init_parser = subparsers.add_parser(
        "basket-input-init", help="create an append-only basket-input store"
    )
    basket_input_init_parser.add_argument("store", type=Path)

    basket_input_ingest_parser = subparsers.add_parser(
        "basket-input-ingest", help="validate and append one canonical basket-input CSV"
    )
    basket_input_ingest_parser.add_argument("store", type=Path)
    basket_input_ingest_parser.add_argument("input", type=Path)
    basket_input_ingest_parser.add_argument("--source", required=True)
    basket_input_ingest_parser.add_argument("--mode", choices=BASKET_INPUT_MODES, required=True)
    basket_input_ingest_parser.add_argument("--ingested-at", required=True)
    basket_input_ingest_parser.add_argument("--reason")
    basket_input_ingest_parser.add_argument("--report", type=Path)

    basket_input_fetch_parser = subparsers.add_parser(
        "basket-input-fetch", help="fetch canonical JSON over HTTPS and append its raw response"
    )
    basket_input_fetch_parser.add_argument("store", type=Path)
    basket_input_fetch_parser.add_argument("endpoint")
    basket_input_fetch_parser.add_argument("--source", required=True)
    basket_input_fetch_parser.add_argument("--mode", choices=BASKET_INPUT_MODES, required=True)
    basket_input_fetch_parser.add_argument("--ingested-at", required=True)
    basket_input_fetch_parser.add_argument("--authorization-env")
    basket_input_fetch_parser.add_argument("--timeout-seconds", type=float, default=15.0)
    basket_input_fetch_parser.add_argument("--reason")
    basket_input_fetch_parser.add_argument("--report", type=Path)

    basket_input_query_parser = subparsers.add_parser(
        "basket-input-query", help="export active basket records known at formation"
    )
    basket_input_query_parser.add_argument("store", type=Path)
    basket_input_query_parser.add_argument("output", type=Path)
    basket_input_query_parser.add_argument("--formation-at", required=True)
    basket_input_query_parser.add_argument("--basket-id")
    basket_input_query_parser.add_argument("--source")
    basket_input_query_parser.add_argument("--dataset-known-at")

    basket_input_materialize_parser = subparsers.add_parser(
        "basket-input-materialize",
        help="resolve a quote/rate schedule into provenance-bound basket replay rows",
    )
    basket_input_materialize_parser.add_argument("store", type=Path)
    basket_input_materialize_parser.add_argument("schedule", type=Path)
    basket_input_materialize_parser.add_argument("output", type=Path)
    basket_input_materialize_parser.add_argument("--basket-id", required=True)
    basket_input_materialize_parser.add_argument("--source")
    basket_input_materialize_parser.add_argument("--maximum-age", default="15min")
    basket_input_materialize_parser.add_argument("--dataset-known-at")

    basket_input_estimate_parser = subparsers.add_parser(
        "basket-input-estimate",
        help="materialize rolling statistical basket inputs with estimate provenance",
    )
    basket_input_estimate_parser.add_argument("store", type=Path)
    basket_input_estimate_parser.add_argument("schedule", type=Path)
    basket_input_estimate_parser.add_argument("output", type=Path)
    basket_input_estimate_parser.add_argument("--basket-id", required=True)
    basket_input_estimate_parser.add_argument("--source")
    basket_input_estimate_parser.add_argument("--maximum-age", default="15min")
    basket_input_estimate_parser.add_argument("--dataset-known-at")
    basket_input_estimate_parser.add_argument("--window-returns", type=int, default=20)
    basket_input_estimate_parser.add_argument("--minimum-returns", type=int, default=10)
    basket_input_estimate_parser.add_argument("--annualization-factor", type=float, default=252.0)
    basket_input_estimate_parser.add_argument("--correlation-shrinkage", type=float, default=0.20)
    basket_input_estimate_parser.add_argument(
        "--estimator", choices=COVARIANCE_ESTIMATORS, default="sample_shrinkage"
    )
    basket_input_estimate_parser.add_argument("--ewma-decay", type=float, default=0.94)
    basket_input_estimate_parser.add_argument("--covariance-shrinkage", type=float, default=0.35)
    basket_input_estimate_parser.add_argument("--volatility-floor", type=float, default=1e-4)

    basket_covariance_select_parser = subparsers.add_parser(
        "basket-covariance-select",
        help="compare covariance estimators on later held-out returns",
    )
    basket_covariance_select_parser.add_argument("store", type=Path)
    basket_covariance_select_parser.add_argument("--basket-id", required=True)
    basket_covariance_select_parser.add_argument("--formation-at", required=True)
    basket_covariance_select_parser.add_argument("--source")
    basket_covariance_select_parser.add_argument("--maximum-age", default="15min")
    basket_covariance_select_parser.add_argument("--dataset-known-at")
    basket_covariance_select_parser.add_argument("--training-window", type=int, default=20)
    basket_covariance_select_parser.add_argument("--minimum-training-returns", type=int, default=10)
    basket_covariance_select_parser.add_argument("--evaluation-returns", type=int, default=10)
    basket_covariance_select_parser.add_argument(
        "--annualization-factor", type=float, default=252.0
    )
    basket_covariance_select_parser.add_argument(
        "--correlation-shrinkage", type=float, default=0.20
    )
    basket_covariance_select_parser.add_argument("--covariance-shrinkage", type=float, default=0.35)
    basket_covariance_select_parser.add_argument("--ewma-decay", type=float, default=0.94)
    basket_covariance_select_parser.add_argument("--volatility-floor", type=float, default=1e-4)
    basket_covariance_select_parser.add_argument("--eigenvalue-floor", type=float, default=1e-10)
    basket_covariance_select_parser.add_argument(
        "--maximum-condition-number", type=float, default=1e8
    )
    basket_covariance_select_parser.add_argument(
        "--minimum-nll-improvement", type=float, default=0.0
    )
    basket_covariance_select_parser.add_argument(
        "--forecast-distribution", choices=FORECAST_DISTRIBUTIONS, default="gaussian"
    )
    basket_covariance_select_parser.add_argument(
        "--student-t-degrees-of-freedom", type=float, default=6.0
    )
    basket_covariance_select_parser.add_argument(
        "--selection-objective", choices=SELECTION_OBJECTIVES, default="mean_nll"
    )
    basket_covariance_select_parser.add_argument("--stress-quantile", type=float, default=0.80)
    basket_covariance_select_parser.add_argument("--stress-weight", type=float, default=0.50)
    basket_covariance_select_parser.add_argument(
        "--minimum-stress-evaluations", type=int, default=2
    )
    basket_covariance_select_parser.add_argument(
        "--maximum-stress-nll-regret", type=float, default=0.0
    )
    basket_covariance_select_parser.add_argument("--report", type=Path)

    basket_covariance_materialize_parser = subparsers.add_parser(
        "basket-input-select-covariance",
        help="materialize replay rows selected by chronological covariance validation",
    )
    basket_covariance_materialize_parser.add_argument("store", type=Path)
    basket_covariance_materialize_parser.add_argument("schedule", type=Path)
    basket_covariance_materialize_parser.add_argument("output", type=Path)
    basket_covariance_materialize_parser.add_argument("--basket-id", required=True)
    basket_covariance_materialize_parser.add_argument("--source")
    basket_covariance_materialize_parser.add_argument("--maximum-age", default="15min")
    basket_covariance_materialize_parser.add_argument("--dataset-known-at")
    basket_covariance_materialize_parser.add_argument("--training-window", type=int, default=20)
    basket_covariance_materialize_parser.add_argument(
        "--minimum-training-returns", type=int, default=10
    )
    basket_covariance_materialize_parser.add_argument("--evaluation-returns", type=int, default=10)
    basket_covariance_materialize_parser.add_argument(
        "--annualization-factor", type=float, default=252.0
    )
    basket_covariance_materialize_parser.add_argument(
        "--correlation-shrinkage", type=float, default=0.20
    )
    basket_covariance_materialize_parser.add_argument(
        "--covariance-shrinkage", type=float, default=0.35
    )
    basket_covariance_materialize_parser.add_argument("--ewma-decay", type=float, default=0.94)
    basket_covariance_materialize_parser.add_argument(
        "--volatility-floor", type=float, default=1e-4
    )
    basket_covariance_materialize_parser.add_argument(
        "--eigenvalue-floor", type=float, default=1e-10
    )
    basket_covariance_materialize_parser.add_argument(
        "--maximum-condition-number", type=float, default=1e8
    )
    basket_covariance_materialize_parser.add_argument(
        "--minimum-nll-improvement", type=float, default=0.0
    )
    basket_covariance_materialize_parser.add_argument(
        "--forecast-distribution", choices=FORECAST_DISTRIBUTIONS, default="gaussian"
    )
    basket_covariance_materialize_parser.add_argument(
        "--student-t-degrees-of-freedom", type=float, default=6.0
    )
    basket_covariance_materialize_parser.add_argument(
        "--selection-objective", choices=SELECTION_OBJECTIVES, default="mean_nll"
    )
    basket_covariance_materialize_parser.add_argument("--stress-quantile", type=float, default=0.80)
    basket_covariance_materialize_parser.add_argument("--stress-weight", type=float, default=0.50)
    basket_covariance_materialize_parser.add_argument(
        "--minimum-stress-evaluations", type=int, default=2
    )
    basket_covariance_materialize_parser.add_argument(
        "--maximum-stress-nll-regret", type=float, default=0.0
    )

    basket_risk_replay_parser = subparsers.add_parser(
        "basket-risk-replay",
        help="run an executable basket risk hedge replay with optional smile risks",
    )
    basket_risk_replay_parser.add_argument("input", type=Path)
    basket_risk_replay_parser.add_argument("deliverable", type=Path)
    basket_risk_replay_parser.add_argument("output", type=Path)
    basket_risk_replay_parser.add_argument("--report", type=Path)
    basket_risk_replay_parser.add_argument(
        "--settlement-method", choices=("physical", "cash"), required=True
    )
    basket_risk_replay_parser.add_argument("--option-type", choices=("call", "put"), required=True)
    basket_risk_replay_parser.add_argument("--strike", type=float, required=True)
    basket_risk_replay_parser.add_argument("--contract-multiplier", type=float, required=True)
    basket_risk_replay_parser.add_argument("--currency", required=True)
    basket_risk_replay_parser.add_argument("--underlying-id")
    basket_risk_replay_parser.add_argument("--expiry", required=True)
    basket_risk_replay_parser.add_argument("--paths", type=int, default=8_000)
    basket_risk_replay_parser.add_argument("--seed", type=int, default=1701)
    basket_risk_replay_parser.add_argument("--component-cost-bps", type=float, default=1.0)
    basket_risk_replay_parser.add_argument("--hedge-option-cost-bps", type=float, default=5.0)
    basket_risk_replay_parser.add_argument("--target-option-cost-bps", type=float, default=5.0)
    basket_risk_replay_parser.add_argument("--include-smile-risks", action="store_true")
    basket_risk_replay_parser.add_argument("--smile-slope-bump", type=float, default=0.01)
    basket_risk_replay_parser.add_argument("--smile-curvature-bump", type=float, default=0.01)
    basket_risk_replay_parser.add_argument(
        "--maximum-absolute-smile-log-moneyness", type=float, default=0.75
    )
    basket_risk_replay_parser.add_argument(
        "--minimum-normalized-singular-value", type=float, default=1e-8
    )
    basket_risk_replay_parser.add_argument("--maximum-condition-number", type=float, default=1e6)
    basket_risk_replay_parser.add_argument(
        "--maximum-absolute-option-position", type=float, default=100.0
    )
    basket_risk_replay_parser.add_argument("--allow-unestimated-inputs", action="store_true")
    basket_risk_replay_parser.add_argument("--require-covariance-selection", action="store_true")
    basket_risk_replay_parser.add_argument("--require-option-quote-provenance", action="store_true")

    basket_input_verify_parser = subparsers.add_parser(
        "basket-input-verify", help="verify basket-input records, archives, and manifest chain"
    )
    basket_input_verify_parser.add_argument("store", type=Path)
    basket_input_verify_parser.add_argument("--report", type=Path)

    replay_parser = subparsers.add_parser(
        "replay",
        help="run point-in-time surface/Heston calibration and listed-hedge replay",
    )
    replay_parser.add_argument("input", type=Path, help="multi-snapshot raw option-chain CSV")
    replay_parser.add_argument("output", type=Path, help="chronological ledger CSV")
    replay_parser.add_argument("--report", type=Path, help="tamper-evident JSON replay audit")
    replay_parser.add_argument("--surface-model", choices=("svi", "ssvi"), default="ssvi")
    replay_parser.add_argument("--strategy", choices=HEDGE_STRATEGIES, default="delta_gamma_vega")
    replay_parser.add_argument("--target-option-type", choices=("call", "put"), default="call")
    replay_parser.add_argument("--target-expiration")
    replay_parser.add_argument("--target-strike", type=float)
    replay_parser.add_argument("--stock-cost-bps", type=float, default=1.0)
    replay_parser.add_argument("--option-cost-bps", type=float, default=5.0)
    replay_parser.add_argument("--max-option-position", type=float, default=50.0)
    replay_parser.add_argument("--condition-limit", type=float, default=1e6)
    replay_parser.add_argument("--cos-terms", type=int, default=96)
    replay_parser.add_argument("--truncation-width", type=float, default=16.0)
    replay_parser.add_argument("--max-function-evaluations", type=int, default=120)
    replay_parser.add_argument("--require-feller", action="store_true")

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="run rolling out-of-sample BSM-versus-Heston pricing comparisons",
    )
    benchmark_parser.add_argument("input", type=Path, help="multi-snapshot raw option-chain CSV")
    benchmark_parser.add_argument("output", type=Path, help="row-level prediction and loss CSV")
    benchmark_parser.add_argument("--report", type=Path, help="hash-chained JSON benchmark audit")
    benchmark_parser.add_argument("--bundle", type=Path, help="content-addressed research ZIP")
    benchmark_parser.add_argument("--training-window", type=int, default=2)
    benchmark_parser.add_argument("--surface-model", choices=("svi", "ssvi"), default="ssvi")
    benchmark_parser.add_argument("--cos-terms", type=int, default=96)
    benchmark_parser.add_argument("--truncation-width", type=float, default=16.0)
    benchmark_parser.add_argument("--max-function-evaluations", type=int, default=120)
    benchmark_parser.add_argument("--require-feller", action="store_true")
    benchmark_parser.add_argument(
        "--comparison-loss",
        choices=COMPARISON_LOSSES,
        default="squared_error",
    )
    benchmark_parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    benchmark_parser.add_argument("--bootstrap-block-length", type=int)
    benchmark_parser.add_argument("--bootstrap-seed", type=int, default=7)
    benchmark_parser.add_argument(
        "--comparison-hac-lags",
        type=int,
        help="Bartlett HAC lags (default: bootstrap block length minus one)",
    )
    benchmark_parser.add_argument("--comparison-confidence-level", type=float, default=0.95)
    benchmark_parser.add_argument("--minimum-comparison-dates", type=int, default=5)

    keygen_parser = subparsers.add_parser(
        "research-keygen", help="generate an Ed25519 research-report signing key pair"
    )
    keygen_parser.add_argument("private_key", type=Path, help="new PKCS8 private-key PEM")
    keygen_parser.add_argument("public_key", type=Path, help="new public-key PEM")
    keygen_security = keygen_parser.add_mutually_exclusive_group(required=True)
    keygen_security.add_argument(
        "--password-file", type=Path, help="file containing the private-key encryption password"
    )
    keygen_security.add_argument(
        "--unencrypted",
        action="store_true",
        help="explicitly permit an unencrypted private key",
    )

    sign_parser = subparsers.add_parser(
        "research-sign", help="create a detached Ed25519 research-bundle attestation"
    )
    sign_parser.add_argument("bundle", type=Path)
    sign_parser.add_argument("private_key", type=Path)
    sign_parser.add_argument("output", type=Path, help="new detached attestation JSON")
    sign_parser.add_argument("--password-file", type=Path)
    sign_parser.add_argument("--issued-at", required=True, help="explicit ISO-8601 timestamp")
    sign_parser.add_argument("--signer", help="optional human-readable signer label")

    verify_parser = subparsers.add_parser(
        "research-verify", help="verify a bundle attestation against a trusted public key"
    )
    verify_parser.add_argument("bundle", type=Path)
    verify_parser.add_argument("attestation", type=Path)
    verify_parser.add_argument("--trusted-public-key", type=Path, required=True)

    history_init_parser = subparsers.add_parser(
        "history-init", help="create an append-only historical event store"
    )
    history_init_parser.add_argument("store", type=Path)

    history_ingest_parser = subparsers.add_parser(
        "history-ingest", help="validate and append one historical CSV batch"
    )
    history_ingest_parser.add_argument("store", type=Path)
    history_ingest_parser.add_argument("input", type=Path)
    history_ingest_parser.add_argument("--source", required=True)
    history_ingest_parser.add_argument("--mode", choices=INGESTION_MODES, required=True)
    history_ingest_parser.add_argument("--ingested-at", required=True)
    history_ingest_parser.add_argument("--reason")
    history_ingest_parser.add_argument("--report", type=Path)

    history_query_parser = subparsers.add_parser(
        "history-query", help="export latest active revisions under explicit knowledge cutoffs"
    )
    history_query_parser.add_argument("store", type=Path)
    history_query_parser.add_argument("output", type=Path)
    history_query_parser.add_argument("--record-type", choices=RECORD_TYPES, required=True)
    history_query_parser.add_argument("--event-start")
    history_query_parser.add_argument("--event-end")
    history_query_parser.add_argument("--received-at")
    history_query_parser.add_argument("--ingested-at")
    history_query_parser.add_argument("--source")

    history_snapshot_parser = subparsers.add_parser(
        "history-snapshot", help="export one or more replay-ready point-in-time chains"
    )
    history_snapshot_parser.add_argument("store", type=Path)
    history_snapshot_parser.add_argument("output", type=Path)
    history_snapshot_parser.add_argument("--snapshot-at", nargs="+", required=True)
    history_snapshot_parser.add_argument("--source")
    history_snapshot_parser.add_argument("--maximum-quote-age", default="15min")
    history_snapshot_parser.add_argument("--dataset-known-at")
    history_snapshot_parser.add_argument(
        "--reference-store", type=Path, help="contract/session reference database"
    )
    history_snapshot_parser.add_argument(
        "--reference-known-at", help="reference-store ingestion cutoff"
    )

    history_verify_parser = subparsers.add_parser(
        "history-verify", help="verify database, raw archives, records, and manifest chain"
    )
    history_verify_parser.add_argument("store", type=Path)
    history_verify_parser.add_argument("--report", type=Path)

    reference_init_parser = subparsers.add_parser(
        "reference-init", help="create an append-only option reference store"
    )
    reference_init_parser.add_argument("store", type=Path)

    reference_ingest_parser = subparsers.add_parser(
        "reference-ingest", help="validate and append one reference CSV batch"
    )
    reference_ingest_parser.add_argument("store", type=Path)
    reference_ingest_parser.add_argument("input", type=Path)
    reference_ingest_parser.add_argument("--kind", choices=REFERENCE_KINDS, required=True)
    reference_ingest_parser.add_argument("--source", required=True)
    reference_ingest_parser.add_argument("--mode", choices=REFERENCE_INGESTION_MODES, required=True)
    reference_ingest_parser.add_argument("--ingested-at", required=True)
    reference_ingest_parser.add_argument("--reason")
    reference_ingest_parser.add_argument("--report", type=Path)

    reference_query_parser = subparsers.add_parser(
        "reference-query", help="export reference records under point-in-time cutoffs"
    )
    reference_query_parser.add_argument("store", type=Path)
    reference_query_parser.add_argument("output", type=Path)
    reference_query_parser.add_argument("--kind", choices=REFERENCE_KINDS, required=True)
    reference_query_parser.add_argument("--effective-at")
    reference_query_parser.add_argument("--published-at")
    reference_query_parser.add_argument("--ingested-at")
    reference_query_parser.add_argument("--source")

    reference_verify_parser = subparsers.add_parser(
        "reference-verify", help="verify reference records, archives, and manifest chain"
    )
    reference_verify_parser.add_argument("store", type=Path)
    reference_verify_parser.add_argument("--report", type=Path)
    return parser


def _price_payload(args: argparse.Namespace) -> dict[str, Any]:
    module = black_scholes if args.model == "bsm" else bachelier
    result = module.price_and_greeks(
        args.option_type,
        args.spot,
        args.strike,
        args.maturity,
        args.volatility,
        args.rate,
        args.dividend_yield,
    )
    lower, upper = module.bounds(
        args.option_type,
        args.spot,
        args.strike,
        args.maturity,
        args.rate,
        args.dividend_yield,
    )
    return {
        "model": args.model,
        "price": result.price,
        "bounds": {"lower": lower, "upper": upper},
        "greeks": asdict(result.greeks),
        "units": {
            "vega": "per absolute volatility unit",
            "rho": "per absolute rate unit",
            "theta": "calendar-time decay per year",
        },
    }


def _read_password(path: Path | None) -> bytes | None:
    if path is None:
        return None
    payload = path.read_bytes()
    if len(payload) > 4096:
        raise ValueError("password file exceeds 4 KiB")
    password = payload.rstrip(b"\r\n")
    if not password:
        raise ValueError("password file must contain a non-empty password")
    return password


def _write_new_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite existing file: {path}") from exc


def _read_bounded_json(path: Path, name: str) -> str:
    if path.stat().st_size > MAX_DELIVERABLE_BYTES:
        raise ValueError(f"{name} exceeds {MAX_DELIVERABLE_BYTES} bytes")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} must be UTF-8 JSON") from exc


def main(argv: list[str] | None = None) -> int:
    """Run the VolForge CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "research-keygen":
            password = _read_password(args.password_file)
            key_pair = generate_signing_key_pair(password)
            if args.private_key.resolve() == args.public_key.resolve():
                raise ValueError("private and public key paths must differ")
            if args.private_key.exists() or args.public_key.exists():
                raise ValueError("refusing to overwrite an existing key file")
            _write_new_file(args.private_key, key_pair.private_key_pem)
            _write_new_file(args.public_key, key_pair.public_key_pem)
            permissions_hardened = False
            if os.name != "nt":
                os.chmod(args.private_key, 0o600)
                permissions_hardened = True
            print(
                json.dumps(
                    {
                        "algorithm": "ed25519",
                        "private_key": str(args.private_key),
                        "private_key_encrypted": key_pair.private_key_encrypted,
                        "private_key_permissions_hardened": permissions_hardened,
                        "public_key": str(args.public_key),
                        "public_key_sha256": key_pair.public_key_sha256,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "research-sign":
            attestation = sign_research_bundle(
                args.bundle.read_bytes(),
                args.private_key.read_bytes(),
                issued_at=args.issued_at,
                signer=args.signer,
                password=_read_password(args.password_file),
            )
            _write_new_file(args.output, attestation)
            payload = json.loads(attestation)
            print(
                json.dumps(
                    {
                        "attestation": str(args.output),
                        "bundle": str(args.bundle),
                        **payload["statement"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "research-verify":
            verification = verify_bundle_attestation(
                args.bundle.read_bytes(),
                args.attestation.read_bytes(),
                trusted_public_key_pem=args.trusted_public_key.read_bytes(),
            )
            print(json.dumps(verification.to_dict(), indent=2, sort_keys=True))
            return 0 if verification.accepted else 2
        if args.command == "price":
            print(json.dumps(_price_payload(args), indent=2, sort_keys=True))
            return 0
        if args.command == "iv":
            result = implied_volatility(
                args.price,
                args.option_type,
                args.spot,
                args.strike,
                args.maturity,
                args.rate,
                args.dividend_yield,
                model=args.model,
                method=args.method,
                initial_guess=args.initial_guess,
            )
            print(json.dumps(asdict(result), indent=2, sort_keys=True))
            return 0 if result.converged else 2
        if args.command == "deliverable-value":
            terms = parse_deliverable_terms(
                _read_bounded_json(args.deliverable, "deliverable_json"),
                settlement_method=args.settlement_method,
                contract_multiplier=args.contract_multiplier,
                currency=args.currency,
                underlying_id=args.underlying_id,
            )
            marks = (
                _read_bounded_json(args.marks, "security_marks") if args.marks is not None else None
            )
            valuation = value_option_settlement(
                terms,
                option_type=args.option_type,
                strike=args.strike,
                security_marks=marks,
                underlying_mark=args.underlying_mark,
                settlement_level=args.settlement_level,
            )
            payload = {
                **valuation.to_dict(),
                "canonical_deliverable_json": terms.canonical_json,
                "required_security_marks": list(terms.required_security_marks),
                "single_factor_pricing_compatible": terms.single_factor_pricing_compatible,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "basket-price":
            terms = parse_deliverable_terms(
                _read_bounded_json(args.deliverable, "deliverable_json"),
                settlement_method=args.settlement_method,
                contract_multiplier=args.contract_multiplier,
                currency=args.currency,
                underlying_id=args.underlying_id,
            )
            market = parse_basket_market(_read_bounded_json(args.market, "basket_market"))
            result = price_basket_option(
                terms,
                market,
                option_type=args.option_type,
                strike=args.strike,
                maturity=args.maturity,
                rate=args.rate,
                settings=BasketMonteCarloSettings(
                    paths=args.paths,
                    seed=args.seed,
                    antithetic=not args.no_antithetic,
                    control_variate=not args.no_control_variate,
                    confidence_level=args.confidence_level,
                ),
            )
            payload = {
                **result.to_dict(),
                "canonical_deliverable_json": terms.canonical_json,
                "canonical_market_json": market.canonical_json,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "generate-basket-inputs":
            records, schedule = generate_synthetic_basket_inputs()
            args.records_output.parent.mkdir(parents=True, exist_ok=True)
            args.schedule_output.parent.mkdir(parents=True, exist_ok=True)
            args.records_output.write_bytes(dataframe_to_csv(records))
            args.schedule_output.write_bytes(dataframe_to_csv(schedule))
            print(
                json.dumps(
                    {
                        "basket_id": SYNTHETIC_BASKET_ID,
                        "records_output": str(args.records_output),
                        "schedule_output": str(args.schedule_output),
                        "records": len(records),
                        "snapshots": len(schedule),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command in {
            "generate-basket-statistical-inputs",
            "generate-basket-smile-inputs",
        }:
            smile_hedges = args.command == "generate-basket-smile-inputs"
            records, schedule = (
                generate_synthetic_smile_basket_inputs()
                if smile_hedges
                else generate_synthetic_statistical_basket_inputs()
            )
            args.records_output.parent.mkdir(parents=True, exist_ok=True)
            args.schedule_output.parent.mkdir(parents=True, exist_ok=True)
            args.records_output.write_bytes(dataframe_to_csv(records))
            args.schedule_output.write_bytes(dataframe_to_csv(schedule))
            print(
                json.dumps(
                    {
                        "basket_id": SYNTHETIC_BASKET_ID,
                        "records_output": str(args.records_output),
                        "schedule_output": str(args.schedule_output),
                        "records": len(records),
                        "snapshots": len(schedule),
                        "statistically_estimated": True,
                        "smile_risk_instruments": smile_hedges,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "generate-basket-option-inputs":
            records, schedule = generate_synthetic_option_quote_inputs()
            args.records_output.parent.mkdir(parents=True, exist_ok=True)
            args.schedule_output.parent.mkdir(parents=True, exist_ok=True)
            args.records_output.write_bytes(dataframe_to_csv(records))
            args.schedule_output.write_bytes(dataframe_to_csv(schedule))
            hedge_ids = sorted(records.loc[records["role"] == "hedge", "instrument_id"].unique())
            print(
                json.dumps(
                    {
                        "basket_id": SYNTHETIC_BASKET_ID,
                        "records_output": str(args.records_output),
                        "schedule_output": str(args.schedule_output),
                        "records": len(records),
                        "formations": len(schedule),
                        "target_instrument_id": SYNTHETIC_TARGET_INSTRUMENT_ID,
                        "hedge_instrument_ids": hedge_ids,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "basket-option-init":
            store = BasketOptionQuoteStore.create(args.store)
            print(
                json.dumps(
                    {
                        "schema_version": BASKET_OPTION_QUOTE_SCHEMA_VERSION,
                        "store": str(store.path),
                        "raw_archive": str(store.raw_directory),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "basket-option-ingest":
            store = BasketOptionQuoteStore(args.store)
            batch = store.ingest_csv(
                args.input,
                source=args.source,
                mode=args.mode,
                ingested_timestamp=args.ingested_at,
                reason=args.reason,
            )
            audit = basket_option_quote_report_json(batch)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_bytes(audit)
            print(audit.decode(), end="")
            return 0
        if args.command == "basket-option-materialize":
            store = BasketOptionQuoteStore(args.store)
            replay = store.materialize_schedule(
                pd.read_csv(args.schedule),
                basket_id=args.basket_id,
                source=args.source,
                target_instrument_id=args.target_instrument_id,
                hedge_instrument_ids=args.hedge_instrument_id,
                maximum_age=args.maximum_age,
                dataset_known_at=args.dataset_known_at,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(dataframe_to_csv(replay))
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "rows": len(replay),
                        "option_quote_provenance_bound": True,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "basket-option-verify":
            store = BasketOptionQuoteStore(args.store)
            verification = store.verify()
            audit = basket_option_quote_report_json(verification)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_bytes(audit)
            print(audit.decode(), end="")
            return 0 if verification.valid else 2
        if args.command == "basket-input-init":
            store = BasketInputStore.create(args.store)
            print(
                json.dumps(
                    {
                        "schema_version": BASKET_INPUT_SCHEMA_VERSION,
                        "store": str(store.path),
                        "raw_archive": str(store.raw_directory),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command in {"basket-input-ingest", "basket-input-fetch"}:
            store = BasketInputStore(args.store)
            if args.command == "basket-input-ingest":
                batch = store.ingest_csv(
                    args.input,
                    source=args.source,
                    mode=args.mode,
                    ingested_timestamp=args.ingested_at,
                    reason=args.reason,
                )
            else:
                batch = store.ingest_adapter(
                    HttpsJsonBasketInputAdapter(
                        args.endpoint,
                        authorization_env=args.authorization_env,
                        timeout_seconds=args.timeout_seconds,
                    ),
                    source=args.source,
                    mode=args.mode,
                    ingested_timestamp=args.ingested_at,
                    reason=args.reason,
                )
            audit = basket_input_report_json(batch)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_bytes(audit)
            print(audit.decode(), end="")
            return 0
        if args.command == "basket-input-query":
            store = BasketInputStore(args.store)
            records = store.records_as_known(
                args.formation_at,
                basket_id=args.basket_id,
                source=args.source,
                dataset_known_at=args.dataset_known_at,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(dataframe_to_csv(records))
            print(json.dumps({"output": str(args.output), "rows": len(records)}, sort_keys=True))
            return 0
        if args.command == "basket-input-materialize":
            store = BasketInputStore(args.store)
            replay = store.materialize_replay(
                pd.read_csv(args.schedule),
                basket_id=args.basket_id,
                source=args.source,
                maximum_age=args.maximum_age,
                dataset_known_at=args.dataset_known_at,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(dataframe_to_csv(replay))
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "rows": len(replay),
                        "store_provenance_bound": True,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "basket-input-estimate":
            store = BasketInputStore(args.store)
            replay = materialize_estimated_replay(
                store,
                pd.read_csv(args.schedule),
                basket_id=args.basket_id,
                source=args.source,
                maximum_age=args.maximum_age,
                dataset_known_at=args.dataset_known_at,
                settings=BasketEstimationSettings(
                    window_returns=args.window_returns,
                    minimum_returns=args.minimum_returns,
                    annualization_factor=args.annualization_factor,
                    correlation_shrinkage=args.correlation_shrinkage,
                    estimator=args.estimator,
                    ewma_decay=args.ewma_decay,
                    covariance_shrinkage=args.covariance_shrinkage,
                    volatility_floor=args.volatility_floor,
                ),
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(dataframe_to_csv(replay))
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "rows": len(replay),
                        "estimate_provenance_bound": True,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "basket-covariance-select":
            store = BasketInputStore(args.store)
            selection = select_basket_covariance(
                store,
                args.basket_id,
                args.formation_at,
                source=args.source,
                maximum_age=args.maximum_age,
                dataset_known_at=args.dataset_known_at,
                settings=BasketCovarianceSelectionSettings(
                    training_window=args.training_window,
                    minimum_training_returns=args.minimum_training_returns,
                    evaluation_returns=args.evaluation_returns,
                    annualization_factor=args.annualization_factor,
                    correlation_shrinkage=args.correlation_shrinkage,
                    covariance_shrinkage=args.covariance_shrinkage,
                    ewma_decay=args.ewma_decay,
                    volatility_floor=args.volatility_floor,
                    eigenvalue_floor=args.eigenvalue_floor,
                    maximum_condition_number=args.maximum_condition_number,
                    minimum_nll_improvement=args.minimum_nll_improvement,
                    forecast_distribution=args.forecast_distribution,
                    student_t_degrees_of_freedom=args.student_t_degrees_of_freedom,
                    selection_objective=args.selection_objective,
                    stress_quantile=args.stress_quantile,
                    stress_weight=args.stress_weight,
                    minimum_stress_evaluations=args.minimum_stress_evaluations,
                    maximum_stress_nll_regret=args.maximum_stress_nll_regret,
                ),
            )
            audit = basket_covariance_selection_report_json(selection)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_bytes(audit)
            print(audit.decode(), end="")
            return 0 if selection.downstream_eligible else 2
        if args.command == "basket-input-select-covariance":
            store = BasketInputStore(args.store)
            replay = materialize_selected_covariance_replay(
                store,
                pd.read_csv(args.schedule),
                basket_id=args.basket_id,
                source=args.source,
                maximum_age=args.maximum_age,
                dataset_known_at=args.dataset_known_at,
                settings=BasketCovarianceSelectionSettings(
                    training_window=args.training_window,
                    minimum_training_returns=args.minimum_training_returns,
                    evaluation_returns=args.evaluation_returns,
                    annualization_factor=args.annualization_factor,
                    correlation_shrinkage=args.correlation_shrinkage,
                    covariance_shrinkage=args.covariance_shrinkage,
                    ewma_decay=args.ewma_decay,
                    volatility_floor=args.volatility_floor,
                    eigenvalue_floor=args.eigenvalue_floor,
                    maximum_condition_number=args.maximum_condition_number,
                    minimum_nll_improvement=args.minimum_nll_improvement,
                    forecast_distribution=args.forecast_distribution,
                    student_t_degrees_of_freedom=args.student_t_degrees_of_freedom,
                    selection_objective=args.selection_objective,
                    stress_quantile=args.stress_quantile,
                    stress_weight=args.stress_weight,
                    minimum_stress_evaluations=args.minimum_stress_evaluations,
                    maximum_stress_nll_regret=args.maximum_stress_nll_regret,
                ),
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(dataframe_to_csv(replay))
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "rows": len(replay),
                        "estimate_provenance_bound": True,
                        "covariance_selection_provenance_bound": True,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "basket-input-verify":
            store = BasketInputStore(args.store)
            verification = store.verify()
            audit = basket_input_report_json(verification)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_bytes(audit)
            print(audit.decode(), end="")
            return 0 if verification.valid else 2
        if args.command == "generate-basket-replay":
            frame = generate_synthetic_basket_replay(paths=args.paths, seed=args.seed)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.deliverable_output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(dataframe_to_csv(frame))
            args.deliverable_output.write_text(
                SYNTHETIC_BASKET_DELIVERABLE + "\n", encoding="utf-8", newline="\n"
            )
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "deliverable_output": str(args.deliverable_output),
                        "expiry_timestamp": SYNTHETIC_BASKET_EXPIRY,
                        "rows": len(frame),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "basket-replay":
            terms = parse_deliverable_terms(
                _read_bounded_json(args.deliverable, "deliverable_json"),
                settlement_method=args.settlement_method,
                contract_multiplier=args.contract_multiplier,
                currency=args.currency,
                underlying_id=args.underlying_id,
            )
            result = run_basket_hedge_replay(
                pd.read_csv(args.input),
                terms,
                option_type=args.option_type,
                strike=args.strike,
                expiry_timestamp=args.expiry,
                settings=BasketReplaySettings(
                    paths=args.paths,
                    seed=args.seed,
                    antithetic=not args.no_antithetic,
                    control_variate=not args.no_control_variate,
                    confidence_level=args.confidence_level,
                    short_option_contracts=args.short_option_contracts,
                    component_cost_bps=args.component_cost_bps,
                    option_cost_bps=args.option_cost_bps,
                    maximum_market_age_seconds=args.maximum_market_age_seconds,
                    require_store_provenance=args.require_store_provenance,
                    require_estimated_inputs=args.require_estimated_inputs,
                ),
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(dataframe_to_csv(result.ledger_frame()))
            audit = basket_replay_report_json(result)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_bytes(audit)
            print(audit.decode(), end="")
            return 0 if result.downstream_eligible else 2
        if args.command == "basket-risk-replay":
            terms = parse_deliverable_terms(
                _read_bounded_json(args.deliverable, "deliverable_json"),
                settlement_method=args.settlement_method,
                contract_multiplier=args.contract_multiplier,
                currency=args.currency,
                underlying_id=args.underlying_id,
            )
            result = run_basket_risk_hedge_replay(
                pd.read_csv(args.input),
                terms,
                option_type=args.option_type,
                strike=args.strike,
                expiry_timestamp=args.expiry,
                settings=BasketRiskHedgeSettings(
                    paths=args.paths,
                    seed=args.seed,
                    component_cost_bps=args.component_cost_bps,
                    hedge_option_cost_bps=args.hedge_option_cost_bps,
                    target_option_cost_bps=args.target_option_cost_bps,
                    include_smile_risks=args.include_smile_risks,
                    smile_slope_bump=args.smile_slope_bump,
                    smile_curvature_bump=args.smile_curvature_bump,
                    maximum_absolute_smile_log_moneyness=(
                        args.maximum_absolute_smile_log_moneyness
                    ),
                    minimum_normalized_singular_value=(args.minimum_normalized_singular_value),
                    maximum_condition_number=args.maximum_condition_number,
                    maximum_absolute_option_position=args.maximum_absolute_option_position,
                    require_estimate_provenance=not args.allow_unestimated_inputs,
                    require_covariance_selection=args.require_covariance_selection,
                    require_option_quote_provenance=args.require_option_quote_provenance,
                ),
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(dataframe_to_csv(result.ledger_frame()))
            audit = basket_risk_hedge_report_json(result)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_bytes(audit)
            print(audit.decode(), end="")
            return 0 if result.downstream_eligible else 2
        if args.command == "history-init":
            store = HistoricalEventStore.create(args.store)
            print(
                json.dumps(
                    {
                        "schema_version": HISTORY_SCHEMA_VERSION,
                        "store": str(store.path),
                        "raw_archive": str(store.raw_directory),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "history-ingest":
            store = HistoricalEventStore(args.store)
            batch = store.ingest_csv(
                args.input,
                source=args.source,
                mode=args.mode,
                ingested_timestamp=args.ingested_at,
                reason=args.reason,
            )
            audit = history_report_json(batch)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_bytes(audit)
            print(audit.decode(), end="")
            return 0
        if args.command == "history-query":
            store = HistoricalEventStore(args.store)
            events = store.events_as_known(
                args.record_type,
                event_start=args.event_start,
                event_end=args.event_end,
                received_at=args.received_at,
                ingested_at=args.ingested_at,
                source=args.source,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(dataframe_to_csv(events))
            print(json.dumps({"output": str(args.output), "rows": len(events)}, sort_keys=True))
            return 0
        if args.command == "history-snapshot":
            store = HistoricalEventStore(args.store)
            reference_store = (
                ReferenceDataStore(args.reference_store) if args.reference_store else None
            )
            snapshots = (
                store.chain_snapshot(
                    args.snapshot_at[0],
                    source=args.source,
                    maximum_quote_age=args.maximum_quote_age,
                    dataset_known_at=args.dataset_known_at,
                    reference_store=reference_store,
                    reference_ingested_at=args.reference_known_at,
                )
                if len(args.snapshot_at) == 1
                else store.replay_chain(
                    args.snapshot_at,
                    source=args.source,
                    maximum_quote_age=args.maximum_quote_age,
                    dataset_known_at=args.dataset_known_at,
                    reference_store=reference_store,
                    reference_ingested_at=args.reference_known_at,
                )
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(dataframe_to_csv(snapshots))
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "rows": len(snapshots),
                        "snapshots": (
                            int(snapshots["quote_timestamp"].nunique())
                            if not snapshots.empty
                            else 0
                        ),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "history-verify":
            store = HistoricalEventStore(args.store)
            verification = store.verify()
            audit = history_report_json(verification)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_bytes(audit)
            print(audit.decode(), end="")
            return 0 if verification.valid else 2
        if args.command == "reference-init":
            store = ReferenceDataStore.create(args.store)
            print(
                json.dumps(
                    {
                        "schema_version": REFERENCE_SCHEMA_VERSION,
                        "store": str(store.path),
                        "raw_archive": str(store.raw_directory),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "reference-ingest":
            store = ReferenceDataStore(args.store)
            batch = store.ingest_csv(
                args.input,
                kind=args.kind,
                source=args.source,
                mode=args.mode,
                ingested_timestamp=args.ingested_at,
                reason=args.reason,
            )
            audit = reference_report_json(batch)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_bytes(audit)
            print(audit.decode(), end="")
            return 0
        if args.command == "reference-query":
            store = ReferenceDataStore(args.store)
            records = store.records_as_known(
                args.kind,
                effective_at=args.effective_at,
                published_at=args.published_at,
                ingested_at=args.ingested_at,
                source=args.source,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(dataframe_to_csv(records))
            print(json.dumps({"output": str(args.output), "rows": len(records)}, sort_keys=True))
            return 0
        if args.command == "reference-verify":
            store = ReferenceDataStore(args.store)
            verification = store.verify()
            audit = reference_report_json(verification)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_bytes(audit)
            print(audit.decode(), end="")
            return 0 if verification.valid else 2
        if args.command == "generate-replay":
            chain = generate_synthetic_replay(
                SyntheticReplayConfig(
                    spot=args.spot,
                    atm_volatility=args.atm_volatility,
                    bid_ask_spread=args.spread,
                )
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(dataframe_to_csv(chain))
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "rows": len(chain),
                        "snapshots": int(chain["quote_timestamp"].nunique()),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "replay":
            if (args.target_expiration is None) != (args.target_strike is None):
                raise ValueError("target-expiration and target-strike must be supplied together")
            raw_chain = pd.read_csv(args.input)
            target = (
                ListedContract(
                    args.target_expiration,
                    args.target_strike,
                    args.target_option_type,
                )
                if args.target_expiration is not None
                else None
            )
            result = run_chronological_replay(
                raw_chain,
                settings=ReplaySettings(
                    surface_model=args.surface_model,
                    strategy=args.strategy,
                    stock_cost_bps=args.stock_cost_bps,
                    option_cost_bps=args.option_cost_bps,
                    max_option_position=args.max_option_position,
                    condition_limit=args.condition_limit,
                    heston_max_function_evaluations=args.max_function_evaluations,
                    require_feller=args.require_feller,
                    cos_terms=args.cos_terms,
                    cos_truncation_width=args.truncation_width,
                ),
                target_contract=target,
                target_option_type=args.target_option_type,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(dataframe_to_csv(result.ledger_frame()))
            audit = replay_report_json(result)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_bytes(audit)
            print(audit.decode(), end="")
            return 0 if result.downstream_eligible else 2
        if args.command == "benchmark":
            raw_chain = pd.read_csv(args.input)
            result = run_chronological_benchmark(
                raw_chain,
                settings=BenchmarkSettings(
                    training_window=args.training_window,
                    surface_model=args.surface_model,
                    heston_max_function_evaluations=args.max_function_evaluations,
                    require_feller=args.require_feller,
                    cos_terms=args.cos_terms,
                    cos_truncation_width=args.truncation_width,
                    comparison_loss=args.comparison_loss,
                    bootstrap_resamples=args.bootstrap_resamples,
                    bootstrap_block_length=args.bootstrap_block_length,
                    bootstrap_seed=args.bootstrap_seed,
                    comparison_hac_lags=args.comparison_hac_lags,
                    comparison_confidence_level=args.comparison_confidence_level,
                    minimum_comparison_dates=args.minimum_comparison_dates,
                ),
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(dataframe_to_csv(result.prediction_frame()))
            audit = benchmark_report_json(result)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_bytes(audit)
            if args.bundle:
                args.bundle.parent.mkdir(parents=True, exist_ok=True)
                args.bundle.write_bytes(build_benchmark_research_bundle(result))
            print(audit.decode(), end="")
            return 0 if result.downstream_eligible else 2
        if args.command == "heston-price":
            parameters = HestonParameters(
                args.kappa,
                args.theta,
                args.vol_of_vol,
                args.rho,
                args.initial_variance,
            )
            result = price_with_diagnostics(
                args.option_type,
                args.spot,
                args.strike,
                args.maturity,
                parameters,
                args.rate,
                args.dividend_yield,
                settings=COSSettings(args.cos_terms, args.truncation_width),
            )
            print(
                json.dumps(
                    {"model": "heston", "parameters": asdict(parameters), **asdict(result)},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "american-binomial":
            result = american_binomial_price(
                args.option_type,
                args.spot,
                args.strike,
                args.maturity,
                args.volatility,
                args.rate,
                args.dividend_yield,
                settings=BinomialSettings(args.steps),
            )
            print(
                json.dumps({"engine": "crr-binomial", **asdict(result)}, indent=2, sort_keys=True)
            )
            return 0
        if args.command in {"american-lsm", "american-pde"}:
            parameters = HestonParameters(
                args.kappa,
                args.theta,
                args.vol_of_vol,
                args.rho,
                args.initial_variance,
            )
            if args.command == "american-lsm":
                result = heston_lsm_price(
                    args.option_type,
                    args.spot,
                    args.strike,
                    args.maturity,
                    parameters,
                    args.rate,
                    args.dividend_yield,
                    settings=LSMSettings(
                        paths=args.paths,
                        steps=args.steps,
                        seed=args.seed,
                        antithetic=not args.no_antithetic,
                    ),
                )
                engine = "heston-longstaff-schwartz"
            else:
                result = heston_american_pde_price(
                    args.option_type,
                    args.spot,
                    args.strike,
                    args.maturity,
                    parameters,
                    args.rate,
                    args.dividend_yield,
                    settings=PDEGridSettings(
                        log_price_points=args.log_price_points,
                        variance_points=args.variance_points,
                        time_steps=args.time_steps,
                        log_price_half_width=args.log_price_half_width,
                        variance_max=args.variance_max,
                        rannacher_steps=args.rannacher_steps,
                    ),
                )
                engine = "heston-ikonen-toivanen-pde"
            print(
                json.dumps(
                    {"engine": engine, "parameters": asdict(parameters), **asdict(result)},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "heston-greeks":
            parameters = HestonParameters(
                args.kappa,
                args.theta,
                args.vol_of_vol,
                args.rho,
                args.initial_variance,
            )
            if args.engine == "mc":
                result = heston_monte_carlo_greeks(
                    args.option_type,
                    args.spot,
                    args.strike,
                    args.maturity,
                    parameters,
                    args.rate,
                    args.dividend_yield,
                    settings=MonteCarloGreekSettings(
                        paths=args.paths,
                        steps=args.mc_steps,
                        seed=args.seed,
                        antithetic=not args.no_antithetic,
                        initial_volatility_relative_bump=args.volatility_bump,
                        maturity_relative_bump=args.maturity_bump,
                    ),
                )
                payload = {
                    "engine": "heston-qe-monte-carlo-greeks",
                    "parameters": asdict(parameters),
                    **asdict(result),
                }
            else:
                result = heston_finite_difference_greeks(
                    args.option_type,
                    args.spot,
                    args.strike,
                    args.maturity,
                    parameters,
                    args.rate,
                    args.dividend_yield,
                    engine=args.engine,
                    settings=FiniteDifferenceSettings(
                        spot_relative_bump=args.spot_bump,
                        initial_volatility_relative_bump=args.volatility_bump,
                        maturity_relative_bump=args.maturity_bump,
                        rate_bump=args.rate_bump,
                    ),
                    cos_settings=COSSettings(args.cos_terms, args.truncation_width),
                    pde_settings=PDEGridSettings(
                        log_price_points=args.log_price_points,
                        variance_points=args.variance_points,
                        time_steps=args.pde_time_steps,
                        log_price_half_width=args.log_price_half_width,
                        variance_max=args.variance_max,
                        rannacher_steps=args.rannacher_steps,
                    ),
                )
                payload = {"parameters": asdict(parameters), **asdict(result)}
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "heston-hedge":
            parameters = HestonParameters(
                args.kappa,
                args.theta,
                args.vol_of_vol,
                args.rho,
                args.initial_variance,
            )
            result = run_heston_hedge_backtest(
                args.option_type,
                args.spot,
                args.strike,
                args.maturity,
                parameters,
                args.rate,
                args.dividend_yield,
                settings=HedgeBacktestSettings(
                    paths=args.paths,
                    rebalancing_steps=args.rebalancing_steps,
                    seed=args.seed,
                    antithetic=not args.no_antithetic,
                    stock_cost_bps=args.stock_cost_bps,
                    option_cost_bps=args.option_cost_bps,
                    hedge_strike_low=args.hedge_strike_low,
                    hedge_strike_high=args.hedge_strike_high,
                    max_option_position=args.max_option_position,
                    condition_limit=args.condition_limit,
                ),
                cos_settings=COSSettings(args.cos_terms, args.truncation_width),
            )
            payload = {
                "engine": "heston-self-financing-hedge-backtest",
                "parameters": asdict(parameters),
                **asdict(result),
            }
            audit = json.dumps(payload, indent=2, sort_keys=True).encode()
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_bytes(audit)
            print(audit.decode())
            return 0
        if args.command == "heston-robustness":
            parameters = HestonParameters(
                args.kappa,
                args.theta,
                args.vol_of_vol,
                args.rho,
                args.initial_variance,
            )
            result = run_hedge_robustness_gate(
                args.option_type,
                args.spot,
                args.strike,
                args.maturity,
                parameters,
                args.rate,
                args.dividend_yield,
                settings=HedgeRobustnessSettings(
                    seeds=tuple(args.seeds),
                    path_counts=tuple(args.path_counts),
                    rebalancing_steps=tuple(args.rebalancing_steps),
                    antithetic=not args.no_antithetic,
                    stock_cost_bps=args.stock_cost_bps,
                    option_cost_bps=args.option_cost_bps,
                    cost_stress_multiplier=args.cost_stress_multiplier,
                    hedge_strike_low=args.hedge_strike_low,
                    hedge_strike_high=args.hedge_strike_high,
                    max_option_position=args.max_option_position,
                    condition_limit=args.condition_limit,
                    physical_annual_return=args.physical_annual_return,
                    physical_kappa_multiplier=args.physical_kappa_multiplier,
                    physical_theta_multiplier=args.physical_theta_multiplier,
                    physical_vol_of_vol_multiplier=args.physical_vol_of_vol_multiplier,
                    physical_rho_shift=args.physical_rho_shift,
                    absolute_tolerance=args.absolute_tolerance,
                    relative_tolerance=args.relative_tolerance,
                    minimum_volatility_reduction=args.minimum_volatility_reduction,
                    maximum_seed_range_z=args.maximum_seed_range_z,
                    maximum_clipped_fraction=args.maximum_clipped_fraction,
                    maximum_fallback_fraction=args.maximum_fallback_fraction,
                    attribution_tolerance=args.attribution_tolerance,
                ),
                cos_settings=COSSettings(args.cos_terms, args.truncation_width),
            )
            audit = hedge_robustness_report_json(result)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_bytes(audit)
            print(audit.decode(), end="")
            return 0 if result.downstream_eligible else 2
        if args.command in {"heston-mc", "heston-pde", "heston-converge"}:
            parameters = HestonParameters(
                args.kappa,
                args.theta,
                args.vol_of_vol,
                args.rho,
                args.initial_variance,
            )
            if args.command == "heston-converge":
                convergence_settings = ConvergenceSettings(
                    absolute_tolerance=args.absolute_tolerance,
                    relative_tolerance=args.relative_tolerance,
                    max_cos_terms=args.max_cos_terms,
                    max_cos_truncation_width=args.max_cos_width,
                    max_monte_carlo_paths=args.max_paths,
                    max_monte_carlo_steps=args.max_mc_steps,
                    seed=args.seed,
                    antithetic=not args.no_antithetic,
                    control_variate=not args.no_control_variate,
                    max_pde_log_price_points=args.max_log_price_points,
                    max_pde_variance_points=args.max_variance_points,
                    max_pde_time_steps=args.max_pde_time_steps,
                    max_pde_log_price_half_width=args.max_log_price_half_width,
                    max_pde_variance=args.max_variance,
                )
                result = assess_heston_convergence(
                    args.option_type,
                    args.spot,
                    args.strike,
                    args.maturity,
                    parameters,
                    args.rate,
                    args.dividend_yield,
                    settings=convergence_settings,
                )
                audit = convergence_report_json(result)
                if args.report:
                    args.report.parent.mkdir(parents=True, exist_ok=True)
                    args.report.write_bytes(audit)
                print(audit.decode(), end="")
                return 0 if result.downstream_eligible else 2
            if args.command == "heston-mc":
                result = heston_monte_carlo(
                    args.option_type,
                    args.spot,
                    args.strike,
                    args.maturity,
                    parameters,
                    args.rate,
                    args.dividend_yield,
                    settings=MonteCarloSettings(
                        paths=args.paths,
                        steps=args.steps,
                        seed=args.seed,
                        antithetic=not args.no_antithetic,
                        control_variate=not args.no_control_variate,
                    ),
                )
                engine = "heston-qe-monte-carlo"
            else:
                result = heston_pde_price(
                    args.option_type,
                    args.spot,
                    args.strike,
                    args.maturity,
                    parameters,
                    args.rate,
                    args.dividend_yield,
                    settings=PDEGridSettings(
                        log_price_points=args.log_price_points,
                        variance_points=args.variance_points,
                        time_steps=args.time_steps,
                        log_price_half_width=args.log_price_half_width,
                        variance_max=args.variance_max,
                        rannacher_steps=args.rannacher_steps,
                    ),
                )
                engine = "heston-finite-difference-pde"
            print(
                json.dumps(
                    {"engine": engine, "parameters": asdict(parameters), **asdict(result)},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "generate":
            config = SyntheticChainConfig(
                seed=args.seed,
                spot=args.spot,
                base_volatility=args.base_volatility,
                skew=args.skew,
                curvature=args.curvature,
                bid_ask_spread=args.spread,
                quote_noise_std=args.noise,
            )
            chain = generate_synthetic_chain(config)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(dataframe_to_csv(chain))
            print(json.dumps({"output": str(args.output), "rows": len(chain)}, sort_keys=True))
            return 0
        if args.command == "surface":
            if args.grid_points < 21 or args.grid_min >= args.grid_max:
                raise ValueError(
                    "surface grid requires at least 21 points and grid-min below grid-max"
                )
            chain = load_csv(args.input, model="bsm")
            quotes = prepare_surface_quotes(chain)
            diagnostic_grid = np.linspace(args.grid_min, args.grid_max, args.grid_points)
            calibration = (
                calibrate_svi_surface(quotes, diagnostic_grid=diagnostic_grid)
                if args.model == "svi"
                else calibrate_ssvi(quotes, diagnostic_grid=diagnostic_grid)
            )
            evaluated = calibration.evaluated_frame(diagnostic_grid)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(dataframe_to_csv(evaluated))
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_bytes(calibration_report_json(calibration))
            print(json.dumps(calibration_summary(calibration), indent=2, sort_keys=True))
            return 0 if calibration.downstream_eligible else 2
        if args.command == "heston-calibrate":
            if args.moneyness_points < 5 or args.moneyness_min >= args.moneyness_max:
                raise ValueError(
                    "Heston grid requires at least five points and minimum below maximum"
                )
            chain = load_csv(args.input, model="bsm")
            quotes = prepare_surface_quotes(chain)
            diagnostic_grid = np.linspace(-1.25, 1.25, 161)
            surface = (
                calibrate_svi_surface(quotes, diagnostic_grid=diagnostic_grid)
                if args.surface_model == "svi"
                else calibrate_ssvi(quotes, diagnostic_grid=diagnostic_grid)
            )
            calibration = calibrate_heston(
                surface,
                log_moneyness=np.linspace(
                    args.moneyness_min, args.moneyness_max, args.moneyness_points
                ),
                settings=COSSettings(args.cos_terms, args.truncation_width),
                require_feller=args.require_feller,
                max_function_evaluations=args.max_function_evaluations,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(dataframe_to_csv(calibration.evaluated_frame()))
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_bytes(heston_calibration_report_json(calibration))
            print(json.dumps(heston_calibration_summary(calibration), indent=2, sort_keys=True))
            return 0 if calibration.calibration_eligible else 2
        chain = load_csv(
            args.input,
            model=args.model,
            valuation_time=args.valuation_time,
            stale_after=args.stale_after,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(dataframe_to_csv(chain))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(quality_report_markdown(chain), encoding="utf-8")
        print(json.dumps(quality_summary(chain), indent=2, sort_keys=True, allow_nan=True))
        return 0
    except (ArithmeticError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"volforge: error: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
