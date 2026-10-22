# Statistical basket inputs and nonlinear risk hedging

VolForge v1.7 adds a stricter research path for adjusted basket options. It estimates volatility
and correlation from point-in-time spot histories, binds those estimates to the append-only input
store, and hedges a short target option with three explicitly quoted basket options plus the
underlying components.

This workflow is separate from the v1.5/v1.6 delta-only replay. Existing replay files remain valid;
callers opt into the statistical-input gate with `require_estimated_inputs` or use the dedicated
advanced ledger, which requires estimate provenance by default.

## Point-in-time estimator

For each formation timestamp, the estimator:

1. verifies the input database and resolves the newest complete executable component snapshot;
2. queries only records whose market and receipt timestamps are no later than formation, subject
   to an optional ingestion-time dataset cutoff;
3. retains only synchronized snapshots containing every current basket component exactly once;
4. takes the last `window_returns + 1` complete price observations;
5. calculates log returns, annualized sample volatility, and sample correlation; and
6. shrinks correlation toward the identity matrix:

   `C_shrunk = (1 - lambda) C_sample + lambda I`.

The annualization factor, rolling window, minimum return count, shrinkage weight, and positive
volatility floor are explicit settings. Missing observations are never forward-filled, incomplete
packages are skipped, ambiguous complete packages at one market time block the estimate, and an
undefined or non-PSD result fails closed.

Every `BasketInputEstimate` records its observation interval and count, sample and shrunk minimum
eigenvalues, estimated market, current execution quotes, contributing record/manifests, resolved
store snapshot, and a canonical SHA-256 self-hash. `materialize_estimated_replay` writes both the
hash and full canonical estimate into each replay row. The delta-only replay can require and verify
that binding with `BasketReplaySettings(require_estimated_inputs=True)`.

The default annualization factor of 252 assumes one consistently spaced trading observation per
return. Production users must choose an observation clock and factor appropriate to their data;
the estimator does not infer irregular-time scaling.

VolForge v1.8 can replace the fixed baseline with a chronologically selected sample-shrinkage,
EWMA, or diagonal-shrinkage covariance estimate. The alternative workflow preserves this estimate
contract and adds a separate fold-by-fold selection audit. See
[`basket-covariance-validation.md`](basket-covariance-validation.md).

## Explicit hedge instruments

Each advanced replay row supplies exactly three `hedge_instruments_json` objects. An object must
contain:

- stable `instrument_id`, `option_type`, `strike`, and timezone-aware `expiry_timestamp`;
- executable `bid` and `ask`;
- `settlement_method`, `contract_multiplier`, `currency`, and `underlying_id`; and
- an explicit `deliverable_json` package.

Instrument definitions must remain unchanged through the replay, use exactly the basket's risk
factors, and expire after the target. Quotes may change at every formation. The ledger buys at ask,
sells at bid, marks at midpoint, applies explicit option costs, and liquidates every hedge option
at the target expiry. A model value is never booked as an executable fill.

## Risk definitions and solve

All bump prices within one snapshot use the same Monte Carlo seed and path controls. For option
value `V`, VolForge reports:

- component pathwise deltas from the correlated-GBM engine;
- uniform relative-spot gamma
  `[V(S(1+h)) - 2V(S) + V(S(1-h))] / h^2`;
- parallel absolute-volatility vega from a centered bump; and
- correlation-strength sensitivity
  `[V(C) - V((1-h)C + hI)] / h`.

The last measure is sensitivity to shrinking all correlations toward zero; it is not a derivative
with respect to one pair. The gamma is with respect to a dimensionless common spot scale, not one
component spot.

Let the three columns of `A` be the hedge instruments' gamma, vega, and correlation risks and let
`r` be the short target's corresponding risk vector. The ledger solves

`A x = N r`,

where `N` is the number of short target contracts. Rows are scaled before conditioning and solving.
The workflow blocks on a non-finite or excessive condition number, an excessive option position,
or a normalized residual above tolerance. Component positions then neutralize the remaining
per-asset option deltas:

`y_i = N delta_target_i - sum_j x_j delta_hedge_ij`.

These controls reject an unstable apparent hedge instead of silently accepting a pseudoinverse or
clipping positions.

## Self-financing ledger

The target short opens at bid. Hedge options and components execute on their adverse quote sides.
Between formations, cash earns the preceding continuous rate and held components accrue their
declared continuous dividend yields. The ledger retains target, hedge-option, and component cash
flows separately from spread loss, explicit transaction costs, and turnover.

Before each rebalance, the existing portfolio is marked using current component spots and current
option midpoints. After trading, the accounting identity is checked:

`post-trade value = pre-trade value - spread cost - transaction cost`.

At target expiry, the explicit target deliverable settles, every hedge option is liquidated at its
supplied bid or ask, and every component position is closed. Each snapshot hash commits to market
and estimate identities, quotes, risk calculations, solver output, trades, accounting, and the
previous audit hash.

## Fail-closed conditions

The advanced result is downstream eligible only if every row passes:

- complete estimated-input provenance and estimate self-hash verification;
- covariance-selection provenance and estimate binding when
  `require_covariance_selection=True`;
- target/hedge option-quote snapshot self-hashes, manifest identities, and exact contract/market
  binding when `require_option_quote_provenance=True`;
- market/receipt/formation ordering and maximum-age checks;
- fixed risk-factor and instrument identities;
- valid PSD market correlation and complete execution quotes;
- price bounds for every bumped valuation;
- solver conditioning, position, and exposure-residual limits;
- explicit target settlement and hedge liquidation; and
- self-financing and audit-chain verification.

The run stops at the first failing snapshot and does not inspect later rows to repair it.

## CLI example

```bash
uv run volforge generate-basket-statistical-inputs \
  data/statistical-basket-inputs.csv data/basket-risk-schedule.csv
uv run volforge basket-input-init data/basket-inputs.sqlite
uv run volforge basket-input-ingest \
  data/basket-inputs.sqlite data/statistical-basket-inputs.csv \
  --source volforge-synthetic-statistical-v1 --mode initial \
  --ingested-at 2026-05-01T00:00:00Z
uv run volforge basket-input-estimate \
  data/basket-inputs.sqlite data/basket-risk-schedule.csv \
  reports/estimated-basket-replay.csv \
  --basket-id VOLFORGE-SYNTHETIC-ADJUSTED \
  --source volforge-synthetic-statistical-v1
uv run volforge basket-risk-replay \
  reports/estimated-basket-replay.csv data/basket-deliverable.json \
  reports/basket-risk-ledger.csv --report reports/basket-risk-audit.json \
  --settlement-method physical --option-type call --strike 60 \
  --contract-multiplier 100 --currency USD --underlying-id ACME \
  --expiry 2026-04-30T20:00:00+00:00
```

The synthetic fixture derives illustrative hedge quotes from the model solely to keep the example
offline and deterministic. Those quotes are not independent market observations and must not be
used as evidence of hedge availability or performance.

To run the v1.8 selected-covariance path, replace `basket-input-estimate` with
`basket-input-select-covariance`, add `--require-covariance-selection` to `basket-risk-replay`, and
review the explicit option-position limit because a different covariance estimate changes the
hedge system. The deterministic fixture passes with
`--maximum-absolute-option-position 150`; that demo limit is not a production recommendation.

VolForge v1.10 can make that selection Student-t scored and regime balanced. The materialized
selection remains subject to the same self-hash and estimate-binding checks; its audit additionally
commits to training-fitted stress thresholds, common regime labels, coverage, balanced scores, and
stress-regret fallback. See [`basket-covariance-validation.md`](basket-covariance-validation.md).

VolForge v1.9 retains this three-factor workflow as the default and adds an opt-in five-factor
mode. `--include-smile-risks` requires five quoted hedge options and inserts dynamic forward-
moneyness skew and curvature scenarios before correlation strength. The solve must demonstrate
full numerical rank under an explicit normalized-singular-value threshold before conditioning,
position, and residual checks. See [`basket-smile-hedging.md`](basket-smile-hedging.md) for the
scenario definition, audit contract, CLI workflow, and limits.

VolForge v1.11 can source the target and hedge bids/asks through the separate append-only
`BasketOptionQuoteStore`. Its materialized schedule retains quote snapshot and manifest lineage
through both statistical-estimation and covariance-selection stages. Add
`--require-option-quote-provenance` to fail before valuation when that lineage is absent, partial,
tampered, or inconsistent with the target terms or ordered hedge payload. See
[`basket-option-quote-store.md`](basket-option-quote-store.md).

## Scope limits

Neutralizing three declared local bump risks does not make the portfolio risk free. The workflow
does not model jumps, smile dynamics, volatility-surface recalibration, instrument basis, discrete
dividends, borrow, funding asymmetry, margin, impact, partial fills, latency, default, or parameter
uncertainty. Correlated GBM remains a transparent research approximation, and sample estimates can
be unstable even after shrinkage. Production research should provide independently controlled
quotes and evaluate additional scenarios outside this local linear system.
