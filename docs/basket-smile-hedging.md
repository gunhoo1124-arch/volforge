# Dynamic basket smile-risk hedging

VolForge v1.9 extends the advanced adjusted-basket replay from three to five declared nonlinear
risks. The mode is optional and preserves the v1.8 three-factor contract by default. It is a local,
model-derived scenario hedge, not a calibrated basket volatility surface or evidence that the
synthetic instruments can be traded.

## Coordinates and scenarios

For each target or hedge option, the replay recomputes forward log-moneyness at every formation:

`k = log(K * M / F_D)`

where `K` is strike per contract unit, `M` is the contract multiplier, and `F_D` is the expected
forward value of that option's explicit deliverable under the current correlated-GBM market. A
configured absolute-`k` limit blocks extrapolative instruments.

The five-factor vector is ordered and audited as:

1. common relative-spot-scale gamma;
2. parallel absolute-volatility vega;
3. smile-skew sensitivity;
4. smile-curvature sensitivity; and
5. correlation-strength sensitivity.

The skew scenario shifts every component volatility by `b_s * k`; the curvature scenario shifts
it by `b_c * k^2`. Each sensitivity is a centered finite difference with respect to its coefficient
bump, using common random numbers. The implementation reduces an effective bump when needed to
keep down-bumped component volatilities non-negative. These are deliberately transparent local
proxies. They do not evolve or recalibrate an implied-volatility surface.

## Instrument coverage and solve gate

Smile mode requires exactly five stable, explicitly defined basket-option instruments with
formation-time bid/ask quotes. Instrument definitions cannot drift during replay. Different
strikes, expiries, option types, and deliverables create distinct dynamic `k` values and risk
profiles; simply supplying five contracts does not establish coverage.

The five risk vectors form the columns of a scaled square system. Before solving, VolForge computes
its singular values. It blocks unless all five normalized singular values exceed the configured
minimum. The existing maximum condition-number, absolute-position, and normalized residual gates
then apply. VolForge uses an exact solve: it does not silently substitute a pseudoinverse, clip a
position, or drop a risk factor. Component shares neutralize residual per-asset deltas only after
the nonlinear solve passes.

At every formation, the audit records the risk-factor identities, target and instrument `k`, all
five sensitivities, numerical rank, smallest normalized singular value, condition number, hedge
positions, residuals, fills, and accounting. Those fields enter the snapshot hash chain. The
result-level `smile_risk_coverage_bound` flag is true only for a completed smile-mode replay whose
pre-expiry snapshots retain full declared coverage.

## CLI example

```bash
uv run volforge generate-basket-smile-inputs \
  data/smile-basket-inputs.csv data/smile-basket-schedule.csv
uv run volforge basket-input-init data/smile-basket.sqlite
uv run volforge basket-input-ingest \
  data/smile-basket.sqlite data/smile-basket-inputs.csv \
  --source volforge-synthetic-statistical-v1 --mode initial \
  --ingested-at 2026-05-01T00:00:00Z
uv run volforge basket-input-select-covariance \
  data/smile-basket.sqlite data/smile-basket-schedule.csv \
  reports/smile-selected-replay.csv \
  --basket-id VOLFORGE-SYNTHETIC-ADJUSTED \
  --source volforge-synthetic-statistical-v1
uv run volforge basket-risk-replay \
  reports/smile-selected-replay.csv data/basket-deliverable.json \
  reports/smile-hedge-ledger.csv --report reports/smile-hedge-audit.json \
  --settlement-method physical --option-type call --strike 60 \
  --contract-multiplier 100 --currency USD --underlying-id ACME \
  --expiry 2026-04-30T20:00:00+00:00 \
  --include-smile-risks --require-covariance-selection
```

Use `--smile-slope-bump`, `--smile-curvature-bump`,
`--maximum-absolute-smile-log-moneyness`, and
`--minimum-normalized-singular-value` to make the scenario and coverage policy explicit. The
synthetic fixture exists for deterministic offline regression only.

## Limitations

A passing solve neutralizes five local finite-difference exposures within numerical tolerance. It
does not cover smile dynamics between formations, surface recalibration, stochastic skew, jumps,
tails, covariance regimes, volatility/correlation dependence, cross-gamma, liquidity, basis,
impact, margin, borrow, or parameter uncertainty. A production study needs independently sourced
option histories and sensitivity analysis across scenario definitions, bumps, position limits,
and covariance methods.
