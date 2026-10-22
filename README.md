# VolForge

[![CI](https://github.com/gunhoo1124-arch/volforge/actions/workflows/ci.yml/badge.svg)](https://github.com/gunhoo1124-arch/volforge/actions/workflows/ci.yml)
[![CodeQL](https://github.com/gunhoo1124-arch/volforge/actions/workflows/codeql.yml/badge.svg)](https://github.com/gunhoo1124-arch/volforge/actions/workflows/codeql.yml)

VolForge is an offline-first, arbitrage-aware derivatives pricing and volatility research
laboratory. Version 1.11 adds independently sourced, append-only target and hedge-option quote
history while preserving heavy-tail covariance validation, dynamic smile-risk coverage,
chronological model selection, and fail-closed single-factor Heston boundaries.

The long-run research question is:

> Does an arbitrage-free volatility surface combined with a stochastic-volatility model produce
> better out-of-sample pricing and hedging results than constant-volatility Black-Scholes?

VolForge is research software. Its outputs are not investment advice, trading recommendations, or
evidence that an apparent arbitrage is executable.

## Why this project is distinct

VolForge adds derivatives pricing and volatility modeling to the portfolio:

- `ts-regressor` focuses on time-series forecasting and chronological equity prediction.
- `RiskForge` focuses on portfolio market risk, Monte Carlo VaR/ES, stress, and allocation.
- `OrderForge` focuses on limit-order-book dynamics, execution, queue position, and costs.
- **VolForge** focuses on option valuation, implied volatility, static-arbitrage checks, model
  calibration, and hedging experiments.

It deliberately does not duplicate alpha forecasting, general portfolio-risk aggregation, or
market-microstructure execution simulation.

## v1.11 capabilities

- Black-Scholes-Merton calls and puts with continuous rates and dividend yields.
- Bachelier calls and puts with normal volatility in price units per square-root year.
- Analytic delta, gamma, vega, theta, and rho.
- Explicit European no-arbitrage lower and upper bounds.
- Numerically safe expiry and near-zero-volatility conventions.
- Brent implied-volatility inversion plus safeguarded Newton with Brent fallback.
- Structured solve results: convergence, iterations, residual, method, and failure reason.
- Required-schema CSV ingestion that retains and explains invalid rows.
- Deterministic multi-expiry, multi-strike call/put chains with configurable smiles, spreads,
  seeded noise, and optional diagnostic faults.
- Put-call parity, quote, bound, duplicate-contract, staleness, and solver diagnostics.
- Deterministic enriched-CSV and Markdown quality-report exports.
- Forward log-moneyness and total implied variance as canonical surface coordinates.
- Per-expiry raw-SVI calibration constrained for positive variance, Lee wing slopes, and sampled
  non-negative risk-neutral density.
- Joint power-law SSVI calibration with monotone ATM variance and Gatheral-Jacquier sufficient
  conditions.
- Butterfly diagnostics through the density factor `g(k)` and adjacent-expiry calendar diagnostics
  through total-variance spreads.
- Structured optimizer, error, parameter, and failure reports with deterministic CSV/JSON exports.
- A mandatory downstream eligibility gate that blocks invalid surfaces from later Heston work.
- Heston European call/put pricing with a stable characteristic function and vectorized COS engine.
- Bounded multi-start Heston calibration to gated surface targets in normalized forward
  coordinates, using vega-scaled price residuals.
- Feller-condition reporting plus an optional parameterization that guarantees compliance.
- Structured Heston optimizer, price-error, implied-volatility-error, and parameter diagnostics.
- Seeded Andersen QE-M Heston Monte Carlo with antithetic sampling, a discounted-underlying
  control variate, standard error, confidence interval, and martingale diagnostics.
- A non-uniform-grid two-dimensional Heston PDE solver with central finite differences,
  Crank-Nicolson stepping, Rannacher damping, and boundary/grid diagnostics.
- Cross-checks of both numerical engines against COS through package, CLI, and Streamlit APIs.
- Automatic three-level refinement, per-engine error estimates, Monte Carlo interval checks,
  cross-engine consensus, deterministic audits, and a hard downstream numerical-convergence gate.
- CRR American-option pricing with early-exercise premium and exercise-boundary diagnostics.
- Seeded Heston Longstaff-Schwartz valuation on full QE-M spot/variance paths with a multifactor
  regression basis and conditional sampling interval.
- A Heston linear-complementarity solver using Ikonen-Toivanen splitting, Crank-Nicolson stepping,
  Rannacher damping, and free-boundary diagnostics.
- Independent constant-volatility and stochastic-volatility American-option cross-checks.
- Five-point COS/PDE Heston delta, gamma, parallel-level vega, calendar theta, and rho with bump
  stability diagnostics.
- Seeded pathwise delta/rho, conditional likelihood-ratio gamma/delta, and paired
  common-random-number vega/theta with antithetic confidence intervals.
- Fixed-grid cubic PDE interpolation for stable bump-and-revalue gamma estimates.
- Seeded common-path comparisons of unhedged, delta, delta-gamma, delta-vega, and
  delta-gamma-vega Heston strategies.
- A self-financing hedge ledger with cash interest, stock dividends, stock/option transaction
  costs, position caps, terminal liquidation, and hedge-system conditioning diagnostics.
- Exact pathwise P&L attribution to delta, gamma, instantaneous vega, theta, financing,
  transaction costs, and residual effects.
- Explicit separation of risk-neutral pricing parameters from physical or stressed Heston path
  parameters and spot drift.
- Multi-seed pooling with joint path/rebalance refinement and sampling-aware stability limits.
- Matched-Q, physical-drift, parameter-misspecification, stressed-cost, and combined scenarios.
- A hard hedge-robustness gate covering worst-scenario risk reduction, seed dispersion, clipping,
  conditioning fallbacks, and attribution reconciliation.
- Exact-timestamp option-chain partitioning and chronological calibration without future quotes.
- First-snapshot-only target selection and fixed, bracketing listed-call hedge identities.
- Daily SVI/SSVI and Heston gates that stop replay before an invalid surface reaches hedge sizing.
- Option buys at ask and sells at bid, midpoint position marks, separate spread and proportional
  cost attribution, self-financing cash/dividends, rebalancing, and final liquidation.
- Canonical snapshot SHA-256 values and chained audit hashes covering model, trade, and P&L state.
- Deterministic multi-date SSVI fixtures plus `generate-replay` and `replay` CLI workflows.
- Vendor-neutral historical quote/trade events with separate event, receipt, and ingestion times.
- Append-only SQLite records and ingestion batches protected against update and deletion.
- Revision-aware corrections and cancellations under independent market-knowledge and
  dataset-version cutoffs.
- Explicit ingestion modes that reject silent backfills and require reasons for corrections and
  historical backfill batches.
- Content-addressed raw CSV archives, file and record SHA-256 values, and chained batch manifests.
- Replay-ready snapshots that select the latest contract quote received by each formation time and
  enforce a maximum quote age.
- Store verification and five `history-*` CLI workflows for ingestion, queries, snapshots, and
  integrity audits.
- A separate append-only product, contract, session, and corporate-action reference store with
  publication-time and ingestion-time knowledge cutoffs.
- Stable contract identities across effective-dated source-symbol and deliverable changes.
- Exact exchange-session intervals with IANA timezones, holidays, early closes, and named session
  segments represented explicitly rather than inferred from weekdays.
- Product-level exercise, settlement, expiration, and last-trade rules plus exact per-contract
  timestamps and structured deliverables.
- A hard historical replay gate that rejects unresolved or ambiguous contracts, mismatched terms,
  future reference publications, closed sessions, after-hours events, and expired listings.
- Reference raw-file/record hashes, chained batch manifests, integrity verification, and four
  `reference-*` CLI workflows.
- Rolling windows of independently fitted daily BSM volatilities and arbitrage-gated Heston
  parameters, with every forecast window ending strictly before its evaluation snapshot.
- Next-snapshot row-level BSM/Heston prices, midpoint RMSE/MAE/bias, vega-scaled errors, bid/ask
  coverage, and zero-inside-spread quote-band loss.
- Model-loss summaries by evaluation date and maturity/forward-moneyness bucket.
- Prediction-panel SHA-256 values linked into a chronological benchmark audit chain.
- Paired, evaluation-date-clustered circular block-bootstrap intervals that preserve cross-sectional
  grouping and local serial dependence rather than treating option rows as independent.
- HLN-corrected Diebold–Mariano statistics with configurable Bartlett-HAC lags plus exact
  evaluation-date sign tests, reported separately from the primary block-bootstrap conclusion.
- Deterministic research ZIPs with per-artifact hashes, a manifest hash, a bundle-root hash, and
  bounded in-memory verification.
- Detached Ed25519 attestations that bind exact bundle bytes and lineage hashes, with separate
  bundle, signature, binding, signer-trust, and acceptance decisions.
- Encrypted PKCS8 key generation plus CLI signing and mandatory trusted-public-key verification;
  self-asserted embedded keys are never silently promoted to trusted identities.
- Multiplier-scaled replay marks, fills, spreads, fees, turnover, and hedge sensitivities for
  standard single-security deliverables; unsupported adjusted deliverables block before calibration.
- Deterministic parsing and canonical SHA-256 identities for standard, adjusted physical, and
  cash-settled deliverables, with bounded strict JSON and duplicate rejection.
- Terminal package valuation for multiple explicitly marked securities, fixed cash, resolved
  cash-in-lieu, and official cash-settlement levels, without inferred prices or FX conversions.
- Separate gross receivable/payable flows, aggregate exercise price, net payoff, and component
  audit rows that keep the contract multiplier distinct from adjusted quantities.
- A `deliverable-value` CLI workflow and interactive deliverable-valuation workspace.
- Constant correlated risk-neutral GBM pricing for European options on explicit multi-security,
  fixed-cash, cash-in-lieu, and cash-settled deliverables.
- Exact risk-factor coverage and positive-semidefinite correlation validation, with bounded strict
  market JSON and canonical SHA-256 identities.
- Seeded antithetic basket Monte Carlo, a discounted-package control variate, confidence intervals,
  crude-versus-adjusted sampling diagnostics, and pathwise component deltas.
- Discounted price bounds and a separately labeled first-two-moment lognormal approximation,
  including one-asset Black-Scholes-Merton equivalence checks.
- A `basket-price` CLI workflow and interactive basket-pricing workspace whose static results stay
  outside the single-factor Heston replay ledger.
- Point-in-time adjusted-option delta-hedge replay with separate market, receipt, and formation
  timestamps, staleness checks, and fixed risk-factor identities.
- Opening option execution at bid, component buys at ask and sells at bid, continuous cash and
  dividend carry, terminal settlement, component liquidation, and exact self-financing checks.
- Separate option/component cash flows, spreads, explicit costs, turnover, marked P&L, and
  component-level trade records.
- Configuration and hash-chained snapshot audits that bind contract terms, settings, canonical
  markets, executable quotes, pricing diagnostics, and ledger state.
- Deterministic `generate-basket-replay` and `basket-replay` CLI workflows plus an interactive
  basket hedge replay workspace.
- Append-only component-market and pairwise-correlation records with independent market, receipt,
  and ingestion timestamps, revision/cancellation history, and complete-package resolution.
- Content-addressed raw archives, canonical record hashes, chained ingestion manifests, SQLite
  append-only triggers, and whole-store integrity verification.
- Canonical CSV plus bounded HTTPS/JSON source adapters with verified TLS, environment-only bearer
  credentials, response limits, and retained non-secret adapter metadata.
- Store-to-replay materialization that binds resolved snapshot and contributing manifest hashes
  into the basket ledger, with an optional hard provenance requirement.
- Six `basket-input-*` CLI workflows and an interactive basket-input storage workspace.
- Rolling synchronized log-return volatility and sample-correlation estimation using only records
  known by formation, with explicit identity shrinkage, PSD diagnostics, and estimate self-hashes.
- Three explicitly quoted basket-option hedges for aggregate relative-spot gamma, parallel vega,
  and correlation-strength sensitivity, plus component positions for residual delta.
- Scaled solver conditioning, option-position and exposure-residual limits, adverse-side fills,
  hedge liquidation, and hash-chained self-financing accounting.
- Statistical-input generation/materialization and advanced risk-replay CLI workflows plus an
  interactive basket risk-hedging workspace.
- Sample-correlation shrinkage, EWMA, and diagonal covariance-shrinkage estimators sharing the
  same point-in-time estimate contract.
- Strictly chronological one-step held-out covariance forecasts scored by Gaussian and covariance-
  consistent multivariate Student-t negative log-likelihood, with raw eigenvalue, stabilized
  conditioning, and explicit promotion diagnostics.
- Training-window-only robust return-energy thresholds, common calm/stress fold labels, weighted
  regime-balanced selection scores, minimum regime coverage, and stress-loss regret gates.
- A conservative estimator gate: unstable candidates are excluded and an alternative must meet a
  configured improvement threshold or the stable v1.7 baseline is retained.
- Canonical covariance-selection self-hashes binding exact folds and store provenance into replay,
  with optional mandatory enforcement by the advanced hedge ledger.
- Covariance-selection CLI workflows and an interactive validation workspace with auditable CSV
  and JSON downloads.
- Optional five-factor basket hedging across common spot-scale gamma, parallel vega, dynamic
  forward-moneyness skew and curvature scenarios, and correlation strength.
- Exact five-instrument solves with singular-value rank coverage, conditioning, position,
  residual, moneyness, provenance, and hash-chain gates.
- A deterministic smile-input generator, CLI controls, and interactive smile-hedging workspace
  with sensitivity, solve, execution, and audit exports.
- A separate append-only basket-option quote store for explicit target and hedge contracts, with
  three timestamp axes, revisions/cancellations, immutable contract identities, content-addressed
  raw artifacts, chained manifests, and point-in-time staleness enforcement.
- Self-hashed option-quote snapshots binding the target contract and bid/ask plus every ordered
  hedge definition and bid/ask into advanced replay, with optional hard enforcement.
- Four `basket-option-*` CLI workflows, a deterministic quote-history generator, and an integrated
  option-quote storage workspace.
- A `benchmark` CLI workflow and interactive model-benchmark workspace.
- Twenty-one Streamlit workspaces and a CLI backed by the same package APIs.
- Numerical, ingestion, CLI, and headless Streamlit tests.

## Quick start

Requirements: [`uv`](https://docs.astral.sh/uv/); offline workflows require no external
credentials. `uv` installs the project's pinned CPython 3.12 interpreter automatically.

```bash
# From the cloned repository
cd volforge
uv sync --locked --dev
uv run streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`. Everything works offline after dependencies are
installed.

Run the quality gates:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv build
```

GitHub Actions enforces at least 85% package coverage, repeats the tests on Linux, Windows, and
macOS, then installs the built wheel in a clean environment and exercises the CLI entry point.

## Streamlit workspaces

1. **Pricing calculator** — select BSM or Bachelier, inspect price/Greek cards, bounds, and the
   value/payoff profile.
2. **Implied-volatility solver** — invert a price with explicit convergence and failure output.
3. **Option-chain explorer** — upload CSV or generate a seeded smile chain, filter expiration
   smiles, inspect invalid rows, and download the enriched chain.
4. **Data-quality diagnostics** — review aggregate failures, per-row reasons, parity residuals,
   and an exportable Markdown report.
5. **Volatility surfaces** — fit constrained SVI slices or joint SSVI, inspect total variance,
   butterfly density factors and calendar spreads, and export the evaluated surface and audit.
6. **Heston calibration** — pass a validated surface through the mandatory gate, fit the five
   Heston parameters with COS pricing, inspect residuals and Feller diagnostics, and export the fit.
7. **Numerical engines** — cross-check COS against seeded QE Monte Carlo and finite-difference PDE
   prices, including sampling uncertainty, discretization controls, and run diagnostics.
8. **American options** — compare Heston Longstaff-Schwartz and obstacle-PDE values, inspect
   early-exercise premiums and boundaries, and retain a CRR constant-volatility benchmark.
9. **Heston Greeks** — compare deterministic COS/PDE bumps with pathwise, likelihood-ratio, and
   common-random-number Monte Carlo estimates and their uncertainty diagnostics.

10. **Hedging laboratory** - compare five self-financing Heston hedge policies on common seeded
    paths, inspect costs and conditioning, and reconcile terminal P&L by risk factor.
11. **Hedge robustness** - run the multi-seed refinement and five-scenario acceptance gate,
    inspect eligible strategies, and download a deterministic audit.
12. **Point-in-time replay** - replay exact-timestamp chains, recalibrate through the surface gate,
    execute fixed listed contracts at bid/ask sides, mark at midpoint, and download the
    hash-chained P&L audit.
13. **Deliverable valuation** - value explicit adjusted physical packages or cash-settled payoffs,
    inspect component cash flows, and download a canonical settlement audit.
14. **Basket pricing** - price an explicit adjusted package under correlated GBM, inspect sampling
    error, bounds, an analytical approximation, and component deltas, and download its audit.
15. **Basket hedge replay** - replay received-by-formation component markets, trade delta hedges at
    explicit bid/ask quotes, settle at expiry, reconcile P&L, and verify the audit chain.
16. **Basket input store** - validate append-only component/correlation packages, resolve them at
    formation cutoffs, bind their manifest identities into replay rows, and inspect the integrated
    ledger.
17. **Basket option quote store** - retain target and hedge-option quotes, resolve every requested
    instrument as known at formation, and bind the verified snapshot into the five-risk replay.
18. **Basket covariance validation** - compare covariance estimators on strictly later held-out
    returns and bind an eligible selection into replay.
19. **Basket risk hedging** - estimate point-in-time volatility/correlation, solve explicit
    gamma/vega/correlation option positions, neutralize residual component delta, and verify the
    self-financing audit.
20. **Basket smile hedging** - solve five explicit option positions for gamma, vega, local smile
    skew/curvature, and correlation risk with full-rank coverage and audit diagnostics.
21. **Model benchmark** - fit rolling lagged BSM/Heston parameters, score untouched next snapshots,
    inspect maturity/moneyness and quote-band losses, and download predictions plus their audit.

Session-state keys are workspace-specific. Pricing formulas and validations live in the package,
not in the Streamlit pages.

## CLI

Price a BSM call:

```bash
uv run volforge price --model bsm --option-type call --spot 100 --strike 100 \
  --maturity 1 --volatility 0.20 --rate 0.05
```

Solve a Bachelier implied normal volatility:

```bash
uv run volforge iv --model bachelier --option-type put --spot 98 --strike 105 \
  --maturity 0.4 --price 9.45 --rate 0.02 --dividend-yield 0.01
```

Generate a chain and then validate it:

```bash
uv run volforge generate data/synthetic.csv --seed 7 --base-volatility 0.22
uv run volforge validate data/synthetic.csv reports/enriched.csv \
  --report reports/quality.md
```

Calibrate an SSVI surface and write an auditable result:

```bash
uv run volforge surface data/synthetic.csv reports/ssvi-surface.csv \
  --model ssvi --report reports/ssvi-audit.json
```

Price with Heston and calibrate it from the gated SSVI surface:

```bash
uv run volforge heston-price --option-type call --spot 100 --strike 100 --maturity 1 \
  --kappa 1.5768 --theta 0.0398 --vol-of-vol 0.5751 --rho -0.5711 \
  --initial-variance 0.0175
uv run volforge heston-calibrate data/synthetic.csv reports/heston-fit.csv \
  --surface-model ssvi --report reports/heston-audit.json
```

Cross-check the same Heston contract with Monte Carlo and a finite-difference PDE:

```bash
uv run volforge heston-mc --option-type call --spot 100 --strike 100 --maturity 1 \
  --kappa 1.5768 --theta 0.0398 --vol-of-vol 0.5751 --rho -0.5711 \
  --initial-variance 0.0175 --paths 100000 --steps 126 --seed 7
uv run volforge heston-pde --option-type call --spot 100 --strike 100 --maturity 1 \
  --kappa 1.5768 --theta 0.0398 --vol-of-vol 0.5751 --rho -0.5711 \
  --initial-variance 0.0175 --log-price-points 121 --variance-points 61 --time-steps 180
```

Enforce the automatic convergence gate and retain its audit:

```bash
uv run volforge heston-converge --option-type call --spot 100 --strike 100 --maturity 1 \
  --kappa 1.5768 --theta 0.0398 --vol-of-vol 0.5751 --rho -0.5711 \
  --initial-variance 0.0175 --report reports/heston-convergence.json
```

The command exits with status `0` only when every engine refinement, Monte Carlo confidence check,
price bound, and cross-engine consensus check passes. A blocked audit is still written when
`--report` is supplied.

Cross-check an American put with Heston simulation and the obstacle PDE:

```bash
uv run volforge american-lsm --option-type put --spot 100 --strike 100 --maturity 1 \
  --rate 0.03 --kappa 1.5768 --theta 0.0398 --vol-of-vol 0.5751 --rho -0.5711 \
  --initial-variance 0.0175 --paths 20000 --steps 64 --seed 7
uv run volforge american-pde --option-type put --spot 100 --strike 100 --maturity 1 \
  --rate 0.03 --kappa 1.5768 --theta 0.0398 --vol-of-vol 0.5751 --rho -0.5711 \
  --initial-variance 0.0175 --log-price-points 121 --variance-points 61 --time-steps 180
```

Estimate Heston Greeks with deterministic COS bumps or direct Monte Carlo estimators:

```bash
uv run volforge heston-greeks --engine cos --option-type call --spot 100 --strike 100 \
  --maturity 1 --rate 0.03 --kappa 1.5768 --theta 0.0398 --vol-of-vol 0.5751 \
  --rho -0.5711 --initial-variance 0.0175
uv run volforge heston-greeks --engine mc --option-type call --spot 100 --strike 100 \
  --maturity 1 --rate 0.03 --kappa 1.5768 --theta 0.0398 --vol-of-vol 0.5751 \
  --rho -0.5711 --initial-variance 0.0175 --paths 20000 --mc-steps 64 --seed 7
```

Compare self-financing Heston hedge strategies and retain an audit:

```bash
uv run volforge heston-hedge --option-type call --spot 100 --strike 100 --maturity 1 \
  --rate 0.03 --kappa 1.5 --theta 0.04 --vol-of-vol 0.4 --rho -0.7 \
  --initial-variance 0.04 --paths 100 --rebalancing-steps 12 \
  --stock-cost-bps 1 --option-cost-bps 5 --report reports/hedging.json
```

Run the hedge robustness gate across seeds, budgets, and stressed dynamics:

```bash
uv run volforge heston-robustness --option-type call --spot 100 --strike 100 --maturity 1 \
  --rate 0.03 --kappa 1.5 --theta 0.04 --vol-of-vol 0.4 --rho -0.7 \
  --initial-variance 0.04 --seeds 7 17 --path-counts 10 20 40 \
  --rebalancing-steps 4 6 8 --report reports/hedge-robustness.json
```

The command exits with status `0` when at least one strategy passes every criterion and all
scenario attributions reconcile; a blocked audit exits with status `2`.

Generate a deterministic multi-date fixture and run the chronological replay:

```bash
uv run volforge generate-replay data/replay.csv
uv run volforge replay data/replay.csv reports/replay-ledger.csv \
  --surface-model ssvi --strategy delta_gamma_vega \
  --report reports/replay-audit.json
```

The replay exits with status `0` only if every snapshot has valid contemporaneous quotes, passes
the surface and Heston gates, retains all selected listed quotes, reconciles its self-financing
ledger, and completes final liquidation. A blocked prefix audit exits with status `2`.

Value an adjusted physical deliverable from explicit component marks:

```bash
uv run volforge deliverable-value data/deliverable.json \
  --marks data/security-marks.json --settlement-method physical \
  --option-type call --strike 60 --contract-multiplier 100 \
  --currency USD --underlying-id ACME
```

The command emits a deterministic settlement audit. See
[`docs/adjusted-deliverables.md`](docs/adjusted-deliverables.md) for the component schema,
cash-settlement form, and valuation limits.

Price the same adjusted package before expiry from explicit component markets and correlation:

```bash
uv run volforge basket-price data/deliverable.json data/basket-market.json \
  --settlement-method physical --option-type call --strike 60 \
  --contract-multiplier 100 --currency USD --underlying-id ACME \
  --maturity 0.5 --rate 0.03 --paths 100000 --seed 19
```

The command emits the seeded Monte Carlo estimate and interval, crude/control-variate diagnostics,
price bounds, a moment-matched approximation, component deltas, and canonical input hashes. See
[`docs/basket-pricing.md`](docs/basket-pricing.md) for assumptions and the replay boundary.

Generate and replay a deterministic adjusted-option hedge fixture:

```bash
uv run volforge generate-basket-replay \
  data/basket-replay.csv data/basket-deliverable.json
uv run volforge basket-replay \
  data/basket-replay.csv data/basket-deliverable.json reports/basket-ledger.csv \
  --settlement-method physical --option-type call --strike 60 \
  --contract-multiplier 100 --currency USD --underlying-id ACME \
  --expiry 2026-04-30T20:00:00+00:00 --paths 20000 --seed 31 \
  --report reports/basket-replay.json
```

The replay exits with status `0` only when every received-by-formation market package, price bound,
execution record, settlement amount, accounting identity, and audit-chain link passes. A blocked
prefix exits with status `2`. See [`docs/basket-replay.md`](docs/basket-replay.md) for the CSV,
self-financing, and provenance conventions.

Build an append-only source boundary and materialize the same replay from normalized records:

```bash
uv run volforge generate-basket-inputs \
  data/basket-inputs.csv data/basket-schedule.csv
uv run volforge basket-input-init data/basket-inputs.sqlite
uv run volforge basket-input-ingest data/basket-inputs.sqlite data/basket-inputs.csv \
  --source volforge-synthetic-v1 --mode initial \
  --ingested-at 2026-05-01T00:00:00Z
uv run volforge basket-input-materialize \
  data/basket-inputs.sqlite data/basket-schedule.csv reports/basket-replay-input.csv \
  --basket-id VOLFORGE-SYNTHETIC-ADJUSTED --source volforge-synthetic-v1
uv run volforge basket-input-verify data/basket-inputs.sqlite \
  --report reports/basket-input-integrity.json
```

Pass `--require-store-provenance` to `basket-replay` to reject unbound CSV inputs. The optional
`basket-input-fetch` command accepts providers that already publish the canonical bounded
HTTPS/JSON contract and reads bearer credentials from a named environment variable. See
[`docs/basket-input-store.md`](docs/basket-input-store.md) for schemas, time axes, revision rules,
adapter controls, and trust limits.

Build the independent executable option-market boundary before materializing the advanced schedule:

```bash
uv run volforge generate-basket-option-inputs \
  data/basket-option-quotes.csv data/basket-option-schedule.csv
uv run volforge basket-option-init data/basket-option-quotes.sqlite
uv run volforge basket-option-ingest \
  data/basket-option-quotes.sqlite data/basket-option-quotes.csv \
  --source licensed-option-source --mode initial \
  --ingested-at 2026-05-01T00:00:00Z
uv run volforge basket-option-materialize \
  data/basket-option-quotes.sqlite data/basket-option-schedule.csv \
  reports/quote-bound-schedule.csv \
  --basket-id VOLFORGE-SYNTHETIC-ADJUSTED \
  --source licensed-option-source \
  --target-instrument-id TARGET-CALL-202604 \
  --hedge-instrument-id LOW-WING-PUT-202606 \
  --hedge-instrument-id VEGA-CALL-202607 \
  --hedge-instrument-id CORR-CALL-202609 \
  --hedge-instrument-id DEEP-LOW-PUT-202610 \
  --hedge-instrument-id HIGH-WING-CALL-202612
uv run volforge basket-option-verify data/basket-option-quotes.sqlite \
  --report reports/basket-option-integrity.json
```

Feed the quote-bound schedule to `basket-input-estimate` or
`basket-input-select-covariance`; both preserve its quote provenance. Pass
`--require-option-quote-provenance` to `basket-risk-replay` to reject missing, partial, tampered,
or contract-mismatched quote lineage. See
[`docs/basket-option-quote-store.md`](docs/basket-option-quote-store.md).

Estimate point-in-time basket inputs and run the explicit nonlinear-risk hedge:

```bash
uv run volforge generate-basket-statistical-inputs \
  data/statistical-basket-inputs.csv data/basket-risk-schedule.csv
uv run volforge basket-input-init data/statistical-basket.sqlite
uv run volforge basket-input-ingest \
  data/statistical-basket.sqlite data/statistical-basket-inputs.csv \
  --source volforge-synthetic-statistical-v1 --mode initial \
  --ingested-at 2026-05-01T00:00:00Z
uv run volforge basket-input-estimate \
  data/statistical-basket.sqlite data/basket-risk-schedule.csv \
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

The advanced replay requires verified estimate provenance by default and exits `2` on an unstable
solver, excessive option position, residual exposure, accounting mismatch, or broken audit link.
See [`docs/basket-risk-hedging.md`](docs/basket-risk-hedging.md) for the estimator, bump-risk,
solver, execution, and limitation conventions.

Compare alternative covariance estimators only on later held-out historical returns, then bind the
selected method into the replay input:

```bash
uv run volforge basket-covariance-select data/statistical-basket.sqlite \
  --basket-id VOLFORGE-SYNTHETIC-ADJUSTED \
  --formation-at 2026-04-01T20:00:00+00:00 \
  --source volforge-synthetic-statistical-v1 \
  --forecast-distribution student_t \
  --selection-objective regime_balanced \
  --report reports/covariance-selection.json
uv run volforge basket-input-select-covariance \
  data/statistical-basket.sqlite data/basket-risk-schedule.csv \
  reports/covariance-selected-replay.csv \
  --basket-id VOLFORGE-SYNTHETIC-ADJUSTED \
  --source volforge-synthetic-statistical-v1 \
  --forecast-distribution student_t \
  --selection-objective regime_balanced
```

The robust policy also exposes Student-t degrees of freedom, stress quantile and weight, minimum
stress-fold coverage, and maximum stress-loss regret. Pass `--require-covariance-selection` to
`basket-risk-replay` to reject missing, tampered, ineligible, or estimate-mismatched selections. See
[`docs/basket-covariance-validation.md`](docs/basket-covariance-validation.md) for the fold design,
loss, stability and promotion rules, provenance contract, and limitations.

Generate five quoted hedge options and run the optional local smile-risk mode:

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

This scenario uses each option's dynamic forward log-moneyness; it is not a calibrated basket
smile. See [`docs/basket-smile-hedging.md`](docs/basket-smile-hedging.md) for definitions, rank
gates, audit fields, and limitations.

Run the rolling next-snapshot model comparison:

```bash
uv run volforge benchmark data/replay.csv reports/model-predictions.csv \
  --training-window 2 --surface-model ssvi \
  --comparison-loss squared_error --comparison-hac-lags 1 \
  --report reports/model-benchmark.json \
  --bundle reports/model-benchmark.zip
```

The benchmark exits with status `0` only when every training surface/Heston fit passes, every
forecast window ends before its evaluation timestamp, all row-level prices are finite, and the
snapshot/prediction audit chain verifies. See
[`docs/model-benchmark.md`](docs/model-benchmark.md) for loss, bucket, and uncertainty conventions,
and [`docs/research-reports.md`](docs/research-reports.md) for bundle verification.

Generate an encrypted signing key, sign the completed bundle, and verify it against a public key
obtained through a trusted channel:

```bash
uv run volforge research-keygen keys/research-private.pem keys/research-public.pem \
  --password-file keys/private-key.password
uv run volforge research-sign reports/model-benchmark.zip keys/research-private.pem \
  reports/model-benchmark.attestation.json \
  --password-file keys/private-key.password \
  --issued-at 2026-10-16T19:00:00Z --signer "VolForge research"
uv run volforge research-verify reports/model-benchmark.zip \
  reports/model-benchmark.attestation.json \
  --trusted-public-key keys/research-public.pem
```

See [`docs/signed-provenance.md`](docs/signed-provenance.md) before managing real signing keys.

Create, ingest, query, snapshot, and verify the local historical store:

```bash
uv run volforge history-init data/options-history.sqlite
uv run volforge history-ingest data/options-history.sqlite data/vendor-quotes.csv \
  --source vendor-name --mode initial --ingested-at 2026-10-08T14:00:00Z \
  --report reports/initial-ingestion.json
uv run volforge history-query data/options-history.sqlite reports/known-quotes.csv \
  --record-type quote --received-at 2025-01-02T16:00:00Z --source vendor-name
uv run volforge history-snapshot data/options-history.sqlite reports/replay-input.csv \
  --snapshot-at 2025-01-02T15:30:00Z 2025-01-03T15:30:00Z \
  --source vendor-name --maximum-quote-age 15min \
  --reference-store data/options-reference.sqlite
uv run volforge history-verify data/options-history.sqlite \
  --report reports/history-integrity.json
```

`event_timestamp` controls market ordering, `received_timestamp` controls what the market-data
source had delivered by a formation time, and batch `ingested_timestamp` controls which VolForge
dataset version is visible. Incremental ingestion cannot add events older than the source
watermark; those rows require an explicit, explained backfill batch.

Create and operate the independent reference master before building a multi-snapshot historical
replay:

```bash
uv run volforge reference-init data/options-reference.sqlite
uv run volforge reference-ingest data/options-reference.sqlite data/products.csv \
  --kind product --source reference-provider --mode initial \
  --ingested-at 2026-10-13T14:00:00Z
uv run volforge reference-query data/options-reference.sqlite reports/products.csv \
  --kind product --effective-at 2026-10-13T14:30:00Z \
  --published-at 2026-10-13T14:30:00Z
uv run volforge reference-verify data/options-reference.sqlite \
  --report reports/reference-integrity.json
```

Products, contracts, sessions, and corporate actions are separate batch kinds. Reference
publication cutoffs are applied at every formation snapshot; reference data published later cannot
be substituted into an earlier replay.

CLI price and solve commands emit JSON. A failed implied-volatility solve or a surface that fails
the static-arbitrage gate exits with status 2 and reports the reason; neither is labeled
successful.

## Numerical conventions

- Inputs are scalar and explicitly validated. Spot and strike must be positive; maturity and
  volatility must be non-negative; every numeric value must be finite.
- Rates and dividend yields are continuously compounded annual decimals.
- Maturity is in ACT/365F years.
- BSM volatility is an annualized lognormal decimal (`0.20` means 20%).
- Bachelier volatility is an annualized absolute standard deviation in underlying price units per
  square-root year (`20.0` means 20 price units per square-root year). It is not 20%.
- Vega and rho are returned per absolute `1.00` change in volatility or rate. Divide by 100 for a
  one-percentage-point sensitivity; the UI does this on its cards.
- Theta is calendar-time decay per year, `-dV/dT`. The UI also shows theta per day as theta/365.
- At expiry, price is intrinsic value. Delta uses a symmetric 0.5/-0.5 convention at the strike;
  higher derivatives and carry sensitivities are reported as zero because the payoff kink is not
  classically differentiable.
- At zero volatility, value is the discounted deterministic forward payoff. At the exercise
  boundary, the same symmetric indicator convention is used and gamma is reported as zero.
- Prices within scaled solver tolerance of the lower bound map explicitly to zero volatility.
  Prices at the upper bound require unbounded volatility and return a failure.
- Surface calibration uses `k = log(K/F)` and total implied variance `w(k,T) = sigma_BS^2 T`.
  Calls and puts for the same strike and expiry are consolidated by median implied volatility.
- Surface interpolation is limited to the calibrated maturity range. Diagnostics are evaluated on
  an explicit log-moneyness grid and therefore do not prove global arbitrage freedom outside that
  domain.
- Heston parameters are risk-neutral. `theta` and `initial_variance` are variances, while
  `vol_of_vol` is the diffusion coefficient of variance. The Feller condition is reported by
  default and can be required during calibration.
- COS calls are recovered from bounded put prices with put-call parity. Finite-range results are
  checked against European price bounds; materially invalid numerical results raise an error.
- QE Monte Carlo is reproducible for identical seeds and settings. Its confidence interval covers
  sampling uncertainty only; time-step bias must be checked separately.
- The PDE solves a bounded problem. Log-price, variance, and time grids plus both domain limits
  must be refined before treating a price as converged.
- The automatic gate evaluates three increasing budgets for every engine. Its tolerance is
  `absolute_tolerance + relative_tolerance * max(abs(COS reference), 1)`; passing the finite ladder
  is evidence at the configured budget, not a mathematical error bound.
- American LSM intervals describe conditional sampling error, not regression or exercise-grid
  bias. American PDE and CRR results require joint space/time refinement.
- Heston vega applies a parallel shift to `sqrt(v0)` and `sqrt(theta)`. Monte Carlo Greek intervals
  capture sampling variation only; bump and time-step bias require separate refinement.
- Hedging vega instead differentiates with respect to current `sqrt(v_t)` while holding `theta`
  fixed. Hedge P&L is per short target option and reconciles delta, gamma, vega, theta, financing,
  transaction cost, and residual components path by path.
- Robustness scenarios retain risk-neutral parameters for pricing and Greeks while independently
  changing the realized spot drift, variance dynamics, or trading costs.
- Replay contract identities are chosen from the first snapshot only. Daily Heston Greeks size the
  hedge, while observed option midpoints determine all trades and marks. Every snapshot must carry
  one exact quote timestamp, and any missing later target or hedge mark blocks the run.
- Snapshot hashes bind normalized source rows; chained audit hashes bind the chronological model,
  position, cash, cost, and P&L records. They are tamper-evident hashes, not digital signatures or
  third-party timestamp attestations.
- Historical queries apply receipt and ingestion cutoffs before selecting the highest active
  revision. Raw source files, normalized records, and ingestion manifests carry separate hashes;
  none is a substitute for a vendor signature or trusted external timestamp.

The detailed equations and finite-difference verification conventions are in
[`docs/mathematics.md`](docs/mathematics.md). The calibration workflow, audit fields, gate, and
interpretation limits are in
[`docs/volatility-surfaces.md`](docs/volatility-surfaces.md). Heston and COS implementation details
are in [`docs/heston-cos.md`](docs/heston-cos.md). Monte Carlo, PDE, diagnostics, and convergence
guidance are in [`docs/numerical-engines.md`](docs/numerical-engines.md).
American exercise methods, diagnostics, and limitations are in
[`docs/american-options.md`](docs/american-options.md).
Greek formulas, conventions, and validation guidance are in
[`docs/heston-greeks.md`](docs/heston-greeks.md).
Self-financing conventions, hedge ratios, cost accounting, attribution, and interpretation limits
are in [`docs/hedging.md`](docs/hedging.md).
Multi-seed pooling, refinement tolerances, scenario definitions, and strategy acceptance are in
[`docs/hedge-robustness.md`](docs/hedge-robustness.md).
Point-in-time assumptions, the listed-contract policy, accounting identity, hash chain, and gate
semantics are in [`docs/point-in-time-replay.md`](docs/point-in-time-replay.md).
Historical schemas, time axes, ingestion modes, knowledge queries, raw archives, and verification
are in [`docs/historical-data.md`](docs/historical-data.md).
Contract/product schemas, effective-time rules, session normalization, corporate-action linkage,
and the replay gate are in [`docs/reference-data.md`](docs/reference-data.md).

## CSV schema

The required header is:

```text
quote_timestamp,expiration,strike,option_type,bid,ask,underlying_spot,risk_free_rate,dividend_yield
```

| Field | Meaning |
|---|---|
| `quote_timestamp` | ISO 8601 quote time; parsed as UTC |
| `expiration` | ISO 8601 expiry date/time; parsed as UTC |
| `strike` | Positive strike in underlying price units |
| `option_type` | `call` or `put` (case-insensitive on input) |
| `bid`, `ask` | Non-negative prices with `bid <= ask` |
| `underlying_spot` | Positive contemporaneous spot |
| `risk_free_rate` | Continuous annual decimal rate |
| `dividend_yield` | Continuous annual decimal yield |

The loader calculates midpoint and ACT/365F maturity, flags stale/negative/crossed/duplicate/bound
failures, and calculates implied volatility only for rows that pass every validation check. Missing
required columns raise a schema error. Malformed rows remain in the enriched output with semicolon-
separated `validation_reasons`. Staleness is measured from an explicit valuation time or,
deterministically, from the newest valid quote in the file.

See [`docs/csv-schema.md`](docs/csv-schema.md) for all derived columns and an example.

## Architecture

```text
src/volforge/
├── models/              # BSM, Bachelier, bounds, types, input validation
├── data/                # CSV validation/enrichment and synthetic chains
├── surfaces/            # SVI, SSVI, static-arbitrage checks, reports, gate
├── calibration/         # Gated Heston fitting, residuals, deterministic audits
├── numerics/            # Seeded Heston Monte Carlo and finite-difference PDE
├── american/             # CRR, Heston LSM, and obstacle-PDE exercise engines
├── greeks/               # Deterministic and Monte Carlo Heston sensitivities
├── hedging/              # Self-financing strategies, costs, diagnostics, attribution
├── robustness/           # Multi-seed refinement, misspecification scenarios, hedge gate
├── replay/               # Point-in-time calibration, listed hedges, P&L and hash-chain audit
├── benchmark/            # Rolling BSM/Heston forecasts, loss panels, prediction audits
├── research/             # Content-addressed benchmark bundles and integrity verification
├── history/              # Append-only events, revisions, provenance, point-in-time snapshots
├── reference/            # Contract/product master, sessions, actions, replay reference gate
├── basket_inputs/        # Append-only component/correlation store and source adapters
├── basket_option_inputs/ # Append-only target/hedge-option quote history and snapshots
├── deliverables.py      # Canonical terms and explicit terminal settlement accounting
├── implied_vol.py       # Model-agnostic bracket/Newton orchestration
├── diagnostics.py       # Put-call parity and quality aggregates
├── reporting.py         # Deterministic CSV and Markdown exports
├── basket.py            # Correlated-GBM adjusted-package pricing and diagnostics
├── basket_covariance.py # Chronological covariance comparison and selection audit
├── basket_estimation.py # Point-in-time rolling volatility/correlation estimates
├── basket_replay.py     # Point-in-time component hedges, settlement, and audit chain
├── basket_risk_hedging.py # Explicit option-risk hedges and self-financing ledger
└── cli.py               # Command-line adapter

streamlit_app.py         # Navigation only
app_pages/               # Streamlit presentation workspaces
tests/                   # Numerical, data, CLI, and AppTest coverage
docs/                    # Mathematics, schema, and architecture decisions
```

Dependencies point inward: Streamlit and CLI layers call package APIs; numerical modules do not
import Streamlit. More detail is in [`docs/architecture.md`](docs/architecture.md).

## Current limitations

- Vanilla European and American calls and puts are priced; barriers and other path-dependent
  contracts are not implemented.
- BSM assumes constant lognormal volatility; Bachelier assumes constant normal volatility. Heston
  uses one time-homogeneous risk-neutral parameter set over the calibrated surface.
- Monte Carlo confidence intervals omit parameter uncertainty, and the automatic gate remains
  limited by its maximum simulation/PDE budgets and bounded domains.
- LSM is an exercise-policy estimate rather than a rigorous bound. Heston Greek intervals omit
  model, parameter, time-step, and bump uncertainty.
- Hedge robustness can use separate physical or stressed dynamics, but those parameters are
  user-defined rather than statistically estimated.
- Replay supports chronological recalibration and fixed listed contracts, but bundled data remain
  synthetic. The history store is vendor-neutral and has no built-in downloader; licensed data
  must be supplied by the user.
- Reference data are user-supplied; VolForge does not bundle licensed OCC, OPRA, vendor, or venue
  files and does not automatically generate future listings from natural-language product rules.
- Explicit settlement, standalone correlated-GBM pricing, and the dedicated basket replay support
  multi-security packages, fixed cash, resolved cash-in-lieu, and cash-settled payoffs. Supplied
  CSV provenance labels are not vendor authentication. The v1.6 store preserves raw source and
  revision evidence, but does not establish upstream accuracy, entitlement, or trusted timestamps;
  the single-factor Heston replay still blocks adjusted contracts.
- The statistical basket estimator assumes a caller-selected observation clock and annualization
  factor. Shrinkage improves conditioning but does not remove sampling or regime uncertainty.
- Covariance selection uses one-return likelihood scores over a small rolling holdout. Optional
  Student-t scoring is a heavy-tail sensitivity rather than an estimated return law, while rolling
  calm/stress labels are not a latent regime model or regime forecast. Longer independent histories
  and sensitivity analysis remain necessary.
- The advanced basket hedge neutralizes three declared local bump risks by default. Its optional
  five-option mode adds forward-moneyness skew and curvature scenarios, but does not calibrate or
  evolve a basket smile and does not remove jumps, basis, liquidity, parameter, or model risk.
  The v1.11 store makes independently supplied quote lineage enforceable, but the bundled
  synthetic hedge quotes remain model-derived demonstrations rather than executable evidence.
- Replay trades options at bid/ask sides, marks at midpoint, and applies standard contract
  multipliers. It does not model market impact, fill probability, latency, margin, or stock borrow.
- Raw-SVI cross-expiry safety is diagnosed on a finite grid; SSVI also enforces analytic sufficient
  conditions at the calibrated ATM-variance nodes. Neither claim extends beyond the calibrated
  maturity horizon.
- Static checks use midpoint prices and do not establish trade executability.
- Reference-gated historical replay uses explicit UTC session intervals, IANA timezones, closures,
  and early closes; ungated synthetic or uploaded CSV studies still trust their supplied timestamps.
- Scalar APIs prioritize transparent diagnostics over large-scale vectorized throughput.
- Synthetic quotes are model demonstrations, not calibrated market data.

## Roadmap

- **v0.1 — complete:** pricing, Greeks, implied volatility, ingestion, synthetic chains, and
  diagnostics.
- **v0.2 — complete:** constrained SVI/SSVI surfaces, total-variance diagnostics, and the
  downstream eligibility gate.
- **v0.3 — complete:** gated Heston calibration, Fourier-cosine pricing, Feller diagnostics, and
  fit audits.
- **v0.4 — complete:** deterministic QE-M Monte Carlo, variance reduction, finite-difference PDE
  pricing, and COS cross-checks.
- **v0.4.1 — complete:** automatic numerical refinement, consensus diagnostics, audit output, and
  a hard downstream convergence gate.
- **v0.5 — complete:** CRR American benchmark, Heston Longstaff-Schwartz, obstacle PDE, exercise
  diagnostics, CLI commands, and an interactive cross-check workspace.
- **v0.5.1 — complete:** five-point COS/PDE Greeks, pathwise and conditional likelihood-ratio
  estimators, common-random-number bumps, uncertainty diagnostics, CLI, and workspace.
- **v0.5.2 — deferred research seam:** automatic-differentiation and adjoint Greek experiments.
- **v0.6 — complete:** common-path delta/gamma/vega hedging, self-financing costs, stability
  diagnostics, exact P&L attribution, CLI audits, and an interactive laboratory.
- **v0.7 — complete:** multi-seed hedge robustness, joint simulation/rebalance refinement, model
  and cost misspecification stresses, physical dynamics, and a hard strategy gate.
- **v0.8 — complete:** point-in-time chain snapshots, chronological calibration replay, fixed
  listed-contract hedge selection, self-financing accounting, and hash-chained daily P&L audits.
- **v0.9 — complete:** append-only historical quote/trade storage, three timestamp axes,
  revision-aware queries, explicit backfills, provenance hashes, and replay-ready snapshots.
- **v0.9.1 — complete:** effective-dated contract/product master, explicit exchange sessions,
  corporate actions, product settlement rules, and a fail-closed historical replay gate.
- **v0.10 — complete:** rolling BSM-versus-Heston benchmark panels, quote-band losses,
  prediction-aware audit chains, and executable bid/ask-side option accounting.
- **v1.0 — complete:** evaluation-date-clustered block-bootstrap comparison uncertainty,
  content-addressed research-report manifests, and standard-deliverable multiplier economics with
  fail-closed adjusted-contract handling.
- **v1.1 — complete:** detached Ed25519 bundle attestations, encrypted key generation, and explicit
  trusted-public-key verification.
- **v1.2 — complete:** HLN-corrected Diebold–Mariano/HAC forecast-accuracy diagnostics, exact
  date-level directional tests, CLI controls, audited exports, and interactive benchmark reporting.
- **v1.3 — complete:** canonical adjusted-deliverable terms, explicit physical-package and
  cash-settlement accounting, CLI/audit output, reference integration, and an interactive
  valuation workspace while preserving the single-factor replay gate.
- **v1.4 — complete:** explicit correlated-GBM basket inputs, seeded antithetic/control-variate
  pricing, sampling intervals, price bounds, a moment-matched cross-check, component deltas,
  CLI/audit output, and an interactive workspace while preserving the chronological replay gate.
- **v1.5 — complete:** received-by-formation basket snapshots, executable component delta hedges,
  carry, expiry settlement, self-financing P&L, configuration-bound audit chaining, deterministic
  CLI fixtures/workflows, and an interactive replay workspace.
- **v1.6 — complete:** append-only point-in-time component/correlation storage, corrections and
  backfills, raw and manifest integrity, bounded CSV/HTTPS adapters, provenance-bound replay
  materialization, CLI workflows, and an interactive storage workspace.
- **v1.7 — complete:** no-lookahead rolling volatility/correlation estimation with explicit
  shrinkage and provenance, three quoted gamma/vega/correlation basket-option hedges, residual
  component delta neutralization, solver/accounting gates, CLI workflows, and a workspace.
- **v1.8 — complete:** chronological held-out comparison of sample-shrinkage, EWMA, and diagonal-
  shrinkage covariance estimators, PSD/conditioning and improvement gates, provenance-bound
  selection, CLI enforcement, and an interactive validation workspace.
- **v1.9 — complete:** dynamic forward-moneyness skew and curvature scenarios, five-instrument
  full-rank hedge coverage, singular-value and residual gates, hash-bound audit output, CLI
  controls, deterministic fixtures, and an interactive workspace.
- **v1.10 — complete:** covariance-consistent Student-t forecast scoring, chronological robust
  calm/stress labels, regime-balanced objectives, stress coverage and regret gates, provenance-
  bound CLI output, and interactive diagnostics.
- **v1.11 — complete:** independent append-only target and hedge-option quote history, immutable
  contract identities, as-known multi-instrument resolution, quote-snapshot self-hashes,
  downstream replay enforcement, CLI workflows, and an integrated storage workspace.

The next research seam is latent-state or transition-aware covariance and basket-smile dynamics,
plus concrete provider adapters for the new canonical option-quote contract. A source label and
retained CSV are not vendor authentication, entitlement proof, or a trusted timestamp.

## Governance

- VolForge is maintained by [`@gunhoo1124-arch`](https://github.com/gunhoo1124-arch) and the
  original project code and documentation are released under the MIT License.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) defines development and numerical-review expectations.
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) defines collaboration and enforcement expectations.
- [`SECURITY.md`](SECURITY.md) explains private vulnerability reporting and data handling.
- [`CHANGELOG.md`](CHANGELOG.md) tracks released behavior starting at v0.1.0.
- [`CITATION.cff`](CITATION.cff) provides machine-readable citation metadata.
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) records dependency and attribution boundaries.
- The project is licensed under the [`MIT License`](LICENSE).
