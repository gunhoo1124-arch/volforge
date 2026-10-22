# v1.11 architecture

## Design goals

VolForge optimizes for numerical transparency, deterministic behavior, explicit failure states,
and adapters that share one domain implementation. v0.8 retains the mandatory surface, numerical,
and robustness gates, chronological point-in-time recalibration, and a hash-chained observed-mark
ledger. v0.9 adds an append-only provenance boundary before replay so revisions, cancellations,
and backfills remain observable rather than rewriting history. v0.9.1 adds a second append-only
boundary for effective-dated contracts, product rules, sessions, and corporate actions. v0.10 adds
a rolling out-of-sample model-comparison boundary and executable bid/ask-side option fills. v1.0
adds dependence-aware inference, content-addressed report bundles, and multiplier-aware replay.
v1.1 adds detached signatures and an explicit trusted-key acceptance boundary around those bundles.
v1.2 adds complementary forecast-accuracy and directional diagnostics without weakening the
date-clustered block-bootstrap conclusion gate.
v1.3 adds a terminal settlement-accounting boundary for adjusted physical and cash-settled
contracts without treating it as a multi-asset pre-expiry Heston model.
v1.4 adds a distinct constant correlated-GBM basket-pricing boundary with sampling diagnostics,
an analytical approximation, component deltas, and canonical input hashes. Its outputs do not
cross the chronological Heston replay gate.
v1.5 adds a separate point-in-time basket hedge ledger. It consumes complete received-by-formation
market packages, executes component deltas at bid/ask, settles the explicit deliverable at expiry,
and binds contract configuration plus every self-financing record into an audit chain.
v1.6 adds an append-only basket-input boundary before that ledger. Canonical CSV or bounded HTTPS
adapters feed revisioned component/correlation records; verified as-known resolution then binds
the selected snapshot and contributing manifest identities into each replay record.
v1.7 adds a statistical-estimation boundary over complete as-known histories, then a separate
multi-instrument hedge ledger. Three explicit basket options span aggregate gamma, vega, and
correlation risk subject to solver gates; components neutralize the remaining deltas. Estimate,
quote, solve, trade, and accounting identities are committed into the audit chain.
v1.8 inserts a chronological covariance-selection boundary between as-known history and the
estimated replay. Three estimators score only later held-out past returns; PSD, conditioning, and
minimum-improvement rules control promotion, and the selected method plus its complete fold audit
is committed into each advanced replay snapshot.
v1.9 adds an optional five-factor hedge boundary. Dynamic forward log-moneyness defines local
skew and curvature scenarios beside gamma, parallel vega, and correlation strength; an explicit
singular-value rank gate proves that five quoted instruments span all five declared risks before
the existing solve, position, residual, execution, accounting, and audit gates can pass.
v1.10 extends the chronological covariance boundary with covariance-consistent Student-t scoring
and candidate-independent calm/stress labels fitted only on each earlier training window. A
regime-balanced objective requires minimum stress coverage and prevents an alternative from being
promoted when its stress loss exceeds the baseline by more than the declared regret limit.
v1.11 adds a second basket-market boundary dedicated to target and hedge-option quotes. Its
append-only records preserve complete contract terms and executable sides independently of the
component/correlation store; as-known snapshots are self-hashed and must bind the exact advanced
replay row when quote provenance is required.

```text
Historical CSV ──► history.store ───────┐
Reference CSV  ──► reference.store ─────┼──► reference-gated snapshots
                                        │
                                        ▼
CSV / synthetic config              scalar contract inputs
          │                                   │
          ▼                                   ▼
 data.chain / data.synthetic       models.black_scholes / models.bachelier
          │                                   │
          ├──────────────► implied_vol ◄──────┘
          │
          ├────► diagnostics / reporting
          │
          ▼
 surface quote preparation ──► SVI / SSVI calibration
                                      │
                                      ▼
                        butterfly / calendar diagnostics
                                      │
                                      ▼
                           downstream eligibility gate
                                      │
                                      ▼
                         Heston calibration / COS pricing
                                      │
                                      ▼
                         Monte Carlo / finite-difference PDE
                                      │
                                      ▼
                      Greeks / self-financing hedge backtests
                                      │
                                      ▼
                   Multi-seed / misspecification robustness gate
                                      │
                                      ▼
                Point-in-time chain replay / listed-contract P&L audit

All presentation-neutral outputs flow to the CLI and Streamlit pages.

```text
Canonical CSV / HTTPS JSON
            │
            ▼
  basket_inputs.adapters
            │ raw bytes + normalized records
            ▼
    basket_inputs.store ──► as-known complete snapshot ──► basket_replay audit chain
       │        │
       │        └── received-time and dataset-version cutoffs
       └── content-addressed archives + chained manifests
```

```text
Canonical option-quote CSV
            │ retained bytes + immutable contracts
            ▼
 basket_option_inputs.store ──► as-known target + ordered hedge snapshot
       │        │                            │
       │        └── receipt/ingestion cutoffs│ self-hash + manifest roots
       └── content-addressed archive         ▼
                                  estimate/covariance materialization
                                               │
                                               ▼
                                  advanced replay quote-lineage gate
```
```

## Package boundaries

- `models.common` owns payoff types, result dataclasses, validation, bounds, and limiting-case
  conventions.
- `models.black_scholes` and `models.bachelier` own scalar prices and analytic Greeks.
- `implied_vol` owns price-bound validation, bracket expansion, safeguarded Newton, Brent fallback,
  residual verification, and structured outcomes.
- `data.chain` owns the CSV contract and row-preserving validation/enrichment pipeline.
- `data.synthetic` owns all pseudo-random generation; one NumPy generator is initialized from the
  explicit seed.
- `history.store` owns the normalized historical event schema, raw archives, append-only SQLite
  records, revision/backfill policies, received/ingested knowledge queries, manifest chaining,
  integrity verification, and replay-ready snapshots.
- `reference.store` owns effective-dated product and contract terms, source-symbol mappings,
  explicit market sessions, corporate-action linkage, reference publication cutoffs, append-only
  provenance, and the fail-closed historical replay enrichment gate.
- `basket_inputs.adapters` owns bounded canonical CSV/HTTPS acquisition, verified TLS defaults,
  environment-only bearer credentials, duplicate-JSON rejection, and non-secret source metadata.
- `basket_inputs.store` owns append-only component/correlation records, correction/backfill rules,
  three timestamp axes, complete point-in-time matrix resolution, raw/payload/manifest integrity,
  and provenance-bound replay materialization.
- `basket_option_inputs.store` owns the separate append-only target/hedge-option quote contract,
  immutable listed-option definitions, as-known multi-instrument resolution, quote snapshot
  self-hashes, raw/payload/manifest integrity, and schedule materialization.
- `basket_estimation` owns synchronized as-known spot panels, rolling log-return volatility,
  sample-correlation shrinkage, EWMA and diagonal-shrinkage estimates, PSD diagnostics, estimate
  self-hashes, and estimated replay materialization.
- `basket_covariance` owns strictly chronological forecast folds, Gaussian and Student-t NLL,
  robust training-window regime classification, balanced calm/stress objectives, eigenvalue,
  condition, coverage and stress-regret gates, conservative promotion, canonical selection
  hashes, and selection-bound replay materialization.
- `diagnostics` owns cross-row parity pairing and aggregate quality measures.
- `reporting` owns byte-stable CSV formatting and deterministic Markdown reports.
- `surfaces.common` owns quote normalization into `(k, w)`, minimum data requirements, and the
  downstream eligibility gate.
- `surfaces.svi` owns raw-SVI functions, derivatives, constrained slice fitting, total-variance
  interpolation, and structured surface results.
- `surfaces.ssvi` owns the power-law SSVI surface, ATM-variance nodes, analytic sufficient
  constraints, and joint fitting.
- `surfaces.arbitrage` owns model-independent density-factor and calendar-spread diagnostics.
- `surfaces.reporting` owns deterministic surface audit JSON and parameter tables.
- `models.heston` owns Heston parameters, Feller diagnostics, characteristic functions, corrected
  cumulants, and vectorized Fourier-cosine European pricing.
- `calibration.heston` owns the gated normalized-forward calibration target, vega-scaled objective,
  optimizer orchestration, final implied-volatility inversion, and deterministic fit reports.
- `numerics.monte_carlo` owns seeded Andersen QE-M path generation, variance reduction, sampling
  uncertainty, and martingale diagnostics.
- `numerics.pde` owns the bounded two-dimensional Heston grid, finite-difference operator,
  Crank-Nicolson/Rannacher stepping, interpolation, and grid diagnostics.
- `numerics.convergence` owns automatic refinement budgets, error estimates, cross-engine
  consensus, deterministic audits, and the downstream numerical eligibility gate.
- `american.binomial` owns the constant-volatility CRR benchmark and discrete boundary.
- `american.monte_carlo` owns Longstaff-Schwartz backward induction and regression diagnostics.
- `american.pde` owns the Heston complementarity problem, Ikonen-Toivanen splitting, and
  free-boundary diagnostics.
- `greeks.finite_difference` owns deterministic COS/PDE stencils and bump diagnostics.
- `greeks.monte_carlo` owns pathwise, conditional likelihood-ratio, and paired common-random-number
  estimators with sampling uncertainty.
- `hedging.backtest` owns common-path hedge construction, the self-financing ledger, costs,
  conditioning safeguards, terminal distributions, and exact P&L attribution.
- `robustness.hedging` owns seed pooling, joint refinement, physical/misspecified path scenarios,
  cost stress, strategy acceptance, and deterministic gate audits.
- `replay.engine` owns exact-timestamp partitioning, first-snapshot contract selection, daily
  surface/Heston gating, bid/ask-side execution, midpoint marking, separate spread/fee accounting,
  standard-deliverable multiplier economics, and SHA-256 audit chaining.
- `deliverables` owns strict component parsing, canonical terms and hashes, explicit security-mark
  requirements, physical exercise exchanges, and cash-settlement payoff accounting.
- `basket` owns validated component markets and positive-semidefinite correlation, exact terminal
  correlated-GBM sampling, antithetic and package-control estimators, moment-matched cross-checks,
  price bounds, component deltas, and deterministic pricing audits.
- `basket_replay` owns the chronological basket CSV contract, three-timestamp knowledge gate,
  explicit component execution, delta rebalancing, cash/dividend carry, terminal settlement,
  accounting reconciliation, deterministic fixtures, and configuration-bound audit chaining.
- `basket_risk_hedging` consumes explicit basket-option definitions and quotes, validates optional
  store snapshot lineage, and owns common-random-
  number bump risks, dynamic forward-moneyness smile scenarios, three- or five-factor scaled
  solves, rank/condition/position/residual gates, residual component-delta hedges, hedge
  liquidation, and its self-financing audit chain.
- `benchmark.engine` owns rolling lagged BSM/Heston fits, next-snapshot pricing, quote-band and
  midpoint losses, maturity/moneyness panels, date-clustered block-bootstrap inference,
  HLN-corrected DM/HAC and exact date-sign diagnostics, and prediction-aware SHA-256 audit chaining.
- `research.report` owns deterministic report packaging, per-artifact hashes, manifest/root hashes,
  and in-memory verification without archive extraction.
- `research.attestation` owns Ed25519 key serialization, detached canonical statements, exact bundle
  binding, and the separation of cryptographic validity from trusted-key acceptance.
- `cli` and `app_pages` translate user inputs into the same APIs. Neither reimplements pricing.

The numerical and data layers have no Streamlit dependency. The application is split into a small
navigation entry point and twenty-one direct page scripts, following Streamlit conventions.

## Failure model

Programmer/input contract violations in scalar pricing raise `ValueError`. An implied-volatility
search returns `ImpliedVolResult` because non-convergence and bound violations are expected research
outcomes. A missing CSV column raises `ChainSchemaError`; row-level problems are retained as flags
and reasons. Surface input insufficiency raises `SurfaceQuoteError`; a completed calibration stores
optimizer and diagnostic failures in its result. `require_downstream_eligible` raises
`SurfaceValidationError` if a caller attempts to pass an invalid surface onward. This distinction
prevents both silent coercion and exception-driven bulk ingestion. Heston calibration invokes the
gate internally before generating targets. Optimizer convergence, final implied-volatility
inversion, Feller policy, fit errors, and failure reasons remain separate fields in the result.
Numerical-engine setting violations raise `ValueError`; undefined martingale corrections,
non-finite solves, and material price-bound failures raise `ArithmeticError`.
`require_numerical_convergence` raises `NumericalConvergenceError` when refinement, statistical,
bounds, or consensus checks block a price from downstream use.
The hedge robustness gate returns a report rather than raising for an ordinary blocked strategy;
invalid scenario construction or settings still raise `ValueError`.
The replay engine likewise returns a completed or blocked chronological prefix. It stops at the
first invalid quote day, failed surface/Heston gate, missing fixed contract, hedge-sizing failure,
or accounting mismatch; no later snapshot is consulted to repair that state.
The benchmark engine also returns a completed or blocked chronological prefix. The current
snapshot is scored before it can become training data, and a failed quote/surface/Heston gate
stops the run rather than skipping forward to a more convenient window.
Forecast inference is explicitly unavailable below its configured date threshold. A degenerate
HAC variance or too few non-tied date signs disables only its corresponding sensitivity diagnostic;
neither condition is converted into artificial statistical confidence.
Historical schema violations raise `HistoricalSchemaError`. Attempts to overwrite logical records,
silently backfill an incremental stream, reuse revisions, or break ingestion ordering raise
`HistoricalIntegrityError`. Corrections and backfills are ordinary append operations only when
their explicit modes, revisions, and reasons satisfy the store invariants.
Reference schema violations raise `ReferenceSchemaError`. Missing or ambiguous contract mappings,
future-only terms, mismatched event attributes, inactive contracts, absent corporate actions,
closed markets, and timestamps outside explicit sessions raise `ReferenceIntegrityError` before a
historical replay chain can reach calibration.
Deliverable schema and valuation violations raise `ValueError`. Missing or extra security marks,
mixed settlement forms, unsupported component types, inconsistent currencies, and absent official
settlement levels fail closed before any payoff is reported.
Basket-market violations also raise `ValueError`. Missing or unused risk factors, invalid spots or
volatilities, non-symmetric/non-PSD correlation, incompatible terms, and invalid simulation
settings fail before sampling. A completed estimate retains its statistical error and bounds gate;
it is never silently promoted to chronological replay eligibility.
Basket replay table-shape, timestamp, and ordering violations raise `BasketReplaySchemaError`.
Snapshot-level look-ahead, staleness, provenance-identity, market, execution, pricing, settlement,
or accounting failures return a blocked chronological prefix and never consult later rows.
Basket-input schema violations raise `BasketInputSchemaError`. Attempts to overwrite records,
silently backfill an incremental stream, weaken revision identity, move receipt time backward, or
resolve an incomplete/ambiguous/stale/non-PSD package raise `BasketInputIntegrityError`. Store
verification is mandatory before materialization. Replay can additionally require bound store
snapshot and manifest hashes rather than accepting an unbound CSV package.
Basket estimation rejects insufficient synchronized history, ambiguous complete packages,
undefined sample correlations, look-ahead, and non-PSD shrunk matrices with
`BasketInputIntegrityError`. The estimate's self-hash binds its settings, window, observations,
market, execution package, and store provenance.
Covariance selection also raises `BasketInputIntegrityError` for insufficient chronological
training/evaluation history, a failed store, or no stable candidate. An alternative is never
promoted unless it passes every PSD/condition fold and its mean held-out NLL improvement reaches
the configured threshold under the compatibility policy; otherwise the stable sample-shrinkage
baseline remains selected. Robust
mode additionally blocks insufficient calm/stress coverage and rejects an alternative promotion
whose stress NLL regret exceeds its declared limit.
Advanced hedge table-shape errors raise `BasketRiskHedgeSchemaError`. Snapshot-level estimate
or covariance-selection tampering, selection/estimate mismatches, instrument drift, invalid
quotes, incomplete declared-risk rank, excessive smile moneyness, ill-conditioned or excessive
solves, residual exposure, settlement, accounting, or audit failures return a blocked
chronological prefix.

## Reproducibility

- Python is constrained to 3.12 and dependencies are locked in `uv.lock`.
- Synthetic generation uses fixed defaults and an explicit NumPy seed.
- Monte Carlo creates a fresh NumPy generator from its explicit run seed and returns that seed in
  every result. The same contract, parameters, settings, and seed reproduce the same estimate.
- Basket pricing canonicalizes contract and market JSON independently and returns both SHA-256
  identities with its explicit seed and estimator configuration.
- Basket replay uses a base seed plus snapshot sequence, canonicalizes every market/execution
  package, and binds complete contract/settings metadata into every linked snapshot hash.
- Basket-input artifacts are archived by content hash. Normalized record payloads and batch
  manifests are canonical, corrections append revisions, and resolved snapshot hashes commit to
  every contributing payload and manifest identity.
- Basket estimates hash their exact rolling window, settings, input records/manifests, market, and
  execution package. Advanced bump valuations use one snapshot seed for common random numbers;
  every instrument risk, solve, position, quote, and trade is bound into the linked audit.
- Covariance-selection audits hash every training/evaluation timestamp, candidate score,
  eigenvalue, condition, promotion decision, and contributing store identity. Accepted selection
  hashes enter the advanced replay chain beside the corresponding estimate hash.
- Robust covariance audits additionally hash both forecast distributions, Student-t degrees of
  freedom, training-fitted energy thresholds, realized energies, common regime labels, balanced
  scores, coverage, stress regret, and every robust gate outcome.
- Smile-mode audits additionally hash the ordered five-factor contract, dynamic target/instrument
  moneyness, skew and curvature sensitivities, numerical rank and minimum singular value, solves,
  positions, and residuals.
- Staleness defaults to the newest valid quote in a file, not wall-clock time.
- CSV exports use LF line endings, omit the index, and use a fixed float representation.
- Historical CSV bytes are archived by content hash. Each normalized event carries a payload hash,
  and each ingestion manifest commits to the prior manifest hash.
- Reference inputs use the same independent raw-file, record-payload, and chained-manifest controls;
  reference publication cutoffs are evaluated separately from local ingestion cutoffs.
- Tests exercise benchmark values, identities, finite differences, round trips, malformed rows,
  deterministic generation, constrained surface calibration, arbitrage counterexamples, CLI
  behavior, Heston benchmark prices and calibration, Monte Carlo determinism and variance
  reduction, PDE refinement and parity, passing and blocked convergence audits, and in-process
  Streamlit pages, plus hedging determinism, costs, conditioning, pathwise reconciliation,
  physical drift, multi-seed refinement, passing/blocked robustness audits, chronological replay,
  missing listed marks, accounting reconciliation, audit tampering, quote/trade ingestion,
  received/ingested revision visibility, cancellation, append-only enforcement, explicit
  backfills, raw-archive tampering, historical CLI round trips, stable contract resolution,
  holiday/after-hours rejection, future corporate-action isolation, and reference CLI workflows.
  Rolling-benchmark tests additionally cover no-look-ahead windows, model-loss panels, invalid
  training snapshots, prediction tampering, date-clustered inference, DM/HAC and date-sign edge
  cases, deterministic/tampered report
  bundles, multiplier-scaled accounting, canonical adjusted-deliverable valuation and rejection,
  correlated-GBM basket pricing, one-asset closed-form equivalence, PSD rejection, variance
  reduction, point-in-time basket timing failures, component execution, cash settlement,
  accounting reconciliation, metadata/snapshot tampering, basket-input timing/revision/package
  and adapter controls, raw-store tampering, statistical no-look-ahead/shrinkage/PSD checks,
  nonlinear exposure neutralization, solver limits, chronological covariance folds, estimator
  promotion/fallback, Gaussian/Student-t scores, common regime labels, stress coverage and regret
  fallback, selection/estimate tampering, five-factor smile neutralization, rank and moneyness
  rejection, smile-audit tampering, CLI output, and the benchmark, deliverable, basket-pricing,
  basket-replay, basket-input, basket-option-input, covariance-validation, basket-risk, and smile
  workspaces. Option-quote tests cover revisions, cutoffs, cancellation, staleness, immutable
  contracts, append-only enforcement, archive tampering, self-hashes, and replay-payload tampering.

## Post-v1.11 extension seam

Future work can add trusted-timestamp or transparency-log integration, stationary-bootstrap
variants, and multiple-comparison model-confidence sets. Basket research can add transition or
latent-state covariance models and calibrated smile dynamics. Concrete vendor adapters may add provider
symbol/pagination/entitlement rules behind the canonical transport contract. None should weaken
the existing storage, estimation, settlement, surface, numerical, reference, Heston replay,
basket replay, benchmark, bundle-integrity, or trusted-key gates.
