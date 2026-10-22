# Changelog

All notable changes to VolForge are documented here. The format follows Keep a Changelog and the
project uses semantic versioning.

## [1.11.0] - 2026-10-21

### Added

- A separate append-only SQLite store for target and hedge-option quote histories, with independent
  market, receipt, and ingestion clocks, immutable contract identities, revisions/cancellations,
  content-addressed raw CSVs, record hashes, chained manifests, and whole-store verification.
- Point-in-time multi-instrument resolution and self-hashed quote snapshots binding the target
  contract/market, ordered hedge contracts/markets, selected records, batches, and manifest roots.
- Optional advanced-replay enforcement and audit fields that reject missing, partial, tampered,
  stale, or contract-mismatched option-quote lineage.
- Deterministic option-quote fixtures, generation/init/ingest/materialization/verification CLI
  workflows, a twenty-first Streamlit workspace, and adversarial package/CLI/UI tests.
- Cross-platform Linux, Windows, and macOS CI with an enforced coverage floor, locked dependency
  installation, clean-wheel smoke testing, and retained build artifacts.
- CodeQL analysis, Dependabot configuration, issue forms, a pull-request checklist, conduct and
  citation metadata, and a tag-driven GitHub release workflow with SHA-256 checksums.
- Explicit maintainer copyright, contribution-rights terms, third-party notices, dependency-license
  review, and repository URLs for the public GitHub release.
- Patched dependency floors for `cryptography`, `certifi`, Pillow, and Streamlit after a pre-release
  vulnerability audit.

### Changed

- Package version advanced to 1.11.0.
- Statistical-estimate and covariance-selection materializers now preserve option-quote provenance
  columns for downstream replay validation.

## [1.10.0] - 2026-10-20

### Added

- Covariance-consistent multivariate Student-t negative log-likelihood beside the existing
  Gaussian score, with explicit degrees of freedom and both losses retained in every fold audit.
- Candidate-independent calm/stress labels from robust return energy fitted only on each fold's
  earlier training window, plus a weighted regime-balanced selection objective.
- Minimum stress-fold coverage and maximum stress-loss-regret gates that fail closed or retain the
  stable sample-shrinkage baseline instead of promoting a fragile alternative.
- CLI controls and expanded covariance workspace diagnostics for scoring law, regime thresholds,
  balanced scores, coverage, stress resilience, and provenance-bound replay materialization.
- Regression coverage for Student-t scoring, common chronological labels, conservative fallback,
  insufficient regime coverage, CLI round trips, and headless UI interaction.

### Changed

- Package version advanced to 1.10.0.
- Gaussian mean-NLL selection remains the package default for backward compatibility; the
  interactive covariance workspace now presents the robust policy first.

## [1.9.0] - 2026-10-20

### Added

- Optional five-factor adjusted-basket hedging across common spot-scale gamma, parallel vega,
  forward-moneyness smile skew, smile curvature, and correlation strength.
- Dynamic per-instrument forward log-moneyness, centered common-random-number smile scenarios, and
  an explicit bound that rejects excessive smile extrapolation.
- Full-rank coverage enforcement using normalized singular values before the existing condition,
  position, and residual gates; all coordinates, risks, ranks, solves, and residuals enter the
  snapshot hash chain.
- A deterministic five-instrument fixture, `generate-basket-smile-inputs` workflow, smile-replay
  CLI controls, and a twentieth Streamlit workspace with downloadable audit artifacts.
- Regression coverage for deterministic neutralization, instrument count, rank, moneyness,
  tampering, CLI round trips, and headless UI interaction.

### Changed

- Package version advanced to 1.9.0.
- The three-factor v1.8 hedge contract remains the default; smile coverage is opt-in and never
  represented as a calibrated basket surface or proof of executable liquidity.

## [1.8.0] - 2026-10-19

### Added

- Sample-correlation shrinkage, exponentially weighted covariance, and diagonal covariance-
  shrinkage estimators behind the existing point-in-time basket estimate contract.
- Strictly chronological one-return forecast folds scored by Gaussian negative log-likelihood,
  with explicit PSD/eigenvalue, conditioning, and minimum-improvement promotion gates.
- Canonical covariance-selection audits binding exact training/evaluation times, candidate losses,
  diagnostics, store records/manifests, selected estimator, and a SHA-256 self-hash.
- `basket-covariance-select` and `basket-input-select-covariance` CLI workflows plus a nineteenth
  Streamlit workspace for comparing candidates and downloading provenance-bound replay inputs.
- Optional advanced-replay enforcement that rejects missing, tampered, ineligible, or estimate-
  mismatched covariance-selection provenance.

### Changed

- Package version advanced to 1.8.0.
- The v1.7 fixed sample-shrinkage estimator remains the default-compatible path; alternatives are
  promoted only when their held-out improvement meets the configured threshold.

## [1.7.0] - 2026-10-19

### Added

- Point-in-time rolling log-return volatility and sample-correlation estimation from complete,
  synchronized append-only basket histories, with explicit identity shrinkage and PSD diagnostics.
- Canonical estimate audits that bind observation windows, source record/manifests, the resolved
  executable snapshot, estimated market, and a SHA-256 self-hash into replay inputs.
- Three explicit quoted basket-option hedge instruments spanning aggregate relative-spot gamma,
  parallel vega, and correlation-strength sensitivity, followed by residual component-delta hedges.
- A scaled 3x3 solver with conditioning, position, and residual gates; adverse-side execution;
  target settlement; hedge liquidation; cash/dividend carry; and self-financing reconciliation.
- `generate-basket-statistical-inputs`, `basket-input-estimate`, and `basket-risk-replay` CLI
  workflows, an eighteenth Streamlit workspace, deterministic fixtures, and regression coverage.

### Changed

- Package version advanced to 1.7.0.
- The v1.6 delta-only replay can now require estimate provenance while remaining backward
  compatible by default.

## [1.6.0] - 2026-10-19

### Added

- A dedicated append-only SQLite basket-input store for canonical component markets, executable
  quotes, and complete pairwise correlation packages.
- Market, receipt, and ingestion time axes; revision/cancellation rules; explicit correction and
  backfill modes; source watermarks; and fail-closed as-known snapshot resolution.
- Content-addressed CSV/JSON archives, record payload hashes, chained batch manifests, database
  append-only triggers, and whole-store integrity verification.
- Canonical local-CSV and bounded HTTPS/JSON adapters. HTTPS requires verified TLS, excludes URL
  credentials/query strings, bounds time and response size, rejects duplicate JSON fields, and
  obtains optional bearer credentials only from a named environment variable.
- Store-to-replay materialization that binds snapshot and contributing manifest hashes into the
  existing basket replay audit chain, plus a `require_store_provenance` policy.
- Six `basket-input-*` CLI workflows, a deterministic normalized fixture, and a seventeenth
  Streamlit workspace for the verified storage-to-replay path.
- Point-in-time, revision, incomplete-package, PSD, append-only, raw-tamper, adapter, CLI, audit,
  and headless Streamlit regression coverage.

### Changed

- Package version advanced to 1.6.0.
- Basket replay audits now retain optional store snapshot and manifest identities; unbound v1.5
  CSV workflows remain available unless the stronger provenance policy is requested.

## [1.5.0] - 2026-10-17

### Added

- A point-in-time adjusted-option hedge replay with separate market, receipt, and formation times,
  maximum-age enforcement, stable risk-factor identities, and structured fail-closed prefixes.
- Explicit component bid/ask execution, opening option sale at bid, delta rebalancing, continuous
  cash/dividend carry, expiry settlement, liquidation, spread loss, costs, turnover, and exact
  self-financing reconciliation.
- Configuration, market, execution, snapshot, and chained audit hashes that bind contract terms,
  replay settings, pricing diagnostics, component trades, and cash state.
- A deterministic four-snapshot multi-asset fixture plus `generate-basket-replay` and
  `basket-replay` CLI workflows.
- A sixteenth Streamlit workspace with P&L visualization, snapshot/component ledgers, explicit
  provenance warning, bounded caching, and downloadable CSV/JSON audits.
- Unit, timing-boundary, cash-settlement, cost, tampering, CLI, and headless Streamlit coverage.

### Changed

- Package version advanced to 1.5.0.
- Adjusted and cash-settled packages now have a dedicated chronological delta-hedge ledger; they
  remain excluded from the single-factor SVI/SSVI-Heston calibration replay.
- Audit verification now rejects result-level replay-setting or contract-metadata tampering, not
  only modifications to individual snapshot records.

## [1.4.0] - 2026-10-16

### Added

- Pre-expiry European pricing for explicit physical and cash-settled deliverables under constant
  correlated risk-neutral GBM, with strict risk-factor coverage and positive-semidefinite
  correlation validation.
- Seeded antithetic Monte Carlo with a discounted-package control variate, crude and adjusted
  standard errors, confidence intervals, and variance-reduction diagnostics.
- Discounted price bounds, a first-two-moment lognormal approximation, pathwise component deltas,
  and exact deterministic handling at expiry, for fixed-cash packages, and at zero volatility.
- Canonical basket-market JSON and SHA-256 audit identity, a `basket-price` CLI workflow, and a
  fifteenth Streamlit workspace with downloadable audit output.
- Unit, CLI, closed-form equivalence, invalid-correlation, deterministic, variance-reduction, and
  headless Streamlit regression coverage.

### Changed

- Package version advanced to 1.4.0.
- Adjusted and cash-settled options can now be analyzed before expiry under explicit standalone
  GBM assumptions, while chronological Heston replay continues to fail closed before calibration.
- Documentation now distinguishes Monte Carlo sampling uncertainty, the moment-matched
  approximation, and the missing point-in-time inputs required for replay integration.

## [1.3.0] - 2026-10-16

### Added

- A deterministic adjusted-deliverable settlement engine with canonical component ordering,
  duplicate-field/component rejection, bounded JSON inputs, and SHA-256 component identities.
- Explicit physical-package accounting for multiple securities, fixed cash, and resolved
  cash-in-lieu amounts, with a required unit mark for every security component.
- Explicit cash-settlement accounting from the official settlement level, strike, option type,
  and contract multiplier.
- Component-level gross values, exercise receivable/payable flows, net long-option payoff, and a
  JSON-ready audit payload.
- A `deliverable-value` CLI workflow and a fourteenth Streamlit workspace for physical and
  cash-settled contract valuation.
- Unit, CLI, reference-store, replay-gate, and headless Streamlit regression coverage.

### Changed

- Package version advanced to 1.3.0.
- Contract multipliers and adjusted deliverable quantities are now separate invariants: the
  multiplier scales aggregate exercise/cash-settlement amounts but never rewrites component units.
- Reference records accept `cash_in_lieu` and `cash_settlement` component types and propagate the
  product settlement currency into replay-ready snapshots.
- Chronological replay uses the shared canonical deliverable parser and continues to fail closed
  before surface calibration for any contract the single-factor Heston engine cannot price.
- Documentation now distinguishes terminal settlement valuation from pre-expiry multi-asset
  option pricing.

## [1.2.0] - 2026-10-16

### Added

- HLN-corrected one-step Diebold–Mariano forecast-accuracy diagnostics using a Bartlett HAC
  long-run variance over evaluation-date mean loss differences.
- Configurable HAC lag selection with an auditable automatic default linked to the bootstrap block
  length and explicit unavailability for zero or numerically invalid long-run variance.
- Two-sided exact binomial date-sign diagnostics with separate Heston-win, BSM-win, and tie counts,
  minimum non-tied-date enforcement, and an explicit independence caveat.
- CLI controls for HAC lags, comparison confidence, and minimum comparison dates, plus an
  interactive comparison-loss selector and diagnostic metrics in the benchmark workspace.
- Regression tests for non-degenerate DM inference, constant-loss variance failure, exact sign
  probabilities, settings validation, CLI serialization, and headless workspace rendering.

### Changed

- Package version advanced to 1.2.0.
- The seeded date-clustered circular block-bootstrap interval remains the primary model-comparison
  conclusion rule; DM/HAC and date-sign results are separately labeled sensitivity diagnostics.
- Research audits and signed content-addressed bundles now carry the expanded comparison record.

## [1.1.0] - 2026-10-16

### Added

- Detached RFC 8032 Ed25519 attestations binding the exact research-bundle bytes, manifest hash,
  bundle-root hash, signing-key fingerprint, explicit issue timestamp, and optional signer label.
- Independent verification fields for bundle integrity, signature validity, exact bundle binding,
  signer trust, and final acceptance; an embedded self-asserted key is never treated as trusted.
- Encrypted PKCS8 private-key and SubjectPublicKeyInfo public-key generation using the maintained
  `cryptography` implementation, with explicit opt-in required for an unencrypted private key.
- `research-keygen`, `research-sign`, and `research-verify` CLI workflows with no-overwrite output
  behavior and mandatory trusted-key verification in the acceptance command.
- Duplicate-JSON-field rejection for bundle manifests and detached attestations.
- Tests covering encrypted keys, deterministic signatures, UTC normalization, self-asserted versus
  trusted verification, wrong trust keys, statement tampering, and the complete CLI workflow.

### Changed

- Package version advanced to 1.1.0.
- `cryptography` is now a pinned runtime dependency for standards-based Ed25519 operations.
- Security guidance now treats private signing keys and password files as secrets and documents
  Windows ACL limitations, key-distribution trust, rotation, and revocation responsibilities.

## [1.0.0] - 2026-10-16

### Added

- Paired BSM-versus-Heston loss inference that clusters the option cross section by evaluation
  timestamp and applies a seeded circular moving-block bootstrap to the chronological date series.
- Configurable comparison loss, resample count, block length, seed, confidence level, and minimum
  date count, with explicit unavailable results for samples too short to support inference.
- Deterministic research ZIPs containing the full audit, row-level predictions, fits, loss tables,
  maturity/moneyness panels, and comparison inference.
- Per-artifact byte counts and SHA-256 values, manifest and bundle-root hashes, bounded in-memory
  bundle verification, CLI export, and a Streamlit download.
- Stable-contract lookup and standard single-security deliverable validation in chronological
  replay, with multiplier and deliverable hashes in every applicable audit record.
- Tests for dependence-aware inference, insufficient histories, deterministic/tampered bundles,
  multiplier-scaled P&L/cost/turnover, and fail-closed structured deliverables.

### Changed

- Package version advanced to 1.0.0.
- Replay now applies contract multipliers to option cash flows, midpoint marks, spread loss,
  explicit fee notional, turnover, and per-contract delta/gamma/vega hedge exposures.
- Adjusted strike/expiry/type terms, multi-component deliverables, cash components, cash-in-lieu,
  and non-underlying securities are rejected before surface calibration until a compatible pricing
  engine exists.
- The Model benchmark workspace now distinguishes point estimates from dependence-aware comparison
  uncertainty and explains why the short synthetic fixture cannot support an interval.

## [0.10.0] - 2026-10-13

### Added

- A strictly chronological rolling BSM-versus-Heston benchmark that fits only trailing snapshots
  and scores the untouched next chain.
- Per-snapshot constant-volatility BSM fits plus Heston calibration behind the mandatory
  SVI/SSVI static-arbitrage gate; rolling forecasts use component-wise parameter medians.
- Row-level dollar, vega-scaled, and zero-inside-spread quote-band losses with overall,
  evaluation-date, maturity, and forward-moneyness summaries.
- Snapshot and prediction-panel SHA-256 values bound into a tamper-evident chronological benchmark
  audit chain.
- A `benchmark` CLI command with deterministic CSV predictions and JSON audit output.
- A thirteenth Streamlit workspace with cached form-submitted runs, next-snapshot comparisons,
  maturity/moneyness heatmaps, audit tables, and downloads.
- Tests for no-look-ahead windows, model-loss panels, invalid training snapshots, prediction
  tampering, CLI output, and the benchmark workspace.

### Changed

- Package version advanced to 0.10.0.
- Replay option sells now execute at bid and buys at ask while open positions remain marked at
  midpoint. Spread loss and proportional transaction fees are separately audited and reconciled.
- Replay turnover and option-fee notionals now use executable quote-side premiums.
- Documentation now distinguishes midpoint valuation, executable-side fills, pricing loss,
  quote-band loss, and remaining execution limitations.
- The next milestone is v1.0 immutable research reports, dependence-aware comparison uncertainty,
  and contract-multiplier/structured-deliverable economics.

## [0.9.1] - 2026-10-13

### Added

- A separate append-only reference database for products, contracts, exchange sessions, and
  corporate actions, preserving source publication and local ingestion time independently.
- Effective-dated source-symbol mappings to stable contract identities, exact last-trade,
  expiration, and settlement timestamps, multipliers, and structured deliverables.
- Product profiles for exercise style, settlement method/style, calendar identity, and explicit
  expiration, last-trade, and settlement rules.
- Explicit open and closed session records with UTC boundaries, IANA timezones, named session
  segments, holiday/early-close reasons, and ambiguity checks.
- Corporate-action records and contract linkage for splits, mergers, spin-offs, distributions,
  symbol changes, and delistings without assuming one universal adjustment formula.
- Raw-file and record SHA-256 values, chained reference batch manifests, append-only triggers, and
  full reference-store integrity verification.
- `reference-init`, `reference-ingest`, `reference-query`, and `reference-verify` CLI workflows.
- Contract/session enrichment fields and provenance hashes on validated historical snapshots.
- Tests for stable contract resolution, mandatory replay references, closed and after-hours
  rejection, future corporate-action isolation, append-only enforcement, tampering, and CLI use.

### Changed

- Package version advanced to 0.9.1.
- Multi-snapshot `HistoricalEventStore.replay_chain` now requires a reference store and fails before
  calibration when contract terms or market sessions are incomplete, ambiguous, or unavailable at
  the formation-time publication cutoff.
- Single historical snapshots remain available for inspection without references but are marked
  `reference_validated=False`.
- The next milestone is v0.10 rolling BSM-versus-Heston benchmarks and executable-price accounting.

## [0.9.0] - 2026-10-08

### Added

- A vendor-neutral historical schema for option quote and trade events with separate market-event,
  source-receipt, and VolForge-ingestion timestamps.
- An append-only SQLite event store with database triggers that reject record or batch mutation.
- Revision-aware `upsert` and `cancel` semantics with independent received-time and ingested-time
  cutoffs, immutable record types, and active-record cancellation checks.
- Explicit `initial`, `incremental`, `backfill`, and `correction` modes; old events cannot enter an
  incremental batch and backfills/corrections require reasons.
- Exact raw-CSV archival, source-file SHA-256 values, per-record payload hashes, and globally
  chained ingestion manifests.
- Integrity verification covering SQLite, manifest linkage, raw artifacts, batch counts, and event
  payload hashes.
- Point-in-time quote selection with configurable maximum age and replay-ready multi-snapshot chain
  construction that uses only quotes received by each formation time.
- `history-init`, `history-ingest`, `history-query`, `history-snapshot`, and `history-verify` CLI
  workflows with deterministic CSV/JSON output.
- Quote/trade ingestion, revision visibility, cancellation, backfill rejection, append-only,
  archive-tampering, snapshot, and CLI tests.

### Changed

- Package version advanced to 0.9.0.
- Historical source files are now intentionally persisted in a content-addressed local archive;
  the security policy documents the resulting data-handling obligations.
- The next release milestone is the contract/reference master and exchange-session normalization.

## [0.8.0] - 2026-10-07

### Added

- Deterministic multi-date SSVI option-chain fixtures with fixed expirations and listed strikes.
- Exact-timestamp snapshot partitioning, per-snapshot validation, constrained SVI/SSVI fitting,
  and Heston recalibration with a hard stop at the first unsafe day.
- First-snapshot-only target and bracketing-call selection; later missing marks block the replay
  rather than triggering look-ahead contract replacement.
- Listed-midpoint self-financing hedge accounting with cash carry, dividends, transaction costs,
  daily rebalancing, final liquidation, and machine-precision reconciliation checks.
- Canonical snapshot SHA-256 hashes and a linked daily audit chain covering calibration status,
  Heston parameters, positions, trades, costs, cash, and P&L.
- Public Heston price/delta/gamma/instantaneous-vega/theta metrics for listed-contract hedge sizing.
- `generate-replay` and `replay` CLI commands with deterministic CSV and JSON artifacts.
- A twelfth Streamlit workspace for synthetic/uploaded replay, P&L, gate diagnostics, and audit
  downloads.
- Determinism, missing-mark, tamper-detection, accounting, CLI, and AppTest coverage.

### Changed

- Package version advanced to 0.8.0.
- The recommended next milestone is a vendor-adapter and benchmark layer for larger rolling,
  genuinely historical out-of-sample studies.

## [0.7.0] - 2026-10-07

### Added

- Explicit separation of risk-neutral pricing parameters from realized Heston path parameters and
  ex-dividend spot drift.
- A deterministic multi-seed, joint path/rebalance refinement ladder for every hedge strategy.
- Matched-Q, physical-drift, variance-parameter misspecification, cost-stress, and combined-stress
  scenarios using common seeds.
- Strategy-level acceptance checks for refinement stability, worst-scenario P&L volatility
  reduction, seed dispersion, position clipping, conditioning fallbacks, and exact attribution.
- A typed downstream robustness report with eligible strategies, failure reasons, and byte-stable
  JSON output.
- A `heston-robustness` CLI command with configurable scenarios, tolerances, budgets, and audit
  output.
- An eleventh Streamlit workspace for refinement, scenario, and gate diagnostics.
- Physical-drift, determinism, pass/block, cost-stress, CLI, and AppTest coverage.

### Changed

- Package version advanced to 0.7.0.
- The recommended next milestone is chronological market replay with point-in-time calibration.

## [0.6.0] - 2026-10-07

### Added

- A common-path, seeded Heston hedging engine for unhedged, delta, delta-gamma, delta-vega, and
  delta-gamma-vega strategies.
- Self-financing cash and dividend accounting, stock/option transaction costs, terminal stock
  liquidation, configurable hedge strikes, and option-position caps.
- State-dependent COS delta, gamma, instantaneous vega, and calendar theta at every path and
  rebalance date.
- Scaled hedge-system conditioning diagnostics, least-squares fallbacks, and clipped-position
  counts.
- Exact pathwise attribution into delta, gamma, vega, theta, financing, transaction cost, and
  residual, with a reported reconciliation gap.
- A `heston-hedge` CLI command with optional JSON audit output.
- A tenth Streamlit workspace with distribution, comparison, attribution, and stability views.
- Determinism, cost, dispersion, reconciliation, CLI, and AppTest coverage.

### Changed

- Package version advanced to 0.6.0.
- The recommended next milestone is robust multi-scenario hedge validation before chronological
  out-of-sample research.

## [0.5.1] - 2026-10-07

### Added

- Five-point COS and PDE Heston delta, gamma, parallel-level vega, calendar theta, and rho with
  explicit bump and three-point discrepancy diagnostics.
- Seeded pathwise delta/rho, conditional likelihood-ratio gamma/delta, and paired
  common-random-number vega/theta estimators with antithetic standard errors.
- Conditional terminal log-spot variance and Gaussian-shock diagnostics on full QE-M paths.
- A `heston-greeks` CLI command with COS, PDE, and Monte Carlo engines.
- A ninth Streamlit workspace for estimator comparison, uncertainty intervals, and method
  disagreement diagnostics.
- Constant-variance Black-Scholes, PDE/COS, determinism, sampling, CLI, and AppTest coverage.

### Changed

- PDE bump-and-revalue can hold the spatial grid fixed and use cubic log-price interpolation,
  preventing a grid-recentering artifact in gamma.
- The recommended next milestone is hedging and P&L attribution.

## [0.5.0] - 2026-10-07

### Added

- A Cox-Ross-Rubinstein American vanilla benchmark with European comparison, early-exercise
  premium, and exercise-boundary diagnostics.
- Reusable seeded QE-M spot/variance path simulation and Heston Longstaff-Schwartz valuation with
  a spot/variance regression basis, conditional sampling interval, and exercise diagnostics.
- A two-dimensional Heston obstacle solver using Ikonen-Toivanen splitting with Crank-Nicolson and
  Rannacher time stepping.
- `american-binomial`, `american-lsm`, and `american-pde` CLI commands.
- An eighth Streamlit workspace for Heston LSM/PDE cross-checks and the CRR reference.
- Benchmark, constant-volatility-limit, determinism, identity, CLI, and AppTest coverage.

### Changed

- The documented next milestone is independent Heston Greek estimators in v0.5.1.

## [0.4.1] - 2026-10-06

### Added

- An automatic three-level refinement ladder for Heston COS, QE Monte Carlo, and finite-difference
  PDE prices.
- A typed convergence report with per-engine error estimates, Monte Carlo interval compatibility,
  cross-engine consensus spread, explicit failure reasons, and deterministic JSON serialization.
- `require_numerical_convergence`, a hard gate that blocks under-resolved prices from downstream
  workflows.
- The `heston-converge` CLI command with configurable tolerances, computation budgets, deterministic
  exit status, and optional JSON audit output.
- An opt-in convergence-gate view in the numerical-engines workspace with refinement history,
  eligibility status, failure diagnostics, and audit download.
- Passing, blocking, determinism, CLI, expiry, validation, and interactive Streamlit tests for the
  convergence workflow.

### Changed

- Package version advanced to 0.4.1 and the numerical workspace defaults to a simulation budget
  that can pass its standard convergence profile for the published Heston benchmark.

## [0.4.0] - 2026-10-06

### Added

- Seeded Heston Monte Carlo valuation with Andersen quadratic-exponential variance transitions,
  a martingale correction, antithetic sampling, and a discounted-underlying control variate.
- Sampling diagnostics including standard error, a 95% confidence interval, control coefficient,
  and discounted terminal-underlying mean.
- A two-dimensional Heston finite-difference solver on non-uniform log-price/variance grids with
  central spatial differences, Crank-Nicolson time stepping, Rannacher damping, and explicit
  zero-/high-variance boundary handling.
- Typed numerical settings and result objects, plus `heston-mc` and `heston-pde` CLI commands.
- A seventh Streamlit workspace that cross-checks Monte Carlo and PDE prices against COS.
- Determinism, variance-reduction, COS cross-check, grid-refinement, parity, CLI, and Streamlit
  tests for the new engines.

### Changed

- Package version advanced to 0.4.0 and numerical-engine diagnostics are now available through
  package, CLI, and Streamlit layers.
- The documented roadmap now treats Monte Carlo and PDE cross-checking as complete.

## [0.3.0] - 2026-10-06

### Added

- Heston risk-neutral parameter types, characteristic function, first and corrected second cumulants,
  and explicit Feller diagnostics.
- Vectorized Fourier-cosine European call/put pricing with bounded-put evaluation, put-call parity,
  configurable terms and truncation width, and numerical bound enforcement.
- Multi-start nonlinear least-squares Heston calibration in normalized forward coordinates using
  vega-scaled price residuals and independently inverted final implied-volatility errors.
- An optional Feller-compliant optimizer parameterization.
- Structured Heston fit points, deterministic CSV/JSON audits, and parameter tables.
- `heston-price` and `heston-calibrate` CLI commands.
- A gated Heston calibration Streamlit workspace with fit, residual, parameter, and Feller views.
- Published COS benchmark, characteristic-function, Black-Scholes limit, parity, calibration,
  gate-rejection, CLI, and Streamlit tests.

### Changed

- Package version advanced to 0.3.0 and Streamlit navigation now contains six workspaces.
- The v0.2 surface gate is now enforced inside the Heston calibration API, not only by adapters.

## [0.2.0] - 2026-10-06

### Added

- Canonical surface-quote preparation in forward log-moneyness and total implied variance.
- Constrained per-expiry raw-SVI calibration with analytic derivatives, variance positivity,
  Lee-wing, and sampled density-factor constraints.
- Joint power-law SSVI calibration with monotone ATM variance and Gatheral-Jacquier sufficient
  butterfly conditions.
- Sampled butterfly and adjacent-expiry calendar-arbitrage reports.
- Structured parameter tables, deterministic calibration JSON, and evaluated-surface CSV exports.
- A downstream eligibility gate that rejects optimizer, butterfly, calendar, or analytic-condition
  failures before stochastic-volatility calibration.
- A volatility-surfaces Streamlit workspace and `volforge surface` CLI command.
- Numerical derivative, exact-fit, invalid-surface, gate, CLI, and Streamlit surface tests.

### Changed

- Streamlit navigation now contains five workspaces.
- Package version advanced to 0.2.0 and documentation now describes the calibrated-domain limits
  of static-arbitrage checks.

## [0.1.0] - 2026-10-06

### Added

- Black-Scholes-Merton and Bachelier European call/put pricing.
- Analytic delta, gamma, vega, theta, and rho with documented units.
- Explicit no-arbitrage bounds and safe expiry/zero-volatility conventions.
- Brent implied-volatility inversion and safeguarded Newton fallback for both models.
- Structured convergence, residual, iteration, method, and failure diagnostics.
- ACT/365F CSV validation/enrichment that retains invalid observations.
- Seeded synthetic option chains with configurable smiles, noise, spreads, and diagnostic faults.
- Put-call parity and aggregate data-quality reports.
- Pricing, implied-volatility, chain-explorer, and diagnostics Streamlit workspaces.
- Shared `price`, `iv`, `generate`, and `validate` CLI commands.
- Ruff, pytest, AppTest smoke coverage, `uv.lock`, and GitHub Actions CI.
- Mathematical, schema, architecture, contribution, and security documentation.
