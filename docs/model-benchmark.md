# Rolling model benchmark

VolForge v1.0 compares a constant-volatility Black-Scholes-Merton baseline with Heston prices
under a strict chronological information boundary. It is designed to answer whether added
stochastic-volatility structure improves the price of the next observed chain, rather than merely
fitting the chain used for calibration.

Bakshi, Cao, and Chen estimate option-model parameters from the previous day's prices and use them
to price the current day's contracts, then report errors across maturity and moneyness categories
([Journal of Finance](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1997.tb02749.x)).
VolForge generalizes that one-day lag to a configurable trailing window while preserving the same
strict ordering. The benchmark also responds to documented look-ahead risk from using information
unavailable at formation time to clean option samples
([Review of Financial Studies](https://academic.oup.com/rfs/advance-article-abstract/doi/10.1093/rfs/hhag061/8722307)).

## Chronological protocol

For evaluation timestamp `t_i`:

1. Take exactly the preceding `training_window` snapshots. Every training timestamp must be
   strictly earlier than `t_i`.
2. For each training snapshot independently, validate every quote, fit a constant BSM volatility,
   fit constrained SVI or SSVI in total implied variance, require the full static-arbitrage gate,
   and only then calibrate Heston.
3. Aggregate daily BSM volatilities and each Heston parameter by the component-wise median. If the
   Feller condition is required, the aggregated Heston vector must satisfy it too.
4. Use current spot, rates, dividend yield, strikes, and maturities with those lagged parameters to
   price every valid current contract. The current snapshot is scored before it can be fitted for
   any later forecast.
5. Fit `t_i` only after scoring it, and only when a later timestamp exists.

The engine stops at the first invalid quote snapshot, surface failure, Heston failure, incomplete
training window, or pricing failure. It does not skip a bad day, repair it with later data, or
reselect observations using future availability. Reference-enriched inputs must have
`reference_validated=True` on every row and must carry `stable_contract_id`.

## Estimation and losses

The BSM baseline is one volatility per training snapshot, estimated by least squares after dividing
dollar price residuals by observed-option BSM vega with a small numerical floor. Heston uses the
same vega-scaled price-residual convention against the arbitrage-gated total-variance surface.
Christoffersen and Jacobs emphasize that option-model conclusions depend on the loss function and
that estimation and comparison losses should be aligned
([Journal of Financial Economics](https://doi.org/10.1016/j.jfineco.2003.02.001)). VolForge
therefore retains both vega-scaled errors and the following directly interpretable dollar losses:

- `rmse`: root mean squared model price minus quote midpoint;
- `mae`: mean absolute midpoint error;
- `bias`: signed mean midpoint error;
- `vega_scaled_rmse` and `vega_scaled_mae`: errors aligned with the calibration normalization;
- `quote_coverage`: fraction of model prices inside the observed bid/ask interval;
- `quote_band_loss`: zero inside the quote and otherwise the distance to the nearest quoted side.

The primary report includes overall, per-evaluation-timestamp, and maturity/moneyness panels.
Maturity buckets are short (`<60` days), medium (`60-180` days), and long (`>=180` days).
Moneyness uses the current forward ratio `F/K`: low strike above `1.03`, near forward from `0.97`
through `1.03`, and high strike below `0.97`. These panels diagnose where a model fails; they do not
constitute a multiple-testing-adjusted statistical claim.

## Dependence-aware comparison uncertainty

The same option appears under two models, many options share one evaluation date, and adjacent
dates need not be independent. VolForge therefore never treats row-level losses as independent
replicates. It pairs BSM and Heston losses contract by contract, computes Heston minus BSM loss,
and averages the complete cross section inside each evaluation timestamp. A negative difference
favors Heston.

The chronological date series is resampled with a circular moving-block bootstrap. Circular blocks
preserve local serial dependence while keeping every date's cross section intact. This follows the
block-resampling principle for dependent stationary observations and the broader equal-predictive-
accuracy framework, which permits non-quadratic, non-Gaussian, serially and contemporaneously
correlated forecast errors
([Politis and Romano, JASA](https://doi.org/10.1080/01621459.1994.10476870);
[Diebold and Mariano, JBES](https://doi.org/10.1080/07350015.1995.10524599)).

The report stores the point difference, bootstrap standard error, percentile interval, fraction of
bootstrap means below zero, seed, resample count, date count, and block length. The default block
length is the rounded cube root of the date count. Fewer than five evaluation dates produce an
explicitly unavailable result, not artificial confidence. The bundled four-snapshot fixture has
only two scored dates and is intentionally below that threshold.

VolForge v1.2 adds two separately labeled sensitivity diagnostics to that same date-level loss
series:

- an HLN-corrected Diebold–Mariano statistic using a Bartlett heteroskedasticity-and-
  autocorrelation-consistent long-run variance; and
- a two-sided exact binomial sign test of whether Heston wins half of the non-tied evaluation
  dates.

The HAC lag count is configurable. By default it is one less than the bootstrap block length, with
a floor of zero. A zero or numerically invalid long-run variance makes the DM diagnostic explicitly
unavailable rather than producing an infinite statistic. The sign test excludes exact ties and is
withheld unless the configured minimum number of non-tied dates is present. Its exact p-value uses
[`scipy.stats.binomtest`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

These diagnostics answer different questions and are not combined into a single model-selection
claim. The DM framework compares forecast accuracy under a chosen loss
([Diebold and Mariano, JBES](https://doi.org/10.1080/07350015.1995.10524599)); the one-step finite-
sample correction follows
[Harvey, Leybourne, and Newbold](https://doi.org/10.1016/S0169-2070(96)00719-4). The exact sign test
is an independence-based directional sensitivity check and is not dependence-robust. The seeded
block-bootstrap interval therefore remains VolForge's primary conclusion rule. Consistent with
[Diebold's retrospective](https://www.nber.org/papers/w18391), results are described as comparative
historical forecast performance, not proof that one structural model is true or universally
superior.

## Tamper-evident audit

Every input snapshot receives a row-order-independent SHA-256 hash. Every chronological audit step
binds the prior audit hash, current snapshot hash, training interval, no-look-ahead result, lagged
forecast parameters, any newly fitted parameters, model RMSE/quote coverage, and a SHA-256 hash of
the complete row-level prediction panel. `verify_benchmark_hash_chain` recomputes both layers.

The chain detects changed artifact content; it is not a digital signature, trusted timestamp, or
proof of vendor provenance. Use the append-only historical and reference stores before the
benchmark when those guarantees matter.

`build_benchmark_research_bundle` packages the audit, predictions, fits, loss tables, panels, and
comparison inference into a deterministic ZIP. Its manifest records every artifact's byte count
and SHA-256, the benchmark settings and final audit hash, a bundle-root SHA-256, and a manifest
SHA-256. `verify_research_bundle` recomputes all of them. This makes the bundle reproducible,
content-addressed, and tamper-evident; it does not make a copied file physically immutable or
cryptographically signed. See [Research-report bundles](research-reports.md).

## Package, CLI, and app

```python
from volforge.benchmark import BenchmarkSettings, run_chronological_benchmark
from volforge.data import generate_synthetic_replay

result = run_chronological_benchmark(
    generate_synthetic_replay(),
    settings=BenchmarkSettings(training_window=2, cos_terms=64),
)
assert result.downstream_eligible
```

```bash
volforge benchmark data/replay.csv reports/model-predictions.csv \
  --training-window 2 --surface-model ssvi \
  --comparison-loss squared_error --comparison-hac-lags 1 \
  --report reports/model-benchmark.json \
  --bundle reports/model-benchmark.zip
```

The Streamlit model-benchmark workspace exposes the same package API. Expensive runs are submitted
as one form and stored in a bounded cache. It displays next-snapshot RMSE, bid/ask coverage,
maturity/moneyness heatmaps, dependence-aware uncertainty, chronological audits, and downloadable
JSON/CSV artifacts plus the complete research bundle.

## Interpretation limits

- The bundled four-snapshot SSVI fixture tests chronology and implementation; it cannot establish
  historical or economic superiority.
- A coordinate-wise parameter median is a robust rolling forecast rule, not a dynamic state-space
  estimator, and parameter uncertainty is not propagated.
- Quote-band coverage is a pricing diagnostic, not proof that a trade could be filled. It omits
  latency, depth, fill probability, impact, fees, margin, and funding constraints.
- Bootstrap intervals quantify sampling variation in the observed date clusters; they do not cover
  parameter uncertainty, non-stationarity, data-selection uncertainty, or multiple comparisons.
- HLN-DM asymptotics can be unreliable in short or strongly dependent samples. The exact sign test
  assumes independent non-tied date signs. Neither diagnostic overrides an unavailable or
  inconclusive block-bootstrap result.
