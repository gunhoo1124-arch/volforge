# Volatility-surface calibration

VolForge v0.2 calibrates Black-Scholes implied-volatility observations in forward
log-moneyness and total implied variance. The numerical package, CLI, and Streamlit workspace use
the same calibration and validation APIs.

## Input pipeline

`prepare_surface_quotes` accepts an enriched BSM chain. It keeps only rows that passed chain
validation and implied-volatility inversion, combines calls and puts at each strike and expiry by
median implied volatility, and calculates:

```text
F = S exp((r-q)T)
k = log(K/F)
w = implied_volatility^2 T
```

At least five unique strikes per expiry and two expiries are required. Missing, invalid, or sparse
inputs fail explicitly instead of being imputed.

## Calibration choices

Choose **SVI slices** for an independently parameterized raw-SVI smile at each observed expiry.
The fitter uses multiple deterministic starts and enforces positive variance, admissible wing
slopes, and sampled non-negative density. This choice is flexible, but calendar consistency is a
post-fit cross-slice diagnostic.

Choose **SSVI** for one joint power-law surface. It shares `rho`, `eta`, and `gamma` across expiry,
fits a nondecreasing ATM total-variance node `theta` at each maturity, and enforces the implemented
Gatheral-Jacquier sufficient constraints during optimization. It is less flexible but couples
maturities directly.

Both models report total-variance RMSE, implied-volatility RMSE, optimizer status, parameter values,
sampled butterfly statistics, sampled calendar statistics, and explicit failure reasons.

## Downstream gate

`require_downstream_eligible(calibration)` is the supported handoff into later model stages. It
rejects a surface when any of these checks fail:

- constrained optimization did not converge;
- an applicable analytic SSVI condition failed;
- any calibrated slice has non-positive variance, a negative sampled density factor, or an invalid
  wing slope; or
- any adjacent-expiry total-variance spread is negative on the diagnostic grid.

The gate returns the unchanged calibration when every check passes and otherwise raises
`SurfaceValidationError` with the accumulated reasons. v0.3 Heston calibration must call this gate
instead of consuming raw observations directly.

## CLI workflow

Generate a raw deterministic chain, calibrate it, and retain both numerical outputs and the audit:

```bash
uv run volforge generate data/synthetic.csv --seed 7
uv run volforge surface data/synthetic.csv reports/surface.csv \
  --model ssvi --report reports/surface-audit.json
```

The evaluated CSV contains model, expiration, maturity, log-moneyness, total variance, and implied
volatility. The JSON report contains parameters, errors, convergence metadata, arbitrage
diagnostics, eligibility, and failure reasons. Exit status is `0` only for an eligible surface and
`2` for rejected calibration or input failures.

## Interpretation limits

Static-arbitrage diagnostics do not demonstrate an executable trade. They use midpoints, ignore
fees and market impact, and evaluate a finite log-moneyness domain. SVI interpolation and SSVI ATM
variance interpolation are restricted to the calibrated maturity horizon; the package refuses
out-of-range maturity evaluation. Surface outputs are model diagnostics, not investment advice.
