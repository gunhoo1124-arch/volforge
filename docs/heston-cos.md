# Heston calibration and COS pricing

VolForge v0.3 adds a risk-neutral Heston model and Fourier-cosine pricing without weakening the
static-arbitrage contract introduced in v0.2. Heston calibration cannot consume a surface until
`require_downstream_eligible` accepts it.

## Model parameters

`HestonParameters` stores:

| Field | Meaning | Public domain |
|---|---|---|
| `kappa` | Variance mean-reversion speed | Positive |
| `theta` | Long-run variance | Positive |
| `vol_of_vol` | Diffusion coefficient of variance | Positive |
| `rho` | Spot/variance Brownian correlation | Strictly between -1 and 1 |
| `initial_variance` | Time-zero instantaneous variance | Positive |

These are risk-neutral calibration parameters, not historical estimates. `theta` and
`initial_variance` are variances rather than volatilities.

The Feller margin is `2*kappa*theta - vol_of_vol**2`. A non-negative value is reported as
`feller_satisfied=True`. Feller compliance is diagnostic by default because useful market fits can
have an attainable zero-variance boundary. Selecting the strict policy parameterizes vol-of-vol as
a fraction of `sqrt(2*kappa*theta)`, making compliance structural rather than a soft penalty.

## COS engine

The COS engine evaluates the stable Heston characteristic function once per maturity and prices a
vector of strikes together. It uses analytic put-payoff coefficients and recovers calls by parity.
This avoids direct integration of the exponentially growing call payoff.

`COSSettings` exposes:

- `terms`, at least 32 and defaulting to 256; and
- `truncation_width`, between 5 and 30 and defaulting to 16.

The truncation interval uses the first cumulant and a corrected analytic second cumulant of log
return. The wider default deliberately prioritizes reliable tails over the smallest possible term
count. Increasing the width without increasing terms can reduce spectral resolution, so accuracy
should be checked by increasing both settings for unusual short-dated or extreme parameter sets.

Every finite-maturity result is compared with European no-arbitrage bounds. Deviations larger than
scaled floating-point tolerance raise `ArithmeticError`; tiny deviations are normalized to the
bound. At expiry the engine returns intrinsic value directly.

Primary references:

- [Heston’s original stochastic-volatility paper](https://doi.org/10.2307/2962057)
- [Fang and Oosterlee’s original COS paper](https://ir.cwi.nl/pub/13283/13283B.pdf)
- [Cui, del Baño Rollin, and Germano on nonlinear least-squares Heston calibration](https://arxiv.org/abs/1511.08718)

## Calibration pipeline

`calibrate_heston` follows this sequence:

1. call the v0.2 downstream gate on the supplied SVI/SSVI result;
2. select at least two maturities and five strictly increasing log-moneyness points;
3. evaluate total variance and convert it to target Black-Scholes implied volatility;
4. generate normalized call targets with `F=1`, `K=exp(k)`, and zero carry;
5. run bounded deterministic multi-start nonlinear least squares;
6. divide each price error by Black-Scholes vega, subject to a wing floor;
7. reprice the best parameters and independently invert every price to implied volatility; and
8. return the parameters, optimizer state, Feller state, price/IV errors, individual fit points,
   and explicit failure reasons.

Normalization removes spot and discount scale without changing forward log-moneyness. It does not
turn the result into a historical or physical-measure parameter estimate.

## CLI examples

Price one option:

```bash
uv run volforge heston-price --option-type call --spot 100 --strike 100 --maturity 1 \
  --kappa 1.5768 --theta 0.0398 --vol-of-vol 0.5751 --rho -0.5711 \
  --initial-variance 0.0175 --cos-terms 512
```

Calibrate from a raw option-chain CSV through SSVI and retain the fit comparison and audit:

```bash
uv run volforge heston-calibrate data/synthetic.csv reports/heston-fit.csv \
  --surface-model ssvi --report reports/heston-audit.json
```

The command exits with status `2` for chain/surface failures, a rejected static-arbitrage gate,
optimizer failure, invalid final prices, or failed final implied-volatility inversion.

## Interpretation limits

One time-homogeneous Heston parameter set may not reproduce every valid surface. Optimizer success
does not imply a small economic fit error, so price and IV RMSE must be reviewed. COS convergence is
numerical and should be checked under more terms and a wider interval for difficult regimes. The
model does not include jumps or parameter term structures. v0.4 supplies independent Monte Carlo
and PDE cross-checks, but not American exercise, path-dependent contracts, or hedging experiments.
Outputs are research diagnostics, not investment advice.
