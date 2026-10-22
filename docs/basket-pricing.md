# Pre-expiry basket pricing

VolForge v1.4 prices a European option on an explicit adjusted deliverable under constant,
correlated risk-neutral geometric Brownian motion (GBM). The engine is a bounded research model,
not an extension of the single-factor Heston replay. Arithmetic basket options generally do not
have a closed-form price, so the primary result is a seeded Monte Carlo estimate; an independently
computed moment-matched lognormal value is reported only as an approximation and cross-check
([Deelstra, Liinev, and Vanmaele](https://openaccess.city.ac.uk/id/eprint/15729/7/basket_option_qf.pdf)).

## Contract and market inputs

The contract uses the canonical component schema in
[adjusted deliverables](adjusted-deliverables.md). For a physical package, every `security`
component becomes a risk factor and `cash` or resolved `cash_in_lieu` stays a fixed amount. A
cash-settled option has one risk factor named by `underlying_id` and uses the declared cash-settlement
quantity, normally the contract multiplier.

The market JSON has this form:

```json
{
  "assets": [
    {"asset_id": "ACME", "spot": 120, "volatility": 0.25, "dividend_yield": 0.01},
    {"asset_id": "SPIN", "spot": 20, "volatility": 0.35, "dividend_yield": 0.00}
  ],
  "correlation": [[1.00, 0.35], [0.35, 1.00]]
}
```

Asset identifiers are canonicalized lexicographically together with the correlation rows and
columns. The engine requires exact equality between contract risk factors and market assets; it
rejects missing or unused assets. Spots must be positive, volatilities non-negative, and all
numeric inputs finite. The correlation matrix must be symmetric, have a unit diagonal, contain
entries in `[-1, 1]`, and be positive semidefinite. No spot, volatility, dividend yield,
correlation, currency conversion, or borrow adjustment is estimated or invented. Inputs are
limited to 16 risk factors and 1 MiB of market JSON.

## Model and payoff

Under the pricing measure, component `i` follows

```text
dS_i / S_i = (r - q_i) dt + sigma_i dW_i
corr(dW_i, dW_j) = rho_ij
```

with constant rate `r`, dividend yield `q_i`, volatility `sigma_i`, and correlation `rho_ij`.
The exact terminal GBM transition is sampled after a positive-semidefinite eigen factorization.
The CERMICS multidimensional Black-Scholes notes give the same correlated-GBM construction and
basket Monte Carlo setting
([CERMICS](https://cermics.enpc.fr/~bl/Halmstad/monte-carlo/td-2/index.html)).

For security quantities `a_i`, fixed cash `c`, contract multiplier `M`, and quoted strike `K`,

```text
Y_T = c + sum_i(a_i S_i(T))
call payoff = exp(-rT) max(Y_T - K M, 0)
put payoff  = exp(-rT) max(K M - Y_T, 0)
```

All component quantities must be positive. Negative-weight spread baskets, Asian averaging,
American exercise, barriers, quanto conversion, stochastic rates, stochastic dividends, local or
stochastic volatility, jumps, default, and discrete corporate-action uncertainty are outside this
engine.

## Estimator and diagnostics

The default estimator uses antithetic normal draws. Each positive/negative shock pair is collapsed
to one independent pair mean before the standard error and control coefficient are estimated; the
audit therefore distinguishes total paths from independent estimator samples. Its control variate
is the discounted delivered package itself,

```text
C = exp(-rT) Y_T
E[C] = sum_i(a_i S_i(0) exp(-q_i T)) + c exp(-rT)
X_cv = X - beta (C - E[C])
```

where `beta` is the sample covariance of the option payoff and control divided by the control
variance. Control variates are a standard basket-option variance-reduction technique; the engine
reports the crude and adjusted standard errors and their variance ratio rather than assuming the
control helped ([Goudenège et al.](https://www.sciencedirect.com/science/article/abs/pii/S0167668713000309)).

The result also includes:

- the seed, path count, control coefficient, confidence level, standard error, and normal sampling
  interval;
- discounted Jensen/no-arbitrage lower and upper bounds, with a five-standard-error tolerance for
  a simulated estimate;
- pathwise component spot deltas and their sampling errors, adjusted by the same control variate
  when enabled;
- a lognormal approximation formed by matching the first two risk-neutral moments of the entire
  delivered package;
- canonical SHA-256 identities for both the contract terms and market input.

At expiry, for fixed cash only, or when every volatility is zero, VolForge uses the exact
deterministic payoff and returns zero sampling error. For a one-asset package without fixed cash,
the moment-matched price reduces to the corresponding Black-Scholes-Merton price; regression tests
also compare Monte Carlo price and delta with that independent closed form.

The reported confidence interval covers sampling error conditional on the supplied model and
parameters. It does not cover parameter estimation, correlation instability, model error,
liquidity, execution, or the approximation error of the separate moment match.

## Package API

```python
from volforge import (
    BasketMonteCarloSettings,
    parse_basket_market,
    parse_deliverable_terms,
    price_basket_option,
)

terms = parse_deliverable_terms(
    [
        {"asset_id": "ACME", "asset_type": "security", "quantity": 50},
        {"asset_id": "SPIN", "asset_type": "security", "quantity": 10},
        {"asset_id": "USD-CIL", "asset_type": "cash_in_lieu", "quantity": 20},
    ],
    settlement_method="physical",
    contract_multiplier=100,
    currency="USD",
    underlying_id="ACME",
)
market = parse_basket_market(
    {
        "assets": [
            {"asset_id": "ACME", "spot": 120, "volatility": 0.25, "dividend_yield": 0.01},
            {"asset_id": "SPIN", "spot": 20, "volatility": 0.35},
        ],
        "correlation": [[1, 0.35], [0.35, 1]],
    }
)
result = price_basket_option(
    terms,
    market,
    option_type="call",
    strike=60,
    maturity=0.5,
    rate=0.03,
    settings=BasketMonteCarloSettings(paths=100_000, seed=19),
)
assert result.bounds_passed
```

## CLI and workspace

Store the component list and market object in separate JSON files:

```bash
volforge basket-price data/deliverable.json data/basket-market.json \
  --settlement-method physical --option-type call --strike 60 \
  --contract-multiplier 100 --currency USD --underlying-id ACME \
  --maturity 0.5 --rate 0.03 --paths 100000 --seed 19
```

The command writes a deterministic JSON audit for a given seed. `--no-antithetic` and
`--no-control-variate` expose the crude alternatives; `--confidence-level` changes the sampling
interval. The Basket pricing Streamlit workspace calls the same package API, places the simulation
behind a form, bounds cached results, and exposes prices, bounds, sampling diagnostics, component
deltas, canonical hashes, and a downloadable audit.

## Replay boundaries

A standalone v1.4 price is not eligibility for the single-factor SVI/SSVI-Heston replay. That
engine still rejects adjusted and cash-settled contracts before surface calibration because its
schema and hedge policy describe one underlying.

The separate v1.5 [basket hedge replay](basket-replay.md) accepts chronological market packages
that explicitly include component spots, volatilities, dividend yields, correlations, and
executable hedge quotes received by each formation time. It never copies one static input backward
through history. The v1.6 [basket-input store](basket-input-store.md) can now preserve revisions,
raw artifacts, and manifest identities and bind resolved packages into replay. A retained store is
still local provenance rather than vendor authentication or a trusted timestamp.
