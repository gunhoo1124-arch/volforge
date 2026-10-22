# Option-chain CSV schema

## Required columns

| Column | Type | Validation |
|---|---|---|
| `quote_timestamp` | ISO 8601 timestamp | Must parse; normalized to UTC |
| `expiration` | ISO 8601 date/time | Must parse as UTC and not precede quote time |
| `strike` | number | Finite and positive |
| `option_type` | text | `call` or `put`, case-insensitive |
| `bid` | number | Finite, non-negative, and no greater than ask |
| `ask` | number | Finite, non-negative, and no less than bid |
| `underlying_spot` | number | Finite and positive |
| `risk_free_rate` | number | Finite continuous annual decimal |
| `dividend_yield` | number | Finite continuous annual decimal |

Example:

```csv
quote_timestamp,expiration,strike,option_type,bid,ask,underlying_spot,risk_free_rate,dividend_yield
2025-01-02T15:30:00Z,2025-04-02T15:30:00Z,100,call,4.95,5.05,100,0.03,0.01
2025-01-02T15:30:00Z,2025-04-02T15:30:00Z,100,put,4.45,4.55,100,0.03,0.01
```

## Derived columns

The enrichment pipeline adds:

- `maturity_years` using ACT/365F;
- `midpoint`, or null for unusable quotes;
- `lower_bound` and `upper_bound`;
- `implied_volatility`, `iv_converged`, `iv_iterations`, `iv_residual`, and
  `iv_failure_reason`;
- boolean flags for negative, crossed, stale, duplicate, below-bound, and above-bound rows;
- `validation_reasons`, a semicolon-separated audit trail; and
- `is_valid`, true only when the row passes validation and implied-volatility inversion.

A duplicate contract has the same quote timestamp, expiration, strike, and option type. Both copies
are flagged. By default, a quote is stale when it is more than 24 hours older than the newest valid
quote timestamp in the file. CLI/API callers can supply an explicit valuation time and threshold.

Only missing required columns abort the entire ingestion. Bad row values are converted to null
where needed for calculation but the row remains in output with reasons.
