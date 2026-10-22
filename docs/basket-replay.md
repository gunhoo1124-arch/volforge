# Point-in-time basket hedge replay

VolForge v1.5+ replays one short European adjusted option from its first executable quote through
explicit expiry settlement. Each pre-expiry snapshot is priced by the v1.4 correlated-GBM basket
engine, and the short option is delta-hedged in every declared risk factor using supplied component
bid/ask quotes. This is a separate multi-asset ledger; it does not route adjusted contracts through
the single-factor SVI/SSVI-Heston replay.

The information boundary is as important as the pricing model. Research designs that filter or
select observations using information unavailable at portfolio formation introduce look-ahead
bias ([Duarte et al.](https://academic.oup.com/rfs/advance-article-abstract/doi/10.1093/rfs/hhag061/8722307)).
VolForge therefore represents market event, receipt, and formation time separately and stops at the
first violation.

## Snapshot CSV contract

Rows must already be in strictly increasing formation-time order. The final formation timestamp
must equal the declared option expiry, and at least one earlier row is required.

| Column | Meaning |
| --- | --- |
| `formation_timestamp` | Time at which the hedge decision is made. Must include a timezone. |
| `market_timestamp` | Event time of the complete option/component/model-input package. |
| `received_timestamp` | Time the package became available to the replay. |
| `source` | Required non-empty supplied provenance label. |
| `source_record_id` | Required source-local identity, unique with `source` in the replay. |
| `option_bid`, `option_ask` | Executable adjusted-option quote per contract. At expiry both must equal settlement payoff. |
| `risk_free_rate` | Continuous annual rate used until the next snapshot. |
| `market_json` | Complete v1.4 component spots, volatilities, dividend yields, and PSD correlation. |
| `execution_json` | Exact component bid/ask set used for hedge execution. |
| `input_store_snapshot_sha256` | Optional v1.6 resolved-store snapshot identity. |
| `input_manifest_hashes_json` | Optional non-empty JSON list of contributing store manifests. |
| `input_estimate_sha256` | Optional v1.7 statistical-estimate self-hash. |
| `input_estimate_json` | Optional complete canonical v1.7 estimate audit. |
| `input_covariance_selection_sha256` | Optional v1.8 covariance-selection self-hash for the advanced ledger. |
| `input_covariance_selection_json` | Optional complete canonical v1.8 selection audit for the advanced ledger. |

The timing gate requires

```text
market_timestamp <= received_timestamp <= formation_timestamp
formation_timestamp - market_timestamp <= maximum_market_age
```

Risk-factor identities cannot change during a replay. Every market spot must lie inside its
component execution quote, and execution quotes must cover exactly the canonical market assets.
The final option quote is treated as an official resolved settlement amount and must equal the
payoff recomputed from the final component spots and canonical deliverable.

`source` and `source_record_id` retain supplied lineage, but they are not authentication. A plain
CSV path does not prove that a vendor issued a record, preserve revisions, or establish that a
supplied volatility or correlation was historically observable. VolForge v1.6 can materialize
these rows from its append-only basket-input store and bind snapshot/manifest hashes. Independent
source identity and trusted timestamps still remain operator/integration responsibilities.

## Self-financing convention

The replay holds `N > 0` short option contracts. On the first snapshot it sells them at option bid
and sets each component position to

```text
x_i = N * component_delta_i
```

so the declared model delta of the short option is offset. On later pre-expiry snapshots it
reprices with a deterministic per-snapshot seed and trades to the new delta targets. Component buys
execute at ask and sells at bid. Explicit component and opening-option costs apply in basis points
to executed notional.

Between snapshots, prior cash accrues at the prior continuous rate. Each prior component position
receives the prior declared continuous dividend yield using the prior spot over the elapsed
ACT/365F interval. This is an explicit discrete-ledger convention, not a reconstruction of actual
dividend events.

At expiry, VolForge:

1. recomputes the terminal payoff from the declared deliverable and final marks;
2. debits that payoff for the short option;
3. liquidates every component at its executable bid or ask; and
4. verifies that final portfolio value equals cash.

At every trade, the marked portfolio identity is

```text
V_after = V_before - option_spread_cost - component_spread_cost - explicit_cost
```

within the configured accounting tolerance. The ledger keeps option cash flow, component trade
cash flow, financing, dividends, spread loss, explicit cost, turnover, cash, marked value, and
accounting gap separate.

## Gates and audit chain

The replay blocks at the first snapshot with:

- future-received, out-of-order, or stale market data;
- duplicate source identities or changing risk factors;
- missing/extra component execution quotes or spots outside their spread;
- invalid or non-PSD market inputs;
- a basket price that fails its discounted bounds gate;
- an expiry quote inconsistent with explicit settlement;
- non-finite ledger state or a self-financing accounting gap.

Each accepted record includes canonical market and execution hashes plus a normalized snapshot
hash. Its audit hash binds all pricing, component trades, cash accounting, the prior audit hash,
and a configuration hash covering the deliverable identity, contract terms, expiry, and complete
replay settings. When present, store snapshot and manifest hashes are included in the normalized
input and audit hashes. `verify_basket_replay_hash_chain` recomputes the configuration and every
link; changing result metadata, a trade, provenance identity, or ordering breaks verification.
This is tamper evidence, not a digital signature, trusted timestamp, or proof of source identity.
When the estimate columns are supplied, both are required and must bind the same formation, market,
store snapshot, and manifests. `require_estimated_inputs=True` rejects older unestimated rows.

## Package API

```python
from volforge import (
    BasketReplaySettings,
    generate_synthetic_basket_replay,
    parse_deliverable_terms,
    run_basket_hedge_replay,
    verify_basket_replay_hash_chain,
)
from volforge.basket_replay import (
    SYNTHETIC_BASKET_DELIVERABLE,
    SYNTHETIC_BASKET_EXPIRY,
)

terms = parse_deliverable_terms(
    SYNTHETIC_BASKET_DELIVERABLE,
    settlement_method="physical",
    contract_multiplier=100,
    currency="USD",
    underlying_id="ACME",
)
result = run_basket_hedge_replay(
    generate_synthetic_basket_replay(),
    terms,
    option_type="call",
    strike=60,
    expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
    settings=BasketReplaySettings(paths=20_000, seed=31),
)
assert result.downstream_eligible
assert verify_basket_replay_hash_chain(result)
```

## CLI and workspace

Generate the deterministic fixture and matching deliverable:

```bash
volforge generate-basket-replay data/basket-replay.csv data/basket-deliverable.json
```

Run it and retain both the flat ledger and nested audit:

```bash
volforge basket-replay \
  data/basket-replay.csv data/basket-deliverable.json reports/basket-ledger.csv \
  --settlement-method physical --option-type call --strike 60 \
  --contract-multiplier 100 --currency USD --underlying-id ACME \
  --expiry 2026-04-30T20:00:00+00:00 --paths 20000 --seed 31 \
  --report reports/basket-replay.json
```

The command exits `0` only for a complete, verified ledger and `2` for a blocked prefix. The Basket
hedge replay Streamlit workspace calls the same package API, batches controls in a form, bounds its
cache, and exposes the P&L path, snapshot ledger, component trades, and downloadable artifacts.

For the v1.6 append-only ingestion, revision, adapter, and materialization workflow, see
[`basket-input-store.md`](basket-input-store.md). Add `--require-store-provenance` to the replay
command when unbound CSV inputs must be rejected.
For the v1.7 statistical estimator and explicit nonlinear-risk hedge ledger, see
[`basket-risk-hedging.md`](basket-risk-hedging.md). Add `--require-estimated-inputs` to keep using
this delta-only ledger while requiring the stronger estimated-market provenance.
The v1.8 covariance-selection columns are enforced and audit-bound by the advanced risk ledger,
not this delta-only ledger; see
[`basket-covariance-validation.md`](basket-covariance-validation.md).

## Interpretation limits

- The included fixture is deterministic synthetic data, not historical performance evidence.
- Supplied volatilities and correlations are used as-is; VolForge does not estimate, smooth, or
  prove their point-in-time availability in this CSV workflow.
- The hedge is component-delta only. Gamma, vega, correlation, cross-gamma, and model-parameter
  exposures remain unhedged.
- Continuous-yield carry is a declared approximation. Discrete dividends, borrow, FX, margin,
  taxes, funding asymmetry, latency, market impact, depth, and fill probability are outside the
  ledger.
- Passing establishes integrity for this finite input and convention. It does not establish model
  adequacy, source authenticity, statistical significance, or permission to trade.
