# Point-in-time calibration replay

VolForge v0.8 turns a multi-date option-chain file into a strictly chronological sequence of
surface fits, Heston calibrations, listed hedge trades, and observed-mark P&L records. The engine
is deliberately fail-closed: the first incomplete or invalid day terminates the replay and no
later observation is used to repair it.

VolForge v0.9.1 adds a reference-gated path into this engine. Multi-snapshot chains constructed
from `HistoricalEventStore` must first resolve stable contracts, effective terms, product rules,
corporate actions, and open exchange sessions through `ReferenceDataStore`. Direct synthetic and
uploaded research fixtures remain supported as explicitly unreferenced inputs.

VolForge v0.10 replaces midpoint trade fills with immediately executable quote sides. Option buys
fill at ask and sells fill at bid, while open positions remain marked at midpoint. The audit
separates spread loss from the existing proportional transaction-cost overlay.

VolForge v1.0 applies contract multipliers and validates structured deliverables before surface
calibration. Standard single-security deliverables enter every option mark, fill, spread cost,
fee notional, turnover amount, and per-contract hedge sensitivity. Multi-asset, cash, cash-in-lieu,
or otherwise adjusted deliverables fail closed because the current Heston ledger cannot value them.

VolForge v1.3 adds a separate terminal settlement engine for adjusted physical packages and
cash-settled contracts. Replay now shares its canonical deliverable parser and hash, but still
blocks those structures before calibration: explicit terminal component marks are not a
multi-asset pre-expiry pricing or hedging model. See
[adjusted deliverables](adjusted-deliverables.md).

VolForge v1.4 adds a standalone correlated-GBM pre-expiry engine for those explicit packages. Its
static user inputs are intentionally not admitted here: this single-factor replay does not consume
component markets, correlations, dividend/borrow inputs, or a complete multi-asset hedge ledger.
See [basket pricing](basket-pricing.md).

VolForge v1.5 adds a separate point-in-time basket hedge replay with received-by-formation market
packages, component bid/ask execution, delta rebalancing, and expiry settlement. It does not alter
this SVI/SSVI-Heston ledger or use its one-factor surface calibration for adjusted packages. See
[basket hedge replay](basket-replay.md).

VolForge v1.6 adds an [append-only basket-input store](basket-input-store.md) before that separate
basket replay. It resolves complete component/correlation packages under receipt and ingestion
cutoffs and binds store identities into the basket audit; it does not weaken or bypass this
single-factor surface/Heston gate.

This separation matters empirically. Duarte, Jones, Khorram, Mo, and Wang document that filtering
option observations with information unavailable at portfolio formation creates look-ahead bias
([Review of Financial Studies](https://academic.oup.com/rfs/advance-article-abstract/doi/10.1093/rfs/hhag061/8722307)).
Bakshi, Cao, and Chen evaluate model prices with parameters inferred from the previous day's option
prices, illustrating the distinct roles of in-sample fit, out-of-sample pricing, and hedging
([Journal of Finance](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1997.tb02749.x)).
The v0.8 fixture is synthetic, so passing its gate establishes implementation integrity rather
than a historical performance claim.

## Information boundary

Input rows are partitioned by their parsed UTC `quote_timestamp`. For snapshot `t_i`, validation,
surface preparation, SVI/SSVI calibration, Heston calibration, contract marking, and hedge sizing
receive only rows whose timestamp is exactly `t_i`. The audit records both the minimum and maximum
source timestamp, and `no_lookahead_passed` requires both to equal the valuation timestamp.

The engine processes timestamps in increasing order. It never fits a full-sample parameter,
interpolates from a later snapshot, backfills a missing quote, or selects a replacement contract
based on future availability.

When a `reference_validated` column is present, every row must be true and the input must carry
`stable_contract_id`, `contract_multiplier`, and `deliverable_json`; otherwise replay rejects the
file before surface preparation. Historical multi-snapshot construction itself cannot emit a
replay chain without a reference store.

## Daily calibration gate

For every snapshot:

1. All raw rows must pass schema, quote, duplicate, no-arbitrage-bound, staleness, and implied-
   volatility validation.
2. Valid calls and puts are consolidated into forward log-moneyness `k = log(K/F)` and total
   implied variance `w = sigma_BS^2 T`.
3. Constrained raw-SVI slices or joint power-law SSVI are fitted and must pass optimizer,
   butterfly, calendar, and analytic-condition diagnostics. The SVI/SSVI construction follows the
   static-arbitrage framework of Gatheral and Jacquier
   ([arXiv](https://arxiv.org/abs/1204.0646)).
4. Heston is calibrated only after the surface gate passes. Optimizer and final implied-volatility
   inversion checks must mark the result calibration-eligible.
5. The target and both fixed hedge contracts must each have exactly one valid two-sided quote.

Any failure creates a final `blocked` audit row and stops processing.

## Listed-contract policy

Automatic selection sees only the first snapshot. It chooses the nearest expiry that contains the
requested target type and at least three listed calls, then selects the strike nearest spot. The
hedges are the nearest listed calls strictly below and above the target strike at the same expiry.

Those three contract identities are frozen for the run. Reference-gated inputs also carry their
stable master identities across source-symbol changes. A later missing, duplicate, invalid, or
unresolved mark blocks the replay. The policy does not search future dates for a contract with
better data and does not roll an expiring position. A caller can instead provide an explicit target
contract; the bracketing hedge calls are still selected from the first snapshot.

For reference-enriched contracts, stable identity is the primary lookup key and strike, expiry,
and option type must remain consistent with the selected pricing contract. The shared parser
canonicalizes physical and cash-settled component terms, but replay eligibility requires exactly
one `security` component whose quantity equals the positive contract multiplier; when
`underlying_id` is present, the component must name that underlying.
OCC product material describes a standard equity/ETF weekly-option unit as 100 shares, while OCC's
adjustment rules permit event-specific changes to the unit, exercise price, underlying, or other
terms ([OCC weekly options](https://www.theocc.com/clearance-and-settlement/clearing/weekly-options);
[OCC Rules, Chapter XXVIII](https://www.theocc.com/getContentAsset/9d3854cd-b782-450f-bcf7-33169b0576ce/dfc3d011-8f63-43f6-9ed8-4b444333a1d0/occ_rules.pdf%3B)).
VolForge therefore consumes explicit reference terms and refuses adjusted structures it cannot
price rather than assuming every listed option represents 100 ordinary shares. Unreferenced
fixtures retain backward-compatible per-unit economics with multiplier one.

## Hedge sizing and observed-mark accounting

Daily calibrated Heston parameters generate per-underlying-unit COS price, delta, gamma,
instantaneous-volatility vega, and calendar theta. Multiplying each sensitivity by its contract
multiplier gives the per-contract exposure. The selected policy solves for stock shares and option
contracts that offset the short target's model sensitivities. The two-by-two gamma/vega system is
scaled before its condition number is measured, uses a bounded least-squares fallback when needed,
and applies the configured option-position cap in contracts.

Model prices never enter the cash ledger. Let `S_i` be observed spot, `E_i(Delta h)` select ask for
a positive option trade and bid for a negative option trade, `x_i` be stock, `h_i` hedge-option
positions, `m_j` their multipliers, and `c_i` cash after trading. A trade from old positions to new
positions applies

```text
c_i = c_i^- - Delta x_i S_i - Delta h_i dot (m_h E_i(Delta h_i))
      - Delta z_i m_z E_i(Delta z_i) - explicit_cost_i
```

where `z=-1` while the target is short. Between snapshots, prior cash earns the prior continuous
rate and prior stock earns the prior continuous dividend yield. Proportional costs apply to stock
notional and executed option premium notional, including opening and closing the target. The final
snapshot closes the target, both hedge options, and stock at the observed executable sides.

Before and after every trade, the portfolio identity is checked:

```text
V_after = V_before - spread_cost - explicit_transaction_cost
```

within the configured tolerance. `spread_cost` is the signed-fill-to-midpoint loss, which equals
absolute option quantity times half the quoted spread for each trade. The first post-trade value is
therefore the negative opening spread and explicit costs; after final liquidation, portfolio value
equals cash and terminal P&L. Stock still executes at the single supplied spot because the chain
schema has no stock bid/ask pair.

## Tamper-evident audit chain

Each raw snapshot is normalized into sorted columns, normalized scalar values, and row-order-
independent canonical JSON, then hashed with SHA-256. Each daily audit hash binds:

- the previous audit hash;
- the current snapshot hash and timestamp boundary;
- surface and Heston eligibility plus Heston parameters;
- target/hedge multipliers and canonical deliverable hashes;
- selected marks, positions, and trades;
- carry, dividends, costs, turnover, cash, and P&L;
- accounting, conditioning, clipping, fallback, and failure diagnostics.

`verify_replay_hash_chain` recomputes every link and detects changed audit content or broken
ordering. This is tamper evidence inside a deterministic artifact, not a digital signature,
trusted timestamp, append-only database, or proof of source-data provenance.

## Package, CLI, and app

```python
from volforge.data import generate_synthetic_replay
from volforge.replay import ReplaySettings, run_chronological_replay

result = run_chronological_replay(
    generate_synthetic_replay(),
    settings=ReplaySettings(strategy="delta_gamma_vega", cos_terms=64),
)
assert result.downstream_eligible
```

```bash
volforge generate-replay data/replay.csv
volforge replay data/replay.csv reports/ledger.csv --report reports/replay.json
```

The Streamlit point-in-time workspace exposes the same package API for the synthetic fixture or an
uploaded CSV. Expensive replay runs are form-submitted and bounded-cache backed; the results show
gate status, chronological P&L, the daily audit table, and downloadable JSON/CSV artifacts.

## Interpretation limits

- The bundled fixture is a deterministic SSVI demonstration, not market history.
- Direct uploaded replay CSV timestamps are trusted as supplied. The history/reference workflow
  separately models vendor receipt, revisions, cancellations, effective terms, and sessions.
- Standard single-security multipliers and deliverables are applied. The standalone settlement
  engine values explicit terminal payoffs, while the separate basket pricing/replay path handles
  declared multi-asset GBM inputs. Adjusted and cash-settled contracts remain blocked from this
  Heston replay. Margin, stock borrow, stock bid/ask execution, latency, fill probability, depth,
  and impact remain outside this ledger.
- A daily Heston fit uses a time-homogeneous parameter set over that snapshot's surface; parameter
  uncertainty is not propagated into hedge P&L.
- Passing the replay gate proves that the finite replay obeyed its data, calibration, accounting, and hash
  gates. It does not establish out-of-sample superiority or permission to trade.
