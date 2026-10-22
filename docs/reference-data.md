# Point-in-time contract and market references

VolForge v0.9.1 adds a reference boundary between append-only market events and chronological
research. A quote symbol is only a source identifier; it is not assumed to be a permanent contract
identity. The reference store resolves that symbol to a stable contract, its effective terms, the
applicable product rules, and an explicit exchange session without modifying the v0.9 event store.

This distinction follows the structure of real listed-option data. Cboe maps venue symbol IDs to
OSI symbols and recommends using published reference data instead of constructing identifiers from
contract attributes
([Cboe symbology reference](https://www.cboe.com/document/tech-spec/document/technical-specifications/cboe-titanium-u.s.-equitiesoptionsfutures-symbology-reference)).
Trading hours also vary by venue, product, session, holiday, and early close
([Cboe hours and holidays](https://www.cboe.com/about/hours/us-options/)). VolForge therefore
ingests explicit UTC session intervals rather than inferring that every weekday is open.

## Separate append-only store

`ReferenceDataStore` uses its own schema-versioned SQLite database and content-addressed raw
archive. Existing v0.9 history databases need no migration. Reference records preserve:

- an effective interval or timestamp describing when a fact applies;
- a source publication timestamp describing when the fact became available;
- a VolForge ingestion timestamp describing when it entered the local dataset;
- append-only revisions and cancellations;
- exact raw CSV bytes, record payload hashes, and a global batch-manifest hash chain.

Knowledge cutoffs are applied before revision and effective-time selection. A contract correction
published after a formation snapshot cannot replace the version available at that snapshot, even
when its declared effective date is earlier.

## Four reference record types

Every input row begins with:

```text
source_record_id,revision,record_action,published_timestamp
```

`record_action` is `upsert` or `cancel`. Revisions increase strictly. A correction cannot change the
business identity of its source record, and cancellation requires an active prior upsert.

### Products

Products contain effective-dated rules shared by a family of contracts:

```text
effective_from,effective_to,product_id,product_symbol,exchange,calendar_id,
underlying_asset_type,exercise_style,settlement_method,settlement_style,currency,
default_multiplier,expiration_rule,last_trade_rule,settlement_rule,status,source_reference
```

Exercise style is `american` or `european`; settlement method is `physical` or `cash`; settlement
style is `am`, `pm`, or `intraday`. The three rule fields are explicit source descriptions, while
each contract stores the exact instantiated timestamps used by replay. This avoids guessing from a
root symbol. For example, Cboe documents cash-settled, European SPX options with both A.M. and P.M.
expirations ([Cboe SPX product information](https://www.cboe.com/tradable-products/sp-500/spx-options/)).

### Contracts

Contracts map a source symbol to a stable economic identity and exact terms:

```text
effective_from,effective_to,contract_id,market_data_source,source_contract_symbol,
product_id,underlying_id,underlying_symbol,option_type,strike,expiration_timestamp,
last_trade_timestamp,settlement_timestamp,contract_multiplier,deliverable_json,status,
corporate_action_id,source_reference
```

`deliverable_json` is a non-empty list. Each component has `asset_id`, `asset_type`, and positive
`quantity`, with optional currency and description. Supported typed records include `security`,
`cash`, `cash_in_lieu`, `cash_settlement`, and source-retained `other` components. Contract terms
can receive a new effective row while retaining the same `contract_id`, allowing a symbol or
deliverable adjustment to preserve position continuity.

### Exchange sessions

Session inputs contain:

```text
calendar_id,session_date,session_name,status,timezone,open_timestamp,close_timestamp,
reason,source_reference
```

Open sessions require exact UTC boundaries and a valid IANA timezone. Closed sessions omit the
boundaries and carry a reason such as a holiday. Multiple named intervals can represent regular,
global, or curb trading. A timestamp matching zero or multiple open intervals is rejected.

### Corporate actions

Corporate actions contain:

```text
corporate_action_id,underlying_id,action_type,effective_timestamp,old_symbol,new_symbol,
description,source_reference
```

Supported categories include splits, reverse splits, mergers, spin-offs, special dividends,
symbol changes, and delistings. A contract that names an action is ineligible until the action is
published and resolvable under the same knowledge cutoff. This is deliberately data-driven: OCC
may adjust the number of contracts, unit of trading, exercise price, underlying security, or a
combination, and many decisions are event-specific
([OCC Rules, Chapter XXVIII](https://www.theocc.com/getContentAsset/9d3854cd-b782-450f-bcf7-33169b0576ce/dfc3d011-8f63-43f6-9ed8-4b444333a1d0/occ_rules.pdf%3B)).

## Replay gate

`HistoricalEventStore.replay_chain` now requires a `ReferenceDataStore`. At each formation time it:

1. selects quotes using the existing receipt and dataset-ingestion cutoffs;
2. selects reference revisions published no later than the formation time;
3. resolves `(market_data_source, source_contract_symbol)` to one active contract;
4. verifies underlying, option type, strike, expiration, exchange, and last-trade time;
5. verifies both the quote event and formation time are in explicit open sessions;
6. resolves the product rules and any linked corporate action; and
7. emits stable identity, terms, session labels, and reference payload hashes.

Missing or ambiguous records, crossed effective boundaries, future publications, inactive or
expired contracts, closed days, and out-of-session timestamps raise `ReferenceIntegrityError`.
A single exploratory `chain_snapshot` may still be exported without a reference store, but it is
marked `reference_validated=False`; multi-snapshot replay cannot bypass the reference gate.

Reference enrichment carries `settlement_method` and the product `settlement_currency` alongside
`contract_multiplier` and `deliverable_json`. The v1.3 settlement engine can value explicit
multi-security, fixed-cash, resolved cash-in-lieu, and cash-settled terminal payoffs. It rejects
`other` components, missing security marks, unresolved FX, and missing official settlement levels.
The separate v1.4 basket engine can price those packages before expiry only when the caller also
supplies a complete, static component market and positive-semidefinite correlation matrix.

The replay ledger remains narrower: it supports exactly one security component whose quantity
equals the multiplier and, when known, whose identity matches `underlying_id`. That multiplier
scales premiums, marks, spread and fee notionals, turnover, and model hedge sensitivities.
Adjusted or cash-settled deliverables still stop replay before surface calibration; terminal
settlement accounting does not imply the current single-underlying Heston engine can price them
before expiry. The separate basket replay can consume explicit received-by-formation component
packages, but it does not change Heston-replay eligibility or authenticate its CSV source labels.
See [adjusted deliverables](adjusted-deliverables.md), [basket pricing](basket-pricing.md), and
[basket hedge replay](basket-replay.md).

## Python API

```python
from volforge.history import HistoricalEventStore
from volforge.reference import ReferenceDataStore

references = ReferenceDataStore.create("data/options-reference.sqlite")
references.ingest_csv(
    "data/products.csv",
    kind="product",
    source="reference-provider",
    mode="initial",
    ingested_timestamp="2026-10-13T14:00:00Z",
)

history = HistoricalEventStore("data/options-history.sqlite")
replay_input = history.replay_chain(
    ["2026-10-12T14:30:00Z", "2026-10-13T14:30:00Z"],
    source="market-feed",
    reference_store=references,
)
assert replay_input["reference_validated"].all()
```

In practice, ingest product, session, corporate-action, and contract files as separate batches.
The batch chain is global, while `initial`/`incremental` state is tracked per source and kind.

## CLI

```bash
volforge reference-init data/options-reference.sqlite
volforge reference-ingest data/options-reference.sqlite data/products.csv \
  --kind product --source reference-provider --mode initial \
  --ingested-at 2026-10-13T14:00:00Z
volforge reference-query data/options-reference.sqlite reports/products.csv \
  --kind product --effective-at 2026-10-13T14:30:00Z \
  --published-at 2026-10-13T14:30:00Z
volforge reference-verify data/options-reference.sqlite \
  --report reports/reference-integrity.json

volforge history-snapshot data/options-history.sqlite reports/replay.csv \
  --snapshot-at 2026-10-12T14:30:00Z 2026-10-13T14:30:00Z \
  --source market-feed --reference-store data/options-reference.sqlite
```

## Limits

- VolForge provides the normalized store and gate, not licensed OCC, OPRA, or exchange data.
- Source adapters must map each vendor's identifiers and revision semantics explicitly.
- Rules are stored and audited; VolForge does not yet generate future listings or expiration dates
  from natural-language rule descriptions.
- Terminal adjusted-deliverable valuation requires explicit component marks or an official
  settlement level. SVI/SSVI-Heston replay remains limited to standard physical deliverables; the
  separate basket replay requires complete point-in-time component market and execution packages.
- Venue halts and unscheduled intraday closures require explicit session revisions.
