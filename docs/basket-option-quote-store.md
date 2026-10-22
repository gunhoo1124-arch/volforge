# Basket option quote store

VolForge v1.11 separates executable target and hedge-option markets from the statistical component
history. The advanced replay can therefore prove which independently ingested quote revision was
known at each formation instead of trusting bid/ask values embedded in a schedule JSON field.

## Canonical record contract

Every CSV row supplies exactly these fields:

| Field | Meaning |
|---|---|
| `market_timestamp` | UTC-aware time represented by the quote |
| `received_timestamp` | UTC-aware time the source quote became observable |
| `source_record_id`, `revision`, `record_action` | stable source event, monotone revision, and `upsert`/`cancel` |
| `basket_id`, `instrument_id`, `role` | basket, immutable instrument key, and `target`/`hedge` role |
| `option_type`, `strike`, `expiry_timestamp` | listed-option definition |
| `bid`, `ask` | executable-side evidence with `0 <= bid <= ask` |
| `settlement_method`, `contract_multiplier`, `currency`, `underlying_id` | contract economics |
| `deliverable_json` | strict, canonicalizable deliverable components |

Ingestion adds a third clock, `ingested_timestamp`, in the batch manifest. A record is eligible only
when its market and receipt times are no later than formation and its ingestion time is no later
than the optional dataset-version cutoff.

## Append-only rules

- `initial` starts an empty source; `incremental` accepts only new identities at non-backfill market
  times; `backfill` requires a reason; `correction` requires existing identities and a reason.
- New identities start with revision zero and an upsert. Corrections increase revisions strictly,
  never move receipt time backward, and cannot alter basket, instrument, role, market time, or any
  contract field.
- An `instrument_id` has one contract definition within a source and basket. A changed strike,
  expiry, deliverable, multiplier, or role needs a new identifier.
- SQLite triggers reject update/delete operations. Each exact CSV artifact is retained under its
  SHA-256 address; normalized records have payload hashes and batch manifests form a hash chain.
- Whole-store verification checks SQLite, record payload/column agreement, raw size/hash and exact
  archive membership, schema version, and every manifest predecessor link.

These controls detect ordinary local tampering. They do not authenticate the named source, prove
entitlement or trusted time, or make a displayed quote executable at a particular size.

## Point-in-time snapshot

`BasketOptionQuoteStore.resolve_snapshot` requires one target instrument and an ordered, unique
hedge-instrument list. It selects the newest active as-known quote for each identity, rejects stale
or missing instruments and role mismatches, and emits:

- the complete target contract and bid/ask;
- canonical `hedge_instruments_json` accepted by the risk replay;
- selected source-record and payload hashes;
- contributing batch and manifest identities; and
- a `snapshot_sha256` over the complete payload.

`materialize_schedule` accepts only formation time and risk-free rate, then adds target bid/ask,
hedge JSON, snapshot JSON/hash, and manifest-hash JSON. The component estimation and covariance
selection materializers preserve those columns unchanged.

## Downstream gate

Set `BasketRiskHedgeSettings.require_option_quote_provenance=True` or pass
`--require-option-quote-provenance`. For every replay row VolForge then verifies the snapshot
self-hash and manifest identities and requires exact binding to:

- formation timestamp;
- target type, strike, expiry, settlement method, multiplier, currency, underlying, deliverable,
  bid, and ask; and
- the ordered hedge contract/quote payload and declared quote count.

The accepted option snapshot hash enters the replay input hash and hash-chained audit. Missing or
partial columns fail schema validation before processing; mismatched or tampered payloads stop the
replay at the first affected snapshot. The result reports `option_quote_provenance_bound` only when
every processed snapshot carries valid lineage.

## CLI workflow

The commands are:

- `generate-basket-option-inputs`: deterministic records plus a formation/rate schedule;
- `basket-option-init`: create a new store and archive directory;
- `basket-option-ingest`: append a canonical CSV under an explicit source/mode/time;
- `basket-option-materialize`: resolve target and repeated `--hedge-instrument-id` arguments; and
- `basket-option-verify`: emit the store integrity audit and return status 2 when invalid.

The generated fixture is model-derived and intended only for deterministic testing. Production
research should add controlled provider-specific adapters, instrument-master mapping, entitlement
checks, and independently retained source evidence without weakening the canonical validation or
as-known cutoff rules.
