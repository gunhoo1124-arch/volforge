# Historical point-in-time data

VolForge v0.9 introduces a local, vendor-neutral history layer for option quotes and trades. Its
purpose is to preserve what happened in the market, when a source delivered it, when VolForge
stored it, and how later corrections changed the dataset—without overwriting an earlier version.

This boundary is necessary for credible empirical options work. Duarte, Jones, Khorram, Mo, and
Wang show that using information unavailable at portfolio formation to filter option observations
can introduce material look-ahead bias
([Review of Financial Studies](https://academic.oup.com/rfs/advance-article-abstract/doi/10.1093/rfs/hhag061/8722307)).
VolForge therefore applies knowledge cutoffs before revision selection and never substitutes a
later correction into an earlier market view.

## Three timestamp axes

Every event preserves three distinct UTC timestamps:

| Timestamp | Meaning | Used for |
|---|---|---|
| `event_timestamp` | When the quote or trade occurred in market time | Event ordering and snapshot age |
| `received_timestamp` | When the upstream source delivered or published it | Market-information cutoff |
| `ingested_timestamp` | When a VolForge batch appended it | Dataset-version cutoff |

`events_as_known` first keeps rows satisfying both the receipt and ingestion cutoffs. It then
chooses the highest eligible revision for each `(source, source_record_id)` and removes records
whose latest eligible action is `cancel`. Event-range filtering is applied after revision
selection, so a correction cannot accidentally expose the replaced revision.

For replay-ready chains, `chain_snapshot` fixes the receipt cutoff to the requested formation time,
selects only quote events at or before that time, enforces a maximum quote age, and retains the
latest eligible event for each listed contract. It emits the v0.8 chain schema with a common
`quote_timestamp` plus source-event, receipt, ingestion, revision, batch, age, and hash metadata.

## Normalized input schema

Every CSV row requires:

```text
record_type,event_timestamp,received_timestamp,source_record_id,revision,record_action,
underlying_symbol,expiration,strike,option_type
```

Common fields:

| Field | Rule |
|---|---|
| `record_type` | `quote` or `trade` |
| `source_record_id` | Non-empty identifier stable across revisions of the same source event |
| `revision` | Non-negative integer that increases strictly for that source record |
| `record_action` | `upsert` or `cancel` |
| `expiration` | UTC timestamp strictly after the event |
| `strike` | Positive finite number |
| `option_type` | `call` or `put` |

An `upsert` quote also requires `bid`, `ask`, `underlying_spot`, `risk_free_rate`, and
`dividend_yield`; it must satisfy `0 <= bid <= ask` and positive spot. Optional quote fields include
`bid_size` and `ask_size`.

An `upsert` trade requires non-negative `trade_price` and positive `trade_size`. Optional metadata
includes `contract_symbol`, `trade_condition`, `exchange`, and `source_sequence`. Extra vendor
columns remain covered by the per-record source payload and raw-file hashes even though they are
not promoted into normalized database columns.

This schema is intentionally independent of a specific feed. Actual exchange feeds define
product, session, sequence, and message semantics that an adapter must map explicitly; Cboe, for
example, publishes separate options market-data and symbology specifications
([Cboe technical specifications](https://www.cboe.com/markets/us/options/support/technical/)).

## Ingestion modes

Each atomic batch declares one mode:

| Mode | Invariant |
|---|---|
| `initial` | The source has no existing history |
| `incremental` | All event times are at or beyond the source watermark and record IDs are new |
| `backfill` | Older, previously absent events may be appended; a reason is mandatory |
| `correction` | Every record ID already exists, revisions increase, and record type stays fixed; a reason is mandatory |

An incremental batch whose earliest event precedes the source watermark is rejected with an
instruction to use `backfill`. An existing source record cannot be changed by another initial,
incremental, or backfill batch; it requires explicit correction mode. A cancellation requires a
currently active prior upsert, so repeated cancellations are rejected. Batch ingestion timestamps
cannot move backward relative to the store.

These checks make historical changes visible. They do not decide whether a business justification
is correct; the mandatory reason is an auditable declaration.

## Append-only and provenance controls

The SQLite store installs triggers that reject `UPDATE` and `DELETE` against market events and
ingestion batches. Every successful CSV ingestion records:

- exact raw bytes in a content-addressed `.raw` directory;
- raw filename, byte length, and SHA-256;
- canonical per-record payload JSON and SHA-256;
- quote/trade counts and event-time bounds;
- ingestion source, mode, timestamp, and reason;
- the previous batch-manifest hash and the current manifest hash.

`history-verify` checks SQLite integrity, every manifest link and hash, batch row counts, archived
file sizes and hashes, and every event payload hash.

These are tamper-evident local controls, not cryptographic authentication. A privileged attacker
who can replace the database, raw archive, and all manifests can construct another internally
consistent store. Long-term research releases should publish final manifest hashes through a
separate trusted channel or signature process.

## Package API

```python
from volforge.history import HistoricalEventStore

store = HistoricalEventStore.create("data/options-history.sqlite")
batch = store.ingest_csv(
    "data/vendor-quotes.csv",
    source="vendor-name",
    mode="initial",
    ingested_timestamp="2026-10-08T14:00:00Z",
)

market_view = store.events_as_known(
    "quote",
    received_at="2025-01-02T16:00:00Z",
    ingested_at="2026-10-08T14:00:00Z",
    source="vendor-name",
)

replay_input = store.replay_chain(
    ["2025-01-02T15:30:00Z", "2025-01-03T15:30:00Z"],
    source="vendor-name",
    maximum_quote_age="15min",
)
assert store.verify().valid
```

`received_at` and `ingested_at` answer different questions. The former reconstructs information
the source had delivered by a market formation time. The latter reconstructs a particular local
dataset version. Omitting `ingested_at` uses the latest stored version; omitting `received_at` from
the low-level query admits every received revision in that dataset version. Replay snapshots never
omit the receipt cutoff—it is fixed to each snapshot time.

## CLI

```bash
volforge history-init data/options-history.sqlite
volforge history-ingest data/options-history.sqlite data/vendor-quotes.csv \
  --source vendor-name --mode initial --ingested-at 2026-10-08T14:00:00Z
volforge history-query data/options-history.sqlite reports/quotes.csv \
  --record-type quote --received-at 2025-01-02T16:00:00Z --source vendor-name
volforge history-snapshot data/options-history.sqlite reports/replay.csv \
  --snapshot-at 2025-01-02T15:30:00Z 2025-01-03T15:30:00Z \
  --source vendor-name --maximum-quote-age 15min
volforge history-verify data/options-history.sqlite --report reports/history-integrity.json
```

## Reference boundary

- No licensed data or vendor downloader is bundled.
- The store trusts the supplied source name, source record identifiers, timestamps, and revision
  semantics after schema validation.
- Contract symbols remain immutable event metadata inside this database. v0.9.1 resolves them
  without mutation through a separate `ReferenceDataStore`.
- Multi-snapshot `replay_chain` requires that reference store and fails closed on unresolved terms,
  missing corporate actions, or invalid sessions. See
  [`reference-data.md`](reference-data.md).
- SQLite is appropriate for an auditable local research store; large tick histories will require a
  partitioned analytical representation without weakening these provenance rules.
- Raw archives may contain licensed or confidential data and are not encrypted automatically.
