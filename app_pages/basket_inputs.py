"""Append-only basket-input storage workspace."""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from volforge.basket_inputs import (
    SYNTHETIC_BASKET_ID,
    BasketInputStore,
    generate_synthetic_basket_inputs,
)
from volforge.basket_replay import (
    SYNTHETIC_BASKET_DELIVERABLE,
    SYNTHETIC_BASKET_EXPIRY,
    BasketReplaySettings,
    basket_replay_report_json,
    run_basket_hedge_replay,
)
from volforge.deliverables import parse_deliverable_terms
from volforge.reporting import dataframe_to_csv


@st.cache_data(max_entries=4, show_spinner=False)
def _run_synthetic_store(maximum_age: str) -> dict[str, object]:
    records, schedule = generate_synthetic_basket_inputs()
    with tempfile.TemporaryDirectory(prefix="volforge-basket-input-") as temporary:
        store = BasketInputStore.create(Path(temporary) / "basket-inputs.sqlite")
        batch = store.ingest_frame(
            records,
            source_file_name="basket-inputs.csv",
            source="volforge-synthetic-v1",
            mode="initial",
            ingested_timestamp="2026-05-01T00:00:00Z",
        )
        verification = store.verify()
        replay_input = store.materialize_replay(
            schedule,
            basket_id=SYNTHETIC_BASKET_ID,
            source="volforge-synthetic-v1",
            maximum_age=maximum_age,
        )
        active_records = store.records_as_known(
            SYNTHETIC_BASKET_EXPIRY,
            basket_id=SYNTHETIC_BASKET_ID,
            source="volforge-synthetic-v1",
        )
    terms = parse_deliverable_terms(
        SYNTHETIC_BASKET_DELIVERABLE,
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="ACME",
    )
    replay = run_basket_hedge_replay(
        replay_input,
        terms,
        option_type="call",
        strike=60,
        expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
        settings=BasketReplaySettings(paths=4_000, require_store_provenance=True),
    )
    return {
        "batch": asdict(batch),
        "verification": asdict(verification),
        "records": active_records,
        "replay_input": replay_input,
        "ledger": replay.ledger_frame(),
        "replay": replay.to_dict(),
        "records_csv": dataframe_to_csv(records),
        "schedule_csv": dataframe_to_csv(schedule),
        "replay_input_csv": dataframe_to_csv(replay_input),
        "replay_report": basket_replay_report_json(replay),
    }


st.title("Basket input store")
st.caption(
    "Resolve effective point-in-time component markets and pairwise correlations from an "
    "append-only database before the basket hedge replay."
)

with st.form("basket_input_store_form"):
    maximum_age = st.selectbox(
        "Maximum market age",
        ("5min", "15min", "30min"),
        index=1,
        help="Each resolved package must be no older than this at portfolio formation.",
    )
    submitted = st.form_submit_button(
        "Build verified replay input",
        type="primary",
        icon=":material/database:",
        key="basket_input_run",
    )

result_slot = st.container()

st.warning(
    "Append-only manifests prove retained bytes and revision history, not that a named vendor "
    "issued them. The HTTPS adapter verifies TLS, bounds responses, and keeps bearer tokens out "
    "of metadata; source entitlement and identity still belong to the operator."
)

if submitted:
    try:
        with result_slot.skeleton():
            result = _run_synthetic_store(maximum_age)
    except (ArithmeticError, OSError, RuntimeError, ValueError) as exc:
        result_slot.error(f"Basket-input gate blocked: {exc}")
    else:
        verification = result["verification"]
        replay = result["replay"]
        assert isinstance(verification, dict)
        assert isinstance(replay, dict)
        with result_slot:
            with st.container(horizontal=True):
                st.metric(
                    "Store integrity",
                    "Passed" if verification["valid"] else "Blocked",
                )
                st.metric("Batches", int(verification["batch_count"]))
                st.metric("Canonical records", int(verification["record_count"]))
                st.metric("Resolved snapshots", int(replay["input_snapshot_count"]))
                st.metric(
                    "Replay provenance",
                    "Bound" if replay["store_provenance_bound"] else "Missing",
                )
            if verification["valid"] and replay["downstream_eligible"]:
                st.success(
                    "Every replay row came from a complete, timely snapshot in a verified "
                    "append-only manifest chain."
                )
            else:
                reasons = [*verification["failure_reasons"], *replay["failure_reasons"]]
                st.error("The storage/replay gate blocked: " + "; ".join(reasons))

            active_records = result["records"]
            replay_input = result["replay_input"]
            ledger = result["ledger"]
            assert isinstance(active_records, pd.DataFrame)
            assert isinstance(replay_input, pd.DataFrame)
            assert isinstance(ledger, pd.DataFrame)
            st.subheader("Resolved canonical records")
            display_records = active_records[
                [
                    "record_type",
                    "snapshot_id",
                    "market_timestamp",
                    "received_timestamp",
                    "revision",
                    "asset_id",
                    "paired_asset_id",
                    "spot",
                    "volatility",
                    "bid",
                    "ask",
                    "correlation",
                ]
            ]
            st.dataframe(
                display_records,
                hide_index=True,
                key="basket_input_records",
                alt="Canonical component and correlation records known by expiry",
                column_config={
                    "market_timestamp": st.column_config.DatetimeColumn(
                        "Market time", format="iso8601"
                    ),
                    "received_timestamp": st.column_config.DatetimeColumn(
                        "Received time", format="iso8601"
                    ),
                    "spot": st.column_config.NumberColumn(format="%.4f"),
                    "volatility": st.column_config.NumberColumn(format="%.4f"),
                    "bid": st.column_config.NumberColumn(format="%.4f"),
                    "ask": st.column_config.NumberColumn(format="%.4f"),
                    "correlation": st.column_config.NumberColumn(format="%.4f"),
                },
            )
            st.subheader("Provenance-bound replay input")
            st.dataframe(
                replay_input[
                    [
                        "formation_timestamp",
                        "market_timestamp",
                        "received_timestamp",
                        "source_record_id",
                        "input_store_snapshot_sha256",
                    ]
                ],
                hide_index=True,
                key="basket_input_materialized",
                alt="Replay schedule resolved from verified basket-input snapshots",
            )
            st.subheader("Integrated replay ledger")
            st.line_chart(
                ledger,
                x="formation_timestamp",
                y="portfolio_value",
                alt="Portfolio value across the provenance-bound basket replay",
            )
            with st.container(horizontal=True):
                st.download_button(
                    "Canonical records",
                    data=result["records_csv"],
                    file_name="volforge-basket-inputs.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
                st.download_button(
                    "Replay schedule",
                    data=result["schedule_csv"],
                    file_name="volforge-basket-schedule.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
                st.download_button(
                    "Materialized replay",
                    data=result["replay_input_csv"],
                    file_name="volforge-basket-replay-input.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
                st.download_button(
                    "Replay audit",
                    data=result["replay_report"],
                    file_name="volforge-basket-replay-audit.json",
                    mime="application/json",
                    icon=":material/download:",
                )

st.caption(
    "The demo is deterministic and offline. Production use should provide a controlled database "
    "path, explicit ingestion timestamps, credential environment variables, and retained raw "
    "provider responses."
)
