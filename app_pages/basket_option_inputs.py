"""Independent basket-option quote storage workspace."""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from volforge.basket_covariance import materialize_selected_covariance_replay
from volforge.basket_estimation import generate_synthetic_smile_basket_inputs
from volforge.basket_inputs import SYNTHETIC_BASKET_ID, BasketInputStore
from volforge.basket_option_inputs import (
    SYNTHETIC_TARGET_INSTRUMENT_ID,
    BasketOptionQuoteStore,
    basket_option_quote_report_json,
    generate_synthetic_option_quote_inputs,
)
from volforge.basket_replay import SYNTHETIC_BASKET_DELIVERABLE, SYNTHETIC_BASKET_EXPIRY
from volforge.basket_risk_hedging import BasketRiskHedgeSettings, run_basket_risk_hedge_replay
from volforge.deliverables import parse_deliverable_terms
from volforge.reporting import dataframe_to_csv

COMPONENT_SOURCE = "volforge-synthetic-statistical-v1"
OPTION_SOURCE = "volforge-synthetic-option-quotes-v1"


@st.cache_data(max_entries=4, show_spinner=False)
def _run_option_quote_workspace(maximum_age: str, paths: int) -> dict[str, object]:
    quote_records, quote_schedule = generate_synthetic_option_quote_inputs()
    component_records, _ = generate_synthetic_smile_basket_inputs()
    hedge_ids = tuple(
        quote_records.loc[quote_records["role"] == "hedge", "instrument_id"].drop_duplicates()
    )
    with tempfile.TemporaryDirectory(prefix="volforge-option-quotes-") as temporary:
        root = Path(temporary)
        quote_store = BasketOptionQuoteStore.create(root / "option-quotes.sqlite")
        quote_batch = quote_store.ingest_frame(
            quote_records,
            source_file_name="basket-option-quotes.csv",
            source=OPTION_SOURCE,
            mode="initial",
            ingested_timestamp="2026-05-01T00:00:00Z",
        )
        verification = quote_store.verify()
        quote_bound_schedule = quote_store.materialize_schedule(
            quote_schedule,
            basket_id=SYNTHETIC_BASKET_ID,
            source=OPTION_SOURCE,
            target_instrument_id=SYNTHETIC_TARGET_INSTRUMENT_ID,
            hedge_instrument_ids=hedge_ids,
            maximum_age=maximum_age,
        )
        active_quotes = quote_store.records_as_known(
            SYNTHETIC_BASKET_EXPIRY,
            basket_id=SYNTHETIC_BASKET_ID,
            source=OPTION_SOURCE,
        )
        component_store = BasketInputStore.create(root / "basket-inputs.sqlite")
        component_store.ingest_frame(
            component_records,
            source_file_name="statistical-basket-inputs.csv",
            source=COMPONENT_SOURCE,
            mode="initial",
            ingested_timestamp="2026-05-01T00:00:00Z",
        )
        replay_input = materialize_selected_covariance_replay(
            component_store,
            quote_bound_schedule,
            basket_id=SYNTHETIC_BASKET_ID,
            source=COMPONENT_SOURCE,
        )
    terms = parse_deliverable_terms(
        SYNTHETIC_BASKET_DELIVERABLE,
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="ACME",
    )
    replay = run_basket_risk_hedge_replay(
        replay_input,
        terms,
        option_type="call",
        strike=60,
        expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
        settings=BasketRiskHedgeSettings(
            paths=paths,
            include_smile_risks=True,
            require_covariance_selection=True,
            require_option_quote_provenance=True,
        ),
    )
    summary = quote_bound_schedule[
        [
            "formation_timestamp",
            "option_bid",
            "option_ask",
            "input_option_quote_snapshot_sha256",
        ]
    ].copy()
    summary["option_mid"] = 0.5 * (summary["option_bid"] + summary["option_ask"])
    return {
        "batch": asdict(quote_batch),
        "verification": asdict(verification),
        "quotes": active_quotes,
        "summary": summary,
        "schedule": quote_bound_schedule,
        "replay": replay.to_dict(),
        "records_csv": dataframe_to_csv(quote_records),
        "schedule_csv": dataframe_to_csv(quote_bound_schedule),
        "verification_report": basket_option_quote_report_json(verification),
    }


st.title("Basket option quote store")
st.caption(
    "Resolve target and smile-hedge option markets from their own append-only, point-in-time "
    "history before combining them with independently estimated component covariance inputs."
)

with st.form("basket_option_quote_form"):
    with st.container(horizontal=True):
        maximum_age = st.selectbox(
            "Maximum quote age",
            ("5min", "15min", "30min"),
            index=1,
            help="Every requested target and hedge quote must meet this formation-time limit.",
        )
        paths = st.selectbox("Monte Carlo paths", (2_000, 4_000), index=0)
    submitted = st.form_submit_button(
        "Build quote-bound replay",
        type="primary",
        icon=":material/price_check:",
        key="basket_option_input_run",
    )

result_slot = st.container()
st.warning(
    "The synthetic quotes are model-derived fixtures. Append-only hashes prove retained bytes, "
    "revision history, and replay binding; they do not authenticate a supplier or prove that a "
    "displayed market was executable at the requested size."
)

if submitted:
    try:
        with result_slot.skeleton():
            workspace = _run_option_quote_workspace(maximum_age, paths)
    except (ArithmeticError, OSError, RuntimeError, ValueError) as exc:
        result_slot.error(f"Option-quote gate blocked: {exc}")
    else:
        verification = workspace["verification"]
        replay = workspace["replay"]
        quotes = workspace["quotes"]
        summary = workspace["summary"]
        schedule = workspace["schedule"]
        assert isinstance(verification, dict)
        assert isinstance(replay, dict)
        assert isinstance(quotes, pd.DataFrame)
        assert isinstance(summary, pd.DataFrame)
        assert isinstance(schedule, pd.DataFrame)
        with result_slot:
            with st.container(horizontal=True):
                st.metric(
                    "Quote-store integrity",
                    "Passed" if verification["valid"] else "Blocked",
                    border=True,
                )
                st.metric("Quote records", int(verification["record_count"]), border=True)
                st.metric(
                    "Option-quote provenance",
                    "Bound" if replay["option_quote_provenance_bound"] else "Missing",
                    border=True,
                )
                st.metric(
                    "Integrated replay",
                    "Passed" if replay["downstream_eligible"] else "Blocked",
                    border=True,
                )
            if verification["valid"] and replay["downstream_eligible"]:
                st.success(
                    "Every target and hedge quote was selected as known at formation, linked to "
                    "a verified raw manifest, and bound into the hash-chained five-risk replay."
                )
            else:
                reasons = [*verification["failure_reasons"], *replay["failure_reasons"]]
                st.error("Quote/replay gate blocked: " + "; ".join(reasons))

            st.subheader("Resolved option market history")
            st.line_chart(
                summary,
                x="formation_timestamp",
                y="option_mid",
                alt="Target basket-option mid quote across formation times",
            )
            st.dataframe(
                quotes[
                    [
                        "market_timestamp",
                        "received_timestamp",
                        "instrument_id",
                        "role",
                        "bid",
                        "ask",
                        "revision",
                        "manifest_hash",
                    ]
                ],
                hide_index=True,
                key="basket_option_quote_records",
                alt="Active target and hedge-option quote records known by expiry",
            )

            st.subheader("Formation-time provenance")
            st.dataframe(
                schedule[
                    [
                        "formation_timestamp",
                        "option_bid",
                        "option_ask",
                        "input_option_quote_snapshot_sha256",
                        "input_option_quote_manifest_hashes_json",
                    ]
                ],
                hide_index=True,
                key="basket_option_quote_snapshots",
                alt="Point-in-time option quote snapshot hashes used by replay",
            )

            with st.container(horizontal=True):
                st.download_button(
                    "Canonical quote history",
                    data=workspace["records_csv"],
                    file_name="volforge-basket-option-quotes.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
                st.download_button(
                    "Quote-bound schedule",
                    data=workspace["schedule_csv"],
                    file_name="volforge-basket-option-schedule.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
                st.download_button(
                    "Store verification",
                    data=workspace["verification_report"],
                    file_name="volforge-basket-option-verification.json",
                    mime="application/json",
                    icon=":material/download:",
                )

st.caption(
    "Production research should ingest retained provider artifacts under controlled source "
    "identities, timestamps, entitlements, and instrument master governance."
)
