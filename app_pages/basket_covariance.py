"""Chronological basket covariance model-selection workspace."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from volforge.basket_covariance import (
    BasketCovarianceSelectionSettings,
    basket_covariance_selection_report_json,
    materialize_selected_covariance_replay,
    select_basket_covariance,
)
from volforge.basket_estimation import generate_synthetic_statistical_basket_inputs
from volforge.basket_inputs import SYNTHETIC_BASKET_ID, BasketInputStore
from volforge.reporting import dataframe_to_csv

SOURCE = "volforge-synthetic-statistical-v1"


@st.cache_data(max_entries=4, show_spinner=False)
def _run_workspace(
    validation_policy: str,
    training_window: int,
    evaluation_returns: int,
    minimum_nll_improvement: float,
    student_t_degrees_of_freedom: float,
    stress_quantile: float,
    stress_weight: float,
) -> dict[str, object]:
    records, schedule = generate_synthetic_statistical_basket_inputs()
    robust = validation_policy == "Student-t regime-balanced"
    settings = BasketCovarianceSelectionSettings(
        training_window=training_window,
        minimum_training_returns=10,
        evaluation_returns=evaluation_returns,
        minimum_nll_improvement=minimum_nll_improvement,
        forecast_distribution="student_t" if robust else "gaussian",
        student_t_degrees_of_freedom=student_t_degrees_of_freedom,
        selection_objective="regime_balanced" if robust else "mean_nll",
        stress_quantile=stress_quantile,
        stress_weight=stress_weight,
    )
    with tempfile.TemporaryDirectory(prefix="volforge-covariance-") as temporary:
        store = BasketInputStore.create(Path(temporary) / "basket-inputs.sqlite")
        store.ingest_frame(
            records,
            source_file_name="statistical-basket-inputs.csv",
            source=SOURCE,
            mode="initial",
            ingested_timestamp="2026-05-01T00:00:00Z",
        )
        first = select_basket_covariance(
            store,
            SYNTHETIC_BASKET_ID,
            schedule.iloc[0]["formation_timestamp"],
            source=SOURCE,
            settings=settings,
        )
        replay = materialize_selected_covariance_replay(
            store,
            schedule,
            basket_id=SYNTHETIC_BASKET_ID,
            source=SOURCE,
            settings=settings,
        )
    selections = [json.loads(value) for value in replay["input_covariance_selection_json"]]
    formation_summary = pd.DataFrame(
        [
            {
                "formation_timestamp": item["formation_timestamp"],
                "best_scoring_estimator": item["best_scoring_estimator"],
                "selected_estimator": item["selected_estimator"],
                "alternative_promotion_passed": item["alternative_promotion_passed"],
                "selection_score_improvement": item["selection_score_improvement"],
                "stress_evaluation_count": item["stress_evaluation_count"],
                "robust_validation_passed": item["robust_validation_passed"],
                "selected_stability_passed": item["selected_stability_passed"],
                "selection_sha256": item["selection_sha256"],
            }
            for item in selections
        ]
    )
    return {
        "selection": first.to_dict(),
        "candidates": first.candidate_frame(),
        "folds": first.fold_frame(),
        "formation_summary": formation_summary,
        "replay": replay,
        "report": basket_covariance_selection_report_json(first),
    }


st.title("Basket covariance validation")
st.caption(
    "Compare sample-correlation shrinkage, EWMA, and diagonal covariance shrinkage using "
    "strictly later held-out returns, with optional heavy-tail and regime-balanced validation, "
    "before admitting one estimator to replay."
)

with st.form("basket_covariance_form"):
    with st.container(horizontal=True):
        validation_policy = st.selectbox(
            "Validation policy",
            ("Student-t regime-balanced", "Gaussian mean NLL"),
            help="The robust policy balances calm and stress folds under a heavy-tailed score.",
        )
        training_window = st.selectbox(
            "Training returns",
            (15, 20, 25),
            index=1,
            help="Every forecast fit uses this many returns ending before its evaluation return.",
        )
        evaluation_returns = st.selectbox(
            "Held-out returns",
            (5, 10),
            index=1,
        )
    with st.container(horizontal=True):
        minimum_nll_improvement = st.selectbox(
            "Minimum NLL improvement",
            (0.0, 0.05, 0.10, 0.25),
            index=0,
            format_func=lambda value: f"{value:.2f}",
            help="An alternative must beat the baseline by at least this declared NLL score.",
        )
        student_t_degrees_of_freedom = st.selectbox(
            "Student-t degrees of freedom",
            (4.0, 6.0, 10.0),
            index=1,
        )
        stress_quantile = st.selectbox(
            "Training stress quantile",
            (0.70, 0.80, 0.90),
            index=1,
        )
        stress_weight = st.selectbox(
            "Stress-regime weight",
            (0.25, 0.50, 0.75),
            index=1,
        )
    submitted = st.form_submit_button(
        "Run covariance validation",
        type="primary",
        icon=":material/query_stats:",
        key="basket_covariance_run",
    )

result_slot = st.container()
st.warning(
    "Student-t scoring is a heavy-tail sensitivity model, not proof of a Student-t return law. "
    "Training-window stress labels and balanced losses do not establish future regime stability; "
    "jumps, liquidity, and structural breaks remain material."
)

if submitted:
    try:
        with result_slot.skeleton():
            workspace = _run_workspace(
                validation_policy,
                training_window,
                evaluation_returns,
                minimum_nll_improvement,
                student_t_degrees_of_freedom,
                stress_quantile,
                stress_weight,
            )
    except (ArithmeticError, OSError, RuntimeError, ValueError) as exc:
        result_slot.error(f"Covariance validation blocked: {exc}")
    else:
        selection = workspace["selection"]
        candidates = workspace["candidates"]
        folds = workspace["folds"]
        formation_summary = workspace["formation_summary"]
        assert isinstance(selection, dict)
        assert isinstance(candidates, pd.DataFrame)
        assert isinstance(folds, pd.DataFrame)
        assert isinstance(formation_summary, pd.DataFrame)
        with result_slot:
            with st.container(horizontal=True):
                st.metric(
                    "Selection gate",
                    "Passed" if selection["downstream_eligible"] else "Blocked",
                    border=True,
                )
                st.metric(
                    "Robust gate",
                    "Passed" if selection["robust_validation_passed"] else "Blocked",
                    border=True,
                )
                st.metric(
                    "Selected estimator",
                    str(selection["selected_estimator"]),
                    border=True,
                )
                st.metric(
                    "Stress folds",
                    f"{int(selection['stress_evaluation_count'])}/{evaluation_returns}",
                    border=True,
                )
                st.metric(
                    "Score improvement",
                    f"{float(selection['selection_score_improvement']):.4f}",
                    border=True,
                )
            if selection["downstream_eligible"]:
                st.success(
                    "The selected estimator passed chronological no-lookahead, PSD, condition, "
                    "regime-coverage, stress-resilience, and promotion gates; its audit is bound "
                    "into every replay row."
                )
            else:
                st.error("Selection blocked: " + "; ".join(selection["failure_reasons"]))

            st.subheader("Held-out candidate comparison")
            st.bar_chart(
                candidates,
                x="estimator",
                y="selection_score",
                alt="Held-out covariance selection score by estimator",
            )
            st.dataframe(
                candidates,
                hide_index=True,
                key="basket_covariance_candidates",
                alt="Covariance estimator losses, eigenvalue diagnostics, and stability gates",
                column_config={
                    "mean_negative_log_likelihood": st.column_config.NumberColumn(format="%.6f"),
                    "median_negative_log_likelihood": st.column_config.NumberColumn(format="%.6f"),
                    "calm_mean_negative_log_likelihood": st.column_config.NumberColumn(
                        format="%.6f"
                    ),
                    "stress_mean_negative_log_likelihood": st.column_config.NumberColumn(
                        format="%.6f"
                    ),
                    "worst_negative_log_likelihood": st.column_config.NumberColumn(format="%.6f"),
                    "selection_score": st.column_config.NumberColumn(format="%.6f"),
                    "minimum_raw_eigenvalue": st.column_config.NumberColumn(format="%.8f"),
                    "maximum_condition_number": st.column_config.NumberColumn(format="%.4f"),
                    "formation_raw_minimum_eigenvalue": st.column_config.NumberColumn(
                        format="%.8f"
                    ),
                    "formation_stabilized_minimum_eigenvalue": st.column_config.NumberColumn(
                        format="%.8f"
                    ),
                    "formation_condition_number": st.column_config.NumberColumn(format="%.4f"),
                },
            )
            st.subheader("Chronological forecast folds")
            st.dataframe(
                folds,
                hide_index=True,
                key="basket_covariance_folds",
                alt="Chronological covariance forecasts with calm and stress diagnostics",
                column_config={
                    "realized_return_energy": st.column_config.NumberColumn(format="%.4f"),
                    "stress_threshold": st.column_config.NumberColumn(format="%.4f"),
                    "gaussian_negative_log_likelihood": st.column_config.NumberColumn(
                        format="%.6f"
                    ),
                    "student_t_negative_log_likelihood": st.column_config.NumberColumn(
                        format="%.6f"
                    ),
                    "negative_log_likelihood": st.column_config.NumberColumn(format="%.6f"),
                },
            )
            st.subheader("Selections bound to the replay schedule")
            st.dataframe(
                formation_summary,
                hide_index=True,
                key="basket_covariance_formations",
                alt="Selected covariance estimator and audit hash at each replay formation",
            )
            with st.container(horizontal=True):
                st.download_button(
                    "Selected replay input",
                    data=dataframe_to_csv(workspace["replay"]),
                    file_name="volforge-covariance-selected-replay.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
                st.download_button(
                    "Selection audit",
                    data=workspace["report"],
                    file_name="volforge-covariance-selection.json",
                    mime="application/json",
                    icon=":material/download:",
                )

st.caption(
    "The deterministic fixture is offline. Production use still requires independently sourced, "
    "point-in-time component histories and explicit review of the selected risk model."
)
