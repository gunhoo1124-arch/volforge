from dataclasses import replace

import pandas as pd
import pytest

from volforge.benchmark import (
    BenchmarkSettings,
    benchmark_report_json,
    model_comparison_inference,
    run_chronological_benchmark,
    verify_benchmark_hash_chain,
)
from volforge.data import generate_synthetic_replay


@pytest.fixture(scope="module")
def benchmark_chain() -> pd.DataFrame:
    return generate_synthetic_replay()


@pytest.fixture(scope="module")
def benchmark_result(benchmark_chain: pd.DataFrame):  # type: ignore[no-untyped-def]
    return run_chronological_benchmark(
        benchmark_chain,
        settings=BenchmarkSettings(
            training_window=2,
            heston_max_function_evaluations=60,
            cos_terms=64,
        ),
    )


def _paired_loss_frame(differences: tuple[float, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sequence, difference in enumerate(differences):
        bsm_loss = 3.0 + sequence * 0.1
        common = {
            "evaluation_sequence": sequence,
            "evaluation_timestamp": f"2025-02-{sequence + 2:02d}T15:30:00+00:00",
            "expiration": "2025-06-20T20:00:00+00:00",
            "strike": 100.0,
            "option_type": "call",
        }
        rows.extend(
            [
                {**common, "model": "bsm", "squared_error": bsm_loss},
                {
                    **common,
                    "model": "heston",
                    "squared_error": bsm_loss + difference,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_rolling_benchmark_is_strictly_out_of_sample_and_audited(benchmark_result) -> None:  # type: ignore[no-untyped-def]
    assert benchmark_result.completed
    assert benchmark_result.downstream_eligible
    assert benchmark_result.evaluation_snapshot_count == 2
    assert verify_benchmark_hash_chain(benchmark_result)
    assert [step.action for step in benchmark_result.steps] == [
        "fit_only",
        "fit_only",
        "score_and_fit",
        "score",
    ]
    assert all(step.no_lookahead_passed for step in benchmark_result.steps)
    predictions = benchmark_result.prediction_frame()
    assert set(predictions["model"]) == {"bsm", "heston"}
    assert (
        pd.to_datetime(predictions["training_end_timestamp"], utc=True)
        < pd.to_datetime(predictions["evaluation_timestamp"], utc=True)
    ).all()
    summary = benchmark_result.summary_frame().set_index("model")
    assert summary.loc["heston", "rmse"] < summary.loc["bsm", "rmse"]
    assert summary.loc["heston", "quote_coverage"] > summary.loc["bsm", "quote_coverage"]
    assert b'"hash_chain_verified": true' in benchmark_report_json(benchmark_result)


def test_benchmark_panels_cover_maturity_and_moneyness(benchmark_result) -> None:  # type: ignore[no-untyped-def]
    panel = benchmark_result.panel_frame()
    assert set(panel["model"]) == {"bsm", "heston"}
    assert set(panel["maturity_bucket"]) == {"short (<60d)", "medium (60-180d)", "long (>=180d)"}
    assert {"low strike", "near forward", "high strike"}.issubset(set(panel["moneyness_bucket"]))
    assert (panel["quote_count"] > 0).all()


def test_benchmark_prediction_mutation_breaks_hash_chain(benchmark_result) -> None:  # type: ignore[no-untyped-def]
    first = benchmark_result.predictions[0]
    changed = replace(first, model_price=first.model_price + 0.01)
    tampered = replace(
        benchmark_result,
        predictions=(changed, *benchmark_result.predictions[1:]),
    )
    assert not verify_benchmark_hash_chain(tampered)
    extra = replace(first, evaluation_sequence=999)
    assert not verify_benchmark_hash_chain(
        replace(benchmark_result, predictions=(*benchmark_result.predictions, extra))
    )


def test_invalid_training_snapshot_blocks_before_heston(benchmark_chain: pd.DataFrame) -> None:
    invalid = benchmark_chain.copy()
    row_index = invalid.index[0]
    invalid.loc[row_index, "bid"] = float(invalid.loc[row_index, "ask"]) + 1.0
    result = run_chronological_benchmark(
        invalid,
        settings=BenchmarkSettings(heston_max_function_evaluations=40, cos_terms=64),
    )
    assert not result.completed
    assert not result.downstream_eligible
    assert result.steps[-1].action == "blocked"
    assert not result.steps[-1].all_quotes_valid
    assert result.steps[-1].surface_downstream_eligible is None
    assert result.steps[-1].heston_calibration_eligible is None
    assert verify_benchmark_hash_chain(result)


def test_reference_failure_and_short_history_are_rejected(
    benchmark_chain: pd.DataFrame,
) -> None:
    invalid = benchmark_chain.copy()
    invalid["reference_validated"] = True
    invalid["stable_contract_id"] = (
        invalid["expiration"].astype(str)
        + "-"
        + invalid["strike"].astype(str)
        + "-"
        + invalid["option_type"]
    )
    invalid.loc[invalid.index[0], "reference_validated"] = False
    with pytest.raises(ValueError, match="failed contract/session reference validation"):
        run_chronological_benchmark(invalid)
    first_two = benchmark_chain.loc[
        benchmark_chain["quote_timestamp"].isin(
            sorted(benchmark_chain["quote_timestamp"].unique())[:2]
        )
    ]
    with pytest.raises(ValueError, match="more snapshots"):
        run_chronological_benchmark(first_two, settings=BenchmarkSettings(training_window=2))


def test_date_clustered_block_bootstrap_detects_consistent_heston_improvement() -> None:
    rows: list[dict[str, object]] = []
    for sequence in range(8):
        for strike in (95.0, 105.0):
            common = {
                "evaluation_sequence": sequence,
                "evaluation_timestamp": f"2025-01-{sequence + 2:02d}T15:30:00+00:00",
                "expiration": "2025-06-20T20:00:00+00:00",
                "strike": strike,
                "option_type": "call",
            }
            rows.extend(
                [
                    {**common, "model": "bsm", "squared_error": 4.0 + sequence * 0.1},
                    {**common, "model": "heston", "squared_error": 1.0 + sequence * 0.1},
                ]
            )
    inference = model_comparison_inference(
        pd.DataFrame(rows),
        settings=BenchmarkSettings(bootstrap_resamples=500, bootstrap_seed=11),
    )
    assert inference.available
    assert inference.evaluation_dates == 8
    assert inference.paired_quote_rows == 16
    assert inference.mean_loss_difference == pytest.approx(-3.0)
    assert inference.confidence_interval_high is not None
    assert inference.confidence_interval_high < 0.0
    assert inference.preferred_model == "heston"
    assert inference.conclusive
    assert not inference.dm_available
    assert "variance" in str(inference.dm_failure_reason)
    assert inference.heston_win_dates == 8
    assert inference.bsm_win_dates == 0
    assert inference.tie_dates == 0
    assert inference.sign_test_available
    assert inference.sign_test_p_value == pytest.approx(0.0078125)
    assert inference.sign_test_rejects_equal_win_rate


def test_hac_diebold_mariano_and_directional_tests_use_date_level_losses() -> None:
    differences = (-0.8, -0.2, -1.1, -0.4, -0.9, -0.1, -1.2, -0.3, -0.7, -0.5)
    inference = model_comparison_inference(
        _paired_loss_frame(differences),
        settings=BenchmarkSettings(
            bootstrap_resamples=500,
            bootstrap_seed=17,
            comparison_hac_lags=2,
        ),
    )

    assert inference.available
    assert inference.hac_lags == 2
    assert inference.dm_available
    assert inference.dm_statistic == pytest.approx(-9.6229783883)
    assert inference.dm_p_value == pytest.approx(4.922515104e-06)
    assert inference.dm_rejects_equal_accuracy
    assert inference.heston_win_dates == 10
    assert inference.bsm_win_dates == 0
    assert inference.sign_test_p_value == pytest.approx(0.001953125)
    assert inference.sign_test_rejects_equal_win_rate


def test_date_sign_test_withholds_small_non_tied_sample() -> None:
    frame = _paired_loss_frame((-1.0, 1.0, 0.0, 0.0, 0.0, 0.0))
    inference = model_comparison_inference(
        frame,
        settings=BenchmarkSettings(bootstrap_resamples=200, comparison_hac_lags=0),
    )

    assert inference.available
    assert (inference.heston_win_dates, inference.bsm_win_dates, inference.tie_dates) == (1, 1, 4)
    assert not inference.sign_test_available
    assert "non-tied" in str(inference.sign_test_failure_reason)
    with pytest.raises(ValueError, match="comparison_hac_lags"):
        model_comparison_inference(
            frame,
            settings=BenchmarkSettings(bootstrap_resamples=200, comparison_hac_lags=6),
        )


def test_comparison_hac_lags_are_validated() -> None:
    with pytest.raises(ValueError, match="comparison_hac_lags"):
        BenchmarkSettings(comparison_hac_lags=-1)
    with pytest.raises(ValueError, match="comparison_hac_lags"):
        BenchmarkSettings(comparison_hac_lags=True)  # type: ignore[arg-type]


def test_comparison_uncertainty_is_withheld_for_too_few_dates(benchmark_result) -> None:  # type: ignore[no-untyped-def]
    inference = benchmark_result.comparison_inference()
    assert not inference.available
    assert inference.evaluation_dates == 2
    assert "requires at least 5" in str(inference.failure_reason)
    assert not inference.dm_available
    assert not inference.sign_test_available
