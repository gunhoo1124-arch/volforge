import pytest

from volforge.hedging import HedgeBacktestSettings, run_heston_hedge_backtest
from volforge.models.heston import HestonParameters


@pytest.fixture(scope="module")
def hedge_result():  # type: ignore[no-untyped-def]
    return run_heston_hedge_backtest(
        "call",
        100.0,
        100.0,
        1.0,
        HestonParameters(1.5, 0.04, 0.4, -0.7, 0.04),
        0.03,
        settings=HedgeBacktestSettings(paths=20, rebalancing_steps=6, seed=11),
    )


def test_heston_hedge_backtest_is_reproducible_and_reconciles(hedge_result) -> None:  # type: ignore[no-untyped-def]
    repeated = run_heston_hedge_backtest(
        "call",
        100.0,
        100.0,
        1.0,
        HestonParameters(1.5, 0.04, 0.4, -0.7, 0.04),
        0.03,
        settings=HedgeBacktestSettings(paths=20, rebalancing_steps=6, seed=11),
    )
    assert repeated == hedge_result
    assert all(result.max_abs_attribution_gap < 1e-9 for result in hedge_result.strategies)
    assert all(
        result.mean_attribution.total == pytest.approx(result.mean_pnl, abs=1e-11)
        for result in hedge_result.strategies
    )


def test_delta_hedging_reduces_sample_pnl_dispersion(hedge_result) -> None:  # type: ignore[no-untyped-def]
    strategies = {result.strategy: result for result in hedge_result.strategies}
    assert (
        strategies["delta"].pnl_standard_deviation < strategies["unhedged"].pnl_standard_deviation
    )
    assert len(strategies["delta"].pnl) == hedge_result.settings.paths
    assert strategies["delta_gamma_vega"].max_condition_number is not None


def test_transaction_costs_reduce_pnl_on_common_paths() -> None:
    parameters = HestonParameters(1.5, 0.04, 0.4, -0.7, 0.04)
    common = dict(paths=10, rebalancing_steps=4, seed=7)
    costless = run_heston_hedge_backtest(
        "put",
        100.0,
        100.0,
        0.75,
        parameters,
        settings=HedgeBacktestSettings(**common, stock_cost_bps=0.0, option_cost_bps=0.0),
    )
    costly = run_heston_hedge_backtest(
        "put",
        100.0,
        100.0,
        0.75,
        parameters,
        settings=HedgeBacktestSettings(**common, stock_cost_bps=2.0, option_cost_bps=10.0),
    )
    for free, with_costs in zip(costless.strategies, costly.strategies, strict=True):
        assert with_costs.mean_pnl <= free.mean_pnl
        assert with_costs.mean_transaction_cost >= 0.0


@pytest.mark.parametrize(
    "settings",
    [
        HedgeBacktestSettings,
        lambda: HedgeBacktestSettings(paths=3),
        lambda: HedgeBacktestSettings(rebalancing_steps=0),
        lambda: HedgeBacktestSettings(hedge_strike_low=1.0),
        lambda: HedgeBacktestSettings(max_option_position=0.0),
        lambda: HedgeBacktestSettings(condition_limit=1.0),
    ],
)
def test_hedge_settings_validation(settings) -> None:  # type: ignore[no-untyped-def]
    if settings is HedgeBacktestSettings:
        assert settings().paths == 100
    else:
        with pytest.raises(ValueError):
            settings()
