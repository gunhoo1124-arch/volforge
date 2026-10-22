from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_all_streamlit_workspaces_smoke() -> None:
    app = Path(__file__).parents[1] / "streamlit_app.py"
    at = AppTest.from_file(app, default_timeout=20).run()
    assert not at.exception
    assert at.title[0].value == "Pricing calculator"

    pages = [
        ("app_pages/implied_volatility.py", "Implied-volatility solver"),
        ("app_pages/chain_explorer.py", "Option-chain explorer"),
        ("app_pages/data_quality.py", "Data-quality diagnostics"),
        ("app_pages/volatility_surfaces.py", "Volatility surfaces"),
        ("app_pages/heston_calibration.py", "Heston calibration"),
        ("app_pages/numerical_engines.py", "Numerical engines"),
        ("app_pages/american_options.py", "American options"),
        ("app_pages/heston_greeks.py", "Heston Greeks"),
        ("app_pages/hedging.py", "Hedging laboratory"),
        ("app_pages/hedge_robustness.py", "Hedge robustness"),
        ("app_pages/point_in_time_replay.py", "Point-in-time replay"),
        ("app_pages/deliverable_valuation.py", "Deliverable valuation"),
        ("app_pages/basket_pricing.py", "Basket pricing"),
        ("app_pages/basket_replay.py", "Basket hedge replay"),
        ("app_pages/basket_inputs.py", "Basket input store"),
        ("app_pages/basket_option_inputs.py", "Basket option quote store"),
        ("app_pages/basket_covariance.py", "Basket covariance validation"),
        ("app_pages/basket_risk_hedging.py", "Basket risk hedging"),
        ("app_pages/basket_smile_hedging.py", "Basket smile hedging"),
        ("app_pages/model_benchmark.py", "Model benchmark"),
    ]
    for path, title in pages:
        at.switch_page(path).run(timeout=20)
        assert not at.exception
        assert at.title[0].value == title


def test_numerical_convergence_gate_ui() -> None:
    app = Path(__file__).parents[1] / "streamlit_app.py"
    at = AppTest.from_file(app, default_timeout=30).run()
    at.switch_page("app_pages/numerical_engines.py").run(timeout=30)
    at.session_state["engine_run_convergence_gate"] = True
    at.button(key="engine_run_cross_check").click().run(timeout=30)
    assert not at.exception
    assert any("eligible for downstream" in alert.value for alert in at.success)
    assert any(metric.label == "Gate status" and metric.value == "Passed" for metric in at.metric)


def test_hedge_robustness_gate_ui() -> None:
    app = Path(__file__).parents[1] / "streamlit_app.py"
    at = AppTest.from_file(app, default_timeout=30).run()
    at.switch_page("app_pages/hedge_robustness.py").run(timeout=30)
    assert not at.exception
    assert any(metric.label == "Gate status" and metric.value == "Passed" for metric in at.metric)
    assert any("passed refinement" in alert.value for alert in at.success)


def test_point_in_time_replay_ui() -> None:
    app = Path(__file__).parents[1] / "streamlit_app.py"
    at = AppTest.from_file(app, default_timeout=30).run()
    at.switch_page("app_pages/point_in_time_replay.py").run(timeout=30)
    at.button(key="replay_run").click().run(timeout=30)
    assert not at.exception
    assert any(metric.label == "Replay gate" and metric.value == "Passed" for metric in at.metric)
    assert any("Every snapshot passed" in alert.value for alert in at.success)


def test_deliverable_valuation_ui() -> None:
    app = Path(__file__).parents[1] / "streamlit_app.py"
    at = AppTest.from_file(app, default_timeout=20).run()
    at.switch_page("app_pages/deliverable_valuation.py").run(timeout=20)
    at.button(key="deliverable_value").click().run(timeout=20)

    assert not at.exception
    assert any(
        metric.label == "Classification" and metric.value == "Adjusted Physical"
        for metric in at.metric
    )
    assert any(
        metric.label == "Net payoff" and metric.value == "220.00 USD" for metric in at.metric
    )
    assert any("ineligible" in alert.value for alert in at.warning)


def test_basket_pricing_ui() -> None:
    app = Path(__file__).parents[1] / "streamlit_app.py"
    at = AppTest.from_file(app, default_timeout=30).run()
    at.switch_page("app_pages/basket_pricing.py").run(timeout=30)
    at.button(key="basket_run").click().run(timeout=30)

    assert not at.exception
    assert any(metric.label == "Bounds gate" and metric.value == "Passed" for metric in at.metric)
    assert any(metric.label == "Monte Carlo price" for metric in at.metric)
    assert any("not admitted" in alert.value for alert in at.warning)


def test_basket_replay_ui() -> None:
    app = Path(__file__).parents[1] / "streamlit_app.py"
    at = AppTest.from_file(app, default_timeout=30).run()
    at.switch_page("app_pages/basket_replay.py").run(timeout=30)
    at.button(key="basket_replay_run").click().run(timeout=30)

    assert not at.exception
    assert any(metric.label == "Replay gate" and metric.value == "Passed" for metric in at.metric)
    assert any("Every supplied snapshot passed" in alert.value for alert in at.success)
    assert any("do not authenticate" in alert.value for alert in at.warning)


def test_basket_input_store_ui() -> None:
    app = Path(__file__).parents[1] / "streamlit_app.py"
    at = AppTest.from_file(app, default_timeout=30).run()
    at.switch_page("app_pages/basket_inputs.py").run(timeout=30)
    at.button(key="basket_input_run").click().run(timeout=30)

    assert not at.exception
    assert any(
        metric.label == "Store integrity" and metric.value == "Passed" for metric in at.metric
    )
    assert any(
        metric.label == "Replay provenance" and metric.value == "Bound" for metric in at.metric
    )
    assert any("verified append-only" in alert.value for alert in at.success)


def test_basket_option_quote_store_ui() -> None:
    app = Path(__file__).parents[1] / "streamlit_app.py"
    at = AppTest.from_file(app, default_timeout=60).run()
    at.switch_page("app_pages/basket_option_inputs.py").run(timeout=60)
    at.button(key="basket_option_input_run").click().run(timeout=60)

    assert not at.exception
    assert any(
        metric.label == "Quote-store integrity" and metric.value == "Passed" for metric in at.metric
    )
    assert any(
        metric.label == "Option-quote provenance" and metric.value == "Bound"
        for metric in at.metric
    )
    assert any("known at formation" in alert.value for alert in at.success)


def test_basket_risk_hedging_ui() -> None:
    app = Path(__file__).parents[1] / "streamlit_app.py"
    at = AppTest.from_file(app, default_timeout=60).run()
    at.switch_page("app_pages/basket_risk_hedging.py").run(timeout=60)
    at.button(key="basket_risk_run").click().run(timeout=60)

    assert not at.exception
    assert any(metric.label == "Replay gate" and metric.value == "Passed" for metric in at.metric)
    assert any(
        metric.label == "Estimated inputs" and metric.value == "Bound" for metric in at.metric
    )
    assert any("nonlinear solver" in alert.value for alert in at.success)


def test_basket_covariance_validation_ui() -> None:
    app = Path(__file__).parents[1] / "streamlit_app.py"
    at = AppTest.from_file(app, default_timeout=30).run()
    at.switch_page("app_pages/basket_covariance.py").run(timeout=30)
    at.button(key="basket_covariance_run").click().run(timeout=30)

    assert not at.exception
    assert any(
        metric.label == "Selection gate" and metric.value == "Passed" for metric in at.metric
    )
    assert any(metric.label == "Robust gate" and metric.value == "Passed" for metric in at.metric)
    assert any(
        metric.label == "Selected estimator" and metric.value == "sample_shrinkage"
        for metric in at.metric
    )
    assert any("stress-resilience" in alert.value for alert in at.success)


def test_basket_smile_hedging_ui() -> None:
    app = Path(__file__).parents[1] / "streamlit_app.py"
    at = AppTest.from_file(app, default_timeout=60).run()
    at.switch_page("app_pages/basket_smile_hedging.py").run(timeout=60)
    at.button(key="basket_smile_run").click().run(timeout=60)

    assert not at.exception
    assert any(metric.label == "Replay gate" and metric.value == "Passed" for metric in at.metric)
    assert any(
        metric.label == "Smile-risk coverage" and metric.value == "Bound" for metric in at.metric
    )
    assert any("five declared nonlinear risks" in alert.value for alert in at.success)


def test_model_benchmark_ui() -> None:
    app = Path(__file__).parents[1] / "streamlit_app.py"
    at = AppTest.from_file(app, default_timeout=30).run()
    at.switch_page("app_pages/model_benchmark.py").run(timeout=30)
    at.button(key="benchmark_run").click().run(timeout=30)
    assert not at.exception
    assert any(
        metric.label == "Benchmark gate" and metric.value == "Passed" for metric in at.metric
    )
    assert any("only earlier fits" in alert.value for alert in at.success)
