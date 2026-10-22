import json
from pathlib import Path

import pandas as pd
import pytest

from volforge.cli import main
from volforge.research import verify_research_bundle


def test_price_cli_emits_json(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(
        [
            "price",
            "--option-type",
            "call",
            "--spot",
            "100",
            "--strike",
            "100",
            "--maturity",
            "1",
            "--volatility",
            "0.2",
            "--rate",
            "0.05",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["price"] == pytest.approx(10.4505835722)


def test_deliverable_value_cli_prices_adjusted_physical_contract(capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    deliverable = tmp_path / "deliverable.json"
    marks = tmp_path / "marks.json"
    deliverable.write_text(
        '[{"asset_id":"ACME","asset_type":"security","quantity":50},'
        '{"asset_id":"SPIN","asset_type":"security","quantity":10},'
        '{"asset_id":"USD-CIL","asset_type":"cash_in_lieu","quantity":20}]',
        encoding="utf-8",
    )
    marks.write_text('{"ACME":120,"SPIN":20}', encoding="utf-8")

    exit_code = main(
        [
            "deliverable-value",
            str(deliverable),
            "--marks",
            str(marks),
            "--settlement-method",
            "physical",
            "--option-type",
            "call",
            "--strike",
            "60",
            "--contract-multiplier",
            "100",
            "--currency",
            "USD",
            "--underlying-id",
            "ACME",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["classification"] == "adjusted_physical"
    assert payload["delivered_value"] == pytest.approx(6_220)
    assert payload["net_payoff"] == pytest.approx(220)
    assert payload["required_security_marks"] == ["ACME", "SPIN"]
    assert not payload["single_factor_pricing_compatible"]


def test_deliverable_value_cli_prices_cash_settlement(capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    deliverable = tmp_path / "cash-settlement.json"
    deliverable.write_text(
        '[{"asset_id":"SPX-SET","asset_type":"cash_settlement","quantity":100}]',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "deliverable-value",
            str(deliverable),
            "--settlement-method",
            "cash",
            "--option-type",
            "put",
            "--strike",
            "600",
            "--settlement-level",
            "595",
            "--contract-multiplier",
            "100",
            "--currency",
            "USD",
            "--underlying-id",
            "SPX",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["classification"] == "cash_settled"
    assert payload["net_payoff"] == pytest.approx(500)


def test_basket_price_cli_prices_adjusted_contract(capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    deliverable = tmp_path / "deliverable.json"
    market = tmp_path / "basket-market.json"
    deliverable.write_text(
        '[{"asset_id":"ACME","asset_type":"security","quantity":50},'
        '{"asset_id":"SPIN","asset_type":"security","quantity":10},'
        '{"asset_id":"USD-CIL","asset_type":"cash_in_lieu","quantity":20}]',
        encoding="utf-8",
    )
    market.write_text(
        '{"assets":['
        '{"asset_id":"ACME","spot":120,"volatility":0.25,"dividend_yield":0.01},'
        '{"asset_id":"SPIN","spot":20,"volatility":0.35,"dividend_yield":0.0}],'
        '"correlation":[[1,0.35],[0.35,1]]}',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "basket-price",
            str(deliverable),
            str(market),
            "--settlement-method",
            "physical",
            "--option-type",
            "call",
            "--strike",
            "60",
            "--contract-multiplier",
            "100",
            "--currency",
            "USD",
            "--underlying-id",
            "ACME",
            "--maturity",
            "0.5",
            "--rate",
            "0.03",
            "--paths",
            "20000",
            "--seed",
            "19",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["classification"] == "adjusted_physical"
    assert payload["engine"] == "correlated-gbm-monte-carlo"
    assert payload["price"] > 0
    assert payload["standard_error"] > 0
    assert payload["bounds_passed"]
    assert [delta["asset_id"] for delta in payload["deltas"]] == ["ACME", "SPIN"]


def test_generate_and_run_basket_replay_cli(capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    snapshots = tmp_path / "basket-replay.csv"
    deliverable = tmp_path / "deliverable.json"
    ledger = tmp_path / "basket-ledger.csv"
    report = tmp_path / "basket-audit.json"

    generated_exit = main(
        [
            "generate-basket-replay",
            str(snapshots),
            str(deliverable),
            "--paths",
            "4000",
            "--seed",
            "101",
        ]
    )
    generated = json.loads(capsys.readouterr().out)

    assert generated_exit == 0
    assert generated["rows"] == 4
    assert generated["expiry_timestamp"] == "2026-04-30T20:00:00+00:00"
    assert snapshots.exists() and deliverable.exists()

    replay_exit = main(
        [
            "basket-replay",
            str(snapshots),
            str(deliverable),
            str(ledger),
            "--report",
            str(report),
            "--settlement-method",
            "physical",
            "--option-type",
            "call",
            "--strike",
            "60",
            "--contract-multiplier",
            "100",
            "--currency",
            "USD",
            "--underlying-id",
            "ACME",
            "--expiry",
            "2026-04-30T20:00:00+00:00",
            "--paths",
            "4000",
        ]
    )
    replay = json.loads(capsys.readouterr().out)

    assert replay_exit == 0
    assert replay["completed"]
    assert replay["downstream_eligible"]
    assert len(pd.read_csv(ledger)) == 4
    assert json.loads(report.read_text(encoding="utf-8"))["final_audit_hash"]


def test_basket_input_store_cli_materializes_verified_replay(capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    records = tmp_path / "basket-inputs.csv"
    schedule = tmp_path / "basket-schedule.csv"
    store = tmp_path / "basket-inputs.sqlite"
    materialized = tmp_path / "materialized.csv"
    query = tmp_path / "known-records.csv"
    verification = tmp_path / "verification.json"
    deliverable = tmp_path / "deliverable.json"
    unused_replay = tmp_path / "unused-replay.csv"
    ledger = tmp_path / "ledger.csv"

    assert main(["generate-basket-inputs", str(records), str(schedule)]) == 0
    generated = json.loads(capsys.readouterr().out)
    assert generated["records"] == 12
    assert generated["snapshots"] == 4
    assert main(["basket-input-init", str(store)]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "basket-input-ingest",
                str(store),
                str(records),
                "--source",
                "volforge-synthetic-v1",
                "--mode",
                "initial",
                "--ingested-at",
                "2026-05-01T00:00:00Z",
            ]
        )
        == 0
    )
    batch = json.loads(capsys.readouterr().out)
    assert batch["row_count"] == 12
    assert (
        main(
            [
                "basket-input-query",
                str(store),
                str(query),
                "--formation-at",
                "2026-04-10T20:00:00Z",
                "--basket-id",
                "VOLFORGE-SYNTHETIC-ADJUSTED",
                "--source",
                "volforge-synthetic-v1",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["rows"] == 6
    assert (
        main(
            [
                "basket-input-materialize",
                str(store),
                str(schedule),
                str(materialized),
                "--basket-id",
                "VOLFORGE-SYNTHETIC-ADJUSTED",
                "--source",
                "volforge-synthetic-v1",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["store_provenance_bound"]
    assert (
        main(
            [
                "basket-input-verify",
                str(store),
                "--report",
                str(verification),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["valid"]
    assert main(["generate-basket-replay", str(unused_replay), str(deliverable)]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "basket-replay",
                str(materialized),
                str(deliverable),
                str(ledger),
                "--settlement-method",
                "physical",
                "--option-type",
                "call",
                "--strike",
                "60",
                "--contract-multiplier",
                "100",
                "--currency",
                "USD",
                "--underlying-id",
                "ACME",
                "--expiry",
                "2026-04-30T20:00:00+00:00",
                "--paths",
                "4000",
                "--require-store-provenance",
            ]
        )
        == 0
    )
    replay = json.loads(capsys.readouterr().out)
    assert replay["completed"]
    assert replay["store_provenance_bound"]


def test_basket_option_quote_store_cli_workflow(capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    records = tmp_path / "option-quotes.csv"
    schedule = tmp_path / "option-schedule.csv"
    store = tmp_path / "option-quotes.sqlite"
    materialized = tmp_path / "quote-bound-schedule.csv"
    verification = tmp_path / "option-verification.json"

    assert main(["generate-basket-option-inputs", str(records), str(schedule)]) == 0
    generated = json.loads(capsys.readouterr().out)
    assert generated["records"] == 24
    assert generated["formations"] == 4
    assert len(generated["hedge_instrument_ids"]) == 5

    assert main(["basket-option-init", str(store)]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == 1
    assert (
        main(
            [
                "basket-option-ingest",
                str(store),
                str(records),
                "--source",
                "volforge-synthetic-option-quotes-v1",
                "--mode",
                "initial",
                "--ingested-at",
                "2026-05-01T00:00:00Z",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["row_count"] == 24
    materialize_arguments = [
        "basket-option-materialize",
        str(store),
        str(schedule),
        str(materialized),
        "--basket-id",
        generated["basket_id"],
        "--source",
        "volforge-synthetic-option-quotes-v1",
        "--target-instrument-id",
        generated["target_instrument_id"],
    ]
    for instrument_id in generated["hedge_instrument_ids"]:
        materialize_arguments.extend(["--hedge-instrument-id", instrument_id])
    assert main(materialize_arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["rows"] == 4
    assert result["option_quote_provenance_bound"]
    assert "input_option_quote_snapshot_json" in pd.read_csv(materialized)

    assert main(["basket-option-verify", str(store), "--report", str(verification)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["valid"]
    assert json.loads(verification.read_text(encoding="utf-8")) == verified


def test_statistical_basket_risk_cli_workflow(capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    records = tmp_path / "statistical-records.csv"
    schedule = tmp_path / "risk-schedule.csv"
    store = tmp_path / "basket.sqlite"
    materialized = tmp_path / "estimated-replay.csv"
    deliverable = tmp_path / "deliverable.json"
    unused = tmp_path / "unused.csv"
    ledger = tmp_path / "risk-ledger.csv"
    report = tmp_path / "risk-audit.json"

    assert main(["generate-basket-statistical-inputs", str(records), str(schedule)]) == 0
    generated = json.loads(capsys.readouterr().out)
    assert generated["records"] > 100
    assert generated["statistically_estimated"]
    assert main(["basket-input-init", str(store)]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "basket-input-ingest",
                str(store),
                str(records),
                "--source",
                "volforge-synthetic-statistical-v1",
                "--mode",
                "initial",
                "--ingested-at",
                "2026-05-01T00:00:00Z",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "basket-input-estimate",
                str(store),
                str(schedule),
                str(materialized),
                "--basket-id",
                "VOLFORGE-SYNTHETIC-ADJUSTED",
                "--source",
                "volforge-synthetic-statistical-v1",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["estimate_provenance_bound"]
    assert main(["generate-basket-replay", str(unused), str(deliverable)]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "basket-risk-replay",
                str(materialized),
                str(deliverable),
                str(ledger),
                "--report",
                str(report),
                "--settlement-method",
                "physical",
                "--option-type",
                "call",
                "--strike",
                "60",
                "--contract-multiplier",
                "100",
                "--currency",
                "USD",
                "--underlying-id",
                "ACME",
                "--expiry",
                "2026-04-30T20:00:00+00:00",
                "--paths",
                "2000",
            ]
        )
        == 0
    )
    replay = json.loads(capsys.readouterr().out)
    assert replay["downstream_eligible"]
    assert replay["estimate_provenance_bound"]
    assert len(pd.read_csv(ledger)) == 4
    assert json.loads(report.read_text(encoding="utf-8"))["final_audit_hash"]


def test_covariance_selection_cli_workflow(capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    records = tmp_path / "statistical-records.csv"
    schedule = tmp_path / "risk-schedule.csv"
    store = tmp_path / "basket.sqlite"
    materialized = tmp_path / "selected-replay.csv"
    selection_report = tmp_path / "selection.json"

    assert main(["generate-basket-statistical-inputs", str(records), str(schedule)]) == 0
    capsys.readouterr()
    assert main(["basket-input-init", str(store)]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "basket-input-ingest",
                str(store),
                str(records),
                "--source",
                "volforge-synthetic-statistical-v1",
                "--mode",
                "initial",
                "--ingested-at",
                "2026-05-01T00:00:00Z",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "basket-covariance-select",
                str(store),
                "--basket-id",
                "VOLFORGE-SYNTHETIC-ADJUSTED",
                "--formation-at",
                "2026-04-01T20:00:00+00:00",
                "--source",
                "volforge-synthetic-statistical-v1",
                "--forecast-distribution",
                "student_t",
                "--selection-objective",
                "regime_balanced",
                "--report",
                str(selection_report),
            ]
        )
        == 0
    )
    selection = json.loads(capsys.readouterr().out)
    assert selection["downstream_eligible"]
    assert selection["selected_estimator"] == "sample_shrinkage"
    assert selection["regime_coverage_passed"]
    assert selection["robust_validation_passed"]
    assert selection["settings"]["forecast_distribution"] == "student_t"
    assert json.loads(selection_report.read_text(encoding="utf-8"))["selection_sha256"]

    assert (
        main(
            [
                "basket-input-select-covariance",
                str(store),
                str(schedule),
                str(materialized),
                "--basket-id",
                "VOLFORGE-SYNTHETIC-ADJUSTED",
                "--source",
                "volforge-synthetic-statistical-v1",
                "--forecast-distribution",
                "student_t",
                "--selection-objective",
                "regime_balanced",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["covariance_selection_provenance_bound"]
    selected = pd.read_csv(materialized)
    assert len(selected) == 4
    assert "input_covariance_selection_json" in selected
    embedded = json.loads(selected.loc[0, "input_covariance_selection_json"])
    assert embedded["robust_validation_passed"]
    assert embedded["settings"]["selection_objective"] == "regime_balanced"


def test_dynamic_smile_hedge_cli_workflow(capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    records = tmp_path / "smile-records.csv"
    schedule = tmp_path / "smile-schedule.csv"
    store = tmp_path / "basket.sqlite"
    materialized = tmp_path / "smile-replay.csv"
    deliverable = tmp_path / "deliverable.json"
    unused = tmp_path / "unused.csv"
    ledger = tmp_path / "smile-ledger.csv"

    assert main(["generate-basket-smile-inputs", str(records), str(schedule)]) == 0
    generated = json.loads(capsys.readouterr().out)
    assert generated["smile_risk_instruments"]
    assert main(["basket-input-init", str(store)]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "basket-input-ingest",
                str(store),
                str(records),
                "--source",
                "volforge-synthetic-statistical-v1",
                "--mode",
                "initial",
                "--ingested-at",
                "2026-05-01T00:00:00Z",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "basket-input-select-covariance",
                str(store),
                str(schedule),
                str(materialized),
                "--basket-id",
                "VOLFORGE-SYNTHETIC-ADJUSTED",
                "--source",
                "volforge-synthetic-statistical-v1",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["generate-basket-replay", str(unused), str(deliverable)]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "basket-risk-replay",
                str(materialized),
                str(deliverable),
                str(ledger),
                "--settlement-method",
                "physical",
                "--option-type",
                "call",
                "--strike",
                "60",
                "--contract-multiplier",
                "100",
                "--currency",
                "USD",
                "--underlying-id",
                "ACME",
                "--expiry",
                "2026-04-30T20:00:00+00:00",
                "--paths",
                "2000",
                "--include-smile-risks",
                "--require-covariance-selection",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["downstream_eligible"]
    assert result["smile_risk_coverage_bound"]
    assert result["risk_factor_ids"][2:4] == ["smile_skew", "smile_curvature"]
    assert len(pd.read_csv(ledger)) == 4


def test_heston_price_cli_matches_published_cos_benchmark(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(
        [
            "heston-price",
            "--option-type",
            "call",
            "--spot",
            "100",
            "--strike",
            "100",
            "--maturity",
            "1",
            "--kappa",
            "1.5768",
            "--theta",
            "0.0398",
            "--vol-of-vol",
            "0.5751",
            "--rho",
            "-0.5711",
            "--initial-variance",
            "0.0175",
            "--cos-terms",
            "512",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["price"] == pytest.approx(5.785155450, abs=2e-6)
    assert not payload["feller_satisfied"]


@pytest.mark.parametrize(
    ("command", "engine", "extra_arguments", "tolerance"),
    [
        (
            "heston-mc",
            "heston-qe-monte-carlo",
            ["--paths", "20000", "--steps", "48", "--seed", "7"],
            0.15,
        ),
        (
            "heston-pde",
            "heston-finite-difference-pde",
            [
                "--log-price-points",
                "81",
                "--variance-points",
                "41",
                "--time-steps",
                "100",
            ],
            0.15,
        ),
    ],
)
def test_heston_numerical_cli_cross_checks_cos(
    capsys, command: str, engine: str, extra_arguments: list[str], tolerance: float
) -> None:  # type: ignore[no-untyped-def]
    arguments = [
        command,
        "--option-type",
        "call",
        "--spot",
        "100",
        "--strike",
        "100",
        "--maturity",
        "1",
        "--kappa",
        "1.5768",
        "--theta",
        "0.0398",
        "--vol-of-vol",
        "0.5751",
        "--rho",
        "-0.5711",
        "--initial-variance",
        "0.0175",
        *extra_arguments,
    ]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"] == engine
    assert payload["price"] == pytest.approx(5.785155450, abs=tolerance)


def test_heston_convergence_cli_enforces_gate_and_writes_audit(capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    common = [
        "--option-type",
        "call",
        "--spot",
        "100",
        "--strike",
        "100",
        "--maturity",
        "1",
        "--kappa",
        "1.5768",
        "--theta",
        "0.0398",
        "--vol-of-vol",
        "0.5751",
        "--rho",
        "-0.5711",
        "--initial-variance",
        "0.0175",
        "--max-cos-terms",
        "256",
        "--max-cos-width",
        "16",
        "--max-paths",
        "20000",
        "--max-mc-steps",
        "64",
        "--max-log-price-points",
        "61",
        "--max-variance-points",
        "31",
        "--max-pde-time-steps",
        "60",
        "--max-log-price-half-width",
        "4",
        "--max-variance",
        "1",
    ]
    audit = tmp_path / "convergence.json"
    assert (
        main(
            [
                "heston-converge",
                *common,
                "--absolute-tolerance",
                "0.25",
                "--relative-tolerance",
                "0",
                "--report",
                str(audit),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["downstream_eligible"]
    assert len(payload["engines"]) == 3
    assert json.loads(audit.read_text(encoding="utf-8")) == payload

    assert (
        main(
            [
                "heston-converge",
                *common,
                "--absolute-tolerance",
                "0.00000001",
                "--relative-tolerance",
                "0",
            ]
        )
        == 2
    )
    blocked = json.loads(capsys.readouterr().out)
    assert not blocked["downstream_eligible"]
    assert blocked["failure_reasons"]


@pytest.mark.parametrize(
    ("arguments", "engine"),
    [
        (
            ["american-binomial", "--volatility", "0.2", "--steps", "250"],
            "crr-binomial",
        ),
        (
            ["american-lsm", "--paths", "4000", "--steps", "24", "--seed", "7"],
            "heston-longstaff-schwartz",
        ),
        (
            [
                "american-pde",
                "--log-price-points",
                "41",
                "--variance-points",
                "21",
                "--time-steps",
                "30",
            ],
            "heston-ikonen-toivanen-pde",
        ),
    ],
)
def test_american_option_cli(capsys, arguments: list[str], engine: str) -> None:  # type: ignore[no-untyped-def]
    command = arguments[0]
    contract = [
        "--option-type",
        "put",
        "--spot",
        "100",
        "--strike",
        "100",
        "--maturity",
        "1",
        "--rate",
        "0.03",
    ]
    heston = [
        "--kappa",
        "1.5768",
        "--theta",
        "0.0398",
        "--vol-of-vol",
        "0.5751",
        "--rho",
        "-0.5711",
        "--initial-variance",
        "0.0175",
    ]
    payload_arguments = [command, *contract]
    if command != "american-binomial":
        payload_arguments.extend(heston)
    payload_arguments.extend(arguments[1:])
    assert main(payload_arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"] == engine
    assert payload["price"] >= 0.0


@pytest.mark.parametrize(
    ("engine", "extra_arguments", "expected_engine"),
    [
        ("cos", [], "heston-cos"),
        (
            "mc",
            ["--paths", "4000", "--mc-steps", "24", "--seed", "7"],
            "heston-qe-monte-carlo-greeks",
        ),
    ],
)
def test_heston_greeks_cli(
    capsys, engine: str, extra_arguments: list[str], expected_engine: str
) -> None:  # type: ignore[no-untyped-def]
    arguments = [
        "heston-greeks",
        "--engine",
        engine,
        "--option-type",
        "call",
        "--spot",
        "100",
        "--strike",
        "100",
        "--maturity",
        "1",
        "--rate",
        "0.03",
        "--kappa",
        "1.5768",
        "--theta",
        "0.0398",
        "--vol-of-vol",
        "0.5751",
        "--rho",
        "-0.5711",
        "--initial-variance",
        "0.0175",
        *extra_arguments,
    ]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"] == expected_engine
    if engine == "mc":
        assert payload["delta"]["standard_error"] > 0.0
    else:
        assert payload["delta"]["estimated_error"] >= 0.0


def test_heston_hedge_cli_writes_reconciled_audit(capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    report = tmp_path / "hedging.json"
    arguments = [
        "heston-hedge",
        "--option-type",
        "call",
        "--spot",
        "100",
        "--strike",
        "100",
        "--maturity",
        "1",
        "--rate",
        "0.03",
        "--kappa",
        "1.5",
        "--theta",
        "0.04",
        "--vol-of-vol",
        "0.4",
        "--rho",
        "-0.7",
        "--initial-variance",
        "0.04",
        "--paths",
        "6",
        "--rebalancing-steps",
        "3",
        "--report",
        str(report),
    ]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"] == "heston-self-financing-hedge-backtest"
    assert len(payload["strategies"]) == 5
    assert max(item["max_abs_attribution_gap"] for item in payload["strategies"]) < 1e-9
    assert json.loads(report.read_text(encoding="utf-8")) == payload


def test_heston_robustness_cli_writes_gate_audit(capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    report = tmp_path / "robustness.json"
    arguments = [
        "heston-robustness",
        "--option-type",
        "call",
        "--spot",
        "100",
        "--strike",
        "100",
        "--maturity",
        "1",
        "--rate",
        "0.03",
        "--kappa",
        "1.5",
        "--theta",
        "0.04",
        "--vol-of-vol",
        "0.4",
        "--rho",
        "-0.7",
        "--initial-variance",
        "0.04",
        "--seeds",
        "7",
        "17",
        "--path-counts",
        "4",
        "6",
        "--rebalancing-steps",
        "2",
        "3",
        "--absolute-tolerance",
        "10",
        "--minimum-volatility-reduction",
        "0",
        "--maximum-clipped-fraction",
        "1",
        "--maximum-fallback-fraction",
        "1",
        "--report",
        str(report),
    ]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["downstream_eligible"]
    assert payload["total_backtest_runs"] == 12
    assert len(payload["scenarios"]) == 5
    assert json.loads(report.read_text(encoding="utf-8")) == payload


def test_generate_and_validate_cli(tmp_path: Path) -> None:
    raw = tmp_path / "chain.csv"
    enriched = tmp_path / "enriched.csv"
    report = tmp_path / "report.md"
    surface = tmp_path / "surface.csv"
    surface_report = tmp_path / "surface.json"
    heston_fit = tmp_path / "heston.csv"
    heston_report = tmp_path / "heston.json"
    assert main(["generate", str(raw), "--seed", "11"]) == 0
    assert main(["validate", str(raw), str(enriched), "--report", str(report)]) == 0
    assert (
        main(
            [
                "surface",
                str(raw),
                str(surface),
                "--model",
                "ssvi",
                "--report",
                str(surface_report),
                "--grid-points",
                "41",
            ]
        )
        == 0
    )
    assert raw.exists() and enriched.exists() and report.exists()
    assert surface.exists() and surface_report.exists()
    assert "VolForge option-chain quality report" in report.read_text(encoding="utf-8")
    assert '"downstream_eligible": true' in surface_report.read_text(encoding="utf-8")
    assert (
        main(
            [
                "heston-calibrate",
                str(raw),
                str(heston_fit),
                "--report",
                str(heston_report),
                "--cos-terms",
                "96",
                "--max-function-evaluations",
                "150",
            ]
        )
        == 0
    )
    assert heston_fit.exists() and heston_report.exists()
    assert '"calibration_eligible": true' in heston_report.read_text(encoding="utf-8")


def test_generate_and_run_point_in_time_replay_cli(capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    raw = tmp_path / "replay.csv"
    ledger = tmp_path / "ledger.csv"
    report = tmp_path / "replay.json"
    assert main(["generate-replay", str(raw)]) == 0
    generated = json.loads(capsys.readouterr().out)
    assert generated["snapshots"] == 4
    assert (
        main(
            [
                "replay",
                str(raw),
                str(ledger),
                "--report",
                str(report),
                "--cos-terms",
                "64",
                "--max-function-evaluations",
                "60",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["downstream_eligible"]
    assert payload["hash_chain_verified"]
    assert ledger.exists()
    assert json.loads(report.read_text(encoding="utf-8")) == payload


def test_run_rolling_model_benchmark_cli(capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    raw = tmp_path / "replay.csv"
    predictions = tmp_path / "predictions.csv"
    report = tmp_path / "benchmark.json"
    bundle = tmp_path / "benchmark.zip"
    assert main(["generate-replay", str(raw)]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "benchmark",
                str(raw),
                str(predictions),
                "--report",
                str(report),
                "--bundle",
                str(bundle),
                "--training-window",
                "2",
                "--cos-terms",
                "64",
                "--max-function-evaluations",
                "60",
                "--comparison-hac-lags",
                "0",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["downstream_eligible"]
    assert payload["hash_chain_verified"]
    assert payload["evaluation_snapshot_count"] == 2
    assert payload["comparison_inference"]["hac_lags"] == 0
    assert payload["comparison_inference"]["dm_method"].startswith("hln-corrected")
    assert not payload["comparison_inference"]["dm_available"]
    assert set(pd.read_csv(predictions)["model"]) == {"bsm", "heston"}
    assert json.loads(report.read_text(encoding="utf-8")) == payload
    assert verify_research_bundle(bundle.read_bytes()).valid

    password_file = tmp_path / "password.txt"
    private_key = tmp_path / "research-private.pem"
    public_key = tmp_path / "research-public.pem"
    attestation = tmp_path / "benchmark.attestation.json"
    password_file.write_text("test signing password\n", encoding="utf-8")
    assert (
        main(
            [
                "research-keygen",
                str(private_key),
                str(public_key),
                "--password-file",
                str(password_file),
            ]
        )
        == 0
    )
    key_payload = json.loads(capsys.readouterr().out)
    assert key_payload["algorithm"] == "ed25519"
    assert key_payload["private_key_encrypted"]
    assert (
        main(
            [
                "research-sign",
                str(bundle),
                str(private_key),
                str(attestation),
                "--password-file",
                str(password_file),
                "--issued-at",
                "2026-10-16T19:00:00Z",
                "--signer",
                "CLI test",
            ]
        )
        == 0
    )
    signed_payload = json.loads(capsys.readouterr().out)
    assert signed_payload["algorithm"] == "ed25519"
    assert (
        main(
            [
                "research-verify",
                str(bundle),
                str(attestation),
                "--trusted-public-key",
                str(public_key),
            ]
        )
        == 0
    )
    verification = json.loads(capsys.readouterr().out)
    assert verification["accepted"]
    assert verification["signer_trusted"]
    with pytest.raises(SystemExit) as overwrite:
        main(
            [
                "research-sign",
                str(bundle),
                str(private_key),
                str(attestation),
                "--password-file",
                str(password_file),
                "--issued-at",
                "2026-10-16T19:00:00Z",
            ]
        )
    assert overwrite.value.code == 2
    assert "refusing to overwrite existing file" in capsys.readouterr().err


def test_historical_store_cli_round_trip(capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    store = tmp_path / "history.sqlite"
    raw = tmp_path / "quotes.csv"
    query_output = tmp_path / "events.csv"
    snapshot_output = tmp_path / "snapshot.csv"
    verification_report = tmp_path / "verification.json"
    pd.DataFrame(
        [
            {
                "record_type": "quote",
                "event_timestamp": "2025-01-02T15:30:00Z",
                "received_timestamp": "2025-01-02T15:30:00.050Z",
                "source_record_id": "q1",
                "revision": 0,
                "record_action": "upsert",
                "underlying_symbol": "XYZ",
                "contract_symbol": "XYZ-20250321-C-100",
                "expiration": "2025-03-21T20:00:00Z",
                "strike": 100.0,
                "option_type": "call",
                "bid": 4.9,
                "ask": 5.1,
                "underlying_spot": 100.0,
                "risk_free_rate": 0.03,
                "dividend_yield": 0.01,
            }
        ]
    ).to_csv(raw, index=False)

    assert main(["history-init", str(store)]) == 0
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["schema_version"] == 1
    assert (
        main(
            [
                "history-ingest",
                str(store),
                str(raw),
                "--source",
                "demo",
                "--mode",
                "initial",
                "--ingested-at",
                "2025-01-02T16:00:00Z",
            ]
        )
        == 0
    )
    ingested = json.loads(capsys.readouterr().out)
    assert ingested["row_count"] == 1

    assert (
        main(
            [
                "history-query",
                str(store),
                str(query_output),
                "--record-type",
                "quote",
                "--received-at",
                "2025-01-02T15:31:00Z",
                "--source",
                "demo",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["rows"] == 1
    assert (
        main(
            [
                "history-snapshot",
                str(store),
                str(snapshot_output),
                "--snapshot-at",
                "2025-01-02T15:30:30Z",
                "--source",
                "demo",
                "--maximum-quote-age",
                "1min",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["snapshots"] == 1
    snapshot = pd.read_csv(snapshot_output)
    assert snapshot.loc[0, "source_record_id"] == "q1"

    assert (
        main(
            [
                "history-verify",
                str(store),
                "--report",
                str(verification_report),
            ]
        )
        == 0
    )
    verified = json.loads(capsys.readouterr().out)
    assert verified["valid"]
    assert json.loads(verification_report.read_text(encoding="utf-8")) == verified


def test_reference_store_cli_round_trip(capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    store = tmp_path / "reference.sqlite"
    raw = tmp_path / "products.csv"
    output = tmp_path / "products-query.csv"
    report = tmp_path / "reference-verification.json"
    pd.DataFrame(
        [
            {
                "source_record_id": "product-xyz",
                "revision": 0,
                "record_action": "upsert",
                "published_timestamp": "2025-01-01T00:00:00Z",
                "effective_from": "2025-01-01T00:00:00Z",
                "product_id": "XYZ-EQUITY-OPTIONS",
                "product_symbol": "XYZ",
                "exchange": "TEST",
                "calendar_id": "TEST-RTH",
                "underlying_asset_type": "equity",
                "exercise_style": "american",
                "settlement_method": "physical",
                "settlement_style": "pm",
                "currency": "USD",
                "default_multiplier": 100,
                "expiration_rule": "listed timestamp",
                "last_trade_rule": "listed timestamp",
                "settlement_rule": "physical delivery",
                "status": "active",
                "source_reference": "test circular",
            }
        ]
    ).to_csv(raw, index=False)

    assert main(["reference-init", str(store)]) == 0
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["schema_version"] == 1
    assert (
        main(
            [
                "reference-ingest",
                str(store),
                str(raw),
                "--kind",
                "product",
                "--source",
                "test-reference",
                "--mode",
                "initial",
                "--ingested-at",
                "2025-01-02T00:00:00Z",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["row_count"] == 1
    assert (
        main(
            [
                "reference-query",
                str(store),
                str(output),
                "--kind",
                "product",
                "--effective-at",
                "2025-01-03T00:00:00Z",
                "--published-at",
                "2025-01-03T00:00:00Z",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["rows"] == 1
    assert pd.read_csv(output).loc[0, "product_id"] == "XYZ-EQUITY-OPTIONS"
    assert main(["reference-verify", str(store), "--report", str(report)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["valid"]
    assert json.loads(report.read_text(encoding="utf-8")) == verified
