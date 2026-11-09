"""Command-line entry point for the climate challenge workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtesting import run_backtest
from .cleaning import clean_annual_indices
from .config import DEFAULT_CONFIG
from .data import FORECAST_CUTOFF, PreparedDataset, prepare_dataset
from .evaluation import evaluate_backtest
from .forecasting import forecast_all_stations
from .oot_eval import (
    OOT_MAX_HORIZON,
    OOT_ORIGIN,
    append_oot_run_log,
    run_oot_evaluation,
)
from .submission import write_submission


def _load_clean_data() -> tuple[PreparedDataset, pd.DataFrame, pd.DataFrame | None]:
    prepared = prepare_dataset()
    cleaned = clean_annual_indices(prepared.annual_indices)
    print("Data quality audit:")
    print(cleaned.audit.to_string())
    daily_path = prepared.data_dir / "daily.parquet"
    daily_observations = (
        pd.read_parquet(daily_path) if daily_path.is_file() else None
    )
    if daily_observations is None:
        print(
            f"Daily feature data not found at {daily_path}; "
            "using annual-only model features."
        )
    else:
        print(f"Using daily feature data from {daily_path}.")
    return prepared, cleaned.data, daily_observations


def _add_backtest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--start-origin", type=int, default=DEFAULT_CONFIG.backtest.start_origin
    )
    parser.add_argument(
        "--end-origin", type=int, default=DEFAULT_CONFIG.backtest.end_origin
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_CONFIG.backtest.horizon)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m climate_challenge",
        description="Prepare climate data, run temporal backtests, and write forecasts.",
    )
    commands = parser.add_subparsers(dest="command")

    commands.add_parser("prepare", help="download and validate source tables")

    backtest_parser = commands.add_parser(
        "backtest", help="run rolling-origin backtests and report MAE"
    )
    _add_backtest_arguments(backtest_parser)

    forecast_parser = commands.add_parser(
        "forecast", help="backtest, select methods, and fill submission.csv"
    )
    _add_backtest_arguments(forecast_parser)
    forecast_parser.add_argument(
        "--template",
        type=Path,
        default=Path(DEFAULT_CONFIG.submission.template_path),
    )
    forecast_parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_CONFIG.submission.output_path),
    )

    oot_parser = commands.add_parser(
        "oot_eval",
        help="evaluate every model at the fixed 2018 origin on later observations",
    )
    oot_parser.add_argument(
        "--data-dir", type=Path, default=Path("data")
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    command = args.command or "prepare"

    if command == "prepare":
        prepared = prepare_dataset()
        years = prepared.annual_indices["Year"]
        print(
            "Downloaded and validated challenge data "
            f"(annual_indices: {len(prepared.annual_indices):,} rows, "
            f"{int(years.min())}-{int(years.max())}; "
            f"city_info: {len(prepared.city_info):,} rows)."
        )
        print(f"Parquet files are stored in {prepared.data_dir}.")
        return

    if command == "oot_eval":
        historical_path = args.data_dir / "annual_indices.parquet"
        oot_annual_path = args.data_dir / "annual_indices_oot.parquet"
        city_info_path = args.data_dir / "city_info.parquet"
        daily_path = args.data_dir / "daily.parquet"
        missing_paths = [
            path
            for path in (
                historical_path,
                oot_annual_path,
                city_info_path,
                daily_path,
            )
            if not path.is_file()
        ]
        if missing_paths:
            parser.error(
                "missing OOT Parquet input(s): "
                + ", ".join(str(path) for path in missing_paths)
                + "; first run python -m climate_challenge.oot_eval.preprocessing"
            )
        historical_annual_indices = pd.read_parquet(historical_path)
        oot_annual_indices = pd.read_parquet(oot_annual_path)
        city_info = pd.read_parquet(city_info_path)
        daily_observations = pd.read_parquet(daily_path)
        baseline_backtest, baseline_scores, audit = run_oot_evaluation(
            historical_annual_indices,
            oot_annual_indices,
            city_info,
            origin=OOT_ORIGIN,
            max_horizon=OOT_MAX_HORIZON,
        )
        daily_backtest, daily_scores, _ = run_oot_evaluation(
            historical_annual_indices,
            oot_annual_indices,
            city_info,
            origin=OOT_ORIGIN,
            max_horizon=OOT_MAX_HORIZON,
            daily_observations=daily_observations,
        )
        repository_root = Path(__file__).resolve().parents[3]
        log_path = append_oot_run_log(
            repository_root / "oot_eval.log",
            baseline_backtest,
            baseline_scores,
            audit,
            origin=OOT_ORIGIN,
            input_paths=(historical_path, oot_annual_path, city_info_path),
            run_label="annual-only baseline",
        )
        append_oot_run_log(
            log_path,
            daily_backtest,
            daily_scores,
            audit,
            origin=OOT_ORIGIN,
            input_paths=(
                historical_path,
                oot_annual_path,
                city_info_path,
                daily_path,
            ),
            run_label="daily feature groups",
        )
        print("OOT data quality audit:")
        print(audit.to_string())
        baseline_pooled = evaluate_backtest(baseline_backtest).rename(
            columns={"mae": "annual_only_mae"}
        )
        daily_pooled = evaluate_backtest(daily_backtest).rename(
            columns={"mae": "daily_features_mae"}
        )
        comparison = baseline_pooled.merge(
            daily_pooled[["metric", "method", "daily_features_mae"]],
            on=["metric", "method"],
            validate="one_to_one",
        )
        comparison["mae_change"] = (
            comparison["daily_features_mae"] - comparison["annual_only_mae"]
        )
        print(
            f"Fixed origin: {OOT_ORIGIN}; evaluated through "
            f"{int(daily_backtest['target_year'].max())} "
            f"({int(daily_backtest['h'].max())} observed forecast leads)."
        )
        print("Pooled MAE comparison (daily_features_mae - annual_only_mae; lower is better):")
        print(
            comparison.to_string(
                index=False, float_format=lambda value: f"{value:.3f}"
            )
        )
        print(f"Appended baseline and daily-feature runs to {log_path}.")
        return

    if args.start_origin > args.end_origin:
        parser.error("--start-origin must not exceed --end-origin")
    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    if command == "forecast" and args.horizon != DEFAULT_CONFIG.forecast.horizon:
        parser.error(
            "forecast submissions require "
            f"--horizon {DEFAULT_CONFIG.forecast.horizon}"
        )
    origins = range(args.start_origin, args.end_origin + 1)

    prepared, cleaned, daily_observations = _load_clean_data()
    latest_year = min(int(cleaned["Year"].max()), FORECAST_CUTOFF.year)
    if args.end_origin + args.horizon > latest_year:
        parser.error(
            f"end origin plus horizon must not exceed last observed year {latest_year}"
        )

    backtest = run_backtest(
        cleaned,
        prepared.city_info,
        origins=origins,
        horizon=args.horizon,
        daily_observations=daily_observations,
    )
    scores = evaluate_backtest(backtest)
    if command == "backtest":
        print(scores.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        return

    forecasts, best_methods = forecast_all_stations(
        cleaned,
        prepared.city_info,
        backtest,
        horizon=args.horizon,
        daily_observations=daily_observations,
    )
    submission = write_submission(forecasts, args.template, args.output)
    print(f"Selected methods by metric: {best_methods}")
    print(f"Wrote {len(submission):,} rows to {args.output}.")


if __name__ == "__main__":
    main()
