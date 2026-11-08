"""Command-line entry point for the climate challenge workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtesting import run_backtest
from .cleaning import clean_annual_indices
from .data import FORECAST_CUTOFF, PreparedDataset, prepare_dataset
from .evaluation import evaluate_backtest
from .forecasting import forecast_all_stations
from .submission import write_submission


def _load_clean_data() -> tuple[PreparedDataset, pd.DataFrame]:
    prepared = prepare_dataset()
    cleaned = clean_annual_indices(prepared.annual_indices)
    print("Data quality audit:")
    print(cleaned.audit.to_string())
    return prepared, cleaned.data


def _add_backtest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-origin", type=int, default=1995)
    parser.add_argument("--end-origin", type=int, default=2011)
    parser.add_argument("--horizon", type=int, default=7)


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
    forecast_parser.add_argument("--template", type=Path, default=Path("submission.csv"))
    forecast_parser.add_argument("--output", type=Path, default=Path("submission.csv"))

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

    if args.start_origin > args.end_origin:
        parser.error("--start-origin must not exceed --end-origin")
    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    if command == "forecast" and args.horizon != 7:
        parser.error("forecast submissions require --horizon 7")
    origins = range(args.start_origin, args.end_origin + 1)

    prepared, cleaned = _load_clean_data()
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
    )
    submission = write_submission(forecasts, args.template, args.output)
    print(f"Selected methods by metric: {best_methods}")
    print(f"Wrote {len(submission):,} rows to {args.output}.")


if __name__ == "__main__":
    main()
