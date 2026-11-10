"""Command-line entry point for the climate challenge workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtesting import append_backtest_run_log, run_backtest
from .cleaning import CleaningResult, clean_annual_indices
from .config import (
    DEFAULT_CONFIG,
    FORECAST_FEATURES,
    FORECAST_METHODS,
    STATIC_FEATURES_BY_METRIC,
)
from .data import (
    FORECAST_CUTOFF,
    PreparedDataset,
    add_static_city_features,
    prepare_dataset,
)
from .evaluation import evaluate_backtest
from .forecasting import forecast_all_stations
from .submission import write_submission
from .us_gdp import preprocess_us_gdp
from .us_personal_income import preprocess_us_personal_income


def _load_clean_data() -> tuple[PreparedDataset, CleaningResult, pd.DataFrame]:
    prepared = prepare_dataset()
    prepared.city_info = add_static_city_features(
        prepared.city_info,
        prepared.elevation,
        prepared.city_climate_divisions,
        prepared.city_coastline_distances,
    )
    cleaned = clean_annual_indices(prepared.annual_indices)
    print("Data quality audit:")
    print(cleaned.audit.to_string())
    return prepared, cleaned, prepared.daily


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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    command = args.command or "prepare"

    if command == "prepare":
        prepared = prepare_dataset()
        gdp_directory = prepared.data_dir / "us_gdp"
        gdp_tables = (
            preprocess_us_gdp(gdp_directory)
            if gdp_directory.is_dir()
            and any(
                archive.name.startswith(("CAGDP", "SAGDP"))
                for archive in gdp_directory.glob("*.zip")
            )
            else ()
        )
        personal_income_directory = prepared.data_dir / "us_personal_income"
        personal_income_tables = (
            preprocess_us_personal_income(personal_income_directory)
            if personal_income_directory.is_dir()
            and any(
                archive.name.startswith(("CAINC", "SAINC"))
                for archive in personal_income_directory.glob("*.zip")
            )
            else ()
        )
        years = prepared.annual_indices["Year"]
        print(
            "Downloaded and validated challenge data "
            f"(annual_indices: {len(prepared.annual_indices):,} rows, "
            f"{int(years.min())}-{int(years.max())}; "
            f"city_info: {len(prepared.city_info):,} rows; "
            f"elevation: {len(prepared.elevation):,} stations; "
            f"city climate divisions: "
            f"{len(prepared.city_climate_divisions):,} stations; "
            f"coastline distances: "
            f"{len(prepared.city_coastline_distances):,} stations)."
        )
        print(f"Parquet files are stored in {prepared.data_dir}.")
        if gdp_tables:
            print(
                f"Preprocessed {len(gdp_tables)} annual BEA GDP tables in "
                f"{gdp_directory}."
            )
        if personal_income_tables:
            print(
                f"Preprocessed {len(personal_income_tables)} annual BEA personal "
                f"income tables in {personal_income_directory}."
            )
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

    prepared, cleaning_result, daily_observations = _load_clean_data()
    cleaned = cleaning_result.data
    latest_year = min(int(cleaned["Year"].max()), FORECAST_CUTOFF.year)
    if args.end_origin + args.horizon > latest_year:
        parser.error(
            f"end origin plus horizon must not exceed last observed year {latest_year}"
        )

    if command == "forecast":
        print(
            "Estimating lead calibration for configured forecast methods only: "
            f"{FORECAST_METHODS}"
        )
        calibration_backtest = run_backtest(
            cleaned,
            prepared.city_info,
            origins=origins,
            horizon=args.horizon,
            daily_observations=daily_observations,
            methods_by_metric=FORECAST_METHODS,
            feature_columns=FORECAST_FEATURES,
            static_features_by_metric=STATIC_FEATURES_BY_METRIC,
        )
        forecasts, selected_methods = forecast_all_stations(
            cleaned,
            prepared.city_info,
            calibration_backtest,
            horizon=args.horizon,
            daily_observations=daily_observations,
            selected_methods=FORECAST_METHODS,
            feature_columns=FORECAST_FEATURES,
            static_features_by_metric=STATIC_FEATURES_BY_METRIC,
        )
        submission = write_submission(forecasts, args.template, args.output)
        print(f"Selected methods by metric: {selected_methods}")
        print(f"Wrote {len(submission):,} rows to {args.output}.")
        return

    backtest = run_backtest(
        cleaned,
        prepared.city_info,
        origins=origins,
        horizon=args.horizon,
        daily_observations=daily_observations,
        static_features_by_metric=STATIC_FEATURES_BY_METRIC,
    )
    scores = evaluate_backtest(backtest)
    if command == "backtest":
        repository_root = Path(__file__).resolve().parents[3]
        results_directory = repository_root / "results" / "backtest"
        results_directory.mkdir(parents=True, exist_ok=True)
        predictions_path = results_directory / "predictions.csv"
        backtest.loc[
            :,
            lambda frame: [
                column for column in frame if not column.startswith("_raw_")
            ],
        ].to_csv(predictions_path, index=False)
        input_paths = (
            prepared.data_dir / "annual_indices.parquet",
            prepared.data_dir / "city_info.parquet",
            prepared.data_dir / "daily.parquet",
        )
        log_path = append_backtest_run_log(
            results_directory / "backtest.log",
            backtest,
            scores,
            cleaning_result.audit,
            origins=tuple(origins),
            horizon=args.horizon,
            input_paths=input_paths,
        )
        print(scores.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        print(f"Appended backtest run details to {log_path}.")
        print(f"Wrote backtest predictions to {predictions_path}.")
        return

if __name__ == "__main__":
    main()
