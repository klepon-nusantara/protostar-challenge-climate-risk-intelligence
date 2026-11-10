"""Command-line entry point for the climate challenge workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtesting import append_backtest_run_log, run_backtest
from .cleaning import CleaningResult, clean_annual_indices
from .config import DEFAULT_CONFIG, FORECAST_FEATURES, FORECAST_METHODS
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


def _load_clean_data() -> tuple[PreparedDataset, CleaningResult, pd.DataFrame | None]:
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
    return prepared, cleaned, daily_observations


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
        backtest, scores, audit = run_oot_evaluation(
            historical_annual_indices,
            oot_annual_indices,
            city_info,
            origin=OOT_ORIGIN,
            max_horizon=OOT_MAX_HORIZON,
            daily_observations=daily_observations,
        )
        repository_root = Path(__file__).resolve().parents[3]
        results_directory = repository_root / "results" / "oot_eval"
        log_path = append_oot_run_log(
            results_directory / "oot_eval.log",
            backtest,
            scores,
            audit,
            origin=OOT_ORIGIN,
            input_paths=(
                historical_path,
                oot_annual_path,
                city_info_path,
                daily_path,
            ),
            run_label="annual + daily features",
        )
        predictions_path = results_directory / "predictions.csv"
        backtest.assign(run_label="annual + daily features").loc[
            :,
            lambda frame: [
                column for column in frame if not column.startswith("_raw_")
            ],
        ].to_csv(predictions_path, index=False)
        print("OOT data quality audit:")
        print(audit.to_string())
        print(
            f"Fixed origin: {OOT_ORIGIN}; evaluated through "
            f"{int(backtest['target_year'].max())} "
            f"({int(backtest['h'].max())} observed forecast leads)."
        )
        print(scores.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        print(f"Appended annual + daily feature run to {log_path}.")
        print(f"Wrote OOT predictions to {predictions_path}.")
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
        )
        forecasts, selected_methods = forecast_all_stations(
            cleaned,
            prepared.city_info,
            calibration_backtest,
            horizon=args.horizon,
            daily_observations=daily_observations,
            selected_methods=FORECAST_METHODS,
            feature_columns=FORECAST_FEATURES,
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
        )
        if daily_observations is not None:
            input_paths += (prepared.data_dir / "daily.parquet",)
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
