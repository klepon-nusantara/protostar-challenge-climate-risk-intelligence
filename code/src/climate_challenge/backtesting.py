"""Rolling-origin backtesting for all challenge metrics and stations."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from .cleaning import TARGET_COLUMNS, metric_matrix, station_metadata
from .config import DEFAULT_CONFIG, GlobalModelConfig
from .evaluation import evaluate_backtest
from .models.global_models import (
    BASE_FEATURES,
    DAILY_FEATURES,
    _daily_feature_matrices,
    build_panel,
    fit_predict_origin,
)


def run_backtest(
    cleaned: pd.DataFrame,
    city_info: pd.DataFrame,
    origins: range | list[int] | tuple[int, ...] = range(
        DEFAULT_CONFIG.backtest.start_origin,
        DEFAULT_CONFIG.backtest.end_origin + 1,
    ),
    horizon: int = DEFAULT_CONFIG.backtest.horizon,
    model_config: GlobalModelConfig = DEFAULT_CONFIG.model,
    daily_observations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Replay fixed historical origins, scoring only observed target years."""

    origin_years = tuple(origins)
    if not origin_years:
        raise ValueError("At least one backtest origin is required")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    latest_year = int(cleaned["Year"].max())
    if any(origin + horizon > latest_year for origin in origin_years):
        raise ValueError(
            "Every backtest origin plus horizon must be within the observed data"
        )

    metadata = station_metadata(city_info)
    outputs: list[pd.DataFrame] = []
    daily_matrices: dict[str, pd.DataFrame] | None = None
    for metric in TARGET_COLUMNS:
        matrix = metric_matrix(cleaned, metric)
        if daily_observations is not None and daily_matrices is None:
            daily_matrices = _daily_feature_matrices(
                daily_observations, matrix.columns
            )
        panel = build_panel(
            matrix,
            metadata,
            horizon,
            model_config,
            daily_feature_matrices=daily_matrices,
        )
        for origin in origin_years:
            predictions, _ = fit_predict_origin(panel, origin, model_config)
            predictions["metric"] = metric
            outputs.append(predictions)
    return pd.concat(outputs, ignore_index=True)


def append_backtest_run_log(
    log_path: str | Path,
    backtest: pd.DataFrame,
    scores: pd.DataFrame,
    audit: pd.DataFrame,
    origins: tuple[int, ...],
    horizon: int,
    model_config: GlobalModelConfig = DEFAULT_CONFIG.model,
    input_paths: tuple[str | Path, ...] = (),
) -> Path:
    """Append model settings, scored period, audit, and results to a run log."""

    required = {"origin", "target_year", "city_id"}
    missing = required.difference(backtest.columns)
    if missing:
        raise ValueError(
            "backtest is missing required log columns: " + ", ".join(sorted(missing))
        )
    if backtest.empty:
        raise ValueError("Cannot log an empty backtest")
    if not origins:
        raise ValueError("At least one backtest origin is required")

    target_rows = backtest[["origin", "target_year", "city_id"]].drop_duplicates()
    methods = sorted(scores["method"].dropna().unique())
    overall_scores = evaluate_backtest(backtest)
    best_scores = (
        overall_scores.dropna(subset=["mae"])
        .sort_values(["metric", "mae", "method"], kind="stable")
        .groupby("metric", sort=True, as_index=False)
        .first()
    )
    model_features = [
        feature
        for feature in (*BASE_FEATURES, *DAILY_FEATURES)
        if feature in backtest.columns
    ]
    if not (set(model_features) - {"h"}):
        model_features = list(BASE_FEATURES)

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as output:
        output.write(f"\n{'=' * 80}\n")
        output.write(f"Backtest run: {datetime.now().astimezone().isoformat()}\n")
        output.write(
            f"Origins: {min(origins)}-{max(origins)} ({len(origins)} origins)\n"
        )
        output.write(f"Forecast horizon: {horizon}\n")
        output.write(
            f"Scored period: {int(backtest['target_year'].min())}-"
            f"{int(backtest['target_year'].max())}\n"
        )
        output.write(
            f"Prediction rows: {len(backtest)}; "
            f"station-origin-targets: {len(target_rows)}; "
            f"stations: {backtest['city_id'].nunique()}\n"
        )
        output.write(f"Models experimented: {', '.join(methods)}\n")
        output.write(
            "Model configuration: "
            + json.dumps(asdict(model_config), sort_keys=True)
            + "\n"
        )
        output.write(
            "Features used by Ridge and LightGBM: "
            + ", ".join(model_features)
            + "\n"
        )
        output.write(
            "Feature note: other listed methods are fixed baselines and do not "
            "consume this feature vector.\n"
        )
        if input_paths:
            output.write(
                "Input files: " + ", ".join(str(path) for path in input_paths) + "\n"
            )
        output.write("Data quality audit:\n")
        output.write(audit.to_string() + "\n")
        output.write("Scores (MAE and scored rows, by metric):\n")
        output.write(
            scores.to_string(index=False, float_format=lambda value: f"{value:.6f}")
            + "\n"
        )
        output.write("Overall pooled MAE across all origins:\n")
        output.write(
            overall_scores.to_string(
                index=False, float_format=lambda value: f"{value:.6f}"
            )
            + "\n"
        )
        output.write("Best pooled MAE by metric:\n")
        output.write(
            best_scores.to_string(
                index=False, float_format=lambda value: f"{value:.6f}"
            )
            + "\n"
        )
    return log_file
