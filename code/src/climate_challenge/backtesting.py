"""Rolling-origin backtesting for all challenge metrics and stations."""

from __future__ import annotations

import pandas as pd

from .cleaning import TARGET_COLUMNS, metric_matrix, station_metadata
from .models.global_models import (
    GlobalModelConfig,
    build_panel,
    fit_predict_origin,
)

MODEL_METHODS = (
    "Last Year",
    "30-Yr Avg",
    "Linear Trend",
    "Half-Trend",
    "Ridge",
    "LightGBM",
    "Ensemble",
)


def run_backtest(
    cleaned: pd.DataFrame,
    city_info: pd.DataFrame,
    origins: range | list[int] | tuple[int, ...] = range(1995, 2012),
    horizon: int = 7,
    model_config: GlobalModelConfig = GlobalModelConfig(),
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
    for metric in TARGET_COLUMNS:
        matrix = metric_matrix(cleaned, metric)
        panel = build_panel(matrix, metadata, horizon, model_config)
        for origin in origin_years:
            predictions, _ = fit_predict_origin(panel, origin, model_config)
            predictions["metric"] = metric
            outputs.append(predictions)
    return pd.concat(outputs, ignore_index=True)
