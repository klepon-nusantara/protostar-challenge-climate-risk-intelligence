"""Rolling-origin backtesting for all challenge metrics and stations."""

from __future__ import annotations

import pandas as pd

from .cleaning import TARGET_COLUMNS, metric_matrix, station_metadata
from .config import DEFAULT_CONFIG, MODEL_METHODS
from .models.global_models import (
    GlobalModelConfig,
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
