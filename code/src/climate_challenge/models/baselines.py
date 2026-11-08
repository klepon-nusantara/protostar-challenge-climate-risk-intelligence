"""Per-station persistence, climatology, and trend forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

BASELINE_METHODS = ("Last Year", "30-Yr Avg", "Linear Trend")


def recent_average(matrix: pd.DataFrame, origin: int, window: int = 30) -> pd.Series:
    return matrix.loc[origin - window + 1 : origin].mean()


def last_observation(matrix: pd.DataFrame, origin: int) -> pd.Series:
    return matrix.loc[:origin].ffill().iloc[-1]


def trend_coefficients(
    matrix: pd.DataFrame, origin: int, window: int = 30, min_points: int = 15
) -> pd.DataFrame:
    """Fit a separate least-squares line to each station's trailing window."""

    history = matrix.loc[origin - window + 1 : origin]
    years = history.index.to_numpy(dtype=float)
    result: dict[object, tuple[float, float]] = {}
    for station_id in history.columns:
        values = history[station_id].to_numpy(dtype=float)
        valid = np.isfinite(values)
        if valid.sum() < min_points:
            result[station_id] = (np.nan, np.nan)
            continue
        slope, intercept = np.polyfit(years[valid], values[valid], 1)
        result[station_id] = (float(slope), float(intercept))
    return pd.DataFrame.from_dict(
        result, orient="index", columns=["slope", "intercept"]
    )


def forecast_baselines(
    matrix: pd.DataFrame,
    origin: int,
    horizon: int,
    min_trend_points: int = 15,
) -> pd.DataFrame:
    """Forecast each station for one lead using the shared baselines."""

    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    avg = recent_average(matrix, origin)
    last = last_observation(matrix, origin)
    coefficients = trend_coefficients(
        matrix, origin, min_points=min_trend_points
    )
    trend = coefficients["slope"] * (origin + horizon) + coefficients["intercept"]
    return pd.DataFrame(
        {
            "city_id": matrix.columns,
            "Last Year": last.reindex(matrix.columns).to_numpy(),
            "30-Yr Avg": avg.reindex(matrix.columns).to_numpy(),
            "Linear Trend": trend.reindex(matrix.columns).to_numpy(),
        }
    )
