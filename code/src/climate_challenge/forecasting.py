"""Final 2019-2025 forecast generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtesting import apply_lead_calibration
from .cleaning import TARGET_COLUMNS, metric_matrix, station_metadata
from .config import (
    BASE_MODEL_METHODS,
    DEFAULT_CONFIG,
    FORECAST_FEATURES,
    FORECAST_METHODS,
)
from .models.global_models import (
    GlobalModelConfig,
    _daily_feature_matrices,
    build_panel,
    feature_columns_for_metric,
    fit_predict_origin,
)

FORECAST_ORIGIN = DEFAULT_CONFIG.forecast.origin
FORECAST_HORIZON = DEFAULT_CONFIG.forecast.horizon


def apply_sparse_station_fallback(
    forecasts: pd.DataFrame,
    cleaned: pd.DataFrame,
    metadata: pd.DataFrame,
    minimum_valid_years: int = DEFAULT_CONFIG.forecast.minimum_valid_years,
    neighbors: int = DEFAULT_CONFIG.forecast.neighbor_count,
    origin: int = FORECAST_ORIGIN,
    baseline_window: int = DEFAULT_CONFIG.model.baseline_window,
) -> pd.DataFrame:
    """Fill stations with sparse recent records using nearby station forecasts."""

    if minimum_valid_years < 1 or neighbors < 1:
        raise ValueError("minimum_valid_years and neighbors must be positive")
    result = forecasts.copy()
    station_ids = result["city_id"].drop_duplicates().tolist()
    matrices = {metric: metric_matrix(cleaned, metric) for metric in TARGET_COLUMNS}
    sparse: set[str] = set()
    for matrix in matrices.values():
        coverage = matrix.loc[origin - baseline_window + 1 : origin].notna().sum()
        sparse.update(coverage[coverage < minimum_valid_years].index.astype(str))

    good_ids = [station for station in station_ids if station not in sparse]
    if not sparse:
        return result
    if len(good_ids) < neighbors:
        raise ValueError(
            f"Only {len(good_ids)} stations have sufficient coverage; "
            f"{neighbors} are required for sparse-station fallback"
        )

    for station_id in sorted(sparse):
        if station_id not in metadata.index:
            raise ValueError(f"Missing coordinates for sparse station {station_id}")
        latitude = metadata.loc[station_id, "Lat"]
        longitude = metadata.loc[station_id, "Lon"]
        lat_delta = metadata.loc[good_ids, "Lat"] - latitude
        lon_delta = (metadata.loc[good_ids, "Lon"] - longitude) * np.cos(
            np.radians(latitude)
        )
        neighbor_ids = (
            pd.Series(np.hypot(lat_delta, lon_delta), index=good_ids)
            .nsmallest(neighbors)
            .index.tolist()
        )

        for metric, matrix in matrices.items():
            station_history = matrix[station_id]
            neighbor_history = matrix[neighbor_ids].mean(axis=1)
            offset = (station_history - neighbor_history).dropna().mean()
            if pd.isna(offset):
                raise ValueError(
                    f"Cannot estimate a historical offset for sparse station "
                    f"{station_id}, metric {metric}"
                )
            neighbors_forecasts = result.loc[
                (result["metric"] == metric)
                & result["city_id"].isin(neighbor_ids)
            ]
            for method in result.columns.intersection(
                ["Last Year", "30-Yr Avg", "Linear Trend", "Half-Trend", "Ridge", "LightGBM", "Ensemble"]
            ):
                neighbor_mean = neighbors_forecasts.groupby("h")[method].mean()
                row_mask = (
                    (result["metric"] == metric)
                    & (result["city_id"] == station_id)
                )
                result.loc[row_mask, method] = (
                    result.loc[row_mask, "h"].map(neighbor_mean) + offset
                )
    return result


def forecast_all_stations(
    cleaned: pd.DataFrame,
    city_info: pd.DataFrame,
    backtest: pd.DataFrame,
    origin: int = FORECAST_ORIGIN,
    horizon: int = FORECAST_HORIZON,
    model_config: GlobalModelConfig = DEFAULT_CONFIG.model,
    use_sparse_fallback: bool = DEFAULT_CONFIG.forecast.use_sparse_fallback,
    minimum_valid_years: int = DEFAULT_CONFIG.forecast.minimum_valid_years,
    neighbors: int = DEFAULT_CONFIG.forecast.neighbor_count,
    daily_observations: pd.DataFrame | None = None,
    selected_methods: dict[str, str] = FORECAST_METHODS,
    feature_columns: tuple[str, ...] | None = FORECAST_FEATURES,
    static_features_by_metric: dict[str, tuple[str, ...]] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fit configured methods at the final origin and return station forecasts."""

    if set(selected_methods) != set(TARGET_COLUMNS):
        raise ValueError("selected_methods must map every target metric exactly once")
    invalid_methods = set(selected_methods.values()).difference(BASE_MODEL_METHODS)
    if invalid_methods:
        raise ValueError(
            "unknown model method(s): " + ", ".join(sorted(invalid_methods))
        )
    metadata = station_metadata(city_info)
    frames: list[pd.DataFrame] = []
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
        method = selected_methods[metric]
        predictions, _ = fit_predict_origin(
            panel,
            origin,
            model_config,
            methods=(method,),
            feature_columns=feature_columns_for_metric(
                metric, feature_columns, static_features_by_metric
            ),
        )
        predictions["metric"] = metric
        frames.append(predictions)
    result = pd.concat(frames, ignore_index=True)

    if use_sparse_fallback:
        result = apply_sparse_station_fallback(
            result,
            cleaned,
            metadata,
            minimum_valid_years=minimum_valid_years,
            neighbors=neighbors,
            origin=origin,
            baseline_window=model_config.baseline_window,
        )

    calibration_predictions = backtest.loc[
        (backtest["origin"] < origin) & (backtest["target_year"] <= origin)
    ]
    result = apply_lead_calibration(
        result,
        calibration_predictions,
        methods=tuple(dict.fromkeys(selected_methods.values())),
    )

    forecast_rows: list[pd.DataFrame] = []
    for metric, method in selected_methods.items():
        selected = result.loc[result["metric"] == metric]
        forecast_rows.append(
            pd.DataFrame(
                {
                    "city_id": selected["city_id"].astype(str),
                    "year": selected["target_year"].astype(int),
                    "metric": metric,
                    "predicted_value": selected[method].astype(float),
                }
            )
        )
    return pd.concat(forecast_rows, ignore_index=True), selected_methods
