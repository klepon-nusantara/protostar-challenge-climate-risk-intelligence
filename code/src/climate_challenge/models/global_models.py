"""Global Ridge and LightGBM models trained across stations and origins."""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import DEFAULT_CONFIG, GlobalModelConfig
from .baselines import recent_average, trend_coefficients

FEATURES = (
    "ly_a",
    "a5_a",
    "a10_a",
    "trend_off",
    "slope",
    "sd30",
    "lat",
    "lon",
    "h",
    "a30",
    "nat_ly",
    "nat_a5",
)


def build_panel(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    horizon: int = DEFAULT_CONFIG.backtest.horizon,
    config: GlobalModelConfig = DEFAULT_CONFIG.model,
) -> pd.DataFrame:
    """Build leakage-safe city/origin/horizon examples for one metric."""

    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if not {"Lat", "Lon"}.issubset(metadata.columns):
        raise ValueError("metadata must contain Lat and Lon columns")
    station_ids = matrix.columns
    missing_metadata = station_ids.difference(metadata.index)
    if len(missing_metadata):
        raise ValueError(
            f"Missing coordinates for {len(missing_metadata)} station IDs"
        )

    final_observed_year = int(matrix.index.max())
    first_origin = max(
        config.first_origin, int(matrix.index.min()) + config.baseline_window - 1
    )
    feature_years = pd.RangeIndex(int(matrix.index.min()), final_observed_year + horizon + 1)
    values = matrix.reindex(feature_years)
    records: list[pd.DataFrame] = []

    for origin in range(first_origin, final_observed_year + 1):
        average_30 = recent_average(values, origin, config.baseline_window)
        average_10 = recent_average(values, origin, config.average_10_years)
        average_5 = recent_average(values, origin, config.average_5_years)
        latest = values.loc[:origin].ffill().iloc[-1]
        variability = values.loc[
            origin - config.baseline_window + 1 : origin
        ].std()
        coefficients = trend_coefficients(
            values,
            origin,
            window=config.baseline_window,
            min_points=config.minimum_trend_points,
        )
        trend_slope = coefficients["slope"]
        national_last = (latest - average_30).mean()
        national_5 = (average_5 - average_30).mean()

        for lead in range(1, horizon + 1):
            target_year = origin + lead
            actual = values.loc[target_year]
            trend = (
                coefficients["slope"] * target_year
                + coefficients["intercept"]
            )
            records.append(
                pd.DataFrame(
                    {
                        "city_id": station_ids,
                        "origin": origin,
                        "h": lead,
                        "target_year": target_year,
                        "y": actual.reindex(station_ids).to_numpy(),
                        "a30": average_30.reindex(station_ids).to_numpy(),
                        "ly_a": (latest - average_30).reindex(station_ids).to_numpy(),
                        "a5_a": (average_5 - average_30).reindex(station_ids).to_numpy(),
                        "a10_a": (average_10 - average_30).reindex(station_ids).to_numpy(),
                        "trend_off": (trend - average_30).reindex(station_ids).to_numpy(),
                        "slope": trend_slope.reindex(station_ids).to_numpy(),
                        "sd30": variability.reindex(station_ids).to_numpy(),
                        "lat": metadata.reindex(station_ids)["Lat"].to_numpy(),
                        "lon": metadata.reindex(station_ids)["Lon"].to_numpy(),
                        "nat_ly": national_last,
                        "nat_a5": national_5,
                        "Last Year": latest.reindex(station_ids).to_numpy(),
                        "30-Yr Avg": average_30.reindex(station_ids).to_numpy(),
                        "Linear Trend": trend.reindex(station_ids).to_numpy(),
                    }
                )
            )

    if not records:
        raise ValueError(
            f"No training origins available (first origin {first_origin}, "
            f"last observed year {final_observed_year})"
        )
    return pd.concat(records, ignore_index=True)


def fit_predict_origin(
    panel: pd.DataFrame,
    origin: int,
    config: GlobalModelConfig = DEFAULT_CONFIG.model,
) -> tuple[pd.DataFrame, lgb.LGBMRegressor]:
    """Fit models using only labels available at origin and forecast its leads."""

    train = panel.loc[
        (panel["target_year"] <= origin)
        & panel["y"].notna()
        & panel["a30"].notna()
        & (panel["origin"] >= config.first_origin)
    ]
    test = panel.loc[panel["origin"] == origin].copy()
    if train.empty:
        raise ValueError(f"No training labels are available at origin {origin}")
    if test.empty:
        raise ValueError(f"No forecast rows are available at origin {origin}")

    training_target = (train["y"] - train["a30"]).to_numpy()
    ridge = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=config.ridge_alpha),
    ).fit(train.loc[:, FEATURES], training_target)
    lightgbm = lgb.LGBMRegressor(
        objective="l1",
        n_estimators=config.lightgbm_estimators,
        learning_rate=config.lightgbm_learning_rate,
        num_leaves=config.lightgbm_num_leaves,
        min_child_samples=config.lightgbm_min_child_samples,
        subsample=config.lightgbm_subsample,
        subsample_freq=config.lightgbm_subsample_freq,
        colsample_bytree=config.lightgbm_colsample_bytree,
        reg_lambda=config.lightgbm_reg_lambda,
        random_state=config.random_state,
        verbosity=-1,
        n_jobs=1,
    ).fit(train.loc[:, FEATURES], training_target)

    test["Ridge"] = test["a30"] + ridge.predict(test.loc[:, FEATURES])
    test["LightGBM"] = test["a30"] + lightgbm.predict(test.loc[:, FEATURES])
    test["Half-Trend"] = 0.5 * (test["30-Yr Avg"] + test["Linear Trend"])
    test["Ensemble"] = (
        test["LightGBM"] + test["Ridge"] + test["Linear Trend"]
    ) / 3
    return test, lightgbm
