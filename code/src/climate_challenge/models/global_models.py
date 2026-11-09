"""Global Ridge and LightGBM models trained across stations and origins."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import DEFAULT_CONFIG, GlobalModelConfig
from .baselines import recent_average, trend_coefficients

REGIONS = ("midwest", "southern", "west_coast", "new_england", "other")
BASE_FEATURES = (
    "ly_a",
    "a5_a",
    "a10_a",
    "trend_off",
    "slope",
    "sd30",
    "years_since_last",
    "slope10",
    "acceleration",
    "spatial_anomaly",
    "regional_anomaly",
    *(f"region_{region}" for region in REGIONS),
    "lat",
    "lon",
    "h",
    "a30",
    "nat_ly",
    "nat_a5",
)
DAILY_SUMMARY_COLUMNS = (
    *(f"{season}_temp" for season in ("djf", "mam", "jja", "son")),
    *(f"{season}_prcp" for season in ("djf", "mam", "jja", "son")),
    "hot_day_fraction",
    "cold_day_fraction",
    "heavy_rain_fraction",
    "wet_day_fraction",
    "wet_day_intensity",
    "max_daily_prcp",
    "longest_dry_spell",
)
DAILY_FEATURES = tuple(
    f"daily_{column}_a5" for column in DAILY_SUMMARY_COLUMNS
)
FEATURES = (*BASE_FEATURES, *DAILY_FEATURES)


def _daily_feature_matrices(
    daily_observations: pd.DataFrame,
    station_ids: pd.Index,
) -> dict[str, pd.DataFrame]:
    """Summarize daily observations into station-year feature matrices."""

    required = {"ID", "Date", "tmax", "tmin", "prcp"}
    missing = required.difference(daily_observations.columns)
    if missing:
        raise ValueError(
            "daily observations are missing required columns: "
            + ", ".join(sorted(missing))
        )
    daily = daily_observations.loc[:, ["ID", "Date", "tmax", "tmin", "prcp"]].copy()
    daily["ID"] = daily["ID"].astype(str)
    daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce")
    if daily[["ID", "Date"]].isna().any().any():
        raise ValueError("daily observations contain missing IDs or invalid dates")
    if daily.duplicated(["ID", "Date"]).any():
        raise ValueError("daily observations contain duplicate station dates")
    daily = daily.loc[daily["ID"].isin(station_ids)].sort_values(
        ["ID", "Date"], ignore_index=True
    )
    if daily.empty:
        raise ValueError("daily observations do not cover any modeled stations")

    for column in ("tmax", "tmin", "prcp"):
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    daily["Year"] = daily["Date"].dt.year
    daily["month"] = daily["Date"].dt.month
    daily["daily_mean_temp"] = (daily["tmax"] + daily["tmin"]) / 2
    daily["season"] = daily["month"].map(
        {
            12: "djf",
            1: "djf",
            2: "djf",
            3: "mam",
            4: "mam",
            5: "mam",
            6: "jja",
            7: "jja",
            8: "jja",
            9: "son",
            10: "son",
            11: "son",
        }
    )

    seasonal = daily.groupby(["ID", "Year", "season"], sort=False).agg(
        temperature=("daily_mean_temp", "mean"),
        precipitation=("prcp", "mean"),
    )
    seasonal_wide = seasonal.unstack("season")
    seasonal_wide.columns = [
        f"{season}_{measure}"
        for measure, season in seasonal_wide.columns
    ]
    seasonal_wide = seasonal_wide.rename(
        columns={
            f"{season}_temperature": f"{season}_temp"
            for season in ("djf", "mam", "jja", "son")
        }
        | {
            f"{season}_precipitation": f"{season}_prcp"
            for season in ("djf", "mam", "jja", "son")
        }
    )

    daily["_hot"] = (daily["tmax"] > 90.0).astype(float).where(
        daily["tmax"].notna()
    )
    daily["_cold"] = (daily["tmin"] < 32.0).astype(float).where(
        daily["tmin"].notna()
    )
    daily["_heavy_rain"] = (daily["prcp"] >= 1.0).astype(float).where(
        daily["prcp"].notna()
    )
    daily["_wet"] = (daily["prcp"] >= 0.01).astype(float).where(
        daily["prcp"].notna()
    )
    daily["_wet_prcp"] = daily["prcp"].where(daily["prcp"] >= 0.01)
    annual = daily.groupby(["ID", "Year"], sort=False).agg(
        hot_day_fraction=("_hot", "mean"),
        cold_day_fraction=("_cold", "mean"),
        heavy_rain_fraction=("_heavy_rain", "mean"),
        wet_day_fraction=("_wet", "mean"),
        wet_day_intensity=("_wet_prcp", "mean"),
        max_daily_prcp=("prcp", "max"),
    )

    dry = daily["prcp"].notna() & daily["prcp"].lt(0.01)
    grouped = daily.groupby(["ID", "Year"], sort=False)
    previous_date = grouped["Date"].shift()
    previous_dry = dry.groupby([daily["ID"], daily["Year"]], sort=False).shift(
        fill_value=False
    )
    new_dry_run = ~(
        dry
        & previous_dry
        & daily["Date"].sub(previous_date).eq(pd.Timedelta(days=1))
    )
    run_number = new_dry_run.astype(int).groupby(
        [daily["ID"], daily["Year"]], sort=False
    ).cumsum()
    dry_runs = daily.loc[dry, ["ID", "Year"]].assign(
        run_number=run_number.loc[dry].to_numpy()
    )
    longest_dry_spell = (
        dry_runs.groupby(["ID", "Year", "run_number"], sort=False)
        .size()
        .groupby(level=["ID", "Year"])
        .max()
        .rename("longest_dry_spell")
    )
    annual = annual.join(longest_dry_spell)
    summary = seasonal_wide.join(annual).reindex(columns=DAILY_SUMMARY_COLUMNS)

    matrices = {
        column: summary[column]
        .unstack("ID")
        .sort_index()
        .reindex(columns=station_ids)
        for column in DAILY_SUMMARY_COLUMNS
    }
    return matrices


def _daily_features_at_origin(
    matrices: dict[str, pd.DataFrame],
    station_ids: pd.Index,
    origin: int,
    config: GlobalModelConfig,
) -> pd.DataFrame:
    """Calculate recent-minus-normal daily features using data through origin."""

    recent_start = origin - config.average_5_years + 1
    baseline_start = origin - config.baseline_window + 1
    features: dict[str, pd.Series] = {}
    for column, matrix in matrices.items():
        history = matrix.loc[baseline_start:origin].reindex(columns=station_ids)
        recent = history.loc[recent_start:origin]
        features[f"daily_{column}_a5"] = recent.mean() - history.mean()
    return pd.DataFrame(features, index=station_ids)


def _station_regions(coordinates: pd.DataFrame) -> pd.Series:
    """Assign approximate broad regions; stations outside the boxes are ``other``.

    The rules use longitude <= -117 and latitude >= 32 for the West Coast;
    latitude >= 41 and longitude -73.5..-66.5 for New England; latitude < 37
    and longitude > -107 for the South; and latitude 37..50 and longitude
    -104..-80 for the Midwest.
    """

    latitude = coordinates["Lat"]
    longitude = coordinates["Lon"]
    regions = pd.Series("other", index=coordinates.index, dtype="object")

    regions.loc[
        (longitude <= -117.0) & (latitude >= 32.0)
    ] = "west_coast"
    regions.loc[
        (latitude >= 41.0) & (longitude >= -73.5) & (longitude <= -66.5)
    ] = "new_england"
    regions.loc[
        (latitude < 37.0) & (longitude > -107.0)
    ] = "southern"
    regions.loc[
        (latitude >= 37.0)
        & (latitude < 50.0)
        & (longitude >= -104.0)
        & (longitude <= -80.0)
    ] = "midwest"
    return regions


def _regional_anomaly(
    anomalies: pd.Series,
    regions: pd.Series,
) -> pd.Series:
    """Return each station's region peer mean, excluding its own anomaly."""

    result = pd.Series(np.nan, index=anomalies.index, dtype=float)
    for region in REGIONS:
        if region == "other":
            continue
        station_ids = regions.index[regions.eq(region)]
        values = anomalies.reindex(station_ids)
        valid_count = values.notna().sum()
        peer_count = valid_count - values.notna().astype(int)
        peer_total = values.fillna(0.0).sum() - values.fillna(0.0)
        result.loc[station_ids] = peer_total.div(peer_count.where(peer_count > 0))
    return result


def _window_slope(
    matrix: pd.DataFrame,
    station_ids: pd.Index,
    window: int,
    min_points: int,
) -> pd.Series:
    """Return a per-station slope over a trailing window, if enough points exist."""

    if window < 1:
        return pd.Series(np.nan, index=station_ids, dtype=float)

    history = matrix.loc[:, station_ids]
    years = history.index.to_numpy(dtype=float)
    slopes: dict[object, float] = {}
    for station_id in station_ids:
        values = history[station_id].to_numpy(dtype=float)
        valid = np.isfinite(values)
        if valid.sum() < min_points:
            slopes[station_id] = np.nan
            continue
        slopes[station_id] = float(np.polyfit(years[valid], values[valid], 1)[0])
    return pd.Series(slopes, index=station_ids, dtype=float)


def _spatial_anomaly(
    anomalies: pd.Series,
    coordinates: pd.DataFrame,
    bandwidth_km: float = 450.0,
) -> pd.Series:
    """Compute a Gaussian distance-weighted anomaly using nearby stations."""

    station_ids = anomalies.index
    coords = coordinates.reindex(station_ids)
    if coords.empty:
        return pd.Series(np.nan, index=station_ids, dtype=float)

    lat1 = np.radians(coords["Lat"].to_numpy()[:, None])
    lon1 = np.radians(coords["Lon"].to_numpy()[:, None])
    lat2 = np.radians(coords["Lat"].to_numpy()[None, :])
    lon2 = np.radians(coords["Lon"].to_numpy()[None, :])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    distance_km = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    weights = np.exp(-0.5 * (distance_km / bandwidth_km) ** 2)
    np.fill_diagonal(weights, 0.0)

    anomaly_values = anomalies.to_numpy(dtype=float)
    valid = np.isfinite(anomaly_values)
    weights[:, ~valid] = 0.0
    totals = weights.sum(axis=1)
    weighted = weights @ np.nan_to_num(anomaly_values, nan=0.0)
    result = np.divide(
        weighted,
        totals,
        out=np.full(len(station_ids), np.nan, dtype=float),
        where=totals > 0,
    )
    return pd.Series(result, index=station_ids, dtype=float)


def build_panel(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    horizon: int = DEFAULT_CONFIG.backtest.horizon,
    config: GlobalModelConfig = DEFAULT_CONFIG.model,
    daily_observations: pd.DataFrame | None = None,
    daily_feature_matrices: dict[str, pd.DataFrame] | None = None,
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
    coordinates = metadata.reindex(station_ids)
    if daily_observations is not None and daily_feature_matrices is not None:
        raise ValueError(
            "Pass daily observations or daily feature matrices, not both"
        )
    daily_matrices = daily_feature_matrices
    if daily_observations is not None:
        daily_matrices = _daily_feature_matrices(daily_observations, station_ids)
    regions = _station_regions(coordinates)
    region_features = pd.DataFrame(
        {
            f"region_{region}": regions.eq(region).astype(float)
            for region in REGIONS
        },
        index=station_ids,
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
        short_history = values.loc[
            origin - config.average_10_years + 1 : origin
        ]
        coefficients = trend_coefficients(
            values,
            origin,
            window=config.baseline_window,
            min_points=config.minimum_trend_points,
        )
        trend_slope = coefficients["slope"]
        short_slope = _window_slope(
            short_history,
            station_ids,
            window=config.average_10_years,
            min_points=min(
                config.average_10_years,
                max(2, config.minimum_trend_points),
            ),
        )

        last_valid_year = pd.Series(
            {
                station_id: (
                    values.loc[:origin, station_id].dropna().index.max()
                    if values.loc[:origin, station_id].notna().any()
                    else np.nan
                )
                for station_id in station_ids
            },
            dtype=float,
        )
        years_since_last = (origin - last_valid_year).astype(float)
        anomalies = latest - average_30
        spatial_anomaly = _spatial_anomaly(
            anomalies.reindex(station_ids),
            coordinates,
        )
        regional_anomaly = _regional_anomaly(anomalies, regions)
        national_last = anomalies.mean()
        national_5 = (average_5 - average_30).mean()
        daily_features = (
            _daily_features_at_origin(
                daily_matrices,
                station_ids,
                origin,
                config,
            )
            if daily_matrices is not None
            else None
        )

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
                        "years_since_last": years_since_last.reindex(station_ids).to_numpy(),
                        "slope10": short_slope.reindex(station_ids).to_numpy(),
                        "acceleration": (
                            short_slope.reindex(station_ids)
                            - trend_slope.reindex(station_ids)
                        ).to_numpy(),
                        "spatial_anomaly": spatial_anomaly.reindex(station_ids).to_numpy(),
                        "regional_anomaly": regional_anomaly.reindex(station_ids).to_numpy(),
                        **{
                            column: region_features[column].to_numpy()
                            for column in region_features
                        },
                        "lat": metadata.reindex(station_ids)["Lat"].to_numpy(),
                        "lon": metadata.reindex(station_ids)["Lon"].to_numpy(),
                        "nat_ly": national_last,
                        "nat_a5": national_5,
                        **(
                            {
                                column: daily_features[column].to_numpy()
                                for column in daily_features
                            }
                            if daily_features is not None
                            else {}
                        ),
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

    feature_columns = [column for column in FEATURES if column in panel.columns]
    if not feature_columns:
        raise ValueError("No model features are available in the training panel")
    training_target = (train["y"] - train["a30"]).to_numpy()
    ridge = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=config.ridge_alpha),
    ).fit(train.loc[:, feature_columns], training_target)
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
    ).fit(train.loc[:, feature_columns], training_target)

    test["Ridge"] = test["a30"] + ridge.predict(test.loc[:, feature_columns])
    test["LightGBM"] = test["a30"] + lightgbm.predict(
        test.loc[:, feature_columns]
    )
    test["Half-Trend"] = 0.5 * (test["30-Yr Avg"] + test["Linear Trend"])
    test["Ensemble"] = (
        test["LightGBM"] + test["Ridge"] + test["Linear Trend"]
    ) / 3
    return test, lightgbm
