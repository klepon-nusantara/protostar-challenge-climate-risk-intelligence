"""Shared default settings for the climate forecasting pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


BASE_MODEL_METHODS = (
    "Last Year",
    "30-Yr Avg",
    "Linear Trend",
    "Half-Trend",
    "Ridge",
    "LightGBM",
    "Ensemble",
)
MODEL_METHODS = BASE_MODEL_METHODS

# Forecasting uses these configured methods instead of selecting them by backtest.
FORECAST_METHODS = {
    "annual_avg_temp": "LightGBM",
    "cdd": "Ridge",
    "hottest_day_temp": "Ridge",
    "total_precip_in": "LightGBM",
}
# Retain static features only for targets where the calibrated 2018 OOT
# evaluation improved MAE. Empty tuples retain the annual/daily baseline.
STATIC_FEATURES_BY_METRIC = {
    "annual_avg_temp": ("distance_to_coast_km", "climate_division_anomaly"),
    "cdd": ("distance_to_coast_km", "climate_division_anomaly"),
    "hottest_day_temp": ("distance_to_coast_km", "climate_division_anomaly"),
    "total_precip_in": ("distance_to_coast_km", "climate_division_anomaly"),
}
# None uses every available non-static feature; static features are selected
# separately by STATIC_FEATURES_BY_METRIC in the configured workflows.
FORECAST_FEATURES: tuple[str, ...] | None = None


@dataclass(frozen=True)
class CleaningConfig:
    missing_day_tolerance: int = 60


@dataclass(frozen=True)
class BacktestConfig:
    start_origin: int = 1995
    end_origin: int = 2011
    horizon: int = 7


@dataclass(frozen=True)
class GlobalModelConfig:
    """Model parameters; tune them only inside temporal validation."""

    first_origin: int = 1925
    baseline_window: int = 30
    average_5_years: int = 5
    average_10_years: int = 10
    minimum_trend_points: int = 15
    ridge_alpha: float = 10.0
    lightgbm_estimators: int = 300
    lightgbm_learning_rate: float = 0.03
    lightgbm_num_leaves: int = 15
    lightgbm_min_child_samples: int = 100
    lightgbm_subsample: float = 0.8
    lightgbm_subsample_freq: int = 1
    lightgbm_colsample_bytree: float = 0.8
    lightgbm_reg_lambda: float = 1.0
    random_state: int = 42


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_samples: int = 1000
    random_state: int = 42


@dataclass(frozen=True)
class ForecastConfig:
    origin: int = 2018
    horizon: int = 7
    use_sparse_fallback: bool = True
    minimum_valid_years: int = 15
    neighbor_count: int = 3


@dataclass(frozen=True)
class SubmissionConfig:
    template_path: str = "submission.csv"
    output_path: str = "submission.csv"


@dataclass(frozen=True)
class PipelineConfig:
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    model: GlobalModelConfig = field(default_factory=GlobalModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    submission: SubmissionConfig = field(default_factory=SubmissionConfig)


DEFAULT_CONFIG = PipelineConfig()
