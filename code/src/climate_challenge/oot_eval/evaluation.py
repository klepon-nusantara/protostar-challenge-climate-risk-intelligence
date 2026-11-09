"""Fixed-origin model evaluation on held-out out-of-time observations."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..backtesting import run_backtest
from ..cleaning import TARGET_COLUMNS, clean_annual_indices
from ..config import DEFAULT_CONFIG, MODEL_METHODS, GlobalModelConfig
from ..evaluation import evaluate_backtest
from ..models.global_models import BASE_FEATURES, DAILY_FEATURES

OOT_ORIGIN = DEFAULT_CONFIG.forecast.origin
OOT_MAX_HORIZON = DEFAULT_CONFIG.forecast.horizon


def run_oot_evaluation(
    historical_annual_indices: pd.DataFrame,
    oot_annual_indices: pd.DataFrame,
    city_info: pd.DataFrame,
    origin: int = OOT_ORIGIN,
    max_horizon: int = OOT_MAX_HORIZON,
    model_config: GlobalModelConfig = DEFAULT_CONFIG.model,
    daily_observations: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate every configured method at a fixed origin on later observations."""

    if max_horizon < 1:
        raise ValueError("max_horizon must be at least 1")
    historical_years = pd.to_numeric(
        historical_annual_indices["Year"], errors="coerce"
    )
    oot_years = pd.to_numeric(oot_annual_indices["Year"], errors="coerce")
    if historical_years.isna().any() or oot_years.isna().any():
        raise ValueError("Historical and OOT annual data must have valid Year values")
    if (historical_years > origin).any():
        raise ValueError(
            f"Historical annual data contains years after the OOT origin {origin}"
        )
    if (oot_years <= origin).any():
        raise ValueError(
            f"OOT annual data must contain only years after origin {origin}"
        )
    if daily_observations is not None:
        required_daily_columns = {"Date"}
        missing_daily_columns = required_daily_columns.difference(
            daily_observations.columns
        )
        if missing_daily_columns:
            raise ValueError(
                "Daily observations are missing required columns: "
                + ", ".join(sorted(missing_daily_columns))
            )
        daily_dates = pd.to_datetime(
            daily_observations["Date"], errors="coerce"
        )
        if daily_dates.isna().any():
            raise ValueError("Daily observations contain invalid dates")
        if (daily_dates.dt.year > origin).any():
            raise ValueError(
                f"Daily feature data contains observations after OOT origin {origin}"
            )
    annual_indices = pd.concat(
        [historical_annual_indices, oot_annual_indices], ignore_index=True
    )
    cleaned = clean_annual_indices(annual_indices)
    latest_year = int(cleaned.data["Year"].max())
    horizon = min(max_horizon, latest_year - origin)
    if horizon < 1:
        raise ValueError(
            f"OOT data must include at least one year after origin {origin}"
        )
    targets = cleaned.data.loc[
        (cleaned.data["Year"] > origin)
        & (cleaned.data["Year"] <= origin + horizon),
        list(TARGET_COLUMNS.values()),
    ]
    if not targets.notna().any().any():
        raise ValueError(
            "No valid cleaned target observations are available after origin"
        )

    backtest = run_backtest(
        cleaned.data,
        city_info,
        origins=[origin],
        horizon=horizon,
        model_config=model_config,
        daily_observations=daily_observations,
    )
    scores = evaluate_backtest(backtest, by_horizon=True)
    if set(scores["method"]) != set(MODEL_METHODS):
        raise RuntimeError("OOT evaluation did not score every configured model method")
    return backtest, scores, cleaned.audit


def append_oot_run_log(
    log_path: str | Path,
    backtest: pd.DataFrame,
    scores: pd.DataFrame,
    audit: pd.DataFrame,
    origin: int,
    model_config: GlobalModelConfig = DEFAULT_CONFIG.model,
    input_paths: tuple[str | Path, ...] = (),
    run_label: str | None = None,
) -> Path:
    """Append model settings, scored period, audit, and results to a run log."""

    required = {"target_year", "city_id"}
    missing = required.difference(backtest.columns)
    if missing:
        raise ValueError(
            "backtest is missing required log columns: " + ", ".join(sorted(missing))
        )
    if backtest.empty:
        raise ValueError("Cannot log an empty OOT backtest")

    target_rows = backtest[["target_year", "city_id"]].drop_duplicates()
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
        output.write(f"OOT evaluation run: {datetime.now().astimezone().isoformat()}\n")
        output.write(f"Origin: {origin}\n")
        output.write(f"Training data cutoff: {origin}\n")
        if run_label:
            output.write(f"Run variant: {run_label}\n")
        output.write(
            f"Scored period: {int(backtest['target_year'].min())}-"
            f"{int(backtest['target_year'].max())}\n"
        )
        output.write(
            f"Target station-years: {len(target_rows)}; "
            f"stations: {target_rows['city_id'].nunique()}\n"
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
        output.write("Scores (MAE and scored rows, by metric and lead):\n")
        output.write(
            scores.to_string(index=False, float_format=lambda value: f"{value:.6f}")
            + "\n"
        )
        output.write("Overall pooled MAE across the full scored period:\n")
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
