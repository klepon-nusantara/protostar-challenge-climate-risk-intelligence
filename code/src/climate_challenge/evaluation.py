"""Metrics and model selection for rolling-origin forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, MODEL_METHODS


def evaluate_backtest(
    backtest: pd.DataFrame, by_horizon: bool = False
) -> pd.DataFrame:
    """Return MAE and scored-row count by metric, method, and optional lead."""

    required = {"metric", "y", "h", *MODEL_METHODS}
    missing = required.difference(backtest.columns)
    if missing:
        raise ValueError(
            "backtest is missing required columns: " + ", ".join(sorted(missing))
        )

    group_columns = ["metric"] + (["h"] if by_horizon else [])
    rows: list[dict[str, object]] = []
    for keys, group in backtest.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_info = dict(zip(group_columns, keys))
        comparable = (
            group[["y", *MODEL_METHODS]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        for method in MODEL_METHODS:
            rows.append(
                {
                    **group_info,
                    "method": method,
                    "mae": float((comparable[method] - comparable["y"]).abs().mean())
                    if not comparable.empty
                    else np.nan,
                    "n": int(len(comparable)),
                }
            )
    return pd.DataFrame(rows)


def block_bootstrap_mae_difference(
    backtest: pd.DataFrame,
    metric: str,
    candidate: str,
    reference: str,
    samples: int = DEFAULT_CONFIG.evaluation.bootstrap_samples,
    seed: int = DEFAULT_CONFIG.evaluation.random_state,
) -> tuple[float, float, float]:
    """Bootstrap candidate-minus-reference MAE by resampling origin blocks."""

    if samples < 1:
        raise ValueError("samples must be positive")
    frame = backtest.loc[backtest["metric"] == metric].copy()
    required = {"origin", "y", candidate, reference}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "backtest is missing required columns: " + ", ".join(sorted(missing))
        )
    frame = frame[["origin", "y", candidate, reference]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if frame.empty:
        raise ValueError(f"No paired predictions are available for {metric}")

    errors = frame.assign(
        candidate_error=lambda data: (data[candidate] - data["y"]).abs(),
        reference_error=lambda data: (data[reference] - data["y"]).abs(),
    )
    by_origin = errors.groupby("origin").agg(
        candidate_error=("candidate_error", "sum"),
        reference_error=("reference_error", "sum"),
        n=("y", "size"),
    )
    rng = np.random.default_rng(seed)
    indices = np.arange(len(by_origin))
    differences = np.empty(samples)
    for sample in range(samples):
        chosen = rng.choice(indices, size=len(indices), replace=True)
        draw = by_origin.iloc[chosen]
        differences[sample] = (
            draw["candidate_error"].sum() - draw["reference_error"].sum()
        ) / draw["n"].sum()
    percentiles = np.percentile(differences, [5, 50, 95])
    return (
        float(percentiles[0]),
        float(percentiles[1]),
        float(percentiles[2]),
    )


def select_best_methods(backtest: pd.DataFrame) -> dict[str, str]:
    """Choose one method per metric using pooled backtest MAE."""

    metrics = evaluate_backtest(backtest)
    selected: dict[str, str] = {}
    for metric, group in metrics.groupby("metric"):
        valid = group.dropna(subset=["mae"])
        if valid.empty:
            raise ValueError(f"No valid backtest scores are available for {metric}")
        best = valid.sort_values(["mae", "method"], kind="stable").iloc[0]
        selected[str(metric)] = str(best["method"])
    return selected
