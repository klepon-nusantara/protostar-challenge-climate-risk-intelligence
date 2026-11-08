"""Challenge-format submission assembly and validation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

SUBMISSION_COLUMNS = ["city_id", "year", "metric", "predicted_value"]
EXPECTED_METRICS = {
    "annual_avg_temp",
    "total_precip_in",
    "hottest_day_temp",
    "cdd",
}
FORECAST_YEARS = set(range(2019, 2026))
KEY_COLUMNS = ["city_id", "year", "metric"]


def assemble_submission(
    forecasts: pd.DataFrame, template: pd.DataFrame
) -> pd.DataFrame:
    """Fill the provided challenge template, preserving its exact row order."""

    if list(template.columns) != SUBMISSION_COLUMNS:
        raise ValueError(
            f"Template columns must be exactly {','.join(SUBMISSION_COLUMNS)}"
        )
    if not set(KEY_COLUMNS + ["predicted_value"]).issubset(forecasts.columns):
        raise ValueError("Forecasts must contain city_id, year, metric, predicted_value")

    template_keys = template[KEY_COLUMNS].copy()
    forecast_values = forecasts[KEY_COLUMNS + ["predicted_value"]].copy()
    template_keys["city_id"] = template_keys["city_id"].astype(str)
    forecast_values["city_id"] = forecast_values["city_id"].astype(str)
    if forecast_values.duplicated(KEY_COLUMNS).any():
        raise ValueError("Forecasts contain duplicate city/year/metric rows")
    template_key_set = set(template_keys.itertuples(index=False, name=None))
    forecast_key_set = set(forecast_values[KEY_COLUMNS].itertuples(index=False, name=None))
    if template_key_set != forecast_key_set:
        missing = len(template_key_set - forecast_key_set)
        unexpected = len(forecast_key_set - template_key_set)
        raise ValueError(
            f"Forecast keys do not match the template ({missing} missing, "
            f"{unexpected} unexpected)"
        )

    assembled = template_keys.merge(
        forecast_values, how="left", on=KEY_COLUMNS, validate="one_to_one", sort=False
    )
    result = assembled[SUBMISSION_COLUMNS]
    validate_submission(result, expected_city_ids=template_keys["city_id"].unique())
    return result


def validate_submission(
    submission: pd.DataFrame, expected_city_ids: Iterable[str] | None = None
) -> None:
    """Raise a useful error if the final CSV fails challenge schema checks."""

    if list(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(
            f"Columns must be exactly {','.join(SUBMISSION_COLUMNS)}"
        )
    if submission[KEY_COLUMNS].isna().any().any():
        raise ValueError("Submission keys must not contain missing values")
    if submission.duplicated(KEY_COLUMNS).any():
        raise ValueError("Submission contains duplicate city/year/metric rows")
    if not submission["year"].isin(FORECAST_YEARS).all():
        raise ValueError("Submission contains years outside 2019-2025")
    if not submission["metric"].isin(EXPECTED_METRICS).all():
        raise ValueError("Submission contains unknown metric codes")

    values = pd.to_numeric(submission["predicted_value"], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Submission forecasts must all be finite numeric values")
    nonnegative_metrics = submission["metric"].isin({"total_precip_in", "cdd"})
    if (values[nonnegative_metrics] < 0).any():
        raise ValueError("Precipitation and cooling-degree-day forecasts must be nonnegative")

    if expected_city_ids is not None:
        expected_ids = set(map(str, expected_city_ids))
        actual_ids = set(submission["city_id"].astype(str))
        if actual_ids != expected_ids:
            raise ValueError("Submission city IDs do not match the template")
        expected_rows = len(expected_ids) * len(FORECAST_YEARS) * len(EXPECTED_METRICS)
        if len(submission) != expected_rows:
            raise ValueError(
                f"Submission has {len(submission)} rows; expected {expected_rows}"
            )


def write_submission(
    forecasts: pd.DataFrame,
    template_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Fill the existing CSV template and write the validated submission."""

    template = pd.read_csv(template_path, dtype={"city_id": str, "metric": str})
    submission = assemble_submission(forecasts, template)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    return submission
