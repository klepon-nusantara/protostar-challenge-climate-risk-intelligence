"""Cleaning and audit helpers for the annual station-index data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET_COLUMNS = {
    "annual_avg_temp": "Avg Temp",
    "total_precip_in": "ToT Prcp",
    "hottest_day_temp": "Max Tmax",
    "cdd": "cdd",
}

MISSING_DAY_COLUMNS = {
    "annual_avg_temp": ("Miss Tmax", "Miss Tmin"),
    "total_precip_in": ("Miss Prcp",),
    "hottest_day_temp": ("Miss Tmax",),
    "cdd": ("Miss Tmax", "Miss Tmin"),
}

REQUIRED_COLUMNS = {
    "ID",
    "Name",
    "Year",
    *(column for columns in MISSING_DAY_COLUMNS.values() for column in columns),
    *TARGET_COLUMNS.values(),
}


@dataclass
class CleaningResult:
    """Cleaned annual indices and per-target missing-data counts."""

    data: pd.DataFrame
    audit: pd.DataFrame


def clean_annual_indices(
    annual_indices: pd.DataFrame, missing_day_tolerance: int = 60
) -> CleaningResult:
    """Replace invalid values and mask years with too many missing days."""

    if missing_day_tolerance < 0:
        raise ValueError("missing_day_tolerance must be nonnegative")
    missing_columns = REQUIRED_COLUMNS.difference(annual_indices.columns)
    if missing_columns:
        raise ValueError(
            "annual_indices is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    frame = annual_indices.copy()
    frame["Year"] = pd.to_numeric(frame["Year"], errors="coerce")
    if frame["Year"].isna().any():
        raise ValueError("annual_indices contains missing or invalid Year values")
    frame["Year"] = frame["Year"].astype(int)
    if frame[["ID", "Name"]].isna().any().any():
        raise ValueError("annual_indices contains missing station IDs or names")
    if frame.duplicated(["ID", "Year"]).any():
        raise ValueError("annual_indices contains duplicate ID/year rows")

    numeric_columns = set(TARGET_COLUMNS.values())
    numeric_columns.update(column for columns in MISSING_DAY_COLUMNS.values() for column in columns)
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[list(numeric_columns)] = frame[list(numeric_columns)].replace(
        [np.inf, -np.inf], np.nan
    )

    raw_missing = {
        metric: int(frame[column].isna().sum())
        for metric, column in TARGET_COLUMNS.items()
    }
    flagged_rows: dict[str, pd.Series] = {}
    for metric, missing_columns in MISSING_DAY_COLUMNS.items():
        flagged = pd.Series(False, index=frame.index)
        for column in missing_columns:
            flagged |= frame[column].isna() | (frame[column] > missing_day_tolerance)
        flagged_rows[metric] = flagged
        frame.loc[flagged, TARGET_COLUMNS[metric]] = np.nan

    audit = pd.DataFrame(
        {
            "raw_missing": pd.Series(raw_missing),
            "flagged_by_missing_days": pd.Series(
                {metric: int(flagged.sum()) for metric, flagged in flagged_rows.items()}
            ),
            "cleaned_missing": pd.Series(
                {
                    metric: int(frame[column].isna().sum())
                    for metric, column in TARGET_COLUMNS.items()
                }
            ),
        }
    )
    return CleaningResult(
        data=frame.sort_values(["ID", "Year"]).reset_index(drop=True),
        audit=audit,
    )


def metric_matrix(cleaned: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return one metric as a year-by-station matrix."""

    if metric not in TARGET_COLUMNS:
        raise ValueError(f"Unknown metric code: {metric!r}")
    column = TARGET_COLUMNS[metric]
    return cleaned.pivot(index="Year", columns="ID", values=column).sort_index()


def station_metadata(city_info: pd.DataFrame) -> pd.DataFrame:
    """Return one coordinate/name row per station ID."""

    required = {"ID", "Name", "Lat", "Lon"}
    missing = required.difference(city_info.columns)
    if missing:
        raise ValueError(
            "city_info is missing required columns: " + ", ".join(sorted(missing))
        )
    metadata = city_info.dropna(subset=["ID", "Lat", "Lon"]).copy()
    metadata["Lat"] = pd.to_numeric(metadata["Lat"], errors="coerce")
    metadata["Lon"] = pd.to_numeric(metadata["Lon"], errors="coerce")
    return (
        metadata.dropna(subset=["Lat", "Lon"])
        .drop_duplicates("ID")
        .set_index("ID")[["Name", "Lat", "Lon"]]
    )
