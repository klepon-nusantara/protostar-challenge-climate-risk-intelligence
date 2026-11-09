"""Preprocess manually supplied 7890488 station CSVs for OOT evaluation."""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import numpy as np
import pandas as pd

DAILY_COLUMNS = ("Date", "tmax", "tmin", "prcp")
OOT_START_DATE = pd.Timestamp("2019-01-01")
OOT_START_YEAR = OOT_START_DATE.year


def _find_csv_directory(input_dir: Path) -> Path:
    if (input_dir / "city_info.csv").is_file():
        return input_dir
    candidates = (
        [
            path
            for path in input_dir.iterdir()
            if path.is_dir() and (path / "city_info.csv").is_file()
        ]
        if input_dir.is_dir()
        else []
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find city_info.csv in {input_dir} or its immediate subfolders"
        )
    raise ValueError(
        f"Found multiple station-data folders in {input_dir}; pass one explicitly"
    )


def _read_city_info(path: Path) -> pd.DataFrame:
    city_info = pd.read_csv(path, dtype={"ID": "string"})
    city_info = city_info.loc[
        :, [column for column in city_info.columns if column.strip()]
    ].copy()
    required = {"Name", "ID", "Lat", "Lon", "Stn.Name", "Stn.stDate", "Stn.edDate"}
    missing = required.difference(city_info.columns)
    if missing:
        raise ValueError(
            "city_info.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )
    city_info["ID"] = city_info["ID"].str.strip()
    for column in ("Lat", "Lon"):
        city_info[column] = pd.to_numeric(city_info[column], errors="coerce")
    if city_info[["ID", "Name", "Lat", "Lon"]].isna().any().any():
        raise ValueError("city_info.csv contains missing IDs, names, or coordinates")
    for station_id, group in city_info.groupby("ID", sort=False):
        for column in ("Name", "Lat", "Lon"):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(
                    f"city_info.csv has inconsistent {column} values for {station_id}"
                )
    for column in ("Stn.stDate", "Stn.edDate"):
        city_info[column] = pd.to_datetime(city_info[column], errors="coerce")
        if city_info[column].isna().any():
            raise ValueError(f"city_info.csv has invalid {column} values")
    return city_info


def _read_station_daily(path: Path) -> pd.DataFrame:
    try:
        daily = pd.read_csv(path, usecols=list(DAILY_COLUMNS), na_values=["NA", ""])
    except ValueError as error:
        raise ValueError(f"{path} must contain columns {DAILY_COLUMNS}") from error
    daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce")
    if daily["Date"].isna().any():
        raise ValueError(f"{path} contains missing or invalid dates")
    if daily["Date"].duplicated().any():
        duplicate = daily.loc[daily["Date"].duplicated(), "Date"].iloc[0].date()
        raise ValueError(f"{path} contains duplicate daily date {duplicate}")

    for column in ("tmax", "tmin", "prcp"):
        original = daily[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & numeric.isna()
        if invalid.any():
            raise ValueError(f"{path} contains invalid numeric values in {column}")
        if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"{path} contains infinite values in {column}")
        daily[column] = numeric
    return daily.sort_values("Date").set_index("Date")


def _annualize_station(path: Path, city_info: pd.DataFrame) -> pd.DataFrame:
    station_id = path.stem
    station_rows = city_info.loc[city_info["ID"] == station_id]
    if station_rows.empty:
        raise ValueError(f"No city_info.csv metadata for station {station_id}")
    daily = _read_station_daily(path)
    if daily.empty:
        raise ValueError(f"{path} contains no daily records")

    years = pd.Index(range(daily.index.min().year, daily.index.max().year + 1))
    complete_dates = pd.date_range(
        f"{years.min()}-01-01", f"{years.max()}-12-31", freq="D"
    )
    daily = daily.reindex(complete_dates)
    daily.index.name = "Date"
    max_mean = daily.groupby(daily.index.year, sort=True)["tmax"].mean()
    min_mean = daily.groupby(daily.index.year, sort=True)["tmin"].mean()
    complete_day_mean = (daily["tmax"] + daily["tmin"]) / 2
    daily["daily_cdd"] = (complete_day_mean - 65).clip(lower=0)
    daily["daily_cdd"] = daily["daily_cdd"].where(complete_day_mean.notna())
    grouped = daily.groupby(daily.index.year, sort=True)
    days_in_year = pd.Series(
        [366 if calendar.isleap(int(year)) else 365 for year in years],
        index=years,
        dtype=int,
    )
    annual = pd.DataFrame(
        {
            "ID": station_id,
            "Name": station_rows["Name"].iloc[0],
            "Lat": station_rows["Lat"].iloc[0],
            "Lon": station_rows["Lon"].iloc[0],
            "Year": years.to_numpy(dtype=int),
            "Avg Temp": (max_mean + min_mean).reindex(years).to_numpy() / 2,
            "ToT Prcp": grouped["prcp"].sum(min_count=1).reindex(years).to_numpy(),
            "Max Tmax": grouped["tmax"].max().reindex(years).to_numpy(),
            "cdd": grouped["daily_cdd"].sum(min_count=1).reindex(years).to_numpy(),
            "Miss Tmax": (
                days_in_year - grouped["tmax"].count().reindex(years).fillna(0)
            ).to_numpy(dtype=int),
            "Miss Tmin": (
                days_in_year - grouped["tmin"].count().reindex(years).fillna(0)
            ).to_numpy(dtype=int),
            "Miss Prcp": (
                days_in_year - grouped["prcp"].count().reindex(years).fillna(0)
            ).to_numpy(dtype=int),
        }
    )
    return annual


def preprocess_oot_eval(
    input_dir: str | Path = "data/7890488",
    output_dir: str | Path = "data",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write only post-2018 station-year observations and station periods."""

    csv_dir = _find_csv_directory(Path(input_dir))
    city_info = _read_city_info(csv_dir / "city_info.csv")
    station_files = sorted(
        path
        for path in csv_dir.glob("*.csv")
        if path.name != "city_info.csv" and not path.name.endswith(".fill.csv")
    )
    if not station_files:
        raise FileNotFoundError(f"No station CSV files found in {csv_dir}")

    file_ids = {path.stem for path in station_files}
    metadata_ids = set(city_info["ID"])
    missing_metadata = sorted(file_ids.difference(metadata_ids))
    missing_files = sorted(metadata_ids.difference(file_ids))
    if missing_metadata or missing_files:
        details = []
        if missing_metadata:
            details.append(f"station files without metadata: {missing_metadata[:5]}")
        if missing_files:
            details.append(f"metadata IDs without station files: {missing_files[:5]}")
        raise ValueError("Station ID mismatch: " + "; ".join(details))

    all_annual_indices = pd.concat(
        [_annualize_station(path, city_info) for path in station_files],
        ignore_index=True,
    ).sort_values(["ID", "Year"], ignore_index=True)
    if all_annual_indices.duplicated(["ID", "Year"]).any():
        raise ValueError("Preprocessing produced duplicate station/year rows")

    annual_indices = all_annual_indices.loc[
        all_annual_indices["Year"] >= OOT_START_YEAR
    ].reset_index(drop=True)
    if annual_indices.empty:
        raise ValueError(
            f"No annual observations from {OOT_START_YEAR} onward were found"
        )

    city_info_oot = city_info.loc[
        city_info["Stn.edDate"] >= OOT_START_DATE
    ].copy()
    city_info_oot["Stn.stDate"] = city_info_oot["Stn.stDate"].clip(
        lower=OOT_START_DATE
    )
    if city_info_oot.empty:
        raise ValueError(
            f"No station metadata periods overlap {OOT_START_DATE.date()} onward"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    annual_indices.to_parquet(output_path / "annual_indices_oot.parquet", index=False)
    city_info_oot.to_parquet(output_path / "city_info_oot.parquet", index=False)
    return annual_indices, city_info_oot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert local daily station CSVs into OOT evaluation Parquet files."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/7890488"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    annual, city_info = preprocess_oot_eval(args.input_dir, args.output_dir)
    years = annual["Year"]
    print(
        f"Wrote annual_indices_oot.parquet ({len(annual):,} OOT station-years, "
        f"{years.min()}-{years.max()}) and city_info_oot.parquet "
        f"({len(city_info):,} OOT station-period rows) to {args.output_dir}."
    )


if __name__ == "__main__":
    main()
