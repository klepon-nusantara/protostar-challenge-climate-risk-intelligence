"""Load and validate challenge data and optional feature sources."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfileobj
from typing import Callable, Iterable
from urllib.request import Request, urlopen

import pandas as pd

TRAINING_REPOSITORY = "CMU-OSPO/protostar-climate-train-to-2018"
TRAINING_REVISION = "bbbdebdab2a18e00f97ef87bd3f35a71e222180b"
FORECAST_CUTOFF = pd.Timestamp("2018-12-31")
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
CLIMATE_DIVISIONS_ARCHIVE = "CONUS_CLIMATE_DIVISIONS.shp.zip"
COASTLINE_ARCHIVE = "ne_10m_coastline.zip"
COASTLINE_DISTANCE_CRS = "EPSG:5070"
COASTLINE_CLIP_BOUNDS = (-140.0, 15.0, -50.0, 60.0)
ORIGINAL_FILES = {
    "annual_indices": "annual_indices/train-00000-of-00001.parquet",
    "city_info": "city_info/train-00000-of-00001.parquet",
}

ANNUAL_REQUIRED_COLUMNS = {"Name", "ID", "Year"}
CITY_REQUIRED_COLUMNS = {"Name", "ID", "Lat", "Lon"}


@dataclass(frozen=True)
class DataSource:
    """A separate data table to include in a prepared dataset.

    The loader receives the forecast cutoff so that date-limited sources can
    apply it to their download query. Weather-dependent sources must identify
    their date or year column; returned records after the cutoff are rejected.
    """

    name: str
    loader: Callable[[pd.Timestamp], pd.DataFrame]
    weather_dependent: bool = True
    date_column: str | None = None


@dataclass
class PreparedDataset:
    """Challenge tables and optional sources, kept separate until joined."""

    annual_indices: pd.DataFrame
    city_info: pd.DataFrame
    elevation: pd.DataFrame
    city_climate_divisions: pd.DataFrame
    city_coastline_distances: pd.DataFrame
    additional_sources: dict[str, pd.DataFrame]
    cutoff: pd.Timestamp
    data_dir: Path


def _validate_columns(
    frame: pd.DataFrame, required: set[str], table_name: str
) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {', '.join(sorted(missing))}"
        )


def _last_complete_year(cutoff: pd.Timestamp) -> int:
    last_complete_year = cutoff.year
    if (cutoff.month, cutoff.day) < (12, 31):
        last_complete_year -= 1
    return last_complete_year


def _validate_annual_cutoff(
    annual_indices: pd.DataFrame, cutoff: pd.Timestamp
) -> None:
    _validate_columns(annual_indices, ANNUAL_REQUIRED_COLUMNS, "annual_indices")
    years = pd.to_numeric(annual_indices["Year"], errors="coerce")
    if years.isna().any():
        raise ValueError("annual_indices contains missing or invalid Year values")
    last_complete_year = _last_complete_year(cutoff)
    if (years > last_complete_year).any():
        latest_year = int(years.max())
        raise ValueError(
            f"annual_indices contains weather data through {latest_year}, "
            f"after the forecast cutoff {cutoff.date()}"
        )


def _validate_source_cutoff(
    frame: pd.DataFrame, source: DataSource, cutoff: pd.Timestamp
) -> None:
    if not source.weather_dependent:
        return
    if source.date_column is None:
        raise ValueError(
            f"Weather-dependent source {source.name!r} must define date_column"
        )
    if source.date_column not in frame.columns:
        raise ValueError(
            f"Source {source.name!r} is missing date column "
            f"{source.date_column!r}"
        )

    values = frame[source.date_column]
    numeric_values = pd.to_numeric(values, errors="coerce")
    is_year_column = numeric_values.notna().all() and numeric_values.between(
        1000, 9999
    ).all()
    if is_year_column:
        last_complete_year = _last_complete_year(cutoff)
        is_after_cutoff = numeric_values > last_complete_year
    else:
        parsed_dates = pd.to_datetime(
            values.astype("string"), errors="coerce", utc=True
        )
        if parsed_dates.isna().any():
            raise ValueError(
                f"Source {source.name!r} has missing or invalid "
                f"{source.date_column!r} values"
            )
        cutoff_utc = (
            cutoff.tz_localize("UTC") if cutoff.tz is None else cutoff.tz_convert("UTC")
        )
        is_after_cutoff = parsed_dates > cutoff_utc

    if is_after_cutoff.any():
        raise ValueError(
            f"Weather-dependent source {source.name!r} contains records "
            f"after the forecast cutoff {cutoff.date()}"
        )


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data"


def _load_original_table(table_name: str, data_dir: Path) -> pd.DataFrame:
    relative_path = ORIGINAL_FILES[table_name]
    local_path = data_dir / f"{table_name}.parquet"
    if not local_path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://huggingface.co/datasets/{TRAINING_REPOSITORY}/resolve/"
            f"{TRAINING_REVISION}/{relative_path}"
        )
        temporary_path = local_path.with_suffix(".parquet.download")
        try:
            with urlopen(url) as response, temporary_path.open("wb") as output:
                copyfileobj(response, output)
            temporary_path.replace(local_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return pd.read_parquet(local_path)


def _city_locations(city_info: pd.DataFrame) -> pd.DataFrame:
    locations = (
        city_info.loc[:, ["ID", "Lat", "Lon"]]
        .drop_duplicates(subset=["ID"])
        .copy()
    )
    locations["ID"] = locations["ID"].astype("string")
    locations["Lat"] = pd.to_numeric(locations["Lat"], errors="coerce")
    locations["Lon"] = pd.to_numeric(locations["Lon"], errors="coerce")
    if locations[["ID", "Lat", "Lon"]].isna().any().any():
        raise ValueError("city_info contains missing or invalid IDs or coordinates")
    if (
        ~locations["Lat"].between(-90, 90)
        | ~locations["Lon"].between(-180, 180)
    ).any():
        raise ValueError("city_info contains coordinates outside valid ranges")
    return locations.reset_index(drop=True)


def _validate_elevation(
    elevation: pd.DataFrame, locations: pd.DataFrame
) -> pd.DataFrame:
    required = {"ID", "elevation_m"}
    _validate_columns(elevation, required, "elevation")
    result = elevation.loc[:, ["ID", "elevation_m"]].copy()
    result["ID"] = result["ID"].astype("string")
    result["elevation_m"] = pd.to_numeric(result["elevation_m"], errors="coerce")
    if result[["ID", "elevation_m"]].isna().any().any():
        raise ValueError("elevation contains missing or invalid IDs or values")
    if not result["elevation_m"].map(math.isfinite).all():
        raise ValueError("elevation contains non-finite elevation_m values")
    if result["ID"].duplicated().any():
        raise ValueError("elevation contains duplicate station IDs")
    if set(result["ID"]) != set(locations["ID"]):
        raise ValueError("elevation station IDs do not match city_info")
    return result


def _load_elevation_table(city_info: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    local_path = data_dir / "elevation.parquet"
    locations = _city_locations(city_info)
    if local_path.is_file():
        return _validate_elevation(pd.read_parquet(local_path), locations)

    payload = {
        "locations": [
            {"latitude": row.Lat, "longitude": row.Lon}
            for row in locations.itertuples(index=False)
        ]
    }
    request = Request(
        OPEN_ELEVATION_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        response_data = json.loads(response.read())

    results = response_data.get("results") if isinstance(response_data, dict) else None
    if not isinstance(results, list) or len(results) != len(locations):
        raise ValueError(
            "Open-Elevation returned an unexpected number of elevation results"
        )
    elevation = locations.loc[:, ["ID"]].copy()
    elevation["elevation_m"] = [
        result.get("elevation") if isinstance(result, dict) else None
        for result in results
    ]
    elevation = _validate_elevation(elevation, locations)
    data_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = local_path.with_suffix(".parquet.download")
    try:
        elevation.to_parquet(temporary_path, index=False)
        temporary_path.replace(local_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return elevation


def _validate_city_climate_divisions(
    city_climate_divisions: pd.DataFrame,
    city_info: pd.DataFrame,
) -> pd.DataFrame:
    required = {"ID", "Name", "Lat", "Lon"}
    _validate_columns(city_climate_divisions, required, "city_climate_divisions")
    division_columns = [
        column
        for column in city_climate_divisions
        if column.startswith("division_")
    ]
    if not division_columns:
        raise ValueError(
            "city_climate_divisions must contain climate division attributes"
        )

    result = city_climate_divisions.copy()
    result["ID"] = result["ID"].astype("string")
    if result["ID"].isna().any() or result["ID"].duplicated().any():
        raise ValueError("city_climate_divisions contains missing or duplicate IDs")
    expected = city_info.drop_duplicates("ID").loc[
        :, ["ID", "Name", "Lat", "Lon"]
    ].copy()
    expected["ID"] = expected["ID"].astype("string")
    if set(result["ID"]) != set(expected["ID"]):
        raise ValueError("city_climate_divisions station IDs do not match city_info")
    comparison = result.merge(
        expected,
        on="ID",
        how="inner",
        suffixes=("", "_expected"),
        validate="one_to_one",
    )
    if not comparison["Name"].astype("string").equals(
        comparison["Name_expected"].astype("string")
    ):
        raise ValueError("city_climate_divisions Name values do not match city_info")
    for column in ("Lat", "Lon"):
        coordinates = pd.to_numeric(comparison[column], errors="coerce")
        expected_coordinates = pd.to_numeric(
            comparison[f"{column}_expected"], errors="coerce"
        )
        if not coordinates.equals(expected_coordinates):
            raise ValueError(
                f"city_climate_divisions {column} values do not match city_info"
            )

    return result.sort_values("ID", ignore_index=True)


def _load_city_climate_divisions(
    city_info: pd.DataFrame, data_dir: Path
) -> pd.DataFrame:
    local_path = data_dir / "city_climate_divisions.parquet"
    if local_path.is_file():
        return _validate_city_climate_divisions(
            pd.read_parquet(local_path), city_info
        )

    archive_path = data_dir / CLIMATE_DIVISIONS_ARCHIVE
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Climate division shapefile archive not found: {archive_path}"
        )

    import geopandas as gpd

    locations = _city_locations(city_info)
    city_names = city_info.drop_duplicates("ID").loc[:, ["ID", "Name"]]
    city_names["ID"] = city_names["ID"].astype("string")
    cities = locations.merge(city_names, on="ID", how="left", validate="one_to_one")
    city_points = gpd.GeoDataFrame(
        cities,
        geometry=gpd.points_from_xy(cities["Lon"], cities["Lat"]),
        crs="EPSG:4326",
    )
    divisions = gpd.read_file(archive_path, engine="pyogrio")
    if divisions.crs is None:
        raise ValueError(
            "Climate division shapefile has no coordinate reference system"
        )
    divisions = divisions.to_crs("EPSG:4326")
    division_columns = [
        column for column in divisions.columns if column != divisions.geometry.name
    ]
    divisions = divisions.loc[:, division_columns + [divisions.geometry.name]].rename(
        columns={column: f"division_{column}" for column in division_columns}
    )

    joined = gpd.sjoin(city_points, divisions, how="left", predicate="within")
    if joined["ID"].duplicated().any():
        raise ValueError(
            "Climate division spatial join matched a station to multiple divisions"
        )
    result = pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))
    result = _validate_city_climate_divisions(result, city_info)

    temporary_path = local_path.with_suffix(".parquet.download")
    try:
        result.to_parquet(temporary_path, index=False)
        temporary_path.replace(local_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return result


def _validate_city_coastline_distances(
    distances: pd.DataFrame,
    city_info: pd.DataFrame,
) -> pd.DataFrame:
    required = {"ID", "Name", "Lat", "Lon", "distance_to_coast_km"}
    _validate_columns(distances, required, "city_coastline_distances")
    result = distances.loc[:, sorted(required)].copy()
    result["ID"] = result["ID"].astype("string")
    result["distance_to_coast_km"] = pd.to_numeric(
        result["distance_to_coast_km"], errors="coerce"
    )
    if result[["ID", "Name", "Lat", "Lon", "distance_to_coast_km"]].isna().any().any():
        raise ValueError("city_coastline_distances contains missing values")
    if (
        not result["distance_to_coast_km"].map(math.isfinite).all()
        or result["distance_to_coast_km"].lt(0).any()
    ):
        raise ValueError(
            "city_coastline_distances contains invalid distance values"
        )
    if result["ID"].duplicated().any():
        raise ValueError("city_coastline_distances contains duplicate station IDs")

    expected = city_info.drop_duplicates("ID").loc[
        :, ["ID", "Name", "Lat", "Lon"]
    ].copy()
    expected["ID"] = expected["ID"].astype("string")
    if set(result["ID"]) != set(expected["ID"]):
        raise ValueError(
            "city_coastline_distances station IDs do not match city_info"
        )
    comparison = result.merge(
        expected,
        on="ID",
        how="inner",
        suffixes=("", "_expected"),
        validate="one_to_one",
    )
    if not comparison["Name"].astype("string").equals(
        comparison["Name_expected"].astype("string")
    ):
        raise ValueError("city_coastline_distances names do not match city_info")
    for column in ("Lat", "Lon"):
        if not pd.to_numeric(comparison[column], errors="coerce").equals(
            pd.to_numeric(comparison[f"{column}_expected"], errors="coerce")
        ):
            raise ValueError(
                f"city_coastline_distances {column} values do not match city_info"
            )
    return result.sort_values("ID", ignore_index=True)


def _load_city_coastline_distances(
    city_info: pd.DataFrame,
    data_dir: Path,
) -> pd.DataFrame:
    local_path = data_dir / "city_coastline_distances.parquet"
    if local_path.is_file():
        return _validate_city_coastline_distances(
            pd.read_parquet(local_path), city_info
        )

    archive_path = data_dir / COASTLINE_ARCHIVE
    if not archive_path.is_file():
        raise FileNotFoundError(f"Coastline archive not found: {archive_path}")

    import geopandas as gpd
    from shapely.geometry import box

    locations = _city_locations(city_info)
    city_names = city_info.drop_duplicates("ID").loc[:, ["ID", "Name"]].copy()
    city_names["ID"] = city_names["ID"].astype("string")
    cities = locations.merge(city_names, on="ID", how="left", validate="one_to_one")
    city_points = gpd.GeoDataFrame(
        cities,
        geometry=gpd.points_from_xy(cities["Lon"], cities["Lat"]),
        crs="EPSG:4326",
    ).to_crs(COASTLINE_DISTANCE_CRS)

    coastline = gpd.read_file(archive_path, engine="pyogrio")
    if coastline.crs is None:
        raise ValueError("Coastline shapefile has no coordinate reference system")
    coastline = coastline.to_crs("EPSG:4326")
    clip_bounds = box(*COASTLINE_CLIP_BOUNDS)
    coastline.geometry = coastline.geometry.intersection(clip_bounds)
    nonempty_geometries = coastline.geometry.apply(
        lambda geometry: geometry is not None and not geometry.is_empty
    )
    coastline = coastline.loc[nonempty_geometries]
    if coastline.empty:
        raise ValueError("Coastline shapefile has no features in the CONUS region")
    coastline = coastline.to_crs(COASTLINE_DISTANCE_CRS)
    nearest = gpd.sjoin_nearest(
        city_points,
        coastline.loc[:, [coastline.geometry.name]],
        how="left",
        distance_col="distance_m",
    )
    nearest_distances = nearest.groupby("ID", sort=False)["distance_m"].min()
    distances = pd.DataFrame(
        {
            "ID": cities["ID"].to_numpy(),
            "Name": cities["Name"].to_numpy(),
            "Lat": cities["Lat"].to_numpy(),
            "Lon": cities["Lon"].to_numpy(),
            "distance_to_coast_km": (
                cities["ID"].map(nearest_distances).to_numpy() / 1000
            ),
        }
    )
    distances = _validate_city_coastline_distances(distances, city_info)

    temporary_path = local_path.with_suffix(".parquet.download")
    try:
        distances.to_parquet(temporary_path, index=False)
        temporary_path.replace(local_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return distances


def prepare_dataset(
    additional_sources: Iterable[DataSource] = (),
    cutoff: str | pd.Timestamp = FORECAST_CUTOFF,
    data_dir: str | Path | None = None,
) -> PreparedDataset:
    """Load challenge tables from ``data_dir``, downloading missing files.

    The challenge tables are stored as Parquet files in the project data
    directory by default. Additional tables remain separate in
    ``PreparedDataset.additional_sources`` until callers join them.
    """

    cutoff_timestamp = pd.Timestamp(cutoff)
    if pd.isna(cutoff_timestamp):
        raise ValueError("cutoff must be a valid date")

    resolved_data_dir = Path(data_dir) if data_dir is not None else _default_data_dir()
    annual_indices = _load_original_table("annual_indices", resolved_data_dir)
    city_info = _load_original_table("city_info", resolved_data_dir)

    _validate_annual_cutoff(annual_indices, cutoff_timestamp)
    _validate_columns(city_info, CITY_REQUIRED_COLUMNS, "city_info")
    elevation = _load_elevation_table(city_info, resolved_data_dir)
    city_climate_divisions = _load_city_climate_divisions(
        city_info, resolved_data_dir
    )
    city_coastline_distances = _load_city_coastline_distances(
        city_info, resolved_data_dir
    )

    source_tables: dict[str, pd.DataFrame] = {}
    for source in additional_sources:
        if not source.name.strip():
            raise ValueError("Additional data source names must not be empty")
        if source.name in source_tables:
            raise ValueError(f"Duplicate additional data source: {source.name!r}")

        frame = source.loader(cutoff_timestamp)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"Loader for source {source.name!r} must return a pandas DataFrame"
            )
        _validate_source_cutoff(frame, source, cutoff_timestamp)
        source_tables[source.name] = frame

    return PreparedDataset(
        annual_indices=annual_indices,
        city_info=city_info,
        elevation=elevation,
        city_climate_divisions=city_climate_divisions,
        city_coastline_distances=city_coastline_distances,
        additional_sources=source_tables,
        cutoff=cutoff_timestamp,
        data_dir=resolved_data_dir,
    )
