# Data card: `data/`

Curated tables used or prepared by the code in `code/`. The annual indices and
city metadata are downloaded from the pinned revision of the challenge's
training dataset by `python -m climate_challenge`. Station elevations are
queried from Open-Elevation and cached by `python -m climate_challenge prepare`.
City locations are spatially joined to the NOAA climate divisions shapefile
archive and cached by the same command.
Daily observations and coverage tables are extracted from the raw station CSVs
in `7890488/` by `python -m climate_challenge.oot_eval.preprocessing`.

## Files

| File | What it holds | Source | License | Years covered | How we prepared it | Used for |
|---|---|---|---|---|---|---|
| `annual_indices.parquet` | Annual weather indices for the selected stations | [`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), `annual_indices` | CC0-1.0 | 1851–2018 | Downloaded unchanged from revision `bbbdebdab2a18e00f97ef87bd3f35a71e222180b`; stored as Parquet | Forecast input |
| `city_info.parquet` | Station names, IDs, coordinates, and station date ranges | [`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), `city_info` | CC0-1.0 | Station metadata; weather observations are not included | Downloaded unchanged from revision `bbbdebdab2a18e00f97ef87bd3f35a71e222180b`; stored as Parquet | Forecast input / city metadata |
| `elevation.parquet` | Elevation in meters for each unique station ID | [Open-Elevation API](https://api.open-elevation.com/api/v1/lookup) | Not specified by the API; check upstream terms before redistribution | Static geographic data | Queried once per station during `prepare`; stores `ID` and `elevation_m` as Parquet and is reused on later runs | Available for feature use; not currently used by forecasting |
| `CONUS_CLIMATE_DIVISIONS.shp.zip` | NOAA CONUS climate division boundaries | [NOAA NCEI climate division monthly data](https://www.ncei.noaa.gov/monitoring-content/data/us/climdiv/monthly/current/) | U.S. government data | Static geographic boundaries | Source shapefile archive supplied in `data/`; reprojected to EPSG:4326 before spatial joining | Input to city climate division table |
| `city_climate_divisions.parquet` | One climate-division record per unique weather station, including division attributes | Derived from `city_info.parquet` and `CONUS_CLIMATE_DIVISIONS.shp.zip` | U.S. government data | Static geographic metadata | `prepare` joins station points to polygons using `within`; contains station ID, name, coordinates, and `division_`-prefixed shapefile fields | Available for feature use; not currently used by forecasting |
| `daily.parquet` | Daily maximum/minimum temperature and precipitation by station | Lai & Dzombak (2019), `7890488/` station CSVs | CC0-1.0 | Through 2018-12-31 | Extracted unchanged, with station ID and name, and restricted to dates through the forecast cutoff | Historical feature exploration; not currently loaded by the forecasting pipeline |
| `coverage_and_stations.parquet` | Annual count of days with both temperatures and station-segment/change counts | Derived from `daily.parquet` and `7890488/city_info.csv` | CC0-1.0 | Through 2018 | Counted complete temperature pairs by station-year; station segment and changes come from station-period start dates | Historical feature exploration; not currently loaded by the forecasting pipeline |
| `daily_oot.parquet` | Daily weather observations by station | Lai & Dzombak (2019), `7890488/` station CSVs | CC0-1.0 | From 2019 onward, as available in the source files | Extracted unchanged, with station ID and name, and kept separate from historical data | Analysis only; never a forecast input |
| `coverage_and_stations_oot.parquet` | Annual coverage and station-segment/change counts | Derived from OOT daily observations and `7890488/city_info.csv` | CC0-1.0 | From 2019 onward, as available in the source files | Same calculation as the historical table, restricted to years from 2019 onward | Analysis only; never a forecast input |
| `annual_indices_oot.parquet`, `city_info_oot.parquet` | Annualized post-2018 evaluation targets and station periods | Derived from `7890488/` station CSVs and `city_info.csv` | CC0-1.0 | From 2019 onward, as available in the source files | Created by the OOT preprocessing command for fixed-origin evaluation | Analysis only; never a forecast input |

## How the files fit together

- **Join keys:** `ID` identifies the station in the annual, city, elevation, and city climate division tables.
- **Units:** The annual table retains the source dataset's units and column names (for example, temperatures in °F, precipitation in inches, and `cdd` in °F-days).
- **Known limitations:** The training data ends in 2018. Annual indices can contain missing values; inspect the `Miss ...` columns before modeling.
- Daily and coverage tables are exported for feature exploration; adding them to a model requires a separate time-based backtest.

## Data rule check

<!-- Forecast inputs: weather records up to 2018-12-31, and data from any year that does not respond to the weather.
2019–2025 data that responds to the weather may be used only for the analysis and policy idea (see the FAQ in the challenge Space). -->

- [x] Every file marked "Forecast input" is weather data up to 2018-12-31, or data that does not respond to the weather.
- [x] Files with 2019–2025 data that responds to the weather are marked "Analysis only" and are not read by the forecasting code.
