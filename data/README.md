# Data card: `data/`

Curated tables used or prepared by the code in `code/`. The annual indices and
city metadata are downloaded from the pinned revision of the challenge's
training dataset by `python -m climate_challenge`. Station elevations are
queried from Open-Elevation and cached by `python -m climate_challenge prepare`.
City locations are spatially joined to the NOAA climate divisions shapefile
archive and cached by the same command.
Distances to the nearest Natural Earth coastline are calculated in the
CONUS Albers equal-area projection and cached in kilometers.
Annual BEA state and county GDP archives supplied in `us_gdp/` are converted
to table-specific Parquet files in the same directory by `prepare`.
Annual BEA state and county personal income archives supplied in
`us_personal_income/` are converted to table-specific Parquet files by
`prepare` as well.
Daily observations and coverage tables are extracted from the raw station CSVs
in `7890488/` by `python -m climate_challenge.oot_eval.preprocessing`.

## Files

| File | What it holds | Source | License | Years covered | How we prepared it | Used for |
|---|---|---|---|---|---|---|
| `annual_indices.parquet` | Annual weather indices for the selected stations | [`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), `annual_indices` | CC0-1.0 | 1851–2018 | Downloaded unchanged from revision `bbbdebdab2a18e00f97ef87bd3f35a71e222180b`; stored as Parquet | Forecast input |
| `city_info.parquet` | Station names, IDs, coordinates, and station date ranges | [`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), `city_info` | CC0-1.0 | Station metadata; weather observations are not included | Downloaded unchanged from revision `bbbdebdab2a18e00f97ef87bd3f35a71e222180b`; stored as Parquet | Forecast input / city metadata |
| `elevation.parquet` | Elevation in meters for each unique station ID | [Open-Elevation API](https://api.open-elevation.com/api/v1/lookup) | Not specified by the API; check upstream terms before redistribution | Static geographic data | Queried once per station during `prepare`; stores `ID` and `elevation_m` as Parquet and is reused on later runs | Available for feature experiments; not enabled by default |
| `CONUS_CLIMATE_DIVISIONS.shp.zip` | NOAA CONUS climate division boundaries | [NOAA NCEI climate division monthly data](https://www.ncei.noaa.gov/monitoring-content/data/us/climdiv/monthly/current/) | U.S. government data | Static geographic boundaries | Source shapefile archive supplied in `data/`; reprojected to EPSG:4326 before spatial joining | Input to city climate division table |
| `city_climate_divisions.parquet` | One climate-division record per unique weather station, including division attributes | Derived from `city_info.parquet` and `CONUS_CLIMATE_DIVISIONS.shp.zip` | U.S. government data | Static geographic metadata | `prepare` joins station points to polygons using `within`; contains station ID, name, coordinates, and `division_`-prefixed shapefile fields | Hottest-day temperature climate-division peer feature |
| `ne_10m_coastline.zip` | Natural Earth 1:10m coastline linework | [Natural Earth 10m Physical Vectors](https://www.naturalearthdata.com/downloads/10m-physical-vectors/) | Public domain | Static geographic boundaries | Source shapefile archive supplied in `data/`; clipped to the CONUS region and projected to EPSG:5070 for distance calculations | Input to city coastline distance table |
| `city_coastline_distances.parquet` | Distance from each unique weather station to the nearest coastline, in kilometers | Derived from `city_info.parquet` and `ne_10m_coastline.zip` | Public domain source data | Static geographic metadata | `prepare` calculates planar nearest distance after projecting both source and points to EPSG:5070; contains station ID, name, coordinates, and `distance_to_coast_km` | Hottest-day temperature feature |
| `daily.parquet` | Daily maximum/minimum temperature and precipitation by station | Lai & Dzombak (2019), `7890488/` station CSVs | CC0-1.0 | Through 2018-12-31 | Extracted unchanged, with station ID and name, and restricted to dates through the forecast cutoff | Historical feature exploration; not currently loaded by the forecasting pipeline |
| `coverage_and_stations.parquet` | Annual count of days with both temperatures and station-segment/change counts | Derived from `daily.parquet` and `7890488/city_info.csv` | CC0-1.0 | Through 2018 | Counted complete temperature pairs by station-year; station segment and changes come from station-period start dates | Historical feature exploration; not currently loaded by the forecasting pipeline |
| `daily_oot.parquet` | Daily weather observations by station | Lai & Dzombak (2019), `7890488/` station CSVs | CC0-1.0 | From 2019 onward, as available in the source files | Extracted unchanged, with station ID and name, and kept separate from historical data | Analysis only; never a forecast input |
| `coverage_and_stations_oot.parquet` | Annual coverage and station-segment/change counts | Derived from OOT daily observations and `7890488/city_info.csv` | CC0-1.0 | From 2019 onward, as available in the source files | Same calculation as the historical table, restricted to years from 2019 onward | Analysis only; never a forecast input |
| `annual_indices_oot.parquet`, `city_info_oot.parquet` | Annualized post-2018 evaluation targets and station periods | Derived from `7890488/` station CSVs and `city_info.csv` | CC0-1.0 | From 2019 onward, as available in the source files | Created by the OOT preprocessing command for fixed-origin evaluation | Analysis only; never a forecast input |
| `us_gdp/*.zip` | BEA regional GDP tables, including county/state annual series, quarterly series, and territory data | [BEA Regional Economic Accounts](https://apps.bea.gov/regional/downloadzip.htm) | See BEA terms and the metadata supplied with each archive | Varies by archive; annual state GDP begins in 1997 and county tables in 2001 | Source ZIP archives supplied in `data/us_gdp/` | Policy analysis |
| `us_gdp/*.parquet` | Parquet conversions of annual state/county GDP tables from `CAGDP` and `SAGDP` archives | Derived from the corresponding BEA ZIP archives | Same as source archives | Varies by table; includes years after 2018 | `prepare` reads each `__ALL_AREAS_` CSV, keeps geography/FIPS and table metadata, converts annual values to numeric, and caches one Parquet per ZIP archive | Policy analysis; not used as forecast features |
| `us_personal_income/*.zip` | BEA regional personal income tables, including county/state annual series, per-capita summaries, and quarterly series | [BEA Regional Economic Accounts](https://apps.bea.gov/regional/downloadzip.htm) | See BEA terms and the metadata supplied with each archive | Varies by archive; annual state series begins in 1929, county detail varies by table | Source ZIP archives supplied in `data/us_personal_income/` | Policy analysis |
| `us_personal_income/*.parquet` | Parquet conversions of annual personal income tables from `CAINC` and `SAINC` archives | Derived from the corresponding BEA ZIP archives | Same as source archives | Varies by table; includes years after 2018 | `prepare` reads each annual `__ALL_AREAS_` CSV, keeps geography/FIPS and table metadata, converts annual values to numeric, and caches one Parquet per ZIP archive | Policy analysis; not used as forecast features |

## How the files fit together

- **Join keys:** `ID` identifies the station in the annual, city, elevation, city climate division, and coastline distance tables.
- **Units:** The annual table retains the source dataset's units and column names (for example, temperatures in °F, precipitation in inches, and `cdd` in °F-days).
- **Known limitations:** The training data ends in 2018. Annual indices can contain missing values; inspect the `Miss ...` columns before modeling.
- **GDP limitations:** State GDP can be associated with the climate stations by state FIPS (all 210 have state assignments); county-level GDP would require a county crosswalk. The provided GDP tables include years after the forecast origin and do not provide historical release vintages, so values available at each forecast origin cannot be verified. They are therefore kept for policy analysis, not used as forecasting features.
- **Personal income limitations:** Personal income is not used as a forecast feature; the provided tables include post-origin years and current-vintage estimates. Geographic and release-vintage matching must be considered for policy analysis.
- Static geography tables provide optional feature inputs as configured in `code/src/climate_challenge/config.py`. The default OOT/forecast configuration uses nearest-coast distance and climate-division peer anomalies for hottest-day temperature only. The climate/state anomalies are recalculated at each origin from observations available through that origin.
- Daily and coverage tables are exported for feature exploration; daily feature use is constrained by each model origin.

## Data rule check

<!-- Forecast inputs: weather records up to 2018-12-31, and data from any year that does not respond to the weather.
2019–2025 data that responds to the weather may be used only for the analysis and policy idea (see the FAQ in the challenge Space). -->

- [x] Every file marked "Forecast input" is weather data up to 2018-12-31, or data that does not respond to the weather.
- [x] Files with 2019–2025 data that responds to the weather are marked "Analysis only" and are not read by the forecasting code.
