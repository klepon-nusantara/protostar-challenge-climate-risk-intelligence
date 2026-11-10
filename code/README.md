# Code

- `src/climate_challenge/data.py` downloads the challenge tables.
- `src/climate_challenge/config.py` contains shared pipeline and model defaults.
- `src/climate_challenge/cleaning.py` cleans annual indices and creates an audit.
- `src/climate_challenge/backtesting.py` runs rolling-origin backtests.
- `src/climate_challenge/models/` contains the baseline and global model code.
- `src/climate_challenge/evaluation.py` scores methods and selects a method per metric.
- `src/climate_challenge/forecasting.py` generates the final forecasts.
- `src/climate_challenge/submission.py` fills and validates the challenge CSV.
- `src/climate_challenge/__main__.py` provides the `prepare`, `backtest`, and
  `forecast` commands.
- `nb/` contains exploratory/example notebooks; the package does not execute them.
- `check_submission.py` validates `submission.csv`.
- `requirements.txt` lists the Python dependencies.

## Preparing data

Install the package in editable mode once from the repository root:

```bash
python -m pip install -e code
```

Run the end-to-end workflows from the repository root:

```bash
python -m climate_challenge prepare
python -m climate_challenge backtest
python -m climate_challenge forecast
```

`prepare` downloads and validates the original tables in `data/`, queries
Open-Elevation for each unique station if `data/elevation.parquet` is missing,
and spatially joins station locations to
`data/CONUS_CLIMATE_DIVISIONS.shp.zip` if
`data/city_climate_divisions.parquet` is missing. It also calculates distance
to the nearest coastline from `data/ne_10m_coastline.zip` when
`data/city_coastline_distances.parquet` is missing. All derived datasets are
cached as Parquet and exposed as `prepared.elevation`,
`prepared.city_climate_divisions`, and `prepared.city_coastline_distances`;
they are not automatically joined into model features.
If `data/us_gdp/` contains BEA `CAGDP` or `SAGDP` ZIP archives, `prepare` also
converts their annual `__ALL_AREAS_` tables to cached, table-specific Parquet
files in that directory. Quarterly and territory archives are not included in
this conversion.
If `data/us_personal_income/` contains BEA `CAINC` or `SAINC` ZIP archives,
`prepare` likewise converts their annual `__ALL_AREAS_` tables to cached
Parquet files in that directory. Quarterly and per-capita tables are not
included in this conversion.

`forecast` uses `FORECAST_METHODS` and `FORECAST_FEATURES` in
`src/climate_challenge/config.py`, estimates lead calibration from historical
backtests for only those configured methods, fits them through the 2018 origin,
predicts 2019–2025, and writes the filled challenge template to `submission.csv`.
It does not run a model-selection backtest. `FORECAST_FEATURES = None` uses all
available features; set it to a tuple of feature names to restrict the feature
list. `STATIC_FEATURES_BY_METRIC` then selects the static geography features
used for each target; the same mapping is used by `backtest` and `oot_eval`.
The configured selection uses coast distance and climate-division peer
anomalies for hottest-day temperature, and no static geography features for
the other targets. This conservative selection retains the only measured
improvement in the calibrated fixed-origin OOT evaluation. Forecast methods are
configured separately by target metric. Lead
calibration is not applied to precipitation forecasts; precipitation is only
clipped at zero to preserve the target's nonnegative domain.
The notebooks are optional examples, not pipeline entry points.

To extract daily observations and annual coverage/station tables from the
manually supplied `data/7890488/` CSVs, run:

```bash
python -m climate_challenge.oot_eval.preprocessing
```

This writes historical `daily.parquet` and `coverage_and_stations.parquet`
through 2018, plus corresponding `_oot` files starting in 2019. The `_oot`
files are for analysis only and must not be used as forecast inputs. This
command also writes the annual and station-period OOT files used by the fixed
origin evaluation.

When `data/daily.parquet` is present, backtesting and forecasting add daily
features for seasonal temperature and precipitation, hot/cold/heavy-rain and
wet-day frequencies, wet-day intensity, daily precipitation maxima, and dry
spell length. Seasonal precipitation features also summarize wet-day frequency,
wet-day intensity, and the 95th-percentile daily precipitation. Precipitation
features include recent five- and ten-year means relative to each station's
trailing 30-year daily-derived baseline. `oot_eval` compares these features
against the annual-only model using only daily observations through the 2018
origin; 2019-onward daily data is never loaded for features.
When the static Parquet tables are present, model panels can include elevation,
distance to coast, and leave-one-out state and climate-division anomaly
features. The configured workflows select these static features by target;
the anomaly features use observations only through each model origin.

Adjust shared defaults in `src/climate_challenge/config.py`. This includes
forecast methods and features, backtest origins and horizon, cleaning tolerance,
model hyperparameters, evaluation resampling, forecast fallback settings, and
submission paths. CLI
options such as `--start-origin`, `--end-origin`, `--horizon`, `--template`, and
`--output` override their corresponding defaults for one run. The data cutoff
remains fixed to the challenge's 2018-12-31 forecast cutoff.

The tables are also available as Python objects:

```python
from climate_challenge.data import prepare_dataset

prepared = prepare_dataset()
annual = prepared.annual_indices
cities = prepared.city_info
```

Missing Parquet files are downloaded directly into the repository's `data/`
directory. Subsequent runs read those local files. Additional feature sources
can also be supplied as `DataSource` objects. Their loaders receive the
forecast cutoff and should apply it to any download query. Weather-dependent
sources must declare their date column; records after the cutoff are rejected.
Sources are returned separately so callers can join each one using an explicit
key.

```python
import pandas as pd

from climate_challenge.data import DataSource, prepare_dataset

site_features = DataSource(
    name="site_features",
    loader=lambda cutoff: pd.read_csv("data/site_features.csv"),
    weather_dependent=False,
)
prepared = prepare_dataset(additional_sources=[site_features])
```

The BEA GDP tables are retained for policy analysis rather than forecast
features. State GDP can be matched to the 210 climate stations by state FIPS,
but the supplied series begin in 1997 (county detail begins in 2001) and the
current downloads include post-2018 observations without historical release
vintages. Those values cannot be verified as information available at each
backtest or forecast origin.
The BEA personal income tables are also retained for policy analysis and are
not forecast inputs.

Run the unit tests from the repository root with:

```bash
python -m unittest discover -s code/tests
```

`AGENTS.md` documents the challenge's data-use rules. Record any curated
external data in [`../data/README.md`](../data/README.md).
