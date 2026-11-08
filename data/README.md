# Data card: `data/`

Curated input tables used by the code in `code/`. Both tables are downloaded
from the pinned revision of the challenge's training dataset by
`python -m climate_challenge`.

## Files

| File | What it holds | Source | License | Years covered | How we prepared it | Used for |
|---|---|---|---|---|---|---|
| `annual_indices.parquet` | Annual weather indices for the selected stations | [`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), `annual_indices` | CC0-1.0 | 1851–2018 | Downloaded unchanged from revision `bbbdebdab2a18e00f97ef87bd3f35a71e222180b`; stored as Parquet | Forecast input |
| `city_info.parquet` | Station names, IDs, coordinates, and station date ranges | [`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), `city_info` | CC0-1.0 | Station metadata; weather observations are not included | Downloaded unchanged from revision `bbbdebdab2a18e00f97ef87bd3f35a71e222180b`; stored as Parquet | Forecast input / city metadata |

## How the files fit together

- **Join key:** `ID` identifies the station in both tables.
- **Units:** The annual table retains the source dataset's units and column names (for example, temperatures in °F, precipitation in inches, and `cdd` in °F-days).
- **Known limitations:** The training data ends in 2018. Annual indices can contain missing values; inspect the `Miss ...` columns before modeling.

## Data rule check

<!-- Forecast inputs: weather records up to 2018-12-31, and data from any year that does not respond to the weather.
2019–2025 data that responds to the weather may be used only for the analysis and policy idea (see the FAQ in the challenge Space). -->

- [x] Every file marked "Forecast input" is weather data up to 2018-12-31, or data that does not respond to the weather.
- [x] Files with 2019–2025 data that responds to the weather are marked "Analysis only" and are not read by the forecasting code.
