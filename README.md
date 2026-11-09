---
pretty_name: "Protostar Climate Challenge Submission: Klepon Nusantara"
license: cc-by-4.0
task_categories:
- time-series-forecasting
tags:
- climate
- forecasting
- protostar-climate-2026
configs:
- config_name: default
  data_files:
  - split: test
    path: submission.csv
---

# Klepon Nusantara: Protostar Climate Challenge Submission

**Negative result:** no

Forecasts of yearly weather for 210 U.S. cities, 2019–2025, made as of the end of 2018 (no data that reveals 2019–2025 weather), for the [Protostar Challenge: Climate Risk Intelligence](https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence).

## Team

| Name | Hugging Face username | Affiliation |
|---|---|---|
| Ardacandra Subiantoro | `ardacandrasubiantoro` | Superbank |
| Bima Priambodo Wahyu Rianto | `bimapriambodo` | Superbank |
| Rizkiyana Putra | `rizkiyanapp` | Superbank |

## Method

- **Model:** Global Ridge and LightGBM models predict each station's deviation from its trailing 30-year mean. Per-metric model selection uses pooled MAE from rolling-origin backtests.
- **Features:** Trailing 5- and 10-year averages relative to the 30-year baseline, 30-year and 10-year trends, trend acceleration, missing-history age, station coordinates, neighboring-station anomaly, regional anomaly, and one-hot region indicators.
- **How we checked it:** Rolling-origin backtests use origins from 1995 through 2011, predicting up to seven years ahead and scoring against observations through 2018. The selected method is chosen separately for each metric. Run `python -m climate_challenge backtest` to reproduce the scores.

## AI Tools and Agents

- **Tools and models:** Copilot in VS Code.
- **What they did:** Helped inspect exploratory notebooks, implement model-feature code, and run tests. It did not generate forecast values.
- **What a person reviewed:** A team member reviewed code changes and outputs.
- **Agent instructions file:** [`code/AGENTS.md`](code/AGENTS.md).

## Policy Idea

To be completed by the team.

## Feedback to the Ecosystem

To be completed by the team.

## Files

| File | Content |
|---|---|
| `submission.csv` | Our forecasts: `city_id,year,metric,predicted_value` |
| `code/` | The reusable forecasting package, tests, and optional example notebooks |
| `data/` | The curated data our code uses; described in its data card, [`data/README.md`](data/README.md) |

**Metric codes:** `annual_avg_temp` (°F), `total_precip_in` (inches), `hottest_day_temp` (°F), `cdd` (°F-days).
**City IDs:** the station `ID` from the `city_info` table of the training dataset (e.g., Pittsburgh = `USW00094823`).

## How to Reproduce

### Environment

Tested with Python 3.11.18. The direct Python dependencies are pinned in [`code/requirements.txt`](code/requirements.txt).

```bash
conda create --name climate-challenge python=3.11.18 pip
conda activate climate-challenge
python -m pip install -r code/requirements.txt
python -m pip install -e code
```

### Run

From the repository root, run the package workflows (the notebooks in
`code/nb/` are examples and are not required):

```bash
python -m climate_challenge prepare
python -m climate_challenge backtest
python -m climate_challenge forecast
```

`prepare` downloads and validates the original tables in `data/`. `backtest`
compares methods using rolling time-based origins. `forecast` repeats the
backtest for method selection, forecasts 2019–2025, and fills the submission
template. To validate the generated submission independently, run:

```bash
python code/check_submission.py
```

## Submit to Hugging Face

Before submitting, make sure `submission.csv` contains the forecasts you intend
to submit and complete the policy idea and ecosystem feedback sections. Validate
it from the repository root:

```bash
python code/check_submission.py
```

Sign in to Hugging Face, create a **public dataset** under your team's
namespace, and authenticate the command-line client:

```bash
python -m pip install -U huggingface_hub
hf auth login
```

The submission dataset is
[`ardacandrasubiantoro/klepon-nusantara-submission`](https://huggingface.co/datasets/ardacandrasubiantoro/klepon-nusantara-submission).
From the repository root, upload the required README, forecasts, and code:

```bash
hf upload ardacandrasubiantoro/klepon-nusantara-submission README.md README.md --repo-type dataset
hf upload ardacandrasubiantoro/klepon-nusantara-submission submission.csv submission.csv --repo-type dataset
hf upload ardacandrasubiantoro/klepon-nusantara-submission code code --repo-type dataset
```

To include the input data and its data card, first run
`python -m climate_challenge prepare`, then upload them explicitly:

```bash
hf upload ardacandrasubiantoro/klepon-nusantara-submission data/README.md data/README.md --repo-type dataset
hf upload ardacandrasubiantoro/klepon-nusantara-submission data/annual_indices.parquet data/annual_indices.parquet --repo-type dataset
hf upload ardacandrasubiantoro/klepon-nusantara-submission data/city_info.parquet data/city_info.parquet --repo-type dataset
```

The Parquet files are ignored by Git, so they are not included by uploading
`code/` or the other files above. Check the dataset page to confirm it is public
and contains the intended files. Finally, paste the dataset URL above in the
competition Space's **Submission** tab. The competition records the dataset
version at submission time; submit again after any changes you want counted.

## Data

- **External data:** No additional external data was used. Forecast inputs come from the challenge's training dataset and end in 2018.
- **Curated data:** see the data card in [`data/README.md`](data/README.md) (each file, its source, license and preparation).

Training data: [`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), from:

- Lai, Y., & Dzombak, D. A. (2019). Compiled historical daily temperature and precipitation data for selected 210 U.S. cities. Carnegie Mellon University. https://doi.org/10.1184/R1/7890488.v6
- Lai, Y., & Dzombak, D. A. (2019). *Use of historical data to assess regional climate change.* Journal of Climate, 32(14), 4299–4320. https://doi.org/10.1175/JCLI-D-18-0630.1
