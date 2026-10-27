---
pretty_name: "Protostar Climate Challenge Submission: <Team Name>"
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

<!--
HOW TO USE THIS TEMPLATE (delete this comment when you are done)

1. Replace every <...> placeholder, including <Team Name> in the header above.
2. Fill in submission.csv: 5,880 rows (210 cities × 7 years × 4 metrics).
3. Optional: put your notebooks or scripts in code/.
4. Make this dataset public and paste its link in the Submission tab of the challenge Space:
   https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence

Keep the tag "protostar-climate-2026" in the header: it is how your submission is found.
-->

# <Team Name>: Protostar Climate Challenge Submission

Forecasts of yearly weather for 210 U.S. cities, 2019–2025, made using only data up to 2018-12-31, for the [Urban Climate AI Challenge](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018).

## Team

| Name | Hugging Face username | Program |
|---|---|---|
| <Name> | <@username> | <e.g., Heinz MSPPM, Tepper BS Business> |

## Method

<!-- A few lines: what model you used, which features, and how you checked it (e.g., trained up to 2010, tested on 2011–2018). -->

- **Model:** <e.g., 10-year average plus a linear trend, per city>
- **Features:** <e.g., the last 30 years of each metric>
- **How we checked it:** <e.g., practice run on 2011–2018: CDD MAE = ... °F-days>

## Policy Idea

<!-- Half a page at most. Answer the three questions however you like. -->

1. **What risk did we notice?** <...>
2. **What would we do about it, and who would do it?** <...>
3. **Why is it sustainable?** <...>

## Files

| File | Content |
|---|---|
| `submission.csv` | Our forecasts: `city_id,year,metric,predicted_value` |
| `code/` | <Our notebooks or scripts, or "not included"> |

**Metric codes:** `annual_avg_temp` (°F), `total_precip_in` (inches), `hottest_day_temp` (°F), `cdd` (°F-days).
**City IDs:** the station `ID` from the `city_info` table of the training dataset (e.g., Pittsburgh = `USW00094823`).

## How to Reproduce

<!-- The steps someone else would follow to recreate submission.csv. -->

1. <e.g., install the packages in code/requirements.txt>
2. <e.g., run code/forecast.ipynb>

## Data

Training data: [`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), from:

- Lai, Y., & Dzombak, D. A. (2019). Compiled historical daily temperature and precipitation data for selected 210 U.S. cities. Carnegie Mellon University. https://doi.org/10.1184/R1/7890488.v6
- Lai, Y., & Dzombak, D. A. (2019). *Use of historical data to assess regional climate change.* Journal of Climate, 32(14), 4299–4320. https://doi.org/10.1175/JCLI-D-18-0630.1
