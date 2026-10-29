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

> 📋 **This is the submission template** for the [Protostar Challenge: Climate Risk Intelligence](https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence). Copy it into your Hugging Face account, fill it in, and **delete this box** when you are done.
>
> 1. **Copy:** `hf download CMU-OSPO/protostar-climate-submission-template --repo-type dataset --local-dir my-submission`, then `hf upload <your-name>/<dataset-name> my-submission . --repo-type dataset` (or create a new dataset on the Hub and upload these files).
> 2. **Fill in** the `predicted_value` column of `submission.csv`: the city, year and metric of all 5,880 rows (210 cities × 7 years × 4 metrics) are already there. Check it with `python code/check_submission.py`.
> 3. **Replace** every `<...>` placeholder in this README, including `<Team Name>` in the header.
> 4. **Add** your notebooks or scripts to `code/`.
> 5. **Make the dataset public** and paste its link in the *Submission* tab of the challenge Space.
>
> Keep the tag `protostar-climate-2026` in the header: it is how submissions are found.

# <Team Name>: Protostar Climate Challenge Submission

**Negative result:** no <!-- Write "yes" if your main finding is why reuse or reproduction did not work. -->

Forecasts of yearly weather for 210 U.S. cities, 2019–2025, made as of the end of 2018 (no data that reveals 2019–2025 weather), for the [Protostar Challenge: Climate Risk Intelligence](https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence).

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

## Feedback to the Ecosystem

<!-- What got in the way, and what would fix it: data, metadata, documentation, licensing, tools, infrastructure, policy. Be specific. -->

- **What got in the way:** <...>
- **What would fix it:** <...>

## Files

| File | Content |
|---|---|
| `submission.csv` | Our forecasts: `city_id,year,metric,predicted_value` |
| `code/` | <Our notebooks or scripts> |

**Metric codes:** `annual_avg_temp` (°F), `total_precip_in` (inches), `hottest_day_temp` (°F), `cdd` (°F-days).
**City IDs:** the station `ID` from the `city_info` table of the training dataset (e.g., Pittsburgh = `USW00094823`).

## How to Reproduce

<!-- The steps someone else would follow to recreate submission.csv. -->

1. <e.g., install the packages in code/requirements.txt>
2. <e.g., run code/forecast.ipynb>

## Data

<!-- List every external source (name, link, years used) and say whether it was used for the forecasts or only for the analysis and policy idea. Forecast inputs must stop at 2018-12-31. -->

- **External data:** <none, or: name, link, years, forecast or analysis>

Training data: [`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), from:

- Lai, Y., & Dzombak, D. A. (2019). Compiled historical daily temperature and precipitation data for selected 210 U.S. cities. Carnegie Mellon University. https://doi.org/10.1184/R1/7890488.v6
- Lai, Y., & Dzombak, D. A. (2019). *Use of historical data to assess regional climate change.* Journal of Climate, 32(14), 4299–4320. https://doi.org/10.1175/JCLI-D-18-0630.1
