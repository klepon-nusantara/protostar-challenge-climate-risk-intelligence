# Data card: `data/`

<!-- The curated data your code (`code/`) reads, as you prepared it. Together, code and data should reproduce `submission.csv`.
Fill in this card for every file here, and delete the example rows and these comments.
If a source is too large, or its license does not allow copying, do not upload it: list it below with its exact version and how to download it. -->

## Files

| File | What it holds | Source | License | Years covered | How we prepared it | Used for |
|---|---|---|---|---|---|---|
| <`annual_valid.csv`> | <One row per city and year: the four metrics> | <[Challenge dataset](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), `annual_indices`> | <CC0-1.0> | <1989–2018> | <Kept years with at most 10 missing days> | <Forecast input> |
| <`city_tree_canopy.csv`> | <Tree canopy share per city> | <NLCD 2016, link> | <Public domain> | <2016> | <Averaged within 5 km of each station; joined by `city_id`> | <Forecast input / analysis only> |

## Not uploaded (download instead)

<!-- Delete this section if every file is uploaded. -->

| Source | Exact version | How to download | Why not uploaded |
|---|---|---|---|
| <Name, link> | <Version, date or DOI> | <Command or steps> | <Too large / license> |

## How the files fit together

- **Join key:** <e.g., `city_id` = the station `ID` in the challenge `city_info` table>
- **Units:** <e.g., °F, inches, °F-days; any conversions you made>
- **Known limitations:** <e.g., missing years, estimated values, coverage gaps>

## Data rule check

<!-- Forecast inputs: weather records up to 2018-12-31, and data from any year that does not respond to the weather.
2019–2025 data that responds to the weather may be used only for the analysis and policy idea (see the FAQ in the challenge Space). -->

- [ ] Every file marked "Forecast input" is weather data up to 2018-12-31, or data that does not respond to the weather.
- [ ] Files with 2019–2025 data that responds to the weather are marked "Analysis only" and are not read by the forecasting code.
