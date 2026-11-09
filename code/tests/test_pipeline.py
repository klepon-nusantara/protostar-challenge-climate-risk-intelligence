import unittest

import numpy as np
import pandas as pd

from climate_challenge.backtesting import run_backtest
from climate_challenge.cleaning import clean_annual_indices, metric_matrix, station_metadata
from climate_challenge.evaluation import evaluate_backtest, select_best_methods
from climate_challenge.forecasting import forecast_all_stations
from climate_challenge.models.global_models import GlobalModelConfig
from climate_challenge.models.global_models import build_panel
from climate_challenge.submission import (
    assemble_submission,
    validate_submission,
)


def make_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    city_records = []
    years = range(1960, 2019)
    for station_number in range(6):
        station_id = f"US{station_number:09d}"
        station_name = f"City {station_number}"
        city_records.append(
            {
                "ID": station_id,
                "Name": station_name,
                "Lat": 30.0 + station_number,
                "Lon": -100.0 + station_number,
            }
        )
        for year in years:
            temperature = 60.0 + station_number + 0.02 * (year - 1960)
            precipitation = 30.0 + station_number + (year % 5)
            hot_day = temperature + 30.0
            cdd = 100.0 + station_number * 10 + (year % 7)
            records.append(
                {
                    "ID": station_id,
                    "Name": station_name,
                    "Year": year,
                    "Miss Tmax": 0.0,
                    "Miss Tmin": 0.0,
                    "Miss Prcp": 0.0,
                    "Avg Temp": temperature,
                    "ToT Prcp": precipitation,
                    "Max Tmax": hot_day,
                    "cdd": cdd,
                }
            )
    return pd.DataFrame(records), pd.DataFrame(city_records)


class CleaningAndEvaluationTests(unittest.TestCase):
    def test_cleaning_masks_inf_and_high_missing_day_years(self) -> None:
        annual, _ = make_dataset()
        annual.loc[0, "Avg Temp"] = np.inf
        annual.loc[1, "Miss Prcp"] = 61

        result = clean_annual_indices(annual)

        self.assertTrue(np.isnan(result.data.loc[result.data.index[0], "Avg Temp"]))
        self.assertTrue(
            result.data.loc[
                (result.data["ID"] == "US000000000") & (result.data["Year"] == 1961),
                "ToT Prcp",
            ].isna().all()
        )
        self.assertEqual(result.audit.loc["total_precip_in", "flagged_by_missing_days"], 1)

    def test_metric_matrix_and_metadata_are_keyed_by_station_id(self) -> None:
        annual, cities = make_dataset()
        cleaned = clean_annual_indices(annual).data

        matrix = metric_matrix(cleaned, "annual_avg_temp")
        metadata = station_metadata(pd.concat([cities, cities.iloc[[0]]]))

        self.assertEqual(matrix.columns.nunique(), 6)
        self.assertEqual(len(metadata), 6)
        self.assertEqual(matrix.loc[1960, "US000000000"], 60.0)

    def test_features_for_an_origin_do_not_depend_on_later_observations(self) -> None:
        annual, cities = make_dataset()
        cleaned = clean_annual_indices(annual).data
        matrix = metric_matrix(cleaned, "annual_avg_temp")
        metadata = station_metadata(cities)
        config = GlobalModelConfig(first_origin=1989, lightgbm_estimators=5)

        original_panel = build_panel(matrix, metadata, horizon=7, config=config)
        changed_matrix = matrix.copy()
        changed_matrix.loc[2011:] = 100_000.0
        changed_panel = build_panel(changed_matrix, metadata, horizon=7, config=config)
        feature_columns = [
            "a30",
            "ly_a",
            "a5_a",
            "a10_a",
            "trend_off",
            "slope",
            "sd30",
            "lat",
            "lon",
            "nat_ly",
            "nat_a5",
            "Last Year",
            "30-Yr Avg",
            "Linear Trend",
        ]
        original = original_panel.loc[
            original_panel["origin"] == 2010, feature_columns
        ].reset_index(drop=True)
        changed = changed_panel.loc[
            changed_panel["origin"] == 2010, feature_columns
        ].reset_index(drop=True)

        pd.testing.assert_frame_equal(original, changed)

    def test_region_features_are_one_hot_and_cover_unassigned_stations(self) -> None:
        annual, cities = make_dataset()
        cleaned = clean_annual_indices(annual).data
        matrix = metric_matrix(cleaned, "annual_avg_temp")
        metadata = station_metadata(cities)
        metadata.loc["US000000000", ["Lat", "Lon"]] = [41.0, -90.0]
        metadata.loc["US000000001", ["Lat", "Lon"]] = [32.0, -90.0]
        metadata.loc["US000000002", ["Lat", "Lon"]] = [37.0, -122.0]
        metadata.loc["US000000003", ["Lat", "Lon"]] = [42.0, -71.0]
        metadata.loc["US000000004", ["Lat", "Lon"]] = [40.0, -110.0]
        metadata.loc["US000000005", ["Lat", "Lon"]] = [32.0, -100.0]
        config = GlobalModelConfig(first_origin=1989)

        panel = build_panel(matrix, metadata, horizon=1, config=config)
        forecast_rows = panel.loc[panel["origin"] == 2010].set_index("city_id")
        region_columns = [
            "region_midwest",
            "region_southern",
            "region_west_coast",
            "region_new_england",
            "region_other",
        ]
        expected_regions = {
            "US000000000": "region_midwest",
            "US000000001": "region_southern",
            "US000000002": "region_west_coast",
            "US000000003": "region_new_england",
            "US000000004": "region_other",
            "US000000005": "region_southern",
        }

        self.assertTrue(forecast_rows[region_columns].sum(axis=1).eq(1).all())
        for station_id, expected_column in expected_regions.items():
            self.assertEqual(forecast_rows.loc[station_id, expected_column], 1.0)
        self.assertTrue(pd.isna(forecast_rows.loc["US000000004", "regional_anomaly"]))
        self.assertTrue(
            np.isfinite(forecast_rows.loc["US000000001", "regional_anomaly"])
        )

    def test_backtest_and_final_submission_pipeline(self) -> None:
        annual, city_info = make_dataset()
        cleaned = clean_annual_indices(annual).data
        config = GlobalModelConfig(
            first_origin=1989,
            lightgbm_estimators=10,
            lightgbm_min_child_samples=5,
        )
        backtest = run_backtest(
            cleaned,
            city_info,
            origins=[2010],
            horizon=7,
            model_config=config,
        )
        metrics = evaluate_backtest(backtest)
        methods = select_best_methods(backtest)
        forecasts, selected = forecast_all_stations(
            cleaned,
            city_info,
            backtest,
            origin=2018,
            horizon=7,
            model_config=config,
            use_sparse_fallback=False,
        )
        template = pd.MultiIndex.from_product(
            [
                ["US000000000", "US000000001"],
                range(2019, 2026),
                ["annual_avg_temp", "total_precip_in", "hottest_day_temp", "cdd"],
            ],
            names=["city_id", "year", "metric"],
        ).to_frame(index=False)
        template["predicted_value"] = np.nan
        submission_forecasts = forecasts.loc[
            forecasts["city_id"].isin(["US000000000", "US000000001"])
        ]
        submission = assemble_submission(submission_forecasts, template)

        self.assertEqual(len(backtest), 6 * 4 * 7)
        self.assertEqual(len(metrics), 4 * 7)
        self.assertEqual(set(methods), set(selected))
        self.assertEqual(len(submission), 2 * 7 * 4)
        self.assertFalse(submission["predicted_value"].isna().any())
        validate_submission(submission, expected_city_ids=["US000000000", "US000000001"])

    def test_submission_rejects_negative_precipitation(self) -> None:
        submission = pd.DataFrame(
            {
                "city_id": ["US000000000"],
                "year": [2019],
                "metric": ["total_precip_in"],
                "predicted_value": [-1.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "must be nonnegative"):
            validate_submission(submission)


if __name__ == "__main__":
    unittest.main()
