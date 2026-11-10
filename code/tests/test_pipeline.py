import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from climate_challenge.backtesting import (
    apply_lead_calibration,
    append_backtest_run_log,
    run_backtest,
)
from climate_challenge.cleaning import clean_annual_indices, metric_matrix, station_metadata
from climate_challenge.config import FORECAST_METHODS, MODEL_METHODS
from climate_challenge.evaluation import evaluate_backtest, select_best_methods
from climate_challenge.forecasting import forecast_all_stations
from climate_challenge.models.global_models import (
    GlobalModelConfig,
    _daily_feature_matrices,
    _daily_features_at_origin,
    build_panel,
)
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
    def test_lead_calibration_uses_historical_bias_by_metric_and_lead(self) -> None:
        calibration = pd.DataFrame(
            {
                "origin": [2000, 2001, 2002, 2000],
                "target_year": [2001, 2002, 2003, 2002],
                "metric": ["annual_avg_temp"] * 4,
                "h": [1, 1, 1, 2],
                "y": [10.0, 20.0, 20.0, 10.0],
                **{
                    method: [10.0, 20.0, 20.0, 10.0]
                    for method in MODEL_METHODS
                },
                "Last Year": [12.0, 24.0, 70.0, 17.0],
            }
        )
        target = pd.DataFrame(
            {
                "metric": ["annual_avg_temp", "annual_avg_temp"],
                "h": [1, 2],
                **{
                    method: [100.0, 200.0]
                    for method in MODEL_METHODS
                },
            }
        )

        calibrated = apply_lead_calibration(target, calibration)

        np.testing.assert_allclose(calibrated["Last Year"], [96.0, 193.0])
        np.testing.assert_allclose(calibrated["_raw_Last Year"], [100.0, 200.0])

    def test_lead_calibration_preserves_nonnegative_targets(self) -> None:
        calibration = pd.DataFrame(
            {
                "origin": [2000],
                "target_year": [2001],
                "metric": ["total_precip_in"],
                "h": [1],
                "y": [1.0],
                **{
                    method: [4.0]
                    for method in MODEL_METHODS
                },
            }
        )
        target = pd.DataFrame(
            {
                "metric": ["total_precip_in"],
                "h": [1],
                **{
                    method: [2.0]
                    for method in MODEL_METHODS
                },
            }
        )

        calibrated = apply_lead_calibration(target, calibration)

        self.assertEqual(calibrated.loc[0, "Last Year"], 0.0)

    def test_backtest_run_log_appends_configuration_and_scores(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "backtest" / "backtest.log"
            backtest = pd.DataFrame(
                {
                    "origin": [2010, 2010],
                    "target_year": [2011, 2012],
                    "city_id": ["US000000001", "US000000001"],
                    "metric": ["annual_avg_temp", "annual_avg_temp"],
                    "h": [1, 2],
                    "y": [10.0, 20.0],
                    **{
                        method: [10.0, 20.0]
                        for method in MODEL_METHODS
                        if method != "Ridge"
                    },
                    "Ridge": [11.0, 23.0],
                }
            )
            scores = evaluate_backtest(backtest)
            audit = pd.DataFrame(
                {"raw_missing": [0]}, index=["annual_avg_temp"]
            )

            append_backtest_run_log(
                log_path,
                backtest,
                scores,
                audit,
                origins=(2010,),
                horizon=2,
                input_paths=("data/annual_indices.parquet",),
            )
            append_backtest_run_log(
                log_path,
                backtest,
                scores,
                audit,
                origins=(2010,),
                horizon=2,
            )

            log_content = log_path.read_text(encoding="utf-8")
            self.assertEqual(log_content.count("Backtest run:"), 2)
            self.assertIn("Origins: 2010-2010 (1 origins)", log_content)
            self.assertIn("Forecast horizon: 2", log_content)
            self.assertIn("Scored period: 2011-2012", log_content)
            self.assertIn("Models experimented:", log_content)
            self.assertIn("Ridge", log_content)
            self.assertIn('"ridge_alpha": 10.0', log_content)
            self.assertIn("annual_avg_temp", log_content)
            self.assertIn("data/annual_indices.parquet", log_content)

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

    def test_daily_features_use_only_observations_through_each_origin(self) -> None:
        annual, cities = make_dataset()
        matrix = metric_matrix(clean_annual_indices(annual).data, "annual_avg_temp")
        metadata = station_metadata(cities)
        daily_rows = []
        for station_id in matrix.columns:
            for year in range(1981, 2012):
                maximum = 90.0 if year == 2010 else 70.0
                if year == 2011:
                    maximum = 110.0
                for month in (1, 7):
                    daily_rows.append(
                        {
                            "ID": station_id,
                            "Date": pd.Timestamp(year=year, month=month, day=15),
                            "tmax": maximum,
                            "tmin": maximum - 20,
                            "prcp": 0.1,
                        }
                    )
        daily = pd.DataFrame(daily_rows)
        changed_future = daily.copy()
        changed_future.loc[
            changed_future["Date"].dt.year == 2011, ["tmax", "tmin", "prcp"]
        ] = [1000.0, 900.0, 100.0]
        config = GlobalModelConfig(first_origin=1989)

        original_panel = build_panel(
            matrix, metadata, horizon=1, config=config, daily_observations=daily
        )
        changed_panel = build_panel(
            matrix,
            metadata,
            horizon=1,
            config=config,
            daily_observations=changed_future,
        )
        daily_columns = [
            column for column in original_panel if column.startswith("daily_")
        ]
        self.assertTrue(daily_columns)
        original = original_panel.loc[
            original_panel["origin"] == 2010, daily_columns
        ].reset_index(drop=True)
        changed = changed_panel.loc[
            changed_panel["origin"] == 2010, daily_columns
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(original, changed)
        self.assertTrue(original["daily_djf_temp_a5"].gt(0).all())
        self.assertIn("daily_jja_prcp_p95_a5", daily_columns)
        self.assertIn("daily_jja_wet_fraction_a10", daily_columns)

    def test_seasonal_precipitation_features_capture_tail_and_recent_change(self) -> None:
        station_id = "US000000001"
        daily = pd.DataFrame(
            [
                {
                    "ID": station_id,
                    "Date": pd.Timestamp(year=year, month=7, day=15),
                    "tmax": 80.0,
                    "tmin": 60.0,
                    "prcp": 5.0 if year == 2010 else 0.0,
                }
                for year in range(1991, 2011)
            ]
        )
        station_ids = pd.Index([station_id])
        matrices = _daily_feature_matrices(daily, station_ids)

        self.assertEqual(matrices["jja_prcp_p95"].loc[2010, station_id], 5.0)
        self.assertEqual(matrices["jja_wet_fraction"].loc[2010, station_id], 1.0)
        self.assertEqual(matrices["jja_wet_intensity"].loc[2010, station_id], 5.0)

        features = _daily_features_at_origin(
            matrices,
            station_ids,
            origin=2010,
            config=GlobalModelConfig(
                baseline_window=20,
                average_5_years=5,
            ),
        )
        self.assertGreater(features.loc[station_id, "daily_jja_prcp_a5"], 0.0)
        self.assertGreater(features.loc[station_id, "daily_jja_prcp_a10"], 0.0)
        self.assertGreater(
            features.loc[station_id, "daily_jja_prcp_p95_a10"], 0.0
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
        self.assertEqual(len(metrics), 4 * len(MODEL_METHODS))
        self.assertEqual(selected, FORECAST_METHODS)
        self.assertEqual(len(submission), 2 * 7 * 4)
        self.assertFalse(submission["predicted_value"].isna().any())
        validate_submission(submission, expected_city_ids=["US000000000", "US000000001"])

    def test_selected_method_backtest_only_calibrates_configured_methods(self) -> None:
        annual, city_info = make_dataset()
        cleaned = clean_annual_indices(annual).data
        backtest = run_backtest(
            cleaned,
            city_info,
            origins=[2010],
            horizon=7,
            model_config=GlobalModelConfig(
                first_origin=1989,
                lightgbm_estimators=5,
                lightgbm_min_child_samples=5,
            ),
            methods_by_metric=FORECAST_METHODS,
            feature_columns=("ly_a",),
        )

        self.assertEqual(
            {
                column.removeprefix("_raw_")
                for column in backtest
                if column.startswith("_raw_")
            },
            set(FORECAST_METHODS.values()),
        )

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
