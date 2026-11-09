import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from climate_challenge.config import MODEL_METHODS
from climate_challenge.models.global_models import GlobalModelConfig
from climate_challenge.oot_eval import append_oot_run_log, run_oot_evaluation
from climate_challenge.oot_eval.preprocessing import preprocess_oot_eval


class OotPreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.raw_dir = self.root / "7890488"
        self.raw_dir.mkdir()
        pd.DataFrame(
            {
                "": [1, 2],
                "Name": ["Example", "Example"],
                "ID": ["US000000001", "US000000001"],
                "Lat": [40.0, 40.0],
                "Lon": [-80.0, -80.0],
                "Stn.Name": ["OLD", "NEW"],
                "Stn.stDate": ["2010-01-01", "2020-01-01"],
                "Stn.edDate": ["2019-12-31", "2020-12-31"],
            }
        ).to_csv(self.raw_dir / "city_info.csv", index=False)
        pd.DataFrame(
            {
                "": [1, 2, 3, 4],
                "Date": [
                    "2018-12-31",
                    "2019-01-01",
                    "2019-01-02",
                    "2020-02-29",
                ],
                "tmax": [100.0, 80.0, 50.0, 70.0],
                "tmin": [90.0, 70.0, np.nan, 60.0],
                "prcp": [5.0, 0.2, 0.1, np.nan],
            }
        ).to_csv(self.raw_dir / "US000000001.csv", index=False)

    def test_preprocesses_full_calendar_years_and_expected_indices(self) -> None:
        annual, city_info = preprocess_oot_eval(
            self.root, self.root / "data"
        )

        self.assertEqual(annual[["ID", "Year"]].values.tolist(), [
            ["US000000001", 2019],
            ["US000000001", 2020],
        ])
        year_2019 = annual.loc[annual["Year"] == 2019].iloc[0]
        self.assertEqual(year_2019["Avg Temp"], 67.5)
        self.assertAlmostEqual(year_2019["ToT Prcp"], 0.3)
        self.assertEqual(year_2019["Max Tmax"], 80.0)
        self.assertEqual(year_2019["cdd"], 10.0)
        self.assertEqual(year_2019["Miss Tmax"], 363)
        self.assertEqual(year_2019["Miss Tmin"], 364)
        self.assertEqual(year_2019["Miss Prcp"], 363)

        year_2020 = annual.loc[annual["Year"] == 2020].iloc[0]
        self.assertEqual(year_2020["Miss Tmax"], 365)
        self.assertEqual(year_2020["Miss Tmin"], 365)
        self.assertTrue(np.isnan(year_2020["ToT Prcp"]))
        self.assertEqual(len(city_info), 2)
        self.assertEqual(city_info["Stn.stDate"].min(), pd.Timestamp("2019-01-01"))
        self.assertTrue((city_info["Stn.edDate"] >= pd.Timestamp("2019-01-01")).all())
        self.assertTrue((annual["Year"] > 2018).all())
        self.assertTrue(
            (self.root / "data" / "annual_indices_oot.parquet").is_file()
        )
        self.assertTrue((self.root / "data" / "city_info_oot.parquet").is_file())

class OotEvaluationTests(unittest.TestCase):
    def test_run_log_appends_models_configuration_and_scores(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "oot_eval.log"
            backtest = pd.DataFrame(
                {
                    "target_year": [2019, 2020],
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
            scores = pd.DataFrame(
                {
                    "metric": ["annual_avg_temp"],
                    "h": [1],
                    "method": ["Ridge"],
                    "mae": [0.5],
                    "n": [2],
                }
            )
            audit = pd.DataFrame(
                {"raw_missing": [0]}, index=["annual_avg_temp"]
            )

            append_oot_run_log(
                log_path,
                backtest,
                scores,
                audit,
                origin=2018,
                input_paths=("data/annual_indices.parquet",),
            )
            append_oot_run_log(
                log_path,
                backtest,
                scores,
                audit,
                origin=2018,
                input_paths=("data/annual_indices.parquet",),
            )

            log_content = log_path.read_text(encoding="utf-8")
            self.assertEqual(log_content.count("OOT evaluation run:"), 2)
            self.assertIn("Origin: 2018", log_content)
            self.assertIn("Scored period: 2019-2020", log_content)
            self.assertIn("Models experimented: Ridge", log_content)
            self.assertIn('"ridge_alpha": 10.0', log_content)
            self.assertIn("Features used by Ridge and LightGBM: ly_a, a5_a", log_content)
            self.assertIn("Feature note: other listed methods are fixed baselines", log_content)
            self.assertIn("annual_avg_temp", log_content)
            self.assertIn("data/annual_indices.parquet", log_content)
            self.assertIn(
                "Overall pooled MAE across the full scored period:", log_content
            )
            self.assertIn("Best pooled MAE by metric:", log_content)
            self.assertRegex(
                log_content,
                r"annual_avg_temp\s+30-Yr Avg\s+0\.000000\s+2",
            )
            self.assertIn("2.000000", log_content)

    def test_fixed_origin_scores_all_methods_by_lead(self) -> None:
        historical_rows = []
        oot_rows = []
        cities = []
        for station_number in range(6):
            station_id = f"US{station_number:09d}"
            name = f"City {station_number}"
            cities.append(
                {
                    "ID": station_id,
                    "Name": name,
                    "Lat": 30.0 + station_number,
                    "Lon": -100.0 + station_number,
                }
            )
            for year in range(1980, 2024):
                temperature = 60 + station_number + 0.03 * (year - 1980)
                row = {
                    "ID": station_id,
                    "Name": name,
                    "Year": year,
                    "Miss Tmax": 0,
                    "Miss Tmin": 0,
                    "Miss Prcp": 0,
                    "Avg Temp": temperature,
                    "ToT Prcp": 25 + station_number + year % 5,
                    "Max Tmax": temperature + 30,
                    "cdd": 80 + station_number * 10 + year % 7,
                }
                (historical_rows if year <= 2018 else oot_rows).append(row)

        backtest, scores, _ = run_oot_evaluation(
            pd.DataFrame(historical_rows),
            pd.DataFrame(oot_rows),
            pd.DataFrame(cities),
            model_config=GlobalModelConfig(
                first_origin=1989,
                lightgbm_estimators=3,
                lightgbm_min_child_samples=5,
            ),
        )

        self.assertEqual(set(backtest["origin"]), {2018})
        self.assertEqual(set(backtest["target_year"]), set(range(2019, 2024)))
        self.assertEqual(set(scores["h"]), set(range(1, 6)))
        self.assertEqual(set(scores["method"]), set(MODEL_METHODS))
        self.assertTrue(scores["n"].gt(0).all())

    def test_rejects_non_oot_years_in_oot_targets(self) -> None:
        historical = pd.DataFrame(
            [
                {
                    "ID": "US000000001",
                    "Name": "Example",
                    "Year": 2018,
                    "Miss Tmax": 0,
                    "Miss Tmin": 0,
                    "Miss Prcp": 0,
                    "Avg Temp": 60.0,
                    "ToT Prcp": 20.0,
                    "Max Tmax": 90.0,
                    "cdd": 50.0,
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "only years after origin"):
            run_oot_evaluation(
                historical,
                historical,
                pd.DataFrame(
                    [{"ID": "US000000001", "Name": "Example", "Lat": 40, "Lon": -80}]
                ),
            )


if __name__ == "__main__":
    unittest.main()
