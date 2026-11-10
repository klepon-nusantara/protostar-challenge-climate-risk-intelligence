import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, box

from climate_challenge.data import DataSource, ORIGINAL_FILES, prepare_dataset


class PrepareDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.data_dir = Path(self.temporary_directory.name)
        self._write_challenge_tables()

    def _write_challenge_tables(self) -> None:
        pd.DataFrame(
            {
                "Name": ["Example"],
                "ID": ["US000000001"],
                "Year": [2018],
            }
        ).to_parquet(self.data_dir / "annual_indices.parquet")
        pd.DataFrame(
            {
                "Name": ["Example"],
                "ID": ["US000000001"],
                "Lat": [40.0],
                "Lon": [-80.0],
            }
        ).to_parquet(self.data_dir / "city_info.parquet")
        pd.DataFrame(
            {"ID": ["US000000001"], "elevation_m": [250.0]}
        ).to_parquet(self.data_dir / "elevation.parquet")
        pd.DataFrame(
            {
                "ID": ["US000000001"],
                "Name": ["Example"],
                "Lat": [40.0],
                "Lon": [-80.0],
                "division_CLIMDIV": ["0101"],
            }
        ).to_parquet(self.data_dir / "city_climate_divisions.parquet")
        pd.DataFrame(
            {
                "ID": ["US000000001"],
                "Name": ["Example"],
                "Lat": [40.0],
                "Lon": [-80.0],
                "distance_to_coast_km": [12.5],
            }
        ).to_parquet(self.data_dir / "city_coastline_distances.parquet")

    def test_loads_challenge_tables_from_data_directory(self) -> None:
        with patch("climate_challenge.data.urlopen") as open_url:
            prepared = prepare_dataset(data_dir=self.data_dir)

        open_url.assert_not_called()
        self.assertEqual(prepared.annual_indices.loc[0, "Year"], 2018)
        self.assertIn("US000000001", prepared.city_info["ID"].tolist())
        self.assertEqual(prepared.elevation.loc[0, "elevation_m"], 250.0)
        self.assertEqual(
            prepared.city_climate_divisions.loc[0, "division_CLIMDIV"], "0101"
        )
        self.assertEqual(
            prepared.city_coastline_distances.loc[0, "distance_to_coast_km"],
            12.5,
        )
        self.assertEqual(len(prepared.additional_sources), 0)
        self.assertEqual(prepared.data_dir, self.data_dir)

    def test_spatially_joins_stations_and_caches_division_attributes(self) -> None:
        (self.data_dir / "city_climate_divisions.parquet").unlink()
        archive_path = self.data_dir / "CONUS_CLIMATE_DIVISIONS.shp.zip"
        archive_path.touch()
        divisions = gpd.GeoDataFrame(
            {"CLIMDIV": ["0101"]},
            geometry=[box(-81.0, 39.0, -79.0, 41.0)],
            crs="EPSG:4326",
        )

        with patch("geopandas.read_file", return_value=divisions) as read_file:
            prepared = prepare_dataset(data_dir=self.data_dir)

        read_file.assert_called_once_with(archive_path, engine="pyogrio")
        self.assertTrue((self.data_dir / "city_climate_divisions.parquet").is_file())
        self.assertEqual(len(prepared.city_climate_divisions), 1)
        self.assertEqual(
            prepared.city_climate_divisions.loc[0, "division_CLIMDIV"], "0101"
        )

    def test_computes_and_caches_nearest_coastline_distance(self) -> None:
        (self.data_dir / "city_coastline_distances.parquet").unlink()
        archive_path = self.data_dir / "ne_10m_coastline.zip"
        archive_path.touch()
        coastline = gpd.GeoDataFrame(
            {"featurecla": ["Coastline"]},
            geometry=[LineString([(-81.0, 39.9), (-79.0, 39.9)])],
            crs="EPSG:4326",
        )

        with patch("geopandas.read_file", return_value=coastline) as read_file:
            prepared = prepare_dataset(data_dir=self.data_dir)

        read_file.assert_called_once_with(archive_path, engine="pyogrio")
        self.assertTrue(
            (self.data_dir / "city_coastline_distances.parquet").is_file()
        )
        self.assertEqual(len(prepared.city_coastline_distances), 1)
        distance = prepared.city_coastline_distances.loc[
            0, "distance_to_coast_km"
        ]
        self.assertGreater(distance, 0)
        self.assertLess(distance, 12)

    def test_additional_source_receives_cutoff_and_is_kept_separate(self) -> None:
        seen_cutoffs: list[pd.Timestamp] = []

        def load_feature(cutoff: pd.Timestamp) -> pd.DataFrame:
            seen_cutoffs.append(cutoff)
            return pd.DataFrame({"city_id": ["US000000001"], "elevation_m": [10]})

        prepared = prepare_dataset(
            [DataSource("elevation", load_feature, weather_dependent=False)],
            data_dir=self.data_dir,
        )

        self.assertEqual(seen_cutoffs, [pd.Timestamp("2018-12-31")])
        self.assertIn("elevation", prepared.additional_sources)
        self.assertNotIn("elevation_m", prepared.annual_indices.columns)

    def test_rejects_weather_dependent_records_after_cutoff(self) -> None:
        source = DataSource(
            "later-weather",
            lambda _cutoff: pd.DataFrame({"year": [2019]}),
            date_column="year",
        )

        with self.assertRaisesRegex(ValueError, "after the forecast cutoff"):
            prepare_dataset([source], data_dir=self.data_dir)

    def test_requires_date_column_for_weather_dependent_sources(self) -> None:
        source = DataSource("weather", lambda _cutoff: pd.DataFrame({"value": [1]}))

        with self.assertRaisesRegex(ValueError, "must define date_column"):
            prepare_dataset([source], data_dir=self.data_dir)

    def test_partial_year_cutoff_rejects_that_year(self) -> None:
        source = DataSource(
            "weather",
            lambda _cutoff: pd.DataFrame({"year": ["2018"]}),
            date_column="year",
        )

        with self.assertRaisesRegex(ValueError, "after the forecast cutoff"):
            prepare_dataset(
                [source], cutoff="2018-06-30", data_dir=self.data_dir
            )

    def test_partial_year_cutoff_rejects_incomplete_annual_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "after the forecast cutoff"):
            prepare_dataset(cutoff="2018-06-30", data_dir=self.data_dir)

    def test_downloads_missing_tables_directly_into_data_directory(self) -> None:
        for path in self.data_dir.glob("*.parquet"):
            path.unlink()
        tables = {
            "annual_indices": pd.DataFrame(
                {"Name": ["Example"], "ID": ["US000000001"], "Year": [2018]}
            ),
            "city_info": pd.DataFrame(
                {
                    "Name": ["Example"],
                    "ID": ["US000000001"],
                    "Lat": [40.0],
                    "Lon": [-80.0],
                }
            ),
        }
        responses: dict[str, bytes] = {}
        for name, frame in tables.items():
            buffer = BytesIO()
            frame.to_parquet(buffer)
            responses[name] = buffer.getvalue()

        def fake_urlopen(request: object, timeout: int = 30) -> BytesIO:
            url = getattr(request, "full_url", request)
            if url == "https://api.open-elevation.com/api/v1/lookup":
                return BytesIO(
                    json.dumps({"results": [{"elevation": 250.0}]}).encode()
                )
            for name, relative_path in ORIGINAL_FILES.items():
                if f"/{relative_path}" in url:
                    return BytesIO(responses[name])
            raise AssertionError(f"Unexpected URL: {url}")

        archive_path = self.data_dir / "CONUS_CLIMATE_DIVISIONS.shp.zip"
        archive_path.touch()
        (self.data_dir / "ne_10m_coastline.zip").touch()
        divisions = gpd.GeoDataFrame(
            {"CLIMDIV": ["0101"]},
            geometry=[box(-81.0, 39.0, -79.0, 41.0)],
            crs="EPSG:4326",
        )
        with (
            patch("climate_challenge.data.urlopen", side_effect=fake_urlopen),
            patch(
                "geopandas.read_file",
                side_effect=[
                    divisions,
                    gpd.GeoDataFrame(
                        {"featurecla": ["Coastline"]},
                        geometry=[LineString([(-81.0, 39.9), (-79.0, 39.9)])],
                        crs="EPSG:4326",
                    ),
                ],
            ),
        ):
            prepared = prepare_dataset(data_dir=self.data_dir)

        self.assertTrue((self.data_dir / "annual_indices.parquet").exists())
        self.assertTrue((self.data_dir / "city_info.parquet").exists())
        self.assertTrue((self.data_dir / "elevation.parquet").exists())
        self.assertTrue((self.data_dir / "city_climate_divisions.parquet").exists())
        self.assertTrue(
            (self.data_dir / "city_coastline_distances.parquet").exists()
        )
        self.assertEqual(prepared.elevation.loc[0, "elevation_m"], 250.0)
        self.assertEqual(len(prepared.city_climate_divisions), 1)
        self.assertEqual(len(prepared.annual_indices), 1)


if __name__ == "__main__":
    unittest.main()
