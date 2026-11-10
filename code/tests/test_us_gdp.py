import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pandas as pd

from climate_challenge.us_gdp import preprocess_us_gdp


class UsGdpPreprocessingTests(unittest.TestCase):
    def test_preprocesses_all_area_data_and_caches_parquet(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            archive_path = data_dir / "CAGDP1.zip"
            all_areas_csv = (
                "GeoFIPS,GeoName,Region,TableName,LineCode,"
                "IndustryClassification,Description,Unit,2001,2002\n"
                '"00000","United States",,CAGDP1 ,1,...,All industry total,'
                "Thousands of dollars,100,110\n"
                '"01001","Autauga County, Alabama",,CAGDP1,1,...,'
                "All industry total,Thousands of dollars,10,12\n"
            )
            state_csv = (
                "GeoFIPS,GeoName,Region,TableName,LineCode,"
                "IndustryClassification,Description,Unit,2001,2002\n"
                '"01000","Alabama",,CAGDP1,1,...,All industry total,'
                "Thousands of dollars,20,22\n"
            )
            with ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "CAGDP1__ALL_AREAS_2001_2002.csv", all_areas_csv
                )
                archive.writestr("CAGDP1_AL_2001_2002.csv", state_csv)

            output_paths = preprocess_us_gdp(data_dir)
            self.assertEqual(output_paths, (data_dir / "CAGDP1.parquet",))
            result = pd.read_parquet(output_paths[0])

            self.assertEqual(len(result), 2)
            self.assertEqual(result["GeoFIPS"].tolist(), ["00000", "01001"])
            self.assertEqual(result.loc[0, "TableName"], "CAGDP1")
            self.assertEqual(result.loc[1, "2002"], 12)
            self.assertEqual(
                result.loc[0, "source_member"],
                "CAGDP1__ALL_AREAS_2001_2002.csv",
            )

            cached_mtime = output_paths[0].stat().st_mtime_ns
            self.assertEqual(preprocess_us_gdp(data_dir), output_paths)
            self.assertEqual(output_paths[0].stat().st_mtime_ns, cached_mtime)

    def test_requires_an_annual_bea_archive(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(FileNotFoundError, "No annual CAGDP"):
                preprocess_us_gdp(temporary_directory)


if __name__ == "__main__":
    unittest.main()
