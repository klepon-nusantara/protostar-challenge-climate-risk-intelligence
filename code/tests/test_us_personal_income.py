import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pandas as pd

from climate_challenge.us_personal_income import preprocess_us_personal_income


class UsPersonalIncomePreprocessingTests(unittest.TestCase):
    def test_preprocesses_annual_cainc_and_sainc_archives(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            csv_content = (
                "GeoFIPS,GeoName,Region,TableName,LineCode,"
                "IndustryClassification,Description,Unit,2001,2002\n"
                '"00000","United States",,CAINC1,1,...,Personal income,'
                "Thousands of dollars,100,110\n"
            )
            for archive_name, member_name in (
                ("CAINC1.zip", "CAINC1__ALL_AREAS_2001_2002.csv"),
                ("SAINC.zip", "SAINC1__ALL_AREAS_2001_2002.csv"),
            ):
                with ZipFile(data_dir / archive_name, "w") as archive:
                    archive.writestr(member_name, csv_content)
                    if archive_name == "SAINC.zip":
                        archive.writestr(
                            "SAINC2__ALL_AREAS_2001_2002.csv",
                            csv_content.replace(
                                '"00000","United States",,CAINC1,1',
                                '"01000","Alabama",1,SAINC2,2',
                            ),
                        )
                    archive.writestr(
                        "CAINC1_AL_2001_2002.csv", csv_content
                    )

            output_paths = preprocess_us_personal_income(data_dir)

            self.assertEqual(
                output_paths,
                (data_dir / "CAINC1.parquet", data_dir / "SAINC.parquet"),
            )
            for path in output_paths:
                result = pd.read_parquet(path)
                expected_rows = 2 if path.name == "SAINC.parquet" else 1
                self.assertEqual(len(result), expected_rows)
                self.assertEqual(result.loc[0, "GeoFIPS"], "00000")
                self.assertEqual(result.loc[0, "2002"], 110)
                self.assertIn("__ALL_AREAS_", result.loc[0, "source_member"])

    def test_requires_a_personal_income_archive(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(FileNotFoundError, "No annual CAINC"):
                preprocess_us_personal_income(temporary_directory)


if __name__ == "__main__":
    unittest.main()
