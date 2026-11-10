"""Shared preprocessing for annual BEA regional archives."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

REQUIRED_COLUMNS = {
    "GeoFIPS",
    "GeoName",
    "TableName",
    "LineCode",
    "Description",
    "Unit",
}


def preprocess_annual_archives(
    data_dir: str | Path,
    archive_prefixes: tuple[str, ...],
    dataset_label: str,
    force: bool = False,
) -> tuple[Path, ...]:
    """Convert annual BEA all-area CSVs from matching ZIP archives to Parquet."""

    source_directory = Path(data_dir)
    if not source_directory.is_dir():
        raise FileNotFoundError(
            f"{dataset_label} directory does not exist: {source_directory}"
        )

    archives = sorted(
        archive
        for archive in source_directory.glob("*.zip")
        if archive.name.startswith(archive_prefixes)
    )
    if not archives:
        raise FileNotFoundError(
            f"No {dataset_label} ZIP archives found in {source_directory}"
        )

    output_paths: list[Path] = []
    for archive in archives:
        output_path = archive.with_suffix(".parquet")
        if (
            not force
            and output_path.is_file()
            and output_path.stat().st_mtime_ns >= archive.stat().st_mtime_ns
        ):
            output_paths.append(output_path)
            continue

        with ZipFile(archive) as source:
            members = sorted(
                name
                for name in source.namelist()
                if "__ALL_AREAS_" in Path(name).name
                and name.lower().endswith(".csv")
            )
            if not members:
                raise ValueError(
                    f"No annual all-area CSV found in BEA archive {archive}"
                )

            tables: list[pd.DataFrame] = []
            for member in members:
                frame = pd.read_csv(
                    source.open(member),
                    dtype={"GeoFIPS": "string"},
                    encoding="latin1",
                    low_memory=False,
                )
                missing = REQUIRED_COLUMNS.difference(frame.columns)
                if missing:
                    raise ValueError(
                        f"{member} is missing required columns: "
                        + ", ".join(sorted(missing))
                    )

                year_columns = [
                    column for column in frame.columns if column.isdigit()
                ]
                if not year_columns:
                    raise ValueError(f"{member} has no annual year columns")
                metadata_columns = [
                    column for column in frame.columns if column not in year_columns
                ]
                for column in metadata_columns:
                    frame[column] = (
                        frame[column].astype("string").str.strip()
                    )
                frame["GeoFIPS"] = frame["GeoFIPS"].str.strip('"')
                for column in year_columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")

                frame.insert(0, "source_member", Path(member).name)
                tables.append(frame)

        result = pd.concat(tables, ignore_index=True)
        temporary_path = output_path.with_suffix(".parquet.tmp")
        try:
            result.to_parquet(temporary_path, index=False)
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        output_paths.append(output_path)

    return tuple(output_paths)
