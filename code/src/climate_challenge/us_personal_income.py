"""Preprocess annual BEA regional personal income archives."""

from __future__ import annotations

from pathlib import Path

from .bea import preprocess_annual_archives


def preprocess_us_personal_income(
    data_dir: str | Path,
    force: bool = False,
) -> tuple[Path, ...]:
    """Convert annual CAINC and SAINC all-area tables to cached Parquet."""

    return preprocess_annual_archives(
        data_dir,
        archive_prefixes=("CAINC", "SAINC"),
        dataset_label="annual CAINC or SAINC",
        force=force,
    )
