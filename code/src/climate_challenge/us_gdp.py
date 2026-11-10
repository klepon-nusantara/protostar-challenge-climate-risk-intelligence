"""Preprocess annual BEA regional GDP archives."""

from __future__ import annotations

from pathlib import Path

from .bea import preprocess_annual_archives


def preprocess_us_gdp(
    data_dir: str | Path,
    force: bool = False,
) -> tuple[Path, ...]:
    """Convert annual CAGDP and SAGDP all-area tables to cached Parquet."""

    return preprocess_annual_archives(
        data_dir,
        archive_prefixes=("CAGDP", "SAGDP"),
        dataset_label="annual CAGDP or SAGDP",
        force=force,
    )
