"""Command-line entry point for downloading the challenge dataset."""

from .data import prepare_dataset


def main() -> None:
    prepared = prepare_dataset()
    years = prepared.annual_indices["Year"]
    print(
        "Downloaded and validated challenge data "
        f"(annual_indices: {len(prepared.annual_indices):,} rows, "
        f"{int(years.min())}-{int(years.max())}; "
        f"city_info: {len(prepared.city_info):,} rows)."
    )
    print(f"Parquet files are stored in {prepared.data_dir}.")


if __name__ == "__main__":
    main()
