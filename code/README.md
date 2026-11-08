# Code

- `src/climate_challenge/data.py` downloads the challenge's `annual_indices` and
  `city_info` tables and validates their schemas and forecast cutoff.
- `nb/` contains exploratory notebooks.
- `check_submission.py` validates `submission.csv`.
- `requirements.txt` lists the Python dependencies.

## Preparing data

Install the package in editable mode once from the repository root:

```bash
python -m pip install -e code
```

Then download and validate the original tables from the repository root:

```bash
python -m climate_challenge
```

The downloaded tables are stored under the project's `data/` directory. In
Python, they can also be loaded with:

```python
from climate_challenge.data import prepare_dataset

prepared = prepare_dataset()
annual = prepared.annual_indices
cities = prepared.city_info
```

Missing Parquet files are downloaded directly into the repository's `data/`
directory. Subsequent runs read those local files without using the
Hugging Face Datasets cache.
Additional feature sources can be supplied as `DataSource` objects. Their
loaders receive the forecast cutoff and should apply it to any download query.
Weather-dependent sources must declare their date column; records after the
cutoff are rejected. Sources are returned separately so callers can join each
one using an explicit key.

```python
import pandas as pd

from climate_challenge.data import DataSource, prepare_dataset

elevation = DataSource(
    name="elevation",
    loader=lambda cutoff: pd.read_csv("data/elevation.csv"),
    weather_dependent=False,
)
prepared = prepare_dataset(additional_sources=[elevation])
```

Run the unit tests from the repository root with:

```bash
python -m unittest discover -s code/tests
```

`AGENTS.md` documents the challenge's data-use rules. Record any curated
external data in [`../data/README.md`](../data/README.md).
