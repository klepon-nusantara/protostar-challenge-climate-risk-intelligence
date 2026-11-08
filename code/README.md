# Code

- `src/climate_challenge/data.py` downloads the challenge tables.
- `src/climate_challenge/cleaning.py` cleans annual indices and creates an audit.
- `src/climate_challenge/backtesting.py` runs rolling-origin backtests.
- `src/climate_challenge/models/` contains the baseline and global model code.
- `src/climate_challenge/evaluation.py` scores methods and selects a method per metric.
- `src/climate_challenge/forecasting.py` generates the final forecasts.
- `src/climate_challenge/submission.py` fills and validates the challenge CSV.
- `src/climate_challenge/__main__.py` provides the `prepare`, `backtest`, and
  `forecast` commands.
- `nb/` contains exploratory/example notebooks; the package does not execute them.
- `check_submission.py` validates `submission.csv`.
- `requirements.txt` lists the Python dependencies.

## Preparing data

Install the package in editable mode once from the repository root:

```bash
python -m pip install -e code
```

Run the end-to-end workflows from the repository root:

```bash
python -m climate_challenge prepare
python -m climate_challenge backtest
python -m climate_challenge forecast
```

`forecast` runs a temporal backtest, selects methods per metric, predicts
2019–2025, and writes the filled challenge template to `submission.csv`.
The notebooks are optional examples, not pipeline entry points.

The tables are also available as Python objects:

```python
from climate_challenge.data import prepare_dataset

prepared = prepare_dataset()
annual = prepared.annual_indices
cities = prepared.city_info
```

Missing Parquet files are downloaded directly into the repository's `data/`
directory. Subsequent runs read those local files. Additional feature sources
can be supplied as `DataSource` objects. Their loaders receive the forecast
cutoff and should apply it to any download query. Weather-dependent sources
must declare their date column; records after the cutoff are rejected. Sources
are returned separately so callers can join each one using an explicit key.

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
