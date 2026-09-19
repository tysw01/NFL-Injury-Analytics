# NFL Injury Analytics Platform

An end-to-end NFL injury analytics application that combines multi-source data engineering, machine learning, and a Flask API to estimate player availability and return-to-play outcomes.

## What It Does

The platform answers two practical questions:

- How likely is an injured player to participate in the next game?
- How quickly is a player likely to return after an injury?

The project ingests public NFL injury reports, IR transactions, player data, snap counts, schedules, and play-by-play data. The data pipeline cleans and joins these sources before loading them into MySQL. A scikit-learn model then uses injury characteristics, report status, practice status, player history, demographics, and game context to estimate next-game availability and return timelines.

## Technical Highlights


## Repository Structure

```text
NFL-Injury-Analytics/
├── injuryAPI.py                 # Flask application and API routes
├── requirements.txt             # Python dependencies
├── .env.example                 # Safe database configuration template
├── field_descriptions_overrides.json
├── nfl_data/
│   ├── config.py                # Environment-driven database settings
│   ├── dictionary_*.csv         # Database field documentation
│   ├── pbp.ipynb                # Parse in-game injury events
│   ├── clean.ipynb              # Clean and merge source data
│   ├── load.ipynb               # Load final data into MySQL
│   └── ml_predict.ipynb          # Model analysis and validation
├── pipeline/
│   ├── run_pipeline.py          # Five-step pipeline orchestrator
│   ├── fetch_nflverse.R         # Download public nflverse data
│   ├── fetch_ir_transactions.py # Scrape IR transaction history
│   ├── fetch_snap_counts.py     # Load snap-count data
│   └── README.md                # Pipeline commands and flags
├── templates/                   # Jinja2 application pages
├── static/                      # CSS and dashboard assets
└── tests/                       # Smoke tests and historical backtesting
```

Large downloaded CSVs, model caches, backups, credentials, and generated outputs are intentionally excluded from this repository. They are created locally by the data pipeline or supplied through a configured database.

## Requirements


## Local Setup

```bash
git clone <your-repository-url>
cd NFL-Injury-Analytics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
```

Edit `.env` with a local database configuration, then export those variables before running the application. The project reads environment variables directly and does not load `.env` files automatically:

```bash
set -a
source .env
set +a
```

The expected variables are:

```text
NFL_DB_HOST=127.0.0.1
NFL_DB_PORT=3306
NFL_DB_USER=root
NFL_DB_PASSWORD=your-local-password
NFL_DB_NAME=nfl_injuries
```

## Build the Database

Run the full pipeline from the repository root:

```bash
python pipeline/run_pipeline.py --full-refresh
```

For regular updates:

```bash
python pipeline/run_pipeline.py
```

The pipeline downloads source files into ignored local paths, executes the cleaning and parsing notebooks, and loads the resulting tables into MySQL. See `pipeline/README.md` for incremental refreshes, skip flags, and scraper options.

## Run the Application

```bash
python injuryAPI.py
```

Open [http://127.0.0.1:5001](http://127.0.0.1:5001). The first model warmup may take several minutes because training data is assembled from the database. A local model cache is generated afterward and is excluded from version control.

## Test and Backtest

Start the API first, then run:

```bash
python tests/run_tests.py
python tests/run_backtest.py
```

The smoke test checks prediction response structure. The backtest compares historical availability predictions against snap-count participation for selected 2024 weeks.

## Data and Modeling Notes

The training labels define availability from whether a player appears in the next game’s snap-count data. The model is intended as an analytics and research tool, not a medical diagnosis or clinical decision system. Public injury reports can be incomplete, and predictions should be interpreted alongside the model’s validation metrics and limitations.

## Future Improvements

- Add a reproducible local database bootstrap with a small permitted sample dataset.
- Replace notebook-driven transformation steps with tested Python pipeline modules.
- Add automated unit and integration tests for API responses and data-quality checks.
- Evaluate temporal cross-validation and calibration drift as new seasons are added.
- Add CI checks for syntax, secrets, and model evaluation artifacts.

## Data Sources

The project uses publicly available NFL data distributed through nflverse and public injury transaction sources. Review each source’s terms before redistributing downloaded data.
