# NFL Injury Data Pipeline

Automated five-step pipeline that keeps the MySQL database up to date.

```
Step 1  fetch_nflverse.R           → pbp.csv, injury_report.csv, schedules.csv, nfl_players.csv
Step 2  fetch_ir_transactions.py   → nfl_injuries_full.csv  (appended)
Step 3  nfl_data/pbp.ipynb         → temp_pbp2.csv  (in-game injury events)
Step 4  nfl_data/clean.ipynb       → cleaned / merged CSVs
Step 5  nfl_data/load.ipynb        → MySQL tables
```

## One-time setup

```bash
# 1. Install Python notebook dependencies
.venv/bin/pip install nbconvert nbformat

# 2. Install the Chromium browser binary for Playwright (IR scraper)
.venv/bin/playwright install chromium
```

R and the `nflreadr` package must already be installed
(`/usr/local/bin/Rscript` and `install.packages("nflreadr")` in R).

---

## Running the pipeline

From the `Capstone/` project root:

```bash
# Weekly incremental update (current season only — fast, ~2-5 min)
python pipeline/run_pipeline.py

# Pull all data from a specific year forward (e.g. first-time historical load)
python pipeline/run_pipeline.py --from-year 2010

# Full rebuild from scratch (1999–present for pbp; 2009+ for injury reports)
python pipeline/run_pipeline.py --full-refresh

# Skip steps you've already run
python pipeline/run_pipeline.py --skip-nflverse --skip-ir

# Specify a custom IR scrape start date
python pipeline/run_pipeline.py --ir-from 2025-01-01

# Override IR end date and overlap window
python pipeline/run_pipeline.py --ir-to 2026-04-14 --ir-backfill-days 21
```

### All flags

| Flag | Description |
|------|-------------|
| `--full-refresh` | Pull all historical data; re-scrape all IR transactions |
| `--from-year YEAR` | Pull nflverse data from this season year (e.g. `2010`) |
| `--skip-nflverse` | Skip Step 1 |
| `--skip-ir` | Skip Step 2 |
| `--skip-pbp` | Skip Step 3 |
| `--skip-clean` | Skip Step 4 |
| `--skip-load` | Skip Step 5 |
| `--ir-from DATE` | Override IR scrape start date (YYYY-MM-DD) |
| `--ir-to DATE` | Override IR scrape end date (YYYY-MM-DD) |
| `--ir-backfill-days N` | In incremental mode, re-scrape N days before watermark |

---

## First historical load (one time)

To pull the entire available history and populate the database from scratch:

```bash
# Export your existing database first as a safety backup
mysqldump -u root -p nfl_injuries > backup_$(date +%Y%m%d).sql

# Then run the full pipeline
python pipeline/run_pipeline.py --full-refresh
```

Expected runtime: 30–90 minutes depending on connection speed.
The nflreadr package caches parquet files locally so re-running is much faster.

---

## Organized file layout

IR scraping now keeps metadata and backups in dedicated locations:

- `pipeline/state/pipeline_state.json`:
	stores IR watermark and last run metadata.
- `nfl_data/backups/`:
	pre-refresh full CSV backups.

## State tracking

`pipeline/state/pipeline_state.json` tracks:

- `ir_last_scraped_date`: max transaction date actually scraped.
- `ir_last_run_date`: when the scraper last ran.
- `ir_last_begin`, `ir_last_end`: last date window used.
- `ir_backfill_days`: overlap window used.

Incremental runs now re-scrape a rolling overlap (`--ir-backfill-days`, default 14)
to catch late source edits while still deduping rows safely.

---

## Data freshness schedule (suggested)

| Frequency | Command |
|-----------|---------|
| Daily or every 6-12h (during season) | `python pipeline/run_pipeline.py --ir-backfill-days 21` |
| Off-season | `python pipeline/run_pipeline.py --skip-ir` |
| New season start | `python pipeline/run_pipeline.py` (incremental) |

---

## Database migration (when phpMyAdmin access expires)

### Recommended: run MySQL locally

The simplest option — keep everything on your own machine.

```bash
# macOS (Homebrew)
brew install mysql
brew services start mysql
mysql_secure_installation

# Import your backup
mysql -u root -p nfl_injuries < backup_YYYYMMDD.sql
```

Update `nfl_data/config.py` to point to `host="127.0.0.1"`.

### Free cloud option: TiDB Cloud Serverless

MySQL-compatible (pymysql works with zero code changes), 5 GB free tier,
no credit card required. https://tidbcloud.com

1. Create a free cluster at tidbcloud.com
2. Download the connection string (host + port + credentials)
3. Update `nfl_data/config.py` with the new host/port/user/password/ssl settings
4. Import: `mysql -u <user> -h <host> -P <port> -p --ssl-mode=VERIFY_IDENTITY nfl_injuries < backup.sql`

### Exporting before graduation

```bash
# Full dump (schema + data)
mysqldump -u <user> -h <host> -p nfl_injuries > nfl_injuries_backup_$(date +%Y%m%d).sql

# Data only (no CREATE TABLE statements)
mysqldump -u <user> -h <host> -p --no-create-info nfl_injuries > data_only.sql
```
