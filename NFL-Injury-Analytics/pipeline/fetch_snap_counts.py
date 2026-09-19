"""
fetch_snap_counts.py
====================
Downloads NFL snap count data from nflverse via nfl_data_py and loads it into
the snap_counts MySQL table.

Usage:
    python pipeline/fetch_snap_counts.py                 # all seasons 2013-2025
    python pipeline/fetch_snap_counts.py 2023 2024       # specific seasons
    python pipeline/fetch_snap_counts.py --csv-only      # just save CSV, skip DB load
    python pipeline/fetch_snap_counts.py --db-only       # load existing CSV into DB

Requires: nfl_data_py, pandas, pymysql
Install:  pip install nfl-data-py --no-deps  (avoids pandas downgrade)
          pip install fastparquet fsspec pandas
"""
import argparse
import os
import sys

# Allow running from repo root or pipeline/  — insert the project root so that
# `from nfl_data import config` resolves regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pymysql
import pymysql.cursors

from nfl_data import config  # DB host/port/user/passwd/db constants

# Absolute path to the local CSV cache for snap counts
CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "nfl_data",
    "snap_counts.csv",
)

# nflverse snap-count data starts from 2013 (when Pro Football Reference began tracking).
# 2025 is set as the upper bound; adjust when a new season begins.
SEASONS = list(range(2013, 2026))


def download_snap_counts(seasons: list[int]) -> pd.DataFrame:
    """
    Download snap count data for the given seasons from nflverse via nfl_data_py.

    Returns a cleaned DataFrame with columns matching our MySQL snap_counts schema.
    All numeric snap/percentage columns are coerced to the correct dtype and NaNs
    are filled with 0 so INSERT statements don't fail on NULL constraints.
    """
    try:
        import nfl_data_py as nfl
    except ImportError:
        # Give a clear fix message if the optional dependency is missing
        print("ERROR: nfl_data_py is not installed.")
        print("  pip install nfl-data-py --no-deps")
        sys.exit(1)

    print(f"Downloading snap counts for {len(seasons)} seasons: {seasons[0]}–{seasons[-1]} ...")
    df = nfl.import_snap_counts(seasons)  # fetches parquet files from nflverse S3

    # Rename columns that differ between nfl_data_py and our DB schema
    rename = {
        "player": "player_name",  # nfl_data_py uses 'player'; our DB uses 'player_name'
    }
    df = df.rename(columns=rename)

    # Coerce snap count columns to integer, filling any NaN gaps with 0
    for col in ["offense_snaps", "defense_snaps", "st_snaps"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Coerce percentage columns to float with 4 decimal places
    for col in ["offense_pct", "defense_pct", "st_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).round(4)

    # Normalise string columns: strip whitespace, uppercase, fill NaN
    df["game_type"] = df["game_type"].fillna("REG").str.upper().str.strip()
    df["pfr_player_id"] = df["pfr_player_id"].fillna("")
    df["position"] = df.get("position", "").fillna("").str.upper().str.strip()
    df["team"] = df["team"].fillna("").str.upper().str.strip()
    df["opponent"] = df["opponent"].fillna("").str.upper().str.strip()

    print(f"  → {len(df):,} rows, {df['season'].nunique()} seasons, {df['player_name'].nunique():,} players")
    return df


def save_csv(df: pd.DataFrame, path: str = CSV_PATH) -> None:
    """Persist the DataFrame to a CSV file, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)  # don't write the DataFrame index as a column
    print(f"Saved to {path}")


def load_csv(path: str = CSV_PATH) -> pd.DataFrame:
    """Load a previously saved snap-count CSV back into a DataFrame."""
    if not os.path.exists(path):
        # Caller must run the download step first; --db-only alone won't work
        raise FileNotFoundError(f"CSV not found: {path}. Run without --db-only first.")
    df = pd.read_csv(path, low_memory=False)  # low_memory=False prevents mixed-type warnings
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def upsert_to_db(df: pd.DataFrame, batch_size: int = 2000) -> None:
    """
    Upsert snap count rows into the MySQL snap_counts table.

    Uses INSERT ... ON DUPLICATE KEY UPDATE so that re-running the script is
    idempotent — existing rows are updated with the latest values rather than
    causing duplicate-key errors.  Rows are batched (default 2000 at a time)
    to avoid single giant transactions that could time out on slow connections.
    """
    conn = pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        passwd=config.passwd,
        db=config.db,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=30,
    )

    # Columns we write explicitly (auto-generated/computed columns in the DB are excluded
    # because MySQL raises an error if you try to INSERT into them).
    db_cols = [
        "game_id", "pfr_game_id", "season", "game_type", "week",
        "player_name", "pfr_player_id", "position", "team", "opponent",
        "offense_snaps", "offense_pct", "defense_snaps", "defense_pct",
        "st_snaps", "st_pct",
    ]

    # Build the SQL statement dynamically so adding a column only requires
    # updating db_cols above rather than touching the SQL string.
    placeholders = ", ".join(["%s"] * len(db_cols))   # one %s per column
    col_list = ", ".join(db_cols)
    # For ON DUPLICATE KEY UPDATE, update every non-key column
    update_clause = ", ".join(
        f"{c}=VALUES({c})"
        for c in db_cols
        if c not in ("game_id", "pfr_player_id")  # skip primary/unique key columns
    )

    sql = (
        f"INSERT INTO snap_counts ({col_list}) VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {update_clause}"
    )

    # Convert DataFrame rows to a list of tuples; replace NaN / empty string with None
    # so pymysql sends proper SQL NULLs instead of the string 'nan'.
    rows = []
    for _, row in df[db_cols].iterrows():
        rows.append(tuple(
            None if pd.isna(v) or v == "" else v
            for v in row
        ))

    try:
        cursor = conn.cursor()
        total = 0
        # Insert in batches to keep transaction size manageable
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            cursor.executemany(sql, batch)  # sends all rows in the batch in a single round-trip
            conn.commit()                   # commit after each batch so partial progress is saved
            total += len(batch)
            pct = int(100 * total / len(rows))
            print(f"\r  Inserted {total:,}/{len(rows):,} rows ({pct}%)", end="", flush=True)
        print()
        print(f"Done — {total:,} rows upserted into snap_counts")
    finally:
        # Always close cursor + connection even if an error occurs mid-batch
        cursor.close()
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NFL snap counts and load into DB")
    parser.add_argument(
        "seasons",
        nargs="*",
        type=int,
        help="Seasons to fetch (e.g. 2023 2024). Defaults to all available seasons.",
    )
    parser.add_argument("--csv-only", action="store_true", help="Save CSV but skip DB load")
    parser.add_argument("--db-only", action="store_true", help="Load CSV into DB, skip download")
    return parser.parse_args()


def main() -> None:
    """Parse CLI arguments and run the download and/or DB load steps."""
    args = parse_args()

    if args.db_only:
        # Skip download; load whatever CSV is already on disk
        df = load_csv()
    else:
        # Determine which seasons to fetch — either from the CLI or the full default range
        seasons = args.seasons if args.seasons else SEASONS
        df = download_snap_counts(seasons)
        save_csv(df)  # always save to CSV first so we have a local backup

    if not args.csv_only:
        # Load (or reload) the CSV data into the MySQL snap_counts table
        print("Loading into database...")
        upsert_to_db(df)
    else:
        print("Skipped DB load (--csv-only).")


if __name__ == "__main__":
    main()
