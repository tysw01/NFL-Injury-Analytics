"""
pipeline/run_pipeline.py

Master data update pipeline. Runs five steps in order:

  1. fetch_nflverse.R           Download pbp, injury reports, schedules, players (R/nflreadr)
  2. fetch_ir_transactions.py   Scrape IR transactions from prosportstransactions.com (Playwright)
  3. nfl_data/pbp.ipynb         Parse in-game injury events from play descriptions
  4. nfl_data/clean.ipynb       Merge + clean all raw CSVs into cleaned CSVs
  5. nfl_data/load.ipynb        Load cleaned CSVs into MySQL

One-time setup (already done — nothing to run):
    The single .venv at the project root has everything installed.

Usage examples:
    # Incremental update - current season only (fastest, use regularly):
    python pipeline/run_pipeline.py

    # Pull everything from a specific year forward:
    python pipeline/run_pipeline.py --from-year 2010

    # Full historical rebuild from scratch:
    python pipeline/run_pipeline.py --full-refresh

    # Skip individual steps:
    python pipeline/run_pipeline.py --skip-nflverse
    python pipeline/run_pipeline.py --skip-ir --skip-pbp

    # Scrape IR from a specific start date only:
    python pipeline/run_pipeline.py --ir-from 2025-01-01

    # Override IR end-date and overlap window:
    python pipeline/run_pipeline.py --ir-to 2026-04-14 --ir-backfill-days 21
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# All paths are derived from this file's location so the pipeline works
# regardless of the working directory from which it is invoked.
PROJECT_ROOT = Path(__file__).parent.parent   # repo root (one level above pipeline/)
PIPELINE_DIR = Path(__file__).parent          # pipeline/ folder
NFL_DATA_DIR = PROJECT_ROOT / "nfl_data"       # where CSVs and notebooks live
VENV_PYTHON  = PROJECT_ROOT / ".venv" / "bin" / "python"  # venv Python executable
RSCRIPT      = "Rscript"                       # must be on PATH (install R + nflreadr first)

# Individual script / notebook paths referenced by the step functions below
FETCH_R      = PIPELINE_DIR / "fetch_nflverse.R"
FETCH_IR_PY  = PIPELINE_DIR / "fetch_ir_transactions.py"
PBP_NB       = NFL_DATA_DIR / "pbp.ipynb"
CLEAN_NB     = NFL_DATA_DIR / "clean.ipynb"
LOAD_NB      = NFL_DATA_DIR / "load.ipynb"


# ── Helpers ───────────────────────────────────────────────────────────────────
def banner(msg: str):
    """Print a prominent 60-char separator bar around a section heading."""
    bar = "─" * 60
    print(f"\n{bar}\n  {msg}\n{bar}")


def run(cmd: list, cwd: Path = PROJECT_ROOT) -> bool:
    """
    Run a shell command as a subprocess and stream its output to stdout.

    Args:
        cmd: Command and arguments as a list (e.g. ['Rscript', 'script.R']).
        cwd: Working directory for the process.  Defaults to the project root.

    Returns:
        True if the command exited with return code 0, False otherwise.
    """
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        print(f"\n[ERROR] Command exited with code {result.returncode}")
        return False
    return True


def check_nbconvert() -> bool:
    """
    Verify that nbconvert and nbformat are importable in the current venv.

    These packages are required to execute Jupyter notebooks from the command
    line (via `jupyter nbconvert --execute`).  If they're missing the notebook
    steps will be skipped and a helpful install command is printed.

    Returns:
        True if both packages are available, False otherwise.
    """
    try:
        import nbconvert  # noqa: F401
        import nbformat   # noqa: F401
        return True
    except ImportError:
        print(
            "\n[WARNING] nbconvert/nbformat not installed."
            f"\n  Fix: {VENV_PYTHON} -m pip install nbconvert nbformat\n"
        )
        return False


def run_notebook(notebook_path: Path) -> bool:
    """
    Execute a Jupyter notebook in-place, saving cell outputs back into the file.

    Uses `jupyter nbconvert --execute --inplace` with a generous 2-hour timeout
    to accommodate large pbp datasets.  The notebook is run from its own parent
    directory so that relative CSV paths inside the notebook resolve correctly.
    """
    if not notebook_path.exists():
        print(f"[ERROR] Notebook not found: {notebook_path}")
        return False
    return run(
        [
            str(VENV_PYTHON), "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=7200",        # 2-hour ceiling per notebook
            "--ExecutePreprocessor.kernel_name=python3",
            str(notebook_path),
        ],
        cwd=notebook_path.parent,   # run from nfl_data/ so relative CSV paths resolve correctly
    )


# ── Individual pipeline steps ─────────────────────────────────────────────────
def step_nflverse(incremental: bool, from_year: int | None) -> bool:
    """
    Step 1: Call the R script that downloads nflverse data (play-by-play,
    injury reports, schedules, player bios) using the nflreadr package.

    In incremental mode no extra flags are passed and R fetches only the current
    season.  With --from-year or --full-refresh, the appropriate flag is passed
    to the R script so it pulls historical data.
    """
    banner("Step 1 — Fetch nflverse data (R)")
    cmd = [RSCRIPT, str(FETCH_R)]
    if incremental:
        cmd.append("--incremental")   # current season only
    elif from_year:
        cmd += ["--from", str(from_year)]  # pull from this year forward
    return run(cmd)


def step_ir_transactions(
    full_refresh: bool,
    ir_from: str | None,
    ir_to: str | None,
    ir_backfill_days: int,
) -> bool:
    """
    Step 2: Run fetch_ir_transactions.py to scrape NFL injured-list moves from
    prosportstransactions.com using Playwright.

    Full refresh: the script scrapes the complete history back to 2009.
    Incremental: only re-scrapes the overlap window (backfill-days before watermark)
                 so late edits from the source site are caught.
    """
    banner("Step 2 — Scrape IR transactions (prosportstransactions.com)")
    cmd = [str(VENV_PYTHON), str(FETCH_IR_PY)]
    if full_refresh:
        cmd.append("--full-refresh")
    else:
        cmd += ["--backfill-days", str(ir_backfill_days)]  # overlap window size
        if ir_from:
            cmd += ["--from", ir_from]  # override start date
    if ir_to:
        cmd += ["--to", ir_to]  # override end date
    return run(cmd)


def step_pbp() -> bool:
    """Step 3: Execute pbp.ipynb to parse in-game injury events from play descriptions."""
    banner("Step 3 — Run pbp.ipynb  (parse in-game injury events)")
    return run_notebook(PBP_NB)


def step_clean() -> bool:
    """Step 4: Execute clean.ipynb to merge and clean all raw CSVs."""
    banner("Step 4 — Run clean.ipynb  (merge + clean raw CSVs)")
    return run_notebook(CLEAN_NB)


def step_load() -> bool:
    """Step 5: Execute load.ipynb to load cleaned data into the MySQL database."""
    banner("Step 5 — Run load.ipynb  (load cleaned data into MySQL)")
    return run_notebook(LOAD_NB)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    """
    Entry point for the master data pipeline.

    Parses CLI flags, resolves the run mode (incremental / from-year / full-refresh),
    checks that nbconvert is installed, then executes each step in order.  Steps
    are gated on the success of their predecessors: clean.ipynb only runs if
    pbp.ipynb passed, and load.ipynb only runs if clean.ipynb passed.
    """
    parser = argparse.ArgumentParser(description="NFL injury data update pipeline")
    parser.add_argument(
        "--full-refresh", action="store_true",
        help="Full historical rebuild: pull all seasons, drop/recreate DB tables.",
    )
    parser.add_argument(
        "--from-year", type=int, default=None, metavar="YEAR",
        help="Pull nflverse data from this season year forward (e.g. 2010).",
    )
    # Skip flags let you re-run only one or two steps after a partial failure
    parser.add_argument("--skip-nflverse", action="store_true", help="Skip nflverse R fetch")
    parser.add_argument("--skip-ir",       action="store_true", help="Skip IR transaction scrape")
    parser.add_argument("--skip-pbp",      action="store_true", help="Skip pbp.ipynb")
    parser.add_argument("--skip-clean",    action="store_true", help="Skip clean.ipynb")
    parser.add_argument("--skip-load",     action="store_true", help="Skip load.ipynb")
    parser.add_argument(
        "--ir-from", default=None, metavar="DATE",
        help="Override IR scrape start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--ir-to", default=None, metavar="DATE",
        help="Override IR scrape end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--ir-backfill-days", type=int, default=14, metavar="N",
        help="In incremental mode, re-scrape N days before the IR watermark.",
    )
    args = parser.parse_args()

    # Incremental = no full-refresh AND no explicit from-year override
    incremental = not args.full_refresh and not args.from_year
    start_time  = datetime.now()
    results: dict = {}  # accumulate step pass/fail for the final summary table

    print(f"\nPipeline started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if args.full_refresh:
        print("Mode: FULL REFRESH (all historical seasons)")
    elif args.from_year:
        print(f"Mode: FROM {args.from_year} to present")
    else:
        print("Mode: INCREMENTAL (current season only)")

    # Check for nbconvert once up front.  If it's missing, all notebook steps
    # will be skipped with a clear error rather than failing mid-pipeline.
    nb_ok = check_nbconvert()

    # ── Step 1: nflverse R fetch ───────────────────────────────────────────────
    if not args.skip_nflverse:
        results["1_nflverse"] = step_nflverse(incremental, args.from_year)
    else:
        print("\n[Skipped] nflverse fetch")
        results["1_nflverse"] = True  # mark True so downstream steps are not blocked

    # ── Step 2: IR transaction scrape ─────────────────────────────────────────
    if not args.skip_ir:
        results["2_ir"] = step_ir_transactions(
            args.full_refresh,
            args.ir_from,
            args.ir_to,
            args.ir_backfill_days,
        )
    else:
        print("\n[Skipped] IR transaction scrape")
        results["2_ir"] = True

    # ── Step 3: pbp.ipynb ─────────────────────────────────────────────────────
    if not args.skip_pbp:
        if nb_ok:
            results["3_pbp"] = step_pbp()
        else:
            results["3_pbp"] = False  # cannot run without nbconvert
    else:
        print("\n[Skipped] pbp.ipynb")
        results["3_pbp"] = True

    # ── Step 4: clean.ipynb ─────────────────────────────────────────────────
    # Only run if pbp.ipynb succeeded (or was explicitly skipped by the user).
    # Skipping clean when pbp failed avoids loading half-processed data.
    if not args.skip_clean:
        if nb_ok and results.get("3_pbp", True):
            results["4_clean"] = step_clean()
        else:
            results["4_clean"] = False
            print("  [Skipped] clean.ipynb — cannot run until pbp.ipynb passes")
    else:
        print("\n[Skipped] clean.ipynb")
        results["4_clean"] = True

    # ── Step 5: load.ipynb ────────────────────────────────────────────────────
    # Only run if clean.ipynb succeeded (or was explicitly skipped).
    if not args.skip_load:
        if nb_ok and results.get("4_clean", True):
            results["5_load"] = step_load()
        else:
            results["5_load"] = False
            print("  [Skipped] load.ipynb — cannot run until clean.ipynb passes")
    else:
        print("\n[Skipped] load.ipynb")
        results["5_load"] = True

    # ── Summary ───────────────────────────────────────────────────────────────
    # Print a concise pass/fail table and exit with code 1 if any step failed.
    elapsed = datetime.now() - start_time
    banner("Pipeline summary")
    labels = {
        "1_nflverse": "nflverse fetch",
        "2_ir":        "IR transactions",
        "3_pbp":       "pbp.ipynb",
        "4_clean":     "clean.ipynb",
        "5_load":      "load.ipynb",
    }
    for key, ok in results.items():
        label  = labels.get(key, key)
        status = "✓ OK" if ok else "✗ FAILED"
        print(f"  {label:<24} {status}")
    print(f"\n  Total time: {elapsed}")

    if not all(results.values()):
        print("\n[PIPELINE FAILED] Review errors above.")
        sys.exit(1)  # non-zero exit so CI/CD tools detect the failure
    print("\n[PIPELINE COMPLETE]")


if __name__ == "__main__":
    main()
