"""
pipeline/fetch_ir_transactions.py

Scrapes NFL injury transactions from prosportstransactions.com using
Playwright and appends/upserts into nfl_data/nfl_injuries_full.csv.

The incremental mode intentionally uses a rolling overlap window
(`--backfill-days`) so late edits from the source are re-ingested.

Usage:
    python fetch_ir_transactions.py
    python fetch_ir_transactions.py --from 2024-01-01
    python fetch_ir_transactions.py --from 2024-01-01 --to 2024-12-31
    python fetch_ir_transactions.py --backfill-days 21
    python fetch_ir_transactions.py --full-refresh
"""

import argparse
import asyncio
import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path

from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent / "nfl_data"  # project's nfl_data/ folder
OUTPUT_CSV    = BASE_DIR / "nfl_injuries_full.csv"          # main injury transaction CSV
STATE_DIR     = Path(__file__).parent / "state"            # pipeline/ state/ subfolder
STATE_FILE    = STATE_DIR / "pipeline_state.json"           # JSON file tracking last run dates
BACKUP_DIR    = BASE_DIR / "backups"                       # where pre-refresh backups are stored
BASE_URL      = "https://www.prosportstransactions.com/football/Search/SearchResults.php"
PAGE_SIZE     = 25                    # results per page on prosportstransactions.com
FULL_REFRESH_START = "2009-06-01"     # earliest date for a full historical pull (2009 pre-season)

# Columns the site returns — must match the order parsed from the HTML table
CSV_COLUMNS   = ["Date", "Team", "Acquired", "Relinquished", "Notes"]

# ── State helpers ─────────────────────────────────────────────────────────────
def load_state() -> dict:
    """
    Load the persistent pipeline state from disk.

    The state file records when the IR scraper last ran and what date range it
    covered.  On first run (no state file yet) returns an empty dict, which
    causes the scraper to fall back to the CSV watermark or a 30-day lookback.
    """
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    """Persist updated pipeline state (watermark dates etc.) to disk."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)  # create state/ dir if it doesn't exist
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def parse_iso_date(value: str) -> date:
    """Parse a YYYY-MM-DD string into a Python date object."""
    return datetime.strptime(value, "%Y-%m-%d").date()


def latest_csv_date(path: Path) -> str | None:
    """
    Scan the output CSV and return the maximum Date string found.

    Used as a watermark so that incremental runs only fetch data that could
    be newer than what we already have on disk.  Returns None if the CSV
    does not exist or has no date values.
    """
    if not path.exists():
        return None
    max_date: str | None = None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = (row.get("Date") or "").strip()
            if not d:
                continue
            # String comparison works here because dates are YYYY-MM-DD (ISO 8601)
            if max_date is None or d > max_date:
                max_date = d
    return max_date


def max_scraped_date(rows: list[dict]) -> str | None:
    """Return the maximum Date string from a list of freshly scraped rows."""
    max_date: str | None = None
    for row in rows:
        d = (row.get("Date") or "").strip()
        if not d:
            continue
        if max_date is None or d > max_date:
            max_date = d
    return max_date


# ── URL builder ───────────────────────────────────────────────────────────────
def build_url(begin: str, end: str, start: int = 0) -> str:
    """
    Construct a paginated search URL for prosportstransactions.com.

    Args:
        begin: Start date in YYYY-MM-DD format.
        end:   End date in YYYY-MM-DD format.
        start: Row offset for pagination (0 means page 1; 25 means page 2, etc.).
    """
    return (
        f"{BASE_URL}"
        f"?Player=&Team=&BeginDate={begin}&EndDate={end}"
        f"&ILChkBx=yes&submit=Search&start={start}"  # ILChkBx=yes limits to IR/injured list moves
    )


# ── HTML parser ───────────────────────────────────────────────────────────────
def parse_table(html: str) -> list[dict]:
    """Extract all data rows from the results table. Returns list of dicts."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"class": "datatable"})
    if table is None:
        # Try to find any table with Date / Team headers
        for t in soup.find_all("table"):
            headers = [th.get_text(strip=True) for th in t.find_all("th")]
            if "Date" in headers and "Team" in headers:
                table = t
                break
    if table is None:
        return []

    rows = []
    # Skip the first <tr> which is the header row (th elements, not td)
    for tr in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) >= 5:  # guard against empty/malformed rows
            rows.append({
                "Date":          cells[0],
                "Team":          cells[1],
                "Acquired":      cells[2],
                "Relinquished":  cells[3],
                "Notes":         cells[4],
            })
    return rows


def has_next_page(html: str) -> bool:
    """
    Return True if the rendered HTML contains a 'Next' pagination link.

    prosportstransactions.com shows a 'Next 25' hyperlink when there are more
    results beyond the current page.  Detecting this tells the scraper whether
    to increment the start offset and fetch another page.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Check all anchor tags for any that contain the word 'next'
    links = soup.find_all("a")
    for link in links:
        text = link.get_text(strip=True).lower()
        if "next" in text:
            return True
    return False


# ── Main scraper ──────────────────────────────────────────────────────────────
async def scrape(begin_date: str, end_date: str) -> list[dict]:
    """Use Playwright to scrape all paginated results for a date range."""
    from playwright.async_api import async_playwright  # noqa: PLC0415

    all_rows: list[dict] = []
    start = 0
    seen_signatures: set[tuple[str, str, str, str]] = set()

    async with async_playwright() as p:
        print("Launching browser...", flush=True)
        browser = await p.chromium.launch(
            headless=True,   # Server-safe mode (no GUI required)
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()

        # Remove webdriver property so we look like a real browser
        await page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        print(f"Scraping: {begin_date} → {end_date}", flush=True)

        while True:
            url = build_url(begin_date, end_date, start)
            print(f"  Fetching page start={start} ...", end=" ", flush=True)

            try:
                # Navigate to the page and wait until the DOM is ready (not full network idle)
                await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
            except Exception as e:
                print(f"\n  Page load error: {e}")
                break

            # If Cloudflare challenge appears, wait for it to auto-solve.
            # A real browser with these settings usually passes automatically within ~30s.
            for _ in range(12):  # poll every 5s, up to 60s total
                title = await page.title()
                if "just a moment" in title.lower() or "attention required" in title.lower():
                    print("  [Cloudflare challenge] waiting 5s...", end=" ", flush=True)
                    await asyncio.sleep(5)
                else:
                    break  # challenge solved (or no challenge appeared)

            html  = await page.content()  # get the fully-rendered HTML
            rows  = parse_table(html)      # extract table rows as dicts
            print(f"{len(rows)} rows")

            if not rows:
                # Empty page means no more results (or an error)
                break

            # Build a signature from first + last row to detect page repetition
            signature = (
                rows[0].get("Date", ""),
                rows[0].get("Team", ""),
                rows[-1].get("Date", ""),
                rows[-1].get("Team", ""),
            )
            if signature in seen_signatures:
                print("  Repeated page signature detected; stopping pagination loop.")
                break
            seen_signatures.add(signature)

            all_rows.extend(rows)

            # Stop if there's no Next link or the page had a partial result set
            if not has_next_page(html) or len(rows) < PAGE_SIZE:
                break

            start += PAGE_SIZE
            # Polite delay — don't hammer the server between pages
            await asyncio.sleep(1.5)

        await browser.close()

    print(f"Total rows fetched: {len(all_rows):,}")
    return all_rows


# ── CSV writer ────────────────────────────────────────────────────────────────
def save_rows(rows: list[dict], full_refresh: bool):
    """
    Write/merge rows into nfl_injuries_full.csv with deterministic deduplication.

    Full-refresh mode:
        Backs up the existing CSV with a timestamp, then writes a fresh file
        containing only the newly scraped rows.

    Incremental mode (default):
        Reads the existing CSV, identifies which rows are genuinely new
        (by comparing a 5-tuple key of Date + Team + Acquired + Relinquished +
        Notes), and appends only the new ones.  Duplicate rows are silently
        skipped.
    """
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    if full_refresh or not OUTPUT_CSV.exists():
        # FULL REFRESH: overwrite the CSV entirely (after backing up the old one)
        if OUTPUT_CSV.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = BACKUP_DIR / f"nfl_injuries_full.pre_refresh_backup_{ts}.csv"
            OUTPUT_CSV.replace(backup_path)  # atomic rename — no read required
            print(f"Backed up previous CSV to {backup_path}")
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows):,} rows to {OUTPUT_CSV}")
    else:
        # INCREMENTAL: append only rows not already in the file.
        # Build a set of existing 5-tuples for O(1) membership testing.
        existing = []
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing = list(reader)

        # Key = (Date, Team, Acquired, Relinquished, Notes) — unique enough to identify a transaction
        existing_keys = {
            (r["Date"], r["Team"], r.get("Acquired", ""), r.get("Relinquished", ""), r.get("Notes", ""))
            for r in existing
        }
        # Keep only rows whose key is not already in the CSV
        new_rows = [
            r for r in rows
            if (r["Date"], r["Team"], r.get("Acquired", ""), r.get("Relinquished", ""), r.get("Notes", ""))
            not in existing_keys
        ]
        print(f"  {len(new_rows):,} new rows (skipped {len(rows) - len(new_rows):,} duplicates)")

        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerows(new_rows)  # append — no header needed
        print(f"Appended {len(new_rows):,} rows to {OUTPUT_CSV}")


# ── CLI entry point ───────────────────────────────────────────────────────────
def main():
    """Parse command-line arguments and orchestrate the scrape + save pipeline."""
    parser = argparse.ArgumentParser(description="Scrape NFL IR transactions")
    parser.add_argument("--from", dest="from_date", default=None,
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--full-refresh", action="store_true",
                        help="Scrape all years (2009 season to present).")
    parser.add_argument("--backfill-days", type=int, default=14,
                        help="In incremental mode, re-scrape this many days before watermark.")
    args = parser.parse_args()

    today = date.today().isoformat()  # default end date: today

    if args.full_refresh:
        # Start from the very beginning of our historical data range
        begin = FULL_REFRESH_START
        end   = today
    else:
        # INCREMENTAL: figure out how far back to reach using the watermark.
        # Prefer the state file's watermark; fall back to the CSV's newest date.
        state = load_state()
        state_watermark = state.get("ir_last_scraped_date")
        csv_watermark = latest_csv_date(OUTPUT_CSV)
        watermark = state_watermark or csv_watermark
        if watermark:
            # Step back --backfill-days before the watermark to pick up late edits
            start_date = parse_iso_date(watermark) - timedelta(days=max(args.backfill_days, 0))
            default_begin = start_date.isoformat()
        else:
            # No prior data at all — default to a 30-day lookback
            default_begin = (date.today() - timedelta(days=30)).isoformat()
        begin = args.from_date or default_begin  # CLI override takes priority
        end   = args.to_date or today

    print(f"Date range: {begin} → {end}")

    # Run the Playwright async scraper via asyncio.run() (synchronous entry point)
    rows = asyncio.run(scrape(begin, end))

    if rows:
        save_rows(rows, full_refresh=args.full_refresh)

        # Update state with the MAX date actually scraped (not simply "today")
        # so the next incremental run starts at the right watermark.
        state = load_state()
        state["ir_last_scraped_date"] = max_scraped_date(rows) or end
        state["ir_last_run_date"] = today
        state["ir_last_begin"] = begin
        state["ir_last_end"] = end
        state["ir_backfill_days"] = args.backfill_days
        save_state(state)
        print(
            "State updated: "
            f"ir_last_scraped_date={state['ir_last_scraped_date']}, "
            f"ir_last_run_date={state['ir_last_run_date']}"
        )
    else:
        print("No rows fetched — nothing saved.")


if __name__ == "__main__":
    main()
