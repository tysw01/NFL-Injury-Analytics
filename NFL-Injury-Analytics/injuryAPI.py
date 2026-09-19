# injuryAPI.py
# =============
# Main Flask application for the NFL Injury Analysis capstone project.
#
# This file serves:
#   - HTML pages (landing, dashboard, statistics, and prediction model)
#   - REST API endpoints (data quality, statistics, explorer, prediction)
#   - The trained ML model for player availability prediction
#
# All database calls go through the MySQL instance defined in nfl_data/config.py.
# The trained model is cached on disk (.model_cache.joblib) and refreshed
# automatically after _MODEL_CACHE_MAX_AGE_HOURS or when manually triggered.
#
# Start the server:  python injuryAPI.py  (runs on port 5001 by default)

import csv
import json
import os
import re
import threading
import time
from io import StringIO
from functools import lru_cache
import joblib

import pymysql
from flask import Flask, Response, jsonify, redirect, render_template, request, url_for

from nfl_data import config  # DB credentials (host, port, user, passwd, db)

# ── Flask app & path constants ────────────────────────────────────────────────
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # absolute path to the project root

# Path where the trained model pickle is cached between server restarts
_MODEL_CACHE_PATH = os.path.join(BASE_DIR, ".model_cache.joblib")
# How many hours before the cached model is considered stale and retrained
_MODEL_CACHE_MAX_AGE_HOURS = 24

# ── Table & field description configuration ──────────────────────────────────
# These are the primary DB tables exposed through the data-statistics page.
MAIN_DATA_TABLES = ["events", "injuryReport", "logs", "pbp", "players"]
# These bridge/auxiliary tables are documented separately.
BRIDGE_TABLES = ["injury_master", "injury_player_bridge", "event_player_bridge"]

# Manually-curated field descriptions per table.
# Add entries here if a column doesn't have a description in the CSV dictionaries.
# Fill these dictionaries as field definitions are provided.
TABLE_FIELD_DESCRIPTIONS = {
    "events": {},
    "injuryReport": {},
    "logs": {},
    "pbp": {},
    "players": {},
}

# Maps each table name to a CSV data-dictionary file (field name + description pairs).
# The CSV files are loaded lazily and cached by load_dictionary_csv().
TABLE_DICTIONARY_FILES = {
    
    "injuryReport": "nfl_data/dictionary_injuries.csv",
    "pbp": "nfl_data/dictionary_pbp.csv",
    "players": "nfl_data/dictionary_players.csv",
}

# JSON file that holds user-entered custom descriptions for any column in any table.
# These override the CSV dictionary values when both are present.
CUSTOM_DESCRIPTIONS_FILE = os.path.join(BASE_DIR, "field_descriptions_overrides.json")


@lru_cache(maxsize=16)
def load_dictionary_csv(file_name: str):
    """
    Load a CSV data-dictionary file and return a {field_name: description} dict.

    Results are cached (up to 16 files) via lru_cache to avoid re-reading the
    same CSV on every request.  The function tolerates several column name
    variants so it works with both nflverse-exported dictionaries and custom ones.
    """
    csv_path = os.path.join(BASE_DIR, file_name)
    if not os.path.exists(csv_path):
        return {}  # return empty dict rather than raising FileNotFoundError

    descriptions = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Accept several column name variants for the field name column
            field_name = (
                row.get("Field")
                or row.get("field")
                or row.get("column_name")
                or row.get("COLUMN_NAME")
                or ""
            ).strip()
            # Accept several column name variants for the description column
            description = (
                row.get("Description")
                or row.get("description")
                or row.get("DESC")
                or ""
            ).strip()
            if field_name and description:
                descriptions[field_name] = description
    return descriptions


def normalize_field_name(field_name: str) -> str:
    """
    Strip all non-alphanumeric characters from a field name and lowercase it.

    Used as a fallback comparison when an exact or case-insensitive match
    fails — e.g. 'Injury Type' and 'injury_type' both normalize to 'injurytype'.
    """
    return re.sub(r"[^a-z0-9]", "", (field_name or "").lower())


@lru_cache(maxsize=1)
def load_custom_field_descriptions():
    """
    Load the user-editable field description overrides from JSON.

    Returns a nested dict: {table_name: {column_name: description}}.
    Results are cached with lru_cache; call load_custom_field_descriptions.cache_clear()
    after saving changes so the next request picks up the new values.
    """
    if not os.path.exists(CUSTOM_DESCRIPTIONS_FILE):
        return {}

    with open(CUSTOM_DESCRIPTIONS_FILE, "r", encoding="utf-8") as fh:
        raw_data = json.load(fh)

    if not isinstance(raw_data, dict):
        return {}

    clean_data = {}
    for table_name, mapping in raw_data.items():
        if not isinstance(mapping, dict):
            continue
        clean_data[table_name] = {
            str(column_name): str(description)
            for column_name, description in mapping.items()
            if str(description).strip()
        }
    return clean_data


def save_custom_field_descriptions(descriptions: dict):
    """
    Atomically write updated custom field descriptions back to disk.

    Uses a write-to-temp-then-rename pattern so the file is never partially
    written if the process crashes mid-write.  Clears the lru_cache so the
    next call to load_custom_field_descriptions() picks up the new values.
    """
    tmp_path = f"{CUSTOM_DESCRIPTIONS_FILE}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(descriptions, fh, indent=2, sort_keys=True)
    os.replace(tmp_path, CUSTOM_DESCRIPTIONS_FILE)  # atomic rename on POSIX systems
    load_custom_field_descriptions.cache_clear()    # invalidate the in-memory cache


def resolve_description_from_map(description_map: dict, column_name: str) -> str:
    """
    Look up a column description using a three-tier fallback strategy:

    1. Exact match on the original column name (fastest path).
    2. Case-insensitive match (handles camelCase vs snake_case differences).
    3. Normalized match after stripping non-alphanumeric chars (broadest fallback).

    Returns an empty string if no match is found at any tier.
    """
    if not column_name:
        return ""

    # Tier 1: exact key lookup
    direct = description_map.get(column_name)
    if direct:
        return direct

    # Tier 2: case-insensitive comparison
    lower_target = column_name.lower()
    for key, value in description_map.items():
        if key.lower() == lower_target:
            return value

    # Tier 3: normalized comparison (removes underscores, dashes, spaces, etc.)
    normalized_target = normalize_field_name(column_name)
    for key, value in description_map.items():
        if normalize_field_name(key) == normalized_target:
            return value

    return ""  # no match found


def get_description_for_column(field_descriptions: dict, column_name: str) -> str:
    """Convenience wrapper: look up a single column description from a pre-built map."""
    return resolve_description_from_map(field_descriptions, column_name)


def get_table_field_descriptions(table_name: str):
    """
    Merge the hardcoded TABLE_FIELD_DESCRIPTIONS with any CSV dictionary file
    for the given table.  The hardcoded dict takes precedence (it is applied
    last in the merge) so manual overrides win over auto-generated descriptions.
    """
    merged = dict(TABLE_FIELD_DESCRIPTIONS.get(table_name, {}))  # start with manual descriptions
    dictionary_file = TABLE_DICTIONARY_FILES.get(table_name)
    if dictionary_file:
        # CSV dictionary values are the base; manual dict overwrites on key collision
        merged = {**load_dictionary_csv(dictionary_file), **merged}
    return merged


def get_combined_descriptions_for_table(table_name: str):
    """
    Build the final field-description map for a table by layering three sources:

    1. CSV data dictionary (lowest priority)
    2. Hardcoded TABLE_FIELD_DESCRIPTIONS (middle priority)
    3. User-entered custom JSON overrides (highest priority)

    The user can override any column description through the data-statistics UI
    without touching code, and those overrides are persisted in
    field_descriptions_overrides.json.
    """
    base_descriptions = get_table_field_descriptions(table_name)   # layers 1+2
    custom_descriptions = load_custom_field_descriptions().get(table_name, {})  # layer 3
    return {
        **base_descriptions,
        **custom_descriptions,  # custom values overwrite base values on key collision
    }


def get_conn():
    """
    Open a new MySQL connection with automatic retry.

    Tries up to 3 times with increasing back-off (0.4s, 0.8s) because the
    remote MySQL server occasionally drops idle connections.  Raises the last
    exception if all attempts fail.

    Returns a pymysql connection with DictCursor so all rows come back as
    {column_name: value} dicts rather than plain tuples.
    """
    last_exc = None
    for attempt in range(3):
        try:
            return pymysql.connect(
                host=config.host,
                port=int(config.port),
                user=config.user,
                passwd=config.passwd,
                db=config.db,
                cursorclass=pymysql.cursors.DictCursor,  # rows returned as dicts
                autocommit=True,          # no explicit COMMIT needed for SELECT queries
                connect_timeout=15,       # fail fast if server is unreachable
                read_timeout=120,         # allow up to 2 minutes for heavy aggregations
                write_timeout=120,
            )
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(0.4 * (attempt + 1))  # back-off: 0.4s, 0.8s
    raise last_exc  # re-raise the last exception after all retries are exhausted


def one(sql: str, params=()):
    """
    Execute a SQL query and return the first row as a dict (or None).

    Opens a fresh connection for each call so there are no long-held connections
    that could time out on the remote server.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)  # params are safely escaped by pymysql
            return cur.fetchone()     # returns a dict or None if the result set is empty
    finally:
        conn.close()  # always return the connection even if an exception occurred


def many(sql: str, params=()):
    """
    Execute a SQL query and return all result rows as a list of dicts.

    Returns an empty list (not None) if the query matches no rows.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()  # returns a list (possibly empty)
    finally:
        conn.close()


def build_quality_payload():
    """
    Build the data-quality summary that powers the /api/quality endpoint.

    Runs several aggregation queries against the events, logs, and injuryReport
    tables to compute:
      - Total, linked, and unlinked event counts
      - Linkage rates (overall and for Achilles specifically as a benchmark)
      - Confidence level distribution
      - Log-to-event linkage rate

    All returned values are raw ints/floats/dicts ready to be JSON-serialized.
    """
    # Count total events and break them down by link status and data completeness
    totals = one(
        """
        SELECT
            COUNT(*) AS total_events,
            SUM(CASE WHEN loggedInGame = 1 THEN 1 ELSE 0 END) AS linked_events,
            SUM(CASE WHEN loggedInGame = 0 THEN 1 ELSE 0 END) AS unlinked_events,
            SUM(CASE WHEN gameId IS NULL OR gameId = '' THEN 1 ELSE 0 END) AS missing_game_id,
            SUM(CASE WHEN playId IS NULL THEN 1 ELSE 0 END) AS missing_play_id,
            SUM(CASE WHEN confidence_level IS NULL OR confidence_level = '' THEN 1 ELSE 0 END) AS missing_confidence,
            SUM(CASE WHEN injury_detail = 'Achilles' THEN 1 ELSE 0 END) AS achilles_total,
            SUM(CASE WHEN injury_detail = 'Achilles' AND loggedInGame = 1 THEN 1 ELSE 0 END) AS achilles_linked
        FROM events
        """
    ) or {}

    confidence = one(
        """
        SELECT
            SUM(CASE WHEN confidence_level = 'high' THEN 1 ELSE 0 END) AS high_confidence,
            SUM(CASE WHEN confidence_level = 'medium' THEN 1 ELSE 0 END) AS medium_confidence,
            SUM(CASE WHEN confidence_level = 'low' THEN 1 ELSE 0 END) AS low_confidence
        FROM events
        """
    ) or {}

    total_events = totals.get("total_events") or 0
    linked_events = totals.get("linked_events") or 0
    achilles_total = totals.get("achilles_total") or 0
    achilles_linked = totals.get("achilles_linked") or 0

    linkage_rate = round((linked_events / total_events) * 100, 2) if total_events else 0.0
    achilles_linkage_rate = round((achilles_linked / achilles_total) * 100, 2) if achilles_total else 0.0

    high_conf = confidence.get("high_confidence") or 0
    medium_conf = confidence.get("medium_confidence") or 0
    low_conf = confidence.get("low_confidence") or 0
    conf_total = high_conf + medium_conf + low_conf

    confidence_pct = {
        "high": round((high_conf / conf_total) * 100, 2) if conf_total else 0.0,
        "medium": round((medium_conf / conf_total) * 100, 2) if conf_total else 0.0,
        "low": round((low_conf / conf_total) * 100, 2) if conf_total else 0.0,
    }

    log_metrics = one(
        """
        SELECT
            COUNT(*) AS total_injury_logs,
            SUM(
                CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM events e
                        WHERE e.loggedInGame = 1
                          AND e.gameId = l.gameId
                          AND e.playId = l.playId
                    ) THEN 1
                    ELSE 0
                END
            ) AS linked_injury_logs
        FROM logs l
        """
    ) or {}

    total_logs = log_metrics.get("total_injury_logs") or 0
    linked_logs = log_metrics.get("linked_injury_logs") or 0
    logs_linkage_rate = round((linked_logs / total_logs) * 100, 2) if total_logs else 0.0

    return {
        "totals": totals,
        "confidence_breakdown": confidence,
        "rates": {
            "linkage_rate_percent": linkage_rate,
            "achilles_linkage_rate_percent": achilles_linkage_rate,
            "logs_linkage_rate_percent": logs_linkage_rate,
        },
        "confidence_pct": confidence_pct,
        "log_metrics": log_metrics,
    }


def normalize_count_mode(count_mode: str | None) -> str:
    return "occurrences" if (count_mode or "").lower() == "occurrences" else "unique"


def normalize_source_mode(source_mode: str | None) -> str:
    value = (source_mode or "all").lower()
    if value in {"weekly_report", "ir_transactions"}:
        return value
    return "all"


def build_statistics_payload(
    scope: str = "all",
    filter_kind: str | None = None,
    filter_value: str | None = None,
    count_mode: str = "unique",
    source_mode: str = "all",
):
    scope = (scope or "all").lower()
    if scope not in {"all", "logged", "unlogged"}:
        scope = "all"
    count_mode = normalize_count_mode(count_mode)
    source_mode = normalize_source_mode(source_mode)

    where_parts = []
    params = []

    if scope == "logged":
        where_parts.append("e.loggedInGame = 1")
    elif scope == "unlogged":
        where_parts.append("e.loggedInGame = 0")

    if filter_kind in {"injury_type", "injury_detail"} and (filter_value or "").strip():
        where_parts.append(f"e.{filter_kind} = %s")
        params.append(filter_value.strip())

    if source_mode == "weekly_report":
        where_parts.append(
            "EXISTS (SELECT 1 FROM injuryReport ir_src WHERE ir_src.injuryId = e.injuryId "
            "AND (ir_src.action IS NULL OR TRIM(ir_src.action) = ''))"
        )
    elif source_mode == "ir_transactions":
        where_parts.append(
            "EXISTS (SELECT 1 FROM injuryReport ir_src WHERE ir_src.injuryId = e.injuryId "
            "AND ir_src.action IS NOT NULL AND TRIM(ir_src.action) <> '')"
        )

    where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    injury_count_expr = "COUNT(DISTINCT e.injuryId)" if count_mode == "unique" else "COUNT(*)"
    linked_count_expr = (
        "COUNT(DISTINCT CASE WHEN e.loggedInGame = 1 THEN e.injuryId END)"
        if count_mode == "unique"
        else "SUM(CASE WHEN e.loggedInGame = 1 THEN 1 ELSE 0 END)"
    )
    total_base_expr = "COUNT(DISTINCT e2.injuryId)" if count_mode == "unique" else "COUNT(*)"

    totals = one(
        f"""
        SELECT
            COUNT(*) AS total_event_rows,
            COUNT(DISTINCT e.injuryId) AS total_injuries,
            SUM(CASE WHEN e.loggedInGame = 1 THEN 1 ELSE 0 END) AS logged_event_rows,
            SUM(CASE WHEN e.loggedInGame = 0 THEN 1 ELSE 0 END) AS unlogged_event_rows,
            COUNT(DISTINCT CASE WHEN e.loggedInGame = 1 THEN e.injuryId END) AS logged_injuries,
            COUNT(DISTINCT CASE WHEN e.loggedInGame = 0 THEN e.injuryId END) AS unlogged_injuries
        FROM events e
        {where_sql}
        """
        ,
        tuple(params),
    ) or {}

    missingness = one(
        f"""
        SELECT
            ROUND(100 * AVG(CASE WHEN e.injury_type IS NULL OR TRIM(e.injury_type) = '' THEN 1 ELSE 0 END), 2) AS injury_type_missing_percent,
            ROUND(100 * AVG(CASE WHEN e.injury_detail IS NULL OR TRIM(e.injury_detail) = '' THEN 1 ELSE 0 END), 2) AS injury_detail_missing_percent
        FROM events e
        {where_sql}
        """,
        tuple(params),
    ) or {}

    season_map_missing = one(
        f"""
        SELECT
            ROUND(100 * AVG(CASE WHEN season_map.season IS NULL THEN 1 ELSE 0 END), 2) AS season_mapped_missing_percent
        FROM events e
        LEFT JOIN (
            SELECT injuryId, MIN(CASE WHEN season IS NOT NULL THEN season END) AS season
            FROM injuryReport
            GROUP BY injuryId
        ) AS season_map ON season_map.injuryId = e.injuryId
        {where_sql}
        """,
        tuple(params),
    ) or {}
    missingness.update(season_map_missing)

    confidence_rows = many(
        f"""
        SELECT
            COALESCE(NULLIF(TRIM(e.confidence_level), ''), 'unknown') AS confidence_level,
            COUNT(*) AS event_count
        FROM events e
        {where_sql}
        GROUP BY COALESCE(NULLIF(TRIM(e.confidence_level), ''), 'unknown')
        ORDER BY event_count DESC
        """
        ,
        tuple(params),
    )

    injury_type_rows = many(
        f"""
        SELECT
            COALESCE(NULLIF(TRIM(e.injury_type), ''), 'Unknown') AS injury_type,
            {injury_count_expr} AS injury_count,
            {linked_count_expr} AS linked_count,
            ROUND(
                100 * {linked_count_expr}
                / NULLIF({injury_count_expr}, 0),
                2
            ) AS linked_rate_percent
        FROM events e
        {where_sql}
        GROUP BY COALESCE(NULLIF(TRIM(e.injury_type), ''), 'Unknown')
        ORDER BY injury_count DESC
        LIMIT 10
        """
        ,
        tuple(params),
    )

    where_sql_e2 = where_sql.replace("e.", "e2.")

    detail_rows = many(
        f"""
        SELECT
            COALESCE(NULLIF(TRIM(e.injury_detail), ''), 'Unknown') AS injury_detail,
            {injury_count_expr} AS injury_count,
            ROUND(
                100 * {injury_count_expr}
                / NULLIF((SELECT {total_base_expr} FROM events e2 {where_sql_e2}), 0),
                2
            ) AS share_percent
        FROM events e
        {where_sql}
        GROUP BY COALESCE(NULLIF(TRIM(e.injury_detail), ''), 'Unknown')
        ORDER BY injury_count DESC
        LIMIT 20
        """,
        tuple(params) + tuple(params),
    )

    season_rows = many(
        f"""
        SELECT
            season_map.season AS season,
            {injury_count_expr} AS injury_count,
            {linked_count_expr} AS linked_count,
            ROUND(
                100 * {linked_count_expr}
                / NULLIF({injury_count_expr}, 0),
                2
            ) AS linked_rate_percent
        FROM events e
        JOIN (
            SELECT injuryId, MIN(CASE WHEN season IS NOT NULL THEN season END) AS season
            FROM injuryReport
            GROUP BY injuryId
        ) AS season_map ON season_map.injuryId = e.injuryId
        {where_sql}
        {'AND' if where_sql else 'WHERE'} season_map.season IS NOT NULL
        GROUP BY season_map.season
        ORDER BY season_map.season
        """
        ,
        tuple(params),
    )

    total_metric = int(
        totals.get("total_injuries") if count_mode == "unique" else totals.get("total_event_rows") or 0
    )
    linked_metric = int(
        totals.get("logged_injuries") if count_mode == "unique" else totals.get("logged_event_rows") or 0
    )
    linkage_rate = round((linked_metric / total_metric) * 100, 2) if total_metric else 0.0

    return {
        "scope": scope,
        "count_mode": count_mode,
        "source_mode": source_mode,
        "filters": {
            "filter_kind": filter_kind if filter_kind in {"injury_type", "injury_detail"} else None,
            "filter_value": filter_value if (filter_value or "").strip() else None,
        },
        "totals": totals,
        "missingness": missingness,
        "linkage_rate_percent": linkage_rate,
        "confidence": confidence_rows,
        "injury_type_breakdown": injury_type_rows,
        "injury_detail_breakdown": detail_rows,
        "season_trend": season_rows,
    }


def build_explorer_payload(
    module: str,
    scope: str = "all",
    granularity: str = "none",
    filter_kind: str | None = None,
    filter_value: str | None = None,
    season_start: int | None = None,
    season_end: int | None = None,
    count_mode: str = "unique",
    source_mode: str = "all",
):
    module = (module or "injury_trends").lower()
    scope = (scope or "all").lower()
    granularity = (granularity or "none").lower()

    if scope not in {"all", "logged", "unlogged"}:
        scope = "all"
    if granularity not in {"none", "season", "week"}:
        granularity = "none"
    count_mode = normalize_count_mode(count_mode)
    source_mode = normalize_source_mode(source_mode)

    if module == "injury_trends" and granularity == "none":
        granularity = "season"

    requested_scope = scope
    context_modules = {
        "field_type",
        "quarter",
        "weather",
        "temperature",
        "location",
        "stadium",
        "roof",
        "opponent",
        "humidity",
        "wind_direction",
    }
    player_modules = {"age", "experience"}

    module_definitions = {
        "injury_trends": "Injury trends over season/week",
        "field_type": "Injuries by field surface",
        "quarter": "Injuries by quarter",
        "transactions": "Transactions by action",
        "weather": "Injuries by weather condition",
        "temperature": "Injuries by temperature bucket",
        "location": "Injuries by game location",
        "stadium": "Injuries by stadium",
        "roof": "Injuries by roof type",
        "position": "Injuries by position",
        "age": "Injuries by player age bucket",
        "experience": "Injuries by years of experience",
        "opponent": "Injuries vs opponent (derived)",
        "humidity": "Injuries by humidity level",
        "wind_direction": "Injuries by wind speed",
    }

    if module not in module_definitions:
        module = "injury_trends"

    # Context-based dimensions only exist for game-linked injuries.
    forced_logged_scope = False
    if module in context_modules and scope != "logged":
        scope = "logged"
        forced_logged_scope = True

    forced_player_link_scope = module in player_modules

    def scope_predicate(alias: str):
        if scope == "logged":
            return f"{alias}.loggedInGame = 1"
        if scope == "unlogged":
            return f"{alias}.loggedInGame = 0"
        return None

    def injury_filter(alias: str):
        if filter_kind in {"injury_type", "injury_detail"} and (filter_value or "").strip():
            return f"{alias}.{filter_kind} = %s", [filter_value.strip()]
        return None, []

    def season_filter(col: str) -> tuple:
        parts: list = []
        sp: list = []
        if season_start:
            parts.append(f"{col} >= %s")
            sp.append(season_start)
        if season_end:
            parts.append(f"{col} <= %s")
            sp.append(season_end)
        return (" AND ".join(parts), sp) if parts else (None, [])

    def source_filter(alias: str, table_kind: str):
        if source_mode == "all":
            return None, []
        if table_kind == "injury_report":
            if source_mode == "weekly_report":
                return f"({alias}.action IS NULL OR TRIM({alias}.action) = '')", []
            return f"({alias}.action IS NOT NULL AND TRIM({alias}.action) <> '')", []

        if source_mode == "weekly_report":
            return (
                f"EXISTS (SELECT 1 FROM injuryReport ir_src WHERE ir_src.injuryId = {alias}.injuryId "
                "AND (ir_src.action IS NULL OR TRIM(ir_src.action) = ''))",
                [],
            )
        return (
            f"EXISTS (SELECT 1 FROM injuryReport ir_src WHERE ir_src.injuryId = {alias}.injuryId "
            "AND ir_src.action IS NOT NULL AND TRIM(ir_src.action) <> '')",
            [],
        )

    rows = []
    params = []

    def count_selects(injury_id_expr: str, linked_expr: str, occurrence_expr: str):
        if count_mode == "unique":
            return (
                f"COUNT(DISTINCT {injury_id_expr})",
                f"COUNT(DISTINCT {linked_expr})",
            )
        return (
            f"COUNT(DISTINCT {occurrence_expr})",
            f"COUNT(DISTINCT CASE WHEN {linked_expr} IS NOT NULL THEN {occurrence_expr} END)",
        )

    if module == "injury_trends":
        where_parts = []
        scope_part = scope_predicate("e")
        if scope_part:
            where_parts.append(scope_part)

        injury_part, injury_params = injury_filter("e")
        if injury_part:
            where_parts.append(injury_part)
            params.extend(injury_params)

        src_part, src_params = source_filter("ir" if count_mode == "occurrences" else "e", "injury_report" if count_mode == "occurrences" else "events")
        if src_part:
            where_parts.append(src_part)
            params.extend(src_params)

        season_part, season_params = season_filter("ir.season")
        if season_part:
            where_parts.append(season_part)
            params.extend(season_params)

        if granularity == "week":
            period_expr = (
                "CASE WHEN ir.week = 22 AND ir.season < 2021 THEN 23 ELSE ir.week END"
            )
            where_parts.append("ir.week IS NOT NULL")
        else:
            period_expr = "ir.season"
            where_parts.append("ir.season IS NOT NULL")

        where_sql = " AND ".join(where_parts) if where_parts else "1=1"
        trend_occurrence_expr = "CONCAT_WS('|', COALESCE(e.injuryId, ''), COALESCE(ir.transactionDate, ''), COALESCE(ir.week, ''), COALESCE(ir.action, ''))"
        injury_count_select, linked_count_select = count_selects(
            "e.injuryId",
            "CASE WHEN e.loggedInGame = 1 THEN e.injuryId END",
            trend_occurrence_expr,
        )
        rows = many(
            f"""
            SELECT
                {period_expr} AS period,
                {injury_count_select} AS injury_count,
                {linked_count_select} AS linked_count,
                ROUND(
                    100 * {linked_count_select}
                    / NULLIF({injury_count_select}, 0),
                    2
                ) AS linked_rate_percent
            FROM events e
            LEFT JOIN injuryReport ir ON ir.injuryId = e.injuryId
            WHERE {where_sql}
            GROUP BY {period_expr}
            ORDER BY {period_expr}
            """,
            tuple(params),
        )
    else:
        where_parts = []
        join_sql = ""
        time_expr = None
        game_count_expr = "NULL"

        if module in {"field_type", "quarter", "weather", "temperature", "location", "stadium", "roof", "humidity", "wind_direction"}:
            join_sql = "JOIN pbp p ON p.gameId = e.gameId AND p.playId = e.playId"
            scope_part = scope_predicate("e")
            if scope_part:
                where_parts.append(scope_part)
            injury_part, injury_params = injury_filter("e")
            if injury_part:
                where_parts.append(injury_part)
                params.extend(injury_params)

            src_part, src_params = source_filter("e", "events")
            if src_part:
                where_parts.append(src_part)
                params.extend(src_params)

            season_part, season_params = season_filter("p.season")
            if season_part:
                where_parts.append(season_part)
                params.extend(season_params)

            if module == "field_type":
                dimension_expr = "COALESCE(NULLIF(TRIM(p.surface), ''), 'Unknown')"
                where_parts.append("p.surface IS NOT NULL AND TRIM(p.surface) <> ''")
            elif module == "quarter":
                dimension_expr = "COALESCE(CAST(p.qtr AS CHAR), 'Unknown')"
                where_parts.append("p.qtr IS NOT NULL")
            elif module == "weather":
                dimension_expr = """
                CASE
                    WHEN REGEXP_LIKE(p.weather, 'indoor|controlled climate|^N/A$|N/A.*indoor|^indoors$|^indoor$|^N/A indoor|^N/A \\\\(', 'i') THEN 'Indoor/Dome'
                    WHEN REGEXP_LIKE(p.weather, 'snow|flurr', 'i') THEN 'Snow'
                    WHEN REGEXP_LIKE(p.weather, 'rain|shower|drizzle|precipit|rainy', 'i') THEN 'Rain'
                    WHEN REGEXP_LIKE(p.weather, 'fog|haze|mist|smoky', 'i') THEN 'Fog/Haze'
                    WHEN REGEXP_LIKE(p.weather, 'cloud|overcast|grey|gray|partly', 'i') THEN 'Cloudy'
                    WHEN REGEXP_LIKE(p.weather, 'sun|clear|fair|bright|crisp|nice', 'i') THEN 'Clear/Sunny'
                    ELSE 'Other'
                END
                """
                where_parts.append("p.weather IS NOT NULL AND TRIM(p.weather) <> ''")
            elif module == "location":
                # Derive true Home / Away / Neutral from the injured player's team
                # vs pbp.home_team / pbp.away_team.
                # pbp.location only ever says 'Home' or 'Neutral' (no 'Away' value),
                # so we must check which side the player's team was on.
                join_sql += (
                    " LEFT JOIN (SELECT injuryId, MIN(team) AS team FROM injuryReport GROUP BY injuryId) ir_loc "
                    "ON ir_loc.injuryId = e.injuryId"
                )
                dimension_expr = """
                CASE
                    WHEN p.location = 'Neutral' THEN 'Neutral'
                    WHEN ir_loc.team = p.home_team THEN 'Home'
                    WHEN ir_loc.team = p.away_team THEN 'Away'
                    ELSE 'Unknown'
                END
                """
                where_parts.append("p.location IS NOT NULL AND TRIM(p.location) <> ''")
                where_parts.append("ir_loc.team IS NOT NULL")
            elif module == "stadium":
                dimension_expr = "COALESCE(NULLIF(TRIM(p.stadium), ''), 'Unknown')"
                where_parts.append("p.stadium IS NOT NULL AND TRIM(p.stadium) <> ''")
            elif module == "roof":
                dimension_expr = "COALESCE(NULLIF(TRIM(p.roof), ''), 'Unknown')"
                where_parts.append("p.roof IS NOT NULL AND TRIM(p.roof) <> ''")
            elif module == "humidity":
                dimension_expr = """
                CASE
                    WHEN REGEXP_LIKE(p.weather, 'indoor|controlled climate|^N/A|^indoors|^indoor', 'i') THEN 'Indoor (N/A)'
                    WHEN LOCATE('Humidity:', p.weather) = 0 THEN 'Not Reported'
                    WHEN CAST(SUBSTRING(p.weather, LOCATE('Humidity: ', p.weather) + 10, 4) AS UNSIGNED) < 30 THEN 'Dry (under 30)'
                    WHEN CAST(SUBSTRING(p.weather, LOCATE('Humidity: ', p.weather) + 10, 4) AS UNSIGNED) < 50 THEN 'Low (30 to 49)'
                    WHEN CAST(SUBSTRING(p.weather, LOCATE('Humidity: ', p.weather) + 10, 4) AS UNSIGNED) < 70 THEN 'Moderate (50 to 69)'
                    WHEN CAST(SUBSTRING(p.weather, LOCATE('Humidity: ', p.weather) + 10, 4) AS UNSIGNED) < 85 THEN 'High (70 to 84)'
                    ELSE 'Very High (85 plus)'
                END
                """
                where_parts.append("p.weather IS NOT NULL AND TRIM(p.weather) <> ''")
            elif module == "wind_direction":
                dimension_expr = """
                CASE
                    WHEN p.wind IS NULL THEN 'Unknown'
                    WHEN p.wind = 0 THEN 'Calm (0 mph)'
                    WHEN p.wind <= 5 THEN 'Light (1-5 mph)'
                    WHEN p.wind <= 10 THEN 'Gentle (6-10 mph)'
                    WHEN p.wind <= 15 THEN 'Moderate (11-15 mph)'
                    WHEN p.wind <= 20 THEN 'Fresh (16-20 mph)'
                    ELSE 'Strong (21+ mph)'
                END
                """
                where_parts.append("p.wind IS NOT NULL")
            else:
                dimension_expr = """
                CASE
                    WHEN p.temp IS NULL THEN 'Unknown'
                    WHEN p.temp < 40 THEN 'Cold (<40F)'
                    WHEN p.temp < 55 THEN 'Cool (40-54F)'
                    WHEN p.temp < 70 THEN 'Mild (55-69F)'
                    WHEN p.temp < 85 THEN 'Warm (70-84F)'
                    ELSE 'Hot (85F+)'
                END
                """
                where_parts.append("p.temp IS NOT NULL")

            if granularity == "season":
                time_expr = "COALESCE(CAST(p.season AS CHAR), 'Unknown')"
            elif granularity == "week":
                time_expr = "COALESCE(CAST(p.week AS CHAR), 'Unknown')"

            source_from = f"events e {join_sql}"
            injury_id_expr = "e.injuryId"
            linked_expr = "CASE WHEN e.loggedInGame = 1 THEN e.injuryId END"
            game_count_expr = "COUNT(DISTINCT p.gameId)"
            occurrence_expr = "CONCAT_WS('|', COALESCE(e.injuryId, ''), COALESCE(e.gameId, ''), COALESCE(CAST(e.playId AS CHAR), ''))"

        elif module in {"transactions", "position"}:
            scope_part = scope_predicate("ir")
            if scope_part:
                where_parts.append(scope_part)
            injury_part, injury_params = injury_filter("ir")
            if injury_part:
                where_parts.append(injury_part)
                params.extend(injury_params)

            src_part, src_params = source_filter("ir", "injury_report")
            if src_part:
                where_parts.append(src_part)
                params.extend(src_params)

            season_part, season_params = season_filter("ir.season")
            if season_part:
                where_parts.append(season_part)
                params.extend(season_params)

            if module == "transactions":
                dimension_expr = "COALESCE(NULLIF(TRIM(ir.action), ''), 'Unknown')"
            else:
                dimension_expr = "COALESCE(NULLIF(TRIM(ir.position), ''), 'Unknown')"

            if granularity == "season":
                time_expr = "COALESCE(CAST(ir.season AS CHAR), 'Unknown')"
            elif granularity == "week":
                time_expr = "COALESCE(CAST(ir.week AS CHAR), 'Unknown')"

            source_from = "injuryReport ir"
            injury_id_expr = "ir.injuryId"
            linked_expr = "CASE WHEN ir.loggedInGame = 1 THEN ir.injuryId END"
            occurrence_expr = "CONCAT_WS('|', COALESCE(ir.injuryId, ''), COALESCE(ir.transactionDate, ''), COALESCE(ir.week, ''), COALESCE(ir.action, ''))"

        elif module in {"age", "experience"}:
            scope_part = scope_predicate("ir")
            if scope_part:
                where_parts.append(scope_part)
            injury_part, injury_params = injury_filter("ir")
            if injury_part:
                where_parts.append(injury_part)
                params.extend(injury_params)

            src_part, src_params = source_filter("ir", "injury_report")
            if src_part:
                where_parts.append(src_part)
                params.extend(src_params)

            season_part, season_params = season_filter("ir.season")
            if season_part:
                where_parts.append(season_part)
                params.extend(season_params)

            if module == "age":
                dimension_expr = """
                CASE
                    WHEN pl.birth_date IS NULL THEN 'Unknown'
                    WHEN TIMESTAMPDIFF(YEAR, pl.birth_date, ir.transactionDate) < 23 THEN '<23'
                    WHEN TIMESTAMPDIFF(YEAR, pl.birth_date, ir.transactionDate) BETWEEN 23 AND 26 THEN '23-26'
                    WHEN TIMESTAMPDIFF(YEAR, pl.birth_date, ir.transactionDate) BETWEEN 27 AND 30 THEN '27-30'
                    WHEN TIMESTAMPDIFF(YEAR, pl.birth_date, ir.transactionDate) BETWEEN 31 AND 34 THEN '31-34'
                    ELSE '35+'
                END
                """
                where_parts.append("pl.birth_date IS NOT NULL")
            else:
                dimension_expr = """
                CASE
                    WHEN pl.years_of_experience IS NULL OR TRIM(pl.years_of_experience) = '' THEN 'Unknown'
                    WHEN CAST(pl.years_of_experience AS SIGNED) <= 1 THEN '0-1'
                    WHEN CAST(pl.years_of_experience AS SIGNED) <= 3 THEN '2-3'
                    WHEN CAST(pl.years_of_experience AS SIGNED) <= 6 THEN '4-6'
                    WHEN CAST(pl.years_of_experience AS SIGNED) <= 9 THEN '7-9'
                    ELSE '10+'
                END
                """
                where_parts.append("pl.years_of_experience IS NOT NULL AND TRIM(pl.years_of_experience) <> ''")

            if granularity == "season":
                time_expr = "COALESCE(CAST(ir.season AS CHAR), 'Unknown')"
            elif granularity == "week":
                time_expr = "COALESCE(CAST(ir.week AS CHAR), 'Unknown')"

            source_from = "injuryReport ir JOIN players pl ON pl.gsis_id = ir.gsis_id"
            injury_id_expr = "ir.injuryId"
            linked_expr = "CASE WHEN ir.loggedInGame = 1 THEN ir.injuryId END"
            occurrence_expr = "CONCAT_WS('|', COALESCE(ir.injuryId, ''), COALESCE(ir.transactionDate, ''), COALESCE(ir.week, ''), COALESCE(ir.action, ''))"

        else:
            scope_part = scope_predicate("e")
            if scope_part:
                where_parts.append(scope_part)
            injury_part, injury_params = injury_filter("e")
            if injury_part:
                where_parts.append(injury_part)
                params.extend(injury_params)

            src_part, src_params = source_filter("e", "events")
            if src_part:
                where_parts.append(src_part)
                params.extend(src_params)

            season_part, season_params = season_filter("game_context.season")
            if season_part:
                where_parts.append(season_part)
                params.extend(season_params)

            where_parts.append("game_context.gameId IS NOT NULL")
            where_parts.append("ir.team IS NOT NULL AND TRIM(ir.team) <> ''")
            where_parts.append("(ir.team = game_context.home_team OR ir.team = game_context.away_team)")

            dimension_expr = """
            CASE
                WHEN ir.team = game_context.home_team THEN game_context.away_team
                WHEN ir.team = game_context.away_team THEN game_context.home_team
                ELSE 'Unknown'
            END
            """

            game_context_sql = """
            LEFT JOIN (
                SELECT
                    gameId,
                    MIN(home_team) AS home_team,
                    MIN(away_team) AS away_team,
                    MIN(season) AS season,
                    MIN(week) AS week
                FROM pbp
                GROUP BY gameId
            ) AS game_context ON game_context.gameId = e.gameId
            """

            if granularity == "season":
                time_expr = "COALESCE(CAST(game_context.season AS CHAR), 'Unknown')"
            elif granularity == "week":
                time_expr = "COALESCE(CAST(game_context.week AS CHAR), 'Unknown')"

            source_from = (
                "events e "
                "LEFT JOIN (SELECT injuryId, MIN(team) AS team FROM injuryReport GROUP BY injuryId) ir "
                "ON ir.injuryId = e.injuryId "
                f"{game_context_sql}"
            )
            injury_id_expr = "e.injuryId"
            linked_expr = "CASE WHEN e.loggedInGame = 1 THEN e.injuryId END"
            game_count_expr = "COUNT(DISTINCT game_context.gameId)"
            occurrence_expr = "CONCAT_WS('|', COALESCE(e.injuryId, ''), COALESCE(e.gameId, ''), COALESCE(CAST(e.playId AS CHAR), ''))"

        period_expr = time_expr if time_expr else "'All'"
        where_sql = " AND ".join(where_parts) if where_parts else "1=1"
        injury_count_select, linked_count_select = count_selects(
            injury_id_expr,
            linked_expr,
            occurrence_expr,
        )
        rows = many(
            f"""
            SELECT
                {period_expr} AS period,
                {dimension_expr} AS dimension,
                {injury_count_select} AS injury_count,
                {linked_count_select} AS linked_count,
                ROUND(
                    100 * {linked_count_select}
                    / NULLIF({injury_count_select}, 0),
                    2
                ) AS linked_rate_percent,
                {game_count_expr} AS game_count
            FROM {source_from}
            WHERE {where_sql}
            GROUP BY {period_expr}, {dimension_expr}
            ORDER BY injury_count DESC
            LIMIT 300
            """,
            tuple(params),
        )

    return {
        "module": module,
        "module_label": module_definitions.get(module, module_definitions["injury_trends"]),
        "count_mode": count_mode,
        "source_mode": source_mode,
        "requested_scope": requested_scope,
        "scope": scope,
        "forced_logged_scope": forced_logged_scope,
        "forced_player_link_scope": forced_player_link_scope,
        "granularity": granularity,
        "filters": {
            "filter_kind": filter_kind if filter_kind in {"injury_type", "injury_detail"} else None,
            "filter_value": filter_value if (filter_value or "").strip() else None,
        },
        "rows": rows,
    }


def build_table_catalog_payload(selected_table: str | None = None):
    tables_raw = many("SHOW TABLES")
    if not tables_raw:
        return {
            "tables": [],
            "data_tables": [],
            "bridge_tables": [],
            "other_tables": [],
            "selected_table": None,
            "selected_columns": [],
            "selected_table_is_main_data": False,
        }

    first_key = next(iter(tables_raw[0].keys()))
    table_names = [r[first_key] for r in tables_raw]
    table_lookup = {name.lower(): name for name in table_names}
    main_data_table_set = {name.lower() for name in MAIN_DATA_TABLES}
    bridge_table_set = {name.lower() for name in BRIDGE_TABLES}
    ordered_table_rank = {
        name.lower(): idx for idx, name in enumerate(MAIN_DATA_TABLES + BRIDGE_TABLES)
    }

    table_stats = []
    for table_name in table_names:
        row_info = one(f"SELECT COUNT(*) AS row_count FROM `{table_name}`") or {}
        col_info = one(
            """
            SELECT COUNT(*) AS column_count
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            """,
            (config.db, table_name),
        ) or {}
        column_count = col_info.get("column_count", 0)

        table_columns = many(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION
            """,
            (config.db, table_name),
        ) or []

        combined_descriptions = get_combined_descriptions_for_table(table_name)
        documented_column_count = sum(
            1
            for row in table_columns
            if (get_description_for_column(combined_descriptions, row.get("COLUMN_NAME")) or "").strip()
        )
        documentation_rate_percent = (
            round((documented_column_count / column_count) * 100, 1) if column_count else 0.0
        )

        table_name_lower = table_name.lower()
        if table_name_lower in main_data_table_set:
            table_group = "data"
        elif table_name_lower in bridge_table_set or "bridge" in table_name_lower:
            table_group = "bridge"
        else:
            table_group = "other"

        table_stats.append(
            {
                "table_name": table_name,
                "row_count": row_info.get("row_count", 0),
                "column_count": column_count,
                "documented_column_count": documented_column_count,
                "documentation_rate_percent": documentation_rate_percent,
                "table_group": table_group,
            }
        )

    selected = None
    if selected_table:
        selected = table_lookup.get(selected_table.lower())

    if not selected:
        selected = next(
            (name for name in table_names if name.lower() in main_data_table_set),
            table_names[0],
        )

    selected_columns = many(
        """
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        ORDER BY ORDINAL_POSITION
        """,
        (config.db, selected),
    ) or []

    dictionary_descriptions = load_dictionary_csv(TABLE_DICTIONARY_FILES[selected]) if selected in TABLE_DICTIONARY_FILES else {}
    manual_descriptions = TABLE_FIELD_DESCRIPTIONS.get(selected, {})
    custom_descriptions = load_custom_field_descriptions().get(selected, {})

    field_descriptions = {
        **dictionary_descriptions,
        **manual_descriptions,
        **custom_descriptions,
    }

    source_counts = {
        "dictionary": 0,
        "manual": 0,
        "custom": 0,
    }

    documented_count = 0
    nullable_count = 0
    key_count = 0

    numeric_types = {
        "tinyint",
        "smallint",
        "mediumint",
        "int",
        "bigint",
        "decimal",
        "numeric",
        "float",
        "double",
        "real",
        "bit",
    }
    numeric_count = 0

    for col in selected_columns:
        column_name = col["COLUMN_NAME"]
        description = get_description_for_column(field_descriptions, column_name)
        col["DESCRIPTION"] = description
        col["HAS_DESCRIPTION"] = bool((description or "").strip())

        if col["HAS_DESCRIPTION"]:
            documented_count += 1

        if col.get("IS_NULLABLE") == "YES":
            nullable_count += 1

        if (col.get("COLUMN_KEY") or "").strip():
            key_count += 1

        if (col.get("DATA_TYPE") or "").lower() in numeric_types:
            numeric_count += 1

        if resolve_description_from_map(custom_descriptions, column_name):
            col["DESCRIPTION_SOURCE"] = "custom"
            source_counts["custom"] += 1
        elif resolve_description_from_map(manual_descriptions, column_name):
            col["DESCRIPTION_SOURCE"] = "manual"
            source_counts["manual"] += 1
        elif resolve_description_from_map(dictionary_descriptions, column_name):
            col["DESCRIPTION_SOURCE"] = "dictionary"
            source_counts["dictionary"] += 1
        else:
            col["DESCRIPTION_SOURCE"] = "missing"

    total_columns = len(selected_columns)
    undocumented_count = max(total_columns - documented_count, 0)
    completion_rate = round((documented_count / total_columns) * 100, 1) if total_columns else 0.0

    selected_profile = {
        "total_columns": total_columns,
        "documented_columns": documented_count,
        "undocumented_columns": undocumented_count,
        "documentation_rate_percent": completion_rate,
        "nullable_columns": nullable_count,
        "key_columns": key_count,
        "numeric_columns": numeric_count,
        "categorical_columns": max(total_columns - numeric_count, 0),
        "source_counts": source_counts,
    }

    def sort_tables_for_display(rows):
        return sorted(
            rows,
            key=lambda row: (
                ordered_table_rank.get(row["table_name"].lower(), 999),
                row["table_name"].lower(),
            ),
        )

    data_tables = sort_tables_for_display(
        [row for row in table_stats if row["table_group"] == "data"]
    )
    bridge_tables = sort_tables_for_display(
        [row for row in table_stats if row["table_group"] == "bridge"]
    )
    other_tables = sort_tables_for_display(
        [row for row in table_stats if row["table_group"] == "other"]
    )

    return {
        "tables": table_stats,
        "data_tables": data_tables,
        "bridge_tables": bridge_tables,
        "other_tables": other_tables,
        "selected_table": selected,
        "selected_columns": selected_columns,
        "selected_table_is_main_data": selected.lower() in main_data_table_set,
        "selected_profile": selected_profile,
    }


def _normalize_status(status: str | None) -> str:
    """Lowercase and strip leading/trailing whitespace from a status string."""
    return (status or "").strip().lower()


def _to_float(value, default=0.0):
    """Safely convert a value to float, returning the default on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value, default=0):
    """
    Safely convert a value to int, returning the default on any conversion error.

    Handles None, strings, and numpy NaN (which raises OverflowError when cast
    to int, unlike regular Python floats which raise ValueError).
    """
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError, OverflowError):
        # numpy NaN raises OverflowError (not ValueError), so catch all three
        return int(default)


def _safe_pct(value: float) -> float:
    """
    Clamp a probability float to [0, 1] and convert to a percentage.

    Prevents out-of-range values from model calibration from propagating
    to the API response (e.g. -0.02 → 0.0%, 1.03 → 100.0%).
    """
    return round(max(0.0, min(1.0, value)) * 100, 2)


# ── Body-region lookup table ───────────────────────────────────────────────────────────
# Maps lowercase injury-detail keywords to a coarse anatomical region label.
# The order matters — the first matching keyword wins, so more specific terms
# (e.g. 'patellar') should appear before generic ones (e.g. 'knee') if needed.
_BODY_REGION_MAP: dict[str, str] = {
    "knee": "Knee", "acl": "Knee", "pcl": "Knee", "mcl": "Knee", "lcl": "Knee",
    "meniscus": "Knee", "patella": "Knee", "patellar": "Knee",
    "ankle": "Ankle",
    "achilles": "Lower Leg", "calf": "Lower Leg", "shin": "Lower Leg",
    "foot": "Foot", "toe": "Foot", "turf toe": "Foot",
    "hamstring": "Hamstring", "quad": "Quad", "hip": "Hip", "groin": "Groin",
    "shoulder": "Shoulder", "rotator": "Shoulder", "labrum": "Shoulder", "ac joint": "Shoulder",
    "elbow": "Elbow", "wrist": "Wrist",
    "hand": "Hand", "finger": "Hand", "thumb": "Hand",
    "back": "Back", "spine": "Back", "disc": "Back", "lumbar": "Back",
    "neck": "Neck", "cervical": "Neck",
    "rib": "Torso", "abdomen": "Torso", "oblique": "Torso", "core": "Torso",
    "concussion": "Head", "head": "Head",
    "chest": "Chest", "pectoral": "Chest",
}


def _injury_body_region(detail_str: str) -> str:
    """Map an injury_detail string to a coarse anatomical region."""
    if not detail_str or detail_str.lower() in ("unknown", ""):
        return "Unknown"
    dl = detail_str.lower()
    for keyword, region in _BODY_REGION_MAP.items():
        if keyword in dl:
            return region
    return "Other"


def _build_model_base_rows(player_id: str | None = None):
    sql = """
        WITH pbp_games AS (
            -- playId=1 uses PRIMARY(gameId,playId) index: one row per game in ~0.01s.
            -- roof/stadium are constant per game, so play 1 is always correct.
            SELECT
                p.gameId,
                p.season,
                p.season_type,
                p.week,
                p.home_team,
                p.away_team,
                p.gameDate AS game_date,
                p.stadium,
                CASE WHEN LOWER(TRIM(COALESCE(p.roof, ''))) IN ('dome', 'closed') THEN 1 ELSE 0 END AS is_dome
            FROM pbp p
            WHERE p.playId = 1
              AND p.season IS NOT NULL
              AND p.season_type IS NOT NULL
              AND p.week IS NOT NULL
              AND p.home_team IS NOT NULL
              AND p.away_team IS NOT NULL
        ),
        team_games_raw AS (
            SELECT season, season_type, week,
                   home_team AS team, away_team AS opponent,
                   game_date, stadium, is_dome, 1 AS is_home
            FROM pbp_games
            UNION ALL
            SELECT season, season_type, week,
                   away_team AS team, home_team AS opponent,
                   game_date, stadium, is_dome, 0 AS is_home
            FROM pbp_games
        ),
        team_games AS (
            -- Pre-compute each team's NEXT game using LEAD() — eliminates the
            -- correlated subquery that caused the 120-second timeout.
            SELECT
                season, season_type, week, team, opponent,
                game_date, stadium, is_dome, is_home,
                LEAD(game_date)   OVER (PARTITION BY team ORDER BY game_date) AS nxt_game_date,
                LEAD(stadium)     OVER (PARTITION BY team ORDER BY game_date) AS nxt_stadium,
                LEAD(opponent)    OVER (PARTITION BY team ORDER BY game_date) AS nxt_opponent,
                LEAD(is_home)     OVER (PARTITION BY team ORDER BY game_date) AS nxt_is_home,
                LEAD(is_dome)     OVER (PARTITION BY team ORDER BY game_date) AS nxt_is_dome,
                LEAD(season)      OVER (PARTITION BY team ORDER BY game_date) AS nxt_season,
                LEAD(week)        OVER (PARTITION BY team ORDER BY game_date) AS nxt_week,
                LEAD(season_type) OVER (PARTITION BY team ORDER BY game_date) AS nxt_season_type
            FROM team_games_raw
        ),
        player_name_base AS (
            -- Normalise all player names so we can join against injury report names
            SELECT
                LOWER(
                    REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
                        COALESCE(NULLIF(TRIM(display_name), ''), NULLIF(TRIM(football_name), ''), TRIM(CONCAT_WS(' ', first_name, last_name))),
                        ' ', ''), '.', ''), CHAR(39), ''), '-', ''), ',', '')
                ) AS normalized_name,
                gsis_id,
                birth_date,
                height,
                weight,
                years_of_experience,
                position_group,
                UPPER(TRIM(COALESCE(latest_team, ''))) AS latest_team,
                status
            FROM players
            WHERE gsis_id IS NOT NULL
              AND TRIM(gsis_id) <> ''
        ),
        player_name_map AS (
            -- Strict match: normalized_name AND latest_team both match the injury report row.
            -- If multiple players share a name, prefer the one on the same team (robust to
            -- common-name collisions like "Josh Williams" or "Chris Jones").
            SELECT
                normalized_name,
                latest_team AS match_team,
                MIN(gsis_id) AS matched_gsis_id,
                MIN(birth_date) AS birth_date,
                MIN(height) AS height,
                MIN(weight) AS weight,
                MIN(years_of_experience) AS years_of_experience,
                MIN(position_group) AS position_group,
                MIN(latest_team) AS latest_team,
                MIN(status) AS status,
                -- How many distinct GSIS IDs share this name+team?  >1 means still ambiguous.
                COUNT(DISTINCT gsis_id) AS name_team_collision_count
            FROM player_name_base
            WHERE normalized_name IS NOT NULL AND normalized_name <> ''
            GROUP BY normalized_name, latest_team
        ),
        player_name_fallback AS (
            -- Fallback (name-only): used when no team-constrained match exists.
            -- Only trust when exactly ONE player has this name.
            SELECT
                normalized_name,
                MIN(gsis_id) AS matched_gsis_id,
                MIN(birth_date) AS birth_date,
                MIN(height) AS height,
                MIN(weight) AS weight,
                MIN(years_of_experience) AS years_of_experience,
                MIN(position_group) AS position_group,
                MIN(latest_team) AS latest_team,
                MIN(status) AS status,
                COUNT(DISTINCT gsis_id) AS name_collision_count
            FROM player_name_base
            WHERE normalized_name IS NOT NULL AND normalized_name <> ''
            GROUP BY normalized_name
            HAVING COUNT(DISTINCT gsis_id) = 1  -- only unambiguous names
        )
        SELECT
            ir.injury_report_row_id,
            ir.injuryId,
            ir.gsis_id,
            COALESCE(
                CASE
                    WHEN ir.gsis_id IS NULL OR TRIM(ir.gsis_id) = '' OR LOWER(TRIM(ir.gsis_id)) = 'na' THEN NULL
                    ELSE TRIM(ir.gsis_id)
                END,
                -- Strict: name + matching team (low collision risk)
                pnm_strict.matched_gsis_id,
                -- Fallback: unambiguous name-only match
                pnm_fallback.matched_gsis_id
            ) AS model_player_id,
            -- Was the name-only fallback the source? Flag for downstream filtering.
            CASE
                WHEN ir.gsis_id IS NOT NULL AND TRIM(ir.gsis_id) <> '' AND LOWER(TRIM(ir.gsis_id)) <> 'na' THEN 'gsis'
                WHEN pnm_strict.matched_gsis_id IS NOT NULL THEN 'name_team'
                WHEN pnm_fallback.matched_gsis_id IS NOT NULL THEN 'name_only'
                ELSE NULL
            END AS id_match_source,
            ir.full_name,
            ir.team,
            ir.season,
            ir.week,
            ir.game_type,
            ir.action,
            ir.injury_type,
            ir.injury_detail,
            ir.severity,
            ir.position,
            ir.report_status,
            ir.practice_status,
            ir.source,
            ir.days_since_prev,
            ir.new_injury_flag,
            ir.on_ir,
            ir.prev_on_ir,
            ir.prev_season,
            ir.loggedInGame,
            ir.transactionDate,
            COALESCE(p_gsis.birth_date, pnm_strict.birth_date, pnm_fallback.birth_date) AS birth_date,
            COALESCE(p_gsis.height, pnm_strict.height, pnm_fallback.height) AS height,
            COALESCE(p_gsis.weight, pnm_strict.weight, pnm_fallback.weight) AS weight,
            COALESCE(p_gsis.years_of_experience, pnm_strict.years_of_experience, pnm_fallback.years_of_experience) AS years_of_experience,
            COALESCE(p_gsis.position_group, pnm_strict.position_group, pnm_fallback.position_group) AS position_group,
            COALESCE(p_gsis.latest_team, pnm_strict.latest_team, pnm_fallback.latest_team) AS latest_team,
            COALESCE(p_gsis.status, pnm_strict.status, pnm_fallback.status) AS status,
            cur_game.game_date AS current_game_date,
            cur_game.nxt_game_date AS next_game_date,
            cur_game.nxt_stadium AS next_game_stadium,
            cur_game.nxt_opponent AS next_game_opponent,
            cur_game.nxt_is_home AS next_game_is_home,
            COALESCE(cur_game.nxt_is_dome, 0) AS next_game_is_dome,
            -- True participation label from snap_counts when available
            CASE WHEN sc.id IS NOT NULL THEN sc.did_play ELSE NULL END AS snap_did_play,
            CASE WHEN sc.id IS NOT NULL THEN sc.offense_pct ELSE NULL END AS snap_offense_pct
        FROM injuryReport ir
        LEFT JOIN players p_gsis
          ON p_gsis.gsis_id = CASE
                WHEN ir.gsis_id IS NULL OR TRIM(ir.gsis_id) = '' OR LOWER(TRIM(ir.gsis_id)) = 'na' THEN NULL
                ELSE TRIM(ir.gsis_id)
            END
        LEFT JOIN player_name_map pnm_strict
          ON pnm_strict.normalized_name = LOWER(
                REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(TRIM(COALESCE(ir.full_name, '')),
                    ' ', ''), '.', ''), CHAR(39), ''), '-', ''), ',', '')
             )
         AND pnm_strict.match_team = UPPER(TRIM(COALESCE(ir.team, '')))
         AND pnm_strict.name_team_collision_count = 1
        LEFT JOIN player_name_fallback pnm_fallback
          ON pnm_fallback.normalized_name = LOWER(
                REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(TRIM(COALESCE(ir.full_name, '')),
                    ' ', ''), '.', ''), CHAR(39), ''), '-', ''), ',', '')
             )
         AND pnm_strict.matched_gsis_id IS NULL
        LEFT JOIN team_games cur_game
          ON cur_game.season = ir.season
         AND cur_game.week = ir.week
         AND cur_game.team = ir.team
         AND cur_game.season_type = (
                CASE
                    WHEN UPPER(COALESCE(ir.game_type, '')) IN ('REG') THEN 'REG'
                    WHEN UPPER(COALESCE(ir.game_type, '')) IN ('POST', 'WC', 'DIV', 'CON', 'SB') THEN 'POST'
                    ELSE 'REG'
                END
            )
        -- snap_counts for the NEXT game: uses LEAD columns directly — no correlated subquery
        LEFT JOIN snap_counts sc
          ON sc.season = cur_game.nxt_season
         AND sc.week = cur_game.nxt_week
         AND sc.game_type = cur_game.nxt_season_type
         AND sc.team = cur_game.team
         AND sc.player_name = TRIM(ir.full_name)
        WHERE (ir.full_name IS NOT NULL AND TRIM(ir.full_name) <> '')
          AND ir.season BETWEEN 2011 AND 2025
    """
    params = []
    if player_id:
        sql += (
            " AND COALESCE("
            "CASE WHEN ir.gsis_id IS NULL OR TRIM(ir.gsis_id) = '' OR LOWER(TRIM(ir.gsis_id)) = 'na' THEN NULL ELSE TRIM(ir.gsis_id) END,"
            " pnm_strict.matched_gsis_id, pnm_fallback.matched_gsis_id) = %s"
        )
        params.append(player_id)

    sql += " ORDER BY model_player_id, ir.full_name, ir.season, ir.week, ir.transactionDate"
    return many(sql, tuple(params))


def _compute_group_outcomes(group_df, np):
    n = len(group_df)
    unavailable = group_df["is_unavailable"].astype(int).to_numpy()
    injury_keys = group_df["injury_key"].astype(str).to_numpy()

    return_distance = np.full(n, np.nan)
    next_available_idx = None
    for idx in range(n - 1, -1, -1):
        if unavailable[idx] == 0:
            next_available_idx = idx
            continue
        if next_available_idx is not None:
            return_distance[idx] = float(next_available_idx - idx)

    body_regions = group_df["injury_body_region"].astype(str).to_numpy() if "injury_body_region" in group_df.columns else injury_keys

    reagg = np.full(n, np.nan)
    for idx in range(n):
        if unavailable[idx] != 1 or np.isnan(return_distance[idx]):
            continue
        ret_idx = idx + int(return_distance[idx])
        # Look up to 3 games after return for a re-injury to the same body region.
        # Widening from 1-game/exact-key to 3-game/same-region gives enough
        # positive examples (~5-15% rate) for the classifier to learn from.
        window_end = min(n, ret_idx + 4)  # ret_idx+1,+2,+3
        risk_flag = 0
        for j in range(ret_idx + 1, window_end):
            if unavailable[j] == 1 and body_regions[j] == body_regions[idx]:
                risk_flag = 1
                break
        reagg[idx] = float(risk_flag)

    return return_distance, reagg


def _engineer_model_dataframe(rows):
    import numpy as np
    import pandas as pd

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ["season", "week", "days_since_prev", "new_injury_flag", "on_ir", "prev_on_ir", "prev_season", "loggedInGame", "height", "weight", "years_of_experience"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["transactionDate"] = pd.to_datetime(df.get("transactionDate"), errors="coerce")
    df["current_game_date"] = pd.to_datetime(df.get("current_game_date"), errors="coerce")
    df["next_game_date"] = pd.to_datetime(df.get("next_game_date"), errors="coerce")

    df["report_status"] = df["report_status"].fillna("").astype(str)
    df["injury_type"] = df["injury_type"].fillna("Unknown")
    df["injury_detail"] = df["injury_detail"].fillna("Unknown")
    df["full_name"] = df.get("full_name", "").fillna("").astype(str)

    normalized_name = (
        df["full_name"]
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )
    df["model_player_key"] = df.get("model_player_id")
    df["model_player_key"] = df["model_player_key"].where(df["model_player_key"].notna() & (df["model_player_key"].astype(str).str.strip() != ""), normalized_name)
    df = df[df["model_player_key"].notna() & (df["model_player_key"].astype(str).str.strip() != "")].copy()

    df["injuryId"] = df.get("injuryId", "").fillna("").astype(str)
    df["injury_instance_no"] = pd.to_numeric(
        df["injuryId"].str.extract(r"_(\d+(?:\.\d+)?)$")[0],
        errors="coerce",
    )
    df["injury_instance_base"] = df["injuryId"].str.replace(r"_(\d+(?:\.\d+)?)$", "", regex=True)
    df["injury_key"] = (
        df["injury_detail"].replace("", "Unknown").fillna("Unknown").astype(str)
        + "::"
        + df["injury_type"].replace("", "Unknown").fillna("Unknown").astype(str)
    )
    df["injury_instance_base"] = df["injury_instance_base"].replace("", np.nan).fillna(df["injury_key"])

    # Derive injury body region from injury_detail (type is broader; detail can be more specific).
    # This gives the model a coarser signal that generalises across severity variants.
    df["injury_body_region"] = df["injury_detail"].apply(_injury_body_region)

    # Normalise action field: NULL / empty → "none".
    # is_ir_transaction = 1 when the transaction places the player on a reserve list
    # (IR, PUP, NFI) or marks them out for the season — the strongest possible signal
    # that a player will not play.  e.g. ~80% of "ACL" rows have action="Placed on IR".
    _IR_PLACEMENT_ACTIONS = {"placed on ir", "placed on pup", "placed on nfi", "out for season"}
    if "action" not in df.columns:
        df["action"] = "none"
    df["action"] = (
        df["action"]
        .fillna("none")
        .astype(str)
        .str.strip()
        .replace("", "none")
    )
    df["is_ir_transaction"] = df["action"].str.lower().isin(_IR_PLACEMENT_ACTIONS).astype(int)

    df["report_status_norm"] = df["report_status"].map(_normalize_status)
    df["is_unavailable"] = (
        df["report_status_norm"].isin({"out", "doubtful"})
        | (df["on_ir"].fillna(0) >= 1)
    ).astype(int)

    df["season_week_index"] = (df["season"].fillna(0) * 30) + df["week"].fillna(0)
    df = df.sort_values(["model_player_key", "season_week_index", "transactionDate", "injury_report_row_id"], na_position="last")

    df["next_game_stadium"] = df.get("next_game_stadium", "").fillna("Unknown").replace("", "Unknown")
    df["next_game_opponent"] = df.get("next_game_opponent", "").fillna("Unknown").replace("", "Unknown")
    df["next_game_is_home"] = pd.to_numeric(df.get("next_game_is_home"), errors="coerce").fillna(0)
    df["next_game_is_dome"] = pd.to_numeric(df.get("next_game_is_dome"), errors="coerce").fillna(0)

    reference_date = df["transactionDate"].fillna(df["current_game_date"])
    df["days_until_next_game"] = (df["next_game_date"] - reference_date).dt.days
    fallback_days = df["days_until_next_game"].dropna().median()
    if pd.isna(fallback_days):
        fallback_days = 7.0
    df["days_until_next_game"] = pd.to_numeric(df["days_until_next_game"], errors="coerce").fillna(float(fallback_days))
    df["days_until_next_game"] = df["days_until_next_game"].clip(lower=0, upper=21)

    grouped = df.groupby("model_player_key", group_keys=False)
    df["next_unavailable"] = grouped["is_unavailable"].shift(-1)
    # Use snap_count ground truth when available; fall back to report-transition signal.
    snap_did_play = pd.to_numeric(df.get("snap_did_play"), errors="coerce")
    df["next_game_available"] = np.where(
        snap_did_play.notna(),
        snap_did_play,  # TRUE ground truth: player actually appeared in the game
        np.where(df["next_unavailable"].isna(), np.nan, (df["next_unavailable"] == 0).astype(float)),
    )
    df["has_snap_label"] = snap_did_play.notna().astype(int)
    # snap_offense_pct measures how heavily a player was used (0–1)
    df["snap_offense_pct"] = pd.to_numeric(df.get("snap_offense_pct"), errors="coerce").fillna(0.0)

    df["prior_reports_count"] = grouped.cumcount()
    df["prior_unavailable_count"] = grouped["is_unavailable"].cumsum() - df["is_unavailable"]
    df["gap_since_prev_report"] = grouped["season_week_index"].diff().fillna(1)
    df["prev3_unavail_rate"] = (
        grouped["is_unavailable"]
        .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    instance_group = df.groupby(["model_player_key", "injury_instance_base"], group_keys=False)
    df["injury_repeat_count"] = df.groupby(["model_player_key", "injury_key"]).cumcount()
    df["prior_same_injury_instances"] = instance_group["injury_instance_no"].transform(lambda s: s.cummax().shift(1)).fillna(0)
    df["new_instance_event"] = (
        (df["prior_same_injury_instances"] > 0)
        & (df["injury_instance_no"].fillna(0) > df["prior_same_injury_instances"])
    ).astype(int)
    first_same_injury_week = instance_group["season_week_index"].transform("min")
    df["weeks_since_first_same_injury"] = (df["season_week_index"] - first_same_injury_week).clip(lower=0)
    df["weeks_remaining"] = (18 - df["week"].fillna(0)).clip(lower=0)

    # Track consecutive unavailable games to represent time already missed.
    miss_streak = grouped["is_unavailable"].transform(
        lambda s: s.groupby((s == 0).cumsum()).cumsum()
    )
    df["time_missed_already"] = miss_streak.where(df["is_unavailable"] == 1, 0).astype(float)

    birth_year = pd.to_datetime(df["birth_date"], errors="coerce").dt.year
    df["age"] = df["season"] - birth_year

    # Compute return-distance outcomes on a per-week deduplicated view so that
    # multiple injury-report entries in the same game-week don't inflate distances.
    # The durations are then broadcast back to all rows for that player-week.
    dedup_cols = ["model_player_key", "season_week_index"]
    df_weekly = (
        df.sort_values(["model_player_key", "season_week_index", "transactionDate", "injury_report_row_id"], na_position="last")
        .groupby(dedup_cols, sort=False)
        .last()
        .reset_index()
    )

    weekly_durations: list[float] = []
    for _, group in df_weekly.groupby("model_player_key", sort=False):
        dist, _ = _compute_group_outcomes(group, np)
        weekly_durations.extend(dist.tolist())

    df_weekly["return_distance_games"] = weekly_durations

    # IR placement within 4 weeks: will a player NOT currently on IR be placed on IR
    # within the next 4 weekly injury-report entries?
    # NOTE: on_ir is unpopulated in the DB; use is_ir_transaction (derived from the
    # action field: "Placed on IR", "Placed on PUP" etc.) as the ground truth.
    df_weekly["ir_placed_next_4_weeks"] = (
        df_weekly.groupby("model_player_key")["is_ir_transaction"]
        .transform(lambda s: s.shift(-1).rolling(4, min_periods=1).max())
        .fillna(0)
        .astype(int)
    )

    # Merge the weekly outcomes back to the full (multi-report-per-week) dataframe.
    df = df.merge(
        df_weekly[dedup_cols + ["return_distance_games", "ir_placed_next_4_weeks"]],
        on=dedup_cols,
        how="left",
    )

    # ── Player career IR history features ────────────────────────────────────
    # These tell the model whether a player has a prior pattern of IR placement,
    # both overall and for the same body region — critical for IR placement prediction
    # because at inference time we don't yet know if the current injury will lead to IR.
    # Sort is already done above; re-sort here to be safe.
    df = df.sort_values(["model_player_key", "season_week_index", "transactionDate", "injury_report_row_id"], na_position="last")
    player_grp = df.groupby("model_player_key", group_keys=False)
    # player_career_ir_count / player_ir_same_region use is_ir_transaction (derived
    # from the action field) because on_ir is unpopulated in the DB.
    df["player_career_ir_count"] = player_grp["is_ir_transaction"].transform(
        lambda s: s.fillna(0).shift(1).fillna(0).cumsum()
    )
    df["player_ir_same_region"] = (
        df.groupby(["model_player_key", "injury_body_region"], group_keys=False)["is_ir_transaction"]
        .transform(lambda s: (s.fillna(0).shift(1).fillna(0).cumsum() > 0).astype(int))
    )

    # ── Team × body-region IR rate ─────────────────────────────────────────
    # How aggressively does this team place players on IR for this type of injury?
    team_region_ir = (
        df.groupby(["team", "injury_body_region"])["is_ir_transaction"]
        .mean()
        .reset_index()
        .rename(columns={"is_ir_transaction": "team_region_ir_rate"})
    )
    df = df.merge(team_region_ir, on=["team", "injury_body_region"], how="left", suffixes=("", "_trg"))
    df["team_region_ir_rate"] = pd.to_numeric(df["team_region_ir_rate"], errors="coerce").fillna(0.08)

    # ── Population-level injury features ────────────────────────────────────
    # Adds explicit league-wide context so the model knows e.g. "historically,
    # 41% of concussion players were listed Out/Doubtful" rather than only
    # relying on the implicit OHE coefficient for injury_detail.
    try:
        pop_rows = many("""
            SELECT
                COALESCE(NULLIF(TRIM(injury_detail),''),'Unknown') AS injury_detail,
                COALESCE(NULLIF(TRIM(position),''),'Unknown')      AS position,
                AVG(CASE WHEN report_status IN ('Out','Doubtful') THEN 1.0 ELSE 0.0 END) AS pop_unavail_rate,
                AVG(CASE WHEN on_ir = 1 THEN 1.0 ELSE 0.0 END)                           AS pop_ir_rate,
                AVG(CASE WHEN LOWER(TRIM(COALESCE(action,''))) IN (
                    'placed on ir','placed on pup','placed on nfi','out for season'
                ) THEN 1.0 ELSE 0.0 END)                                                  AS pop_action_ir_rate
            FROM injuryReport
            WHERE season >= 2011
            GROUP BY
                COALESCE(NULLIF(TRIM(injury_detail),''),'Unknown'),
                COALESCE(NULLIF(TRIM(position),''),'Unknown')
            HAVING COUNT(*) >= 5
        """)
        if pop_rows:
            pop_df = pd.DataFrame(pop_rows)
            pop_df["pop_unavail_rate"]    = pd.to_numeric(pop_df["pop_unavail_rate"], errors="coerce")
            pop_df["pop_ir_rate"]         = pd.to_numeric(pop_df["pop_ir_rate"], errors="coerce")
            pop_df["pop_action_ir_rate"]  = pd.to_numeric(pop_df["pop_action_ir_rate"], errors="coerce")
            pop_df["injury_detail"] = pop_df["injury_detail"].astype(str)
            pop_df["position"]      = pop_df["position"].astype(str)
            df["_mrg_detail"] = df["injury_detail"].fillna("Unknown").astype(str)
            df["_mrg_pos"]    = df["position"].fillna("Unknown").astype(str)
            df = df.merge(
                pop_df[["injury_detail", "position", "pop_unavail_rate", "pop_ir_rate", "pop_action_ir_rate"]],
                left_on=["_mrg_detail", "_mrg_pos"],
                right_on=["injury_detail", "position"],
                how="left",
                suffixes=("", "_pop"),
            )
            df.drop(columns=["_mrg_detail", "_mrg_pos",
                              "injury_detail_pop", "position_pop"], errors="ignore", inplace=True)
            df["pop_unavail_rate"]   = pd.to_numeric(df["pop_unavail_rate"], errors="coerce").fillna(0.15)
            df["pop_ir_rate"]        = pd.to_numeric(df["pop_ir_rate"], errors="coerce").fillna(0.05)
            df["pop_action_ir_rate"] = pd.to_numeric(df["pop_action_ir_rate"], errors="coerce").fillna(0.05)
        else:
            df["pop_unavail_rate"]   = 0.15
            df["pop_ir_rate"]        = 0.05
            df["pop_action_ir_rate"] = 0.05
    except Exception:
        df["pop_unavail_rate"]   = 0.15
        df["pop_ir_rate"]        = 0.05
        df["pop_action_ir_rate"] = 0.05
    # ────────────────────────────────────────────────────────────────────────

    # ── Population-level average return days (from injury_return_labels) ────
    # Leakage-safe: uses historical population averages from PAST cases,
    # not the current injury's own return_days target.
    try:
        return_rows = many("""
            SELECT
                LOWER(TRIM(COALESCE(e.injury_detail, ''))) AS injury_detail,
                AVG(l.return_days)                         AS pop_avg_return_days,
                COUNT(*)                                   AS n
            FROM injury_return_labels l
            JOIN events e ON e.injuryId = l.injury_id
            WHERE l.return_days IS NOT NULL
              AND l.return_days BETWEEN 0 AND 180
            GROUP BY LOWER(TRIM(COALESCE(e.injury_detail, '')))
            HAVING COUNT(*) >= 10
        """)
        if return_rows:
            ret_detail_df = pd.DataFrame(return_rows)
            ret_detail_df["pop_avg_return_days"] = pd.to_numeric(ret_detail_df["pop_avg_return_days"], errors="coerce")
            ret_detail_df["injury_detail_key"] = ret_detail_df["injury_detail"].astype(str)

            # fallback: type-level averages
            return_type_rows = many("""
                SELECT
                    LOWER(TRIM(COALESCE(e.injury_type, ''))) AS injury_type,
                    AVG(l.return_days)                       AS pop_avg_return_days_type,
                    COUNT(*)                                 AS n
                FROM injury_return_labels l
                JOIN events e ON e.injuryId = l.injury_id
                WHERE l.return_days IS NOT NULL
                  AND l.return_days BETWEEN 0 AND 180
                GROUP BY LOWER(TRIM(COALESCE(e.injury_type, '')))
                HAVING COUNT(*) >= 10
            """) or []
            ret_type_df = pd.DataFrame(return_type_rows) if return_type_rows else pd.DataFrame(columns=["injury_type", "pop_avg_return_days_type"])
            ret_type_df["pop_avg_return_days_type"] = pd.to_numeric(ret_type_df.get("pop_avg_return_days_type"), errors="coerce")

            df["_mrg_detail_ret"] = df["injury_detail"].fillna("").str.lower().str.strip()
            df["_mrg_type_ret"]   = df["injury_type"].fillna("").str.lower().str.strip()

            df = df.merge(
                ret_detail_df[["injury_detail_key", "pop_avg_return_days"]],
                left_on="_mrg_detail_ret",
                right_on="injury_detail_key",
                how="left",
            )
            df.drop(columns=["injury_detail_key"], errors="ignore", inplace=True)

            if not ret_type_df.empty:
                df = df.merge(
                    ret_type_df[["injury_type", "pop_avg_return_days_type"]],
                    left_on="_mrg_type_ret",
                    right_on="injury_type",
                    how="left",
                    suffixes=("", "_rt"),
                )
                df.drop(columns=["injury_type_rt"], errors="ignore", inplace=True)
                # Fill detail-level gaps with type-level averages
                df["pop_avg_return_days"] = df["pop_avg_return_days"].fillna(df["pop_avg_return_days_type"])
                df.drop(columns=["pop_avg_return_days_type"], errors="ignore", inplace=True)

            df.drop(columns=["_mrg_detail_ret", "_mrg_type_ret"], errors="ignore", inplace=True)
            df["pop_avg_return_days"] = pd.to_numeric(df["pop_avg_return_days"], errors="coerce").fillna(14.0)
        else:
            df["pop_avg_return_days"] = 14.0
    except Exception:
        df["pop_avg_return_days"] = 14.0

    # Derived feature: how many days remain in the expected recovery window
    df["days_remaining_estimate"] = (df["pop_avg_return_days"] - df["time_missed_already"]).clip(lower=0)
    # ────────────────────────────────────────────────────────────────────────

    return df


def _train_binary_pipeline(df, feature_cols, categorical_cols, target_col):
    """
    Train a binary classification pipeline to predict a yes/no target column.

    Architecture:
      - HistGradientBoostingClassifier: gradient-boosted trees that handle
        missing values natively (no imputation required for numeric columns)
      - OrdinalEncoder: converts categorical columns to integer codes
      - CalibratedClassifierCV: wraps the classifier with Platt scaling so
        predict_proba() returns well-calibrated probabilities rather than
        raw scores
      - Temporal train/test split: seasons \u2264 2023 train, seasons \u2265 2024 test

    Returns a dict with 'ready', 'model', 'metrics', and supporting info,
    or a dict with 'ready': False and a 'message' if training fails.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    # Drop rows where the target label is unknown (can't learn from them)
    local_df = df.dropna(subset=[target_col]).copy()
    if local_df.empty or local_df[target_col].nunique() < 2:
        return {
            "ready": False,
            "message": f"Insufficient class variety for {target_col}",
        }

    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    available_cols = [col for col in feature_cols if col in local_df.columns]
    categorical_cols = [col for col in categorical_cols if col in available_cols]
    numeric_cols = [col for col in numeric_cols if col in available_cols]

    # Limit cardinality for high-cardinality categorical columns.
    # Keep only the 50 most frequent values; rare values are grouped as 'Other'
    # to prevent overfitting on team names, stadiums, etc.
    for col in categorical_cols:
        local_df[col] = local_df[col].fillna("Unknown").astype(str)
        top_values = local_df[col].value_counts(dropna=False).head(50).index
        local_df[col] = local_df[col].where(local_df[col].isin(top_values), "Other")

    # Temporal split: seasons \u2264 2023 train, seasons \u2265 2024 test.
    # This prevents data leakage by never training on data that postdates the test set.
    train_mask = local_df["season"].fillna(0) <= 2023
    test_mask = local_df["season"].fillna(0) >= 2024
    test_cutoff = 2024
    while (int(train_mask.sum()) < 200 or int(test_mask.sum()) < 100) and test_cutoff > 2016:
        test_cutoff -= 1
        train_mask = local_df["season"].fillna(0) <= (test_cutoff - 1)
        test_mask = local_df["season"].fillna(0) >= test_cutoff

    if int(train_mask.sum()) < 200 or int(test_mask.sum()) < 100:
        split_idx = int(len(local_df) * 0.8)
        ranked = local_df.sort_values(
            ["season", "week", "transactionDate"],
            na_position="last",
        ).index
        train_mask = local_df.index.isin(ranked[:split_idx])
        test_mask = ~train_mask

    X_train = local_df.loc[train_mask, available_cols]
    y_train = local_df.loc[train_mask, target_col].astype(int)
    X_test = local_df.loc[test_mask, available_cols]
    y_test = local_df.loc[test_mask, target_col].astype(int)

    if X_train.empty or X_test.empty or y_train.nunique() < 2:
        return {
            "ready": False,
            "message": f"Unable to split train/test for {target_col}",
        }

    # HistGradientBoostingClassifier wrapped with isotonic calibration.
    # CalibratedClassifierCV(method='isotonic') fits a monotonic mapping from raw scores →
    # probabilities using 5-fold CV, correcting overconfident/underconfident outputs so that
    # predict_proba values are reliably interpretable (e.g. "73% chance of playing").
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    )),
                ]),
                categorical_cols,
            ),
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_cols,
            ),
        ],
        remainder="drop",
    )

    n_cat = len(categorical_cols)
    inner_pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=400,
                    learning_rate=0.05,
                    max_depth=6,
                    min_samples_leaf=40,
                    l2_regularization=0.1,
                    class_weight="balanced",
                    categorical_features=list(range(n_cat)) if n_cat else None,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline = CalibratedClassifierCV(inner_pipeline, method="isotonic", cv=5)

    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    auc = 0.5
    if y_test.nunique() > 1:
        auc = roc_auc_score(y_test, y_prob)

    # Find the probability threshold that maximises Youden's J (sensitivity + specificity - 1)
    # on the test set. Using 0.5 on imbalanced tasks causes F1=0
    # because the calibrated model never emits probabilities that high for the rare class.
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    j_scores = tpr - fpr
    opt_threshold = float(thresholds[j_scores.argmax()]) if len(thresholds) > 0 else 0.5
    opt_threshold = max(0.1, min(0.9, opt_threshold))  # clamp to a sane range
    y_pred = (y_prob >= opt_threshold).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "roc_auc": round(float(auc), 4),
        "threshold": round(opt_threshold, 4),
    }

    return {
        "ready": True,
        "model": pipeline,
        "threshold": opt_threshold,
        "features": available_cols,
        "categorical_features": categorical_cols,
        "numeric_features": numeric_cols,
        "metrics": metrics,
        "row_count": int(len(local_df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate": float(local_df[target_col].mean()),
    }


def _extract_top_features(trained_model, limit=12):
    model = trained_model.get("model")
    if not model:
        return []

    try:
        # Handle CalibratedClassifierCV wrapping an inner Pipeline.
        # Structure: CalibratedClassifierCV → calibrated_classifiers[0].estimator → Pipeline(prep, clf)
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.pipeline import Pipeline as SkPipeline
        if isinstance(model, CalibratedClassifierCV):
            # Pull the first fold's estimator (all folds share same structure/importances are averaged)
            inner = model.calibrated_classifiers_[0].estimator
            if isinstance(inner, SkPipeline):
                prep = inner.named_steps["prep"]
                clf = inner.named_steps["clf"]
            else:
                return []
        elif hasattr(model, "named_steps"):
            prep = model.named_steps["prep"]
            clf = model.named_steps["clf"]
        else:
            return []

        names = prep.get_feature_names_out()
        # Tree models expose feature_importances_; linear models expose coef_
        if hasattr(clf, "feature_importances_"):
            scores = clf.feature_importances_
            use_importance = True
        elif hasattr(clf, "coef_"):
            scores = clf.coef_[0]
            use_importance = False
        else:
            return []
    except Exception:
        return []

    if not len(scores):
        return []

    indexed = sorted(
        zip(names, scores),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:limit]

    top = []
    for raw_name, score in indexed:
        clean_name = str(raw_name).replace("cat__", "").replace("num__", "")
        score_val = float(score)
        if use_importance:
            top.append({
                "feature": clean_name,
                "coefficient": round(score_val, 4),
                "direction": "important feature",
            })
        else:
            top.append({
                "feature": clean_name,
                "coefficient": round(score_val, 4),
                "direction": "raises probability" if score_val > 0 else "lowers probability",
            })
    return top


def _mode_or_default(values, default_value):
    if values is None:
        return default_value
    clean = values.dropna()
    if clean.empty:
        return default_value
    if clean.dtype == object:
        clean = clean.astype(str).str.strip()
        clean = clean[clean != ""]
        if clean.empty:
            return default_value
    mode = clean.mode(dropna=True)
    if mode.empty:
        return default_value
    return mode.iloc[0]


def _fetch_pop_features(injury_detail: str, position: str = ""):
    """Fast per-injury-type population stats used at inference time.
    Returns (pop_unavail_rate, pop_ir_rate, pop_action_ir_rate) as floats.
    Tries injury×position first; falls back to injury-only if insufficient rows.
    """
    try:
        detail = (injury_detail or "Unknown").strip() or "Unknown"
        pos    = (position    or "Unknown").strip() or "Unknown"

        row = one(
            """
            SELECT
                AVG(CASE WHEN report_status IN ('Out','Doubtful') THEN 1.0 ELSE 0.0 END) AS pop_unavail_rate,
                AVG(CASE WHEN on_ir = 1 THEN 1.0 ELSE 0.0 END)                           AS pop_ir_rate,
                AVG(CASE WHEN LOWER(TRIM(COALESCE(action,''))) IN (
                    'placed on ir','placed on pup','placed on nfi','out for season'
                ) THEN 1.0 ELSE 0.0 END)                                                  AS pop_action_ir_rate,
                COUNT(*) AS n
            FROM injuryReport
            WHERE COALESCE(NULLIF(TRIM(injury_detail),''),'Unknown') = %s
              AND COALESCE(NULLIF(TRIM(position),''),'Unknown')      = %s
            HAVING COUNT(*) >= 5
            """,
            (detail, pos),
        )
        if row and row.get("pop_unavail_rate") is not None:
            return float(row["pop_unavail_rate"]), float(row["pop_ir_rate"] or 0.0), float(row["pop_action_ir_rate"] or 0.0)

        # Fallback: injury-only (across all positions)
        row = one(
            """
            SELECT
                AVG(CASE WHEN report_status IN ('Out','Doubtful') THEN 1.0 ELSE 0.0 END) AS pop_unavail_rate,
                AVG(CASE WHEN on_ir = 1 THEN 1.0 ELSE 0.0 END)                           AS pop_ir_rate,
                AVG(CASE WHEN LOWER(TRIM(COALESCE(action,''))) IN (
                    'placed on ir','placed on pup','placed on nfi','out for season'
                ) THEN 1.0 ELSE 0.0 END)                                                  AS pop_action_ir_rate
            FROM injuryReport
            WHERE COALESCE(NULLIF(TRIM(injury_detail),''),'Unknown') = %s
            HAVING COUNT(*) >= 5
            """,
            (detail,),
        )
        if row and row.get("pop_unavail_rate") is not None:
            return float(row["pop_unavail_rate"]), float(row["pop_ir_rate"] or 0.0), float(row["pop_action_ir_rate"] or 0.0)
    except Exception:
        pass
    return 0.15, 0.05, 0.05  # global population fallback


def _fetch_team_ir_features(team: str, body_region: str) -> float:
    """Return this team's historical IR-placement rate for the given body region.
    Falls back to league-wide rate for that region, then to a global default.
    """
    try:
        t   = (team        or "Unknown").strip() or "Unknown"
        rgn = (body_region or "Unknown").strip() or "Unknown"
        # Match body region by grouping all injury_detail values that map to it.
        # We store region in Python but not in the DB, so filter by known keywords.
        row = one(
            """
            SELECT
                AVG(CASE WHEN LOWER(TRIM(COALESCE(action,''))) IN (
                    'placed on ir','placed on pup','placed on nfi','out for season'
                ) THEN 1.0 ELSE 0.0 END) AS team_region_ir_rate,
                COUNT(*) AS n
            FROM injuryReport
            WHERE COALESCE(NULLIF(TRIM(team),''),'Unknown') = %s
              AND season >= 2011
            HAVING COUNT(*) >= 10
            """,
            (t,),
        )
        # We can't easily filter by body_region in SQL (it's a Python mapping), so
        # we get team-level overall rate as the best available DB signal.
        if row and row.get("team_region_ir_rate") is not None:
            return float(row["team_region_ir_rate"])
    except Exception:
        pass
    return 0.08  # global fallback


def _fetch_player_ir_history(gsis_id: str, body_region: str):
    """Return (player_career_ir_count, player_ir_same_region) for a resolved player.
    player_career_ir_count: total past IR placements for this player.
    player_ir_same_region:  1 if any past IR placement involved the same body region.
    Falls back to (0, 0) when the player is unknown or has no history.
    """
    try:
        pid = (gsis_id or "").strip()
        if not pid or pid.lower() in ("unknown", "na", ""):
            return 0, 0

        row = one(
            """
            SELECT
                SUM(CASE WHEN LOWER(TRIM(COALESCE(action,''))) IN (
                    'placed on ir','placed on pup','placed on nfi','out for season'
                ) THEN 1 ELSE 0 END) AS career_ir_count
            FROM injuryReport
            WHERE gsis_id = %s
              AND season >= 2011
            """,
            (pid,),
        )
        career_ir = int(row["career_ir_count"] or 0) if row else 0

        # For same-region check we pull player's injury_detail history and test
        # each against our Python region map — cheaper than replicating the map in SQL.
        detail_rows = many(
            """
            SELECT DISTINCT injury_detail
            FROM injuryReport
            WHERE gsis_id = %s
              AND LOWER(TRIM(COALESCE(action,''))) IN (
                    'placed on ir','placed on pup','placed on nfi','out for season')
              AND season >= 2011
            """,
            (pid,),
        )
        same_region = 0
        for dr in (detail_rows or []):
            if _injury_body_region(str(dr.get("injury_detail") or "")) == body_region:
                same_region = 1
                break

        return career_ir, same_region
    except Exception:
        pass
    return 0, 0


def _build_inference_defaults(df, feature_cols, categorical_cols):
    defaults = {}
    for col in feature_cols:
        if col not in df.columns:
            defaults[col] = "Unknown" if col in categorical_cols else 0.0
            continue
        if col in categorical_cols:
            defaults[col] = str(_mode_or_default(df[col], "Unknown") or "Unknown")
        else:
            median_val = df[col].dropna().median() if hasattr(df[col], "dropna") else 0.0
            defaults[col] = _to_float(median_val, 0.0)
    return defaults


def _build_model_input_options(df):
    options = {}
    if df is None or df.empty:
        return {
            "positions": [],
            "injury_types": [],
            "injury_details": [],
            "stadiums": [],
            "report_statuses": [],
            "practice_statuses": [],
            "severity_levels": [],
            "game_types": [],
        }

    config = {
        "positions": ("position", 60),
        "injury_types": ("injury_type", 60),
        "injury_details": ("injury_detail", 200),
        "stadiums": ("next_game_stadium", 80),
        "report_statuses": ("report_status", 25),
        "practice_statuses": ("practice_status", 25),
        "severity_levels": ("severity", 20),
        "game_types": ("game_type", 10),
    }

    for key, (col, max_items) in config.items():
        if col not in df.columns:
            options[key] = []
            continue
        series = (
            df[col]
            .dropna()
            .astype(str)
            .map(lambda x: x.strip())
        )
        series = series[series != ""]
        if series.empty:
            options[key] = []
            continue
        ranked = series.value_counts().head(max_items).index.tolist()
        options[key] = sorted(ranked)

    return options


# ── Model bundle in-memory cache and refresh state ─────────────────────────────
# The trained model bundle is loaded once per server process and held in
# _model_bundle_cache.  A threading.Lock prevents multiple threads from
# training simultaneously (which could exhaust memory and CPU).
_model_bundle_cache: dict | None = None
_model_refresh_lock = threading.Lock()
# Tracks whether a background refresh is currently running and its outcome.
_model_refresh_state = {
    "in_progress": False,    # True while _refresh_model_bundle_async is running
    "started_at": None,      # Unix timestamp when the refresh began
    "finished_at": None,     # Unix timestamp when the refresh completed
    "last_error": None,      # Error message if the refresh failed
    "last_message": None,    # Human-readable status for the API response
}


def _snapshot_model_refresh_state():
    """Return a copy of the current model-refresh state for safe read outside the lock."""
    return {
        "in_progress": bool(_model_refresh_state.get("in_progress")),
        "started_at": _model_refresh_state.get("started_at"),
        "finished_at": _model_refresh_state.get("finished_at"),
        "last_error": _model_refresh_state.get("last_error"),
        "last_message": _model_refresh_state.get("last_message"),
    }


def _refresh_model_bundle_async():
    """
    Train and cache a new model bundle in the background.

    This function is meant to be run in a daemon thread so the HTTP server
    keeps responding while training happens.  It acquires _model_refresh_lock
    to update the global _model_bundle_cache when training is complete.
    """
    with _model_refresh_lock:
        _model_refresh_state["in_progress"] = True
        _model_refresh_state["started_at"] = int(time.time())
        _model_refresh_state["finished_at"] = None
        _model_refresh_state["last_error"] = None
        _model_refresh_state["last_message"] = "Model refresh started."

    try:
        bundle = get_advanced_model_bundle(force_retrain=True)
        if bundle.get("ready"):
            message = bundle.get("message", "Model refresh completed.")
            last_error = None
        else:
            message = bundle.get("message", "Model refresh finished, but the bundle is not ready.")
            last_error = message
    except Exception as exc:
        message = f"Model refresh failed: {str(exc)}"
        last_error = message
    finally:
        build_model_overview_payload.cache_clear()
        build_model_input_options_payload.cache_clear()
        with _model_refresh_lock:
            _model_refresh_state["in_progress"] = False
            _model_refresh_state["finished_at"] = int(time.time())
            _model_refresh_state["last_error"] = last_error
            _model_refresh_state["last_message"] = message


def _start_model_refresh_if_needed():
    """
    Trigger a background model refresh if one is not already in progress.

    Returns True if a new refresh thread was started, False if one was already
    running (so the caller knows not to wait for a new result).
    """
    with _model_refresh_lock:
        if _model_refresh_state.get("in_progress"):
            return False
    refresh_thread = threading.Thread(
        target=_refresh_model_bundle_async,
        name="model-refresh",
        daemon=True,
    )
    refresh_thread.start()
    return True


def _warmup_model_bundle():
    """Load model from disk cache (or train if no valid cache) in a background thread at startup."""
    global _model_bundle_cache
    with _model_refresh_lock:
        _model_refresh_state["in_progress"] = True
        _model_refresh_state["started_at"] = int(time.time())
        _model_refresh_state["last_message"] = "Loading model from cache..."
        _model_refresh_state["last_error"] = None
        _model_refresh_state["finished_at"] = None
    try:
        bundle = get_advanced_model_bundle(force_retrain=False)
        message = "Model ready." if bundle.get("ready") else bundle.get("message", "Model not ready.")
        last_error = None if bundle.get("ready") else message
    except Exception as exc:
        message = f"Model warmup failed: {str(exc)}"
        last_error = message
    finally:
        build_model_overview_payload.cache_clear()
        build_model_input_options_payload.cache_clear()
        with _model_refresh_lock:
            _model_refresh_state["in_progress"] = False
            _model_refresh_state["finished_at"] = int(time.time())
            _model_refresh_state["last_message"] = message
            _model_refresh_state["last_error"] = last_error


def get_advanced_model_bundle(force_retrain: bool = False):
    """
    Build (or return cached) the ML model bundle.
    Results are cached only on success — if training fails (DB down, empty rows, etc.)
    the next call will retry automatically.
    Trained bundle is also persisted to disk via joblib so server restarts load
    instantly without a full retrain.
    """
    global _model_bundle_cache
    if not force_retrain and _model_bundle_cache is not None and _model_bundle_cache.get("ready"):
        return _model_bundle_cache

    # Try loading from disk cache — avoids full retrain on server restart.
    try:
        if (not force_retrain) and os.path.exists(_MODEL_CACHE_PATH):
            age_hours = (time.time() - os.path.getmtime(_MODEL_CACHE_PATH)) / 3600
            if age_hours < _MODEL_CACHE_MAX_AGE_HOURS:
                # Load in a daemon thread so we can enforce a timeout safely from
                # any thread (SIGALRM only works from the main thread on macOS).
                _result: list = []
                def _load_cache():
                    try:
                        _result.append(joblib.load(_MODEL_CACHE_PATH))
                    except Exception as e:
                        _result.append(e)
                _t = threading.Thread(target=_load_cache, daemon=True)
                _t.start()
                _t.join(timeout=30)
                if _result and isinstance(_result[0], dict) and _result[0].get("ready"):
                    cached = _result[0]
                    _model_bundle_cache = cached
                    return cached
    except Exception:
        pass  # corrupt or incompatible — fall through to retrain
    try:
        import pandas as pd
    except Exception as exc:
        return {
            "ready": False,
            "message": f"Model libraries unavailable: {str(exc)}",
        }

    try:
        rows = _build_model_base_rows()
    except Exception as exc:
        return {
            "ready": False,
            "message": f"Database unavailable — could not load training data: {str(exc)}",
        }

    if not rows:
        return {
            "ready": False,
            "message": "No training rows returned from the database (check DB connection and season filter).",
        }

    try:
        df = _engineer_model_dataframe(rows)
    except Exception as exc:
        return {
            "ready": False,
            "message": f"Feature engineering failed: {str(exc)}",
        }

    if df.empty:
        return {
            "ready": False,
            "message": "No model-ready rows available after feature engineering.",
        }

    feature_cols = [
        "season",
        "week",
        "team",
        "game_type",
        "injury_type",
        "injury_detail",
        "injury_body_region",
        "severity",
        "position",
        "position_group",
        "report_status",
        "practice_status",
        "source",
        "next_game_stadium",
        "next_game_opponent",
        "next_game_is_home",
        "next_game_is_dome",
        "days_until_next_game",
        "time_missed_already",
        "injury_instance_no",
        "prior_same_injury_instances",
        "new_instance_event",
        "weeks_since_first_same_injury",
        "days_since_prev",
        "new_injury_flag",
        "on_ir",
        "prev_on_ir",
        "years_of_experience",
        "age",
        "prior_reports_count",
        "prior_unavailable_count",
        "prev3_unavail_rate",
        "gap_since_prev_report",
        "injury_repeat_count",
        "weeks_remaining",
        "pop_unavail_rate",       # league-wide unavailability rate for this injury×position
        "pop_ir_rate",            # league-wide IR rate for this injury×position
        "pop_action_ir_rate",     # league-wide rate of IR-placement action for this injury×position
        "pop_avg_return_days",    # historical avg return days for this injury_detail (leakage-safe population stat)
        "days_remaining_estimate",# pop_avg_return_days - time_missed_already (clipped ≥ 0)
        "action",             # transaction action type: "Placed on IR", "none", etc.
        "is_ir_transaction",  # 1 if action indicates a reserve-list placement (IR/PUP/NFI)
        "team_region_ir_rate",  # this team's historical IR rate for this body region
        "player_career_ir_count",  # number of times this player was on IR before this record
        "player_ir_same_region",   # 1 if player has ever been on IR for the same body region
    ]

    categorical_cols = [
        "team",
        "game_type",
        "injury_type",
        "injury_detail",
        "injury_body_region",
        "severity",
        "position",
        "position_group",
        "report_status",
        "practice_status",
        "source",
        "next_game_stadium",
        "next_game_opponent",
        "action",
    ]

    # 2013 is the first season with snap-count ground-truth labels (nflverse/PFR coverage).
    # All sub-models use the same floor so training populations are consistent.
    ML_SEASON_FLOOR = 2013

    availability_train = df[(df["season"] >= ML_SEASON_FLOOR)].dropna(subset=["next_game_available"]).copy()
    availability_model = _train_binary_pipeline(
        availability_train,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        target_col="next_game_available",
    )

    duration_base = df[
        (df["season"] >= ML_SEASON_FLOOR)
        & (df["is_unavailable"] == 1)
        & (df["return_distance_games"].notna())
    ].copy()
    # No upper cap — season-ending injuries (large return_distance_games) are valid
    # negative examples for return_within_X horizons and improve model discrimination.

    duration_models = {}
    for horizon in (1, 2, 3):
        col = f"return_within_{horizon}"
        duration_base[col] = (duration_base["return_distance_games"] <= horizon).astype(int)
        duration_models[horizon] = _train_binary_pipeline(
            duration_base,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            target_col=col,
        )

    ir_train = df[
        (df["season"] >= ML_SEASON_FLOOR)
        & (df["is_ir_transaction"] == 0)   # exclude rows that ARE an IR placement; predict future placement
        & (df["ir_placed_next_4_weeks"].notna())
    ].copy()
    ir_placement_model = _train_binary_pipeline(
        ir_train,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        target_col="ir_placed_next_4_weeks",
    )

    ready = bool(
        availability_model.get("ready")
        and any(m.get("ready") for m in duration_models.values())
    )
    # ir_placement_model is optional — on_ir may be unpopulated in some DB states

    bundle = {
        "ready": ready,
        "message": "Advanced multi-task player availability models trained from injury history.",
        "dataset": {
            "row_count": int(len(df)),
            "players": int(df["model_player_key"].nunique()),
            "seasons": [int(df["season"].min()), int(df["season"].max())],
            "availability_rows": int(len(availability_train)),
            "duration_rows": int(len(duration_base)),
            "ir_rows": int(len(ir_train)),
            "feature_count": int(len(feature_cols)),
        },
        "feature_cols": feature_cols,
        "availability_model": availability_model,
        "duration_models": duration_models,
        "ir_placement_model": ir_placement_model,
        "top_features": _extract_top_features(availability_model, limit=12),
        "inference_defaults": _build_inference_defaults(df, feature_cols, categorical_cols),
        "input_options": _build_model_input_options(df),
    }
    if bundle["ready"]:
        _model_bundle_cache = bundle
        try:
            import signal as _signal
            def _dump_timeout(signum, frame):
                raise TimeoutError("joblib.dump hung")
            _old = _signal.signal(_signal.SIGALRM, _dump_timeout)
            _signal.alarm(60)  # 60-second limit on cache write
            try:
                joblib.dump(bundle, _MODEL_CACHE_PATH)
            finally:
                _signal.alarm(0)
                _signal.signal(_signal.SIGALRM, _old)
        except Exception:
            pass  # disk save failure is non-fatal
    return bundle


@lru_cache(maxsize=1)
def build_model_overview_payload():
    bundle = get_advanced_model_bundle()
    if not bundle.get("ready"):
        return {
            "ready": False,
            "message": bundle.get("message", "Model pipeline not ready."),
            "dataset": bundle.get("dataset", {}),
            "tasks": {},
            "top_features": [],
        }

    availability = bundle.get("availability_model", {})
    duration_models = bundle.get("duration_models", {})
    ir_placement = bundle.get("ir_placement_model", {})

    duration_summary = {}
    for horizon, model_obj in duration_models.items():
        if model_obj.get("ready"):
            duration_summary[str(horizon)] = model_obj.get("metrics", {})

    return {
        "ready": True,
        "message": bundle.get("message", "Model overview fetched."),
        "dataset": bundle.get("dataset", {}),
        "tasks": {
            "next_game_availability": {
                "label": "Play Next Game",
                "metrics": availability.get("metrics", {}),
                "positive_rate_percent": _safe_pct(availability.get("positive_rate", 0.0)),
                "rows": availability.get("row_count", 0),
            },
            "return_timeline": {
                "label": "Return Within N Games",
                "horizon_metrics": duration_summary,
                # Summarise across horizons: use the median AUC and best F1 across the 4 models.
                # This lets the overview endpoint report consistent top-level metrics.
                "metrics": {
                    "roc_auc": round(
                        float(
                            sorted(
                                [
                                    v.get("roc_auc", 0)
                                    for v in duration_summary.values()
                                    if v.get("roc_auc") is not None
                                ]
                            )[len(duration_summary) // 2]
                        ),
                        4,
                    ) if duration_summary else None,
                    "f1": round(
                        float(
                            max(
                                (v.get("f1", 0) for v in duration_summary.values()),
                                default=0,
                            )
                        ),
                        4,
                    ) if duration_summary else None,
                    "f1_macro": round(
                        float(
                            max(
                                (v.get("f1_macro", 0) for v in duration_summary.values()),
                                default=0,
                            )
                        ),
                        4,
                    ) if duration_summary else None,
                },
                "rows": bundle.get("dataset", {}).get("duration_rows", 0),
            },
            "ir_placement": {
                "label": "IR Placement Within 4 Weeks",
                "metrics": ir_placement.get("metrics", {}),
                "positive_rate_percent": _safe_pct(ir_placement.get("positive_rate", 0.0)),
                "rows": bundle.get("dataset", {}).get("ir_rows", 0),
            },
        },
        "top_features": bundle.get("top_features", []),
    }


def _prepare_player_feature_row(player_id: str):
    bundle = get_advanced_model_bundle()
    rows = _build_model_base_rows(player_id=player_id)
    df = _engineer_model_dataframe(rows)
    if df.empty:
        return None, bundle
    latest = df.sort_values(["season_week_index", "transactionDate", "injury_report_row_id"], na_position="last").iloc[-1]
    return latest, bundle


def _sanitize_model_inputs(raw_inputs: dict | None):
    data = raw_inputs or {}
    severity_raw = str(data.get("severity") or "").strip()
    severity_key = severity_raw.lower()
    # Accept UI tier labels and map them to model-native values.
    severity_map = {
        "minor": "Sprain",
        "moderate": "Dislocated",
        "serious": "Season-Ending",
        "season ending": "Season-Ending",
    }
    severity_clean = severity_map.get(severity_key, severity_raw)

    report_raw = str(data.get("report_status") or "").strip()
    report_map = {
        "questionable": "Questionable",
        "doubtful": "Doubtful",
        "out": "Out",
    }
    report_clean = report_map.get(report_raw.lower(), report_raw)

    practice_raw = str(data.get("practice_status") or "").strip()
    practice_map = {
        "did not practice": "Did Not Participate",
        "dnp": "Did Not Participate",
        "limited": "Limited",
        "full": "Full",
        "not listed": "Not Listed",
    }
    practice_clean = practice_map.get(practice_raw.lower(), practice_raw)

    clean = {
        "player_name": str(data.get("player_name") or "").strip(),
        "position": str(data.get("position") or "").strip(),
        "injury_type": str(data.get("injury_type") or "").strip(),
        "injury_detail": str(data.get("injury_detail") or "").strip(),
        "next_game_stadium": str(data.get("next_game_stadium") or "").strip(),
        "days_until_next_game": _to_float(data.get("days_until_next_game"), 0.0),
        "time_missed_already": _to_float(data.get("time_missed_already"), 0.0),
        "report_status": report_clean,
        "practice_status": practice_clean,
        "severity": severity_clean,
        "game_type": str(data.get("game_type") or "").strip(),
    }
    return clean


def _lookup_player_by_name(player_name: str):
    token = (player_name or "").strip().lower()
    if not token:
        return None

    rows = many(
        """
        SELECT
            p.gsis_id,
            COALESCE(NULLIF(TRIM(p.display_name), ''), NULLIF(TRIM(p.football_name), ''), TRIM(CONCAT_WS(' ', p.first_name, p.last_name)), p.short_name, p.gsis_id) AS full_name,
            COALESCE(p.position, '') AS position,
            CASE
                WHEN EXISTS (
                    SELECT 1
                    FROM injuryReport ir
                    WHERE ir.gsis_id IS NOT NULL
                      AND TRIM(ir.gsis_id) <> ''
                      AND LOWER(TRIM(ir.gsis_id)) <> 'na'
                      AND TRIM(ir.gsis_id) = TRIM(p.gsis_id)
                    LIMIT 1
                )
                THEN 1
                ELSE 0
            END AS has_injury_history
        FROM players p
        WHERE p.gsis_id IS NOT NULL
          AND TRIM(p.gsis_id) <> ''
          AND (
               LOWER(COALESCE(p.display_name, '')) = %s
            OR LOWER(COALESCE(p.football_name, '')) = %s
            OR LOWER(TRIM(CONCAT_WS(' ', p.first_name, p.last_name))) = %s
            OR LOWER(COALESCE(p.last_name, '')) LIKE %s
            OR LOWER(COALESCE(p.first_name, '')) LIKE %s
          )
        ORDER BY
            CASE
                WHEN LOWER(COALESCE(p.display_name, '')) = %s THEN 0
                WHEN LOWER(COALESCE(p.football_name, '')) = %s THEN 1
                WHEN LOWER(TRIM(CONCAT_WS(' ', p.first_name, p.last_name))) = %s THEN 2
                WHEN LOWER(COALESCE(p.last_name, '')) LIKE %s THEN 3
                WHEN LOWER(COALESCE(p.first_name, '')) LIKE %s THEN 4
                ELSE 5
            END,
            has_injury_history DESC,
            COALESCE(p.display_name, ''),
            p.gsis_id
        LIMIT 1
        """,
        (
            token,
            token,
            token,
            f"{token}%",
            f"{token}%",
            token,
            token,
            token,
            f"{token}%",
            f"{token}%",
        ),
    ) or []

    if not rows:
        return None
    row = rows[0]
    return {
        "gsis_id": row.get("gsis_id"),
        "full_name": row.get("full_name") or "Unknown",
        "position": row.get("position") or "",
        "has_injury_history": bool(row.get("has_injury_history")),
    }


_schedules_df: "pd.DataFrame | None" = None


def _load_schedules_data() -> "pd.DataFrame":
    """Load nfl_data/schedules.csv into a module-level DataFrame (cached after first load)."""
    global _schedules_df
    if _schedules_df is not None:
        return _schedules_df
    csv_path = os.path.join(os.path.dirname(__file__), "nfl_data", "schedules.csv")
    if not os.path.exists(csv_path):
        import pandas as _pd
        _schedules_df = _pd.DataFrame()
        return _schedules_df
    _schedules_df = pd.read_csv(csv_path, low_memory=False)
    return _schedules_df


def _get_next_game_from_schedules(team: str, season: int, week: int, game_type: str = "REG") -> "dict | None":
    """Return the first game for *team* in *season* after *week*, or None if not found.

    Returns a dict with keys: stadium, opponent, is_home, is_dome, game_date, next_week.
    is_dome is derived from the roof column (dome / closed / retractable → 1).
    """
    sched = _load_schedules_data()
    if sched.empty:
        return None
    mask = (
        (sched["season"].astype(int) == int(season))
        & (sched["game_type"].astype(str) == str(game_type))
        & (sched["week"].astype(int) > int(week))
        & (
            (sched["home_team"].astype(str) == str(team))
            | (sched["away_team"].astype(str) == str(team))
        )
    )
    next_games = sched[mask].sort_values("week")
    if next_games.empty:
        return None
    row = next_games.iloc[0]
    is_home = int(str(row.get("home_team", "")) == str(team))
    opponent = str(row.get("away_team" if is_home else "home_team", "Unknown"))
    stadium = str(row.get("stadium") or "Unknown")
    roof = str(row.get("roof") or "outdoors").lower()
    is_dome = int(roof in ("dome", "closed", "retractable"))
    gameday = row.get("gameday")
    return {
        "stadium": stadium,
        "opponent": opponent,
        "is_home": is_home,
        "is_dome": is_dome,
        "game_date": str(gameday) if pd.notna(gameday) else None,
        "next_week": int(row.get("week", week + 1)),
    }


def _apply_inference_overrides(feature_row: dict, model_inputs: dict):
    if feature_row is None:
        return feature_row

    mapping = {
        "position": "position",
        "injury_type": "injury_type",
        "injury_detail": "injury_detail",
        "next_game_stadium": "next_game_stadium",
        "report_status": "report_status",
        "practice_status": "practice_status",
        "severity": "severity",
        "game_type": "game_type",
    }

    for incoming, feature in mapping.items():
        val = str(model_inputs.get(incoming) or "").strip()
        if val:
            feature_row[feature] = val

    injury_detail = str(feature_row.get("injury_detail") or "").strip()
    injury_type = str(feature_row.get("injury_type") or "").strip()
    feature_row["days_until_next_game"] = max(0.0, _to_float(model_inputs.get("days_until_next_game"), _to_float(feature_row.get("days_until_next_game"), 7.0)))
    feature_row["time_missed_already"] = max(0.0, _to_float(model_inputs.get("time_missed_already"), _to_float(feature_row.get("time_missed_already"), 0.0)))
    # Carry through dome/home flags if caller supplied them; otherwise preserve DB value
    if model_inputs.get("next_game_is_dome") is not None:
        feature_row["next_game_is_dome"] = _to_int(model_inputs.get("next_game_is_dome"), 0)
    if model_inputs.get("next_game_is_home") is not None:
        feature_row["next_game_is_home"] = _to_int(model_inputs.get("next_game_is_home"), 0)

    # Auto-fill game context from schedules when caller did not supply a stadium.
    current_stadium = str(feature_row.get("next_game_stadium") or "").strip()
    if not current_stadium or current_stadium in ("Unknown", ""):
        team = str(feature_row.get("team") or "").strip()
        season = _to_int(feature_row.get("season"), 0)
        week = _to_int(feature_row.get("week"), 0)
        if team and season and week:
            sched = _get_next_game_from_schedules(team, season, week)
            if sched:
                feature_row["next_game_stadium"] = sched.get("stadium", "Unknown")
                feature_row["next_game_is_dome"] = sched.get("is_dome", 0)
                feature_row["next_game_is_home"] = sched.get("is_home", 0)
                feature_row["next_game_opponent"] = sched.get("opponent", "Unknown")

    feature_row["injury_detail"] = injury_detail or injury_type or "Unknown"
    feature_row["injury_type"] = injury_type or "Unknown"
    feature_row["injury_key"] = f"{feature_row['injury_detail']}::{feature_row['injury_type']}"
    # Derive body region from the (possibly overridden) injury_detail
    if not feature_row.get("injury_body_region") or feature_row["injury_body_region"] == "Unknown":
        feature_row["injury_body_region"] = _injury_body_region(feature_row["injury_detail"])
    # Always (re)compute population features for the active injury_detail×position
    # so inference reflects the real league-wide stats, not whatever was in training cache.
    pop_avail, pop_ir, pop_action_ir = _fetch_pop_features(
        feature_row.get("injury_detail", "Unknown"),
        str(feature_row.get("position") or ""),
    )
    feature_row["pop_unavail_rate"]   = pop_avail
    feature_row["pop_ir_rate"]        = pop_ir
    feature_row["pop_action_ir_rate"] = pop_action_ir
    # Population avg return days (leakage-safe: historical population, not this injury)
    _ret_bm = _lookup_return_time_benchmark(
        feature_row.get("injury_detail", ""),
        feature_row.get("injury_type", ""),
    )
    _pop_avg_ret = float(_ret_bm.get("average_return_days") or 14.0)
    feature_row["pop_avg_return_days"] = _pop_avg_ret
    feature_row["days_remaining_estimate"] = max(0.0, _pop_avg_ret - float(feature_row.get("time_missed_already") or 0.0))
    # Team × body-region IR rate
    feature_row["team_region_ir_rate"] = _fetch_team_ir_features(
        str(feature_row.get("team") or ""),
        str(feature_row.get("injury_body_region") or "Unknown"),
    )
    # Player career IR history for same body region
    # resolved_player_id is not directly in scope here, so use gsis_id from the row.
    _gsis = str(feature_row.get("gsis_id") or "").strip()
    _body_rgn = str(feature_row.get("injury_body_region") or "Unknown")
    _career_ir, _same_rgn_ir = _fetch_player_ir_history(_gsis, _body_rgn)
    # Only overwrite if the row doesn't already have valid historical values
    # (training rows carry their own shifted cumulative counts from the DB).
    if not feature_row.get("player_career_ir_count"):
        feature_row["player_career_ir_count"] = float(_career_ir)
    if not feature_row.get("player_ir_same_region"):
        feature_row["player_ir_same_region"] = int(_same_rgn_ir)
    # Recompute is_ir_transaction based on current action value
    _IR_PLACEMENT_ACTIONS = {"placed on ir", "placed on pup", "placed on nfi", "out for season"}
    feature_row["is_ir_transaction"] = int(
        str(feature_row.get("action") or "none").strip().lower() in _IR_PLACEMENT_ACTIONS
    )
    return feature_row


def _build_fallback_feature_row(bundle, model_inputs: dict):
    defaults = dict(bundle.get("inference_defaults") or {})
    seasons = (bundle.get("dataset") or {}).get("seasons") or [2025, 2025]
    latest_season = _to_int(seasons[-1] if seasons else 2025, 2025)

    row = {
        "gsis_id": "",
        "full_name": model_inputs.get("player_name") or "Unknown",
        "team": defaults.get("team", "Unknown"),
        "position_group": defaults.get("position_group", "Unknown"),
        "season": _to_int(defaults.get("season"), latest_season),
        "week": _to_int(defaults.get("week"), 1),
        "source": defaults.get("source", "injury_report"),
        "next_game_stadium": str(defaults.get("next_game_stadium") or "Unknown"),
        "next_game_opponent": str(defaults.get("next_game_opponent") or "Unknown"),
        "next_game_is_home": _to_int(defaults.get("next_game_is_home"), 0),
        "next_game_is_dome": _to_int(defaults.get("next_game_is_dome"), 0),
        "days_until_next_game": _to_float(defaults.get("days_until_next_game"), 7.0),
        "time_missed_already": _to_float(defaults.get("time_missed_already"), 0.0),
        "days_since_prev": _to_float(defaults.get("days_since_prev"), 7.0),
        "new_injury_flag": _to_int(defaults.get("new_injury_flag"), 1),
        "on_ir": _to_int(defaults.get("on_ir"), 0),
        "prev_on_ir": _to_int(defaults.get("prev_on_ir"), 0),
        "years_of_experience": _to_float(defaults.get("years_of_experience"), 4.0),
        "age": _to_float(defaults.get("age"), 27.0),
        "prior_reports_count": _to_float(defaults.get("prior_reports_count"), 0.0),
        "prior_unavailable_count": _to_float(defaults.get("prior_unavailable_count"), 0.0),
        "injury_body_region": str(defaults.get("injury_body_region") or "Unknown"),
        "prev3_unavail_rate": _to_float(defaults.get("prev3_unavail_rate"), 0.0),
        "gap_since_prev_report": _to_float(defaults.get("gap_since_prev_report"), 1.0),
        "injury_repeat_count": _to_float(defaults.get("injury_repeat_count"), 0.0),
    }
    row["weeks_remaining"] = max(0, 18 - _to_int(row.get("week"), 1))

    row["game_type"] = str(defaults.get("game_type") or "REG")
    row["injury_type"] = str(defaults.get("injury_type") or "Unknown")
    row["injury_detail"] = str(defaults.get("injury_detail") or "Unknown")
    row["severity"] = str(defaults.get("severity") or "Unknown")
    row["position"] = str(defaults.get("position") or "Unknown")
    row["report_status"] = str(defaults.get("report_status") or "Questionable")
    row["practice_status"] = str(defaults.get("practice_status") or "Limited")
    # Fallback-path numeric features not set by the DB row
    row["injury_instance_no"]          = 1.0
    row["prior_same_injury_instances"] = 0.0
    row["new_instance_event"]          = 1.0
    row["weeks_since_first_same_injury"] = 0.0
    row["pop_unavail_rate"]            = 0.15
    row["pop_ir_rate"]                 = 0.05
    row["pop_action_ir_rate"]          = 0.05
    row["pop_avg_return_days"]         = 14.0
    row["days_remaining_estimate"]     = 14.0
    row["action"]                      = "none"
    row["is_ir_transaction"]           = 0
    row["team_region_ir_rate"]         = 0.08
    row["player_career_ir_count"]      = 0.0
    row["player_ir_same_region"]       = 0

    return _apply_inference_overrides(row, model_inputs)


def _log_prediction_to_db(result: dict, feature_row: dict, model_inputs: dict) -> None:
    """Persist a prediction to model_predictions_log for auditability.  Non-fatal on failure."""
    try:
        player = result.get("player") or {}
        next_game = result.get("next_game") or {}
        rt = result.get("return_timeline", {})
        rw = rt.get("return_within", {})
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_predictions_log (
                        player_gsis_id, player_name, player_team, player_position,
                        season, week,
                        injury_type, injury_detail, severity,
                        next_game_stadium, days_until_next_game, time_missed_already,
                        play_probability, play_classification,
                        return_within_1, return_within_2, return_within_3, return_within_4,
                        projected_games_out,
                        match_source, used_fallback_profile
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s, %s, %s, %s,
                        %s,
                        %s, %s
                    )
                    """,
                    (
                        player.get("gsis_id") or None,
                        player.get("full_name") or None,
                        player.get("team") or None,
                        player.get("position") or None,
                        player.get("season") or None,
                        player.get("week") or None,
                        player.get("injury_type") or None,
                        player.get("injury_detail") or None,
                        str(model_inputs.get("severity") or feature_row.get("severity") or "") or None,
                        str(model_inputs.get("next_game_stadium") or feature_row.get("next_game_stadium") or "") or None,
                        _to_float(model_inputs.get("days_until_next_game"), None),
                        _to_float(model_inputs.get("time_missed_already"), None),
                        next_game.get("play_probability"),
                        next_game.get("classification"),
                        rw.get("1_game"),
                        rw.get("2_games"),
                        rw.get("3_games"),
                        rw.get("4_games"),
                        rt.get("projected_games_out"),
                        (result.get("match") or {}).get("match_source"),
                        int(bool((result.get("match") or {}).get("used_fallback_profile"))),
                    ),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass  # Never let logging break a prediction response


def _generate_plain_english_explanation(result: dict, feature_row: dict) -> dict:
    """
    Translate model outputs + key feature values into plain English sentences
    suitable for the web UI. Returns a dict with:
      - summary: one-sentence headline
      - factors: list of up to 5 factor strings explaining the prediction
      - confidence_note: sentence about model confidence
      - stats_context: dict of numeric context values displayed in the stats section
    """
    player_name = (result.get("player") or {}).get("full_name") or "The player"
    play_prob   = (result.get("next_game") or {}).get("play_probability") or 0.0
    label       = (result.get("next_game") or {}).get("classification") or ""
    rw          = (result.get("return_timeline") or {}).get("return_within") or {}
    proj_out    = (result.get("return_timeline") or {}).get("projected_games_out") or 0.0

    injury_detail   = str(feature_row.get("injury_detail") or "Unknown")
    injury_region   = str(feature_row.get("injury_body_region") or "Unknown")
    severity        = str(feature_row.get("severity") or "").strip()
    report_status   = str(feature_row.get("report_status") or "").strip()
    practice_status = str(feature_row.get("practice_status") or "").strip()
    time_missed     = float(feature_row.get("time_missed_already") or 0)
    on_ir           = _to_int(feature_row.get("on_ir"), 0)
    is_home         = _to_int(feature_row.get("next_game_is_home"), 0)
    is_dome         = _to_int(feature_row.get("next_game_is_dome"), 0)
    days_until      = float(feature_row.get("days_until_next_game") or 7)
    opponent        = str(feature_row.get("next_game_opponent") or "their opponent")
    repeat_count    = _to_int(feature_row.get("injury_repeat_count"), 0)
    weeks_missed    = round(time_missed / 7) if time_missed > 0 else 0

    # ── Headline ──────────────────────────────────────────────────────────────
    pct = round(play_prob * 100)
    if label == "Likely To Play":
        summary = (
            f"{player_name} is likely to play ({pct}% probability). "
            f"{injury_region} injuries at this severity return in the next game {pct}% of the time historically."
        )
    else:
        games_word = "game" if round(proj_out) == 1 else "games"
        summary = (
            f"{player_name} is likely out ({100 - pct}% probability of missing). "
            f"Similar {injury_region} injuries project {round(proj_out):.0f} {games_word} missed on average."
        )

    # ── Factor bullets ────────────────────────────────────────────────────────
    factors = []

    # Practice/report status
    if practice_status:
        status_map = {
            "DNP": "Did not practice — strong indicator of absence",
            "Limited": "Limited in practice — uncertainty remains",
            "Full": "Full practice participation — positive return signal",
        }
        factors.append(status_map.get(practice_status, f"Practice status: {practice_status}"))
    elif report_status:
        rs_map = {
            "Out": "Listed as Out on the official injury report",
            "Doubtful": "Listed as Doubtful — historically misses ~75% of games",
            "Questionable": "Listed as Questionable — roughly 50/50 historically",
            "Probable": "Listed as Probable — historically plays ~85% of the time",
        }
        factors.append(rs_map.get(report_status, f"Report status: {report_status}"))

    # Severity
    if severity and severity.lower() not in ("unknown", ""):
        factors.append(f"Injury severity classified as '{severity}' for this {injury_detail} injury")

    # Time missed
    if on_ir:
        factors.append("Player is on Injured Reserve — return requires activation")
    elif weeks_missed >= 3:
        factors.append(f"Already missed approximately {weeks_missed} weeks, which affects return probability")
    elif weeks_missed == 1:
        factors.append("Missed one week so far — early in the injury timeline")

    # Repeat injury
    if repeat_count >= 2:
        factors.append(
            f"This is a repeat {injury_region} injury (recurrence #{repeat_count}) — "
            "prior re-injury history detected"
        )

    # Game context
    venue_parts = []
    if is_home:
        venue_parts.append("home game")
    else:
        venue_parts.append(f"away game vs. {opponent}")
    if is_dome:
        venue_parts.append("indoor/dome venue (controlled surface, no weather factor)")
    if days_until < 5:
        venue_parts.append(f"only {round(days_until)} days until kickoff — limited recovery time")
    if venue_parts:
        factors.append(". ".join(p.capitalize() for p in venue_parts))

    # Return timeline
    r1 = round((rw.get("1_game") or 0.0) * 100)
    r2 = round((rw.get("2_games") or 0.0) * 100)
    if r1 >= 60:
        factors.append(f"{r1}% historical return rate by next game for comparable injuries")
    elif r2 >= 60:
        factors.append(f"Return is more likely by game 2 ({r2}%) than game 1 ({r1}%)")

    # Cap at 5 most relevant factors
    factors = factors[:5]

    # ── Confidence note ───────────────────────────────────────────────────────
    used_fallback = (result.get("match") or {}).get("used_fallback_profile", False)
    match_src     = (result.get("match") or {}).get("match_source", "")
    if used_fallback:
        confidence_note = (
            "Prediction is based on league-wide injury averages for this injury type — "
            "no prior injury history was found for this player."
        )
    elif match_src == "matched_injury_history":
        confidence_note = (
            "Prediction is personalised using this player's injury history and snap-count return data."
        )
    else:
        confidence_note = "Prediction uses player profile data with league-average injury baselines."

    # ── Stats context (for the statistics section of the UI) ──────────────────
    stats_context = {
        "play_probability_pct": pct,
        "return_within_1_game_pct": round((rw.get("1_game") or 0.0) * 100),
        "return_within_2_games_pct": round((rw.get("2_games") or 0.0) * 100),
        "return_within_3_games_pct": round((rw.get("3_games") or 0.0) * 100),
        "return_within_4_games_pct": round((rw.get("4_games") or 0.0) * 100),
        "projected_games_out": round(proj_out, 1),
        "weeks_already_missed": weeks_missed,
        "days_until_next_game": round(days_until, 1),
        "is_home_game": bool(is_home),
        "is_dome_venue": bool(is_dome),
        "injury_body_region": injury_region,
        "repeat_injury_count": repeat_count,
    }

    return {
        "summary": summary,
        "factors": factors,
        "confidence_note": confidence_note,
        "stats_context": stats_context,
    }


def _lookup_return_time_benchmark(injury_detail: str, injury_type: str) -> dict:
    """
    Return benchmark statistics from persisted labels table for the selected injury.
    Prefers injury_detail-level matches and falls back to injury_type-level matches.
    """
    detail = (injury_detail or "").strip()
    inj_type = (injury_type or "").strip()

    if not detail and not inj_type:
        return {
            "scope": "none",
            "label": "No injury benchmark available",
            "sample_size": 0,
            "average_return_days": None,
        }

    where_common = """
        l.return_days IS NOT NULL
        AND l.return_days BETWEEN 0 AND 180
    """

    # Most specific benchmark: injury_detail
    if detail:
        detail_row = one(
            f"""
            SELECT
                COUNT(*) AS n,
                AVG(l.return_days) AS avg_days
            FROM injury_return_labels l
            JOIN events e ON e.injuryId = l.injury_id
            WHERE {where_common}
              AND LOWER(TRIM(COALESCE(e.injury_detail, ''))) = LOWER(TRIM(%s))
            """,
            (detail,),
        ) or {}

        n_detail = int(detail_row.get("n") or 0)
        if n_detail >= 10:
            avg_days = detail_row.get("avg_days")
            return {
                "scope": "injury_detail",
                "label": f"Average return for {detail}",
                "sample_size": n_detail,
                "average_return_days": round(float(avg_days), 1) if avg_days is not None else None,
            }

    # Fallback benchmark: injury_type
    if inj_type:
        type_row = one(
            f"""
            SELECT
                COUNT(*) AS n,
                AVG(l.return_days) AS avg_days
            FROM injury_return_labels l
            JOIN events e ON e.injuryId = l.injury_id
            WHERE {where_common}
              AND LOWER(TRIM(COALESCE(e.injury_type, ''))) = LOWER(TRIM(%s))
            """,
            (inj_type,),
        ) or {}

        n_type = int(type_row.get("n") or 0)
        if n_type > 0:
            avg_days = type_row.get("avg_days")
            return {
                "scope": "injury_type",
                "label": f"Average return for {inj_type} injuries",
                "sample_size": n_type,
                "average_return_days": round(float(avg_days), 1) if avg_days is not None else None,
            }

    return {
        "scope": "none",
        "label": "No injury benchmark available",
        "sample_size": 0,
        "average_return_days": None,
    }


def _predict_player_outcomes(player_id: str | None = None, model_inputs: dict | None = None):
    import pandas as pd

    clean_inputs = _sanitize_model_inputs(model_inputs)

    # Check the in-memory cache FIRST — never trigger a training run from a predict
    # request. If the bundle isn't ready, return 503 immediately instead of blocking
    # for minutes while the warmup thread trains in the background.
    if _model_bundle_cache is None or not _model_bundle_cache.get("ready"):
        return {
            "ready": False,
            "message": "Model is still loading, please retry shortly.",
        }

    bundle = _model_bundle_cache

    if not bundle.get("ready"):
        return {
            "ready": False,
            "message": bundle.get("message", "Model pipeline not ready."),
        }

    resolved_player = None
    latest_row = None

    resolved_player_id = str(player_id or "").strip()
    if not resolved_player_id and clean_inputs.get("player_name"):
        resolved_player = _lookup_player_by_name(clean_inputs.get("player_name"))
        if resolved_player:
            resolved_player_id = str(resolved_player.get("gsis_id") or "").strip()

    if resolved_player_id:
        latest_row, _ = _prepare_player_feature_row(resolved_player_id)

    used_fallback_profile = latest_row is None
    if latest_row is None:
        latest_row = _build_fallback_feature_row(bundle, clean_inputs)
    else:
        latest_row = _apply_inference_overrides(latest_row, clean_inputs)
        if clean_inputs.get("player_name") and not str(latest_row.get("full_name") or "").strip():
            latest_row["full_name"] = clean_inputs.get("player_name")

    availability_model = bundle.get("availability_model", {})
    duration_models = bundle.get("duration_models", {})
    ir_placement_model = bundle.get("ir_placement_model", {})
    feature_cols = bundle.get("feature_cols", [])

    x = pd.DataFrame([latest_row])[feature_cols]

    avail_prob = float(availability_model["model"].predict_proba(x)[:, 1][0])

    # ── Domain-knowledge hard caps applied AFTER model output ─────────────────
    # The training data under-represents season-ending injuries and definitive
    # "OUT" designations (players placed on IR disappear from injury reports,
    # so the model never saw their "did not play" snap-count outcomes).
    # These caps enforce known football reality without retraining.
    _report   = str(latest_row.get("report_status")   or clean_inputs.get("report_status")   or "").strip().upper()
    _practice = str(latest_row.get("practice_status") or clean_inputs.get("practice_status") or "").strip().upper()
    _severity = str(latest_row.get("severity")        or clean_inputs.get("severity")         or "").strip().upper()
    _detail   = str(latest_row.get("injury_detail")   or clean_inputs.get("injury_detail")    or "").strip().upper()

    # Season-ending / structural injuries — player cannot play regardless of other signals
    _SEASON_ENDING_KEYWORDS = {
        "ACL", "TORN ACL", "PCL", "TORN PCL", "ACHILLES", "TORN ACHILLES",
        "PATELLAR TENDON", "TORN PATELLAR", "LISFRANC",
        "FRACTURE", "BROKEN", "SURGERY", "TORN", "RUPTURED",
        "SEASON-ENDING", "SEASON ENDING",
    }
    _is_season_ending = any(kw in _detail for kw in _SEASON_ENDING_KEYWORDS) or any(kw in _severity for kw in _SEASON_ENDING_KEYWORDS)

    # Time sensitivity: if injury just occurred and is season-ending, even harder cap
    _days = _to_float(latest_row.get("days_until_next_game"), 7.0)
    _missed = _to_float(latest_row.get("time_missed_already"), 0.0)

    if _report == "OUT":
        # Official "Out" designation means the team has ruled the player out
        avail_prob = min(avail_prob, 0.04)
    elif _is_season_ending and _missed < 7:
        # Season-ending injury that just occurred — essentially impossible to play
        avail_prob = min(avail_prob, 0.02)
    elif _is_season_ending:
        # Season-ending injury, already missed time — still very unlikely
        avail_prob = min(avail_prob, 0.08)
    elif _report == "DOUBTFUL":
        # "Doubtful" should be pessimistic, but still reflect user-input context.
        # We avoid a single flat cap because it made minor and serious cases collapse
        # to the same probability.
        _is_minor_input = _severity in {"MILD", "SPRAIN", "STRAIN", "MINOR"}
        _is_full_or_limited = _practice in {"FULL", "LIMITED"}
        if _is_season_ending:
            avail_prob = min(avail_prob, 0.08)
        elif _is_minor_input and _is_full_or_limited and _missed >= 2:
            # Mild issue, already missed time, and practicing: can be closer to a coin flip.
            avail_prob = min(avail_prob, 0.55)
        elif _is_minor_input and _is_full_or_limited:
            avail_prob = min(avail_prob, 0.45)
        elif _is_full_or_limited and _missed >= 2:
            avail_prob = min(avail_prob, 0.40)
        else:
            avail_prob = min(avail_prob, 0.25)
    elif _report in ("QUESTIONABLE", "LIMITED"):
        # Questionable with a severe structural injury: cap lower
        _SEVERE_STRUCTURAL = {"ACL", "PCL", "ACHILLES", "LISFRANC", "FRACTURE", "BROKEN", "PATELLAR TENDON"}
        if any(kw in _detail for kw in _SEVERE_STRUCTURAL):
            avail_prob = min(avail_prob, 0.30)

    avail_label = "Likely To Play" if avail_prob >= 0.5 else "Likely Out"

    # Game-1 cumulative = avail_prob (the canonical availability model).
    # Horizons 2-4 come from their respective duration sub-models.
    # Using avail_prob for game_1 ensures "Play Probability" and
    # "Return within 1 game" are always the same number.
    cumulative_return = [avail_prob]
    for horizon in (2, 3, 4):
        model_obj = duration_models.get(horizon, {})
        if not model_obj.get("ready"):
            cumulative_return.append(cumulative_return[-1])  # carry last value forward
            continue
        cumulative_return.append(float(model_obj["model"].predict_proba(x)[:, 1][0]))

    # ── Apply the same domain-knowledge caps to the return-timeline sub-models ──
    # The duration models have the same survivor-bias problem: IR players are
    # absent from training data, so they wildly over-estimate return probability
    # for season-ending injuries across all horizons, not just game 1.
    if _is_season_ending and _missed < 7:
        # Just occurred — essentially no one returns within 4 games from a season-ender
        _timeline_caps = (0.02, 0.05, 0.08, 0.10)
    elif _is_season_ending:
        # Already missed time but still a season-ender
        _timeline_caps = (0.08, 0.12, 0.15, 0.18)
    elif _report == "OUT":
        # Officially ruled out — most return within the season but not in 1-2 games
        _timeline_caps = (0.04, 0.55, 0.75, 0.85)
    elif _report == "DOUBTFUL":
        _timeline_caps = (0.25, 0.65, 0.80, 0.88)
    else:
        _timeline_caps = (1.0, 1.0, 1.0, 1.0)  # no cap for other statuses

    cumulative_return = [
        min(raw, cap)
        for raw, cap in zip(cumulative_return, _timeline_caps)
    ]

    # Enforce monotone non-decreasing: P(return within N) >= P(return within N-1)
    monotonic = []
    running = 0.0
    for val in cumulative_return:
        running = max(running, min(1.0, max(0.0, val)))
        monotonic.append(running)

    per_game = {
        "game_1": monotonic[0],
        "game_2": max(0.0, monotonic[1] - monotonic[0]),
        "game_3": max(0.0, monotonic[2] - monotonic[1]),
        "game_4": max(0.0, monotonic[3] - monotonic[2]),
        "game_5_plus": max(0.0, 1.0 - monotonic[3]),
    }

    expected_games_out = (
        per_game["game_1"] * 1
        + per_game["game_2"] * 2
        + per_game["game_3"] * 3
        + per_game["game_4"] * 4
        + per_game["game_5_plus"] * 5
    )

    ir_prob = 0.0
    ir_band = "Low"
    if ir_placement_model.get("ready"):
        ir_prob = float(ir_placement_model["model"].predict_proba(x)[:, 1][0])
        # Input-aware IR calibration: preserve model signal but enforce obvious
        # directional behavior for severe vs. mild scenarios.
        _is_minor_input = _severity in {"MILD", "SPRAIN", "STRAIN", "MINOR"}
        _is_full_or_limited = _practice in {"FULL", "LIMITED"}
        if _is_season_ending and _missed < 2:
            ir_prob = max(ir_prob, 0.45)
        elif _is_season_ending:
            ir_prob = max(ir_prob, 0.35)
        elif _report == "OUT":
            ir_prob = max(ir_prob, 0.30)
        elif _report == "DOUBTFUL" and _practice.startswith("DID NOT"):
            ir_prob = max(ir_prob, 0.22)

        if _is_minor_input and _is_full_or_limited and _missed >= 2:
            ir_prob = min(ir_prob, 0.12)
        elif _is_minor_input and _is_full_or_limited:
            ir_prob = min(ir_prob, 0.18)

        ir_prob = max(0.0, min(1.0, ir_prob))
        ir_band = "High" if ir_prob >= 0.30 else "Moderate" if ir_prob >= 0.12 else "Low"

    match_source = "fallback_profile"
    if resolved_player_id and not used_fallback_profile:
        match_source = "matched_injury_history"
    elif resolved_player_id and used_fallback_profile:
        match_source = "matched_player_no_injury_history"

    display_name = str(latest_row.get("full_name") or "").strip() or clean_inputs.get("player_name") or "Unknown"
    return_benchmark = _lookup_return_time_benchmark(
        str(latest_row.get("injury_detail") or clean_inputs.get("injury_detail") or ""),
        str(latest_row.get("injury_type") or clean_inputs.get("injury_type") or ""),
    )

    result = {
        "ready": True,
        "match": {
            "input_player_name": clean_inputs.get("player_name") or "",
            "resolved_gsis_id": resolved_player_id,
            "used_fallback_profile": bool(used_fallback_profile),
            "match_source": match_source,
        },
        "player": {
            "gsis_id": str(latest_row.get("gsis_id") or resolved_player_id),
            "full_name": display_name,
            "team": str(latest_row.get("team") or "Unknown"),
            "position": str(latest_row.get("position") or "Unknown"),
            "season": _to_int(latest_row.get("season")),
            "week": _to_int(latest_row.get("week")),
            "injury_type": str(latest_row.get("injury_type") or "Unknown"),
            "injury_detail": str(latest_row.get("injury_detail") or "Unknown"),
            "report_status": str(latest_row.get("report_status") or "Unknown"),
        },
        "next_game": {
            "play_probability": round(avail_prob, 4),
            "classification": avail_label,
        },
        "return_timeline": {
            "return_detection": {
                "method": "snap_count_ground_truth_or_report_transition",
                "notes": (
                    "When snap_counts data exists for the next game, return is determined by "
                    "actual game participation (did_play). Otherwise inferred from injury-report "
                    "status transitions and IR flags."
                ),
            },
            "return_within": {
                "1_game": round(monotonic[0], 4),
                "2_games": round(monotonic[1], 4),
                "3_games": round(monotonic[2], 4),
                "4_games": round(monotonic[3], 4),
            },
            "return_on_each_upcoming_game": {
                "game_1": round(per_game["game_1"], 4),
                "game_2": round(per_game["game_2"], 4),
                "game_3": round(per_game["game_3"], 4),
                "game_4": round(per_game["game_4"], 4),
                "game_5_plus": round(per_game["game_5_plus"], 4),
            },
            "projected_games_out": round(float(expected_games_out), 2),
        },
        "ir_placement": {
            "probability": round(ir_prob, 4),
            "risk_band": ir_band,
        },
        "return_benchmark": return_benchmark,
    }

    result["explanation"] = _generate_plain_english_explanation(result, latest_row)
    _log_prediction_to_db(result, latest_row, clean_inputs)
    return result


def search_players_by_prefix(prefix: str, limit: int = 12):
    token = (prefix or "").strip().lower()
    if len(token) < 1:
        return []

    capped_limit = max(1, min(int(limit or 12), 30))
    like = f"{token}%"

    rows = many(
        """
        SELECT
            p.gsis_id,
            COALESCE(NULLIF(TRIM(p.display_name), ''), NULLIF(TRIM(p.football_name), ''), TRIM(CONCAT_WS(' ', p.first_name, p.last_name)), p.short_name, p.gsis_id) AS full_name,
            COALESCE(p.first_name, '') AS first_name,
            COALESCE(p.last_name, '') AS last_name,
            COALESCE(p.position, '') AS position,
            COALESCE(p.latest_team, '') AS team,
            CASE
                WHEN EXISTS (
                    SELECT 1
                    FROM injuryReport ir
                    WHERE ir.gsis_id IS NOT NULL
                      AND TRIM(ir.gsis_id) <> ''
                      AND LOWER(TRIM(ir.gsis_id)) <> 'na'
                      AND TRIM(ir.gsis_id) = TRIM(p.gsis_id)
                    LIMIT 1
                )
                THEN 1
                ELSE 0
            END AS has_injury_history
        FROM players p
        WHERE p.gsis_id IS NOT NULL
          AND TRIM(p.gsis_id) <> ''
          AND (
                LOWER(COALESCE(p.first_name, '')) LIKE %s
             OR LOWER(COALESCE(p.last_name, '')) LIKE %s
             OR LOWER(COALESCE(p.display_name, '')) LIKE %s
             OR LOWER(COALESCE(p.football_name, '')) LIKE %s
          )
        ORDER BY
            CASE
                WHEN LOWER(COALESCE(p.last_name, '')) LIKE %s THEN 0
                WHEN LOWER(COALESCE(p.first_name, '')) LIKE %s THEN 1
                WHEN LOWER(COALESCE(p.display_name, '')) LIKE %s THEN 2
                WHEN LOWER(COALESCE(p.football_name, '')) LIKE %s THEN 3
                ELSE 4
            END,
            COALESCE(p.last_name, ''),
            COALESCE(p.first_name, ''),
            COALESCE(p.display_name, ''),
            p.gsis_id
        LIMIT %s
        """,
        (like, like, like, like, like, like, like, like, capped_limit),
    ) or []

    return [
        {
            "gsis_id": r.get("gsis_id"),
            "full_name": r.get("full_name") or "Unknown",
            "has_injury_history": bool(r.get("has_injury_history")),
        }
        for r in rows
    ]


@lru_cache(maxsize=1)
def build_model_input_options_payload():
    """
    Query the database directly for dropdown values — does NOT depend on model
    training succeeding.  Falls back to empty lists if DB is unreachable.
    """
    def _distinct(sql):
        try:
            rows = many(sql)
            # DictCursor returns dicts; grab the first (and only) column value
            return sorted(
                str(v).strip()
                for r in rows
                for v in [next(iter(r.values()))]
                if v is not None and str(v).strip() != ""
            )
        except Exception:
            return []

    return {
        "injury_details": _distinct(
            "SELECT DISTINCT injury_detail FROM injuryReport "
            "WHERE injury_detail IS NOT NULL AND TRIM(injury_detail) <> '' "
            "ORDER BY injury_detail"
        ),
        "severity_levels": _distinct(
            "SELECT DISTINCT severity FROM injuryReport "
            "WHERE severity IS NOT NULL AND TRIM(severity) <> '' "
            "ORDER BY severity"
        ),
        "stadiums": _distinct(
            "SELECT DISTINCT stadium FROM pbp "
            "WHERE stadium IS NOT NULL AND TRIM(stadium) <> '' AND playId = 1 "
            "ORDER BY stadium"
        ),
        "injury_types": _distinct(
            "SELECT DISTINCT injury_type FROM injuryReport "
            "WHERE injury_type IS NOT NULL AND TRIM(injury_type) <> '' "
            "ORDER BY injury_type"
        ),
        "positions": _distinct(
            "SELECT DISTINCT position FROM injuryReport "
            "WHERE position IS NOT NULL AND TRIM(position) <> '' "
            "ORDER BY position"
        ),
        "report_statuses": _distinct(
            "SELECT DISTINCT report_status FROM injuryReport "
            "WHERE report_status IS NOT NULL AND TRIM(report_status) <> '' "
            "ORDER BY report_status"
        ),
        "practice_statuses": _distinct(
            "SELECT DISTINCT practice_status FROM injuryReport "
            "WHERE practice_status IS NOT NULL AND TRIM(practice_status) <> '' "
            "ORDER BY practice_status"
        ),
        "game_types": _distinct(
            "SELECT DISTINCT game_type FROM injuryReport "
            "WHERE game_type IS NOT NULL AND TRIM(game_type) <> '' "
            "ORDER BY game_type"
        ),
    }


def build_injury_player_linkage_payload():
    counts = one(
        """
        SELECT
            COUNT(*) AS total_injury_rows,
            SUM(
                CASE
                    WHEN ir.gsis_id IS NULL
                      OR TRIM(ir.gsis_id) = ''
                      OR LOWER(TRIM(ir.gsis_id)) = 'na'
                    THEN 1
                    ELSE 0
                END
            ) AS missing_or_invalid_gsis,
            SUM(
                CASE
                    WHEN ir.gsis_id IS NOT NULL
                      AND TRIM(ir.gsis_id) <> ''
                      AND LOWER(TRIM(ir.gsis_id)) <> 'na'
                      AND p.gsis_id IS NOT NULL
                    THEN 1
                    ELSE 0
                END
            ) AS linked_to_players,
            SUM(
                CASE
                    WHEN ir.gsis_id IS NOT NULL
                      AND TRIM(ir.gsis_id) <> ''
                      AND LOWER(TRIM(ir.gsis_id)) <> 'na'
                      AND p.gsis_id IS NULL
                    THEN 1
                    ELSE 0
                END
            ) AS unmatched_valid_gsis
        FROM injuryReport ir
        LEFT JOIN players p ON TRIM(p.gsis_id) = TRIM(ir.gsis_id)
        """
    ) or {}

    duplicate_ids = one(
        """
        SELECT COUNT(*) AS duplicate_player_ids
        FROM (
            SELECT gsis_id
            FROM players
            WHERE gsis_id IS NOT NULL AND TRIM(gsis_id) <> ''
            GROUP BY gsis_id
            HAVING COUNT(*) > 1
        ) d
        """
    ) or {}

    recovered_by_name = one(
        """
        SELECT COUNT(*) AS matched_missing_by_name
        FROM injuryReport ir
        WHERE (
            ir.gsis_id IS NULL
            OR TRIM(ir.gsis_id) = ''
            OR LOWER(TRIM(ir.gsis_id)) = 'na'
        )
          AND ir.full_name IS NOT NULL
          AND TRIM(ir.full_name) <> ''
          AND LOWER(
                REPLACE(
                    REPLACE(
                        REPLACE(
                            REPLACE(
                                REPLACE(TRIM(ir.full_name), ' ', ''),
                            '.', ''),
                        CHAR(39), ''),
                    '-', ''),
                ',', '')
              ) IN (
                SELECT normalized_name
                FROM (
                    SELECT DISTINCT
                        LOWER(
                            REPLACE(
                                REPLACE(
                                    REPLACE(
                                        REPLACE(
                                            REPLACE(TRIM(name_value), ' ', ''),
                                        '.', ''),
                                    CHAR(39), ''),
                                '-', ''),
                            ',', '')
                        ) AS normalized_name
                    FROM (
                        SELECT display_name AS name_value FROM players
                        UNION ALL
                        SELECT football_name AS name_value FROM players
                        UNION ALL
                        SELECT CONCAT_WS(' ', first_name, last_name) AS name_value FROM players
                    ) names
                    WHERE name_value IS NOT NULL
                      AND TRIM(name_value) <> ''
                ) normalized_players
            )
        """
    ) or {}

    total_rows = int(counts.get("total_injury_rows") or 0)
    missing_or_invalid = int(counts.get("missing_or_invalid_gsis") or 0)
    unmatched_valid = int(counts.get("unmatched_valid_gsis") or 0)
    linked = int(counts.get("linked_to_players") or 0)
    unlinked_total = missing_or_invalid + unmatched_valid
    duplicate_player_ids = int(duplicate_ids.get("duplicate_player_ids") or 0)
    matched_missing_by_name = int(recovered_by_name.get("matched_missing_by_name") or 0)
    unresolved_after_name_match = max(0, unlinked_total - matched_missing_by_name)

    return {
        "total_injury_rows": total_rows,
        "linked_to_players": linked,
        "unable_to_link_definitively": unlinked_total,
        "breakdown": {
            "missing_or_invalid_gsis": missing_or_invalid,
            "unmatched_valid_gsis": unmatched_valid,
        },
        "name_match_backfill": {
            "matched_missing_by_name": matched_missing_by_name,
            "remaining_unlinked_after_name_match": unresolved_after_name_match,
        },
        "linkage_rate_percent": round((linked / total_rows) * 100, 2) if total_rows else 0.0,
        "player_table_duplicate_gsis_ids": duplicate_player_ids,
    }


def build_undocumented_rows(table_name: str):
    columns = many(
        """
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        ORDER BY ORDINAL_POSITION
        """,
        (config.db, table_name),
    ) or []

    descriptions = get_combined_descriptions_for_table(table_name)
    missing_rows = []
    for col in columns:
        if not (get_description_for_column(descriptions, col.get("COLUMN_NAME")) or "").strip():
            missing_rows.append(
                {
                    "table": table_name,
                    "column_name": col.get("COLUMN_NAME"),
                    "data_type": col.get("DATA_TYPE"),
                    "is_nullable": col.get("IS_NULLABLE"),
                    "column_key": col.get("COLUMN_KEY"),
                    "description": "",
                }
            )
    return missing_rows


@app.get("/")
def landing_page():
    """Render the project landing page."""
    return render_template("landing.html")


@app.get("/statistics")
def statistics_page():
    """Render the main statistics/analysis page."""
    return render_template("statistics.html")


@app.get("/dashboard")
def dashboard_page():
    """Redirect /dashboard to the statistics page (legacy URL alias)."""
    return redirect(url_for("statistics_page"))


@app.route("/data-statistics", methods=["GET", "POST"])
def data_statistics_page():
    """
    Render the data-dictionary viewer page.

    GET:  Display the selected table's columns with their current descriptions.
    POST: Save user-edited field descriptions for the selected table to the
          custom JSON overrides file, then re-render the page.

    Form field names follow the pattern 'desc__<column_name>' so we can
    distinguish them from the other form controls (e.g. 'table' selector).
    Only columns that actually exist in INFORMATION_SCHEMA are accepted to
    prevent arbitrary key injection into the JSON file.
    """
    try:
        if request.method == "POST":
            selected_table = request.form.get("table")
            if selected_table:
                # Fetch the real column list from the DB so we can whitelist input keys
                valid_columns = many(
                    """
                    SELECT COLUMN_NAME
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                    """,
                    (config.db, selected_table),
                )
                valid_column_names = {row["COLUMN_NAME"] for row in valid_columns}

                all_custom = dict(load_custom_field_descriptions())
                selected_custom = dict(all_custom.get(selected_table, {}))

                # Process each description field from the submitted form
                for key, value in request.form.items():
                    if not key.startswith("desc__"):
                        continue  # skip non-description form fields
                    column_name = key.replace("desc__", "", 1)
                    if column_name not in valid_column_names:
                        continue  # silently ignore unknown column names for security
                    text = (value or "").strip()
                    if text:
                        selected_custom[column_name] = text  # add or update description
                    else:
                        selected_custom.pop(column_name, None)  # remove blank descriptions

                if selected_custom:
                    all_custom[selected_table] = selected_custom
                else:
                    all_custom.pop(selected_table, None)

                save_custom_field_descriptions(all_custom)

                return redirect(url_for("data_statistics_page", table=selected_table, saved="1"))

        quality_data = build_quality_payload()
        selected_table = request.args.get("table")
        catalog = build_table_catalog_payload(selected_table)
        saved = request.args.get("saved") == "1"
        return render_template(
            "data_statistics.html",
            quality=quality_data,
            catalog=catalog,
            saved=saved,
        )
    except Exception as exc:
        return render_template("data_statistics.html", error=str(exc), quality=None, catalog=None), 500


@app.get("/data-statistics/export-undocumented")
def export_undocumented_descriptions():
    try:
        table_name = request.args.get("table")
        if not table_name:
            return jsonify({"success": False, "message": "Missing table parameter"}), 400

        catalog = build_table_catalog_payload(table_name)
        selected_table = catalog.get("selected_table")
        if not selected_table:
            return jsonify({"success": False, "message": "Invalid table"}), 400

        rows = build_undocumented_rows(selected_table)
        buffer = StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=["table", "column_name", "data_type", "is_nullable", "column_key", "description"],
        )
        writer.writeheader()
        writer.writerows(rows)

        csv_output = buffer.getvalue()
        buffer.close()

        return Response(
            csv_output,
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=undocumented_{selected_table}.csv"
            },
        )
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


@app.get("/endpoints")
def endpoints_page():
    return render_template("endpoints.html")


@app.get("/prediction-model")
def prediction_model_page():
    return render_template("prediction_model.html")


@app.get("/api/model/overview")
def model_overview():
    try:
        refresh_requested = request.args.get("refresh") == "1"
        refresh_started = False
        refresh_status_code = 200

        if refresh_requested:
            refresh_started = _start_model_refresh_if_needed()
            refresh_status_code = 202

        # If the in-memory bundle is not ready yet, don't block — return 503 and
        # kick off a background load so the client can poll.
        if _model_bundle_cache is None or not _model_bundle_cache.get("ready"):
            if not _model_refresh_state.get("in_progress"):
                _start_model_refresh_if_needed()
            refresh_snap = _snapshot_model_refresh_state()
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Model is loading, please retry shortly.",
                        "responseObject": {
                            "ready": False,
                            "refresh": {
                                **refresh_snap,
                                "requested": refresh_requested,
                                "started": refresh_started,
                            },
                        },
                        "statusCode": 503,
                    }
                ),
                503,
            )

        payload = build_model_overview_payload()
        payload["refresh"] = {
            **_snapshot_model_refresh_state(),
            "requested": refresh_requested,
            "started": refresh_started,
        }
        return (
            jsonify(
                {
                    "success": True,
                    "message": (
                        "Model refresh started"
                        if refresh_started
                        else "Model refresh already in progress"
                        if refresh_requested and payload["refresh"].get("in_progress")
                        else "Model overview fetched"
                    ),
                    "responseObject": payload,
                    "statusCode": refresh_status_code,
                }
            ),
            refresh_status_code,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Failed to build model overview: {str(exc)}",
                    "responseObject": None,
                    "statusCode": 500,
                }
            ),
            500,
        )


@app.get("/api/model/input-options")
def model_input_options():
    try:
        if request.args.get("refresh") == "1":
            build_model_input_options_payload.cache_clear()
            global _model_bundle_cache; _model_bundle_cache = None
            try:
                if os.path.exists(_MODEL_CACHE_PATH):
                    os.remove(_MODEL_CACHE_PATH)
            except Exception:
                pass

        payload = build_model_input_options_payload()
        return (
            jsonify(
                {
                    "success": True,
                    "message": "Model input options fetched",
                    "responseObject": payload,
                    "statusCode": 200,
                }
            ),
            200,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Failed to fetch model input options: {str(exc)}",
                    "responseObject": None,
                    "statusCode": 500,
                }
            ),
            500,
        )


@app.get("/api/model/player-search")
def model_player_search():
    """
    Autocomplete player name search for the prediction form.

    Query parameters:
      q     - search prefix (at least 1 character required)
      limit - max number of results to return (default 12)

    Returns a list of player objects with gsis_id, name, position, and team
    so the frontend can populate a name-lookup dropdown.
    """
    try:
        query = (request.args.get("q") or "").strip()
        limit = request.args.get("limit", default=12, type=int)

        if len(query) < 1:
            return (
                jsonify(
                    {
                        "success": True,
                        "message": "Type at least 1 character",
                        "responseObject": {"players": []},
                        "statusCode": 200,
                    }
                ),
                200,
            )

        players = search_players_by_prefix(query, limit=limit)
        return (
            jsonify(
                {
                    "success": True,
                    "message": f"Found {len(players)} player matches",
                    "responseObject": {
                        "query": query,
                        "players": players,
                    },
                    "statusCode": 200,
                }
            ),
            200,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Failed to search players: {str(exc)}",
                    "responseObject": None,
                    "statusCode": 500,
                }
            ),
            500,
        )


@app.get("/api/model/linkage-summary")
def model_linkage_summary():
    """
    Return a summary of how many injury report rows were successfully linked
    to a player record in the players table via gsis_id or name matching.

    Used on the prediction model page to show data quality context.
    """
    try:
        payload = build_injury_player_linkage_payload()
        return (
            jsonify(
                {
                    "success": True,
                    "message": "Injury-to-player linkage summary fetched",
                    "responseObject": payload,
                    "statusCode": 200,
                }
            ),
            200,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Failed to compute linkage summary: {str(exc)}",
                    "responseObject": None,
                    "statusCode": 500,
                }
            ),
            500,
        )


@app.post("/api/model/predict")
def model_predict_player():
    """
    Generate an injury availability prediction for a single player.

    Accepts a JSON body with at minimum one of:
      - gsis_id       (preferred — enables full per-player feature lookup)
      - player_name   (triggers a name-based DB lookup to find the gsis_id)

    Optional overrides (all will override the DB-looked-up values if provided):
      position, injury_type, injury_detail, report_status, practice_status,
      severity, next_game_stadium, days_until_next_game, time_missed_already,
      game_type

    Returns a full prediction payload including:
      - play_probability (0–1 float)
      - projected_games_out
      - return_timeline
      - top feature importances
      - plain-English explanation
    """
    try:
        req = request.get_json(silent=True) or {}
        player_id = str(req.get("gsis_id") or "").strip()
        player_name = str(req.get("player_name") or "").strip()

        if not player_id and not player_name:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "player_name is required when gsis_id is not provided",
                        "responseObject": None,
                        "statusCode": 400,
                    }
                ),
                400,
            )

        model_inputs = {
            "player_name": player_name,
            "position": req.get("position"),
            "injury_type": req.get("injury_type"),
            "injury_detail": req.get("injury_detail"),
            "next_game_stadium": req.get("next_game_stadium"),
            "days_until_next_game": req.get("days_until_next_game"),
            "time_missed_already": req.get("time_missed_already"),
            "report_status": req.get("report_status"),
            "practice_status": req.get("practice_status"),
            "severity": req.get("severity"),
            "game_type": req.get("game_type"),
        }

        payload = _predict_player_outcomes(player_id=player_id, model_inputs=model_inputs)
        status_code = 200 if payload.get("ready") else 404
        return (
            jsonify(
                {
                    "success": bool(payload.get("ready")),
                    "message": "Player prediction generated" if payload.get("ready") else payload.get("message", "Prediction unavailable"),
                    "responseObject": payload,
                    "statusCode": status_code,
                }
            ),
            status_code,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Failed to generate player prediction: {str(exc)}",
                    "responseObject": None,
                    "statusCode": 500,
                }
            ),
            500,
        )


@app.post("/api/model/backtest")
def model_backtest():
    """
    Run predictions against a historical week's full injury report and compare
    against actual snap-count outcomes.

    POST body (JSON):
      { "season": 2024, "week": 12 }   — required
      { "season": 2024, "week": 12, "game_type": "REG" }  — optional game_type filter

    Returns per-player predictions with actual outcome (did_play from snap_counts)
    plus aggregate accuracy metrics.
    """
    try:
        import pandas as pd

        req = request.get_json(silent=True) or {}
        season = int(req.get("season") or 0)
        week   = int(req.get("week") or 0)
        game_type = str(req.get("game_type") or "REG").strip().upper()

        if not season or not week:
            return jsonify({"success": False, "message": "season and week are required", "responseObject": None, "statusCode": 400}), 400

        # Pull every injury report row for this week that has a snap-count outcome
        rows = many(
            """
            SELECT
                ir.gsis_id,
                ir.full_name,
                ir.team,
                ir.position,
                ir.injury_type,
                ir.injury_detail,
                ir.severity,
                ir.report_status,
                ir.practice_status,
                sc.did_play AS actual_did_play,
                sc.offense_pct AS actual_offense_pct
            FROM injuryReport ir
            JOIN snap_counts sc
              ON sc.season = ir.season
             AND sc.week   = ir.week
             AND sc.game_type = ir.game_type
             AND sc.team   = ir.team
             AND sc.player_name = TRIM(ir.full_name)
            WHERE ir.season    = %s
              AND ir.week      = %s
              AND ir.game_type = %s
            GROUP BY ir.gsis_id, ir.full_name, ir.team, ir.position,
                     ir.injury_type, ir.injury_detail, ir.severity,
                     ir.report_status, ir.practice_status,
                     sc.did_play, sc.offense_pct
            ORDER BY ir.full_name
            """,
            (season, week, game_type),
        )

        if not rows:
            return jsonify({
                "success": False,
                "message": f"No injury report rows with snap-count outcomes found for {season} Week {week} ({game_type}).",
                "responseObject": None,
                "statusCode": 404,
            }), 404

        results = []
        correct = 0
        total   = 0

        for row in rows:
            player_id   = str(row.get("gsis_id") or "").strip()
            player_name = str(row.get("full_name") or "").strip()
            model_inputs = {
                "player_name":    player_name,
                "position":       row.get("position"),
                "injury_type":    row.get("injury_type"),
                "injury_detail":  row.get("injury_detail"),
                "severity":       row.get("severity"),
                "report_status":  row.get("report_status"),
                "practice_status": row.get("practice_status"),
                "game_type":      game_type,
            }

            # Pass player_id=None to skip per-player DB lookup (uses fast fallback feature row)
            pred = _predict_player_outcomes(player_id=None, model_inputs=model_inputs)
            if not pred.get("ready"):
                continue

            actual       = int(row.get("actual_did_play") or 0)
            play_prob    = (pred.get("next_game") or {}).get("play_probability") or 0.0
            predicted    = 1 if play_prob >= 0.5 else 0
            match        = predicted == actual

            if match:
                correct += 1
            total += 1

            results.append({
                "player_name":       player_name,
                "team":              row.get("team"),
                "position":          row.get("position"),
                "injury_type":       row.get("injury_type"),
                "injury_detail":     row.get("injury_detail"),
                "report_status":     row.get("report_status"),
                "practice_status":   row.get("practice_status"),
                "play_probability":  round(play_prob, 4),
                "predicted_plays":   bool(predicted),
                "actual_did_play":   bool(actual),
                "actual_offense_pct": row.get("actual_offense_pct"),
                "correct":           match,
                "explanation":       (pred.get("explanation") or {}).get("summary", ""),
            })

        accuracy = round(correct / total, 4) if total > 0 else None

        return jsonify({
            "success": True,
            "message": f"Backtest complete: {season} Week {week} ({game_type}) — {total} players evaluated",
            "responseObject": {
                "season":    season,
                "week":      week,
                "game_type": game_type,
                "metrics": {
                    "total_evaluated": total,
                    "correct":         correct,
                    "accuracy":        accuracy,
                    "accuracy_pct":    round(accuracy * 100, 1) if accuracy is not None else None,
                },
                "players": results,
            },
            "statusCode": 200,
        }), 200

    except Exception as exc:
        return jsonify({
            "success": False,
            "message": f"Backtest failed: {str(exc)}",
            "responseObject": None,
            "statusCode": 500,
        }), 500


@app.get("/health")
def health():
    """
    Lightweight health check endpoint.

    Returns HTTP 200 with {"status": "ok"} so load balancers, Docker health
    checks, and monitoring tools can confirm the server is responsive.
    """
    return jsonify({"status": "ok"}), 200


@app.get("/api/quality")
def quality():
    """
    Return data-quality metrics for the injury events dataset.

    Metrics include:
      - Total / linked / unlinked event counts
      - Achilles linkage rate (used as a benchmark for high-profile injuries)
      - Confidence level distribution (high / medium / low)
      - Injury-log to event linkage rate

    Returns JSON with the quality payload nested under 'responseObject'.
    """
    try:
        payload = build_quality_payload()

        return (
            jsonify(
                {
                    "success": True,
                    "message": "Data quality metrics fetched",
                    "responseObject": payload,
                    "statusCode": 200,
                }
            ),
            200,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Failed to fetch quality metrics: {str(exc)}",
                    "responseObject": None,
                    "statusCode": 500,
                }
            ),
            500,
        )


@app.get("/api/trends/time")
def trends_time():
    """
    Return injury counts aggregated over time (by season or by week).

    Query parameters:
      mode        - 'season' (default) or 'week'
      count_mode  - 'unique' (default, distinct injuryIds) or 'occurrences' (all rows)
      source_mode - 'all' | 'weekly_report' | 'ir_transactions'
      scope       - 'all' | 'logged' | 'unlogged'
      filter_kind - optional 'injury_type' or 'injury_detail' to narrow results
      filter_value- the value to filter on (used with filter_kind)
      season      - specific season year to filter weeks within
    """
    try:
        mode = (request.args.get("mode") or "season").lower()
        if mode not in {"season", "week"}:
            mode = "season"
        count_mode = normalize_count_mode(request.args.get("count_mode"))
        source_mode = normalize_source_mode(request.args.get("source_mode"))

        filter_kind = request.args.get("filter_kind")
        filter_value = request.args.get("filter_value")
        season_filter = request.args.get("season")
        scope = (request.args.get("scope") or "all").lower()
        if scope not in {"all", "logged", "unlogged"}:
            scope = "all"

        where_parts = []
        params = []

        if count_mode == "occurrences":
            from_sql = "FROM injuryReport ir"
            if mode == "week":
                period_expr = "CASE WHEN ir.week = 22 AND ir.season < 2021 THEN 23 ELSE ir.week END"
            else:
                period_expr = "ir.season"

            where_parts.append("ir.season IS NOT NULL")
            where_parts.append("ir.season < 2025")
            where_parts.append("(ir.action IS NULL OR TRIM(ir.action) = '' OR LOWER(ir.action) NOT LIKE 'activated from%')")

            if mode == "week":
                where_parts.append("ir.week IS NOT NULL")
                where_parts.append("ir.week BETWEEN 1 AND 22")

            if season_filter and str(season_filter).isdigit() and mode == "week":
                where_parts.append("ir.season = %s")
                params.append(int(season_filter))

            if scope == "logged":
                where_parts.append("ir.loggedInGame = 1")
            elif scope == "unlogged":
                where_parts.append("(ir.loggedInGame = 0 OR ir.loggedInGame IS NULL)")

            if source_mode == "weekly_report":
                where_parts.append("(ir.action IS NULL OR TRIM(ir.action) = '')")
            elif source_mode == "ir_transactions":
                where_parts.append("(ir.action IS NOT NULL AND TRIM(ir.action) <> '')")

            if filter_kind in {"injury_type", "injury_detail"} and filter_value:
                where_parts.append(f"ir.{filter_kind} = %s")
                params.append(filter_value)

            where_sql = " AND ".join(where_parts) if where_parts else "1=1"

            rows = many(
                f"""
                SELECT
                    {period_expr} AS period,
                    COUNT(*) AS injury_count,
                    SUM(CASE WHEN ir.loggedInGame = 1 THEN 1 ELSE 0 END) AS linked_count
                {from_sql}
                WHERE {where_sql}
                GROUP BY {period_expr}
                ORDER BY {period_expr}
                """,
                tuple(params),
            )
        else:
            if mode == "week":
                period_select = """
                CASE
                    WHEN week_map.week = 22 AND week_map.season < 2021 THEN 23
                    ELSE week_map.week
                END AS period
                """
                group_order_sql = """
                GROUP BY
                    CASE
                        WHEN week_map.week = 22 AND week_map.season < 2021 THEN 23
                        ELSE week_map.week
                    END
                ORDER BY
                    CASE
                        WHEN week_map.week = 22 AND week_map.season < 2021 THEN 23
                        ELSE week_map.week
                    END
                """
                season_having = ""
                if season_filter and str(season_filter).isdigit():
                    season_having = f"AND MIN(CASE WHEN season IS NOT NULL THEN season END) = {int(season_filter)}"
                join_sql = f"""
                JOIN (
                    SELECT
                        injuryId,
                        MIN(CASE WHEN season IS NOT NULL THEN season END) AS season,
                        MIN(CASE WHEN season IS NOT NULL AND week BETWEEN 1 AND 22 THEN week END) AS week
                    FROM injuryReport
                    GROUP BY injuryId
                    HAVING SUM(
                        CASE
                            WHEN action IS NULL OR TRIM(action) = '' OR LOWER(action) NOT LIKE 'activated from%%'
                                THEN 1
                            ELSE 0
                        END
                    ) > 0
                    {season_having}
                ) AS week_map ON week_map.injuryId = e.injuryId
                """
                where_parts.append("week_map.week IS NOT NULL")
                where_parts.append("week_map.season < 2025")
            else:
                period_select = "season_map.season AS period"
                group_order_sql = "GROUP BY season_map.season ORDER BY season_map.season"
                join_sql = """
                JOIN (
                    SELECT injuryId, MIN(CASE WHEN season IS NOT NULL THEN season END) AS season
                    FROM injuryReport
                    GROUP BY injuryId
                    HAVING SUM(
                        CASE
                            WHEN action IS NULL OR TRIM(action) = '' OR LOWER(action) NOT LIKE 'activated from%%'
                                THEN 1
                            ELSE 0
                        END
                    ) > 0
                ) AS season_map ON season_map.injuryId = e.injuryId
                """
                where_parts.append("season_map.season IS NOT NULL")
                where_parts.append("season_map.season < 2025")

            if scope == "logged":
                where_parts.append("e.loggedInGame = 1")
            elif scope == "unlogged":
                where_parts.append("e.loggedInGame = 0")

            if source_mode == "weekly_report":
                where_parts.append(
                    "EXISTS (SELECT 1 FROM injuryReport ir_src WHERE ir_src.injuryId = e.injuryId "
                    "AND (ir_src.action IS NULL OR TRIM(ir_src.action) = ''))"
                )
            elif source_mode == "ir_transactions":
                where_parts.append(
                    "EXISTS (SELECT 1 FROM injuryReport ir_src WHERE ir_src.injuryId = e.injuryId "
                    "AND ir_src.action IS NOT NULL AND TRIM(ir_src.action) <> '')"
                )

            if filter_kind in {"injury_type", "injury_detail"} and filter_value:
                where_parts.append(f"e.{filter_kind} = %s")
                params.append(filter_value)

            where_sql = " AND ".join(where_parts)

            rows = many(
                f"""
                SELECT
                    {period_select},
                    COUNT(DISTINCT e.injuryId) AS injury_count,
                    COUNT(DISTINCT CASE WHEN e.loggedInGame = 1 THEN e.injuryId END) AS linked_count
                FROM events e
                {join_sql}
                WHERE {where_sql}
                {group_order_sql}
                """,
                tuple(params),
            )

        total_counted_events = sum((int(row.get("injury_count") or 0) for row in rows))

        return (
            jsonify(
                {
                    "success": True,
                    "message": "Injury time trends fetched",
                    "responseObject": {
                        "granularity": mode,
                        "scope": scope,
                        "count_mode": count_mode,
                        "source_mode": source_mode,
                        "filters": {
                            "filter_kind": filter_kind,
                            "filter_value": filter_value,
                        },
                        "total_counted_events": total_counted_events,
                        "rows": rows,
                    },
                    "statusCode": 200,
                }
            ),
            200,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Failed to fetch trends: {str(exc)}",
                    "responseObject": None,
                    "statusCode": 500,
                }
            ),
            500,
        )


@app.get("/api/trends/values")
def trends_values():
    """
    Return the distinct values for a given filter dimension.

    Query parameters:
      filter_kind - 'injury_type' (default) or 'injury_detail'

    Used to populate the filter dropdown on the statistics charts so users
    can drill down into a specific injury category.
    """
    try:
        filter_kind = request.args.get("filter_kind", "injury_type")
        if filter_kind not in {"injury_type", "injury_detail"}:
            filter_kind = "injury_type"

        rows = many(
            f"""
            SELECT DISTINCT {filter_kind} AS value
            FROM events
            WHERE {filter_kind} IS NOT NULL AND {filter_kind} <> ''
            ORDER BY {filter_kind}
            """
        )

        return (
            jsonify(
                {
                    "success": True,
                    "message": "Trend filter values fetched",
                    "responseObject": {
                        "filter_kind": filter_kind,
                        "values": [r["value"] for r in rows],
                    },
                    "statusCode": 200,
                }
            ),
            200,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Failed to fetch trend values: {str(exc)}",
                    "responseObject": None,
                    "statusCode": 500,
                }
            ),
            500,
        )


@app.get("/api/statistics/overview")
def statistics_overview():
    """
    Return a comprehensive statistics summary for the injury events table.

    Query parameters:
      scope       - 'all' | 'logged' | 'unlogged'  (filter by in-game link status)
      count_mode  - 'unique' (distinct injuries) or 'occurrences' (all report rows)
      source_mode - 'all' | 'weekly_report' | 'ir_transactions'
      filter_kind - optional 'injury_type' or 'injury_detail'
      filter_value- value to filter on

    Returns breakdown by injury type, injury detail, season trend, confidence,
    and missingness metrics.
    """
    try:
        scope = request.args.get("scope", "all")
        filter_kind = request.args.get("filter_kind")
        filter_value = request.args.get("filter_value")
        count_mode = normalize_count_mode(request.args.get("count_mode"))
        source_mode = normalize_source_mode(request.args.get("source_mode"))
        payload = build_statistics_payload(scope, filter_kind, filter_value, count_mode, source_mode)
        return (
            jsonify(
                {
                    "success": True,
                    "message": "Statistics summary fetched",
                    "responseObject": payload,
                    "statusCode": 200,
                }
            ),
            200,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Failed to fetch statistics summary: {str(exc)}",
                    "responseObject": None,
                    "statusCode": 500,
                }
            ),
            500,
        )


@app.get("/api/analytics/modules")
def analytics_modules():
    """
    Return the list of available analytics explorer modules.

    Each module represents a different dimension by which injuries can be
    explored (e.g. field surface, weather, player age).  This endpoint
    powers the module selector dropdown in the analytics explorer UI.
    """
    modules = [
        {"key": "field_type", "label": "Field Type"},
        {"key": "quarter", "label": "Quarter of Game"},
        {"key": "location", "label": "Game Location"},
        {"key": "stadium", "label": "Stadium"},
        {"key": "roof", "label": "Roof Type"},
        {"key": "transactions", "label": "Transactions"},
        {"key": "weather", "label": "Weather Condition"},
        {"key": "humidity", "label": "Humidity"},
        {"key": "wind_direction", "label": "Wind Speed"},
        {"key": "temperature", "label": "Temperature"},
        {"key": "age", "label": "Age Buckets"},
        {"key": "experience", "label": "Experience Buckets"},
        {"key": "position", "label": "Position"},
        {"key": "opponent", "label": "Opponent Team (Derived)"},
    ]
    return (
        jsonify(
            {
                "success": True,
                "message": "Analytics modules fetched",
                "responseObject": {"modules": modules},
                "statusCode": 200,
            }
        ),
        200,
    )


@app.get("/api/analytics/explorer")
def analytics_explorer():
    """
    Run the analytics explorer for the requested module and return aggregated data.

    This is the main data endpoint for the interactive analytics charts on the
    statistics page.  It supports 14 different analysis modules (field_type,
    weather, position, age, etc.) that each aggregate injuries along a different
    dimension.

    Query parameters:
      module      - which analysis dimension to run (see analytics_modules)
      scope       - 'all' | 'logged' | 'unlogged'
      granularity - optional time breakdown: 'none' | 'season' | 'week'
      filter_kind - optional 'injury_type' or 'injury_detail'
      filter_value- value to filter on
      season_start- earliest season to include
      season_end  - latest season to include
      count_mode  - 'unique' (default) or 'occurrences'
      source_mode - 'all' | 'weekly_report' | 'ir_transactions'
    """
    try:
        module = request.args.get("module", "injury_trends")
        scope = request.args.get("scope", "all")
        granularity = request.args.get("granularity", "none")
        filter_kind = request.args.get("filter_kind")
        filter_value = request.args.get("filter_value")
        season_start = request.args.get("season_start", type=int)
        season_end = request.args.get("season_end", type=int)
        count_mode = normalize_count_mode(request.args.get("count_mode"))
        source_mode = normalize_source_mode(request.args.get("source_mode"))

        payload = build_explorer_payload(
            module=module,
            scope=scope,
            granularity=granularity,
            filter_kind=filter_kind,
            filter_value=filter_value,
            season_start=season_start,
            season_end=season_end,
            count_mode=count_mode,
            source_mode=source_mode,
        )

        return (
            jsonify(
                {
                    "success": True,
                    "message": "Analytics explorer data fetched",
                    "responseObject": payload,
                    "statusCode": 200,
                }
            ),
            200,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Failed to fetch analytics explorer data: {str(exc)}",
                    "responseObject": None,
                    "statusCode": 500,
                }
            ),
            500,
        )


if __name__ == "__main__":
    # Warm the model bundle from disk cache in a background thread at startup.
    # This avoids blocking the first HTTP request — which could take 30+ seconds
    # if the model bundle needs to be retrained.  Requests that arrive before the
    # warmup completes receive a 503 with a Retry-After header.
    threading.Thread(target=_warmup_model_bundle, name="model-warmup", daemon=True).start()
    # use_reloader=False prevents the development server from starting a second
    # process that would try to train the model twice on startup.
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)
