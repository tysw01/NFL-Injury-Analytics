#!/usr/bin/env python3
"""
tests/run_backtest.py
=====================
Standalone backtest runner for 2024 weeks 8-12.

Instead of calling the /api/model/predict endpoint, this script loads the
trained model bundle directly from the injuryAPI module and runs batch
inference across entire weeks of injury-report data.  This is much faster
than per-player API calls because:
  1. Population stats are loaded once and reused via a lookup dict.
  2. All feature rows for a week are built in Python (no per-player DB calls).
  3. model.predict_proba() is called on the entire week's DataFrame at once.

Ground truth is determined by whether the player appeared in snap_counts
for that week (did_play = 1 means the player participated in at least 1 snap).

Usage:
    python tests/run_backtest.py   (must be run from the project root)
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")  # suppress sklearn and pandas deprecation warnings

# Add project root to sys.path so we can import injuryAPI even when running
# this script from the tests/ subdirectory.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from injuryAPI import (
    many,                     # utility: run a SELECT and return all rows
    get_advanced_model_bundle,# loads/caches the trained availability model
    _injury_body_region,      # maps injury detail string → body region category
    _to_int,                  # safely cast a value to int (returns default on failure)
    _to_float,                # safely cast a value to float (returns default on failure)
)

# ── Backtest configuration ────────────────────────────────────────────────────
SEASON    = 2024           # NFL season to backtest against
WEEKS     = [8, 9, 10, 11, 12]  # Regular-season weeks to evaluate
GAME_TYPE = "REG"         # Include only regular-season games (not preseason/playoffs)

# ── Position → position_group mapping (mirrors training data) ─────────────────
# Maps individual NFL position codes to broad positional groups.  These groups
# must exactly match the categories used during model training or the one-hot
# encoding will create unseen categories and degrade performance.
_POS_GROUP = {
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "OT": "OL", "OG": "OL", "OC": "OL", "G": "OL", "T": "OL", "C": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
    "CB": "DB", "S": "DB", "SS": "DB", "FS": "DB", "DB": "DB",
    "K": "K", "P": "P", "LS": "LS",
}

def _pos_group(pos: str) -> str:
    """Return the broad position group for a raw position string (e.g. 'OT' → 'OL')."""
    return _POS_GROUP.get((pos or "").strip().upper(), "Unknown")


# Set of lowercased action strings that indicate the player was placed on IR.
# Used to derive the is_ir_transaction binary feature for the model.
_IR_PLACEMENT_ACTIONS = {"placed on ir", "placed on pup", "placed on nfi", "out for season"}


def _build_feature_row(row: dict, season: int, week: int, game_type: str, pop_lookup: dict | None = None) -> dict:
    """
    Convert a single injuryReport DB row into a flat feature dictionary
    suitable for passing to the trained sklearn pipeline.

    Hard-coded defaults (e.g. days_until_next_game=7.0, age=27.0) are used for
    contextual features that cannot be back-filled from historical DB records.
    Population stats (pop_unavail_rate etc.) are looked up from pop_lookup
    by injury_detail so we avoid making extra DB calls per player.

    The returned dict includes two private keys (prefixed with '_') that hold
    player name and ground truth label; these are stripped before model inference.
    """
    pos          = str(row.get("position") or "Unknown").strip()
    inj_detail   = str(row.get("injury_detail") or "Unknown").strip() or "Unknown"
    inj_type     = str(row.get("injury_type")   or "Unknown").strip() or "Unknown"
    # How many regular-season weeks are left after the current one
    weeks_remain = max(0, 18 - week)
    # Normalise the transaction action field; empty strings become 'none'
    action_raw   = str(row.get("action") or "").strip()
    action_norm  = action_raw if action_raw else "none"
    # Binary flag: 1 if the action puts the player on IR/PUP/NFI, else 0
    is_ir_txn    = int(action_norm.lower() in _IR_PLACEMENT_ACTIONS)
    # Population lookup: real historical averages keyed by injury_detail;
    # fall back to conservative defaults if this injury type is unseen.
    pop          = (pop_lookup or {}).get(inj_detail, {})
    return {
        # identifiers (not used by model)
        "_player_name": str(row.get("full_name") or "").strip(),
        "_actual":      int(row.get("actual_did_play") or 0),
        # ── model features ───────────────────────────────────────────────────
        "season":                       season,
        "week":                         week,
        "team":                         str(row.get("team") or "Unknown").strip() or "Unknown",
        "game_type":                    game_type,
        "injury_type":                  inj_type,
        "injury_detail":                inj_detail,
        "injury_body_region":           _injury_body_region(inj_detail),
        "severity":                     str(row.get("severity") or "Unknown").strip() or "Unknown",
        "position":                     pos or "Unknown",
        "position_group":               _pos_group(pos),
        "report_status":                str(row.get("report_status")   or "Questionable").strip() or "Questionable",
        "practice_status":              str(row.get("practice_status") or "Limited").strip()      or "Limited",
        "source":                       "injury_report",
        "next_game_stadium":            "Unknown",
        "next_game_opponent":           "Unknown",
        "next_game_is_home":            0,
        "next_game_is_dome":            0,
        "days_until_next_game":         7.0,
        "time_missed_already":          0.0,
        "injury_instance_no":           1.0,
        "prior_same_injury_instances":  0.0,
        "new_instance_event":           1.0,
        "weeks_since_first_same_injury":0.0,
        "days_since_prev":              7.0,
        "new_injury_flag":              1,
        "on_ir":                        0,
        "prev_on_ir":                   0,
        "years_of_experience":          4.0,
        "age":                          27.0,
        "prior_reports_count":          0.0,
        "prior_unavailable_count":      0.0,
        "prev3_unavail_rate":           0.0,
        "gap_since_prev_report":        1.0,
        "injury_repeat_count":          0.0,
        "weeks_remaining":              weeks_remain,
        "pop_unavail_rate":             pop.get("pop_unavail_rate",   0.15),
        "pop_ir_rate":                  pop.get("pop_ir_rate",         0.05),
        "pop_action_ir_rate":           pop.get("pop_action_ir_rate", 0.05),
        "action":                       action_norm,
        "is_ir_transaction":            is_ir_txn,
    }


# ── Load model bundle once ────────────────────────────────────────────────────
# get_advanced_model_bundle() checks if a cached model pickle is fresh enough;
# if not it retrains from the DB.  Loading the bundle once here avoids repeated
# disk reads or retraining during the per-week loop below.
print("Loading model bundle...", flush=True)
bundle = get_advanced_model_bundle()
if not bundle.get("ready"):
    # Model training failed — print the error and exit so pytest can catch it.
    print("ERROR: model not ready:", bundle.get("message"))
    sys.exit(1)

# Unpack the parts we need from the bundle
avail_model  = bundle["availability_model"]["model"]  # trained sklearn Pipeline
feature_cols = bundle["feature_cols"]                  # ordered list of feature names the model expects
ds           = bundle.get("dataset", {})               # metadata about the training dataset
print(f"Model ready  rows={ds.get('row_count')}  features={ds.get('feature_count')}", flush=True)

# ── Pre-load population stats by injury_detail (one query, no per-player calls) ─
# Instead of querying population averages inside _build_feature_row() for every
# player (which would be O(n) DB queries), we run one aggregation query here and
# store results in a dict keyed by injury_detail.  The HAVING COUNT(*) >= 5 clause
# excludes injury types with too few examples to produce reliable averages.
print("Loading population stats...", flush=True)
pop_rows = many("""
    SELECT
        COALESCE(NULLIF(TRIM(injury_detail),''),'Unknown') AS injury_detail,
        AVG(CASE WHEN report_status IN ('Out','Doubtful') THEN 1.0 ELSE 0.0 END) AS pop_unavail_rate,
        AVG(CASE WHEN on_ir = 1 THEN 1.0 ELSE 0.0 END)                           AS pop_ir_rate,
        AVG(CASE WHEN LOWER(TRIM(COALESCE(action,''))) IN (
            'placed on ir','placed on pup','placed on nfi','out for season'
        ) THEN 1.0 ELSE 0.0 END)                                                  AS pop_action_ir_rate
    FROM injuryReport
    WHERE season >= 2011
    GROUP BY COALESCE(NULLIF(TRIM(injury_detail),''),'Unknown')
    HAVING COUNT(*) >= 5
""")
pop_lookup = {
    r["injury_detail"]: {
        "pop_unavail_rate":   float(r["pop_unavail_rate"]   or 0.15),
        "pop_ir_rate":        float(r["pop_ir_rate"]        or 0.05),
        "pop_action_ir_rate": float(r["pop_action_ir_rate"] or 0.05),
    }
    for r in (pop_rows or [])
}
print(f"Pop lookup loaded: {len(pop_lookup)} distinct injuries\n", flush=True)

all_results = {}

# ── Per-week inference loop ──────────────────────────────────────────────────
# For each week, fetch all injury report rows from the DB, build a DataFrame of
# model features in one shot, call predict_proba, then compare predictions to the
# actual snap-count-based ground truth.
for week in WEEKS:
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
            ir.action,
            -- NULL snap count means player was on IR but did NOT play
            CASE WHEN sc.did_play IS NOT NULL THEN sc.did_play ELSE 0 END AS actual_did_play,
            sc.offense_pct AS actual_offense_pct
        FROM injuryReport ir
        LEFT JOIN snap_counts sc
          ON sc.season    = ir.season
         AND sc.week      = ir.week
         AND sc.game_type = ir.game_type
         AND sc.team      = ir.team
         AND sc.player_name = TRIM(ir.full_name)
        WHERE ir.season    = %s
          AND ir.week      = %s
          AND ir.game_type = %s
        GROUP BY ir.gsis_id, ir.full_name, ir.team, ir.position,
                 ir.injury_type, ir.injury_detail, ir.severity,
                 ir.report_status, ir.practice_status, ir.action,
                 sc.did_play, sc.offense_pct
        ORDER BY ir.full_name
        """,
        (SEASON, week, GAME_TYPE),
    )

    print(f"W{week}: fetched {len(rows)} rows from IR ({sum(1 for r in rows if r.get('actual_did_play'))} played / {sum(1 for r in rows if not r.get('actual_did_play'))} sat out)", flush=True)
    if not rows:
        print(f"W{week}: 0 injury report rows found", flush=True)
        all_results[week] = None
        continue

    # Build one feature dict per player, then strip out the private metadata
    # fields (_player_name, _actual) before passing to the model.
    feature_rows = [_build_feature_row(r, SEASON, week, GAME_TYPE, pop_lookup) for r in rows]
    player_names = [fr.pop("_player_name") for fr in feature_rows]  # labels for display
    actuals      = [fr.pop("_actual")      for fr in feature_rows]   # ground-truth labels (1=played, 0=sat)

    # Reindex the DataFrame to model's expected column order (missing cols → NaN)
    X = pd.DataFrame(feature_rows)[feature_cols]
    # predict_proba returns [[P(class=0), P(class=1)], ...]; we want P(class=1) = P(played)
    probs = avail_model.predict_proba(X)[:, 1]

    correct = 0
    total   = len(rows)
    wrong   = []  # will hold tuples of (name, prob, predicted, actual) for misclassified players

    # First pass: count correct predictions
    for i, (name, actual, prob) in enumerate(zip(player_names, actuals, probs)):
        predicted  = int(prob >= 0.5)  # threshold at 50% probability
        is_correct = predicted == actual
        if is_correct:
            correct += 1
        else:
            wrong.append((name, prob, predicted, actual))

    # Second pass: print a sample of correct predictions + every wrong prediction
    # (helps quickly spot systematic biases — e.g. always predicting Out for Doubtful)
    shown = 0  # track how many correct predictions we've displayed
    for i, (name, actual, prob) in enumerate(zip(player_names, actuals, probs)):
        predicted  = int(prob >= 0.5)
        is_correct = predicted == actual
        flag = "✓" if is_correct else "✗"
        if is_correct and shown < 3:
            # Show up to 3 correct examples as a sanity check
            print(f"  {flag} {name:<24} prob={prob:.3f} pred={'Play' if predicted else 'Out '} actual={'Play' if actual else 'Out '}", flush=True)
            shown += 1
        elif not is_correct:
            # Always show wrong predictions so we can inspect failures
            print(f"  {flag} {name:<24} prob={prob:.3f} pred={'Play' if predicted else 'Out '} actual={'Play' if actual else 'Out '}", flush=True)

    # Summarise the week's results
    accuracy = round(correct / total * 100, 1) if total else None
    n_played  = sum(1 for a in actuals if a == 1)  # players who actually played
    n_sat     = sum(1 for a in actuals if a == 0)  # players who sat out
    all_results[week] = {"n": total, "correct": correct, "accuracy": accuracy, "played": n_played, "sat": n_sat}
    print(f"\n2024 Week {week}: n={total} ({n_played} played / {n_sat} sat)  correct={correct}  accuracy={accuracy}%\n{'─'*50}", flush=True)

# ── Summary table ──────────────────────────────────────────────────────────────
# Aggregate all per-week results into a single ASCII table that makes it easy
# to see if accuracy degrades later in the season (e.g. due to IR transactions
# accumulating and changing the distribution of injury severities).
print("\n╔══════════════════════════════════════════════════════╗")
print("║  2024 Backtest Summary (W8-12)                       ║")
print("╠════════╦══════╦═════════╦════════════╦═══════════════╣")
print("║  Week  ║  N   ║ Correct ║ Accuracy   ║ Played↑/Sat↓  ║")
print("╠════════╬══════╬═════════╬════════════╬═══════════════╣")
total_n = 0
total_correct = 0
for week in WEEKS:
    r = all_results.get(week)
    if r:
        w_str = f"W{week}"
        print(f"║  {w_str:<5} ║ {r['n']:4d} ║  {r['correct']:5d}  ║  {r['accuracy']:6.1f}%  ║  {r['played']:3d}↑ {r['sat']:3d}↓  ║")
        total_n       += r["n"]
        total_correct += r["correct"]
    else:
        print(f"║  W{week}   ║  N/A ║   N/A   ║    N/A     ║  N/A          ║")
print("╠════════╬══════╬═════════╬════════════╬═══════════════╣")
overall = round(total_correct / total_n * 100, 1) if total_n else None
print(f"║ Overall║ {total_n:4d} ║  {total_correct:5d}  ║  {str(overall)+'%':>7s}   ║               ║")
print("╚════════╩══════╩═════════╩════════════╩═══════════════╝")
