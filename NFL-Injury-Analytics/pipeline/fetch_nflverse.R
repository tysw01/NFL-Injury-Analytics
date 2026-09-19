#!/usr/bin/env Rscript
# pipeline/fetch_nflverse.R
#
# Downloads nflverse data (pbp, weekly injury report, schedules, players)
# and writes CSVs into nfl_data/.
#
# Modes:
#   Rscript fetch_nflverse.R                        full rebuild (1999–present)
#   Rscript fetch_nflverse.R --incremental          current season only (fast)
#   Rscript fetch_nflverse.R --from 2010            from a specific start year
#
# nflreadr caches parquet files locally, so repeat full runs are faster after
# the first download.

suppressMessages({
  library(nflreadr)
  library(readr)
  library(dplyr)
})

args        <- commandArgs(trailingOnly = TRUE)
incremental <- "--incremental" %in% args

# --from YEAR flag
from_year <- NULL
from_idx  <- which(args == "--from")
if (length(from_idx) > 0 && length(args) >= from_idx + 1) {
  from_year <- as.integer(args[from_idx + 1])
}

# ── Determine season range ────────────────────────────────────────────────────
current <- nflreadr::most_recent_season()
cat(sprintf("Most recent NFL season: %d\n", current))

if (incremental) {
  seasons <- current
  cat("Mode: incremental (current season only)\n")
} else if (!is.null(from_year)) {
  seasons <- from_year:current
  cat(sprintf("Mode: from %d to %d\n", from_year, current))
} else {
  # nflverse pbp goes back to 1999; injury reports to ~2009
  seasons <- 1999:current
  cat(sprintf("Mode: full refresh (seasons %d–%d)\n", min(seasons), max(seasons)))
}

data_dir <- "nfl_data"
dir.create(data_dir, showWarnings = FALSE)

# ── Play-by-play ──────────────────────────────────────────────────────────────
cat("Fetching play-by-play...\n")
pbp_new <- nflreadr::load_pbp(seasons = seasons)
cat(sprintf("  Downloaded: %d rows\n", nrow(pbp_new)))

pbp_path <- file.path(data_dir, "pbp.csv")
if (incremental && file.exists(pbp_path)) {
  pbp_existing <- readr::read_csv(pbp_path, show_col_types = FALSE)
  # Drop existing rows for the current season then append fresh ones
  pbp_existing <- pbp_existing %>% filter(.data$season != current)
  pbp_combined <- bind_rows(pbp_existing, pbp_new)
  readr::write_csv(pbp_combined, pbp_path)
  cat(sprintf("  Merged into existing pbp.csv → %d total rows\n", nrow(pbp_combined)))
} else {
  readr::write_csv(pbp_new, pbp_path)
  cat(sprintf("  Wrote pbp.csv → %d rows\n", nrow(pbp_new)))
}

# ── Weekly injury report ──────────────────────────────────────────────────────
cat("Fetching weekly injury reports...\n")
inj_new <- nflreadr::load_injuries(seasons = seasons)
cat(sprintf("  Downloaded: %d rows\n", nrow(inj_new)))

inj_path <- file.path(data_dir, "injury_report.csv")
if (incremental && file.exists(inj_path)) {
  inj_existing <- readr::read_csv(inj_path, show_col_types = FALSE)
  inj_existing <- inj_existing %>% filter(.data$season != current)
  inj_combined <- bind_rows(inj_existing, inj_new)
  readr::write_csv(inj_combined, inj_path)
  cat(sprintf("  Merged into existing injury_report.csv → %d total rows\n", nrow(inj_combined)))
} else {
  readr::write_csv(inj_new, inj_path)
  cat(sprintf("  Wrote injury_report.csv → %d rows\n", nrow(inj_new)))
}

# ── Players ───────────────────────────────────────────────────────────────────
# Players is a static registry — always full refresh (fast, ~24k rows)
cat("Fetching player registry...\n")
players <- nflreadr::load_players()
readr::write_csv(players, file.path(data_dir, "nfl_players.csv"))
cat(sprintf("  Wrote nfl_players.csv → %d rows\n", nrow(players)))

# ── Schedules ─────────────────────────────────────────────────────────────────
# Full season schedule: home/away, stadium, kickoff, game_type.
# Covers past + future games for the requested seasons.
cat("Fetching schedules...\n")
sched_seasons <- if (incremental) current else max(2009, min(seasons)):current
sched_new <- nflreadr::load_schedules(seasons = sched_seasons) %>%
  select(
    season, game_type, week,
    home_team, away_team,
    stadium, roof,
    gameday, gametime,
    game_id
  )
cat(sprintf("  Downloaded: %d rows\n", nrow(sched_new)))

sched_path <- file.path(data_dir, "schedules.csv")
if (incremental && file.exists(sched_path)) {
  sched_existing <- readr::read_csv(sched_path, show_col_types = FALSE)
  sched_existing <- sched_existing %>% filter(.data$season != current)
  sched_combined <- bind_rows(sched_existing, sched_new)
  readr::write_csv(sched_combined, sched_path)
  cat(sprintf("  Merged into existing schedules.csv → %d total rows\n", nrow(sched_combined)))
} else {
  readr::write_csv(sched_new, sched_path)
  cat(sprintf("  Wrote schedules.csv → %d rows\n", nrow(sched_new)))
}

cat("\nnflverse fetch complete.\n")
