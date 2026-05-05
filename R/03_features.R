# R/03_features.R
# Engineers all model inputs from raw match and odds data.
#
# Inputs:  data/raw/historical_matches.csv
#          data/raw/upcoming_matches.csv
#          data/raw/odds_consensus.csv
#
# Outputs: data/processed/team_long.rds         — long format (one row per team per match)
#          data/processed/team_strength.csv      — season-level Dixon-Coles parameters
#          data/processed/match_features.csv     — wide feature matrix for model training
#          data/processed/upcoming_features.csv  — upcoming fixtures + features + odds

source("config.R")
source("00_setup.R")
library(slider)

PROCESSED_DIR <- file.path("data", "processed")
dir.create(PROCESSED_DIR, recursive = TRUE, showWarnings = FALSE)

nan_to_na <- function(x) replace(x, is.nan(x), NA_real_)

# ── 1. Load ───────────────────────────────────────────────────────────────────

historical <- read_csv(file.path(DATA_DIR, "historical_matches.csv"), show_col_types = FALSE) |>
  filter(!is.na(home_goals), !is.na(away_goals))

upcoming   <- read_csv(file.path(DATA_DIR, "upcoming_matches.csv"),   show_col_types = FALSE)
odds       <- read_csv(file.path(DATA_DIR, "odds_consensus.csv"),     show_col_types = FALSE)

message(glue("Loaded {nrow(historical)} historical matches, {nrow(upcoming)} upcoming fixtures."))

# ── 2. Long format ────────────────────────────────────────────────────────────
# One row per team per match, from that team's perspective.
# team_result: W/D/L from the team's point of view.

message("Building long format...")

team_long <- bind_rows(
  historical |>
    transmute(
      match_id, competition, season, matchday, utc_date,
      team     = home_team, opponent = away_team,
      goals_for = home_goals, goals_against = away_goals,
      is_home  = TRUE,
      team_result = case_when(
        result == "H" ~ "W", result == "D" ~ "D", result == "A" ~ "L",
        TRUE ~ NA_character_
      )
    ),
  historical |>
    transmute(
      match_id, competition, season, matchday, utc_date,
      team     = away_team, opponent = home_team,
      goals_for = away_goals, goals_against = home_goals,
      is_home  = FALSE,
      team_result = case_when(
        result == "A" ~ "W", result == "D" ~ "D", result == "H" ~ "L",
        TRUE ~ NA_character_
      )
    )
) |>
  mutate(
    team_points = case_when(
      team_result == "W" ~ 3, team_result == "D" ~ 1, team_result == "L" ~ 0,
      TRUE ~ NA_real_
    )
  ) |>
  arrange(team, utc_date)

# ── 3. Rolling form (last 5 matches, cross-competition, pre-match) ────────────
# Window: [i-5, i-1] — the 5 matches before the current one.
# .after = -1 excludes the current row so there is no data leakage.
# .complete = FALSE allows partial windows for early-career rows.

message("Computing rolling form (last 5 matches)...")

team_long <- team_long |>
  group_by(team) |>
  arrange(utc_date, .by_group = TRUE) |>
  mutate(
    form_pts_5 = slide_dbl(team_points,   mean,   .before = 5, .after = -1, .complete = FALSE),
    form_gf_5  = slide_dbl(goals_for,     mean,   .before = 5, .after = -1, .complete = FALSE),
    form_ga_5  = slide_dbl(goals_against, mean,   .before = 5, .after = -1, .complete = FALSE),
    form_gd_5  = form_gf_5 - form_ga_5,
    form_n     = slide_int(team_points,   length, .before = 5, .after = -1, .complete = FALSE)
  ) |>
  mutate(across(starts_with("form_"), nan_to_na)) |>
  ungroup()

saveRDS(team_long, file.path(PROCESSED_DIR, "team_long.rds"))
message(glue("  → {nrow(team_long)} team-match rows saved."))

# ── 4. Attack / defence strength (Dixon-Coles parameters) ────────────────────
# For team i in a given season/competition:
#   home_att = (team's avg home goals)   / (league avg home goals)   > 1 = above-avg attack
#   home_def = (team's avg home conceded)/ (league avg away goals)   < 1 = above-avg defence
#   away_att = (team's avg away goals)   / (league avg away goals)
#   away_def = (team's avg away conceded)/ (league avg home goals)
#
# These multiply together in the Dixon-Coles model:
#   λ_home = lg_avg_home * home_att_home_team * home_def_away_team
#   λ_away = lg_avg_away * away_att_away_team * away_def_home_team

message("Computing season-level attack/defence strengths...")

compute_season_strength <- function(df) {
  league_avgs <- df |>
    group_by(competition, season) |>
    summarise(
      lg_avg_home = mean(home_goals, na.rm = TRUE),
      lg_avg_away = mean(away_goals, na.rm = TRUE),
      n_matches   = n(),
      .groups = "drop"
    )

  home_stats <- df |>
    group_by(team = home_team, competition, season) |>
    summarise(
      mean_home_gf = mean(home_goals, na.rm = TRUE),
      mean_home_ga = mean(away_goals, na.rm = TRUE),
      home_n       = n(),
      .groups = "drop"
    )

  away_stats <- df |>
    group_by(team = away_team, competition, season) |>
    summarise(
      mean_away_gf = mean(away_goals, na.rm = TRUE),
      mean_away_ga = mean(home_goals, na.rm = TRUE),
      away_n       = n(),
      .groups = "drop"
    )

  home_stats |>
    full_join(away_stats, by = c("team", "competition", "season")) |>
    left_join(league_avgs, by = c("competition", "season")) |>
    mutate(
      home_att = mean_home_gf / lg_avg_home,
      home_def = mean_home_ga / lg_avg_away,
      away_att = mean_away_gf / lg_avg_away,
      away_def = mean_away_ga / lg_avg_home
    ) |>
    select(team, competition, season,
           home_att, home_def, away_att, away_def,
           lg_avg_home, lg_avg_away, home_n, away_n)
}

team_strength <- compute_season_strength(historical)
write_csv(team_strength, file.path(PROCESSED_DIR, "team_strength.csv"))
message(glue("  → {nrow(team_strength)} team-season strength rows saved."))

# ── 5. Match feature matrix (model training data) ────────────────────────────
# Wide format: one row per historical match with pre-match features for both
# teams and the observed goals / result as targets.

message("Assembling match feature matrix...")

home_form <- team_long |>
  filter(is_home) |>
  select(match_id,
         home_form_pts = form_pts_5, home_form_gf = form_gf_5,
         home_form_ga  = form_ga_5,  home_form_gd  = form_gd_5,
         home_form_n   = form_n)

away_form <- team_long |>
  filter(!is_home) |>
  select(match_id,
         away_form_pts = form_pts_5, away_form_gf = form_gf_5,
         away_form_ga  = form_ga_5,  away_form_gd  = form_gd_5,
         away_form_n   = form_n)

# Use the same season's strength parameters for each match
home_str <- team_strength |>
  select(team, competition, season,
         home_att, home_def, lg_avg_home, lg_avg_away)

away_str <- team_strength |>
  select(team, competition, season, away_att, away_def)

match_features <- historical |>
  select(match_id, competition, season, stage, matchday, utc_date,
         home_team, away_team, home_goals, away_goals, result) |>
  left_join(home_form, by = "match_id") |>
  left_join(away_form, by = "match_id") |>
  left_join(home_str,  by = c("home_team" = "team", "competition", "season")) |>
  left_join(away_str,  by = c("away_team" = "team", "competition", "season")) |>
  mutate(
    # Pre-match expected goals under the Dixon-Coles formulation
    expected_home = lg_avg_home * home_att * away_def,
    expected_away = lg_avg_away * away_att * home_def
  )

write_csv(match_features, file.path(PROCESSED_DIR, "match_features.csv"))
message(glue("  → {nrow(match_features)} rows, {ncol(match_features)} columns saved."))

# ── 6. Team name normalisation (football-data.org ↔ The Odds API) ────────────
# football-data.org: "Arsenal FC"  /  The Odds API: "Arsenal"

normalize_name <- function(x) {
  x |>
    str_replace_all("\\bFC$",  "") |>
    str_replace_all("^FC\\b",  "") |>
    str_replace_all("\\bCF$",  "") |>
    str_replace_all("\\bAFC$", "") |>
    str_replace_all("\\bSC$",  "") |>
    str_trim() |>
    str_to_lower()
}

# ── 7. Upcoming fixture features + odds ──────────────────────────────────────

message("Building upcoming fixture features...")

# Current form: most recent form snapshot per team
current_form <- team_long |>
  group_by(team) |>
  slice_max(utc_date, n = 1, with_ties = FALSE) |>
  ungroup() |>
  select(team,
         form_pts = form_pts_5, form_gf = form_gf_5,
         form_ga  = form_ga_5,  form_gd  = form_gd_5)

# Most recent season's strength parameters per team per competition
current_strength <- team_strength |>
  group_by(team, competition) |>
  slice_max(season, n = 1, with_ties = FALSE) |>
  ungroup()

# Normalise names for joining
odds_norm <- odds |>
  mutate(
    home_norm = normalize_name(home_team),
    away_norm = normalize_name(away_team)
  )

upcoming_norm <- upcoming |>
  mutate(
    home_norm = normalize_name(home_team),
    away_norm = normalize_name(away_team)
  )

upcoming_features <- upcoming_norm |>
  # Attach odds
  left_join(
    odds_norm |>
      select(home_norm, away_norm, competition,
             n_bookmakers,
             home_odds_avg, draw_odds_avg, away_odds_avg,
             implied_home_prob, implied_draw_prob, implied_away_prob),
    by = c("home_norm", "away_norm", "competition")
  ) |>
  # Attach home team's current form
  left_join(
    current_form |> rename_with(~paste0("home_", .), .cols = -team),
    by = c("home_team" = "team")
  ) |>
  # Attach away team's current form
  left_join(
    current_form |> rename_with(~paste0("away_", .), .cols = -team),
    by = c("away_team" = "team")
  ) |>
  # Attach home team's strength parameters
  left_join(
    current_strength |>
      select(team, competition, home_att, home_def, lg_avg_home, lg_avg_away),
    by = c("home_team" = "team", "competition")
  ) |>
  # Attach away team's strength parameters
  left_join(
    current_strength |> select(team, competition, away_att, away_def),
    by = c("away_team" = "team", "competition")
  ) |>
  mutate(
    expected_home = lg_avg_home * home_att * away_def,
    expected_away = lg_avg_away * away_att * home_def
  ) |>
  select(-home_norm, -away_norm) |>
  arrange(utc_date)

write_csv(upcoming_features, file.path(PROCESSED_DIR, "upcoming_features.csv"))
message(glue("  → {nrow(upcoming_features)} upcoming fixtures saved."))

# ── 8. Diagnostics ───────────────────────────────────────────────────────────

odds_matched    <- sum(!is.na(upcoming_features$implied_home_prob))
odds_unmatched  <- sum(is.na(upcoming_features$implied_home_prob))
strength_missing <- sum(is.na(upcoming_features$home_att) | is.na(upcoming_features$away_att))

message(glue("\n--- Diagnostics ---"))
message(glue("Fixtures with odds:           {odds_matched}"))
message(glue("Fixtures WITHOUT odds:        {odds_unmatched}  (name mismatch or no market yet)"))
message(glue("Fixtures with missing strength: {strength_missing}  (team not in historical data)"))

if (odds_unmatched > 0) {
  message("\nUnmatched fixtures (no odds):")
  upcoming_features |>
    filter(is.na(implied_home_prob)) |>
    select(utc_date, competition, home_team, away_team) |>
    print(n = Inf)
}

message("\n=== Feature engineering complete ===")
message(glue("All outputs saved to {PROCESSED_DIR}/"))
