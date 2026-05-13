# R/06_ensemble.R
# Blends Dixon-Coles (Bayesian Poisson) and Elo/Logistic predictions, then
# computes how differently the two models see each fixture.
#
# Two questions answered here:
#   1. Where do the models agree? (high confidence zone)
#   2. Where do they diverge? (investigate before acting on value bets)
#
# Divergence metric: Total Variation (TV) distance between the two probability
# distributions over {H, D, A}. TV = 0.5 * sum(|p_i - q_i|), range [0, 1].
# TV = 0: models are identical. TV = 0.1: differ by ~10 pp on average per outcome.
#
# Ensemble method: weighted arithmetic average (equal weights to start).
# Weights can be tuned once we have evaluation data from 07_evaluate.R.
#
# Outputs:
#   outputs/ensemble_predictions.csv  — ensemble probs + divergence for all fixtures
#   outputs/ensemble_value_bets.csv   — ranked value bets from the ensemble

source("config.R")
source("00_setup.R")

OUTPUT_DIR <- "outputs"
run_date   <- format(Sys.Date(), "%m-%d-%Y")
dated_dir  <- file.path(OUTPUT_DIR, run_date)
dir.create(dated_dir, recursive = TRUE, showWarnings = FALSE)

DC_WEIGHT  <- 0.5   # relative weight for Dixon-Coles model
ELO_WEIGHT <- 0.5   # relative weight for Elo/logistic model
MIN_EDGE   <- 0.05  # minimum edge (probability pp) to flag as value

# ── 1. Load both model predictions ───────────────────────────────────────────

# Both files are written (possibly empty) by 04 and 05 even when there are no
# upcoming fixtures, so a missing file means those scripts haven't been run yet.
req <- c(file.path(OUTPUT_DIR, "all_predictions.csv"),
         file.path(OUTPUT_DIR, "elo_predictions.csv"))
missing <- req[!file.exists(req)]
if (length(missing) > 0) {
  stop("Ensemble requires these files (run 04 and 05 first):\n",
       paste(" ", missing, collapse = "\n"))
}

message("Loading model predictions...")
dc_preds  <- read_csv(file.path(OUTPUT_DIR, "all_predictions.csv"),  show_col_types = FALSE)
elo_preds <- read_csv(file.path(OUTPUT_DIR, "elo_predictions.csv"),  show_col_types = FALSE)

if (nrow(dc_preds) == 0 || nrow(elo_preds) == 0) {
  message("No upcoming fixtures to ensemble — writing empty output files.")
  write_csv(tibble(), file.path(dated_dir, "predictions.csv"))
  write_csv(tibble(), file.path(dated_dir, "full_output.csv"))
} else {

elo_slim <- elo_preds |>
  select(match_id,
         lr_home_prob, lr_draw_prob, lr_away_prob,
         elo_home, elo_away, elo_diff)

combined <- dc_preds |>
  left_join(elo_slim, by = "match_id") |>
  filter(!is.na(lr_home_prob))

message(glue("{nrow(combined)} fixtures with predictions from both models."))

# ── 2. Ensemble + divergence ──────────────────────────────────────────────────

most_likely <- function(h, d, a) {
  case_when(h >= d & h >= a ~ "Home",
            d >= a          ~ "Draw",
            TRUE            ~ "Away")
}

combined <- combined |>
  mutate(
    # Ensemble: weighted average of the two probability distributions
    ens_home_prob = DC_WEIGHT * model_home_prob + ELO_WEIGHT * lr_home_prob,
    ens_draw_prob = DC_WEIGHT * model_draw_prob + ELO_WEIGHT * lr_draw_prob,
    ens_away_prob = DC_WEIGHT * model_away_prob + ELO_WEIGHT * lr_away_prob,

    # Total Variation distance: how different are the two models' distributions?
    tv_distance = 0.5 * (abs(model_home_prob - lr_home_prob) +
                          abs(model_draw_prob - lr_draw_prob) +
                          abs(model_away_prob - lr_away_prob)),

    # Most likely outcome according to each model
    dc_pick  = most_likely(model_home_prob, model_draw_prob, model_away_prob),
    lr_pick  = most_likely(lr_home_prob,    lr_draw_prob,    lr_away_prob),
    ens_pick = most_likely(ens_home_prob,   ens_draw_prob,   ens_away_prob),

    # Do the models pick the same most-likely outcome?
    models_agree = dc_pick == lr_pick
  )

# ── 3. Model comparison table ─────────────────────────────────────────────────
# Sorted by divergence so the most disagreed-upon fixtures appear first.
# Fixtures where models pick DIFFERENT outcomes are flagged — these deserve
# extra scrutiny before acting on any value signal.

message("\n", strrep("=", 70))
message("MODEL COMPARISON  (sorted by divergence, highest first)")
message(strrep("=", 70))
message(glue(
  "Weights: Dixon-Coles = {DC_WEIGHT}, Elo/Logistic = {ELO_WEIGHT}\n"
))

fmt_pct <- function(x) sprintf("%5.1f%%", x * 100)

combined |>
  arrange(desc(tv_distance)) |>
  mutate(
    date    = format(as.Date(utc_date), "%d %b"),
    fixture = paste0(str_trunc(home_team, 16, "right"), " v ", str_trunc(away_team, 16, "right")),
    flag    = if_else(!models_agree, "  !!!", "")
  ) |>
  rowwise() |>
  mutate(
    row_out = glue(
      "{date}  {str_pad(competition, 4)}  {str_pad(fixture, 36)}",
      "  DC: {fmt_pct(model_home_prob)} / {fmt_pct(model_draw_prob)} / {fmt_pct(model_away_prob)}",
      "  ELO: {fmt_pct(lr_home_prob)} / {fmt_pct(lr_draw_prob)} / {fmt_pct(lr_away_prob)}",
      "  TV={round(tv_distance, 3)}{flag}"
    )
  ) |>
  ungroup() |>
  pull(row_out) |>
  walk(message)

n_disagree <- sum(!combined$models_agree)
message(glue(
  "\n{n_disagree} of {nrow(combined)} fixtures have models picking DIFFERENT most-likely outcomes (!!!)"
))
message("Header: DC probs = H% / D% / A%,  ELO probs = H% / D% / A%,  TV = total variation distance")

# ── 4. Where each model is more "confident" ───────────────────────────────────
# A model is more confident on a fixture if its max-outcome probability
# is higher than the other model's.

combined <- combined |>
  mutate(
    dc_max_prob  = pmax(model_home_prob, model_draw_prob, model_away_prob),
    lr_max_prob  = pmax(lr_home_prob,    lr_draw_prob,    lr_away_prob),
    more_confident = if_else(dc_max_prob >= lr_max_prob, "DC", "ELO")
  )

message(glue("\nDC more confident on:  {sum(combined$more_confident == 'DC')} fixtures"))
message(glue("ELO more confident on: {sum(combined$more_confident == 'ELO')} fixtures"))

# ── 5. Ensemble value bets ────────────────────────────────────────────────────

ensemble_bets <- bind_rows(
  combined |> transmute(
    utc_date, competition, home_team, away_team,
    bet_type     = "Home Win",
    odds         = home_odds_avg,
    ens_prob     = ens_home_prob,
    dc_prob      = model_home_prob,
    lr_prob      = lr_home_prob,
    implied_prob = implied_home_prob,
    tv_distance,
    models_agree,
    edge         = ens_home_prob - implied_home_prob,
    ev           = (ens_home_prob * home_odds_avg) - 1
  ),
  combined |> transmute(
    utc_date, competition, home_team, away_team,
    bet_type     = "Draw",
    odds         = draw_odds_avg,
    ens_prob     = ens_draw_prob,
    dc_prob      = model_draw_prob,
    lr_prob      = lr_draw_prob,
    implied_prob = implied_draw_prob,
    tv_distance,
    models_agree,
    edge         = ens_draw_prob - implied_draw_prob,
    ev           = (ens_draw_prob * draw_odds_avg) - 1
  ),
  combined |> transmute(
    utc_date, competition, home_team, away_team,
    bet_type     = "Away Win",
    odds         = away_odds_avg,
    ens_prob     = ens_away_prob,
    dc_prob      = model_away_prob,
    lr_prob      = lr_away_prob,
    implied_prob = implied_away_prob,
    tv_distance,
    models_agree,
    edge         = ens_away_prob - implied_away_prob,
    ev           = (ens_away_prob * away_odds_avg) - 1
  )
) |>
  filter(!is.na(implied_prob), edge > MIN_EDGE, ev > 0) |>
  arrange(desc(ev))

# ── 6. Full fixture EV table (all games, no threshold) ───────────────────────
# One row per fixture showing model probabilities and EV for all three outcomes.
# Sorted by best EV available in that game so you can browse any fixture.

fixture_ev <- combined |>
  mutate(
    ev_home = if_else(!is.na(home_odds_avg), (ens_home_prob * home_odds_avg) - 1, NA_real_),
    ev_draw = if_else(!is.na(draw_odds_avg), (ens_draw_prob * draw_odds_avg) - 1, NA_real_),
    ev_away = if_else(!is.na(away_odds_avg), (ens_away_prob * away_odds_avg) - 1, NA_real_),
    best_ev  = pmax(ev_home, ev_draw, ev_away, na.rm = TRUE),
    best_bet = case_when(
      !is.na(ev_home) & ev_home == best_ev ~ "Home",
      !is.na(ev_draw) & ev_draw == best_ev ~ "Draw",
      !is.na(ev_away) & ev_away == best_ev ~ "Away",
      TRUE ~ ens_pick  # no odds available — fall back to ensemble model's predicted outcome
    ),
    match_date = as.Date(utc_date)
  ) |>
  arrange(desc(best_ev))

# predictions.csv — clean summary for browsing
fmt_ev <- function(x) ifelse(is.na(x), NA_character_, sprintf("%.2f%%", x * 100))

fixture_ev |>
  mutate(
    ev_home = fmt_ev(ev_home),
    ev_draw = fmt_ev(ev_draw),
    ev_away = fmt_ev(ev_away),
    best_ev = fmt_ev(best_ev)
  ) |>
  select(
    competition, stage, match_date,
    home_team, away_team,
    dc_pick, lr_pick, ens_pick,
    models_agree, more_confident,
    ev_home, ev_draw, ev_away, best_ev, best_bet
  ) |>
  write_csv(file.path(dated_dir, "predictions.csv"))

# full_output.csv — every model column rounded, for historical reference
fixture_ev |>
  mutate(across(where(is.numeric), \(x) round(x, 3))) |>
  write_csv(file.path(dated_dir, "full_output.csv"))

message("\n", strrep("=", 70))
message("ALL FIXTURES — sorted by best available EV")
message(strrep("=", 70))
message("Columns: H%/D%/A% = ensemble model probs | mH/mD/mA = market implied probs")
message("         EV-H/D/A = expected value per £1 stake | Best = highest EV outcome\n")

fixture_ev |>
  mutate(
    date    = format(as.Date(utc_date), "%d %b"),
    fixture = paste0(str_trunc(home_team, 13, "right"), " v ", str_trunc(away_team, 13, "right")),
    `H%`    = paste0(round(ens_home_prob * 100), "%"),
    `D%`    = paste0(round(ens_draw_prob * 100), "%"),
    `A%`    = paste0(round(ens_away_prob * 100), "%"),
    mH      = paste0(round(implied_home_prob * 100), "%"),
    mD      = paste0(round(implied_draw_prob * 100), "%"),
    mA      = paste0(round(implied_away_prob * 100), "%"),
    `EV-H`  = ifelse(is.na(ev_home), "  n/a", sprintf("%+.1f%%", ev_home * 100)),
    `EV-D`  = ifelse(is.na(ev_draw), "  n/a", sprintf("%+.1f%%", ev_draw * 100)),
    `EV-A`  = ifelse(is.na(ev_away), "  n/a", sprintf("%+.1f%%", ev_away * 100)),
    Best    = if_else(is.na(best_bet), "-", best_bet)
  ) |>
  select(date, Comp = competition, fixture,
         `H%`, `D%`, `A%`, mH, mD, mA,
         `EV-H`, `EV-D`, `EV-A`, Best) |>
  as.data.frame() |>
  print(row.names = FALSE)

# ── 7. Value bet output ───────────────────────────────────────────────────────

message("\n", strrep("=", 70))
message(glue("ENSEMBLE VALUE BETS  (edge > {MIN_EDGE * 100}pp, EV > 0)"))
message(strrep("=", 70))

if (nrow(ensemble_bets) == 0) {
  message("No value bets found above threshold.")
} else {
  ensemble_bets |>
    mutate(
      date     = format(as.Date(utc_date), "%d %b"),
      fixture  = paste0(str_trunc(home_team, 14, "right"), " v ", str_trunc(away_team, 14, "right")),
      agree    = if_else(models_agree, "YES", " NO"),
      ens_pct  = paste0(round(ens_prob     * 100, 1), "%"),
      dc_pct   = paste0(round(dc_prob      * 100, 1), "%"),
      lr_pct   = paste0(round(lr_prob      * 100, 1), "%"),
      mkt_pct  = paste0(round(implied_prob * 100, 1), "%"),
      edge_pct = paste0("+", round(edge * 100, 1), "pp"),
      ev_pct   = paste0("+", round(ev   * 100, 1), "%"),
      tv       = round(tv_distance, 3)
    ) |>
    select(date, comp = competition, fixture, bet_type, odds,
           ens_pct, dc_pct, lr_pct, mkt_pct, edge_pct, ev_pct,
           agree, tv) |>
    as.data.frame() |>
    print(row.names = FALSE)

  message(glue("\nColumns: ens=ensemble, dc=Dixon-Coles, lr=Elo/logistic, mkt=market implied"))
  message("agree=YES: both models favour this outcome over the market")
  message("tv: total variation — higher means models are less aligned on this fixture")
}

# ── 7. Save ───────────────────────────────────────────────────────────────────

message(glue("\n=== Ensemble complete ==="))
message(glue("  predictions.csv  : {nrow(fixture_ev)} fixtures"))
message(glue("  full_output.csv  : {nrow(fixture_ev)} fixtures (all columns)"))
message(glue("  Folder: outputs/{run_date}/"))
message("\nNext steps:")
message("  - Run source('R/07_evaluate.R') after several matchweeks to assess model accuracy")
message("  - Adjust DC_WEIGHT / ELO_WEIGHT in this script based on evaluation results")

} # end if (nrow > 0)
