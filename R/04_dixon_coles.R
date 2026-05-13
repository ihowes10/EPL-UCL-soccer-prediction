# R/04_dixon_coles.R
# Bayesian bivariate Poisson model for soccer prediction (Dixon-Coles style).
#
# Model structure:
#   log(E[home_goals]) = b0_home + b_competition + att[home_team] + def[away_team]
#   log(E[away_goals]) = b0_away + b_competition + att[away_team] + def[home_team]
#
#   - b0_home / b0_away     : overall scoring intercepts (captures home advantage implicitly)
#   - b_competition         : adjusts for CL vs PL average goal rates
#   - att[k], def[k]        : zero-mean random effects per team
#   - Matches are weighted by exponential time-decay (half-life = 1 year)
#     so that form from 3 seasons ago counts far less than last month.
#
# Value identification:
#   edge = model P(outcome) - market-implied P(outcome)
#   EV   = model P(outcome) * decimal_odds - 1
#   A bet is flagged as "value" when edge > MIN_EDGE and EV > 0.

source("config.R")
source("00_setup.R")

PROCESSED_DIR <- file.path("data", "processed")
MODEL_DIR     <- "models"
OUTPUT_DIR    <- "outputs"
dir.create(MODEL_DIR,  recursive = TRUE, showWarnings = FALSE)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

N_SIM    <- 4000   # posterior draws for scoreline simulation
MIN_EDGE <- 0.05   # minimum probability edge to flag a value bet (5 pp)
HALF_LIFE_DAYS <- 365  # time-decay: a match 1 year old has 50% weight

# ── 1. Load data ──────────────────────────────────────────────────────────────

message("Loading processed features...")
match_features    <- read_csv(file.path(PROCESSED_DIR, "match_features.csv"),    show_col_types = FALSE)
upcoming_features <- read_csv(file.path(PROCESSED_DIR, "upcoming_features.csv"), show_col_types = FALSE)

# ── 2. Prepare training data ──────────────────────────────────────────────────

train_data <- match_features |>
  filter(!is.na(home_goals), !is.na(away_goals)) |>
  rename(homegoals = home_goals, awaygoals = away_goals) |>
  mutate(
    competition = factor(competition),
    # Exponential time-decay: older matches count less
    days_ago     = as.numeric(Sys.Date() - as.Date(utc_date)),
    decay_weight = exp(-log(2) * days_ago / HALF_LIFE_DAYS),
    # Separate grouping columns for each team-role random effect
    att_home = home_team,
    def_away = away_team,
    att_away = away_team,
    def_home = home_team
  )

message(glue(
  "Training data: {nrow(train_data)} matches | ",
  "{n_distinct(c(train_data$home_team, train_data$away_team))} teams | ",
  "competitions: {paste(levels(train_data$competition), collapse = ', ')}"
))

# ── 3. Model specification ────────────────────────────────────────────────────
# Bivariate Poisson (home goals and away goals modelled as independent
# Poisson given their rates — the standard Dixon-Coles approximation).
#
# Random effects:
#   att_home / att_away : how many goals a team scores (attack strength)
#   def_home / def_away : how many goals a team concedes (defence quality,
#                         parameterised as extra goals allowed, so lower = better defence)

home_bf <- bf(
  homegoals | weights(decay_weight) ~ 1 + competition + (1 | att_home) + (1 | def_away),
  family = poisson()
)
away_bf <- bf(
  awaygoals | weights(decay_weight) ~ 1 + competition + (1 | att_away) + (1 | def_home),
  family = poisson()
)

# Weakly informative priors:
#   Intercept ~ N(0, 1) on log scale: implied avg goals in [0.37, 2.72] — sensible range
#   sd        ~ N(0, 0.5) half-normal: prevents extreme team effects
#   b         ~ N(0, 0.5): competition effect (CL vs PL)
priors <- c(
  prior(normal(0, 1),   class = Intercept, resp = homegoals),
  prior(normal(0, 1),   class = Intercept, resp = awaygoals),
  prior(normal(0, 0.5), class = b,         resp = homegoals),
  prior(normal(0, 0.5), class = b,         resp = awaygoals),
  prior(normal(0, 0.5), class = sd,        resp = homegoals),
  prior(normal(0, 0.5), class = sd,        resp = awaygoals)
)

# ── 4. Fit (or reload from disk cache) ───────────────────────────────────────

model_path <- file.path(MODEL_DIR, "dixon_coles_brms.rds")

if (file.exists(model_path)) {
  message("Loading cached model from ", model_path, " ...")
  fit_dc <- readRDS(model_path)
} else {
  message("Fitting Bayesian model — Stan compilation + sampling (~5-10 min on first run)...")
  fit_dc <- brm(
    formula = home_bf + away_bf + set_rescor(FALSE),
    data    = train_data,
    prior   = priors,
    chains  = 4,
    iter    = 2000,
    warmup  = 1000,
    cores   = min(4L, parallel::detectCores()),
    seed    = 42,
    control = list(adapt_delta = 0.95)
  )
  saveRDS(fit_dc, model_path)
  message("Model saved to ", model_path)
}

# ── 5. Convergence diagnostics ────────────────────────────────────────────────
# R-hat should be < 1.01 and bulk ESS > 400 for reliable inference.

diag_fixed <- as.data.frame(summary(fit_dc)$fixed)
max_rhat   <- max(diag_fixed$Rhat,     na.rm = TRUE)
min_ess    <- min(diag_fixed$Bulk_ESS, na.rm = TRUE)

message("\n--- Convergence Diagnostics ---")
message(glue("Max R-hat    : {round(max_rhat, 4)}  (want < 1.01)"))
message(glue("Min Bulk ESS : {round(min_ess, 0)}  (want > 400)"))

if (max_rhat > 1.01) warning("R-hat > 1.01 — chains may not have converged. Consider more iterations.")
if (min_ess  < 400)  warning("Low ESS — consider increasing iter or thinning.")

# Print top team effects for a sense-check.
# ranef() returns a 3D array [teams, stats, terms]; rownames() is unreliable
# on 3D arrays — use dimnames()[[1]] for team names and as.numeric() to strip
# array attributes before handing to dplyr.
home_att <- ranef(fit_dc)$homegoals$att_home

message("\n--- Top 10 Home Attack Strengths ---")
tibble(
  team     = dimnames(home_att)[[1]],
  estimate = as.numeric(home_att[, "Estimate", 1])
) |>
  arrange(desc(estimate)) |>
  head(10) |>
  mutate(estimate = round(estimate, 3)) |>
  print()

# ── 6. Predict all upcoming fixtures ─────────────────────────────────────────
# All fixtures are passed to posterior_linpred in one call (vectorised).
# allow_new_levels = TRUE: teams not seen in training sample their effect
# from the prior (league-average ability with calibrated uncertainty).

if (nrow(upcoming_features) == 0) {
  message("No upcoming fixtures found — writing empty output files.")
  write_csv(tibble(), file.path(OUTPUT_DIR, "all_predictions.csv"))
  write_csv(tibble(), file.path(OUTPUT_DIR, "value_bets.csv"))
} else {

message(glue("\nPredicting {nrow(upcoming_features)} upcoming fixtures..."))

pred_data <- upcoming_features |>
  mutate(
    homegoals    = 0L,
    awaygoals    = 0L,
    decay_weight = 1,
    # Cup competitions (FAC, ELC) aren't in training data — map them to PL
    # so the model uses PL-level goal rates rather than erroring on unknown levels.
    # as.character() is required first: ifelse() on a factor returns integer codes.
    competition  = factor(
      if_else(as.character(competition) %in% levels(train_data$competition),
              as.character(competition), "PL"),
      levels = levels(train_data$competition)
    ),
    att_home     = home_team,
    def_away     = away_team,
    att_away     = away_team,
    def_home     = home_team
  )

# Draw posterior expected-goal rates for all fixtures simultaneously
lambda_mat <- posterior_linpred(
  fit_dc,
  newdata           = pred_data,
  resp              = "homegoals",
  transform         = TRUE,   # exp(linear predictor) → λ
  allow_new_levels  = TRUE,
  sample_new_levels = "uncertainty",
  ndraws            = N_SIM
)   # shape: (N_SIM × n_fixtures)

mu_mat <- posterior_linpred(
  fit_dc,
  newdata           = pred_data,
  resp              = "awaygoals",
  transform         = TRUE,
  allow_new_levels  = TRUE,
  sample_new_levels = "uncertainty",
  ndraws            = N_SIM
)

n_fix <- nrow(pred_data)

# Simulate one scoreline per (draw × fixture) cell
sim_home <- matrix(rpois(N_SIM * n_fix, as.vector(lambda_mat)), nrow = N_SIM)
sim_away <- matrix(rpois(N_SIM * n_fix, as.vector(mu_mat)),     nrow = N_SIM)

predictions <- upcoming_features |>
  mutate(
    model_home_prob = colMeans(sim_home > sim_away),
    model_draw_prob = colMeans(sim_home == sim_away),
    model_away_prob = colMeans(sim_home < sim_away),
    exp_home_goals  = colMeans(lambda_mat),
    exp_away_goals  = colMeans(mu_mat),
    # 90% credible interval on expected home goals (shows model uncertainty)
    exp_home_lo     = apply(lambda_mat, 2, quantile, 0.05),
    exp_home_hi     = apply(lambda_mat, 2, quantile, 0.95)
  )

# ── 7. Value calculation ──────────────────────────────────────────────────────

predictions <- predictions |>
  mutate(
    edge_home = model_home_prob - implied_home_prob,
    edge_draw = model_draw_prob - implied_draw_prob,
    edge_away = model_away_prob - implied_away_prob,
    ev_home   = (model_home_prob * home_odds_avg) - 1,
    ev_draw   = (model_draw_prob * draw_odds_avg) - 1,
    ev_away   = (model_away_prob * away_odds_avg) - 1
  )

# Reshape to one row per bet type for clean output
value_bets <- bind_rows(
  predictions |> transmute(
    utc_date, competition, home_team, away_team, exp_home_goals, exp_away_goals,
    bet_type = "Home Win",
    odds     = home_odds_avg,
    model_prob   = model_home_prob,
    implied_prob = implied_home_prob,
    edge     = edge_home,
    ev       = ev_home
  ),
  predictions |> transmute(
    utc_date, competition, home_team, away_team, exp_home_goals, exp_away_goals,
    bet_type = "Draw",
    odds     = draw_odds_avg,
    model_prob   = model_draw_prob,
    implied_prob = implied_draw_prob,
    edge     = edge_draw,
    ev       = ev_draw
  ),
  predictions |> transmute(
    utc_date, competition, home_team, away_team, exp_home_goals, exp_away_goals,
    bet_type = "Away Win",
    odds     = away_odds_avg,
    model_prob   = model_away_prob,
    implied_prob = implied_away_prob,
    edge     = edge_away,
    ev       = ev_away
  )
) |>
  filter(!is.na(implied_prob), edge > MIN_EDGE, ev > 0) |>
  arrange(desc(ev))

# ── 8. Save outputs ───────────────────────────────────────────────────────────

write_csv(predictions, file.path(OUTPUT_DIR, "all_predictions.csv"))
write_csv(value_bets,  file.path(OUTPUT_DIR, "value_bets.csv"))

message(glue("\n=== Dixon-Coles Model Complete ==="))
message(glue("Full predictions : {nrow(predictions)} fixtures → {file.path(OUTPUT_DIR, 'all_predictions.csv')}"))
message(glue("Value bets found : {nrow(value_bets)} (edge > {MIN_EDGE * 100}pp, EV > 0) → {file.path(OUTPUT_DIR, 'value_bets.csv')}"))

if (nrow(value_bets) > 0) {
  message("\n--- Top value bets (ranked by Expected Value) ---")
  value_bets |>
    mutate(
      utc_date     = format(as.Date(utc_date), "%d %b"),
      model_prob   = paste0(round(model_prob   * 100, 1), "%"),
      implied_prob = paste0(round(implied_prob * 100, 1), "%"),
      edge         = paste0(round(edge         * 100, 1), "pp"),
      ev           = paste0("+", round(ev * 100, 1), "%"),
      odds         = round(odds, 2),
      exp_goals    = paste0(round(exp_home_goals, 2), " - ", round(exp_away_goals, 2))
    ) |>
    select(utc_date, competition, home_team, away_team,
           bet_type, odds, model_prob, implied_prob, edge, ev, exp_goals) |>
    print(n = 20)
} else {
  message("No value bets found above the threshold. Try lowering MIN_EDGE or waiting for more fixtures.")
}

} # end if (nrow(upcoming_features) > 0)
