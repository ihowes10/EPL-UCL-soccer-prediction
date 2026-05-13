# R/05_elo_logistic.R
# Elo rating system + multinomial logistic regression for match outcome prediction.
#
# This model is deliberately independent from 04_dixon_coles.R so the two
# can disagree — disagreement between models is informative when evaluating
# prediction confidence and diagnosing where each model is weak.
#
# Pipeline:
#   1. Compute Elo ratings sequentially over all historical matches
#      - Home advantage applied as a +100-point bonus in the expected-score formula
#      - K-factor scales with goal difference (larger wins = more rating movement)
#   2. Fit a multinomial logistic regression: result ~ elo_diff + form features
#   3. Hold out the last 365 days as a test set — saves predictions for later evaluation
#   4. Predict upcoming fixtures using current Elo + current form
#
# Outputs:
#   data/processed/elo_history.csv    — pre-match Elo for every historical match
#   data/processed/elo_current.csv    — current rating for every team
#   models/elo_logistic.rds           — production model (fit on all data)
#   models/elo_logistic_eval.rds      — eval model (fit on training split only)
#   outputs/elo_predictions.csv       — upcoming fixture probabilities
#   outputs/lr_test_predictions.csv   — held-out test predictions for evaluation

source("config.R")
source("00_setup.R")
library(nnet)

PROCESSED_DIR <- file.path("data", "processed")
MODEL_DIR     <- "models"
OUTPUT_DIR    <- "outputs"

K_FACTOR       <- 20    # base Elo update speed
HOME_ADV_ELO   <- 100   # home team treated as +100 pts stronger for expected-score calc
ELO_INIT       <- 1500  # starting rating for all teams

# ── 1. Load data ──────────────────────────────────────────────────────────────

message("Loading processed features...")
match_features    <- read_csv(file.path(PROCESSED_DIR, "match_features.csv"),    show_col_types = FALSE)
upcoming_features <- read_csv(file.path(PROCESSED_DIR, "upcoming_features.csv"), show_col_types = FALSE)

# Elo must be computed in strict chronological order
historical_sorted <- match_features |>
  filter(!is.na(result), !is.na(home_goals)) |>
  arrange(utc_date)

message(glue("Computing Elo over {nrow(historical_sorted)} matches..."))

# ── 2. Elo computation ────────────────────────────────────────────────────────
# For each match we record the pre-match rating of both teams, then update.
# Goal-difference multiplier: a 3-0 win carries more signal than 1-0.
#   k_adj = K * (1 + 0.5 * log1p(|gd|))
#   gd=0 → k_adj=K, gd=1 → k_adj≈1.35K, gd=3 → k_adj≈1.69K

compute_elo <- function(matches, K = K_FACTOR, home_adv = HOME_ADV_ELO, init = ELO_INIT) {
  teams <- unique(c(matches$home_team, matches$away_team))
  elo   <- setNames(rep(init, length(teams)), teams)

  n             <- nrow(matches)
  pre_elo_home  <- numeric(n)
  pre_elo_away  <- numeric(n)

  for (i in seq_len(n)) {
    home <- matches$home_team[i]
    away <- matches$away_team[i]

    if (!home %in% names(elo)) elo[home] <- init
    if (!away %in% names(elo)) elo[away] <- init

    r_h <- elo[home]
    r_a <- elo[away]

    pre_elo_home[i] <- r_h
    pre_elo_away[i] <- r_a

    # Expected score for home team; home_adv shifts the effective rating
    e_h <- 1 / (1 + 10^((r_a - (r_h + home_adv)) / 400))

    # Actual score (1 = win, 0.5 = draw, 0 = loss)
    s_h <- switch(matches$result[i], "H" = 1, "D" = 0.5, "A" = 0, 0)

    # Goal-difference multiplier
    gd    <- abs(matches$home_goals[i] - matches$away_goals[i])
    k_adj <- K * (1 + 0.5 * log1p(gd))

    elo[home] <- r_h + k_adj * (s_h       - e_h)
    elo[away] <- r_a + k_adj * ((1 - s_h) - (1 - e_h))
  }

  list(
    pre_match = tibble(
      match_id = matches$match_id,
      elo_home = pre_elo_home,
      elo_away = pre_elo_away,
      elo_diff = pre_elo_home - pre_elo_away   # home advantage NOT included here
    ),
    current = tibble(
      team = names(elo),
      elo  = unname(elo)
    ) |> arrange(desc(elo))
  )
}

elo_result  <- compute_elo(historical_sorted)
current_elo <- elo_result$current

write_csv(elo_result$pre_match, file.path(PROCESSED_DIR, "elo_history.csv"))
write_csv(current_elo,          file.path(PROCESSED_DIR, "elo_current.csv"))

message("\n--- Current Top 20 Elo Ratings ---")
print(head(current_elo, 20))

# ── 3. Build training data ────────────────────────────────────────────────────

# Impute missing form values with league-average estimates rather than dropping rows.
# A team's first few matches have no form history — 1.5 pts/game and 0 GD are
# reasonable neutral priors for an "unknown" team.

train_data <- historical_sorted |>
  left_join(elo_result$pre_match, by = "match_id") |>
  mutate(
    result        = factor(result, levels = c("H", "D", "A")),
    competition   = factor(competition),
    home_form_pts = replace_na(home_form_pts, 1.5),
    away_form_pts = replace_na(away_form_pts, 1.5),
    home_form_gd  = replace_na(home_form_gd, 0),
    away_form_gd  = replace_na(away_form_gd, 0)
  ) |>
  filter(!is.na(elo_diff))

message(glue("Training rows after Elo join: {nrow(train_data)}"))

# ── 4. Temporal train / test split ───────────────────────────────────────────
# Last 365 days held out. We fit two versions of the model:
#   - eval model  : trained on the earlier split, tested on the held-out year
#   - production  : trained on all data (used for upcoming predictions)
# Both are saved so we can run formal evaluation later.

cutoff      <- max(train_data$utc_date) - lubridate::days(365)
train_split <- filter(train_data, utc_date <  cutoff)
test_split  <- filter(train_data, utc_date >= cutoff)

message(glue("Train split: {nrow(train_split)} matches | Test split: {nrow(test_split)} matches"))

# ── 5. Multinomial logistic regression ───────────────────────────────────────
# elo_diff is by far the strongest predictor; form features add marginal signal.
# competition adjusts for PL vs CL baseline win/draw/loss rates.

lr_formula <- result ~ elo_diff +
              home_form_pts + away_form_pts +
              home_form_gd  + away_form_gd  +
              competition

message("Fitting logistic regression (eval split)...")
fit_lr_eval <- multinom(lr_formula, data = train_split, trace = FALSE)

message("Fitting logistic regression (full data)...")
fit_lr      <- multinom(lr_formula, data = train_data,  trace = FALSE)

saveRDS(fit_lr,      file.path(MODEL_DIR, "elo_logistic.rds"))
saveRDS(fit_lr_eval, file.path(MODEL_DIR, "elo_logistic_eval.rds"))

message("Models saved.")

# ── 6. Evaluation on held-out test set ───────────────────────────────────────
# Log-loss: proper scoring rule. Lower = better.
#   Naive baseline (always predict league-average probs) ≈ log(3) ≈ 1.099
# Brier score: mean squared error of predicted probs vs one-hot outcome.
#   Range [0, 2]; lower = better; random baseline = 0.667

log_loss <- function(prob_mat, actuals) {
  idx          <- cbind(seq_len(nrow(prob_mat)), as.integer(actuals))
  correct_prob <- prob_mat[idx]
  -mean(log(pmax(correct_prob, 1e-15)))
}

brier_score <- function(prob_mat, actuals) {
  n       <- nrow(prob_mat)
  k       <- ncol(prob_mat)
  one_hot <- matrix(0, n, k)
  one_hot[cbind(seq_len(n), as.integer(actuals))] <- 1
  mean(rowSums((prob_mat - one_hot)^2))
}

test_probs <- predict(fit_lr_eval, newdata = test_split, type = "probs")

ll <- log_loss(test_probs, test_split$result)
bs <- brier_score(test_probs, test_split$result)

pred_class  <- levels(test_split$result)[apply(test_probs, 1, which.max)]
accuracy    <- mean(pred_class == as.character(test_split$result))

message(glue("\n--- Held-out Evaluation ({nrow(test_split)} matches, last 365 days) ---"))
message(glue("Log-loss    : {round(ll, 4)}  (naive baseline: {round(log(3), 4)})"))
message(glue("Brier score : {round(bs, 4)}  (random baseline: 0.667)"))
message(glue("Accuracy    : {round(accuracy * 100, 1)}%  (most-likely outcome correct)"))

# Save test predictions — used by the evaluation script to compare models
test_eval_out <- test_split |>
  select(match_id, utc_date, competition, home_team, away_team, result) |>
  bind_cols(
    as_tibble(test_probs) |>
      rename(lr_prob_H = H, lr_prob_D = D, lr_prob_A = A)
  )
write_csv(test_eval_out, file.path(OUTPUT_DIR, "lr_test_predictions.csv"))
message(glue("Test predictions saved to {file.path(OUTPUT_DIR, 'lr_test_predictions.csv')}"))

# ── 7. Predict upcoming fixtures ──────────────────────────────────────────────

if (nrow(upcoming_features) == 0) {
  message("No upcoming fixtures — writing empty output file.")
  write_csv(tibble(), file.path(OUTPUT_DIR, "elo_predictions.csv"))
} else {

message(glue("\nPredicting {nrow(upcoming_features)} upcoming fixtures..."))

upcoming_elo <- upcoming_features |>
  left_join(current_elo |> rename(elo_home = elo), by = c("home_team" = "team")) |>
  left_join(current_elo |> rename(elo_away = elo), by = c("away_team" = "team")) |>
  mutate(
    # Teams not yet in Elo system get the neutral starting rating
    elo_home      = replace_na(elo_home, ELO_INIT),
    elo_away      = replace_na(elo_away, ELO_INIT),
    elo_diff      = elo_home - elo_away,
    competition   = factor(
      if_else(as.character(competition) %in% levels(train_data$competition),
              as.character(competition), "PL"),
      levels = levels(train_data$competition)
    ),
    home_form_pts = replace_na(home_form_pts, 1.5),
    away_form_pts = replace_na(away_form_pts, 1.5),
    home_form_gd  = replace_na(home_form_gd, 0),
    away_form_gd  = replace_na(away_form_gd, 0)
  )

probs_upcoming <- predict(fit_lr, newdata = upcoming_elo, type = "probs")

elo_predictions <- upcoming_features |>
  mutate(
    lr_home_prob = probs_upcoming[, "H"],
    lr_draw_prob = probs_upcoming[, "D"],
    lr_away_prob = probs_upcoming[, "A"],
    elo_home     = upcoming_elo$elo_home,
    elo_away     = upcoming_elo$elo_away,
    elo_diff     = upcoming_elo$elo_diff
  )

write_csv(elo_predictions, file.path(OUTPUT_DIR, "elo_predictions.csv"))

message(glue("\n=== Elo + Logistic model complete ==="))
message(glue("Predictions saved for {nrow(elo_predictions)} upcoming fixtures."))
message("\nNext step: source('R/06_ensemble.R') to blend DC and Elo predictions.")

} # end if (nrow(upcoming_features) > 0)
