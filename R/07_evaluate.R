# R/07_evaluate.R
# Evaluates and compares all three models on held-out test data.
#
# Test period: last 365 days of historical data (same cutoff as 05_elo_logistic.R).
#
# A note on fairness:
#   Elo/Logistic — true out-of-sample: trained on data BEFORE the test period.
#   Dixon-Coles  — slightly optimistic: the fitted model saw test-period matches.
#                  Set USE_LOO = TRUE for the fully rigorous DC evaluation via
#                  PSIS-LOO (adds ~5-10 min but gives a genuine out-of-sample score).
#
# Metrics
#   Log-loss   : proper scoring rule; lower = better; naive baseline ≈ 1.099
#   Brier score: mean squared prob error; lower = better; naive baseline = 0.667
#   Accuracy   : fraction of matches where the top-probability outcome was correct
#
# Outputs
#   outputs/dc_test_predictions.csv
#   outputs/evaluation_summary.csv
#   outputs/plots/calibration.png     — reliability diagrams for both models
#   outputs/plots/model_agreement.png — DC vs Elo probability scatter

source("config.R")
source("00_setup.R")
library(brms)

PROCESSED_DIR <- file.path("data", "processed")
OUTPUT_DIR    <- "outputs"
PLOTS_DIR     <- file.path(OUTPUT_DIR, "plots")
dir.create(PLOTS_DIR, recursive = TRUE, showWarnings = FALSE)

USE_LOO    <- FALSE  # TRUE = proper LOO-CV for DC (~5-10 min extra)
N_SIM      <- 2000   # posterior draws for DC test predictions
DC_WEIGHT  <- 0.5
ELO_WEIGHT <- 0.5

# ── 1. Load data and models ───────────────────────────────────────────────────

message("Loading data and models...")
match_features <- read_csv(file.path(PROCESSED_DIR, "match_features.csv"), show_col_types = FALSE)
lr_test        <- read_csv(file.path(OUTPUT_DIR, "lr_test_predictions.csv"), show_col_types = FALSE)
fit_dc         <- readRDS(file.path("models", "dixon_coles_brms.rds"))

# Reproduce the same cutoff as 05_elo_logistic.R
cutoff      <- max(filter(match_features, !is.na(result))$utc_date, na.rm = TRUE) - days(365)
test_matches <- match_features |>
  filter(!is.na(result), !is.na(home_goals), utc_date >= cutoff)

message(glue("Test period: {format(as.Date(cutoff), '%d %b %Y')} to present — {nrow(test_matches)} matches"))

# ── 2. Generate DC predictions on the test period ─────────────────────────────

message("Generating DC posterior predictions on test set (this may take a minute)...")

dc_test_data <- test_matches |>
  rename(homegoals = home_goals, awaygoals = away_goals) |>
  mutate(
    competition  = factor(competition),
    decay_weight = 1,
    att_home     = home_team,
    def_away     = away_team,
    att_away     = away_team,
    def_home     = home_team
  )

lambda_mat <- posterior_linpred(
  fit_dc, newdata = dc_test_data, resp = "homegoals",
  transform = TRUE, allow_new_levels = TRUE, ndraws = N_SIM
)
mu_mat <- posterior_linpred(
  fit_dc, newdata = dc_test_data, resp = "awaygoals",
  transform = TRUE, allow_new_levels = TRUE, ndraws = N_SIM
)

n_test   <- nrow(dc_test_data)
sim_home <- matrix(rpois(N_SIM * n_test, as.vector(lambda_mat)), nrow = N_SIM)
sim_away <- matrix(rpois(N_SIM * n_test, as.vector(mu_mat)),     nrow = N_SIM)

dc_test_preds <- test_matches |>
  select(match_id, utc_date, competition, home_team, away_team, result) |>
  mutate(
    dc_prob_H = colMeans(sim_home > sim_away),
    dc_prob_D = colMeans(sim_home == sim_away),
    dc_prob_A = colMeans(sim_home < sim_away)
  )

write_csv(dc_test_preds, file.path(OUTPUT_DIR, "dc_test_predictions.csv"))

# ── 3. Combine and compute ensemble on test set ───────────────────────────────

test_eval <- dc_test_preds |>
  inner_join(
    lr_test |> select(match_id, lr_prob_H, lr_prob_D, lr_prob_A),
    by = "match_id"
  ) |>
  mutate(
    result    = factor(result, levels = c("H", "D", "A")),
    ens_prob_H = DC_WEIGHT * dc_prob_H + ELO_WEIGHT * lr_prob_H,
    ens_prob_D = DC_WEIGHT * dc_prob_D + ELO_WEIGHT * lr_prob_D,
    ens_prob_A = DC_WEIGHT * dc_prob_A + ELO_WEIGHT * lr_prob_A
  )

message(glue("Matched {nrow(test_eval)} matches across both models for evaluation."))

# ── 4. Metric functions ───────────────────────────────────────────────────────

log_loss <- function(prob_mat, actuals) {
  idx          <- cbind(seq_len(nrow(prob_mat)), as.integer(actuals))
  -mean(log(pmax(prob_mat[idx], 1e-15)))
}

brier_score <- function(prob_mat, actuals) {
  n       <- nrow(prob_mat)
  k       <- ncol(prob_mat)
  one_hot <- matrix(0, n, k)
  one_hot[cbind(seq_len(n), as.integer(actuals))] <- 1
  mean(rowSums((prob_mat - one_hot)^2))
}

accuracy <- function(prob_mat, actuals) {
  pred_class <- levels(actuals)[apply(prob_mat, 1, which.max)]
  mean(pred_class == as.character(actuals))
}

eval_model <- function(prob_H, prob_D, prob_A, actuals) {
  mat <- cbind(prob_H, prob_D, prob_A)
  tibble(
    log_loss    = round(log_loss(mat, actuals),    4),
    brier       = round(brier_score(mat, actuals), 4),
    accuracy    = round(accuracy(mat, actuals),    4)
  )
}

# ── 5. Summary table ──────────────────────────────────────────────────────────

n_H <- sum(test_eval$result == "H")
n_D <- sum(test_eval$result == "D")
n_A <- sum(test_eval$result == "A")

# Naive baseline: always predict observed class frequencies
naive_H <- n_H / nrow(test_eval)
naive_D <- n_D / nrow(test_eval)
naive_A <- n_A / nrow(test_eval)
naive_mat <- matrix(c(naive_H, naive_D, naive_A), nrow = nrow(test_eval), ncol = 3, byrow = TRUE)

summary_table <- bind_rows(
  eval_model(test_eval$dc_prob_H,  test_eval$dc_prob_D,  test_eval$dc_prob_A,  test_eval$result) |> mutate(model = "Dixon-Coles (in-sample)", .before = 1),
  eval_model(test_eval$lr_prob_H,  test_eval$lr_prob_D,  test_eval$lr_prob_A,  test_eval$result) |> mutate(model = "Elo/Logistic (out-of-sample)", .before = 1),
  eval_model(test_eval$ens_prob_H, test_eval$ens_prob_D, test_eval$ens_prob_A, test_eval$result) |> mutate(model = "Ensemble (mixed)", .before = 1),
  tibble(model = "Naive baseline",
         log_loss = round(log_loss(naive_mat, test_eval$result), 4),
         brier    = round(brier_score(naive_mat, test_eval$result), 4),
         accuracy = round(accuracy(naive_mat, test_eval$result), 4))
)

message("\n", strrep("=", 60))
message(glue("EVALUATION SUMMARY  (n = {nrow(test_eval)} matches)"))
message(glue("Outcome distribution:  H={n_H}  D={n_D}  A={n_A}"))
message(strrep("=", 60))
print(as.data.frame(summary_table), row.names = FALSE)
message("\nlog-loss: lower = better | naive baseline ≈ ", round(log(3), 4))
message("brier:    lower = better | random = 0.667")
message("accuracy: higher = better (most-likely outcome correct)")
message("\nNote: DC is evaluated slightly in-sample. Set USE_LOO=TRUE for rigorous DC scoring.")

write_csv(summary_table, file.path(OUTPUT_DIR, "evaluation_summary.csv"))

# ── 6. Optional LOO-CV for DC model ──────────────────────────────────────────

if (USE_LOO) {
  message("\nRunning PSIS-LOO for Dixon-Coles (this takes several minutes)...")
  loo_dc <- loo(fit_dc)
  message("\nLOO-CV result for Dixon-Coles:")
  print(loo_dc)
}

# ── 7. Calibration plots ──────────────────────────────────────────────────────
# Reliability diagrams: if the model says 70%, does it happen 70% of the time?
# Points on the diagonal = perfect calibration.
# Points above diagonal = model underestimates (outcome more frequent than predicted).
# Points below diagonal = model overestimates.

message("\nGenerating calibration plots...")

calibrate_model <- function(probs_H, probs_D, probs_A, actuals, model_name, n_bins = 8) {
  bind_rows(
    tibble(predicted = probs_H, actual = as.integer(actuals == "H"), outcome = "Home Win"),
    tibble(predicted = probs_D, actual = as.integer(actuals == "D"), outcome = "Draw"),
    tibble(predicted = probs_A, actual = as.integer(actuals == "A"), outcome = "Away Win")
  ) |>
    mutate(
      bin = cut(predicted,
                breaks = quantile(predicted, seq(0, 1, length.out = n_bins + 1), na.rm = TRUE),
                include.lowest = TRUE)
    ) |>
    group_by(outcome, bin) |>
    summarise(n = n(), mean_pred = mean(predicted), obs_freq = mean(actual), .groups = "drop") |>
    mutate(model = model_name)
}

calib_dc  <- calibrate_model(test_eval$dc_prob_H, test_eval$dc_prob_D, test_eval$dc_prob_A, test_eval$result, "Dixon-Coles")
calib_lr  <- calibrate_model(test_eval$lr_prob_H, test_eval$lr_prob_D, test_eval$lr_prob_A, test_eval$result, "Elo/Logistic")
calib_ens <- calibrate_model(test_eval$ens_prob_H,test_eval$ens_prob_D,test_eval$ens_prob_A,test_eval$result, "Ensemble")

calib_all <- bind_rows(calib_dc, calib_lr, calib_ens) |>
  mutate(outcome = factor(outcome, levels = c("Home Win", "Draw", "Away Win")))

calib_plot <- ggplot(calib_all, aes(x = mean_pred, y = obs_freq, colour = model)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "grey50", linewidth = 0.7) +
  geom_line(linewidth = 0.8) +
  geom_point(aes(size = n)) +
  scale_colour_manual(values = c("Dixon-Coles" = "#2166ac", "Elo/Logistic" = "#d6604d", "Ensemble" = "#4dac26")) +
  scale_size_continuous(range = c(1.5, 5), guide = "none") +
  scale_x_continuous(labels = scales::percent_format(1), limits = c(0, 1)) +
  scale_y_continuous(labels = scales::percent_format(1), limits = c(0, 1)) +
  facet_wrap(~outcome, ncol = 3) +
  labs(
    title    = "Calibration (Reliability Diagrams)",
    subtitle = glue("Test period: {format(as.Date(cutoff), '%d %b %Y')} – present  |  n = {nrow(test_eval)} matches"),
    x        = "Predicted probability",
    y        = "Observed frequency",
    colour   = "Model",
    caption  = "Points above the diagonal: model underestimates this outcome.\nPoint size ∝ number of matches in bin."
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(file.path(PLOTS_DIR, "calibration.png"), calib_plot, width = 11, height = 5, dpi = 150)
message(glue("  → calibration.png saved"))

# ── 8. Model agreement scatter ────────────────────────────────────────────────
# Where do DC and Elo agree or diverge on the test set?
# Each point is a match. Colour = actual result.

agree_long <- test_eval |>
  select(match_id, result,
         home_dc = dc_prob_H, home_lr = lr_prob_H,
         draw_dc = dc_prob_D, draw_lr = lr_prob_D,
         away_dc = dc_prob_A, away_lr = lr_prob_A) |>
  pivot_longer(
    cols      = -c(match_id, result),
    names_to  = c("outcome", "model"),
    names_sep = "_"
  ) |>
  pivot_wider(names_from = model, values_from = value) |>
  mutate(
    outcome = recode(outcome, home = "Home Win", draw = "Draw", away = "Away Win"),
    outcome = factor(outcome, levels = c("Home Win", "Draw", "Away Win")),
    correct = case_when(
      outcome == "Home Win" & result == "H" ~ "Correct",
      outcome == "Draw"     & result == "D" ~ "Correct",
      outcome == "Away Win" & result == "A" ~ "Correct",
      TRUE ~ "Wrong"
    )
  )

agree_plot <- ggplot(agree_long, aes(x = dc, y = lr, colour = correct)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "grey60") +
  geom_point(alpha = 0.45, size = 1.2) +
  scale_colour_manual(values = c("Correct" = "#1a9850", "Wrong" = "#d73027")) +
  scale_x_continuous(labels = scales::percent_format(1), limits = c(0, 1)) +
  scale_y_continuous(labels = scales::percent_format(1), limits = c(0, 1)) +
  facet_wrap(~outcome, ncol = 3) +
  labs(
    title    = "Where DC and Elo/Logistic agree vs diverge (test set)",
    subtitle = "Points on the diagonal = identical predictions. Colour = whether the outcome occurred.",
    x        = "Dixon-Coles probability",
    y        = "Elo/Logistic probability",
    colour   = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(file.path(PLOTS_DIR, "model_agreement.png"), agree_plot, width = 11, height = 5, dpi = 150)
message(glue("  → model_agreement.png saved"))

message(glue("\n=== Evaluation complete ==="))
message(glue("Outputs in {OUTPUT_DIR}/"))
message("  evaluation_summary.csv  — metrics table")
message("  dc_test_predictions.csv — DC probs on test period")
message(glue("  plots/calibration.png   — reliability diagrams"))
message(glue("  plots/model_agreement.png — DC vs Elo agreement scatter"))
message("\nTo tune ensemble weights, compare log-loss across models and adjust")
message("DC_WEIGHT / ELO_WEIGHT in 06_ensemble.R accordingly.")
