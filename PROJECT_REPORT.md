---
output:
  pdf_document:
    latex_engine: xelatex
  html_document: default
---
# Soccer Match Outcome Prediction & Value Betting System

## Overview

An end-to-end sports analytics pipeline that ingests live match and odds data, fits two independent predictive models, blends them in a weighted ensemble, and identifies fixtures where the model's probability estimate meaningfully exceeds the market-implied probability — a condition known as a "value bet."

The project is built entirely in R and covers the full data science lifecycle: API integration, feature engineering, Bayesian model fitting, frequentist modelling, ensemble design, and probabilistic evaluation.

---

## Motivation

Bookmaker odds embed a margin (the "vig") that systematically underprices certain outcomes. If a well-calibrated model consistently assigns a higher probability to an outcome than the market implies after removing the vig, the expected value (EV) of that bet is positive over the long run:

```
EV = model_probability × decimal_odds − 1
```

The core challenge is building a model that is more accurate than the collective market — a high bar that requires rigorous feature engineering and honest evaluation.

---

## Data Sources

| Source | API | Usage |
|---|---|---|
| football-data.org (v4) | REST/JSON | Historical results, upcoming fixtures, competition metadata |
| The Odds API (v4) | REST/JSON | Live bookmaker odds (h2h markets, EU region) |

Four seasons of Premier League and UEFA Champions League data are fetched (currently 2022–2025), with the current season's upcoming fixtures used for prediction. Vig-adjusted implied probabilities are computed by normalising each bookmaker's raw probabilities by the overround, then averaging across all available bookmakers.

---

## Pipeline Architecture

```
01_fetch_matches.R   →   raw match data (historical + upcoming)
02_fetch_odds.R      →   bookmaker odds + implied probabilities
03_features.R        →   feature engineering
04_dixon_coles.R     →   Bayesian Poisson model
05_elo_logistic.R    →   Elo ratings + logistic regression
06_ensemble.R        →   blend predictions, flag value bets
07_evaluate.R        →   model evaluation and calibration
```

Each script is independently runnable and writes well-defined output files, making the pipeline modular and easy to extend.

---

## Feature Engineering (`03_features.R`)

- **Rolling form** (last 5 matches, cross-competition): points per game, goals scored/conceded, goal difference — computed with the `slider` package using `.after = -1` to strictly exclude the current match and prevent data leakage.
- **Attack / defence strength** (Dixon-Coles parameters): season-level ratios of each team's goals scored/conceded relative to the league average, separately for home and away roles.
- **Team name normalisation**: strips suffixes (FC, AFC, CF, SC) and lowercases names to join across the two APIs, which use inconsistent naming conventions.
- **Odds consensus**: average decimal odds and vig-adjusted implied probabilities across all available bookmakers per fixture.

---

## Model 1 — Bayesian Bivariate Poisson (Dixon-Coles)

**File:** `R/04_dixon_coles.R` | **Library:** `brms` / Stan

### Model structure

```
log(E[home_goals]) = β₀_home + β_competition + att[home_team] + def[away_team]
log(E[away_goals]) = β₀_away + β_competition + att[away_team] + def[home_team]
```

Home and away goals are modelled as independent Poisson random variables conditional on their expected rates. Team attack and defence effects are zero-mean random effects (partial pooling), allowing the model to borrow strength across teams while still learning team-specific parameters.

### Key design choices

- **Exponential time-decay**: each match is weighted by `exp(-log(2) × days_ago / 365)`, giving matches from one year ago 50% of the weight of a match played today. This means form from three seasons ago has minimal influence.
- **Competition fixed effect**: adjusts for systematic differences in goal rates between the Premier League and Champions League.
- **Weakly informative priors**: `Normal(0, 1)` on intercepts (implied average goals ∈ [0.37, 2.72]) and `Normal(0, 0.5)` half-normal on standard deviations, preventing implausibly extreme team effects.
- **Posterior simulation**: 4,000 posterior draws of expected goal rates per fixture are used to simulate scorelines, from which win/draw/loss probabilities are derived with full uncertainty propagation.
- **Model caching**: the fitted Stan model is saved to disk and reloaded on subsequent runs, avoiding costly recompilation.

### Convergence diagnostics

On the current dataset (~1,600 matches): Max R-hat = 1.002 (target < 1.01), Min Bulk ESS = 3,162 (target > 400) — indicating well-mixed chains and reliable inference.

---

## Model 2 — Elo Ratings + Multinomial Logistic Regression

**File:** `R/05_elo_logistic.R` | **Library:** `nnet`

### Elo rating system

Ratings are updated sequentially in strict chronological order after each match. Key design choices:

- **Home advantage**: the home team is treated as 100 Elo points stronger in the expected-score calculation.
- **Goal-difference multiplier**: `k_adj = K × (1 + 0.5 × log1p(|goal_diff|))` — larger winning margins produce bigger rating updates, as they carry more signal about true team quality.
- **New team handling**: teams entering the dataset start at 1,500 (the neutral baseline) and are imputed from the prior for predictions.

### Logistic regression

A multinomial logistic model (`nnet::multinom`) predicts the three-way match outcome (H/D/A) using:

```
result ~ elo_diff + home_form_pts + away_form_pts + home_form_gd + away_form_gd + competition
```

Two model versions are trained: one on all data (used for live predictions) and one on a training split excluding the last 365 days (used for honest held-out evaluation).

---

## Ensemble & Value Identification (`06_ensemble.R`)

Predictions from both models are blended using a weighted arithmetic average (equal weights by default, tunable once evaluation data accumulates):

```
P_ensemble(outcome) = 0.5 × P_dixon_coles(outcome) + 0.5 × P_elo_logistic(outcome)
```

### Divergence metric

Total Variation (TV) distance measures how differently the two models view a fixture:

```
TV = 0.5 × (|P_DC_H − P_ELO_H| + |P_DC_D − P_ELO_D| + |P_DC_A − P_ELO_A|)
```

TV = 0 means identical predictions; TV = 0.1 means the models differ by ~10 percentage points on average per outcome. High-TV fixtures where the models also pick different winners are flagged for extra scrutiny before acting on any value signal.

### Output

A single CSV (`full_ev_table.csv`) covers every upcoming fixture with model picks, agreement flags, and EV for all three outcomes formatted as percentages.

---

## Evaluation Framework (`07_evaluate.R`)

Model quality is assessed on a held-out test set (last 365 days of historical data) using proper probabilistic scoring rules:

| Metric | Description | Naive baseline |
|---|---|---|
| **Log-loss** | Penalises confident wrong predictions heavily | log(3) ≈ 1.099 |
| **Brier score** | Mean squared error of predicted vs actual | 0.667 |
| **Accuracy** | Most-likely outcome correct | ~0.52 (league average) |

**Elo/logistic held-out results** (570 matches): Log-loss 0.987, Brier 0.589, Accuracy 54.6% — all meaningfully better than the naive baseline.

Calibration reliability diagrams (3-panel: Home/Draw/Away) are produced to check whether predicted probabilities match observed frequencies — a necessary condition for the EV framework to be valid.

---

## Technical Skills Demonstrated

| Area | Details |
|---|---|
| **Bayesian inference** | Hierarchical Poisson model via brms/Stan; posterior predictive simulation; convergence diagnostics (R-hat, ESS) |
| **Frequentist ML** | Multinomial logistic regression; temporal train/test split; proper scoring rules |
| **API integration** | Rate-limited REST API calls with retry logic and graceful error handling (httr2) |
| **Data engineering** | Long/wide reshaping; cross-API name normalisation; rolling window features without data leakage |
| **Ensemble methods** | Weighted model blending; divergence quantification via Total Variation distance |
| **Probabilistic evaluation** | Log-loss, Brier score, calibration curves |
| **Software design** | Modular pipeline; disk-cached models; empty-output handling; informative console diagnostics |

---

## Repository Structure

```
soccer_value/
├── config.R                  # API keys and global constants
├── 00_setup.R                # Package installation and loading
├── R/
│   ├── 01_fetch_matches.R    # Match data ingestion
│   ├── 02_fetch_odds.R       # Odds ingestion + vig adjustment
│   ├── 03_features.R         # Feature engineering
│   ├── 04_dixon_coles.R      # Bayesian Poisson model
│   ├── 05_elo_logistic.R     # Elo + logistic model
│   ├── 06_ensemble.R         # Blending + value identification
│   └── 07_evaluate.R         # Model evaluation
├── data/
│   ├── raw/                  # API outputs
│   └── processed/            # Engineered features
├── models/                   # Cached fitted models (.rds)
└── outputs/                  # Predictions, value bets, plots
```

---

## Possible Extensions

- **Kelly criterion staking**: size bets proportionally to edge rather than flat-staking.
- **Asian handicap / over-under markets**: extend to goal totals using the existing Poisson goal distributions.
- **Team-specific home advantage**: allow home advantage to vary by team in the Bayesian model.
- **Dynamic ensemble weights**: use rolling log-loss to weight models adaptively based on recent accuracy.
- **Automated pipeline**: schedule daily refreshes via cron or a workflow orchestrator.
