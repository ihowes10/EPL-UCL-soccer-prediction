<<<<<<< HEAD
# Soccer Match Outcome Prediction & Value Betting System

Note: This project was built to practice statistical methods with something I am interested in, not to actually make sports prediction bets.

An end-to-end sports analytics pipeline built in R that identifies positive expected value (+EV) betting opportunities 
in the Premier League and UEFA Champions League. The system fetches live match and odds data from public APIs, fits 
two independent predictive models, blends them in a weighted ensemble, and surfaces fixtures where the model's 
probability meaningfully exceeds the market-implied probability.

---

## How It Works

Bookmakers embed a margin (the "vig") into their odds. After removing it, the market-implied probability for each 
outcome can be estimated. When a well-calibrated model assigns a higher probability to an outcome than the market 
implies, the expected value of that bet is positive:

```
EV = model_probability × decimal_odds − 1
```

The pipeline runs two structurally independent models — a Bayesian Poisson model and an Elo-based logistic regression, 
then blends them into an ensemble. Fixtures where the models disagree are flagged so they can be scrutinised before 
acting on any signal.

---

## Models

### Bayesian Bivariate Poisson (Dixon-Coles)
Home and away goals are modelled as independent Poisson random variables with log-linear expected rates that decompose 
into team-level attack and defence random effects, a competition fixed effect, and an overall intercept. Matches are 
weighted by exponential time-decay (half-life = 1 year). Fitted via `brms` / Stan with 4 chains × 1,000 post-warmup 
draws. Convergence: Max R-hat = 1.002, Min Bulk ESS = 3,162.

### Elo + Multinomial Logistic Regression
Elo ratings are updated sequentially after every match using a goal-difference multiplier and a fixed home 
advantage bonus (+100 points). A multinomial logistic regression (`nnet::multinom`) then maps the pre-match Elo 
differential and recent form features to a 3-way outcome probability. Two versions are trained: one on all historical 
data (for live prediction) and one on a temporal training split (for honest evaluation).

### Ensemble
The two model distributions are blended via weighted arithmetic average (equal weights by default). Total Variation 
distance between the two distributions is computed per fixture as a divergence diagnostic.

---

## Pipeline

| Script | Role |
|---|---|
| `R/01_fetch_matches.R` | Fetches historical results and upcoming fixtures from football-data.org |
| `R/02_fetch_odds.R` | Fetches bookmaker odds from The Odds API and computes vig-adjusted implied probabilities |
| `R/03_features.R` | Engineers rolling form, attack/defence strength, and odds features |
| `R/04_dixon_coles.R` | Fits (or reloads) the Bayesian Poisson model and predicts upcoming fixtures |
| `R/05_elo_logistic.R` | Computes Elo ratings and fits the logistic regression |
| `R/06_ensemble.R` | Blends predictions, computes EV, and writes the output table |
| `R/07_evaluate.R` | Evaluates models on a held-out test set with calibration plots |

Run scripts in order from an RStudio session with the working directory set to the project root.

---

## Output

A single CSV (`outputs/full_ev_table.csv`) is produced for every upcoming fixture:

| Column | Description |
|---|---|
| `competition` | PL or CL |
| `stage` | e.g. REGULAR_SEASON, SEMI_FINAL |
| `match_date` | Date of fixture |
| `home_team` / `away_team` | Team names |
| `dc_pick` / `lr_pick` / `ens_pick` | Most likely outcome per model |
| `models_agree` | TRUE if both models pick the same outcome |
| `more_confident` | Which model has the higher max-outcome probability |
| `ev_home` / `ev_draw` / `ev_away` | Expected value per outcome (%) |
| `best_ev` / `best_bet` | Highest EV available and the corresponding outcome |

---

## Data Sources

- **[football-data.org](https://www.football-data.org/)** (v4 API) — match results, fixtures, competition metadata
- **[The Odds API](https://the-odds-api.com/)** (v4 API) — live h2h bookmaker odds (EU region)

API keys are stored in `config.R`, which is excluded from version control.

---

## Evaluation

Models are evaluated on a held-out test set (last 365 days of historical data) using proper probabilistic scoring 
rules:

| Metric | Elo/Logistic | Naive Baseline |
|---|---|---|
| Log-loss | 0.987 | 1.099 |
| Brier score | 0.589 | 0.667 |
| Accuracy | 54.6% | ~52% |

Calibration reliability diagrams are saved to `outputs/plots/` after running `07_evaluate.R`.

---

## Requirements

- R ≥ 4.2
- Stan / CmdStan (for `brms`) — see [mc-stan.org](https://mc-stan.org/cmdstanr/)
- R packages: `tidyverse`, `httr2`, `lubridate`, `glue`, `slider`, `brms`, `tidybayes`, `nnet`

Install all packages by running `source("00_setup.R")`.

---

## Project Structure

```
soccer_value/
├── config.R                  # API keys and global constants (not committed)
├── 00_setup.R                # Package installation
├── R/
│   ├── 01_fetch_matches.R
│   ├── 02_fetch_odds.R
│   ├── 03_features.R
│   ├── 04_dixon_coles.R
│   ├── 05_elo_logistic.R
│   ├── 06_ensemble.R
│   └── 07_evaluate.R
├── data/
│   ├── raw/                  # API outputs (not committed)
│   └── processed/            # Engineered features (not committed)
├── models/                   # Cached fitted models (not committed)
├── outputs/                  # Predictions and plots
└── PROJECT_REPORT.md         # Detailed technical write-up
```

---

## Limitations & Planned Extensions

- **In-sample DC evaluation**: the Bayesian model is evaluated on data it was trained on; `07_evaluate.R` includes a `USE_LOO` flag for rigorous PSIS-LOO cross-validation once compute allows.
- **Equal ensemble weights**: weights are currently fixed at 0.5/0.5 and should be tuned once sufficient evaluation data accumulates.
- **Odds availability**: CL knockout fixtures often have thinner markets; fixtures without odds still receive model predictions but no EV estimate.
- **Planned**: Kelly criterion staking, Asian handicap markets, dynamic ensemble weighting, automated daily pipeline refresh.
=======
>>>>>>> b8b4d5c9713d81cdd1323e8d26afa37d047cf89a
