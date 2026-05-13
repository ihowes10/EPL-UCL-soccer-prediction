# config.R — API keys and global constants
# Keys are loaded from .Renviron — copy .Renviron.example to .Renviron and fill in your values.

# Explicitly load .Renviron in case the R session was started before the file existed
local({
  renviron <- file.path(getwd(), ".Renviron")
  if (file.exists(renviron)) readRenviron(renviron)
})

FOOTBALL_API_KEY <- Sys.getenv("FOOTBALL_API_KEY")
ODDS_API_KEY     <- Sys.getenv("ODDS_API_KEY")

if (FOOTBALL_API_KEY == "") stop("FOOTBALL_API_KEY not set — check your .Renviron")
if (ODDS_API_KEY     == "") stop("ODDS_API_KEY not set — check your .Renviron")

FOOTBALL_BASE_URL <- "https://api.football-data.org/v4"
ODDS_BASE_URL     <- "https://api.the-odds-api.com/v4"

# Competitions used for model training and historical data storage.
# Only add competitions here if you want them in the Dixon-Coles/Elo training data.
TARGET_COMPETITIONS <- c(
  premier_league   = "PL",
  champions_league = "CL"
)

# Cup competitions — fetched for UPCOMING PREDICTIONS ONLY within the lookahead
# window. Never stored in historical data or used for model training, so the
# trained model's competition factor levels stay clean (PL + CL only).
CUP_COMPETITIONS <- c(
  fa_cup      = "FAC",
  carabao_cup = "ELC"
)

# The Odds API sport keys — must match their sport slugs exactly
ODDS_SPORT_KEYS <- c(
  PL  = "soccer_epl",
  CL  = "soccer_uefa_champs_league",
  FAC = "soccer_england_fa_cup",
  ELC = "soccer_england_efl_cup"
)

# How many days ahead to predict. 10 days captures the next PL matchweek
# plus any midweek UCL or cup fixtures in the same window.
# Increase temporarily if you want to look further ahead.
LOOKAHEAD_DAYS <- 10

# Seasons to fetch for model training (football-data.org uses start year)
# Update CURRENT_SEASON each summer when the new season kicks off.
SEASONS        <- 2022:2025
CURRENT_SEASON <- 2025

DATA_DIR <- file.path("data", "raw")
