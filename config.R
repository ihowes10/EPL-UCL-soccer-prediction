# config.R — API keys and global constants
# Keys are loaded from .Renviron — copy .Renviron.example to .Renviron and fill in your values.

FOOTBALL_API_KEY <- Sys.getenv("FOOTBALL_API_KEY")
ODDS_API_KEY     <- Sys.getenv("ODDS_API_KEY")

if (FOOTBALL_API_KEY == "") stop("FOOTBALL_API_KEY not set — check your .Renviron")
if (ODDS_API_KEY     == "") stop("ODDS_API_KEY not set — check your .Renviron")

FOOTBALL_BASE_URL <- "https://api.football-data.org/v4"
ODDS_BASE_URL     <- "https://api.the-odds-api.com/v4"

TARGET_COMPETITIONS <- c(
  premier_league   = "PL",
  champions_league = "CL"
)

# The Odds API sport keys — must match their sport slugs exactly
ODDS_SPORT_KEYS <- c(
  PL = "soccer_epl",
  CL = "soccer_uefa_champs_league"
)

# Seasons to fetch for model training (football-data.org uses start year)
# Update CURRENT_SEASON each summer when the new season kicks off.
SEASONS        <- 2022:2025
CURRENT_SEASON <- 2025

DATA_DIR <- file.path("data", "raw")
