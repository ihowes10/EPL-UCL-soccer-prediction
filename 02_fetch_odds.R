# R/02_fetch_odds.R
# Pulls current bookmaker odds for upcoming fixtures from The Odds API.
# Outputs: data/raw/odds_raw.csv, data/raw/odds_consensus.csv

source("config.R")
source("00_setup.R")

# ── helpers ───────────────────────────────────────────────────────────────────

get_price <- function(outcomes, team_name) {
  # Safely extract decimal price for a named outcome
  match <- keep(outcomes, ~.x$name == team_name)
  if (length(match) == 0) NA_real_ else match[[1]]$price
}

# Adjust raw implied probs to sum to 1 (removes bookmaker overround/vig)
add_implied_probs <- function(df) {
  df |>
    mutate(
      raw_home = 1 / home_odds,
      raw_draw = 1 / draw_odds,
      raw_away = 1 / away_odds,
      overround = raw_home + raw_draw + raw_away,
      implied_home_prob = raw_home / overround,
      implied_draw_prob = raw_draw / overround,
      implied_away_prob = raw_away / overround
    ) |>
    select(-raw_home, -raw_draw, -raw_away)
}

# ── main fetch ────────────────────────────────────────────────────────────────

fetch_odds <- function(sport_key) {
  message(glue("  Fetching odds for {sport_key}..."))

  resp <- request(glue("{ODDS_BASE_URL}/sports/{sport_key}/odds")) |>
    req_url_query(
      apiKey     = ODDS_API_KEY,
      regions    = "eu",
      markets    = "h2h",
      oddsFormat = "decimal"
    ) |>
    req_error(is_error = \(r) FALSE) |>
    req_retry(max_tries = 3) |>
    req_perform()

  status <- resp_status(resp)

  # Log remaining API quota from response headers
  remaining <- resp_header(resp, "x-requests-remaining")
  used      <- resp_header(resp, "x-requests-used")
  if (!is.null(remaining)) message(glue("  API quota — used: {used}, remaining: {remaining}"))

  if (status != 200) {
    warning(glue("Status {status} fetching odds for {sport_key}."))
    return(tibble())
  }

  events_raw <- resp_body_json(resp)

  if (length(events_raw) == 0) {
    message(glue("  No upcoming odds found for {sport_key}."))
    return(tibble())
  }

  message(glue("  → {length(events_raw)} events returned."))

  # Parse every bookmaker line for every event
  map_dfr(events_raw, function(event) {
    map_dfr(event$bookmakers, function(bk) {
      h2h <- keep(bk$markets, ~.x$key == "h2h")
      if (length(h2h) == 0) return(tibble())

      outcomes <- h2h[[1]]$outcomes

      tibble(
        event_id        = event$id,
        sport_key       = event$sport_key,
        commence_time   = ymd_hms(event$commence_time),
        home_team       = event$home_team,
        away_team       = event$away_team,
        bookmaker       = bk$key,
        bookmaker_title = bk$title,
        home_odds       = get_price(outcomes, event$home_team),
        draw_odds       = get_price(outcomes, "Draw"),
        away_odds       = get_price(outcomes, event$away_team)
      )
    })
  })
}

# ── run ───────────────────────────────────────────────────────────────────────

message("=== Fetching odds data ===")

all_odds <- map_dfr(names(ODDS_SPORT_KEYS), function(comp_code) {
  sport_key <- ODDS_SPORT_KEYS[[comp_code]]
  odds      <- fetch_odds(sport_key)
  if (nrow(odds) > 0) odds <- mutate(odds, competition = comp_code)
  Sys.sleep(2)
  odds
})

if (nrow(all_odds) == 0) {
  message("No odds data retrieved. Check API key or upcoming fixture schedule.")
} else {
  all_odds <- add_implied_probs(all_odds)

  # Consensus view: average odds/implied probs across bookmakers per event
  consensus_odds <- all_odds |>
    group_by(event_id, sport_key, competition, commence_time, home_team, away_team) |>
    summarise(
      n_bookmakers      = n(),
      home_odds_avg     = mean(home_odds, na.rm = TRUE),
      draw_odds_avg     = mean(draw_odds, na.rm = TRUE),
      away_odds_avg     = mean(away_odds, na.rm = TRUE),
      implied_home_prob = mean(implied_home_prob, na.rm = TRUE),
      implied_draw_prob = mean(implied_draw_prob, na.rm = TRUE),
      implied_away_prob = mean(implied_away_prob, na.rm = TRUE),
      .groups = "drop"
    ) |>
    arrange(commence_time)

  dir.create(DATA_DIR, recursive = TRUE, showWarnings = FALSE)
  write_csv(all_odds,       file.path(DATA_DIR, "odds_raw.csv"))
  write_csv(consensus_odds, file.path(DATA_DIR, "odds_consensus.csv"))

  message(glue(
    "\nDone. Odds for {nrow(consensus_odds)} fixtures ",
    "across {n_distinct(all_odds$bookmaker)} bookmakers saved to {DATA_DIR}/"
  ))

  message("\nSample upcoming fixtures with market-implied probabilities:")
  print(select(consensus_odds, commence_time, home_team, away_team,
               implied_home_prob, implied_draw_prob, implied_away_prob))
}
