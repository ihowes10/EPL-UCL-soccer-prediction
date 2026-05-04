# R/01_fetch_matches.R
# Pulls historical results and upcoming fixtures from football-data.org.
# Outputs: data/raw/historical_matches.csv, data/raw/upcoming_matches.csv

source("config.R")
source("00_setup.R")

# ── helpers ──────────────────────────────────────────────────────────────────

safe_int <- function(x) if (is.null(x)) NA_integer_ else as.integer(x)
safe_chr <- function(x) if (is.null(x)) NA_character_ else as.character(x)

parse_matches <- function(matches_raw, competition_code, season) {
  map_dfr(matches_raw, function(m) {
    winner <- safe_chr(m$score$winner)
    result <- case_when(
      winner == "HOME_TEAM" ~ "H",
      winner == "AWAY_TEAM" ~ "A",
      winner == "DRAW"      ~ "D",
      TRUE                  ~ NA_character_
    )
    tibble(
      match_id     = as.integer(m$id),
      competition  = competition_code,
      season       = season,
      stage        = safe_chr(m$stage),
      matchday     = safe_int(m$matchday),
      utc_date     = ymd_hms(m$utcDate),
      status       = safe_chr(m$status),
      home_team    = m$homeTeam$name,
      away_team    = m$awayTeam$name,
      home_team_id = as.integer(m$homeTeam$id),
      away_team_id = as.integer(m$awayTeam$id),
      home_goals   = safe_int(m$score$fullTime$home),
      away_goals   = safe_int(m$score$fullTime$away),
      result       = result
    )
  })
}

# ── main fetch ────────────────────────────────────────────────────────────────

fetch_matches <- function(competition_code, season) {
  message(glue("  Fetching {competition_code} / season {season}..."))

  resp <- request(glue("{FOOTBALL_BASE_URL}/competitions/{competition_code}/matches")) |>
    req_headers("X-Auth-Token" = FOOTBALL_API_KEY) |>
    req_url_query(season = season) |>
    req_error(is_error = \(r) FALSE) |>
    req_retry(max_tries = 3) |>
    req_perform()

  status <- resp_status(resp)

  if (status == 403) {
    warning(glue("403 Forbidden for {competition_code} — this competition may require a higher API tier."))
    return(tibble())
  }
  if (status != 200) {
    warning(glue("Unexpected status {status} for {competition_code} season {season}."))
    return(tibble())
  }

  body        <- resp_body_json(resp)
  matches_raw <- body$matches

  if (length(matches_raw) == 0) {
    message(glue("  No matches found for {competition_code} season {season}."))
    return(tibble())
  }

  message(glue("  → {length(matches_raw)} matches returned."))
  parse_matches(matches_raw, competition_code, season)
}

# ── run ───────────────────────────────────────────────────────────────────────

message("=== Fetching match data ===")
message(glue("Competitions: {paste(TARGET_COMPETITIONS, collapse=', ')}"))
message(glue("Seasons: {paste(SEASONS, collapse=', ')}"))

all_matches <- map_dfr(SEASONS, function(season) {
  map_dfr(TARGET_COMPETITIONS, function(comp_code) {
    result <- fetch_matches(comp_code, season)
    Sys.sleep(7)  # respect 10 req/min rate limit
    result
  })
})

historical_matches <- filter(all_matches, status == "FINISHED")
upcoming_matches   <- filter(all_matches, status %in% c("SCHEDULED", "TIMED"))

dir.create(DATA_DIR, recursive = TRUE, showWarnings = FALSE)
write_csv(historical_matches, file.path(DATA_DIR, "historical_matches.csv"))
write_csv(upcoming_matches,   file.path(DATA_DIR, "upcoming_matches.csv"))
saveRDS(all_matches,          file.path(DATA_DIR, "all_matches.rds"))

message(glue(
  "\nDone. {nrow(historical_matches)} historical matches, ",
  "{nrow(upcoming_matches)} upcoming fixtures saved to {DATA_DIR}/"
))
