# 00_setup.R — install and load required packages

required_packages <- c(
  "httr2",      # modern HTTP client
  "tidyverse",  # dplyr, purrr, readr, ggplot2, etc.
  "lubridate",  # date/time parsing
  "glue",       # string interpolation
  "slider"      # rolling window aggregations
)

# brms requires Stan (C++ compiler needed).
# If you haven't installed Stan yet, run one of:
#   install.packages("rstan")                    # classic backend
#   install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
#   cmdstanr::install_cmdstan()                  # faster backend (recommended)
model_packages <- c(
  "brms",      # Bayesian regression models via Stan
  "tidybayes"  # tidy extraction of posterior draws
)

install_if_missing <- function(pkgs) {
  missing <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
  if (length(missing) > 0) {
    message("Installing missing packages: ", paste(missing, collapse = ", "))
    install.packages(missing)
  }
}

install_if_missing(required_packages)
install_if_missing(model_packages)

invisible(lapply(c(required_packages, model_packages), library, character.only = TRUE))
message("All packages loaded.")
