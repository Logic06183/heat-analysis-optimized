# =============================================================================
# SA1: R DLNM validation, ERA5-Land variant
# =============================================================================
# Wrapper around sa1_dlnm_validation.R that points at the ERA5-Land-flavoured
# input CSV and writes outputs to mcd_outputs_era5_land/. Same cross-basis
# specification (ns df=4 for exposure, ns df=4 with log-knots for lag) so the
# comparison with primary ERA5 is methodologically apples-to-apples.
# =============================================================================

# Override paths before sourcing the main script.
# When run via Rscript, get this file's path from commandArgs().
args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- normalizePath(sub("^--file=", "", file_arg[1]))
repo_root <- dirname(dirname(script_path))

input_csv_override <- file.path(repo_root, "mcd_outputs_era5_land", "dlnm_r_input.csv")
output_dir_override <- file.path(repo_root, "mcd_outputs_era5_land", "sensitivity", "sa1_dlnm")

if (!dir.exists(output_dir_override)) dir.create(output_dir_override, recursive=TRUE)

# Read the original R script and patch paths.
src <- readLines(file.path(repo_root, "dlnm_r", "sa1_dlnm_validation.R"))
src <- sub(
  'input_csv  <- file\\.path\\(repo_root, "mcd_outputs", "dlnm_r_input.csv"\\)',
  paste0('input_csv  <- "', input_csv_override, '"'),
  src
)
src <- gsub(
  'file\\.path\\(repo_root, "mcd_outputs", "stage1"',
  paste0('file.path(repo_root, "mcd_outputs_era5_land", "stage1"'),
  src
)
src <- gsub(
  'file\\.path\\(repo_root, "mcd_outputs", "sensitivity"',
  paste0('file.path(repo_root, "mcd_outputs_era5_land", "sensitivity"'),
  src
)

# Write a temp file and source it.
tmp <- tempfile(fileext = ".R")
writeLines(src, tmp)
cat("Running R DLNM under ERA5-Land mode...\n")
cat("  input:  ", input_csv_override, "\n")
cat("  output: ", output_dir_override, "\n\n")
source(tmp)
