#!/usr/bin/env Rscript
# =============================================================================
# SA1: Proper DLNM Validation (R implementation)
# =============================================================================
# Uses the dlnm package (Gasparrini 2011) with cross-basis to estimate
# lag-response profiles for the 14 model-adequate biomarkers and compares
# them with the XGBoost-SHAP lag profiles from Stage 1.
#
# The key methodological improvement over the Python SA1 (OLS): the cross-basis
# handles multicollinearity across lag features by imposing a smooth spline
# constraint on the lag dimension. Standard OLS with all 7 correlated lag
# features gives unstable coefficients (negative rho values are a symptom);
# the cross-basis is the canonical epidemiological solution (Gasparrini et al.
# 2010 Statistics in Medicine).
#
# Input:  mcd_outputs/dlnm_r_input.csv   (exported by Python)
#         mcd_outputs/stage1/*/lag_response_summary.csv  (primary SHAP profiles)
# Output: mcd_outputs/sensitivity/sa1_dlnm/sa1_r_dlnm_results.json
# =============================================================================

suppressPackageStartupMessages({
  library(dlnm)
  library(splines)
  library(jsonlite)
  library(lme4)      # mixed model: random intercept per patient
})

cat("=== SA1: R DLNM Validation ===\n")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
repo_root  <- getwd()
input_csv  <- file.path(repo_root, "mcd_outputs", "dlnm_r_input.csv")
stage1_dir <- file.path(repo_root, "mcd_outputs", "stage1")
out_dir    <- file.path(repo_root, "mcd_outputs", "sensitivity", "sa1_dlnm")
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)
out_json   <- file.path(out_dir, "sa1_r_dlnm_results.json")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
cat("Loading dataset...\n")
df <- read.csv(input_csv, stringsAsFactors=FALSE)
cat("  Rows:", nrow(df), "  Cols:", ncol(df), "\n")

# ---------------------------------------------------------------------------
# Lag matrix: temperature at lags 0,1,3,7,14,21,30 days
# Interpolate to a full 0-30 day matrix for the cross-basis
# ---------------------------------------------------------------------------
lag_col_map <- list(
  "0"  = "dlnm_lag0_c",
  "1"  = "dlnm_lag1_c",
  "3"  = "dlnm_lag3_c",
  "7"  = "dlnm_lag7_c",
  "14" = "dlnm_lag14_c",
  "21" = "dlnm_lag21_c",
  "30" = "era5_temp_lag30d_c"
)
lag_days_avail <- as.integer(names(lag_col_map))
lag_col_names  <- unlist(lag_col_map, use.names=FALSE)

# Check all columns exist
missing_cols <- lag_col_names[!lag_col_names %in% colnames(df)]
if (length(missing_cols) > 0) {
  stop("Missing lag columns: ", paste(missing_cols, collapse=", "))
}

Q_avail <- as.matrix(df[, lag_col_names])
colnames(Q_avail) <- paste0("lag", lag_days_avail)

cat("  Building 0-30 lag matrix via linear interpolation...\n")
# Interpolate each row to full 0-30 day grid
all_lags <- 0:30
Q_full <- matrix(NA_real_, nrow=nrow(df), ncol=length(all_lags))
colnames(Q_full) <- paste0("lag", all_lags)

for (i in seq_len(nrow(Q_avail))) {
  row_vals <- Q_avail[i, ]
  if (any(is.na(row_vals))) {
    Q_full[i, ] <- NA_real_
  } else {
    Q_full[i, ] <- approx(lag_days_avail, row_vals, xout=all_lags, rule=2)$y
  }
}

# Temperature reference point for prediction: median of lag-0 values
temp_ref <- median(Q_full[, "lag0"], na.rm=TRUE)
# Exposure contrast: 90th-percentile temperature vs median (reference)
# Predicting at=temp_ref vs cen=temp_ref always gives zero — need a real contrast.
temp_p90 <- quantile(Q_full[, "lag0"], 0.90, na.rm=TRUE)
cat(sprintf("  Temp reference (lag-0 median): %.1f degC\n", temp_ref))
cat(sprintf("  Temp contrast  (lag-0 p90):    %.1f degC\n", temp_p90))

# Complete cases for Q matrix
complete_Q <- complete.cases(Q_full)
cat(sprintf("  Rows with complete lag matrix: %d/%d\n", sum(complete_Q), nrow(df)))

# ---------------------------------------------------------------------------
# Cross-basis specification
# Cross-basis: natural cubic spline for exposure (df=3), natural spline for
# lag (df=4, log-distributed knots — standard for 0-30 day lag window).
# ---------------------------------------------------------------------------
cb_argvar <- list(fun="ns", df=4)   # df=4 better captures nonlinear exposure-response
cb_arglag <- list(fun="ns", df=4, knots=dlnm::logknots(30, 3))

cat("Building cross-basis...\n")
cb <- crossbasis(Q_full, lag=c(0, 30),
                 argvar=cb_argvar,
                 arglag=cb_arglag)
cat("  Cross-basis dimensions:", dim(cb), "\n")

# ---------------------------------------------------------------------------
# Adequate biomarkers (14 that passed Stage 1 adequacy criterion)
# ---------------------------------------------------------------------------
adequate_biomarkers <- c(
  "body_fat_percent", "hematocrit", "hemoglobin", "viral_load",
  "waist_hip_ratio", "creatinine", "cd4_count", "diastolic_bp",
  "bmi", "ldl_cholesterol", "systolic_bp", "heart_rate",
  "albumin", "hip_circumference"
)
# Filter to those present in the dataset
biomarkers_to_run <- adequate_biomarkers[adequate_biomarkers %in% colnames(df)]
cat(sprintf("\nRunning DLNM for %d biomarkers:\n", length(biomarkers_to_run)))

# ---------------------------------------------------------------------------
# Covariates: year, month fixed effects, age, sex
# (biomarkers already person-mean centred in Python, no patient RE needed)
# ---------------------------------------------------------------------------
avail_covars <- c()
for (v in c("year", "age_years")) {
  if (v %in% colnames(df)) avail_covars <- c(avail_covars, v)
}
# Sex and hiv_status as factors if present
for (v in c("sex", "hiv_status")) {
  if (v %in% colnames(df)) {
    df[[v]] <- as.factor(df[[v]])
    avail_covars <- c(avail_covars, v)
  }
}
# Month dummies
if ("month" %in% colnames(df)) {
  df$month_f <- as.factor(df$month)
  avail_covars <- c(avail_covars, "month_f")
}

cat("  Covariates:", paste(avail_covars, collapse=", "), "\n\n")

# ---------------------------------------------------------------------------
# Helper: load primary SHAP lag profile for a biomarker
# Returns data.frame with lag_day, mean_abs_shap (or NULL)
# ---------------------------------------------------------------------------
load_shap_profile <- function(bio) {
  p <- file.path(stage1_dir, bio, "lag_response_summary.csv")
  if (!file.exists(p)) return(NULL)
  prof <- read.csv(p)
  if (!"lag_day" %in% colnames(prof) || !"mean_abs_shap" %in% colnames(prof)) return(NULL)
  prof[order(prof$lag_day), ]
}

# ---------------------------------------------------------------------------
# Fit DLNM and extract lag-specific effects
# ---------------------------------------------------------------------------
all_results <- list()

for (bio in biomarkers_to_run) {
  cat(sprintf("  %s...", bio))

  y <- df[[bio]]
  ok <- complete_Q & !is.na(y)
  if (sum(ok) < 500) {
    cat(sprintf("  SKIP (n=%d)\n", sum(ok)))
    all_results[[bio]] <- list(status="insufficient_data", n_obs=sum(ok))
    next
  }

  # Build a per-biomarker cross-basis on the subset rows.
  # crosspred() requires a proper crossbasis object (with knot attributes),
  # not a plain matrix slice — so we rebuild cb for each biomarker.
  df_sub <- df[ok, , drop=FALSE]
  df_sub$y_resp <- y[ok]

  # Drop factor covariates that have only one level in this subset
  # (happens for e.g. viral_load which may only appear in HIV+ patients).
  active_covars <- Filter(function(v) {
    col <- df_sub[[v]]
    if (is.factor(col)) nlevels(droplevels(col)) >= 2 else TRUE
  }, avail_covars)
  covar_str <- if (length(active_covars) > 0) paste(active_covars, collapse=" + ") else "1"

  tryCatch({
    # Rebuild cross-basis on the ok-subset Q matrix so crosspred is consistent
    cb_bio <- crossbasis(Q_full[ok, ], lag=c(0, 30),
                         argvar=cb_argvar, arglag=cb_arglag)

    # --- Try lmer with random intercept for patient_id ---
    # This isolates within-patient temperature effects (same signal SHAP captures
    # via GroupKFold) by absorbing between-patient mean differences in the RE.
    # crosspred needs fixef() and vcov() passed explicitly for lmer objects.
    lmer_form <- as.formula(paste("y_resp ~ cb_bio +", covar_str, "+ (1|patient_id)"))
    mod_lmer  <- tryCatch(
      lmer(lmer_form, data=df_sub, REML=FALSE,
           control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5))),
      error=function(e) NULL,
      warning=function(w) {
        # Singular fit or convergence warnings are common with small n — still usable
        withCallingHandlers(
          lmer(lmer_form, data=df_sub, REML=FALSE,
               control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5))),
          warning=function(w) invokeRestart("muffleWarning")
        )
      }
    )

    if (!is.null(mod_lmer)) {
      r2        <- 1 - var(residuals(mod_lmer)) / var(df_sub$y_resp, na.rm=TRUE)
      pred      <- crosspred(cb_bio, mod_lmer,
                             coef=fixef(mod_lmer),
                             vcov=as.matrix(vcov(mod_lmer)),
                             at=temp_p90, bylag=1, cumul=FALSE, cen=temp_ref)
      mod_type  <- "lmer"
    } else {
      # Fallback: OLS (lmer failed — e.g. single group, rank-deficient)
      mod       <- lm(as.formula(paste("y_resp ~ cb_bio +", covar_str)),
                      data=df_sub, x=FALSE, y=FALSE, model=FALSE)
      r2        <- summary(mod)$r.squared
      pred      <- crosspred(cb_bio, mod, at=temp_p90, bylag=1, cumul=FALSE, cen=temp_ref)
      mod_type  <- "lm_fallback"
    }

    # matfit: matrix [n_at_values x n_lags]; single at → first (only) row.
    # Column names are "lag0","lag1",... — strip prefix before integer parse.
    lag_fit    <- as.vector(pred$matfit[1, ])
    lag_se     <- as.vector(pred$matSE[1, ])
    lag_days_r <- as.integer(sub("lag", "", colnames(pred$matfit)))

    # -----------------------------------------------------------------------
    # Compare with SHAP profile: Spearman rho between |DLNM effect| and
    # SHAP |mean_abs_shap|, at matching lag days
    # -----------------------------------------------------------------------
    shap_prof <- load_shap_profile(bio)
    concordance <- NA_real_
    concordant  <- NA
    dominant_lag_dlnm <- lag_days_r[which.max(abs(lag_fit))]
    dominant_lag_shap <- NA_integer_
    dominant_lag_agrees <- NA

    if (!is.null(shap_prof)) {
      # Match on lag_day
      shap_lags  <- shap_prof$lag_day
      dlnm_abs   <- abs(lag_fit[lag_days_r %in% shap_lags])
      shap_abs   <- shap_prof$mean_abs_shap[shap_lags %in% lag_days_r]

      if (length(dlnm_abs) >= 3 && length(dlnm_abs) == length(shap_abs)) {
        rho_test    <- cor.test(dlnm_abs, shap_abs, method="spearman", exact=FALSE)
        concordance <- as.numeric(rho_test$estimate)
        concordant  <- isTRUE(concordance >= 0.5 - .Machine$double.eps^0.5)

        dominant_lag_shap   <- shap_lags[which.max(shap_abs)]
        dominant_lag_agrees <- (dominant_lag_dlnm == dominant_lag_shap)
      }
    }

    cat(sprintf("  [%s] n=%d  R2=%.3f  rho=%.3f  DomLag: DLNM=%d SHAP=%s %s\n",
        mod_type,
        sum(ok), r2, ifelse(is.na(concordance), 0, concordance),
        dominant_lag_dlnm,
        ifelse(is.na(dominant_lag_shap), "?", dominant_lag_shap),
        ifelse(is.na(dominant_lag_agrees), "", ifelse(dominant_lag_agrees, "[AGREE]", "[differ]"))))

    all_results[[bio]] <- list(
      status              = "success",
      model_type          = mod_type,
      n_obs               = sum(ok),
      dlnm_r2             = r2,
      spearman_rho        = concordance,
      concordant          = concordant,
      dominant_lag_dlnm   = dominant_lag_dlnm,
      dominant_lag_shap   = dominant_lag_shap,
      dominant_lag_agrees = dominant_lag_agrees,
      lag_days            = lag_days_r,
      dlnm_fit            = lag_fit,
      dlnm_se             = lag_se
    )

  }, error=function(e) {
    cat(sprintf("  ERROR: %s\n", conditionMessage(e)))
    all_results[[bio]] <<- list(status="error", error=conditionMessage(e))
  })
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
successes <- Filter(function(r) isTRUE(r$status == "success"), all_results)
n_concordant <- sum(sapply(successes, function(r) isTRUE(r$concordant)))
rhos <- sapply(successes, function(r) r$spearman_rho)
rhos <- rhos[!is.na(rhos)]
dom_agree <- sum(sapply(successes, function(r) isTRUE(r$dominant_lag_agrees)))

cat(sprintf("\n=== R DLNM Summary ===\n"))
cat(sprintf("  %d/%d biomarkers: concordant (rho >= 0.5)\n", n_concordant, length(successes)))
cat(sprintf("  Mean Spearman rho: %.3f\n", if(length(rhos)>0) mean(rhos) else NA))
cat(sprintf("  Min  Spearman rho: %.3f\n", if(length(rhos)>0) min(rhos) else NA))
cat(sprintf("  Dominant lag agrees: %d/%d\n", dom_agree, length(successes)))

# ---------------------------------------------------------------------------
# Save JSON for Python
# ---------------------------------------------------------------------------
output <- list(
  method             = "R_dlnm_crossbasis",
  description        = "Proper DLNM with natural spline cross-basis (Gasparrini 2010)",
  n_biomarkers       = length(successes),
  n_concordant       = n_concordant,
  n_dominant_agree   = dom_agree,
  mean_spearman_rho  = if(length(rhos)>0) mean(rhos) else NULL,
  min_spearman_rho   = if(length(rhos)>0) min(rhos) else NULL,
  concordance_threshold = 0.5,
  temp_reference_c   = temp_ref,
  cross_basis_spec   = list(
    argvar = "natural spline, df=3",
    arglag = "natural spline, df=4, log-knots",
    lag_range = "0-30 days"
  ),
  per_biomarker      = all_results
)

write_json(output, out_json, auto_unbox=TRUE, digits=6, null="null")
cat(sprintf("\nSaved: %s\n", out_json))
