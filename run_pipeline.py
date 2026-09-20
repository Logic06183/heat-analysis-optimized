"""MCD Pipeline Orchestrator."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers, BIOMARKER_SYSTEMS
from mcd_pipeline.utils.reproducibility import set_global_seeds, capture_environment


def setup_logging():
    log_dir = config.OUTPUT_ROOT / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        force=True,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / 'pipeline.log'),
        ],
    )
    return logging.getLogger('mcd_pipeline')


def run_stage0(logger, fast=False):
    logger.info('=' * 60)
    logger.info('STAGE 0: Data Preparation & Feature Engineering')
    logger.info('=' * 60)
    from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
    from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features
    from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers as _get_bio
    df = build_analysis_dataset()
    bio_cols = [b.column for b in _get_bio().values() if b.column and b.column in df.columns]
    df = engineer_features(df, bio_cols)
    logger.info('Stage 0 complete: %d rows, %d columns', len(df), len(df.columns))
    return df


def run_stage1(logger, df=None, fast=False):
    logger.info('=' * 60)
    logger.info('STAGE 1: Lag-Response Profiling')
    if fast:
        config.N_BOOTSTRAP_REPLICATES = config.N_BOOTSTRAP_FAST
    logger.info('=' * 60)
    from mcd_pipeline.stage1_lag_profiling.run_stage1 import run_stage1 as _run
    return _run(df=df)


def run_stage2(logger, df=None, fast=False):
    logger.info('=' * 60)
    logger.info('STAGE 2: Interaction Detection')
    logger.info('=' * 60)
    from mcd_pipeline.stage2_interactions.run_stage2 import run_stage2 as _run
    return _run(df=df)


def run_stage3(logger, df=None, fast=False):
    from mcd_pipeline.stage3_clustering.run_stage3 import run_stage3 as _run
    return _run(df=df)


def run_posthoc(logger, df=None, tiers=None, biomarkers=None):
    from mcd_pipeline.posthoc.run_posthoc import run_posthoc as _run
    return _run(df=df, tiers=tiers, biomarkers=biomarkers)


# ---------------------------------------------------------------------------
# Final-revision (peer-review) additional analyses
# ---------------------------------------------------------------------------

def run_revision_per_cohort_meta(logger, df=None, biomarkers=None,
                                 n_top_cohorts=6, min_cohort_n=200):
    """SA9: per-cohort meta-analysis on cluster-driving biomarkers."""
    logger.info('=' * 60)
    logger.info('REVISION SA9: Per-cohort meta-analysis')
    logger.info('=' * 60)
    from mcd_pipeline.sensitivity.sa9_per_cohort_meta import run_per_cohort_meta
    if df is None:
        df = run_stage0(logger)
    return run_per_cohort_meta(
        df=df, biomarkers=biomarkers,
        n_top_cohorts=n_top_cohorts, min_cohort_n=min_cohort_n,
    )


def run_revision_humidity(logger, df=None, include_wet_bulb=True):
    """SA10: humidity-aware exposure sensitivity."""
    logger.info('=' * 60)
    logger.info('REVISION SA10: Humidity sensitivity')
    logger.info('=' * 60)
    from mcd_pipeline.sensitivity.sa10_humidity import run_humidity_sensitivity
    if df is None:
        df = run_stage0(logger)
    return run_humidity_sensitivity(df=df, include_wet_bulb=include_wet_bulb)


def run_revision_cluster_cis(logger, n_bootstrap=2000):
    """Stage 3 cluster-proportion bootstrap 95% CIs."""
    logger.info('=' * 60)
    logger.info('REVISION: Cluster-proportion 95%% CIs')
    logger.info('=' * 60)
    from mcd_pipeline.stage3_clustering.cluster_proportion_cis import (
        run_cluster_proportion_cis,
    )
    return run_cluster_proportion_cis(n_bootstrap=n_bootstrap)


def run_revision_parsimony(logger, k_min=2, k_max=8, n_references=50):
    """Stage 3 parsimony diagnostics: gap statistic + PC loadings."""
    logger.info('=' * 60)
    logger.info('REVISION: Clustering parsimony diagnostics')
    logger.info('=' * 60)
    from mcd_pipeline.stage3_clustering.parsimony_diagnostics import (
        run_parsimony_diagnostics,
    )
    return run_parsimony_diagnostics(
        k_values=range(k_min, k_max + 1),
        n_references=n_references,
    )


def run_revision_all(logger, df=None):
    """Run all four final-revision peer-review additions."""
    if df is None:
        df = run_stage0(logger)
    out = {
        'sa9_per_cohort_meta': run_revision_per_cohort_meta(logger, df=df),
        'sa10_humidity': run_revision_humidity(logger, df=df),
        'cluster_cis': run_revision_cluster_cis(logger),
        'parsimony': run_revision_parsimony(logger),
    }
    logger.info('All four final-revision analyses complete')
    return out


def main():
    parser = argparse.ArgumentParser(description='MCD Pipeline')
    parser.add_argument('--stage', nargs='*', type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--posthoc', action='store_true')
    parser.add_argument('--posthoc-tier', nargs='*', type=int, choices=[1, 2, 3])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--validate', action='store_true')

    # Final-revision additions (peer-review)
    rev = parser.add_argument_group('final revision (Lancet Planetary Health response)')
    rev.add_argument('--revision', action='store_true',
                     help='Run all four revision additions in one call.')
    rev.add_argument('--revision-sa9', action='store_true',
                     help='Per-cohort meta-analysis (SA9).')
    rev.add_argument('--revision-sa10', action='store_true',
                     help='Humidity-aware exposure sensitivity (SA10).')
    rev.add_argument('--revision-cluster-cis', action='store_true',
                     help='Bootstrap 95%% CIs for cluster proportions (Stage 3).')
    rev.add_argument('--revision-parsimony', action='store_true',
                     help='Gap statistic + PC loadings for cluster parsimony.')
    rev.add_argument('--revision-bootstrap', type=int, default=2000,
                     help='Bootstrap iterations for cluster CIs (default 2000).')

    args = parser.parse_args()

    logger = setup_logging()
    logger.info('MCD Pipeline starting')

    # When only revision flags are given (no --stage), skip the main pipeline
    # stages so that e.g. `--revision` doesn't re-run the full 50-bootstrap
    # Stage 1 before reaching the four peer-review additions.
    any_revision = (
        args.revision or args.revision_sa9 or args.revision_sa10
        or args.revision_cluster_cis or args.revision_parsimony
    )
    logger.info('Project root: %s', config.PROJECT_ROOT)

    set_global_seeds()
    env = capture_environment()
    logger.info('Environment captured: %d packages', env.get('n_packages', 0))

    available = get_available_biomarkers()
    logger.info('Available biomarkers: %d across %d systems', len(available), len(BIOMARKER_SYSTEMS))

    if args.dry_run:
        logger.info('Dry run complete.')
        return

    if args.stage is not None:
        stages = args.stage
    elif any_revision:
        stages = []   # don't re-run the main pipeline for revision-only runs
    else:
        stages = [0, 1, 2, 3]
    df = None

    if 0 in stages:
        df = run_stage0(logger, fast=args.fast)

    if 1 in stages:
        results = run_stage1(logger, df=df, fast=args.fast)
        logger.info('Stage 1: %d biomarkers processed', len(results) if results else 0)

    if 2 in stages:
        results2 = run_stage2(logger, df=df, fast=args.fast)
        logger.info('Stage 2: %d biomarkers processed',
                    sum(1 for r in results2.values() if 'error' not in r))

    if 3 in stages:
        results3 = run_stage3(logger, df=df, fast=args.fast)
        logger.info('Stage 3 complete: k=%d, silhouette=%.3f, stability ARI=%.3f',
                    results3.get('optimal_k', -1),
                    results3.get('optimal_silhouette', float('nan')),
                    results3.get('stability_mean_ari', float('nan')))

    if args.posthoc:
        tiers = args.posthoc_tier or [1, 2, 3]
        run_posthoc(logger, df=df, tiers=tiers)

    if args.revision:
        run_revision_all(logger, df=df)
    else:
        if args.revision_sa9:
            run_revision_per_cohort_meta(logger, df=df)
        if args.revision_sa10:
            run_revision_humidity(logger, df=df)
        if args.revision_cluster_cis:
            run_revision_cluster_cis(logger, n_bootstrap=args.revision_bootstrap)
        if args.revision_parsimony:
            run_revision_parsimony(logger)

    logger.info('Pipeline complete')


if __name__ == '__main__':
    main()
