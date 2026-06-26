"""Ablation study: full classifier + quality pipeline with word_count and sentence_count excluded."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from modules.classifier import (
    LEVEL_NAMES as _CLF_LEVEL_NAMES,
    categorize_fragility,
    compute_fold_shap_correlations,
    plot_auc_comparison,
    plot_shap_importance,
    run_experiment,
    save_outputs,
)
from quality_pipeline import (
    aggregate_results,
    build_subsets,
    collect_cv_predictions,
    collect_score_distributions,
    compute_degradation_summary,
    compute_permutation_importance,
    get_score_range,
    plot_degradation_summary,
    plot_permutation_importance,
    plot_qwk_degradation,
    plot_qwk_heatmap,
    plot_scatter_pred_actual,
    plot_score_distributions,
    run_experiments_per_prompt,
    run_quality_experiments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FEATURES_PATH    = Path("features.parquet")
OUTPUT_DIR       = Path("Results Ablation")
RANDOM_SEED      = 42
N_FOLDS          = 5
ABLATED_FEATURES = frozenset({"word_count", "sentence_count"})


def load_and_ablate(features_path: Path) -> pd.DataFrame:
    """Load features.parquet and drop ABLATED_FEATURES; logs what was removed."""
    df      = pd.read_parquet(features_path)
    dropped = [c for c in ABLATED_FEATURES if c in df.columns]
    missing = ABLATED_FEATURES - set(df.columns)
    if missing:
        logger.warning("Ablated features not found in parquet: %s", missing)
    df = df.drop(columns=dropped)
    logger.info("Loaded %d rows. Dropped %s. Remaining columns: %d",
                len(df), sorted(dropped), df.shape[1])
    return df


def run_classifier_pipeline(df: pd.DataFrame, output_dir: Path, random_seed: int = RANDOM_SEED) -> Path:
    """Run binary classification (SRQ1/SRQ2) on the ablated matrix; return path to shap_importance.csv."""
    logger.info("=== CLASSIFIER PIPELINE (ablated: %s) ===", sorted(ABLATED_FEATURES))

    experiment_results: dict[int, dict] = {}
    for level_k in [1, 2, 3]:
        logger.info("=== Level %d (%s) vs. original ===", level_k, _CLF_LEVEL_NAMES[level_k])
        result = run_experiment(df, level_k, random_seed)
        if result is not None:
            experiment_results[level_k] = result

    if not experiment_results:
        logger.error("No classifier experiments completed. Aborting.")
        return output_dir / "shap_importance.csv"

    available_levels = sorted(experiment_results.keys())

    result_rows = []
    for level_k, res in experiment_results.items():
        for model_name, auc_k, std_auc_k, f1_k, std_f1_k in [
            ("XGBoost",      "xgb_mean_auc", "xgb_std_auc", "xgb_mean_f1", "xgb_std_f1"),
            ("RandomForest", "rf_mean_auc",  "rf_std_auc",  "rf_mean_f1",  "rf_std_f1"),
        ]:
            result_rows.append({
                "level":      level_k,
                "model_name": model_name,
                "mean_auc":   res[auc_k],
                "std_auc":    res[std_auc_k],
                "mean_f1":    res[f1_k],
                "std_f1":     res[std_f1_k],
            })
    results_df = pd.DataFrame(result_rows)

    feature_cols = experiment_results[available_levels[0]]["feature_cols"]
    shap_records: dict[str, dict] = {feat: {"feature_name": feat} for feat in feature_cols}
    for level_k in available_levels:
        for feat, imp in zip(feature_cols, experiment_results[level_k]["mean_shap"]):
            shap_records[feat][f"mean_shap_level{level_k}"] = float(imp)

    shap_df = pd.DataFrame(shap_records.values()).reset_index(drop=True)
    shap_df["mean_shap_overall"] = shap_df[[f"mean_shap_level{k}" for k in available_levels]].mean(axis=1)

    if len(available_levels) < 3:
        logger.warning("Only %d/3 levels available — fragility labels are provisional.", len(available_levels))
    shap_df = categorize_fragility(shap_df)

    corr_dfs = []
    for level_k in available_levels:
        corr          = compute_fold_shap_correlations(experiment_results[level_k]["fold_shap_list"], feature_cols)
        corr["level"] = level_k
        corr_dfs.append(corr)
    fold_corr_df = pd.concat(corr_dfs, ignore_index=True)

    for level_k in available_levels:
        sub = fold_corr_df[fold_corr_df["level"] == level_k]["spearman_r"]
        logger.info("Level %d (%s) fold-SHAP Spearman r: mean=%.3f min=%.3f max=%.3f",
                    level_k, _CLF_LEVEL_NAMES[level_k], sub.mean(), sub.min(), sub.max())

    save_outputs(results_df, shap_df, fold_corr_df, output_dir)
    plot_shap_importance(shap_df, output_dir / "shap_importance_plot.png")
    plot_auc_comparison(results_df, output_dir / "auc_comparison_plot.png")
    logger.info("Classifier pipeline complete. Outputs in '%s'.", output_dir)
    return output_dir / "shap_importance.csv"


def run_quality_pipeline(
    df: pd.DataFrame,
    shap_path: Path,
    output_dir: Path,
    random_seed: int = RANDOM_SEED,
    n_folds: int = N_FOLDS,
) -> None:
    logger.info("=== QUALITY PIPELINE (ablated: %s) ===", sorted(ABLATED_FEATURES))

    shap_df = pd.read_csv(shap_path)
    logger.info("Loaded %d features. Fragility breakdown:\n%s",
                len(shap_df), shap_df["fragility_label"].value_counts().to_string())

    score_min, score_max = get_score_range(df)
    subsets              = build_subsets(df, shap_df)

    fold_df    = run_quality_experiments(df, subsets, score_min, score_max, random_seed, n_folds)
    results_df = aggregate_results(fold_df)
    logger.info("Quality results summary:\n%s", results_df.to_string(index=False))

    results_df.to_csv(output_dir / "quality_results.csv", index=False)
    plot_qwk_degradation(results_df, output_dir / "qwk_degradation_plot.png")

    deg_summary = compute_degradation_summary(fold_df)
    deg_summary.to_csv(output_dir / "qwk_degradation_summary.csv", index=False)
    plot_degradation_summary(deg_summary, output_dir / "qwk_degradation_summary_plot.png")

    per_prompt_df = run_experiments_per_prompt(df, subsets, score_min, score_max, random_seed, n_folds)
    plot_qwk_heatmap(per_prompt_df, output_dir / "qwk_per_prompt_heatmap.png")

    cv_preds = collect_cv_predictions(df, subsets, score_min, score_max, random_seed, n_folds)
    plot_scatter_pred_actual(cv_preds, results_df, output_dir / "scatter_pred_actual.png", score_min, score_max)

    perm_df = compute_permutation_importance(df, subsets, score_min, score_max, random_seed)
    if not perm_df.empty:
        perm_df.to_csv(output_dir / "robust_feature_permutation.csv", index=False)
        plot_permutation_importance(perm_df, output_dir / "robust_feature_permutation_plot.png")

    score_dists = collect_score_distributions(df, subsets, score_min, score_max, random_seed)
    plot_score_distributions(score_dists, output_dir / "score_distribution_plot.png", score_min, score_max)

    logger.info("Quality pipeline complete. Outputs in '%s'.", output_dir)


def main(
    features_path: Path = FEATURES_PATH,
    output_dir: Path    = OUTPUT_DIR,
    random_seed: int    = RANDOM_SEED,
    n_folds: int        = N_FOLDS,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(random_seed)
    df       = load_and_ablate(features_path)
    shap_csv = run_classifier_pipeline(df, output_dir, random_seed)
    run_quality_pipeline(df, shap_csv, output_dir, random_seed, n_folds)
    logger.info("Ablation pipeline complete. All outputs in '%s'.", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Ablation study excluding {sorted(ABLATED_FEATURES)} from the feature matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--features-path", default=str(FEATURES_PATH))
    parser.add_argument("--output-dir",    default=str(OUTPUT_DIR))
    parser.add_argument("--random-seed",   type=int, default=RANDOM_SEED)
    parser.add_argument("--n-folds",       type=int, default=N_FOLDS)
    args = parser.parse_args()
    main(Path(args.features_path), Path(args.output_dir), args.random_seed, args.n_folds)