"""Entry point for binary classification pipeline (SRQ1: detection accuracy, SRQ2: feature fragility)."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from modules.classifier import (
    LEVEL_NAMES,
    categorize_fragility,
    compute_fold_shap_correlations,
    load_features,
    plot_auc_comparison,
    plot_shap_importance,
    run_experiment,
    save_outputs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FEATURES_PATH = Path("features.parquet")
OUTPUT_DIR    = Path("results")
RANDOM_SEED   = 42


def relabel_only(output_dir: Path = OUTPUT_DIR) -> None:
    """Re-apply fragility labels to an existing shap_importance.csv and regenerate plots without rerunning experiments."""
    shap_csv    = output_dir / "shap_importance.csv"
    results_csv = output_dir / "classification_results.csv"

    if not shap_csv.exists():
        logger.error("'%s' not found — run the full pipeline first.", shap_csv)
        return

    logger.info("=== Relabel-only mode — loading '%s' ===", shap_csv)
    shap_df = pd.read_csv(shap_csv)
    shap_df = categorize_fragility(shap_df)

    shap_level_cols = [c for c in shap_df.columns if c.startswith("mean_shap_level")]
    shap_out_cols   = ["feature_name"] + shap_level_cols + ["mean_shap_overall", "fragility_label"]
    shap_df[shap_out_cols].to_csv(shap_csv, index=False)
    logger.info("Overwrote '%s' with updated fragility labels", shap_csv)

    plot_shap_importance(shap_df, output_dir / "shap_importance_plot.png")

    if results_csv.exists():
        plot_auc_comparison(pd.read_csv(results_csv), output_dir / "auc_comparison_plot.png")
    else:
        logger.warning("'%s' not found — skipping AUC comparison plot.", results_csv)

    logger.info("Relabel-only pass complete. Outputs in '%s'.", output_dir)


def main(
    features_path: Path = FEATURES_PATH,
    output_dir: Path = OUTPUT_DIR,
    random_seed: int = RANDOM_SEED,
) -> None:
    """Run three GroupKFold experiments (L1-L3 vs. L0), aggregate SHAP, label fragility, and save all outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(random_seed)

    df = load_features(features_path)

    experiment_results: dict[int, dict] = {}
    for level_k in [1, 2, 3]:
        logger.info("=== Level %d (%s) vs. original ===", level_k, LEVEL_NAMES[level_k])
        result = run_experiment(df, level_k, random_seed)
        if result is not None:
            experiment_results[level_k] = result

    if not experiment_results:
        logger.error("No experiments completed. Aborting.")
        return

    available_levels = sorted(experiment_results.keys())
    logger.info("Completed experiments for levels: %s", available_levels)

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

    feature_cols  = experiment_results[available_levels[0]]["feature_cols"]
    shap_records: dict[str, dict] = {feat: {"feature_name": feat} for feat in feature_cols}
    for level_k in available_levels:
        for feat, importance in zip(feature_cols, experiment_results[level_k]["mean_shap"]):
            shap_records[feat][f"mean_shap_level{level_k}"] = float(importance)

    shap_df = pd.DataFrame(shap_records.values()).reset_index(drop=True)
    available_shap_cols          = [f"mean_shap_level{k}" for k in available_levels]
    shap_df["mean_shap_overall"] = shap_df[available_shap_cols].mean(axis=1)

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
        logger.info(
            "Level %d (%s) fold-SHAP Spearman r: mean=%.3f min=%.3f max=%.3f",
            level_k, LEVEL_NAMES[level_k], sub.mean(), sub.min(), sub.max(),
        )

    save_outputs(results_df, shap_df, fold_corr_df, output_dir)
    plot_shap_importance(shap_df, output_dir / "shap_importance_plot.png")
    plot_auc_comparison(results_df, output_dir / "auc_comparison_plot.png")

    logger.info("Pipeline complete. All outputs written to '%s'.", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Binary classification pipeline (SRQ1/SRQ2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--features-path", default=str(FEATURES_PATH))
    parser.add_argument("--output-dir",    default=str(OUTPUT_DIR))
    parser.add_argument("--random-seed",   type=int, default=RANDOM_SEED)
    parser.add_argument("--relabel-only",  action="store_true", default=False,
        help="Skip experiments; reload shap_importance.csv, relabel, and replot.")

    args = parser.parse_args()
    if args.relabel_only:
        relabel_only(Path(args.output_dir))
    else:
        main(Path(args.features_path), Path(args.output_dir), args.random_seed)