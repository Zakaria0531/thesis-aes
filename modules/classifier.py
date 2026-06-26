"""Binary classification and SHAP-based fragility analysis for AES intervention detection."""

import logging
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

_META_COLS: frozenset[str] = frozenset({"essay_id", "prompt_name", "level", "score"})

LEVEL_NAMES: dict[int, str] = {1: "Light", 2: "Medium", 3: "Heavy"}
N_SPLITS = 5
TOP_N_FEATURES = 26

FRAGILITY_COLORS: dict[str, str] = {
    "fragile": "#d62728",
    "robust":  "#2ca02c",
    "neutral": "#7f7f7f",
}
MODEL_COLORS: dict[str, str] = {
    "XGBoost":      "#1f77b4",
    "RandomForest": "#ff7f0e",
}


def load_features(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in _META_COLS]
    logger.info(
        "Loaded features: %d rows | %d features | levels: %s | prompts: %d",
        len(df), len(feature_cols),
        sorted(df["level"].unique().tolist()),
        df["prompt_name"].nunique(),
    )
    return df


def prepare_binary_dataset(
    df: pd.DataFrame,
    level_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    subset = df[df["level"].isin([0, level_k])].copy()
    feature_cols = [c for c in df.columns if c not in _META_COLS]
    X      = subset[feature_cols].to_numpy(dtype=np.float64)
    y      = (subset["level"] == level_k).to_numpy(dtype=np.int32)
    groups = subset["essay_id"].to_numpy()
    logger.info(
        "Level %d binary dataset: %d samples (pos=%d neg=%d) %d features",
        level_k, len(y), int(y.sum()), int((1 - y).sum()), len(feature_cols),
    )
    return X, y, groups, feature_cols


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        logger.warning("Single-class validation fold — returning AUC = 0.5")
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def run_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_seed: int,
) -> dict:
    imputer = SimpleImputer(strategy="mean")
    X_tr  = imputer.fit_transform(X_tr)
    X_val = imputer.transform(X_val)

    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())

    xgb_clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=random_seed,
        eval_metric="auc",
        verbosity=0,
    )
    xgb_clf.fit(X_tr, y_tr)
    xgb_prob = xgb_clf.predict_proba(X_val)[:, 1]
    xgb_pred = xgb_clf.predict(X_val)

    rf_clf = RandomForestClassifier(n_estimators=300, random_state=random_seed, n_jobs=-1)
    rf_clf.fit(X_tr, y_tr)
    rf_prob = rf_clf.predict_proba(X_val)[:, 1]
    rf_pred = rf_clf.predict(X_val)

    explainer   = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(X_val)
    # older SHAP returns a list of per-class arrays; newer returns the positive class directly
    shap_arr = shap_values[1] if isinstance(shap_values, list) else shap_values

    return {
        "xgb_auc":       _safe_auc(y_val, xgb_prob),
        "xgb_f1":        float(f1_score(y_val, xgb_pred, average="macro", zero_division=0)),
        "rf_auc":        _safe_auc(y_val, rf_prob),
        "rf_f1":         float(f1_score(y_val, rf_pred, average="macro", zero_division=0)),
        "mean_abs_shap": np.abs(shap_arr).mean(axis=0),
    }


def run_experiment(
    df: pd.DataFrame,
    level_k: int,
    random_seed: int,
) -> dict | None:
    """Run 5-fold GroupKFold CV for one intervention level; groups by essay_id to prevent leakage."""
    df_level = df[df["level"].isin([0, level_k])]
    if df_level["level"].nunique() < 2:
        logger.warning("Level %d skipped: only one class present", level_k)
        return None

    X, y, groups, feature_cols = prepare_binary_dataset(df_level, level_k)
    gkf = GroupKFold(n_splits=N_SPLITS)

    xgb_aucs, xgb_f1s, rf_aucs, rf_f1s = [], [], [], []
    fold_shap_list: list[np.ndarray] = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        logger.info(
            "  Level %d (%s) | fold %d/%d (train=%d val=%d pos_rate_train=%.2f)",
            level_k, LEVEL_NAMES[level_k], fold_idx, N_SPLITS,
            len(train_idx), len(val_idx), y[train_idx].mean(),
        )
        fold_result = run_fold(X[train_idx], y[train_idx], X[val_idx], y[val_idx], random_seed)
        xgb_aucs.append(fold_result["xgb_auc"])
        xgb_f1s.append(fold_result["xgb_f1"])
        rf_aucs.append(fold_result["rf_auc"])
        rf_f1s.append(fold_result["rf_f1"])
        fold_shap_list.append(fold_result["mean_abs_shap"])

    logger.info(
        "Level %d  XGB AUC %.3f±%.3f  RF AUC %.3f±%.3f",
        level_k, np.mean(xgb_aucs), np.std(xgb_aucs), np.mean(rf_aucs), np.std(rf_aucs),
    )
    return {
        "level":          level_k,
        "feature_cols":   feature_cols,
        "fold_shap_list": fold_shap_list,
        "mean_shap":      np.mean(fold_shap_list, axis=0),
        "xgb_mean_auc":   float(np.mean(xgb_aucs)),
        "xgb_std_auc":    float(np.std(xgb_aucs)),
        "xgb_mean_f1":    float(np.mean(xgb_f1s)),
        "xgb_std_f1":     float(np.std(xgb_f1s)),
        "rf_mean_auc":    float(np.mean(rf_aucs)),
        "rf_std_auc":     float(np.std(rf_aucs)),
        "rf_mean_f1":     float(np.mean(rf_f1s)),
        "rf_std_f1":      float(np.std(rf_f1s)),
    }


def categorize_fragility(shap_df: pd.DataFrame) -> pd.DataFrame:
    """Label features fragile / neutral / robust using Q1/Q3 cutoffs on mean_shap_overall."""
    shap_df = shap_df.copy()
    s = shap_df["mean_shap_overall"]

    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    logger.info("Fragility thresholds: robust ≤ %.5f | fragile ≥ %.5f", q1, q3)

    def _label(v: float) -> str:
        if v >= q3:
            return "fragile"
        if v <= q1:
            return "robust"
        return "neutral"

    shap_df["fragility_label"] = shap_df["mean_shap_overall"].map(_label)
    counts = shap_df["fragility_label"].value_counts().to_dict()
    logger.info(
        "Fragility labels: fragile=%d neutral=%d robust=%d",
        counts.get("fragile", 0), counts.get("neutral", 0), counts.get("robust", 0),
    )
    return shap_df


def compute_fold_shap_correlations(
    fold_shap_list: list[np.ndarray],
    feature_cols: list[str],
) -> pd.DataFrame:
    records = []
    for (i, shap_a), (j, shap_b) in combinations(enumerate(fold_shap_list), r=2):
        r, p = spearmanr(shap_a, shap_b)
        records.append({"fold_a": i, "fold_b": j, "spearman_r": float(r), "p_value": float(p)})
    return pd.DataFrame(records)


def plot_shap_importance(shap_df: pd.DataFrame, output_path: Path) -> None:
    plot_data = shap_df.sort_values("mean_shap_overall", ascending=True)
    colors    = [FRAGILITY_COLORS[lbl] for lbl in plot_data["fragility_label"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_data) * 0.45)))
    ax.barh(
        plot_data["feature_name"], plot_data["mean_shap_overall"],
        color=colors, edgecolor="white", linewidth=0.5,
    )
    ax.set_xlabel("Mean |SHAP value| (averaged across intervention levels)")
    ax.set_title(f"Feature SHAP Importance (n={len(plot_data)})\nColoured by Fragility Label", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        mpatches.Patch(color=FRAGILITY_COLORS["fragile"], label="Fragile"),
        mpatches.Patch(color=FRAGILITY_COLORS["robust"],  label="Robust"),
        mpatches.Patch(color=FRAGILITY_COLORS["neutral"], label="Neutral"),
    ], loc="lower right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SHAP importance plot → '%s'", output_path)


def plot_auc_comparison(results_df: pd.DataFrame, output_path: Path) -> None:
    levels      = sorted(results_df["level"].unique())
    model_order = ["XGBoost", "RandomForest"]
    bar_width   = 0.22
    x_centers   = np.arange(len(levels), dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    for m_idx, model_name in enumerate(model_order):
        subset = results_df[results_df["model_name"] == model_name].sort_values("level").reset_index(drop=True)
        offset = (m_idx - (len(model_order) - 1) / 2) * bar_width
        ax.bar(
            x_centers + offset, subset["mean_auc"].values,
            width=bar_width, label=model_name, color=MODEL_COLORS[model_name],
            yerr=subset["std_auc"].values, capsize=3,
            error_kw={"elinewidth": 1.2, "ecolor": "black", "capthick": 1.2},
            edgecolor="white", linewidth=0.5,
        )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="Random Guess (AUC = 0.5)")
    ax.set_xticks(x_centers)
    ax.set_xticklabels([LEVEL_NAMES[l] for l in levels])
    ax.set_xlabel("Intervention level")
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Classifier AUC by Intervention Level\n(error bars = ±1 SD across 5 folds)", fontsize=12)
    ax.legend(framealpha=0.9, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved AUC comparison plot → '%s'", output_path)


def save_outputs(
    results_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    fold_corr_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    results_df.to_csv(output_dir / "classification_results.csv", index=False)

    shap_level_cols = [c for c in shap_df.columns if c.startswith("mean_shap_level")]
    shap_out_cols   = ["feature_name"] + shap_level_cols + ["mean_shap_overall", "fragility_label"]
    shap_df[shap_out_cols].to_csv(output_dir / "shap_importance.csv", index=False)

    fold_corr_df.to_csv(output_dir / "fold_shap_correlations.csv", index=False)
    logger.info("Saved outputs to '%s'", output_dir)