"""Quality assessment pipeline (SRQ3): QWK stability across feature subsets and intervention levels."""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FEATURES_PATH = Path("features.parquet")
SHAP_PATH     = Path("results/shap_importance.csv")
OUTPUT_DIR    = Path("results")
RANDOM_SEED   = 42
N_FOLDS       = 5

LEVEL_NAMES   = {0: "original", 1: "light", 2: "medium", 3: "heavy"}
QWK_THRESHOLD = 0.70

_CB_COLORS = {
    "all":     "#0072B2",
    "robust":  "#009E73",
    "fragile": "#D55E00",
}

_XGB_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, verbosity=0,
)


def load_data(features_path: Path, shap_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading features from '%s' ...", features_path)
    df = pd.read_parquet(features_path)
    logger.info("Loaded %d rows. Levels present: %s", len(df), sorted(df["level"].unique()))

    logger.info("Loading SHAP importances from '%s' ...", shap_path)
    shap_df = pd.read_csv(shap_path)
    logger.info("Loaded %d features. Fragility breakdown:\n%s",
                len(shap_df), shap_df["fragility_label"].value_counts().to_string())
    return df, shap_df


def build_subsets(df: pd.DataFrame, shap_df: pd.DataFrame) -> dict[str, list[str]]:
    meta_cols    = {"essay_id", "prompt_name", "level", "score"}
    all_features = [c for c in df.columns if c not in meta_cols]

    def _filter(label: str) -> list[str]:
        return [f for f in shap_df.loc[shap_df["fragility_label"] == label, "feature_name"] if f in all_features]

    subsets = {"all": all_features, "robust": _filter("robust"), "fragile": _filter("fragile")}
    for name, cols in subsets.items():
        logger.info("Subset '%s': %d features", name, len(cols))
    return subsets


def get_score_range(df: pd.DataFrame) -> tuple[int, int]:
    originals = df[df["level"] == 0]["score"]
    score_min, score_max = int(originals.min()), int(originals.max())
    logger.info("Score range from original essays: [%d, %d]", score_min, score_max)
    return score_min, score_max


def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray, score_min: int, score_max: int) -> float | None:
    y_pred_r = np.clip(np.round(y_pred), score_min, score_max).astype(int)
    y_true_r = np.clip(np.round(y_true), score_min, score_max).astype(int)
    try:
        return float(cohen_kappa_score(y_true_r, y_pred_r, weights="quadratic"))
    except ValueError as exc:
        logger.warning("QWK undefined (%s) — fold skipped.", exc)
        return None


def _fit_xgb(X: np.ndarray, y: np.ndarray, random_seed: int) -> XGBRegressor:
    model = XGBRegressor(**_XGB_PARAMS, random_state=random_seed)
    model.fit(X, y)
    return model


def run_quality_experiments(
    df: pd.DataFrame,
    subsets: dict[str, list[str]],
    score_min: int,
    score_max: int,
    random_seed: int = RANDOM_SEED,
    n_folds: int = N_FOLDS,
) -> pd.DataFrame:
    """Run GroupKFold CV per feature subset; train on level 0, evaluate on levels 0-3."""
    df_orig = df[df["level"] == 0].copy().reset_index(drop=True)
    df_ai   = {lvl: df[df["level"] == lvl].copy().reset_index(drop=True) for lvl in [1, 2, 3]}
    gkf     = GroupKFold(n_splits=n_folds)
    groups  = df_orig["essay_id"].values
    y_orig  = df_orig["score"].values
    records: list[dict] = []

    for subset_name, feature_cols in subsets.items():
        if not feature_cols:
            logger.warning("Subset '%s' has no features — skipping.", subset_name)
            continue
        logger.info("=== Subset: %-8s (%d features) ===", subset_name, len(feature_cols))
        X_orig = df_orig[feature_cols].values

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_orig, y_orig, groups)):
            model = _fit_xgb(X_orig[train_idx], y_orig[train_idx], random_seed)

            qwk_0 = compute_qwk(y_orig[test_idx], model.predict(X_orig[test_idx]), score_min, score_max)
            if qwk_0 is not None:
                records.append({"feature_subset": subset_name, "level": 0, "fold": fold_idx, "qwk": qwk_0})

            for lvl, df_lvl in df_ai.items():
                if df_lvl.empty:
                    continue
                qwk_k = compute_qwk(df_lvl["score"].values, model.predict(df_lvl[feature_cols].values), score_min, score_max)
                if qwk_k is not None:
                    records.append({"feature_subset": subset_name, "level": lvl, "fold": fold_idx, "qwk": qwk_k})

            logger.info(
                "  fold %d/%d  L0=%.3f  L1=%.3f  L2=%.3f  L3=%.3f",
                fold_idx + 1, n_folds,
                *(next((r["qwk"] for r in records
                        if r["feature_subset"] == subset_name and r["level"] == l and r["fold"] == fold_idx),
                       float("nan")) for l in [0, 1, 2, 3]),
            )

    return pd.DataFrame(records)


def aggregate_results(fold_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        fold_df.groupby(["feature_subset", "level"])["qwk"]
        .agg(mean_qwk="mean", std_qwk="std", n_folds="count")
        .reset_index()
    )
    agg["std_qwk"] = agg["std_qwk"].fillna(0.0)
    return agg


def save_quality_results(results_df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "quality_results.csv"
    results_df.to_csv(path, index=False)
    logger.info("Saved quality results → '%s' (%d rows)", path, len(results_df))
    return path


def plot_qwk_degradation(results_df: pd.DataFrame, output_path: Path) -> None:
    subset_labels = {"all": "All features", "robust": "Robust only", "fragile": "Fragile only"}
    x_labels = [f"{LEVEL_NAMES[l].capitalize()}\n(level {l})" for l in [0, 1, 2, 3]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for subset in ["all", "robust", "fragile"]:
        sub = results_df[results_df["feature_subset"] == subset].sort_values("level")
        if sub.empty:
            continue
        means, stds, levels = sub["mean_qwk"].values, sub["std_qwk"].values, sub["level"].values
        ax.plot(levels, means, marker="o", color=_CB_COLORS[subset], label=subset_labels[subset], linewidth=2)
        ax.fill_between(levels, means - stds, means + stds, color=_CB_COLORS[subset], alpha=0.15)

    ax.axhline(QWK_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
               label=f"Acceptability threshold (QWK = {QWK_THRESHOLD:.2f})")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Intervention level", labelpad=8)
    ax.set_ylabel("Quadratic Weighted Kappa (QWK)")
    ax.set_title("AES quality model performance across intervention levels\n(trained on original essays)")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved QWK degradation plot → '%s'", output_path)


def compute_degradation_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-fold absolute and relative QWK drop from level 0 to level 3."""
    pivot = (
        fold_df[fold_df["level"].isin([0, 3])]
        .pivot_table(index=["feature_subset", "fold"], columns="level", values="qwk")
        .reset_index()
    )
    pivot.columns.name = None
    pivot = pivot.rename(columns={0: "qwk_level0", 3: "qwk_level3"}).dropna(subset=["qwk_level0", "qwk_level3"])
    pivot["abs_drop"]     = pivot["qwk_level0"] - pivot["qwk_level3"]
    pivot["rel_drop_pct"] = pivot["abs_drop"] / pivot["qwk_level0"].abs() * 100

    summary = (
        pivot.groupby("feature_subset")
        .agg(
            mean_qwk_level0  =("qwk_level0",  "mean"),
            mean_qwk_level3  =("qwk_level3",  "mean"),
            mean_abs_drop    =("abs_drop",     "mean"),
            std_abs_drop     =("abs_drop",     "std"),
            mean_rel_drop_pct=("rel_drop_pct", "mean"),
            std_rel_drop_pct =("rel_drop_pct", "std"),
        )
        .reset_index()
    )
    summary[["std_abs_drop", "std_rel_drop_pct"]] = summary[["std_abs_drop", "std_rel_drop_pct"]].fillna(0.0)
    return summary


def plot_degradation_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    subset_order  = ["all", "robust", "fragile"]
    subset_labels = ["All features", "Robust only", "Fragile only"]
    colors        = [_CB_COLORS[s] for s in subset_order]
    y_pos         = np.arange(len(subset_order))
    indexed       = summary_df.set_index("feature_subset")

    def _get(col: str, subset: str) -> float:
        return float(indexed.loc[subset, col]) if subset in indexed.index else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, mean_col, std_col, xlabel, title in [
        (axes[0], "mean_abs_drop",     "std_abs_drop",     "QWK drop (level 0 → level 3)",        "Absolute QWK degradation"),
        (axes[1], "mean_rel_drop_pct", "std_rel_drop_pct", "Relative QWK drop (% of level-0 QWK)", "Relative QWK degradation"),
    ]:
        vals = [_get(mean_col, s) for s in subset_order]
        stds = [_get(std_col,  s) for s in subset_order]
        ax.barh(y_pos, vals, xerr=stds, color=colors, capsize=5, height=0.5, error_kw={"elinewidth": 1.2})
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(subset_labels if ax is axes[0] else [])
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)

    fig.suptitle("QWK degradation: original → heavily AI-assisted essays (level 3)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved degradation summary plot → '%s'", output_path)


def run_experiments_per_prompt(
    df: pd.DataFrame,
    subsets: dict[str, list[str]],
    score_min: int,
    score_max: int,
    random_seed: int = RANDOM_SEED,
    n_folds: int = N_FOLDS,
) -> pd.DataFrame:
    """Same CV setup as run_quality_experiments, but records QWK per prompt."""
    df_orig = df[df["level"] == 0].copy().reset_index(drop=True)
    df_ai   = {lvl: df[df["level"] == lvl].copy().reset_index(drop=True) for lvl in [1, 2, 3]}
    gkf     = GroupKFold(n_splits=n_folds)
    groups  = df_orig["essay_id"].values
    y_orig  = df_orig["score"].values
    records: list[dict] = []

    for subset_name, feature_cols in subsets.items():
        if not feature_cols:
            continue
        logger.info("Per-prompt experiment: subset '%s' ...", subset_name)
        X_orig = df_orig[feature_cols].values

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_orig, y_orig, groups)):
            model = _fit_xgb(X_orig[train_idx], y_orig[train_idx], random_seed)

            test_slice = df_orig.iloc[test_idx].copy()
            test_slice["_pred"] = model.predict(X_orig[test_idx])
            for prompt, grp in test_slice.groupby("prompt_name"):
                q = compute_qwk(grp["score"].values, grp["_pred"].values, score_min, score_max)
                if q is not None:
                    records.append({"feature_subset": subset_name, "prompt": prompt, "level": 0, "fold": fold_idx, "qwk": q})

            for lvl, df_lvl in df_ai.items():
                if df_lvl.empty:
                    continue
                lvl_slice = df_lvl.copy()
                lvl_slice["_pred"] = model.predict(df_lvl[feature_cols].values)
                for prompt, grp in lvl_slice.groupby("prompt_name"):
                    q = compute_qwk(grp["score"].values, grp["_pred"].values, score_min, score_max)
                    if q is not None:
                        records.append({"feature_subset": subset_name, "prompt": prompt, "level": lvl, "fold": fold_idx, "qwk": q})

    return pd.DataFrame(records)


def _sanitize_prompt(name: str) -> str:
    return name.replace("_", " ").replace("-", " ")


def plot_qwk_heatmap(per_prompt_df: pd.DataFrame, output_path: Path) -> None:
    subsets       = ["all", "robust", "fragile"]
    subset_labels = {"all": "All features", "robust": "Robust only", "fragile": "Fragile only"}
    levels        = [0, 1, 2, 3]
    level_labels  = [f"Level {l}\n({LEVEL_NAMES[l]})" for l in levels]

    agg = (
        per_prompt_df.groupby(["feature_subset", "prompt", "level"])["qwk"]
        .mean().reset_index()
    )
    prompts       = sorted(agg["prompt"].unique())
    prompt_labels = [_sanitize_prompt(p) for p in prompts]
    norm          = TwoSlopeNorm(vmin=min(agg["qwk"].min(), -0.1), vcenter=QWK_THRESHOLD, vmax=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    im_ref = None
    for ax, subset in zip(axes, subsets):
        sub    = agg[agg["feature_subset"] == subset]
        matrix = np.full((len(prompts), len(levels)), np.nan)
        for i, p in enumerate(prompts):
            for j, lvl in enumerate(levels):
                cell = sub[(sub["prompt"] == p) & (sub["level"] == lvl)]["qwk"]
                if not cell.empty:
                    matrix[i, j] = cell.iloc[0]

        im = ax.imshow(matrix, aspect="auto", norm=norm, cmap="RdYlGn")
        im_ref = im
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(level_labels, fontsize=8)
        ax.set_title(subset_labels[subset], fontweight="bold", pad=8)
        if ax is axes[0]:
            ax.set_yticks(range(len(prompts)))
            ax.set_yticklabels(prompt_labels, fontsize=8)
        else:
            ax.set_yticks([])

        for i in range(len(prompts)):
            for j in range(len(levels)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, fontweight="bold")

    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes, orientation="vertical", fraction=0.018, pad=0.03)
        cbar.set_label("Mean QWK", labelpad=8)
        cbar.ax.axhline(QWK_THRESHOLD, color="black", linewidth=1.5, linestyle="--")
        cbar.ax.text(1.6, QWK_THRESHOLD, f" {QWK_THRESHOLD:.2f}", va="center", ha="left",
                     fontsize=7, transform=cbar.ax.transData)

    fig.suptitle("Per-prompt QWK by intervention level (mean across CV folds)", fontweight="bold")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-prompt QWK heatmap → '%s'", output_path)


def collect_cv_predictions(
    df: pd.DataFrame,
    subsets: dict[str, list[str]],
    score_min: int,
    score_max: int,
    random_seed: int = RANDOM_SEED,
    n_folds: int = N_FOLDS,
) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Collect out-of-fold predictions (L0) and mean across-fold predictions (L3) for scatter plots."""
    df_orig = df[df["level"] == 0].copy().reset_index(drop=True)
    df_lv3  = df[df["level"] == 3].copy().reset_index(drop=True)
    gkf     = GroupKFold(n_splits=n_folds)
    groups  = df_orig["essay_id"].values
    y_orig  = df_orig["score"].values
    result: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}

    for subset_name in ("all", "robust"):
        feature_cols = subsets.get(subset_name, [])
        if not feature_cols:
            continue
        logger.info("Collecting CV predictions for scatter: subset '%s' ...", subset_name)

        X_orig        = df_orig[feature_cols].values
        X_lv3         = df_lv3[feature_cols].values if not df_lv3.empty else None
        y_lv3         = df_lv3["score"].values      if not df_lv3.empty else None
        pred_lv0      = np.full(len(df_orig), np.nan)
        preds_lv3_acc: list[np.ndarray] = []

        for train_idx, test_idx in gkf.split(X_orig, y_orig, groups):
            model = _fit_xgb(X_orig[train_idx], y_orig[train_idx], random_seed)
            pred_lv0[test_idx] = model.predict(X_orig[test_idx])
            if X_lv3 is not None:
                preds_lv3_acc.append(model.predict(X_lv3))

        def _clip_round(arr: np.ndarray) -> np.ndarray:
            return np.clip(np.round(arr), score_min, score_max).astype(int)

        subset_preds: dict[int, tuple[np.ndarray, np.ndarray]] = {
            0: (_clip_round(y_orig), _clip_round(pred_lv0)),
        }
        if preds_lv3_acc:
            subset_preds[3] = (_clip_round(y_lv3), _clip_round(np.mean(preds_lv3_acc, axis=0)))

        result[subset_name] = subset_preds
    return result


def plot_scatter_pred_actual(
    cv_predictions: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
    results_df: pd.DataFrame,
    output_path: Path,
    score_min: int,
    score_max: int,
) -> None:
    subset_labels = {"all": "All features", "robust": "Robust only"}
    level_labels  = {0: "Original (level 0)", 3: "Heavy AI-assisted (level 3)"}
    rng    = np.random.default_rng(RANDOM_SEED)
    jitter = 0.18

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for row_idx, lvl in enumerate([0, 3]):
        for col_idx, subset in enumerate(["all", "robust"]):
            ax = axes[row_idx][col_idx]
            if subset not in cv_predictions or lvl not in cv_predictions[subset]:
                ax.set_visible(False)
                continue

            y_true, y_pred = cv_predictions[subset][lvl]
            ax.scatter(
                y_true.astype(float) + rng.uniform(-jitter, jitter, len(y_true)),
                y_pred.astype(float) + rng.uniform(-jitter, jitter, len(y_pred)),
                alpha=0.06, s=2, color=_CB_COLORS[subset], rasterized=True,
            )
            diag = np.arange(score_min, score_max + 1)
            ax.plot(diag, diag, color="black", linewidth=1.2, linestyle="--", zorder=5)

            qwk_row = results_df[(results_df["feature_subset"] == subset) & (results_df["level"] == lvl)]
            qwk_val = qwk_row["mean_qwk"].values[0] if not qwk_row.empty else float("nan")

            ax.set_title(f"{subset_labels[subset]} — {level_labels[lvl]}\nQWK = {qwk_val:.3f}", fontsize=9)
            ax.set_xlabel("Actual score")
            ax.set_ylabel("Predicted score")
            ax.set_xticks(range(score_min, score_max + 1))
            ax.set_yticks(range(score_min, score_max + 1))
            ax.set_xlim(score_min - 0.5, score_max + 0.5)
            ax.set_ylim(score_min - 0.5, score_max + 0.5)
            ax.set_aspect("equal")

    fig.suptitle(
        "Predicted vs actual score\n(all-features and robust-only models; "
        "level 0 = CV holdout, level 3 = mean across folds)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved predicted vs actual scatter → '%s'", output_path)


def _make_qwk_scorer(score_min: int, score_max: int):
    def _scorer(estimator, X, y):
        y_pred = np.clip(np.round(estimator.predict(X)), score_min, score_max).astype(int)
        y_true = np.clip(np.round(y), score_min, score_max).astype(int)
        try:
            return cohen_kappa_score(y_true, y_pred, weights="quadratic")
        except ValueError:
            return 0.0
    return _scorer


def compute_permutation_importance(
    df: pd.DataFrame,
    subsets: dict[str, list[str]],
    score_min: int,
    score_max: int,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Train the robust-only model on all level-0 data and compute permutation importance (n_repeats=10)."""
    feature_cols = subsets.get("robust", [])
    if not feature_cols:
        logger.warning("No robust features — skipping permutation importance.")
        return pd.DataFrame()

    df_orig = df[df["level"] == 0].copy()
    X, y    = df_orig[feature_cols].values, df_orig["score"].values
    model   = _fit_xgb(X, y, random_seed)

    logger.info("Computing permutation importance (%d features, %d essays) ...", len(feature_cols), len(df_orig))
    result = permutation_importance(
        model, X, y, scoring=_make_qwk_scorer(score_min, score_max),
        n_repeats=10, random_state=random_seed, n_jobs=-1,
    )
    return (
        pd.DataFrame({
            "feature_name":    feature_cols,
            "mean_importance": result.importances_mean,
            "std_importance":  result.importances_std,
        })
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_permutation_importance(perm_df: pd.DataFrame, output_path: Path) -> None:
    if perm_df.empty:
        logger.warning("No permutation importance data — skipping plot.")
        return

    perm_sorted = perm_df.sort_values("mean_importance")
    y_pos       = np.arange(len(perm_sorted))

    fig, ax = plt.subplots(figsize=(7, max(3.0, len(perm_sorted) * 0.5)))
    ax.barh(y_pos, perm_sorted["mean_importance"], xerr=perm_sorted["std_importance"],
            color=_CB_COLORS["robust"], capsize=4, height=0.6, error_kw={"elinewidth": 1.2})
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace("_", " ") for f in perm_sorted["feature_name"]])
    ax.set_xlabel("Mean QWK decrease (permutation importance)")
    ax.set_title("Permutation importance of robust features\n(QWK scorer, trained on original essays, n_repeats=10)")
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved permutation importance plot → '%s'", output_path)


def collect_score_distributions(
    df: pd.DataFrame,
    subsets: dict[str, list[str]],
    score_min: int,
    score_max: int,
    random_seed: int = RANDOM_SEED,
) -> dict:
    distributions: dict = {}
    df_orig = df[df["level"] == 0].copy()

    for subset_name in ("all", "robust"):
        feature_cols = subsets.get(subset_name, [])
        if not feature_cols:
            continue
        logger.info("Collecting score distributions for subset '%s' ...", subset_name)
        model = _fit_xgb(df_orig[feature_cols].values, df_orig["score"].values, random_seed)
        distributions[subset_name] = {
            lvl: np.clip(model.predict(df[df["level"] == lvl][feature_cols].values), score_min, score_max)
            for lvl in [0, 1, 2, 3] if not df[df["level"] == lvl].empty
        }

    distributions["actual_original"] = df_orig["score"].values
    return distributions


def plot_score_distributions(distributions: dict, output_path: Path, score_min: int, score_max: int) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)
    actual_ref = distributions.get("actual_original")

    for ax, lvl in zip(axes, [0, 1, 2, 3]):
        if actual_ref is not None:
            sns.kdeplot(actual_ref, ax=ax, color="black", linestyle="--",
                        linewidth=1.2, label="Actual (original)", fill=False)
        for subset_name in ("all", "robust"):
            if subset_name not in distributions or lvl not in distributions[subset_name]:
                continue
            sns.kdeplot(distributions[subset_name][lvl], ax=ax, color=_CB_COLORS[subset_name],
                        linewidth=1.5, label="All features" if subset_name == "all" else "Robust only",
                        fill=True, alpha=0.15)
        ax.set_xlabel("Score")
        ax.set_title(f"Level {lvl}: {LEVEL_NAMES[lvl].capitalize()}")
        ax.set_xlim(score_min - 0.5, score_max + 0.5)
        if ax is axes[0]:
            ax.set_ylabel("Density")
            ax.legend(fontsize=7.5, framealpha=0.9)
        else:
            ax.set_ylabel("")
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)

    fig.suptitle(
        "Predicted score distributions by intervention level\n"
        "(reference: actual scores of original essays — dashed black)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved score distribution plot → '%s'", output_path)


def main(
    features_path: Path = FEATURES_PATH,
    shap_path: Path     = SHAP_PATH,
    output_dir: Path    = OUTPUT_DIR,
    random_seed: int    = RANDOM_SEED,
    n_folds: int        = N_FOLDS,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(random_seed)

    df, shap_df          = load_data(features_path, shap_path)
    score_min, score_max = get_score_range(df)
    subsets              = build_subsets(df, shap_df)

    fold_df    = run_quality_experiments(df, subsets, score_min, score_max, random_seed, n_folds)
    results_df = aggregate_results(fold_df)
    logger.info("Quality results summary:\n%s", results_df.to_string(index=False))
    save_quality_results(results_df, output_dir)

    plot_qwk_degradation(results_df, output_dir / "qwk_degradation_plot.png")

    degradation_summary = compute_degradation_summary(fold_df)
    degradation_summary.to_csv(output_dir / "qwk_degradation_summary.csv", index=False)
    plot_degradation_summary(degradation_summary, output_dir / "qwk_degradation_summary_plot.png")

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

    logger.info("Pipeline complete. All outputs written to '%s'.", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quality assessment pipeline (SRQ3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--features-path", default=str(FEATURES_PATH))
    parser.add_argument("--shap-path",     default=str(SHAP_PATH))
    parser.add_argument("--output-dir",    default=str(OUTPUT_DIR))
    parser.add_argument("--random-seed",   type=int, default=RANDOM_SEED)
    parser.add_argument("--n-folds",       type=int, default=N_FOLDS)
    args = parser.parse_args()
    main(Path(args.features_path), Path(args.shap_path), Path(args.output_dir), args.random_seed, args.n_folds)