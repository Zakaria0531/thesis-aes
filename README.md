# Feature Robustness in Automated Essay Scoring Under AI Writing Assistance

This thesis investigates which features used in Automated Essay Scoring (AES) are robust to AI writing assistance. Using the ASAP 2.0 dataset of approximately 24,000 student essays, each essay is transformed into three AI-assisted variants via GPT-4o-mini: grammar correction, style enhancement, and substantive revision. Binary classifiers (XGBoost, Random Forest) are trained to distinguish original from AI-assisted essays per intervention level. SHAP feature importance is used to identify which AES features are most affected (fragile) versus stable (robust) under AI assistance. A final quality prediction model using only robust features is evaluated using Quadratic Weighted Kappa to assess whether meaningful essay scoring remains feasible when students use AI tools.

---

## Research Questions

**RQ: To what extent can automated essay scoring features distinguish AI-assisted from human-written text, and which features remain robust for quality assessment under AI assistance?**

- **SRQ1:** How well can binary classifiers distinguish original student essays from AI-assisted variants across three intervention levels (grammar correction, style enhancement, substantive revision)?
- **SRQ2:** Which AES features show the highest importance for detecting AI assistance versus the lowest importance?
- **SRQ3:** Can quality assessment models using only robust features show less performance degradation than all-feature models when applied to AI-assisted essays?

---

## Repository Structure

```
.
├── data/                                    # AI-assisted essay variants (not tracked in git)
├── results/
│   ├── classification_results.csv           # AUC-ROC and macro-F1 by model and intervention level
│   ├── shap_importance.csv                  # Mean absolute SHAP values per feature across folds
│   ├── fold_shap_correlations.csv           # Rank correlations of SHAP importance across CV folds
│   ├── quality_results.csv                  # QWK per feature subset and intervention level
│   ├── qwk_degradation_summary.csv          # Absolute and relative QWK drop (level 0 → level 3)
│   ├── robust_feature_permutation.csv       # Permutation importance for robust features
│   ├── auc_comparison_plot.png              # AUC-ROC by model and intervention level
│   ├── shap_importance_plot.png             # SHAP feature importance bar chart
│   ├── shap_vs_cv.png                       # SHAP importance vs. cross-fold stability
│   ├── qwk_degradation_plot.png             # QWK by intervention level per feature subset
│   ├── qwk_degradation_summary_plot.png     # Absolute and relative QWK degradation bar chart
│   ├── qwk_per_prompt_heatmap.png           # Per-prompt QWK heatmap across intervention levels
│   ├── scatter_pred_actual.png              # Predicted vs actual score scatter (level 0 and 3)
│   ├── robust_feature_permutation_plot.png  # Permutation importance bar chart (robust subset)
│   ├── score_distribution_plot.png          # KDE of predicted scores by intervention level
│   ├── score_correlation.png                # Score correlation across conditions
│   ├── feature_distributions.png            # Feature value distributions across intervention levels
│   ├── feature_shift.png                    # Feature shift from original to AI-assisted
│   └── strategy_comparison.png             # Comparison across feature subset strategies
├── Results Ablation/                        # Same outputs as results/ for the length-feature ablation
├── modules/
│   ├── classifier.py                        # CV logic, SHAP computation, and fragility labelling
│   ├── features.py                          # Feature extraction: surface (12), readability (7), coherence (2), syntactic (5)
│   └── prompts.py                           # GPT-4o-mini system prompt templates per intervention level
├── interventions.py                         # Generate AI-assisted essay variants via GPT-4o-mini
├── feature_pipeline.py                      # Extract AES features and write features.parquet and feature_delta.parquet
├── classifier_pipeline.py                   # Binary classification pipeline (SRQ1 + SRQ2)
├── quality_pipeline.py                      # Quality assessment pipeline (SRQ3)
├── ablation_pipeline.py                     # Ablation study: classifier + quality pipeline excluding length features
├── features.parquet                         # Extracted features for all essays and variants
├── feature_delta.parquet                    # Per-feature deltas between original and AI-assisted variants
├── feature_analysis.ipynb                   # Exploratory feature analysis and visualizations
└── requirements.txt
```

> **Note:** The `data/` directory is not tracked in git due to file size. See the [`data/` section](#data) below for details and how to request access.

### `modules/`

The `modules/` directory is the shared library for all pipelines. Entry-point scripts import from here and contain no logic of their own.

- **`features.py`:** Linguistic feature extraction across four families: surface (12 features), readability (7), coherence (2), and syntactic (5). Core public API: `get_features(text, nlp)` where `nlp` is loaded once by the caller via `load_spacy_model()`.
- **`classifier.py`:** Binary classification utilities: essay-ID grouped `GroupKFold` CV, XGBoost and Random Forest training, fold-local mean imputation, SHAP global importance via `TreeExplainer`, fragility labelling via Q1/Q3 cutoffs on mean SHAP importance, and Spearman fold-stability correlations.
- **`prompts.py`:** GPT-4o-mini system prompt templates for each of the three intervention levels (light / medium / heavy).

### `results/` and `Results Ablation/`

`results/` contains all outputs from the main experimental pipeline. `Results Ablation/` mirrors the same schema for the ablation study, which excludes `word_count` and `sentence_count` from all models to verify that findings are not driven by essay length alone.

### `data/`

Contains the ASAP 2.0 essay corpus alongside the three AI-assisted variants generated for each essay (grammar-corrected, style-enhanced, substantively revised). This directory is **not tracked in git** due to file size. The data is available on request — contact the author at zakariahader@gmail.com.

---

## Requirements

```
lexicalrichness==0.5.1
matplotlib==3.10.9
numpy==2.4.6
openai==2.38.0
pandas==3.0.3
python-dotenv==1.2.2
scikit_learn==1.8.0
scipy==1.17.1
shap==0.51.0
spacy==3.8.13
textstat==0.7.13
tqdm==4.67.3
xgboost==3.2.0
```

Install with:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Usage

```bash
# 1. Generate AI-assisted essay variants
python interventions.py

# 2. Extract linguistic features
python feature_pipeline.py

# 3. Run binary classification pipeline (SRQ1 + SRQ2)
python classifier_pipeline.py

# 4. Run quality assessment pipeline (SRQ3)
python quality_pipeline.py

# 5. Run ablation study (optional)
python ablation_pipeline.py
```

Each script accepts `--help` for available options.
