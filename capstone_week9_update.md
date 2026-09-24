# Capstone Week 9 Update — ML Component

**Project:** ClaimGuard — Insurance Fraud Detection Platform  
**Author:** Shantel Wambui  
**Date:** September 2026  

---

## Which ML algorithm did you choose and why?

I trained two models: **Random Forest** and **XGBoost**, and selected **XGBoost** as the primary production model.

**Random Forest** was chosen as the baseline because it is robust to noisy sensor data, handles mixed feature types well, and produces reliable feature importance scores out of the box. It served as a strong benchmark.

**XGBoost** was chosen as the final model for three reasons:
1. It consistently achieved higher ROC-AUC scores across all 5 cross-validation folds
2. Its `scale_pos_weight` parameter directly addresses class imbalance at the algorithm level
3. It integrates natively with SHAP's `TreeExplainer`, producing faster and more accurate Shapley value computations than ensemble methods

In the ClaimGuard context, XGBoost mirrors what we use for fraud scoring — a gradient boosting model that learns iteratively from claim patterns, with each tree correcting errors from the previous one.

---

## How did you handle class imbalance?

The dataset had approximately **8% failure rate** (92:8 split) — a realistic imbalance for equipment failure scenarios. I applied a **two-layer imbalance strategy**:

**Layer 1 — SMOTE (Synthetic Minority Over-sampling Technique):**  
Applied to the training set only (after the train/test split to prevent data leakage). SMOTE generates synthetic minority-class samples by interpolating between existing failure observations in feature space. This gave the model balanced exposure to both classes during training without simply duplicating real observations.

**Layer 2 — Algorithm-level correction:**
- Random Forest: `class_weight='balanced'` — automatically adjusts sample weights inversely proportional to class frequency
- XGBoost: `scale_pos_weight = 92/8 ≈ 11.5` — tells the gradient boosting process to penalise missed failures more heavily

**Why not accuracy as a metric?**  
A naive model that always predicts "no failure" would achieve 92% accuracy while being completely useless. All evaluation used **Precision, Recall, F1-Score, and ROC-AUC** — metrics that expose the model's actual ability to catch failures.

---

## What is one insight from your Feature Importance analysis that surprised you?

The most surprising finding was that **`maintenance_lag_days`** (days since last service) ranked higher than `rotational_speed_rpm` in both models' feature importance.

I expected vibration and temperature to dominate entirely. They do rank 1st and 2nd respectively — but maintenance history as the 3rd driver was unexpected. It suggests that **operational discipline is as predictive as sensor readings**: machines that go too long without a service are more likely to fail even when their real-time sensor values look normal.

This has a direct implication for ClaimGuard: in insurance fraud detection, **claim history patterns** (how recently a policyholder filed, time gaps between claims) can be as informative as the claim details themselves. Temporal features deserve more attention in the fraud model's feature engineering.

The SHAP beeswarm confirmed this directionally — high `maintenance_lag_days` values (red dots) consistently pushed predictions to the right (toward failure), independently of other sensor readings.

---

## Artefacts added this week

| File | Description |
|---|---|
| `week9_operational_ml.ipynb` | Full ML notebook — 7 parts |
| `equipment_sensor_data.csv` | Synthetic dataset (3,000 rows) |
| `shap_summary.png` | SHAP beeswarm — global explainability |
| `shap_force_plot.png` | SHAP force plot — single high-risk machine |
| `shap_waterfall.png` | SHAP waterfall — feature contribution breakdown |
| `feature_importance.png` | RF and XGBoost importance comparison |
| `roc_curves.png` | ROC-AUC comparison |
| `Week9_Model_Explainer_Shantel.md` | Written model explainer report |
