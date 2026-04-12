# 🏠 House Price Prediction
### In-depth Regression Analysis — Without Log Transform on Target

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square) ![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-green?style=flat-square) ![Pandas](https://img.shields.io/badge/Pandas-Data-purple?style=flat-square) ![Seaborn](https://img.shields.io/badge/Seaborn-Viz-teal?style=flat-square) ![Random Forest](https://img.shields.io/badge/Random_Forest-Ensemble-blue?style=flat-square) ![Feature Engineering](https://img.shields.io/badge/Feature_Engineering-✓-blueviolet?style=flat-square)

A complete ML pipeline for predicting residential house prices — covering EDA, feature engineering, outlier handling, target encoding, and model comparison across Linear Regression, Random Forest, and XGBoost.

---

## 📊 Dataset Overview

| Attribute | Value |
|---|---|
| Total Rows | 4,600 |
| Original Features | 18 |
| Null Values | 0 |
| Target Column | `price` (float64) |
| Location | Washington State, USA |
| Unique ZIP Codes | 77 |

---

## 🔁 Pipeline Steps

| # | Step | Details |
|---|---|---|
| 1 | **Data Loading & EDA** | `df.info()`, `df.shape`, `value_counts` on city, statezip, country |
| 2 | **Column Pruning** | Dropped `street` (unique per row), `country` (all "USA"), `statezip` (replaced by `zip_code`) |
| 3 | **Type Conversions** | `date` → datetime64, `city` & `zip_code` → category, `floors` → category |
| 4 | **Visualization** | Histograms + KDE, box plots (numerical), count plots (categorical), scatter vs price, correlation heatmap |
| 5 | **Feature Engineering** | `has_basement`, `house_age` (2026 − yr_built), `is_renovated`, `total_sqft` |
| 6 | **Outlier Handling** | Removed bedrooms ≤ 0 or ≥ 7; capped price at 99th percentile |
| 7 | **Target Encoding** | `zip_code` and `city` replaced with mean price per group (fixes OHE noise) |
| 8 | **Leakage Removal** | `price_per_sqft` dropped — derived from target variable |
| 9 | **Model Training & Evaluation** | Linear Regression, Random Forest (×2 iterations), XGBoost — with 5-fold CV |

---

## 🤖 Model Results

| Model | Test R² | CV Mean R² | Notes |
|---|---|---|---|
| Linear Regression | ❌ Broken | — | Train/test scale mismatch bug |
| Random Forest (OHE) | 0.664 | — | Baseline |
| Random Forest (Target Enc.) | 0.716 | 0.701 | Improved with target encoding |
| **XGBoost** | **0.724** | **0.683** | Best performing model |

> ⚠️ One CV fold consistently underperforms (~0.27–0.32 R²), suggesting a non-uniform distribution split — worth investigating with stratified folds.

---

## 🏆 Top Features (Random Forest / XGBoost)

| Feature | Importance | Notes |
|---|---|---|
| `sqft_living` | ~44% | Strongest predictor |
| `zip_code_encoded` | ~35% | Location signal via target encoding |
| `price_per_sqft` | — | ❌ Removed — data leakage |
| `total_sqft` | ~6% | Engineered feature |
| `house_age` | ~4% | Engineered from `yr_built` |
| `city_encoded` | ~3% | Location signal via target encoding |
| `view`, `bathrooms`, `sqft_lot`, `bedrooms` | < 3% each | Supporting features |

---

## 💡 Key Decisions & Learnings

✅ **Target encoding outperformed OHE** — One-hot encoding `zip_code` and `city` created 70+ sparse columns with near-zero importance, adding noise without signal. Switching to target encoding improved R² from 0.664 → 0.716.

⚠️ **Log transform intentionally skipped** — The notebook title reflects this design choice. The commented-out `np.log1p()` cells preserve the option to compare both approaches.

✅ **`yr_built` → `house_age`** — Converted for better model interpretability and a more linear relationship with price.

⚠️ **Data leakage caught & fixed** — `price_per_sqft` inflated R² artificially to ~0.91 because it contained the target variable `price` in its formula. Correctly identified and removed before final evaluation.

---

## ⚙️ How to Run

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost imbalanced-learn
```

Place `House_Prices.csv` in the same directory, then open and run all cells in:

```
House_Price_Prediction_Without_Log_Transform_on_the_Target.ipynb
```

---

## 📁 Project Structure

```
├── House_Price_Prediction_Without_Log_Transform_on_the_Target.ipynb
├── House_Prices.csv
└── README.md
```
