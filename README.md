# House Price Prediction
> In-depth regression analysis — without log transform on target

![Python](https://img.shields.io/badge/Python-3.x-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-green)

A complete ML pipeline for predicting residential house prices — covering EDA, feature engineering, outlier handling, target encoding, and model comparison.

## Dataset
- 4,600 rows · 18 original features · 0 null values
- Target: `price` (float64) · Location: Washington state, USA · 77 unique ZIP codes

## Pipeline
1. **Data loading & EDA** — shape, dtypes, value counts
2. **Column pruning** — dropped `street`, `country`, `statezip`
3. **Type conversions** — `date` → datetime, `city`/`zip_code` → category
4. **Visualization** — histograms, box plots, scatter vs price, correlation heatmap
5. **Feature engineering** — `has_basement`, `house_age`, `is_renovated`, `total_sqft`
6. **Outlier handling** — bedrooms clipped, price capped at 99th percentile
7. **Target encoding** — `zip_code` and `city` encoded as mean price per group
8. **Leakage removal** — dropped `price_per_sqft` (derived from target)
9. **Model training** — Linear Regression, Random Forest (×2), XGBoost with 5-fold CV

## Results

| Model | Test R² | CV Mean R² |
|---|---|---|
| Linear Regression | ❌ (bug) | — |
| Random Forest (OHE) | 0.664 | — |
| Random Forest (Target Enc.) | 0.716 | 0.701 |
| **XGBoost** | **0.724** | **0.683** |

## Key Learnings
- Target encoding on `zip_code`/`city` significantly outperformed one-hot encoding
- `price_per_sqft` caused data leakage (inflated R² to ~0.91) — correctly identified and removed
- Log transform on price intentionally skipped — preserved as commented code for comparison

## Setup
```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost imbalanced-learn
```
Place `House_Prices.csv` in the same directory and run all cells.
