# House_Price_Detection
🏡House-Price_Prediction - End-to-end Machine Learning project for predicting house prices using advanced EDA, feature engineering, and regression models.   Includes data cleaning, outlier handling, feature selection, encoding, and model building with performance evaluation.


# 🏠 House Price Prediction - Machine Learning Project

## 📌 Overview
This project focuses on predicting house prices using a complete end-to-end Machine Learning pipeline. It covers all key stages of a real-world ML workflow, including data cleaning, exploratory data analysis (EDA), feature engineering, feature selection, and model building.

---

## 📊 Dataset
The dataset contains housing-related features such as:
- Bedrooms, Bathrooms
- Square footage (living area, lot, basement)
- Floors, Condition, View
- Location details (ZIP code)
- Year built and renovation details
- Waterfront availability

---

## ⚙️ Data Preprocessing

### 🔹 Column Removal
- Removed irrelevant and redundant columns such as `street`, `country`, and `sqft_above`
- Eliminated multicollinearity by dropping highly correlated features

### 🔹 Feature Engineering
- Extracted `zip_code` from `statezip`
- Created `house_age` from `yr_built`
- Created binary features:
  - `has_basement`
  - `is_renovated`

---

## 📈 Exploratory Data Analysis (EDA)

- Visualized distributions of all numerical features
- Identified skewness and outliers using histograms and boxplots
- Analyzed relationships between features and target (`price`) using scatter plots
- Used correlation heatmap to identify important and redundant features

---

## 🔥 Key Insights

- `sqft_living` is the strongest predictor of house price
- Location (`zip_code`) plays a significant role
- Features like `waterfront` and `view` have strong impact despite low correlation values
- Many features showed right-skewed distributions and required transformation
- Outliers significantly affected the dataset and were handled accordingly

---

## 🔄 Data Transformation

- Applied **log transformation** to skewed features:
  - `price`, `sqft_living`, `sqft_lot`, `sqft_basement`
- Removed extreme outliers using quantile-based filtering
- Converted categorical features (`zip_code`, `floors`) appropriately

---

## 🎯 Feature Selection

### ✅ Selected Features:
- sqft_living, bathrooms, bedrooms
- floors, waterfront, view, condition
- sqft_basement, has_basement
- house_age, is_renovated
- zip_code

### ❌ Removed Features:
- sqft_above (high correlation)
- yr_built, yr_renovated (replaced with engineered features)
- sqft_lot (low predictive power)

---

## 🤖 Model Building

Implemented multiple regression models:
- Linear Regression
- Random Forest Regressor

---

## 📊 Model Evaluation

- Evaluated models using:
  - R² Score
  - Root Mean Squared Error (RMSE)

- Random Forest performed better due to:
  - Handling non-linear relationships
  - Robustness to outliers and feature interactions

---

## 🚀 Conclusion

This project demonstrates a complete ML workflow from raw data to model evaluation. It highlights the importance of:
- Proper feature engineering
- Handling skewness and outliers
- Understanding feature relationships
- Selecting the right model

---

## 🔮 Future Improvements

- Implement XGBoost for improved performance
- Perform hyperparameter tuning
- Add cross-validation
- Deploy the model using a web application

---

## 🧠 Key Skills Demonstrated

- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Feature Selection  
- Regression Modeling  
- Model Evaluation  
