# 🍷 Wine Quality Prediction using Machine Learning

This project predicts the **quality of red wine** based on its physicochemical properties using multiple machine learning models.  
The goal is to identify which model best estimates wine quality and explore how chemical attributes such as acidity, alcohol, and sulphates influence wine ratings.

---

## 📊 Dataset Overview

- **Source:** UCI Machine Learning Repository – *Wine Quality Data Set*  
- **Samples:** 1,599 wines (reduced to 1,359 after removing duplicates)  
- **Features:** 11 physicochemical variables + 1 target (`quality`)  
- **Target Variable:** `quality` (scores range from 3–8)

| Feature | Description |
|----------|-------------|
| fixed acidity | concentration of nonvolatile acids |
| volatile acidity | amount of acetic acid |
| citric acid | adds freshness and flavor |
| residual sugar | leftover sugar after fermentation |
| chlorides | salt content |
| free sulfur dioxide | prevents microbial growth |
| total sulfur dioxide | total SO₂ concentration |
| density | relative density of the wine |
| pH | measure of acidity |
| sulphates | stability enhancer |
| alcohol | alcohol content by volume |

---

## 🧹 Data Preprocessing

- Checked for **missing values** → none found  
- Removed **240 duplicate rows**  
- Detected **class imbalance** (most wines rated 5 or 6)  
- Created **correlation heatmaps** to identify feature relationships  
- Applied **StandardScaler** for SVR model input normalization  

---

## 🧠 Models Implemented

| Model | Description | R² Score | MAE |
|--------|--------------|----------|------|
| **SVR (RBF Kernel)** | Captures non-linear relationships | 0.41 | 0.47 |
| **Random Forest Regressor** | Ensemble of decision trees | **0.45** | **0.44** |
| **XGBoost Regressor** | Gradient-boosted trees | 0.44 | 0.42 |
| **Gradient Boosting Regressor** | Sequential additive trees | 0.39 | 0.48 |

🟢 **Best Model:** Random Forest Regressor (R² = 0.45, MAE = 0.44)

---

## ⚙️ Model Optimization

- **GridSearchCV** used to tune SVR parameters:
  ```python
  {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']}
