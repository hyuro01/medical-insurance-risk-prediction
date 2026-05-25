# medical-insurance-risk-classification-and-prediction

---

## Project Overview & Business Impact
In the health insurance sector, predicting financial liability and identifying high-risk policyholders are critical for actuarial pricing, risk management, and product design. 

This project delivers an end-to-end machine learning pipeline utilizing a Kaggle dataset of **100,000 records and 54 features**. It implements a dual-track modeling approach:
### 1. Regression Track: 
Predicting precise annual medical costs to optimize premium alignment.
### 2. Classification Track: 
Segmenting individuals into structured medical risk levels to enable automated underwriting and proactive health interventions.

---

## Core Architecture & Modeling Strategies

### 1. Regression Framework (Annual Medical Cost Prediction)
To capture both linear baselines and complex non-linear interactions in financial claims, a diverse suite of regression algorithms was deployed:
* **Models Evaluated**: Lasso Regression, Ridge Regression, Gradient Boosting Regressor (GBR), Random Forest Regressor, and XGBoost Regressor.
* **Performance Insights**: **Gradient Boosting Regressor (GBR)** and **Random Forest** outperformed the other models, demonstrating superior capability in handling non-linear claim distributions and minimizing prediction error (MAE/RMSE).

### 2. Classification Framework (Personal Medical Risk Segmentation)
To accurately categorize policyholders into distinct risk tiers for automated underwriting:
* **Models Evaluated**: Decision Tree, Random Forest Classifier, and XGBoost Classifier.
* **Performance Insights**: **XGBoost Classifier** achieved the highest performance metrics (Accuracy, F1-Score, and AUC), proving highly effective at capturing cross-feature interactions within high-dimensional data (54 features).

---

## Key Insights & Feature Importance
Post-modeling feature importance analysis yielded significant business intelligence for health insurance underwriting:
### 1. Primary Risk Drivers: 
**Age** and **Number of Chronic Diseases** emerged as the dominant features driving both cost escalation and high-risk classification.
### 2. Actuarial Application:
These findings provide a data-driven foundation for insurance firms to design targeted, age-stratified policies and specialized plans for individuals with pre-existing chronic conditions, shifting the business model from reactive payouts to proactive risk mitigation.
