# Dataset provenance

This directory contains the dataset used by the notebook.

- **Dataset:** Medical Insurance Cost Prediction
- **Creator:** Mohan Krishna Thalla
- **Source:** https://www.kaggle.com/datasets/mohankrishnathalla/medical-insurance-cost-prediction
- **Kaggle reference:** `mohankrishnathalla/medical-insurance-cost-prediction`
- **Version:** 1
- **Version date:** 2025-10-10
- **License:** CC0: Public Domain
- **Local filename:** `medical_insurance.csv`
- **Shape:** 100,000 records × 54 columns
- **SHA-256:** `94cbbfaf9468a668a0cd5395a6a2bda9e9041551fedc54b82f326808a746752a`

The CC0 license permits copying and redistribution. Attribution is retained here for provenance even though it is not legally required by CC0.

## Modeling boundary

The Kaggle data card states that predictors describe information available before the period represented by `annual_medical_cost`. The file itself does not provide row-level timestamps with which to verify this ordering. For that reason, the primary analysis excludes fields that are direct outcomes or strong target proxies and treats utilization/procedure counts as a separate sensitivity set.

Excluded from the primary models:

- record identifier;
- annual and monthly premiums;
- claims count, average claim amount, and total claims paid;
- the published `risk_score` and `is_high_risk` fields;
- the prediction target itself.

Refer to the notebook for the complete feature audit.
