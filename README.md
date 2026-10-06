# Leakage-Aware Medical Cost and High-Cost Risk Modeling

An empirical study of medical-cost regression and high-cost population identification using 100,000 insurance records. The project focuses on a question that is easy to overlook in tabular machine learning: **which fields would actually be available at prediction time?**

The original course version produced high scores while retaining premium, claims-summary, and derived risk fields. This revision replaces those results with an explicit prediction contract, a conservative feature audit, fixed train/validation/test splits, segment-level error analysis, and a separately evaluated high-cost classifier.

## Research questions

1. How well can annual medical cost be estimated using only conservative pre-period features?
2. How much apparent performance changes when ambiguous utilization-history variables are added?
3. Can a model identify a useful share of future high-cost cases without treating the prediction as an automated decision?

## Data

- **Source:** [Medical Insurance Cost Prediction](https://www.kaggle.com/datasets/mohankrishnathalla/medical-insurance-cost-prediction) by Mohan Krishna Thalla
- **Kaggle version:** v1, updated 2025-10-10
- **Size:** 100,000 rows and 54 columns
- **License:** [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
- **Target:** `annual_medical_cost`

The Kaggle data card states that predictor fields represent information available before the prediction period and that `annual_medical_cost` represents expenditure in the following year. Because the table does not contain row-level timestamps, this project still treats utilization and procedure counts as ambiguous and reports them only in a sensitivity analysis.

See [`data/README.md`](data/README.md) for provenance, integrity information, and feature-use boundaries.

## Prediction contract

Features are assigned before model fitting:

- **Strict features:** demographics, socioeconomic and lifestyle factors, baseline clinical measurements, disease indicators, and plan characteristics.
- **Ambiguous history:** healthcare utilization and procedure counts. These appear only in the sensitivity analysis.
- **Excluded target proxies:** identifiers, premiums, claims aggregates, undocumented derived risk labels/scores, and the target itself.

All preprocessing is learned from training data. Models are selected on a validation set, followed by one evaluation on the untouched test set. The fixed split is 60% training, 20% validation, and 20% testing with random seed 42.

## Results

### Annual-cost regression

| Evaluation | Model / comparison | R² | RMSE | MAE |
|---|---|---:|---:|---:|
| Strict primary estimate | Ridge | 0.1242 | 2,935.69 | 1,830.03 |
| Strict + ambiguous history | Ridge | 0.1793 | 2,841.75 | 1,769.79 |

- The strict Ridge model reduced test RMSE by **10.4%** relative to the median baseline.
- Adding ambiguous utilization history reduced RMSE by a further **3.2%**. This is a sensitivity result, not the primary prospective estimate.
- For the training-defined top 5% cost segment, MAE rose to **8,877.14**, showing that extreme-cost cases remain the main regression failure mode.

### High-cost identification

The high-cost label is derived using the 95th percentile of the **training** target distribution. A logistic-regression threshold is selected on validation data and then frozen for test evaluation.

| ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier score | Threshold |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7026 | 0.3819 | 0.3329 | 0.5907 | 0.4258 | 0.1458 | 0.205 |

At this operating point, the model retrieves 59.1% of high-cost cases, but only one-third of flagged cases are positive. A defensible use would therefore be **risk triage**: route the highest-risk cases to human review and use broader, lower-cost outreach for the remaining potential-risk population—not automatic eligibility, pricing, or clinical decisions.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── medical_insurance.csv
├── notebooks/
│   └── medical_insurance_leakage_aware.ipynb
└── scripts/
    └── build_revised_notebook.py
```

The notebook is committed with outputs so the reported evidence can be inspected without rerunning the full workflow. `scripts/build_revised_notebook.py` deterministically regenerates the notebook source.

## Reproduce the analysis

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/medical_insurance_leakage_aware.ipynb` and run all cells from the repository root. The notebook searches for the dataset at `data/medical_insurance.csv` and `../data/medical_insurance.csv`.

To regenerate the notebook structure before execution:

```bash
python scripts/build_revised_notebook.py
```

## Limitations

- The dataset appears to be constructed for modeling exercises; no clinical collection protocol or external population is documented on the data page.
- The data-card timeline is not independently verifiable from row-level timestamps.
- The study is a single-dataset holdout evaluation, not external validation.
- Feature importance is associational and should not be interpreted as a causal effect.
- The reported classifier threshold reflects one recall/precision trade-off and would require cost-sensitive validation before operational use.
- Nothing in this repository establishes clinical validity, fairness across protected groups, or deployment readiness.

## Version history

The original course-project state is retained in Git history and tagged as `v1-course-project`. The current `main` branch contains the leakage-aware revision and supersedes the earlier performance claims.
