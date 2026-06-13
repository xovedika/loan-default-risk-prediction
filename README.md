# Loan Default Risk Prediction with Explainable ML

An end-to-end machine learning project for predicting whether a loan applicant is likely to default. The project emphasizes statistics, class imbalance, model comparison, probability-focused evaluation, and interpretability.

## Problem Statement

Credit teams need more than a binary approve/reject model. They need calibrated default probabilities, a way to compare model tradeoffs, and explanations for the main drivers behind each risk score. This project builds a reproducible ML pipeline that predicts loan default risk and converts probabilities into Low, Medium, and High risk tiers.

## Dataset

The training script first attempts to load the public `credit-g` German Credit dataset from OpenML. If OpenML is unavailable, it falls back to a deterministic synthetic credit-risk dataset with realistic signals such as debt-to-income ratio, credit utilization, delinquency history, credit history length, home ownership, and loan purpose.

Primary dataset link: [German Credit Data on OpenML](https://www.openml.org/search?type=data&sort=runs&id=31)

## Approach

1. Exploratory data analysis
   - Class imbalance and default-rate analysis
   - Numeric feature distributions by default status
   - Correlation analysis against the target
   - Welch t-tests and chi-square tests for statistical signal checks

2. Feature engineering
   - Missing-value imputation
   - One-hot encoding for categorical variables
   - Standard scaling for numeric variables
   - Derived credit-risk features such as debt-to-income and credit-utilization proxies
   - PCA in the logistic-regression baseline

3. Modeling
   - Logistic Regression baseline
   - Random Forest
   - XGBoost when installed, otherwise scikit-learn HistGradientBoosting
   - SMOTE when installed, otherwise class weighting
   - GridSearchCV with stratified cross-validation

4. Evaluation
   - ROC-AUC
   - Average precision / PR curve
   - F1-score
   - Calibration curve
   - Confusion matrix
   - Business-cost framing where false negatives cost more than false positives

5. Interpretability
   - SHAP summary plot when SHAP works in the local environment
   - Permutation importance fallback
   - Top risk factor report

6. Risk segmentation
   - Low risk: probability < 25%
   - Medium risk: 25-55%
   - High risk: > 55%

## Repository Structure

```text
loan-default-risk-prediction/
├── README.md
├── app.py
├── data/
├── images/
├── notebooks/
│   └── analysis.ipynb
├── requirements.txt
└── src/
    ├── data.py
    ├── evaluate.py
    ├── preprocessing.py
    └── train.py
```

Generated outputs are written to `images/`, `models/`, and `reports/`.

## How to Run

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Train models and generate reports:

```bash
python src/train.py
```

Launch the optional Streamlit demo:

```bash
streamlit run app.py
```

## Expected Outputs

After training, the project generates:

- `reports/model_leaderboard.csv`
- `reports/hypothesis_tests.csv`
- `reports/feature_correlations.csv`
- `reports/resume_line.txt`
- `models/best_model.joblib`
- ROC, PR, calibration, confusion matrix, class balance, feature importance, and risk segmentation plots in `images/`

## Current Results

On the OpenML German Credit dataset, the best model from the latest run was Logistic Regression with PCA and SMOTE:

| Model | ROC-AUC | Average Precision | F1 | Business Cost |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.800 | 0.637 | 0.648 | 103 |
| XGBoost | 0.793 | 0.631 | 0.604 | 127 |
| Random Forest | 0.789 | 0.629 | 0.586 | 134 |

Top risk factors from the interpretability report included checking account status, credit history, loan duration, purpose, and credit amount.

## Resume Line

Use the generated `reports/resume_line.txt` after training. Template:

> Built an end-to-end loan default prediction pipeline on German Credit Data, comparing Logistic Regression, Random Forest, and boosted trees with probability-focused evaluation and SHAP/permutation interpretability, achieving X.XXX ROC-AUC while addressing class imbalance through SMOTE/class weighting and cost-sensitive analysis.

## Why This Project Fits Amazon ML Summer School

This project demonstrates the fundamentals tested in selection rounds: probability, statistics, linear algebra via PCA, supervised learning, model evaluation beyond accuracy, and applied problem solving. It also shows practical ML engineering through reproducible scripts, modular preprocessing, interpretable results, and a deployable demo.
