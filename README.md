# Loan Default Risk Prediction

A machine learning project that predicts whether a loan applicant is likely to default, using the German Credit dataset. The goal wasn't just to get a good accuracy number — I wanted to build something that mimics how a credit team might actually use a model: getting a calibrated probability, understanding why the model flagged someone as risky, and seeing how different models trade off against each other.

## Why this dataset

The German Credit dataset is small (1,000 rows) and a bit messy, which made it a good fit for practicing the full pipeline — handling categorical features, dealing with class imbalance (only ~30% of applicants defaulted), and figuring out which model actually generalizes vs. just looks good on paper.

## What's in the pipeline

I started with exploratory analysis — checking the class imbalance, looking at how numeric features (credit amount, duration, age, etc.) differ between defaulters and non-defaulters, and running some basic statistical tests (Welch's t-test, chi-square) to see which features actually had a meaningful relationship with default.

For preprocessing, I handled missing values, one-hot encoded categorical fields, scaled numeric features, and added a couple of derived features like debt-to-income and credit utilization. For the Logistic Regression model, I also applied PCA to reduce dimensionality.

On the modeling side, I compared three models: Logistic Regression (with PCA), Random Forest, and XGBoost (falling back to scikit-learn's HistGradientBoosting if XGBoost isn't installed). To deal with the class imbalance, I used SMOTE where available, with class weighting as a fallback. All models were tuned with GridSearchCV using stratified cross-validation.

For evaluation, accuracy alone doesn't tell you much on an imbalanced dataset, so I focused on ROC-AUC, average precision, F1, and calibration. I also added a simple business-cost framing — false negatives (missed defaulters) are weighted more heavily than false positives, since that's closer to how a real lender would think about risk.

## Results

| Model | ROC-AUC | Avg. Precision | F1 | Business Cost |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression (PCA + SMOTE) | 0.800 | 0.637 | 0.648 | 103 |
| XGBoost | 0.793 | 0.631 | 0.604 | 127 |
| Random Forest | 0.789 | 0.629 | 0.586 | 134 |

Logistic Regression came out on top — slightly surprising given it's the simplest model here, but it had the best calibration and the lowest business cost. With only 1,000 rows, the more flexible tree-based models likely had less room to show their advantage and were more prone to overfitting on the cross-validation folds.

## Interpretability

Beyond just predicting a probability, I wanted the model to explain why. I used SHAP where it worked in the environment, with permutation importance as a fallback. Across models, the strongest signals were checking account status, credit history, loan duration, loan purpose, and credit amount — which lines up with intuition about what lenders actually care about.

Probabilities are converted into three risk tiers: Low (<25%), Medium (25-55%), and High (>55%).

## The app

There's a Streamlit app (`app.py`) that lets you enter an applicant's profile and get a live prediction, along with the risk tier and a breakdown of which features influenced the result most.

## Running it locally

```bash
pip install -r requirements.txt
python src/train.py        # trains models, generates reports and plots
streamlit run app.py        # launches the demo
```

Training generates:

- `reports/model_leaderboard.csv` — model comparison
- `reports/hypothesis_tests.csv` — statistical test results
- `reports/feature_correlations.csv`
- `models/best_model.joblib` — the saved best model
- Various plots in `images/` (ROC, PR curve, calibration, confusion matrix, feature importance, etc.)

Note: the trained model isn't included in this repo (it's gitignored due to size), so you'll need to run `train.py` once before launching the app.

## Notes / what I'd do differently

The synthetic fallback dataset (used if OpenML is unavailable) has a different feature schema than the real German Credit data, which I'd unify if I revisited this. I'd also like to add per-prediction SHAP explanations in the app instead of just showing global feature importance — that would make the "explainability" angle much stronger.

## Repository structure

```
loan-default-risk-prediction/
├── app.py
├── requirements.txt
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── analysis.ipynb
├── images/        # generated plots and reports
├── data/          # dataset (gitignored)
├── models/        # trained models (gitignored)
└── reports/       # generated CSV reports (gitignored)
```