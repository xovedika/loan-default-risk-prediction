# Loan Default Risk Prediction

A machine learning project that predicts whether a loan applicant is likely to default using the German Credit dataset. The idea was not just to build the most accurate model, but to explore how different models behave on a real credit-risk problem and understand the trade-offs between performance, interpretability, and business impact.

## Project Overview

Credit risk is a classic machine learning problem where the cost of mistakes is not equal. Missing a risky applicant can be much more expensive than incorrectly flagging a safe one.

In this project, I built an end-to-end pipeline that:

* Analyzes applicant and loan characteristics
* Predicts the probability of default
* Compares multiple machine learning models
* Evaluates models using both ML metrics and business-oriented costs
* Explains model decisions using feature importance techniques
* Provides an interactive Streamlit application for testing predictions

---

## Dataset

I used the German Credit dataset, which contains information about 1,000 loan applicants and whether they were considered good or bad credit risks.

The dataset is relatively small and contains a mix of numerical and categorical features, making it useful for practicing:

* Data preprocessing
* Feature engineering
* Handling class imbalance
* Model selection and evaluation
* Explainable AI techniques

One challenge was that only about 30% of the applicants belonged to the default class, which creates an imbalanced classification problem.

---

## Exploratory Data Analysis

Before training models, I explored the data to understand which factors were related to default risk.

Some of the analyses included:

* Class distribution analysis
* Numerical feature distributions
* Correlation analysis
* Welch's t-tests for numerical features
* Chi-square tests for categorical features

Features such as checking account status, credit history, loan duration, and credit amount showed stronger relationships with default outcomes than others.

---

## Data Preprocessing

The preprocessing pipeline includes:

* Missing value handling
* One-hot encoding of categorical variables
* Feature scaling for numerical columns
* Train-test splitting with stratification

I also experimented with a few engineered features, including:

* Debt-to-income ratio
* Credit utilization

For Logistic Regression, PCA was applied after preprocessing to reduce dimensionality and remove some redundancy from the encoded feature space.

---

## Models Evaluated

I compared three different approaches:

### Logistic Regression + PCA

A simple and interpretable baseline model.

### Random Forest

An ensemble tree-based model capable of capturing nonlinear relationships.

### XGBoost

A gradient boosting approach designed to improve predictive performance on structured tabular data.

If XGBoost is not available in the environment, the project automatically falls back to HistGradientBoosting from scikit-learn.

---

## Handling Class Imbalance

Because the default class is underrepresented, I experimented with techniques to improve minority-class detection.

Primary approach:

* SMOTE (Synthetic Minority Oversampling Technique)

Fallback approach:

* Class-weighted training

This helped improve recall for default cases without relying entirely on the majority class.

---

## Model Tuning

All models were tuned using:

* GridSearchCV
* Stratified cross-validation

This ensured that class proportions remained consistent across folds and reduced the risk of selecting a model based on a favorable train-test split.

---

## Evaluation Metrics

Accuracy alone can be misleading for imbalanced datasets, so I evaluated models using:

* ROC-AUC
* Average Precision
* F1 Score
* Calibration performance

I also introduced a simple business-cost metric where:

* False negatives (missed defaulters) carry a higher penalty
* False positives carry a lower penalty

This provides a more realistic view of how the models would perform in a lending scenario.

---

## Results

| Model                             | ROC-AUC | Average Precision | F1 Score | Business Cost |
| --------------------------------- | ------- | ----------------- | -------- | ------------- |
| Logistic Regression (PCA + SMOTE) | 0.800   | 0.637             | 0.648    | 103           |
| XGBoost                           | 0.793   | 0.631             | 0.604    | 127           |
| Random Forest                     | 0.789   | 0.629             | 0.586    | 134           |

Interestingly, Logistic Regression performed best overall.

I initially expected the more complex tree-based models to outperform it, but the simpler model produced the lowest business cost and better-calibrated probabilities. Given the dataset size, this suggests that the additional flexibility of boosting and random forests may not provide much advantage.

---

## Model Explainability

To understand why predictions were being made, I used:

* SHAP (when available)
* Permutation importance as a fallback

The most influential features across models were:

* Checking account status
* Credit history
* Loan duration
* Loan purpose
* Credit amount

These factors consistently appeared near the top of the importance rankings and align with common lending considerations.

---

## Risk Categories

Predicted probabilities are converted into three risk groups:

| Risk Tier   | Probability |
| ----------- | ----------- |
| Low Risk    | < 25%       |
| Medium Risk | 25% – 55%   |
| High Risk   | > 55%       |

This makes the output easier to interpret than a raw probability alone.

---

## Streamlit Application

The project includes a Streamlit dashboard where users can:

* Enter applicant information
* Receive a default-risk prediction
* View the predicted probability
* See the assigned risk tier
* Explore feature importance information

This provides a simple way to interact with the trained model without running notebooks or scripts manually.

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the models:

```bash
python src/train.py
```

Launch the application:

```bash
streamlit run app.py
```

---

## Generated Outputs

Training produces several artifacts:

```text
reports/
├── model_leaderboard.csv
├── hypothesis_tests.csv
└── feature_correlations.csv

models/
└── best_model.joblib

images/
├── ROC curves
├── Precision-Recall curves
├── Calibration plots
├── Confusion matrices
└── Feature importance visualizations
```

Note: Trained models are excluded from version control, so the training pipeline must be run before using the Streamlit application.

---

## Project Structure

```text
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
├── images/
├── data/
├── models/
└── reports/
```

---

## Future Improvements

A few things I would improve if I continued working on the project:

* Add per-applicant SHAP explanations directly inside the Streamlit app
* Unify the schema between the fallback synthetic dataset and the German Credit dataset
* Experiment with probability threshold optimization instead of using a fixed threshold
* Try more advanced calibration techniques for probability estimates
* Evaluate fairness metrics across demographic groups

---

## Key Takeaways

This project was a good exercise in going beyond simply training a classifier. It involved data analysis, feature engineering, class imbalance handling, model comparison, explainability, and deployment in a single workflow.

One of the biggest lessons was that a simpler model can sometimes outperform more complex alternatives, especially when data is limited and probability calibration matters as much as raw predictive power.
