from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from data import TARGET, add_derived_features, load_credit_data
from evaluate import evaluate_classifier, explain_model, save_risk_segments
from preprocessing import build_preprocessor, split_feature_types

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:
    SMOTE = None
    ImbPipeline = Pipeline


RANDOM_STATE = 42


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    image_dir = project_root / "images"
    model_dir = project_root / "models"
    report_dir = project_root / "reports"
    for directory in [image_dir, model_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    data, info = load_credit_data(project_root / "data")
    data = add_derived_features(data)
    x = data.drop(columns=[TARGET])
    y = data[TARGET].astype(int)

    run_eda(data, image_dir, report_dir)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(x_train)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    models = build_model_grids(preprocessor, y_train)

    leaderboard = []
    fitted_models = {}
    for name, pipeline, params in models:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x_train, y_train)
        fitted_models[name] = search.best_estimator_
        metrics = evaluate_classifier(search.best_estimator_, x_test, y_test, image_dir, name)
        metrics["cv_roc_auc"] = search.best_score_
        metrics["best_params"] = json.dumps(search.best_params_)
        leaderboard.append(metrics)

    leaderboard_df = pd.DataFrame(leaderboard).sort_values("roc_auc", ascending=False)
    leaderboard_df.to_csv(report_dir / "model_leaderboard.csv", index=False)

    best_name = leaderboard_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    probabilities = best_model.predict_proba(x_test)[:, 1]
    save_risk_segments(probabilities, image_dir)
    explain_model(best_model, x_test, y_test, image_dir, top_n=10)

    joblib.dump(best_model, model_dir / "best_model.joblib")
    write_project_metadata(project_root, info, x_train, best_name, leaderboard_df.iloc[0].to_dict())

    print(f"Best model: {best_name}")
    print(leaderboard_df[["model", "roc_auc", "average_precision", "f1", "business_cost"]].to_string(index=False))


def build_model_grids(preprocessor, y_train) -> list[tuple[str, Pipeline, dict]]:
    imbalance_ratio = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    logistic_steps = [("preprocess", preprocessor)]
    if SMOTE is not None:
        logistic_steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    logistic_steps.extend(
        [
            ("pca", PCA(random_state=RANDOM_STATE)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=None if SMOTE is not None else "balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=250,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    try:
        from xgboost import XGBClassifier

        boosted_model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=RANDOM_STATE,
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=imbalance_ratio,
        )
        boosted_name = "xgboost"
        boosted_params = {"model__max_depth": [3, 4], "model__learning_rate": [0.03, 0.07]}
    except Exception:
        boosted_model = HistGradientBoostingClassifier(random_state=RANDOM_STATE, learning_rate=0.05)
        boosted_name = "hist_gradient_boosting"
        boosted_params = {"model__max_leaf_nodes": [15, 31], "model__learning_rate": [0.03, 0.07]}

    boosted = Pipeline([("preprocess", preprocessor), ("model", boosted_model)])

    return [
        ("logistic_regression", ImbPipeline(logistic_steps), {"pca__n_components": [0.9, 0.95], "model__C": [0.3, 1.0]}),
        ("random_forest", random_forest, {"model__max_depth": [5, 10, None], "model__min_samples_leaf": [2, 5]}),
        (boosted_name, boosted, boosted_params),
    ]


def run_eda(data: pd.DataFrame, image_dir: Path, report_dir: Path) -> None:
    report = []
    default_rate = data[TARGET].mean()
    report.append(f"Rows: {len(data)}")
    report.append(f"Columns: {data.shape[1]}")
    report.append(f"Default rate: {default_rate:.3f}")

    plt.figure(figsize=(5.5, 4))
    sns.countplot(data=data, x=TARGET, hue=TARGET, palette=["#4d908e", "#c44536"], legend=False)
    plt.xticks([0, 1], ["Repaid", "Default"])
    plt.title("Class Balance")
    plt.tight_layout()
    plt.savefig(image_dir / "class_balance.png", dpi=160)
    plt.close()

    numeric_features, categorical_features = split_feature_types(data.drop(columns=[TARGET]))
    if numeric_features:
        correlations = data[numeric_features + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
        correlations.to_csv(report_dir / "feature_correlations.csv", header=["correlation_with_default"])
        plt.figure(figsize=(8, max(4, len(correlations.head(12)) * 0.35)))
        correlations.head(12).sort_values().plot(kind="barh", color="#326c88")
        plt.title("Numeric Feature Correlation with Default")
        plt.tight_layout()
        plt.savefig(image_dir / "correlations.png", dpi=160)
        plt.close()

        for feature in numeric_features[:6]:
            plt.figure(figsize=(7, 4))
            sns.kdeplot(data=data, x=feature, hue=TARGET, common_norm=False, fill=True, alpha=0.35)
            plt.title(f"{feature} Distribution by Default Status")
            plt.tight_layout()
            plt.savefig(image_dir / f"distribution_{feature}.png", dpi=160)
            plt.close()

    hypothesis_tests = run_hypothesis_tests(data, numeric_features, categorical_features)
    hypothesis_tests.to_csv(report_dir / "hypothesis_tests.csv", index=False)
    report_dir.joinpath("eda_summary.txt").write_text("\n".join(report), encoding="utf-8")


def run_hypothesis_tests(data: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
    rows = []
    try:
        from scipy.stats import chi2_contingency, ttest_ind

        for feature in numeric_features[:12]:
            good = data.loc[data[TARGET] == 0, feature].dropna()
            bad = data.loc[data[TARGET] == 1, feature].dropna()
            if len(good) > 2 and len(bad) > 2:
                stat, p_value = ttest_ind(good, bad, equal_var=False)
                rows.append({"feature": feature, "test": "Welch t-test", "statistic": stat, "p_value": p_value})

        for feature in categorical_features[:12]:
            table = pd.crosstab(data[feature], data[TARGET])
            if table.shape[0] > 1 and table.shape[1] > 1:
                stat, p_value, _, _ = chi2_contingency(table)
                rows.append({"feature": feature, "test": "Chi-square", "statistic": stat, "p_value": p_value})
    except Exception as exc:
        rows.append({"feature": "hypothesis_tests", "test": "skipped", "statistic": 0, "p_value": 1, "note": str(exc)})

    return pd.DataFrame(rows)


def write_project_metadata(project_root: Path, info, x_train: pd.DataFrame, best_name: str, best_metrics: dict) -> None:
    categorical_values = {
        column: sorted(x_train[column].dropna().astype(str).unique().tolist())
        for column in x_train.select_dtypes(exclude=["number", "bool"]).columns
    }
    numeric_defaults = {
        column: float(x_train[column].median())
        for column in x_train.select_dtypes(include=["number", "bool"]).columns
    }
    metadata = {
        "dataset": info.__dict__,
        "features": x_train.columns.tolist(),
        "categorical_values": categorical_values,
        "numeric_defaults": numeric_defaults,
        "best_model": best_name,
        "best_metrics": {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in best_metrics.items()
            if key != "best_params"
        },
    }
    (project_root / "reports" / "project_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    resume_line = (
        "Built an end-to-end loan default prediction pipeline on "
        f"{info.name}, comparing Logistic Regression, Random Forest, and boosted trees with "
        f"probability-focused evaluation and SHAP/permutation interpretability, achieving "
        f"{best_metrics['roc_auc']:.3f} ROC-AUC while addressing class imbalance through "
        "SMOTE/class weighting and cost-sensitive analysis."
    )
    (project_root / "reports" / "resume_line.txt").write_text(resume_line, encoding="utf-8")


if __name__ == "__main__":
    main()
