from __future__ import annotations

from pathlib import Path

import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def evaluate_classifier(model, x_test, y_test, output_dir: str | Path, model_name: str) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "model": model_name,
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(y_test, probabilities),
        "f1": f1_score(y_test, predictions),
    }

    report_path = output_dir / f"{model_name}_classification_report.txt"
    report_path.write_text(classification_report(y_test, predictions), encoding="utf-8")

    plot_roc(y_test, probabilities, output_dir / f"{model_name}_roc.png", model_name)
    plot_precision_recall(y_test, probabilities, output_dir / f"{model_name}_pr.png", model_name)
    plot_calibration(y_test, probabilities, output_dir / f"{model_name}_calibration.png", model_name)
    plot_confusion(y_test, predictions, output_dir / f"{model_name}_confusion.png", model_name)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    metrics["business_cost"] = int((5 * fn) + fp)
    metrics["false_negatives"] = int(fn)
    metrics["false_positives"] = int(fp)
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    return metrics


def plot_roc(y_true, probabilities, path: Path, model_name: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc = roc_auc_score(y_true, probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{model_name} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_precision_recall(y_true, probabilities, path: Path, model_name: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    ap = average_precision_score(y_true, probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"{model_name} AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_calibration(y_true, probabilities, path: Path, model_name: str) -> None:
    prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=10, strategy="uniform")
    plt.figure(figsize=(7, 5))
    plt.plot(prob_pred, prob_true, marker="o", label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed default rate")
    plt.title("Probability Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_confusion(y_true, predictions, path: Path, model_name: str) -> None:
    plt.figure(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, predictions, display_labels=["Repaid", "Default"])
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_risk_segments(probabilities: np.ndarray, output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    tiers = pd.cut(
        probabilities,
        bins=[0, 0.25, 0.55, 1],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    distribution = tiers.value_counts().rename_axis("risk_tier").reset_index(name="applications")
    distribution.to_csv(output_dir / "risk_segments.csv", index=False)

    plt.figure(figsize=(7, 4))
    sns.barplot(data=distribution, x="risk_tier", y="applications", hue="risk_tier", palette="viridis", legend=False)
    plt.title("Portfolio Risk Distribution")
    plt.xlabel("Risk tier")
    plt.ylabel("Applications")
    plt.tight_layout()
    plt.savefig(output_dir / "risk_segments.png", dpi=160)
    plt.close()
    return distribution


def explain_model(best_model, x_test: pd.DataFrame, y_test, output_dir: str | Path, top_n: int = 10) -> pd.DataFrame:
    output_dir = Path(output_dir)
    try:
        import shap

        sample = x_test.sample(min(200, len(x_test)), random_state=42)
        explainer = shap.Explainer(best_model.predict_proba, sample)
        values = explainer(sample)
        importance = np.abs(values.values[:, :, 1]).mean(axis=0)
        summary = pd.DataFrame({"feature": sample.columns, "importance": importance}).sort_values("importance", ascending=False)
        shap.plots.beeswarm(values[:, :, 1], show=False, max_display=top_n)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
        plt.close()
    except Exception:
        result = permutation_importance(
            best_model,
            x_test,
            y_test,
            n_repeats=5,
            random_state=42,
            scoring="roc_auc",
        )
        summary = pd.DataFrame({"feature": x_test.columns, "importance": result.importances_mean}).sort_values(
            "importance", ascending=False
        )

    summary.head(top_n).to_csv(output_dir / "top_risk_factors.csv", index=False)
    plt.figure(figsize=(8, 5))
    top = summary.head(top_n).sort_values("importance")
    plt.barh(top["feature"], top["importance"], color="#326c88")
    plt.xlabel("Importance")
    plt.title("Top Risk Factors")
    plt.tight_layout()
    plt.savefig(output_dir / "top_risk_factors.png", dpi=160)
    plt.close()
    return summary
