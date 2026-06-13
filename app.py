from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "reports" / "project_metadata.json"
RISK_FACTORS_PATH = ROOT / "images" / "top_risk_factors.csv"
DATA_PATH = ROOT / "data" / "credit_data.csv"

FEATURE_LABELS = {
    "checking_status": "Checking account status",
    "duration": "Loan duration (months)",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "credit_amount": "Credit amount",
    "savings_status": "Savings status",
    "employment": "Employment length",
    "installment_commitment": "Installment rate",
    "personal_status": "Personal status",
    "other_parties": "Co-applicant / guarantor",
    "residence_since": "Years at residence",
    "property_magnitude": "Property owned",
    "age": "Age",
    "other_payment_plans": "Other payment plans",
    "housing": "Housing",
    "existing_credits": "Existing credits",
    "job": "Job type",
    "num_dependents": "Dependents",
    "own_telephone": "Telephone",
    "foreign_worker": "Foreign worker",
}

FEATURE_GROUPS = {
    "Credit profile": [
        "checking_status",
        "credit_history",
        "savings_status",
        "existing_credits",
        "other_payment_plans",
    ],
    "Loan terms": [
        "credit_amount",
        "duration",
        "purpose",
        "installment_commitment",
        "other_parties",
    ],
    "Applicant background": [
        "employment",
        "job",
        "age",
        "housing",
        "property_magnitude",
        "residence_since",
        "personal_status",
        "num_dependents",
        "own_telephone",
        "foreign_worker",
    ],
}


st.set_page_config(page_title="Loan Default Risk", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --risk-green: #1b7f5a;
        --risk-amber: #b26a00;
        --risk-red: #b42318;
        --ink: #16202a;
        --muted: #5f6f7d;
        --line: #d9e2e8;
        --panel: #f7fafb;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1180px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .eyebrow {
        color: #315266;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    .hero-copy {
        color: var(--muted);
        font-size: 1rem;
        margin-bottom: 1.1rem;
        max-width: 760px;
    }
    .metric-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.7rem;
        margin: 1rem 0 1.2rem;
    }
    .metric-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        background: white;
    }
    .metric-label {
        color: var(--muted);
        font-size: 0.78rem;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        color: var(--ink);
        font-size: 1.28rem;
        font-weight: 750;
    }
    .risk-panel {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        background: var(--panel);
    }
    .risk-score {
        font-size: 3rem;
        line-height: 1;
        font-weight: 800;
        color: var(--ink);
    }
    .risk-tier {
        display: inline-block;
        border-radius: 999px;
        padding: 0.25rem 0.65rem;
        color: white;
        font-weight: 700;
        font-size: 0.82rem;
        margin-top: 0.45rem;
    }
    .risk-low { background: var(--risk-green); }
    .risk-medium { background: var(--risk-amber); }
    .risk-high { background: var(--risk-red); }
    .probability-track {
        height: 0.7rem;
        width: 100%;
        border-radius: 999px;
        background: #dce8ee;
        overflow: hidden;
        margin-top: 0.9rem;
    }
    .probability-fill {
        height: 100%;
        border-radius: 999px;
    }
    .fill-low { background: var(--risk-green); }
    .fill-medium { background: var(--risk-amber); }
    .fill-high { background: var(--risk-red); }
    .threshold-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.75rem;
    }
    .chip {
        border-radius: 999px;
        padding: 0.22rem 0.55rem;
        color: white;
        font-size: 0.76rem;
        font-weight: 700;
    }
    .decision-row {
        display: flex;
        gap: 0.55rem;
        align-items: center;
        justify-content: space-between;
        border-top: 1px solid var(--line);
        padding-top: 0.8rem;
        margin-top: 0.8rem;
        color: var(--muted);
        font-size: 0.9rem;
    }
    .factor-row {
        display: grid;
        grid-template-columns: minmax(120px, 1fr) 90px;
        gap: 0.6rem;
        align-items: center;
        font-size: 0.88rem;
        margin-bottom: 0.52rem;
    }
    .bar {
        height: 0.48rem;
        border-radius: 999px;
        background: #dce8ee;
        overflow: hidden;
    }
    .bar span {
        display: block;
        height: 100%;
        background: #326c88;
        border-radius: 999px;
    }
    .factor-value {
        color: var(--muted);
        font-size: 0.78rem;
        text-align: right;
    }
    .waterfall-row {
        display: grid;
        grid-template-columns: minmax(130px, 1fr) 110px;
        gap: 0.65rem;
        align-items: center;
        margin-bottom: 0.58rem;
        font-size: 0.86rem;
    }
    .waterfall-track {
        height: 0.55rem;
        border-radius: 999px;
        background: #e2ebf0;
        overflow: hidden;
    }
    .waterfall-fill {
        display: block;
        height: 100%;
        border-radius: 999px;
    }
    .push-risk { background: var(--risk-red); }
    .lower-risk { background: var(--risk-green); }
    .delta-value {
        color: var(--muted);
        font-size: 0.78rem;
        text-align: right;
        margin-top: 0.08rem;
    }
    .waterfall-summary {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        gap: 0.5rem;
        align-items: center;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.65rem;
        margin-bottom: 0.85rem;
        background: white;
    }
    .summary-number {
        color: var(--ink);
        font-weight: 800;
        font-size: 1.15rem;
    }
    .summary-label {
        color: var(--muted);
        font-size: 0.72rem;
    }
    .summary-arrow {
        color: var(--muted);
        font-weight: 800;
    }
    div[data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: white;
    }
    div[data-testid="stForm"] {
        border: none;
    }
    div[data-testid="stFormSubmitButton"] button {
        background: #326c88;
        border-color: #326c88;
        color: white;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background: #28586f;
        border-color: #28586f;
        color: white;
    }
    @media (max-width: 820px) {
        .metric-strip {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .risk-score {
            font-size: 2.4rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Explainable credit risk model</div>', unsafe_allow_html=True)
st.title("Loan Default Risk Prediction")
st.markdown(
    '<p class="hero-copy">Estimate applicant default probability, review the assigned risk tier, and connect the prediction back to model evidence from the German Credit dataset.</p>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.warning("Train the model first with `python src/train.py`.")
    st.stop()

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
features = metadata["features"]
categorical_values = metadata.get("categorical_values", {})
numeric_defaults = metadata.get("numeric_defaults", {})
metrics = metadata.get("best_metrics", {})

st.markdown(
    f"""
    <div class="metric-strip">
      <div class="metric-card"><div class="metric-label">Dataset rows</div><div class="metric-value">{metadata['dataset']['rows']:,}</div></div>
      <div class="metric-card"><div class="metric-label">Observed default rate</div><div class="metric-value">{metadata['dataset']['default_rate']:.1%}</div></div>
      <div class="metric-card"><div class="metric-label">Best ROC-AUC</div><div class="metric-value">{metrics.get('roc_auc', 0):.3f}</div></div>
      <div class="metric-card"><div class="metric-label">Best model</div><div class="metric-value">{metadata.get('best_model', 'model').replace('_', ' ').title()}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)


def label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def collect_feature(feature: str):
    if feature in categorical_values:
        if len(categorical_values[feature]) <= 3:
            return st.radio(label(feature), categorical_values[feature], key=feature, horizontal=True)
        return st.selectbox(label(feature), categorical_values[feature], key=feature)

    default = float(numeric_defaults.get(feature, 0.0))
    step = 1.0
    if feature in {"credit_amount"}:
        step = 100.0
    return st.number_input(label(feature), value=default, step=step, key=feature)


def score_sample(values: dict) -> tuple[float, str, str]:
    sample = pd.DataFrame([values])
    probability = float(model.predict_proba(sample)[0, 1])
    if probability < 0.25:
        return probability, "Low", "risk-low fill-low"
    if probability < 0.55:
        return probability, "Medium", "risk-medium fill-medium"
    return probability, "High", "risk-high fill-high"


def load_top_factors() -> pd.DataFrame:
    if RISK_FACTORS_PATH.exists():
        return pd.read_csv(RISK_FACTORS_PATH).head(5)
    return pd.DataFrame({"feature": [], "importance": []})


@st.cache_data
def load_baseline_profile(features_to_use: tuple[str, ...]) -> dict:
    if not DATA_PATH.exists():
        baseline = {}
        for feature in features_to_use:
            if feature in categorical_values:
                baseline[feature] = categorical_values[feature][0]
            else:
                baseline[feature] = float(numeric_defaults.get(feature, 0.0))
        return baseline

    data = pd.read_csv(DATA_PATH)
    baseline = {}
    for feature in features_to_use:
        if feature not in data.columns:
            baseline[feature] = categorical_values.get(feature, [numeric_defaults.get(feature, 0.0)])[0]
        elif feature in categorical_values:
            baseline[feature] = str(data[feature].mode(dropna=True).iloc[0])
        else:
            baseline[feature] = float(data[feature].median())
    return baseline


def predict_probability(values: dict) -> float:
    return float(model.predict_proba(pd.DataFrame([values]))[0, 1])


def local_waterfall(values: dict, top_n: int = 7) -> tuple[float, float, pd.DataFrame]:
    baseline = load_baseline_profile(tuple(features))
    base_probability = predict_probability(baseline)

    independent_effects = []
    for feature in features:
        counterfactual = baseline.copy()
        counterfactual[feature] = values[feature]
        effect = predict_probability(counterfactual) - base_probability
        independent_effects.append((feature, abs(effect)))

    ordered_features = [feature for feature, _ in sorted(independent_effects, key=lambda item: item[1], reverse=True)]
    current = baseline.copy()
    previous_probability = base_probability
    rows = []
    for feature in ordered_features:
        current[feature] = values[feature]
        next_probability = predict_probability(current)
        delta = next_probability - previous_probability
        if abs(delta) >= 0.002:
            rows.append(
                {
                    "feature": feature,
                    "label": label(feature),
                    "value": values[feature],
                    "delta": delta,
                    "probability": next_probability,
                }
            )
        previous_probability = next_probability

    final_probability = predict_probability(values)
    explanation = pd.DataFrame(rows)
    if not explanation.empty:
        explanation = explanation.sort_values("delta", key=lambda series: series.abs(), ascending=False).head(top_n)
    return base_probability, final_probability, explanation


left, right = st.columns([1.45, 1], gap="large")

with left:
    st.subheader("Applicant profile")
    with st.form("risk_form"):
        inputs = {}
        for group, group_features in FEATURE_GROUPS.items():
            with st.expander(group, expanded=True):
                grid = st.columns(3)
                for index, feature in enumerate(group_features):
                    if feature not in features:
                        continue
                    with grid[index % 3]:
                        inputs[feature] = collect_feature(feature)

        missing = [feature for feature in features if feature not in inputs]
        for feature in missing:
            inputs[feature] = collect_feature(feature)

        submitted = st.form_submit_button("Predict default risk", type="primary", use_container_width=True)

if "last_inputs" not in st.session_state or submitted:
    st.session_state.last_inputs = inputs

probability, tier, tier_class = score_sample(st.session_state.last_inputs)

with right:
    st.subheader("Risk assessment")
    tier_badge_class, tier_fill_class = tier_class.split(" ")
    probability_width = int(round(probability * 100))
    st.markdown(
        f"""
        <div class="risk-panel">
          <div class="metric-label">Predicted default probability</div>
          <div class="risk-score">{probability:.0%}</div>
          <div class="risk-tier {tier_badge_class}">{tier} risk</div>
          <div class="probability-track"><div class="probability-fill {tier_fill_class}" style="width: {probability_width}%"></div></div>
          <div class="threshold-chips">
            <span class="chip risk-low">Low &lt;25%</span>
            <span class="chip risk-medium">Medium 25-55%</span>
            <span class="chip risk-high">High &gt;55%</span>
          </div>
          <div class="decision-row">
            <span>Model output</span>
            <strong>Probability rounded for display</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Model evidence")
    st.caption("Applicant-specific contribution path from a typical profile to this prediction.")
    base_probability, final_probability, explanation = local_waterfall(st.session_state.last_inputs)
    st.markdown(
        f"""
        <div class="waterfall-summary">
          <div><div class="summary-label">Typical profile</div><div class="summary-number">{base_probability:.0%}</div></div>
          <div class="summary-arrow">→</div>
          <div><div class="summary-label">Selected applicant</div><div class="summary-number">{final_probability:.0%}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if explanation.empty:
        st.caption("This applicant is close to the typical baseline profile.")
    else:
        max_delta = explanation["delta"].abs().max()
        for row in explanation.itertuples(index=False):
            width = 0 if max_delta == 0 else int((abs(row.delta) / max_delta) * 100)
            direction_class = "push-risk" if row.delta > 0 else "lower-risk"
            sign = "+" if row.delta > 0 else ""
            direction = "pushed risk up" if row.delta > 0 else "lowered risk"
            st.markdown(
                f"""
                <div class="waterfall-row">
                  <div>{row.label}<br><span class="metric-label">{direction}</span></div>
                  <div>
                    <div class="waterfall-track"><span class="waterfall-fill {direction_class}" style="width: {width}%"></span></div>
                    <div class="delta-value">{sign}{row.delta:.1%}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with st.expander("Global model importance", expanded=False):
        st.caption("Training-set importance, useful for portfolio-level interpretation.")
        factors = load_top_factors()
        if factors.empty:
            st.caption("Run training to generate feature importance.")
        else:
            max_importance = factors["importance"].max()
            for row in factors.itertuples(index=False):
                width = 0 if max_importance == 0 else int((row.importance / max_importance) * 100)
                st.markdown(
                    f"""
                    <div class="factor-row">
                      <div>{label(row.feature)}</div>
                      <div>
                        <div class="bar"><span style="width: {width}%"></span></div>
                        <div class="factor-value">{row.importance:.3f}</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with st.expander("Selected applicant snapshot", expanded=False):
        st.dataframe(pd.DataFrame([st.session_state.last_inputs]).T.rename(columns={0: "Value"}), use_container_width=True)

st.caption(
    "This demo is for ML portfolio use, not real lending decisions. The model is trained on the German Credit dataset and should be validated before any production use."
)
