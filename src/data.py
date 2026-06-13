from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


TARGET = "default"


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    source: str
    rows: int
    columns: int
    default_rate: float


def load_credit_data(cache_dir: str | Path = "data") -> tuple[pd.DataFrame, DatasetInfo]:
    """Load German Credit from OpenML, falling back to a reproducible synthetic credit dataset."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "credit_data.csv"

    if cached.exists():
        data = pd.read_csv(cached)
        return data, _info(data, "Cached credit risk dataset", str(cached))

    try:
        raw = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
        data = raw.frame.copy()
        data[TARGET] = (data["class"].astype(str).str.lower() == "bad").astype(int)
        data = data.drop(columns=["class"])
        source = "OpenML credit-g / German Credit Data"
    except Exception:
        data = make_synthetic_credit_data()
        source = "Synthetic fallback generated from credit-risk assumptions"

    data.to_csv(cached, index=False)
    return data, _info(data, "German Credit Risk" if "checking_status" in data.columns else "Synthetic Credit Risk", source)


def make_synthetic_credit_data(n_rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    annual_income = rng.lognormal(mean=10.7, sigma=0.55, size=n_rows).round(2)
    loan_amount = rng.lognormal(mean=8.7, sigma=0.75, size=n_rows).round(2)
    credit_history_months = rng.integers(6, 241, size=n_rows)
    open_credit_lines = rng.poisson(5, size=n_rows) + 1
    delinquencies_2yrs = rng.poisson(0.35, size=n_rows)
    revolving_balance = rng.lognormal(mean=7.4, sigma=0.9, size=n_rows).round(2)
    credit_limit = revolving_balance + rng.lognormal(mean=8.2, sigma=0.7, size=n_rows)
    employment_length = rng.choice(["<1 year", "1-3 years", "3-7 years", "7+ years"], size=n_rows, p=[0.18, 0.28, 0.32, 0.22])
    home_ownership = rng.choice(["rent", "mortgage", "own"], size=n_rows, p=[0.43, 0.42, 0.15])
    purpose = rng.choice(["debt_consolidation", "credit_card", "home_improvement", "small_business", "education"], size=n_rows)

    dti = loan_amount / np.maximum(annual_income, 1)
    utilization = revolving_balance / np.maximum(credit_limit, 1)
    logit = (
        -2.6
        + 3.8 * dti
        + 1.7 * utilization
        + 0.32 * delinquencies_2yrs
        - 0.005 * credit_history_months
        + np.where(home_ownership == "rent", 0.25, 0)
        + np.where(purpose == "small_business", 0.35, 0)
        + rng.normal(0, 0.45, n_rows)
    )
    probability = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "annual_income": annual_income,
            "loan_amount": loan_amount,
            "credit_history_months": credit_history_months,
            "open_credit_lines": open_credit_lines,
            "delinquencies_2yrs": delinquencies_2yrs,
            "revolving_balance": revolving_balance,
            "credit_limit": credit_limit.round(2),
            "employment_length": employment_length,
            "home_ownership": home_ownership,
            "purpose": purpose,
            TARGET: default,
        }
    )


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"loan_amount", "annual_income"}.issubset(out.columns):
        out["debt_to_income"] = out["loan_amount"] / out["annual_income"].clip(lower=1)
    elif {"amount", "duration"}.issubset(out.columns):
        out["installment_proxy"] = out["amount"] / out["duration"].clip(lower=1)

    if {"revolving_balance", "credit_limit"}.issubset(out.columns):
        out["credit_utilization"] = out["revolving_balance"] / out["credit_limit"].clip(lower=1)

    if {"amount", "duration"}.issubset(out.columns):
        out["amount_per_month"] = out["amount"] / out["duration"].clip(lower=1)

    return out.replace([np.inf, -np.inf], np.nan)


def _info(data: pd.DataFrame, name: str, source: str) -> DatasetInfo:
    return DatasetInfo(
        name=name,
        source=source,
        rows=len(data),
        columns=data.shape[1],
        default_rate=float(data[TARGET].mean()),
    )
