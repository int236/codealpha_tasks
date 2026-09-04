"""
generate_data.py
-----------------
Generates a realistic synthetic credit-scoring dataset.

This exists so the pipeline is runnable out-of-the-box for grading/demo
purposes. For your actual submission, swap this out for a real dataset
(e.g. the UCI "German Credit Data" or the Kaggle "Give Me Some Credit"
dataset) by placing a CSV at data/credit_data.csv with the same column
names, or point --data-path at your file.
"""

import numpy as np
import pandas as pd


def generate_credit_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Create a synthetic but statistically realistic credit dataset.

    The target `creditworthy` is generated from a weighted, noisy function
    of the features so that the resulting classification problem is
    non-trivial (not perfectly separable), like real credit data.
    """
    rng = np.random.default_rng(random_state)

    age = rng.integers(21, 70, n_samples)
    income = rng.gamma(shape=5.0, scale=9000, size=n_samples) + 15000
    employment_length = np.clip(rng.normal(7, 5, n_samples), 0, 40)
    num_credit_lines = rng.integers(0, 15, n_samples)
    num_late_payments_2yr = rng.poisson(0.8, n_samples)
    existing_debt = np.clip(rng.normal(income * 0.35, income * 0.15), 0, None)
    loan_amount = np.clip(rng.normal(15000, 8000, n_samples), 500, None)
    credit_utilization = np.clip(rng.beta(2, 5, n_samples), 0, 1)  # fraction of credit limit used
    has_bankruptcy = rng.binomial(1, 0.05, n_samples)
    savings = np.clip(rng.normal(income * 0.1, income * 0.08), 0, None)

    debt_to_income = existing_debt / (income + 1)

    # Latent "creditworthiness score" — weighted combination + noise.
    score = (
        0.9 * (income / 50000)
        - 2.5 * debt_to_income
        - 0.6 * num_late_payments_2yr
        - 1.8 * credit_utilization
        - 2.2 * has_bankruptcy
        + 0.05 * employment_length
        + 0.4 * (savings / 10000)
        - 0.15 * (loan_amount / 10000)
        + rng.normal(0, 0.6, n_samples)  # noise
    )

    creditworthy = (score > np.median(score)).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": income.round(2),
        "employment_length_years": employment_length.round(1),
        "num_credit_lines": num_credit_lines,
        "num_late_payments_2yr": num_late_payments_2yr,
        "existing_debt": existing_debt.round(2),
        "loan_amount_requested": loan_amount.round(2),
        "credit_utilization": credit_utilization.round(3),
        "has_bankruptcy": has_bankruptcy,
        "savings": savings.round(2),
        "creditworthy": creditworthy,  # target: 1 = good credit risk, 0 = bad
    })
    return df


if __name__ == "__main__":
    df = generate_credit_dataset()
    df.to_csv("data/credit_data.csv", index=False)
    print(f"Generated {len(df)} rows -> data/credit_data.csv")
    print(df["creditworthy"].value_counts(normalize=True))
