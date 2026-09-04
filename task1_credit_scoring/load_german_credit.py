"""
load_german_credit.py
----------------------
Loads the real UCI German Credit dataset (german.data) and converts it
into a model-ready CSV: numeric columns kept as-is, categorical columns
one-hot encoded, target renamed/remapped to match credit_scoring.py's
expectations (creditworthy: 1 = good, 0 = bad).

Usage:
    python load_german_credit.py
    # writes data/german_credit_processed.csv
"""

import pandas as pd

# Column names from german.doc, in file order.
COLUMNS = [
    "checking_status",       # categorical (A11-A14)
    "duration_months",       # numeric
    "credit_history",        # categorical (A30-A34)
    "purpose",                # categorical (A40-A410)
    "credit_amount",          # numeric
    "savings_status",         # categorical (A61-A65)
    "employment_since",       # categorical (A71-A75)
    "installment_rate_pct",   # numeric
    "personal_status_sex",    # categorical (A91-A95)
    "other_debtors",          # categorical (A101-A103)
    "residence_since",        # numeric
    "property",                # categorical (A121-A124)
    "age",                     # numeric
    "other_installment_plans", # categorical (A141-A143)
    "housing",                 # categorical (A151-A153)
    "num_existing_credits",    # numeric
    "job",                     # categorical (A171-A174)
    "num_dependents",          # numeric
    "telephone",                # categorical (A191-A192)
    "foreign_worker",           # categorical (A201-A202)
    "target",                    # 1 = good credit, 2 = bad credit
]

CATEGORICAL_COLUMNS = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment_since", "personal_status_sex", "other_debtors",
    "property", "other_installment_plans", "housing", "job",
    "telephone", "foreign_worker",
]


def load_and_process(raw_path: str = "data/german.data", out_path: str = "data/german_credit_processed.csv") -> pd.DataFrame:
    # german.data is whitespace-separated with no header row.
    df = pd.read_csv(raw_path, sep=r"\s+", header=None, names=COLUMNS)

    # Target: file uses 1=good, 2=bad. Remap to match credit_scoring.py's
    # convention: creditworthy = 1 (good), 0 (bad).
    df["creditworthy"] = (df["target"] == 1).astype(int)
    df = df.drop(columns=["target"])

    # One-hot encode every categorical column. drop_first=True avoids
    # redundant columns (e.g. with 2 categories, one column already
    # implies the other — encoding both would be duplicate information).
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)

    # get_dummies produces True/False; convert to 1/0 so every column
    # in the dataset is numeric.
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    df.to_csv(out_path, index=False)
    print(f"Processed dataset: {df.shape[0]} rows, {df.shape[1]} columns -> {out_path}")
    print(f"Class balance:\n{df['creditworthy'].value_counts(normalize=True)}")
    return df


if __name__ == "__main__":
    load_and_process()
