"""
credit_scoring.py
------------------
CodeAlpha Task 1: Credit Scoring Model

Predicts an individual's creditworthiness from financial history using
Logistic Regression, Decision Tree, and Random Forest, and compares them
on Precision, Recall, F1, and ROC-AUC.

Usage:
    python credit_scoring.py                     # uses/generates data/credit_data.csv
    python credit_scoring.py --data-path my.csv   # use your own dataset
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from generate_data import generate_credit_dataset

TARGET = "creditworthy"
RANDOM_STATE = 42


def load_data(data_path: str) -> pd.DataFrame:
    if os.path.exists(data_path):
        print(f"Loading dataset from {data_path}")
        return pd.read_csv(data_path)
    print(f"No dataset found at {data_path} — generating a synthetic demo dataset instead.")
    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
    df = generate_credit_dataset()
    df.to_csv(data_path, index=False)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive standard credit-risk ratios from raw financial fields."""
    df = df.copy()

    if {"existing_debt", "annual_income"}.issubset(df.columns):
        df["debt_to_income_ratio"] = df["existing_debt"] / (df["annual_income"] + 1)

    if {"loan_amount_requested", "annual_income"}.issubset(df.columns):
        df["loan_to_income_ratio"] = df["loan_amount_requested"] / (df["annual_income"] + 1)

    if {"savings", "annual_income"}.issubset(df.columns):
        df["savings_rate"] = df["savings"] / (df["annual_income"] + 1)

    if "num_late_payments_2yr" in df.columns:
        df["has_late_payments"] = (df["num_late_payments_2yr"] > 0).astype(int)

    return df


def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    results = []
    roc_curves = {}
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        }
        results.append(metrics)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves[name] = (fpr, tpr, metrics["ROC-AUC"])
        fitted[name] = model

        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, target_names=["Not Creditworthy", "Creditworthy"]))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    return pd.DataFrame(results), roc_curves, fitted


def plot_roc_curves(roc_curves, out_path: str):
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Credit Scoring Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve plot -> {out_path}")


def plot_feature_importance(model, feature_names, out_path: str):
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh")
    plt.title("Random Forest — Feature Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved feature importance plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Credit Scoring Model")
    parser.add_argument("--data-path", default="data/credit_data.csv")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load & feature engineer
    df = load_data(args.data_path)
    df = engineer_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 2. Split & scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # 3. Train & evaluate
    results_df, roc_curves, fitted = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)

    print("\n================ SUMMARY ================")
    print(results_df.sort_values("ROC-AUC", ascending=False).to_string(index=False))

    results_df.to_csv(os.path.join(args.out_dir, "model_comparison.csv"), index=False)

    # 4. Plots
    plot_roc_curves(roc_curves, os.path.join(args.out_dir, "roc_curves.png"))
    plot_feature_importance(
        fitted["Random Forest"], X_train_scaled.columns, os.path.join(args.out_dir, "feature_importance.png")
    )

    best_model_name = results_df.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
    print(f"\nBest model by ROC-AUC: {best_model_name}")


if __name__ == "__main__":
    main()
