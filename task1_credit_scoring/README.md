# Credit Scoring Model

Predicts whether an individual is creditworthy from financial history,
comparing Logistic Regression, Decision Tree, and Random Forest.

## What it does

1. **Loads data** from `data/credit_data.csv`. If the file doesn't exist,
   it auto-generates a realistic 5,000-row synthetic dataset (`generate_data.py`)
   so the pipeline runs immediately with no setup.
2. **Feature engineering**: derives `debt_to_income_ratio`,
   `loan_to_income_ratio`, `savings_rate`, and `has_late_payments` from
   the raw fields.
3. **Trains** Logistic Regression, Decision Tree, and Random Forest on a
   stratified 80/20 split, after standardizing features.
4. **Evaluates** each model on Accuracy, Precision, Recall, F1-Score, and
   ROC-AUC, and prints a full classification report + confusion matrix.
5. **Saves outputs** to `outputs/`: `model_comparison.csv`,
   `roc_curves.png`, `feature_importance.png`.

## Run it

```bash
pip install -r requirements.txt
python credit_scoring.py
```

To use a real dataset (recommended for your final submission — e.g. the
UCI "German Credit Data" or Kaggle "Give Me Some Credit"), place a CSV
with a `creditworthy` (0/1) target column at `data/credit_data.csv`, or:

```bash
python credit_scoring.py --data-path path/to/your_data.csv
```

## Results (synthetic demo data)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.818 | 0.813 | 0.826 | 0.819 | 0.892 |
| Random Forest | 0.805 | 0.809 | 0.798 | 0.804 | 0.879 |
| Decision Tree | 0.749 | 0.754 | 0.740 | 0.747 | 0.827 |

Numbers on a real dataset will differ — this table is from the bundled
synthetic data and mainly demonstrates the pipeline works end-to-end.
