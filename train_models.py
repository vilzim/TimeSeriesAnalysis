"""Time-series up/down prediction: initial classical ML benchmark.

Use after producing data/features/SYMBOL_features.csv with more_data.py.
Example:
    python train_models.py --file data/features/AAPL_features.csv

Requires: pandas, numpy, scikit-learn; xgboost optional.
"""

import argparse
import json
from pathlib import Path

from joblib import dump

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURE_COLUMNS = [
    "return_1d", "return_1d_1lag", "return_1d_2lag", "return_1d_3lag",
    "price_to_SMA_5", "price_to_SMA_10", "price_to_SMA_20",
    "rsi", "macd", "macd_signal", "macd_histogram",
    "volatility_10", "volume_change",
]


def load_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError("Dataset contains duplicate timestamps; fix them before training.")
    missing = sorted(set(FEATURE_COLUMNS + ["target"]) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Keep only 13 engineered features and the target: no future information as X.
    for col in FEATURE_COLUMNS + ["target"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    original_count = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + ["target"])
    if not df["target"].isin((0, 1)).all():
        raise ValueError("Target must be binary (0 or 1).")
    df["target"] = df["target"].astype(int)
    print(f"Loaded {len(df)} usable rows ({original_count - len(df)} rows excluded).")
    return df


def split_chronologically(df, train_end, val_end, test_end):
    train_end, val_end, test_end = map(pd.Timestamp, (train_end, val_end, test_end))
    if not train_end < val_end < test_end:
        raise ValueError("Require train_end < val_end < test_end.")

    train = df.loc[df.index < train_end].copy()
    val = df.loc[(df.index >= train_end) & (df.index < val_end)].copy()
    test = df.loc[(df.index >= val_end) & (df.index < test_end)].copy()

    # target[t] depends on the price at t+1. Purge the last training/validation
    # observation so its label never depends on the following split's prices.
    train = train.iloc[:-1]
    val = val.iloc[:-1]

    for name, part in (("Train", train), ("Validation", val), ("Test", test)):
        if part.empty:
            raise ValueError(f"{name} has no data. Adjust the date boundaries.")
        print(f"{name:10}: {len(part):4} rows, {part.index.min().date()}–{part.index.max().date()}, "
              f"positive target={part['target'].mean():.1%}")
    if train["target"].nunique() < 2:
        raise ValueError("Training data must contain both target classes.")
    return train, val, test


def build_models():
    models = {
        "Always majority": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=6, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0)),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            objective="binary:logistic", n_estimators=250,
            max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=2,
        )
    except ImportError:
        print("XGBoost is not installed: skipping it (pip install xgboost).")
    return models


def evaluate(model, x, y):
    predicted = model.predict(x)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(x)[:, 1]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(x)
    else:
        score = None

    result = {
        "accuracy": accuracy_score(y, predicted),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "f1": f1_score(y, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y, score) if score is not None and len(np.unique(y)) == 2 else np.nan,
    }
    return result, predicted


def main():
    parser = argparse.ArgumentParser(description="Compare classifiers on historical financial features")
    parser.add_argument("--file", default="data/features/AAPL_features.csv")
    parser.add_argument("--train-end", default="2022-01-01", help="Validation starts on this date")
    parser.add_argument("--val-end", default="2023-01-01", help="Test starts on this date")
    parser.add_argument("--test-end", default="2024-01-01", help="Exclusive end of test")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    df = load_data(args.file)
    train, val, test = split_chronologically(df, args.train_end, args.val_end, args.test_end)
    x_train, y_train = train[FEATURE_COLUMNS], train["target"]

    output_dir = Path(args.output)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    test_predictions = pd.DataFrame({"actual_target": test["target"]}, index=test.index)
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        dump(model, model_dir / f"{name.lower().replace(' ', '_')}.joblib")
        for split_name, split in (("validation", val), ("test", test)):
            metrics, predictions = evaluate(model, split[FEATURE_COLUMNS], split["target"])
            metrics_rows.append({"model": name, "split": split_name, **metrics})
            if split_name == "test":
                test_predictions[name] = predictions

    metrics_df = pd.DataFrame(metrics_rows)
    metadata = {"feature_columns": FEATURE_COLUMNS, "target": "next-day up/down",
                "train_end_exclusive": args.train_end, "validation_end_exclusive": args.val_end,
                "test_end_exclusive": args.test_end,
                "note": "Saved benchmark models were trained on the train partition only."}
    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index_label="timestamp")

    print("\nValidation (used to compare models and choose settings):")
    print(metrics_df[metrics_df["split"] == "validation"].round(4).to_string(index=False))
    print("\nTest (unseen historical period; do not tune based on it):")
    print(metrics_df[metrics_df["split"] == "test"].round(4).to_string(index=False))
    print(f"\nSaved metrics, predictions, and benchmark models in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
