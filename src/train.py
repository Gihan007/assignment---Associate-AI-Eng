import json
import os
from pathlib import Path
from typing import Optional, List

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from .data_prep import build_preprocessor, engineer_features


def save_confusion_plot(cm, labels, out_path: Path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def train_models(train_csv: str, target: str = "Exited", drop_cols: Optional[List[str]] = None, model_out: Optional[str] = None, metrics_out: Optional[str] = None):
    df = pd.read_csv(train_csv)

    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")

    df = engineer_features(df)

    feature_cols = [c for c in df.columns if c != target]
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target], test_size=0.2, stratify=df[target], random_state=42)

    feature_engineering = FunctionTransformer(engineer_features, validate=False)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    log_reg = Pipeline([
        ("features", feature_engineering),
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    rf = Pipeline([
        ("features", feature_engineering),
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ])

    lgbm = Pipeline([
        ("features", feature_engineering),
        ("preprocessor", preprocessor),
        ("model", LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    cat = Pipeline([
        ("features", feature_engineering),
        ("preprocessor", preprocessor),
        ("model", CatBoostClassifier(
            iterations=400,
            depth=8,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            random_seed=42,
            thread_count=-1,
        )),
    ])

    models = {
        "log_reg": log_reg,
        "random_forest": rf,
        "lightgbm": lgbm,
        "catboost": cat,
    }
    metrics = {}
    confusion = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        confusion[name] = cm
        metrics[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(y_test, preds, zero_division=0),
        }

    best_name = max(metrics, key=lambda m: metrics[m]["f1"])
    best_model = models[best_name]

    if model_out is None:
        model_out = os.getenv("MODEL_PATH", "models/churn_model.joblib")
    if metrics_out is None:
        metrics_out = os.getenv("METRICS_PATH", "models/metrics.json")

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_out)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "metrics": metrics}, f, indent=2)

    cm_path = Path(metrics_out).with_name("confusion_matrix.png")
    save_confusion_plot(confusion[best_name], labels=[0, 1], out_path=cm_path)

    return best_name, metrics, model_out, metrics_out, cm_path


def main():
    load_dotenv()
    train_csv = os.getenv("TRAIN_CSV", "data/Churn_Modelling.csv")
    target = os.getenv("TRAIN_TARGET", "Exited")
    drop_cols_env = os.getenv("EXCLUDE_COLS", "RowNumber,CustomerId,Surname")
    drop_cols = [c for c in drop_cols_env.split(",") if c]

    best_name, metrics, model_path, metrics_path, cm_path = train_models(train_csv, target=target, drop_cols=drop_cols)
    print(f"Best model: {best_name}")
    for name, m in metrics.items():
        print(f"\nModel: {name}")
        print(f"Accuracy: {m['accuracy']:.3f}, Precision: {m['precision']:.3f}, Recall: {m['recall']:.3f}, F1: {m['f1']:.3f}")
        print("Confusion matrix:", m["confusion_matrix"])
        print(m["classification_report"])
    print(f"Saved best model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix plot to: {cm_path}")


if __name__ == "__main__":
    main()
