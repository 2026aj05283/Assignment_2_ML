"""
ML Assignment 2 - Telecom Customer Churn Classification
Author : Kunal Vaishnav
BITS ID: 2026aj05283

Trains five classifiers on the same dataset, evaluates them on a held-out
test split, and saves:
  * model/<model_key>.joblib      - fitted sklearn Pipelines (preprocessing + model)
  * model/metrics.json            - evaluation metrics on the test split
  * model/feature_meta.json       - feature list / target name used by the app
  * test_data.csv                 - the held-out test split (features + target)

Run:  python train_models.py
"""

from __future__ import annotations

import json
import os
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")  # inherited by joblib workers
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "telecom_churn.csv"
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"

TARGET = "churn"
RANDOM_STATE = 2026  # fixed seed for reproducibility

# SVC(probability=True) emits a FutureWarning on scikit-learn >= 1.9; it is still
# fully supported and needed here for predict_proba (AUC), so silence the noise.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
TEST_SIZE = 0.20


def build_candidates() -> dict[str, tuple[str, Pipeline, dict]]:
    """Return {key: (display_name, pipeline, param_grid)} for the five models."""
    return {
        "logistic_regression": (
            "Logistic Regression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
            ]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]},
        ),
        "decision_tree": (
            "Decision Tree",
            Pipeline([
                ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]),
            {
                "clf__max_depth": [4, 6, 8, 10, None],
                "clf__min_samples_leaf": [1, 5, 10],
                "clf__criterion": ["gini", "entropy"],
            },
        ),
        "knn": (
            "KNN",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier()),
            ]),
            {"clf__n_neighbors": [3, 5, 7, 9, 11, 15], "clf__weights": ["uniform", "distance"]},
        ),
        "naive_bayes": (
            "Naive Bayes",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GaussianNB()),
            ]),
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
        ),
        "svm": (
            "Support Vector Machine",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            ]),
            {"clf__C": [1.0, 3.0, 10.0], "clf__gamma": ["scale", 0.03], "clf__class_weight": [None, "balanced"]},
        ),
    }


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute the six required metrics (binary, positive class = 1 / churn)."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    print(f"Dataset: {df.shape[0]} rows, {X.shape[1]} features, churn rate={y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    MODEL_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results: dict[str, dict] = {}
    for key, (name, pipe, grid) in build_candidates().items():
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        metrics = {k: round(float(v), 4) for k, v in evaluate(best, X_test, y_test).items()}
        results[key] = {
            "name": name,
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_f1": round(float(search.best_score_), 4),
            "test_metrics": metrics,
        }
        joblib.dump(best, MODEL_DIR / f"{key}.joblib", compress=3)
        print(f"{name:<24} {metrics}  params={results[key]['best_params']}")

    # Persist artefacts used by the Streamlit app
    (MODEL_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    (MODEL_DIR / "feature_meta.json").write_text(json.dumps({
        "features": list(X.columns),
        "target": TARGET,
        "class_names": {"0": "No Churn", "1": "Churn"},
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }, indent=2))

    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    test_df.to_csv(TEST_CSV, index=False)
    print(f"Saved {len(test_df)} test rows -> {TEST_CSV.name}")


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
