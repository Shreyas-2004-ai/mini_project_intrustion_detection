"""
random_forest.py
================
Random Forest baseline for node-level binary intrusion detection on UNSW-NB15.

Purpose
-------
Provides a conventional-ML comparison point for the GraphSAGE baseline.
Uses EXACTLY the same 42 preprocessed input features as GraphSAGE.
No graph structure is used — each flow record is an independent sample.

Feature input
-------------
The same sklearn ColumnTransformer pipeline fitted in Phase 1 is loaded here.
Features are the 42 columns output by that pipeline:
  - 39 scaled numerical features
  - 3 ordinal-encoded categorical features (proto, service, state)
Neither id, label, nor attack_cat appears in X.

Class imbalance
---------------
Handled via class_weight='balanced' in RandomForestClassifier, which
internally sets each class weight to n_samples / (n_classes * n_samples_c).
This mirrors the approach used in the GraphSAGE weighted CrossEntropyLoss.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ── Default configuration ──────────────────────────────────────────────────────

RF_DEFAULTS = {
    "n_estimators":   200,
    "max_depth":      None,      # grow full trees
    "min_samples_leaf": 1,
    "max_features":   "sqrt",    # standard RF heuristic
    "class_weight":   "balanced",
    "n_jobs":         -1,        # use all CPU cores
    "random_state":   42,
}


# ── Training ───────────────────────────────────────────────────────────────────

def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg:     Dict,
) -> RandomForestClassifier:
    """
    Fit a RandomForestClassifier on the training data.

    Parameters
    ----------
    X_train : (N_train, 42) feature matrix
    y_train : (N_train,) binary labels
    cfg     : hyperparameter dict (see RF_DEFAULTS)

    Returns
    -------
    Fitted RandomForestClassifier
    """
    clf = RandomForestClassifier(
        n_estimators    = cfg["n_estimators"],
        max_depth       = cfg["max_depth"],
        min_samples_leaf= cfg["min_samples_leaf"],
        max_features    = cfg["max_features"],
        class_weight    = cfg["class_weight"],
        n_jobs          = cfg["n_jobs"],
        random_state    = cfg["random_state"],
    )
    clf.fit(X_train, y_train)
    return clf


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_rf(
    clf:    RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """
    Evaluate a fitted classifier and return a metrics dict.
    """
    preds = clf.predict(X_test)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)
    cm   = confusion_matrix(y_test, preds, labels=[0, 1]).tolist()
    cr   = classification_report(
        y_test, preds,
        labels=[0, 1],
        target_names=["Normal (0)", "Attack (1)"],
        zero_division=0,
    )

    n_normal = int((y_test == 0).sum())
    n_attack = int((y_test == 1).sum())

    return {
        "total_nodes":           len(y_test),
        "n_normal":              n_normal,
        "n_attack":              n_attack,
        "accuracy":              round(float(acc),  4),
        "precision":             round(float(prec), 4),
        "recall":                round(float(rec),  4),
        "f1":                    round(float(f1),   4),
        "confusion_matrix":      cm,
        "classification_report": cr,
    }


# ── Persistence ────────────────────────────────────────────────────────────────

def save_rf(clf: RandomForestClassifier, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "random_forest.pkl"
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    return path


def load_rf(path: Path) -> RandomForestClassifier:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Random Forest model not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Console report ─────────────────────────────────────────────────────────────

def print_rf_report(metrics: Dict, feature_names: list, cfg: Dict) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("  RANDOM FOREST — FINAL TEST SET EVALUATION")
    print("  UNSW-NB15  (per-flow, no graph structure)")
    print(sep)
    print(f"\n  Configuration:")
    for k, v in cfg.items():
        print(f"    {k:<20}: {v}")
    print(f"\n  Input features ({len(feature_names)}) — same as GraphSAGE x:")
    for i, n in enumerate(feature_names):
        print(f"    [{i:>2}] {n}")
    print(f"\n  Test set composition:")
    print(f"    Total samples : {metrics['total_nodes']:>10,}")
    print(f"    Normal        : {metrics['n_normal']:>10,}")
    print(f"    Attack        : {metrics['n_attack']:>10,}")
    print(f"\n  Metrics (per-flow binary classification):")
    print(f"    Accuracy      : {metrics['accuracy']:.4f}")
    print(f"    Precision     : {metrics['precision']:.4f}")
    print(f"    Recall        : {metrics['recall']:.4f}")
    print(f"    F1-score      : {metrics['f1']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion matrix (rows=True, cols=Predicted):")
    print(f"                    Pred Normal  Pred Attack")
    print(f"    True Normal  : {cm[0][0]:>10,}  {cm[0][1]:>10,}")
    print(f"    True Attack  : {cm[1][0]:>10,}  {cm[1][1]:>10,}")
    print(f"\n  Classification report:")
    print(metrics["classification_report"])
    print(sep)
    print("  NOTE: This is the conventional ML baseline.")
    print("  GraphSAGE comparison uses the same 42 features on graph structure.")
    print(f"{sep}\n")
