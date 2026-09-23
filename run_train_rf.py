"""
run_train_rf.py
===============
Entry-point for the Random Forest conventional-ML baseline.

Usage
-----
    python run_train_rf.py [--n_estimators N] [--max_depth D] [--seed S]

What it does
------------
1. Loads raw training and testing CSVs.
2. Applies the SAME preprocessing pipeline fitted in Phase 1 (no re-fit).
3. Extracts the 42-feature matrix and binary labels — identical feature set
   used by GraphSAGE (same pipeline output, same column order).
4. Verifies id / label / attack_cat are NOT in X.
5. Trains a RandomForestClassifier on training data only.
6. Evaluates ONCE on the official test set.
7. Saves model, metrics JSON, classification report, and config to disk.
8. Prints the full evaluation report including all 42 feature names.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    TRAIN_RAW_PATH,
    TEST_RAW_PATH,
    TARGET_COLS,
    IDENTIFIER_COLS,
)
from src.data_loader import load_dataset
from src.logger import get_logger
from src.models.random_forest import (
    RF_DEFAULTS,
    evaluate_rf,
    load_rf,
    print_rf_report,
    save_rf,
    train_rf,
)
from src.preprocessor import load_pipeline, transform_split

log = get_logger("run_train_rf")

RF_MODELS_DIR  = Path("models") / "random_forest"
RF_RESULTS_DIR = Path("results") / "ml"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    d = RF_DEFAULTS
    p = argparse.ArgumentParser(description="Random Forest baseline — UNSW-NB15")
    p.add_argument("--n_estimators",    type=int,   default=d["n_estimators"])
    p.add_argument("--max_depth",       type=int,   default=None)
    p.add_argument("--min_samples_leaf",type=int,   default=d["min_samples_leaf"])
    p.add_argument("--max_features",    type=str,   default=d["max_features"])
    p.add_argument("--seed",            type=int,   default=d["random_state"])
    return p.parse_args()


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_split(raw_path: Path, pipeline, split_name: str):
    """
    Load a raw CSV, apply the fitted pipeline, return (X, y, feature_names).
    Verifies that id / label / attack_cat are excluded from X.
    """
    raw_df = load_dataset(raw_path)
    proc_df = transform_split(pipeline, raw_df, name=split_name)

    # Feature columns = pipeline output columns (no targets, no id)
    exclude = set(TARGET_COLS + IDENTIFIER_COLS)
    feature_names = [c for c in proc_df.columns if c not in exclude]

    # Hard assertions — same checks as GNN pre-flight
    assert len(feature_names) == 42, (
        f"Expected 42 features, got {len(feature_names)}: {feature_names}"
    )
    assert "label"      not in feature_names, "label found in feature columns"
    assert "attack_cat" not in feature_names, "attack_cat found in feature columns"
    assert "id"         not in feature_names, "id found in feature columns"

    X = proc_df[feature_names].values.astype("float32")
    y = proc_df["label"].values.astype("int32")

    log.info(
        "  %s: X=%s  y=%s  (normal=%d  attack=%d)",
        split_name, X.shape, y.shape,
        (y == 0).sum(), (y == 1).sum(),
    )
    return X, y, feature_names


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = {
        "n_estimators":    args.n_estimators,
        "max_depth":       args.max_depth,
        "min_samples_leaf":args.min_samples_leaf,
        "max_features":    args.max_features,
        "class_weight":    "balanced",
        "n_jobs":          -1,
        "random_state":    args.seed,
    }

    log.info("=" * 65)
    log.info("GNN Intrusion Detection — Random Forest Baseline")
    log.info("=" * 65)
    log.info("Config: %s", cfg)

    RF_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RF_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load pipeline (fitted on training data only in Phase 1) ───────────
    log.info("Loading preprocessing pipeline ...")
    pipeline = load_pipeline()

    # ── Prepare data ──────────────────────────────────────────────────────
    log.info("Preparing training data ...")
    X_train, y_train, feature_names = prepare_split(TRAIN_RAW_PATH, pipeline, "train")

    log.info("Preparing test data ...")
    X_test,  y_test,  _             = prepare_split(TEST_RAW_PATH,  pipeline, "test")

    # ── Verify feature lists match ────────────────────────────────────────
    _, _, test_feat_names = prepare_split(TEST_RAW_PATH, pipeline, "test-verify")
    assert feature_names == test_feat_names, \
        "Feature columns differ between train and test — pipeline mismatch."

    # ── Train ─────────────────────────────────────────────────────────────
    log.info("Training Random Forest (%d estimators) ...", cfg["n_estimators"])
    t0 = time.time()
    clf = train_rf(X_train, y_train, cfg)
    elapsed = time.time() - t0
    log.info("Training complete in %.1f s", elapsed)

    # ── Save model ────────────────────────────────────────────────────────
    model_path = save_rf(clf, RF_MODELS_DIR)
    log.info("Model saved: %s", model_path)

    # ── Evaluate on official test set (used ONCE) ─────────────────────────
    log.info("Evaluating on official test set ...")
    metrics = evaluate_rf(clf, X_test, y_test)

    # ── Save results ──────────────────────────────────────────────────────
    metrics_clean = {k: v for k, v in metrics.items() if k != "classification_report"}
    metrics_path = RF_RESULTS_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_clean, f, indent=2)

    report_path = RF_RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(metrics["classification_report"])

    cfg_path = RF_RESULTS_DIR / "model_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    feat_path = RF_RESULTS_DIR / "feature_names.json"
    with open(feat_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    log.info("Results saved to: %s", RF_RESULTS_DIR)

    # ── Print report ──────────────────────────────────────────────────────
    print_rf_report(metrics, feature_names, cfg)

    # ── Feature importance top-20 ─────────────────────────────────────────
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:20]
    print("  Top-20 feature importances:")
    print(f"  {'Rank':<5} {'Feature':<25} {'Importance':>10}")
    print("  " + "-" * 43)
    for rank, i in enumerate(idx, 1):
        print(f"  {rank:<5} {feature_names[i]:<25} {importances[i]:>10.5f}")
    print()

    # Save full importances
    imp_df_path = RF_RESULTS_DIR / "feature_importances.json"
    imp_dict = {feature_names[i]: round(float(importances[i]), 6) for i in range(len(feature_names))}
    with open(imp_df_path, "w") as f:
        json.dump(imp_dict, f, indent=2)
    log.info("Feature importances saved: %s", imp_df_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("Random Forest training failed: %s", exc, exc_info=True)
        sys.exit(1)
