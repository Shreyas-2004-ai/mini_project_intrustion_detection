"""
run_preprocessing.py
====================
Entry-point script for Stage 1: Data Inspection & Preprocessing.

Usage
-----
    python run_preprocessing.py

What it does
------------
1. Loads both raw UNSW-NB15 CSV files with validation.
2. Inspects each dataset and prints a structured report.
3. Fits a preprocessing pipeline on the training set.
4. Transforms both splits consistently and saves them to data/processed/.
5. Prints a final concise summary.
"""

import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader  import load_train_test
from src.inspector    import inspect_dataset, print_report
from src.preprocessor import run_preprocessing
from src.logger       import get_logger

log = get_logger("run_preprocessing")


def main() -> None:
    log.info("=" * 65)
    log.info("GNN Intrusion Detection — Stage 1: Preprocessing")
    log.info("=" * 65)

    # ── 1. Load ────────────────────────────────────────────────────────────
    log.info("Step 1/3 — Loading raw datasets ...")
    train_df, test_df = load_train_test()

    # ── 2. Inspect ─────────────────────────────────────────────────────────
    log.info("Step 2/3 — Inspecting datasets ...")
    train_report = inspect_dataset(train_df, name="UNSW-NB15 Training Set")
    test_report  = inspect_dataset(test_df,  name="UNSW-NB15 Testing Set")

    print_report(train_report)
    print_report(test_report)

    # ── 3. Preprocess ──────────────────────────────────────────────────────
    log.info("Step 3/3 — Running preprocessing pipeline ...")
    train_proc, test_proc = run_preprocessing(train_df, test_df)

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PREPROCESSING COMPLETE")
    print("=" * 65)
    print(f"  Train processed : {train_proc.shape[0]:>7,} rows  x  {train_proc.shape[1]} cols")
    print(f"  Test  processed : {test_proc.shape[0]:>7,} rows  x  {test_proc.shape[1]} cols")
    print(f"  Saved to        : data/processed/")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("Preprocessing failed: %s", exc, exc_info=True)
        sys.exit(1)
