"""
Dataset inspection and summary reporting.
Produces structured summaries without modifying the data.
"""

from typing import Dict, Any

import numpy as np
import pandas as pd

from .config import TARGET_COLS, ATTACK_CATEGORIES
from .data_loader import get_feature_cols
from .logger import get_logger

log = get_logger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def inspect_dataset(df: pd.DataFrame, name: str = "dataset") -> Dict[str, Any]:
    """
    Perform a comprehensive inspection of a raw DataFrame.

    Returns a dict with all findings so callers can log, assert, or display them.
    """
    log.info("=== Inspecting: %s ===", name)
    report: Dict[str, Any] = {"name": name}

    # Shape
    report["n_rows"], report["n_cols"] = df.shape
    log.info("  Shape: %d rows x %d cols", report["n_rows"], report["n_cols"])

    # Column names
    report["columns"] = list(df.columns)

    # Data types
    report["dtypes"] = df.dtypes.astype(str).to_dict()

    # Duplicate rows
    n_dup = int(df.duplicated().sum())
    report["duplicate_rows"] = n_dup
    log.info("  Duplicate rows: %d", n_dup)

    # Missing values
    missing = df.isnull().sum()
    report["missing_per_col"] = missing[missing > 0].to_dict()
    report["total_missing"]   = int(missing.sum())
    if report["total_missing"] > 0:
        log.warning("  Missing values found: %s", report["missing_per_col"])
    else:
        log.info("  No missing values detected.")

    # Column roles
    report["col_roles"] = get_feature_cols(df)
    log.info(
        "  Numerical features: %d | Categorical features: %d",
        len(report["col_roles"]["numerical"]),
        len(report["col_roles"]["categorical"]),
    )

    # Target analysis
    report["targets"] = _inspect_targets(df)

    # Leakage check
    report["leakage_warnings"] = _check_leakage(df)

    return report


def _inspect_targets(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyse label and attack_cat distributions."""
    targets = {}

    for col in TARGET_COLS:
        if col not in df.columns:
            log.warning("  Target column '%s' not found in DataFrame.", col)
            continue

        info: Dict[str, Any] = {}
        info["missing"]       = int(df[col].isnull().sum())
        info["unique_values"] = sorted(df[col].dropna().unique().tolist())
        vc = df[col].value_counts(dropna=False)
        info["value_counts"]  = vc.to_dict()

        if info["missing"] > 0:
            log.warning("  Target '%s' has %d missing values!", col, info["missing"])
        else:
            log.info("  Target '%s': %d unique values, no missing.", col, len(info["unique_values"]))

        targets[col] = info

    # Consistency check: every attack where label==1 should have a non-Normal attack_cat
    if "label" in df.columns and "attack_cat" in df.columns:
        mismatch = df[(df["label"] == 0) & (df["attack_cat"] != "Normal")]
        if len(mismatch) > 0:
            log.warning(
                "  %d rows have label=0 but attack_cat != 'Normal' — potential inconsistency.",
                len(mismatch),
            )
        else:
            log.info("  label / attack_cat consistency check passed.")
        targets["label_attack_cat_mismatch"] = len(mismatch)

    return targets


def _check_leakage(df: pd.DataFrame) -> list:
    """
    Heuristic leakage checks.
    Returns a list of warning strings (empty if none found).
    """
    warnings = []

    # Columns whose names suggest they encode the target directly
    suspicious_patterns = ["label", "attack", "class", "category", "target"]
    feature_cols = (
        get_feature_cols(df)["numerical"] + get_feature_cols(df)["categorical"]
    )
    for col in feature_cols:
        for pat in suspicious_patterns:
            if pat in col.lower():
                msg = f"Possible leakage: feature column '{col}' contains '{pat}'"
                warnings.append(msg)
                log.warning("  %s", msg)

    if not warnings:
        log.info("  No obvious leakage columns detected in feature set.")

    return warnings


def print_report(report: Dict[str, Any]) -> None:
    """Pretty-print an inspection report to stdout."""
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  DATASET REPORT — {report['name'].upper()}")
    print(sep)

    print(f"\n  Rows : {report['n_rows']:>10,}")
    print(f"  Cols : {report['n_cols']:>10,}")
    print(f"  Duplicates  : {report['duplicate_rows']:>10,}")
    print(f"  Total missing: {report['total_missing']:>9,}")

    roles = report["col_roles"]
    print(f"\n  Column roles:")
    print(f"    Identifier  ({len(roles['identifier']):>2}): {roles['identifier']}")
    print(f"    Target      ({len(roles['target']):>2}): {roles['target']}")
    print(f"    Categorical ({len(roles['categorical']):>2}): {roles['categorical']}")
    print(f"    Numerical   ({len(roles['numerical']):>2}): {roles['numerical']}")

    print("\n  Target distributions:")
    for col, info in report["targets"].items():
        if col == "label_attack_cat_mismatch":
            continue
        print(f"\n    [{col}]  missing={info['missing']}")
        for val, cnt in sorted(info["value_counts"].items(), key=lambda x: -x[1] if isinstance(x[1], int) else 0):
            print(f"      {str(val):<25} {cnt:>8,}")

    if report.get("leakage_warnings"):
        print("\n  ⚠ Leakage warnings:")
        for w in report["leakage_warnings"]:
            print(f"    - {w}")

    print(f"\n{sep}\n")
