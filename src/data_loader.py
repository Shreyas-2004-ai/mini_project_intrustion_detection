"""
Data loading and basic validation for UNSW-NB15 CSV files.
Reads raw CSVs, verifies shape/columns, and returns DataFrames unchanged.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import (
    CATEGORICAL_COLS,
    EXPECTED_COLS,
    IDENTIFIER_COLS,
    TARGET_COLS,
    TRAIN_RAW_PATH,
    TEST_RAW_PATH,
)
from .logger import get_logger

log = get_logger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load a single UNSW-NB15 CSV file with validation.

    Parameters
    ----------
    path : Path
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame, unmodified.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the column count or required columns are wrong.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    log.info("Loading dataset: %s", path.name)
    df = pd.read_csv(path, low_memory=False)
    log.info("  Loaded %d rows x %d columns", *df.shape)

    # Column count check
    if df.shape[1] != EXPECTED_COLS:
        log.warning(
            "  Expected %d columns, found %d — verify the file.", EXPECTED_COLS, df.shape[1]
        )

    # Required columns check
    required = set(TARGET_COLS + IDENTIFIER_COLS + CATEGORICAL_COLS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing from {path.name}: {missing}")

    return df


def load_train_test() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both training and testing sets and return them as a tuple."""
    train = load_dataset(TRAIN_RAW_PATH)
    test  = load_dataset(TEST_RAW_PATH)
    return train, test


def get_numerical_cols(df: pd.DataFrame) -> list:
    """
    Derive numerical feature columns dynamically from the DataFrame.
    Excludes identifiers, targets, and categorical columns.
    """
    exclude = set(IDENTIFIER_COLS + TARGET_COLS + CATEGORICAL_COLS)
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def get_feature_cols(df: pd.DataFrame) -> dict:
    """
    Return a dict grouping columns by role:
        {
            'identifier':   [...],
            'target':       [...],
            'categorical':  [...],
            'numerical':    [...],
        }
    """
    num_cols = get_numerical_cols(df)
    return {
        "identifier":  IDENTIFIER_COLS,
        "target":      TARGET_COLS,
        "categorical": CATEGORICAL_COLS,
        "numerical":   num_cols,
    }
