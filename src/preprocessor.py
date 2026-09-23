"""
Preprocessing pipeline for UNSW-NB15.

Strategy
--------
- Numerical : StandardScaler  (handles varied magnitudes in traffic features)
- Categorical: OrdinalEncoder  (memory-efficient; compatible with tree/GNN models)
- Identifier columns (id): dropped
- Target columns (label, attack_cat): preserved as-is, never transformed

The fitted pipeline is serialised to data/processed/preprocessor.pkl so the
same transformations can be applied to any new CSV without re-fitting.
"""

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .config import (
    CATEGORICAL_COLS,
    IDENTIFIER_COLS,
    PREPROCESSOR_PATH,
    TARGET_COLS,
    TRAIN_PROCESSED_PATH,
    TEST_PROCESSED_PATH,
)
from .data_loader import get_numerical_cols
from .logger import get_logger

log = get_logger(__name__)


# ── Pipeline builder ──────────────────────────────────────────────────────────

def build_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    """
    Build (but do not fit) a ColumnTransformer based on the column layout of df.

    Parameters
    ----------
    df : pd.DataFrame
        A raw DataFrame (used only to resolve column names).

    Returns
    -------
    ColumnTransformer
    """
    num_cols = get_numerical_cols(df)
    log.info("  Numerical features  (%d): %s", len(num_cols), num_cols)
    log.info("  Categorical features(%d): %s", len(CATEGORICAL_COLS), CATEGORICAL_COLS)

    num_pipeline = Pipeline([("scaler", StandardScaler())])

    # handle_unknown='use_encoded_value' + unknown_value=-1 keeps the pipeline
    # safe if the test set contains unseen category values.
    cat_pipeline = Pipeline(
        [
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float64,
                ),
            )
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, CATEGORICAL_COLS),
        ],
        remainder="drop",   # drops id; targets are handled separately
        verbose_feature_names_out=False,
    )
    return transformer


# ── Fit / transform helpers ───────────────────────────────────────────────────

def fit_pipeline(train_df: pd.DataFrame) -> ColumnTransformer:
    """Fit the ColumnTransformer on the training set and return it."""
    log.info("Fitting preprocessing pipeline on training data ...")
    pipeline = build_pipeline(train_df)
    pipeline.fit(train_df)
    log.info("Pipeline fitted successfully.")
    return pipeline


def transform_split(
    pipeline: ColumnTransformer,
    df: pd.DataFrame,
    name: str = "split",
) -> pd.DataFrame:
    """
    Apply a fitted pipeline to df and return a DataFrame with original
    feature names plus the untouched target columns.

    Parameters
    ----------
    pipeline : fitted ColumnTransformer
    df       : raw DataFrame to transform
    name     : label for logging

    Returns
    -------
    pd.DataFrame  — processed features + target columns
    """
    log.info("Transforming '%s' (%d rows) ...", name, len(df))

    X_arr   = pipeline.transform(df)
    feat_names = pipeline.get_feature_names_out()
    X_df    = pd.DataFrame(X_arr, columns=feat_names, index=df.index)

    # Re-attach target columns unchanged
    for col in TARGET_COLS:
        if col in df.columns:
            X_df[col] = df[col].values

    log.info("  Transformed shape: %s", X_df.shape)
    return X_df


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_pipeline(pipeline: ColumnTransformer, path: Path = PREPROCESSOR_PATH) -> None:
    """Serialise the fitted pipeline to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    log.info("Pipeline saved to: %s", path)


def load_pipeline(path: Path = PREPROCESSOR_PATH) -> ColumnTransformer:
    """Load a previously saved pipeline."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {path}. Run preprocessing first.")
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    log.info("Pipeline loaded from: %s", path)
    return pipeline


# ── End-to-end convenience ────────────────────────────────────────────────────

def run_preprocessing(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit on train, transform both splits, save results.

    Returns
    -------
    (train_processed, test_processed)
    """
    TRAIN_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    pipeline      = fit_pipeline(train_df)
    train_proc    = transform_split(pipeline, train_df, name="train")
    test_proc     = transform_split(pipeline, test_df,  name="test")

    # Persist
    train_proc.to_csv(TRAIN_PROCESSED_PATH, index=False)
    test_proc.to_csv(TEST_PROCESSED_PATH,   index=False)
    log.info("Processed train saved to: %s", TRAIN_PROCESSED_PATH)
    log.info("Processed test  saved to: %s", TEST_PROCESSED_PATH)

    save_pipeline(pipeline)
    return train_proc, test_proc
