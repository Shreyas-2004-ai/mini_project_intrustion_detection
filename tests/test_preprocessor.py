"""
Tests for src/preprocessor.py

Run with:  python -m pytest tests/ -v
"""

import pickle
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessor import (
    build_pipeline,
    fit_pipeline,
    transform_split,
    save_pipeline,
    load_pipeline,
    run_preprocessing,
)
from src.config import (
    TARGET_COLS,
    CATEGORICAL_COLS,
    IDENTIFIER_COLS,
    TRAIN_RAW_PATH,
    TEST_RAW_PATH,
)
from src.data_loader import get_numerical_cols


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_train():
    return pd.read_csv(TRAIN_RAW_PATH, nrows=300)


@pytest.fixture(scope="module")
def raw_test():
    return pd.read_csv(TEST_RAW_PATH, nrows=100)


@pytest.fixture(scope="module")
def fitted_pipeline(raw_train):
    return fit_pipeline(raw_train)


@pytest.fixture(scope="module")
def train_transformed(fitted_pipeline, raw_train):
    return transform_split(fitted_pipeline, raw_train, name="train")


@pytest.fixture(scope="module")
def test_transformed(fitted_pipeline, raw_test):
    return transform_split(fitted_pipeline, raw_test, name="test")


# ── Tests: build_pipeline ─────────────────────────────────────────────────────

class TestBuildPipeline:

    def test_returns_column_transformer(self, raw_train):
        from sklearn.compose import ColumnTransformer
        ct = build_pipeline(raw_train)
        assert hasattr(ct, "fit")
        assert hasattr(ct, "transform")

    def test_has_num_and_cat_transformers(self, raw_train):
        ct = build_pipeline(raw_train)
        names = [t[0] for t in ct.transformers]
        assert "num" in names
        assert "cat" in names


# ── Tests: fit_pipeline ───────────────────────────────────────────────────────

class TestFitPipeline:

    def test_pipeline_is_fitted(self, fitted_pipeline):
        from sklearn.utils.validation import check_is_fitted
        # Should not raise
        check_is_fitted(fitted_pipeline)

    def test_feature_names_available(self, fitted_pipeline):
        names = fitted_pipeline.get_feature_names_out()
        assert len(names) > 0


# ── Tests: transform_split ────────────────────────────────────────────────────

class TestTransformSplit:

    def test_output_is_dataframe(self, train_transformed):
        assert isinstance(train_transformed, pd.DataFrame)

    def test_targets_preserved(self, train_transformed):
        for col in TARGET_COLS:
            assert col in train_transformed.columns

    def test_identifier_dropped(self, train_transformed):
        for col in IDENTIFIER_COLS:
            assert col not in train_transformed.columns

    def test_no_nan_in_features(self, train_transformed, raw_train):
        num_cols = get_numerical_cols(raw_train)
        for col in num_cols:
            if col in train_transformed.columns:
                assert train_transformed[col].isnull().sum() == 0, f"NaN in {col}"

    def test_numerical_features_scaled(self, train_transformed, raw_train):
        """After StandardScaler, each numerical column should have ~mean=0, std=1."""
        num_cols = get_numerical_cols(raw_train)
        for col in num_cols:
            if col in train_transformed.columns:
                mean = train_transformed[col].mean()
                std  = train_transformed[col].std()
                assert abs(mean) < 0.1, f"Column {col} mean={mean:.4f}, expected ~0"
                assert abs(std - 1.0) < 0.1, f"Column {col} std={std:.4f}, expected ~1"

    def test_categorical_columns_encoded(self, train_transformed):
        for col in CATEGORICAL_COLS:
            assert col in train_transformed.columns
            assert pd.api.types.is_numeric_dtype(train_transformed[col])

    def test_same_columns_train_test(self, train_transformed, test_transformed):
        assert list(train_transformed.columns) == list(test_transformed.columns)

    def test_row_count_preserved_train(self, train_transformed, raw_train):
        assert len(train_transformed) == len(raw_train)

    def test_row_count_preserved_test(self, test_transformed, raw_test):
        assert len(test_transformed) == len(raw_test)

    def test_target_values_unchanged(self, train_transformed, raw_train):
        """Targets must not be modified by the pipeline."""
        pd.testing.assert_series_equal(
            train_transformed["label"].reset_index(drop=True),
            raw_train["label"].reset_index(drop=True),
        )

    def test_unseen_categories_handled(self, fitted_pipeline, raw_train):
        """OrdinalEncoder should not crash on unseen category values."""
        df_new = raw_train.copy()
        df_new["proto"] = "UNSEEN_PROTOCOL"
        result = transform_split(fitted_pipeline, df_new, name="unseen_test")
        # Unseen values encoded as -1.0
        assert (result["proto"] == -1.0).all()


# ── Tests: pipeline persistence ───────────────────────────────────────────────

class TestPipelinePersistence:

    def test_save_and_load(self, fitted_pipeline, tmp_path):
        path = tmp_path / "pipeline.pkl"
        save_pipeline(fitted_pipeline, path)
        assert path.exists()
        loaded = load_pipeline(path)
        # Verify loaded pipeline produces the same output
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(loaded)

    def test_load_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pipeline(tmp_path / "missing.pkl")
