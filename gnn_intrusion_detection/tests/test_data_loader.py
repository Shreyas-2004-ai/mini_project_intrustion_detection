"""
Tests for src/data_loader.py

Run with:  python -m pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (
    load_dataset,
    get_numerical_cols,
    get_feature_cols,
)
from src.config import (
    TRAIN_RAW_PATH,
    TEST_RAW_PATH,
    TARGET_COLS,
    CATEGORICAL_COLS,
    IDENTIFIER_COLS,
    EXPECTED_COLS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def train_df():
    """Load a small slice of the real training CSV."""
    return pd.read_csv(TRAIN_RAW_PATH, nrows=500)


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv(TEST_RAW_PATH, nrows=200)


def _make_minimal_csv(tmp_path: Path, rows: int = 10) -> Path:
    """Create a minimal valid CSV with all required columns."""
    cols = {
        "id": range(rows),
        "dur": np.random.rand(rows),
        "proto": ["tcp"] * rows,
        "service": ["-"] * rows,
        "state": ["FIN"] * rows,
        "spkts": np.random.randint(1, 10, rows),
        "dpkts": np.random.randint(1, 10, rows),
        "sbytes": np.random.randint(100, 1000, rows),
        "dbytes": np.random.randint(100, 1000, rows),
        "rate": np.random.rand(rows) * 100,
        "sttl": np.random.randint(50, 255, rows),
        "dttl": np.random.randint(50, 255, rows),
        "sload": np.random.rand(rows) * 1000,
        "dload": np.random.rand(rows) * 1000,
        "sloss": np.zeros(rows, int),
        "dloss": np.zeros(rows, int),
        "sinpkt": np.random.rand(rows) * 50,
        "dinpkt": np.random.rand(rows) * 50,
        "sjit": np.random.rand(rows) * 20,
        "djit": np.random.rand(rows) * 20,
        "swin": [255] * rows,
        "stcpb": np.random.randint(0, 2**31, rows),
        "dtcpb": np.random.randint(0, 2**31, rows),
        "dwin": [255] * rows,
        "tcprtt": np.zeros(rows),
        "synack": np.zeros(rows),
        "ackdat": np.zeros(rows),
        "smean": np.random.randint(40, 100, rows),
        "dmean": np.random.randint(40, 100, rows),
        "trans_depth": np.zeros(rows, int),
        "response_body_len": np.zeros(rows, int),
        "ct_srv_src": np.ones(rows, int),
        "ct_state_ttl": np.zeros(rows, int),
        "ct_dst_ltm": np.ones(rows, int),
        "ct_src_dport_ltm": np.ones(rows, int),
        "ct_dst_sport_ltm": np.ones(rows, int),
        "ct_dst_src_ltm": np.ones(rows, int),
        "is_ftp_login": np.zeros(rows, int),
        "ct_ftp_cmd": np.zeros(rows, int),
        "ct_flw_http_mthd": np.zeros(rows, int),
        "ct_src_ltm": np.ones(rows, int),
        "ct_srv_dst": np.ones(rows, int),
        "is_sm_ips_ports": np.zeros(rows, int),
        "attack_cat": ["Normal"] * rows,
        "label": np.zeros(rows, int),
    }
    df = pd.DataFrame(cols)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ── Tests: load_dataset ───────────────────────────────────────────────────────

class TestLoadDataset:

    def test_loads_real_train_file(self, train_df):
        assert isinstance(train_df, pd.DataFrame)
        assert len(train_df) == 500

    def test_loads_real_test_file(self, test_df):
        assert isinstance(test_df, pd.DataFrame)
        assert len(test_df) == 200

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nonexistent.csv")

    def test_loads_minimal_csv(self, tmp_path):
        csv_path = _make_minimal_csv(tmp_path, rows=20)
        df = load_dataset(csv_path)
        assert df.shape == (20, EXPECTED_COLS)

    def test_raises_on_missing_required_columns(self, tmp_path):
        # CSV without target columns
        df = pd.DataFrame({"id": [1], "dur": [0.1]})
        bad_csv = tmp_path / "bad.csv"
        df.to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Required columns missing"):
            load_dataset(bad_csv)

    def test_correct_column_count(self, train_df):
        assert train_df.shape[1] == EXPECTED_COLS

    def test_target_columns_present(self, train_df):
        for col in TARGET_COLS:
            assert col in train_df.columns

    def test_categorical_columns_present(self, train_df):
        for col in CATEGORICAL_COLS:
            assert col in train_df.columns

    def test_identifier_columns_present(self, train_df):
        for col in IDENTIFIER_COLS:
            assert col in train_df.columns


# ── Tests: get_numerical_cols ─────────────────────────────────────────────────

class TestGetNumericalCols:

    def test_excludes_targets(self, train_df):
        num = get_numerical_cols(train_df)
        for col in TARGET_COLS:
            assert col not in num

    def test_excludes_identifiers(self, train_df):
        num = get_numerical_cols(train_df)
        for col in IDENTIFIER_COLS:
            assert col not in num

    def test_excludes_categoricals(self, train_df):
        num = get_numerical_cols(train_df)
        for col in CATEGORICAL_COLS:
            assert col not in num

    def test_returns_only_numeric_dtypes(self, train_df):
        num = get_numerical_cols(train_df)
        for col in num:
            assert pd.api.types.is_numeric_dtype(train_df[col]), f"{col} is not numeric"

    def test_consistent_between_train_and_test(self, train_df, test_df):
        assert get_numerical_cols(train_df) == get_numerical_cols(test_df)


# ── Tests: get_feature_cols ───────────────────────────────────────────────────

class TestGetFeatureCols:

    def test_returns_four_role_keys(self, train_df):
        roles = get_feature_cols(train_df)
        assert set(roles.keys()) == {"identifier", "target", "categorical", "numerical"}

    def test_no_overlap_between_roles(self, train_df):
        roles = get_feature_cols(train_df)
        all_cols = []
        for lst in roles.values():
            all_cols.extend(lst)
        # No column should appear in two categories
        assert len(all_cols) == len(set(all_cols)), "Duplicate columns across roles"

    def test_union_covers_all_columns(self, train_df):
        roles = get_feature_cols(train_df)
        accounted = set()
        for lst in roles.values():
            accounted.update(lst)
        assert accounted == set(train_df.columns)
