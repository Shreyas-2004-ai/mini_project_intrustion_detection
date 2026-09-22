"""
Tests for src/models/random_forest.py

Run with:  python -m pytest tests/test_random_forest.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.random_forest import (
    RF_DEFAULTS,
    evaluate_rf,
    load_rf,
    save_rf,
    train_rf,
)
from src.config import TRAIN_RAW_PATH, TEST_RAW_PATH, TARGET_COLS, IDENTIFIER_COLS
from src.preprocessor import load_pipeline, transform_split
from src.data_loader import load_dataset


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    return load_pipeline()


@pytest.fixture(scope="module")
def small_train(pipeline):
    """200 rows from the training CSV, preprocessed."""
    import pandas as pd
    raw = pd.read_csv(TRAIN_RAW_PATH, nrows=200)
    proc = transform_split(pipeline, raw, name="test_train")
    feat = [c for c in proc.columns if c not in set(TARGET_COLS + IDENTIFIER_COLS)]
    X = proc[feat].values.astype("float32")
    y = proc["label"].values.astype("int32")
    return X, y, feat


@pytest.fixture(scope="module")
def small_test(pipeline):
    """Rows from the testing CSV containing both classes, preprocessed."""
    import pandas as pd
    raw = pd.read_csv(TEST_RAW_PATH, nrows=500)
    # Ensure both classes are present
    proc = transform_split(pipeline, raw, name="test_test")
    feat = [c for c in proc.columns if c not in set(TARGET_COLS + IDENTIFIER_COLS)]
    X = proc[feat].values.astype("float32")
    y = proc["label"].values.astype("int32")
    assert set(y.tolist()).issuperset({0, 1}), "Test fixture must contain both classes"
    return X, y, feat


@pytest.fixture(scope="module")
def fitted_clf(small_train):
    X, y, _ = small_train
    cfg = dict(RF_DEFAULTS)
    cfg["n_estimators"] = 10   # fast for tests
    return train_rf(X, y, cfg)


# ── Tests: feature integrity ───────────────────────────────────────────────────

class TestFeatureIntegrity:

    def test_feature_count_is_42(self, small_train):
        _, _, feat = small_train
        assert len(feat) == 42, f"Expected 42 features, got {len(feat)}"

    def test_no_label_in_features(self, small_train):
        _, _, feat = small_train
        assert "label" not in feat

    def test_no_attack_cat_in_features(self, small_train):
        _, _, feat = small_train
        assert "attack_cat" not in feat

    def test_no_id_in_features(self, small_train):
        _, _, feat = small_train
        assert "id" not in feat

    def test_train_test_features_identical(self, small_train, small_test):
        _, _, train_feat = small_train
        _, _, test_feat  = small_test
        assert train_feat == test_feat, "Feature columns differ between train and test"

    def test_features_match_pipeline_output(self, pipeline, small_train):
        """RF feature names must be identical to the pipeline's output columns."""
        pipeline_names = list(pipeline.get_feature_names_out())
        _, _, feat = small_train
        assert feat == pipeline_names

    def test_x_shape(self, small_train):
        X, y, _ = small_train
        assert X.shape == (200, 42)
        assert y.shape == (200,)

    def test_y_is_binary(self, small_train):
        _, y, _ = small_train
        assert set(y.tolist()).issubset({0, 1})


# ── Tests: train_rf ────────────────────────────────────────────────────────────

class TestTrainRF:

    def test_returns_classifier(self, fitted_clf):
        assert isinstance(fitted_clf, RandomForestClassifier)

    def test_is_fitted(self, fitted_clf):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(fitted_clf)

    def test_n_estimators(self, small_train):
        X, y, _ = small_train
        cfg = dict(RF_DEFAULTS)
        cfg["n_estimators"] = 5
        clf = train_rf(X, y, cfg)
        assert len(clf.estimators_) == 5

    def test_seed_reproducibility(self, small_train):
        """Same seed must produce same predictions."""
        X, y, _ = small_train
        cfg = dict(RF_DEFAULTS)
        cfg["n_estimators"] = 5
        clf1 = train_rf(X, y, cfg)
        clf2 = train_rf(X, y, cfg)
        assert (clf1.predict(X) == clf2.predict(X)).all()

    def test_class_weight_balanced(self, fitted_clf):
        assert fitted_clf.class_weight == "balanced"


# ── Tests: evaluate_rf ─────────────────────────────────────────────────────────

class TestEvaluateRF:

    def test_returns_required_keys(self, fitted_clf, small_test):
        X, y, _ = small_test
        m = evaluate_rf(fitted_clf, X, y)
        for key in ["accuracy", "precision", "recall", "f1",
                    "confusion_matrix", "classification_report",
                    "total_nodes", "n_normal", "n_attack"]:
            assert key in m

    def test_node_counts_correct(self, fitted_clf, small_test):
        X, y, _ = small_test
        m = evaluate_rf(fitted_clf, X, y)
        assert m["total_nodes"] == len(y)
        assert m["n_normal"] + m["n_attack"] == len(y)

    def test_accuracy_in_range(self, fitted_clf, small_test):
        X, y, _ = small_test
        m = evaluate_rf(fitted_clf, X, y)
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_confusion_matrix_shape(self, fitted_clf, small_test):
        X, y, _ = small_test
        m = evaluate_rf(fitted_clf, X, y)
        cm = m["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2

    def test_confusion_matrix_sums_to_total(self, fitted_clf, small_test):
        X, y, _ = small_test
        m = evaluate_rf(fitted_clf, X, y)
        cm = m["confusion_matrix"]
        total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
        assert total == len(y)

    def test_test_data_excluded_from_training(self, small_train, small_test):
        """Train and test X must not be identical (different raw CSV rows)."""
        X_tr, _, _ = small_train
        X_te, _, _ = small_test
        # They may have different row counts — just check no full-matrix equality
        if X_tr.shape == X_te.shape:
            assert not (X_tr == X_te).all()


# ── Tests: save / load ─────────────────────────────────────────────────────────

class TestPersistence:

    def test_save_creates_file(self, fitted_clf, tmp_path):
        path = save_rf(fitted_clf, tmp_path)
        assert path.exists()
        assert path.suffix == ".pkl"

    def test_load_roundtrip(self, fitted_clf, tmp_path, small_test):
        save_rf(fitted_clf, tmp_path)
        loaded = load_rf(tmp_path / "random_forest.pkl")
        X, y, _ = small_test
        p1 = fitted_clf.predict(X)
        p2 = loaded.predict(X)
        assert (p1 == p2).all()

    def test_load_raises_on_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_rf(tmp_path / "nonexistent.pkl")
