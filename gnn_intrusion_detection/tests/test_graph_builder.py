"""
Tests for src/graph_builder.py

Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CT_EDGE_FEATURES,
    GRAPH_K,
    GRAPH_WINDOW_SIZE,
    PREPROCESSOR_PATH,
    TARGET_COLS,
    TRAIN_RAW_PATH,
    TEST_RAW_PATH,
)
from src.graph_builder import (
    _window_to_graph,
    build_graphs,
    save_graphs,
    load_graphs,
)
from src.preprocessor import load_pipeline, transform_split


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    return load_pipeline(PREPROCESSOR_PATH)


@pytest.fixture(scope="module")
def small_train_proc(pipeline):
    """200 preprocessed training records."""
    df = pd.read_csv(TRAIN_RAW_PATH, nrows=200)
    df = df.sort_values("id").reset_index(drop=True)
    return transform_split(pipeline, df, name="test_fixture")


@pytest.fixture(scope="module")
def feature_cols(small_train_proc):
    return [c for c in small_train_proc.columns if c not in TARGET_COLS]


@pytest.fixture(scope="module")
def ct_indices(feature_cols):
    return [feature_cols.index(c) for c in CT_EDGE_FEATURES if c in feature_cols]


@pytest.fixture(scope="module")
def single_graph(small_train_proc, feature_cols, ct_indices):
    """One graph from a 100-row window."""
    window = small_train_proc.iloc[:100]
    return _window_to_graph(window, feature_cols, ct_indices, k=5)


# ── Tests: _window_to_graph ───────────────────────────────────────────────────

class TestWindowToGraph:

    def test_returns_pyg_data(self, single_graph):
        from torch_geometric.data import Data
        assert isinstance(single_graph, Data)

    def test_node_count(self, single_graph):
        assert single_graph.num_nodes == 100

    def test_feature_dim(self, single_graph):
        assert single_graph.x.shape == (100, 42)

    def test_feature_dtype(self, single_graph):
        assert single_graph.x.dtype == torch.float32

    def test_edge_index_shape(self, single_graph):
        # edge_index must be (2, E)
        assert single_graph.edge_index.shape[0] == 2

    def test_edge_index_dtype(self, single_graph):
        assert single_graph.edge_index.dtype == torch.long

    def test_edge_attr_shape(self, single_graph):
        E = single_graph.edge_index.shape[1]
        assert single_graph.edge_attr.shape == (E, 1)

    def test_edge_attr_range(self, single_graph):
        # Cosine similarity is in [-1, 1]; we clamped self-loops to -1 before
        # selecting top-k, so actual selected weights should be > -1 for most.
        assert (single_graph.edge_attr <= 1.0 + 1e-5).all()
        assert (single_graph.edge_attr >= -1.0 - 1e-5).all()

    def test_labels_y_present(self, single_graph):
        assert single_graph.y is not None
        assert single_graph.y.shape[0] == 100

    def test_labels_binary(self, single_graph):
        assert set(single_graph.y.numpy().tolist()).issubset({0, 1})

    def test_attack_cat_present(self, single_graph):
        assert single_graph.attack_cat is not None
        assert single_graph.attack_cat.shape[0] == 100

    def test_no_self_loops(self, single_graph):
        src = single_graph.edge_index[0]
        dst = single_graph.edge_index[1]
        assert (src != dst).all(), "Self-loops found in edge_index"

    def test_label_not_in_x(self, single_graph, small_train_proc, feature_cols):
        """Verify label columns are not part of node features."""
        assert "label" not in feature_cols
        assert "attack_cat" not in feature_cols

    def test_k_neighbours_per_node(self, small_train_proc, feature_cols, ct_indices):
        """Each node should have exactly k outgoing edges (for a full window)."""
        window = small_train_proc.iloc[:100]
        k = 5
        g = _window_to_graph(window, feature_cols, ct_indices, k=k)
        src_counts = torch.bincount(g.edge_index[0], minlength=100)
        # Every node should have exactly k outgoing edges
        assert (src_counts == k).all(), f"Not all nodes have exactly k={k} neighbours"

    def test_configurable_k(self, small_train_proc, feature_cols, ct_indices):
        for k in [3, 5, 10]:
            window = small_train_proc.iloc[:100]
            g = _window_to_graph(window, feature_cols, ct_indices, k=k)
            n_edges_expected = 100 * k
            assert g.edge_index.shape[1] == n_edges_expected, \
                f"k={k}: expected {n_edges_expected} edges, got {g.edge_index.shape[1]}"

    def test_partial_window(self, small_train_proc, feature_cols, ct_indices):
        """Final partial window should still produce a valid graph."""
        window = small_train_proc.iloc[:37]  # 37-node partial window
        g = _window_to_graph(window, feature_cols, ct_indices, k=5)
        assert g.num_nodes == 37
        assert g.x.shape == (37, 42)

    def test_no_leakage_edges_use_no_labels(
        self, small_train_proc, feature_cols, ct_indices
    ):
        """
        Construct graph from a modified DF where label is flipped.
        Edge structure should be identical — proving labels don't affect edges.
        """
        window_orig = small_train_proc.iloc[:100].copy()
        window_flip = window_orig.copy()
        window_flip["label"] = 1 - window_flip["label"]  # flip all labels

        g_orig = _window_to_graph(window_orig, feature_cols, ct_indices, k=5)
        g_flip = _window_to_graph(window_flip, feature_cols, ct_indices, k=5)

        # Edge structure must be identical regardless of label values
        assert torch.equal(g_orig.edge_index, g_flip.edge_index), \
            "Edge index changed when labels were flipped — leakage detected!"
        assert torch.allclose(g_orig.edge_attr, g_flip.edge_attr), \
            "Edge weights changed when labels were flipped — leakage detected!"


# ── Tests: build_graphs ───────────────────────────────────────────────────────

class TestBuildGraphs:

    @pytest.fixture(scope="class")
    def small_raw(self):
        return pd.read_csv(TRAIN_RAW_PATH, nrows=350)

    def test_graph_count_exact_multiple(self, small_raw):
        # 300 rows / window=100 = 3 graphs exactly
        graphs = build_graphs(small_raw.iloc[:300], "test", window_size=100, k=5)
        assert len(graphs) == 3

    def test_graph_count_with_remainder(self, small_raw):
        # 350 rows / window=100 = 3 full + 1 partial = 4 graphs
        graphs = build_graphs(small_raw, "test", window_size=100, k=5)
        assert len(graphs) == 4

    def test_final_partial_graph_size(self, small_raw):
        graphs = build_graphs(small_raw, "test", window_size=100, k=5)
        assert graphs[-1].num_nodes == 50  # 350 % 100 = 50

    def test_train_test_independence(self):
        """Training and testing graphs must be built independently."""
        train_raw = pd.read_csv(TRAIN_RAW_PATH, nrows=100)
        test_raw  = pd.read_csv(TEST_RAW_PATH,  nrows=100)
        train_g = build_graphs(train_raw, "train", window_size=100, k=5)
        test_g  = build_graphs(test_raw,  "test",  window_size=100, k=5)
        # Different data — feature matrices should not be equal
        assert not torch.equal(train_g[0].x, test_g[0].x), \
            "Train and test graphs produced identical features — check independence"

    def test_feature_dim_all_graphs(self, small_raw):
        graphs = build_graphs(small_raw, "test", window_size=100, k=5)
        for i, g in enumerate(graphs):
            assert g.x.shape[1] == 42, f"Graph {i}: wrong feature dim {g.x.shape[1]}"

    def test_ct_features_present(self):
        """All 8 ct_* features must be in the feature set."""
        pipeline = load_pipeline(PREPROCESSOR_PATH)
        feat_names = list(pipeline.get_feature_names_out())
        for col in CT_EDGE_FEATURES:
            assert col in feat_names, f"ct_* feature '{col}' missing from pipeline output"


# ── Tests: save and load ──────────────────────────────────────────────────────

class TestSaveLoadGraphs:

    def test_save_and_reload(self, tmp_path, small_train_proc, feature_cols, ct_indices):
        window = small_train_proc.iloc[:100]
        g = _window_to_graph(window, feature_cols, ct_indices, k=5)
        graphs = [g]

        save_graphs(graphs, tmp_path)
        loaded = load_graphs(tmp_path)

        assert len(loaded) == 1
        assert torch.equal(loaded[0].x, g.x)
        assert torch.equal(loaded[0].edge_index, g.edge_index)
        assert torch.equal(loaded[0].y, g.y)

    def test_load_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_graphs(tmp_path / "nonexistent")
