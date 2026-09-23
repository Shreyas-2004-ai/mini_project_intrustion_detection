"""
Tests for src/graph_validator.py

Run with:  python -m pytest tests/ -v
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph_validator import (
    graph_stats,
    dataset_stats,
    aggregate_stats,
    validate_no_leakage,
    save_stats,
)
from src.config import CT_EDGE_FEATURES, TARGET_COLS, TRAIN_RAW_PATH, PREPROCESSOR_PATH
from src.preprocessor import load_pipeline, transform_split
from src.graph_builder import _window_to_graph


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_graphs():
    """Three graphs built from real data for validator tests."""
    pipeline = load_pipeline(PREPROCESSOR_PATH)
    df = pd.read_csv(TRAIN_RAW_PATH, nrows=300)
    df = df.sort_values("id").reset_index(drop=True)
    proc = transform_split(pipeline, df, name="validator_fixture")
    feature_cols = [c for c in proc.columns if c not in TARGET_COLS]
    ct_indices   = [feature_cols.index(c) for c in CT_EDGE_FEATURES if c in feature_cols]
    graphs = []
    for i in range(3):
        window = proc.iloc[i*100:(i+1)*100]
        graphs.append(_window_to_graph(window, feature_cols, ct_indices, k=5))
    return graphs


@pytest.fixture(scope="module")
def single_graph(sample_graphs):
    return sample_graphs[0]


# ── Tests: graph_stats ────────────────────────────────────────────────────────

class TestGraphStats:

    def test_returns_dict(self, single_graph):
        s = graph_stats(single_graph, 0)
        assert isinstance(s, dict)

    def test_required_keys(self, single_graph):
        s = graph_stats(single_graph, 0)
        expected_keys = {
            "graph_id", "n_nodes", "n_edges", "avg_degree",
            "min_degree", "max_degree", "n_isolated",
            "ew_min", "ew_max", "ew_mean",
            "n_attack", "n_normal",
        }
        assert expected_keys.issubset(s.keys())

    def test_n_nodes_correct(self, single_graph):
        s = graph_stats(single_graph, 0)
        assert s["n_nodes"] == 100

    def test_n_edges_positive(self, single_graph):
        s = graph_stats(single_graph, 0)
        assert s["n_edges"] > 0

    def test_avg_degree_positive(self, single_graph):
        s = graph_stats(single_graph, 0)
        assert s["avg_degree"] > 0

    def test_attack_plus_normal_equals_nodes(self, single_graph):
        s = graph_stats(single_graph, 0)
        assert s["n_attack"] + s["n_normal"] == s["n_nodes"]

    def test_ew_min_leq_ew_max(self, single_graph):
        s = graph_stats(single_graph, 0)
        assert s["ew_min"] <= s["ew_max"]

    def test_edge_weights_in_cosine_range(self, single_graph):
        s = graph_stats(single_graph, 0)
        assert s["ew_min"] >= -1.0 - 1e-5
        assert s["ew_max"] <=  1.0 + 1e-5

    def test_n_isolated_non_negative(self, single_graph):
        s = graph_stats(single_graph, 0)
        assert s["n_isolated"] >= 0


# ── Tests: dataset_stats ──────────────────────────────────────────────────────

class TestDatasetStats:

    def test_returns_dataframe(self, sample_graphs):
        df = dataset_stats(sample_graphs, "train")
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self, sample_graphs):
        df = dataset_stats(sample_graphs, "train")
        assert len(df) == len(sample_graphs)

    def test_graph_ids_sequential(self, sample_graphs):
        df = dataset_stats(sample_graphs, "train")
        assert list(df["graph_id"]) == list(range(len(sample_graphs)))


# ── Tests: aggregate_stats ────────────────────────────────────────────────────

class TestAggregateStats:

    def test_returns_dict(self, sample_graphs):
        df  = dataset_stats(sample_graphs, "train")
        agg = aggregate_stats(df, "train")
        assert isinstance(agg, dict)

    def test_n_graphs_correct(self, sample_graphs):
        df  = dataset_stats(sample_graphs, "train")
        agg = aggregate_stats(df, "train")
        assert agg["n_graphs"] == len(sample_graphs)

    def test_total_nodes(self, sample_graphs):
        df  = dataset_stats(sample_graphs, "train")
        agg = aggregate_stats(df, "train")
        assert agg["total_nodes"] == sum(g.num_nodes for g in sample_graphs)

    def test_attack_pct_in_range(self, sample_graphs):
        df  = dataset_stats(sample_graphs, "train")
        agg = aggregate_stats(df, "train")
        assert 0.0 <= agg["attack_node_pct"] <= 100.0


# ── Tests: validate_no_leakage ────────────────────────────────────────────────

class TestValidateNoLeakage:

    def test_passes_on_valid_graphs(self, sample_graphs):
        assert validate_no_leakage(sample_graphs, []) is True

    def test_fails_on_wrong_feature_dim(self):
        # Graph with 43 features (wrong) — should raise
        bad_graph = Data(
            x=torch.zeros(10, 43),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, 1),
            y=torch.zeros(10, dtype=torch.long),
            attack_cat=torch.zeros(10, dtype=torch.long),
        )
        with pytest.raises(ValueError, match="42 node features"):
            validate_no_leakage([bad_graph], [])

    def test_fails_on_missing_y(self):
        bad_graph = Data(
            x=torch.zeros(10, 42),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, 1),
            y=None,
            attack_cat=torch.zeros(10, dtype=torch.long),
        )
        with pytest.raises(ValueError, match="missing node labels"):
            validate_no_leakage([bad_graph], [])


# ── Tests: save_stats ─────────────────────────────────────────────────────────

class TestSaveStats:

    def test_csv_files_created(self, tmp_path, sample_graphs, monkeypatch):
        import src.graph_validator as gv
        train_path = tmp_path / "train_stats.csv"
        test_path  = tmp_path / "test_stats.csv"
        overall    = tmp_path / "overall.json"

        monkeypatch.setattr(gv, "TRAIN_GRAPH_STATS_PATH", train_path)
        monkeypatch.setattr(gv, "TEST_GRAPH_STATS_PATH",  test_path)
        monkeypatch.setattr(gv, "OVERALL_STATS_PATH",     overall)

        df = dataset_stats(sample_graphs, "train")
        gv.save_stats(df, df)

        assert train_path.exists()
        assert test_path.exists()
        assert overall.exists()

    def test_json_has_train_and_test_keys(self, tmp_path, sample_graphs, monkeypatch):
        import src.graph_validator as gv
        overall = tmp_path / "overall.json"
        monkeypatch.setattr(gv, "TRAIN_GRAPH_STATS_PATH", tmp_path / "t.csv")
        monkeypatch.setattr(gv, "TEST_GRAPH_STATS_PATH",  tmp_path / "t2.csv")
        monkeypatch.setattr(gv, "OVERALL_STATS_PATH",     overall)

        df = dataset_stats(sample_graphs, "train")
        gv.save_stats(df, df)

        with open(overall) as f:
            data = json.load(f)
        assert "train" in data
        assert "test" in data
