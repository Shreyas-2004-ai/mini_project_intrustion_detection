"""
Tests for src/inference.py

Run with:  python -m pytest tests/test_inference.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import (
    build_inference_graphs,
    load_graphsage,
    load_inference_csv,
    predict_csv,
    run_inference,
    summarise,
    DEFAULT_MODEL_PATH,
)
from src.config import TRAIN_RAW_PATH, TARGET_COLS, IDENTIFIER_COLS, CATEGORICAL_COLS
from src.gnn_config import GNN_INPUT_DIM


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_raw_csv(tmp_path_factory):
    """Write a small slice of the real training CSV to a temp file."""
    tmp = tmp_path_factory.mktemp("csv")
    df  = pd.read_csv(TRAIN_RAW_PATH, nrows=250)
    p   = tmp / "sample.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture(scope="module")
def csv_without_labels(tmp_path_factory):
    """CSV with label and attack_cat columns removed (simulates production upload)."""
    tmp = tmp_path_factory.mktemp("csv_no_labels")
    df  = pd.read_csv(TRAIN_RAW_PATH, nrows=150)
    df  = df.drop(columns=["label", "attack_cat"])
    p   = tmp / "no_labels.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture(scope="module")
def loaded_df(small_raw_csv):
    return load_inference_csv(small_raw_csv)


@pytest.fixture(scope="module")
def inference_graphs(loaded_df):
    graphs, order = build_inference_graphs(loaded_df, window_size=100, k=5)
    return graphs, order


@pytest.fixture(scope="module")
def model():
    return load_graphsage(DEFAULT_MODEL_PATH)


# ── Tests: load_inference_csv ─────────────────────────────────────────────────

class TestLoadInferenceCSV:

    def test_loads_real_csv(self, small_raw_csv):
        df = load_inference_csv(small_raw_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 250

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_inference_csv(tmp_path / "nonexistent.csv")

    def test_labels_injected_when_absent(self, csv_without_labels):
        df = load_inference_csv(csv_without_labels)
        assert "label"      in df.columns
        assert "attack_cat" in df.columns
        assert (df["label"] == 0).all()

    def test_labels_preserved_when_present(self, small_raw_csv):
        df = load_inference_csv(small_raw_csv)
        assert "label" in df.columns

    def test_id_column_present(self, small_raw_csv):
        df = load_inference_csv(small_raw_csv)
        assert "id" in df.columns

    def test_id_injected_when_absent(self, tmp_path):
        df_orig = pd.read_csv(TRAIN_RAW_PATH, nrows=50).drop(columns=["id"])
        p = tmp_path / "no_id.csv"
        df_orig.to_csv(p, index=False)
        df = load_inference_csv(p)
        assert "id" in df.columns
        assert len(df) == 50

    def test_raises_on_missing_feature_columns(self, tmp_path):
        df = pd.DataFrame({"id": [1, 2], "dur": [0.1, 0.2]})
        p  = tmp_path / "bad.csv"
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="Required feature columns missing"):
            load_inference_csv(p)


# ── Tests: build_inference_graphs ─────────────────────────────────────────────

class TestBuildInferenceGraphs:

    def test_returns_graphs_and_order(self, loaded_df):
        graphs, order = build_inference_graphs(loaded_df, window_size=100, k=5)
        assert isinstance(graphs, list)
        assert len(graphs) > 0
        assert len(order) == len(loaded_df)

    def test_graph_count_250_rows(self, loaded_df):
        # 250 rows / window=100 → 2 full + 1 partial = 3 graphs
        graphs, _ = build_inference_graphs(loaded_df, window_size=100, k=5)
        assert len(graphs) == 3

    def test_node_feature_dim(self, inference_graphs):
        graphs, _ = inference_graphs
        for g in graphs:
            assert g.x.shape[1] == GNN_INPUT_DIM

    def test_no_edge_leakage_uses_labels(self, loaded_df):
        """
        Graphs built from label-flipped input must have identical edge_index —
        proving labels are not used for edge construction.
        """
        df_flip = loaded_df.copy()
        df_flip["label"] = 1 - df_flip["label"].fillna(0).astype(int)
        graphs_orig, _ = build_inference_graphs(loaded_df,  window_size=100, k=5)
        graphs_flip, _ = build_inference_graphs(df_flip, window_size=100, k=5)
        for go, gf in zip(graphs_orig, graphs_flip):
            assert torch.equal(go.edge_index, gf.edge_index)

    def test_partial_final_window(self, loaded_df):
        graphs, _ = build_inference_graphs(loaded_df, window_size=100, k=5)
        assert graphs[-1].num_nodes == 50   # 250 % 100 = 50

    def test_works_without_labels_in_csv(self, csv_without_labels):
        df = load_inference_csv(csv_without_labels)
        graphs, _ = build_inference_graphs(df, window_size=100, k=5)
        assert len(graphs) > 0
        for g in graphs:
            assert g.x.shape[1] == GNN_INPUT_DIM

    def test_feature_count_matches_gnn_input_dim(self, inference_graphs):
        graphs, _ = inference_graphs
        for g in graphs:
            assert g.x.shape[1] == GNN_INPUT_DIM, \
                f"Expected {GNN_INPUT_DIM} features, got {g.x.shape[1]}"


# ── Tests: load_graphsage ─────────────────────────────────────────────────────

class TestLoadGraphSAGE:

    def test_loads_model(self, model):
        from src.models.graphsage import GraphSAGE
        assert isinstance(model, GraphSAGE)

    def test_model_in_eval_mode(self, model):
        assert not model.training

    def test_raises_on_missing_checkpoint(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_graphsage(tmp_path / "no_model.pt")

    def test_parameter_count(self, model):
        assert model.count_parameters() == 13826


# ── Tests: run_inference ──────────────────────────────────────────────────────

class TestRunInference:

    def test_returns_predictions_and_probs(self, model, inference_graphs):
        graphs, _ = inference_graphs
        preds, probs = run_inference(model, graphs, batch_size=4)
        total = sum(g.num_nodes for g in graphs)
        assert preds.shape == (total,)
        assert probs.shape == (total, 2)

    def test_predictions_are_binary(self, model, inference_graphs):
        graphs, _ = inference_graphs
        preds, _ = run_inference(model, graphs, batch_size=4)
        assert set(preds.tolist()).issubset({0, 1})

    def test_probabilities_sum_to_one(self, model, inference_graphs):
        graphs, _ = inference_graphs
        _, probs = run_inference(model, graphs, batch_size=4)
        sums = probs.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_probabilities_in_range(self, model, inference_graphs):
        graphs, _ = inference_graphs
        _, probs = run_inference(model, graphs, batch_size=4)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_model_stays_in_eval_after_inference(self, model, inference_graphs):
        graphs, _ = inference_graphs
        run_inference(model, graphs, batch_size=4)
        assert not model.training


# ── Tests: summarise ──────────────────────────────────────────────────────────

class TestSummarise:

    def test_all_normal(self):
        s = summarise(np.zeros(100, dtype=int))
        assert s["total_flows"]  == 100
        assert s["normal_flows"] == 100
        assert s["attack_flows"] == 0
        assert s["attack_pct"]   == 0.0

    def test_all_attack(self):
        s = summarise(np.ones(100, dtype=int))
        assert s["total_flows"]  == 100
        assert s["normal_flows"] == 0
        assert s["attack_flows"] == 100
        assert s["attack_pct"]   == 100.0

    def test_mixed(self):
        preds = np.array([0, 0, 1, 1, 1])
        s = summarise(preds)
        assert s["total_flows"]  == 5
        assert s["normal_flows"] == 2
        assert s["attack_flows"] == 3
        assert s["attack_pct"]   == 60.0

    def test_counts_sum_to_total(self):
        preds = np.random.randint(0, 2, size=500)
        s = summarise(preds)
        assert s["normal_flows"] + s["attack_flows"] == s["total_flows"]


# ── Tests: predict_csv (end-to-end) ───────────────────────────────────────────

class TestPredictCSV:

    def test_returns_all_keys(self, small_raw_csv):
        result = predict_csv(small_raw_csv)
        for key in ["predictions", "probabilities", "summary", "n_graphs", "n_flows"]:
            assert key in result

    def test_n_flows_matches_csv(self, small_raw_csv):
        result = predict_csv(small_raw_csv)
        assert result["n_flows"] == 250

    def test_prediction_length_matches_flows(self, small_raw_csv):
        result = predict_csv(small_raw_csv)
        assert len(result["predictions"]) == result["n_flows"]

    def test_probability_length_matches_flows(self, small_raw_csv):
        result = predict_csv(small_raw_csv)
        assert result["probabilities"].shape == (result["n_flows"], 2)

    def test_summary_counts_consistent(self, small_raw_csv):
        result = predict_csv(small_raw_csv)
        s = result["summary"]
        assert s["normal_flows"] + s["attack_flows"] == s["total_flows"]

    def test_works_without_labels(self, csv_without_labels):
        result = predict_csv(csv_without_labels)
        assert result["n_flows"] == 150
        assert len(result["predictions"]) == 150
