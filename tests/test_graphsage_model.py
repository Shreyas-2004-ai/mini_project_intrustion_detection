"""
Tests for src/models/graphsage.py and src/training.py / src/evaluation.py

Run with:  python -m pytest tests/test_graphsage_model.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
import numpy as np
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.graphsage import GraphSAGE
from src.gnn_config import GNN_INPUT_DIM, GNN_DEFAULTS
from src.training import compute_class_weights, set_seed, _run_epoch
from src.evaluation import compute_metrics, predict_all


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_graph(n_nodes: int = 20, n_attack: int = 10) -> Data:
    """Create a small synthetic graph for unit tests."""
    x          = torch.randn(n_nodes, GNN_INPUT_DIM)
    # Simple ring edges
    src        = torch.arange(n_nodes)
    dst        = (torch.arange(n_nodes) + 1) % n_nodes
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr  = torch.ones(n_nodes, 1)
    y          = torch.zeros(n_nodes, dtype=torch.long)
    y[:n_attack] = 1
    attack_cat = torch.zeros(n_nodes, dtype=torch.long)
    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        y=y, attack_cat=attack_cat, num_nodes=n_nodes,
    )


@pytest.fixture
def small_graph():
    return _make_graph(n_nodes=20, n_attack=10)


@pytest.fixture
def graph_list():
    return [_make_graph(n_nodes=20, n_attack=10) for _ in range(8)]


@pytest.fixture
def default_model():
    return GraphSAGE(
        in_channels=GNN_INPUT_DIM,
        hidden_dim=32,
        num_classes=2,
        num_layers=2,
        dropout=0.0,   # disable dropout for deterministic tests
    )


# ── Tests: GraphSAGE model ────────────────────────────────────────────────────

class TestGraphSAGEModel:

    def test_output_shape(self, default_model, small_graph):
        logits = default_model(small_graph.x, small_graph.edge_index)
        assert logits.shape == (20, 2)

    def test_output_dtype(self, default_model, small_graph):
        logits = default_model(small_graph.x, small_graph.edge_index)
        assert logits.dtype == torch.float32

    def test_predict_shape(self, default_model, small_graph):
        preds = default_model.predict(small_graph.x, small_graph.edge_index)
        assert preds.shape == (20,)

    def test_predict_binary(self, default_model, small_graph):
        preds = default_model.predict(small_graph.x, small_graph.edge_index)
        assert set(preds.tolist()).issubset({0, 1})

    def test_predict_proba_sums_to_one(self, default_model, small_graph):
        proba = default_model.predict_proba(small_graph.x, small_graph.edge_index)
        assert proba.shape == (20, 2)
        sums = proba.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(20), atol=1e-5)

    def test_configurable_hidden_dim(self, small_graph):
        for hdim in [16, 64, 128]:
            m = GraphSAGE(GNN_INPUT_DIM, hidden_dim=hdim)
            logits = m(small_graph.x, small_graph.edge_index)
            assert logits.shape == (20, 2)

    def test_configurable_num_layers(self, small_graph):
        for nl in [1, 2, 3]:
            m = GraphSAGE(GNN_INPUT_DIM, num_layers=nl, dropout=0.0)
            logits = m(small_graph.x, small_graph.edge_index)
            assert logits.shape == (20, 2)

    def test_invalid_num_layers_raises(self):
        with pytest.raises(ValueError):
            GraphSAGE(GNN_INPUT_DIM, num_layers=0)

    def test_parameter_count_positive(self, default_model):
        assert default_model.count_parameters() > 0

    def test_train_eval_mode_difference(self, small_graph):
        """Dropout should cause different outputs in train vs eval mode."""
        set_seed(0)
        model = GraphSAGE(GNN_INPUT_DIM, hidden_dim=32, dropout=0.9)
        model.train()
        out_train = model(small_graph.x, small_graph.edge_index).detach()
        model.eval()
        with torch.no_grad():
            out_eval = model(small_graph.x, small_graph.edge_index)
        # With dropout=0.9 outputs should differ between train and eval
        assert not torch.equal(out_train, out_eval)

    def test_x_dimension_matches_input(self, small_graph):
        """Model must accept exactly GNN_INPUT_DIM features."""
        model = GraphSAGE(in_channels=GNN_INPUT_DIM)
        out = model(small_graph.x, small_graph.edge_index)
        assert out.shape[0] == small_graph.num_nodes

    def test_label_not_in_features(self, small_graph):
        """Sanity: y is NOT part of x."""
        assert small_graph.x.shape[1] == GNN_INPUT_DIM
        # label column count would make it 43
        assert small_graph.x.shape[1] != 43

    def test_attack_cat_not_in_features(self, small_graph):
        """Sanity: attack_cat is a separate attribute, not inside x."""
        assert hasattr(small_graph, "attack_cat")
        assert small_graph.x.shape[1] == GNN_INPUT_DIM  # still 42, not 43


# ── Tests: class weights ──────────────────────────────────────────────────────

class TestClassWeights:

    def test_returns_tensor_of_two(self, graph_list):
        device = torch.device("cpu")
        w = compute_class_weights(graph_list, device)
        assert w.shape == (2,)

    def test_weights_positive(self, graph_list):
        w = compute_class_weights(graph_list, torch.device("cpu"))
        assert (w > 0).all()

    def test_majority_class_lower_weight(self, graph_list):
        """Attack (majority) should receive lower weight than Normal."""
        w = compute_class_weights(graph_list, torch.device("cpu"))
        # graph_list has equal attack/normal so weights should be equal
        # Use imbalanced fixture to test direction
        imb = [_make_graph(20, 18) for _ in range(4)]  # 18/20 attack
        w_imb = compute_class_weights(imb, torch.device("cpu"))
        assert w_imb[0] > w_imb[1], "Normal (minority) should have higher weight"


# ── Tests: compute_metrics ────────────────────────────────────────────────────

class TestComputeMetrics:

    def test_perfect_prediction(self):
        labels = np.array([0, 0, 1, 1, 1])
        preds  = np.array([0, 0, 1, 1, 1])
        m = compute_metrics(preds, labels)
        assert m["accuracy"]  == 1.0
        assert m["precision"] == 1.0
        assert m["recall"]    == 1.0
        assert m["f1"]        == 1.0

    def test_all_wrong(self):
        labels = np.array([0, 0, 1, 1])
        preds  = np.array([1, 1, 0, 0])
        m = compute_metrics(preds, labels)
        assert m["accuracy"] == 0.0

    def test_required_keys(self):
        labels = np.array([0, 1, 1, 0])
        preds  = np.array([0, 1, 0, 0])
        m = compute_metrics(preds, labels)
        for key in ["accuracy", "precision", "recall", "f1",
                    "confusion_matrix", "classification_report",
                    "total_nodes", "n_normal", "n_attack"]:
            assert key in m

    def test_node_counts(self):
        labels = np.array([0, 0, 0, 1, 1])
        preds  = np.array([0, 0, 1, 1, 1])
        m = compute_metrics(preds, labels)
        assert m["n_normal"] == 3
        assert m["n_attack"] == 2
        assert m["total_nodes"] == 5

    def test_confusion_matrix_shape(self):
        labels = np.array([0, 1, 1, 0])
        preds  = np.array([0, 1, 0, 0])
        m = compute_metrics(preds, labels)
        assert len(m["confusion_matrix"]) == 2
        assert len(m["confusion_matrix"][0]) == 2


# ── Tests: predict_all ────────────────────────────────────────────────────────

class TestPredictAll:

    def test_returns_correct_length(self, default_model, graph_list):
        device = torch.device("cpu")
        preds, labels = predict_all(default_model, graph_list, device, batch_size=4)
        total_nodes = sum(g.num_nodes for g in graph_list)
        assert len(preds) == total_nodes
        assert len(labels) == total_nodes

    def test_predictions_are_binary(self, default_model, graph_list):
        device = torch.device("cpu")
        preds, _ = predict_all(default_model, graph_list, device, batch_size=4)
        assert set(preds.tolist()).issubset({0, 1})

    def test_model_in_eval_mode_after(self, default_model, graph_list):
        device = torch.device("cpu")
        predict_all(default_model, graph_list, device, batch_size=4)
        assert not default_model.training   # eval() should be active


# ── Tests: reproducibility ────────────────────────────────────────────────────

class TestReproducibility:

    def test_same_seed_same_output(self, small_graph):
        """Two models initialised with the same seed must produce identical output."""
        set_seed(42)
        m1 = GraphSAGE(GNN_INPUT_DIM, hidden_dim=32, dropout=0.0)
        set_seed(42)
        m2 = GraphSAGE(GNN_INPUT_DIM, hidden_dim=32, dropout=0.0)
        o1 = m1(small_graph.x, small_graph.edge_index)
        o2 = m2(small_graph.x, small_graph.edge_index)
        assert torch.allclose(o1, o2)
