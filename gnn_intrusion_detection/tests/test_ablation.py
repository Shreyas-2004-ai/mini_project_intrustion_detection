"""
Tests for src/ablation.py — randomise_edges() and validate_random_graph().

Run with:  python -m pytest tests/test_ablation.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ablation import randomise_edges, strip_edges, validate_random_graph


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_knn_graph(n: int = 20, k: int = 5, seed: int = 0) -> Data:
    """
    Synthetic k-NN-style graph: every node has exactly k outgoing edges,
    no self-loops, no duplicates — mirrors the real UNSW-NB15 graph structure.
    """
    rng = np.random.default_rng(seed)
    src_list, dst_list = [], []
    for i in range(n):
        candidates = [j for j in range(n) if j != i]
        chosen = rng.choice(candidates, size=k, replace=False).tolist()
        src_list.extend([i] * k)
        dst_list.extend(chosen)

    ei = torch.tensor([src_list, dst_list], dtype=torch.long)
    ea = torch.ones(len(src_list), 1)
    x  = torch.randn(n, 42)
    y  = torch.randint(0, 2, (n,), dtype=torch.long)
    ac = torch.zeros(n, dtype=torch.long)
    return Data(x=x, edge_index=ei, edge_attr=ea, y=y, attack_cat=ac, num_nodes=n)


def _make_irregular_graph(n: int = 15) -> Data:
    """Graph where nodes have different out-degrees (not uniform k-NN)."""
    edges = [(0,1),(0,2),(0,3),(1,4),(1,5),(2,6),(3,7),(3,8),(4,9),(5,10)]
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    ei  = torch.tensor([src, dst], dtype=torch.long)
    ea  = torch.ones(len(edges), 1)
    x   = torch.randn(n, 42)
    y   = torch.zeros(n, dtype=torch.long)
    ac  = torch.zeros(n, dtype=torch.long)
    return Data(x=x, edge_index=ei, edge_attr=ea, y=y, attack_cat=ac, num_nodes=n)


@pytest.fixture
def knn_graph():
    return _make_knn_graph(n=20, k=5)


@pytest.fixture
def knn_graphs():
    return [_make_knn_graph(n=20, k=5, seed=i) for i in range(10)]


@pytest.fixture
def rand_graphs(knn_graphs):
    return randomise_edges(knn_graphs, seed=42)


# ── Core structural properties ────────────────────────────────────────────────

class TestRandomiseEdgesStructure:

    def test_edge_count_preserved(self, knn_graphs, rand_graphs):
        for orig, rand in zip(knn_graphs, rand_graphs):
            assert orig.edge_index.shape[1] == rand.edge_index.shape[1], \
                "Edge count not preserved"

    def test_no_self_loops(self, rand_graphs):
        for i, g in enumerate(rand_graphs):
            src, dst = g.edge_index
            n_self = (src == dst).sum().item()
            assert n_self == 0, f"Graph {i}: {n_self} self-loop(s) found"

    def test_no_duplicate_edges(self, rand_graphs):
        for i, g in enumerate(rand_graphs):
            src, dst = g.edge_index
            pairs = list(zip(src.tolist(), dst.tolist()))
            n_dup = len(pairs) - len(set(pairs))
            assert n_dup == 0, f"Graph {i}: {n_dup} duplicate edge(s) found"

    def test_out_degree_sequence_preserved(self, knn_graphs, rand_graphs):
        for i, (orig, rand) in enumerate(zip(knn_graphs, rand_graphs)):
            n = orig.num_nodes
            orig_deg = np.bincount(orig.edge_index[0].numpy(), minlength=n)
            rand_deg = np.bincount(rand.edge_index[0].numpy(), minlength=n)
            assert np.array_equal(orig_deg, rand_deg), \
                f"Graph {i}: out-degree sequence differs"

    def test_uniform_knn_degree_preserved(self, knn_graphs, rand_graphs):
        """For uniform k=5 graphs, every node must still have out-degree 5."""
        k = 5
        for i, g in enumerate(rand_graphs):
            n = g.num_nodes
            deg = np.bincount(g.edge_index[0].numpy(), minlength=n)
            assert (deg == k).all(), \
                f"Graph {i}: not all nodes have degree {k}. Degrees: {deg}"

    def test_destinations_are_randomised(self, knn_graphs, rand_graphs):
        """Destinations must differ from the original — rewiring actually happened."""
        changed = 0
        for orig, rand in zip(knn_graphs, rand_graphs):
            if not torch.equal(orig.edge_index[1], rand.edge_index[1]):
                changed += 1
        # With 10 graphs and k=5 on n=20, virtually all must change
        assert changed >= 9, \
            f"Only {changed}/10 graphs had destinations changed — rewiring too weak"

    def test_edge_weights_are_uniform_one(self, rand_graphs):
        for i, g in enumerate(rand_graphs):
            assert g.edge_attr is not None
            assert g.edge_attr.shape == (g.edge_index.shape[1], 1), \
                f"Graph {i}: edge_attr shape wrong"
            assert (g.edge_attr == 1.0).all(), \
                f"Graph {i}: edge_attr not all 1.0"


# ── Labels and features are untouched ────────────────────────────────────────

class TestNoLabelLeakage:

    def test_x_unchanged(self, knn_graphs, rand_graphs):
        for orig, rand in zip(knn_graphs, rand_graphs):
            assert torch.equal(orig.x, rand.x), "x was modified"

    def test_y_unchanged(self, knn_graphs, rand_graphs):
        for orig, rand in zip(knn_graphs, rand_graphs):
            assert torch.equal(orig.y, rand.y), "y was modified"

    def test_attack_cat_unchanged(self, knn_graphs, rand_graphs):
        for orig, rand in zip(knn_graphs, rand_graphs):
            assert torch.equal(orig.attack_cat, rand.attack_cat), \
                "attack_cat was modified"

    def test_rewiring_independent_of_labels(self):
        """
        Same graph, labels flipped → edges must be identical (labels not used).
        """
        g_orig  = _make_knn_graph(n=20, k=5, seed=7)
        g_flip  = Data(
            x=g_orig.x.clone(),
            edge_index=g_orig.edge_index.clone(),
            edge_attr=g_orig.edge_attr.clone(),
            y=1 - g_orig.y,        # flip all labels
            attack_cat=g_orig.attack_cat.clone(),
            num_nodes=g_orig.num_nodes,
        )
        rand_orig = randomise_edges([g_orig], seed=99)[0]
        rand_flip = randomise_edges([g_flip], seed=99)[0]

        assert torch.equal(rand_orig.edge_index, rand_flip.edge_index), \
            "Edge index differs when labels are flipped — labels were used!"


# ── Validate_random_graph helper ──────────────────────────────────────────────

class TestValidateRandomGraph:

    def test_passes_on_valid_rewired_graph(self, knn_graphs, rand_graphs):
        for orig, rand in zip(knn_graphs, rand_graphs):
            validate_random_graph(orig, rand)   # must not raise

    def test_fails_on_wrong_edge_count(self, knn_graph):
        # Remove one edge
        bad = Data(
            x=knn_graph.x,
            edge_index=knn_graph.edge_index[:, :-1],
            edge_attr=knn_graph.edge_attr[:-1],
            y=knn_graph.y,
            attack_cat=knn_graph.attack_cat,
            num_nodes=knn_graph.num_nodes,
        )
        with pytest.raises(AssertionError, match="Edge count"):
            validate_random_graph(knn_graph, bad)

    def test_fails_on_self_loop(self, knn_graph):
        ei = knn_graph.edge_index.clone()
        ei[1, 0] = ei[0, 0]   # force a self-loop
        bad = Data(
            x=knn_graph.x, edge_index=ei,
            edge_attr=knn_graph.edge_attr,
            y=knn_graph.y, attack_cat=knn_graph.attack_cat,
            num_nodes=knn_graph.num_nodes,
        )
        with pytest.raises(AssertionError, match="Self-loop"):
            validate_random_graph(knn_graph, bad)

    def test_fails_on_modified_x(self, knn_graph):
        bad_x = knn_graph.x.clone()
        bad_x[0, 0] += 1.0
        bad = Data(
            x=bad_x, edge_index=knn_graph.edge_index,
            edge_attr=knn_graph.edge_attr,
            y=knn_graph.y, attack_cat=knn_graph.attack_cat,
            num_nodes=knn_graph.num_nodes,
        )
        with pytest.raises(AssertionError, match="Node features"):
            validate_random_graph(knn_graph, bad)

    def test_fails_on_modified_y(self, knn_graph):
        bad_y = 1 - knn_graph.y
        bad = Data(
            x=knn_graph.x, edge_index=knn_graph.edge_index,
            edge_attr=knn_graph.edge_attr,
            y=bad_y, attack_cat=knn_graph.attack_cat,
            num_nodes=knn_graph.num_nodes,
        )
        with pytest.raises(AssertionError, match="Labels"):
            validate_random_graph(knn_graph, bad)


# ── Irregular degree sequence ─────────────────────────────────────────────────

class TestIrregularDegreeGraph:

    def test_irregular_degree_sequence_preserved(self):
        orig = _make_irregular_graph(n=15)
        rand = randomise_edges([orig], seed=7)[0]

        n = orig.num_nodes
        orig_deg = np.bincount(orig.edge_index[0].numpy(), minlength=n)
        rand_deg = np.bincount(rand.edge_index[0].numpy(), minlength=n)
        assert np.array_equal(orig_deg, rand_deg)

    def test_irregular_no_self_loops(self):
        orig = _make_irregular_graph(n=15)
        rand = randomise_edges([orig], seed=7)[0]
        src, dst = rand.edge_index
        assert (src != dst).all()

    def test_irregular_no_duplicates(self):
        orig = _make_irregular_graph(n=15)
        rand = randomise_edges([orig], seed=7)[0]
        src, dst = rand.edge_index
        pairs = list(zip(src.tolist(), dst.tolist()))
        assert len(pairs) == len(set(pairs))


# ── Empty / degenerate graphs ─────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_graph_returns_empty_edges(self):
        g = Data(
            x=torch.randn(5, 42),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 1)),
            y=torch.zeros(5, dtype=torch.long),
            attack_cat=torch.zeros(5, dtype=torch.long),
            num_nodes=5,
        )
        rand = randomise_edges([g], seed=0)[0]
        assert rand.edge_index.shape[1] == 0

    def test_single_node_graph(self):
        g = Data(
            x=torch.randn(1, 42),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 1)),
            y=torch.zeros(1, dtype=torch.long),
            attack_cat=torch.zeros(1, dtype=torch.long),
            num_nodes=1,
        )
        rand = randomise_edges([g], seed=0)[0]
        assert rand.edge_index.shape[1] == 0


# ── strip_edges sanity ────────────────────────────────────────────────────────

class TestStripEdges:

    def test_strip_removes_all_edges(self, knn_graphs):
        stripped = strip_edges(knn_graphs)
        for g in stripped:
            assert g.edge_index.shape[1] == 0

    def test_strip_preserves_x_y(self, knn_graphs):
        stripped = strip_edges(knn_graphs)
        for orig, s in zip(knn_graphs, stripped):
            assert torch.equal(orig.x, s.x)
            assert torch.equal(orig.y, s.y)

    def test_strip_preserves_num_nodes(self, knn_graphs):
        stripped = strip_edges(knn_graphs)
        for orig, s in zip(knn_graphs, stripped):
            assert orig.num_nodes == s.num_nodes
