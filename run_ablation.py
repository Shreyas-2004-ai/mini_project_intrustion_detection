"""
run_ablation.py
===============
Entry-point for the GraphSAGE ablation study.

Usage
-----
    python run_ablation.py

Runs three experiments with IDENTICAL model architecture, features,
training procedure, seed, and train/val split:

  A — Original k=5 cosine-similarity graph
  B — No-message graph (edges removed)
  C — Degree-preserving random rewired graph (same edge count, same out-degrees)

Pre-flight structural checks are run on Experiment C graphs before any
training starts to verify correctness of the rewiring.

Official test graphs are used for final evaluation only, after training.
Results are saved to results/gnn/ablation/.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ablation import randomise_edges, run_ablation, print_ablation_report, validate_random_graph
from src.config import TRAIN_GRAPHS_DIR, TEST_GRAPHS_DIR
from src.gnn_config import GNN_DEFAULTS
from src.graph_builder import load_graphs
from src.logger import get_logger

log = get_logger("run_ablation")


def preflight_random_graph(train_graphs, test_graphs, seed: int) -> None:
    """
    Verify that degree-preserving rewiring satisfies all structural requirements
    BEFORE any training begins.  Raises AssertionError on failure.
    """
    log.info("Pre-flight: verifying Experiment C graph properties ...")

    # Build C graphs for a small sample to keep pre-flight fast
    sample_orig  = train_graphs[:20]
    sample_rand  = randomise_edges(sample_orig, seed=seed)

    total_edges_orig = total_edges_rand = 0
    total_self_loops = total_duplicates  = 0
    degree_mismatches = 0

    for orig, rand in zip(sample_orig, sample_rand):
        n_orig = orig.edge_index.shape[1]
        n_rand = rand.edge_index.shape[1]
        total_edges_orig += n_orig
        total_edges_rand += n_rand

        src_r = rand.edge_index[0].numpy()
        dst_r = rand.edge_index[1].numpy()

        # Self-loops
        total_self_loops += int((src_r == dst_r).sum())

        # Duplicates
        pairs = list(zip(src_r.tolist(), dst_r.tolist()))
        total_duplicates += len(pairs) - len(set(pairs))

        # Out-degree sequence
        n = orig.num_nodes
        od = np.bincount(orig.edge_index[0].numpy(), minlength=n)
        rd = np.bincount(src_r, minlength=n)
        if not np.array_equal(od, rd):
            degree_mismatches += 1

    assert total_edges_orig == total_edges_rand, (
        f"Edge count mismatch: orig={total_edges_orig} rand={total_edges_rand}"
    )
    assert total_self_loops == 0, (
        f"Self-loops found in Experiment C graphs: {total_self_loops}"
    )
    assert total_duplicates == 0, (
        f"Duplicate edges found in Experiment C graphs: {total_duplicates}"
    )
    assert degree_mismatches == 0, (
        f"Out-degree sequence not preserved in {degree_mismatches} graphs"
    )

    log.info(
        "  Pre-flight PASSED on %d sample graphs: "
        "edges=%d, self_loops=0, duplicates=0, degree_preserved=True",
        len(sample_orig), total_edges_rand,
    )


def main() -> None:
    log.info("=" * 65)
    log.info("GNN Intrusion Detection — Ablation Study")
    log.info("  Experiments: A (original), B (no edges), C (degree-preserving random)")
    log.info("=" * 65)

    cfg = dict(GNN_DEFAULTS)
    log.info("Shared config: %s", cfg)

    log.info("Loading training graphs ...")
    train_graphs = load_graphs(TRAIN_GRAPHS_DIR)

    log.info("Loading test graphs ...")
    test_graphs = load_graphs(TEST_GRAPHS_DIR)

    log.info(
        "Loaded %d train graphs, %d test graphs",
        len(train_graphs), len(test_graphs),
    )

    # Structural verification BEFORE any training
    preflight_random_graph(train_graphs, test_graphs, seed=cfg["seed"])

    results = run_ablation(train_graphs, test_graphs, cfg)
    print_ablation_report(results)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("Ablation study failed: %s", exc, exc_info=True)
        sys.exit(1)

