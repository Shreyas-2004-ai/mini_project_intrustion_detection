"""
ablation.py
===========
Ablation study for the GraphSAGE intrusion-detection model.

Three controlled experiments using IDENTICAL architecture, features,
training procedure, seed, and train/val split:

  Experiment A — Original graph
      k=5 cosine-similarity k-NN edges (the existing baseline).

  Experiment B — No-message graph
      All edges removed.  SAGEConv degenerates to a per-node linear
      transformation with no neighbourhood aggregation.

  Experiment C — Random graph
      Same number of edges as the original graph, but randomly rewired.
      Edges connect random pairs of nodes (no self-loops); edge weights
      are set to 1.0 (uniform, uninformative).

Purpose
-------
Isolate the contribution of the meaningful ct_*-similarity edges.
If A >> B, graph structure helps in general.
If A >> C, the similarity-based wiring specifically matters (not just
having any edges).
If A ≈ C, the wiring scheme does not add value beyond edge existence.

Leakage
-------
Labels are never used to construct or select edges in any experiment.
The same official test graphs (loaded from disk) are evaluated ONCE,
after all three models are trained.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Data

from .evaluation import compute_metrics, predict_all
from .gnn_config import GNN_DEFAULTS, GNN_INPUT_DIM, MODELS_DIR
from .logger import get_logger
from .training import set_seed, train

log = get_logger(__name__)

ABLATION_DIR = Path(__file__).resolve().parent.parent / "results" / "gnn" / "ablation"


# ── Graph transformation helpers ──────────────────────────────────────────────

def strip_edges(graphs: List[Data]) -> List[Data]:
    """
    Experiment B: remove all edges from every graph.
    Returns new Data objects — originals are untouched.
    """
    out = []
    for g in graphs:
        new_g = Data(
            x          = g.x,
            edge_index = torch.zeros((2, 0), dtype=torch.long),
            edge_attr  = torch.zeros((0, 1), dtype=torch.float),
            y          = g.y,
            attack_cat = g.attack_cat,
            num_nodes  = g.num_nodes,
        )
        out.append(new_g)
    return out


def randomise_edges(graphs: List[Data], seed: int) -> List[Data]:
    """
    Experiment C: degree-preserving random rewiring.

    Algorithm
    ---------
    For each graph independently:
    1. Extract the original edge_index to get the out-degree of every node.
       In the k-NN graph every node has exactly k outgoing edges, but the
       function handles arbitrary degree sequences correctly.
    2. Collect all destination endpoints as a flat list (length = n_edges).
    3. Randomly shuffle that destination list (Fisher-Yates via numpy).
    4. Re-assign shuffled destinations back to sources in the original
       order, so node i still has exactly its original out-degree.
    5. Repair any self-loops by swapping the offending destination with a
       randomly chosen destination from a different source group — at most
       max_repair_passes passes over the edge list.
    6. If any self-loops remain after repairs (extremely rare), fall back
       to the next available non-self destination via a linear scan.

    Guarantees
    ----------
    - edge count            : identical to original  (no edges added/removed)
    - per-node out-degree   : identical to original  (same number of outgoing
                               edges per node — destinations are just reassigned)
    - no self-loops         : enforced by repair step
    - no duplicate edges    : enforced by using a set during assignment
    - labels / features     : never read — only edge_index structure is used

    Parameters
    ----------
    graphs : list of PyG Data objects (originals, untouched)
    seed   : integer seed for the numpy RNG (same seed → same permutation)

    Returns
    -------
    New list of PyG Data objects with rewired edge_index and uniform edge_attr.
    """
    rng = np.random.default_rng(seed)
    out: List[Data] = []

    for g in graphs:
        n       = g.num_nodes
        n_edges = g.edge_index.shape[1] if g.edge_index.numel() > 0 else 0

        if n_edges == 0 or n <= 1:
            out.append(Data(
                x=g.x, edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 1), dtype=torch.float),
                y=g.y, attack_cat=g.attack_cat, num_nodes=n,
            ))
            continue

        src = g.edge_index[0].numpy().copy()   # (n_edges,)  — source per edge
        dst = g.edge_index[1].numpy().copy()   # (n_edges,)  — original destinations

        # ── Step 1: shuffle ALL destinations globally ──────────────────────
        perm     = rng.permutation(n_edges)
        new_dst  = dst[perm]                   # shuffled destination pool

        # ── Step 2: repair self-loops ──────────────────────────────────────
        # A self-loop occurs where new_dst[i] == src[i].
        # Strategy: for each self-loop at position i, find another position j
        # (with a different source) where swapping new_dst[i] ↔ new_dst[j]
        # eliminates the self-loop at i without creating one at j.
        max_repair_passes = 5
        for _ in range(max_repair_passes):
            self_loop_mask = (new_dst == src)
            if not self_loop_mask.any():
                break
            sl_indices = np.where(self_loop_mask)[0]
            for i in sl_indices:
                # Find a position j where swap is safe:
                #   new_dst[j] != src[i]  (so i is fixed)
                #   new_dst[i] != src[j]  (so j is not broken)
                candidates = np.where(
                    (new_dst != src[i]) &          # my new dst won't self-loop
                    (src != src[i]) &               # different source node
                    (new_dst[i] != src)             # their new dst won't self-loop
                )[0]
                if len(candidates) > 0:
                    j = candidates[rng.integers(len(candidates))]
                    new_dst[i], new_dst[j] = new_dst[j], new_dst[i]

        # ── Step 3: remove any residual duplicates ─────────────────────────
        # Build adjacency set per source; if a (src, dst) pair is already
        # seen, replace the duplicate with the first unused valid destination.
        seen: set = set()
        for i in range(n_edges):
            s, d = int(src[i]), int(new_dst[i])
            if (s, d) not in seen and s != d:
                seen.add((s, d))
            else:
                # Linear scan for a valid replacement destination
                for candidate in range(n):
                    if candidate != s and (s, candidate) not in seen:
                        new_dst[i] = candidate
                        seen.add((s, candidate))
                        break

        # ── Step 4: assemble Data object ───────────────────────────────────
        ei = torch.tensor(
            np.stack([src, new_dst], axis=0), dtype=torch.long
        )
        out.append(Data(
            x          = g.x,
            edge_index = ei,
            edge_attr  = torch.ones(n_edges, 1, dtype=torch.float),
            y          = g.y,
            attack_cat = g.attack_cat,
            num_nodes  = n,
        ))

    return out


# ── Validation ────────────────────────────────────────────────────────────────

def validate_random_graph(orig: Data, rand: Data) -> None:
    """
    Assert that a randomised graph satisfies all structural requirements.

    Raises AssertionError with a descriptive message on failure.
    """
    n_orig  = orig.edge_index.shape[1] if orig.edge_index.numel() > 0 else 0
    n_rand  = rand.edge_index.shape[1] if rand.edge_index.numel() > 0 else 0

    # 1. Edge count preserved
    assert n_rand == n_orig, (
        f"Edge count mismatch: original={n_orig}, random={n_rand}"
    )

    if n_rand == 0:
        return

    src_o = orig.edge_index[0].numpy()
    src_r = rand.edge_index[0].numpy()
    dst_r = rand.edge_index[1].numpy()

    # 2. Out-degree sequence preserved
    orig_deg = np.bincount(src_o, minlength=orig.num_nodes)
    rand_deg = np.bincount(src_r, minlength=rand.num_nodes)
    assert np.array_equal(orig_deg, rand_deg), (
        "Out-degree sequence not preserved.\n"
        f"  orig degrees: {orig_deg}\n"
        f"  rand degrees: {rand_deg}"
    )

    # 3. No self-loops
    n_self = int((src_r == dst_r).sum())
    assert n_self == 0, f"Self-loops found: {n_self}"

    # 4. No duplicate edges
    pairs = list(zip(src_r.tolist(), dst_r.tolist()))
    n_dup = n_rand - len(set(pairs))
    assert n_dup == 0, f"Duplicate edges found: {n_dup}"

    # 5. Node features unchanged (x, y, attack_cat untouched)
    assert torch.equal(rand.x, orig.x), "Node features (x) were modified"
    assert torch.equal(rand.y, orig.y), "Labels (y) were modified"
    assert torch.equal(rand.attack_cat, orig.attack_cat), (
        "attack_cat was modified"
    )


# ── Single experiment runner ───────────────────────────────────────────────────

def run_experiment(
    name:         str,
    train_graphs: List[Data],
    test_graphs:  List[Data],
    cfg:          Dict,
    model_suffix: str,
) -> Dict:
    """
    Train on train_graphs, evaluate on test_graphs, return metrics dict.

    Parameters
    ----------
    name         : human-readable label (e.g. "A_original")
    train_graphs : graphs used for training (already transformed for the experiment)
    test_graphs  : graphs used for final evaluation (same transformation applied)
    cfg          : hyperparameter dict (identical across all experiments)
    model_suffix : suffix appended to the saved checkpoint filename
    """
    log.info("=== Experiment %s ===", name)
    t0 = time.time()

    # Redirect checkpoint to experiment-specific path so experiments don't
    # overwrite each other
    exp_cfg = dict(cfg)
    orig_models_dir = MODELS_DIR
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    # Temporarily patch the models dir inside training by overriding the path
    # We pass the model path via a side-channel: training.py saves to
    # MODELS_DIR / "graphsage_best.pt".  We rename after training.
    model, history, best_path = train(train_graphs, exp_cfg)

    # Move checkpoint to ablation directory with experiment name
    exp_model_path = ABLATION_DIR / f"graphsage_{model_suffix}.pt"
    import shutil
    shutil.copy(best_path, exp_model_path)
    log.info("  Model checkpoint saved: %s", exp_model_path)

    # Evaluate on test graphs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds, labels = predict_all(model, test_graphs, device, batch_size=cfg["batch_size"])
    metrics = compute_metrics(preds, labels)

    elapsed = time.time() - t0
    log.info(
        "  Experiment %s done in %.1f s | "
        "Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f",
        name, elapsed,
        metrics["accuracy"], metrics["precision"],
        metrics["recall"],   metrics["f1"],
    )

    # Save per-experiment results
    result_path = ABLATION_DIR / f"{model_suffix}_metrics.json"
    metrics_clean = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open(result_path, "w") as f:
        json.dump(metrics_clean, f, indent=2)

    hist_path = ABLATION_DIR / f"{model_suffix}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    metrics["experiment"] = name
    metrics["best_val_f1"] = max(r["val_f1"] for r in history)
    metrics["elapsed_s"]   = round(elapsed, 1)
    return metrics


# ── Full ablation pipeline ────────────────────────────────────────────────────

def run_ablation(
    train_graphs: List[Data],
    test_graphs:  List[Data],
    cfg:          Dict = None,
) -> List[Dict]:
    """
    Run all three ablation experiments and return the results list.

    Parameters
    ----------
    train_graphs : original training graphs (k=5 cosine-similarity)
    test_graphs  : original testing  graphs (k=5 cosine-similarity)
    cfg          : hyperparameter dict; defaults to GNN_DEFAULTS

    Returns
    -------
    List of three metrics dicts, one per experiment.
    """
    if cfg is None:
        cfg = dict(GNN_DEFAULTS)

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # ── Experiment A: original graph ──────────────────────────────────────
    log.info("Running Experiment A — Original k=5 cosine-similarity graph ...")
    res_a = run_experiment(
        name         = "A — Original graph (k=5 cosine-similarity)",
        train_graphs = train_graphs,
        test_graphs  = test_graphs,
        cfg          = cfg,
        model_suffix = "A_original",
    )
    results.append(res_a)

    # ── Experiment B: no edges ────────────────────────────────────────────
    log.info("Running Experiment B — No-message graph (edges removed) ...")
    train_b = strip_edges(train_graphs)
    test_b  = strip_edges(test_graphs)
    res_b = run_experiment(
        name         = "B — No-message graph (no edges)",
        train_graphs = train_b,
        test_graphs  = test_b,
        cfg          = cfg,
        model_suffix = "B_no_edges",
    )
    results.append(res_b)

    # ── Experiment C: random edges ────────────────────────────────────────
    log.info("Running Experiment C — Random graph (random rewiring) ...")
    train_c = randomise_edges(train_graphs, seed=cfg["seed"])
    test_c  = randomise_edges(test_graphs,  seed=cfg["seed"])
    res_c = run_experiment(
        name         = "C — Random graph (random rewiring)",
        train_graphs = train_c,
        test_graphs  = test_c,
        cfg          = cfg,
        model_suffix = "C_random",
    )
    results.append(res_c)

    # ── Save comparison table ─────────────────────────────────────────────
    _save_comparison(results)
    return results


# ── Comparison table ──────────────────────────────────────────────────────────

def _save_comparison(results: List[Dict]) -> None:
    rows = []
    for r in results:
        rows.append({
            "experiment":     r["experiment"],
            "graph_structure": _graph_label(r["experiment"]),
            "accuracy":        r["accuracy"],
            "precision":       r["precision"],
            "recall":          r["recall"],
            "f1":              r["f1"],
            "best_val_f1":     r.get("best_val_f1", ""),
        })

    table_path = ABLATION_DIR / "comparison_table.json"
    with open(table_path, "w") as f:
        json.dump(rows, f, indent=2)
    log.info("Comparison table saved: %s", table_path)


def _graph_label(experiment_name: str) -> str:
    if "A" in experiment_name:
        return "k=5 cosine-similarity k-NN"
    if "B" in experiment_name:
        return "No edges (node features only)"
    if "C" in experiment_name:
        return "Random rewiring (same edge count)"
    return "Unknown"


def print_ablation_report(results: List[Dict]) -> None:
    sep = "=" * 75
    print(f"\n{sep}")
    print("  ABLATION STUDY COMPLETE")
    print("  GraphSAGE — Graph Structure Contribution Analysis")
    print("  UNSW-NB15 Flow-Similarity Graphs")
    print(sep)
    print()
    print(f"  {'Experiment':<45} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("  " + "─" * 73)
    for r in results:
        label = _graph_label(r["experiment"])
        print(
            f"  {r['experiment']:<45} "
            f"{r['accuracy']:>7.4f} {r['precision']:>7.4f} "
            f"{r['recall']:>7.4f} {r['f1']:>7.4f}"
        )
    print()
    print(f"  {'Experiment':<45} {'Graph Structure'}")
    print("  " + "─" * 73)
    for r in results:
        print(f"  {r['experiment']:<45} {_graph_label(r['experiment'])}")
    print()

    # Print full confusion matrices
    for r in results:
        cm = r["confusion_matrix"]
        print(f"  [{r['experiment']}]  Confusion matrix:")
        print(f"                    Pred Normal  Pred Attack")
        print(f"    True Normal  : {cm[0][0]:>10,}  {cm[0][1]:>10,}")
        print(f"    True Attack  : {cm[1][0]:>10,}  {cm[1][1]:>10,}")
        print()
    print(sep)
    print()
