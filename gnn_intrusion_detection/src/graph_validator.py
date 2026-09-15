"""
graph_validator.py
==================
Validation and statistics reporting for flow-similarity graph datasets.

For every PyG Data object this module reports:
  - number of nodes
  - number of edges
  - average / min / max node degree
  - number of isolated nodes  (degree == 0)
  - edge-weight min / max / mean
  - number of attack nodes  (label == 1)
  - number of normal nodes  (label == 0)

Aggregate statistics across all graphs in a split are also computed and
saved to CSV / JSON for inspection without loading graph objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

from .config import (
    OVERALL_STATS_PATH,
    TRAIN_GRAPH_STATS_PATH,
    TEST_GRAPH_STATS_PATH,
)
from .logger import get_logger

log = get_logger(__name__)


# ── Per-graph statistics ──────────────────────────────────────────────────────

def graph_stats(graph: Data, graph_idx: int = 0) -> Dict:
    """
    Compute per-graph statistics for one PyG Data object.

    Parameters
    ----------
    graph     : PyG Data object
    graph_idx : integer index used as the 'graph_id' field in the result

    Returns
    -------
    dict with scalar statistics
    """
    n_nodes = graph.num_nodes
    n_edges = graph.edge_index.shape[1] if graph.edge_index.numel() > 0 else 0

    # Node degrees
    if n_edges > 0:
        deg = degree(graph.edge_index[0], num_nodes=n_nodes).numpy()
    else:
        deg = np.zeros(n_nodes, dtype=np.float32)

    avg_deg  = float(deg.mean())
    min_deg  = float(deg.min())
    max_deg  = float(deg.max())
    n_isolated = int((deg == 0).sum())

    # Edge weights
    if n_edges > 0 and graph.edge_attr is not None:
        w = graph.edge_attr.numpy().flatten()
        ew_min  = float(w.min())
        ew_max  = float(w.max())
        ew_mean = float(w.mean())
    else:
        ew_min = ew_max = ew_mean = float("nan")

    # Node labels
    if graph.y is not None:
        labels = graph.y.numpy()
        n_attack = int((labels == 1).sum())
        n_normal = int((labels == 0).sum())
    else:
        n_attack = n_normal = 0

    return {
        "graph_id":   graph_idx,
        "n_nodes":    n_nodes,
        "n_edges":    n_edges,
        "avg_degree": round(avg_deg, 4),
        "min_degree": min_deg,
        "max_degree": max_deg,
        "n_isolated": n_isolated,
        "ew_min":     round(ew_min, 6),
        "ew_max":     round(ew_max, 6),
        "ew_mean":    round(ew_mean, 6),
        "n_attack":   n_attack,
        "n_normal":   n_normal,
    }


# ── Dataset-level statistics ──────────────────────────────────────────────────

def dataset_stats(graphs: List[Data], split_name: str) -> pd.DataFrame:
    """
    Compute per-graph stats for an entire list and return as DataFrame.

    Parameters
    ----------
    graphs     : list of PyG Data objects
    split_name : "train" or "test" — used for logging

    Returns
    -------
    pd.DataFrame  (one row per graph)
    """
    log.info("Computing statistics for %d '%s' graphs ...", len(graphs), split_name)
    rows = [graph_stats(g, i) for i, g in enumerate(graphs)]
    df = pd.DataFrame(rows)
    log.info("  Done.")
    return df


def aggregate_stats(stats_df: pd.DataFrame, split_name: str) -> Dict:
    """
    Summarise a stats DataFrame into scalar aggregate metrics.

    Returns a dict suitable for JSON serialisation.
    """
    numeric_cols = [
        "n_nodes", "n_edges", "avg_degree", "min_degree", "max_degree",
        "n_isolated", "ew_min", "ew_max", "ew_mean", "n_attack", "n_normal",
    ]
    agg: Dict = {"split": split_name, "n_graphs": len(stats_df)}
    for col in numeric_cols:
        agg[f"{col}_mean"]   = round(float(stats_df[col].mean()),   4)
        agg[f"{col}_median"] = round(float(stats_df[col].median()), 4)
        agg[f"{col}_min"]    = round(float(stats_df[col].min()),     4)
        agg[f"{col}_max"]    = round(float(stats_df[col].max()),     4)

    total_nodes  = int(stats_df["n_nodes"].sum())
    total_attack = int(stats_df["n_attack"].sum())
    total_normal = int(stats_df["n_normal"].sum())
    agg["total_nodes"]       = total_nodes
    agg["total_attack_nodes"] = total_attack
    agg["total_normal_nodes"] = total_normal
    agg["attack_node_pct"]   = round(100.0 * total_attack / total_nodes, 2) if total_nodes else 0.0
    return agg


def validate_no_leakage(graphs: List[Data], feature_cols: List[str]) -> bool:
    """
    Sanity-check that neither 'label' nor 'attack_cat' index appears in x.

    This is a structural check only — it verifies column count matches
    expectation (42 features) and that graph.y is stored separately.

    Returns True if no leakage detected, raises ValueError otherwise.
    """
    for i, g in enumerate(graphs):
        if g.x.shape[1] != 42:
            raise ValueError(
                f"Graph {i}: expected 42 node features, got {g.x.shape[1]}. "
                "Possible leakage — check feature_cols."
            )
        if not hasattr(g, "y") or g.y is None:
            raise ValueError(f"Graph {i}: missing node labels 'y'.")
        if not hasattr(g, "attack_cat") or g.attack_cat is None:
            raise ValueError(f"Graph {i}: missing 'attack_cat'.")
    log.info("  Leakage check passed: all graphs have 42 node features, labels stored separately.")
    return True


# ── Save / print helpers ──────────────────────────────────────────────────────

def save_stats(
    train_stats: pd.DataFrame,
    test_stats:  pd.DataFrame,
) -> None:
    """Save per-graph CSV stats and overall JSON to data/processed/graphs/."""
    TRAIN_GRAPH_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_stats.to_csv(TRAIN_GRAPH_STATS_PATH, index=False)
    test_stats.to_csv(TEST_GRAPH_STATS_PATH,   index=False)
    log.info("  Per-graph stats saved.")

    overall = {
        "train": aggregate_stats(train_stats, "train"),
        "test":  aggregate_stats(test_stats,  "test"),
    }
    with open(OVERALL_STATS_PATH, "w") as f:
        json.dump(overall, f, indent=2)
    log.info("  Overall stats saved to: %s", OVERALL_STATS_PATH)


def print_stats_report(
    train_stats: pd.DataFrame,
    test_stats:  pd.DataFrame,
) -> None:
    """Print a human-readable summary to stdout."""
    sep = "=" * 65

    for stats_df, name in [(train_stats, "TRAINING"), (test_stats, "TESTING")]:
        agg = aggregate_stats(stats_df, name)
        print(f"\n{sep}")
        print(f"  GRAPH STATISTICS — {name} SET")
        print(sep)
        print(f"  Total graphs          : {agg['n_graphs']:>8,}")
        print(f"  Total nodes           : {agg['total_nodes']:>8,}")
        print(f"  Total attack nodes    : {agg['total_attack_nodes']:>8,}  "
              f"({agg['attack_node_pct']:.1f}%)")
        print(f"  Total normal nodes    : {agg['total_normal_nodes']:>8,}")
        print()
        print(f"  Per-graph averages:")
        print(f"    Nodes               : {agg['n_nodes_mean']:>8.1f}")
        print(f"    Edges               : {agg['n_edges_mean']:>8.1f}")
        print(f"    Avg degree          : {agg['avg_degree_mean']:>8.3f}")
        print(f"    Min degree          : {agg['min_degree_mean']:>8.3f}")
        print(f"    Max degree          : {agg['max_degree_mean']:>8.3f}")
        print(f"    Isolated nodes      : {agg['n_isolated_mean']:>8.3f}")
        print(f"    Edge weight mean    : {agg['ew_mean_mean']:>8.5f}")
        print(f"    Edge weight min     : {agg['ew_min_min']:>8.5f}")
        print(f"    Edge weight max     : {agg['ew_max_max']:>8.5f}")
        print(f"\n  Per-graph range (nodes): "
              f"min={agg['n_nodes_min']:.0f}  max={agg['n_nodes_max']:.0f}")
        print(f"  Per-graph range (edges): "
              f"min={agg['n_edges_min']:.0f}  max={agg['n_edges_max']:.0f}")
        print(sep)
