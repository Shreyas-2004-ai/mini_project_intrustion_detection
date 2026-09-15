"""
graph_builder.py
================
Constructs flow-similarity graphs from preprocessed UNSW-NB15 flow records.

Design
------
Each CSV row is one graph node.  The dataset is split into non-overlapping
temporal windows of GRAPH_WINDOW_SIZE consecutive records (ordered by `id`).
Each window becomes one PyTorch Geometric `Data` object.

Edges are constructed using cosine similarity over the 8 ct_* neighbourhood
features.  For each node the k most-similar neighbours are connected
(k-nearest-neighbour graph, configurable).  The cosine similarity score is
stored as an edge weight in `edge_attr`.

IMPORTANT — academic terminology
---------------------------------
These edges represent *flow-similarity relationships* based on the UNSW-NB15
ct_* neighbourhood statistics.  They do NOT represent real IP communication
links.  The dataset contains no srcip / dstip columns; none are invented here.

Leakage prevention
------------------
- `label` and `attack_cat` are NEVER used when computing similarity or edges.
- The preprocessing pipeline is fitted on training data only and loaded here.
- Training and testing windows are built from their respective CSVs
  independently; no records cross the split boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data

from .config import (
    CT_EDGE_FEATURES,
    GRAPH_K,
    GRAPH_WINDOW_SIZE,
    PREPROCESSOR_PATH,
    TARGET_COLS,
    TRAIN_GRAPHS_DIR,
    TEST_GRAPHS_DIR,
)
from .data_loader import load_train_test
from .logger import get_logger
from .preprocessor import load_pipeline, transform_split

log = get_logger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _attack_cat_to_int(series: pd.Series) -> torch.Tensor:
    """Encode attack_cat strings as integer codes (alphabetical order)."""
    codes = pd.Categorical(series).codes          # -1 for NaN, ≥0 otherwise
    return torch.tensor(codes, dtype=torch.long)


def _window_to_graph(
    window_df: pd.DataFrame,
    feature_cols: List[str],
    ct_indices: List[int],
    k: int,
) -> Data:
    """
    Convert one window DataFrame (already preprocessed, targets attached) into
    a PyTorch Geometric Data object.

    Parameters
    ----------
    window_df   : preprocessed DataFrame slice, including label / attack_cat
    feature_cols: ordered list of node-feature column names (no id/targets)
    ct_indices  : column indices within feature_cols for the 8 ct_* features
    k           : number of nearest neighbours per node

    Returns
    -------
    torch_geometric.data.Data
    """
    n = len(window_df)

    # ── Node feature matrix  (N × 42) ────────────────────────────────────────
    X = window_df[feature_cols].values.astype(np.float32)   # (N, 42)
    x = torch.tensor(X, dtype=torch.float)

    # ── Edge construction via ct_* cosine similarity ──────────────────────────
    # Extract the 8 ct_* columns for similarity computation only
    ct_matrix = X[:, ct_indices]                             # (N, 8)

    # Cosine similarity matrix  (N × N); diagonal = 1.0
    sim_matrix = cosine_similarity(ct_matrix)                # (N, N)
    np.fill_diagonal(sim_matrix, -1.0)                       # exclude self-loops

    # Clamp k so it never exceeds n-1 (important for small final windows)
    k_eff = min(k, n - 1)

    # For each node pick the top-k neighbours (argsort descending)
    src_list, dst_list, weight_list = [], [], []
    for i in range(n):
        top_k_idx = np.argpartition(sim_matrix[i], -k_eff)[-k_eff:]
        for j in top_k_idx:
            src_list.append(i)
            dst_list.append(int(j))
            weight_list.append(float(sim_matrix[i, j]))

    if len(src_list) == 0:
        # Degenerate window with a single node — no edges
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long
        )
        edge_attr = torch.tensor(weight_list, dtype=torch.float).unsqueeze(1)

    # ── Labels  (targets — never used for edge construction) ─────────────────
    y = torch.tensor(window_df["label"].values, dtype=torch.long)
    attack_cat = _attack_cat_to_int(window_df["attack_cat"])

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        attack_cat=attack_cat,
        num_nodes=n,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def build_graphs(
    raw_df: pd.DataFrame,
    split_name: str,
    window_size: int = GRAPH_WINDOW_SIZE,
    k: int = GRAPH_K,
) -> List[Data]:
    """
    Build a list of flow-similarity graphs from a raw UNSW-NB15 DataFrame.

    The DataFrame is sorted by `id`, windowed into non-overlapping slices of
    `window_size` records, and each slice is converted to a PyG Data object.

    Parameters
    ----------
    raw_df      : raw (unprocessed) DataFrame from load_dataset()
    split_name  : "train" or "test" — used for logging only
    window_size : number of records per graph window
    k           : number of nearest neighbours per node

    Returns
    -------
    List[torch_geometric.data.Data]
    """
    log.info(
        "Building %s graphs  (window=%d, k=%d, rows=%d) ...",
        split_name, window_size, k, len(raw_df),
    )

    # Sort by id to ensure temporal order
    df = raw_df.sort_values("id").reset_index(drop=True)

    # Apply the preprocessing pipeline (fitted on training data only)
    pipeline = load_pipeline(PREPROCESSOR_PATH)
    proc_df  = transform_split(pipeline, df, name=split_name)

    # Determine ordered feature columns (no id, no targets)
    feature_cols = [c for c in proc_df.columns if c not in TARGET_COLS]

    # Indices of ct_* columns within feature_cols
    ct_indices = [feature_cols.index(c) for c in CT_EDGE_FEATURES if c in feature_cols]
    if len(ct_indices) != len(CT_EDGE_FEATURES):
        missing = [c for c in CT_EDGE_FEATURES if c not in feature_cols]
        raise ValueError(f"ct_* edge features missing from processed data: {missing}")

    log.info(
        "  Feature columns: %d  |  ct_* edge indices: %s",
        len(feature_cols), ct_indices,
    )

    # Build windows
    graphs: List[Data] = []
    n_rows   = len(proc_df)
    n_full   = n_rows // window_size
    n_remain = n_rows %  window_size

    for i in range(n_full):
        start = i * window_size
        end   = start + window_size
        window = proc_df.iloc[start:end]
        graphs.append(_window_to_graph(window, feature_cols, ct_indices, k))

    # Handle the final partial window explicitly
    if n_remain > 0:
        window = proc_df.iloc[n_full * window_size:]
        log.info(
            "  Partial final window: %d records (index %d)",
            n_remain, n_full,
        )
        graphs.append(_window_to_graph(window, feature_cols, ct_indices, k))

    log.info("  Generated %d graphs for '%s' split.", len(graphs), split_name)
    return graphs


def save_graphs(graphs: List[Data], output_dir: Path) -> None:
    """
    Serialise each graph as an individual .pt file.

    Files are named graph_0000.pt, graph_0001.pt, …
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, graph in enumerate(graphs):
        path = output_dir / f"graph_{idx:05d}.pt"
        torch.save(graph, path)
    log.info("  Saved %d graph files to: %s", len(graphs), output_dir)


def load_graphs(input_dir: Path) -> List[Data]:
    """Load all .pt graph files from a directory, sorted by filename."""
    input_dir = Path(input_dir)
    paths = sorted(input_dir.glob("graph_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No graph files found in {input_dir}")
    graphs = [torch.load(p, weights_only=False) for p in paths]
    log.info("  Loaded %d graphs from: %s", len(graphs), input_dir)
    return graphs


def run_graph_construction(
    window_size: int = GRAPH_WINDOW_SIZE,
    k: int = GRAPH_K,
) -> Tuple[List[Data], List[Data]]:
    """
    End-to-end graph construction for both train and test splits.

    Returns
    -------
    (train_graphs, test_graphs)
    """
    train_raw, test_raw = load_train_test()

    train_graphs = build_graphs(train_raw, "train", window_size=window_size, k=k)
    test_graphs  = build_graphs(test_raw,  "test",  window_size=window_size, k=k)

    save_graphs(train_graphs, TRAIN_GRAPHS_DIR)
    save_graphs(test_graphs,  TEST_GRAPHS_DIR)

    return train_graphs, test_graphs
