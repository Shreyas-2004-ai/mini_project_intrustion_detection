"""
inference.py
============
Inference pipeline for new uploaded CSV files.

Accepts any CSV that follows the UNSW-NB15 feature schema and produces
node-level Normal / Attack predictions using the trained GraphSAGE model.

Pipeline
--------
1. Load and validate the CSV (same column checks as the training pipeline).
2. Apply the saved preprocessing pipeline (fitted on training data — never
   re-fitted here).
3. Build flow-similarity graphs using the same window=100, k=5 construction
   as training (reuses graph_builder._window_to_graph).
4. Run inference with the saved GraphSAGE checkpoint.
5. Return per-flow predictions, probabilities, and summary statistics.

Label and attack_cat handling
------------------------------
In production, uploaded CSVs may not contain `label` or `attack_cat`.
The inference pipeline handles both cases:
- If present, they are stripped before building node features (same as
  training).
- If absent, dummy zero-valued columns are inserted so the preprocessing
  pipeline receives the expected schema.  The dummy values are never used
  as features.

No model weights are modified.  No preprocessing is re-fitted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .config import (
    CATEGORICAL_COLS,
    CT_EDGE_FEATURES,
    GRAPH_K,
    GRAPH_WINDOW_SIZE,
    IDENTIFIER_COLS,
    PREPROCESSOR_PATH,
    TARGET_COLS,
)
from .gnn_config import GNN_DEFAULTS, GNN_INPUT_DIM, MODELS_DIR
from .graph_builder import _window_to_graph
from .logger import get_logger
from .models.graphsage import GraphSAGE
from .preprocessor import load_pipeline, transform_split

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = MODELS_DIR / "graphsage_best.pt"
REQUIRED_FEATURE_COLS = (
    # All 45 original columns minus the two target columns.
    # The model expects the 42 preprocessed features that the pipeline produces.
    # Validated dynamically against the pipeline output at runtime.
)


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_inference_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load an uploaded CSV and prepare it for the inference pipeline.

    Accepts CSVs with or without `label` / `attack_cat` columns.
    If those columns are absent, dummy zero-valued placeholders are inserted
    so the preprocessing pipeline can run without modification.

    Parameters
    ----------
    csv_path : Path to the uploaded CSV file.

    Returns
    -------
    pd.DataFrame — ready to pass into the preprocessing pipeline.

    Raises
    ------
    FileNotFoundError : if csv_path does not exist.
    ValueError        : if required feature columns are missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    log.info("Loaded CSV: %s  (%d rows x %d cols)", csv_path.name, *df.shape)

    # Ensure id column is present (used for temporal ordering)
    if "id" not in df.columns:
        log.warning("'id' column missing — assigning sequential index as id.")
        df.insert(0, "id", range(1, len(df) + 1))

    # Inject dummy targets if absent (never used as features)
    for col in TARGET_COLS:
        if col not in df.columns:
            log.info("  '%s' not in CSV — inserting dummy zeros.", col)
            df[col] = 0

    # Verify the categorical and numerical feature columns are present
    required = set(CATEGORICAL_COLS + ["id"])
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Required feature columns missing from input CSV: {sorted(missing)}"
        )

    return df


# ── Graph construction for inference ─────────────────────────────────────────

def build_inference_graphs(
    raw_df:      pd.DataFrame,
    window_size: int = GRAPH_WINDOW_SIZE,
    k:           int = GRAPH_K,
) -> Tuple[List[Data], List[int]]:
    """
    Preprocess the input DataFrame and build flow-similarity graphs.

    Reuses the saved preprocessing pipeline and the same windowing /
    k-NN construction used during training.

    Parameters
    ----------
    raw_df      : DataFrame from load_inference_csv()
    window_size : records per graph window (default: 100)
    k           : k-NN edges per node (default: 5)

    Returns
    -------
    (graphs, original_row_order)
      graphs            : list of PyG Data objects
      original_row_order: row indices from raw_df in the order they were
                          processed (after sorting by id), so that predictions
                          can be re-aligned to the input CSV row order.
    """
    log.info(
        "Building inference graphs (window=%d, k=%d, rows=%d) ...",
        window_size, k, len(raw_df),
    )

    # Sort by id — same as training
    df = raw_df.sort_values("id").reset_index(drop=True)
    original_row_order = df.index.tolist()

    # Apply the saved preprocessing pipeline (never re-fitted)
    pipeline    = load_pipeline(PREPROCESSOR_PATH)
    proc_df     = transform_split(pipeline, df, name="inference")

    # Feature columns — same derivation as graph_builder.build_graphs
    feature_cols = [c for c in proc_df.columns if c not in TARGET_COLS]

    # ct_* column indices within feature_cols
    ct_indices = [feature_cols.index(c) for c in CT_EDGE_FEATURES if c in feature_cols]
    if len(ct_indices) != len(CT_EDGE_FEATURES):
        missing = [c for c in CT_EDGE_FEATURES if c not in feature_cols]
        raise ValueError(f"ct_* edge features missing after preprocessing: {missing}")

    # Validate feature dimension
    if len(feature_cols) != GNN_INPUT_DIM:
        raise ValueError(
            f"Expected {GNN_INPUT_DIM} feature columns, got {len(feature_cols)}. "
            "Ensure the CSV has the correct UNSW-NB15 schema."
        )

    # Build windows
    graphs: List[Data] = []
    n_rows   = len(proc_df)
    n_full   = n_rows // window_size
    n_remain = n_rows % window_size

    for i in range(n_full):
        window = proc_df.iloc[i * window_size:(i + 1) * window_size]
        graphs.append(_window_to_graph(window, feature_cols, ct_indices, k))

    if n_remain > 0:
        window = proc_df.iloc[n_full * window_size:]
        graphs.append(_window_to_graph(window, feature_cols, ct_indices, k))

    log.info("  Generated %d inference graphs.", len(graphs))
    return graphs, original_row_order


# ── Model loading ─────────────────────────────────────────────────────────────

def load_graphsage(
    model_path: Path = DEFAULT_MODEL_PATH,
    cfg:        Dict = None,
) -> GraphSAGE:
    """
    Load the saved GraphSAGE checkpoint into eval mode.

    Parameters
    ----------
    model_path : path to the .pt checkpoint file
    cfg        : model hyperparameters; defaults to GNN_DEFAULTS

    Returns
    -------
    GraphSAGE in eval mode on CPU (moved to GPU inside run_inference if available)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "Run run_train_gnn.py first to train and save the model."
        )

    if cfg is None:
        cfg = GNN_DEFAULTS

    model = GraphSAGE(
        in_channels=GNN_INPUT_DIM,
        hidden_dim=cfg.get("hidden_dim", 64),
        num_classes=2,
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.3),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state  = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    log.info("Loaded GraphSAGE from: %s  (device=%s)", model_path, device)
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model:       GraphSAGE,
    graphs:      List[Data],
    batch_size:  int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a list of graphs and return predictions and probabilities.

    Parameters
    ----------
    model      : loaded GraphSAGE in eval mode
    graphs     : output of build_inference_graphs()
    batch_size : graphs per batch

    Returns
    -------
    (predictions, probabilities)
      predictions  : int array of shape (N,) — 0 = Normal, 1 = Attack
      probabilities: float array of shape (N, 2) — softmax probabilities
                     [:, 0] = P(Normal), [:, 1] = P(Attack)
    """
    device = next(model.parameters()).device
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    all_preds, all_probs = [], []
    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index)          # (N_batch, 2)
        probs  = F.softmax(logits, dim=-1).cpu().numpy()
        preds  = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_probs.append(probs)

    predictions   = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probs)
    return predictions, probabilities


# ── Summary statistics ────────────────────────────────────────────────────────

def summarise(predictions: np.ndarray) -> Dict:
    """
    Compute summary statistics from inference predictions.

    Parameters
    ----------
    predictions : int array — 0 = Normal, 1 = Attack

    Returns
    -------
    dict with total_flows, normal_flows, attack_flows, attack_pct
    """
    total   = len(predictions)
    n_atk   = int((predictions == 1).sum())
    n_norm  = int((predictions == 0).sum())
    atk_pct = round(100.0 * n_atk / total, 2) if total > 0 else 0.0

    return {
        "total_flows":   total,
        "normal_flows":  n_norm,
        "attack_flows":  n_atk,
        "attack_pct":    atk_pct,
    }


# ── End-to-end pipeline ───────────────────────────────────────────────────────

def predict_csv(
    csv_path:   Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    window_size: int = GRAPH_WINDOW_SIZE,
    k:          int  = GRAPH_K,
    batch_size: int  = 32,
) -> Dict:
    """
    Full inference pipeline: CSV → predictions + summary.

    Parameters
    ----------
    csv_path    : path to the uploaded CSV file
    model_path  : path to the GraphSAGE checkpoint
    window_size : records per graph window
    k           : k-NN edges per node
    batch_size  : inference batch size

    Returns
    -------
    dict:
      predictions   : np.ndarray (N,) — 0=Normal / 1=Attack per flow
      probabilities : np.ndarray (N, 2) — P(Normal), P(Attack) per flow
      summary       : dict — total_flows, normal_flows, attack_flows, attack_pct
      n_graphs      : number of graphs generated
      n_flows       : total flow records processed
    """
    # 1. Load CSV
    raw_df = load_inference_csv(csv_path)
    n_flows = len(raw_df)

    # 2. Build graphs
    graphs, _ = build_inference_graphs(raw_df, window_size=window_size, k=k)

    # 3. Load model
    model = load_graphsage(model_path)

    # 4. Inference
    predictions, probabilities = run_inference(model, graphs, batch_size=batch_size)

    # 5. Summary
    summary = summarise(predictions)

    log.info(
        "Inference complete: %d flows | %d normal | %d attack (%.1f%%)",
        summary["total_flows"], summary["normal_flows"],
        summary["attack_flows"], summary["attack_pct"],
    )

    return {
        "predictions":   predictions,
        "probabilities": probabilities,
        "summary":       summary,
        "n_graphs":      len(graphs),
        "n_flows":       n_flows,
    }
