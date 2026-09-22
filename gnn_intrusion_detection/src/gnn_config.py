"""
gnn_config.py
=============
GNN-specific configuration constants.
Kept separate from config.py to avoid IDE autofix conflicts.
"""

from pathlib import Path

# Resolve project root relative to this file (src/ -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input feature dimension — 42 preprocessed features per node
GNN_INPUT_DIM = 42

# Binary classification: Normal=0 / Attack=1
GNN_NUM_CLASSES = 2

MODELS_DIR  = _PROJECT_ROOT / "models"
RESULTS_DIR = _PROJECT_ROOT / "results" / "gnn"

# Default GraphSAGE hyperparameters — all overridable via CLI
GNN_DEFAULTS = {
    "hidden_dim":   64,
    "num_layers":   2,
    "dropout":      0.3,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "epochs":       50,
    "batch_size":   32,
    "seed":         42,
    "val_split":    0.15,
}
