"""
evaluation.py
=============
Final evaluation of a trained GraphSAGE model on the official UNSW-NB15
test graphs.

Rules
-----
- The test graphs are used ONCE, here, after training is complete.
- No hyperparameter tuning is performed against these results.
- Metrics are computed at the NODE level (one prediction per flow record).
- Results are saved to results/gnn/ as JSON and a plain-text report.
- Plots (training history + confusion matrix) are also saved here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .gnn_config import RESULTS_DIR
from .logger import get_logger
from .models.graphsage import GraphSAGE

log = get_logger(__name__)


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_all(
    model:  GraphSAGE,
    graphs: List[Data],
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference over a list of graphs.

    Returns
    -------
    (all_preds, all_labels) — numpy arrays of length = total nodes
    """
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    all_preds, all_labels = [], []
    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        preds  = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch.y.cpu().numpy().tolist())

    return np.array(all_preds), np.array(all_labels)


# ── Metric computation ────────────────────────────────────────────────────────

def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
) -> Dict:
    """Compute the full set of classification metrics."""
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    cm   = confusion_matrix(labels, preds).tolist()
    cr   = classification_report(
        labels, preds,
        target_names=["Normal (0)", "Attack (1)"],
        zero_division=0,
    )

    n_normal = int((labels == 0).sum())
    n_attack = int((labels == 1).sum())
    total    = len(labels)

    return {
        "total_nodes":  total,
        "n_normal":     n_normal,
        "n_attack":     n_attack,
        "accuracy":     round(float(acc),  4),
        "precision":    round(float(prec), 4),
        "recall":       round(float(rec),  4),
        "f1":           round(float(f1),   4),
        "confusion_matrix": cm,
        "classification_report": cr,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_training_history(history: List[Dict], out_dir: Path) -> None:
    """Save training and validation loss + F1 curves."""
    epochs      = [r["epoch"]     for r in history]
    train_loss  = [r["train_loss"] for r in history]
    val_loss    = [r["val_loss"]   for r in history]
    train_f1    = [r["train_f1"]   for r in history]
    val_f1      = [r["val_f1"]     for r in history]
    val_prec    = [r["val_precision"] for r in history]
    val_rec     = [r["val_recall"]    for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GraphSAGE Training History — UNSW-NB15", fontsize=13, fontweight="bold")

    # Loss
    axes[0].plot(epochs, train_loss, label="Train loss", color="#2c7bb6", linewidth=1.5)
    axes[0].plot(epochs, val_loss,   label="Val loss",   color="#d7191c", linewidth=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.4)

    # F1 / Precision / Recall
    axes[1].plot(epochs, val_f1,   label="Val F1",        color="#1a9641", linewidth=1.8)
    axes[1].plot(epochs, val_prec, label="Val Precision",  color="#fdae61", linewidth=1.2, linestyle="--")
    axes[1].plot(epochs, val_rec,  label="Val Recall",     color="#a6611a", linewidth=1.2, linestyle=":")
    axes[1].plot(epochs, train_f1, label="Train F1",       color="#2c7bb6", linewidth=1.0, alpha=0.6)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics"); axes[1].legend(); axes[1].grid(alpha=0.4)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    path = out_dir / "training_history.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Training history plot saved: %s", path)


def plot_confusion_matrix(cm: List[List[int]], out_dir: Path) -> None:
    """Save a labelled confusion-matrix heatmap."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle("GraphSAGE — Confusion Matrix (Test Set)", fontsize=12, fontweight="bold")

    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    classes = ["Normal (0)", "Attack (1)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)

    thresh = cm_arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm_arr[i, j]:,}",
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
                fontsize=13, fontweight="bold",
            )

    plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Confusion matrix plot saved: %s", path)


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(
    metrics:      Dict,
    history:      List[Dict],
    cfg:          Dict,
    out_dir:      Path,
) -> None:
    """Persist metrics JSON, history CSV, and config JSON."""
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)

    # Evaluation metrics
    metrics_path = out_dir / "test_metrics.json"
    # classification_report is a string — keep it separate
    metrics_clean = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open(metrics_path, "w") as f:
        json.dump(metrics_clean, f, indent=2)
    log.info("  Metrics saved: %s", metrics_path)

    # Classification report as text
    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(metrics["classification_report"])
    log.info("  Classification report saved: %s", report_path)

    # Training history
    hist_df = pd.DataFrame(history)
    hist_path = out_dir / "training_history.csv"
    hist_df.to_csv(hist_path, index=False)
    log.info("  Training history saved: %s", hist_path)

    # Config snapshot
    cfg_path = out_dir / "model_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    log.info("  Model config saved: %s", cfg_path)


# ── Console report ────────────────────────────────────────────────────────────

def print_evaluation_report(metrics: Dict, cfg: Dict) -> None:
    """Print the final test-set evaluation to stdout."""
    sep = "=" * 65
    print(f"\n{sep}")
    print("  GRAPHSAGE — FINAL TEST SET EVALUATION")
    print("  UNSW-NB15 Flow-Similarity Graphs")
    print(sep)
    print(f"\n  Model configuration:")
    print(f"    hidden_dim  : {cfg.get('hidden_dim')}")
    print(f"    num_layers  : {cfg.get('num_layers')}")
    print(f"    dropout     : {cfg.get('dropout')}")
    print(f"    lr          : {cfg.get('lr')}")
    print(f"    weight_decay: {cfg.get('weight_decay')}")
    print(f"    epochs      : {cfg.get('epochs')}")
    print(f"    batch_size  : {cfg.get('batch_size')}")
    print(f"    seed        : {cfg.get('seed')}")
    print(f"\n  Test set composition:")
    print(f"    Total nodes  : {metrics['total_nodes']:>10,}")
    print(f"    Normal nodes : {metrics['n_normal']:>10,}")
    print(f"    Attack nodes : {metrics['n_attack']:>10,}")
    print(f"\n  Metrics (node-level binary classification):")
    print(f"    Accuracy     : {metrics['accuracy']:.4f}")
    print(f"    Precision    : {metrics['precision']:.4f}")
    print(f"    Recall       : {metrics['recall']:.4f}")
    print(f"    F1-score     : {metrics['f1']:.4f}")
    print(f"\n  Confusion matrix (rows=True, cols=Predicted):")
    cm = metrics["confusion_matrix"]
    print(f"                  Pred Normal  Pred Attack")
    print(f"    True Normal : {cm[0][0]:>10,}  {cm[0][1]:>10,}")
    print(f"    True Attack : {cm[1][0]:>10,}  {cm[1][1]:>10,}")
    print(f"\n  Classification report:")
    print(metrics["classification_report"])
    print(sep)
    print("  NOTE: This is the GNN baseline. Comparison with conventional")
    print("  ML models will follow in a separate experiment.")
    print(f"{sep}\n")


# ── End-to-end evaluation entry point ────────────────────────────────────────

def evaluate(
    model:       GraphSAGE,
    test_graphs: List[Data],
    history:     List[Dict],
    cfg:         Dict,
    batch_size:  int = 32,
) -> Dict:
    """
    Run final evaluation on test_graphs, save all artefacts, print report.

    Returns the metrics dict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Evaluating on %d test graphs ...", len(test_graphs))
    preds, labels = predict_all(model, test_graphs, device, batch_size)
    metrics       = compute_metrics(preds, labels)

    # Save artefacts
    save_results(metrics, history, cfg, RESULTS_DIR)
    plot_training_history(history, RESULTS_DIR)
    plot_confusion_matrix(metrics["confusion_matrix"], RESULTS_DIR)

    print_evaluation_report(metrics, cfg)
    return metrics
