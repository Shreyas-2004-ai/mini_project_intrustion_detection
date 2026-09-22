"""
training.py
===========
Training loop for the GraphSAGE node-classification model on UNSW-NB15
flow-similarity graphs.

Key design decisions
--------------------
- Class imbalance is handled via a weighted CrossEntropyLoss.
  Normal : 56,000 nodes  (~32%)
  Attack : 119,341 nodes (~68%)
  The inverse-frequency weight for the Normal class is raised so the model
  does not trivially predict Attack for every node.

- Validation graphs are split OFF from the training graphs by index
  (the last `val_split` fraction of training graphs).  The official test
  graphs are never loaded here.

- The best model (highest validation F1) is saved to disk and reloaded
  for final test evaluation.

- Training history (loss, accuracy, precision, recall, F1 per epoch) is
  saved as a CSV for later plotting.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .gnn_config import MODELS_DIR, RESULTS_DIR, GNN_INPUT_DIM
from .logger import get_logger
from .models.graphsage import GraphSAGE

log = get_logger(__name__)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Class-weight computation ──────────────────────────────────────────────────

def compute_class_weights(graphs: List[Data], device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a list of graphs.
    Returns a tensor of shape (2,) for use in CrossEntropyLoss.
    """
    total_normal = sum((g.y == 0).sum().item() for g in graphs)
    total_attack = sum((g.y == 1).sum().item() for g in graphs)
    total        = total_normal + total_attack

    # weight_c = total / (num_classes * count_c)
    w_normal = total / (2.0 * total_normal) if total_normal > 0 else 1.0
    w_attack = total / (2.0 * total_attack) if total_attack > 0 else 1.0

    log.info(
        "  Class weights — Normal: %.4f  Attack: %.4f  "
        "(normal=%d  attack=%d)",
        w_normal, w_attack, total_normal, total_attack,
    )
    return torch.tensor([w_normal, w_attack], dtype=torch.float, device=device)


# ── Epoch-level helpers ───────────────────────────────────────────────────────

def _run_epoch(
    model:      GraphSAGE,
    loader:     DataLoader,
    criterion:  nn.CrossEntropyLoss,
    optimizer:  torch.optim.Optimizer | None,
    device:     torch.device,
    train_mode: bool,
) -> Dict[str, float]:
    """
    Run one full pass (train or eval) over a DataLoader.
    Returns a dict of aggregated metrics for the epoch.
    """
    model.train(train_mode)

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            batch = batch.to(device)

            logits = model(batch.x, batch.edge_index)   # (N_batch, 2)
            loss   = criterion(logits, batch.y)

            if train_mode and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.num_nodes
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch.y.cpu().numpy().tolist())

    n          = len(all_labels)
    avg_loss   = total_loss / n if n > 0 else float("nan")
    acc        = accuracy_score(all_labels, all_preds)
    prec       = precision_score(all_labels, all_preds, zero_division=0)
    rec        = recall_score(all_labels, all_preds, zero_division=0)
    f1         = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "loss":      round(avg_loss, 6),
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
    }


# ── Main training function ────────────────────────────────────────────────────

def train(
    train_graphs:  List[Data],
    cfg:           Dict,
) -> Tuple[GraphSAGE, List[Dict], Path]:
    """
    Train a GraphSAGE model on the provided training graphs.

    Parameters
    ----------
    train_graphs : list of PyG Data objects (from the training split only)
    cfg          : hyperparameter dict — see GNN_DEFAULTS in config.py

    Returns
    -------
    (best_model, history, best_model_path)
      best_model      : GraphSAGE loaded with best-validation-F1 weights
      history         : list of per-epoch dicts (train + val metrics)
      best_model_path : Path where best model was saved
    """
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Val split (taken from the END of training graphs — no shuffling across boundary)
    n_total = len(train_graphs)
    n_val   = max(1, int(n_total * cfg["val_split"]))
    n_train = n_total - n_val

    # Shuffle graph order with a fixed seed before splitting
    rng = random.Random(cfg["seed"])
    indices = list(range(n_total))
    rng.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    tr_graphs  = [train_graphs[i] for i in train_idx]
    val_graphs = [train_graphs[i] for i in val_idx]

    log.info(
        "Split: %d train graphs / %d val graphs  (val_split=%.2f)",
        len(tr_graphs), len(val_graphs), cfg["val_split"],
    )

    # ── DataLoaders
    tr_loader  = DataLoader(tr_graphs,  batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=cfg["batch_size"], shuffle=False)

    # ── Model
    model = GraphSAGE(
        in_channels=GNN_INPUT_DIM,
        hidden_dim=cfg["hidden_dim"],
        num_classes=2,
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    log.info("Model:\n%s", model)

    # ── Loss — weighted to address class imbalance
    class_weights = compute_class_weights(tr_graphs, device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimiser
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    # ── Output dirs
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_f1       = -1.0
    best_model_path   = MODELS_DIR / "graphsage_best.pt"
    history: List[Dict] = []

    log.info("Starting training for %d epochs ...", cfg["epochs"])
    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        tr_metrics  = _run_epoch(model, tr_loader,  criterion, optimizer, device, train_mode=True)
        val_metrics = _run_epoch(model, val_loader, criterion, None,      device, train_mode=False)

        record = {
            "epoch":          epoch,
            "train_loss":     tr_metrics["loss"],
            "train_acc":      tr_metrics["accuracy"],
            "train_precision":tr_metrics["precision"],
            "train_recall":   tr_metrics["recall"],
            "train_f1":       tr_metrics["f1"],
            "val_loss":       val_metrics["loss"],
            "val_acc":        val_metrics["accuracy"],
            "val_precision":  val_metrics["precision"],
            "val_recall":     val_metrics["recall"],
            "val_f1":         val_metrics["f1"],
        }
        history.append(record)

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_model_path)

        if epoch % 5 == 0 or epoch == 1:
            log.info(
                "Epoch %3d/%d | "
                "train loss=%.4f acc=%.4f f1=%.4f | "
                "val loss=%.4f acc=%.4f f1=%.4f%s",
                epoch, cfg["epochs"],
                tr_metrics["loss"],  tr_metrics["accuracy"],  tr_metrics["f1"],
                val_metrics["loss"], val_metrics["accuracy"], val_metrics["f1"],
                "  ← best" if val_metrics["f1"] == best_val_f1 else "",
            )

    elapsed = time.time() - t0
    log.info("Training complete in %.1f s  |  best val F1 = %.4f", elapsed, best_val_f1)

    # Reload best weights
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()

    return model, history, best_model_path
