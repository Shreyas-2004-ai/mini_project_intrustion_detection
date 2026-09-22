"""
run_train_gnn.py
================
Entry-point for Phase 3: GraphSAGE training and evaluation.

Usage
-----
    python run_train_gnn.py [options]

All hyperparameters have defaults from src/config.GNN_DEFAULTS and can be
overridden on the command line.

What it does
------------
1. Pre-flight checks: verifies x does not contain label / attack_cat,
   and that test graphs are kept separate.
2. Loads the pre-built training graphs from data/processed/graphs/train/.
3. Trains a GraphSAGE model (train/val split from training graphs only).
4. Saves the best checkpoint to models/graphsage_best.pt.
5. Loads the official test graphs from data/processed/graphs/test/.
6. Evaluates the best model ONCE on the test graphs.
7. Saves metrics, history, and plots to results/gnn/.
8. Prints the complete evaluation report.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.gnn_config import GNN_DEFAULTS, GNN_INPUT_DIM, MODELS_DIR, RESULTS_DIR
from src.config import TRAIN_GRAPHS_DIR, TEST_GRAPHS_DIR
from src.evaluation import evaluate
from src.graph_builder import load_graphs
from src.logger import get_logger
from src.training import train

log = get_logger("run_train_gnn")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    d = GNN_DEFAULTS
    p = argparse.ArgumentParser(description="Phase 3: GraphSAGE training + evaluation")
    p.add_argument("--hidden_dim",    type=int,   default=d["hidden_dim"])
    p.add_argument("--num_layers",    type=int,   default=d["num_layers"])
    p.add_argument("--dropout",       type=float, default=d["dropout"])
    p.add_argument("--lr",            type=float, default=d["lr"])
    p.add_argument("--weight_decay",  type=float, default=d["weight_decay"])
    p.add_argument("--epochs",        type=int,   default=d["epochs"])
    p.add_argument("--batch_size",    type=int,   default=d["batch_size"])
    p.add_argument("--seed",          type=int,   default=d["seed"])
    p.add_argument("--val_split",     type=float, default=d["val_split"])
    return p.parse_args()


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def preflight_checks(train_graphs, test_graphs) -> None:
    """
    Structural safety checks before training starts.
    Raises AssertionError if any check fails.
    """
    # 1. Node features must be exactly 42 — no label / attack_cat leakage
    for i, g in enumerate(train_graphs[:5]):
        assert g.x.shape[1] == GNN_INPUT_DIM, (
            f"Train graph {i}: x has {g.x.shape[1]} features, expected {GNN_INPUT_DIM}. "
            "label or attack_cat may have leaked into x."
        )
    for i, g in enumerate(test_graphs[:5]):
        assert g.x.shape[1] == GNN_INPUT_DIM, (
            f"Test graph {i}: x has {g.x.shape[1]} features, expected {GNN_INPUT_DIM}."
        )

    # 2. Labels y must be binary {0, 1}
    for i, g in enumerate(train_graphs[:5]):
        unique = g.y.unique().tolist()
        assert all(v in (0, 1) for v in unique), (
            f"Train graph {i}: y contains non-binary values {unique}."
        )

    # 3. attack_cat must be a separate attribute, not inside x
    for i, g in enumerate(train_graphs[:3]):
        assert hasattr(g, "attack_cat"), f"Train graph {i}: missing attack_cat attribute."

    # 4. Training and test directories must be different
    assert str(TRAIN_GRAPHS_DIR) != str(TEST_GRAPHS_DIR), \
        "Train and test graph directories are the same — aborting."

    log.info("Pre-flight checks passed.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = vars(args)

    log.info("=" * 65)
    log.info("GNN Intrusion Detection — Phase 3: GraphSAGE Training")
    log.info("=" * 65)
    log.info("Config: %s", cfg)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load graphs ────────────────────────────────────────────────────────
    log.info("Loading training graphs from: %s", TRAIN_GRAPHS_DIR)
    train_graphs = load_graphs(TRAIN_GRAPHS_DIR)
    log.info("Loading test graphs from: %s", TEST_GRAPHS_DIR)
    test_graphs  = load_graphs(TEST_GRAPHS_DIR)

    # ── Pre-flight ─────────────────────────────────────────────────────────
    log.info("Running pre-flight checks ...")
    preflight_checks(train_graphs, test_graphs)

    # ── Train ──────────────────────────────────────────────────────────────
    log.info("Step 1/2 — Training GraphSAGE ...")
    model, history, best_path = train(train_graphs, cfg)
    log.info("Best model saved to: %s", best_path)

    # ── Evaluate ───────────────────────────────────────────────────────────
    log.info("Step 2/2 — Evaluating on official test graphs (used ONCE) ...")
    evaluate(model, test_graphs, history, cfg, batch_size=cfg["batch_size"])

    log.info("All results saved to: %s", RESULTS_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("Training failed: %s", exc, exc_info=True)
        sys.exit(1)
