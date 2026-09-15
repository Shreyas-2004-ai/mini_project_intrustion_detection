"""
run_graph_construction.py
=========================
Entry-point for Phase 2: Flow-Similarity Graph Construction.

Usage
-----
    python run_graph_construction.py [--window WINDOW] [--k K]

Defaults are read from src/config.py (GRAPH_WINDOW_SIZE, GRAPH_K).

What it does
------------
1. Loads raw UNSW-NB15 training and testing CSVs.
2. Applies the preprocessing pipeline fitted in Phase 1 (no re-fitting).
3. Splits each dataset into non-overlapping temporal windows.
4. Within each window computes cosine similarity over the 8 ct_* features.
5. Constructs a k-NN flow-similarity graph per window.
6. Saves graphs as .pt files under data/processed/graphs/{train,test}/.
7. Validates graphs (leakage check, shape checks).
8. Computes and saves per-graph and aggregate statistics.
9. Prints a final dataset report.

IMPORTANT: edges represent flow-similarity relationships, NOT IP communication
links.  The dataset contains no srcip/dstip columns.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import GRAPH_WINDOW_SIZE, GRAPH_K
from src.graph_builder import run_graph_construction
from src.graph_validator import dataset_stats, print_stats_report, save_stats, validate_no_leakage
from src.logger import get_logger

log = get_logger("run_graph_construction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: UNSW-NB15 flow-similarity graph construction"
    )
    parser.add_argument(
        "--window", type=int, default=GRAPH_WINDOW_SIZE,
        help=f"Window size (records per graph). Default: {GRAPH_WINDOW_SIZE}",
    )
    parser.add_argument(
        "--k", type=int, default=GRAPH_K,
        choices=[3, 5, 10],
        help=f"k for k-NN edge construction. Default: {GRAPH_K}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 65)
    log.info("GNN Intrusion Detection — Phase 2: Graph Construction")
    log.info("  window_size = %d  |  k = %d", args.window, args.k)
    log.info("=" * 65)

    # ── 1. Build graphs ────────────────────────────────────────────────────
    log.info("Step 1/3 — Building flow-similarity graphs ...")
    train_graphs, test_graphs = run_graph_construction(
        window_size=args.window,
        k=args.k,
    )

    # ── 2. Validate ────────────────────────────────────────────────────────
    log.info("Step 2/3 — Validating graphs ...")
    validate_no_leakage(train_graphs, feature_cols=[])  # structural check
    validate_no_leakage(test_graphs,  feature_cols=[])

    # ── 3. Statistics ──────────────────────────────────────────────────────
    log.info("Step 3/3 — Computing and saving statistics ...")
    train_stats = dataset_stats(train_graphs, "train")
    test_stats  = dataset_stats(test_graphs,  "test")
    save_stats(train_stats, test_stats)

    print_stats_report(train_stats, test_stats)

    print("\n" + "=" * 65)
    print("  GRAPH CONSTRUCTION COMPLETE")
    print("=" * 65)
    print(f"  Training graphs : {len(train_graphs):>6,}  →  data/processed/graphs/train/")
    print(f"  Testing graphs  : {len(test_graphs):>6,}  →  data/processed/graphs/test/")
    print(f"  Stats CSV       : data/processed/graphs/train_graph_stats.csv")
    print(f"                    data/processed/graphs/test_graph_stats.csv")
    print(f"  Overall JSON    : data/processed/graphs/overall_stats.json")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("Graph construction failed: %s", exc, exc_info=True)
        sys.exit(1)
