"""
run_graph_quality.py
====================
Entry-point for Phase 2b: Graph Quality Sensitivity Analysis.

Usage
-----
    python run_graph_quality.py [--window WINDOW] [--k 3 5 10]

Defaults: window=100, k values = [3, 5, 10]

What it does
------------
1. Loads both raw UNSW-NB15 CSVs.
2. For each k in {3, 5, 10} rebuilds flow-similarity graphs using the
   existing graph_builder, then collects all edge cosine-similarity values.
3. Computes structural statistics (degree, isolation) and full similarity
   distributions (percentiles, buckets).
4. Saves histograms and comparison plots to data/processed/graph_analysis/.
5. Prints a structured sensitivity table and an academic quality conclusion.

No target information (label / attack_cat) is used anywhere.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import GRAPH_WINDOW_SIZE
from src.data_loader import load_train_test
from src.graph_quality import run_quality_analysis
from src.logger import get_logger

log = get_logger("run_graph_quality")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2b: Graph quality sensitivity analysis"
    )
    parser.add_argument(
        "--window", type=int, default=GRAPH_WINDOW_SIZE,
        help=f"Window size (records per graph). Default: {GRAPH_WINDOW_SIZE}",
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=[3, 5, 10],
        help="k values to evaluate. Default: 3 5 10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 65)
    log.info("GNN Intrusion Detection — Phase 2b: Graph Quality Analysis")
    log.info("  window_size = %d  |  k values = %s", args.window, args.k)
    log.info("=" * 65)

    log.info("Loading raw datasets ...")
    train_raw, test_raw = load_train_test()

    run_quality_analysis(
        train_raw=train_raw,
        test_raw=test_raw,
        window_size=args.window,
        k_values=args.k,
    )

    print("\nOutputs saved to: data/processed/graph_analysis/\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("Graph quality analysis failed: %s", exc, exc_info=True)
        sys.exit(1)
