"""
run_inference.py
================
Entry-point: run the GraphSAGE inference pipeline on an uploaded CSV.

Usage
-----
    python run_inference.py --csv path/to/input.csv [options]

Options
-------
  --csv         Path to input CSV (required)
  --model       Path to GraphSAGE checkpoint (default: models/graphsage_best.pt)
  --window      Records per graph window (default: 100)
  --k           k-NN edges per node (default: 5)
  --batch_size  Inference batch size (default: 32)
  --out         Optional path to save prediction CSV

Output CSV columns (when --out is specified)
----------------------------------------------
  flow_index    : row index in input CSV (sorted by id)
  prediction    : 0 = Normal, 1 = Attack
  prob_normal   : P(Normal)
  prob_attack   : P(Attack)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.gnn_config import GNN_DEFAULTS
from src.inference import predict_csv, DEFAULT_MODEL_PATH
from src.logger import get_logger

log = get_logger("run_inference")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GraphSAGE inference on uploaded CSV")
    p.add_argument("--csv",        required=True,  type=Path, help="Input CSV path")
    p.add_argument("--model",      default=DEFAULT_MODEL_PATH, type=Path)
    p.add_argument("--window",     default=100,    type=int)
    p.add_argument("--k",          default=5,      type=int)
    p.add_argument("--batch_size", default=32,     type=int)
    p.add_argument("--out",        default=None,   type=Path,
                   help="Optional: save per-flow predictions to this CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 65)
    log.info("GNN Intrusion Detection — Inference Pipeline")
    log.info("  CSV    : %s", args.csv)
    log.info("  Model  : %s", args.model)
    log.info("  Window : %d  |  k : %d", args.window, args.k)
    log.info("=" * 65)

    result = predict_csv(
        csv_path    = args.csv,
        model_path  = args.model,
        window_size = args.window,
        k           = args.k,
        batch_size  = args.batch_size,
    )

    s = result["summary"]
    sep = "=" * 65
    print(f"\n{sep}")
    print("  INFERENCE SUMMARY")
    print(sep)
    print(f"  Total flows    : {s['total_flows']:>10,}")
    print(f"  Normal flows   : {s['normal_flows']:>10,}")
    print(f"  Attack flows   : {s['attack_flows']:>10,}")
    print(f"  Attack %       : {s['attack_pct']:>9.2f}%")
    print(f"  Graphs created : {result['n_graphs']:>10,}")
    print(sep + "\n")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        preds = result["predictions"]
        probs = result["probabilities"]
        out_df = pd.DataFrame({
            "flow_index":  range(len(preds)),
            "prediction":  preds,
            "prob_normal": probs[:, 0].round(6),
            "prob_attack": probs[:, 1].round(6),
        })
        out_df.to_csv(out_path, index=False)
        log.info("Predictions saved to: %s", out_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("Inference failed: %s", exc, exc_info=True)
        sys.exit(1)
