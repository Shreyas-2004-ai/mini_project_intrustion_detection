"""
Verification script for project_progress.ipynb
Run from gnn_intrusion_detection/ directory.
"""
import sys, json, os, tempfile
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

RESULTS_GNN  = Path("results/gnn")
RESULTS_ML   = Path("results/ml")
ABLATION_DIR = RESULTS_GNN / "ablation"
GRAPHS_DIR   = Path("data/processed/graphs")
ANALYSIS_DIR = Path("data/processed/graph_analysis")

def check(label, condition, detail=""):
    if condition:
        print(f"  OK   {label}" + (f"  ({detail})" if detail else ""))
    else:
        print(f"  FAIL {label}" + (f"  ({detail})" if detail else ""))
    return condition

results = []

# ── Section 1: Dataset ────────────────────────────────────────────────────────
from src.config import TRAIN_RAW_PATH, TEST_RAW_PATH
train_raw = pd.read_csv(TRAIN_RAW_PATH, low_memory=False)
test_raw  = pd.read_csv(TEST_RAW_PATH,  low_memory=False)
results.append(check("train shape",  train_raw.shape == (175341, 45), str(train_raw.shape)))
results.append(check("test shape",   test_raw.shape  == (82332, 45),  str(test_raw.shape)))
results.append(check("no missing",   train_raw.isnull().sum().sum() == 0))
results.append(check("no duplicates",train_raw.duplicated().sum() == 0))

# ── Section 2: Preprocessor ───────────────────────────────────────────────────
from src.preprocessor import load_pipeline, transform_split
pipeline = load_pipeline()
feats = list(pipeline.get_feature_names_out())
results.append(check("42 features", len(feats) == 42, str(len(feats))))

# ── Section 3: CT features ────────────────────────────────────────────────────
from src.config import CT_EDGE_FEATURES
results.append(check("8 ct_* features", len(CT_EDGE_FEATURES) == 8))

# ── Section 4: Graph statistics ───────────────────────────────────────────────
with open(GRAPHS_DIR / "overall_stats.json") as f:
    overall = json.load(f)
results.append(check("1754 train graphs", overall["train"]["n_graphs"] == 1754))
results.append(check("824 test graphs",   overall["test"]["n_graphs"]  == 824))

import torch
g = torch.load(GRAPHS_DIR / "train/graph_00001.pt", weights_only=False)
results.append(check("graph x dim=42",   g.x.shape[1] == 42,                 str(g.x.shape)))
results.append(check("edge_index 2-row", g.edge_index.shape[0] == 2))

# ── Section 5: Quality analysis ───────────────────────────────────────────────
with open(ANALYSIS_DIR / "quality_analysis.json") as f:
    quality = json.load(f)
results.append(check("quality keys 3/5/10", set(quality.keys()) == {"3","5","10"}))

# ── Section 6: Model config + instantiation ───────────────────────────────────
with open(RESULTS_GNN / "model_config.json") as f:
    cfg = json.load(f)
results.append(check("hidden_dim=64", cfg["hidden_dim"] == 64))
results.append(check("epochs=50",     cfg["epochs"] == 50))

from src.models.graphsage import GraphSAGE
from src.gnn_config import GNN_INPUT_DIM
model = GraphSAGE(in_channels=GNN_INPUT_DIM, hidden_dim=cfg["hidden_dim"],
                  num_classes=2, num_layers=cfg["num_layers"], dropout=cfg["dropout"])
results.append(check("13826 params", model.count_parameters() == 13826,
                     str(model.count_parameters())))

# ── Section 7: Training history ───────────────────────────────────────────────
hist = pd.read_csv(RESULTS_GNN / "training_history.csv")
results.append(check("50 training epochs", len(hist) == 50, str(len(hist))))
best_ep = hist.loc[hist["val_f1"].idxmax()]
results.append(check("val_f1 > 0.9", float(best_ep["val_f1"]) > 0.9,
                     str(round(float(best_ep["val_f1"]),4))))

# ── Section 8: GNN test metrics ───────────────────────────────────────────────
with open(RESULTS_GNN / "test_metrics.json") as f:
    gnn_m = json.load(f)
results.append(check("GNN acc=0.8817", gnn_m["accuracy"] == 0.8817))
results.append(check("GNN f1=0.8994",  gnn_m["f1"]       == 0.8994))
results.append(check("CM shape 2x2",   len(gnn_m["confusion_matrix"]) == 2))

# ── Section 9: RF metrics + importances ───────────────────────────────────────
with open(RESULTS_ML / "test_metrics.json") as f:
    rf_m = json.load(f)
results.append(check("RF acc=0.8711", rf_m["accuracy"] == 0.8711))
results.append(check("RF f1=0.8941",  rf_m["f1"]       == 0.8941))

with open(RESULTS_ML / "feature_importances.json") as f:
    imp = json.load(f)
results.append(check("42 importances", len(imp) == 42))

# ── Section 10: Ablation ──────────────────────────────────────────────────────
with open(ABLATION_DIR / "A_original_metrics.json") as f: a_m = json.load(f)
with open(ABLATION_DIR / "B_no_edges_metrics.json") as f: b_m = json.load(f)
with open(ABLATION_DIR / "C_random_metrics.json")   as f: c_m = json.load(f)
with open(ABLATION_DIR / "comparison_table.json")   as f: comp = json.load(f)
results.append(check("A f1=0.8991",  a_m["f1"] == 0.8991))
results.append(check("B f1=0.8688",  b_m["f1"] == 0.8688))
results.append(check("C f1=0.9546",  c_m["f1"] == 0.9546))
results.append(check("3 exp in table", len(comp) == 3))
for fname in ["A_original_history.json","B_no_edges_history.json","C_random_history.json"]:
    with open(ABLATION_DIR / fname) as f: h = json.load(f)
    results.append(check(f"{fname} 50 epochs", len(h) == 50))

# ── Section 13: Inference pipeline ────────────────────────────────────────────
from src.inference import predict_csv
sample = pd.read_csv(TRAIN_RAW_PATH, nrows=200)
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="") as tmp:
    sample.to_csv(tmp, index=False)
    tmp_path = tmp.name
result = predict_csv(tmp_path)
os.unlink(tmp_path)
results.append(check("inference 200 flows",
                     result["n_flows"] == 200, str(result["n_flows"])))
results.append(check("normal+attack=total",
                     result["summary"]["normal_flows"] + result["summary"]["attack_flows"] == 200))
results.append(check("probs shape (200,2)",
                     result["probabilities"].shape == (200, 2),
                     str(result["probabilities"].shape)))

# ── PNG files ─────────────────────────────────────────────────────────────────
results.append(check("training_history.png exists",
                     (RESULTS_GNN / "training_history.png").exists()))
results.append(check("confusion_matrix.png exists",
                     (RESULTS_GNN / "confusion_matrix.png").exists()))

# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(results)
total  = len(results)
print()
print(f"Result: {passed}/{total} checks passed")
if passed == total:
    print("NOTEBOOK VERIFICATION: ALL CHECKS PASSED")
else:
    print("NOTEBOOK VERIFICATION: SOME CHECKS FAILED")
