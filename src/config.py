"""
Central configuration: paths, column definitions, and constants.
All values are derived from actual inspection of the UNSW-NB15 dataset.
"""

from pathlib import Path

# ── Project root (two levels up from this file: src/ -> gnn_intrusion_detection/) ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ──
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_RAW_PATH = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_RAW_PATH  = RAW_DIR / "UNSW_NB15_testing-set.csv"

TRAIN_PROCESSED_PATH = PROCESSED_DIR / "train_processed.csv"
TEST_PROCESSED_PATH  = PROCESSED_DIR / "test_processed.csv"

PREPROCESSOR_PATH = PROCESSED_DIR / "preprocessor.pkl"

# ── Column definitions (verified from actual dataset inspection) ──

# Row identifier — must be dropped before any ML step
IDENTIFIER_COLS = ["id"]

# Target columns — never used as input features
TARGET_COLS = ["label", "attack_cat"]

# Categorical features (dtype object in the raw CSV)
CATEGORICAL_COLS = ["proto", "service", "state"]

# All remaining columns are numerical features
# (derived at runtime in data_loader.py to stay in sync with the real CSV)
# Excluded: IDENTIFIER_COLS + TARGET_COLS + CATEGORICAL_COLS

# attack_cat classes present in UNSW-NB15
ATTACK_CATEGORIES = [
    "Normal", "Generic", "Exploits", "Fuzzers", "DoS",
    "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms",
]

# Expected dataset shape (used for validation warnings, not hard failures)
EXPECTED_TRAIN_ROWS = 175_341
EXPECTED_TEST_ROWS  =  82_332
EXPECTED_COLS       =  45

# ── Graph construction settings ───────────────────────────────────────────────

# Window size: number of consecutive flow records per graph
GRAPH_WINDOW_SIZE = 100

# k for k-nearest-neighbour edge construction (configurable; default = 5)
# Must be one of the supported values: 3, 5, 10
GRAPH_K = 5

# The 8 ct_* features used ONLY for cosine-similarity edge construction.
# These encode neighbourhood statistics and are the closest proxy for
# network topology available in UNSW-NB15 (no srcip/dstip present).
# They are also kept as node features — they encode structural context.
CT_EDGE_FEATURES = [
    "ct_srv_src",
    "ct_srv_dst",
    "ct_src_ltm",
    "ct_dst_ltm",
    "ct_dst_src_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_state_ttl",
]

# Output directories for serialised PyG graph datasets
GRAPHS_DIR       = PROCESSED_DIR / "graphs"
TRAIN_GRAPHS_DIR = GRAPHS_DIR / "train"
TEST_GRAPHS_DIR  = GRAPHS_DIR / "test"

# Statistics output paths
TRAIN_GRAPH_STATS_PATH = GRAPHS_DIR / "train_graph_stats.csv"
TEST_GRAPH_STATS_PATH  = GRAPHS_DIR / "test_graph_stats.csv"
OVERALL_STATS_PATH     = GRAPHS_DIR / "overall_stats.json"
