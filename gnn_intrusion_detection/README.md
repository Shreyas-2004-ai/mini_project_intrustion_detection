# Graph Neural Network-Based Intrusion Detection

Master's-level mini project — Phase 1: Data Inspection & Preprocessing | Phase 2: Graph Construction

---

## Dataset

**UNSW-NB15** — network traffic records labelled with attack categories.

| File | Rows | Columns |
|------|------|---------|
| `data/raw/UNSW_NB15_training-set.csv` | 175,341 | 45 |
| `data/raw/UNSW_NB15_testing-set.csv`  |  82,332 | 45 |

Columns confirmed by direct inspection (no IP address columns exist in this dataset):

| Role | Columns |
|------|---------|
| Identifier | `id` |
| Targets | `label`, `attack_cat` |
| Categorical | `proto`, `service`, `state` |
| Numerical | all remaining 40 columns |

---

## Project Structure

```
gnn_intrusion_detection/
├── data/
│   ├── raw/                        # Original CSV files (never modified)
│   └── processed/
│       ├── train_processed.csv
│       ├── test_processed.csv
│       ├── preprocessor.pkl        # Fitted sklearn pipeline
│       └── graphs/
│           ├── train/              # graph_00000.pt … graph_01753.pt
│           ├── test/               # graph_00000.pt … graph_00823.pt
│           ├── train_graph_stats.csv
│           ├── test_graph_stats.csv
│           └── overall_stats.json
├── src/
│   ├── config.py                   # Paths, column definitions, graph constants
│   ├── logger.py                   # Shared logging factory
│   ├── data_loader.py              # CSV loading + column-role helpers
│   ├── inspector.py                # Inspection, statistics, report printing
│   ├── preprocessor.py             # Sklearn pipeline: fit, transform, save/load
│   ├── graph_builder.py            # Flow-similarity graph construction (PyG)
│   └── graph_validator.py          # Graph statistics and leakage validation
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_graph_builder.py
│   └── test_graph_validator.py
├── run_preprocessing.py            # Phase 1 entry-point
├── run_graph_construction.py       # Phase 2 entry-point
└── README.md
```

---

## Setup

Install dependencies (Python 3.8+):

```bash
pip install pandas numpy scikit-learn pytest torch torch_geometric
```

---

## Stage 1 — Data Inspection & Preprocessing

### Run the full pipeline

From the `gnn_intrusion_detection/` directory:

```bash
python run_preprocessing.py
```

This will:
1. Load and validate both raw CSV files.
2. Inspect each dataset (shape, dtypes, missing values, duplicates, target distributions, leakage check).
3. Print a structured report to the console.
4. Fit a `StandardScaler` + `OrdinalEncoder` pipeline on the training set only.
5. Transform both splits with the fitted pipeline.
6. Save results to `data/processed/`.

### Run tests

```bash
python -m pytest tests/ -v
```

---

## Preprocessing Design

### Numerical features (40 columns)
Transformed with `StandardScaler` — zero mean, unit variance.
Handles the wide range of magnitudes present in network traffic metrics.

### Categorical features (`proto`, `service`, `state`)
Transformed with `OrdinalEncoder`.
Unseen category values at inference time are encoded as `-1` (no crash).

### What is never touched
- `id` — dropped (identifier, not a feature).
- `label`, `attack_cat` — passed through unchanged (targets, not input features).
- The original raw CSV files — the pipeline always reads from `data/raw/` and writes only to `data/processed/`.

### Reproducibility
The fitted `ColumnTransformer` is serialised to `data/processed/preprocessor.pkl`.
Load it for any future training/inference run:

```python
from src.preprocessor import load_pipeline
pipeline = load_pipeline()
X = pipeline.transform(new_df)
```

---

## Target Column Notes

- `label` — binary: `0` = Normal, `1` = Attack.
- `attack_cat` — multiclass: 10 categories (`Normal`, `Generic`, `Exploits`, `Fuzzers`, `DoS`, `Reconnaissance`, `Analysis`, `Backdoor`, `Shellcode`, `Worms`).
- No missing values observed in either target column in the UNSW-NB15 dataset.

---

## Data Leakage Policy

`label` and `attack_cat` are **never** passed into the `ColumnTransformer`.
The `remainder="drop"` setting in the pipeline ensures that `id` and targets
cannot accidentally appear as features.

---

## Phase 2 — Flow-Similarity Graph Construction

### Important terminology

These graphs are **flow-similarity graphs**.  Edges connect flow records that
have similar `ct_*` neighbourhood statistics.  They do **not** represent real
IP communication links — the UNSW-NB15 dataset contains no `srcip` or `dstip`
columns and none are invented here.

### Graph definition

| Element | Definition |
|---------|-----------|
| Node | One flow record (one CSV row) within a 100-record window |
| Node features | All 42 input features (39 numerical + 3 encoded categorical) |
| Edge | Directed edge from node *i* to its k nearest neighbours by cosine similarity over the 8 `ct_*` features |
| Edge weight | Cosine similarity score in `ct_*` feature space (stored in `edge_attr`) |
| Node label `y` | Binary `label` column (0 = Normal, 1 = Attack) |
| Node label `attack_cat` | 10-class attack category (integer encoded, stored separately) |

### The 8 ct_* edge features

| Feature | Meaning |
|---------|---------|
| `ct_srv_src` | Connections from same source to same service (last 100) |
| `ct_srv_dst` | Connections to same destination with same service (last 100) |
| `ct_src_ltm` | Connections from same source (last 100) |
| `ct_dst_ltm` | Connections to same destination (last 100) |
| `ct_dst_src_ltm` | Connections between same source-destination pair (last 100) |
| `ct_src_dport_ltm` | Connections from same source to same destination port (last 100) |
| `ct_dst_sport_ltm` | Connections to same destination from same source port (last 100) |
| `ct_state_ttl` | Connections with same protocol, state, TTL (last 100) |

### k-NN construction

For each node in a window, the k most similar nodes (by cosine similarity in
`ct_*` space) are connected as directed edges.  Self-loops are excluded.
k is configurable: supported values are **3, 5, 10**.  Default is 5 (set in
`src/config.py` as `GRAPH_K`).

### Leakage prevention

- `label` and `attack_cat` are **never** used when computing similarity or constructing edges.
- The preprocessing pipeline is fitted on training data only and only loaded here.
- Training and testing windows are built from their respective CSVs independently.
- No records cross the train/test boundary at any stage.

### Graph statistics (default: window=100, k=5)

| | Training | Testing |
|--|--|--|
| Graphs | 1,754 | 824 |
| Total nodes | 175,341 | 82,332 |
| Attack nodes | 119,341 (68.1%) | 45,332 (55.1%) |
| Avg edges per graph | 499.8 | 499.6 |
| Avg degree per node | 5.0 | 5.0 |
| Isolated nodes | 0 | 0 |
| Avg edge weight | 0.980 | 0.985 |

### Run graph construction

Phase 1 (preprocessing) must be run first.

```bash
# Default: window=100, k=5
python run_graph_construction.py

# Custom window and k
python run_graph_construction.py --window 100 --k 3
python run_graph_construction.py --window 100 --k 10
```

Output is saved to `data/processed/graphs/`.

### PyG Data object fields

Each `.pt` file is a `torch_geometric.data.Data` object with:

```
graph.x           — float32 tensor  (N × 42)   node features
graph.edge_index  — long tensor     (2 × E)    k-NN connectivity
graph.edge_attr   — float32 tensor  (E × 1)    cosine similarity weights
graph.y           — long tensor     (N,)        binary node labels
graph.attack_cat  — long tensor     (N,)        multiclass labels (int encoded)
graph.num_nodes   — int             N
```

### Load graphs for downstream use

```python
from src.graph_builder import load_graphs
from src.config import TRAIN_GRAPHS_DIR, TEST_GRAPHS_DIR

train_graphs = load_graphs(TRAIN_GRAPHS_DIR)
test_graphs  = load_graphs(TEST_GRAPHS_DIR)
```
