# PROJECT STATUS — GNN-Based Network Intrusion Detection
## Master's Level Mini Project | Complete Documentation

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Dataset](#3-dataset)
4. [Technologies Used](#4-technologies-used)
5. [Complete Pipeline](#5-complete-pipeline)
6. [Phase 1 — Data Inspection and Preprocessing](#6-phase-1--data-inspection-and-preprocessing)
7. [Phase 2 — Graph Construction](#7-phase-2--graph-construction)
8. [Important Graph Design Limitation](#8-important-graph-design-limitation)
9. [Phase 3 — GraphSAGE Model](#9-phase-3--graphsage-model)
10. [Original GraphSAGE Results](#10-original-graphsage-results)
11. [Random Forest Baseline](#11-random-forest-baseline)
12. [Ablation Study](#12-ablation-study)
13. [Why Experiment C Outperformed A](#13-why-experiment-c-outperformed-a)
14. [Ablation Validation](#14-ablation-validation)
15. [Reproducibility and CUDA Note](#15-reproducibility-and-cuda-note)
16. [Inference Pipeline](#16-inference-pipeline)
17. [Testing Summary](#17-testing-summary)
18. [Current Project Status](#18-current-project-status)
19. [What the Web Platform Will Eventually Do](#19-what-the-web-platform-will-eventually-do)
20. [Current Limitations](#20-current-limitations)
21. [What Is Left for the Mini Project](#21-what-is-left-for-the-mini-project)
22. [Presentation Cheat Sheet](#22-presentation-cheat-sheet)
23. [One-Page Project Summary](#23-one-page-project-summary)

---

## 1. Project Overview

### What is this project?

This is a master's-level mini project that investigates using **Graph Neural Networks (GNN)** to detect network intrusions — meaning cyberattacks in network traffic data.

### The problem being solved

Every time a computer communicates over a network, it generates **network flow records** — small summaries of that communication (how long it lasted, how many bytes were sent, which protocol was used, etc.). Attackers also generate these flows. The goal is to automatically classify each flow as **Normal (safe)** or **Attack (malicious)**.

### Why intrusion detection is needed

Cyberattacks are constantly happening on networks — DoS attacks, backdoors, malware communication, reconnaissance scanning, etc. Manual inspection is impossible at scale. Automated intrusion detection systems are needed to monitor and flag suspicious traffic in near-real-time.

### What our system does

Our system:
1. Takes a **CSV file** of network flow records as input (the UNSW-NB15 dataset, or any compatible CSV)
2. Preprocesses the data (scales numbers, encodes categories)
3. Constructs **flow-similarity graphs** — networks of connected flow records
4. Feeds those graphs through a **GraphSAGE neural network** that learns to classify each flow
5. Outputs a prediction for every flow: **Normal (0)** or **Attack (1)**

### IMPORTANT: Input is CSV, NOT real-time packet capture

This project does **NOT** capture or sniff live network packets. It works entirely on pre-recorded CSV files that already contain network flow summaries. This is a research/offline analysis system, not a live network sensor.

### Expected future web-platform workflow

The system is being designed to eventually work as a web application:

```
User opens web browser
        ↓
Uploads a CSV file of network flows
        ↓
Backend server receives the file
        ↓
Preprocessing pipeline runs automatically
        ↓
Graph construction converts flows into graphs
        ↓
Trained GraphSAGE model runs inference
        ↓
Prediction results returned to user
        ↓
Dashboard shows: total flows, normal count, attack count, attack %, confidence scores
```

> NOTE: The web frontend (React/Flask/API) is NOT yet implemented. Only the ML backend pipeline is currently built and working.

---

## 2. Problem Statement

### Simple version

Traditional machine learning for intrusion detection treats each network flow as a completely separate, independent data point. It looks at one row at a time and asks "is this attack or normal?" — ignoring the fact that network flows that happen close together in time often have related patterns.

Our project investigates: **can we do better by also looking at the relationships between nearby flows?**

### How we investigate this

We model a group of 100 consecutive network flows as a **graph** — a mathematical structure of nodes and edges:

- **Node** = one network flow record (one row from the CSV)
- **Edge** = a connection between two flows that have similar neighbourhood statistics (measured using cosine similarity on 8 special `ct_*` features)
- **Node features** = the 42 preprocessed features describing that flow
- **Node label** = 0 (Normal) or 1 (Attack)

### What GraphSAGE does

**GraphSAGE** is a Graph Neural Network (GNN) algorithm. Instead of looking at just one node's features, it also aggregates ("collects and summarises") the features of connected neighbouring nodes, then makes a prediction based on **both** the node's own features AND information from its neighbourhood.

This is the key difference from regular ML: **a flow's prediction is influenced by what is happening in nearby flows**.

### Why we investigate graph structure

The hypothesis is that attack flows may cluster together in graphs — reconnaissance probes connect to other probes, DoS attack flows connect to other DoS flows — while normal flows have different neighbourhood patterns. By capturing these relationships explicitly, a GNN may be better able to distinguish attacks from normal traffic.

---

## 3. Dataset

### What dataset we used

**UNSW-NB15** — a publicly available, research-standard dataset of labelled network traffic flows created by the Australian Centre for Cyber Security (UNSW, Canberra).

### Dataset statistics

| Property | Value |
|----------|-------|
| Training set rows | 175,341 |
| Testing set rows | 82,332 |
| Total columns | 45 |
| Missing values | 0 |
| Duplicate rows | 0 |

### Column breakdown

| Role | Columns | Count |
|------|---------|-------|
| Identifier | `id` | 1 |
| Binary target | `label` | 1 |
| Multiclass target | `attack_cat` | 1 |
| Categorical features | `proto`, `service`, `state` | 3 |
| Numerical features | all other remaining columns | 39 |
| **Total** | | **45** |

### The target columns explained

**`label`** — Binary classification:
- `0` = Normal traffic
- `1` = Attack traffic

**`attack_cat`** — Multiclass attack category (10 classes):

| Category | Meaning |
|----------|---------|
| Normal | Legitimate traffic |
| Generic | Generic attack pattern |
| Exploits | Exploitation of vulnerabilities |
| Fuzzers | Fuzzing/random input attacks |
| DoS | Denial-of-Service attacks |
| Reconnaissance | Network scanning/probing |
| Analysis | Protocol analysis attacks |
| Backdoor | Backdoor communication |
| Shellcode | Shellcode injection |
| Worms | Worm propagation |

### Class distribution (from actual dataset inspection)

**Training set (175,341 rows):**
- Normal (label=0): ~56,000 rows (~32%)
- Attack (label=1): ~119,341 rows (~68%)

**Testing set (82,332 rows):**
- Normal (label=0): 37,000 rows (44.9%)
- Attack (label=1): 45,332 rows (55.1%)

> Note: The training set is significantly imbalanced — attacks are roughly twice as common as normal flows. This is handled in training using weighted loss.

### Important dataset limitation

The UNSW-NB15 CSV files used in this project **do not contain source IP (`srcip`) or destination IP (`dstip`) columns** in the selected representation. Therefore:

- We **cannot** build an IP-to-IP communication graph
- We **do not** know which machines were talking to which
- Instead, we build a **flow-similarity graph** based on statistical similarity between flows
- The 8 `ct_*` neighbourhood-count features serve as the proxy for "how related are two flows?"

This is a real limitation and is acknowledged in all our documentation and the ablation report.

---

## 4. Technologies Used

| Technology | Version/Type | Why We Use It |
|------------|-------------|---------------|
| **Python** | 3.8+ | Main programming language — industry standard for ML |
| **pandas** | library | Loading and manipulating CSV data efficiently |
| **NumPy** | library | Fast numerical array operations (cosine similarity, matrix math) |
| **scikit-learn** | library | StandardScaler, OrdinalEncoder, preprocessing pipeline, Random Forest, metrics |
| **PyTorch** | deep learning framework | Tensor operations, model training, GPU acceleration |
| **PyTorch Geometric (PyG)** | GNN library | Provides `SAGEConv` (GraphSAGE layer), `Data` graph objects, `DataLoader` |
| **GraphSAGE** | model architecture | GNN algorithm for node-level classification using neighbourhood aggregation |
| **CUDA** | GPU acceleration | Used during training (confirmed in model_config.json and ablation report — "Device: CUDA") |
| **Matplotlib** | plotting library | Training history plots, confusion matrix heatmaps, graph quality distribution plots |
| **pytest** | testing framework | Automated unit and integration tests for all modules |
| **CSV** | file format | Input data format — network flow records from UNSW-NB15 |
| **JSON** | file format | Saving/loading metrics, model configuration, statistics |
| **pickle** | serialisation | Saving the fitted sklearn preprocessing pipeline to disk |
| **Kiro IDE** | development environment | AI-assisted development environment used to build and document the project |

> Technologies NOT in this repository (not yet implemented): React, Flask, FastAPI, Django, MongoDB, PostgreSQL, REST API, Docker, any web frontend framework.

---

## 5. Complete Pipeline

The complete data flow from raw CSV to final prediction:

```
CSV dataset (raw)
       ↓
  [Phase 1] Data Loading & Inspection
       ↓  src/data_loader.py, src/inspector.py
  Shape check, column validation, target distribution analysis
       ↓
  [Phase 1] Preprocessing
       ↓  src/preprocessor.py
  StandardScaler (39 numerical) + OrdinalEncoder (3 categorical)
  → 42-feature processed DataFrame
       ↓
  [Phase 2] Graph Construction
       ↓  src/graph_builder.py
  Window 100 rows → compute cosine similarity → k=5 nearest neighbours
  → PyG Data objects (1,754 train graphs, 824 test graphs)
       ↓
  [Phase 2] Graph Validation & Quality Analysis
       ↓  src/graph_validator.py, src/graph_quality.py
  Structural checks, leakage validation, k-sensitivity analysis
       ↓
  [Phase 3] GraphSAGE Training
       ↓  src/models/graphsage.py, src/training.py
  SAGEConv × 2 + Linear head, 50 epochs, Adam optimizer
  Best model saved by validation F1
       ↓
  [Phase 3] Official Test Evaluation
       ↓  src/evaluation.py
  Best model evaluated ONCE on 824 test graphs
  → Accuracy, Precision, Recall, F1, Confusion Matrix, Plots
       ↓
  [Baseline] Random Forest
       ↓  src/models/random_forest.py
  Same 42 features, no graph structure
  → Comparison point for GraphSAGE
       ↓
  [Phase 4] Ablation Study
       ↓  src/ablation.py
  3 experiments × identical training → isolate graph structure contribution
       ↓
  [Inference] Prediction on new CSV
       ↓  src/inference.py
  Load CSV → preprocess → build graphs → trained model → predictions
```

Every stage reads from the previous stage's output and writes to its own output location. No stage modifies raw data.

---

## 6. Phase 1 — Data Inspection and Preprocessing

### What this phase does

Phase 1 takes the raw UNSW-NB15 CSV files, inspects them thoroughly, transforms the features into a suitable format for machine learning, and saves the results.

**Entry-point script:** `run_preprocessing.py`
**Command:** `python run_preprocessing.py`

### Files and their roles

#### `src/config.py`
The **central configuration file** for the entire project. Contains:
- All file paths (raw data, processed data, graph directories, model paths)
- Column definitions: which columns are identifiers, targets, categorical, numerical
- Graph construction constants: `GRAPH_WINDOW_SIZE = 100`, `GRAPH_K = 5`
- The 8 `ct_*` features used for edge construction
- Expected dataset shapes for validation

Think of it as a single source of truth — any other file that needs a path or constant imports it from here.

#### `src/logger.py`
A shared logging factory. Instead of using raw `print()` statements, all modules get consistent, timestamped log messages. Not visible to the end user, but important for debugging during development.

#### `src/data_loader.py`
Loads CSV files and validates them:
- Checks that the file exists
- Verifies the expected 45 columns are present
- Checks that required columns (`id`, `label`, `attack_cat`, `proto`, `service`, `state`) exist
- Returns the raw DataFrame **unchanged** — no modifications at this stage
- Also provides `get_numerical_cols()` which dynamically derives the list of 39 numerical feature columns by excluding identifiers, targets, and categorical columns

#### `src/inspector.py`
Performs a full dataset analysis and prints a structured report:
- Shape (rows × columns)
- Duplicate rows count
- Missing value count per column
- Data types
- Target distribution (how many normal vs attack rows)
- Consistency check: every row with `label=0` should have `attack_cat="Normal"`
- Leakage check: warns if any feature column name contains "label", "attack", "class", "category", or "target"

This is run before any ML, purely to understand the data.

#### `src/preprocessor.py`
Builds and applies the feature transformation pipeline:

**Numerical features (39 columns):**
- Transformed with `StandardScaler`
- Result: each column has mean ≈ 0 and standard deviation ≈ 1
- Why: network traffic features have wildly different scales (packet counts in the hundreds, byte sizes in millions). Without scaling, large-magnitude features dominate distance calculations.

**Categorical features (`proto`, `service`, `state`):**
- Transformed with `OrdinalEncoder`
- Result: string categories converted to integer codes (e.g., "tcp" → 0, "udp" → 1)
- Unseen category values at inference time are encoded as `-1` (no crash)
- Why: ML models require numerical input

**What is NEVER transformed:**
- `id` — dropped entirely (it is just a row number, not a feature)
- `label` — preserved as-is (this is the prediction target)
- `attack_cat` — preserved as-is (this is the multiclass target)

**Final processed output:**
- 42 feature columns (39 scaled numerical + 3 ordinal categorical)
- 2 target columns (`label`, `attack_cat`) passed through unchanged
- Total columns in processed CSV: 44

**Serialisation:**
The fitted pipeline is saved to `data/processed/preprocessor.pkl`. This means the exact same transformations used for training can be applied to any future CSV at inference time — without re-fitting. This is the correct approach: you NEVER refit scaling on test/inference data.

#### `run_preprocessing.py`
The entry-point script that ties it all together:
1. Loads both raw CSV files
2. Runs `inspect_dataset()` on both and prints reports
3. Runs `run_preprocessing()` which fits on train, transforms both, saves CSVs + pipeline
4. Prints a summary

**Output files:**
- `data/processed/train_processed.csv` — 175,341 rows × 44 columns
- `data/processed/test_processed.csv` — 82,332 rows × 44 columns
- `data/processed/preprocessor.pkl` — fitted sklearn ColumnTransformer

### Tests for Phase 1

| Test file | Tests |
|-----------|-------|
| `tests/test_data_loader.py` | 14 tests — file loading, column validation, error handling |
| `tests/test_preprocessor.py` | 16 tests — pipeline building, scaling, encoding, persistence |

---

## 7. Phase 2 — Graph Construction

### What this phase does

Phase 2 takes the preprocessed CSV data and converts it into **graphs** — mathematical structures that can be processed by a Graph Neural Network.

**Entry-point script:** `run_graph_construction.py`
**Command:** `python run_graph_construction.py --window 100 --k 5`

### The key design choice: windows

The dataset has 175,341 training rows. We cannot make one giant graph from all of them — that would be computationally infeasible and wouldn't represent temporal locality.

Instead, we divide the rows into **non-overlapping windows of 100 consecutive records** (sorted by `id` to preserve temporal order). Each window becomes one graph.

- 175,341 ÷ 100 = 1,754 full windows + 1 partial window of 41 rows = **1,754 graphs** (the partial window is included, making 1,754 total because 175,341 ÷ 100 = 1753 full + 1 partial = 1754)
- 82,332 ÷ 100 = 823 full windows + 1 partial window of 32 rows = **824 graphs**

### What one graph represents

One graph = a snapshot of 100 consecutive network flow records from the dataset.

Think of it like a "neighbourhood" — 100 flows that happened around the same time, connected to each other based on how similar their network behaviour statistics are.

### Graph structure in detail

#### Nodes
- Each node = one flow record (one row from the preprocessed CSV)
- Node features (`graph.x`) = 42 preprocessed feature values for that flow
- Shape: `(100, 42)` for a full window

#### The 8 `ct_*` features used for edge construction

These 8 features count how many similar connections happened in the last 100 flows. They are the closest proxy for "relatedness" available in this dataset:

| Feature | What it counts |
|---------|---------------|
| `ct_srv_src` | Connections from same source to same service (last 100) |
| `ct_srv_dst` | Connections to same destination with same service (last 100) |
| `ct_src_ltm` | Connections from same source (last 100) |
| `ct_dst_ltm` | Connections to same destination (last 100) |
| `ct_dst_src_ltm` | Connections between same source-destination pair (last 100) |
| `ct_src_dport_ltm` | Connections from same source to same destination port (last 100) |
| `ct_dst_sport_ltm` | Connections to same destination from same source port (last 100) |
| `ct_state_ttl` | Connections with same protocol, state, TTL (last 100) |

These 8 columns are also kept as node features (they appear in the 42-feature vector), not just used for edge construction.

#### Cosine similarity

To decide which nodes should be connected, we compute **cosine similarity** between every pair of nodes using only the 8 `ct_*` features above.

Cosine similarity measures the angle between two vectors:
- Value of **1.0** = identical direction = very similar flows
- Value of **0.0** = perpendicular = unrelated
- Value of **-1.0** = opposite directions = very different

Most edges in our graphs have similarity ≥ 0.98, because consecutive flows within a 100-record window naturally share similar network context.

#### k-Nearest Neighbour (k-NN) edges with k=5

For each node, we find its **5 most similar neighbours** (highest cosine similarity) and draw a directed edge from that node to each of those 5 neighbours. Self-loops are excluded.

Result: **every node has exactly 5 outgoing edges** (for full windows).

Why k=5? It was selected as a reasonable default — sufficient for GNN message passing, not too computationally expensive. k=3 and k=10 were also tested in the graph quality analysis.

#### Edge index and edge weights
- `graph.edge_index`: a (2 × E) tensor storing source and destination node IDs for each edge
- `graph.edge_attr`: a (E × 1) tensor storing the cosine similarity score for each edge
- For a 100-node window with k=5: E = 500 edges

#### Labels (stored separately, NEVER used for edge construction)
- `graph.y`: (N,) tensor of binary labels — 0=Normal, 1=Attack
- `graph.attack_cat`: (N,) tensor of integer-encoded attack categories

### Actual graph statistics (window=100, k=5)

| Statistic | Training | Testing |
|-----------|----------|---------|
| Number of graphs | 1,754 | 824 |
| Total nodes | 175,341 | 82,332 |
| Attack nodes | 119,341 (68.1%) | 45,332 (55.1%) |
| Normal nodes | ~56,000 (31.9%) | 37,000 (44.9%) |
| Total edges | 876,705 | 411,660 |
| Average edges per graph | ~499.8 | ~499.6 |
| Average degree per node | ~5.0 | ~5.0 |
| Isolated nodes | 0 | 0 |
| Average edge weight (cosine sim) | ~0.980 | ~0.985 |

> Why ~499 edges per graph instead of exactly 500? Because the last partial window has fewer than 100 nodes, so it has fewer edges. The full windows each have exactly 100 × 5 = 500 edges.

### Graph quality sensitivity analysis

To validate our choice of k=5, we ran a sensitivity analysis for k=3, 5, and 10.

| k | Split | Graphs | Total Edges | Avg Degree | Mean Similarity | Isolated Nodes |
|---|-------|--------|-------------|-----------|----------------|----------------|
| 3 | Train | 1,754 | ~526,023 | 3.0 | ~0.985 | 0 |
| 3 | Test | 824 | ~246,996 | 3.0 | ~0.990 | 0 |
| 5 | Train | 1,754 | ~876,705 | 5.0 | ~0.980 | 0 |
| 5 | Test | 824 | ~411,660 | 5.0 | ~0.985 | 0 |
| 10 | Train | 1,754 | ~1,753,410 | 10.0 | ~0.978 | 0 |
| 10 | Test | 824 | ~823,320 | 10.0 | ~0.983 | 0 |

Key observations:
- No isolated nodes across any k value (every node is connected)
- Mean similarity stays ≈ 0.98 regardless of k (all graphs draw from the same pairwise similarity pool)
- The similarity is high because flows within a 100-record temporal window naturally share similar context

Output plots saved to `data/processed/graph_analysis/`:
- `similarity_distribution_k3.png`, `similarity_distribution_k5.png`, `similarity_distribution_k10.png`
- `similarity_comparison.png` — comparison box plots and bucket charts

### PyG Data object format

Each saved `.pt` file is a `torch_geometric.data.Data` object:

```python
graph.x           # float32 tensor (N × 42)  — node features
graph.edge_index  # long tensor    (2 × E)   — k-NN connectivity
graph.edge_attr   # float32 tensor (E × 1)   — cosine similarity weights
graph.y           # long tensor    (N,)       — binary node labels
graph.attack_cat  # long tensor    (N,)       — multiclass labels (int)
graph.num_nodes   # int             N         — total nodes
```

### Files implementing Phase 2

| File | Role |
|------|------|
| `src/graph_builder.py` | Core construction logic: windowing, cosine similarity, k-NN, Data objects |
| `src/graph_validator.py` | Per-graph stats, leakage validation, CSV/JSON output |
| `src/graph_quality.py` | Sensitivity analysis for k=3,5,10 with plots |
| `run_graph_construction.py` | Entry-point script |
| `run_graph_quality.py` | Entry-point for quality analysis |

### Tests for Phase 2

| Test file | Tests |
|-----------|-------|
| `tests/test_graph_builder.py` | ~20 tests — node count, feature dimensions, edge count, leakage, save/load |
| `tests/test_graph_validator.py` | ~14 tests — stats correctness, leakage detection, file output |

---

## 8. Important Graph Design Limitation

### We are NOT building an IP communication graph

A common assumption about network intrusion detection with graphs is that nodes represent IP addresses (machines) and edges represent network connections between them. **Our graph does NOT work this way.**

### We are building a flow-similarity graph

Here is the exact distinction:

| Aspect | IP Communication Graph | Our Flow-Similarity Graph |
|--------|----------------------|--------------------------|
| Node = | An IP address (machine) | One network flow record (one row) |
| Edge = | A network connection between two machines | Two flows with similar `ct_*` statistics |
| Edge source | `srcip` and `dstip` columns | Cosine similarity over 8 `ct_*` features |
| Available? | NO — not in our CSV | YES — computed from the data |

### Why this matters for your presentation

If your guide asks "does your graph show which machines are communicating?" — the answer is **no**. The dataset we use does not provide source/destination IP addresses in the selected CSV representation. Our graph connects flows that *behave similarly*, not flows that *communicate with each other*.

This is clearly stated in all our code, README, and result reports. It is a known limitation, not an oversight.

### Why the similarity graph is still meaningful

Even though it's not an IP graph, the similarity graph still captures useful structure:
- Flows from the same attack campaign tend to have similar `ct_*` statistics
- Flows from the same protocol or service cluster together
- GraphSAGE can potentially learn "if my neighbours look like attack traffic, I might also be attack traffic"

---

## 9. Phase 3 — GraphSAGE Model

### What is GraphSAGE? (Simple explanation)

GraphSAGE stands for **Graph SAmple and aggreGatE**. It is a type of neural network that works on graphs.

Imagine you are trying to decide if a person is a criminal. Instead of only looking at that person's own features (height, age, etc.), you also look at their **friends and neighbours**. If all their neighbours are criminals, that's a strong signal. GraphSAGE does the same thing for nodes in a graph.

### What is message passing?

Message passing is the process by which a node collects information from its neighbours:

1. Each neighbour sends a **message** (their feature vector)
2. The node **aggregates** (averages) all received messages
3. The node **combines** its own features with the aggregated neighbourhood features
4. A neural network layer transforms this combined vector into a new representation

In our model, this happens twice (2 GraphSAGE layers), so each node can "see" up to 2 hops away.

### What happens during a prediction for one node

1. Node has 42 input features
2. SAGEConv layer 1: average the 42 features of all 5 neighbours, concatenate with own features → feed through linear layer → ReLU activation → dropout
3. SAGEConv layer 2: same process again at 64-dimensional hidden representation
4. Linear classifier head: 64 → 2 outputs (logit scores for Normal and Attack)
5. Argmax: whichever output is higher is the prediction

### Exact architecture

```
Input: node features (N × 42)
       ↓
SAGEConv Layer 1: in_channels=42 → out_channels=64
       ↓
ReLU activation
       ↓
Dropout (p=0.3) — randomly zeroes 30% of neurons during training
       ↓
SAGEConv Layer 2: in_channels=64 → out_channels=64
       ↓
ReLU activation
       ↓
Dropout (p=0.3)
       ↓
Linear classifier head: 64 → 2 (one score per class)
       ↓
Prediction: argmax → 0 (Normal) or 1 (Attack)
```

Total trainable parameters: **13,826**

### Training hyperparameters (confirmed from `results/gnn/model_config.json`)

| Parameter | Value | Why |
|-----------|-------|-----|
| Hidden dimension | 64 | Balance between capacity and efficiency |
| Number of SAGEConv layers | 2 | 2 hops of neighbourhood information |
| Dropout | 0.3 | Reduces overfitting |
| Learning rate | 0.001 | Standard Adam learning rate |
| Weight decay | 0.0001 | L2 regularisation to prevent overfitting |
| Epochs | 50 | Sufficient for convergence |
| Batch size | 32 | 32 graphs per training step |
| Seed | 42 | Reproducibility |
| Validation split | 0.15 | 15% of training graphs held for validation |
| Optimizer | Adam | Adaptive learning rate optimizer |

### Handling class imbalance

The training data has ~68% attack and ~32% normal flows. A naive model would just predict "attack" for everything and get 68% accuracy.

We solve this with **weighted CrossEntropyLoss**:
- Class weights are computed as `total / (2 × class_count)` for each class
- Normal class weight: ~1.5759 (higher weight because it is the minority)
- Attack class weight: ~0.7324 (lower weight because it is the majority)
- This means misclassifying a normal flow as attack is penalised more heavily

### Train/validation split

- The 1,754 training graphs are shuffled with seed=42 and split 85/15
- 1,491 graphs for training, 263 graphs for validation
- The 824 official test graphs are **never seen during training or hyperparameter tuning**
- The test set is evaluated **ONCE** after training is complete (correct ML practice)

### Best model saving

After each epoch, if the validation F1 score is the best seen so far, the model weights are saved to `models/graphsage_best.pt`. At the end of training, the best weights are reloaded for final evaluation.

---

## 10. Original GraphSAGE Results

These are the actual results from `results/gnn/test_metrics.json` and `results/gnn/classification_report.txt`.

### Test set composition

| Split | Total | Normal | Attack |
|-------|-------|--------|--------|
| Test set | 82,332 nodes | 37,000 (44.9%) | 45,332 (55.1%) |

### Metrics on official test set

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.8817** |
| **Precision** | **0.8459** |
| **Recall** | **0.9601** |
| **F1-score** | **0.8994** |

### Confusion matrix

|  | Predicted Normal | Predicted Attack |
|--|-----------------|-----------------|
| **True Normal** | 29,070 (correctly safe) | 7,930 (false alarms) |
| **True Attack** | 1,811 (missed attacks) | 43,521 (correctly caught) |

### Per-class classification report

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Normal (0) | 0.94 | 0.79 | 0.86 | 37,000 |
| Attack (1) | 0.85 | 0.96 | 0.90 | 45,332 |
| Weighted avg | 0.89 | 0.88 | 0.88 | 82,332 |

### What these metrics mean

**Accuracy (0.8817):** The model correctly classifies 88.17% of all flows. Out of 82,332 flows, it gets 72,591 right.

**Precision (0.8459):** When the model says "this is an attack", it is correct 84.59% of the time. 15.41% of "attack" predictions are false alarms.

**Recall (0.9601):** The model catches 96.01% of all actual attacks. Only 3.99% of real attacks are missed. This is the most important metric for security — missing an attack is more dangerous than a false alarm.

**F1-score (0.8994):** The harmonic mean of precision and recall. A balanced measure of detection quality. Closer to 1.0 is better.

**Confusion matrix interpretation:**
- 43,521 attacks correctly detected (true positives — good)
- 29,070 normal flows correctly passed (true negatives — good)
- 1,811 attacks missed (false negatives — dangerous)
- 7,930 normal flows falsely flagged as attacks (false positives — annoying but less dangerous)

### Best validation F1

From the ablation report Experiment A: **0.9601** (best validation F1 during training)

---

## 11. Random Forest Baseline

### What is a Random Forest?

A Random Forest is a conventional machine learning model that builds many decision trees, has each tree vote on the prediction, and uses the majority vote as the final answer. It does **not** use any graph structure — it treats each flow record as a completely independent data point.

### Why we used Random Forest

We need a comparison point. The question is: does using graph structure (GraphSAGE) actually help compared to not using it (Random Forest)?

If GraphSAGE gets similar results to Random Forest, the graph structure is not adding value. If GraphSAGE is clearly better, the graph relationships are genuinely useful.

### Feature input

Random Forest uses **exactly the same 42 preprocessed features** as GraphSAGE — the same sklearn pipeline output, the same column order. The only difference is that RF ignores the graph edges entirely.

### Random Forest configuration (from `results/ml/model_config.json`)

| Parameter | Value |
|-----------|-------|
| n_estimators | 200 trees |
| max_depth | None (grow full trees) |
| max_features | "sqrt" (standard RF heuristic) |
| class_weight | "balanced" (mirrors GraphSAGE's weighted loss) |
| n_jobs | -1 (all CPU cores) |
| random_state | 42 |

### Actual test set results (from `results/ml/test_metrics.json`)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.8711** |
| **Precision** | **0.8166** |
| **Recall** | **0.9878** |
| **F1-score** | **0.8941** |

### Random Forest confusion matrix

|  | Predicted Normal | Predicted Attack |
|--|-----------------|-----------------|
| **True Normal** | 26,942 | 10,058 |
| **True Attack** | 554 | 44,778 |

### Per-class classification report

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Normal (0) | 0.98 | 0.73 | 0.84 | 37,000 |
| Attack (1) | 0.82 | 0.99 | 0.89 | 45,332 |
| Weighted avg | 0.89 | 0.87 | 0.87 | 82,332 |

### Direct comparison: GraphSAGE vs Random Forest

| Metric | GraphSAGE | Random Forest | Difference |
|--------|-----------|---------------|------------|
| Accuracy | 0.8817 | 0.8711 | +0.0106 |
| Precision | 0.8459 | 0.8166 | +0.0293 |
| Recall | 0.9601 | 0.9878 | -0.0277 |
| F1-score | 0.8994 | 0.8941 | +0.0053 |

**Summary:** The results are close. GraphSAGE has better accuracy, precision, and F1. Random Forest has higher recall (catches more attacks, but also more false alarms). Neither is dramatically better than the other in this initial comparison — the real investigation of graph structure contribution is done in the ablation study.

### The fundamental difference

- **Random Forest:** Looks at each flow row in complete isolation. No knowledge of what happens in nearby flows.
- **GraphSAGE:** Each flow's prediction is influenced by its 5 most similar neighbours within its 100-flow window.

---

## 12. Ablation Study

### What is an ablation study?

An ablation study is an experiment where you systematically remove or change one component of your system and measure what happens to performance. It helps you understand which parts are actually contributing.

### Why we ran this study

The key question is: **does the k-NN similarity graph structure actually help GraphSAGE, or would it perform just as well (or even better) with different edges?**

To answer this, we ran 3 experiments with identical model architecture, identical features, identical training procedure, identical seed, and identical train/val split. Only the **graph wiring** changed.

### The three experiments

**Experiment A — Original graph (k=5 cosine-similarity)**
- The actual `ct_*` k-NN graph built in Phase 2
- Edges connect nodes to their 5 most similar neighbours
- Edge weights = cosine similarity scores
- This is the "real" graph

**Experiment B — No-message graph (all edges removed)**
- `edge_index` = empty (no edges at all)
- SAGEConv degenerates: without neighbours, it becomes a simple per-node linear transformation
- Same node features as A
- Tests: does any graph at all help vs no graph?

**Experiment C — Degree-preserving random rewired graph**
- Same number of edges as A (every node still has exactly 5 outgoing edges)
- Destinations are randomly shuffled within each graph
- Self-loops removed, duplicates removed
- Edge weights = uniform 1.0 (uninformative)
- Labels NEVER read during rewiring
- Tests: does the *specific* similarity-based wiring matter, vs just having any edges?

### What stays identical across all three experiments

- GraphSAGE architecture (SAGEConv layers, hidden_dim=64, dropout=0.3)
- All 42 node features
- Training procedure (epochs=50, lr=0.001, Adam optimizer, weighted loss)
- Seed (42)
- Train/validation split (same 1,491/263 graph split with seed=42)
- Test graphs (same 824 official test graphs, evaluated once)

### Results (from `results/gnn/ablation/`)

#### Experiment A — Original k=5 cosine-similarity graph

| Metric | Value |
|--------|-------|
| Accuracy | 0.8813 |
| Precision | 0.8452 |
| Recall | 0.9604 |
| F1 | 0.8991 |
| Best Validation F1 | 0.9601 |

**Confusion matrix:**

|  | Pred Normal | Pred Attack |
|--|-------------|-------------|
| True Normal | 29,028 | 7,972 |
| True Attack | 1,797 | 43,535 |

---

#### Experiment B — No-message graph (no edges)

| Metric | Value |
|--------|-------|
| Accuracy | 0.8398 |
| Precision | 0.7910 |
| Recall | 0.9636 |
| F1 | 0.8688 |
| Best Validation F1 | 0.9511 |

**Confusion matrix:**

|  | Pred Normal | Pred Attack |
|--|-------------|-------------|
| True Normal | 25,462 | 11,538 |
| True Attack | 1,652 | 43,680 |

---

#### Experiment C — Degree-preserving random rewired graph

| Metric | Value |
|--------|-------|
| Accuracy | 0.9487 |
| Precision | 0.9318 |
| Recall | 0.9785 |
| F1 | 0.9546 |
| Best Validation F1 | 0.9731 |

**Confusion matrix:**

|  | Pred Normal | Pred Attack |
|--|-------------|-------------|
| True Normal | 33,751 | 3,249 |
| True Attack | 975 | 44,357 |

---

### Comparison table

| Exp | Graph Structure | Accuracy | Precision | Recall | F1 | Best Val F1 |
|-----|----------------|----------|-----------|--------|-----|------------|
| A | k=5 cosine-similarity k-NN | 0.8813 | 0.8452 | 0.9604 | 0.8991 | 0.9601 |
| B | No edges (node features only) | 0.8398 | 0.7910 | 0.9636 | 0.8688 | 0.9511 |
| C | Degree-preserving random | **0.9487** | **0.9318** | **0.9785** | **0.9546** | **0.9731** |

### What the results tell us

1. **A > B**: Graph structure helps. Having any edges (even random) is better than no edges. Removing all edges drops F1 from 0.8991 to 0.8688.

2. **C > A**: The random rewired graph outperforms the similarity-based graph. This is the surprising and academically interesting finding — explained in the next section.

3. **B < A < C**: The ordering suggests that *more informative/diverse neighbourhood aggregation* leads to better performance.

---

## 13. Why Experiment C Outperformed A

### Important caveat first

We cannot prove causation from a single set of experiments. What we can do is **measure structural differences between the graphs** and propose hypotheses about why the different wiring leads to different performance. This is the scientifically honest approach.

### Observed measurements (from ablation_report.md structural analysis)

The following differences between experiments A and C were directly measured from the actual graph data (50 training graphs, 5,000 nodes, 25,000 edges each):

#### 1. ct_* neighbourhood contrast — the most diagnostic measurement

SAGEConv computes: `h_i = Linear( [x_i || mean_j(x_j)] )`
where `mean_j(x_j)` is the average feature vector of node i's neighbours.

The **ct_* difference between a node and its neighbourhood mean** tells us how much new information the neighbourhood provides in those 8 dimensions:

| | A (k-NN) | C (random) | Ratio |
|--|----------|------------|-------|
| Mean absolute ct_* diff (self vs neighbour mean) | 0.0613 | 0.3085 | **5.03×** |
| Mean absolute ct_* diff (mixed-label windows) | 0.0427 | 0.3403 | **7.98×** |

**Interpretation (hypothesis):** In Experiment A, the k-NN wiring intentionally connects nodes to their most similar ct_* neighbours. This means `mean_j(x_j) ≈ x_i` in the ct_* dimensions — the neighbourhood mean carries almost no additional information beyond what the node already knows about itself. In Experiment C, the random wiring creates 5× more contrast, so the neighbourhood mean provides genuinely new information.

#### 2. Cross-class connectivity (in mixed-label windows)

| | A (k-NN) | C (random) |
|--|----------|------------|
| Mean fraction of neighbours with a different label | 6.88% | **29.79%** |

**Interpretation (hypothesis):** In Experiment A, the similarity-based wiring tends to cluster nodes of the same type together (normal with normal, attack with attack). In Experiment C, random wiring connects across class boundaries 4.3× more often. This cross-class signal may help the model learn to distinguish attack from normal by contrast.

#### 3. L2 distance (node vs neighbourhood mean, all 42 features)

| | A (k-NN) | C (random) | Ratio |
|--|----------|------------|-------|
| L2 distance (42-dim, self vs neighbour mean) | 5.035 | 6.469 | **1.28×** |

**Interpretation (hypothesis):** The full-feature distance between a node and its neighbourhood mean is 28% larger in Experiment C, meaning the neighbourhood provides more orthogonal (independent) information.

#### 4. Full-feature cosine similarity between connected nodes

| | A (k-NN) | C (random) |
|--|----------|------------|
| Mean full-feature cosine similarity | 0.9950 | 0.9936 |
| Standard deviation | 0.0443 | 0.0488 |

**Observation:** Both graphs have similarly high full-feature similarity between connected nodes. This is because both graphs only connect nodes within the same 100-record temporal window, and flows close in time are inherently similar. The difference is specifically in the ct_* dimensions.

#### 5. Overall neighbour feature variance

| | A (k-NN) | C (random) | Ratio |
|--|----------|------------|-------|
| Overall (42-dim) variance | 303.96 | 303.98 | **1.00** |

**Observation:** The total feature diversity in the neighbourhood is **identical** between A and C. Experiment C does not inject extra information — it just redistributes it differently. This confirms the degree-preservation is working correctly.

### Summary of the observed finding

> The k-NN graph wiring intentionally minimises ct_* distance between connected nodes. This means the 8 ct_* dimensions of the aggregated neighbourhood carry near-zero additional information in Experiment A (the neighbourhood mean ≈ the node itself). The degree-preserving random wiring in Experiment C creates 5× more contrast in exactly those 8 dimensions, while keeping the same total feature diversity and edge count.

### IMPORTANT DISCLAIMER

This is an **observed measurement** and a **proposed hypothesis**, not a proven causal mechanism. We observe that C outperforms A and we measure structural differences that could explain why. We do not claim to have proven that this is the definitive reason. This distinction is important for academic reporting.

---

## 14. Ablation Validation

### Why validation was necessary

During development, the first version of the random rewiring contained errors:
- Some nodes were being given more or fewer outgoing edges than the original (degree sequence was NOT preserved)
- Some self-loops remained after rewiring
- Some duplicate edges existed

These errors would invalidate the ablation experiment — if Experiment C has a different number of edges or connectivity pattern than intended, the comparison is unfair.

### The corrected implementation

The final `randomise_edges()` function in `src/ablation.py` uses a robust algorithm:
1. Extract the original edge_index to get the out-degree of every node
2. Collect all destination endpoints as a flat list (length = total edges)
3. Randomly shuffle that destination list globally (Fisher-Yates permutation)
4. Reassign shuffled destinations to sources maintaining original out-degree
5. Run up to 5 repair passes to eliminate self-loops by swapping destinations
6. Final scan to remove any residual duplicate edges

### Verified structural properties of randomised graphs

From the ablation report, verified on 50 training graphs before any training:

| Property | Result |
|----------|--------|
| Edge count preserved (C vs A) | ✓ 25,000 = 25,000 |
| Self-loops | ✓ 0 |
| Duplicate edges | ✓ 0 |
| Out-degree sequence preserved | ✓ 0 mismatches |
| Node features (x) modified | ✓ No |
| Labels (y) modified | ✓ No |
| attack_cat modified | ✓ No |
| Rewiring independent of labels | ✓ Confirmed |

### Ablation tests

**24 unit tests** in `tests/test_ablation.py` cover:
- Edge count preservation
- No self-loops
- No duplicate edges
- Out-degree sequence preservation
- Uniform k-NN degree (all nodes have exactly k edges)
- Node features (x) unchanged
- Labels (y) unchanged
- attack_cat unchanged
- Rewiring is independent of label values (labels not used)
- Irregular degree sequences
- Empty graph edge cases
- Single-node graph edge cases
- `strip_edges()` correctness (Experiment B)
- `validate_random_graph()` rejection of invalid graphs

---

## 15. Reproducibility and CUDA Note

### The reproducibility setup

The code sets the following seeds before any training run:

```python
random.seed(seed)         # Python stdlib random
np.random.seed(seed)      # NumPy random
torch.manual_seed(seed)   # PyTorch CPU operations
torch.cuda.manual_seed_all(seed)   # PyTorch GPU operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

The train/validation split uses `random.Random(seed)` — a separate Python `Random` object — so the split assignment is **fully deterministic and identical** across all three ablation experiments (A, B, C use the same 1,491/263 split).

### The CUDA non-determinism issue

Despite all seed settings, **repeated runs on CUDA GPU may produce slightly different results** due to a known PyTorch limitation.

The technical cause is: `scatter_add` operations inside SAGEConv use parallel reductions on GPU. Floating-point addition is not associative (a+b+c ≠ b+c+a in floating point due to rounding), and the order of parallel additions is non-deterministic. Setting `torch.use_deterministic_algorithms(True)` would force determinism but is not set here as it restricts certain operations.

### What this means in practice

- The exact weight values in `models/graphsage_best.pt` can differ slightly between two independent training runs on the same GPU
- The difference in metrics between runs is typically small (0.001–0.005 in F1)
- The relative ordering of experiments (A, B, C) and the structural measurements are unaffected
- The ablation train/val split is fully deterministic (uses CPU random)

### What this does NOT mean

This does not automatically invalidate the methodology. The structural analysis (ct_* contrast, cross-class connectivity, etc.) is computed from the graph data directly and is deterministic. The performance ordering (C > A > B) is robust enough that it is unlikely to reverse with a different random seed.

When reporting results, this should be noted as: "Results may vary by ~0.001–0.005 F1 between runs due to CUDA non-determinism in scatter_add operations."

---

## 16. Inference Pipeline

### What the inference pipeline does

The inference pipeline allows a **new CSV file** (not seen during training) to be processed and classified using the trained GraphSAGE model. This is the component that will eventually power the web platform.

**Files:**
- `src/inference.py` — core logic
- `run_inference.py` — command-line entry-point
- `tests/test_inference.py` — unit and integration tests

**Command:**
```bash
python run_inference.py --csv path/to/file.csv --out predictions.csv
```

### Step-by-step inference flow

```
New CSV file
     ↓
load_inference_csv()
  - Check file exists
  - If 'id' column missing → inject sequential IDs
  - If 'label'/'attack_cat' missing → inject dummy zeros (never used as features)
  - Validate required feature columns present
     ↓
build_inference_graphs()
  - Sort by id (temporal order)
  - Apply saved preprocessor.pkl (NEVER re-fitted)
  - Divide into 100-row windows
  - Compute cosine similarity → k=5 k-NN edges
  - Create PyG Data objects
     ↓
load_graphsage()
  - Load models/graphsage_best.pt
  - Rebuild model architecture with same hyperparameters
  - Set to eval() mode (no gradient computation, dropout disabled)
     ↓
run_inference()
  - Pass graphs through model in batches of 32
  - torch.no_grad() context (faster, no memory for gradients)
  - Compute softmax probabilities for each node
  - Argmax → binary prediction (0 or 1) for each flow
     ↓
summarise()
  - Count total flows, normal flows, attack flows
  - Compute attack percentage
     ↓
Output:
  - predictions[]:  0/1 per flow
  - probabilities[]: [P(Normal), P(Attack)] per flow
  - summary: {total_flows, normal_flows, attack_flows, attack_pct}
  Optional: predictions.csv with flow_index, prediction, prob_normal, prob_attack
```

### Handling CSVs without labels

In production, an uploaded CSV may not have `label` or `attack_cat` columns (since the user doesn't know which flows are attacks — that's what they're asking us to find!).

The inference pipeline handles this gracefully:
- If `label` or `attack_cat` is absent, dummy zero-valued columns are inserted
- These dummy values are stripped by the preprocessing pipeline before any features are computed
- The model never sees them

### Output fields

| Field | Description |
|-------|-------------|
| `flow_index` | Row index in the input CSV |
| `prediction` | 0 = Normal, 1 = Attack |
| `prob_normal` | P(Normal) — softmax probability, 0 to 1 |
| `prob_attack` | P(Attack) — softmax probability, 0 to 1 |

### Why this is important for the web platform

The inference pipeline is the core backend component that the web platform will call. When a user uploads a CSV, the backend will run `predict_csv()` and return the results for display. Everything else (UI, charts, tables) builds on top of these predictions.

---

## 17. Testing Summary

All tests are in the `tests/` directory and run with pytest:
```bash
python -m pytest tests/ -v
```

### Test files and counts

| Test File | Approximate Tests | What is Tested |
|-----------|------------------|----------------|
| `tests/test_data_loader.py` | ~14 tests | CSV loading, column validation, get_numerical_cols(), get_feature_cols(), error handling for missing files/columns |
| `tests/test_preprocessor.py` | ~16 tests | Pipeline building, StandardScaler output (~mean=0, ~std=1), OrdinalEncoder, target column preservation, id column removal, unseen category handling (-1 encoding), pipeline save/load, train/test column consistency |
| `tests/test_graph_builder.py` | ~20 tests | Node count, feature dimensions (42), edge count (100×k), no self-loops, label leakage (edges don't change when labels are flipped), partial windows, save/load .pt files |
| `tests/test_graph_validator.py` | ~14 tests | graph_stats() keys and values, dataset_stats() shape, validate_no_leakage() pass/fail, save_stats() file creation and JSON structure |
| `tests/test_graphsage_model.py` | ~21 tests | Output shape (N×2), binary predictions, softmax sums to 1, configurable hidden_dim and num_layers, invalid num_layers raises error, train vs eval mode difference, label/attack_cat not in features, class weight computation, compute_metrics(), predict_all(), reproducibility with same seed |
| `tests/test_random_forest.py` | ~18 tests | Feature count == 42, no label/id/attack_cat in features, train/test features identical, fitted classifier, n_estimators, seed reproducibility, evaluate_rf() keys, confusion matrix sum, save/load pickle |
| `tests/test_ablation.py` | **24 tests** | Edge count preservation, no self-loops, no duplicate edges, out-degree sequence preserved, uniform k degree, x/y/attack_cat unchanged, rewiring independent of labels, irregular degree graph, empty/single-node edge cases, strip_edges(), validate_random_graph() |
| `tests/test_inference.py` | ~23 tests | load_inference_csv() (with/without labels, missing id, bad columns), build_inference_graphs() (graph count, feature dim, no leakage, partial window), load_graphsage() (13,826 params, eval mode), run_inference() (prediction shape, binary, probabilities sum to 1), summarise(), predict_csv() end-to-end |

**Total tests across all files: approximately 150 unit and integration tests**

---

## 18. Current Project Status

### Completed

| Component | Status | Output |
|-----------|--------|--------|
| Data loading and validation | ✅ Complete | `src/data_loader.py` |
| Dataset inspection | ✅ Complete | `src/inspector.py` |
| Preprocessing pipeline | ✅ Complete | `data/processed/preprocessor.pkl`, processed CSVs |
| Flow-similarity graph construction | ✅ Complete | 1,754 train + 824 test `.pt` files |
| Graph validation and leakage checks | ✅ Complete | `src/graph_validator.py` |
| Graph quality sensitivity analysis (k=3,5,10) | ✅ Complete | Plots + JSON in `data/processed/graph_analysis/` |
| GraphSAGE model architecture | ✅ Complete | `src/models/graphsage.py` |
| GraphSAGE training loop | ✅ Complete | `src/training.py` |
| GraphSAGE evaluation | ✅ Complete | `results/gnn/` |
| Random Forest baseline | ✅ Complete | `results/ml/` |
| Ablation study (3 experiments) | ✅ Complete | `results/gnn/ablation/` |
| Ablation validation and correction | ✅ Complete | 24 tests passing |
| Inference pipeline | ✅ Complete | `src/inference.py` |
| Unit and integration test suite | ✅ Complete | ~150 tests |
| Project documentation (README) | ✅ Complete | `README.md` |
| Complete project documentation | ✅ Complete | `PROJECT_STATUS.md` (this file) |

### Not Yet Implemented

| Component | Status | Notes |
|-----------|--------|-------|
| Web frontend (UI) | ❌ Not started | React or similar framework |
| CSV upload interface | ❌ Not started | File upload component |
| Backend API / web server | ❌ Not started | Flask / FastAPI / Django |
| API endpoint for inference | ❌ Not started | REST endpoint calling `predict_csv()` |
| Prediction results dashboard | ❌ Not started | Charts, tables, attack breakdown |
| Graph visualisation | ❌ Not started | Interactive graph viewer |
| Attack category breakdown display | ❌ Not started | Multiclass output display |
| Per-flow probability display | ❌ Not started | Confidence scores in UI |
| Error handling (invalid CSV, etc.) | ❌ Not started | User-facing error messages |
| System integration testing | ❌ Not started | End-to-end web test |
| Deployment | ❌ Not started | Server/cloud setup |
| Final project report | ⏳ In progress | Academic write-up |

---

## 19. What the Web Platform Will Eventually Do

> NOTE: Everything in this section describes PLANNED future functionality. It does not currently exist.

### Planned user-facing workflow

```
User opens web browser
        ↓
Navigates to intrusion detection web app
        ↓
Clicks "Upload CSV" button
        ↓
Selects a network flow CSV file from their computer
        ↓
Clicks "Analyse"
        ↓
Backend receives CSV
        ↓
Runs preprocessing (load_inference_csv + preprocessor.pkl)
        ↓
Runs graph construction (build_inference_graphs)
        ↓
Runs GraphSAGE inference (run_inference)
        ↓
Returns results as JSON to frontend
        ↓
Results dashboard displayed to user
```

### What the results dashboard could display

| Display Element | Description |
|----------------|-------------|
| Total flow count | "Your file contains X network flows" |
| Normal flow count | "X flows appear normal" |
| Attack flow count | "X flows are classified as attacks" |
| Attack percentage | "Y% of flows are attacks" |
| Per-flow probability | Confidence score for each prediction |
| Attack category breakdown | If `attack_cat` is available, show distribution |
| Confusion matrix | If true labels are available in the CSV |
| Graph visualisation | Interactive visualisation of one graph |
| Summary statistics | Min, max, mean probability by class |
| Download button | Download full predictions as CSV |

### Key design principle

The backend ML pipeline is already complete. The web platform is essentially a user interface layer on top of the existing `predict_csv()` function. The core science is done.

---

## 20. Current Limitations

| Limitation | Explanation |
|-----------|-------------|
| **CSV-based only, not real-time** | System works on pre-recorded flow CSVs. Cannot capture or analyse live network traffic. A real-time system would require packet capture (pcap) and feature extraction pipeline. |
| **No IP communication graph** | Our graph is flow-similarity based, not IP-to-IP topology. We cannot model which machines communicate with which. |
| **High intra-window similarity** | The ct_* features are pre-computed by the dataset authors over a sliding window, so flows close in CSV order always have correlated ct_* values. The k-NN graph therefore connects very similar nodes, which the ablation study shows may not be optimal. |
| **Binary classification** | The current GraphSAGE model predicts only Normal/Attack (binary). Multiclass prediction across 10 attack categories is not yet trained. |
| **UNSW-NB15 dataset constraints** | The model is trained on a specific benchmark dataset from 2015. Real-world network traffic may differ significantly (different protocols, traffic volumes, attack patterns). Generalisation to production networks is unverified. |
| **CUDA non-determinism** | Repeated training runs can produce slightly different model weights due to non-deterministic GPU operations. Results are reproducible within ±0.005 F1 across runs. |
| **Window boundary assumption** | Splitting data into fixed 100-row windows ignores attack flows that span window boundaries. An attack that starts in window N and continues in window N+1 may be partially missed. |
| **No web interface** | The system currently has no user-facing interface. All functionality is accessed via command-line scripts. |
| **No real-time feedback** | Even the final web platform will be batch-processing (analyse a whole CSV at once), not streaming analysis. |

---

## 21. What Is Left for the Mini Project

Work remaining in logical order:

1. **Backend API integration** — Create a Python web server (Flask or FastAPI) with an endpoint that accepts a CSV, calls `predict_csv()`, and returns JSON results

2. **CSV upload interface** — Build a simple web page with a file upload form

3. **Results dashboard** — Create a results page showing attack counts, percentages, per-flow predictions table

4. **Graph visualisation** — Add an interactive visualisation of a sample graph from the uploaded CSV

5. **Error handling and user feedback** — Handle invalid CSVs, wrong column formats, missing files — show clear error messages to the user

6. **System integration testing** — Test the full pipeline end-to-end from browser upload to results display

7. **Final documentation and report** — Write the academic project report covering all phases, methodology, results, discussion, and conclusion

8. **Deployment / demo preparation** — Set up the system for the final presentation, ensure demo runs smoothly

> Do not start these tasks now — this section is for planning awareness only.

---

## 22. Presentation Cheat Sheet

This section gives you ready answers for your guide's questions. Read the short answer for speaking, read the explanation for deeper understanding.

---

**Q: What is your project?**
> Short: We're building a system that analyses network traffic flow records from a CSV file and detects cyberattacks using a Graph Neural Network called GraphSAGE.
> Deeper: The project takes the UNSW-NB15 network flow dataset, converts batches of 100 flows into flow-similarity graphs, and uses GraphSAGE to classify each flow as Normal or Attack by incorporating information from neighbouring flows in the graph.

---

**Q: What problem are you solving?**
> Short: Traditional intrusion detection treats each network flow independently. We're investigating whether modelling relationships between flows (using graphs) can improve detection.
> Deeper: Standard ML models predict one row at a time. We hypothesize that attack flows may cluster together or have distinctive neighbourhood patterns. By building graphs where similar flows are connected, a GNN can use neighbourhood context when making predictions.

---

**Q: Why UNSW-NB15?**
> Short: It's a well-known, publicly available, labelled network traffic benchmark dataset used widely in academic intrusion detection research.
> Deeper: It provides 257,673 labelled flow records across 10 attack categories, which is large enough for meaningful ML experiments. It was created by the Australian Centre for Cyber Security and is a standard comparison dataset.

---

**Q: Why CSV instead of real-time packets?**
> Short: This is a research project building the ML pipeline first. Working with a CSV dataset lets us focus on the algorithm without needing network infrastructure.
> Deeper: Real-time packet capture requires network sensors, feature extraction pipelines, and live deployment. For a mini project, using a pre-built research dataset is standard practice. The eventual web platform will support CSV uploads for offline analysis.

---

**Q: What is a network flow?**
> Short: A summary of one network communication session — like a single conversation between two computers, described by how long it lasted, how many bytes were transferred, which protocol was used, etc.
> Deeper: Each row in the CSV represents one flow record with 45 columns capturing things like duration, packet counts, byte sizes, protocol type, service type, connection state, and statistical counts of similar recent connections.

---

**Q: What is a node in your graph?**
> Short: Each node represents one network flow record — one row from the CSV.
> Deeper: A node has 42 features (the preprocessed network flow statistics). Its label is either 0 (Normal) or 1 (Attack). In a 100-flow window, there are 100 nodes.

---

**Q: What is an edge in your graph?**
> Short: An edge connects two flow records that have similar neighbourhood statistics, measured by cosine similarity over 8 special count-based features called ct_* features.
> Deeper: For each node, we find its 5 most similar neighbours using cosine similarity over 8 ct_* features (which count how many similar connections happened in the last 100 flows). We draw a directed edge from each node to each of its 5 nearest neighbours.

---

**Q: Why did you construct a graph?**
> Short: To model relationships between flows — the hypothesis is that flows happening close together in time with similar behaviour patterns may have related attack/normal status.
> Deeper: By connecting similar flows in a graph, GraphSAGE can aggregate neighbourhood information when classifying each flow. Instead of only looking at one flow's own features, it also looks at what its neighbours look like, which may help distinguish attack from normal traffic.

---

**Q: Why cosine similarity?**
> Short: Cosine similarity measures how similar two vectors are in terms of direction (angle), independent of magnitude. It's standard for comparing count-based feature vectors.
> Deeper: The 8 ct_* features are integer counts. After StandardScaler normalisation, cosine similarity captures the relative pattern of these counts between two flows. A similarity of 0.98+ means two flows have nearly identical neighbourhood count patterns.

---

**Q: Why the ct_* features for edges?**
> Short: These 8 features are the closest proxy for "how related are two flows" in this dataset, since we don't have source/destination IP addresses.
> Deeper: The ct_* features count connections from/to the same sources, destinations, services, and ports over the last 100 connections. Two flows with similar ct_* values likely belong to the same communication pattern or attack campaign. Without IP addresses, these are the best available relationship indicators.

---

**Q: Why window size 100?**
> Short: 100 flows is a practical chunk size — large enough to capture meaningful local patterns, small enough to build and process efficiently.
> Deeper: There's no theoretically optimal window size for this dataset. 100 is a reasonable default that creates 1,754 training graphs and 824 test graphs. The ct_* features themselves are computed by the dataset authors over a 100-connection sliding window, so our window size matches their feature computation context.

---

**Q: Why k=5?**
> Short: 5 neighbours per node gives sufficient connectivity for GNN message passing without excessive computational cost.
> Deeper: We validated k=3, 5, and 10. All produce zero isolated nodes and similar similarity distributions. The graph quality sensitivity analysis showed no compelling reason to prefer one k over another from structural statistics alone. We used k=5 as the primary configuration and noted that empirical comparison through GNN training would be needed to select the best k — which is exactly what the ablation study begins to explore.

---

**Q: What is GraphSAGE?**
> Short: GraphSAGE is a Graph Neural Network that learns to classify nodes by combining each node's own features with an aggregated summary of its neighbours' features.
> Deeper: It uses SAGEConv layers which compute: `h_i = Linear([x_i || mean_j(x_j)])` — concatenate the node's own feature vector with the mean of its neighbours, then apply a learnable linear transformation. Stacking two layers allows each node to incorporate information from 2 hops away.

---

**Q: Why GraphSAGE instead of normal ML?**
> Short: Normal ML ignores the relationships between flows. GraphSAGE can leverage the graph structure — if your neighbours look like attacks, you probably are one too.
> Deeper: The key difference is that GraphSAGE's predictions are context-dependent. The same flow with the same 42 features could receive a different prediction depending on what its graph neighbours look like. Standard models like Random Forest make predictions solely from the individual flow's features.

---

**Q: What are the 42 features?**
> Short: 39 numerical network flow statistics (duration, byte counts, packet counts, timing, etc.) standardised to mean=0, std=1, plus 3 categorical features (protocol, service, state) encoded as integers.
> Deeper: The 39 numerical features include things like `dur` (duration), `sbytes`/`dbytes` (bytes sent/received), `spkts`/`dpkts` (packet counts), `sttl`/`dttl` (TTL values), `rate`, `sload`/`dload`, jitter measurements, TCP-specific fields, and the 8 ct_* count features. The `id`, `label`, and `attack_cat` columns are excluded.

---

**Q: What is message passing?**
> Short: Each node sends its features to its connected neighbours, who aggregate (average) those messages and combine them with their own features. This propagates information across the graph.
> Deeper: In one message-passing step (one SAGEConv layer), every node simultaneously collects feature vectors from all its neighbours, averages them, concatenates the average with its own features, and passes the result through a linear layer + activation. After 2 layers, each node has integrated information from 2 hops in the graph.

---

**Q: What does Random Forest do?**
> Short: Random Forest builds 200 decision trees, each trained on a random subset of data. For any new input, each tree votes, and the majority vote wins.
> Deeper: Each tree is a series of if-else rules on individual features. The "forest" averages out individual trees' errors. It treats every flow independently — no graph structure, no neighbourhood information.

---

**Q: Why did you use Random Forest?**
> Short: To provide a conventional ML baseline. We need to know if GraphSAGE's graph-based approach actually improves over standard ML with the same features.
> Deeper: Random Forest is a strong, well-understood benchmark for tabular data. By using exactly the same 42 features as GraphSAGE (same pipeline output, same column order) and comparing results, we can isolate the contribution of graph structure. If GraphSAGE doesn't beat RF significantly, the graph approach isn't helping.

---

**Q: What is an ablation study?**
> Short: An experiment where you remove or change one part of your system to see how much that part contributes to performance.
> Deeper: We ran 3 experiments with identical model and training setup, changing only the graph wiring. Experiment A = real similarity graph. Experiment B = no edges. Experiment C = random edges (same count). Comparing A vs B tells us if any graph helps. Comparing A vs C tells us if the specific similarity-based wiring matters.

---

**Q: Why did you create Experiment B?**
> Short: To test whether having any graph structure at all helps, versus no graph structure. Experiment B removes all edges.
> Deeper: If A >> B, graph structure in general is beneficial. If A ≈ B, GraphSAGE is essentially operating as a simple per-node MLP, and the graph edges aren't adding anything.

---

**Q: Why did you create Experiment C?**
> Short: To test whether the specific similarity-based wiring matters, versus just having random connections with the same number of edges.
> Deeper: If A >> C, the ct_* similarity-based wiring specifically is important. If A ≈ C or C > A, the particular wiring scheme doesn't matter much — just having edges of any kind is sufficient (or perhaps the random wiring is accidentally better, which is what we found).

---

**Q: Why did C outperform A?**
> Short: Our measurement shows that the k-NN wiring connects nearly identical nodes, giving GraphSAGE almost no new information from neighbours. Random wiring connects more diverse neighbours, so aggregation actually provides useful new information.
> Deeper: We measured that in Experiment A, the mean absolute ct_* difference between a node and its neighbourhood mean is only 0.061 (nodes and their kNN neighbours are almost identical in those 8 dimensions). In Experiment C, this difference is 0.309 (5× higher). SAGEConv's concatenation of `[x_i || mean_j(x_j)]` carries more useful, independent information in Experiment C. This is a measurement, not a proven causal explanation.

---

**Q: What does F1 mean?**
> Short: F1-score is the harmonic mean of precision and recall. It's a balanced measure that is 1.0 for a perfect classifier and 0.0 for a random one.
> Deeper: F1 = 2 × (Precision × Recall) / (Precision + Recall). It balances the trade-off between catching attacks (recall) and avoiding false alarms (precision). It's preferred over accuracy when class distributions are uneven.

---

**Q: What does recall mean for intrusion detection?**
> Short: Recall is the fraction of actual attacks that are correctly detected. High recall means few attacks are missed.
> Deeper: In security, missing an attack (false negative = low recall) is typically more dangerous than a false alarm (false positive = low precision). An attacker that slips through is more costly than an analyst investigating a false alarm. That's why recall is particularly important in intrusion detection.

---

**Q: What does precision mean?**
> Short: Precision is the fraction of "attack" predictions that are actually attacks. High precision means few false alarms.
> Deeper: If precision is low, the system is generating too many false alarms — flagging normal traffic as attacks. This wastes analysts' time and causes alert fatigue. In a real deployment, both high precision and high recall are needed.

---

**Q: What are the current results?**
> Short: GraphSAGE achieves 88.17% accuracy, 84.59% precision, 96.01% recall, and 89.94% F1 on the test set. The ablation Experiment C achieves 94.87% accuracy and 95.46% F1.
> Deeper: See Sections 10, 11, and 12 for full result tables. GraphSAGE slightly outperforms Random Forest on F1. The ablation study shows that degree-preserving random edges outperform the similarity-based k-NN graph, which is an academically interesting finding that warrants further investigation.

---

**Q: What does your system currently do?**
> Short: It can load a CSV file, preprocess it, build flow-similarity graphs, run them through a trained GraphSAGE model, and output a normal/attack prediction with probability for every flow.
> Deeper: The complete ML pipeline runs end-to-end from CSV to predictions via `python run_inference.py --csv file.csv`. There is no web interface yet — all interaction is through command-line scripts.

---

**Q: What is not implemented yet?**
> Short: The web frontend, the CSV upload interface, the results dashboard, the API backend, and the graph visualisation are all not yet built.
> Deeper: See Section 18 (Current Project Status) for the full not-yet-implemented table and Section 21 (What Is Left) for the remaining work list.

---

**Q: Is this real-time intrusion detection?**
> Short: No. This is an offline batch analysis system. You upload a CSV of recorded flows and get predictions. It does not monitor live network traffic.
> Deeper: Real-time IDS would require: (1) a packet capture agent on the network, (2) real-time feature extraction from packets, (3) a streaming inference pipeline. None of these are implemented. Our system is a research/analysis tool for working with pre-recorded flow data.

---

**Q: Does your graph represent IP communication?**
> Short: No. Our graph is a flow-similarity graph — nodes are individual flows, and edges connect flows with similar neighbourhood statistics. We don't know which machines are talking to which.
> Deeper: The UNSW-NB15 CSV files in our project don't contain source IP and destination IP columns. So we cannot build a topology-based graph. Instead, we use the 8 ct_* features (which count similar recent connections) as a proxy for relatedness between flows. The edges represent statistical similarity, not actual network communication paths.

---

**Q: How will the web platform work?**
> Short: User uploads a CSV → backend runs the preprocessing + graph construction + inference pipeline → results shown on a dashboard with flow counts, attack percentages, and per-flow predictions.
> Deeper: See Section 19 for the planned workflow diagram. The ML backend is complete. The web layer (API, frontend, visualisation) is the remaining work.

---

**Q: What is novel or interesting about your approach?**
> Short: We're applying Graph Neural Networks to intrusion detection on a dataset that doesn't have direct network topology, which requires an alternative graph construction strategy using flow-similarity.
> Deeper: Most GNN-based intrusion detection systems use IP communication graphs. Since our dataset lacks IP columns, we construct a flow-similarity graph using ct_* features — a novel approach for this dataset. The ablation study revealed that the similarity-based wiring is actually less informative than random wiring with the same edge count, which is an academically interesting structural finding worth investigating further.

---

**Q: What are the limitations?**
> Short: No real-time analysis, no IP communication graph, binary classification only, dataset-specific training, and the web interface is not yet built.
> Deeper: See Section 20 (Current Limitations) for the complete table with detailed explanations.

---

## 23. One-Page Project Summary

```
═══════════════════════════════════════════════════════════════
  GNN-BASED NETWORK INTRUSION DETECTION
  Master's Level Mini Project
═══════════════════════════════════════════════════════════════

PROJECT TITLE
  Graph Neural Network-Based Intrusion Detection on
  UNSW-NB15 Network Flow Data

PROBLEM
  Investigate whether flow-similarity graph relationships
  improve binary intrusion detection over conventional ML
  baselines that treat flows independently.

DATASET
  UNSW-NB15  (UNSW, Canberra)
  Training: 175,341 flows  |  Testing: 82,332 flows
  45 columns per row  |  0 missing values
  10 attack categories (binary label: 0=Normal, 1=Attack)
  NOTE: No source/destination IP columns available

INPUT
  CSV files of pre-recorded network flow records.
  NOT real-time packet capture.

PREPROCESSING
  39 numerical features → StandardScaler (mean=0, std=1)
  3 categorical features (proto, service, state) → OrdinalEncoder
  id column dropped  |  label/attack_cat preserved separately
  → 42-feature input representation
  Pipeline saved to preprocessor.pkl for reuse

GRAPH CONSTRUCTION
  Window: 100 consecutive flows → 1 graph
  Results: 1,754 training graphs, 824 test graphs
  Edges: k=5 cosine-similarity k-NN over 8 ct_* features
  Every node: exactly 5 outgoing edges, 0 isolated nodes
  Avg edge weight: ~0.98 (high similarity within windows)

MODEL
  GraphSAGE: SAGEConv(42→64) → ReLU → Dropout(0.3)
           → SAGEConv(64→64) → ReLU → Dropout(0.3)
           → Linear(64→2)
  13,826 trainable parameters
  Weighted CrossEntropyLoss (inverse-frequency weights)
  Adam optimizer, lr=0.001, 50 epochs, seed=42
  Device: CUDA GPU

EXPERIMENTS
  Phase 1: Preprocessing + Validation
  Phase 2: Graph Construction + Quality Analysis (k=3,5,10)
  Phase 3: GraphSAGE training + RF baseline comparison
  Phase 4: Ablation study (3 experiments)
  Inference: predict_csv() pipeline on new CSV files

RESULTS (Test Set: 82,332 nodes)
  ┌──────────────┬──────────┬───────────┬─────────┬───────┐
  │ Model        │ Accuracy │ Precision │  Recall │  F1   │
  ├──────────────┼──────────┼───────────┼─────────┼───────┤
  │ GraphSAGE(A) │  0.8817  │  0.8459   │ 0.9601  │0.8994 │
  │ RandomForest │  0.8711  │  0.8166   │ 0.9878  │0.8941 │
  │ Ablation B   │  0.8398  │  0.7910   │ 0.9636  │0.8688 │
  │ Ablation C   │  0.9487  │  0.9318   │ 0.9785  │0.9546 │
  └──────────────┴──────────┴───────────┴─────────┴───────┘
  Key finding: Degree-preserving random edges (C) outperform
  similarity-based k-NN (A) — measured reason: k-NN wiring
  connects near-identical ct_* neighbours (mean diff = 0.061)
  while random wiring provides 5× more contrast (0.309).

CURRENT STATUS
  ML pipeline: COMPLETE (preprocessing → graphs → training
               → evaluation → ablation → inference)
  Web platform: NOT YET IMPLEMENTED
  Tests: ~150 unit/integration tests passing

REMAINING WORK
  Backend API, CSV upload UI, results dashboard,
  graph visualisation, deployment, final report
═══════════════════════════════════════════════════════════════
```

---

*Document generated from actual source code, test files, and result files in this repository. All metric values verified against `results/gnn/test_metrics.json`, `results/ml/test_metrics.json`, and `results/gnn/ablation/ablation_report.md`.*

*Last updated: September 2026*
