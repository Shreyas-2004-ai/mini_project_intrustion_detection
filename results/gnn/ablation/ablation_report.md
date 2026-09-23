# Ablation Study Report
## GraphSAGE — Graph Structure Contribution Analysis
### Dataset: UNSW-NB15 | Model: GraphSAGE | Phase: 4

---

## 1. Purpose

This ablation study isolates the contribution of the cosine-similarity
k-NN graph structure to GraphSAGE intrusion-detection performance.
Three experiments are run with identical model architecture, features,
training procedure, seed, and train/val split.  Only the graph wiring
changes between experiments.

---

## 2. Shared Configuration

| Parameter | Value |
|-----------|-------|
| Model | GraphSAGE |
| Architecture | SAGEConv(42→64) → ReLU → Dropout → SAGEConv(64→64) → ReLU → Dropout → Linear(64→2) |
| Trainable parameters | 13,826 |
| Node features | 42 (39 scaled numerical + 3 ordinal categorical) |
| Target | `label`  (0 = Normal, 1 = Attack) |
| Hidden dim | 64 |
| Num layers | 2 |
| Dropout | 0.3 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Epochs | 50 |
| Batch size | 32 |
| Seed | 42 |
| Val split | 0.15 (263 val graphs from 1,754 train graphs) |
| Train/val shuffle | `random.Random(42)` — deterministic, identical across A/B/C |
| Class weighting | Inverse-frequency: Normal = 1.5759, Attack = 0.7324 |
| Loss | Weighted CrossEntropyLoss |
| Optimiser | Adam |
| Device | CUDA |
| Test graphs | 824 official test graphs (never seen during training) |

---

## 3. Experiment Definitions

### Experiment A — Original graph (k=5 cosine-similarity k-NN)
- Each node connects to its 5 nearest neighbours by cosine similarity
  over 8 ct_* features: `ct_srv_src`, `ct_srv_dst`, `ct_src_ltm`,
  `ct_dst_ltm`, `ct_dst_src_ltm`, `ct_src_dport_ltm`,
  `ct_dst_sport_ltm`, `ct_state_ttl`
- Edge weight = cosine similarity score (stored in `edge_attr`)
- Graphs loaded directly from `data/processed/graphs/train/` and
  `data/processed/graphs/test/` — unmodified

### Experiment B — No-message graph (all edges removed)
- All edges stripped; `edge_index` = empty, `edge_attr` = empty
- SAGEConv degenerates to a per-node linear transformation with
  no neighbourhood aggregation
- Same node features as A

### Experiment C — Degree-preserving random rewired graph
- Same edge count and out-degree sequence as A (every node keeps k=5
  outgoing edges)
- Destinations are globally shuffled within each graph independently,
  then self-loops are repaired and duplicates removed
- Edge weights set to uniform 1.0 (uninformative)
- Labels and node features are never read during rewiring
- Seed = 42 for the numpy RNG

---

## 4. Graph Validation Results (Experiment C)

Verified on 50 training graphs before any training began.

| Property | Value |
|----------|-------|
| Edge count match (C vs A) | ✓  25,000 = 25,000 |
| Self-loops | ✓  0 |
| Duplicate edges | ✓  0 |
| Out-degree mismatches | ✓  0 |
| Node features (x) modified | ✓  No |
| Labels (y) modified | ✓  No |
| attack_cat modified | ✓  No |

24 unit tests in `tests/test_ablation.py` all pass, covering
structural properties, label-independence, irregular degree sequences,
and edge cases.

---

## 5. Test Set Results

Test set: 82,332 nodes across 824 graphs.
Normal nodes: 37,000 | Attack nodes: 45,332.

### Experiment A — Original k=5 similarity graph

| Metric | Value |
|--------|-------|
| Accuracy | 0.8813 |
| Precision | 0.8452 |
| Recall | 0.9604 |
| F1 | 0.8991 |
| Best validation F1 | 0.9601 |

**Confusion matrix:**

|  | Pred Normal | Pred Attack |
|--|-------------|-------------|
| True Normal | 29,028 | 7,972 |
| True Attack | 1,797 | 43,535 |

### Experiment B — No-message graph

| Metric | Value |
|--------|-------|
| Accuracy | 0.8398 |
| Precision | 0.7910 |
| Recall | 0.9636 |
| F1 | 0.8688 |
| Best validation F1 | 0.9511 |

**Confusion matrix:**

|  | Pred Normal | Pred Attack |
|--|-------------|-------------|
| True Normal | 25,462 | 11,538 |
| True Attack | 1,652 | 43,680 |

### Experiment C — Degree-preserving random rewired graph

| Metric | Value |
|--------|-------|
| Accuracy | 0.9487 |
| Precision | 0.9318 |
| Recall | 0.9785 |
| F1 | 0.9546 |
| Best validation F1 | 0.9731 |

**Confusion matrix:**

|  | Pred Normal | Pred Attack |
|--|-------------|-------------|
| True Normal | 33,751 | 3,249 |
| True Attack | 975 | 44,357 |

### Comparison table

| Exp | Graph Structure | Accuracy | Precision | Recall | F1 | Best Val F1 |
|-----|----------------|----------|-----------|--------|-----|------------|
| A | k=5 cosine-similarity k-NN | 0.8813 | 0.8452 | 0.9604 | 0.8991 | 0.9601 |
| B | No edges | 0.8398 | 0.7910 | 0.9636 | 0.8688 | 0.9511 |
| C | Degree-preserving random | **0.9487** | **0.9318** | **0.9785** | **0.9546** | **0.9731** |

---

## 6. Structural Statistics: A vs C

Measured on 50 training graphs (5,000 nodes, 25,000 edges each).

### ct_* edge weight distribution (Experiment A, original stored weights)

| Stat | Value |
|------|-------|
| Mean | 0.9762 |
| Std | 0.0387 |
| P1 | 0.8374 |
| P50 | 0.9889 |
| P99 | 1.0000 |

The k-NN graph selects near-identical flows: 75% of edges have
ct_* similarity > 0.975.

### Full 42-feature cosine similarity between connected nodes

| | A (k-NN) | C (random) |
|--|----------|------------|
| Mean | 0.9950 | 0.9936 |
| Std | 0.0443 | 0.0488 |
| P50 | 0.9994 | 0.9989 |

Both graphs have similarly high full-feature similarity between
connected nodes, since all connections remain within the same
100-record temporal window.

### ct_* feature difference: node vs its neighbourhood mean

This is the most diagnostic statistic.
SAGEConv computes `h_i = Linear([x_i ‖ mean_j(x_j)])`.
If `mean_j(x_j) ≈ x_i` in the ct_* dimensions, those dimensions
of the second half of the 84-dim input carry no additional information.

| | A (k-NN) | C (random) | C/A ratio |
|--|----------|------------|-----------|
| Mean absolute ct_* diff (self vs nbr mean) | 0.0613 | 0.3085 | **5.03** |
| Mean absolute ct_* diff (mixed-label windows) | 0.0427 | 0.3403 | **7.98** |
| L2 distance (full 42-dim, self vs nbr mean) | 5.035 | 6.469 | 1.28 |

### Cross-class connectivity (mixed-label windows only, 50 graphs)

| | A (k-NN) | C (random) |
|--|----------|------------|
| Mean fraction of neighbours with different label | 0.0688 | **0.2979** |

In windows containing both Normal and Attack flows, Experiment C
connects nodes across the class boundary 4.3× more often than A.

### Overall neighbour feature variance

| | A (k-NN) | C (random) | C/A ratio |
|--|----------|------------|-----------|
| Overall (42-dim) | 303.96 | 303.98 | 1.00 |
| ct_* dimensions only | 0.330 | 0.405 | 1.23 |

Total feature variance in the neighbourhood is identical across A
and C — confirming that degree preservation is working correctly
and that no additional information is being injected.

---

## 7. Measured Reasons C Differs from A

The following differences between A and C are measured from the actual
graph data.  No causal claims are made beyond what is directly observed.

1. **ct_* contrast in neighbourhood aggregation.**
   A's kNN wiring intentionally minimises ct_* distance between source
   and neighbours (mean diff = 0.061).  C's random wiring produces
   5.0× more ct_* contrast (mean diff = 0.309).  The 8 ct_* dimensions
   of `mean_j(x_j)` carry near-zero additional information in A.

2. **Cross-class connectivity.**
   In mixed-label windows, C connects nodes across the Normal/Attack
   boundary 4.3× more frequently than A (C: 29.8%, A: 6.9%).
   A's similarity-based wiring clusters same-type flows together.

3. **New information in SAGEConv input.**
   L2 distance between `x_i` and `mean_nbr` is 28% larger in C
   (6.47 vs 5.04).  The orthogonal component of `mean_nbr` relative
   to `x_i` is 27% larger in C (0.049 vs 0.038).

4. **Overall neighbour feature variance is equal.**
   C does not inject more total feature diversity — the C/A ratio of
   overall variance is 1.00.  The difference is specifically in which
   neighbours are selected, not in the total information pool.

---

## 8. CUDA Non-determinism Note

Despite setting `torch.manual_seed(42)`, `cudnn.deterministic=True`,
and `cudnn.benchmark=False`, repeated runs of the same experiment
may produce slightly different weights due to non-deterministic
`scatter_add` operations in SAGEConv on CUDA.  This is a known
PyTorch limitation when `torch.use_deterministic_algorithms(True)` is
not set.  The `random.Random(42)` train/val split is fully deterministic
and is identical across A, B, and C.

---

## 9. Reproducibility

```bash
# Phase 1 — preprocessing (must run first)
python run_preprocessing.py

# Phase 2 — graph construction
python run_graph_construction.py --window 100 --k 5

# Phase 4 — ablation study
python run_ablation.py
```

All results saved to `results/gnn/ablation/`:
- `A_original_metrics.json`
- `B_no_edges_metrics.json`
- `C_random_metrics.json`
- `A_original_history.json`
- `B_no_edges_history.json`
- `C_random_history.json`
- `comparison_table.json`
- `graphsage_A_original.pt`
- `graphsage_B_no_edges.pt`
- `graphsage_C_random.pt`
