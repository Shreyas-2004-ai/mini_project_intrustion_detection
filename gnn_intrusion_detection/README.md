# Graph Neural Network-Based Intrusion Detection

Master's-level mini project — Stage 1: Data Inspection & Preprocessing

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
│   ├── raw/                   # Original CSV files (never modified)
│   └── processed/             # Output of preprocessing
│       ├── train_processed.csv
│       ├── test_processed.csv
│       └── preprocessor.pkl   # Serialised fitted pipeline
├── src/
│   ├── config.py              # Paths, column definitions, constants
│   ├── logger.py              # Shared logging factory
│   ├── data_loader.py         # CSV loading + column-role helpers
│   ├── inspector.py           # Inspection, statistics, report printing
│   └── preprocessor.py        # Sklearn pipeline: fit, transform, save/load
├── tests/
│   ├── test_data_loader.py
│   └── test_preprocessor.py
├── run_preprocessing.py       # Entry-point: inspect + preprocess + report
└── README.md
```

---

## Setup

Install dependencies (Python 3.8+):

```bash
pip install pandas numpy scikit-learn pytest
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
