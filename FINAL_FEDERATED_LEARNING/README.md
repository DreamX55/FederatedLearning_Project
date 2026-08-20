# Canonical Federated Learning Research Project

## Overview
This repository contains the complete, canonical implementation of the Federated Learning with Differential Privacy framework applied to Human Activity Recognition (HAR), as presented in the research paper *"Robust Human Activity Recognition through Federated Learning with Differential Privacy: A Comparison of Baseline and Centralized Models"*.

---

## Directory Structure

```text
FINAL_FEDERATED_LEARNING/
├── src/                        # Core source Python modules
│   ├── config/                 # Privacy and training configuration schemas
│   ├── datasets/               # HAR dataset loaders and client partitioning
│   ├── evaluation/             # Metrics calculation and ablation study scripts
│   ├── federated/              # Federated Server and Client FedAvg logic
│   ├── models/                 # FNN (fnn.py) and CNN_LSTM_Attn (mobile_optimized.py)
│   ├── optimization/           # Adaptive client selection
│   ├── privacy/                # Parameter delta DP noise addition & clipping
│   └── utils/                  # Logging and statistical utilities
├── scripts/                    # Entry point training and evaluation scripts
│   ├── train_federated.py      # Main FL + DP training loop
│   ├── train_federated_nodp.py # Baseline FL training without DP
│   ├── train_centralized.py    # Centralized FNN model training
│   ├── evaluate_centralized.py # Centralized evaluation script
│   ├── evaluate_federated_nodp.py # FL baseline evaluation script
│   ├── analysis.ipynb          # Jupyter Notebook for EDA, RF baseline, & SHAP/LIME XAI
│   └── run.sh                  # Execution shell script
├── models/                     # Trained baseline model binaries (.pt)
├── data/                       # Dataset storage
│   ├── processed/              # 30-subject pre-processed feature arrays
│   └── raw_uci_har/            # Raw UCI HAR dataset files
├── results/                    # Output experimental artifacts
│   ├── tables/                 # summary_results.csv, round_metrics.csv, client_metrics.csv
│   ├── figures/                # 6 PNG benchmark plots
│   └── checkpoints/            # 50 global model round checkpoints (.pt)
└── requirements.txt            # Python environment dependencies
```

---

## Quick Start & Reproduction

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Federated Training with Differential Privacy
```bash
python scripts/train_federated.py
```

### 3. Run Baselines
```bash
python scripts/train_federated_nodp.py
python scripts/train_centralized.py
```

### 4. Evaluate Models
```bash
python scripts/evaluate_centralized.py
python scripts/evaluate_federated_nodp.py
```
