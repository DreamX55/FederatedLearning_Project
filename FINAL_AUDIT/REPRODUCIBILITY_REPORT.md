# Research Reproducibility Audit Report

## Component Validation Summary

| Research Component | Storage Path in Final Project | Validation Status | Evidence |
| ------------------ | ----------------------------- | ----------------- | -------- |
| **Dataset** | `FINAL_FEDERATED_LEARNING/data/processed/` & `data/raw_uci_har/` | **COMPLETE** | 30 Subject feature arrays ($N \times 561$) present |
| **Preprocessing** | `FINAL_FEDERATED_LEARNING/src/datasets/har_dataset.py` | **COMPLETE** | Automated subject-based client data partitioning |
| **Baselines** | `FINAL_FEDERATED_LEARNING/scripts/train_centralized.py` & `analysis.ipynb` | **COMPLETE** | FNN Baseline (85.87%) & Random Forest Baseline (84.44%) |
| **Federated Learning** | `FINAL_FEDERATED_LEARNING/scripts/train_federated_nodp.py` | **COMPLETE** | FedAvg on 30 clients (93.59% accuracy) |
| **Differential Privacy** | `FINAL_FEDERATED_LEARNING/src/privacy/differential_privacy.py` | **COMPLETE** | Parameter delta Gaussian noise & L2 clipping (88.93% accuracy) |
| **Centralized LSTM-CNN** | `FINAL_FEDERATED_LEARNING/src/models/mobile_optimized.py` | **COMPLETE** | `CNN_LSTM_Attn` model preserved (84.59% accuracy) |
| **Evaluation** | `FINAL_FEDERATED_LEARNING/scripts/evaluate_*.py` | **COMPLETE** | Metric calculation scripts (Accuracy, F1, Recall) |
| **XAI (SHAP & LIME)** | `FINAL_FEDERATED_LEARNING/scripts/analysis.ipynb` | **COMPLETE** | SHAP summary plots & LIME local explanation plots |
| **Result Tables & Checkpoints** | `FINAL_FEDERATED_LEARNING/results/` | **COMPLETE** | `summary_results.csv` & 50 global model `.pt` checkpoints |
| **Figures** | `SPRINGER_LATEX/figures/` & `FINAL_FEDERATED_LEARNING/results/figures/` | **COMPLETE** | All 10 paper figures present |
| **Paper Source** | `SPRINGER_LATEX/` | **COMPLETE** | Modular LaTeX project compiles cleanly to 12-page PDF |
| **Springer Compilation** | `pdflatex` / `bibtex` toolchain | **COMPLETE** | Tested and verified without external dependencies |
| **Environment Dependencies** | `FINAL_FEDERATED_LEARNING/requirements.txt` | **COMPLETE** | PyTorch, Scikit-learn, SHAP, LIME, Matplotlib |
