# GitHub Provenance and Git History Audit

## 1. Remote Repositories Overview

| Repository Folder | Remote Origin URL | Main Branch | Latest Commit | Role in Research Lifecycle |
| ----------------- | ----------------- | ----------- | ------------- | -------------------------- |
| `FederatedLearning_Project/` | `https://github.com/Cypher9802/FederatedLearning_Project.git` | `main` | `6459672` (Sun Apr 5 2026) | **Primary Canonical Repository** |
| `FL_Project/` | `https://github.com/Cypher9802/FL_Project.git` | `main` | `c8bbba9` | Experimental Prototype (Raw CNN-LSTM) |
| `privacy/` | `https://github.com/tensorflow/privacy.git` | `master` | `f27d79d` | External Dependency (TF Privacy Reference) |
| `opacus/` | `https://github.com/pytorch/opacus.git` | `main` | External | External Dependency (PyTorch Opacus Submodule) |

---

## 2. Commit Sequence of Canonical Repository (`FederatedLearning_Project`)

The commit log of `https://github.com/Cypher9802/FederatedLearning_Project.git` details the step-by-step progress of the research:

1. `561e845` — *All necessary code*
   * Initial project setup, directory creation.
2. `dd04bce` — *Created run.sh and added data into gitignore*
   * Execution scripting and environment boundaries.
3. `a75db34` — *Final complete code-93% accuracy*
   * Milestone commit: Reached 93.6% FL accuracy on UCI HAR.
4. `3c9d67a` — *Changed and fixed noise and clip_norm. also added the code to provide and print the DP noise*
   * Integration of parameter delta noise scale logging.
5. `164ed3c` — *Uses DP as delta for easier privacy management*
   * Switched from raw gradient DP hooks to parameter delta DP ($\Delta W$).
6. `315fafa` — *Plots and Maps*
   * Added `scripts/analysis.ipynb` for feature analysis and XAI (SHAP/LIME).
7. `071208e` — *Updated results/*
   * Added 6 PNG figures (`ablation_heatmap.png`, `confusion_matrix.png`, `learning_curves.png`, `method_comparison.png`, `noise_vs_utility.png`, `client_variability_boxplot.png`) and 3 CSV result tables (`summary_results.csv`, `client_metrics.csv`, `round_metrics.csv`).
8. `0b77159` — *Centralised & Federated(No DP) commits*
   * Added `train_centralized.py`, `train_federated_nodp.py`, and baseline trained models `models/centralized_model.pt` & `models/federated_nodp_model.pt`.
9. `6459672` — *Update documentation and research findings in README.md*
   * Updated repository `README.md` with benchmark summaries.

---

## 3. Local Untracked Additions vs GitHub Remote

The snapshot downloaded from GitHub (`FederatedLearning_Project-main`) represented remote commit `6459672`. 
However, the local workspace directory `FederatedLearning_Project/` contained critical untracked assets created after the last Git push:

* **LaTeX Manuscript (`docs/paper_draft/`):** Complete Springer LNCS LaTeX manuscript (`main.tex`, 8 sections, `references.bib`, 12 figures, compiled `main.pdf`).
* **Checkpoints (`results/checkpoints/`):** 50+ trained PyTorch `.pt` model binaries (`global_model_round_0.pt` through `global_model_round_49.pt`).
* **Springer Template (`docs/springer_template/`):** Official LNCS macros, styles, and documentation.
* **Migration Specs (`MIGRATION_SPEC.md`, `ARCHITECTURE_SPEC.md`):** Complete migration guides from MS Word to LaTeX.

---

## 4. Preservation Strategy

All remote Git commits and local untracked additions have been merged into the unified `FINAL_FEDERATED_LEARNING/` and `SPRINGER_LATEX/` folders, preserving complete historical provenance and full reproducibility.
