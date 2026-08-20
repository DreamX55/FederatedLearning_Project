# Final Canonical Project Manifest

## Canonical Research Artifacts Mapping

### Canonical Code
* **Path:** `ROOT/FINAL_FEDERATED_LEARNING/src/` & `ROOT/FINAL_FEDERATED_LEARNING/scripts/`
* **Primary Entry Points:** 
  * `scripts/train_federated.py` (FL with Differential Privacy)
  * `scripts/train_federated_nodp.py` (FL without Differential Privacy)
  * `scripts/train_centralized.py` (Centralized FNN Baseline)
  * `scripts/analysis.ipynb` (Exploratory Data Analysis, Random Forest Baseline, SHAP/LIME XAI)

### Canonical Dataset
* **Path:** `ROOT/FINAL_FEDERATED_LEARNING/data/`
* **Raw Signals:** `data/raw_uci_har/` (UCI Human Activity Recognition Dataset)
* **Processed Features:** `data/processed/` (Subject-partitioned 561-dimensional numpy arrays for Subjects 1–30)

### Canonical Models
* **Path:** `ROOT/FINAL_FEDERATED_LEARNING/src/models/` & `ROOT/FINAL_FEDERATED_LEARNING/models/`
* **Architectures:**
  * `fnn.py`: 3-layer Feed-Forward Neural Network (`561 -> 128 -> 64 -> 32 -> 6`)
  * `mobile_optimized.py`: CNN-LSTM with Attention (`CNN_LSTM_Attn`)
* **Saved Model Weights:**
  * `models/centralized_model.pt` (Centralized FNN model binary)
  * `models/federated_nodp_model.pt` (Federated model binary)

### Canonical Results
* **Path:** `ROOT/FINAL_FEDERATED_LEARNING/results/`
* **Tables:** `results/tables/summary_results.csv`, `round_metrics.csv`, `client_metrics.csv`, `evaluation_results.txt`
* **Figures:** `results/figures/` (`confusion_matrix.png`, `learning_curves.png`, `noise_vs_utility.png`, `method_comparison.png`, `ablation_heatmap.png`, `client_variability_boxplot.png`)
* **Checkpoints:** `results/checkpoints/` (50 global round checkpoints: `global_model_round_0.pt` – `global_model_round_49.pt`)

### Canonical Paper Source
* **Path:** `ROOT/SPRINGER_LATEX/`
* **Main Document:** `SPRINGER_LATEX/main.tex`
* **Sections:** `sections/00_metadata.tex` through `08_conclusion.tex`
* **Style Assets:** `llncs.cls`, `splncs04.bst`, `styles/packages.tex`, `styles/macros.tex`, `styles/commands.tex`
* **Figures:** `figures/fig1.png` through `figures/fig10b.png`
* **Compiled PDF:** `SPRINGER_LATEX/main.pdf` (12 pages, compiled cleanly via `pdflatex`)

### Canonical Paper PDF Reference
* **Path:** `ROOT/PAPER_REFERENCE/latest_manuscript.pdf`
* **Source:** `Federated_Learning_Differential_Privacy_HAR_LNCS 23.11.32.pdf`

### Bibliography
* **Path:** `ROOT/SPRINGER_LATEX/references.bib`

### Dependencies
* **Path:** `ROOT/FINAL_FEDERATED_LEARNING/requirements.txt`
* **Key Packages:** `torch`, `numpy`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `shap`, `lime`, `pyyaml`

### GitHub Provenance
* **URL:** `https://github.com/Cypher9802/FederatedLearning_Project.git`
* **Branch:** `main`
* **Latest Commit:** `6459672 Update documentation and research findings in README.md`
